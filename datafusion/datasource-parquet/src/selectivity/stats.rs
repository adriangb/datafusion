// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Per-filter selectivity counters (`SelectivityStats`) and the Welford
//! running-mean / running-variance accumulator used to drive
//! confidence-interval gating in the tracker.
//!
//! Each filter's stats live behind a per-filter `Mutex` inside
//! [`super::tracker::SelectivityTracker`], so the hot per-batch update
//! path contends only on the cheap inner mutex; multiple filters update
//! in parallel.

/// Tracks selectivity statistics for a single filter expression.
#[derive(Debug, Clone, Default, Copy, PartialEq)]
pub(super) struct SelectivityStats {
    /// Number of rows that matched (passed) the filter
    pub(super) rows_matched: u64,
    /// Total number of rows evaluated
    pub(super) rows_total: u64,
    /// Cumulative evaluation time in nanoseconds
    pub(super) eval_nanos: u64,
    /// Cumulative bytes across batches this filter has been evaluated on
    pub(super) bytes_seen: u64,
    /// Welford's online algorithm: number of per-batch effectiveness samples
    pub(super) sample_count: u64,
    /// Welford's online algorithm: running mean of per-batch effectiveness
    pub(super) eff_mean: f64,
    /// Welford's online algorithm: running sum of squared deviations (M2)
    pub(super) eff_m2: f64,
    /// Whether the underlying expression is wrapped in
    /// `OptionalFilterPhysicalExpr`. Cached here (rather than looked up
    /// in `SelectivityTracker::is_optional`) so the per-batch hot path
    /// in `SelectivityTracker::update` can skip the
    /// SKIP_FLAG/CI-bound work entirely for non-optional filters with a
    /// single field load on the already-held stats lock — no extra
    /// HashMap or `RwLock::read()` per batch.
    pub(super) is_optional: bool,
}

impl SelectivityStats {
    /// Returns the cumulative effectiveness as an opaque ordering score
    /// (higher = run first).
    ///
    /// Computed from `eff_mean` so it matches the Welford-tracked metric
    /// fed to CI bounds: per-batch scatter-aware bytes-saved-per-second.
    /// Callers should not assume the unit.
    pub(super) fn effectiveness(&self) -> Option<f64> {
        if self.sample_count == 0 {
            return None;
        }
        Some(self.eff_mean)
    }

    /// Returns the lower bound of a confidence interval on mean effectiveness.
    ///
    /// Uses Welford's online variance to compute a one-sided CI:
    /// `mean - z * stderr`. Returns `None` if fewer than 2 samples.
    pub(super) fn confidence_lower_bound(&self, confidence_z: f64) -> Option<f64> {
        if self.sample_count < 2 {
            return None;
        }
        let variance = self.eff_m2 / (self.sample_count - 1) as f64;
        let stderr = (variance / self.sample_count as f64).sqrt();
        Some(self.eff_mean - confidence_z * stderr)
    }

    /// Returns the upper bound of a confidence interval on mean effectiveness.
    ///
    /// Uses Welford's online variance: `mean + z * stderr`.
    /// Returns `None` if fewer than 2 samples.
    pub(super) fn confidence_upper_bound(&self, confidence_z: f64) -> Option<f64> {
        if self.sample_count < 2 {
            return None;
        }
        let variance = self.eff_m2 / (self.sample_count - 1) as f64;
        let stderr = (variance / self.sample_count as f64).sqrt();
        Some(self.eff_mean + confidence_z * stderr)
    }

    /// Update stats with new observations.
    ///
    /// `skippable_bytes` is the caller's already-computed estimate of
    /// non-filter projection bytes that late-materialization would
    /// actually save for this batch — see
    /// [`super::skippable::count_skippable_bytes`] for the windowed
    /// scatter calculation. The Welford accumulator tracks
    /// `skippable_bytes × 1e9 / eval_nanos` (= scatter-aware
    /// bytes-saved-per-second), which is what the promote/demote
    /// gates compare against `min_bytes_per_sec`.
    pub(super) fn update(
        &mut self,
        matched: u64,
        total: u64,
        eval_nanos: u64,
        skippable_bytes: u64,
    ) {
        self.rows_matched += matched;
        self.rows_total += total;
        self.eval_nanos += eval_nanos;
        self.bytes_seen += skippable_bytes;

        if total > 0 && eval_nanos > 0 {
            let batch_eff = skippable_bytes as f64 * 1e9 / eval_nanos as f64;

            self.sample_count += 1;
            let delta = batch_eff - self.eff_mean;
            self.eff_mean += delta / self.sample_count as f64;
            let delta2 = batch_eff - self.eff_mean;
            self.eff_m2 += delta * delta2;
        }
    }
}
