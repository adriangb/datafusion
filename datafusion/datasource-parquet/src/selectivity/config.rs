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

//! [`TrackerConfig`] — immutable knobs for [`super::tracker::SelectivityTracker`].
//!
//! Use the builder methods to customise, then call
//! [`build()`](TrackerConfig::build) to produce a ready-to-use tracker.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;

use parking_lot::{Mutex, RwLock};

use super::tracker::{SelectivityTracker, SelectivityTrackerInner};

/// Immutable configuration for a [`SelectivityTracker`].
#[doc(hidden)]
pub struct TrackerConfig {
    /// Minimum bytes/sec throughput for promoting a filter (default: INFINITY = disabled).
    pub min_bytes_per_sec: f64,
    /// Byte-ratio threshold for initial filter placement (row-level vs post-scan).
    /// Computed as `filter_compressed_bytes / projection_compressed_bytes`.
    /// When low, the filter columns are small relative to the projection,
    /// so row-level placement enables large late-materialization savings.
    /// When high, the filter columns dominate the projection, so there's
    /// little benefit from late materialization.
    /// Default is 0.20.
    pub byte_ratio_threshold: f64,
    /// Z-score for confidence intervals on filter effectiveness.
    /// Lower values (e.g. 1.0 or 0.0) will make the tracker more aggressive about promotion/demotion based on limited data.
    /// Higher values (e.g. 3.0) will require more confidence before changing filter states.
    /// Default is 2.0, corresponding to ~97.5% one-sided confidence.
    /// Set to <= 0.0 to disable confidence intervals and promote/demote based on point estimates alone (not recommended).
    /// Set to INFINITY to disable promotion entirely (overrides `min_bytes_per_sec`).
    pub confidence_z: f64,
    /// Initial-placement prior threshold: if per-conjunct row-group
    /// statistics pruning prunes ≥ this fraction of the file's row
    /// groups, place the filter at row-level on first encounter. Set
    /// to >1.0 to disable the prior. Default 0.5.
    pub prior_promote_threshold: f64,
    /// Initial-placement prior threshold: if per-conjunct row-group
    /// statistics pruning prunes ≤ this fraction of the file's row
    /// groups, place the filter at post-scan on first encounter. Set
    /// to <0.0 to disable the prior. Default 0.05.
    pub prior_demote_threshold: f64,
    /// Per-fetch latency baseline in milliseconds — at this average
    /// per-fetch RTT the tracker uses the unmodified `confidence_z`.
    /// Above this, `confidence_z` is shrunk proportionally so the
    /// tracker becomes more aggressive about state changes when
    /// per-request cost is high. 0.0 disables. Default 5.0.
    pub latency_z_baseline_ms: f64,
    /// Maximum scale factor for the latency-aware z shrink. Default 8.0.
    pub latency_z_max_scale: f64,
}

impl TrackerConfig {
    pub fn new() -> Self {
        Self {
            min_bytes_per_sec: f64::INFINITY,
            byte_ratio_threshold: 0.20,
            confidence_z: 2.0,
            prior_promote_threshold: 0.5,
            prior_demote_threshold: 0.05,
            latency_z_baseline_ms: 5.0,
            latency_z_max_scale: 8.0,
        }
    }

    pub fn with_min_bytes_per_sec(mut self, v: f64) -> Self {
        self.min_bytes_per_sec = v;
        self
    }

    pub fn with_byte_ratio_threshold(mut self, v: f64) -> Self {
        self.byte_ratio_threshold = v;
        self
    }

    pub fn with_confidence_z(mut self, v: f64) -> Self {
        self.confidence_z = v;
        self
    }

    pub fn with_prior_promote_threshold(mut self, v: f64) -> Self {
        self.prior_promote_threshold = v;
        self
    }

    pub fn with_prior_demote_threshold(mut self, v: f64) -> Self {
        self.prior_demote_threshold = v;
        self
    }

    pub fn with_latency_z_baseline_ms(mut self, v: f64) -> Self {
        self.latency_z_baseline_ms = v;
        self
    }

    pub fn with_latency_z_max_scale(mut self, v: f64) -> Self {
        self.latency_z_max_scale = v;
        self
    }

    pub fn build(self) -> SelectivityTracker {
        SelectivityTracker {
            config: self,
            filter_stats: RwLock::new(HashMap::new()),
            skip_flags: RwLock::new(HashMap::new()),
            inner: Mutex::new(SelectivityTrackerInner::new()),
            total_fetch_ns: AtomicU64::new(0),
            total_fetches: AtomicU64::new(0),
        }
    }
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self::new()
    }
}
