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

//! Scatter-aware "bytes that late-materialization could skip" calculation.
//!
//! Used by both the row-filter path (after evaluating a predicate against a
//! batch, to *measure* what was skipped) and by the post-scan path
//! (predicting what *would* be skipped if the filter were promoted to
//! row-level). See [`count_skippable_bytes`] for the per-window logic and
//! the interpretation under each call site.

use arrow::array::BooleanArray;

/// Window size for the per-batch scatter analysis fed to
/// [`count_skippable_bytes`]. Approximates a parquet data page so that
/// "windows with zero survivors" tracks "pages a row-level decoder
/// could skip". Hardcoded for now; making this configurable (or
/// deriving it from per-row-group page metadata) is a natural follow-up.
pub(crate) const SKIP_WINDOW_ROWS: usize = 8192;

/// Compute the bytes that late-materialization can plausibly skip for a
/// batch given the predicate output `bool_arr` and the total non-filter
/// projection bytes for that batch.
///
/// Splits `bool_arr` into [`SKIP_WINDOW_ROWS`]-sized windows; each window
/// with zero survivors represents a page-sized chunk whose
/// other-projection columns the row-level decoder can skip outright.
/// Returns `total_other_bytes × (empty_windows / total_windows)` —
/// scatter-discounted skippable bytes.
///
/// Interpretation depends on which side calls this:
///
/// - **Post-scan path**: a *prediction* of bytes-saved-per-sec the
///   row-level path would achieve. The bool_arr we see is over the wide
///   batch in the same row order the decoder would emit, so for single-
///   predicate filters the prediction is faithful (modulo `W` matching
///   the actual parquet page size).
///
/// - **Row-level path**: a conservative *measurement* of what the
///   decoder actually skipped — within-window RowSelection narrowing is
///   an additional uncounted bonus. So at row-level this is a *lower
///   bound* of real savings, which is the safe direction for the
///   demote-or-not decision.
pub(crate) fn count_skippable_bytes(
    bool_arr: &BooleanArray,
    total_other_bytes: u64,
) -> u64 {
    let n = bool_arr.len();
    if n == 0 || total_other_bytes == 0 {
        return 0;
    }
    // Short-circuit on the two extremes: avoids a redundant per-window
    // SIMD scan over the same buffer when the answer is already
    // determined by the batch-level total. The whole helper otherwise
    // costs ~2× per-batch `true_count` for nothing.
    let total_matched = bool_arr.true_count();
    if total_matched == 0 {
        // Every window empty: full skippable.
        return total_other_bytes;
    }
    if total_matched == n {
        // No window empty: nothing skippable.
        return 0;
    }
    let total_windows = n.div_ceil(SKIP_WINDOW_ROWS);
    if total_windows == 1 {
        // One-window batch with mixed matches → not skippable. Avoids
        // a wasted slice+`true_count`.
        return 0;
    }
    let mut empty_windows: u64 = 0;
    for i in 0..total_windows {
        let start = i * SKIP_WINDOW_ROWS;
        let len = SKIP_WINDOW_ROWS.min(n - start);
        if bool_arr.slice(start, len).true_count() == 0 {
            empty_windows += 1;
        }
    }
    ((total_other_bytes as f64 * empty_windows as f64) / total_windows as f64) as u64
}
