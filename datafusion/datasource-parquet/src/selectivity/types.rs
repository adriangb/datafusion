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

//! Public types in the selectivity tracker's API surface.
//!
//! - [`FilterId`] — stable per-conjunct identifier assigned by
//!   `ParquetSource::with_predicate`.
//! - [`FilterState`] — the lifecycle states a filter moves through
//!   (RowFilter / PostScan / Dropped).
//! - [`PartitionedFilters`] — the tracker's output, consumed by
//!   `ParquetOpener::open` to wire row-level vs post-scan filters.
//!
//! [`PartitionResult`] is the internal contract between the outer
//! [`super::tracker::SelectivityTracker`] (which manages locks) and the
//! inner state machine (which does the partitioning logic).

use std::sync::Arc;

use datafusion_physical_expr_common::physical_expr::PhysicalExpr;

/// Stable identifier for a filter conjunct, assigned by `ParquetSource::with_predicate`.
pub type FilterId = usize;

/// Per-filter lifecycle state in the adaptive filter system.
///
/// State transitions:
/// - **(unseen)** → [`RowFilter`](Self::RowFilter) or [`PostScan`](Self::PostScan)
///   on first encounter in `SelectivityTracker::partition_filters`.
/// - [`PostScan`](Self::PostScan) → [`RowFilter`](Self::RowFilter) when
///   effectiveness ≥ `min_bytes_per_sec` and enough rows have been observed.
/// - [`RowFilter`](Self::RowFilter) → [`PostScan`](Self::PostScan) when
///   effectiveness is below threshold (mandatory filter).
/// - [`RowFilter`](Self::RowFilter) → [`Dropped`](Self::Dropped) when
///   effectiveness is below threshold and the filter is optional
///   (`OptionalFilterPhysicalExpr`).
/// - [`RowFilter`](Self::RowFilter) → [`PostScan`](Self::PostScan)/[`Dropped`](Self::Dropped)
///   on periodic re-evaluation if effectiveness drops below threshold after
///   CI upper bound drops below threshold.
/// - **Any state** → re-evaluated when a dynamic filter's
///   `snapshot_generation` changes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FilterState {
    /// Currently a row filter.
    RowFilter,
    /// Currently a post-scan filter.
    PostScan,
    /// Dropped entirely (insufficient throughput and optional).
    Dropped,
}

/// Result of partitioning filters into row filters vs post-scan.
///
/// Produced by `SelectivityTracker::partition_filters`, consumed by
/// `ParquetOpener::open` to build row-level predicates and post-scan filters.
///
/// Filters are partitioned based on their effectiveness threshold.
///
/// This type is `pub` to support the [selectivity tracker benchmark
/// harness](../../benches/selectivity_tracker.rs); treat the layout as
/// unstable from outside the crate.
#[derive(Debug, Clone, Default)]
#[doc(hidden)]
pub struct PartitionedFilters {
    /// Filters promoted past collection — individual chained ArrowPredicates
    pub row_filters: Vec<(FilterId, Arc<dyn PhysicalExpr>)>,
    /// Filters demoted to post-scan (fast path only)
    pub post_scan: Vec<(FilterId, Arc<dyn PhysicalExpr>)>,
}

/// Internal carrier between the outer
/// [`super::tracker::SelectivityTracker::partition_filters`] (which holds
/// the locks) and the inner state-machine partitioning logic. The outer
/// half uses `new_optional_flags` to insert per-filter
/// `Mutex<SelectivityStats>` entries during a brief Phase 2 write lock.
pub(super) struct PartitionResult {
    pub(super) partitioned: PartitionedFilters,
    /// `(FilterId, is_optional)` entries observed for the first time in
    /// this `partition_filters` call.
    pub(super) new_optional_flags: Vec<(FilterId, bool)>,
}
