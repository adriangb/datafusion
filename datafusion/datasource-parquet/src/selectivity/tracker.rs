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

//! [`SelectivityTracker`] — public API and locking shell — and
//! [`SelectivityTrackerInner`], the per-filter state machine it wraps.
//!
//! See the module-level docs on [`SelectivityTracker`] for the locking
//! design (two layers: a shared-read map of per-filter `Mutex`es plus an
//! inner state-machine `Mutex`) and the filter state diagram.

use arrow::datatypes::SchemaRef;
use log::debug;
use parking_lot::{Mutex, RwLock};
use parquet::file::metadata::ParquetMetaData;
use parquet::schema::types::SchemaDescriptor;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr_common::physical_expr::{
    OptionalFilterPhysicalExpr, PhysicalExpr, snapshot_generation,
};

use super::config::TrackerConfig;
use super::stats::SelectivityStats;
use super::types::{FilterId, FilterState, PartitionResult, PartitionedFilters};

/// # Filter state machine
///
/// ```text
///                        ┌─────────┐
///                        │   New   │
///                        └─────────┘
///                             │
///                             ▼
///                ┌────────────────────────┐
///                │     Estimated Cost     │
///                │Bytes needed for filter │
///                └────────────────────────┘
///                             │
///          ┌──────────────────┴──────────────────┐
/// ┌────────▼────────┐                   ┌────────▼────────┐
/// │    Post-scan    │                   │   Row filter    │
/// │                 │                   │                 │
/// └─────────────────┘                   └─────────────────┘
///          │                                     │
///          ▼                                     ▼
/// ┌─────────────────┐                   ┌─────────────────┐
/// │  Effectiveness  │                   │  Effectiveness  │
/// │  Bytes pruned   │                   │  Bytes pruned   │
/// │       per       │                   │       per       │
/// │Second of compute│                   │Second of compute│
/// └─────────────────┘                   └─────────────────┘
///          │                                     │
///          └──────────────────┬──────────────────┘
///                             ▼
///     ┌───────────────────────────────────────────────┐
///     │                   New Scan                    │
///     │     Move filters based on effectiveness.      │
///     │    Promote (move post-scan -> row filter).    │
///     │    Demote (move row-filter -> post-scan).     │
///     │   Disable (for optional filters; either row   │
///     │             filter or disabled).              │
///     └───────────────────────────────────────────────┘
///                             │
///          ┌──────────────────┴──────────────────┐
/// ┌────────▼────────┐                   ┌────────▼────────┐
/// │    Post-scan    │                   │   Row filter    │
/// │                 │                   │                 │
/// └─────────────────┘                   └─────────────────┘
/// ```
///
/// See `TrackerConfig` for configuration knobs.
pub struct SelectivityTracker {
    pub(super) config: TrackerConfig,
    /// Cumulative wall time spent inside `AsyncFileReader::get_byte_ranges`
    /// across all openers using this tracker.
    pub(super) total_fetch_ns: AtomicU64,
    /// Number of byte-range fetches recorded.
    pub(super) total_fetches: AtomicU64,
    /// Per-filter selectivity statistics, each individually `Mutex`-protected.
    ///
    /// The outer `RwLock` is almost always read-locked: both `update()` (hot,
    /// per-batch) and `partition_filters()` (cold, per-file-open) only need
    /// shared access to look up existing entries.  The write lock is taken
    /// only when `partition_filters()` inserts entries for newly-seen filter
    /// IDs — a brief, infrequent operation.
    ///
    /// Each inner `Mutex<SelectivityStats>` protects a single filter's
    /// counters, so concurrent `update()` calls on *different* filters
    /// proceed in parallel with zero contention.
    pub(super) filter_stats: RwLock<HashMap<FilterId, Mutex<SelectivityStats>>>,
    /// Per-filter "skip" flags — when set, the corresponding filter is
    /// treated as a no-op by both the row-filter
    /// (`DatafusionArrowPredicate::evaluate`) and the post-scan path
    /// (`apply_post_scan_filters_with_stats`). This is the mid-stream
    /// equivalent of dropping an optional filter: once the per-batch
    /// `update()` path proves an `OptionalFilterPhysicalExpr` is
    /// CPU-dominated and ineffective, it flips the flag and subsequent
    /// batches stop paying the evaluation cost. The decoder still decodes
    /// the filter columns (we cannot rebuild it mid-scan), so I/O is not
    /// reclaimed; only the predicate evaluation is skipped.
    ///
    /// Only ever set for filters whose `is_optional` flag (cached on the
    /// per-filter [`SelectivityStats`]) is `true` — mandatory filters
    /// must always execute or queries return wrong rows.
    pub(super) skip_flags: RwLock<HashMap<FilterId, Arc<AtomicBool>>>,
    /// Filter lifecycle state machine and dynamic-filter generation tracking.
    ///
    /// Only `partition_filters()` acquires this lock (once per file open).
    /// `update()` never touches it, so the hot per-batch path is completely
    /// decoupled from the cold state-machine path.
    pub(super) inner: Mutex<SelectivityTrackerInner>,
}

impl std::fmt::Debug for SelectivityTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectivityTracker")
            .field("config.min_bytes_per_sec", &self.config.min_bytes_per_sec)
            .finish()
    }
}

impl Default for SelectivityTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl SelectivityTracker {
    /// Create a new tracker with default settings (feature disabled).
    pub fn new() -> Self {
        TrackerConfig::new().build()
    }

    /// Record one batch of `get_byte_ranges` activity (latency-aware z input).
    pub fn record_fetch(&self, ranges: usize, elapsed_ns: u64) {
        if ranges == 0 || elapsed_ns == 0 {
            return;
        }
        self.total_fetch_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        self.total_fetches
            .fetch_add(ranges as u64, Ordering::Relaxed);
    }

    fn avg_fetch_ms(&self) -> f64 {
        let fetches = self.total_fetches.load(Ordering::Relaxed);
        if fetches == 0 {
            return 0.0;
        }
        let ns = self.total_fetch_ns.load(Ordering::Relaxed) as f64;
        ns / fetches as f64 / 1_000_000.0
    }

    fn effective_z(&self) -> f64 {
        let z = self.config.confidence_z;
        if self.config.latency_z_baseline_ms <= 0.0 {
            return z;
        }
        let avg = self.avg_fetch_ms();
        if avg <= self.config.latency_z_baseline_ms {
            return z;
        }
        let factor = (avg / self.config.latency_z_baseline_ms)
            .clamp(1.0, self.config.latency_z_max_scale);
        z / factor
    }

    /// Update stats for a filter after processing a batch.
    ///
    /// **Locking:** acquires `filter_stats.read()` (shared) then a per-filter
    /// `Mutex`.  Never touches `inner`, so this hot per-batch path cannot
    /// contend with the cold per-file-open `partition_filters()` path.
    ///
    /// Silently skips unknown filter IDs (can occur if `update()` is called
    /// before `partition_filters()` has registered the filter — in practice
    /// this cannot happen because `partition_filters()` runs during file open
    /// before any batches are processed).
    ///
    /// **Mid-stream drop:** after every `SKIP_FLAG_CHECK_INTERVAL`'th batch
    /// we evaluate the CI upper bound; if it falls below
    /// `min_bytes_per_sec` and the filter is wrapped in
    /// `OptionalFilterPhysicalExpr`, we set the per-filter skip flag.
    /// Subsequent calls to `DatafusionArrowPredicate::evaluate` (row-level)
    /// and `apply_post_scan_filters_with_stats` (post-scan) observe the
    /// flag and short-circuit their work for that filter. Mandatory
    /// filters are never flagged because doing so would change the result
    /// set.
    #[doc(hidden)]
    pub fn update(
        &self,
        id: FilterId,
        matched: u64,
        total: u64,
        eval_nanos: u64,
        batch_bytes: u64,
    ) {
        let stats_map = self.filter_stats.read();
        let Some(entry) = stats_map.get(&id) else {
            return;
        };
        let mut stats = entry.lock();
        stats.update(matched, total, eval_nanos, batch_bytes);

        // Fast path for non-optional filters: nothing else to do. The
        // SKIP_FLAG mid-stream drop only applies to
        // `OptionalFilterPhysicalExpr`-wrapped filters (hash-join /
        // TopK dynamic), and `is_optional` is cached inline on
        // `SelectivityStats` at filter registration so this is a single
        // field load on the already-held lock.
        if !stats.is_optional {
            return;
        }

        // Optional filter: do the SKIP_FLAG check every batch — there's
        // no SKIP_FLAG_CHECK_INTERVAL gate here on purpose. We want
        // join/TopK skip flags to fire as soon as stats prove the
        // filter's selectivity has collapsed, even mid-row-group. The
        // CI-bound calc is cheap arithmetic on already-locked stats.
        if !self.config.min_bytes_per_sec.is_finite() {
            return;
        }
        let z = self.effective_z();
        let Some(ub) = stats.confidence_upper_bound(z) else {
            return;
        };
        if ub >= self.config.min_bytes_per_sec {
            return;
        }
        drop(stats);
        drop(stats_map);

        if let Some(flag) = self.skip_flags.read().get(&id)
            && !flag.swap(true, Ordering::Release)
        {
            debug!(
                "FilterId {id}: mid-stream skip — CI upper bound {ub} < {} bytes/sec",
                self.config.min_bytes_per_sec
            );
        }
    }

    /// Returns the shared skip flag for `id`, creating one if absent.
    ///
    /// Cloned into [`crate::row_filter::DatafusionArrowPredicate`] so the
    /// row-filter path can short-circuit when the per-batch update path
    /// decides the filter has stopped pulling its weight. The post-scan
    /// path uses [`Self::is_filter_skipped`] instead — it does not need a
    /// long-lived handle.
    pub(crate) fn skip_flag(&self, id: FilterId) -> Arc<AtomicBool> {
        if let Some(existing) = self.skip_flags.read().get(&id) {
            return Arc::clone(existing);
        }
        let mut write = self.skip_flags.write();
        Arc::clone(
            write
                .entry(id)
                .or_insert_with(|| Arc::new(AtomicBool::new(false))),
        )
    }

    /// Returns `true` when `id` has been mid-stream-dropped by the tracker.
    ///
    /// Cheap: a single `RwLock::read` plus an atomic load. Called from the
    /// post-scan filter loop in `apply_post_scan_filters_with_stats`.
    pub(crate) fn is_filter_skipped(&self, id: FilterId) -> bool {
        self.skip_flags
            .read()
            .get(&id)
            .is_some_and(|f| f.load(Ordering::Acquire))
    }

    /// Partition filters into row-level predicates vs post-scan filters.
    ///
    /// Called once per file open (cold path).
    ///
    /// **Locking — two phases:**
    /// 1. Acquires `inner` (exclusive) and `filter_stats` (shared read) for
    ///    all decision logic — promotion, demotion, initial placement, and
    ///    sorting by effectiveness.  Because `filter_stats` is only
    ///    read-locked, concurrent `update()` calls proceed unblocked.
    /// 2. If new filter IDs were seen, briefly acquires `filter_stats` (write)
    ///    to insert per-filter `Mutex` entries so that future `update()` calls
    ///    can find them.
    #[doc(hidden)]
    #[expect(clippy::too_many_arguments)]
    pub fn partition_filters(
        &self,
        filters: Vec<(FilterId, Arc<dyn PhysicalExpr>)>,
        projection_columns: &std::collections::HashSet<usize>,
        projection_scan_size: usize,
        metadata: &ParquetMetaData,
        arrow_schema: &SchemaRef,
        parquet_schema: &SchemaDescriptor,
        page_pruning_rates: &HashMap<FilterId, f64>,
    ) -> PartitionedFilters {
        // Phase 1: inner.lock() + filter_stats.read() → all decision logic
        let z_eff = self.effective_z();
        let mut guard = self.inner.lock();
        let stats_map = self.filter_stats.read();
        let result = guard.partition_filters(
            filters,
            projection_columns,
            projection_scan_size,
            metadata,
            arrow_schema,
            parquet_schema,
            &self.config,
            z_eff,
            page_pruning_rates,
            &stats_map,
        );
        drop(stats_map);
        drop(guard);

        // Phase 2: if new filters were seen, briefly acquire write locks
        // to insert per-filter `Mutex<SelectivityStats>` (with
        // `is_optional` cached inline so the per-batch `update()` hot
        // path can fast-return for mandatory filters) and an
        // `AtomicBool` skip-flag (only consulted for optional filters).
        if !result.new_optional_flags.is_empty() {
            let mut stats_write = self.filter_stats.write();
            let mut skip_write = self.skip_flags.write();
            for (id, is_optional) in result.new_optional_flags {
                stats_write.entry(id).or_insert_with(|| {
                    Mutex::new(SelectivityStats {
                        is_optional,
                        ..Default::default()
                    })
                });
                skip_write
                    .entry(id)
                    .or_insert_with(|| Arc::new(AtomicBool::new(false)));
            }
        }

        result.partitioned
    }

    /// Test-only convenience that derives `arrow_schema` / `parquet_schema`
    /// from the parquet metadata and forwards to the public
    /// [`Self::partition_filters`]. Lets test code keep its existing call
    /// sites without threading two more arguments through every test.
    #[doc(hidden)]
    pub fn partition_filters_for_test(
        &self,
        filters: Vec<(FilterId, Arc<dyn PhysicalExpr>)>,
        projection_columns: &std::collections::HashSet<usize>,
        projection_scan_size: usize,
        metadata: &ParquetMetaData,
    ) -> PartitionedFilters {
        let parquet_schema = metadata.file_metadata().schema_descr_ptr();
        let arrow_schema: SchemaRef = match parquet::arrow::parquet_to_arrow_schema(
            parquet_schema.as_ref(),
            None,
        ) {
            Ok(s) => Arc::new(s),
            Err(_) => Arc::new(arrow::datatypes::Schema::empty()),
        };
        self.partition_filters(
            filters,
            projection_columns,
            projection_scan_size,
            metadata,
            &arrow_schema,
            parquet_schema.as_ref(),
            &HashMap::new(),
        )
    }

    /// Test helper: ensure a stats entry exists for the given filter ID.
    /// In production, `partition_filters()` inserts entries for new filters.
    /// Tests that call `update()` without prior `partition_filters()` need this.
    #[cfg(test)]
    pub(super) fn ensure_stats_entry(&self, id: FilterId) {
        let map = self.filter_stats.read();
        if map.get(&id).is_none() {
            drop(map);
            self.filter_stats
                .write()
                .entry(id)
                .or_insert_with(|| Mutex::new(SelectivityStats::default()));
        }
    }
}

/// Filter state-machine and generation tracking, guarded by the `Mutex`
/// inside [`SelectivityTracker`].
///
/// This struct intentionally does **not** contain per-filter stats — those
/// live in the separate `filter_stats` lock so that the hot `update()` path
/// can modify stats without acquiring this lock.  Only the cold
/// `partition_filters()` path (once per file open) needs this lock.
#[derive(Debug)]
pub(super) struct SelectivityTrackerInner {
    /// Per-filter lifecycle state (RowFilter / PostScan / Dropped).
    pub(super) filter_states: HashMap<FilterId, FilterState>,
    /// Last-seen snapshot generation per filter, for detecting when a dynamic
    /// filter's selectivity has changed (e.g. hash-join build side grew).
    pub(super) snapshot_generations: HashMap<FilterId, u64>,
}

impl SelectivityTrackerInner {
    pub(super) fn new() -> Self {
        Self {
            filter_states: HashMap::new(),
            snapshot_generations: HashMap::new(),
        }
    }

    /// Check and update the snapshot generation for a filter.
    pub(super) fn note_generation(
        &mut self,
        id: FilterId,
        generation: u64,
        stats_map: &HashMap<FilterId, Mutex<SelectivityStats>>,
    ) {
        if generation == 0 {
            return;
        }
        match self.snapshot_generations.get(&id) {
            Some(&prev_generation) if prev_generation == generation => {}
            Some(_) => {
                let current_state = self.filter_states.get(&id).copied();
                // Always reset stats since selectivity changed with new generation.
                if let Some(entry) = stats_map.get(&id) {
                    *entry.lock() = SelectivityStats::default();
                }
                self.snapshot_generations.insert(id, generation);

                // Optional/dynamic filters only get more selective over time
                // (hash join build side accumulates more values). So if the
                // filter was already working (RowFilter or PostScan), preserve
                // its state. Only un-drop Dropped filters back to PostScan
                // so they get another chance with the new selectivity.
                if current_state == Some(FilterState::Dropped) {
                    debug!("FilterId {id} generation changed, un-dropping to PostScan");
                    self.filter_states.insert(id, FilterState::PostScan);
                } else {
                    debug!(
                        "FilterId {id} generation changed, resetting stats but preserving state {current_state:?}"
                    );
                }
            }
            None => {
                self.snapshot_generations.insert(id, generation);
            }
        }
    }

    /// Get the effectiveness for a filter by ID.
    fn get_effectiveness_by_id(
        &self,
        id: FilterId,
        stats_map: &HashMap<FilterId, Mutex<SelectivityStats>>,
    ) -> Option<f64> {
        stats_map
            .get(&id)
            .and_then(|entry| entry.lock().effectiveness())
    }

    /// Demote a filter to post-scan or drop it entirely if optional.
    fn demote_or_drop(
        &mut self,
        id: FilterId,
        expr: &Arc<dyn PhysicalExpr>,
        post_scan: &mut Vec<(FilterId, Arc<dyn PhysicalExpr>)>,
        stats_map: &HashMap<FilterId, Mutex<SelectivityStats>>,
    ) {
        if expr.downcast_ref::<OptionalFilterPhysicalExpr>().is_none() {
            self.filter_states.insert(id, FilterState::PostScan);
            post_scan.push((id, Arc::clone(expr)));
            // Reset stats for this filter so it can be re-evaluated as a post-scan filter.
            if let Some(entry) = stats_map.get(&id) {
                *entry.lock() = SelectivityStats::default();
            }
        } else {
            self.filter_states.insert(id, FilterState::Dropped);
        }
    }

    /// Promote a filter to row-level.
    fn promote(
        &mut self,
        id: FilterId,
        expr: Arc<dyn PhysicalExpr>,
        row_filters: &mut Vec<(FilterId, Arc<dyn PhysicalExpr>)>,
        stats_map: &HashMap<FilterId, Mutex<SelectivityStats>>,
    ) {
        row_filters.push((id, expr));
        self.filter_states.insert(id, FilterState::RowFilter);
        // Reset stats for this filter since it will be evaluated at row-level now.
        if let Some(entry) = stats_map.get(&id) {
            *entry.lock() = SelectivityStats::default();
        }
    }

    /// Partition filters into collecting / promoted / post-scan buckets.
    #[expect(clippy::too_many_arguments)]
    fn partition_filters(
        &mut self,
        filters: Vec<(FilterId, Arc<dyn PhysicalExpr>)>,
        projection_columns: &std::collections::HashSet<usize>,
        projection_scan_size: usize,
        metadata: &ParquetMetaData,
        arrow_schema: &SchemaRef,
        parquet_schema: &SchemaDescriptor,
        config: &TrackerConfig,
        z_eff: f64,
        page_pruning_rates: &HashMap<FilterId, f64>,
        stats_map: &HashMap<FilterId, Mutex<SelectivityStats>>,
    ) -> PartitionResult {
        let mut new_optional_flags: Vec<(FilterId, bool)> = Vec::new();

        // If min_bytes_per_sec is INFINITY -> all filters are post-scan.
        if config.min_bytes_per_sec.is_infinite() {
            debug!(
                "Filter promotion disabled via min_bytes_per_sec=INFINITY; all {} filters post-scan",
                filters.len()
            );
            // Register all filter IDs so update() can find them
            for (id, expr) in &filters {
                if !stats_map.contains_key(id) {
                    new_optional_flags.push((*id, is_optional_filter(expr)));
                }
            }
            return PartitionResult {
                partitioned: PartitionedFilters {
                    row_filters: Vec::new(),
                    post_scan: filters,
                },
                new_optional_flags,
            };
        }
        // If min_bytes_per_sec is 0 -> all filters are promoted.
        if config.min_bytes_per_sec == 0.0 {
            debug!(
                "All filters promoted via min_bytes_per_sec=0; all {} filters row-level",
                filters.len()
            );
            // Register all filter IDs so update() can find them
            for (id, expr) in &filters {
                if !stats_map.contains_key(id) {
                    new_optional_flags.push((*id, is_optional_filter(expr)));
                }
            }
            return PartitionResult {
                partitioned: PartitionedFilters {
                    row_filters: filters,
                    post_scan: Vec::new(),
                },
                new_optional_flags,
            };
        }

        // Note snapshot generations for dynamic filter detection.
        // This clears stats for any filter whose generation has changed since the last scan.
        // This must be done before any other logic since it can change filter states and stats.
        for &(id, ref expr) in &filters {
            let generation = snapshot_generation(expr);
            self.note_generation(id, generation, stats_map);
        }

        // Separate into row filters and post-scan filters based on effectiveness and state.
        let mut row_filters: Vec<(FilterId, Arc<dyn PhysicalExpr>)> = Vec::new();
        let mut post_scan_filters: Vec<(FilterId, Arc<dyn PhysicalExpr>)> = Vec::new();

        // Use the latency-aware effective z (clamped to <= config.confidence_z).
        let confidence_z = z_eff;
        for (id, expr) in filters {
            let state = self.filter_states.get(&id).copied();

            let Some(state) = state else {
                // New filter: decide initial placement.
                //
                // We start at row-level only when the filter pulls in a
                // small amount of *extra* I/O — bytes for filter columns
                // **not already in the user projection** — relative to the
                // projection. These are the cases where the row-level
                // I/O cost is bounded and late materialization on a
                // selective filter is a clear win (think a small int
                // column predicate against a heavy string projection).
                //
                // Two cases default to post-scan instead, with the
                // tracker free to promote later if measured
                // bytes-saved-per-sec exceeds `min_bytes_per_sec`:
                //
                // - `extra_bytes == 0`: filter cols are entirely in the
                //   projection (e.g. `WHERE col <> '' GROUP BY col`).
                //   There's no I/O to save; the only payoff is late
                //   materialization on the *non*-filter projection
                //   columns, which depends on selectivity we don't know
                //   yet. Empirically (ClickBench Q10/11/13/14/26)
                //   defaulting these to row-level loses to post-scan
                //   because predicate-cache eviction on heavy string
                //   columns means the filter column is decoded twice.
                //
                // - `byte_ratio > byte_ratio_threshold`: extra I/O is
                //   too high to justify before we have evidence the
                //   filter is selective.
                //
                // Pre-existing snapshot-generation handling
                // ([`SelectivityTrackerInner::note_generation`]) keeps
                // dynamic filters (hash-join, TopK) at post-scan when
                // they re-arm with new values — those rely on row-group
                // statistics pruning rather than row-level I/O savings,
                // so post-scan is correct for them too.
                let filter_columns: Vec<usize> = collect_columns(&expr)
                    .iter()
                    .map(|col| col.index())
                    .collect();
                let extra_columns: Vec<usize> = filter_columns
                    .iter()
                    .copied()
                    .filter(|c| !projection_columns.contains(c))
                    .collect();
                let extra_bytes =
                    crate::row_filter::total_compressed_bytes(&extra_columns, metadata);
                let byte_ratio = if projection_scan_size > 0 {
                    extra_bytes as f64 / projection_scan_size as f64
                } else {
                    1.0
                };

                if !stats_map.contains_key(&id) {
                    new_optional_flags.push((id, is_optional_filter(&expr)));
                }

                // Selectivity prior from page-index pruning that the
                // opener already ran on this file (see
                // `PagePruningAccessPlanFilter::prune_plan_with_per_conjunct_stats`).
                // No extra pruning work is done here — we just look up
                // this filter's per-conjunct rate. When no rate is
                // available (page index disabled, predicate not
                // single-column, or schema mismatch), we fall back to
                // the existing byte-ratio heuristic.
                //
                // **Dynamic-filter refresh**: when this conjunct is a
                // populated DynamicFilter (snapshot_generation > 0)
                // we evaluate a per-conjunct `PruningPredicate` against
                // the file's row-group stats *now*, because the
                // side-effect rates captured at file open were taken
                // when the filter was still a placeholder. This is
                // targeted re-evaluation — only for dynamic conjuncts
                // that have updated since file open — so it doesn't
                // count as an "extra pruning run" on the static path.
                let dynamic_rate = if snapshot_generation(&expr) > 0 {
                    fresh_rate_for_dynamic_conjunct(
                        &expr,
                        arrow_schema,
                        parquet_schema,
                        metadata,
                    )
                } else {
                    None
                };
                let prior = dynamic_rate.or_else(|| page_pruning_rates.get(&id).copied());

                let row_level = match prior {
                    Some(p) if p >= config.prior_promote_threshold => {
                        debug!(
                            "FilterId {id}: New filter → Row filter via page-prior (pruned_rate={p:.3} >= {}) — {expr}",
                            config.prior_promote_threshold
                        );
                        true
                    }
                    Some(p) if p <= config.prior_demote_threshold => {
                        debug!(
                            "FilterId {id}: New filter → Post-scan via page-prior (pruned_rate={p:.3} <= {}) — {expr}",
                            config.prior_demote_threshold
                        );
                        false
                    }
                    _ => {
                        let r =
                            extra_bytes > 0 && byte_ratio <= config.byte_ratio_threshold;
                        debug!(
                            "FilterId {id}: New filter → {} via byte_ratio (byte_ratio={byte_ratio:.4}, extra_bytes={extra_bytes}, prior={prior:?}) — {expr}",
                            if r { "Row filter" } else { "Post-scan" }
                        );
                        r
                    }
                };

                if row_level {
                    self.filter_states.insert(id, FilterState::RowFilter);
                    row_filters.push((id, expr));
                } else {
                    self.filter_states.insert(id, FilterState::PostScan);
                    post_scan_filters.push((id, expr));
                }
                continue;
            };

            match state {
                FilterState::RowFilter => {
                    // Should we demote this filter based on CI upper bound?
                    if let Some(entry) = stats_map.get(&id) {
                        let stats = entry.lock();
                        if let Some(ub) = stats.confidence_upper_bound(confidence_z)
                            && ub < config.min_bytes_per_sec
                        {
                            drop(stats);
                            debug!(
                                "FilterId {id}: Row filter → Post-scan via CI upper bound {ub} < {} bytes/sec — {expr}",
                                config.min_bytes_per_sec
                            );
                            self.demote_or_drop(
                                id,
                                &expr,
                                &mut post_scan_filters,
                                stats_map,
                            );
                            continue;
                        }
                    }
                    // If not demoted, keep as row filter.
                    row_filters.push((id, expr));
                }
                FilterState::PostScan => {
                    // Single gate: scatter-aware CI lower bound on
                    // bytes-saved-per-sec ≥ `min_bytes_per_sec`.
                    //
                    // The metric (see [`SelectivityStats::update`])
                    // counts only sub-batch windows the filter empties
                    // out, so a 50% uniform filter scores ~0 and stays
                    // at post-scan; a TopK / hash-join / `Title LIKE`
                    // style filter where most batches drop entirely
                    // blows past the threshold.
                    //
                    // Earlier revisions also required `prune_rate ≥ 99%`
                    // on the theory that arrow-rs's row-level path
                    // double-decoded heavy string columns when the
                    // filter and projection overlapped. EXPLAIN ANALYZE
                    // on the ClickBench Q23 workload (URL LIKE
                    // `%google%`) showed the predicate cache is in fact
                    // active (`predicate_cache_inner_records=8.76M`)
                    // and the filter column is decoded once. The gate
                    // was removed; the residual ClickBench regressions
                    // we attributed to it (Q26 / Q31) trace to a
                    // different cause: post-scan filtering inside the
                    // opener changes batch-arrival order at downstream
                    // TopK, shifting the convergence point of TopK's
                    // dynamic filter and slightly weakening file-stats
                    // pruning. That has nothing to do with the
                    // promotion decision.
                    if let Some(entry) = stats_map.get(&id) {
                        let stats = entry.lock();
                        if let Some(lb) = stats.confidence_lower_bound(confidence_z)
                            && lb >= config.min_bytes_per_sec
                        {
                            drop(stats);
                            debug!(
                                "FilterId {id}: Post-scan → Row filter via CI lower bound {lb} >= {} bytes/sec — {expr}",
                                config.min_bytes_per_sec
                            );
                            self.promote(id, expr, &mut row_filters, stats_map);
                            continue;
                        }
                    }
                    // Should we drop this filter if it's optional and ineffective?
                    // Non-optional filters must stay as post-scan regardless.
                    if let Some(entry) = stats_map.get(&id) {
                        let stats = entry.lock();
                        if let Some(ub) = stats.confidence_upper_bound(confidence_z)
                            && ub < config.min_bytes_per_sec
                            && expr.downcast_ref::<OptionalFilterPhysicalExpr>().is_some()
                        {
                            drop(stats);
                            debug!(
                                "FilterId {id}: Post-scan → Dropped via CI upper bound {ub} < {} bytes/sec — {expr}",
                                config.min_bytes_per_sec
                            );
                            self.filter_states.insert(id, FilterState::Dropped);
                            continue;
                        }
                    }
                    // Keep as post-scan filter (don't reset stats for mandatory filters).
                    post_scan_filters.push((id, expr));
                }
                FilterState::Dropped => continue,
            }
        }

        // Sort row filters by:
        // - Effectiveness (descending, higher = better) if available for both filters.
        // - Scan size (ascending, cheapest first) as fallback — cheap filters prune
        //   rows before expensive ones, reducing downstream evaluation cost.
        let cmp_row_filters =
            |(id_a, expr_a): &(FilterId, Arc<dyn PhysicalExpr>),
             (id_b, expr_b): &(FilterId, Arc<dyn PhysicalExpr>)| {
                let eff_a = self.get_effectiveness_by_id(*id_a, stats_map);
                let eff_b = self.get_effectiveness_by_id(*id_b, stats_map);
                if let (Some(eff_a), Some(eff_b)) = (eff_a, eff_b) {
                    eff_b
                        .partial_cmp(&eff_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    let size_a = filter_scan_size(expr_a, metadata);
                    let size_b = filter_scan_size(expr_b, metadata);
                    size_a.cmp(&size_b)
                }
            };
        row_filters.sort_by(cmp_row_filters);
        // Post-scan filters: same logic (cheaper post-scan filters first to reduce
        // the batch size for subsequent filters).
        post_scan_filters.sort_by(cmp_row_filters);

        debug!(
            "Partitioned filters: {} row-level, {} post-scan",
            row_filters.len(),
            post_scan_filters.len()
        );
        PartitionResult {
            partitioned: PartitionedFilters {
                row_filters,
                post_scan: post_scan_filters,
            },
            new_optional_flags,
        }
    }
}

/// Returns `true` if `expr` is wrapped in [`OptionalFilterPhysicalExpr`].
fn is_optional_filter(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.downcast_ref::<OptionalFilterPhysicalExpr>().is_some()
}

/// Calculate the estimated number of bytes needed to evaluate a filter based on the columns
/// it references as if it were applied to the entire file.
/// This is used for initial placement of new filters before any stats are available, and as a fallback for filters without stats.
fn filter_scan_size(expr: &Arc<dyn PhysicalExpr>, metadata: &ParquetMetaData) -> usize {
    let columns: Vec<usize> = collect_columns(expr)
        .iter()
        .map(|col| col.index())
        .collect();

    crate::row_filter::total_compressed_bytes(&columns, metadata)
}

// (Per-conjunct page-pruning rates are now extracted as a side-effect
// of the opener's existing page-index pruning pass — see
// `PagePruningAccessPlanFilter::prune_plan_with_per_conjunct_stats`.
// `partition_filters` reads them through its `page_pruning_rates`
// parameter; no extra pruning runs happen on the static path.)

/// Compute a fresh row-group pruning rate for a single dynamic
/// conjunct, evaluated against the file's row-group statistics
/// *now*. Used by `partition_filters` to refresh the prior for
/// dynamic filters that were placeholders when the side-effect
/// rates were captured at file open and have since been populated
/// by the join build side.
///
/// Returns `None` when the conjunct doesn't translate into a
/// usable pruning predicate (e.g. always-true after rewriting,
/// references columns missing from the schema, contains
/// hash_lookup-style nodes the rewriter can't handle).
fn fresh_rate_for_dynamic_conjunct(
    expr: &Arc<dyn PhysicalExpr>,
    arrow_schema: &SchemaRef,
    parquet_schema: &SchemaDescriptor,
    metadata: &ParquetMetaData,
) -> Option<f64> {
    use datafusion_pruning::PruningPredicate;
    // Unwrap OptionalFilterPhysicalExpr — pruning should evaluate
    // the underlying predicate, not the marker.
    let inner = if let Some(opt) = expr.downcast_ref::<OptionalFilterPhysicalExpr>() {
        opt.inner()
    } else {
        Arc::clone(expr)
    };
    let groups = metadata.row_groups();
    if groups.is_empty() {
        return None;
    }
    let stats = crate::row_group_filter::RowGroupPruningStatistics {
        parquet_schema,
        row_group_metadatas: groups.iter().collect(),
        arrow_schema: arrow_schema.as_ref(),
    };

    // First try: build a PruningPredicate from the whole conjunct.
    if let Ok(pp) =
        PruningPredicate::try_new(Arc::clone(&inner), Arc::clone(arrow_schema))
        && !pp.always_true()
        && let Ok(kept) = pp.prune(&stats)
        && !kept.is_empty()
    {
        let total = kept.len();
        let pruned = total - kept.iter().filter(|b| **b).count();
        return Some(pruned as f64 / total as f64);
    }

    // Second try (the AND-with-hash-lookup case): snapshot the
    // dynamic filter to materialize its current inner expression,
    // then split the AND inside. `split_conjunction` doesn't descend
    // into DynamicFilterPhysicalExpr wrappers, so without this step
    // the split would return `[dynamic_filter]` and miss the
    // prunable parts inside. We take the *max* pruning rate across
    // sub-parts as a *promote* signal — if any sub-conjunct prunes
    // a high fraction, the whole AND prunes at least that much. We
    // deliberately do NOT use this as a demote signal.
    let snapshot_result =
        datafusion_physical_expr_common::physical_expr::snapshot_physical_expr_opt(
            Arc::clone(&inner),
        )
        .ok()?;
    let snapshotted = snapshot_result.data;
    let parts = datafusion_physical_expr::split_conjunction(&snapshotted);
    if parts.len() < 2 {
        return None;
    }
    let mut max_rate: Option<f64> = None;
    for part in parts {
        let Ok(pp) =
            PruningPredicate::try_new(Arc::clone(part), Arc::clone(arrow_schema))
        else {
            continue;
        };
        if pp.always_true() {
            continue;
        }
        let Ok(kept) = pp.prune(&stats) else { continue };
        if kept.is_empty() {
            continue;
        }
        let total = kept.len();
        let pruned = total - kept.iter().filter(|b| **b).count();
        let rate = pruned as f64 / total as f64;
        max_rate = Some(max_rate.map_or(rate, |m| m.max(rate)));
    }
    // Promote-only semantics: only return when the partial-AND rate
    // is high enough to be a confident promote signal. Below that we
    // return None and let the standard prior / byte-ratio fallback
    // run, which won't be misled by an undercounted rate.
    max_rate.filter(|&r| r >= 0.5)
}
