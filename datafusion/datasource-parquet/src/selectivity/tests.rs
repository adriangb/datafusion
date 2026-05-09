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

//! Tests for the [`super`] selectivity tracker. Exercises the public
//! API plus selected internal items re-exported from
//! [`super::mod`](super) for test access.

use super::types::FilterState;
use super::*;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use parquet::basic::Type as PhysicalType;
use parquet::file::metadata::{
    ColumnChunkMetaData, FileMetaData, ParquetMetaData, RowGroupMetaData,
};
use parquet::schema::types::SchemaDescPtr;
use parquet::schema::types::Type as SchemaType;
use std::sync::Arc;

mod helper_functions {
    use super::*;

    /// Creates test ParquetMetaData with specified row groups and column sizes.
    ///
    /// # Arguments
    /// * `specs` - Vec of (num_rows, vec![compressed_size]) tuples for each row group
    pub fn create_test_metadata(specs: Vec<(i64, Vec<usize>)>) -> ParquetMetaData {
        // Get the maximum number of columns from all specs
        let num_columns = specs
            .iter()
            .map(|(_, sizes)| sizes.len())
            .max()
            .unwrap_or(1);
        let schema_descr = get_test_schema_descr_with_columns(num_columns);

        let row_group_metadata: Vec<_> = specs
            .into_iter()
            .map(|(num_rows, column_sizes)| {
                let columns = column_sizes
                    .into_iter()
                    .enumerate()
                    .map(|(col_idx, size)| {
                        ColumnChunkMetaData::builder(schema_descr.column(col_idx))
                            .set_num_values(num_rows)
                            .set_total_compressed_size(size as i64)
                            .build()
                            .unwrap()
                    })
                    .collect();

                RowGroupMetaData::builder(schema_descr.clone())
                    .set_num_rows(num_rows)
                    .set_column_metadata(columns)
                    .build()
                    .unwrap()
            })
            .collect();

        let total_rows: i64 = row_group_metadata.iter().map(|rg| rg.num_rows()).sum();
        let file_metadata =
            FileMetaData::new(1, total_rows, None, None, schema_descr.clone(), None);

        ParquetMetaData::new(file_metadata, row_group_metadata)
    }

    /// Creates a simple column expression with given name and index.
    pub fn col_expr(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new(name, index))
    }

    /// Create schema with specified number of columns, each named "a", "b", etc.
    pub fn get_test_schema_descr_with_columns(num_columns: usize) -> SchemaDescPtr {
        use parquet::basic::LogicalType;

        let fields: Vec<_> = (0..num_columns)
            .map(|i| {
                let col_name = format!("{}", (b'a' + i as u8) as char);
                SchemaType::primitive_type_builder(&col_name, PhysicalType::BYTE_ARRAY)
                    .with_logical_type(Some(LogicalType::String))
                    .build()
                    .unwrap()
            })
            .map(Arc::new)
            .collect();

        let schema = SchemaType::group_type_builder("schema")
            .with_fields(fields)
            .build()
            .unwrap();
        Arc::new(parquet::schema::types::SchemaDescriptor::new(Arc::new(
            schema,
        )))
    }
}

mod selectivity_stats_tests {
    use super::*;

    #[test]
    fn test_effectiveness_basic_calculation() {
        let mut stats = SelectivityStats::default();

        // skippable_bytes is now caller-computed (= rows_pruned *
        // bytes_per_row in the simple case), so passing 5000 directly
        // models the same scenario the old test described:
        // "100 rows total, 50 pruned, 100 bytes/row → 5000 saved".
        stats.update(50, 100, 1_000_000_000, 5_000);

        let eff = stats.effectiveness().unwrap();
        assert!((eff - 5000.0).abs() < 0.1);
    }

    #[test]
    fn test_effectiveness_zero_rows_total() {
        let mut stats = SelectivityStats::default();
        stats.update(0, 0, 1_000_000_000, 10_000);

        assert_eq!(stats.effectiveness(), None);
    }

    #[test]
    fn test_effectiveness_zero_eval_nanos() {
        let mut stats = SelectivityStats::default();
        stats.update(50, 100, 0, 10_000);

        assert_eq!(stats.effectiveness(), None);
    }

    #[test]
    fn test_effectiveness_zero_bytes_seen() {
        // A batch with zero skippable_bytes is a legitimate sample
        // ("filter ran, late-mat had nothing to save") — Welford
        // records it as eff=0 rather than discarding it, so the
        // demotion path can see "CPU spent, no payoff."
        let mut stats = SelectivityStats::default();
        stats.update(50, 100, 1_000_000_000, 0);

        assert_eq!(stats.effectiveness(), Some(0.0));
    }

    #[test]
    fn test_effectiveness_all_rows_matched() {
        let mut stats = SelectivityStats::default();
        // All rows matched (no pruning) — caller computes
        // skippable_bytes = rows_pruned * bytes_per_row = 0.
        stats.update(100, 100, 1_000_000_000, 0);

        let eff = stats.effectiveness().unwrap();
        assert_eq!(eff, 0.0);
    }

    #[test]
    fn test_confidence_bounds_single_sample() {
        let mut stats = SelectivityStats::default();
        stats.update(50, 100, 1_000_000_000, 10_000);

        // Single sample returns None for confidence bounds
        assert_eq!(stats.confidence_lower_bound(2.0), None);
        assert_eq!(stats.confidence_upper_bound(2.0), None);
    }

    #[test]
    fn test_welford_identical_samples() {
        let mut stats = SelectivityStats::default();

        // Add two identical samples
        stats.update(50, 100, 1_000_000_000, 10_000);
        stats.update(50, 100, 1_000_000_000, 10_000);

        // Variance should be 0
        assert_eq!(stats.sample_count, 2);
        let lb = stats.confidence_lower_bound(2.0).unwrap();
        let ub = stats.confidence_upper_bound(2.0).unwrap();

        // Both should be equal to the mean since variance is 0
        assert!((lb - ub).abs() < 0.01);
    }

    #[test]
    fn test_welford_variance_calculation() {
        let mut stats = SelectivityStats::default();

        // Add samples that produce effectiveness values 5000, 6000, 7000
        // (caller-computed skippable_bytes is the lever now).
        stats.update(50, 100, 1_000_000_000, 5_000); // eff = 5000
        stats.update(40, 100, 1_000_000_000, 6_000); // eff = 6000
        stats.update(30, 100, 1_000_000_000, 7_000); // eff = 7000

        // We should have 3 samples
        assert_eq!(stats.sample_count, 3);

        // Mean should be 6000
        assert!((stats.eff_mean - 6000.0).abs() < 1.0);

        // Both bounds should be defined
        let lb = stats.confidence_lower_bound(1.0).unwrap();
        let ub = stats.confidence_upper_bound(1.0).unwrap();

        assert!(lb < stats.eff_mean);
        assert!(ub > stats.eff_mean);
    }

    #[test]
    fn test_confidence_bounds_asymmetry() {
        let mut stats = SelectivityStats::default();

        stats.update(50, 100, 1_000_000_000, 10_000);
        stats.update(40, 100, 1_000_000_000, 10_000);

        let lb = stats.confidence_lower_bound(2.0).unwrap();
        let ub = stats.confidence_upper_bound(2.0).unwrap();

        // Bounds should be symmetric around the mean
        let lower_dist = stats.eff_mean - lb;
        let upper_dist = ub - stats.eff_mean;

        assert!((lower_dist - upper_dist).abs() < 0.01);
    }

    #[test]
    fn test_welford_incremental_vs_batch() {
        // Create two identical stats objects
        let mut stats_incremental = SelectivityStats::default();
        let mut stats_batch = SelectivityStats::default();

        // Incremental: add one at a time
        stats_incremental.update(50, 100, 1_000_000_000, 10_000);
        stats_incremental.update(40, 100, 1_000_000_000, 10_000);
        stats_incremental.update(30, 100, 1_000_000_000, 10_000);

        // Batch: simulate batch update (all at once)
        stats_batch.update(120, 300, 3_000_000_000, 30_000);

        // Both should produce the same overall statistics
        assert_eq!(stats_incremental.rows_total, stats_batch.rows_total);
        assert_eq!(stats_incremental.rows_matched, stats_batch.rows_matched);

        // Means should be close
        assert!((stats_incremental.eff_mean - stats_batch.eff_mean).abs() < 100.0);
    }

    #[test]
    fn test_effectiveness_numerical_stability() {
        let mut stats = SelectivityStats::default();

        // Test with large values to ensure numerical stability
        stats.update(
            500_000_000,
            1_000_000_000,
            10_000_000_000_000,
            1_000_000_000_000,
        );

        let eff = stats.effectiveness();
        assert!(eff.is_some());
        assert!(eff.unwrap() > 0.0);
        assert!(!eff.unwrap().is_nan());
        assert!(!eff.unwrap().is_infinite());
    }
}

mod tracker_config_tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TrackerConfig::default();

        assert!(config.min_bytes_per_sec.is_infinite());
        assert_eq!(config.byte_ratio_threshold, 0.20);
        assert_eq!(config.confidence_z, 2.0);
    }

    #[test]
    fn test_with_min_bytes_per_sec() {
        let config = TrackerConfig::new().with_min_bytes_per_sec(1000.0);

        assert_eq!(config.min_bytes_per_sec, 1000.0);
    }

    #[test]
    fn test_with_byte_ratio_threshold() {
        let config = TrackerConfig::new().with_byte_ratio_threshold(0.5);

        assert_eq!(config.byte_ratio_threshold, 0.5);
    }

    #[test]
    fn test_with_confidence_z() {
        let config = TrackerConfig::new().with_confidence_z(3.0);

        assert_eq!(config.confidence_z, 3.0);
    }

    #[test]
    fn test_builder_chain() {
        let config = TrackerConfig::new()
            .with_min_bytes_per_sec(500.0)
            .with_byte_ratio_threshold(0.3)
            .with_confidence_z(1.5);

        assert_eq!(config.min_bytes_per_sec, 500.0);
        assert_eq!(config.byte_ratio_threshold, 0.3);
        assert_eq!(config.confidence_z, 1.5);
    }

    #[test]
    fn test_build_creates_tracker() {
        let tracker = TrackerConfig::new().with_min_bytes_per_sec(1000.0).build();

        // Tracker should be created and functional
        assert_eq!(tracker.config.min_bytes_per_sec, 1000.0);
    }
}

mod state_machine_tests {
    use super::helper_functions::*;
    use super::*;

    #[test]
    fn test_initial_placement_low_byte_ratio() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(1000.0)
            .with_byte_ratio_threshold(0.2)
            .build();

        // Create metadata: 1 row group, 100 rows, 1000 bytes for column
        let metadata = create_test_metadata(vec![(100, vec![1000])]);

        // Filter using column 0 (1000 bytes out of 1000 projection = 100% ratio > 0.2)
        // So this should be placed in post-scan initially
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr)];

        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // With 100% byte ratio, should go to post-scan
        assert_eq!(result.row_filters.len(), 0);
        assert_eq!(result.post_scan.len(), 1);
    }

    #[test]
    fn test_initial_placement_filter_in_projection_low_ratio() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(1000.0)
            .with_byte_ratio_threshold(0.5)
            .build();

        // Create metadata: 1 row group, 100 rows, 100 bytes for column
        let metadata = create_test_metadata(vec![(100, vec![100])]);

        // Filter using column 0 which IS in the projection.
        // filter_bytes=100, projection=1000, ratio=0.10 <= 0.5 → RowFilter
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr)];

        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        assert_eq!(result.row_filters.len(), 1);
        assert_eq!(result.post_scan.len(), 0);
    }

    #[test]
    fn test_initial_placement_high_byte_ratio() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(1000.0)
            .with_byte_ratio_threshold(0.5)
            .build();

        // Create metadata: 1 row group, 100 rows, 100 bytes for column
        let metadata = create_test_metadata(vec![(100, vec![100])]);

        // Filter using column 0 (100 bytes / 1000 projection = 10% ratio <= 0.5)
        // So this should be placed in row-filter immediately
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr)];

        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // With 10% byte ratio, should go to row-filter
        assert_eq!(result.row_filters.len(), 1);
        assert_eq!(result.post_scan.len(), 0);
    }

    #[test]
    fn test_min_bytes_per_sec_infinity_disables_promotion() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(f64::INFINITY)
            .build();

        let metadata = create_test_metadata(vec![(100, vec![100])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr)];

        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // All filters should go to post_scan when min_bytes_per_sec is INFINITY
        assert_eq!(result.row_filters.len(), 0);
        assert_eq!(result.post_scan.len(), 1);
    }

    #[test]
    fn test_min_bytes_per_sec_zero_promotes_all() {
        let tracker = TrackerConfig::new().with_min_bytes_per_sec(0.0).build();

        let metadata = create_test_metadata(vec![(100, vec![1000])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr)];

        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // All filters should be promoted to row_filters when min_bytes_per_sec is 0
        assert_eq!(result.row_filters.len(), 1);
        assert_eq!(result.post_scan.len(), 0);
    }

    #[test]
    fn test_promotion_via_confidence_lower_bound() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(1000.0)
            .with_byte_ratio_threshold(0.5) // Force to PostScan initially
            .with_confidence_z(0.5) // Lower z for easier promotion
            .build();

        let metadata = create_test_metadata(vec![(100, vec![1000])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr.clone())];

        // First partition: goes to PostScan (high byte ratio)
        let result = tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.post_scan.len(), 1);
        assert_eq!(result.row_filters.len(), 0);

        // Feed high effectiveness stats
        for _ in 0..5 {
            tracker.update(1, 1, 100, 100_000, 1000); // high effectiveness
        }

        // Second partition: should be promoted to RowFilter
        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.row_filters.len(), 1);
        assert_eq!(result.post_scan.len(), 0);
    }

    #[test]
    fn test_demotion_via_confidence_upper_bound() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(10000.0)
            .with_byte_ratio_threshold(0.1) // Force to RowFilter initially
            .with_confidence_z(0.5) // Lower z for easier demotion
            .build();

        let metadata = create_test_metadata(vec![(100, vec![100])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr.clone())];

        // First partition: goes to RowFilter (low byte ratio)
        let result = tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.row_filters.len(), 1);
        assert_eq!(result.post_scan.len(), 0);

        // Feed low effectiveness stats — all rows matched, no rows
        // pruned, so caller-computed skippable_bytes is 0.
        for _ in 0..5 {
            tracker.update(1, 100, 100, 100_000, 0);
        }

        // Second partition: should be demoted to PostScan
        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.row_filters.len(), 0);
        assert_eq!(result.post_scan.len(), 1);
    }

    #[test]
    fn test_demotion_resets_stats() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(10000.0)
            .with_byte_ratio_threshold(0.1)
            .with_confidence_z(0.5)
            .build();

        let metadata = create_test_metadata(vec![(100, vec![100])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr.clone())];

        // Start as RowFilter
        tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // Add stats — no pruning, so skippable_bytes = 0
        tracker.update(1, 100, 100, 100_000, 0);
        tracker.update(1, 100, 100, 100_000, 0);

        // Demote
        tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // Stats should be zeroed after demotion
        let stats_map = tracker.filter_stats.read();
        assert_eq!(
            *stats_map.get(&1).unwrap().lock(),
            SelectivityStats::default()
        );
    }

    #[test]
    fn test_promotion_resets_stats() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(100.0)
            .with_byte_ratio_threshold(0.5)
            .with_confidence_z(0.5)
            .build();

        let metadata = create_test_metadata(vec![(100, vec![1000])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr.clone())];

        // Start as PostScan
        tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // Add stats with high prune_rate so the selectivity gate
        // (>= 0.99) lets the promotion fire.
        for _ in 0..3 {
            tracker.update(1, 1, 100, 100_000, 1000);
        }

        // Promote
        tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // Stats should be zeroed after promotion
        let stats_map = tracker.filter_stats.read();
        assert_eq!(
            *stats_map.get(&1).unwrap().lock(),
            SelectivityStats::default()
        );
    }

    #[test]
    fn test_optional_filter_dropping() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(10000.0)
            .with_byte_ratio_threshold(0.5)
            .with_confidence_z(0.5)
            .build();

        let metadata = create_test_metadata(vec![(100, vec![1000])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr.clone())];

        // Start as PostScan
        tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // Feed poor effectiveness stats — no pruning, no skippable_bytes
        for _ in 0..5 {
            tracker.update(1, 100, 100, 100_000, 0);
        }

        // Next partition: should stay as PostScan (not dropped because not optional)
        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.post_scan.len(), 1);
        assert_eq!(result.row_filters.len(), 0);
    }

    #[test]
    fn test_persistent_dropped_state() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(10000.0)
            .with_byte_ratio_threshold(0.5)
            .build();

        let metadata = create_test_metadata(vec![(100, vec![1000])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr.clone())];

        // Mark filter as dropped by manually setting state
        tracker
            .inner
            .lock()
            .filter_states
            .insert(1, FilterState::Dropped);

        // On next partition, dropped filters should not reappear
        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.row_filters.len(), 0);
        assert_eq!(result.post_scan.len(), 0);
    }
}

mod filter_ordering_tests {
    use super::helper_functions::*;
    use super::*;

    #[test]
    fn test_filters_get_partitioned() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(1.0) // Very low threshold
            .build();

        let metadata = create_test_metadata(vec![(100, vec![100, 100, 100])]);
        let filters = vec![
            (1, col_expr("a", 0)),
            (2, col_expr("a", 1)),
            (3, col_expr("a", 2)),
        ];

        // Partition should process all filters
        let result = tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // With min_bytes_per_sec=1.0, filters should be partitioned
        assert!(result.row_filters.len() + result.post_scan.len() > 0);

        // Add stats and partition again
        tracker.update(1, 60, 100, 1_000_000, 100);
        tracker.update(2, 1, 100, 1_000_000, 100);
        tracker.update(3, 40, 100, 1_000_000, 100);

        let result2 = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // Filters should still be partitioned
        assert!(result2.row_filters.len() + result2.post_scan.len() > 0);
    }

    #[test]
    fn test_filters_processed_without_stats() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(1.0) // Very low threshold
            .build();

        // Different column sizes: 300, 200, 100 bytes
        let metadata = create_test_metadata(vec![(100, vec![300, 200, 100])]);
        let filters = vec![
            (1, col_expr("a", 0)),
            (2, col_expr("a", 1)),
            (3, col_expr("a", 2)),
        ];

        // First partition - no stats yet
        let result = tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // All filters should be processed (partitioned into row/post-scan)
        assert!(result.row_filters.len() + result.post_scan.len() > 0);

        // Filters should be consistent on repeated calls
        let result2 = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(
            result.row_filters.len() + result.post_scan.len(),
            result2.row_filters.len() + result2.post_scan.len()
        );
    }

    #[test]
    fn test_filters_with_partial_stats() {
        let tracker = TrackerConfig::new().with_min_bytes_per_sec(1.0).build();

        // Give filter 2 larger bytes so it's prioritized when falling back to byte ratio
        let metadata = create_test_metadata(vec![(100, vec![100, 300, 100])]);
        let filters = vec![
            (1, col_expr("a", 0)),
            (2, col_expr("a", 1)),
            (3, col_expr("a", 2)),
        ];

        // First partition
        let result1 = tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert!(result1.row_filters.len() + result1.post_scan.len() > 0);

        // Only add stats for filters 1 and 3, not 2
        tracker.update(1, 60, 100, 1_000_000, 100);
        tracker.update(3, 60, 100, 1_000_000, 100);

        // Second partition with partial stats
        let result2 = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert!(result2.row_filters.len() + result2.post_scan.len() > 0);
    }

    #[test]
    fn test_ordering_stability_with_identical_values() {
        let tracker = TrackerConfig::new().with_min_bytes_per_sec(0.0).build();

        let metadata = create_test_metadata(vec![(100, vec![100, 100, 100])]);
        let filters = vec![
            (1, col_expr("a", 0)),
            (2, col_expr("a", 1)),
            (3, col_expr("a", 2)),
        ];

        let result1 = tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        let result2 = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // Without stats and with identical byte sizes, order should be stable
        assert_eq!(result1.row_filters[0].0, result2.row_filters[0].0);
        assert_eq!(result1.row_filters[1].0, result2.row_filters[1].0);
        assert_eq!(result1.row_filters[2].0, result2.row_filters[2].0);
    }
}

mod dynamic_filter_tests {
    use super::helper_functions::*;
    use super::*;

    #[test]
    fn test_generation_zero_ignored() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(1000.0)
            .with_byte_ratio_threshold(0.5)
            .build();

        let metadata = create_test_metadata(vec![(100, vec![1000])]);

        // Create two filters with same ID but generation 0 and 1
        // Generation 0 should be ignored
        let expr1 = col_expr("a", 0);
        let filters1 = vec![(1, expr1)];

        tracker.partition_filters(
            filters1,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        tracker.update(1, 50, 100, 100_000, 1000);

        // Generation 0 doesn't trigger state reset
        let snapshot_gen = tracker.inner.lock().snapshot_generations.get(&1).copied();
        assert_eq!(snapshot_gen, None);
    }

    #[test]
    fn test_generation_change_clears_stats() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(1000.0)
            .with_byte_ratio_threshold(0.5)
            .build();

        // Pre-populate stats entry so update() can find it
        tracker.ensure_stats_entry(1);

        // Initialize generation to 100
        {
            let mut inner = tracker.inner.lock();
            let stats = tracker.filter_stats.read();
            inner.note_generation(1, 100, &stats);
        }

        // Add stats
        tracker.update(1, 50, 100, 100_000, 1000);
        tracker.update(1, 50, 100, 100_000, 1000);

        let stats_before = {
            let stats_map = tracker.filter_stats.read();
            *stats_map.get(&1).unwrap().lock() != SelectivityStats::default()
        };
        assert!(stats_before);

        // Simulate generation change to a different value
        {
            let mut inner = tracker.inner.lock();
            let stats = tracker.filter_stats.read();
            inner.note_generation(1, 101, &stats);
        }

        // Stats should be zeroed on generation change
        let stats_after = {
            let stats_map = tracker.filter_stats.read();
            *stats_map.get(&1).unwrap().lock() == SelectivityStats::default()
        };
        assert!(stats_after);
    }

    #[test]
    fn test_generation_unchanged_preserves_stats() {
        let tracker = TrackerConfig::new().with_min_bytes_per_sec(1000.0).build();

        // Pre-populate stats entry so update() can find it
        tracker.ensure_stats_entry(1);

        // Manually set generation
        {
            let mut inner = tracker.inner.lock();
            let stats = tracker.filter_stats.read();
            inner.note_generation(1, 100, &stats);
        }

        // Add stats
        tracker.update(1, 50, 100, 100_000, 1000);
        tracker.update(1, 50, 100, 100_000, 1000);

        let sample_count_before = {
            let stats_map = tracker.filter_stats.read();
            stats_map.get(&1).map(|s| s.lock().sample_count)
        };
        assert_eq!(sample_count_before, Some(2));

        // Call note_generation with same generation
        {
            let mut inner = tracker.inner.lock();
            let stats = tracker.filter_stats.read();
            inner.note_generation(1, 100, &stats);
        }

        // Stats should be preserved
        let sample_count_after = {
            let stats_map = tracker.filter_stats.read();
            stats_map.get(&1).map(|s| s.lock().sample_count)
        };
        assert_eq!(sample_count_after, Some(2));
    }

    #[test]
    fn test_generation_change_preserves_state() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(1000.0)
            .with_byte_ratio_threshold(0.1)
            .build();

        let metadata = create_test_metadata(vec![(100, vec![100])]);

        // First partition: goes to RowFilter
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr)];
        tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        let state_before = tracker.inner.lock().filter_states.get(&1).copied();
        assert_eq!(state_before, Some(FilterState::RowFilter));

        // Simulate generation change
        {
            let mut inner = tracker.inner.lock();
            let stats = tracker.filter_stats.read();
            inner.note_generation(1, 100, &stats);
        }

        // State should be preserved despite stats being cleared
        let state_after = tracker.inner.lock().filter_states.get(&1).copied();
        assert_eq!(state_after, Some(FilterState::RowFilter));
    }

    #[test]
    fn test_generation_change_undrops_dropped_filter() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(1000.0)
            .with_byte_ratio_threshold(0.1)
            .build();

        // Manually set filter state to Dropped
        tracker
            .inner
            .lock()
            .filter_states
            .insert(1, FilterState::Dropped);
        {
            let mut inner = tracker.inner.lock();
            let stats = tracker.filter_stats.read();
            inner.note_generation(1, 100, &stats);
        }

        // Simulate generation change
        {
            let mut inner = tracker.inner.lock();
            let stats = tracker.filter_stats.read();
            inner.note_generation(1, 101, &stats);
        }

        // Dropped filter should be un-dropped to PostScan
        let state_after = tracker.inner.lock().filter_states.get(&1).copied();
        assert_eq!(state_after, Some(FilterState::PostScan));
    }

    #[test]
    fn test_multiple_filters_independent_generation_tracking() {
        let tracker = TrackerConfig::new().with_min_bytes_per_sec(1000.0).build();

        // Pre-populate stats entries so update() can find them
        tracker.ensure_stats_entry(1);
        tracker.ensure_stats_entry(2);

        // Set generations for multiple filters
        {
            let mut inner = tracker.inner.lock();
            let stats = tracker.filter_stats.read();
            inner.note_generation(1, 100, &stats);
            inner.note_generation(2, 200, &stats);
        }

        // Add stats to both
        tracker.update(1, 50, 100, 100_000, 1000);
        tracker.update(2, 50, 100, 100_000, 1000);

        // Change generation of filter 1 only
        {
            let mut inner = tracker.inner.lock();
            let stats = tracker.filter_stats.read();
            inner.note_generation(1, 101, &stats);
        }

        // Filter 1 stats should be zeroed, filter 2 preserved
        let stats_map = tracker.filter_stats.read();
        assert_eq!(
            *stats_map.get(&1).unwrap().lock(),
            SelectivityStats::default()
        );
        assert_ne!(
            *stats_map.get(&2).unwrap().lock(),
            SelectivityStats::default()
        );
    }
}

mod integration_tests {
    use super::helper_functions::*;
    use super::*;

    #[test]
    fn test_full_promotion_lifecycle() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(500.0)
            .with_byte_ratio_threshold(0.5) // Force initial PostScan
            .with_confidence_z(0.5)
            .build();

        let metadata = create_test_metadata(vec![(100, vec![1000])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr.clone())];

        // Step 1: Initial placement (PostScan)
        let result = tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.post_scan.len(), 1);
        assert_eq!(result.row_filters.len(), 0);

        // Step 2: Accumulate high effectiveness stats
        for _ in 0..5 {
            tracker.update(1, 1, 100, 100_000, 1000); // high effectiveness
        }

        // Step 3: Promotion should occur
        let result = tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.row_filters.len(), 1);
        assert_eq!(result.post_scan.len(), 0);

        // Step 4: Continue to partition without additional updates
        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.row_filters.len(), 1);
        assert_eq!(result.post_scan.len(), 0);
    }

    #[test]
    fn test_full_demotion_lifecycle() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(10000.0)
            .with_byte_ratio_threshold(0.1) // Force initial RowFilter
            .with_confidence_z(0.5)
            .build();

        let metadata = create_test_metadata(vec![(100, vec![100])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr.clone())];

        // Step 1: Initial placement (RowFilter)
        let result = tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.row_filters.len(), 1);
        assert_eq!(result.post_scan.len(), 0);

        // Step 2: Accumulate low effectiveness stats — no pruning,
        // so skippable_bytes = 0
        for _ in 0..5 {
            tracker.update(1, 100, 100, 100_000, 0);
        }

        // Step 3: Demotion should occur
        let result = tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.row_filters.len(), 0);
        assert_eq!(result.post_scan.len(), 1);

        // Step 4: Continue to partition without additional updates
        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.row_filters.len(), 0);
        assert_eq!(result.post_scan.len(), 1);
    }

    #[test]
    fn test_multiple_filters_mixed_states() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(1000.0)
            .with_byte_ratio_threshold(0.4) // Force PostScan initially (500/1000=0.5 > 0.4)
            .with_confidence_z(0.5)
            .build();

        let metadata = create_test_metadata(vec![(100, vec![500, 500])]);
        let filters = vec![(1, col_expr("a", 0)), (2, col_expr("a", 1))];

        // Initial partition: both go to PostScan (500/1000 = 0.5 > 0.4)
        let result = tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.post_scan.len(), 2);

        // Filter 1: high effectiveness — 99/100 rows pruned out of
        // 500 batch bytes ≈ 495 skippable bytes
        for _ in 0..3 {
            tracker.update(1, 1, 100, 100_000, 495);
        }

        // Filter 2: low effectiveness — no rows pruned, so 0 skippable
        for _ in 0..3 {
            tracker.update(2, 100, 100, 100_000, 0);
        }

        // Next partition: Filter 1 promoted, Filter 2 stays PostScan
        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.row_filters.len(), 1);
        assert_eq!(result.post_scan.len(), 1);
        assert_eq!(result.row_filters[0].0, 1);
        assert_eq!(result.post_scan[0].0, 2);
    }

    #[test]
    fn test_empty_filter_list() {
        let tracker = TrackerConfig::new().build();
        let metadata = create_test_metadata(vec![(100, vec![1000])]);
        let filters = vec![];

        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        assert_eq!(result.row_filters.len(), 0);
        assert_eq!(result.post_scan.len(), 0);
    }

    #[test]
    fn test_single_filter() {
        let tracker = TrackerConfig::new().with_min_bytes_per_sec(0.0).build();

        let metadata = create_test_metadata(vec![(100, vec![100])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr)];

        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        assert_eq!(result.row_filters.len(), 1);
        assert_eq!(result.post_scan.len(), 0);
    }

    #[test]
    fn test_zero_effectiveness_stays_at_boundary() {
        let tracker = TrackerConfig::new()
            .with_min_bytes_per_sec(100.0)
            .with_byte_ratio_threshold(0.1)
            .with_confidence_z(0.5)
            .build();

        let metadata = create_test_metadata(vec![(100, vec![100])]);
        let expr = col_expr("a", 0);
        let filters = vec![(1, expr.clone())];

        // Start as RowFilter
        tracker.partition_filters(
            filters.clone(),
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );

        // All rows match (zero effectiveness) — no rows pruned, so
        // skippable_bytes = 0
        for _ in 0..5 {
            tracker.update(1, 100, 100, 100_000, 0);
        }

        // Should demote due to CI upper bound being 0
        let result = tracker.partition_filters(
            filters,
            &std::collections::HashSet::new(),
            1000,
            &metadata,
        );
        assert_eq!(result.row_filters.len(), 0);
        assert_eq!(result.post_scan.len(), 1);
    }

    #[test]
    fn test_confidence_z_parameter_stored() {
        // Test that different confidence_z values are properly stored in config
        let tracker_conservative = TrackerConfig::new()
            .with_min_bytes_per_sec(1000.0)
            .with_byte_ratio_threshold(0.5)
            .with_confidence_z(3.0) // Harder to promote
            .build();

        let tracker_aggressive = TrackerConfig::new()
            .with_min_bytes_per_sec(1000.0)
            .with_byte_ratio_threshold(0.5)
            .with_confidence_z(0.5) // Easier to promote
            .build();

        // Verify configs are stored correctly
        assert_eq!(tracker_conservative.config.confidence_z, 3.0);
        assert_eq!(tracker_aggressive.config.confidence_z, 0.5);

        // The z-score affects confidence intervals during promotion/demotion decisions.
        // With identical stats, higher z requires narrower confidence intervals,
        // making promotion harder. With lower z, confidence intervals are wider,
        // making promotion easier. This is tested in other integration tests
        // that verify actual promotion/demotion behavior.
    }
}
