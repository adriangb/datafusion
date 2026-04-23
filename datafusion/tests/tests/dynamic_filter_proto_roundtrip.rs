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

//! End-to-end tests asserting that `datafusion-proto` round-trip preserves
//! the identity + shared mutable state of `DynamicFilterPhysicalExpr`
//! instances attached to operators that produce them (`HashJoinExec` and
//! `SortExec`'s TopK).
//!
//! Each test plans `EXPLAIN ANALYZE <query>` via SQL, round-trips the
//! resulting physical plan through proto, executes the deserialized plan,
//! and snapshots the `EXPLAIN ANALYZE` output text with insta. With
//! `datafusion.explain.analyze_categories = 'rows'`, the text contains only
//! deterministic row-count metrics — no timing — so the snapshot captures
//! plan shape, dynamic-filter state after execution, and scan-level pruning
//! all in one.
//!
//! These tests live in a standalone crate (`datafusion-tests`) because they
//! need dev-dependencies on both `datafusion` and `datafusion-proto`; putting
//! them in either of those crates' own `tests/` directory would close a
//! dev-dependency cycle caught by the workspace's circular-dependency check.

use std::sync::Arc;

use arrow::util::pretty::pretty_format_batches;
use datafusion::physical_plan::collect;
use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
use datafusion_proto::bytes::{
    physical_plan_from_bytes_with_proto_converter,
    physical_plan_to_bytes_with_proto_converter,
};
use datafusion_proto::physical_plan::{
    DeduplicatingProtoConverter, DefaultPhysicalExtensionCodec,
};

/// Execute a SQL statement for its side effect (table creation, `COPY`,
/// etc.), discarding any result batches.
async fn exec(ctx: &SessionContext, sql: &str) -> datafusion_common::Result<()> {
    ctx.sql(sql).await?.collect().await?;
    Ok(())
}

/// Round-trip an EXPLAIN ANALYZE plan through proto, execute it, and return
/// the rendered text.
async fn roundtrip_and_explain_analyze(
    ctx: &SessionContext,
    sql: &str,
) -> datafusion_common::Result<String> {
    let explain_sql = format!("EXPLAIN ANALYZE {sql}");
    let plan = ctx.sql(&explain_sql).await?.create_physical_plan().await?;

    let codec = DefaultPhysicalExtensionCodec {};
    let converter = DeduplicatingProtoConverter {};
    let bytes = physical_plan_to_bytes_with_proto_converter(
        Arc::clone(&plan),
        &codec,
        &converter,
    )?;
    let result_plan = physical_plan_from_bytes_with_proto_converter(
        bytes.as_ref(),
        ctx.task_ctx().as_ref(),
        &codec,
        &converter,
    )?;

    let batches = collect(result_plan, ctx.task_ctx()).await?;
    Ok(pretty_format_batches(&batches)?.to_string())
}

/// Filters applied to insta snapshots to strip non-deterministic text
/// (tempdir paths, workspace-absolute paths).
fn settings() -> insta::Settings {
    let mut settings = insta::Settings::clone_current();
    // Replace absolute paths ending in `foo.parquet` with just `foo.parquet`
    // so snapshots are stable across machines and across tempdir runs.
    settings.add_filter(r"[^\s\[\]]*/([A-Za-z0-9_\-]+\.parquet)", "$1");
    settings
}

/// End-to-end: a SQL hash join with a selective WHERE on the build side
/// produces a dynamic filter that's pushed into the probe-side
/// `ParquetSource`. After proto round-trip, the `HashJoinExec`'s dynamic
/// filter Arc and the pushed predicate still share mutable state, so
/// build-side `update()` during execution is visible to the scan and the
/// scan prunes rows. The snapshot shows the plan shape + the dynamic filter
/// expression after execution + `output_rows` on every operator.
#[tokio::test]
async fn hash_join_dynamic_filter_prunes_via_sql() -> datafusion_common::Result<()> {
    let config = SessionConfig::new()
        .set_bool("datafusion.execution.parquet.pushdown_filters", true)
        .set_str("datafusion.explain.analyze_categories", "rows");
    let ctx = SessionContext::new_with_config(config);
    let parquet_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../core/tests/data/tpch_nation_small.parquet"
    );
    ctx.register_parquet("build", parquet_path, ParquetReadOptions::default())
        .await?;
    ctx.register_parquet("probe", parquet_path, ParquetReadOptions::default())
        .await?;

    // Self-join with a selective WHERE on the build side. The build side
    // reduces to a single row; the dynamic filter derived from it is pushed
    // into the probe-side scan.
    let sql = "SELECT p.n_name FROM probe p \
               INNER JOIN build b ON p.n_nationkey = b.n_nationkey \
               WHERE b.n_nationkey = 5";
    let output = roundtrip_and_explain_analyze(&ctx, sql).await?;

    settings().bind(|| insta::assert_snapshot!(output, @"
    +-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | plan_type         | plan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
    +-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Plan with Metrics | HashJoinExec: mode=CollectLeft, join_type=Inner, on=[(n_nationkey@0, n_nationkey@0)], projection=[n_name@1], metrics=[output_rows=1, output_batches=1, array_map_created_count=1, build_input_batches=1, build_input_rows=1, input_batches=1, input_rows=1, avg_fanout=100% (1/1), probe_hit_rate=100% (1/1)]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
    |                   |   DataSourceExec: file_groups={1 group: [[tpch_nation_small.parquet]]}, projection=[n_nationkey, n_name], file_type=parquet, predicate=n_nationkey@0 = 5, pruning_predicate=n_nationkey_null_count@2 != row_count@3 AND n_nationkey_min@0 <= 5 AND 5 <= n_nationkey_max@1, required_guarantees=[n_nationkey in (5)], metrics=[output_rows=1, output_batches=1, files_ranges_pruned_statistics=1 total → 1 matched, row_groups_pruned_statistics=1 total → 1 matched, row_groups_pruned_bloom_filter=1 total → 1 matched, page_index_pages_pruned=0 total → 0 matched, page_index_rows_pruned=0 total → 0 matched, limit_pruned_row_groups=0 total → 0 matched, batches_split=0, file_open_errors=0, file_scan_errors=0, files_opened=1, files_processed=1, num_predicate_creation_errors=0, predicate_evaluation_errors=0, pushdown_rows_matched=1, pushdown_rows_pruned=19, predicate_cache_inner_records=20, predicate_cache_records=1, scan_efficiency_ratio=10.56% (461/4.36 K)]                                                                                                                                                                                                                                                                                                                                           |
    |                   |   RepartitionExec: partitioning=RoundRobinBatch(12), input_partitions=1, metrics=[output_rows=1, output_batches=1, spill_count=0, spilled_rows=0]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
    |                   |     DataSourceExec: file_groups={1 group: [[tpch_nation_small.parquet]]}, projection=[n_nationkey], file_type=parquet, predicate=n_nationkey@0 = 5 AND DynamicFilter [ n_nationkey@0 >= 5 AND n_nationkey@0 <= 5 AND n_nationkey@0 IN (SET) ([5]) ], pruning_predicate=n_nationkey_null_count@2 != row_count@3 AND n_nationkey_min@0 <= 5 AND 5 <= n_nationkey_max@1 AND n_nationkey_null_count@2 != row_count@3 AND n_nationkey_max@1 >= 5 AND n_nationkey_null_count@2 != row_count@3 AND n_nationkey_min@0 <= 5 AND n_nationkey_null_count@2 != row_count@3 AND n_nationkey_min@0 <= 5 AND 5 <= n_nationkey_max@1, required_guarantees=[n_nationkey in (5)], metrics=[output_rows=1, output_batches=1, files_ranges_pruned_statistics=1 total → 1 matched, row_groups_pruned_statistics=1 total → 1 matched, row_groups_pruned_bloom_filter=1 total → 1 matched, page_index_pages_pruned=0 total → 0 matched, page_index_rows_pruned=0 total → 0 matched, limit_pruned_row_groups=0 total → 0 matched, batches_split=0, file_open_errors=0, file_scan_errors=0, files_opened=1, files_processed=1, num_predicate_creation_errors=0, predicate_evaluation_errors=0, pushdown_rows_matched=1, pushdown_rows_pruned=19, predicate_cache_inner_records=20, predicate_cache_records=2, scan_efficiency_ratio=4.42% (193/4.36 K)] |
    |                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
    +-------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    "));

    Ok(())
}

/// End-to-end: an `ORDER BY ... LIMIT 1` over two single-row parquet files
/// (`a.parquet` key=1, `b.parquet` key=2). With `target_partitions=1`, both
/// files are read sequentially; after `a.parquet` TopK's filter tightens and
/// `b.parquet` gets pruned by row-group statistics — the scan never yields
/// b's row. The snapshot captures the updated dynamic filter expression, the
/// single emitted row, and the `row_groups_pruned` metric.
#[tokio::test]
async fn topk_dynamic_filter_proto_round_trip() -> datafusion_common::Result<()> {
    let tmp = tempfile::TempDir::new()?;
    let tmp_path = tmp.path().to_str().unwrap();

    let config = SessionConfig::new()
        .set_bool("datafusion.execution.parquet.pushdown_filters", true)
        .set_str("datafusion.explain.analyze_categories", "rows")
        .with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config);

    // Write two single-row parquet files and register the directory as a
    // listing table — all via SQL.
    exec(&ctx, &format!(
        "COPY (SELECT 1 AS id, 'foo' AS name) TO '{tmp_path}/a.parquet' STORED AS PARQUET"
    ))
    .await?;
    exec(&ctx, &format!(
        "COPY (SELECT 2 AS id, 'bar' AS name) TO '{tmp_path}/b.parquet' STORED AS PARQUET"
    ))
    .await?;
    exec(&ctx, &format!(
        "CREATE EXTERNAL TABLE t (id bigint, name text) STORED AS PARQUET LOCATION '{tmp_path}'"
    ))
    .await?;

    let sql = "SELECT name FROM t ORDER BY id ASC LIMIT 1";
    let output = roundtrip_and_explain_analyze(&ctx, sql).await?;

    settings().bind(|| insta::assert_snapshot!(output, @"
    +-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | plan_type         | plan                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
    +-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Plan with Metrics | ProjectionExec: expr=[name@0 as name], metrics=[output_rows=1, output_batches=1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
    |                   |   SortExec: TopK(fetch=1), expr=[id@1 ASC NULLS LAST], preserve_partitioning=[false], filter=[id@1 < 1], metrics=[output_rows=1, output_batches=1, row_replacements=1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
    |                   |     DataSourceExec: file_groups={1 group: [[a.parquet, b.parquet]]}, projection=[name, id], file_type=parquet, predicate=DynamicFilter [ id@1 < 1 ], pruning_predicate=id_null_count@1 != row_count@2 AND id_min@0 < 1, required_guarantees=[], metrics=[output_rows=1, output_batches=1, files_ranges_pruned_statistics=2 total → 1 matched, row_groups_pruned_statistics=1 total → 1 matched, row_groups_pruned_bloom_filter=1 total → 1 matched, page_index_pages_pruned=0 total → 0 matched, page_index_rows_pruned=0 total → 0 matched, limit_pruned_row_groups=0 total → 0 matched, batches_split=0, file_open_errors=0, file_scan_errors=0, files_opened=2, files_processed=2, num_predicate_creation_errors=0, predicate_evaluation_errors=0, pushdown_rows_matched=1, pushdown_rows_pruned=0, predicate_cache_inner_records=0, predicate_cache_records=0, scan_efficiency_ratio=0% (0/742)] |
    |                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
    +-------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    "));
    Ok(())
}
