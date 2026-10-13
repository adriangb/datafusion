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

//! [`InlineAutomaticCtes`] inlines the automatic materialized CTEs for which
//! materialization is not expected to help.

use datafusion_common::Result;
use datafusion_common::tree_node::{Transformed, TreeNodeRecursion};
use datafusion_expr::{LogicalPlan, MaterializedCte};

use crate::{OptimizerConfig, OptimizerRule};

/// Decides, for each automatic [`MaterializedCte`], if the CTE stays
/// materialized or is inlined at each reference.
///
/// With `datafusion.optimizer.enable_automatic_cte_materialization`, the SQL
/// planner creates an automatic [`MaterializedCte`] for each CTE that has no
/// `MATERIALIZED` or `NOT MATERIALIZED` directive. Materialization computes
/// the body one time, but it buffers the full output of the body, and it stops
/// the pushdown of the filters and projections of each reference into the
/// body. This rule replaces the node with [`MaterializedCte::inline`] when one
/// of these conditions is true:
///
/// 1. The CTE has zero or one reference. A scan in a subquery expression or
///    in the body of another CTE is a reference.
/// 2. The body has a volatile expression. Each reference of an inlined CTE
///    evaluates such an expression again, and materialization would change
///    the result of the query. An automatic decision must not change results.
/// 3. The body has an outer reference. Such a body cannot be computed one
///    time.
/// 4. The body is cheap: it has only projections, filters and aliases on one
///    table scan, materialized CTE scan, `VALUES` list or empty relation, and
///    it has no subquery expression. To compute such a body again costs
///    about the same as to read it from the buffer, and the inlined copies
///    get filter and projection pushdown.
///
/// In all other cases the CTE stays materialized. The rule does not use
/// statistics or a cost model.
///
/// A node from `AS MATERIALIZED` is not automatic and is never changed.
///
/// The rule decides the innermost CTE first. A later CTE of a `WITH` clause is
/// nested in the continuation of an earlier one, so when a later CTE that
/// reads an earlier CTE is inlined, the earlier CTE is decided with the new
/// number of references.
#[derive(Default, Debug)]
pub struct InlineAutomaticCtes;

impl InlineAutomaticCtes {
    #[expect(missing_docs)]
    pub fn new() -> Self {
        Self {}
    }
}

impl OptimizerRule for InlineAutomaticCtes {
    fn name(&self) -> &str {
        "inline_automatic_ctes"
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Most plans have no automatic CTE. Do not rebuild them.
        if !has_automatic_cte(&plan)? {
            return Ok(Transformed::no(plan));
        }
        plan.transform_up_with_subqueries(|node| match node {
            LogicalPlan::MaterializedCte(cte)
                if cte.is_automatic() && should_inline(&cte)? =>
            {
                Ok(Transformed::yes(cte.inline()?))
            }
            _ => Ok(Transformed::no(node)),
        })
    }
}

fn has_automatic_cte(plan: &LogicalPlan) -> Result<bool> {
    let mut found = false;
    plan.apply_with_subqueries(|node| {
        if matches!(node, LogicalPlan::MaterializedCte(cte) if cte.is_automatic()) {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

/// See the conditions on [`InlineAutomaticCtes`].
fn should_inline(cte: &MaterializedCte) -> Result<bool> {
    let body = cte.body();
    Ok(cte.scans()?.len() <= 1
        || is_volatile(body)?
        || !body.all_out_ref_exprs().is_empty()
        || is_cheap(body))
}

/// Returns `true` if `plan` has a volatile expression, also in a subquery.
fn is_volatile(plan: &LogicalPlan) -> Result<bool> {
    let mut volatile = false;
    plan.apply_with_subqueries(|node| {
        node.apply_expressions(|expr| {
            if expr.is_volatile() {
                volatile = true;
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        })
    })?;
    Ok(volatile)
}

/// Returns `true` if `plan` is only projections, filters and aliases, without
/// subquery expressions, on one leaf that reads rows which already exist.
fn is_cheap(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Projection(_)
        | LogicalPlan::Filter(_)
        | LogicalPlan::SubqueryAlias(_) => {
            !has_subquery_expr(plan) && is_cheap(plan.inputs()[0])
        }
        LogicalPlan::TableScan(_)
        | LogicalPlan::MaterializedCteScan(_)
        | LogicalPlan::Values(_)
        | LogicalPlan::EmptyRelation(_) => !has_subquery_expr(plan),
        _ => false,
    }
}

fn has_subquery_expr(plan: &LogicalPlan) -> bool {
    let mut found = false;
    plan.apply_subqueries(|_| {
        found = true;
        Ok(TreeNodeRecursion::Stop)
    })
    // The closure always returns Ok
    .unwrap();
    found
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::assert_optimized_plan_eq_snapshot;
    use crate::test::test_table_scan;
    use crate::test::udfs::PlacementTestUDF;
    use crate::{OptimizerContext, OptimizerRule};
    use datafusion_expr::logical_plan::builder::LogicalPlanBuilder;
    use datafusion_expr::{
        JoinType, MaterializedCteId, MaterializedCteScan, ScalarUDF, Volatility, col,
        in_subquery, lit,
    };
    use datafusion_functions_aggregate::expr_fn::sum;

    macro_rules! assert_optimized_plan_equal {
        ($plan:expr, @ $expected:literal $(,)?) => {{
            let optimizer_ctx = OptimizerContext::new().with_max_passes(1);
            let rules: Vec<Arc<dyn OptimizerRule + Send + Sync>> =
                vec![Arc::new(InlineAutomaticCtes::new())];
            assert_optimized_plan_eq_snapshot!(optimizer_ctx, rules, $plan, @ $expected,)
        }};
    }

    /// A CTE named `name` with the body `body`. `continuation` builds the
    /// query from a function that returns a scan of the CTE with an alias.
    fn cte(
        name: &str,
        body: LogicalPlan,
        automatic: bool,
        continuation: impl FnOnce(&dyn Fn(&str) -> Result<LogicalPlan>) -> Result<LogicalPlan>,
    ) -> Result<LogicalPlan> {
        let body = LogicalPlanBuilder::from(body).alias(name)?.build()?;
        let id = MaterializedCteId::next();
        let schema = Arc::clone(body.schema());
        let scan = |alias: &str| {
            LogicalPlanBuilder::from(LogicalPlan::MaterializedCteScan(
                MaterializedCteScan::new(id, name, Arc::clone(&schema)),
            ))
            .alias(alias)?
            .build()
        };
        let continuation = continuation(&scan)?;
        Ok(LogicalPlan::MaterializedCte(
            MaterializedCte::try_new(id, name, Arc::new(body), Arc::new(continuation))?
                .with_automatic(automatic),
        ))
    }

    /// `SELECT a, sum(b) AS s FROM test GROUP BY a`
    fn aggregate_body() -> Result<LogicalPlan> {
        LogicalPlanBuilder::from(test_table_scan()?)
            .aggregate(vec![col("a")], vec![sum(col("b")).alias("s")])?
            .build()
    }

    /// `SELECT a, b AS s FROM test WHERE c > 1`
    fn cheap_body() -> Result<LogicalPlan> {
        LogicalPlanBuilder::from(test_table_scan()?)
            .filter(col("c").gt(lit(1u32)))?
            .project(vec![col("a"), col("b").alias("s")])?
            .build()
    }

    /// `SELECT x.a FROM <scan> x JOIN <scan> y ON x.s = y.s`
    fn self_join(scan: &dyn Fn(&str) -> Result<LogicalPlan>) -> Result<LogicalPlan> {
        LogicalPlanBuilder::from(scan("x")?)
            .join(
                scan("y")?,
                JoinType::Inner,
                (vec!["x.s"], vec!["y.s"]),
                None,
            )?
            .project(vec![col("x.a")])?
            .build()
    }

    #[test]
    fn keeps_two_references_to_an_aggregate() -> Result<()> {
        let plan = cte("c", aggregate_body()?, true, self_join)?;
        assert_optimized_plan_equal!(plan, @"
        MaterializedCte: name=c, automatic=true
          SubqueryAlias: c
            Aggregate: groupBy=[[test.a]], aggr=[[sum(test.b) AS s]]
              TableScan: test
          Projection: x.a
            Inner Join: x.s = y.s
              SubqueryAlias: x
                MaterializedCteScan: name=c
              SubqueryAlias: y
                MaterializedCteScan: name=c
        ")
    }

    #[test]
    fn inlines_one_reference() -> Result<()> {
        let plan = cte("c", aggregate_body()?, true, |scan| {
            LogicalPlanBuilder::from(scan("x")?)
                .filter(col("x.a").eq(lit(1u32)))?
                .build()
        })?;
        assert_optimized_plan_equal!(plan, @"
        Filter: x.a = UInt32(1)
          SubqueryAlias: x
            SubqueryAlias: c
              Aggregate: groupBy=[[test.a]], aggr=[[sum(test.b) AS s]]
                TableScan: test
        ")
    }

    #[test]
    fn removes_a_cte_without_reference() -> Result<()> {
        let plan = cte("c", aggregate_body()?, true, |_| test_table_scan())?;
        assert_optimized_plan_equal!(plan, @"TableScan: test")
    }

    #[test]
    fn inlines_a_cheap_body() -> Result<()> {
        let plan = cte("c", cheap_body()?, true, self_join)?;
        assert_optimized_plan_equal!(plan, @"
        Projection: x.a
          Inner Join: x.s = y.s
            SubqueryAlias: x
              SubqueryAlias: c
                Projection: test.a, test.b AS s
                  Filter: test.c > UInt32(1)
                    TableScan: test
            SubqueryAlias: y
              SubqueryAlias: c
                Projection: test.a, test.b AS s
                  Filter: test.c > UInt32(1)
                    TableScan: test
        ")
    }

    #[test]
    fn inlines_a_volatile_body() -> Result<()> {
        let volatile = ScalarUDF::new_from_impl(
            PlacementTestUDF::new().with_volatility(Volatility::Volatile),
        );
        let body = LogicalPlanBuilder::from(aggregate_body()?)
            .project(vec![col("a"), volatile.call(vec![col("s")]).alias("s")])?
            .build()?;
        let plan = cte("c", body, true, self_join)?;
        assert_optimized_plan_equal!(plan, @"
        Projection: x.a
          Inner Join: x.s = y.s
            SubqueryAlias: x
              SubqueryAlias: c
                Projection: test.a, leaf_udf(s) AS s
                  Aggregate: groupBy=[[test.a]], aggr=[[sum(test.b) AS s]]
                    TableScan: test
            SubqueryAlias: y
              SubqueryAlias: c
                Projection: test.a, leaf_udf(s) AS s
                  Aggregate: groupBy=[[test.a]], aggr=[[sum(test.b) AS s]]
                    TableScan: test
        ")
    }

    #[test]
    fn never_inlines_an_explicit_cte() -> Result<()> {
        // One reference and a cheap body, but the user asked for the CTE.
        let plan = cte("c", cheap_body()?, false, |scan| scan("x"))?;
        assert_optimized_plan_equal!(plan, @"
        MaterializedCte: name=c
          SubqueryAlias: c
            Projection: test.a, test.b AS s
              Filter: test.c > UInt32(1)
                TableScan: test
          SubqueryAlias: x
            MaterializedCteScan: name=c
        ")
    }

    #[test]
    fn counts_a_scan_in_a_subquery_expression() -> Result<()> {
        let plan = cte("c", aggregate_body()?, true, |scan| {
            let subquery = LogicalPlanBuilder::from(scan("y")?)
                .project(vec![col("y.s")])?
                .build()?;
            LogicalPlanBuilder::from(scan("x")?)
                .filter(in_subquery(col("x.s"), Arc::new(subquery)))?
                .build()
        })?;
        assert_optimized_plan_equal!(plan, @"
        MaterializedCte: name=c, automatic=true
          SubqueryAlias: c
            Aggregate: groupBy=[[test.a]], aggr=[[sum(test.b) AS s]]
              TableScan: test
          Filter: x.s IN (<subquery>)
            Subquery:
              Projection: y.s
                SubqueryAlias: y
                  MaterializedCteScan: name=c
            SubqueryAlias: x
              MaterializedCteScan: name=c
        ")
    }

    #[test]
    fn counts_the_references_again_after_another_cte_is_inlined() -> Result<()> {
        // `c` has one reference, in the body of `d`. `d` is cheap and has
        // two references, so it is inlined, and then `c` has two references.
        let plan = cte("c", aggregate_body()?, true, |scan_c| {
            let body_d = LogicalPlanBuilder::from(scan_c("c")?)
                .filter(col("c.s").gt(lit(1u64)))?
                .build()?;
            cte("d", body_d, true, self_join)
        })?;
        assert_optimized_plan_equal!(plan, @"
        MaterializedCte: name=c, automatic=true
          SubqueryAlias: c
            Aggregate: groupBy=[[test.a]], aggr=[[sum(test.b) AS s]]
              TableScan: test
          Projection: x.a
            Inner Join: x.s = y.s
              SubqueryAlias: x
                SubqueryAlias: d
                  Filter: c.s > UInt64(1)
                    SubqueryAlias: c
                      MaterializedCteScan: name=c
              SubqueryAlias: y
                SubqueryAlias: d
                  Filter: c.s > UInt64(1)
                    SubqueryAlias: c
                      MaterializedCteScan: name=c
        ")
    }

    #[test]
    fn inlines_a_cte_into_the_body_of_another_cte() -> Result<()> {
        // `c` has one reference, in the body of `d`, which stays materialized.
        let plan = cte("c", aggregate_body()?, true, |scan_c| {
            let body_d = LogicalPlanBuilder::from(scan_c("c")?)
                .aggregate(vec![col("c.s")], vec![sum(col("c.a")).alias("a")])?
                .build()?;
            cte("d", body_d, true, self_join)
        })?;
        assert_optimized_plan_equal!(plan, @"
        MaterializedCte: name=d, automatic=true
          SubqueryAlias: d
            Aggregate: groupBy=[[c.s]], aggr=[[sum(c.a) AS a]]
              SubqueryAlias: c
                SubqueryAlias: c
                  Aggregate: groupBy=[[test.a]], aggr=[[sum(test.b) AS s]]
                    TableScan: test
          Projection: x.a
            Inner Join: x.s = y.s
              SubqueryAlias: x
                MaterializedCteScan: name=d
              SubqueryAlias: y
                MaterializedCteScan: name=d
        ")
    }
}
