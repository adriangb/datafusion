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

//! Structures for Morsel Driven IO.
//!
//! Morsel Driven IO is a technique for parallelizing the reading of large files
//! by dividing them into smaller "morsels" that are processed independently.
//!
//! It is inspired by the paper [Morsel-Driven Parallelism: A NUMA-Aware Query
//! Evaluation Framework for the Many-Core Age](https://db.in.tum.de/~leis/papers/morsels.pdf).

use std::fmt::Debug;

use crate::PartitionedFile;
use arrow::array::RecordBatch;
use datafusion_common::Result;
use futures::Future;
use futures::future::BoxFuture;
use futures::stream::BoxStream;

/// A Morsel of work ready to resolve to a stream of [`RecordBatch`]es.
///
/// This represents a single morsel of work that is ready to be processed. It
/// has all data necessary (does not need any I/O) and is ready to be turned
/// into a stream of [`RecordBatch`]es for processing by the execution engine.
pub trait Morsel: Send + Debug {
    /// Consume this morsel and produce a stream of [`RecordBatch`]es for processing.
    ///
    /// This should not do any I/O work, such as reading from the file.
    fn into_stream(self: Box<Self>) -> BoxStream<'static, Result<RecordBatch>>;
}

/// A Morselizer takes a single [`PartitionedFile`] and creates the initial planner
/// for that file.
///
/// This is the entry point for morsel driven I/O.
pub trait Morselizer: Send + Sync + Debug {
    /// Return the initial [`MorselPlanner`] for this file.
    ///
    /// "Morselzing" a file may involve CPU work, such as parsing parquet
    /// metadata and evaluating pruning predicates. It should NOT do any I/O
    /// work, such as reading from the file. If I/O is required, it should
    /// return a future that the caller can poll to drive the I/O work to
    /// completion, and once the future is complete, the caller can call
    /// `plan_file` again for a different file.
    fn plan_file(&self, file: PartitionedFile) -> Result<Box<dyn MorselPlanner>>;
}

/// A Morsel Planner is responsible for creating morsels for a given scan.
///
/// The [`MorselPlanner`] is the unit of I/O. There is only ever a single I/O
/// outstanding for a specific planner. DataFusion may run
/// multiple planners in parallel, which corresponds to multiple parallel
/// I/O requests.
///
/// It is not a Rust `Stream` so that it can explicitly separate CPU bound
/// work from I/O work.
///
/// The design is similar to `ParquetPushDecoder`: when `plan` is called, it
/// should do CPU work to produce the next morsels or discover the next I/O
/// phase.
///
/// Best practice is to spawn I/O in a Tokio task on a separate runtime to
/// ensure that CPU work doesn't block or slow down I/O work, but this is not
/// strictly required by the API.
pub trait MorselPlanner: Send + Debug {
    /// Advance this planner by one step.
    ///
    /// This may involve CPU work, such as parsing parquet metadata and
    /// evaluating pruning predicates.
    ///
    /// It should NOT do any I/O work, such as reading from the file. If I/O is
    /// required, the returned [`MorselPlan`] may contain a
    /// [`BlockedPlannerContinuation`], whose future owns the blocked continuation and
    /// resolves to the next [`MorselPlanner`].
    ///
    /// Note this function is **not async** to make it explicitly clear that I/O
    /// must be driven by the returned [`BlockedPlannerContinuation`].
    ///
    /// # Lifecycle
    ///
    /// A planner moves between CPU-ready and I/O-blocked states:
    ///
    /// ```text
    /// Box<dyn MorselPlanner>
    ///     |
    ///     | step()
    ///     v
    /// PlannerStep::Ready(MorselPlan)
    ///     |
    ///     | may contain:
    ///     | - morsels
    ///     | - child planners
    ///     | - ready_continuation
    ///     | - blocked_continuation
    ///     v
    /// BlockedPlannerContinuation
    ///     |
    ///     | poll / run I/O
    ///     v
    /// Box<dyn MorselPlanner>
    /// ```
    ///
    /// Returns [`PlannerStep::Done`] if the planner has no more work to do.
    fn step(self: Box<Self>) -> Result<PlannerStep>;
}

/// Result of advancing a [`MorselPlanner`].
pub enum PlannerStep {
    /// CPU work produced morsels and/or child planners immediately.
    Ready(MorselPlan),
    /// The planner has no more work to do.
    Done,
}

impl Debug for PlannerStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready(plan) => f.debug_tuple("PlannerStep::Ready").field(plan).finish(),
            Self::Done => f.debug_tuple("PlannerStep::Done").finish(),
        }
    }
}

/// A named future that owns the blocked continuation of a [`MorselPlanner`].
///
/// This is not just "some I/O future". It is the suspended remainder of the
/// planner state machine: once the required I/O completes, polling this future
/// yields the next CPU-ready planner.
pub struct BlockedPlannerContinuation {
    future: BoxFuture<'static, Result<Box<dyn MorselPlanner>>>,
}

impl BlockedPlannerContinuation {
    /// Create a new blocked continuation future.
    pub fn new(future: BoxFuture<'static, Result<Box<dyn MorselPlanner>>>) -> Self {
        Self { future }
    }
}

impl Debug for BlockedPlannerContinuation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockedPlannerContinuation")
            .finish_non_exhaustive()
    }
}

impl Future for BlockedPlannerContinuation {
    type Output = Result<Box<dyn MorselPlanner>>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.future.as_mut().poll(cx)
    }
}

/// Return result of [`PlannerStep::Ready`].
///
/// # Logical Ordering
///
/// For plans where the output order of rows is maintained, the output order of
/// a [`MorselPlanner`] is logically defined as follows:
/// 1. All morsels that are directly produced
/// 2. Recursively, all morsels produced by the returned `planners`
///
/// # Scheduler View
///
/// A plan may hand the scheduler four kinds of work at once:
///
/// ```text
/// MorselPlan
/// ├── morsels: ready output work
/// ├── planners: ready child CPU work
/// ├── ready_continuation: same planner, still CPU-ready
/// └── blocked_continuation: same planner, blocked on I/O
/// ```
///
/// This lets a single `step()` return ready work immediately while also handing
/// off the blocked remainder to a more concurrent I/O executor.
#[derive(Default)]
pub struct MorselPlan {
    /// Any morsels that are ready for processing.
    morsels: Vec<Box<dyn Morsel>>,
    /// Any newly-created planners that are ready for CPU work.
    planners: Vec<Box<dyn MorselPlanner>>,
    /// The same planner's immediate CPU-ready continuation.
    ready_continuation: Option<Box<dyn MorselPlanner>>,
    /// A blocked continuation that can be polled independently while ready work
    /// is being consumed.
    blocked_continuation: Option<BlockedPlannerContinuation>,
}

impl Debug for MorselPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MorselPlan")
            .field("morsels", &self.morsels.len())
            .field("planners", &self.planners.len())
            .field(
                "ready_continuation",
                &self.ready_continuation.as_ref().map(|_| "<planner>"),
            )
            .field(
                "blocked_continuation",
                &self.blocked_continuation.as_ref().map(|_| "<blocked>"),
            )
            .finish()
    }
}

impl MorselPlan {
    /// Create an empty morsel plan.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ready morsels.
    pub fn with_morsels(mut self, morsels: Vec<Box<dyn Morsel>>) -> Self {
        self.morsels = morsels;
        self
    }

    /// Set the ready child planners.
    pub fn with_planners(mut self, planners: Vec<Box<dyn MorselPlanner>>) -> Self {
        self.planners = planners;
        self
    }

    /// Set the same planner's immediate CPU-ready continuation.
    pub fn with_ready_continuation(
        mut self,
        ready_continuation: Box<dyn MorselPlanner>,
    ) -> Self {
        self.ready_continuation = Some(ready_continuation);
        self
    }

    /// Set the blocked continuation that can be polled independently.
    pub fn with_blocked_continuation(
        mut self,
        blocked_continuation: BlockedPlannerContinuation,
    ) -> Self {
        self.blocked_continuation = Some(blocked_continuation);
        self
    }

    /// Take the ready morsels.
    pub fn take_morsels(&mut self) -> Vec<Box<dyn Morsel>> {
        std::mem::take(&mut self.morsels)
    }

    /// Take the ready child planners.
    pub fn take_planners(&mut self) -> Vec<Box<dyn MorselPlanner>> {
        std::mem::take(&mut self.planners)
    }

    /// Take the same planner's immediate CPU-ready continuation, if any.
    pub fn take_ready_continuation(&mut self) -> Option<Box<dyn MorselPlanner>> {
        self.ready_continuation.take()
    }

    /// Take the blocked continuation, if any.
    pub fn take_blocked_continuation(&mut self) -> Option<BlockedPlannerContinuation> {
        self.blocked_continuation.take()
    }

    /// Returns `true` if this plan contains a CPU-ready continuation.
    pub fn has_ready_continuation(&self) -> bool {
        self.ready_continuation.is_some()
    }

    /// Returns `true` if this plan contains a blocked continuation.
    pub fn has_blocked_continuation(&self) -> bool {
        self.blocked_continuation.is_some()
    }
}
