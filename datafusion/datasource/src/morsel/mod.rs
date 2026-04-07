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
    /// Attempt to plan morsels. This may involve CPU work, such as parsing
    /// parquet metadata and evaluating pruning predicates.
    ///
    /// It should NOT do any I/O work, such as reading from the file. If I/O is
    /// required, the returned [`MorselPlan`] should contain a future that owns
    /// the blocked continuation and resolves to the next [`MorselPlanner`].
    ///
    /// Taking ownership of `self` encodes that contract in the type system:
    /// once `plan` returns an `io_future`, there is no planner value left that
    /// a caller could accidentally poll again before the I/O completes.
    ///
    /// Note this function is **not async** to make it explicitly clear that if
    /// I/O is required, it should be done in the returned `io_future`.
    ///
    /// Returns `None` if the planner has no more work to do.
    ///
    /// # Empty Morsel Plans
    ///
    /// It may return `None`, which means no batches will be read from the file
    /// (e.g. due to late-pruning based on statistics).
    ///
    /// # Output Ordering
    ///
    /// See the comments on [`MorselPlan`] for the logical output order.
    fn plan(self: Box<Self>) -> Result<Option<MorselPlan>>;
}

/// A named future that owns the blocked continuation of a [`MorselPlanner`].
///
/// This is not just "some I/O future". It is the suspended remainder of the
/// planner state machine: once the required I/O completes, polling this future
/// yields the next CPU-ready planner.
///
/// This avoids the previous runtime protocol of "planner is waiting, so don't
/// call `plan` again yet": the blocked continuation has moved into this future.
pub struct PendingMorselPlanner {
    future: BoxFuture<'static, Result<Box<dyn MorselPlanner>>>,
}

impl PendingMorselPlanner {
    /// Create a new blocked continuation future.
    pub fn new(future: BoxFuture<'static, Result<Box<dyn MorselPlanner>>>) -> Self {
        Self { future }
    }
}

impl Debug for PendingMorselPlanner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingMorselPlanner")
            .finish_non_exhaustive()
    }
}

impl Future for PendingMorselPlanner {
    type Output = Result<Box<dyn MorselPlanner>>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        self.future.as_mut().poll(cx)
    }
}

/// Return result of [`MorselPlanner::plan`].
///
/// # Logical Ordering
///
/// For plans where the output order of rows is maintained, the output order of
/// a [`MorselPlanner`] is logically defined as follows:
/// 1. All morsels that are directly produced
/// 2. Recursively, all morsels produced by the returned `planners`
#[derive(Default)]
pub struct MorselPlan {
    /// Any morsels that are ready for processing.
    morsels: Vec<Box<dyn Morsel>>,
    /// Any newly-created planners that are ready for CPU work.
    planners: Vec<Box<dyn MorselPlanner>>,
    /// A future that will drive any I/O work to completion and yield the next
    /// CPU-ready planner.
    ///
    /// DataFusion will poll this future occasionally to drive the I/O work to
    /// completion. Once the future resolves, the returned planner is ready for
    /// another call to `plan`.
    io_future: Option<PendingMorselPlanner>,
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

    /// Set the pending I/O future.
    pub fn with_io_future(mut self, io_future: PendingMorselPlanner) -> Self {
        self.io_future = Some(io_future);
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

    /// Take the pending I/O future, if any.
    pub fn take_io_future(&mut self) -> Option<PendingMorselPlanner> {
        self.io_future.take()
    }

    /// Set the pending I/O future.
    pub fn set_io_future(&mut self, io_future: PendingMorselPlanner) {
        self.io_future = Some(io_future);
    }

    /// Returns `true` if this plan contains an I/O future.
    pub fn has_io_future(&self) -> bool {
        self.io_future.is_some()
    }
}
