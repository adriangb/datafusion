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

use datafusion_common::internal_datafusion_err;
use std::collections::VecDeque;
use std::task::{Context, Poll};

use crate::morsel::{Morsel, MorselPlanner, Morselizer, PendingMorselPlanner};
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result};
use datafusion_physical_plan::metrics::ScopedTimerGuard;
use futures::stream::BoxStream;
use futures::{FutureExt as _, StreamExt as _};

use super::work_source::WorkSource;
use super::{FileStreamMetrics, OnError};

/// Applies `on_error` to an error from opening a file.
///
/// Evaluates to `Some` with the value to return from `poll_scan`, or `None`
/// to skip the file. It is a macro, not a method, because `poll_scan` holds a
/// borrow of `metrics.time_processing` and this touches other metric fields.
macro_rules! open_error {
    ($this:ident, $err:expr) => {{
        $this.metrics.file_open_errors.add(1);
        $this.metrics.time_opening.stop();
        match $this.on_error {
            OnError::Skip => {
                $this.metrics.files_processed.add(1);
                None
            }
            OnError::Fail => Some(ScanAndReturn::Error($err)),
        }
    }};
}

/// State [`FileStreamState::Scan`].
///
/// There is one `ScanState` per `FileStream`, and thus per output partition.
///
/// It groups together the lifecycle of scanning that partition's files:
/// unopened files, the files that are opening, the active reader, and the
/// metrics associated with processing that work.
///
/// # I/O
///
/// The `ScanState` opens up to `open_ahead` files at the same time. Each
/// opening file has at most one planner I/O outstanding, so at most
/// `open_ahead` planner I/Os are outstanding. Only the file at the front of
/// `open_files` turns its morsels into the active reader. Thus the stream
/// decodes one file at a time, and it keeps the order of `work_source`.
///
/// # State Transitions
///
/// ```text
/// work_source
///    |
///    v
/// morselizer.plan_file(file)  (up to open_ahead files)
///    |
///    v
/// open_files[i]: ready_planners ---> plan() ---> morsels
///                     ^               |            |
///                     |               v            | front file only
///                     +------- pending_planner     v
///                                         into_stream() ---> reader ---> RecordBatches
/// ```
///
/// [`FileStreamState::Scan`]: super::FileStreamState::Scan
pub(super) struct ScanState {
    /// Unopened files that still need to be planned for this stream.
    work_source: WorkSource,
    /// Remaining row limit, if any.
    remain: Option<usize>,
    /// The morselizer used to plan files.
    morselizer: Box<dyn Morselizer>,
    /// Behavior if opening or scanning a file fails.
    on_error: OnError,
    /// The maximum length of `open_files`.
    open_ahead: usize,
    /// Files that are open, in `work_source` order.
    open_files: VecDeque<OpenFile>,
    /// The active reader, if any. It reads a morsel of the front file.
    reader: Option<BoxStream<'static, Result<RecordBatch>>>,
    /// Metrics for the active scan queues.
    metrics: FileStreamMetrics,
}

/// The planning state of one open file.
#[derive(Default)]
struct OpenFile {
    /// CPU-ready planners for this file.
    ready_planners: VecDeque<Box<dyn MorselPlanner>>,
    /// The single planner of this file that waits on I/O, if any.
    pending_planner: Option<PendingMorselPlanner>,
    /// Morsels of this file that wait to become the reader.
    morsels: VecDeque<Box<dyn Morsel>>,
}

impl OpenFile {
    /// Returns true when this file has no planning work and no morsels left.
    fn is_exhausted(&self) -> bool {
        self.ready_planners.is_empty()
            && self.pending_planner.is_none()
            && self.morsels.is_empty()
    }

    /// Polls planner I/O and, if `plan` is true, runs CPU planning for this
    /// file until it waits on I/O or has no planners.
    ///
    /// If `plan` is true and this function returns `Ok`, either
    /// `pending_planner` is set and was polled (so a waker is registered), or
    /// `ready_planners` is empty.
    fn drive(
        &mut self,
        cx: &mut Context<'_>,
        metrics: &FileStreamMetrics,
        plan: bool,
    ) -> Result<()> {
        loop {
            if let Some(mut pending_planner) = self.pending_planner.take() {
                match pending_planner.poll_unpin(cx) {
                    Poll::Pending => {
                        self.pending_planner = Some(pending_planner);
                        return Ok(());
                    }
                    Poll::Ready(Ok(planner)) => self.ready_planners.push_back(planner),
                    Poll::Ready(Err(err)) => return Err(err),
                }
            }

            if !plan {
                return Ok(());
            }
            let Some(planner) = self.ready_planners.pop_front() else {
                return Ok(());
            };
            match planner.plan()? {
                Some(mut plan) => {
                    self.morsels.extend(plan.take_morsels());
                    self.ready_planners.extend(plan.take_ready_planners());
                    if let Some(pending_planner) = plan.take_pending_planner() {
                        if self.pending_planner.is_some() {
                            return Err(internal_datafusion_err!(
                                "Conflicting pending planner state in FileStream ScanState"
                            ));
                        }
                        self.pending_planner = Some(pending_planner);
                    }
                }
                None => {
                    // The planner pruned the rest of its file.
                    metrics.files_processed.add(1);
                }
            }
        }
    }
}

impl ScanState {
    pub(super) fn new(
        work_source: WorkSource,
        remain: Option<usize>,
        morselizer: Box<dyn Morselizer>,
        on_error: OnError,
        metrics: FileStreamMetrics,
        open_ahead: usize,
    ) -> Self {
        Self {
            work_source,
            remain,
            morselizer,
            on_error,
            open_ahead: open_ahead.max(1),
            open_files: Default::default(),
            reader: None,
            metrics,
        }
    }

    /// Updates how scan errors are handled while the stream is still active.
    pub(super) fn set_on_error(&mut self, on_error: OnError) {
        self.on_error = on_error;
    }

    /// Drives one iteration of the active scan state.
    ///
    /// Work is attempted in this order:
    /// 1. open unopened files until `open_files` holds `open_ahead` files
    /// 2. run planning and planner I/O for every open file
    /// 3. poll the active reader
    /// 4. turn a morsel of the front file into the active reader
    ///
    /// The return [`ScanAndReturn`] tells `poll_inner` how to update the
    /// outer `FileStreamState`.
    pub(super) fn poll_scan(&mut self, cx: &mut Context<'_>) -> ScanAndReturn {
        let _processing_timer: ScopedTimerGuard<'_> =
            self.metrics.time_processing.timer();

        // Open more files. A morselizer only does CPU work here.
        while self.open_files.len() < self.open_ahead {
            if !self.open_files.is_empty()
                && (!self.morselizer.can_open_ahead()
                    || !self.work_source.may_open_ahead(self.open_files.len()))
            {
                break;
            }
            let Some(part_file) = self.work_source.pop_front() else {
                break;
            };
            if self.metrics.time_opening.start.is_none() {
                self.metrics.time_opening.start();
            }
            match self.morselizer.plan_file(part_file) {
                Ok(planner) => {
                    self.metrics.files_opened.add(1);
                    let mut open_file = OpenFile::default();
                    open_file.ready_planners.push_back(planner);
                    self.open_files.push_back(open_file);
                }
                Err(err) => {
                    if let Some(ret) = open_error!(self, err) {
                        return ret;
                    }
                }
            }
        }

        // Drive every open file. Each file can have one planner I/O
        // outstanding, so the open files load metadata in parallel. The files
        // behind the front file also run CPU planning now, so that their next
        // I/O starts while the front file streams. The front file only polls
        // its I/O here and plans below, when no reader is active.
        let mut i = 0;
        while i < self.open_files.len() {
            match self.open_files[i].drive(cx, &self.metrics, i > 0) {
                Ok(()) => i += 1,
                Err(err) => {
                    self.open_files.remove(i);
                    if let Some(ret) = open_error!(self, err) {
                        return ret;
                    }
                }
            }
        }

        // Next try and get the next batch from the active reader, if any.
        if let Some(reader) = self.reader.as_mut() {
            match reader.poll_next_unpin(cx) {
                // Morsels should ideally only expose ready-to-decode streams,
                // but tolerate pending readers here.
                Poll::Pending => return ScanAndReturn::Return(Poll::Pending),
                Poll::Ready(Some(Ok(batch))) => {
                    self.metrics.time_scanning_until_data.stop();
                    self.metrics.time_scanning_total.stop();
                    // Apply any remaining row limit.
                    let (batch, finished) = match &mut self.remain {
                        Some(remain) => {
                            if *remain > batch.num_rows() {
                                *remain -= batch.num_rows();
                                self.metrics.time_scanning_total.start();
                                (batch, false)
                            } else {
                                let batch = batch.slice(0, *remain);
                                let done = 1
                                    + self.work_source.skipped_on_limit()
                                    + self.open_files.len().saturating_sub(1);
                                self.metrics.files_processed.add(done);
                                *remain = 0;
                                (batch, true)
                            }
                        }
                        None => {
                            self.metrics.time_scanning_total.start();
                            (batch, false)
                        }
                    };
                    return if finished {
                        ScanAndReturn::Done(Some(Ok(batch)))
                    } else {
                        ScanAndReturn::Return(Poll::Ready(Some(Ok(batch))))
                    };
                }
                Poll::Ready(Some(Err(err))) => {
                    self.reader = None;
                    self.metrics.file_scan_errors.add(1);
                    self.metrics.time_scanning_until_data.stop();
                    self.metrics.time_scanning_total.stop();
                    return match self.on_error {
                        OnError::Skip => {
                            // Drop the rest of the failed file.
                            self.open_files.pop_front();
                            self.metrics.files_processed.add(1);
                            ScanAndReturn::Continue
                        }
                        OnError::Fail => ScanAndReturn::Error(err),
                    };
                }
                Poll::Ready(None) => {
                    self.reader = None;
                    self.metrics.files_processed.add(1);
                    self.metrics.time_scanning_until_data.stop();
                    self.metrics.time_scanning_total.stop();
                    return ScanAndReturn::Continue;
                }
            }
        }

        // No active reader. Only the front file may supply the next reader,
        // which keeps the output in file order.
        let Some(front) = self.open_files.front_mut() else {
            return ScanAndReturn::Done(None);
        };
        if let Some(morsel) = front.morsels.pop_front() {
            self.metrics.time_opening.stop();
            self.metrics.time_scanning_until_data.start();
            self.metrics.time_scanning_total.start();
            self.reader = Some(morsel.into_stream());
            return ScanAndReturn::Continue;
        }
        if let Err(err) = front.drive(cx, &self.metrics, true) {
            self.open_files.pop_front();
            return open_error!(self, err).unwrap_or(ScanAndReturn::Continue);
        }
        if !front.morsels.is_empty() {
            return ScanAndReturn::Continue;
        }
        if front.is_exhausted() {
            // Each reader counted itself as processed when it ended, and
            // `OpenFile::drive` counted each pruned planner.
            self.open_files.pop_front();
            return ScanAndReturn::Continue;
        }

        // The front file waits on planner I/O, and `drive` registered a waker.
        ScanAndReturn::Return(Poll::Pending)
    }
}

/// What should be done on the next iteration of [`ScanState::poll_scan`]?
pub(super) enum ScanAndReturn {
    /// Poll again.
    Continue,
    /// Return the provided result without changing the outer state.
    Return(Poll<Option<Result<RecordBatch>>>),
    /// Update the outer `FileStreamState` to `Done` and return the provided result.
    Done(Option<Result<RecordBatch>>),
    /// Update the outer `FileStreamState` to `Error` and return the provided error.
    Error(DataFusionError),
}
