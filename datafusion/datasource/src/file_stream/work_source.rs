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

use std::collections::VecDeque;
use std::sync::Arc;

use crate::PartitionedFile;
use crate::file_groups::FileGroup;
use crate::file_scan_config::FileScanConfig;
use parking_lot::Mutex;

/// Source of work for `ScanState`.
///
/// Streams that may share work across siblings use [`WorkSource::Shared`],
/// while streams that can not share work (e.g. because they must preserve file
/// order) use  [`WorkSource::Local`].
#[derive(Debug, Clone)]
pub(super) enum WorkSource {
    /// Files this stream will plan locally without sharing them.
    Local(VecDeque<PartitionedFile>),
    /// Files shared with sibling streams.
    Shared(SharedWorkSource),
}

impl WorkSource {
    /// Pop the next file to plan from this work source.
    pub(super) fn pop_front(&mut self) -> Option<PartitionedFile> {
        match self {
            Self::Local(files) => files.pop_front(),
            Self::Shared(shared) => shared.pop_front(),
        }
    }

    /// Whether a stream that has `open` files open may open one more ahead.
    ///
    /// A local source belongs to this stream alone. A shared source holds
    /// the work of all sibling streams: a stream that opens files ahead takes
    /// them from its siblings. Thus a stream opens another file ahead only
    /// while the queue still holds `open` files for each sibling, its fair
    /// share. Without this, one stream can take most of the queue early and
    /// run it alone while its siblings have nothing to steal.
    pub(super) fn may_open_ahead(&self, open: usize) -> bool {
        match self {
            Self::Local(_) => true,
            Self::Shared(shared) => shared.len() >= open.saturating_mul(shared.streams()),
        }
    }

    /// Return how many queued files should be counted as already processed
    /// when this stream stops early after hitting a global limit.
    pub(super) fn skipped_on_limit(&self) -> usize {
        match self {
            Self::Local(files) => files.len(),
            Self::Shared(_) => 0,
        }
    }
}

/// Shared source of work for sibling `FileStream`s
///
/// The queue is created once per execution and shared by all reorderable
/// sibling streams for that execution. Whichever stream becomes idle first may
/// take the next unopened file from the front of the queue.
///
/// It uses a [`Mutex`] internally to provide thread-safe access
/// to the shared file queue.
#[derive(Debug, Clone)]
pub(crate) struct SharedWorkSource {
    inner: Arc<SharedWorkSourceInner>,
}

#[derive(Debug, Default)]
pub(super) struct SharedWorkSourceInner {
    files: Mutex<VecDeque<PartitionedFile>>,
    /// The number of sibling streams that share the queue.
    streams: usize,
}

impl SharedWorkSource {
    /// Create a shared work source containing the provided unopened files.
    pub(crate) fn new(files: impl IntoIterator<Item = PartitionedFile>) -> Self {
        Self::with_streams(files, 1)
    }

    /// Create a shared work source for `streams` sibling streams.
    pub(crate) fn with_streams(
        files: impl IntoIterator<Item = PartitionedFile>,
        streams: usize,
    ) -> Self {
        let files = files.into_iter().collect();
        Self {
            inner: Arc::new(SharedWorkSourceInner {
                files: Mutex::new(files),
                streams: streams.max(1),
            }),
        }
    }

    /// Create a shared work source for the unopened files in `config`.
    ///
    /// Files are reordered by the file source (e.g. by statistics for TopK)
    /// before being placed in the shared queue, so the most promising files
    /// are processed first across all partitions.
    pub(crate) fn from_config(config: &FileScanConfig) -> Self {
        let files: Vec<_> = config
            .file_groups
            .iter()
            .flat_map(FileGroup::iter)
            .cloned()
            .collect();
        let files = config.file_source.reorder_files(files);
        Self::with_streams(files, config.file_groups.len())
    }

    /// Pop the next file from the shared work queue.
    ///
    /// Returns `None` if the queue is empty
    fn pop_front(&self) -> Option<PartitionedFile> {
        self.inner.files.lock().pop_front()
    }

    /// The number of files left in the queue.
    fn len(&self) -> usize {
        self.inner.files.lock().len()
    }

    fn streams(&self) -> usize {
        self.inner.streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(n: usize) -> Vec<PartitionedFile> {
        (0..n)
            .map(|i| PartitionedFile::new(format!("f{i}.parquet"), 10))
            .collect()
    }

    /// Verifies that a stream may open files ahead from a shared queue only
    /// while the queue holds its fair share for every sibling stream.
    #[test]
    fn shared_source_limits_open_ahead_to_a_fair_share() {
        let shared = WorkSource::Shared(SharedWorkSource::with_streams(files(8), 4));
        assert!(shared.may_open_ahead(1), "8 queued >= 1 x 4 streams");
        assert!(shared.may_open_ahead(2), "8 queued >= 2 x 4 streams");
        assert!(!shared.may_open_ahead(3), "8 queued < 3 x 4 streams");

        let local = WorkSource::Local(files(1).into());
        assert!(local.may_open_ahead(100), "a local queue is never shared");
    }
}
