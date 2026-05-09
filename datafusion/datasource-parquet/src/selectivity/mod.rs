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

//! Adaptive filter selectivity tracking for the Parquet decoder.
//!
//! - [`SelectivityTracker`] — main entry point. Decides per filter whether
//!   it runs as a row-level predicate or a post-scan filter, and re-evaluates
//!   that placement at each row-group boundary.
//! - [`TrackerConfig`] — knobs (min bytes/sec for promotion, byte-ratio
//!   threshold for initial placement, confidence z-score).
//! - [`PartitionedFilters`] — output consumed by `ParquetOpener::open`.
//! - [`FilterId`] — stable per-conjunct identifier assigned by
//!   `ParquetSource::with_predicate`.
//!
//! Internally split across:
//! - [`stats`] — `SelectivityStats` Welford accumulator.
//! - [`config`] — `TrackerConfig` builder.
//! - [`tracker`] — `SelectivityTracker` outer API and its inner state machine.
//! - [`types`] — `FilterState`, `PartitionedFilters`, `PartitionResult`.
//! - [`skippable`] — `count_skippable_bytes` scatter calculation, used both
//!   here and by the row-filter / post-scan paths.

mod config;
mod skippable;
mod stats;
mod tracker;
mod types;

pub use config::TrackerConfig;
pub use tracker::SelectivityTracker;
pub use types::{FilterId, PartitionedFilters};

pub(crate) use skippable::count_skippable_bytes;

// Private re-exports so the in-crate test module can write
// `use super::*;` without reaching into the submodules.
#[cfg(test)]
use stats::SelectivityStats;

#[cfg(test)]
mod tests;
