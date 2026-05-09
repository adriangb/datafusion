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

//! [`TrackerConfig`] — immutable knobs for [`super::tracker::SelectivityTracker`].
//!
//! Use the builder methods to customise, then call
//! [`build()`](TrackerConfig::build) to produce a ready-to-use tracker.

use std::collections::HashMap;

use parking_lot::{Mutex, RwLock};

use super::tracker::{SelectivityTracker, SelectivityTrackerInner};

/// Immutable configuration for a [`SelectivityTracker`].
#[doc(hidden)]
pub struct TrackerConfig {
    /// Minimum bytes/sec throughput for promoting a filter (default: INFINITY = disabled).
    pub min_bytes_per_sec: f64,
    /// Byte-ratio threshold for initial filter placement (row-level vs post-scan).
    /// Computed as `filter_compressed_bytes / projection_compressed_bytes`.
    /// When low, the filter columns are small relative to the projection,
    /// so row-level placement enables large late-materialization savings.
    /// When high, the filter columns dominate the projection, so there's
    /// little benefit from late materialization.
    /// Default is 0.20.
    pub byte_ratio_threshold: f64,
    /// Z-score for confidence intervals on filter effectiveness.
    /// Lower values (e.g. 1.0 or 0.0) will make the tracker more aggressive about promotion/demotion based on limited data.
    /// Higher values (e.g. 3.0) will require more confidence before changing filter states.
    /// Default is 2.0, corresponding to ~97.5% one-sided confidence.
    /// Set to <= 0.0 to disable confidence intervals and promote/demote based on point estimates alone (not recommended).
    /// Set to INFINITY to disable promotion entirely (overrides `min_bytes_per_sec`).
    pub confidence_z: f64,
}

impl TrackerConfig {
    pub fn new() -> Self {
        Self {
            min_bytes_per_sec: f64::INFINITY,
            byte_ratio_threshold: 0.20,
            confidence_z: 2.0,
        }
    }

    pub fn with_min_bytes_per_sec(mut self, v: f64) -> Self {
        self.min_bytes_per_sec = v;
        self
    }

    pub fn with_byte_ratio_threshold(mut self, v: f64) -> Self {
        self.byte_ratio_threshold = v;
        self
    }

    pub fn with_confidence_z(mut self, v: f64) -> Self {
        self.confidence_z = v;
        self
    }

    pub fn build(self) -> SelectivityTracker {
        SelectivityTracker {
            config: self,
            filter_stats: RwLock::new(HashMap::new()),
            skip_flags: RwLock::new(HashMap::new()),
            inner: Mutex::new(SelectivityTrackerInner::new()),
        }
    }
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self::new()
    }
}
