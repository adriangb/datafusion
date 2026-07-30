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

//! **DO NOT MERGE.** Deliberate memory spikes, injected into an operator to
//! validate that the benchmark harness reports them.
//!
//! Two spikes are injected, separately sized, because the harness has two
//! independent reporting channels and they measure different things:
//!
//! - `DF_INJECT_POOL_BYTES` grows a [`MemoryReservation`] against the task's
//!   [`MemoryPool`] without allocating anything. It should appear in
//!   `pool_peak_bytes` and *not* in peak RSS.
//! - `DF_INJECT_ALLOC_BYTES` allocates and writes a buffer without telling the
//!   pool about it — what an operator that under-accounts actually does. It
//!   should appear in peak RSS and *not* in `pool_peak_bytes`.
//!
//! Setting both at once therefore separates the channels in a single run: each
//! number moves by exactly one of the two knobs.
//!
//! Both are unset by default, so a build carrying this file behaves normally
//! until the environment asks for a spike.
//!
//! # One injection at a time
//!
//! [`MemoryInjection::install`] is called per stream, and how many streams a
//! plan has depends on the query and the partition count. A process-wide flag
//! keeps at most one injection live, so the spike is one fixed size rather than
//! a multiple of however many operators happened to run — otherwise a 1 GiB
//! knob turns into 12 GiB on a 12-partition plan and exhausts the pool.
//!
//! The reservation is released when the injection is dropped, i.e. when the
//! stream holding it is dropped, which frees the flag for the next query.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};

/// Whether an injection is currently live anywhere in the process.
static ACTIVE: AtomicBool = AtomicBool::new(false);

/// Name the injected reservation registers under, so it is identifiable in
/// `TrackConsumersPool` output rather than looking like a real consumer.
const CONSUMER_NAME: &str = "InjectedMemoryProbe";

/// A live memory spike. Both halves are released on drop.
pub(crate) struct MemoryInjection {
    /// Pool accounting with no backing allocation. Never read: held so the
    /// reservation lives as long as this value, and released by its `Drop`.
    #[cfg_attr(not(test), expect(dead_code, reason = "held for its Drop, never read"))]
    reservation: Option<MemoryReservation>,
    /// Backing allocation with no pool accounting. Never read, for the same
    /// reason — holding it is what keeps the pages resident.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "held to keep the pages resident, never read")
    )]
    buffer: Option<Vec<u8>>,
}

impl MemoryInjection {
    /// Install a spike for the lifetime of the returned value, or return
    /// `None` if neither knob is set or another injection is already live.
    pub(crate) fn install(pool: &Arc<dyn MemoryPool>) -> Option<Self> {
        let pool_bytes = env_bytes("DF_INJECT_POOL_BYTES");
        let alloc_bytes = env_bytes("DF_INJECT_ALLOC_BYTES");
        if pool_bytes.is_none() && alloc_bytes.is_none() {
            return None;
        }

        // Claim the process-wide slot, or leave it to whoever holds it.
        if ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return None;
        }

        let reservation = pool_bytes.map(|bytes| {
            let reservation = MemoryConsumer::new(CONSUMER_NAME).register(pool);
            // `grow` rather than `try_grow`: the point is to record a peak, not
            // to test the limit, and a failure here would fail the query.
            reservation.grow(bytes);
            reservation
        });

        // `vec![1u8; n]` writes every byte, so the pages are resident and the
        // spike reaches RSS. A `with_capacity` would only reserve address space.
        let buffer = alloc_bytes.map(|bytes| vec![1u8; bytes]);

        Some(Self {
            reservation,
            buffer,
        })
    }
}

impl Drop for MemoryInjection {
    fn drop(&mut self) {
        ACTIVE.store(false, Ordering::Release);
    }
}

/// Read a byte count from `key`, accepting a plain number of bytes or a `K`,
/// `M`, or `G` suffix (`1G`, `512M`). Unparseable values are ignored.
fn env_bytes(key: &str) -> Option<usize> {
    let raw = std::env::var(key).ok()?;
    let raw = raw.trim();
    let (digits, multiplier) = match raw.as_bytes().last()? {
        b'K' | b'k' => (&raw[..raw.len() - 1], 1 << 10),
        b'M' | b'm' => (&raw[..raw.len() - 1], 1 << 20),
        b'G' | b'g' => (&raw[..raw.len() - 1], 1 << 30),
        _ => (raw, 1),
    };
    let value: usize = digits.trim().parse().ok()?;
    value.checked_mul(multiplier).filter(|bytes| *bytes > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_execution::memory_pool::GreedyMemoryPool;

    /// `install` is driven by process-wide environment variables and a
    /// process-wide flag, so the tests that touch either run under one lock
    /// rather than in parallel.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn pool() -> Arc<dyn MemoryPool> {
        Arc::new(GreedyMemoryPool::new(1 << 30))
    }

    #[test]
    fn parses_suffixes_and_rejects_junk() {
        unsafe {
            std::env::set_var("DF_TEST_BYTES", "1G");
        }
        assert_eq!(env_bytes("DF_TEST_BYTES"), Some(1 << 30));
        unsafe {
            std::env::set_var("DF_TEST_BYTES", "512M");
        }
        assert_eq!(env_bytes("DF_TEST_BYTES"), Some(512 << 20));
        unsafe {
            std::env::set_var("DF_TEST_BYTES", "1024");
        }
        assert_eq!(env_bytes("DF_TEST_BYTES"), Some(1024));
        // Zero is indistinguishable from "no spike", so it is treated as one.
        unsafe {
            std::env::set_var("DF_TEST_BYTES", "0");
        }
        assert_eq!(env_bytes("DF_TEST_BYTES"), None);
        unsafe {
            std::env::set_var("DF_TEST_BYTES", "not-a-size");
        }
        assert_eq!(env_bytes("DF_TEST_BYTES"), None);
        unsafe {
            std::env::remove_var("DF_TEST_BYTES");
        }
        assert_eq!(env_bytes("DF_TEST_BYTES"), None);
    }

    #[test]
    fn no_injection_without_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let pool = pool();
        assert!(MemoryInjection::install(&pool).is_none());
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn pool_knob_reserves_without_allocating() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("DF_INJECT_POOL_BYTES", "4M");
        }
        let pool = pool();
        let injection = MemoryInjection::install(&pool).expect("installed");
        assert_eq!(pool.reserved(), 4 << 20);
        assert!(injection.reservation.is_some());
        assert!(injection.buffer.is_none());
        drop(injection);
        assert_eq!(pool.reserved(), 0);
        unsafe {
            std::env::remove_var("DF_INJECT_POOL_BYTES");
        }
    }

    #[test]
    fn alloc_knob_allocates_without_reserving() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("DF_INJECT_ALLOC_BYTES", "4M");
        }
        let pool = pool();
        let injection = MemoryInjection::install(&pool).expect("installed");
        assert_eq!(pool.reserved(), 0);
        assert!(injection.reservation.is_none());
        assert_eq!(injection.buffer.as_ref().map(Vec::len), Some(4 << 20));
        unsafe {
            std::env::remove_var("DF_INJECT_ALLOC_BYTES");
        }
    }

    #[test]
    fn only_one_injection_is_live_at_a_time() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe {
            std::env::set_var("DF_INJECT_POOL_BYTES", "4M");
        }
        let pool = pool();
        let first = MemoryInjection::install(&pool).expect("installed");
        assert!(MemoryInjection::install(&pool).is_none());
        assert_eq!(pool.reserved(), 4 << 20);
        drop(first);
        // The slot is free again once the holder is dropped.
        let second = MemoryInjection::install(&pool).expect("installed");
        assert_eq!(pool.reserved(), 4 << 20);
        drop(second);
        unsafe {
            std::env::remove_var("DF_INJECT_POOL_BYTES");
        }
    }
}
