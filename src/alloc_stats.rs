//! Allocation counter — compiled only with `--features alloc-stats`.
//!
//! Wraps the system allocator and counts every alloc/realloc made by the Rust
//! side of the extension. Used by `scripts/metrics.py` to verify the
//! "zero allocation" claim; normal builds do not include this module.

use pyo3::prelude::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static BASE: AtomicU64 = AtomicU64::new(0);

#[inline]
fn grow(n: u64) {
    let live = LIVE.fetch_add(n, Relaxed) + n;
    PEAK.fetch_max(live, Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        grow(l.size() as u64);
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(l.size() as u64, Relaxed);
        grow(l.size() as u64);
        System.alloc_zeroed(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        LIVE.fetch_sub(l.size() as u64, Relaxed);
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(new as u64, Relaxed);
        if new > l.size() {
            grow((new - l.size()) as u64);
        } else {
            LIVE.fetch_sub((l.size() - new) as u64, Relaxed);
        }
        System.realloc(p, l, new)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Reset counters; peak is reset to the current live size.
#[pyfunction]
pub fn _alloc_reset() {
    ALLOCS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    let live = LIVE.load(Relaxed);
    BASE.store(live, Relaxed);
    PEAK.store(live, Relaxed);
}

/// (alloc_count, alloc_bytes, peak_live_bytes_above_reset)
#[pyfunction]
pub fn _alloc_stats() -> (u64, u64, u64) {
    let peak = PEAK.load(Relaxed).saturating_sub(BASE.load(Relaxed));
    (ALLOCS.load(Relaxed), BYTES.load(Relaxed), peak)
}
