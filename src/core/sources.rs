//! Concrete stream sources.
//!
//! Each source is a zero-allocation handle — no heap allocation at construction.
//! The slice sources hold a *reference*, so they are lifetime-bound; the owned
//! variants hold a `Vec` and drive the Rust ownership model cleanly.

use super::traits::ZStream;

// ---------------------------------------------------------------------------
// Slice source — borrows data, zero allocation, best for hot loops
// ---------------------------------------------------------------------------

pub struct SliceStream<'a, T> {
    data: &'a [T],
    pos: usize,
}

impl<'a, T> SliceStream<'a, T> {
    pub fn new(data: &'a [T]) -> Self {
        SliceStream { data, pos: 0 }
    }

    pub fn remaining(&self) -> &[T] {
        &self.data[self.pos..]
    }
}

impl<'a, T: Clone> ZStream for SliceStream<'a, T> {
    type Item = T;

    #[inline(always)]
    fn next_item(&mut self) -> Option<T> {
        let item = self.data.get(self.pos)?;
        self.pos += 1;
        Some(item.clone())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len() - self.pos;
        (remaining, Some(remaining))
    }

    fn is_vectorizable(&self) -> bool {
        true // contiguous typed data — SIMD eligible
    }
}

// ---------------------------------------------------------------------------
// Ref-item variant — no clone, item is a reference (useful for large structs)
// ---------------------------------------------------------------------------

pub struct SliceRefStream<'a, T> {
    data: &'a [T],
    pos: usize,
}

impl<'a, T> SliceRefStream<'a, T> {
    pub fn new(data: &'a [T]) -> Self {
        SliceRefStream { data, pos: 0 }
    }
}

impl<'a, T> ZStream for SliceRefStream<'a, T> {
    type Item = &'a T;

    #[inline(always)]
    fn next_item(&mut self) -> Option<&'a T> {
        let item = self.data.get(self.pos)?;
        self.pos += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.data.len() - self.pos;
        (r, Some(r))
    }

    fn is_vectorizable(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Owned Vec source — takes ownership, useful when we already own the data
// ---------------------------------------------------------------------------

/// Wraps `std::vec::IntoIter<T>` as a `ZStream`.
///
/// The previous implementation used `unsafe { ptr::read + set_len(0) }` to
/// move elements out without running Drop on consumed slots, then manually
/// called `drop_in_place` on the tail.  This is exactly what `Vec::into_iter`
/// does internally — and the standard library version is verified, audited,
/// and handles all edge cases (ZSTs, panics in Drop, etc.).
/// There is no reason to reimplement it.
pub struct VecStream<T> {
    inner: std::vec::IntoIter<T>,
}

impl<T> VecStream<T> {
    pub fn new(data: Vec<T>) -> Self {
        VecStream {
            inner: data.into_iter(),
        }
    }

    pub fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        VecStream {
            inner: iter.into_iter().collect::<Vec<_>>().into_iter(),
        }
    }
}

impl<T> ZStream for VecStream<T> {
    type Item = T;

    #[inline(always)]
    fn next_item(&mut self) -> Option<T> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // IntoIter implements ExactSizeIterator — both bounds are exact.
        let n = self.inner.len();
        (n, Some(n))
    }

    fn is_vectorizable(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Range source — integer ranges without materializing into a Vec
// ---------------------------------------------------------------------------

pub struct RangeStream {
    current: i64,
    end: i64,
    step: i64,
}

impl RangeStream {
    pub fn new(start: i64, end: i64) -> Self {
        RangeStream {
            current: start,
            end,
            step: 1,
        }
    }

    pub fn with_step(start: i64, end: i64, step: i64) -> Self {
        assert_ne!(step, 0, "step cannot be zero");
        RangeStream {
            current: start,
            end,
            step,
        }
    }
}

impl ZStream for RangeStream {
    type Item = i64;

    #[inline(always)]
    fn next_item(&mut self) -> Option<i64> {
        if (self.step > 0 && self.current >= self.end)
            || (self.step < 0 && self.current <= self.end)
        {
            return None;
        }
        let v = self.current;
        self.current = self.current.wrapping_add(self.step);
        Some(v)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = if self.step > 0 {
            ((self.end - self.current).max(0) as usize).div_ceil(self.step as usize)
        } else {
            ((self.current - self.end).max(0) as usize).div_ceil((-self.step) as usize)
        };
        (remaining, Some(remaining))
    }
}

// ---------------------------------------------------------------------------
// Repeat/Once sources
// ---------------------------------------------------------------------------

pub struct RepeatN<T: Clone> {
    value: T,
    remaining: usize,
}

impl<T: Clone> RepeatN<T> {
    pub fn new(value: T, n: usize) -> Self {
        RepeatN {
            value,
            remaining: n,
        }
    }
}

impl<T: Clone> ZStream for RepeatN<T> {
    type Item = T;

    #[inline(always)]
    fn next_item(&mut self) -> Option<T> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(self.value.clone())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

pub struct Once<T> {
    value: Option<T>,
}

impl<T> Once<T> {
    pub fn new(value: T) -> Self {
        Once { value: Some(value) }
    }
}

impl<T> ZStream for Once<T> {
    type Item = T;

    #[inline(always)]
    fn next_item(&mut self) -> Option<T> {
        self.value.take()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.value.is_some() as usize;
        (n, Some(n))
    }
}

// ---------------------------------------------------------------------------
// Concatenated chunks — efficient over pre-split data
// ---------------------------------------------------------------------------

pub struct ChunkedStream<T> {
    chunks: Vec<Vec<T>>,
    chunk_idx: usize,
    item_idx: usize,
}

impl<T> ChunkedStream<T> {
    pub fn new(chunks: Vec<Vec<T>>) -> Self {
        ChunkedStream {
            chunks,
            chunk_idx: 0,
            item_idx: 0,
        }
    }
}

impl<T: Clone> ZStream for ChunkedStream<T> {
    type Item = T;

    #[inline]
    fn next_item(&mut self) -> Option<T> {
        loop {
            let chunk = self.chunks.get(self.chunk_idx)?;
            if let Some(item) = chunk.get(self.item_idx) {
                self.item_idx += 1;
                return Some(item.clone());
            }
            self.chunk_idx += 1;
            self.item_idx = 0;
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let total: usize = self.chunks.iter().map(|c| c.len()).sum();
        let consumed: usize = self.chunks[..self.chunk_idx]
            .iter()
            .map(|c| c.len())
            .sum::<usize>()
            + self.item_idx;
        let remaining = total - consumed;
        (remaining, Some(remaining))
    }
}
