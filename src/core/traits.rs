//! Core stream trait and all operator combinators.
//!
//! The key design invariant: ALL types here are fully monomorphized — there is
//! no dynamic dispatch anywhere in this module.  The trait object boundary only
//! appears when we cross into the Python layer (`python::query`).
//!
//! Iterator fusion happens implicitly through Rust's type system.  A chain
//!   SliceStream → Map → Filter → Take
//! compiles to a single state machine with no heap allocation.

use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// Core trait
// ---------------------------------------------------------------------------

/// The zero-allocation stream abstraction.
///
/// Deliberately mirrors `std::Iterator` in shape, but lives separately so we
/// can attach fusion-specific methods, size hints, and vectorization flags
/// without needing a `From<Iterator>` blanket that would pull in all of std.
pub trait ZStream: Sized {
    type Item;

    fn next_item(&mut self) -> Option<Self::Item>;

    /// Bounds on remaining items.  Implementations must provide a tight lower
    /// bound — we use it to pre-allocate collection buffers.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }

    /// True when this pipeline can be batched into SIMD lanes.  Only
    /// implementations over contiguous typed arrays set this.
    fn is_vectorizable(&self) -> bool {
        false
    }

    // ------------------------------------------------------------------
    // Lazy combinators — each returns a *concrete* type, zero allocation
    // ------------------------------------------------------------------

    #[inline(always)]
    fn zmap<B, F: FnMut(Self::Item) -> B>(self, f: F) -> Map<Self, F> {
        Map { source: self, f }
    }

    #[inline(always)]
    fn zfilter<F: FnMut(&Self::Item) -> bool>(self, pred: F) -> Filter<Self, F> {
        Filter { source: self, pred }
    }

    #[inline(always)]
    fn zmap_filter<B, F: FnMut(Self::Item) -> Option<B>>(self, f: F) -> MapFilter<Self, F> {
        MapFilter { source: self, f }
    }

    #[inline(always)]
    fn ztake(self, n: usize) -> Take<Self> {
        Take {
            source: self,
            remaining: n,
        }
    }

    #[inline(always)]
    fn zskip(self, n: usize) -> Skip<Self> {
        Skip {
            source: self,
            to_skip: n,
        }
    }

    #[inline(always)]
    fn ztake_while<F: FnMut(&Self::Item) -> bool>(self, pred: F) -> TakeWhile<Self, F> {
        TakeWhile {
            source: self,
            pred,
            done: false,
        }
    }

    #[inline(always)]
    fn zskip_while<F: FnMut(&Self::Item) -> bool>(self, pred: F) -> SkipWhile<Self, F> {
        SkipWhile {
            source: self,
            pred,
            skipping: true,
        }
    }

    #[inline(always)]
    fn zenumerate(self) -> Enumerate<Self> {
        Enumerate {
            source: self,
            index: 0,
        }
    }

    #[inline(always)]
    fn zzip<Other: ZStream>(self, other: Other) -> Zip<Self, Other> {
        Zip { a: self, b: other }
    }

    #[inline(always)]
    fn zchain<Other: ZStream<Item = Self::Item>>(self, other: Other) -> Chain<Self, Other> {
        Chain {
            first: self,
            second: other,
            first_done: false,
        }
    }

    #[inline(always)]
    fn zflat_map<B, Inner, F>(self, f: F) -> FlatMap<Self, F, Inner>
    where
        Inner: ZStream<Item = B>,
        F: FnMut(Self::Item) -> Inner,
    {
        FlatMap {
            source: self,
            f,
            current: None,
        }
    }

    // ------------------------------------------------------------------
    // Terminal operations — consume the stream
    // ------------------------------------------------------------------

    #[inline]
    fn zcollect_vec(mut self) -> Vec<Self::Item> {
        let (lower, upper) = self.size_hint();
        // If we know the exact count avoid reallocations entirely.
        let cap = upper.unwrap_or(lower);
        let mut out = Vec::with_capacity(cap);
        while let Some(item) = self.next_item() {
            out.push(item);
        }
        out
    }

    #[inline]
    fn zfold<B, F: FnMut(B, Self::Item) -> B>(mut self, init: B, mut f: F) -> B {
        let mut acc = init;
        while let Some(item) = self.next_item() {
            acc = f(acc, item);
        }
        acc
    }

    #[inline]
    fn zfor_each<F: FnMut(Self::Item)>(mut self, mut f: F) {
        while let Some(item) = self.next_item() {
            f(item);
        }
    }

    #[inline]
    fn zcount(mut self) -> usize {
        let mut n = 0usize;
        while self.next_item().is_some() {
            n += 1;
        }
        n
    }

    #[inline]
    fn zfirst(mut self) -> Option<Self::Item> {
        self.next_item()
    }

    #[inline]
    fn zlast(mut self) -> Option<Self::Item> {
        let mut last = None;
        while let Some(item) = self.next_item() {
            last = Some(item);
        }
        last
    }

    #[inline]
    fn zany<F: FnMut(Self::Item) -> bool>(mut self, mut pred: F) -> bool {
        while let Some(item) = self.next_item() {
            if pred(item) {
                return true;
            }
        }
        false
    }

    #[inline]
    fn zall<F: FnMut(Self::Item) -> bool>(mut self, mut pred: F) -> bool {
        while let Some(item) = self.next_item() {
            if !pred(item) {
                return false;
            }
        }
        true
    }

    #[inline]
    fn zmin_by_key<B: Ord, F: FnMut(&Self::Item) -> B>(mut self, mut key: F) -> Option<Self::Item> {
        let mut min_item = self.next_item()?;
        let mut min_key = key(&min_item);
        while let Some(item) = self.next_item() {
            let k = key(&item);
            if k < min_key {
                min_key = k;
                min_item = item;
            }
        }
        Some(min_item)
    }

    #[inline]
    fn zmax_by_key<B: Ord, F: FnMut(&Self::Item) -> B>(mut self, mut key: F) -> Option<Self::Item> {
        let mut max_item = self.next_item()?;
        let mut max_key = key(&max_item);
        while let Some(item) = self.next_item() {
            let k = key(&item);
            if k > max_key {
                max_key = k;
                max_item = item;
            }
        }
        Some(max_item)
    }

    /// Erase the concrete stream type behind a dynamic dispatch wrapper.
    /// This is the boundary between the Rust type system and the Python layer.
    fn erase(self) -> ErasedStream<Self::Item>
    where
        Self: Send + 'static,
    {
        ErasedStream {
            inner: Box::new(self),
            _marker: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Adapter: std::Iterator → ZStream (zero cost)
// ---------------------------------------------------------------------------

pub struct FromIter<I: Iterator> {
    inner: I,
}

impl<I: Iterator> FromIter<I> {
    pub fn new(iter: I) -> Self {
        FromIter { inner: iter }
    }
}

impl<I: Iterator> ZStream for FromIter<I> {
    type Item = I::Item;

    #[inline(always)]
    fn next_item(&mut self) -> Option<I::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

// ZStream → std::Iterator bridge (allows use in for loops, std combinators)
pub struct IntoStdIter<S: ZStream>(pub S);

impl<S: ZStream> Iterator for IntoStdIter<S> {
    type Item = S::Item;
    #[inline(always)]
    fn next(&mut self) -> Option<S::Item> {
        self.0.next_item()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

// ---------------------------------------------------------------------------
// Operator: Map
// ---------------------------------------------------------------------------

pub struct Map<S: ZStream, F> {
    source: S,
    f: F,
}

impl<S, F, B> ZStream for Map<S, F>
where
    S: ZStream,
    F: FnMut(S::Item) -> B,
{
    type Item = B;

    #[inline(always)]
    fn next_item(&mut self) -> Option<B> {
        self.source.next_item().map(|x| (self.f)(x))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.source.size_hint()
    }
}

// ---------------------------------------------------------------------------
// Operator: Filter
// ---------------------------------------------------------------------------

pub struct Filter<S: ZStream, F> {
    source: S,
    pred: F,
}

impl<S, F> ZStream for Filter<S, F>
where
    S: ZStream,
    F: FnMut(&S::Item) -> bool,
{
    type Item = S::Item;

    #[inline(always)]
    fn next_item(&mut self) -> Option<S::Item> {
        loop {
            let item = self.source.next_item()?;
            if (self.pred)(&item) {
                return Some(item);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // Lower bound: 0 (all may be filtered), upper: source upper
        let (_, upper) = self.source.size_hint();
        (0, upper)
    }
}

// ---------------------------------------------------------------------------
// Operator: MapFilter (filter_map) — avoids double boxing vs Map+Filter
// ---------------------------------------------------------------------------

pub struct MapFilter<S: ZStream, F> {
    source: S,
    f: F,
}

impl<S, F, B> ZStream for MapFilter<S, F>
where
    S: ZStream,
    F: FnMut(S::Item) -> Option<B>,
{
    type Item = B;

    #[inline(always)]
    fn next_item(&mut self) -> Option<B> {
        loop {
            let item = self.source.next_item()?;
            if let Some(v) = (self.f)(item) {
                return Some(v);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Operator: Take
// ---------------------------------------------------------------------------

pub struct Take<S: ZStream> {
    source: S,
    remaining: usize,
}

impl<S: ZStream> ZStream for Take<S> {
    type Item = S::Item;

    #[inline(always)]
    fn next_item(&mut self) -> Option<S::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        self.source.next_item()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.source.size_hint();
        let r = self.remaining;
        (lower.min(r), Some(upper.map_or(r, |u| u.min(r))))
    }
}

// ---------------------------------------------------------------------------
// Operator: Skip
// ---------------------------------------------------------------------------

pub struct Skip<S: ZStream> {
    source: S,
    to_skip: usize,
}

impl<S: ZStream> ZStream for Skip<S> {
    type Item = S::Item;

    #[inline]
    fn next_item(&mut self) -> Option<S::Item> {
        while self.to_skip > 0 {
            self.source.next_item()?;
            self.to_skip -= 1;
        }
        self.source.next_item()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.source.size_hint();
        let s = self.to_skip;
        (lower.saturating_sub(s), upper.map(|u| u.saturating_sub(s)))
    }
}

// ---------------------------------------------------------------------------
// Operator: TakeWhile / SkipWhile
// ---------------------------------------------------------------------------

pub struct TakeWhile<S: ZStream, F> {
    source: S,
    pred: F,
    done: bool,
}

impl<S: ZStream, F: FnMut(&S::Item) -> bool> ZStream for TakeWhile<S, F> {
    type Item = S::Item;

    #[inline(always)]
    fn next_item(&mut self) -> Option<S::Item> {
        if self.done {
            return None;
        }
        let item = self.source.next_item()?;
        if (self.pred)(&item) {
            Some(item)
        } else {
            self.done = true;
            None
        }
    }
}

pub struct SkipWhile<S: ZStream, F> {
    source: S,
    pred: F,
    skipping: bool,
}

impl<S: ZStream, F: FnMut(&S::Item) -> bool> ZStream for SkipWhile<S, F> {
    type Item = S::Item;

    #[inline]
    fn next_item(&mut self) -> Option<S::Item> {
        loop {
            let item = self.source.next_item()?;
            if self.skipping && (self.pred)(&item) {
                continue;
            }
            self.skipping = false;
            return Some(item);
        }
    }
}

// ---------------------------------------------------------------------------
// Operator: Enumerate
// ---------------------------------------------------------------------------

pub struct Enumerate<S: ZStream> {
    source: S,
    index: usize,
}

impl<S: ZStream> ZStream for Enumerate<S> {
    type Item = (usize, S::Item);

    #[inline(always)]
    fn next_item(&mut self) -> Option<(usize, S::Item)> {
        let item = self.source.next_item()?;
        let i = self.index;
        self.index += 1;
        Some((i, item))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.source.size_hint()
    }
}

// ---------------------------------------------------------------------------
// Operator: Zip
// ---------------------------------------------------------------------------

pub struct Zip<A: ZStream, B: ZStream> {
    a: A,
    b: B,
}

impl<A: ZStream, B: ZStream> ZStream for Zip<A, B> {
    type Item = (A::Item, B::Item);

    #[inline(always)]
    fn next_item(&mut self) -> Option<(A::Item, B::Item)> {
        Some((self.a.next_item()?, self.b.next_item()?))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (la, ua) = self.a.size_hint();
        let (lb, ub) = self.b.size_hint();
        let lower = la.min(lb);
        let upper = match (ua, ub) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        (lower, upper)
    }
}

// ---------------------------------------------------------------------------
// Operator: Chain
// ---------------------------------------------------------------------------

pub struct Chain<A: ZStream, B: ZStream<Item = A::Item>> {
    first: A,
    second: B,
    first_done: bool,
}

impl<A: ZStream, B: ZStream<Item = A::Item>> ZStream for Chain<A, B> {
    type Item = A::Item;

    #[inline(always)]
    fn next_item(&mut self) -> Option<A::Item> {
        if !self.first_done {
            if let Some(item) = self.first.next_item() {
                return Some(item);
            }
            self.first_done = true;
        }
        self.second.next_item()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (la, ua) = self.first.size_hint();
        let (lb, ub) = self.second.size_hint();
        let lower = la.saturating_add(lb);
        let upper = ua.zip(ub).map(|(a, b)| a.saturating_add(b));
        (lower, upper)
    }
}

// ---------------------------------------------------------------------------
// Operator: FlatMap
// ---------------------------------------------------------------------------

pub struct FlatMap<S: ZStream, F, Inner: ZStream> {
    source: S,
    f: F,
    current: Option<Inner>,
}

impl<S, F, Inner> ZStream for FlatMap<S, F, Inner>
where
    S: ZStream,
    Inner: ZStream,
    F: FnMut(S::Item) -> Inner,
{
    type Item = Inner::Item;

    #[inline]
    fn next_item(&mut self) -> Option<Inner::Item> {
        loop {
            if let Some(ref mut inner) = self.current {
                if let Some(item) = inner.next_item() {
                    return Some(item);
                }
            }
            let outer = self.source.next_item()?;
            self.current = Some((self.f)(outer));
        }
    }
}

// ---------------------------------------------------------------------------
// Type erasure — used only at the PyO3 boundary
// ---------------------------------------------------------------------------

/// Concrete wrapper that erases the stream type.
pub struct ErasedStream<T> {
    inner: Box<dyn ErasedStreamInner<Item = T>>,
    _marker: PhantomData<T>,
}

trait ErasedStreamInner: Send {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);
}

impl<S: ZStream + Send + 'static> ErasedStreamInner for S {
    type Item = S::Item;
    fn next(&mut self) -> Option<S::Item> {
        self.next_item()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        ZStream::size_hint(self)
    }
}

impl<T> ZStream for ErasedStream<T> {
    type Item = T;

    #[inline]
    fn next_item(&mut self) -> Option<T> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}
