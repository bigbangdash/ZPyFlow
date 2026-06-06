//! Typed numeric pipeline — the fast path that can release the GIL.
//!
//! ## Design: truly lazy, ZLinq-style
//!
//! ZLinq's key insight: operators must not execute anything until a terminal
//! call (ToArray, Sum, Count…).  Each operator just appends to an `ops` list;
//! the single fused pass runs only at `execute()`.
//!
//! We replicate this with `Arc<Vec<f64>>`: the source data is shared without
//! copying across every `.filter()` / `.map()` call.  Only `execute()` reads
//! it, applies every op in one scan, and writes to a single output Vec.
//!
//! Analogy to ZLinq interfaces:
//!   NumericOp list   ≈  nested struct chain `Where<Select<FromArray<T>,…>,…>`
//!   execute()        ≈  terminal `Sum()` / `ToArray()` triggering TryGetNext loop
//!   Arc<Vec<f64>>    ≈  `ReadOnlySpan<T>` borrow (TryGetSpan → SIMD path)

use super::simd;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Expression DSL — encodes operations as data, enabling Rust-side execution
// ---------------------------------------------------------------------------

/// A single numeric transformation or predicate.
/// These are the operations we can fuse and optionally vectorize.
#[derive(Debug, Clone, PartialEq)]
pub enum NumericOp {
    // Filters
    FilterGt(f64),
    FilterGe(f64),
    FilterLt(f64),
    FilterLe(f64),
    FilterEq(f64),
    FilterNe(f64),
    FilterBetween(f64, f64), // inclusive
    FilterNotNan,
    FilterNan,
    FilterFinite,
    FilterInf,            // x == x: passes when NOT NaN (arrow null-drop pattern)

    // Scalar maps
    MapMulScalar(f64),
    MapAddScalar(f64),
    MapSubScalar(f64),
    MapDivScalar(f64),
    MapPowScalar(f64),
    MapMod(f64),
    MapFloorDiv(f64),

    // Unary maps
    MapAbs,
    MapNeg,
    MapSqrt,
    MapFloor,
    MapCeil,
    MapRound,
    MapReciprocal,
    MapLog,
    MapLog2,
    MapLog10,
    MapExp,
    MapSigmoid,

    // Binary maps
    MapClamp(f64, f64),

    // Control flow
    Take(usize),
    Skip(usize),
}

#[derive(Debug, Clone, Copy)]
enum FilterKind {
    Gt(f64),
    Ge(f64),
    Lt(f64),
    Le(f64),
    Between(f64, f64),
}

fn detect_sole_filter_kind(ops: &[NumericOp]) -> Option<FilterKind> {
    if ops.len() != 1 {
        return None;
    }
    match ops[0] {
        NumericOp::FilterGt(t) => Some(FilterKind::Gt(t)),
        NumericOp::FilterGe(t) => Some(FilterKind::Ge(t)),
        NumericOp::FilterLt(t) => Some(FilterKind::Lt(t)),
        NumericOp::FilterLe(t) => Some(FilterKind::Le(t)),
        NumericOp::FilterBetween(lo, hi) => Some(FilterKind::Between(lo, hi)),
        _ => None,
    }
}

fn ops_are_filters_only(ops: &[NumericOp]) -> bool {
    ops.iter().all(|op| {
        matches!(
            op,
            NumericOp::FilterGt(_)
                | NumericOp::FilterGe(_)
                | NumericOp::FilterLt(_)
                | NumericOp::FilterLe(_)
                | NumericOp::FilterEq(_)
                | NumericOp::FilterNe(_)
                | NumericOp::FilterBetween(_, _)
                | NumericOp::FilterNotNan
                | NumericOp::FilterNan
                | NumericOp::FilterFinite
                | NumericOp::FilterInf
        )
    })
}

fn compute_stats_single_filter_simd_f64(data: &[f64], kind: FilterKind) -> Option<(usize, f64, f64, f64)> {
    match kind {
        FilterKind::Gt(t) => simd::simd_filter_stats_gt(data, t),
        FilterKind::Ge(t) => simd::simd_filter_stats_ge(data, t),
        FilterKind::Lt(t) => simd::simd_filter_stats_lt(data, t),
        FilterKind::Le(t) => simd::simd_filter_stats_le(data, t),
        FilterKind::Between(lo, hi) => simd::simd_filter_stats_between(data, lo, hi),
    }
}

/// A fused sequence of numeric operations over a f64 array.
///
/// # Lazy evaluation (ZLinq-equivalent)
///
/// `push_op()` / `filter()` / `map()` only append to `self.ops`.
/// No data is read until `execute()` or `execute_parallel()` is called.
///
/// # Zero intermediate allocation
///
/// `data` is stored behind `Arc` — every chained `PyQuery` shares the same
/// backing Vec without copying it.  The `ops` Vec is cheap to clone (it holds
/// enums, not the data).  Result is written to a single output `Vec` in one
/// scan at terminal time.
pub struct NumericPipeline {
    data: Arc<Vec<f64>>, // shared, never cloned for ops-only changes
    ops: Vec<NumericOp>,
}

impl NumericPipeline {
    pub fn new(data: Vec<f64>) -> Self {
        NumericPipeline {
            data: Arc::new(data),
            ops: Vec::new(),
        }
    }

    /// Build from an existing Arc (zero copy when branching pipelines).
    pub fn from_arc(data: Arc<Vec<f64>>) -> Self {
        NumericPipeline {
            data,
            ops: Vec::new(),
        }
    }

    pub fn with_ops(mut self, ops: Vec<NumericOp>) -> Self {
        self.ops = ops;
        self
    }

    /// Append one op — the only mutation; data is never touched.
    ///
    /// Adjacent ops of compatible kinds are merged in-place at build time:
    /// - Scalar maps: `MapMulScalar(2) + MapMulScalar(3)` → `MapMulScalar(6)`
    /// - Same-side filters: `FilterGt(5) + FilterGt(10)` → `FilterGt(10)` (tighter wins)
    /// - Opposing bounds: `FilterGe(lo) + FilterLe(hi)` → `FilterBetween(lo, hi)`
    /// - Idempotent flags: `FilterNotNan + FilterNotNan` → `FilterNotNan`
    ///
    /// Merges are only attempted when the last op and the new op are both filters
    /// (or both maps of the same kind), so map→filter and filter→map sequences
    /// are never incorrectly fused.
    pub fn push_op(mut self, op: NumericOp) -> Self {
        use NumericOp::*;
        let folded = match (self.ops.last_mut(), &op) {
            // Scalar map folding (existing)
            (Some(MapMulScalar(a)), MapMulScalar(b)) => { *a *= b; true }
            (Some(MapAddScalar(a)), MapAddScalar(b)) => { *a += b; true }
            (Some(MapSubScalar(a)), MapSubScalar(b)) => { *a += b; true }
            (Some(MapDivScalar(a)), MapDivScalar(b)) => { *a *= b; true }
            // Same-side tightening: keep the stricter bound
            (Some(FilterGt(a)), FilterGt(b)) => { *a = a.max(*b); true }
            (Some(FilterGe(a)), FilterGe(b)) => { *a = a.max(*b); true }
            (Some(FilterLt(a)), FilterLt(b)) => { *a = a.min(*b); true }
            (Some(FilterLe(a)), FilterLe(b)) => { *a = a.min(*b); true }
            // Opposing bounds: Ge(lo) + Le(hi) → Between(lo, hi)
            // Between semantics: lo <= x <= hi; when lo > hi no element passes (correct empty).
            (Some(last @ FilterGe(_)), FilterLe(hi)) => {
                if let FilterGe(lo) = last { *last = FilterBetween(*lo, *hi); }
                true
            }
            (Some(last @ FilterLe(_)), FilterGe(lo)) => {
                if let FilterLe(hi) = last { *last = FilterBetween(*lo, *hi); }
                true
            }
            // Idempotent flags
            (Some(FilterNotNan), FilterNotNan) => true,
            (Some(FilterNan),    FilterNan)    => true,
            (Some(FilterFinite), FilterFinite) => true,
            (Some(FilterInf),    FilterInf)    => true,
            _ => false,
        };
        if !folded {
            self.ops.push(op);
        }
        self
    }

    /// Cheap: clone the Arc pointer (8 bytes + refcount bump), not the data.
    pub fn arc(&self) -> Arc<Vec<f64>> {
        Arc::clone(&self.data)
    }

    /// Clone only the ops list (small, O(ops) not O(data)).
    pub fn clone_ops(&self) -> Vec<NumericOp> {
        self.ops.clone()
    }

    /// For callers that still need a Vec<f64> (e.g. SIMD path hand-off).
    /// Tries to unwrap the Arc first (zero copy if we're the sole owner).
    pub fn into_data_vec(self) -> Vec<f64> {
        Arc::try_unwrap(self.data).unwrap_or_else(|arc| (*arc).clone())
    }

    /// Execute all ops in a single fused pass — equivalent to ZLinq's terminal
    /// `ToArray()` / `Sum()` triggering the `TryGetNext` loop through the
    /// nested struct chain.
    ///
    /// Reads `self.data` once.  No intermediate Vec is created.
    /// SIMD path is chosen automatically when all ops qualify.
    pub fn execute(self) -> Vec<f64> {
        execute_fused_f64(&self.data, &self.ops)
    }

    #[cfg(feature = "parallel")]
    pub fn execute_parallel(self) -> Vec<f64> {
        let (out_skip, out_take) = extract_skip_take_from_ops(&self.ops);
        self.execute_parallel_with_skip_take(out_skip, out_take)
    }

    #[cfg(feature = "parallel")]
    pub fn execute_parallel_with_skip_take(self, out_skip: usize, out_take: Option<usize>) -> Vec<f64> {
        use rayon::prelude::*;

        let ops = Arc::new(self.ops); // share ops across rayon threads
        let data = self.data; // Arc pointer clone, not data clone

        if !ops_contain_filter(ops.as_ref()) {
            let (start, end) = compute_output_slice_range(data.len(), out_skip, out_take);
            return data[start..end]
                .par_iter()
                .copied()
                .map(|val| {
                    let mut v = val;
                    for op in ops.iter() {
                        if let ScalarResult::Value(new_v) = apply_scalar_op(v, op) {
                            v = new_v;
                        }
                    }
                    v
                })
                .collect();
        }

        let mut result: Vec<f64> = data
            .par_iter()
            .copied()
            .filter_map(|val| {
                let mut v = val;
                for op in ops.iter() {
                    match apply_scalar_op(v, op) {
                        ScalarResult::Value(new_v) => v = new_v,
                        ScalarResult::Filtered => return None,
                    }
                }
                Some(v)
            })
            .collect();

        let start = out_skip.min(result.len());
        if start > 0 {
            result.drain(..start);
        }
        if let Some(n) = out_take {
            result.truncate(n);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Shared fused execution kernel (used by execute() and query.rs)
// ---------------------------------------------------------------------------

/// Accumulate a new `Take(n)` into the current minimum.
#[inline]
fn combine_take_bounds(current: Option<usize>, n: usize) -> Option<usize> {
    Some(current.map_or(n, |prev| prev.min(n)))
}

/// Scan f64 ops for Skip/Take bounds — O(ops) time, zero allocation.
/// Skip/Take remain in the ops slice; the hot loop treats them as neutral.
fn extract_skip_take_from_ops(ops: &[NumericOp]) -> (usize, Option<usize>) {
    let mut skip: usize = 0;
    let mut take: Option<usize> = None;
    for op in ops {
        match op {
            NumericOp::Skip(n) => skip += n,
            NumericOp::Take(n) => take = combine_take_bounds(take, *n),
            _ => {}
        }
    }
    (skip, take)
}

/// Fold consecutive scalar map ops of the same kind into a single op.
///
/// E.g. `MapMulScalar(2.0) → MapMulScalar(3.0)` becomes `MapMulScalar(6.0)`.
/// Filters and non-collapsible ops pass through unchanged.
/// Ops that cancel out (e.g. `MapAddScalar(0.0)`) are stripped.
pub fn collapse_ops(ops: &[NumericOp]) -> Vec<NumericOp> {
    if ops.len() <= 1 {
        return ops.to_vec();
    }
    let mut out: Vec<NumericOp> = Vec::with_capacity(ops.len());
    for op in ops {
        match (out.last_mut(), op) {
            (Some(NumericOp::MapMulScalar(a)), NumericOp::MapMulScalar(b)) => *a *= b,
            (Some(NumericOp::MapAddScalar(a)), NumericOp::MapAddScalar(b)) => *a += b,
            (Some(NumericOp::MapSubScalar(a)), NumericOp::MapSubScalar(b)) => *a += b,
            (Some(NumericOp::MapDivScalar(a)), NumericOp::MapDivScalar(b)) => *a *= b,
            _ => out.push(op.clone()),
        }
    }
    // Strip identity ops that result from collapsing
    out.retain(|op| match op {
        NumericOp::MapMulScalar(v) | NumericOp::MapDivScalar(v) => (*v - 1.0).abs() > f64::EPSILON,
        NumericOp::MapAddScalar(v) | NumericOp::MapSubScalar(v) => v.abs() > f64::EPSILON,
        _ => true,
    });
    out
}

/// Pure Rust API path — ops may contain Skip/Take embedded by `push_op`.
/// Bounds are extracted inline with zero allocation; ops are passed as-is to
/// `execute_fused_f64_with_skip_take`, where Skip/Take are handled as neutral ops.
pub fn execute_fused_f64(data: &[f64], ops: &[NumericOp]) -> Vec<f64> {
    let (src_skip, out_take) = extract_skip_take_from_ops(ops);
    execute_fused_f64_with_skip_take(data, ops, src_skip, out_take)
}

/// Core execution kernel — used by both the Python and pure Rust paths.
///
/// `out_skip` elements that pass all ops are discarded before collecting;
/// `out_take` caps the output length.  Both bounds are output-level (applied
/// after all filters/maps), matching the semantics of `.skip(n).take(m)` in
/// a lazy pipeline where the user expects skip to follow any preceding filter.
pub fn execute_fused_f64_with_skip_take(
    data: &[f64],
    ops: &[NumericOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> Vec<f64> {
    // Pagination hot path: if no filter changes cardinality, apply skip/take
    // to the source slice before maps. This avoids scanning skipped items.
    if !ops_contain_filter(ops) {
        let (start, end) = compute_output_slice_range(data.len(), out_skip, out_take);
        let window = &data[start..end];
        if pipeline_is_simd_eligible(ops) {
            return simd::execute_simd_pipeline(window, ops);
        }

        let mut out = Vec::with_capacity(window.len());
        for &raw in window {
            let mut val = raw;
            for op in ops {
                if let ScalarResult::Value(v) = apply_scalar_op(val, op) {
                    val = v;
                }
            }
            out.push(val);
        }
        return out;
    }

    // SIMD path: only when there is no take limit.
    // With take, the scalar path breaks early and is faster when take << data.len().
    if pipeline_is_simd_eligible(ops) && out_take.is_none() {
        let mut result = simd::execute_simd_pipeline(data, ops);
        let start = out_skip.min(result.len());
        if start > 0 {
            result.drain(..start);
        }
        return result;
    }

    // Scalar fused path — one pass, zero intermediate allocations.
    let est = out_take.unwrap_or(data.len() / 2 + 1);
    let mut out = Vec::with_capacity(est);
    let mut skipped = 0usize;

    'outer: for &raw in data {
        let mut val = raw;
        for op in ops {
            match apply_scalar_op(val, op) {
                ScalarResult::Value(v) => val = v,
                ScalarResult::Filtered => continue 'outer,
            }
        }
        if skipped < out_skip {
            skipped += 1;
            continue;
        }
        out.push(val);
        if out_take.is_some_and(|n| out.len() >= n) {
            break;
        }
    }
    out
}

/// Count elements that pass all ops — never allocates an output Vec.
/// Uses SIMD for the common case of a single filter op with no skip/take.
pub fn count_fused_f64_with_skip_take(
    data: &[f64],
    ops: &[NumericOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> usize {
    // SIMD fast path: single filter op, no skip/take
    if out_skip == 0 && out_take.is_none() && ops.len() == 1 {
        match ops[0] {
            NumericOp::FilterGt(t) => return simd::simd_count_gt(data, t),
            NumericOp::FilterGe(t) => return simd::simd_count_ge(data, t),
            NumericOp::FilterLt(t) => return simd::simd_count_lt(data, t),
            NumericOp::FilterLe(t) => return simd::simd_count_le(data, t),
            NumericOp::FilterBetween(l, h) => return simd::simd_count_between(data, l, h),
            _ => {}
        }
    }
    // Scalar fallback
    let mut count = 0usize;
    let mut skipped = 0usize;
    'outer: for &raw in data {
        let mut val = raw;
        for op in ops {
            match apply_scalar_op(val, op) {
                ScalarResult::Value(v) => val = v,
                ScalarResult::Filtered => continue 'outer,
            }
        }
        if skipped < out_skip {
            skipped += 1;
            continue;
        }
        count += 1;
        if out_take.is_some_and(|n| count >= n) {
            break;
        }
    }
    count
}

/// Fused filter+sum: single SIMD pass, no intermediate Vec allocation.
/// Only handles single filter ops; falls back to execute+sum for complex pipelines.
pub fn filter_sum_fused_f64(data: &[f64], ops: &[NumericOp]) -> Option<f64> {
    if ops.len() != 1 {
        return None;
    }
    Some(match ops[0] {
        NumericOp::FilterGt(t) => simd::simd_filter_sum_gt(data, t),
        NumericOp::FilterGe(t) => simd::simd_filter_sum_ge(data, t),
        NumericOp::FilterLt(t) => simd::simd_filter_sum_lt(data, t),
        NumericOp::FilterLe(t) => simd::simd_filter_sum_le(data, t),
        NumericOp::FilterBetween(l, h) => simd::simd_filter_sum_between(data, l, h),
        _ => return None,
    })
}

/// Fused filter+mean: single SIMD pass, no intermediate Vec.
/// Returns None when the op is not supported (caller falls back).
/// Returns Some(None) when no element passes the filter.
/// Returns Some(Some(mean)) on success.
pub fn filter_mean_fused_f64(data: &[f64], ops: &[NumericOp]) -> Option<Option<f64>> {
    if ops.len() != 1 {
        return None;
    }
    Some(match ops[0] {
        NumericOp::FilterGt(t) => simd::simd_filter_mean_gt(data, t),
        NumericOp::FilterGe(t) => simd::simd_filter_mean_ge(data, t),
        NumericOp::FilterLt(t) => simd::simd_filter_mean_lt(data, t),
        NumericOp::FilterLe(t) => simd::simd_filter_mean_le(data, t),
        NumericOp::FilterBetween(l, h) => simd::simd_filter_mean_between(data, l, h),
        _ => return None,
    })
}

/// Fused filter+variance (population, denominator N): single SIMD pass.
/// Returns None when the op is not supported; Some(None) when no element passes.
pub fn filter_var_fused_f64(data: &[f64], ops: &[NumericOp]) -> Option<Option<f64>> {
    if ops.len() != 1 {
        return None;
    }
    Some(match ops[0] {
        NumericOp::FilterGt(t) => simd::simd_filter_var_gt(data, t),
        NumericOp::FilterGe(t) => simd::simd_filter_var_ge(data, t),
        NumericOp::FilterLt(t) => simd::simd_filter_var_lt(data, t),
        NumericOp::FilterLe(t) => simd::simd_filter_var_le(data, t),
        NumericOp::FilterBetween(l, h) => simd::simd_filter_var_between(data, l, h),
        _ => return None,
    })
}

/// Fused filter+max: single SIMD pass, no intermediate Vec.
/// Returns None when the op is not supported; Some(None) when no element passes.
pub fn filter_max_fused_f64(data: &[f64], ops: &[NumericOp]) -> Option<Option<f64>> {
    if ops.len() != 1 {
        return None;
    }
    Some(match ops[0] {
        NumericOp::FilterGt(t) => simd::simd_filter_max_gt(data, t),
        NumericOp::FilterGe(t) => simd::simd_filter_max_ge(data, t),
        NumericOp::FilterLt(t) => simd::simd_filter_max_lt(data, t),
        NumericOp::FilterLe(t) => simd::simd_filter_max_le(data, t),
        _ => return None,
    })
}

/// Fused filter+min: single SIMD pass, no intermediate Vec.
/// Returns None when the op is not supported; Some(None) when no element passes.
pub fn filter_min_fused_f64(data: &[f64], ops: &[NumericOp]) -> Option<Option<f64>> {
    if ops.len() != 1 {
        return None;
    }
    Some(match ops[0] {
        NumericOp::FilterGt(t) => simd::simd_filter_min_gt(data, t),
        NumericOp::FilterGe(t) => simd::simd_filter_min_ge(data, t),
        NumericOp::FilterLt(t) => simd::simd_filter_min_lt(data, t),
        NumericOp::FilterLe(t) => simd::simd_filter_min_le(data, t),
        _ => return None,
    })
}

/// Fused single-pass multi-stat: count + sum + min + max in one scan.
///
/// Returns `None` if no element passes the filter (min/max undefined).
/// Mean is `sum / count` — the caller can derive it without an extra pass.
///
/// Handles arbitrary op chains (not just single-filter); the hot loop applies
/// `eval_filter_f64` for each op per element so no intermediate Vec is created.
pub fn filter_multi_stat_f64(
    data: &[f64],
    ops: &[NumericOp],
) -> Option<(usize, f64, f64, f64)> {
    if let Some(kind) = detect_sole_filter_kind(ops) {
        return compute_stats_single_filter_simd_f64(data, kind);
    }

    let mut count = 0usize;
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    'outer: for &x in data {
        for op in ops {
            if !eval_filter_f64(x, op) {
                continue 'outer;
            }
        }
        count += 1;
        sum += x;
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    if count == 0 {
        None
    } else {
        Some((count, sum, min, max))
    }
}

/// Fused bounded stats over the final pipeline values.
///
/// Unlike `filter_multi_stat_f64`, this applies both filters and maps, and it
/// honors output-level skip/take. It never materializes the filtered output.
pub fn stats_fused_f64_with_skip_take(
    data: &[f64],
    ops: &[NumericOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> Option<(usize, f64, f64, f64)> {
    if out_skip == 0 && out_take.is_none() && ops_are_filters_only(ops) {
        return filter_multi_stat_f64(data, ops);
    }

    let mut count = 0usize;
    let mut skipped = 0usize;
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    'outer: for &raw in data {
        let mut val = raw;
        for op in ops {
            match apply_scalar_op(val, op) {
                ScalarResult::Value(v) => val = v,
                ScalarResult::Filtered => continue 'outer,
            }
        }
        if skipped < out_skip {
            skipped += 1;
            continue;
        }
        count += 1;
        sum += val;
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
        if out_take.is_some_and(|n| count >= n) {
            break;
        }
    }

    if count == 0 {
        None
    } else {
        Some((count, sum, min, max))
    }
}

/// Fused bounded sum over the final pipeline values. Never allocates.
pub fn sum_fused_f64_with_skip_take(
    data: &[f64],
    ops: &[NumericOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> f64 {
    if out_skip == 0 && out_take.is_none() {
        if let Some(s) = filter_sum_fused_f64(data, ops) {
            return s;
        }
        if ops.is_empty() {
            return simd::simd_sum_f64(data);
        }
    }
    stats_fused_f64_with_skip_take(data, ops, out_skip, out_take)
        .map(|(_, sum, _, _)| sum)
        .unwrap_or(0.0)
}

/// Fused bounded mean over the final pipeline values. Never allocates.
pub fn mean_fused_f64_with_skip_take(
    data: &[f64],
    ops: &[NumericOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> Option<f64> {
    if out_skip == 0 && out_take.is_none() {
        if let Some(mean) = filter_mean_fused_f64(data, ops) {
            return mean;
        }
    }
    stats_fused_f64_with_skip_take(data, ops, out_skip, out_take)
        .map(|(count, sum, _, _)| sum / count as f64)
}

/// Fused bounded population variance over the final pipeline values.
///
/// Uses two direct scans to match the existing mean-then-ssq behavior without
/// materializing an intermediate Vec.
pub fn var_fused_f64_with_skip_take(
    data: &[f64],
    ops: &[NumericOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> Option<f64> {
    if out_skip == 0 && out_take.is_none() {
        if let Some(var) = filter_var_fused_f64(data, ops) {
            return var;
        }
    }
    let (count, sum, _, _) = stats_fused_f64_with_skip_take(data, ops, out_skip, out_take)?;
    let mean = sum / count as f64;
    let mut seen = 0usize;
    let mut skipped = 0usize;
    let mut ssq = 0.0f64;

    'outer: for &raw in data {
        let mut val = raw;
        for op in ops {
            match apply_scalar_op(val, op) {
                ScalarResult::Value(v) => val = v,
                ScalarResult::Filtered => continue 'outer,
            }
        }
        if skipped < out_skip {
            skipped += 1;
            continue;
        }
        let d = val - mean;
        ssq += d * d;
        seen += 1;
        if out_take.is_some_and(|n| seen >= n) {
            break;
        }
    }

    Some(ssq / count as f64)
}

/// Fused bounded minimum over the final pipeline values. Never allocates.
pub fn min_fused_f64_with_skip_take(
    data: &[f64],
    ops: &[NumericOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> Option<f64> {
    if out_skip == 0 && out_take.is_none() {
        if let Some(min) = filter_min_fused_f64(data, ops) {
            return min;
        }
    }
    stats_fused_f64_with_skip_take(data, ops, out_skip, out_take).map(|(_, _, min, _)| min)
}

/// Fused bounded maximum over the final pipeline values. Never allocates.
pub fn max_fused_f64_with_skip_take(
    data: &[f64],
    ops: &[NumericOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> Option<f64> {
    if out_skip == 0 && out_take.is_none() {
        if let Some(max) = filter_max_fused_f64(data, ops) {
            return max;
        }
        if ops.is_empty() {
            return simd::simd_max_f64(data);
        }
    }
    stats_fused_f64_with_skip_take(data, ops, out_skip, out_take).map(|(_, _, _, max)| max)
}

/// Count variant for the i64 fast path.
/// Uses SIMD for the common case of a single filter op with no skip/take.
pub fn count_fused_i64_with_skip_take(
    data: &[i64],
    ops: &[IntOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> usize {
    // SIMD fast path: single filter op, no skip/take
    if out_skip == 0 && out_take.is_none() && ops.len() == 1 {
        match ops[0] {
            IntOp::FilterGt(t) => return simd::simd_count_i64_gt(data, t),
            IntOp::FilterGe(t) => return simd::simd_count_i64_ge(data, t),
            IntOp::FilterLt(t) => return simd::simd_count_i64_lt(data, t),
            IntOp::FilterLe(t) => return simd::simd_count_i64_le(data, t),
            _ => {}
        }
    }
    // Scalar fallback
    let mut count = 0usize;
    let mut skipped = 0usize;
    'outer: for &raw in data {
        let mut val = raw;
        for op in ops {
            match apply_int_op(val, op) {
                Some(v) => val = v,
                None => continue 'outer,
            }
        }
        if skipped < out_skip {
            skipped += 1;
            continue;
        }
        count += 1;
        if out_take.is_some_and(|n| count >= n) {
            break;
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Per-element scalar application
// ---------------------------------------------------------------------------

pub(crate) enum ScalarResult {
    Value(f64),
    Filtered,
}

#[inline(always)]
pub(crate) fn apply_scalar_op(val: f64, op: &NumericOp) -> ScalarResult {
    match op {
        NumericOp::FilterGt(t) => {
            if val > *t {
                ScalarResult::Value(val)
            } else {
                ScalarResult::Filtered
            }
        }
        NumericOp::FilterGe(t) => {
            if val >= *t {
                ScalarResult::Value(val)
            } else {
                ScalarResult::Filtered
            }
        }
        NumericOp::FilterLt(t) => {
            if val < *t {
                ScalarResult::Value(val)
            } else {
                ScalarResult::Filtered
            }
        }
        NumericOp::FilterLe(t) => {
            if val <= *t {
                ScalarResult::Value(val)
            } else {
                ScalarResult::Filtered
            }
        }
        NumericOp::FilterEq(t) => {
            if (val - t).abs() < f64::EPSILON {
                ScalarResult::Value(val)
            } else {
                ScalarResult::Filtered
            }
        }
        NumericOp::FilterNe(t) => {
            if (val - t).abs() >= f64::EPSILON {
                ScalarResult::Value(val)
            } else {
                ScalarResult::Filtered
            }
        }
        NumericOp::FilterBetween(lo, hi) => {
            if val >= *lo && val <= *hi {
                ScalarResult::Value(val)
            } else {
                ScalarResult::Filtered
            }
        }
        NumericOp::FilterNotNan => {
            if !val.is_nan() { ScalarResult::Value(val) } else { ScalarResult::Filtered }
        }
        NumericOp::FilterNan => {
            if val.is_nan() { ScalarResult::Value(val) } else { ScalarResult::Filtered }
        }
        NumericOp::FilterFinite => {
            if val.is_finite() { ScalarResult::Value(val) } else { ScalarResult::Filtered }
        }
        NumericOp::FilterInf => {
            if val.is_infinite() { ScalarResult::Value(val) } else { ScalarResult::Filtered }
        }

        NumericOp::MapMulScalar(s) => ScalarResult::Value(val * s),
        NumericOp::MapAddScalar(s) => ScalarResult::Value(val + s),
        NumericOp::MapSubScalar(s) => ScalarResult::Value(val - s),
        NumericOp::MapDivScalar(s) => ScalarResult::Value(val / s),
        NumericOp::MapPowScalar(s) => ScalarResult::Value(val.powf(*s)),
        NumericOp::MapMod(s) => ScalarResult::Value(val % s),
        NumericOp::MapFloorDiv(s) => ScalarResult::Value((val / s).floor()),
        NumericOp::MapAbs => ScalarResult::Value(val.abs()),
        NumericOp::MapNeg => ScalarResult::Value(-val),
        NumericOp::MapSqrt => ScalarResult::Value(val.sqrt()),
        NumericOp::MapFloor => ScalarResult::Value(val.floor()),
        NumericOp::MapCeil => ScalarResult::Value(val.ceil()),
        NumericOp::MapRound => ScalarResult::Value(val.round()),
        NumericOp::MapReciprocal => ScalarResult::Value(1.0 / val),
        NumericOp::MapLog => ScalarResult::Value(val.ln()),
        NumericOp::MapLog2 => ScalarResult::Value(val.log2()),
        NumericOp::MapLog10 => ScalarResult::Value(val.log10()),
        NumericOp::MapExp => ScalarResult::Value(val.exp()),
        NumericOp::MapSigmoid => ScalarResult::Value(1.0 / (1.0 + (-val).exp())),
        NumericOp::MapClamp(lo, hi) => ScalarResult::Value(val.clamp(*lo, *hi)),

        // Skip/Take are bounds, not value transforms — pass the value through.
        // Bounds are enforced by the loop in execute_fused_f64_with_skip_take, not here.
        NumericOp::Take(_) | NumericOp::Skip(_) => ScalarResult::Value(val),
    }
}

/// Returns true when all *value* ops are amenable to SIMD execution.
/// Skip/Take are bounds, not value ops — they are treated as neutral here
/// so their presence does not incorrectly disable the SIMD path.
fn pipeline_is_simd_eligible(ops: &[NumericOp]) -> bool {
    ops.iter().all(|op| {
        matches!(
            op,
            NumericOp::FilterGt(_)
        | NumericOp::FilterGe(_)
        | NumericOp::FilterLt(_)
        | NumericOp::FilterLe(_)
        | NumericOp::FilterBetween(_, _)
        | NumericOp::MapMulScalar(_)
        | NumericOp::MapAddScalar(_)
        | NumericOp::MapSubScalar(_)
        | NumericOp::MapDivScalar(_)
        | NumericOp::MapPowScalar(_)
        | NumericOp::MapAbs
        | NumericOp::MapNeg
        | NumericOp::MapSqrt
        | NumericOp::MapFloor
        | NumericOp::MapCeil
        | NumericOp::MapRound
        | NumericOp::MapReciprocal
        | NumericOp::MapLog
        | NumericOp::MapLog2
        | NumericOp::MapLog10
        | NumericOp::MapExp
        | NumericOp::MapSigmoid
        | NumericOp::MapClamp(_, _)
        | NumericOp::MapMod(_)
        | NumericOp::MapFloorDiv(_)
        | NumericOp::Skip(_)
        | NumericOp::Take(_)
        )
    })
}

fn ops_contain_filter(ops: &[NumericOp]) -> bool {
    ops.iter().any(|op| {
        matches!(
            op,
            NumericOp::FilterGt(_)
                | NumericOp::FilterGe(_)
                | NumericOp::FilterLt(_)
                | NumericOp::FilterLe(_)
                | NumericOp::FilterEq(_)
                | NumericOp::FilterNe(_)
                | NumericOp::FilterBetween(_, _)
                | NumericOp::FilterNotNan
                | NumericOp::FilterNan
                | NumericOp::FilterFinite
                | NumericOp::FilterInf
        )
    })
}

fn compute_output_slice_range(len: usize, skip: usize, take: Option<usize>) -> (usize, usize) {
    let start = skip.min(len);
    let remaining = len - start;
    let count = take.unwrap_or(remaining).min(remaining);
    (start, start + count)
}

// ---------------------------------------------------------------------------
// F32 variant — f32 fast path (ML embeddings, feature arrays)
// ---------------------------------------------------------------------------

pub(crate) enum ScalarResult32 {
    Value(f32),
    Filtered,
}

#[inline(always)]
pub(crate) fn apply_scalar_op_f32(val: f32, op: &NumericOp) -> ScalarResult32 {
    match op {
        NumericOp::FilterGt(t) => {
            if val > *t as f32 { ScalarResult32::Value(val) } else { ScalarResult32::Filtered }
        }
        NumericOp::FilterGe(t) => {
            if val >= *t as f32 { ScalarResult32::Value(val) } else { ScalarResult32::Filtered }
        }
        NumericOp::FilterLt(t) => {
            if val < *t as f32 { ScalarResult32::Value(val) } else { ScalarResult32::Filtered }
        }
        NumericOp::FilterLe(t) => {
            if val <= *t as f32 { ScalarResult32::Value(val) } else { ScalarResult32::Filtered }
        }
        NumericOp::FilterEq(t) => {
            if (val - *t as f32).abs() < f32::EPSILON { ScalarResult32::Value(val) } else { ScalarResult32::Filtered }
        }
        NumericOp::FilterNe(t) => {
            if (val - *t as f32).abs() >= f32::EPSILON { ScalarResult32::Value(val) } else { ScalarResult32::Filtered }
        }
        NumericOp::FilterBetween(lo, hi) => {
            if val >= *lo as f32 && val <= *hi as f32 { ScalarResult32::Value(val) } else { ScalarResult32::Filtered }
        }
        NumericOp::FilterNotNan => {
            if !val.is_nan() { ScalarResult32::Value(val) } else { ScalarResult32::Filtered }
        }
        NumericOp::FilterNan => {
            if val.is_nan() { ScalarResult32::Value(val) } else { ScalarResult32::Filtered }
        }
        NumericOp::FilterFinite => {
            if val.is_finite() { ScalarResult32::Value(val) } else { ScalarResult32::Filtered }
        }
        NumericOp::FilterInf => {
            if val.is_infinite() { ScalarResult32::Value(val) } else { ScalarResult32::Filtered }
        }
        NumericOp::MapMulScalar(s) => ScalarResult32::Value(val * *s as f32),
        NumericOp::MapAddScalar(s) => ScalarResult32::Value(val + *s as f32),
        NumericOp::MapSubScalar(s) => ScalarResult32::Value(val - *s as f32),
        NumericOp::MapDivScalar(s) => ScalarResult32::Value(val / *s as f32),
        NumericOp::MapPowScalar(s) => ScalarResult32::Value(val.powf(*s as f32)),
        NumericOp::MapAbs => ScalarResult32::Value(val.abs()),
        NumericOp::MapNeg => ScalarResult32::Value(-val),
        NumericOp::MapSqrt => ScalarResult32::Value(val.sqrt()),
        NumericOp::MapFloor => ScalarResult32::Value(val.floor()),
        NumericOp::MapCeil => ScalarResult32::Value(val.ceil()),
        NumericOp::MapRound => ScalarResult32::Value(val.round()),
        NumericOp::MapReciprocal => ScalarResult32::Value(1.0 / val),
        NumericOp::MapLog => ScalarResult32::Value(val.ln()),
        NumericOp::MapLog2 => ScalarResult32::Value(val.log2()),
        NumericOp::MapLog10 => ScalarResult32::Value(val.log10()),
        NumericOp::MapExp => ScalarResult32::Value(val.exp()),
        NumericOp::MapSigmoid => ScalarResult32::Value(1.0 / (1.0 + (-val).exp())),
        NumericOp::MapClamp(lo, hi) => ScalarResult32::Value(val.clamp(*lo as f32, *hi as f32)),
        NumericOp::MapMod(s) => ScalarResult32::Value(val % *s as f32),
        NumericOp::MapFloorDiv(s) => ScalarResult32::Value((val / *s as f32).floor()),
        NumericOp::Take(_) | NumericOp::Skip(_) => ScalarResult32::Value(val),
    }
}

fn ops_contain_filter_f32(ops: &[NumericOp]) -> bool {
    ops_contain_filter(ops) // same predicate variants
}

fn pipeline_is_simd_f32_eligible(ops: &[NumericOp]) -> bool {
    ops.iter().all(|op| {
        matches!(
            op,
            NumericOp::FilterGt(_)
                | NumericOp::FilterGe(_)
                | NumericOp::FilterLt(_)
                | NumericOp::FilterLe(_)
                | NumericOp::FilterBetween(_, _)
                | NumericOp::MapMulScalar(_)
                | NumericOp::MapAddScalar(_)
                | NumericOp::MapSubScalar(_)
                | NumericOp::MapDivScalar(_)
                | NumericOp::MapAbs
                | NumericOp::MapNeg
                | NumericOp::MapSqrt
                | NumericOp::Skip(_)
                | NumericOp::Take(_)
        )
    })
}

/// Single-pass fused execution for f32 data (mirrors execute_fused_f64_with_skip_take).
pub fn execute_fused_f32_with_skip_take(
    data: &[f32],
    ops: &[NumericOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> Vec<f32> {
    if !ops_contain_filter_f32(ops) {
        let (start, end) = compute_output_slice_range(data.len(), out_skip, out_take);
        let window = &data[start..end];
        if pipeline_is_simd_f32_eligible(ops) {
            return simd::execute_simd_f32_pipeline(window, ops);
        }
        let mut out = Vec::with_capacity(window.len());
        for &raw in window {
            let mut val = raw;
            for op in ops {
                if let ScalarResult32::Value(v) = apply_scalar_op_f32(val, op) {
                    val = v;
                }
            }
            out.push(val);
        }
        return out;
    }

    if pipeline_is_simd_f32_eligible(ops) && out_take.is_none() {
        let mut result = simd::execute_simd_f32_pipeline(data, ops);
        let start = out_skip.min(result.len());
        if start > 0 {
            result.drain(..start);
        }
        return result;
    }

    let est = out_take.unwrap_or(data.len() / 2 + 1);
    let mut out = Vec::with_capacity(est);
    let mut skipped = 0usize;

    'outer: for &raw in data {
        let mut val = raw;
        for op in ops {
            match apply_scalar_op_f32(val, op) {
                ScalarResult32::Value(v) => val = v,
                ScalarResult32::Filtered => continue 'outer,
            }
        }
        if skipped < out_skip {
            skipped += 1;
            continue;
        }
        out.push(val);
        if out_take.is_some_and(|n| out.len() >= n) {
            break;
        }
    }
    out
}

/// Count elements passing all ops for f32 data.
pub fn count_fused_f32_with_skip_take(
    data: &[f32],
    ops: &[NumericOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> usize {
    if out_skip == 0 && out_take.is_none() && ops.len() == 1 {
        match ops[0] {
            NumericOp::FilterGt(t) => return simd::simd_count_f32_gt(data, t as f32),
            NumericOp::FilterGe(t) => return simd::simd_count_f32_ge(data, t as f32),
            NumericOp::FilterLt(t) => return simd::simd_count_f32_lt(data, t as f32),
            NumericOp::FilterLe(t) => return simd::simd_count_f32_le(data, t as f32),
            _ => {}
        }
    }
    let mut count = 0usize;
    let mut skipped = 0usize;
    'outer: for &raw in data {
        let mut val = raw;
        for op in ops {
            match apply_scalar_op_f32(val, op) {
                ScalarResult32::Value(v) => val = v,
                ScalarResult32::Filtered => continue 'outer,
            }
        }
        if skipped < out_skip {
            skipped += 1;
            continue;
        }
        count += 1;
        if out_take.is_some_and(|n| count >= n) {
            break;
        }
    }
    count
}

// ---------------------------------------------------------------------------
// I64 variant (integer fast path)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum IntOp {
    FilterGt(i64),
    FilterGe(i64),
    FilterLt(i64),
    FilterLe(i64),
    FilterEq(i64),
    FilterNe(i64),
    MapMulScalar(i64),
    MapAddScalar(i64),
    MapSubScalar(i64),
    MapAbs,
    MapNeg,
    Take(usize),
    Skip(usize),
}

/// Integer pipeline — same Arc-based lazy design as `NumericPipeline`.
/// `Arc<Vec<i64>>` means `.filter()` / `.map()` chains share the data
/// without O(N) cloning; only `execute()` reads the data once.
pub struct IntPipeline {
    data: Arc<Vec<i64>>,
    ops: Vec<IntOp>,
}

impl IntPipeline {
    pub fn new(data: Vec<i64>) -> Self {
        IntPipeline {
            data: Arc::new(data),
            ops: Vec::new(),
        }
    }

    pub fn from_arc(data: Arc<Vec<i64>>) -> Self {
        IntPipeline {
            data,
            ops: Vec::new(),
        }
    }

    pub fn with_ops(mut self, ops: Vec<IntOp>) -> Self {
        self.ops = ops;
        self
    }

    /// Same-side filter tightening, mirroring `NumericPipeline::push_op`.
    pub fn push_op(mut self, op: IntOp) -> Self {
        use IntOp::*;
        let folded = match (self.ops.last_mut(), &op) {
            (Some(FilterGt(a)), FilterGt(b)) => { *a = (*a).max(*b); true }
            (Some(FilterGe(a)), FilterGe(b)) => { *a = (*a).max(*b); true }
            (Some(FilterLt(a)), FilterLt(b)) => { *a = (*a).min(*b); true }
            (Some(FilterLe(a)), FilterLe(b)) => { *a = (*a).min(*b); true }
            (Some(MapMulScalar(a)), MapMulScalar(b)) => { *a = a.saturating_mul(*b); true }
            (Some(MapAddScalar(a)), MapAddScalar(b)) => { *a = a.saturating_add(*b); true }
            (Some(MapSubScalar(a)), MapSubScalar(b)) => { *a = a.saturating_add(*b); true }
            _ => false,
        };
        if !folded {
            self.ops.push(op);
        }
        self
    }

    /// Arc pointer clone — O(1), does not copy the Vec.
    pub fn arc(&self) -> Arc<Vec<i64>> {
        Arc::clone(&self.data)
    }

    pub fn clone_ops(&self) -> Vec<IntOp> {
        self.ops.clone()
    }

    /// Execute all ops in a single fused pass.
    /// Skip/Take embedded in ops are extracted before the loop (pure Rust API path).
    pub fn execute(self) -> Vec<i64> {
        let (skip_count, take_count, value_ops) = extract_skip_take_from_int_ops(self.ops);
        execute_fused_i64_with_skip_take(&self.data, &value_ops, skip_count, take_count)
    }

    #[cfg(feature = "parallel")]
    pub fn execute_parallel(self) -> Vec<i64> {
        let (skip_count, take_count, value_ops) = extract_skip_take_from_int_ops(self.ops);
        IntPipeline {
            data: self.data,
            ops: value_ops,
        }
        .execute_parallel_with_skip_take(skip_count, take_count)
    }

    #[cfg(feature = "parallel")]
    pub fn execute_parallel_with_skip_take(self, out_skip: usize, out_take: Option<usize>) -> Vec<i64> {
        use rayon::prelude::*;

        let ops = Arc::new(self.ops);
        let data = self.data;

        if !has_filter_i64(ops.as_ref()) {
            let (start, end) = compute_output_slice_range(data.len(), out_skip, out_take);
            return data[start..end]
                .par_iter()
                .copied()
                .map(|val| {
                    let mut v = val;
                    for op in ops.iter() {
                        if let Some(new_v) = apply_int_op(v, op) {
                            v = new_v;
                        }
                    }
                    v
                })
                .collect();
        }

        let mut result: Vec<i64> = data
            .par_iter()
            .copied()
            .filter_map(|val| {
                let mut v = val;
                for op in ops.iter() {
                    match apply_int_op(v, op) {
                        Some(new_v) => v = new_v,
                        None => return None,
                    }
                }
                Some(v)
            })
            .collect();

        let start = out_skip.min(result.len());
        if start > 0 {
            result.drain(..start);
        }
        if let Some(n) = out_take {
            result.truncate(n);
        }
        result
    }
}

fn extract_skip_take_from_int_ops(ops: Vec<IntOp>) -> (usize, Option<usize>, Vec<IntOp>) {
    let mut skip: usize = 0;
    let mut take: Option<usize> = None;
    let value_ops = ops
        .into_iter()
        .filter(|op| match op {
            IntOp::Take(n) => {
                take = combine_take_bounds(take, *n);
                false
            }
            IntOp::Skip(n) => {
                skip += n;
                false
            }
            _ => true,
        })
        .collect();
    (skip, take, value_ops)
}

/// Python path: out_skip and out_take are output-level bounds — no Skip/Take in ops.
pub fn execute_fused_i64_with_skip_take(
    data: &[i64],
    ops: &[IntOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> Vec<i64> {
    // Pagination hot path: with no filters, skip/take can be applied before
    // integer maps because maps do not change cardinality.
    if !has_filter_i64(ops) {
        let (start, end) = compute_output_slice_range(data.len(), out_skip, out_take);
        let window = &data[start..end];
        let mut out = Vec::with_capacity(window.len());
        for &raw in window {
            let mut val = raw;
            for op in ops {
                if let Some(v) = apply_int_op(val, op) {
                    val = v;
                }
            }
            out.push(val);
        }
        return out;
    }

    // SIMD fast path: single filter op, no skip/take
    if out_skip == 0 && out_take.is_none() && ops.len() == 1 {
        match ops[0] {
            IntOp::FilterGt(t) => return simd::simd_filter_i64_gt(data, t),
            IntOp::FilterGe(t) => return simd::simd_filter_i64_ge(data, t),
            IntOp::FilterLt(t) => return simd::simd_filter_i64_lt(data, t),
            IntOp::FilterLe(t) => return simd::simd_filter_i64_le(data, t),
            _ => {}
        }
    }
    // Scalar fused path
    let est = out_take.unwrap_or(data.len() / 2 + 1);
    let mut out = Vec::with_capacity(est);
    let mut skipped = 0usize;

    'outer: for &raw in data {
        let mut val = raw;
        for op in ops {
            match apply_int_op(val, op) {
                Some(v) => val = v,
                None => continue 'outer,
            }
        }
        if skipped < out_skip {
            skipped += 1;
            continue;
        }
        out.push(val);
        if out_take.is_some_and(|n| out.len() >= n) {
            break;
        }
    }
    out
}

fn has_filter_i64(ops: &[IntOp]) -> bool {
    ops.iter().any(|op| {
        matches!(
            op,
            IntOp::FilterGt(_)
                | IntOp::FilterGe(_)
                | IntOp::FilterLt(_)
                | IntOp::FilterLe(_)
                | IntOp::FilterEq(_)
                | IntOp::FilterNe(_)
        )
    })
}

#[inline(always)]
pub(crate) fn apply_int_op(val: i64, op: &IntOp) -> Option<i64> {
    match op {
        IntOp::FilterGt(t) => {
            if val > *t {
                Some(val)
            } else {
                None
            }
        }
        IntOp::FilterGe(t) => {
            if val >= *t {
                Some(val)
            } else {
                None
            }
        }
        IntOp::FilterLt(t) => {
            if val < *t {
                Some(val)
            } else {
                None
            }
        }
        IntOp::FilterLe(t) => {
            if val <= *t {
                Some(val)
            } else {
                None
            }
        }
        IntOp::FilterEq(t) => {
            if val == *t {
                Some(val)
            } else {
                None
            }
        }
        IntOp::FilterNe(t) => {
            if val != *t {
                Some(val)
            } else {
                None
            }
        }
        IntOp::MapMulScalar(s) => Some(val.wrapping_mul(*s)),
        IntOp::MapAddScalar(s) => Some(val.wrapping_add(*s)),
        IntOp::MapSubScalar(s) => Some(val.wrapping_sub(*s)),
        IntOp::MapAbs => Some(val.wrapping_abs()),
        IntOp::MapNeg => Some(val.wrapping_neg()),
        IntOp::Take(_) | IntOp::Skip(_) => Some(val),
    }
}

/// Evaluate a single filter op against one f64 value. True = passes the predicate.
#[inline(always)]
pub fn eval_filter_f64(x: f64, op: &NumericOp) -> bool {
    match op {
        NumericOp::FilterGt(t) => x > *t,
        NumericOp::FilterGe(t) => x >= *t,
        NumericOp::FilterLt(t) => x < *t,
        NumericOp::FilterLe(t) => x <= *t,
        NumericOp::FilterEq(t) => x == *t,
        NumericOp::FilterNe(t) => x != *t,
        NumericOp::FilterBetween(lo, hi) => x >= *lo && x <= *hi,
        NumericOp::FilterNotNan => !x.is_nan(),
        _ => true,
    }
}

/// Evaluate a single filter op against one i64 value. True = passes the predicate.
#[inline(always)]
pub fn eval_filter_i64(x: i64, op: &IntOp) -> bool {
    match op {
        IntOp::FilterGt(t) => x > *t,
        IntOp::FilterGe(t) => x >= *t,
        IntOp::FilterLt(t) => x < *t,
        IntOp::FilterLe(t) => x <= *t,
        IntOp::FilterEq(t) => x == *t,
        IntOp::FilterNe(t) => x != *t,
        _ => true,
    }
}

// ---------------------------------------------------------------------------
// Pure-Rust unit tests — runnable under Miri (no PyO3 FFI)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f64_pipeline(data: Vec<f64>, ops: Vec<NumericOp>) -> Vec<f64> {
        NumericPipeline::new(data).with_ops(ops).execute()
    }

    fn make_i64_pipeline(data: Vec<i64>, ops: Vec<IntOp>) -> Vec<i64> {
        IntPipeline::new(data).with_ops(ops).execute()
    }

    // --- f64 filter ops ---

    #[test]
    fn f64_filter_gt() {
        let result = make_f64_pipeline(vec![1.0, 2.0, 3.0, 4.0], vec![NumericOp::FilterGt(2.0)]);
        assert_eq!(result, vec![3.0, 4.0]);
    }

    #[test]
    fn f64_filter_ge() {
        let result = make_f64_pipeline(vec![1.0, 2.0, 3.0], vec![NumericOp::FilterGe(2.0)]);
        assert_eq!(result, vec![2.0, 3.0]);
    }

    #[test]
    fn f64_filter_lt() {
        let result = make_f64_pipeline(vec![1.0, 2.0, 3.0], vec![NumericOp::FilterLt(2.0)]);
        assert_eq!(result, vec![1.0]);
    }

    #[test]
    fn f64_filter_le() {
        let result = make_f64_pipeline(vec![1.0, 2.0, 3.0], vec![NumericOp::FilterLe(2.0)]);
        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn f64_filter_eq() {
        let result = make_f64_pipeline(vec![1.0, 2.0, 3.0], vec![NumericOp::FilterEq(2.0)]);
        assert_eq!(result, vec![2.0]);
    }

    #[test]
    fn f64_filter_ne() {
        let result = make_f64_pipeline(vec![1.0, 2.0, 3.0], vec![NumericOp::FilterNe(2.0)]);
        assert_eq!(result, vec![1.0, 3.0]);
    }

    #[test]
    fn f64_filter_between() {
        let result = make_f64_pipeline(
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![NumericOp::FilterBetween(1.0, 3.0)],
        );
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    // --- f64 map ops ---

    #[test]
    fn f64_map_mul_scalar() {
        let result = make_f64_pipeline(vec![1.0, 2.0, 3.0], vec![NumericOp::MapMulScalar(2.0)]);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn f64_map_add_scalar() {
        let result = make_f64_pipeline(vec![1.0, 2.0], vec![NumericOp::MapAddScalar(10.0)]);
        assert_eq!(result, vec![11.0, 12.0]);
    }

    #[test]
    fn f64_map_abs() {
        let result = make_f64_pipeline(vec![-1.0, 2.0, -3.0], vec![NumericOp::MapAbs]);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn f64_map_neg() {
        let result = make_f64_pipeline(vec![1.0, -2.0], vec![NumericOp::MapNeg]);
        assert_eq!(result, vec![-1.0, 2.0]);
    }

    #[test]
    fn f64_map_sqrt() {
        let result = make_f64_pipeline(vec![4.0, 9.0], vec![NumericOp::MapSqrt]);
        assert_eq!(result, vec![2.0, 3.0]);
    }

    // --- f64 skip / take ---

    #[test]
    fn f64_skip() {
        let result = make_f64_pipeline(vec![1.0, 2.0, 3.0, 4.0], vec![NumericOp::Skip(2)]);
        assert_eq!(result, vec![3.0, 4.0]);
    }

    #[test]
    fn f64_take() {
        let result = make_f64_pipeline(vec![1.0, 2.0, 3.0, 4.0], vec![NumericOp::Take(2)]);
        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn f64_filter_then_take() {
        let result = make_f64_pipeline(
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![NumericOp::FilterGt(0.0), NumericOp::Take(3)],
        );
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn f64_filter_stats_gt_uses_predicate_semantics() {
        let data = vec![f64::NAN, -1.0, 0.0, 1.0, 2.0, 3.0];
        let result = filter_multi_stat_f64(&data, &[NumericOp::FilterGt(0.0)]).unwrap();
        assert_eq!(result.0, 3);
        assert_eq!(result.1, 6.0);
        assert_eq!(result.2, 1.0);
        assert_eq!(result.3, 3.0);
    }

    #[test]
    fn f64_filter_stats_order_predicates_match_execute() {
        let data = vec![f64::NAN, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let cases = vec![
            NumericOp::FilterGt(0.0),
            NumericOp::FilterGe(0.0),
            NumericOp::FilterLt(1.0),
            NumericOp::FilterLe(1.0),
            NumericOp::FilterBetween(-1.0, 2.0),
        ];

        for op in cases {
            let values = execute_fused_f64_with_skip_take(&data, &[op.clone()], 0, None);
            let expected = if values.is_empty() {
                None
            } else {
                Some((
                    values.len(),
                    values.iter().sum::<f64>(),
                    values.iter().copied().reduce(f64::min).unwrap(),
                    values.iter().copied().reduce(f64::max).unwrap(),
                ))
            };
            assert_eq!(filter_multi_stat_f64(&data, &[op]), expected);
        }
    }

    #[test]
    fn f64_filter_stats_between_is_inclusive() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let result = filter_multi_stat_f64(&data, &[NumericOp::FilterBetween(1.0, 3.0)]).unwrap();
        assert_eq!(result, (3, 6.0, 1.0, 3.0));
    }

    #[test]
    fn f64_with_skip_take_stats_falls_back_for_skip_take() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let result = stats_fused_f64_with_skip_take(&data, &[NumericOp::FilterGe(1.0)], 1, Some(2)).unwrap();
        assert_eq!(result, (2, 5.0, 2.0, 3.0));
    }

    #[test]
    fn f64_count_single_filter_predicates_match_execute() {
        let data = vec![f64::NAN, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let cases = vec![
            NumericOp::FilterGt(0.0),
            NumericOp::FilterGe(0.0),
            NumericOp::FilterLt(1.0),
            NumericOp::FilterLe(1.0),
            NumericOp::FilterBetween(-1.0, 2.0),
        ];

        for op in cases {
            let values = execute_fused_f64_with_skip_take(&data, &[op.clone()], 0, None);
            assert_eq!(
                count_fused_f64_with_skip_take(&data, &[op], 0, None),
                values.len()
            );
        }
    }

    #[test]
    fn f64_sum_mean_single_filter_predicates_match_execute() {
        let data = vec![f64::NAN, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let cases = vec![
            NumericOp::FilterGt(0.0),
            NumericOp::FilterGe(0.0),
            NumericOp::FilterLt(1.0),
            NumericOp::FilterLe(1.0),
            NumericOp::FilterBetween(-1.0, 2.0),
        ];

        for op in cases {
            let values = execute_fused_f64_with_skip_take(&data, &[op.clone()], 0, None);
            let expected_sum = values.iter().sum::<f64>();
            let expected_mean = if values.is_empty() {
                None
            } else {
                Some(expected_sum / values.len() as f64)
            };

            assert_eq!(filter_sum_fused_f64(&data, &[op.clone()]), Some(expected_sum));
            assert_eq!(filter_mean_fused_f64(&data, &[op]), Some(expected_mean));
        }
    }

    // --- f64 chained ops ---

    #[test]
    fn f64_chained_filter_map() {
        let result = make_f64_pipeline(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![NumericOp::FilterGt(1.5), NumericOp::MapMulScalar(10.0)],
        );
        assert_eq!(result, vec![20.0, 30.0, 40.0]);
    }

    #[test]
    fn f64_empty_data() {
        let result = make_f64_pipeline(vec![], vec![NumericOp::FilterGt(0.0)]);
        assert!(result.is_empty());
    }

    #[test]
    fn f64_no_ops() {
        let result = make_f64_pipeline(vec![1.0, 2.0, 3.0], vec![]);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    // --- i64 filter ops ---

    #[test]
    fn i64_filter_gt() {
        let result = make_i64_pipeline(vec![1, 2, 3, 4], vec![IntOp::FilterGt(2)]);
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn i64_filter_eq() {
        let result = make_i64_pipeline(vec![1, 2, 3], vec![IntOp::FilterEq(2)]);
        assert_eq!(result, vec![2]);
    }

    #[test]
    fn i64_map_mul() {
        let result = make_i64_pipeline(vec![1, 2, 3], vec![IntOp::MapMulScalar(3)]);
        assert_eq!(result, vec![3, 6, 9]);
    }

    #[test]
    fn i64_map_abs() {
        let result = make_i64_pipeline(vec![-5, 3, -1], vec![IntOp::MapAbs]);
        assert_eq!(result, vec![5, 3, 1]);
    }

    #[test]
    fn i64_take() {
        let result = make_i64_pipeline(vec![10, 20, 30, 40], vec![IntOp::Take(2)]);
        assert_eq!(result, vec![10, 20]);
    }

    // --- eval_filter_* helpers ---

    #[test]
    fn eval_filter_f64_between_inclusive() {
        assert!(eval_filter_f64(1.0, &NumericOp::FilterBetween(1.0, 3.0)));
        assert!(eval_filter_f64(3.0, &NumericOp::FilterBetween(1.0, 3.0)));
        assert!(!eval_filter_f64(0.9, &NumericOp::FilterBetween(1.0, 3.0)));
        assert!(!eval_filter_f64(3.1, &NumericOp::FilterBetween(1.0, 3.0)));
    }

    #[test]
    fn eval_filter_i64_ne() {
        assert!(eval_filter_i64(5, &IntOp::FilterNe(3)));
        assert!(!eval_filter_i64(3, &IntOp::FilterNe(3)));
    }

    // ── parallel execution ────────────────────────────────────────────────────

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_f64_filter_matches_sequential() {
        let data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
        let ops = vec![NumericOp::FilterGt(5000.0)];

        let sequential = NumericPipeline::new(data.clone()).with_ops(ops.clone()).execute();
        let parallel = NumericPipeline::new(data).with_ops(ops).execute_parallel();

        assert_eq!(sequential.len(), parallel.len());
        // Order may differ after rayon; compare sorted results
        let mut seq_sorted = sequential;
        let mut par_sorted = parallel;
        seq_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        par_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(seq_sorted, par_sorted);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_f64_map_matches_sequential() {
        let data: Vec<f64> = (0..1_000).map(|i| i as f64).collect();
        let ops = vec![NumericOp::MapMulScalar(2.0)];

        let sequential = NumericPipeline::new(data.clone()).with_ops(ops.clone()).execute();
        let mut parallel = NumericPipeline::new(data).with_ops(ops).execute_parallel();

        // rayon map-only path preserves relative order within chunks but not globally;
        // sort both for a stable comparison
        let mut seq_sorted = sequential;
        seq_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        parallel.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(seq_sorted, parallel);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_i64_filter_matches_sequential() {
        let data: Vec<i64> = (0..10_000).collect();
        let ops = vec![IntOp::FilterGt(5000)];

        let sequential = IntPipeline::new(data.clone()).with_ops(ops.clone()).execute();
        let parallel = IntPipeline::new(data).with_ops(ops).execute_parallel();

        let mut seq_sorted = sequential;
        let mut par_sorted = parallel;
        seq_sorted.sort();
        par_sorted.sort();
        assert_eq!(seq_sorted, par_sorted);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_empty_input() {
        let parallel = NumericPipeline::new(vec![]).with_ops(vec![NumericOp::FilterGt(0.0)]).execute_parallel();
        assert!(parallel.is_empty());
    }
}
