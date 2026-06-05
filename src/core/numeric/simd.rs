//! SIMD-accelerated numeric operations using the `wide` crate.
//!
//! `wide` provides stable (non-nightly) SIMD through explicit vector types:
//!   f64x4  → 4 × f64 in 256-bit AVX register
//!   f32x8  → 8 × f32 in 256-bit AVX register
//!   i64x4  → 4 × i64 in 256-bit AVX register
//!
//! The pattern for every SIMD operation:
//!   1. Process as many 4-wide (or 8-wide) chunks as possible with SIMD.
//!   2. Handle the tail (<4 elements) with scalar code.
//!   3. Never allocate intermediate Vecs — accumulate directly into output.
//!
//! Filtering with SIMD is inherently awkward because the output length is
//! variable.  We use a write-pointer approach: write into a pre-allocated
//! output buffer and track how many elements were written.

use super::pipeline::NumericOp;
use wide::{f32x8, f64x4, i64x4, CmpEq, CmpGe, CmpGt, CmpLe, CmpLt};

// ---------------------------------------------------------------------------
// Public entry point — dispatch to SIMD sub-operations
// ---------------------------------------------------------------------------

/// Execute a sequence of numeric ops using SIMD where possible.
/// Caller guarantees that `can_use_simd_path` returned true for `ops`.
pub fn execute_simd_pipeline(data: &[f64], ops: &[NumericOp]) -> Vec<f64> {
    if ops.is_empty() {
        return data.to_vec();
    }

    // Find the first filter op.  Skip/Take are neutral bounds handled by the
    // caller; map ops are "value ops" that require the data to be copied first.
    let first_filter = ops.iter().position(|op| {
        matches!(
            op,
            NumericOp::FilterGt(_)
                | NumericOp::FilterGe(_)
                | NumericOp::FilterLt(_)
                | NumericOp::FilterLe(_)
                | NumericOp::FilterBetween(_, _)
        )
    });

    let Some(fi) = first_filter else {
        // Pure map chain — one copy, applied in-place.
        let mut result = data.to_vec();
        flush_maps(&mut result, &collect_map_ops(ops));
        return result;
    };

    // Check whether any map op precedes the first filter.
    let has_leading_maps = ops[..fi].iter().any(|op| to_map_op(op).is_some());

    // Build the initial filtered result.
    // Filter-first: filter directly from the input slice — no upfront full copy.
    // This halves allocations (1 instead of 2) for the common filter-only case.
    let mut result = if has_leading_maps {
        let mut r = data.to_vec();
        flush_maps(&mut r, &collect_map_ops(&ops[..fi]));
        apply_filter_op(&r, &ops[fi])
    } else {
        apply_filter_op(data, &ops[fi])
    };

    // Apply remaining ops on the now-smaller result.
    let mut pending_maps: Vec<MapOp> = Vec::new();
    for op in &ops[fi + 1..] {
        match op {
            NumericOp::FilterGt(_)
            | NumericOp::FilterGe(_)
            | NumericOp::FilterLt(_)
            | NumericOp::FilterLe(_)
            | NumericOp::FilterBetween(_, _) => {
                flush_maps(&mut result, &pending_maps);
                pending_maps.clear();
                result = apply_filter_op(&result, op);
            }
            _ => {
                if let Some(m) = to_map_op(op) {
                    pending_maps.push(m);
                }
            }
        }
    }
    if !pending_maps.is_empty() {
        flush_maps(&mut result, &pending_maps);
    }
    result
}

fn apply_filter_op(data: &[f64], op: &NumericOp) -> Vec<f64> {
    match op {
        NumericOp::FilterGt(t) => simd_filter_gt(data, *t),
        NumericOp::FilterGe(t) => simd_filter_ge(data, *t),
        NumericOp::FilterLt(t) => simd_filter_lt(data, *t),
        NumericOp::FilterLe(t) => simd_filter_le(data, *t),
        NumericOp::FilterBetween(l, h) => simd_filter_between(data, *l, *h),
        _ => unreachable!(),
    }
}

fn collect_map_ops(ops: &[NumericOp]) -> Vec<MapOp> {
    ops.iter().filter_map(to_map_op).collect()
}

fn to_map_op(op: &NumericOp) -> Option<MapOp> {
    match op {
        NumericOp::MapMulScalar(s) => Some(MapOp::Mul(*s)),
        NumericOp::MapAddScalar(s) => Some(MapOp::Add(*s)),
        NumericOp::MapSubScalar(s) => Some(MapOp::Sub(*s)),
        NumericOp::MapDivScalar(s) => Some(MapOp::Div(*s)),
        NumericOp::MapAbs => Some(MapOp::Abs),
        NumericOp::MapNeg => Some(MapOp::Neg),
        NumericOp::MapSqrt => Some(MapOp::Sqrt),
        NumericOp::MapFloor => Some(MapOp::Floor),
        NumericOp::MapCeil => Some(MapOp::Ceil),
        NumericOp::MapRound => Some(MapOp::Round),
        NumericOp::MapPowScalar(s) => Some(MapOp::Pow(*s)),
        NumericOp::MapReciprocal => Some(MapOp::Reciprocal),
        NumericOp::MapLog => Some(MapOp::Log),
        NumericOp::MapLog2 => Some(MapOp::Log2),
        NumericOp::MapLog10 => Some(MapOp::Log10),
        NumericOp::MapExp => Some(MapOp::Exp),
        NumericOp::MapSigmoid => Some(MapOp::Sigmoid),
        NumericOp::MapClamp(lo, hi) => Some(MapOp::Clamp(*lo, *hi)),
        NumericOp::MapMod(s) => Some(MapOp::Mod(*s)),
        NumericOp::MapFloorDiv(s) => Some(MapOp::FloorDiv(*s)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Scalar map op enum (internal)
// ---------------------------------------------------------------------------

enum MapOp {
    Mul(f64),
    Add(f64),
    Sub(f64),
    Div(f64),
    Abs,
    Neg,
    Sqrt,
    Floor,
    Ceil,
    Round,
    Pow(f64),
    Reciprocal,
    Log,
    Log2,
    Log10,
    Exp,
    Sigmoid,
    Clamp(f64, f64),
    Mod(f64),
    FloorDiv(f64),
}

/// Apply a batch of map operations in-place using SIMD.
fn flush_maps(data: &mut Vec<f64>, maps: &[MapOp]) {
    for map in maps {
        match map {
            MapOp::Mul(s) => simd_map_mul_inplace(data, *s),
            MapOp::Add(s) => simd_map_add_inplace(data, *s),
            MapOp::Sub(s) => simd_map_sub_inplace(data, *s),
            MapOp::Div(s) => simd_map_div_inplace(data, *s),
            MapOp::Abs => simd_map_abs_inplace(data),
            MapOp::Neg => simd_map_neg_inplace(data),
            MapOp::Sqrt => simd_map_sqrt_inplace(data),
            MapOp::Floor => simd_map_floor_inplace(data),
            MapOp::Ceil => simd_map_ceil_inplace(data),
            MapOp::Round => simd_map_round_inplace(data),
            MapOp::Pow(s) => simd_map_pow_inplace(data, *s),
            MapOp::Reciprocal => simd_map_reciprocal_inplace(data),
            MapOp::Log => { for v in data.iter_mut() { *v = v.ln(); } }
            MapOp::Log2 => { for v in data.iter_mut() { *v = v.log2(); } }
            MapOp::Log10 => { for v in data.iter_mut() { *v = v.log10(); } }
            MapOp::Exp => { for v in data.iter_mut() { *v = v.exp(); } }
            MapOp::Sigmoid => { for v in data.iter_mut() { *v = 1.0 / (1.0 + (-*v).exp()); } }
            MapOp::Clamp(lo, hi) => { for v in data.iter_mut() { *v = v.clamp(*lo, *hi); } }
            MapOp::Mod(s) => { for v in data.iter_mut() { *v %= s; } }
            MapOp::FloorDiv(s) => { for v in data.iter_mut() { *v = (*v / s).floor(); } }
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD map: in-place multiplication
// ---------------------------------------------------------------------------

/// Multiply all elements by `scalar` using 256-bit SIMD (4 × f64 per cycle).
pub fn simd_map_mul_inplace(data: &mut [f64], scalar: f64) {
    let scalar_v = f64x4::splat(scalar);
    let chunks = data.chunks_exact_mut(4);
    let _remainder = chunks.into_remainder(); // need to split differently

    // chunks_exact_mut doesn't return remainder separately in one pass.
    // Use index-based chunking instead:
    let n = data.len();
    let full = n / 4 * 4;

    // SIMD path for full 4-wide chunks
    let (left, right) = data.split_at_mut(full);
    for chunk in left.chunks_exact_mut(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let result = v * scalar_v;
        let arr: [f64; 4] = result.into();
        chunk.copy_from_slice(&arr);
    }

    // Scalar tail
    for x in right.iter_mut() {
        *x *= scalar;
    }
}

/// Add scalar to all elements.
pub fn simd_map_add_inplace(data: &mut [f64], scalar: f64) {
    let scalar_v = f64x4::splat(scalar);
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at_mut(full);
    for chunk in left.chunks_exact_mut(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let result = v + scalar_v;
        let arr: [f64; 4] = result.into();
        chunk.copy_from_slice(&arr);
    }
    for x in right.iter_mut() {
        *x += scalar;
    }
}

/// Subtract scalar from all elements.
pub fn simd_map_sub_inplace(data: &mut [f64], scalar: f64) {
    let scalar_v = f64x4::splat(scalar);
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at_mut(full);
    for chunk in left.chunks_exact_mut(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let result = v - scalar_v;
        let arr: [f64; 4] = result.into();
        chunk.copy_from_slice(&arr);
    }
    for x in right.iter_mut() {
        *x -= scalar;
    }
}

/// Divide all elements by scalar.
pub fn simd_map_div_inplace(data: &mut [f64], scalar: f64) {
    // Multiply by reciprocal — one fewer latency cycle than division
    simd_map_mul_inplace(data, 1.0 / scalar);
}

/// Negate all elements.
pub fn simd_map_neg_inplace(data: &mut [f64]) {
    let neg_one = f64x4::splat(-1.0);
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at_mut(full);
    for chunk in left.chunks_exact_mut(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let result = v * neg_one;
        let arr: [f64; 4] = result.into();
        chunk.copy_from_slice(&arr);
    }
    for x in right.iter_mut() {
        *x = -*x;
    }
}

/// Absolute value of all elements.
pub fn simd_map_abs_inplace(data: &mut [f64]) {
    // abs = x & 0x7FFFFFFFFFFFFFFF (clear sign bit)
    // wide doesn't have a direct abs for f64x4 in all versions; use scalar
    // until we confirm the API.  The loop vectorizes cleanly with LLVM anyway.
    for x in data.iter_mut() {
        *x = x.abs();
    }
}

/// Square root of all elements (element-wise sqrt, not sum).
pub fn simd_map_sqrt_inplace(data: &mut [f64]) {
    // LLVM auto-vectorizes this into vsqrtpd instructions with -O3
    for x in data.iter_mut() {
        *x = x.sqrt();
    }
}

pub fn simd_map_floor_inplace(data: &mut [f64]) {
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at_mut(full);
    for chunk in left.chunks_exact_mut(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let arr: [f64; 4] = v.floor().into();
        chunk.copy_from_slice(&arr);
    }
    for x in right.iter_mut() {
        *x = x.floor();
    }
}

pub fn simd_map_ceil_inplace(data: &mut [f64]) {
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at_mut(full);
    for chunk in left.chunks_exact_mut(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let arr: [f64; 4] = v.ceil().into();
        chunk.copy_from_slice(&arr);
    }
    for x in right.iter_mut() {
        *x = x.ceil();
    }
}

pub fn simd_map_round_inplace(data: &mut [f64]) {
    // f64x4::round() uses hardware round-to-even; f64::round() rounds half
    // away from zero.  Use scalar to preserve Rust semantics (-0.5 → -1.0).
    for x in data.iter_mut() {
        *x = x.round();
    }
}

pub fn simd_map_pow_inplace(data: &mut [f64], scalar: f64) {
    // wide::f64x4 does not expose element-wise powf; use scalar with LLVM auto-vectorization
    for x in data.iter_mut() {
        *x = x.powf(scalar);
    }
}

pub fn simd_map_reciprocal_inplace(data: &mut [f64]) {
    let one = f64x4::splat(1.0);
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at_mut(full);
    for chunk in left.chunks_exact_mut(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let arr: [f64; 4] = (one / v).into();
        chunk.copy_from_slice(&arr);
    }
    for x in right.iter_mut() {
        *x = 1.0 / *x;
    }
}

// ---------------------------------------------------------------------------
// SIMD filter: returns a new Vec (variable-length output)
// ---------------------------------------------------------------------------

/// Keep only elements > threshold.
///
/// Implementation: process 4 elements per SIMD iteration, use bitmask
/// to conditionally copy survivors into output buffer.
pub fn simd_filter_gt(data: &[f64], threshold: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let threshold_v = f64x4::splat(threshold);
    let n = data.len();
    let full = n / 4 * 4;

    let (left, right) = data.split_at(full);
    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        // cmp_gt returns lanes with all bits set (NaN-safe in wide)
        let mask = v.cmp_gt(threshold_v);
        let bits = mask.move_mask();
        // bits is a u8 where bit i means lane i passed
        if bits == 0b1111 {
            // All 4 pass — bulk copy
            out.extend_from_slice(chunk);
        } else if bits != 0 {
            // Selective copy
            for (i, &val) in chunk.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    out.push(val);
                }
            }
        }
        // bits == 0: all filtered, skip
    }

    // Scalar tail
    for &val in right {
        if val > threshold {
            out.push(val);
        }
    }

    out
}

pub fn simd_filter_ge(data: &[f64], threshold: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let threshold_v = f64x4::splat(threshold);
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at(full);

    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mask = v.cmp_ge(threshold_v);
        let bits = mask.move_mask();
        if bits == 0b1111 {
            out.extend_from_slice(chunk);
        } else if bits != 0 {
            for (i, &val) in chunk.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    out.push(val);
                }
            }
        }
    }
    for &val in right {
        if val >= threshold {
            out.push(val);
        }
    }
    out
}

pub fn simd_filter_lt(data: &[f64], threshold: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let threshold_v = f64x4::splat(threshold);
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at(full);

    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mask = v.cmp_lt(threshold_v);
        let bits = mask.move_mask();
        if bits == 0b1111 {
            out.extend_from_slice(chunk);
        } else if bits != 0 {
            for (i, &val) in chunk.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    out.push(val);
                }
            }
        }
    }
    for &val in right {
        if val < threshold {
            out.push(val);
        }
    }
    out
}

pub fn simd_filter_le(data: &[f64], threshold: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let threshold_v = f64x4::splat(threshold);
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at(full);

    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mask = v.cmp_le(threshold_v);
        let bits = mask.move_mask();
        if bits == 0b1111 {
            out.extend_from_slice(chunk);
        } else if bits != 0 {
            for (i, &val) in chunk.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    out.push(val);
                }
            }
        }
    }
    for &val in right {
        if val <= threshold {
            out.push(val);
        }
    }
    out
}

/// Keep elements in [lo, hi] (inclusive on both ends).
/// SIMD mask: cmp_ge(lo) AND cmp_le(hi) — 2 compares per 4 elements.
pub fn simd_filter_between(data: &[f64], lo: f64, hi: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let lo_v = f64x4::splat(lo);
    let hi_v = f64x4::splat(hi);
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at(full);

    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mask = v.cmp_ge(lo_v) & v.cmp_le(hi_v);
        let bits = mask.move_mask();
        if bits == 0b1111 {
            out.extend_from_slice(chunk);
        } else if bits != 0 {
            for (i, &val) in chunk.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    out.push(val);
                }
            }
        }
    }
    for &val in right {
        if val >= lo && val <= hi {
            out.push(val);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// f32x8 variants — useful for large ML feature arrays where f32 precision is ok
// ---------------------------------------------------------------------------

pub fn simd_filter_gt_f32(data: &[f32], threshold: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let threshold_v = f32x8::splat(threshold);
    let n = data.len();
    let full = n / 8 * 8;
    let (left, right) = data.split_at(full);

    for chunk in left.chunks_exact(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let mask = v.cmp_gt(threshold_v);
        let bits = mask.move_mask();
        if bits == 0xFF {
            out.extend_from_slice(chunk);
        } else if bits != 0 {
            for (i, &val) in chunk.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    out.push(val);
                }
            }
        }
    }
    for &val in right {
        if val > threshold {
            out.push(val);
        }
    }
    out
}

pub fn simd_map_mul_f32_inplace(data: &mut [f32], scalar: f32) {
    let scalar_v = f32x8::splat(scalar);
    let n = data.len();
    let full = n / 8 * 8;
    let (left, right) = data.split_at_mut(full);
    for chunk in left.chunks_exact_mut(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let result = v * scalar_v;
        let arr: [f32; 8] = result.into();
        chunk.copy_from_slice(&arr);
    }
    for x in right.iter_mut() {
        *x *= scalar;
    }
}

pub fn simd_filter_ge_f32(data: &[f32], threshold: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let threshold_v = f32x8::splat(threshold);
    let n = data.len();
    let full = n / 8 * 8;
    let (left, right) = data.split_at(full);
    for chunk in left.chunks_exact(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let mask = v.cmp_ge(threshold_v);
        let bits = mask.move_mask();
        if bits == 0xFF {
            out.extend_from_slice(chunk);
        } else if bits != 0 {
            for (i, &val) in chunk.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    out.push(val);
                }
            }
        }
    }
    for &val in right {
        if val >= threshold {
            out.push(val);
        }
    }
    out
}

pub fn simd_filter_lt_f32(data: &[f32], threshold: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let threshold_v = f32x8::splat(threshold);
    let n = data.len();
    let full = n / 8 * 8;
    let (left, right) = data.split_at(full);
    for chunk in left.chunks_exact(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let mask = v.cmp_lt(threshold_v);
        let bits = mask.move_mask();
        if bits == 0xFF {
            out.extend_from_slice(chunk);
        } else if bits != 0 {
            for (i, &val) in chunk.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    out.push(val);
                }
            }
        }
    }
    for &val in right {
        if val < threshold {
            out.push(val);
        }
    }
    out
}

pub fn simd_filter_le_f32(data: &[f32], threshold: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let threshold_v = f32x8::splat(threshold);
    let n = data.len();
    let full = n / 8 * 8;
    let (left, right) = data.split_at(full);
    for chunk in left.chunks_exact(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let mask = v.cmp_le(threshold_v);
        let bits = mask.move_mask();
        if bits == 0xFF {
            out.extend_from_slice(chunk);
        } else if bits != 0 {
            for (i, &val) in chunk.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    out.push(val);
                }
            }
        }
    }
    for &val in right {
        if val <= threshold {
            out.push(val);
        }
    }
    out
}

pub fn simd_filter_between_f32(data: &[f32], lo: f32, hi: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let lo_v = f32x8::splat(lo);
    let hi_v = f32x8::splat(hi);
    let n = data.len();
    let full = n / 8 * 8;
    let (left, right) = data.split_at(full);
    for chunk in left.chunks_exact(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let mask = v.cmp_ge(lo_v) & v.cmp_le(hi_v);
        let bits = mask.move_mask();
        if bits == 0xFF {
            out.extend_from_slice(chunk);
        } else if bits != 0 {
            for (i, &val) in chunk.iter().enumerate() {
                if bits & (1 << i) != 0 {
                    out.push(val);
                }
            }
        }
    }
    for &val in right {
        if val >= lo && val <= hi {
            out.push(val);
        }
    }
    out
}

pub fn simd_map_add_f32_inplace(data: &mut [f32], scalar: f32) {
    let s = f32x8::splat(scalar);
    let n = data.len();
    let full = n / 8 * 8;
    let (left, right) = data.split_at_mut(full);
    for chunk in left.chunks_exact_mut(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let arr: [f32; 8] = (v + s).into();
        chunk.copy_from_slice(&arr);
    }
    for x in right.iter_mut() {
        *x += scalar;
    }
}

pub fn simd_map_sub_f32_inplace(data: &mut [f32], scalar: f32) {
    let s = f32x8::splat(scalar);
    let n = data.len();
    let full = n / 8 * 8;
    let (left, right) = data.split_at_mut(full);
    for chunk in left.chunks_exact_mut(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let arr: [f32; 8] = (v - s).into();
        chunk.copy_from_slice(&arr);
    }
    for x in right.iter_mut() {
        *x -= scalar;
    }
}

pub fn simd_map_neg_f32_inplace(data: &mut [f32]) {
    let neg_one = f32x8::splat(-1.0);
    let n = data.len();
    let full = n / 8 * 8;
    let (left, right) = data.split_at_mut(full);
    for chunk in left.chunks_exact_mut(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let arr: [f32; 8] = (v * neg_one).into();
        chunk.copy_from_slice(&arr);
    }
    for x in right.iter_mut() {
        *x = -*x;
    }
}

pub fn simd_map_abs_f32_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = x.abs();
    }
}

pub fn simd_map_sqrt_f32_inplace(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = x.sqrt();
    }
}

pub fn simd_map_div_f32_inplace(data: &mut [f32], scalar: f32) {
    simd_map_mul_f32_inplace(data, 1.0 / scalar);
}

pub fn simd_sum_f32(data: &[f32]) -> f32 {
    let n = data.len();
    let full = n / 8 * 8;
    let (left, right) = data.split_at(full);
    let mut acc = f32x8::splat(0.0);
    for chunk in left.chunks_exact(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        acc += v;
    }
    let arr: [f32; 8] = acc.into();
    let mut total = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];
    for &x in right {
        total += x;
    }
    total
}

pub fn simd_max_f32(data: &[f32]) -> Option<f32> {
    if data.is_empty() {
        return None;
    }
    let mut max_val = f32::NEG_INFINITY;
    for &x in data {
        if x > max_val {
            max_val = x;
        }
    }
    Some(max_val)
}

pub fn simd_count_f32_gt(data: &[f32], threshold: f32) -> usize {
    let thresh = f32x8::splat(threshold);
    let mut count = 0usize;
    let (left, right) = data.split_at(data.len() / 8 * 8);
    for chunk in left.chunks_exact(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        count += v.cmp_gt(thresh).move_mask().count_ones() as usize;
    }
    for &x in right {
        count += (x > threshold) as usize;
    }
    count
}

pub fn simd_count_f32_ge(data: &[f32], threshold: f32) -> usize {
    let thresh = f32x8::splat(threshold);
    let mut count = 0usize;
    let (left, right) = data.split_at(data.len() / 8 * 8);
    for chunk in left.chunks_exact(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        count += v.cmp_ge(thresh).move_mask().count_ones() as usize;
    }
    for &x in right {
        count += (x >= threshold) as usize;
    }
    count
}

pub fn simd_count_f32_lt(data: &[f32], threshold: f32) -> usize {
    let thresh = f32x8::splat(threshold);
    let mut count = 0usize;
    let (left, right) = data.split_at(data.len() / 8 * 8);
    for chunk in left.chunks_exact(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        count += v.cmp_lt(thresh).move_mask().count_ones() as usize;
    }
    for &x in right {
        count += (x < threshold) as usize;
    }
    count
}

pub fn simd_count_f32_le(data: &[f32], threshold: f32) -> usize {
    let thresh = f32x8::splat(threshold);
    let mut count = 0usize;
    let (left, right) = data.split_at(data.len() / 8 * 8);
    for chunk in left.chunks_exact(8) {
        let v = f32x8::from([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        count += v.cmp_le(thresh).move_mask().count_ones() as usize;
    }
    for &x in right {
        count += (x <= threshold) as usize;
    }
    count
}

/// Execute a sequence of NumericOp over f32 data using f32x8 SIMD.
/// Mirrors `execute_simd_pipeline` but operates on `&[f32]`.
pub fn execute_simd_f32_pipeline(data: &[f32], ops: &[NumericOp]) -> Vec<f32> {
    if ops.is_empty() {
        return data.to_vec();
    }

    let first_filter = ops.iter().position(|op| {
        matches!(
            op,
            NumericOp::FilterGt(_)
                | NumericOp::FilterGe(_)
                | NumericOp::FilterLt(_)
                | NumericOp::FilterLe(_)
                | NumericOp::FilterBetween(_, _)
        )
    });

    let Some(fi) = first_filter else {
        let mut result = data.to_vec();
        flush_maps_f32(&mut result, ops);
        return result;
    };

    let has_leading_maps = ops[..fi].iter().any(|op| is_map_op_f32(op));

    let mut result = if has_leading_maps {
        let mut r = data.to_vec();
        flush_maps_f32(&mut r, &ops[..fi]);
        apply_filter_op_f32(&r, &ops[fi])
    } else {
        apply_filter_op_f32(data, &ops[fi])
    };

    for op in &ops[fi + 1..] {
        match op {
            NumericOp::FilterGt(_)
            | NumericOp::FilterGe(_)
            | NumericOp::FilterLt(_)
            | NumericOp::FilterLe(_)
            | NumericOp::FilterBetween(_, _) => {
                result = apply_filter_op_f32(&result, op);
            }
            _ if is_map_op_f32(op) => flush_maps_f32(&mut result, std::slice::from_ref(op)),
            _ => {}
        }
    }
    result
}

fn apply_filter_op_f32(data: &[f32], op: &NumericOp) -> Vec<f32> {
    match op {
        NumericOp::FilterGt(t) => simd_filter_gt_f32(data, *t as f32),
        NumericOp::FilterGe(t) => simd_filter_ge_f32(data, *t as f32),
        NumericOp::FilterLt(t) => simd_filter_lt_f32(data, *t as f32),
        NumericOp::FilterLe(t) => simd_filter_le_f32(data, *t as f32),
        NumericOp::FilterBetween(l, h) => simd_filter_between_f32(data, *l as f32, *h as f32),
        _ => unreachable!(),
    }
}

fn is_map_op_f32(op: &NumericOp) -> bool {
    matches!(
        op,
        NumericOp::MapMulScalar(_)
            | NumericOp::MapAddScalar(_)
            | NumericOp::MapSubScalar(_)
            | NumericOp::MapDivScalar(_)
            | NumericOp::MapAbs
            | NumericOp::MapNeg
            | NumericOp::MapSqrt
    )
}

fn flush_maps_f32(data: &mut Vec<f32>, ops: &[NumericOp]) {
    for op in ops {
        match op {
            NumericOp::MapMulScalar(s) => simd_map_mul_f32_inplace(data, *s as f32),
            NumericOp::MapAddScalar(s) => simd_map_add_f32_inplace(data, *s as f32),
            NumericOp::MapSubScalar(s) => simd_map_sub_f32_inplace(data, *s as f32),
            NumericOp::MapDivScalar(s) => simd_map_div_f32_inplace(data, *s as f32),
            NumericOp::MapAbs => simd_map_abs_f32_inplace(data),
            NumericOp::MapNeg => simd_map_neg_f32_inplace(data),
            NumericOp::MapSqrt => simd_map_sqrt_f32_inplace(data),
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Reduction operations — useful as terminal steps
// ---------------------------------------------------------------------------

/// Sum all f64 elements using SIMD horizontal reduction.
pub fn simd_sum_f64(data: &[f64]) -> f64 {
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at(full);

    let mut acc = f64x4::splat(0.0);
    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        acc += v;
    }

    // Horizontal sum: add all 4 lanes
    let arr: [f64; 4] = acc.into();
    let mut total = arr[0] + arr[1] + arr[2] + arr[3];

    // Scalar tail
    for &x in right {
        total += x;
    }
    total
}

/// Dot product of two f64 slices using SIMD fused multiply-add.
pub fn simd_dot_product_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot product requires equal-length slices");
    let n = a.len();
    let full = n / 4 * 4;

    let mut acc = f64x4::splat(0.0);
    for (ca, cb) in a[..full].chunks_exact(4).zip(b[..full].chunks_exact(4)) {
        let va = f64x4::from([ca[0], ca[1], ca[2], ca[3]]);
        let vb = f64x4::from([cb[0], cb[1], cb[2], cb[3]]);
        acc += va * vb;
    }

    let arr: [f64; 4] = acc.into();
    let mut total = arr[0] + arr[1] + arr[2] + arr[3];
    for i in full..n {
        total += a[i] * b[i];
    }
    total
}

/// Compute max of all elements.
pub fn simd_max_f64(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let n = data.len();
    let full = n / 4 * 4;
    let (left, right) = data.split_at(full);

    let mut acc = f64x4::splat(f64::NEG_INFINITY);
    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        acc = acc.max(v);
    }

    let arr: [f64; 4] = acc.into();
    let mut max_val = arr[0].max(arr[1]).max(arr[2]).max(arr[3]);
    for &x in right {
        if x > max_val {
            max_val = x;
        }
    }
    Some(max_val)
}

// ---------------------------------------------------------------------------
// SIMD count — no allocation, uses move_mask() + popcount
// ---------------------------------------------------------------------------

macro_rules! simd_count_fn {
    ($name:ident, $cmp_method:ident, $scalar_op:tt) => {
        pub fn $name(data: &[f64], threshold: f64) -> usize {
            let thresh = f64x4::splat(threshold);
            let mut count = 0usize;
            let (left, right) = data.split_at(data.len() / 4 * 4);
            for chunk in left.chunks_exact(4) {
                let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
                count += v.$cmp_method(thresh).move_mask().count_ones() as usize;
            }
            for &x in right {
                count += (x $scalar_op threshold) as usize;
            }
            count
        }
    };
}

simd_count_fn!(simd_count_gt, cmp_gt, >);
simd_count_fn!(simd_count_ge, cmp_ge, >=);
simd_count_fn!(simd_count_lt, cmp_lt, <);
simd_count_fn!(simd_count_le, cmp_le, <=);

pub fn simd_count_between(data: &[f64], lo: f64, hi: f64) -> usize {
    let lo_v = f64x4::splat(lo);
    let hi_v = f64x4::splat(hi);
    let mut count = 0usize;
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mask = v.cmp_ge(lo_v) & v.cmp_le(hi_v);
        count += mask.move_mask().count_ones() as usize;
    }
    for &x in right {
        count += (x >= lo && x <= hi) as usize;
    }
    count
}

// ---------------------------------------------------------------------------
// SIMD fused filter+sum — single pass, no intermediate Vec
// ---------------------------------------------------------------------------

macro_rules! simd_filter_sum_fn {
    ($name:ident, $cmp_method:ident, $scalar_op:tt) => {
        pub fn $name(data: &[f64], threshold: f64) -> f64 {
            let thresh = f64x4::splat(threshold);
            let mut acc = f64x4::ZERO;
            let (left, right) = data.split_at(data.len() / 4 * 4);
            for chunk in left.chunks_exact(4) {
                let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
                // blend(t, f): selects t where mask is all-1s, f where all-0s
                acc += v.$cmp_method(thresh).blend(v, f64x4::ZERO);
            }
            let arr: [f64; 4] = acc.into();
            let mut sum = arr[0] + arr[1] + arr[2] + arr[3];
            for &x in right {
                if x $scalar_op threshold { sum += x; }
            }
            sum
        }
    };
}

simd_filter_sum_fn!(simd_filter_sum_gt, cmp_gt, >);
simd_filter_sum_fn!(simd_filter_sum_ge, cmp_ge, >=);
simd_filter_sum_fn!(simd_filter_sum_lt, cmp_lt, <);
simd_filter_sum_fn!(simd_filter_sum_le, cmp_le, <=);

pub fn simd_filter_sum_between(data: &[f64], lo: f64, hi: f64) -> f64 {
    let lo_v = f64x4::splat(lo);
    let hi_v = f64x4::splat(hi);
    let mut acc = f64x4::ZERO;
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mask = v.cmp_ge(lo_v) & v.cmp_le(hi_v);
        acc += mask.blend(v, f64x4::ZERO);
    }
    let arr: [f64; 4] = acc.into();
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];
    for &x in right {
        if x >= lo && x <= hi {
            sum += x;
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// SIMD fused filter+mean — single pass, no intermediate Vec
// Returns None when no element passes the filter.
// ---------------------------------------------------------------------------

macro_rules! simd_filter_mean_fn {
    ($name:ident, $cmp_method:ident, $scalar_op:tt) => {
        pub fn $name(data: &[f64], threshold: f64) -> Option<f64> {
            let thresh = f64x4::splat(threshold);
            let mut acc = f64x4::ZERO;
            let mut cnt: usize = 0;
            let (left, right) = data.split_at(data.len() / 4 * 4);
            for chunk in left.chunks_exact(4) {
                let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let mask = v.$cmp_method(thresh);
                acc += mask.blend(v, f64x4::ZERO);
                cnt += mask.move_mask().count_ones() as usize;
            }
            let arr: [f64; 4] = acc.into();
            let mut sum = arr[0] + arr[1] + arr[2] + arr[3];
            for &x in right {
                if x $scalar_op threshold { sum += x; cnt += 1; }
            }
            if cnt == 0 { None } else { Some(sum / cnt as f64) }
        }
    };
}

simd_filter_mean_fn!(simd_filter_mean_gt, cmp_gt, >);
simd_filter_mean_fn!(simd_filter_mean_ge, cmp_ge, >=);
simd_filter_mean_fn!(simd_filter_mean_lt, cmp_lt, <);
simd_filter_mean_fn!(simd_filter_mean_le, cmp_le, <=);

pub fn simd_filter_mean_between(data: &[f64], lo: f64, hi: f64) -> Option<f64> {
    let lo_v = f64x4::splat(lo);
    let hi_v = f64x4::splat(hi);
    let mut acc = f64x4::ZERO;
    let mut cnt: usize = 0;
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mask = v.cmp_ge(lo_v) & v.cmp_le(hi_v);
        acc += mask.blend(v, f64x4::ZERO);
        cnt += mask.move_mask().count_ones() as usize;
    }
    let arr: [f64; 4] = acc.into();
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3];
    for &x in right {
        if x >= lo && x <= hi {
            sum += x;
            cnt += 1;
        }
    }
    if cnt == 0 {
        None
    } else {
        Some(sum / cnt as f64)
    }
}

// ---------------------------------------------------------------------------
// SIMD fused filter+variance — single pass (sum + sum_sq + count), no Vec.
// Returns population variance (denominator N).  None when no element passes.
// ---------------------------------------------------------------------------

macro_rules! simd_filter_var_fn {
    ($name:ident, $cmp_method:ident, $scalar_op:tt) => {
        pub fn $name(data: &[f64], threshold: f64) -> Option<f64> {
            let thresh = f64x4::splat(threshold);
            let mut sum_acc = f64x4::ZERO;
            let mut ssq_acc = f64x4::ZERO;
            let mut cnt: usize = 0;
            let (left, right) = data.split_at(data.len() / 4 * 4);
            for chunk in left.chunks_exact(4) {
                let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let mask = v.$cmp_method(thresh);
                let f = mask.blend(v, f64x4::ZERO);
                sum_acc += f;
                ssq_acc += f * f;
                cnt += mask.move_mask().count_ones() as usize;
            }
            let sa: [f64; 4] = sum_acc.into();
            let qa: [f64; 4] = ssq_acc.into();
            let mut sum = sa[0] + sa[1] + sa[2] + sa[3];
            let mut ssq = qa[0] + qa[1] + qa[2] + qa[3];
            for &x in right {
                if x $scalar_op threshold { sum += x; ssq += x * x; cnt += 1; }
            }
            if cnt == 0 { return None; }
            let n = cnt as f64;
            let mean = sum / n;
            Some((ssq / n) - mean * mean)
        }
    };
}

simd_filter_var_fn!(simd_filter_var_gt, cmp_gt, >);
simd_filter_var_fn!(simd_filter_var_ge, cmp_ge, >=);
simd_filter_var_fn!(simd_filter_var_lt, cmp_lt, <);
simd_filter_var_fn!(simd_filter_var_le, cmp_le, <=);

pub fn simd_filter_var_between(data: &[f64], lo: f64, hi: f64) -> Option<f64> {
    let lo_v = f64x4::splat(lo);
    let hi_v = f64x4::splat(hi);
    let mut sum_acc = f64x4::ZERO;
    let mut ssq_acc = f64x4::ZERO;
    let mut cnt: usize = 0;
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mask = v.cmp_ge(lo_v) & v.cmp_le(hi_v);
        let f = mask.blend(v, f64x4::ZERO);
        sum_acc += f;
        ssq_acc += f * f;
        cnt += mask.move_mask().count_ones() as usize;
    }
    let sa: [f64; 4] = sum_acc.into();
    let qa: [f64; 4] = ssq_acc.into();
    let mut sum = sa[0] + sa[1] + sa[2] + sa[3];
    let mut ssq = qa[0] + qa[1] + qa[2] + qa[3];
    for &x in right {
        if x >= lo && x <= hi {
            sum += x;
            ssq += x * x;
            cnt += 1;
        }
    }
    if cnt == 0 {
        return None;
    }
    let n = cnt as f64;
    let mean = sum / n;
    Some((ssq / n) - mean * mean)
}

// ---------------------------------------------------------------------------
// SIMD fused filter+max — single pass, no intermediate Vec
// ---------------------------------------------------------------------------

macro_rules! simd_filter_max_fn {
    ($name:ident, $cmp_method:ident, $scalar_op:tt) => {
        pub fn $name(data: &[f64], threshold: f64) -> Option<f64> {
            let thresh = f64x4::splat(threshold);
            let mut max_v = f64x4::splat(f64::NEG_INFINITY);
            let mut found = false;
            let (left, right) = data.split_at(data.len() / 4 * 4);
            for chunk in left.chunks_exact(4) {
                let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let mask = v.$cmp_method(thresh);
                if mask.move_mask() != 0 {
                    found = true;
                    // Filtered lanes get NEG_INFINITY — neutral for max accumulation
                    max_v = max_v.max(mask.blend(v, f64x4::splat(f64::NEG_INFINITY)));
                }
            }
            let arr: [f64; 4] = max_v.into();
            let mut result = arr[0].max(arr[1]).max(arr[2]).max(arr[3]);
            for &x in right {
                if x $scalar_op threshold { result = result.max(x); found = true; }
            }
            if found { Some(result) } else { None }
        }
    };
}

simd_filter_max_fn!(simd_filter_max_gt, cmp_gt, >);
simd_filter_max_fn!(simd_filter_max_ge, cmp_ge, >=);
simd_filter_max_fn!(simd_filter_max_lt, cmp_lt, <);
simd_filter_max_fn!(simd_filter_max_le, cmp_le, <=);

// ---------------------------------------------------------------------------
// SIMD fused filter+min — single pass, no intermediate Vec
// ---------------------------------------------------------------------------

macro_rules! simd_filter_min_fn {
    ($name:ident, $cmp_method:ident, $scalar_op:tt) => {
        pub fn $name(data: &[f64], threshold: f64) -> Option<f64> {
            let thresh = f64x4::splat(threshold);
            let mut min_v = f64x4::splat(f64::INFINITY);
            let mut found = false;
            let (left, right) = data.split_at(data.len() / 4 * 4);
            for chunk in left.chunks_exact(4) {
                let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let mask = v.$cmp_method(thresh);
                if mask.move_mask() != 0 {
                    found = true;
                    // Filtered lanes get INFINITY — neutral for min accumulation
                    min_v = min_v.min(mask.blend(v, f64x4::splat(f64::INFINITY)));
                }
            }
            let arr: [f64; 4] = min_v.into();
            let mut result = arr[0].min(arr[1]).min(arr[2]).min(arr[3]);
            for &x in right {
                if x $scalar_op threshold { result = result.min(x); found = true; }
            }
            if found { Some(result) } else { None }
        }
    };
}

simd_filter_min_fn!(simd_filter_min_gt, cmp_gt, >);
simd_filter_min_fn!(simd_filter_min_ge, cmp_ge, >=);
simd_filter_min_fn!(simd_filter_min_lt, cmp_lt, <);
simd_filter_min_fn!(simd_filter_min_le, cmp_le, <=);

// ---------------------------------------------------------------------------
// SIMD fused filter+stats — count + sum + min + max, no intermediate Vec
// ---------------------------------------------------------------------------

macro_rules! simd_filter_stats_fn {
    ($name:ident, $cmp_method:ident, $scalar_op:tt) => {
        pub fn $name(data: &[f64], threshold: f64) -> Option<(usize, f64, f64, f64)> {
            let thresh = f64x4::splat(threshold);
            let mut sum_v = f64x4::ZERO;
            let mut min_v = f64x4::splat(f64::INFINITY);
            let mut max_v = f64x4::splat(f64::NEG_INFINITY);
            let mut count = 0usize;
            let (left, right) = data.split_at(data.len() / 4 * 4);

            for chunk in left.chunks_exact(4) {
                let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let mask = v.$cmp_method(thresh);
                let mask_bits = mask.move_mask();
                if mask_bits != 0 {
                    count += mask_bits.count_ones() as usize;
                    sum_v += mask.blend(v, f64x4::ZERO);
                    min_v = min_v.min(mask.blend(v, f64x4::splat(f64::INFINITY)));
                    max_v = max_v.max(mask.blend(v, f64x4::splat(f64::NEG_INFINITY)));
                }
            }

            let sum_arr: [f64; 4] = sum_v.into();
            let min_arr: [f64; 4] = min_v.into();
            let max_arr: [f64; 4] = max_v.into();
            let mut sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
            let mut min = min_arr[0].min(min_arr[1]).min(min_arr[2]).min(min_arr[3]);
            let mut max = max_arr[0].max(max_arr[1]).max(max_arr[2]).max(max_arr[3]);

            for &x in right {
                if x $scalar_op threshold {
                    count += 1;
                    sum += x;
                    min = min.min(x);
                    max = max.max(x);
                }
            }

            if count == 0 {
                None
            } else {
                Some((count, sum, min, max))
            }
        }
    };
}

simd_filter_stats_fn!(simd_filter_stats_gt, cmp_gt, >);
simd_filter_stats_fn!(simd_filter_stats_ge, cmp_ge, >=);
simd_filter_stats_fn!(simd_filter_stats_lt, cmp_lt, <);
simd_filter_stats_fn!(simd_filter_stats_le, cmp_le, <=);

pub fn simd_filter_stats_between(data: &[f64], lo: f64, hi: f64) -> Option<(usize, f64, f64, f64)> {
    let lo_v = f64x4::splat(lo);
    let hi_v = f64x4::splat(hi);
    let mut sum_v = f64x4::ZERO;
    let mut min_v = f64x4::splat(f64::INFINITY);
    let mut max_v = f64x4::splat(f64::NEG_INFINITY);
    let mut count = 0usize;
    let (left, right) = data.split_at(data.len() / 4 * 4);

    for chunk in left.chunks_exact(4) {
        let v = f64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mask = v.cmp_ge(lo_v) & v.cmp_le(hi_v);
        let mask_bits = mask.move_mask();
        if mask_bits != 0 {
            count += mask_bits.count_ones() as usize;
            sum_v += mask.blend(v, f64x4::ZERO);
            min_v = min_v.min(mask.blend(v, f64x4::splat(f64::INFINITY)));
            max_v = max_v.max(mask.blend(v, f64x4::splat(f64::NEG_INFINITY)));
        }
    }

    let sum_arr: [f64; 4] = sum_v.into();
    let min_arr: [f64; 4] = min_v.into();
    let max_arr: [f64; 4] = max_v.into();
    let mut sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
    let mut min = min_arr[0].min(min_arr[1]).min(min_arr[2]).min(min_arr[3]);
    let mut max = max_arr[0].max(max_arr[1]).max(max_arr[2]).max(max_arr[3]);

    for &x in right {
        if x >= lo && x <= hi {
            count += 1;
            sum += x;
            min = min.min(x);
            max = max.max(x);
        }
    }

    if count == 0 {
        None
    } else {
        Some((count, sum, min, max))
    }
}

// ---------------------------------------------------------------------------
// i64 SIMD filter — returns a new Vec<i64>
//
// Uses i64x4 for the comparison (4 lanes in parallel).
// Mask extraction uses Into<[i64; 4]>: -1 (all bits set) = lane passed.
//
// i64x4 has cmp_gt, cmp_lt, cmp_eq but NOT cmp_ge / cmp_le.
// ge = gt | eq,  le = lt | eq.
// ---------------------------------------------------------------------------

fn push_masked_i64(out: &mut Vec<i64>, chunk: &[i64], mask: i64x4) {
    let m: [i64; 4] = mask.into();
    for (&val, &lane) in chunk.iter().zip(m.iter()) {
        if lane != 0 {
            out.push(val);
        }
    }
}

/// Count lanes that match the mask (each lane is -1 if true, 0 if false).
#[inline(always)]
fn count_mask_i64(mask: i64x4) -> usize {
    let m: [i64; 4] = mask.into();
    (m[0] != 0) as usize + (m[1] != 0) as usize + (m[2] != 0) as usize + (m[3] != 0) as usize
}

pub fn simd_count_i64_gt(data: &[i64], threshold: i64) -> usize {
    let thresh = i64x4::splat(threshold);
    let mut count = 0usize;
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = i64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        count += count_mask_i64(v.cmp_gt(thresh));
    }
    for &x in right {
        count += (x > threshold) as usize;
    }
    count
}

pub fn simd_count_i64_ge(data: &[i64], threshold: i64) -> usize {
    let thresh = i64x4::splat(threshold);
    let mut count = 0usize;
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = i64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        count += count_mask_i64(v.cmp_gt(thresh) | v.cmp_eq(thresh));
    }
    for &x in right {
        count += (x >= threshold) as usize;
    }
    count
}

pub fn simd_count_i64_lt(data: &[i64], threshold: i64) -> usize {
    let thresh = i64x4::splat(threshold);
    let mut count = 0usize;
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = i64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        count += count_mask_i64(v.cmp_lt(thresh));
    }
    for &x in right {
        count += (x < threshold) as usize;
    }
    count
}

pub fn simd_count_i64_le(data: &[i64], threshold: i64) -> usize {
    let thresh = i64x4::splat(threshold);
    let mut count = 0usize;
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = i64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        count += count_mask_i64(v.cmp_lt(thresh) | v.cmp_eq(thresh));
    }
    for &x in right {
        count += (x <= threshold) as usize;
    }
    count
}

pub fn simd_filter_i64_gt(data: &[i64], threshold: i64) -> Vec<i64> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let thresh = i64x4::splat(threshold);
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = i64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        push_masked_i64(&mut out, chunk, v.cmp_gt(thresh));
    }
    for &val in right {
        if val > threshold {
            out.push(val);
        }
    }
    out
}

pub fn simd_filter_i64_ge(data: &[i64], threshold: i64) -> Vec<i64> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let thresh = i64x4::splat(threshold);
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = i64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        push_masked_i64(&mut out, chunk, v.cmp_gt(thresh) | v.cmp_eq(thresh));
    }
    for &val in right {
        if val >= threshold {
            out.push(val);
        }
    }
    out
}

pub fn simd_filter_i64_lt(data: &[i64], threshold: i64) -> Vec<i64> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let thresh = i64x4::splat(threshold);
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = i64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        push_masked_i64(&mut out, chunk, v.cmp_lt(thresh));
    }
    for &val in right {
        if val < threshold {
            out.push(val);
        }
    }
    out
}

pub fn simd_filter_i64_le(data: &[i64], threshold: i64) -> Vec<i64> {
    let mut out = Vec::with_capacity(data.len() / 2);
    let thresh = i64x4::splat(threshold);
    let (left, right) = data.split_at(data.len() / 4 * 4);
    for chunk in left.chunks_exact(4) {
        let v = i64x4::from([chunk[0], chunk[1], chunk[2], chunk[3]]);
        push_masked_i64(&mut out, chunk, v.cmp_lt(thresh) | v.cmp_eq(thresh));
    }
    for &val in right {
        if val <= threshold {
            out.push(val);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn float_eq(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-10)
    }

    fn range_f64(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64).collect()
    }

    // ── filter_gt ────────────────────────────────────────────────────────────

    #[test]
    fn test_filter_gt_basic() {
        let data = vec![1.0, 5.0, 3.0, 7.0, 2.0];
        let result = simd_filter_gt(&data, 3.0);
        assert_eq!(result, vec![5.0, 7.0]);
    }

    #[test]
    fn test_filter_gt_empty() {
        assert_eq!(simd_filter_gt(&[], 0.0), Vec::<f64>::new());
    }

    #[test]
    fn test_filter_gt_all_pass() {
        let data = vec![5.0, 6.0, 7.0];
        assert_eq!(simd_filter_gt(&data, 4.0), data);
    }

    #[test]
    fn test_filter_gt_none_pass() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(simd_filter_gt(&data, 10.0), Vec::<f64>::new());
    }

    #[test]
    fn test_filter_gt_simd_width_boundary() {
        // 9 elements: exercises both 4-wide SIMD chunks and scalar tail
        let data = range_f64(9);
        let result = simd_filter_gt(&data, 4.0);
        let expected: Vec<f64> = vec![5.0, 6.0, 7.0, 8.0];
        assert!(float_eq(&result, &expected));
    }

    // ── filter_ge ────────────────────────────────────────────────────────────

    #[test]
    fn test_filter_ge_boundary() {
        let data = vec![3.0, 4.0, 5.0];
        assert_eq!(simd_filter_ge(&data, 4.0), vec![4.0, 5.0]);
    }

    #[test]
    fn test_filter_ge_empty() {
        assert_eq!(simd_filter_ge(&[], 0.0), Vec::<f64>::new());
    }

    // ── filter_lt / filter_le ────────────────────────────────────────────────

    #[test]
    fn test_filter_lt_basic() {
        let data = vec![1.0, 5.0, 3.0, 7.0, 2.0];
        let result = simd_filter_lt(&data, 4.0);
        assert_eq!(result, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn test_filter_le_boundary() {
        let data = vec![2.0, 3.0, 4.0, 5.0];
        assert_eq!(simd_filter_le(&data, 3.0), vec![2.0, 3.0]);
    }

    // ── filter_between ───────────────────────────────────────────────────────

    #[test]
    fn test_filter_between_basic() {
        let data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let result = simd_filter_between(&data, 3.0, 7.0);
        assert_eq!(result, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_filter_between_empty_input() {
        assert_eq!(simd_filter_between(&[], 0.0, 10.0), Vec::<f64>::new());
    }

    #[test]
    fn test_filter_between_none_pass() {
        let data = vec![1.0, 2.0];
        assert_eq!(simd_filter_between(&data, 5.0, 10.0), Vec::<f64>::new());
    }

    // ── map_mul_inplace ──────────────────────────────────────────────────────

    #[test]
    fn test_map_mul_basic() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        simd_map_mul_inplace(&mut data, 2.0);
        assert!(float_eq(&data, &[2.0, 4.0, 6.0, 8.0, 10.0]));
    }

    #[test]
    fn test_map_mul_empty() {
        let mut data: Vec<f64> = vec![];
        simd_map_mul_inplace(&mut data, 3.0);
        assert_eq!(data, Vec::<f64>::new());
    }

    #[test]
    fn test_map_mul_simd_boundary() {
        // 9 elements: 2 SIMD chunks (4+4) + 1 scalar
        let mut data = vec![1.0; 9];
        simd_map_mul_inplace(&mut data, 3.0);
        assert!(float_eq(&data, &vec![3.0; 9]));
    }

    // ── map_add / map_sub / map_neg ──────────────────────────────────────────

    #[test]
    fn test_map_add_basic() {
        let mut data = vec![1.0, 2.0, 3.0];
        simd_map_add_inplace(&mut data, 10.0);
        assert!(float_eq(&data, &[11.0, 12.0, 13.0]));
    }

    #[test]
    fn test_map_sub_basic() {
        let mut data = vec![5.0, 6.0, 7.0];
        simd_map_sub_inplace(&mut data, 3.0);
        assert!(float_eq(&data, &[2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_map_neg_basic() {
        let mut data = vec![1.0, -2.0, 3.0];
        simd_map_neg_inplace(&mut data);
        assert!(float_eq(&data, &[-1.0, 2.0, -3.0]));
    }

    #[test]
    fn test_map_neg_empty() {
        let mut data: Vec<f64> = vec![];
        simd_map_neg_inplace(&mut data);
        assert_eq!(data, Vec::<f64>::new());
    }

    // ── NaN handling ─────────────────────────────────────────────────────────

    #[test]
    fn test_filter_gt_nan_excluded() {
        // NaN comparisons always return false, so NaN is excluded; valid floats pass normally
        let data = vec![1.0, f64::NAN, 5.0];
        let result = simd_filter_gt(&data, 0.0);
        // 1.0 > 0.0 = true, NaN > 0.0 = false (excluded), 5.0 > 0.0 = true
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 5.0);
    }

    // ── consistency: large data uses SIMD path ────────────────────────────────

    #[test]
    fn test_filter_gt_large_matches_scalar() {
        // Compare SIMD result against a simple scalar reference
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let simd_result = simd_filter_gt(&data, 500.0);
        let scalar_result: Vec<f64> = data.iter().copied().filter(|&x| x > 500.0).collect();
        assert_eq!(simd_result, scalar_result);
    }

    #[test]
    fn test_filter_between_large_matches_scalar() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let simd_result = simd_filter_between(&data, 200.0, 800.0);
        let scalar_result: Vec<f64> = data.iter().copied().filter(|&x| x >= 200.0 && x <= 800.0).collect();
        assert_eq!(simd_result, scalar_result);
    }

    #[test]
    fn test_map_mul_large_matches_scalar() {
        let mut simd_data: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let mut scalar_data = simd_data.clone();
        simd_map_mul_inplace(&mut simd_data, 1.5);
        for x in &mut scalar_data { *x *= 1.5; }
        assert!(float_eq(&simd_data, &scalar_data));
    }
}
