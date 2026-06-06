//! Python-facing `Query` class.
//!
//! # Dispatch strategy
//!
//! When the user calls `Query(data)`, we inspect the data:
//!
//!   1. numpy f64 array (or list[float])  → `QueryInner::F64(NumericPipeline)`
//!   2. numpy i64 array (or list[int])    → `QueryInner::I64(IntPipeline)`
//!   3. Anything else                     → `QueryInner::Py(Vec<PyObject>)`
//!
//! For fast-path (F64/I64) operations, when the user supplies an `Expr`
//! (expression DSL) we translate it into a `NumericOp` and queue it.
//! The GIL is released during execution.
//!
//! For Python-callback operations (lambda, custom function), we fall back
//! to the `Py` path and hold the GIL.
//!
//! # GIL management
//!
//! Python callbacks: GIL held.  We call each Python callable per element.
//! Pure Rust numeric: GIL released via `py.allow_threads(|| ...)`.
//! Parallel numeric:  GIL released, rayon work-stealing, re-acquired at
//!                    collection boundary.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use super::agg::{AggSpecKind, GroupKey};
use super::columnar::{apply_columnar_ops, ColumnarData, columnar_indices_to_py_list};
use super::conversion::rust_row_to_py;
use super::expr::{ExprOp, PyExpr, PyFieldExpr};
use super::fastpath::{
    execute_lazy_float_list, execute_numpy_f32, execute_numpy_f64, filter_by_field,
    filter_by_field_py,
};
use crate::io::ParsedOutput;
use crate::core::{
    eval_filter_i64, execute_fused_f64_with_skip_take, execute_fused_i64_with_skip_take, execute_obj_pipeline, IntOp, IntPipeline, NumericOp, NumericPipeline, ObjOp, RustRow,
    RustValue,
};
use ahash::AHashMap;
use std::sync::Arc;

fn native_slice_as_bytes<T>(values: &[T]) -> &[u8] {
    let byte_len = values.len() * std::mem::size_of::<T>();
    // Native numeric Vecs are plain contiguous POD values. PyBytes copies this
    // slice once; callers reconstruct a typed array from those bytes in Python.
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, byte_len) }
}

// ---------------------------------------------------------------------------
// Core Query state
// ---------------------------------------------------------------------------

/// A single deferred operation in the lazy Python-object pipeline.
#[derive(Clone)]
enum PyPipelineOp {
    Filter(PyObject), // Python callable predicate
    Map(PyObject),    // Python callable transform
}

impl PyPipelineOp {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            PyPipelineOp::Filter(f) => PyPipelineOp::Filter(f.clone_ref(py)),
            PyPipelineOp::Map(f) => PyPipelineOp::Map(f.clone_ref(py)),
        }
    }
}

enum QueryInner {
    /// Typed f64 pipeline — GIL-free execution path
    F64(NumericPipeline),
    /// Typed i64 pipeline — GIL-free execution path
    I64(IntPipeline),
    /// Compact bool/uint8 pipeline — values are numeric 0..255 and promote to i64 for maps.
    U8 { data: Arc<Vec<u8>>, ops: Vec<IntOp> },
    /// Materialized Python objects — used for F64/I64 map fallbacks
    Py(Vec<PyObject>),
    /// Lazy Python-object pipeline.  Nothing executes until a terminal
    /// call — filter/map just accumulate ops; take short-circuits iteration.
    Obj {
        source: PyObject,
        ops: Vec<PyPipelineOp>,
    },
    /// Python list[float] stored without upfront extraction.
    /// At execution time: if take << N, iterate lazily via unsafe ob_fval reads
    /// (GIL held, early exit); otherwise materialize to Vec<f64> and use SIMD.
    LazyFloatList {
        source: PyObject,
        ops: Vec<NumericOp>,
    },
    /// Numpy f64 array stored without upfront copy.
    /// Buffer-protocol slice is passed directly to SIMD at terminal time (zero intermediate Vec).
    NumpyF64 {
        source: Py<PyAny>,
        ops: Vec<NumericOp>,
    },
    /// Numpy f32 array — f32x8 SIMD, GIL released.
    /// Results promoted to f64 / Python float at collection time.
    NumpyF32 {
        source: Py<PyAny>,
        ops: Vec<NumericOp>,
    },
    /// Rust-native object pipeline.
    /// Python dicts are converted to Arc<RustRow> once at construction (GIL held).
    /// All filter/count/sum ops run inside py.allow_threads() without GIL.
    /// Results are converted back to Python dicts at to_list() time (GIL held).
    RustObj {
        data: Arc<Vec<RustRow>>,
        ops: Vec<ObjOp>,
    },
    /// Fast field-filter path for list[dict] — avoids full dict→RustRow conversion.
    ///
    /// Execution at terminal time:
    ///   1. [GIL] Extract field values into Vec<f64> via C API (one field only)
    ///   2. [GIL released] SIMD filter → Vec<usize> of matching indices
    ///   3. [GIL] Collect original dict refs by index (zero dict copy)
    ///
    /// All ops must filter the SAME field_name.  When a different field or a lambda
    /// arrives, this path materializes and falls back to Obj/RustObj.
    ObjField {
        source: PyObject,     // original Python list[dict]
        field_name: Arc<str>, // the field being filtered
        ops: Vec<NumericOp>,  // f64 comparison ops (no field name needed — field extracted once)
    },
    /// Fast path for non-numeric field comparisons (string, bool, int equality).
    ///
    /// Unlike ObjField (which extracts to f64 + SIMD), this path loops in Rust
    /// using the C API (PyDict_GetItem + PyObject comparison) — no Python function
    /// call frames, no Python lambda overhead.
    ///
    /// Optional `map_field`: if set, `to_list()` extracts that field from each
    /// matching dict instead of returning the whole dict.  This enables full fusion
    /// of filter+map_field+take in a single Rust loop.
    ObjFieldPy {
        source: PyObject,
        ops: Vec<ObjOp>,
        map_field: Option<Arc<str>>,
    },
    /// Columnar layout for list-of-dicts (spec-082 T3).
    ///
    /// The source list-of-dicts is converted to `ColumnarData` once (at `.preload()`
    /// or implicitly on the first `field()` DSL filter).  Subsequent `field()` ops
    /// accumulate as `ObjOp` entries executed in a single Rust loop that accesses
    /// typed column slices directly — no per-row Python dict lookup.
    ColumnarObj {
        data: Arc<ColumnarData>,
        ops: Vec<ObjOp>,
    },
    /// Already-consumed or empty
    #[allow(dead_code)]
    Empty,
}

/// Lazy query pipeline.  Operations are deferred until a terminal call
/// (`.to_list()`, `.to_dict()`, `.count()`, etc.).
#[pyclass(name = "Query")]
pub struct PyQuery {
    inner: QueryInner,
    /// Pending take/skip applied at execution time
    take: Option<usize>,
    skip: usize,
    /// Whether parallel execution was requested
    parallel: bool,
}

impl PyQuery {
    fn from_inner(inner: QueryInner) -> Self {
        PyQuery {
            inner,
            take: None,
            skip: 0,
            parallel: false,
        }
    }
}

/// Fold-and-push a `NumericOp` into a Vec, collapsing consecutive scalar maps.
#[inline]
fn push_numeric_op(ops: &mut Vec<NumericOp>, op: NumericOp) {
    let folded = match (ops.last_mut(), &op) {
        (Some(NumericOp::MapMulScalar(a)), NumericOp::MapMulScalar(b)) => { *a *= b; true }
        (Some(NumericOp::MapAddScalar(a)), NumericOp::MapAddScalar(b)) => { *a += b; true }
        (Some(NumericOp::MapSubScalar(a)), NumericOp::MapSubScalar(b)) => { *a += b; true }
        (Some(NumericOp::MapDivScalar(a)), NumericOp::MapDivScalar(b)) => { *a *= b; true }
        _ => false,
    };
    if !folded {
        ops.push(op);
    }
}

pub mod construct;
pub mod filter;
pub mod map_ops;
pub mod terminal;
pub mod transform;

// ---------------------------------------------------------------------------
// Cross-module callable helpers (used by sub-modules that can't call #[pymethods] methods)

pub(super) fn collect_to_pylist(q: &PyQuery, py: Python<'_>) -> PyResult<Py<pyo3::types::PyList>> {
    use pyo3::types::PyList;
    match &q.inner {
        QueryInner::F64(pipeline) => {
            let result = execute_f64(py, pipeline, q.skip, q.take, q.parallel);
            Ok(PyList::new_bound(py, &result).into())
        }
        QueryInner::I64(pipeline) => {
            let result = execute_i64(py, pipeline, q.skip, q.take, q.parallel);
            Ok(PyList::new_bound(py, &result).into())
        }
        QueryInner::U8 { data, ops } => {
            let result = execute_u8(py, data, ops, q.skip, q.take);
            Ok(PyList::new_bound(py, &result).into())
        }
        QueryInner::Py(items) => {
            let sliced = apply_skip_take(items, q.skip, q.take);
            Ok(PyList::new_bound(py, sliced).into())
        }
        QueryInner::Obj { source, ops } => {
            let items = collect_py_lazy(py, source, ops, q.skip, q.take)?;
            Ok(PyList::new_bound(py, &items).into())
        }
        QueryInner::RustObj { data, ops } => {
            let data = std::sync::Arc::clone(data);
            let ops = ops.clone();
            let (skip, take) = (q.skip, q.take);
            let rows = py.allow_threads(move || execute_obj_pipeline(&data, &ops, skip, take));
            let list = PyList::empty_bound(py);
            for row in &rows {
                list.append(rust_row_to_py(py, row))?;
            }
            Ok(list.into())
        }
        QueryInner::LazyFloatList { source, ops } => {
            let result = execute_lazy_float_list(py, source.as_ptr(), ops, q.skip, q.take);
            Ok(PyList::new_bound(py, &result).into())
        }
        QueryInner::NumpyF64 { source, ops } => {
            let result = execute_numpy_f64(py, source, ops, q.skip, q.take)?;
            Ok(PyList::new_bound(py, &result).into())
        }
        QueryInner::NumpyF32 { source, ops } => {
            let result = execute_numpy_f32(py, source, ops, q.skip, q.take)?;
            let result_f64: Vec<f64> = result.iter().map(|&x| x as f64).collect();
            Ok(PyList::new_bound(py, &result_f64).into())
        }
        QueryInner::ObjField { source, field_name, ops } => {
            let items = filter_by_field(py, source, field_name, ops, q.skip, q.take)?;
            Ok(PyList::new_bound(py, &items).into())
        }
        QueryInner::ObjFieldPy { source, ops, map_field } => {
            let items = filter_by_field_py(py, source, ops, q.skip, q.take, map_field.as_deref())?;
            Ok(PyList::new_bound(py, items.iter().map(|o| o.bind(py))).into())
        }
        QueryInner::ColumnarObj { data, ops } => {
            let indices = apply_columnar_ops(data, ops, q.skip, q.take);
            let rows = columnar_indices_to_py_list(py, data, &indices)?;
            Ok(PyList::new_bound(py, rows.iter().map(|o| o.bind(py))).into())
        }
        QueryInner::Empty => Ok(PyList::empty_bound(py).into()),
    }
}

pub(super) fn take_query(q: &PyQuery, py: Python<'_>, n: usize) -> PyQuery {
    let inner = clone_inner(py, &q.inner);
    PyQuery {
        inner,
        take: Some(match q.take { Some(e) => e.min(n), None => n }),
        skip: q.skip,
        parallel: q.parallel,
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

// ── group_agg helpers ──────────────────────────────────────────────────────

/// Initialise per-group accumulator slots (2 f64 slots per spec).
///
/// Layout per spec i:  slot[i*2] = primary value, slot[i*2+1] = secondary (count for Mean).
/// Max/Min are initialised to ±infinity so the first real value always wins.
fn agg_init_acc(kinds: &[AggSpecKind]) -> Vec<f64> {
    let mut v = vec![0.0f64; kinds.len() * 2];
    for (i, k) in kinds.iter().enumerate() {
        match k {
            AggSpecKind::Max(_) => v[i * 2] = f64::NEG_INFINITY,
            AggSpecKind::Min(_) => v[i * 2] = f64::INFINITY,
            _ => {}
        }
    }
    v
}

/// Update all accumulators for one item (GIL held — calls Python field extractors).
fn agg_update(
    py: Python<'_>,
    acc: &mut [f64],
    kinds: &[AggSpecKind],
    item: &Bound<'_, PyAny>,
) -> PyResult<()> {
    for (i, kind) in kinds.iter().enumerate() {
        let s = i * 2;
        match kind {
            AggSpecKind::Count => {
                acc[s] += 1.0;
            }
            AggSpecKind::Sum(field_fn) => {
                let val: f64 = field_fn.bind(py).call1((item,))?.extract()?;
                acc[s] += val;
            }
            AggSpecKind::Mean(field_fn) => {
                let val: f64 = field_fn.bind(py).call1((item,))?.extract()?;
                acc[s] += val;
                acc[s + 1] += 1.0;
            }
            AggSpecKind::Max(field_fn) => {
                let val: f64 = field_fn.bind(py).call1((item,))?.extract()?;
                if val > acc[s] {
                    acc[s] = val;
                }
            }
            AggSpecKind::Min(field_fn) => {
                let val: f64 = field_fn.bind(py).call1((item,))?.extract()?;
                if val < acc[s] {
                    acc[s] = val;
                }
            }
        }
    }
    Ok(())
}

/// Convert accumulated state into Python `list[dict]`.
fn agg_build_result(
    py: Python<'_>,
    keys: Vec<PyObject>,
    accs: Vec<Vec<f64>>,
    names: &[String],
    kinds: &[AggSpecKind],
) -> PyResult<PyObject> {
    let result = PyList::empty_bound(py);
    for (key, acc) in keys.into_iter().zip(accs.iter()) {
        let row = PyDict::new_bound(py);
        row.set_item("_key", key)?;
        for (i, (name, kind)) in names.iter().zip(kinds.iter()).enumerate() {
            let s = i * 2;
            let val: PyObject = match kind {
                AggSpecKind::Count => (acc[s] as u64).into_py(py),
                AggSpecKind::Sum(_) => acc[s].into_py(py),
                AggSpecKind::Mean(_) => {
                    if acc[s + 1] == 0.0 {
                        f64::NAN.into_py(py)
                    } else {
                        (acc[s] / acc[s + 1]).into_py(py)
                    }
                }
                AggSpecKind::Max(_) | AggSpecKind::Min(_) => acc[s].into_py(py),
            };
            row.set_item(name, val)?;
        }
        result.append(row)?;
    }
    Ok(result.into())
}

fn group_agg_field_count(
    query: &PyQuery,
    py: Python<'_>,
    field_name: Arc<str>,
    names: &[String],
    kinds: &[AggSpecKind],
) -> PyResult<PyObject> {
    // ── RustObj path: GIL-free pipeline → AHashMap<GroupKey> (no Python objects in hot loop) ──
    if let QueryInner::RustObj { data, ops } = &query.inner {
        let data = Arc::clone(data);
        let ops = ops.clone();
        let (skip, take) = (query.skip, query.take);
        let rows = py.allow_threads(move || execute_obj_pipeline(&data, &ops, skip, take));

        let mut key_to_idx: AHashMap<GroupKey, usize> = AHashMap::new();
        let mut keys: Vec<GroupKey> = Vec::new();
        let mut accs: Vec<Vec<f64>> = Vec::new();

        for row in &rows {
            let key = row
                .get(field_name.as_ref())
                .map(group_key_from_rust_value)
                .unwrap_or(GroupKey::Null);
            let idx = match key_to_idx.get(&key).copied() {
                Some(i) => i,
                None => {
                    let i = keys.len();
                    key_to_idx.insert(key.clone(), i);
                    keys.push(key);
                    accs.push(agg_init_acc(kinds));
                    i
                }
            };
            for (i, kind) in kinds.iter().enumerate() {
                if matches!(kind, AggSpecKind::Count) {
                    accs[idx][i * 2] += 1.0;
                }
            }
        }
        return agg_build_result(py, group_keys_to_py(py, keys), accs, names, kinds);
    }

    // ── Obj path: iterate Python dicts directly, no dict→RustRow conversion ──
    // Uses PyDict for key→index map (Python hash/eq handles any hashable key type).
    match &query.inner {
        QueryInner::Obj { .. }
        | QueryInner::ObjField { .. }
        | QueryInner::ObjFieldPy { .. }
        | QueryInner::ColumnarObj { .. } => {}
        QueryInner::Empty => return agg_build_result(py, vec![], vec![], names, kinds),
        _ => {
            return Err(PyValueError::new_err(
                "group_agg(field(...)) requires object/dict rows",
            ))
        }
    }

    let key_to_idx = PyDict::new_bound(py);
    let mut keys: Vec<PyObject> = Vec::new();
    let mut accs: Vec<Vec<f64>> = Vec::new();

    let mut accumulate = |item: Bound<'_, PyAny>| -> PyResult<()> {
        let key: PyObject = if let Ok(dict) = item.downcast::<PyDict>() {
            dict.get_item(field_name.as_ref())?
                .map_or_else(|| py.None(), |v| v.unbind())
        } else {
            py.None()
        };
        let idx = match key_to_idx.get_item(&key)? {
            Some(v) => v.extract::<usize>()?,
            None => {
                let i = keys.len();
                keys.push(key.clone_ref(py));
                accs.push(agg_init_acc(kinds));
                key_to_idx.set_item(&key, i)?;
                i
            }
        };
        for (j, kind) in kinds.iter().enumerate() {
            if matches!(kind, AggSpecKind::Count) {
                accs[idx][j * 2] += 1.0;
            }
        }
        Ok(())
    };

    // Direct path: Obj with no pending ops and no skip/take → iterate source without materialising.
    let is_direct = matches!(&query.inner, QueryInner::Obj { ops, .. } if ops.is_empty())
        && query.skip == 0
        && query.take.is_none();

    if is_direct {
        if let QueryInner::Obj { source, .. } = &query.inner {
            for item_res in source.bind(py).iter()? {
                accumulate(item_res?)?;
            }
        }
    } else {
        let list = collect_to_pylist(&query, py)?;
        for item in list.bind(py).iter() {
            accumulate(item)?;
        }
    }

    agg_build_result(py, keys, accs, names, kinds)
}

fn group_key_from_rust_value(v: &RustValue) -> GroupKey {
    match v {
        RustValue::Null => GroupKey::Null,
        RustValue::Bool(b) => GroupKey::Bool(*b),
        RustValue::Int(i) => GroupKey::Int(*i),
        RustValue::Float(f) => GroupKey::Float(f.to_bits()),
        RustValue::Str(s) => GroupKey::Str(Arc::clone(s)),
    }
}

fn group_key_to_py(py: Python<'_>, key: &GroupKey) -> PyObject {
    match key {
        GroupKey::Null => py.None(),
        GroupKey::Bool(b) => b.into_py(py),
        GroupKey::Int(i) => i.into_py(py),
        GroupKey::Float(bits) => f64::from_bits(*bits).into_py(py),
        GroupKey::Str(s) => s.as_ref().into_py(py),
    }
}

fn group_keys_to_py(py: Python<'_>, keys: Vec<GroupKey>) -> Vec<PyObject> {
    keys.iter().map(|key| group_key_to_py(py, key)).collect()
}

// ── end group_agg helpers ──────────────────────────────────────────────────

/// Clone a QueryInner cheaply: F64/I64 via Arc (O(1)), Py via clone_ref.
fn clone_inner(py: Python<'_>, inner: &QueryInner) -> QueryInner {
    match inner {
        QueryInner::F64(p) => QueryInner::F64(branch_f64_pipeline(p)),
        QueryInner::I64(p) => QueryInner::I64(branch_i64_pipeline(p)),
        QueryInner::U8 { data, ops } => QueryInner::U8 {
            data: Arc::clone(data),
            ops: ops.clone(),
        },
        QueryInner::Py(items) => QueryInner::Py(items.iter().map(|o| o.clone_ref(py)).collect()),
        QueryInner::Obj { source, ops } => QueryInner::Obj {
            source: source.clone_ref(py),
            ops: ops.iter().map(|op| op.clone_ref(py)).collect(),
        },
        QueryInner::RustObj { data, ops } => QueryInner::RustObj {
            data: Arc::clone(data),
            ops: ops.clone(),
        },
        QueryInner::LazyFloatList { source, ops } => QueryInner::LazyFloatList {
            source: source.clone_ref(py),
            ops: ops.clone(),
        },
        QueryInner::NumpyF64 { source, ops } => QueryInner::NumpyF64 {
            source: source.clone_ref(py),
            ops: ops.clone(),
        },
        QueryInner::NumpyF32 { source, ops } => QueryInner::NumpyF32 {
            source: source.clone_ref(py),
            ops: ops.clone(),
        },
        QueryInner::ObjField {
            source,
            field_name,
            ops,
        } => QueryInner::ObjField {
            source: source.clone_ref(py),
            field_name: Arc::clone(field_name),
            ops: ops.clone(),
        },
        QueryInner::ObjFieldPy {
            source,
            ops,
            map_field,
        } => QueryInner::ObjFieldPy {
            source: source.clone_ref(py),
            ops: ops.clone(),
            map_field: map_field.as_ref().map(Arc::clone),
        },
        QueryInner::ColumnarObj { data, ops } => QueryInner::ColumnarObj {
            data: Arc::clone(data),
            ops: ops.clone(),
        },
        QueryInner::Empty => QueryInner::Empty,
    }
}

/// Lazily execute the Obj pipeline: iterate source, apply ops, respect skip/take.
/// Stops as soon as `take` items have been collected — no wasted work.
/// Bound (pre-resolved) version of a pipeline op.
/// Pre-binding callables before the element loop avoids one `bind(py)` call
/// (= one atomic refcount increment) per element per op.
enum BoundOp<'py> {
    Filter(Bound<'py, PyAny>),
    Map(Bound<'py, PyAny>),
}

fn bind_ops<'py>(py: Python<'py>, ops: &[PyPipelineOp]) -> Vec<BoundOp<'py>> {
    // bind(py) returns &Bound — clone() to get owned Bound (one refcount bump per
    // op, vs the previous approach of one bump per element per op).
    ops.iter()
        .map(|op| match op {
            PyPipelineOp::Filter(f) => BoundOp::Filter(f.bind(py).clone()),
            PyPipelineOp::Map(f) => BoundOp::Map(f.bind(py).clone()),
        })
        .collect()
}

fn collect_py_lazy(
    py: Python<'_>,
    source: &PyObject,
    ops: &[PyPipelineOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<Vec<PyObject>> {
    // Pre-bind callables once — avoids bind(py) refcount bump per element per op.
    let bound = bind_ops(py, ops);

    let iter = source.bind(py).iter()?;
    let mut out: Vec<PyObject> = Vec::new();
    let mut skipped = 0usize;

    'outer: for item_res in iter {
        let mut item: PyObject = item_res?.into();

        for op in &bound {
            match op {
                BoundOp::Filter(pred) => {
                    if !pred.call1((item.clone_ref(py),))?.is_truthy()? {
                        continue 'outer;
                    }
                }
                BoundOp::Map(f) => {
                    item = f.call1((item,))?.into();
                }
            }
        }

        if skipped < skip {
            skipped += 1;
            continue;
        }
        out.push(item);
        if take.is_some_and(|n| out.len() >= n) {
            break;
        }
    }
    Ok(out)
}

/// Count-only variant: same as collect_py_lazy but never builds the output Vec.
fn count_py_lazy(
    py: Python<'_>,
    source: &PyObject,
    ops: &[PyPipelineOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<usize> {
    let bound = bind_ops(py, ops);

    let iter = source.bind(py).iter()?;
    let mut count = 0usize;
    let mut skipped = 0usize;

    'outer: for item_res in iter {
        let mut item: PyObject = item_res?.into();

        for op in &bound {
            match op {
                BoundOp::Filter(pred) => {
                    if !pred.call1((item.clone_ref(py),))?.is_truthy()? {
                        continue 'outer;
                    }
                }
                BoundOp::Map(f) => {
                    item = f.call1((item,))?.into();
                }
            }
        }

        if skipped < skip {
            skipped += 1;
            continue;
        }
        count += 1;
        if take.is_some_and(|n| count >= n) {
            break;
        }
    }
    Ok(count)
}

/// Single-pass filter + group: applies pending ops and groups results by key_fn.
/// Returns a Python dict {key: [items]}, avoiding the double-pass overhead of
/// to_list() followed by Python-side GroupBy construction.
fn group_by_collect(
    py: Python<'_>,
    source: &PyObject,
    ops: &[PyPipelineOp],
    key_fn: &Bound<'_, PyAny>,
    skip: usize,
    take: Option<usize>,
) -> PyResult<PyObject> {
    let bound = bind_ops(py, ops);
    let dict = PyDict::new_bound(py);

    let iter = source.bind(py).iter()?;
    let mut skipped = 0usize;
    let mut collected = 0usize;

    'outer: for item_res in iter {
        let mut item: PyObject = item_res?.into();

        for op in &bound {
            match op {
                BoundOp::Filter(pred) => {
                    if !pred.call1((item.clone_ref(py),))?.is_truthy()? {
                        continue 'outer;
                    }
                }
                BoundOp::Map(f) => {
                    item = f.call1((item,))?.into();
                }
            }
        }

        if skipped < skip {
            skipped += 1;
            continue;
        }

        // Compute the group key and append item to the group list.
        let key = key_fn.call1((item.clone_ref(py),))?;
        match dict.get_item(&key)? {
            Some(list) => list.downcast::<PyList>()?.append(item.bind(py))?,
            None => dict.set_item(&key, PyList::new_bound(py, [item]))?,
        }

        collected += 1;
        if take.is_some_and(|n| collected >= n) {
            break;
        }
    }

    Ok(dict.into())
}

/// Execute a `NumericPipeline`, releasing the GIL.
///
/// The key fix vs the original design: `pipeline.arc()` returns a clone of
/// the Arc pointer (8 bytes), NOT a clone of the Vec<f64> (O(N) bytes).
/// All ops are accumulated as a small Vec<NumericOp> and applied in ONE scan.
fn execute_f64(
    py: Python<'_>,
    pipeline: &NumericPipeline,
    skip: usize,
    take: Option<usize>,
    parallel: bool,
) -> Vec<f64> {
    let data = pipeline.arc();
    let ops = pipeline.clone_ops(); // value ops only — no Skip/Take mixed in

    // Release GIL — pure Rust from here
    py.allow_threads(move || {
        #[cfg(feature = "parallel")]
        if parallel {
            return NumericPipeline::from_arc(Arc::clone(&data))
                .with_ops(ops)
                .execute_parallel_with_skip_take(skip, take);
        }
        // skip and take are passed as explicit bounds — NOT inserted into ops.
        // Semantics: skip = source-level (before filters), take = output-level (after filters).
        // This avoids the O(n) ops.insert(0, ...) and clarifies execution order.
        execute_fused_f64_with_skip_take(&data, &ops, skip, take)
    })
}

fn execute_i64(
    py: Python<'_>,
    pipeline: &IntPipeline,
    skip: usize,
    take: Option<usize>,
    parallel: bool,
) -> Vec<i64> {
    let data = pipeline.arc();
    let ops = pipeline.clone_ops();

    py.allow_threads(move || {
        #[cfg(feature = "parallel")]
        if parallel {
            return IntPipeline::from_arc(Arc::clone(&data))
                .with_ops(ops)
                .execute_parallel_with_skip_take(skip, take);
        }
        execute_fused_i64_with_skip_take(&data, &ops, skip, take)
    })
}

fn execute_u8(
    py: Python<'_>,
    data: &Arc<Vec<u8>>,
    ops: &[IntOp],
    skip: usize,
    take: Option<usize>,
) -> Vec<i64> {
    let data = Arc::clone(data);
    let ops = ops.to_vec();
    py.allow_threads(move || execute_u8_with_skip_take(&data, &ops, skip, take))
}

fn execute_u8_with_skip_take(
    data: &[u8],
    ops: &[IntOp],
    out_skip: usize,
    out_take: Option<usize>,
) -> Vec<i64> {
    let est = out_take.unwrap_or(data.len());
    let mut out = Vec::with_capacity(est.min(data.len()));
    let mut skipped = 0usize;

    'outer: for &raw in data {
        let val = raw as i64;
        for op in ops {
            if !eval_filter_i64(val, op) {
                continue 'outer;
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

fn count_u8_with_skip_take(data: &[u8], ops: &[IntOp], out_skip: usize, out_take: Option<usize>) -> usize {
    // Fast path: single filter op, no skip/take — compare u8 directly, no i64 cast.
    // The compiler auto-vectorizes these simple byte-level filter+count loops.
    if out_skip == 0 && out_take.is_none() && ops.len() == 1 {
        return match ops[0] {
            IntOp::FilterGt(t) => {
                if t < 0 {
                    data.len()
                } else if t >= 255 {
                    0
                } else {
                    let t = t as u8;
                    data.iter().filter(|&&x| x > t).count()
                }
            }
            IntOp::FilterGe(t) => {
                if t <= 0 {
                    data.len()
                } else if t > 255 {
                    0
                } else {
                    let t = t as u8;
                    data.iter().filter(|&&x| x >= t).count()
                }
            }
            IntOp::FilterLt(t) => {
                if t <= 0 {
                    0
                } else if t > 255 {
                    data.len()
                } else {
                    let t = t as u8;
                    data.iter().filter(|&&x| x < t).count()
                }
            }
            IntOp::FilterLe(t) => {
                if t < 0 {
                    0
                } else if t >= 255 {
                    data.len()
                } else {
                    let t = t as u8;
                    data.iter().filter(|&&x| x <= t).count()
                }
            }
            IntOp::FilterEq(t) => {
                if t < 0 || t > 255 {
                    0
                } else {
                    let t = t as u8;
                    data.iter().filter(|&&x| x == t).count()
                }
            }
            IntOp::FilterNe(t) => {
                if t < 0 || t > 255 {
                    data.len()
                } else {
                    let t = t as u8;
                    data.iter().filter(|&&x| x != t).count()
                }
            }
            _ => data.len(), // map ops pass all elements
        };
    }
    // Scalar fallback for multi-op or skip/take
    let mut count = 0usize;
    let mut skipped = 0usize;

    'outer: for &raw in data {
        let val = raw as i64;
        for op in ops {
            if !eval_filter_i64(val, op) {
                continue 'outer;
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

fn sum_u8_with_skip_take(data: &[u8], ops: &[IntOp], out_skip: usize, out_take: Option<usize>) -> i64 {
    let mut sum = 0i64;
    let mut emitted = 0usize;
    let mut skipped = 0usize;

    'outer: for &raw in data {
        let val = raw as i64;
        for op in ops {
            if !eval_filter_i64(val, op) {
                continue 'outer;
            }
        }
        if skipped < out_skip {
            skipped += 1;
            continue;
        }
        sum += val;
        emitted += 1;
        if out_take.is_some_and(|n| emitted >= n) {
            break;
        }
    }
    sum
}

fn collect_f64_pipeline(
    py: Python<'_>,
    pipeline: &NumericPipeline,
    skip: usize,
    take: Option<usize>,
) -> PyResult<Vec<f64>> {
    Ok(execute_f64(py, pipeline, skip, take, false))
}

fn collect_i64_pipeline(
    py: Python<'_>,
    pipeline: &IntPipeline,
    skip: usize,
    take: Option<usize>,
) -> PyResult<Vec<i64>> {
    Ok(execute_i64(py, pipeline, skip, take, false))
}

/// Batch size for chunked Python predicate calls.
/// Reduces Rust→Python FFI roundtrips from N to N/FILTER_CHUNK_SIZE.
const FILTER_CHUNK_SIZE: usize = 256;

/// Get Python builtins `filter` and `list` callables (cached inline).
fn py_filter_list<'py>(
    py: Python<'py>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let b = py.import_bound("builtins")?;
    Ok((b.getattr("filter")?, b.getattr("list")?))
}

/// Apply a Python filter predicate to f64 data.
///
/// Processes data in chunks of FILTER_CHUNK_SIZE, calling `list(filter(pred, chunk))`
/// once per chunk. This moves N predicate invocations from Rust's call1 machinery
/// into Python's C evaluation loop, reducing FFI roundtrips by ~256x.
fn apply_py_filter_f64(
    py: Python<'_>,
    data: Vec<f64>,
    pred: &Bound<'_, PyAny>,
) -> PyResult<Vec<f64>> {
    let (filter_fn, list_fn) = py_filter_list(py)?;
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks(FILTER_CHUNK_SIZE) {
        let py_chunk = PyList::new_bound(py, chunk.iter().map(|&v| v));
        let filtered = list_fn.call1((filter_fn.call1((pred, &py_chunk))?,))?;
        for item in filtered.downcast::<PyList>()? {
            out.push(item.extract::<f64>()?);
        }
    }
    Ok(out)
}

fn apply_py_filter_i64(
    py: Python<'_>,
    data: Vec<i64>,
    pred: &Bound<'_, PyAny>,
) -> PyResult<Vec<i64>> {
    let (filter_fn, list_fn) = py_filter_list(py)?;
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks(FILTER_CHUNK_SIZE) {
        let py_chunk = PyList::new_bound(py, chunk.iter().map(|&v| v));
        let filtered = list_fn.call1((filter_fn.call1((pred, &py_chunk))?,))?;
        for item in filtered.downcast::<PyList>()? {
            out.push(item.extract::<i64>()?);
        }
    }
    Ok(out)
}

fn apply_py_filter(
    py: Python<'_>,
    items: &[PyObject],
    pred: &Bound<'_, PyAny>,
    skip: usize,
    take: Option<usize>,
) -> PyResult<Vec<PyObject>> {
    let items = &items[skip.min(items.len())..];
    let cap = take.unwrap_or(items.len());
    let mut out = Vec::with_capacity(cap);

    let (filter_fn, list_fn) = py_filter_list(py)?;
    'outer: for chunk in items.chunks(FILTER_CHUNK_SIZE) {
        let py_chunk = PyList::new_bound(py, chunk.iter().map(|o| o.clone_ref(py)));
        let filtered = list_fn.call1((filter_fn.call1((pred, &py_chunk))?,))?;
        for item in filtered.downcast::<PyList>()? {
            out.push(item.unbind());
            if let Some(n) = take {
                if out.len() >= n {
                    break 'outer;
                }
            }
        }
    }
    Ok(out)
}

enum MappedResult {
    F64(Vec<f64>),
    Py(Vec<PyObject>),
}

fn apply_py_map_f64(
    py: Python<'_>,
    data: Vec<f64>,
    f: &Bound<'_, PyAny>,
) -> PyResult<MappedResult> {
    let mut float_out: Vec<f64> = Vec::with_capacity(data.len());
    let mut py_out: Option<Vec<PyObject>> = None;

    for (_i, val) in data.iter().enumerate() {
        let result = f.call1((val.into_py(py),))?;
        if py_out.is_none() {
            if let Ok(fv) = result.extract::<f64>() {
                float_out.push(fv);
                continue;
            }
            // First non-float output — switch to Py mode
            let mut pyvec: Vec<PyObject> = float_out.drain(..).map(|v| v.into_py(py)).collect();
            pyvec.push(result.into());
            py_out = Some(pyvec);
        } else {
            py_out.as_mut().unwrap().push(result.into());
        }
    }

    Ok(match py_out {
        Some(v) => MappedResult::Py(v),
        None => MappedResult::F64(float_out),
    })
}

enum MappedResultI {
    I64(Vec<i64>),
    F64(Vec<f64>),
    Py(Vec<PyObject>),
}

fn apply_py_map_i64(
    py: Python<'_>,
    data: Vec<i64>,
    f: &Bound<'_, PyAny>,
) -> PyResult<MappedResultI> {
    let mut int_out: Vec<i64> = Vec::with_capacity(data.len());
    let mut float_out: Option<Vec<f64>> = None;
    let mut py_out: Option<Vec<PyObject>> = None;

    for val in data {
        let result = f.call1((val.into_py(py),))?;
        if py_out.is_none() && float_out.is_none() {
            if let Ok(iv) = result.extract::<i64>() {
                int_out.push(iv);
                continue;
            }
            if let Ok(fv) = result.extract::<f64>() {
                let fvec: Vec<f64> = int_out.drain(..).map(|v| v as f64).collect();
                float_out = Some(fvec);
                float_out.as_mut().unwrap().push(fv);
                continue;
            }
            let mut pyvec: Vec<PyObject> = int_out.drain(..).map(|v| v.into_py(py)).collect();
            pyvec.push(result.into());
            py_out = Some(pyvec);
        } else if let Some(ref mut fvec) = float_out {
            if let Ok(fv) = result.extract::<f64>() {
                fvec.push(fv);
            } else {
                let mut pyvec: Vec<PyObject> = fvec.drain(..).map(|v| v.into_py(py)).collect();
                pyvec.push(result.into());
                py_out = Some(pyvec);
                float_out = None;
            }
        } else if let Some(ref mut pvec) = py_out {
            pvec.push(result.into());
        }
    }

    Ok(if let Some(v) = py_out {
        MappedResultI::Py(v)
    } else if let Some(v) = float_out {
        MappedResultI::F64(v)
    } else {
        MappedResultI::I64(int_out)
    })
}

fn apply_py_map(
    py: Python<'_>,
    items: &[PyObject],
    f: &Bound<'_, PyAny>,
    skip: usize,
    take: Option<usize>,
) -> PyResult<Vec<PyObject>> {
    let mut out = Vec::new();
    let mut skipped = 0;
    for item in items {
        if skipped < skip {
            skipped += 1;
            continue;
        }
        let mapped = f.call1((item.clone_ref(py),))?;
        out.push(mapped.into());
        if let Some(n) = take {
            if out.len() >= n {
                break;
            }
        }
    }
    Ok(out)
}

/// Try to parse a Python lambda to a DSL Expr via `zpyflow._lambda_parser`.
/// Returns Some(Expr) if the lambda is a simple comparison/arithmetic; None otherwise.
/// Never panics — all Python errors are swallowed.
fn try_lambda_to_dsl_expr<'py>(
    py: Python<'py>,
    pred: &Bound<'py, PyAny>,
) -> Option<Bound<'py, PyAny>> {
    // Only attempt for plain Python functions (not C callables, not Expr objects)
    let builtins = py.import_bound("builtins").ok()?;
    let callable = builtins.getattr("callable").ok()?;
    if !callable.call1((pred,)).ok()?.is_truthy().unwrap_or(false) {
        return None;
    }
    // Skip if it's already a ZPyFlow DSL object
    if pred.extract::<PyRef<PyExpr>>().is_ok() || pred.extract::<PyRef<PyFieldExpr>>().is_ok() {
        return None;
    }

    let parser = py.import_bound("zpyflow._lambda_parser").ok()?;
    let result = parser
        .getattr("try_lambda_to_expr")
        .ok()?
        .call1((pred,))
        .ok()?;
    if result.is_none() {
        return None;
    }
    Some(result)
}

fn apply_skip_take<'a>(items: &'a [PyObject], skip: usize, take: Option<usize>) -> &'a [PyObject] {
    let start = skip.min(items.len());
    let sliced = &items[start..];
    match take {
        Some(n) => &sliced[..n.min(sliced.len())],
        None => sliced,
    }
}

/// Clone the pipeline descriptor (Arc pointer + small ops Vec).
/// Data is NOT copied — Arc refcount is bumped only.
fn branch_f64_pipeline(pipeline: &NumericPipeline) -> NumericPipeline {
    NumericPipeline::from_arc(pipeline.arc()).with_ops(pipeline.clone_ops())
}

fn branch_i64_pipeline(pipeline: &IntPipeline) -> IntPipeline {
    IntPipeline::from_arc(pipeline.arc()).with_ops(pipeline.clone_ops())
}

fn expr_to_int_op(op: &ExprOp) -> Option<IntOp> {
    match op {
        ExprOp::Gt(t) => Some(IntOp::FilterGt(*t as i64)),
        ExprOp::Ge(t) => Some(IntOp::FilterGe(*t as i64)),
        ExprOp::Lt(t) => Some(IntOp::FilterLt(*t as i64)),
        ExprOp::Le(t) => Some(IntOp::FilterLe(*t as i64)),
        ExprOp::Eq(t) => Some(IntOp::FilterEq(*t as i64)),
        ExprOp::Ne(t) => Some(IntOp::FilterNe(*t as i64)),
        ExprOp::MulScalar(s) => Some(IntOp::MapMulScalar(*s as i64)),
        ExprOp::AddScalar(s) => Some(IntOp::MapAddScalar(*s as i64)),
        ExprOp::SubScalar(s) => Some(IntOp::MapSubScalar(*s as i64)),
        ExprOp::Abs => Some(IntOp::MapAbs),
        ExprOp::Neg => Some(IntOp::MapNeg),
        _ => None, // No integer equivalent for sqrt, floor, etc.
    }
}

/// Convert a `ParsedOutput` to a `PyQuery`. GIL must be held for `Strs` path.
fn parsed_to_query(py: Python<'_>, out: ParsedOutput) -> PyQuery {
    PyQuery::from_inner(match out {
        ParsedOutput::F64(v) => QueryInner::F64(NumericPipeline::new(v)),
        ParsedOutput::I64(v) => QueryInner::I64(IntPipeline::new(v)),
        ParsedOutput::Rows(v) => QueryInner::RustObj {
            data: Arc::new(v),
            ops: Vec::new(),
        },
        ParsedOutput::Strs(v) => {
            let objs: Vec<PyObject> = v.into_iter().map(|s| s.into_py(py)).collect();
            QueryInner::Py(objs)
        }
    })
}
