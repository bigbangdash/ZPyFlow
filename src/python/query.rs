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

use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use super::agg::{AggSpecKind, GroupKey, PyAggSpec};
use super::conversion::{rust_row_to_py, try_convert_to_rust_obj};
use super::expr::{ExprOp, PyExpr, PyFieldExpr};
use super::fastpath::{
    count_by_field, count_by_field_py, count_lazy_float_list, count_numpy_f32, count_numpy_f64,
    execute_lazy_float_list, execute_numpy_f32, execute_numpy_f64, filter_by_field,
    filter_by_field_py, objop_is_field_filter, objop_to_numeric, sum_field_by_field,
};
use super::io_bridge::{csv_col_spec, parse_csv, parse_jsonl};
use crate::io::ParsedOutput;
use crate::pipeline::numeric::{
    count_fused_f64_bounded, count_fused_i64_bounded, eval_filter_f64, eval_filter_i64,
    execute_fused_f64_bounded, execute_fused_i64_bounded, filter_max_fused_f64,
    filter_mean_fused_f64, filter_min_fused_f64, filter_multi_stat_f64, filter_sum_fused_f64,
    filter_var_fused_f64, IntOp, IntPipeline, NumericOp, NumericPipeline,
};
use crate::pipeline::obj::{
    count_obj_pipeline, execute_obj_pipeline, row_passes, sum_field_obj_pipeline, ObjOp, RustRow,
    RustValue,
};
use ahash::AHashMap;
use std::sync::Arc;

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

#[pymethods]
impl PyQuery {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    #[new]
    #[pyo3(signature = (data, /))]
    fn new(_py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(list) = data.downcast::<pyo3::types::PyList>() {
            if list.is_empty() {
                return Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(
                    vec![],
                ))));
            }
            let first = list.get_item(0)?;
            // list[dict] → Obj (lazy; converts to RustObj on first field() DSL op)
            if first.is_instance_of::<PyDict>() {
                let source = data.unbind();
                return Ok(PyQuery::from_inner(QueryInner::Obj {
                    source,
                    ops: Vec::new(),
                }));
            }
            // list[float] → lazy extraction
            if first.is_instance_of::<pyo3::types::PyFloat>() {
                return Ok(PyQuery::from_inner(QueryInner::LazyFloatList {
                    source: data.clone().unbind(),
                    ops: Vec::new(),
                }));
            }
            // list[int] → try i64 before f64 (Python ints coerce to f64, so order matters)
            if first.is_instance_of::<pyo3::types::PyInt>() {
                if let Ok(v) = data.extract::<Vec<i64>>() {
                    return Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(v))));
                }
            }
        }
        // Non-list typed extraction (numpy arrays, tuples, etc.)
        if let Ok(v) = data.extract::<Vec<f64>>() {
            return Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(
                v,
            ))));
        }
        if let Ok(v) = data.extract::<Vec<i64>>() {
            return Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(v))));
        }
        // Generic Python iterable fallback
        let source = data.unbind();
        Ok(PyQuery::from_inner(QueryInner::Obj {
            source,
            ops: Vec::new(),
        }))
    }

    // ------------------------------------------------------------------
    // Buffer protocol constructors (fast path for numpy arrays)
    //
    // Uses PyObject_GetBuffer (Py_LIMITED_API) via RawBuffer — works with abi3.
    // PyBUF_C_CONTIGUOUS guarantees 1-D C-contiguous layout; non-contiguous or
    // F-order arrays will raise BufferError at this call.
    // Python side: Query._from_buffer_f64(arr)  (arr must be C-contiguous)
    // ------------------------------------------------------------------

    #[staticmethod]
    fn _from_buffer_f64(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        use super::fastpath::RawBuffer;
        let buf = unsafe { RawBuffer::get(py, obj.as_ptr()) }?;
        if buf.ndim() != 1 {
            return Err(PyValueError::new_err("_from_buffer_f64: expected 1-D array"));
        }
        // C-contiguity is guaranteed by PyBUF_C_CONTIGUOUS; buf dropped here.
        Ok(PyQuery::from_inner(QueryInner::NumpyF64 {
            source: obj.clone().unbind(),
            ops: Vec::new(),
        }))
    }

    #[staticmethod]
    fn _from_buffer_i64(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        use super::fastpath::RawBuffer;
        let buf = unsafe { RawBuffer::get(py, obj.as_ptr()) }?;
        if buf.ndim() != 1 {
            return Err(PyValueError::new_err("_from_buffer_i64: expected 1-D array"));
        }
        let n = buf.item_count();
        let src = buf.buf_ptr::<i64>() as usize;
        let mut data = vec![0i64; n];
        let dst = data.as_mut_ptr() as usize;
        py.allow_threads(move || unsafe {
            std::ptr::copy_nonoverlapping(src as *const i64, dst as *mut i64, n);
        });
        Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(data))))
    }

    #[staticmethod]
    fn _from_buffer_f32(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        use super::fastpath::RawBuffer;
        let buf = unsafe { RawBuffer::get(py, obj.as_ptr()) }?;
        if buf.ndim() != 1 {
            return Err(PyValueError::new_err("_from_buffer_f32: expected 1-D array"));
        }
        let _ = py;
        Ok(PyQuery::from_inner(QueryInner::NumpyF32 {
            source: obj.clone().unbind(),
            ops: Vec::new(),
        }))
    }

    #[staticmethod]
    fn _from_buffer_u8(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        use super::fastpath::RawBuffer;
        let buf = unsafe { RawBuffer::get(py, obj.as_ptr()) }?;
        if buf.ndim() != 1 {
            return Err(PyValueError::new_err("_from_buffer_u8: expected 1-D array"));
        }
        let n = buf.item_count();
        let src = buf.buf_ptr::<u8>() as usize;
        let mut data = vec![0u8; n];
        let dst = data.as_mut_ptr() as usize;
        py.allow_threads(move || unsafe {
            std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, n);
        });
        Ok(PyQuery::from_inner(QueryInner::U8 {
            data: Arc::new(data),
            ops: Vec::new(),
        }))
    }

    // ------------------------------------------------------------------
    // Explicit typed constructors — for mixed-type lists or type coercion
    // ------------------------------------------------------------------

    /// Construct a Query with explicit f64 coercion.
    ///
    /// Use this when your list contains mixed numeric types (e.g. `[1, 2, 3.0]`)
    /// and you want to guarantee the f64 fast path (SIMD, GIL-free).
    ///
    /// Raises ``ValueError`` if any element cannot be converted to float.
    ///
    /// Example:
    /// ```python
    /// Query.f64([1, 2, 3.0]).filter(col > 1).to_list()  # → [2.0, 3.0]
    /// ```
    #[staticmethod]
    fn f64(py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<Self> {
        let list = data.iter()?;
        let mut out = Vec::new();
        for item in list {
            let item = item?;
            let v: f64 = item.extract().map_err(|_| {
                let repr = item.repr().map(|s| s.to_string()).unwrap_or_else(|_| "?".to_string());
                PyValueError::new_err(format!("Query.f64(): cannot convert {repr} to float"))
            })?;
            out.push(v);
        }
        let _ = py;
        Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(out))))
    }

    /// Construct a Query with explicit i64 coercion.
    ///
    /// Use this when your list contains mixed integer types and you want to
    /// guarantee the i64 fast path (GIL-free).
    ///
    /// Raises ``ValueError`` if any element cannot be converted to int.
    ///
    /// Example:
    /// ```python
    /// Query.i64([1, 2, 3]).filter(col > 1).to_list()  # → [2, 3]
    /// ```
    #[staticmethod]
    fn i64(py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<Self> {
        let list = data.iter()?;
        let mut out = Vec::new();
        for item in list {
            let item = item?;
            let v: i64 = item.extract().map_err(|_| {
                let repr = item.repr().map(|s| s.to_string()).unwrap_or_else(|_| "?".to_string());
                PyValueError::new_err(format!("Query.i64(): cannot convert {repr} to int"))
            })?;
            out.push(v);
        }
        let _ = py;
        Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(out))))
    }

    // ------------------------------------------------------------------
    // CSV / JSON Lines constructors (GIL-free parse)
    //
    // Path variants:  Rust opens and reads the file inside allow_threads.
    // Bytes variants: caller read the content (GIL); Rust parses it GIL-free.
    //
    // column_name / column_idx: mutually exclusive; both None → all rows (dict).
    // field_name (JSONL): None → all rows; Some → extract that field.
    // dtype: "auto" | "float" | "int" | "str"
    // ------------------------------------------------------------------

    #[staticmethod]
    #[pyo3(signature = (path, column_name=None, column_idx=None, dtype="auto", delimiter=",", has_header=true))]
    fn _from_csv_path(
        py: Python<'_>,
        path: &str,
        column_name: Option<String>,
        column_idx: Option<usize>,
        dtype: &str,
        delimiter: &str,
        has_header: bool,
    ) -> PyResult<Self> {
        let path = path.to_string();
        let dtype = dtype.to_string();
        let delim = delimiter.as_bytes().first().copied().unwrap_or(b',');
        let col = csv_col_spec(column_name, column_idx);

        let out = py
            .allow_threads(move || -> Result<ParsedOutput, String> {
                let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
                parse_csv(bytes, col, &dtype, delim, has_header)
            })
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(parsed_to_query(py, out))
    }

    #[staticmethod]
    #[pyo3(signature = (data, column_name=None, column_idx=None, dtype="auto", delimiter=",", has_header=true))]
    fn _from_csv_bytes(
        py: Python<'_>,
        data: &[u8],
        column_name: Option<String>,
        column_idx: Option<usize>,
        dtype: &str,
        delimiter: &str,
        has_header: bool,
    ) -> PyResult<Self> {
        let bytes = data.to_vec(); // copy under GIL; parse runs GIL-free
        let dtype = dtype.to_string();
        let delim = delimiter.as_bytes().first().copied().unwrap_or(b',');
        let col = csv_col_spec(column_name, column_idx);

        let out = py
            .allow_threads(move || -> Result<ParsedOutput, String> {
                parse_csv(bytes, col, &dtype, delim, has_header)
            })
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(parsed_to_query(py, out))
    }

    #[staticmethod]
    #[pyo3(signature = (path, field_name=None, dtype="auto"))]
    fn _from_jsonl_path(
        py: Python<'_>,
        path: &str,
        field_name: Option<String>,
        dtype: &str,
    ) -> PyResult<Self> {
        let path = path.to_string();
        let dtype = dtype.to_string();
        let out = py
            .allow_threads(move || -> Result<ParsedOutput, String> {
                let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
                parse_jsonl(bytes, field_name, &dtype)
            })
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(parsed_to_query(py, out))
    }

    #[staticmethod]
    #[pyo3(signature = (data, field_name=None, dtype="auto"))]
    fn _from_jsonl_bytes(
        py: Python<'_>,
        data: &[u8],
        field_name: Option<String>,
        dtype: &str,
    ) -> PyResult<Self> {
        let bytes = data.to_vec();
        let dtype = dtype.to_string();
        let out = py
            .allow_threads(move || -> Result<ParsedOutput, String> {
                parse_jsonl(bytes, field_name, &dtype)
            })
            .map_err(|e| PyValueError::new_err(e))?;

        Ok(parsed_to_query(py, out))
    }

    // ------------------------------------------------------------------
    // filter(expr_or_callable)
    // ------------------------------------------------------------------

    fn filter(&self, py: Python<'_>, pred: Bound<'_, PyAny>) -> PyResult<PyQuery> {
        match &self.inner {
            // Fast path: Expr DSL
            QueryInner::F64(pipeline) => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if let Some(op) = expr.op.to_f64_op() {
                        if self.skip == 0 && self.take.is_none() {
                            let new_inner = branch_f64_pipeline(pipeline).push_op(op);
                            return Ok(PyQuery::from_inner(QueryInner::F64(new_inner)));
                        }
                        // Pending skip/take: materialize to preserve pipeline-order semantics.
                        // filter(col>0).skip(5).filter(col<9) must skip AFTER the first filter.
                        let data = collect_f64_pipeline(py, pipeline, self.skip, self.take)?;
                        return Ok(PyQuery::from_inner(QueryInner::F64(
                            NumericPipeline::new(data).push_op(op),
                        )));
                    }
                }
                // Lambda AST fast path: safe only when no pending skip/take.
                if self.skip == 0 && self.take.is_none() {
                    if let Some(dsl) = try_lambda_to_dsl_expr(py, &pred) {
                        if let Ok(expr) = dsl.extract::<PyRef<PyExpr>>() {
                            if let Some(op) = expr.op.to_f64_op() {
                                let new_inner = branch_f64_pipeline(pipeline).push_op(op);
                                return Ok(PyQuery::from_inner(QueryInner::F64(new_inner)));
                            }
                        }
                    }
                }
                // Fall through to Python callback path for f64
                let data = collect_f64_pipeline(py, pipeline, self.skip, self.take)?;
                let filtered = apply_py_filter_f64(py, data, &pred)?;
                Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(
                    filtered,
                ))))
            }
            QueryInner::I64(pipeline) => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    let int_op = expr_to_int_op(&expr.op);
                    if let Some(op) = int_op {
                        if self.skip == 0 && self.take.is_none() {
                            let new_inner = branch_i64_pipeline(pipeline).push_op(op);
                            return Ok(PyQuery::from_inner(QueryInner::I64(new_inner)));
                        }
                        let data = collect_i64_pipeline(py, pipeline, self.skip, self.take)?;
                        return Ok(PyQuery::from_inner(QueryInner::I64(
                            IntPipeline::new(data).push_op(op),
                        )));
                    }
                }
                // Python callback on i64
                let data = collect_i64_pipeline(py, pipeline, self.skip, self.take)?;
                let filtered = apply_py_filter_i64(py, data, &pred)?;
                Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(
                    filtered,
                ))))
            }
            QueryInner::U8 { data, ops } => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if expr.op.is_filter() {
                        if let Some(op) = expr_to_int_op(&expr.op) {
                            let mut new_ops = ops.clone();
                            new_ops.push(op);
                            return Ok(PyQuery::from_inner(QueryInner::U8 {
                                data: Arc::clone(data),
                                ops: new_ops,
                            }));
                        }
                    }
                }
                let data = execute_u8(py, data, ops, self.skip, self.take);
                let filtered = apply_py_filter_i64(py, data, &pred)?;
                Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(
                    filtered,
                ))))
            }
            QueryInner::Py(items) => {
                let filtered = apply_py_filter(py, items, &pred, self.skip, self.take)?;
                Ok(PyQuery::from_inner(QueryInner::Py(filtered)))
            }
            QueryInner::RustObj { data, ops } => {
                // FieldExpr DSL → ObjOp, stays GIL-free
                if let Ok(fexpr) = pred.extract::<PyRef<PyFieldExpr>>() {
                    if let Some(op) = fexpr.op.clone() {
                        let mut new_ops = ops.clone();
                        new_ops.push(op);
                        return Ok(PyQuery::from_inner(QueryInner::RustObj {
                            data: Arc::clone(data),
                            ops: new_ops,
                        }));
                    }
                }
                // Python lambda fallback: run existing ObjOps GIL-free, materialize,
                // then hand off to the Obj pipeline with the new lambda.
                let materialized = execute_obj_pipeline(data, ops, self.skip, self.take);
                let mat_list = PyList::empty_bound(py);
                for row in &materialized {
                    mat_list.append(rust_row_to_py(py, row))?;
                }
                Ok(PyQuery::from_inner(QueryInner::Obj {
                    source: mat_list.unbind().into_py(py),
                    ops: vec![PyPipelineOp::Filter(pred.unbind())],
                }))
            }
            QueryInner::Obj { source, ops } => {
                // FieldExpr on fresh Obj → fast field path (no Python lambda overhead)
                if ops.is_empty() && self.skip == 0 && self.take.is_none() {
                    if let Ok(fexpr) = pred.extract::<PyRef<PyFieldExpr>>() {
                        if let Some(fop) = &fexpr.op {
                            // Numeric → ObjField (f64 extract + SIMD)
                            if let Some((fname, nop)) = objop_to_numeric(fop) {
                                return Ok(PyQuery::from_inner(QueryInner::ObjField {
                                    source: source.clone_ref(py),
                                    field_name: fname,
                                    ops: vec![nop],
                                }));
                            }
                            // Non-numeric (string, bool, int Eq/Ne) → ObjFieldPy (C-level, no Python frame)
                            if objop_is_field_filter(fop) {
                                return Ok(PyQuery::from_inner(QueryInner::ObjFieldPy {
                                    source: source.clone_ref(py),
                                    ops: vec![fop.clone()],
                                    map_field: None,
                                }));
                            }
                        }
                    }
                }
                // Lambda or non-fresh: stay as Obj (FieldExpr has __call__ as fallback)
                let mut new_ops: Vec<PyPipelineOp> =
                    ops.iter().map(|op| op.clone_ref(py)).collect();
                new_ops.push(PyPipelineOp::Filter(pred.unbind()));
                Ok(PyQuery::from_inner(QueryInner::Obj {
                    source: source.clone_ref(py),
                    ops: new_ops,
                }))
            }
            QueryInner::ObjFieldPy {
                source,
                ops,
                map_field,
            } => {
                // Accumulate additional field filter ops (or materialize for lambda/map_field set)
                if self.skip == 0 && self.take.is_none() && map_field.is_none() {
                    if let Ok(fexpr) = pred.extract::<PyRef<PyFieldExpr>>() {
                        if let Some(fop) = &fexpr.op {
                            if objop_is_field_filter(fop) {
                                let mut new_ops = ops.clone();
                                new_ops.push(fop.clone());
                                return Ok(PyQuery::from_inner(QueryInner::ObjFieldPy {
                                    source: source.clone_ref(py),
                                    ops: new_ops,
                                    map_field: None,
                                }));
                            }
                        }
                    }
                }
                // Lambda or map_field already set: materialize then fall back
                let items = filter_by_field_py(
                    py,
                    source,
                    ops,
                    self.skip,
                    self.take,
                    map_field.as_deref(),
                )?;
                let mat = PyList::new_bound(py, items.iter().map(|o| o.bind(py)));
                let mut new_ops: Vec<PyPipelineOp> = Vec::new();
                new_ops.push(PyPipelineOp::Filter(pred.unbind()));
                Ok(PyQuery::from_inner(QueryInner::Obj {
                    source: mat.unbind().into_py(py),
                    ops: new_ops,
                }))
            }
            QueryInner::ObjField {
                source,
                field_name,
                ops,
            } => {
                // Same field numeric op → accumulate (no materialization needed)
                if self.skip == 0 && self.take.is_none() {
                    if let Ok(fexpr) = pred.extract::<PyRef<PyFieldExpr>>() {
                        if let Some(fop) = &fexpr.op {
                            if let Some((fname, nop)) = objop_to_numeric(fop) {
                                if fname.as_ref() == field_name.as_ref() {
                                    let mut new_ops = ops.clone();
                                    new_ops.push(nop);
                                    return Ok(PyQuery::from_inner(QueryInner::ObjField {
                                        source: source.clone_ref(py),
                                        field_name: Arc::clone(field_name),
                                        ops: new_ops,
                                    }));
                                }
                            }
                        }
                    }
                }
                // Different field, non-numeric, skip/take set, or lambda: materialize first
                let items = filter_by_field(py, source, field_name, ops, self.skip, self.take)?;
                let mat = PyList::new_bound(py, &items);
                let mut new_ops: Vec<PyPipelineOp> = Vec::new();
                new_ops.push(PyPipelineOp::Filter(pred.unbind()));
                Ok(PyQuery::from_inner(QueryInner::Obj {
                    source: mat.unbind().into_py(py),
                    ops: new_ops,
                }))
            }
            QueryInner::LazyFloatList { source, ops } => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if let Some(op) = expr.op.to_f64_op() {
                        if self.skip == 0 && self.take.is_none() {
                            let mut new_ops = ops.clone();
                            new_ops.push(op);
                            return Ok(PyQuery::from_inner(QueryInner::LazyFloatList {
                                source: source.clone_ref(py),
                                ops: new_ops,
                            }));
                        }
                        let data =
                            execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                        return Ok(PyQuery::from_inner(QueryInner::F64(
                            NumericPipeline::new(data).push_op(op),
                        )));
                    }
                }
                if self.skip == 0 && self.take.is_none() {
                    if let Some(dsl) = try_lambda_to_dsl_expr(py, &pred) {
                        if let Ok(expr) = dsl.extract::<PyRef<PyExpr>>() {
                            if let Some(op) = expr.op.to_f64_op() {
                                let mut new_ops = ops.clone();
                                new_ops.push(op);
                                return Ok(PyQuery::from_inner(QueryInner::LazyFloatList {
                                    source: source.clone_ref(py),
                                    ops: new_ops,
                                }));
                            }
                        }
                    }
                }
                let v = execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                let filtered = apply_py_filter_f64(py, v, &pred)?;
                Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(
                    filtered,
                ))))
            }
            QueryInner::NumpyF64 { source, ops } => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if let Some(op) = expr.op.to_f64_op() {
                        if self.skip == 0 && self.take.is_none() {
                            let mut new_ops = ops.clone();
                            push_numeric_op(&mut new_ops, op);
                            return Ok(PyQuery::from_inner(QueryInner::NumpyF64 {
                                source: source.clone_ref(py),
                                ops: new_ops,
                            }));
                        }
                        let data = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                        return Ok(PyQuery::from_inner(QueryInner::F64(
                            NumericPipeline::new(data).push_op(op),
                        )));
                    }
                }
                if self.skip == 0 && self.take.is_none() {
                    if let Some(dsl) = try_lambda_to_dsl_expr(py, &pred) {
                        if let Ok(expr) = dsl.extract::<PyRef<PyExpr>>() {
                            if let Some(op) = expr.op.to_f64_op() {
                                let mut new_ops = ops.clone();
                                push_numeric_op(&mut new_ops, op);
                                return Ok(PyQuery::from_inner(QueryInner::NumpyF64 {
                                    source: source.clone_ref(py),
                                    ops: new_ops,
                                }));
                            }
                        }
                    }
                }
                let v = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                let filtered = apply_py_filter_f64(py, v, &pred)?;
                Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(
                    filtered,
                ))))
            }
            QueryInner::NumpyF32 { source, ops } => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if let Some(op) = expr.op.to_f64_op() {
                        if self.skip == 0 && self.take.is_none() {
                            let mut new_ops = ops.clone();
                            push_numeric_op(&mut new_ops, op);
                            return Ok(PyQuery::from_inner(QueryInner::NumpyF32 {
                                source: source.clone_ref(py),
                                ops: new_ops,
                            }));
                        }
                        let data = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                        let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                        return Ok(PyQuery::from_inner(QueryInner::F64(
                            NumericPipeline::new(data_f64).push_op(op),
                        )));
                    }
                }
                // Python callback: materialize f32 → f64, then apply
                let data = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                let filtered = apply_py_filter_f64(py, data_f64, &pred)?;
                Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(filtered))))
            }
            QueryInner::Empty => Ok(PyQuery::from_inner(QueryInner::Empty)),
        }
    }

    // ------------------------------------------------------------------
    // map(expr_or_callable)
    // ------------------------------------------------------------------

    fn map(&self, py: Python<'_>, f: Bound<'_, PyAny>) -> PyResult<PyQuery> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                if let Ok(expr) = f.extract::<PyRef<PyExpr>>() {
                    if let Some(op) = expr.op.to_f64_op() {
                        if !expr.op.is_filter() {
                            if self.skip == 0 && self.take.is_none() {
                                let new_inner = branch_f64_pipeline(pipeline).push_op(op);
                                return Ok(PyQuery::from_inner(QueryInner::F64(new_inner)));
                            }
                            // Pending skip/take: materialize first to preserve pipeline-order semantics.
                            let data = collect_f64_pipeline(py, pipeline, self.skip, self.take)?;
                            return Ok(PyQuery::from_inner(QueryInner::F64(
                                NumericPipeline::new(data).push_op(op),
                            )));
                        }
                    }
                }
                // Lambda AST fast path: safe only when no pending skip/take.
                if self.skip == 0 && self.take.is_none() {
                    if let Some(dsl) = try_lambda_to_dsl_expr(py, &f) {
                        if let Ok(expr) = dsl.extract::<PyRef<PyExpr>>() {
                            if let Some(op) = expr.op.to_f64_op() {
                                if !expr.op.is_filter() {
                                    let new_inner = branch_f64_pipeline(pipeline).push_op(op);
                                    return Ok(PyQuery::from_inner(QueryInner::F64(new_inner)));
                                }
                            }
                        }
                    }
                }
                // Python callback: must hold GIL, but we still avoid intermediate Vec
                let data = collect_f64_pipeline(py, pipeline, self.skip, self.take)?;
                let result = apply_py_map_f64(py, data, &f)?;
                match result {
                    MappedResult::F64(v) => Ok(PyQuery::from_inner(QueryInner::F64(
                        NumericPipeline::new(v),
                    ))),
                    MappedResult::Py(v) => Ok(PyQuery::from_inner(QueryInner::Py(v))),
                }
            }
            QueryInner::I64(pipeline) => {
                if let Ok(expr) = f.extract::<PyRef<PyExpr>>() {
                    if let Some(op) = expr_to_int_op(&expr.op) {
                        if self.skip == 0 && self.take.is_none() {
                            let new_inner = branch_i64_pipeline(pipeline).push_op(op);
                            return Ok(PyQuery::from_inner(QueryInner::I64(new_inner)));
                        }
                        // Pending skip/take: materialize first.
                        let data = collect_i64_pipeline(py, pipeline, self.skip, self.take)?;
                        return Ok(PyQuery::from_inner(QueryInner::I64(
                            IntPipeline::new(data).push_op(op),
                        )));
                    }
                }
                let data = collect_i64_pipeline(py, pipeline, self.skip, self.take)?;
                let result = apply_py_map_i64(py, data, &f)?;
                match result {
                    MappedResultI::I64(v) => {
                        Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(v))))
                    }
                    MappedResultI::F64(v) => Ok(PyQuery::from_inner(QueryInner::F64(
                        NumericPipeline::new(v),
                    ))),
                    MappedResultI::Py(v) => Ok(PyQuery::from_inner(QueryInner::Py(v))),
                }
            }
            QueryInner::U8 { data, ops } => {
                let current = execute_u8(py, data, ops, self.skip, self.take);
                if let Ok(expr) = f.extract::<PyRef<PyExpr>>() {
                    if let Some(op) = expr_to_int_op(&expr.op) {
                        return Ok(PyQuery::from_inner(QueryInner::I64(
                            IntPipeline::new(current).push_op(op),
                        )));
                    }
                }
                let result = apply_py_map_i64(py, current, &f)?;
                match result {
                    MappedResultI::I64(v) => {
                        Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(v))))
                    }
                    MappedResultI::F64(v) => Ok(PyQuery::from_inner(QueryInner::F64(
                        NumericPipeline::new(v),
                    ))),
                    MappedResultI::Py(v) => Ok(PyQuery::from_inner(QueryInner::Py(v))),
                }
            }
            QueryInner::Py(items) => {
                let mapped = apply_py_map(py, items, &f, self.skip, self.take)?;
                Ok(PyQuery::from_inner(QueryInner::Py(mapped)))
            }
            QueryInner::Obj { source, ops } => {
                let mut new_ops: Vec<PyPipelineOp> =
                    ops.iter().map(|op| op.clone_ref(py)).collect();
                new_ops.push(PyPipelineOp::Map(f.unbind()));
                Ok(PyQuery::from_inner(QueryInner::Obj {
                    source: source.clone_ref(py),
                    ops: new_ops,
                }))
            }
            QueryInner::LazyFloatList { source, ops } => {
                if let Ok(expr) = f.extract::<PyRef<PyExpr>>() {
                    if let Some(op) = expr.op.to_f64_op() {
                        if !expr.op.is_filter() {
                            if self.skip == 0 && self.take.is_none() {
                                let mut new_ops = ops.clone();
                                new_ops.push(op);
                                return Ok(PyQuery::from_inner(QueryInner::LazyFloatList {
                                    source: source.clone_ref(py),
                                    ops: new_ops,
                                }));
                            }
                            // Pending skip/take: materialize first.
                            let data = execute_lazy_float_list(
                                py,
                                source.as_ptr(),
                                ops,
                                self.skip,
                                self.take,
                            );
                            return Ok(PyQuery::from_inner(QueryInner::F64(
                                NumericPipeline::new(data).push_op(op),
                            )));
                        }
                    }
                }
                if self.skip == 0 && self.take.is_none() {
                    if let Some(dsl) = try_lambda_to_dsl_expr(py, &f) {
                        if let Ok(expr) = dsl.extract::<PyRef<PyExpr>>() {
                            if let Some(op) = expr.op.to_f64_op() {
                                if !expr.op.is_filter() {
                                    let mut new_ops = ops.clone();
                                    new_ops.push(op);
                                    return Ok(PyQuery::from_inner(QueryInner::LazyFloatList {
                                        source: source.clone_ref(py),
                                        ops: new_ops,
                                    }));
                                }
                            }
                        }
                    }
                }
                let v = execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                let result = apply_py_map_f64(py, v, &f)?;
                match result {
                    MappedResult::F64(v) => Ok(PyQuery::from_inner(QueryInner::F64(
                        NumericPipeline::new(v),
                    ))),
                    MappedResult::Py(v) => Ok(PyQuery::from_inner(QueryInner::Py(v))),
                }
            }
            QueryInner::NumpyF64 { source, ops } => {
                if let Ok(expr) = f.extract::<PyRef<PyExpr>>() {
                    if let Some(op) = expr.op.to_f64_op() {
                        if !expr.op.is_filter() {
                            if self.skip == 0 && self.take.is_none() {
                                let mut new_ops = ops.clone();
                                push_numeric_op(&mut new_ops, op);
                                return Ok(PyQuery::from_inner(QueryInner::NumpyF64 {
                                    source: source.clone_ref(py),
                                    ops: new_ops,
                                }));
                            }
                            // Pending skip/take: materialize first.
                            let data = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                            return Ok(PyQuery::from_inner(QueryInner::F64(
                                NumericPipeline::new(data).push_op(op),
                            )));
                        }
                    }
                }
                if self.skip == 0 && self.take.is_none() {
                    if let Some(dsl) = try_lambda_to_dsl_expr(py, &f) {
                        if let Ok(expr) = dsl.extract::<PyRef<PyExpr>>() {
                            if let Some(op) = expr.op.to_f64_op() {
                                if !expr.op.is_filter() {
                                    let mut new_ops = ops.clone();
                                    push_numeric_op(&mut new_ops, op);
                                    return Ok(PyQuery::from_inner(QueryInner::NumpyF64 {
                                        source: source.clone_ref(py),
                                        ops: new_ops,
                                    }));
                                }
                            }
                        }
                    }
                }
                let v = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                let result = apply_py_map_f64(py, v, &f)?;
                match result {
                    MappedResult::F64(v) => Ok(PyQuery::from_inner(QueryInner::F64(
                        NumericPipeline::new(v),
                    ))),
                    MappedResult::Py(v) => Ok(PyQuery::from_inner(QueryInner::Py(v))),
                }
            }
            QueryInner::NumpyF32 { source, ops } => {
                if let Ok(expr) = f.extract::<PyRef<PyExpr>>() {
                    if let Some(op) = expr.op.to_f64_op() {
                        if !expr.op.is_filter() {
                            if self.skip == 0 && self.take.is_none() {
                                let mut new_ops = ops.clone();
                                push_numeric_op(&mut new_ops, op);
                                return Ok(PyQuery::from_inner(QueryInner::NumpyF32 {
                                    source: source.clone_ref(py),
                                    ops: new_ops,
                                }));
                            }
                            let data = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                            let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                            return Ok(PyQuery::from_inner(QueryInner::F64(
                                NumericPipeline::new(data_f64).push_op(op),
                            )));
                        }
                    }
                }
                // Python callback: promote to f64 first
                let v = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                let v_f64: Vec<f64> = v.iter().map(|&x| x as f64).collect();
                let result = apply_py_map_f64(py, v_f64, &f)?;
                match result {
                    MappedResult::F64(v) => Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(v)))),
                    MappedResult::Py(v) => Ok(PyQuery::from_inner(QueryInner::Py(v))),
                }
            }
            QueryInner::RustObj { data, ops } => {
                // Materialize GIL-free, then apply the lambda under GIL as an Obj pipeline.
                let data = Arc::clone(data);
                let ops_c = ops.clone();
                let (skip, take) = (self.skip, self.take);
                let rows =
                    py.allow_threads(move || execute_obj_pipeline(&data, &ops_c, skip, take));
                let mat_list = PyList::empty_bound(py);
                for row in &rows {
                    mat_list.append(rust_row_to_py(py, row))?;
                }
                Ok(PyQuery::from_inner(QueryInner::Obj {
                    source: mat_list.unbind().into_py(py),
                    ops: vec![PyPipelineOp::Map(f.unbind())],
                }))
            }
            QueryInner::ObjField {
                source,
                field_name,
                ops,
            } => {
                // Materialize filtered items, then apply map as Obj pipeline.
                let items = filter_by_field(py, source, field_name, ops, self.skip, self.take)?;
                let mat = PyList::new_bound(py, &items);
                Ok(PyQuery::from_inner(QueryInner::Obj {
                    source: mat.unbind().into_py(py),
                    ops: vec![PyPipelineOp::Map(f.unbind())],
                }))
            }
            QueryInner::ObjFieldPy {
                source,
                ops,
                map_field,
            } => {
                // Materialize filtered items, then apply map as Obj pipeline.
                let items = filter_by_field_py(
                    py,
                    source,
                    ops,
                    self.skip,
                    self.take,
                    map_field.as_deref(),
                )?;
                let mat = PyList::new_bound(py, items.iter().map(|o| o.bind(py)));
                Ok(PyQuery::from_inner(QueryInner::Obj {
                    source: mat.unbind().into_py(py),
                    ops: vec![PyPipelineOp::Map(f.unbind())],
                }))
            }
            QueryInner::Empty => Ok(PyQuery::from_inner(QueryInner::Empty)),
        }
    }

    // ------------------------------------------------------------------
    // take / skip
    // ------------------------------------------------------------------

    fn take(&self, py: Python<'_>, n: usize) -> PyQuery {
        let inner = clone_inner(py, &self.inner);
        PyQuery {
            inner,
            take: Some(match self.take {
                Some(e) => e.min(n),
                None => n,
            }),
            skip: self.skip,
            parallel: self.parallel,
        }
    }

    fn skip(&self, py: Python<'_>, n: usize) -> PyQuery {
        let inner = clone_inner(py, &self.inner);
        PyQuery {
            inner,
            take: self.take,
            skip: self.skip + n,
            parallel: self.parallel,
        }
    }

    // ------------------------------------------------------------------
    // parallel execution hint
    // ------------------------------------------------------------------

    fn parallel(&self, py: Python<'_>) -> PyQuery {
        let inner = clone_inner(py, &self.inner);
        PyQuery {
            inner,
            take: self.take,
            skip: self.skip,
            parallel: true,
        }
    }

    // ------------------------------------------------------------------
    // Terminal operations
    // ------------------------------------------------------------------

    fn to_list(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                let result = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                let list = PyList::new_bound(py, &result);
                Ok(list.into())
            }
            QueryInner::I64(pipeline) => {
                let result = execute_i64(py, pipeline, self.skip, self.take, self.parallel);
                let list = PyList::new_bound(py, &result);
                Ok(list.into())
            }
            QueryInner::U8 { data, ops } => {
                let result = execute_u8(py, data, ops, self.skip, self.take);
                let list = PyList::new_bound(py, &result);
                Ok(list.into())
            }
            QueryInner::Py(items) => {
                let sliced = apply_skip_take(items, self.skip, self.take);
                let list = PyList::new_bound(py, sliced);
                Ok(list.into())
            }
            QueryInner::Obj { source, ops } => {
                let items = collect_py_lazy(py, source, ops, self.skip, self.take)?;
                Ok(PyList::new_bound(py, &items).into())
            }
            QueryInner::RustObj { data, ops } => {
                let data = Arc::clone(data);
                let ops = ops.clone();
                let (skip, take) = (self.skip, self.take);
                // Execute GIL-free, then convert results under GIL
                let rows = py.allow_threads(move || execute_obj_pipeline(&data, &ops, skip, take));
                let list = PyList::empty_bound(py);
                for row in &rows {
                    list.append(rust_row_to_py(py, row))?;
                }
                Ok(list.into())
            }
            QueryInner::LazyFloatList { source, ops } => {
                let result =
                    execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                Ok(PyList::new_bound(py, &result).into())
            }
            QueryInner::NumpyF64 { source, ops } => {
                let result = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                Ok(PyList::new_bound(py, &result).into())
            }
            QueryInner::NumpyF32 { source, ops } => {
                let result = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                // Promote to f64 for Python (Python floats are always f64)
                let result_f64: Vec<f64> = result.iter().map(|&x| x as f64).collect();
                Ok(PyList::new_bound(py, &result_f64).into())
            }
            QueryInner::ObjField {
                source,
                field_name,
                ops,
            } => {
                let items = filter_by_field(py, source, field_name, ops, self.skip, self.take)?;
                Ok(PyList::new_bound(py, &items).into())
            }
            QueryInner::ObjFieldPy {
                source,
                ops,
                map_field,
            } => {
                let items = filter_by_field_py(
                    py,
                    source,
                    ops,
                    self.skip,
                    self.take,
                    map_field.as_deref(),
                )?;
                Ok(PyList::new_bound(py, items.iter().map(|o| o.bind(py))).into())
            }
            QueryInner::Empty => Ok(PyList::empty_bound(py).into()),
        }
    }

    /// Return the pipeline result as raw `bytes` (little-endian f64 values).
    /// Avoids boxing individual floats — only one bulk memcpy.
    /// Use with numpy: `np.frombuffer(q.to_bytes(), dtype=np.float64)`
    /// Only supported for f64 pipelines; raises ValueError otherwise.
    fn to_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                let result = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                let raw: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        result.as_ptr() as *const u8,
                        result.len() * std::mem::size_of::<f64>(),
                    )
                };
                Ok(PyBytes::new_bound(py, raw).into())
            }
            QueryInner::LazyFloatList { source, ops } => {
                let result =
                    execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                let raw: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        result.as_ptr() as *const u8,
                        result.len() * std::mem::size_of::<f64>(),
                    )
                };
                Ok(PyBytes::new_bound(py, raw).into())
            }
            QueryInner::NumpyF64 { source, ops } => {
                let result = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                let raw: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        result.as_ptr() as *const u8,
                        result.len() * std::mem::size_of::<f64>(),
                    )
                };
                Ok(PyBytes::new_bound(py, raw).into())
            }
            _ => Err(PyValueError::new_err(
                "to_bytes() requires an f64 pipeline; use to_list() for other types",
            )),
        }
    }

    /// Return the pipeline result as a numpy ndarray.
    ///
    /// Transfers the Rust Vec's buffer directly to numpy (zero copy for f64/i64/u8).
    /// No per-element Python float boxing — equivalent to `to_bytes()` + `frombuffer`
    /// but returns a writable, owned numpy array in a single step.
    ///
    /// Dtype mapping:
    ///   f64 pipeline → np.float64
    ///   i64 pipeline → np.int64
    ///   u8 pipeline  → np.uint8
    ///
    /// Raises ValueError for object/Py pipelines — use `to_list()` + `np.array()`.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                let result = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                Ok(result.into_pyarray_bound(py).into_any().unbind())
            }
            QueryInner::I64(pipeline) => {
                let result = execute_i64(py, pipeline, self.skip, self.take, self.parallel);
                Ok(result.into_pyarray_bound(py).into_any().unbind())
            }
            QueryInner::U8 { data, ops } => {
                let data = Arc::clone(data);
                let ops = ops.clone();
                let skip = self.skip;
                let take = self.take;
                let result = py.allow_threads(move || execute_u8_bounded(&data, &ops, skip, take));
                Ok(result.into_pyarray_bound(py).into_any().unbind())
            }
            QueryInner::LazyFloatList { source, ops } => {
                let result =
                    execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                Ok(result.into_pyarray_bound(py).into_any().unbind())
            }
            QueryInner::NumpyF64 { source, ops } => {
                let result = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                Ok(result.into_pyarray_bound(py).into_any().unbind())
            }
            QueryInner::NumpyF32 { source, ops } => {
                let result = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                Ok(result.into_pyarray_bound(py).into_any().unbind())
            }
            QueryInner::Empty => {
                let empty: Vec<f64> = vec![];
                Ok(empty.into_pyarray_bound(py).into_any().unbind())
            }
            _ => Err(PyValueError::new_err(
                "to_numpy() is only supported for numeric pipelines (f64/i64/u8). \
                 Use to_list() for object pipelines, or preload() to materialise first.",
            )),
        }
    }

    /// Group elements by key_fn in a single pass, applying pending filter/map ops.
    /// Returns a Python dict {key: [items]} for use by the Python GroupBy wrapper.
    /// This avoids the double-pass overhead of to_list() + Python-side GroupBy.
    /// Expose the raw components of an object-path pipeline for pure-Python iteration.
    ///
    /// Returns `None` for numeric/materialized paths (caller should use `to_list()`).
    /// For `QueryInner::Obj`, returns `[source, ops_list, skip, take_or_None]` where
    /// `ops_list` is `[[is_filter: bool, callable], ...]` preserving original order.
    ///
    /// Python's `__iter__` uses this to run the hot loop entirely in CPython's eval
    /// loop, avoiding the PyO3 boundary crossing per element that makes lambda
    /// operations slow.
    fn _iter_parts(&self, py: Python<'_>) -> PyResult<PyObject> {
        let QueryInner::Obj { source, ops } = &self.inner else {
            return Ok(py.None());
        };

        let ops_list = PyList::empty_bound(py);
        for op in ops {
            let (is_filter, f) = match op {
                PyPipelineOp::Filter(f) => (true, f),
                PyPipelineOp::Map(f) => (false, f),
            };
            let pair = PyList::new_bound(py, [is_filter.into_py(py), f.clone_ref(py)]);
            ops_list.append(&pair)?;
        }
        let take_obj: PyObject = match self.take {
            Some(n) => n.into_py(py),
            None => py.None(),
        };
        Ok(PyList::new_bound(
            py,
            [
                source.clone_ref(py),
                ops_list.into_py(py),
                self.skip.into_py(py),
                take_obj,
            ],
        )
        .into())
    }

    // ------------------------------------------------------------------
    // ZStream combinators — lazy operators exposed to Python
    // ------------------------------------------------------------------

    /// Stop emitting elements as soon as `pred` returns False.
    /// DSL fast path for F64/I64/RustObj; lambda fallback materializes first.
    fn take_while(&self, py: Python<'_>, pred: Bound<'_, PyAny>) -> PyResult<PyQuery> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if expr.op.is_filter() {
                        if let Some(op) = expr.op.to_f64_op() {
                            let data = execute_f64(py, pipeline, self.skip, self.take, false);
                            let result = py.allow_threads(move || {
                                let n = data
                                    .iter()
                                    .position(|&x| !eval_filter_f64(x, &op))
                                    .unwrap_or(data.len());
                                data[..n].to_vec()
                            });
                            return Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(
                                result,
                            ))));
                        }
                    }
                }
                // Lambda fallback
                let data = execute_f64(py, pipeline, self.skip, self.take, false);
                let mut out = Vec::new();
                for val in data {
                    if !pred.call1((val.into_py(py),))?.is_truthy()? {
                        break;
                    }
                    out.push(val);
                }
                Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(
                    out,
                ))))
            }
            QueryInner::I64(pipeline) => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if expr.op.is_filter() {
                        if let Some(_op) = expr.op.to_f64_op() {
                            let int_op = expr_to_int_op(&expr.op);
                            if let Some(iop) = int_op {
                                let data = execute_i64(py, pipeline, self.skip, self.take, false);
                                let result = py.allow_threads(move || {
                                    let n = data
                                        .iter()
                                        .position(|&x| !eval_filter_i64(x, &iop))
                                        .unwrap_or(data.len());
                                    data[..n].to_vec()
                                });
                                return Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(
                                    result,
                                ))));
                            }
                        }
                    }
                }
                let data = execute_i64(py, pipeline, self.skip, self.take, false);
                let mut out = Vec::new();
                for val in data {
                    if !pred.call1((val.into_py(py),))?.is_truthy()? {
                        break;
                    }
                    out.push(val);
                }
                Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(out))))
            }
            QueryInner::RustObj { data, ops } => {
                if let Ok(fexpr) = pred.extract::<PyRef<PyFieldExpr>>() {
                    if let Some(fop) = fexpr.op.clone() {
                        let data = Arc::clone(data);
                        let ops_c = ops.clone();
                        let (skip, take) = (self.skip, self.take);
                        let result = py.allow_threads(move || {
                            let rows = execute_obj_pipeline(&data, &ops_c, skip, take);
                            let n = rows
                                .iter()
                                .position(|r| !row_passes(r, &fop))
                                .unwrap_or(rows.len());
                            rows[..n].to_vec()
                        });
                        return Ok(PyQuery::from_inner(QueryInner::RustObj {
                            data: Arc::new(result),
                            ops: Vec::new(),
                        }));
                    }
                }
                // Lambda fallback
                let data = Arc::clone(data);
                let ops_c = ops.clone();
                let (skip, take) = (self.skip, self.take);
                let rows =
                    py.allow_threads(move || execute_obj_pipeline(&data, &ops_c, skip, take));
                let mat = PyList::empty_bound(py);
                for row in &rows {
                    let py_row = rust_row_to_py(py, row);
                    if !pred.call1((py_row.clone_ref(py),))?.is_truthy()? {
                        break;
                    }
                    mat.append(py_row)?;
                }
                Ok(PyQuery::from_inner(QueryInner::Obj {
                    source: mat.unbind().into_py(py),
                    ops: Vec::new(),
                }))
            }
            _ => {
                let list = self.to_list(py)?;
                let mut out: Vec<PyObject> = Vec::new();
                for item in list.bind(py).iter() {
                    if !pred.call1((item.clone(),))?.is_truthy()? {
                        break;
                    }
                    out.push(item.unbind());
                }
                Ok(PyQuery::from_inner(QueryInner::Py(out)))
            }
        }
    }

    /// Skip elements as long as `pred` returns True; emit all remaining elements.
    fn skip_while(&self, py: Python<'_>, pred: Bound<'_, PyAny>) -> PyResult<PyQuery> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if expr.op.is_filter() {
                        if let Some(op) = expr.op.to_f64_op() {
                            let data = execute_f64(py, pipeline, self.skip, self.take, false);
                            let result = py.allow_threads(move || {
                                let start = data
                                    .iter()
                                    .position(|&x| !eval_filter_f64(x, &op))
                                    .unwrap_or(data.len());
                                data[start..].to_vec()
                            });
                            return Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(
                                result,
                            ))));
                        }
                    }
                }
                let data = execute_f64(py, pipeline, self.skip, self.take, false);
                let mut skipping = true;
                let mut out = Vec::new();
                for val in data {
                    if skipping && pred.call1((val.into_py(py),))?.is_truthy()? {
                        continue;
                    }
                    skipping = false;
                    out.push(val);
                }
                Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(
                    out,
                ))))
            }
            QueryInner::I64(pipeline) => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if expr.op.is_filter() {
                        if let Some(iop) = expr_to_int_op(&expr.op) {
                            let data = execute_i64(py, pipeline, self.skip, self.take, false);
                            let result = py.allow_threads(move || {
                                let start = data
                                    .iter()
                                    .position(|&x| !eval_filter_i64(x, &iop))
                                    .unwrap_or(data.len());
                                data[start..].to_vec()
                            });
                            return Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(
                                result,
                            ))));
                        }
                    }
                }
                let data = execute_i64(py, pipeline, self.skip, self.take, false);
                let mut skipping = true;
                let mut out = Vec::new();
                for val in data {
                    if skipping && pred.call1((val.into_py(py),))?.is_truthy()? {
                        continue;
                    }
                    skipping = false;
                    out.push(val);
                }
                Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(out))))
            }
            QueryInner::RustObj { data, ops } => {
                if let Ok(fexpr) = pred.extract::<PyRef<PyFieldExpr>>() {
                    if let Some(fop) = fexpr.op.clone() {
                        let data = Arc::clone(data);
                        let ops_c = ops.clone();
                        let (skip, take) = (self.skip, self.take);
                        let result = py.allow_threads(move || {
                            let rows = execute_obj_pipeline(&data, &ops_c, skip, take);
                            let start = rows
                                .iter()
                                .position(|r| !row_passes(r, &fop))
                                .unwrap_or(rows.len());
                            rows[start..].to_vec()
                        });
                        return Ok(PyQuery::from_inner(QueryInner::RustObj {
                            data: Arc::new(result),
                            ops: Vec::new(),
                        }));
                    }
                }
                // Lambda fallback
                let data = Arc::clone(data);
                let ops_c = ops.clone();
                let (skip, take) = (self.skip, self.take);
                let rows =
                    py.allow_threads(move || execute_obj_pipeline(&data, &ops_c, skip, take));
                let mut skipping = true;
                let mat = PyList::empty_bound(py);
                for row in &rows {
                    let py_row = rust_row_to_py(py, row);
                    if skipping && pred.call1((py_row.clone_ref(py),))?.is_truthy()? {
                        continue;
                    }
                    skipping = false;
                    mat.append(py_row)?;
                }
                Ok(PyQuery::from_inner(QueryInner::Obj {
                    source: mat.unbind().into_py(py),
                    ops: Vec::new(),
                }))
            }
            _ => {
                let list = self.to_list(py)?;
                let mut skipping = true;
                let mut out: Vec<PyObject> = Vec::new();
                for item in list.bind(py).iter() {
                    if skipping && pred.call1((item.clone(),))?.is_truthy()? {
                        continue;
                    }
                    skipping = false;
                    out.push(item.unbind());
                }
                Ok(PyQuery::from_inner(QueryInner::Py(out)))
            }
        }
    }

    /// Concatenate this query with `other`. F64+F64 and I64+I64 stay typed; others become Py.
    fn chain(&self, py: Python<'_>, other: PyRef<'_, PyQuery>) -> PyResult<PyQuery> {
        match (&self.inner, &other.inner) {
            (QueryInner::F64(a), QueryInner::F64(b)) => {
                let a_data = a.arc();
                let a_ops = a.clone_ops();
                let b_data = b.arc();
                let b_ops = b.clone_ops();
                let (a_skip, a_take) = (self.skip, self.take);
                let (b_skip, b_take) = (other.skip, other.take);
                let result = py.allow_threads(move || {
                    let mut out = execute_fused_f64_bounded(&a_data, &a_ops, a_skip, a_take);
                    out.extend_from_slice(&execute_fused_f64_bounded(
                        &b_data, &b_ops, b_skip, b_take,
                    ));
                    out
                });
                Ok(PyQuery::from_inner(QueryInner::F64(NumericPipeline::new(
                    result,
                ))))
            }
            (QueryInner::I64(a), QueryInner::I64(b)) => {
                let a_data = a.arc();
                let a_ops = a.clone_ops();
                let b_data = b.arc();
                let b_ops = b.clone_ops();
                let (a_skip, a_take) = (self.skip, self.take);
                let (b_skip, b_take) = (other.skip, other.take);
                let result = py.allow_threads(move || {
                    let mut out = execute_fused_i64_bounded(&a_data, &a_ops, a_skip, a_take);
                    out.extend_from_slice(&execute_fused_i64_bounded(
                        &b_data, &b_ops, b_skip, b_take,
                    ));
                    out
                });
                Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(
                    result,
                ))))
            }
            _ => {
                let mut a_list = self
                    .to_list(py)?
                    .bind(py)
                    .iter()
                    .map(|x| x.unbind())
                    .collect::<Vec<PyObject>>();
                let b_list = other.to_list(py)?;
                a_list.extend(b_list.bind(py).iter().map(|x| x.unbind()));
                Ok(PyQuery::from_inner(QueryInner::Py(a_list)))
            }
        }
    }

    /// Yield `(index, item)` tuples (zero-based). Always returns a Py pipeline.
    fn enumerate(&self, py: Python<'_>) -> PyResult<PyQuery> {
        use pyo3::types::PyTuple;
        let list = self.to_list(py)?;
        let mut out: Vec<PyObject> = Vec::with_capacity(list.bind(py).len());
        for (i, item) in list.bind(py).iter().enumerate() {
            let t = PyTuple::new_bound(py, [i.into_py(py), item.unbind()]);
            out.push(t.into());
        }
        Ok(PyQuery::from_inner(QueryInner::Py(out)))
    }

    /// Yield `(a, b)` pairs from self and `other`, stopping at the shorter sequence.
    fn zip(&self, py: Python<'_>, other: PyRef<'_, PyQuery>) -> PyResult<PyQuery> {
        use pyo3::types::PyTuple;
        let a = self.to_list(py)?;
        let b = other.to_list(py)?;
        let n = a.bind(py).len().min(b.bind(py).len());
        let mut out: Vec<PyObject> = Vec::with_capacity(n);
        for (x, y) in a.bind(py).iter().zip(b.bind(py).iter()) {
            let t = PyTuple::new_bound(py, [x.unbind(), y.unbind()]);
            out.push(t.into());
        }
        Ok(PyQuery::from_inner(QueryInner::Py(out)))
    }

    /// Apply `f` to each element and flatten the resulting iterables.
    fn flat_map(&self, py: Python<'_>, f: Bound<'_, PyAny>) -> PyResult<PyQuery> {
        let list = self.to_list(py)?;
        let mut out: Vec<PyObject> = Vec::new();
        for item in list.bind(py).iter() {
            let sub = f.call1((item,))?;
            for elem in sub.iter()? {
                out.push(elem?.unbind());
            }
        }
        Ok(PyQuery::from_inner(QueryInner::Py(out)))
    }

    /// Convert a dict-list Obj pipeline to RustObj eagerly.
    /// Use this to pay the import cost once when the same dataset will be queried many times:
    ///   q = Query(logs).preload()    # converts once
    ///   q.filter(field("x") > 5).count()  # GIL-free, runs in microseconds
    fn preload(&self, py: Python<'_>) -> PyResult<PyQuery> {
        if let QueryInner::Obj { source, ops } = &self.inner {
            if ops.is_empty() {
                if let Ok(list) = source.bind(py).downcast::<PyList>() {
                    if let Some(rows) = try_convert_to_rust_obj(py, list)? {
                        return Ok(PyQuery {
                            inner: QueryInner::RustObj {
                                data: Arc::new(rows),
                                ops: Vec::new(),
                            },
                            skip: self.skip,
                            take: self.take,
                            parallel: self.parallel,
                        });
                    }
                }
            }
        }
        Ok(PyQuery {
            inner: clone_inner(py, &self.inner),
            skip: self.skip,
            take: self.take,
            parallel: self.parallel,
        })
    }

    fn group_by_collect(&self, py: Python<'_>, key_fn: Bound<'_, PyAny>) -> PyResult<PyObject> {
        match &self.inner {
            QueryInner::Obj { source, ops } => {
                group_by_collect(py, source, ops, &key_fn, self.skip, self.take)
            }
            _ => {
                // For numeric/materialized paths, fall back to to_list() + group.
                let dict = PyDict::new_bound(py);
                let list = self.to_list(py)?;
                for item in list.bind(py).iter() {
                    let key = key_fn.call1((item.clone(),))?;
                    match dict.get_item(&key)? {
                        Some(lst) => lst.downcast::<PyList>()?.append(item)?,
                        None => dict.set_item(&key, PyList::new_bound(py, [item]))?,
                    }
                }
                Ok(dict.into())
            }
        }
    }

    fn to_dict(
        &self,
        py: Python<'_>,
        key: Bound<'_, PyAny>,
        value: Bound<'_, PyAny>,
    ) -> PyResult<Py<PyDict>> {
        let list = self.to_list(py)?;
        let dict = PyDict::new_bound(py);
        for item in list.bind(py).iter() {
            let k = key.call1((item.clone(),))?;
            let v = value.call1((item,))?;
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }

    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                let data = pipeline.arc();
                let ops = pipeline.clone_ops();
                let skip = self.skip;
                let take = self.take;
                Ok(py.allow_threads(move || count_fused_f64_bounded(&data, &ops, skip, take)))
            }
            QueryInner::I64(pipeline) => {
                let data = pipeline.arc();
                let ops = pipeline.clone_ops();
                let skip = self.skip;
                let take = self.take;
                Ok(py.allow_threads(move || count_fused_i64_bounded(&data, &ops, skip, take)))
            }
            QueryInner::U8 { data, ops } => {
                let data = Arc::clone(data);
                let ops = ops.clone();
                let skip = self.skip;
                let take = self.take;
                Ok(py.allow_threads(move || count_u8_bounded(&data, &ops, skip, take)))
            }
            QueryInner::Py(items) => Ok(apply_skip_take(items, self.skip, self.take).len()),
            QueryInner::Obj { source, ops } => count_py_lazy(py, source, ops, self.skip, self.take),
            QueryInner::RustObj { data, ops } => {
                let data = Arc::clone(data);
                let ops = ops.clone();
                let (skip, take) = (self.skip, self.take);
                Ok(py.allow_threads(move || count_obj_pipeline(&data, &ops, skip, take)))
            }
            QueryInner::LazyFloatList { source, ops } => Ok(count_lazy_float_list(
                source.as_ptr(),
                ops,
                self.skip,
                self.take,
            )),
            QueryInner::NumpyF64 { source, ops } => {
                count_numpy_f64(py, source, ops, self.skip, self.take)
            }
            QueryInner::NumpyF32 { source, ops } => {
                count_numpy_f32(py, source, ops, self.skip, self.take)
            }
            QueryInner::ObjField {
                source,
                field_name,
                ops,
            } => count_by_field(py, source, field_name, ops, self.skip, self.take),
            QueryInner::ObjFieldPy { source, ops, .. } => {
                count_by_field_py(py, source, ops, self.skip, self.take)
            }
            QueryInner::Empty => Ok(0),
        }
    }

    fn first(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = self.take(py, 1).to_list(py)?;
        let list = list.bind(py);
        if list.is_empty() {
            Ok(py.None())
        } else {
            Ok(list.get_item(0)?.into())
        }
    }

    fn last(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = self.to_list(py)?;
        let list = list.bind(py);
        let n = list.len();
        if n == 0 {
            Ok(py.None())
        } else {
            Ok(list.get_item(n - 1)?.into())
        }
    }

    fn sum(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                // Fast path: single filter op, no skip/take → fused SIMD pass, no Vec
                if self.skip == 0 && self.take.is_none() && !self.parallel {
                    let data = pipeline.arc();
                    let ops = pipeline.clone_ops();
                    if let Some(s) = py.allow_threads(|| filter_sum_fused_f64(&data, &ops)) {
                        return Ok(s.into_py(py));
                    }
                }
                let data = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                let s = crate::simd::simd_sum_f64(&data);
                Ok(s.into_py(py))
            }
            QueryInner::I64(pipeline) => {
                let data = execute_i64(py, pipeline, self.skip, self.take, self.parallel);
                let s: i64 = py.allow_threads(|| data.iter().sum());
                Ok(s.into_py(py))
            }
            QueryInner::U8 { data, ops } => {
                let data = Arc::clone(data);
                let ops = ops.clone();
                let skip = self.skip;
                let take = self.take;
                let s: i64 = py.allow_threads(move || sum_u8_bounded(&data, &ops, skip, take));
                Ok(s.into_py(py))
            }
            QueryInner::Py(_)
            | QueryInner::Obj { .. }
            | QueryInner::ObjField { .. }
            | QueryInner::ObjFieldPy { .. } => {
                let list = self.to_list(py)?;
                let builtins = py.import_bound("builtins")?;
                let result = builtins.getattr("sum")?.call1((list,))?;
                Ok(result.into())
            }
            QueryInner::RustObj { .. } => {
                // sum() on object path without field() arg → not meaningful; return 0
                Ok(0i64.into_py(py))
            }
            QueryInner::LazyFloatList { source, ops } => {
                let v = execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                let s = py.allow_threads(|| crate::simd::simd_sum_f64(&v));
                Ok(s.into_py(py))
            }
            QueryInner::NumpyF64 { source, ops } => {
                let v = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                let s = py.allow_threads(|| crate::simd::simd_sum_f64(&v));
                Ok(s.into_py(py))
            }
            QueryInner::NumpyF32 { source, ops } => {
                let v = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                let s = py.allow_threads(|| crate::simd::simd_sum_f32(&v));
                Ok((s as f64).into_py(py))
            }
            QueryInner::Empty => Ok(0i64.into_py(py)),
        }
    }

    /// Sum a specific field over matching rows — GIL-free for RustObj path.
    /// Usage: `Query(rows).filter(field("active") == True).sum_field("price")`
    fn sum_field(&self, py: Python<'_>, field_name: &str) -> PyResult<PyObject> {
        match &self.inner {
            QueryInner::RustObj { data, ops } => {
                let data = Arc::clone(data);
                let ops = ops.clone();
                let field_name = field_name.to_string();
                let (skip, take) = (self.skip, self.take);
                let s = py.allow_threads(move || {
                    sum_field_obj_pipeline(&data, &field_name, &ops, skip, take)
                });
                Ok(s.into_py(py))
            }
            // Obj (dict list): fused single-pass — no intermediate Vec<PyObject>.
            QueryInner::Obj { source, ops }
                if ops.is_empty() && self.skip == 0 && self.take.is_none() =>
            {
                // Direct path: no filter ops, just sum field over all rows.
                let list = source.bind(py);
                let mut acc = 0.0f64;
                for item_res in list.iter()? {
                    let item = item_res?;
                    if let Ok(dict) = item.downcast::<PyDict>() {
                        if let Ok(Some(v)) = dict.get_item(field_name) {
                            acc += v
                                .extract::<f64>()
                                .or_else(|_| v.extract::<i64>().map(|i| i as f64))
                                .unwrap_or(0.0);
                        }
                    }
                }
                Ok(acc.into_py(py))
            }
            QueryInner::Obj { source, ops } => {
                // Has filter/map ops: materialize then sum (uncommon path).
                let items = collect_py_lazy(py, source, ops, self.skip, self.take)?;
                let mut acc = 0.0f64;
                for item in &items {
                    if let Ok(dict) = item.bind(py).downcast::<PyDict>() {
                        if let Ok(Some(v)) = dict.get_item(field_name) {
                            acc += v
                                .extract::<f64>()
                                .or_else(|_| v.extract::<i64>().map(|i| i as f64))
                                .unwrap_or(0.0);
                        }
                    }
                }
                Ok(acc.into_py(py))
            }
            // ObjField: fused filter+sum, no Vec<PyObject> materialisation.
            QueryInner::ObjField {
                source,
                field_name: filter_field,
                ops,
            } => {
                let s = sum_field_by_field(
                    py,
                    source,
                    filter_field,
                    ops,
                    field_name,
                    self.skip,
                    self.take,
                )?;
                Ok(s.into_py(py))
            }
            QueryInner::ObjFieldPy {
                source,
                ops,
                map_field,
            } => {
                let items = filter_by_field_py(
                    py,
                    source,
                    ops,
                    self.skip,
                    self.take,
                    map_field.as_deref(),
                )?;
                let mut acc = 0.0f64;
                for item in &items {
                    if let Ok(dict) = item.bind(py).downcast::<PyDict>() {
                        if let Ok(Some(v)) = dict.get_item(field_name) {
                            acc += v
                                .extract::<f64>()
                                .or_else(|_| v.extract::<i64>().map(|i| i as f64))
                                .unwrap_or(0.0);
                        }
                    }
                }
                Ok(acc.into_py(py))
            }
            _ => {
                let list = self.to_list(py)?;
                let builtins = py.import_bound("builtins")?;
                let result = builtins.getattr("sum")?.call1((list,))?;
                Ok(result.into())
            }
        }
    }

    /// Extract a single field from every dict in the pipeline result.
    ///
    /// For `ObjFieldPy` (string/bool field filter), fuses filter + field extraction
    /// into a single Rust loop — no Python function call overhead, no intermediate list.
    /// For all other paths, falls back to `map(operator.itemgetter(field_name))`.
    fn map_field(&self, py: Python<'_>, field_name: &str) -> PyResult<Self> {
        match &self.inner {
            QueryInner::ObjFieldPy {
                source,
                ops,
                map_field: None,
            } => {
                // Fuse: single Rust loop will do filter + field extraction
                Ok(PyQuery::from_inner(QueryInner::ObjFieldPy {
                    source: source.clone_ref(py),
                    ops: ops.clone(),
                    map_field: Some(Arc::from(field_name)),
                }))
            }
            _ => {
                // Generic path: use operator.itemgetter (C-level, faster than lambda)
                let op = py
                    .import_bound("operator")?
                    .getattr("itemgetter")?
                    .call1((field_name,))?;
                self.map(py, op)
            }
        }
    }

    /// Mean of the sequence.  Returns `None` (Python `None`) on empty input.
    /// For f64 with a single filter op: single SIMD pass, zero intermediate Vec.
    fn mean(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                if self.skip == 0 && self.take.is_none() && !self.parallel {
                    let data = pipeline.arc();
                    let ops = pipeline.clone_ops();
                    if let Some(result) = py.allow_threads(|| filter_mean_fused_f64(&data, &ops)) {
                        return Ok(result.map_or_else(|| py.None(), |v| v.into_py(py)));
                    }
                }
                let data = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                if data.is_empty() {
                    return Ok(py.None());
                }
                let mean = py.allow_threads(|| {
                    let s = crate::simd::simd_sum_f64(&data);
                    s / data.len() as f64
                });
                Ok(mean.into_py(py))
            }
            QueryInner::I64(pipeline) => {
                let data = execute_i64(py, pipeline, self.skip, self.take, self.parallel);
                if data.is_empty() {
                    return Ok(py.None());
                }
                let mean = py.allow_threads(|| {
                    let s: i64 = data.iter().sum();
                    s as f64 / data.len() as f64
                });
                Ok(mean.into_py(py))
            }
            QueryInner::U8 { data, ops } => {
                let data = Arc::clone(data);
                let ops = ops.clone();
                let skip = self.skip;
                let take = self.take;
                let mean = py.allow_threads(move || {
                    let s = sum_u8_bounded(&data, &ops, skip, take);
                    let n = count_u8_bounded(&data, &ops, skip, take);
                    if n == 0 {
                        None
                    } else {
                        Some(s as f64 / n as f64)
                    }
                });
                Ok(mean.map_or_else(|| py.None(), |v| v.into_py(py)))
            }
            QueryInner::LazyFloatList { source, ops } => {
                let v = execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                if v.is_empty() {
                    return Ok(py.None());
                }
                let mean = py.allow_threads(|| {
                    let s = crate::simd::simd_sum_f64(&v);
                    s / v.len() as f64
                });
                Ok(mean.into_py(py))
            }
            QueryInner::NumpyF64 { source, ops } => {
                let v = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                if v.is_empty() {
                    return Ok(py.None());
                }
                let mean = py.allow_threads(|| {
                    let s = crate::simd::simd_sum_f64(&v);
                    s / v.len() as f64
                });
                Ok(mean.into_py(py))
            }
            QueryInner::Empty => Ok(py.None()),
            _ => {
                // Obj / RustObj / Py: materialise then compute in Python
                let list = self.to_list(py)?;
                let list = list.bind(py);
                if list.is_empty() {
                    return Ok(py.None());
                }
                let len = list.len();
                let builtins = py.import_bound("builtins")?;
                let s = builtins.getattr("sum")?.call1((list,))?;
                let mean = s.extract::<f64>()? / len as f64;
                Ok(mean.into_py(py))
            }
        }
    }

    /// Population variance (denominator N).  Returns `None` on empty / all-filtered.
    /// For f64 with a single filter op: SIMD single-pass (sum + sum² + count).
    fn var(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                if self.skip == 0 && self.take.is_none() && !self.parallel {
                    let data = pipeline.arc();
                    let ops = pipeline.clone_ops();
                    if let Some(r) = py.allow_threads(|| filter_var_fused_f64(&data, &ops)) {
                        return Ok(r.map_or_else(|| py.None(), |v| v.into_py(py)));
                    }
                }
                let data = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                if data.is_empty() {
                    return Ok(py.None());
                }
                let v = py.allow_threads(|| {
                    let n = data.len() as f64;
                    let mean = crate::simd::simd_sum_f64(&data) / n;
                    let ssq: f64 = data
                        .iter()
                        .map(|&x| {
                            let d = x - mean;
                            d * d
                        })
                        .sum();
                    ssq / n
                });
                Ok(v.into_py(py))
            }
            QueryInner::I64(pipeline) => {
                let data = execute_i64(py, pipeline, self.skip, self.take, self.parallel);
                if data.is_empty() {
                    return Ok(py.None());
                }
                let v = py.allow_threads(|| {
                    let n = data.len() as f64;
                    let mean = data.iter().sum::<i64>() as f64 / n;
                    let ssq: f64 = data
                        .iter()
                        .map(|&x| {
                            let d = x as f64 - mean;
                            d * d
                        })
                        .sum();
                    ssq / n
                });
                Ok(v.into_py(py))
            }
            QueryInner::LazyFloatList { source, ops } => {
                let v = execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                if v.is_empty() {
                    return Ok(py.None());
                }
                let var = py.allow_threads(|| {
                    let n = v.len() as f64;
                    let mean = crate::simd::simd_sum_f64(&v) / n;
                    let ssq: f64 = v
                        .iter()
                        .map(|&x| {
                            let d = x - mean;
                            d * d
                        })
                        .sum();
                    ssq / n
                });
                Ok(var.into_py(py))
            }
            QueryInner::NumpyF64 { source, ops } => {
                let v = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                if v.is_empty() {
                    return Ok(py.None());
                }
                let var = py.allow_threads(|| {
                    let n = v.len() as f64;
                    let mean = crate::simd::simd_sum_f64(&v) / n;
                    let ssq: f64 = v
                        .iter()
                        .map(|&x| {
                            let d = x - mean;
                            d * d
                        })
                        .sum();
                    ssq / n
                });
                Ok(var.into_py(py))
            }
            QueryInner::Empty => Ok(py.None()),
            _ => {
                let list = self.to_list(py)?;
                let list = list.bind(py);
                if list.is_empty() {
                    return Ok(py.None());
                }
                let n = list.len();
                let builtins = py.import_bound("builtins")?;
                let sum_val = builtins.getattr("sum")?.call1((list,))?.extract::<f64>()?;
                let mean = sum_val / n as f64;
                let ssq: f64 = (0..n)
                    .map(|i| {
                        let x = list
                            .get_item(i)
                            .and_then(|v| v.extract::<f64>())
                            .unwrap_or(0.0);
                        let d = x - mean;
                        d * d
                    })
                    .sum();
                Ok((ssq / n as f64).into_py(py))
            }
        }
    }

    /// Population standard deviation (square root of `var()`).
    /// Returns `None` on empty / all-filtered.
    fn std(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.var(py)? {
            ref v if v.is_none(py) => Ok(py.None()),
            v => {
                let variance: f64 = v.extract(py)?;
                Ok(variance.sqrt().into_py(py))
            }
        }
    }

    fn min(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner {
            QueryInner::LazyFloatList { source, ops } => {
                let v = execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                let result = py.allow_threads(|| v.iter().copied().reduce(f64::min));
                match result {
                    Some(val) => Ok(val.into_py(py)),
                    None => Ok(py.None()),
                }
            }
            QueryInner::NumpyF64 { source, ops } => {
                let v = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                let result = py.allow_threads(|| v.iter().copied().reduce(f64::min));
                match result {
                    Some(val) => Ok(val.into_py(py)),
                    None => Ok(py.None()),
                }
            }
            QueryInner::NumpyF32 { source, ops } => {
                let v = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                let result = py.allow_threads(|| v.iter().copied().reduce(f32::min));
                match result {
                    Some(val) => Ok((val as f64).into_py(py)),
                    None => Ok(py.None()),
                }
            }
            QueryInner::F64(pipeline) => {
                if self.skip == 0 && self.take.is_none() && !self.parallel {
                    let data = pipeline.arc();
                    let ops = pipeline.clone_ops();
                    if let Some(result) = py.allow_threads(|| filter_min_fused_f64(&data, &ops)) {
                        return Ok(result.map_or_else(|| py.None(), |v| v.into_py(py)));
                    }
                }
                let data = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                let result = py.allow_threads(|| data.iter().copied().reduce(f64::min));
                match result {
                    Some(v) => Ok(v.into_py(py)),
                    None => Ok(py.None()),
                }
            }
            _ => {
                let list = self.to_list(py)?;
                let builtins = py.import_bound("builtins")?;
                let result = builtins.getattr("min")?.call1((list,))?;
                Ok(result.into())
            }
        }
    }

    fn max(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner {
            QueryInner::LazyFloatList { source, ops } => {
                let v = execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                let result = py.allow_threads(|| crate::simd::simd_max_f64(&v));
                match result {
                    Some(val) => Ok(val.into_py(py)),
                    None => Ok(py.None()),
                }
            }
            QueryInner::NumpyF64 { source, ops } => {
                let v = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                let result = py.allow_threads(|| crate::simd::simd_max_f64(&v));
                match result {
                    Some(val) => Ok(val.into_py(py)),
                    None => Ok(py.None()),
                }
            }
            QueryInner::NumpyF32 { source, ops } => {
                let v = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                let result = py.allow_threads(|| crate::simd::simd_max_f32(&v));
                match result {
                    Some(val) => Ok((val as f64).into_py(py)),
                    None => Ok(py.None()),
                }
            }
            QueryInner::F64(pipeline) => {
                if self.skip == 0 && self.take.is_none() && !self.parallel {
                    let data = pipeline.arc();
                    let ops = pipeline.clone_ops();
                    if let Some(result) = py.allow_threads(|| filter_max_fused_f64(&data, &ops)) {
                        return Ok(result.map_or_else(|| py.None(), |v| v.into_py(py)));
                    }
                }
                let data = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                let result = py.allow_threads(|| crate::simd::simd_max_f64(&data));
                match result {
                    Some(v) => Ok(v.into_py(py)),
                    None => Ok(py.None()),
                }
            }
            _ => {
                let list = self.to_list(py)?;
                let builtins = py.import_bound("builtins")?;
                let result = builtins.getattr("max")?.call1((list,))?;
                Ok(result.into())
            }
        }
    }

    /// Single-pass fused stats: count + sum + mean + min + max.
    ///
    /// Returns a Python dict with keys "count", "sum", "mean", "min", "max".
    /// "mean", "min", "max" are None when no element passes the filter.
    ///
    /// Use this instead of calling count()/sum()/mean()/min()/max() separately —
    /// each separate call re-scans the data, while stats() does a single pass.
    fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Helper: pack a (count, sum, min, max) result into a Python dict.
        let to_dict = |py: Python<'_>,
                       result: Option<(usize, f64, f64, f64)>|
         -> PyResult<PyObject> {
            let d = PyDict::new_bound(py);
            match result {
                None => {
                    d.set_item("count", 0usize)?;
                    d.set_item("sum", 0.0f64)?;
                    d.set_item("mean", py.None())?;
                    d.set_item("min", py.None())?;
                    d.set_item("max", py.None())?;
                }
                Some((cnt, sum, min, max)) => {
                    d.set_item("count", cnt)?;
                    d.set_item("sum", sum)?;
                    d.set_item("mean", sum / cnt as f64)?;
                    d.set_item("min", min)?;
                    d.set_item("max", max)?;
                }
            }
            Ok(d.into())
        };

        match &self.inner {
            QueryInner::F64(pipeline) => {
                if self.skip == 0 && self.take.is_none() && !self.parallel {
                    let data = pipeline.arc();
                    let ops = pipeline.clone_ops();
                    let result =
                        py.allow_threads(|| filter_multi_stat_f64(&data, &ops));
                    return to_dict(py, result);
                }
                let data = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                let result = py.allow_threads(|| filter_multi_stat_f64(&data, &[]));
                to_dict(py, result)
            }
            QueryInner::LazyFloatList { source, ops } => {
                let v = execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                let result = py.allow_threads(|| filter_multi_stat_f64(&v, &[]));
                to_dict(py, result)
            }
            QueryInner::NumpyF64 { source, ops } => {
                let v = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                let result = py.allow_threads(|| filter_multi_stat_f64(&v, &[]));
                to_dict(py, result)
            }
            QueryInner::NumpyF32 { source, ops } => {
                let v = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                let v_f64: Vec<f64> = v.iter().map(|&x| x as f64).collect();
                let result = py.allow_threads(|| filter_multi_stat_f64(&v_f64, &[]));
                to_dict(py, result)
            }
            _ => {
                let list = self.to_list(py)?;
                let list = list.bind(py);
                let mut cnt = 0usize;
                let mut sum = 0.0f64;
                let mut min = f64::INFINITY;
                let mut max = f64::NEG_INFINITY;
                for item in list.iter() {
                    if let Ok(v) = item.extract::<f64>() {
                        cnt += 1;
                        sum += v;
                        if v < min {
                            min = v;
                        }
                        if v > max {
                            max = v;
                        }
                    }
                }
                to_dict(py, if cnt == 0 { None } else { Some((cnt, sum, min, max)) })
            }
        }
    }

    fn reduce(
        &self,
        py: Python<'_>,
        f: Bound<'_, PyAny>,
        initial: Option<PyObject>,
    ) -> PyResult<PyObject> {
        let list = self.to_list(py)?;
        let list = list.bind(py);
        let mut acc = match initial {
            Some(ref v) => v.clone(),
            None => {
                if list.is_empty() {
                    return Err(PyValueError::new_err(
                        "reduce() on empty sequence with no initial value",
                    ));
                }
                list.get_item(0)?.into()
            }
        };
        let start = if initial.is_none() { 1 } else { 0 };
        for i in start..list.len() {
            let item = list.get_item(i)?;
            acc = f.call1((acc, item))?.into();
        }
        Ok(acc)
    }

    fn for_each(&self, py: Python<'_>, f: Bound<'_, PyAny>) -> PyResult<()> {
        let list = self.to_list(py)?;
        let list = list.bind(py);
        for item in list.iter() {
            f.call1((item,))?;
        }
        Ok(())
    }

    fn any(&self, py: Python<'_>, pred: Bound<'_, PyAny>) -> PyResult<bool> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                // DSL fast path: col > 0 etc. → count > 0, fully GIL-free
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if expr.op.is_filter() {
                        if let Some(op) = expr.op.to_f64_op() {
                            let data = pipeline.arc();
                            let base_ops = pipeline.clone_ops();
                            let skip = self.skip;
                            let take = self.take;
                            let n = py.allow_threads(|| {
                                if skip == 0 && take.is_none() {
                                    let mut ops = base_ops;
                                    ops.push(op);
                                    count_fused_f64_bounded(&data, &ops, 0, None)
                                } else {
                                    let bounded =
                                        execute_fused_f64_bounded(&data, &base_ops, skip, take);
                                    count_fused_f64_bounded(&bounded, &[op], 0, None)
                                }
                            });
                            return Ok(n > 0);
                        }
                    }
                }
                let data = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                for val in data {
                    if pred.call1((val.into_py(py),))?.is_truthy()? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            QueryInner::I64(pipeline) => {
                // DSL fast path: col > 0 etc. -> count > 0, fully GIL-free
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if expr.op.is_filter() {
                        if let Some(op) = expr_to_int_op(&expr.op) {
                            let data = pipeline.arc();
                            let base_ops = pipeline.clone_ops();
                            let skip = self.skip;
                            let take = self.take;
                            let n = py.allow_threads(|| {
                                if skip == 0 && take.is_none() {
                                    let mut ops = base_ops;
                                    ops.push(op);
                                    count_fused_i64_bounded(&data, &ops, 0, None)
                                } else {
                                    let bounded =
                                        execute_fused_i64_bounded(&data, &base_ops, skip, take);
                                    count_fused_i64_bounded(&bounded, &[op], 0, None)
                                }
                            });
                            return Ok(n > 0);
                        }
                    }
                }
                let data = execute_i64(py, pipeline, self.skip, self.take, false);
                for val in data {
                    if pred.call1((val.into_py(py),))?.is_truthy()? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            QueryInner::U8 { data, ops } => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if expr.op.is_filter() {
                        if let Some(op) = expr_to_int_op(&expr.op) {
                            let data = Arc::clone(data);
                            let mut check_ops = ops.clone();
                            check_ops.push(op);
                            let (skip, take) = (self.skip, self.take);
                            let n = py.allow_threads(move || {
                                count_u8_bounded(&data, &check_ops, skip, take)
                            });
                            return Ok(n > 0);
                        }
                    }
                }
                let data = execute_u8(py, data, ops, self.skip, self.take);
                for val in data {
                    if pred.call1((val.into_py(py),))?.is_truthy()? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            QueryInner::Py(items) => {
                for item in apply_skip_take(items, self.skip, self.take) {
                    if pred.call1((item.clone_ref(py),))?.is_truthy()? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            QueryInner::Obj { source, ops } => {
                let bound = bind_ops(py, ops);
                let iter = source.bind(py).iter()?;
                let mut skipped = 0usize;
                let mut count = 0usize;
                'outer: for item_res in iter {
                    let mut item: PyObject = item_res?.into();
                    for op in &bound {
                        match op {
                            BoundOp::Filter(f) => {
                                if !f.call1((item.clone_ref(py),))?.is_truthy()? {
                                    continue 'outer;
                                }
                            }
                            BoundOp::Map(f) => {
                                item = f.call1((item,))?.into();
                            }
                        }
                    }
                    if skipped < self.skip {
                        skipped += 1;
                        continue;
                    }
                    if pred.call1((item,))?.is_truthy()? {
                        return Ok(true);
                    }
                    count += 1;
                    if self.take.is_some_and(|n| count >= n) {
                        break;
                    }
                }
                Ok(false)
            }
            QueryInner::LazyFloatList { source, ops } => {
                let data = execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                for val in data {
                    if pred.call1((val.into_py(py),))?.is_truthy()? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            QueryInner::NumpyF64 { source, ops } => {
                let data = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                for val in data {
                    if pred.call1((val.into_py(py),))?.is_truthy()? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            QueryInner::NumpyF32 { source, ops } => {
                let data = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                for val in data {
                    if pred.call1(((val as f64).into_py(py),))?.is_truthy()? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            QueryInner::RustObj { data, ops } => {
                // FieldExpr DSL fast path: fully GIL-free count check
                if let Ok(fexpr) = pred.extract::<PyRef<PyFieldExpr>>() {
                    if let Some(extra_op) = fexpr.op.clone() {
                        let data = Arc::clone(data);
                        let mut check_ops = ops.clone();
                        check_ops.push(extra_op);
                        let (skip, take) = (self.skip, self.take);
                        let n = py.allow_threads(move || {
                            count_obj_pipeline(&data, &check_ops, skip, take)
                        });
                        return Ok(n > 0);
                    }
                }
                // Lambda fallback: materialize GIL-free, then check under GIL
                let data = Arc::clone(data);
                let ops_c = ops.clone();
                let (skip, take) = (self.skip, self.take);
                let rows =
                    py.allow_threads(move || execute_obj_pipeline(&data, &ops_c, skip, take));
                for row in &rows {
                    if pred.call1((rust_row_to_py(py, row),))?.is_truthy()? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            QueryInner::ObjField { .. } | QueryInner::ObjFieldPy { .. } => {
                let list = self.to_list(py)?;
                for item in list.bind(py).iter() {
                    if pred.call1((item,))?.is_truthy()? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            QueryInner::Empty => Ok(false),
        }
    }

    fn all(&self, py: Python<'_>, pred: Bound<'_, PyAny>) -> PyResult<bool> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                // DSL fast path: all pass pred ↔ count_with_pred == count_total, GIL-free
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if expr.op.is_filter() {
                        if let Some(op) = expr.op.to_f64_op() {
                            let data = pipeline.arc();
                            let base_ops = pipeline.clone_ops();
                            let skip = self.skip;
                            let take = self.take;
                            let (total, matching) = py.allow_threads(|| {
                                if skip == 0 && take.is_none() {
                                    let mut filter_ops = base_ops.clone();
                                    filter_ops.push(op);
                                    (
                                        count_fused_f64_bounded(&data, &base_ops, 0, None),
                                        count_fused_f64_bounded(&data, &filter_ops, 0, None),
                                    )
                                } else {
                                    let bounded =
                                        execute_fused_f64_bounded(&data, &base_ops, skip, take);
                                    (
                                        bounded.len(),
                                        count_fused_f64_bounded(&bounded, &[op], 0, None),
                                    )
                                }
                            });
                            return Ok(total == matching);
                        }
                    }
                }
                let data = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                for val in data {
                    if !pred.call1((val.into_py(py),))?.is_truthy()? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            QueryInner::I64(pipeline) => {
                // DSL fast path: all pass pred <-> count_with_pred == count_total.
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if expr.op.is_filter() {
                        if let Some(op) = expr_to_int_op(&expr.op) {
                            let data = pipeline.arc();
                            let base_ops = pipeline.clone_ops();
                            let skip = self.skip;
                            let take = self.take;
                            let (total, matching) = py.allow_threads(|| {
                                if skip == 0 && take.is_none() {
                                    let mut filter_ops = base_ops.clone();
                                    filter_ops.push(op);
                                    (
                                        count_fused_i64_bounded(&data, &base_ops, 0, None),
                                        count_fused_i64_bounded(&data, &filter_ops, 0, None),
                                    )
                                } else {
                                    let bounded =
                                        execute_fused_i64_bounded(&data, &base_ops, skip, take);
                                    (
                                        bounded.len(),
                                        count_fused_i64_bounded(&bounded, &[op], 0, None),
                                    )
                                }
                            });
                            return Ok(total == matching);
                        }
                    }
                }
                let data = execute_i64(py, pipeline, self.skip, self.take, false);
                for val in data {
                    if !pred.call1((val.into_py(py),))?.is_truthy()? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            QueryInner::U8 { data, ops } => {
                if let Ok(expr) = pred.extract::<PyRef<PyExpr>>() {
                    if expr.op.is_filter() {
                        if let Some(op) = expr_to_int_op(&expr.op) {
                            let data = Arc::clone(data);
                            let base_ops = ops.clone();
                            let mut check_ops = base_ops.clone();
                            check_ops.push(op);
                            let (skip, take) = (self.skip, self.take);
                            let (total, matching) = py.allow_threads(move || {
                                (
                                    count_u8_bounded(&data, &base_ops, skip, take),
                                    count_u8_bounded(&data, &check_ops, skip, take),
                                )
                            });
                            return Ok(total == matching);
                        }
                    }
                }
                let data = execute_u8(py, data, ops, self.skip, self.take);
                for val in data {
                    if !pred.call1((val.into_py(py),))?.is_truthy()? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            QueryInner::Py(items) => {
                for item in apply_skip_take(items, self.skip, self.take) {
                    if !pred.call1((item.clone_ref(py),))?.is_truthy()? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            QueryInner::Obj { source, ops } => {
                let bound = bind_ops(py, ops);
                let iter = source.bind(py).iter()?;
                let mut skipped = 0usize;
                let mut count = 0usize;
                'outer: for item_res in iter {
                    let mut item: PyObject = item_res?.into();
                    for op in &bound {
                        match op {
                            BoundOp::Filter(f) => {
                                if !f.call1((item.clone_ref(py),))?.is_truthy()? {
                                    continue 'outer;
                                }
                            }
                            BoundOp::Map(f) => {
                                item = f.call1((item,))?.into();
                            }
                        }
                    }
                    if skipped < self.skip {
                        skipped += 1;
                        continue;
                    }
                    if !pred.call1((item,))?.is_truthy()? {
                        return Ok(false);
                    }
                    count += 1;
                    if self.take.is_some_and(|n| count >= n) {
                        break;
                    }
                }
                Ok(true)
            }
            QueryInner::LazyFloatList { source, ops } => {
                let data = execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                for val in data {
                    if !pred.call1((val.into_py(py),))?.is_truthy()? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            QueryInner::NumpyF64 { source, ops } => {
                let data = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                for val in data {
                    if !pred.call1((val.into_py(py),))?.is_truthy()? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            QueryInner::NumpyF32 { source, ops } => {
                let data = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                for val in data {
                    if !pred.call1(((val as f64).into_py(py),))?.is_truthy()? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            QueryInner::RustObj { data, ops } => {
                // FieldExpr DSL fast path: total == matching → all pass
                if let Ok(fexpr) = pred.extract::<PyRef<PyFieldExpr>>() {
                    if let Some(extra_op) = fexpr.op.clone() {
                        let data = Arc::clone(data);
                        let base_ops = ops.clone();
                        let mut check_ops = base_ops.clone();
                        check_ops.push(extra_op);
                        let (skip, take) = (self.skip, self.take);
                        let (total, matching) = py.allow_threads(move || {
                            (
                                count_obj_pipeline(&data, &base_ops, skip, take),
                                count_obj_pipeline(&data, &check_ops, skip, take),
                            )
                        });
                        return Ok(total == matching);
                    }
                }
                // Lambda fallback
                let data = Arc::clone(data);
                let ops_c = ops.clone();
                let (skip, take) = (self.skip, self.take);
                let rows =
                    py.allow_threads(move || execute_obj_pipeline(&data, &ops_c, skip, take));
                for row in &rows {
                    if !pred.call1((rust_row_to_py(py, row),))?.is_truthy()? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            QueryInner::ObjField { .. } | QueryInner::ObjFieldPy { .. } => {
                let list = self.to_list(py)?;
                for item in list.bind(py).iter() {
                    if !pred.call1((item,))?.is_truthy()? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            QueryInner::Empty => Ok(true),
        }
    }

    /// Expose pipeline info for debugging / benchmarking
    fn __repr__(&self) -> String {
        let kind = match &self.inner {
            QueryInner::F64(_) => "f64",
            QueryInner::I64(_) => "i64",
            QueryInner::U8 { .. } => "u8",
            QueryInner::Py(_) => "py",
            QueryInner::Obj { ops, .. } => {
                if ops.is_empty() {
                    "obj"
                } else {
                    "obj(lazy)"
                }
            }
            QueryInner::RustObj { ops, .. } => {
                if ops.is_empty() {
                    "rust_obj"
                } else {
                    "rust_obj(lazy)"
                }
            }
            QueryInner::ObjField { .. } => "obj_field",
            QueryInner::ObjFieldPy { map_field, .. } => {
                if map_field.is_some() {
                    "obj_field_py(mapped)"
                } else {
                    "obj_field_py"
                }
            }
            QueryInner::LazyFloatList { .. } => "lazy_f64_list",
            QueryInner::NumpyF64 { .. } => "numpy_f64",
            QueryInner::NumpyF32 { .. } => "numpy_f32",
            QueryInner::Empty => "empty",
        };
        format!(
            "Query<{}>(skip={}, take={:?}, parallel={})",
            kind, self.skip, self.take, self.parallel
        )
    }

    /// Human-readable explanation of what execution path this query will take.
    ///
    /// Reports pipeline kind, queued ops, skip/take bounds, parallel flag,
    /// GIL-free classification, and allocation estimate.
    fn explain(&self) -> String {
        let (kind, gil_free, ops_str) = match &self.inner {
            QueryInner::F64(p) => {
                let ops = p.clone_ops();
                let ops_repr = if ops.is_empty() {
                    "(none)".to_string()
                } else {
                    ops.iter()
                        .map(|o| format!("{o:?}"))
                        .collect::<Vec<_>>()
                        .join(" → ")
                };
                ("f64", true, ops_repr)
            }
            QueryInner::I64(p) => {
                let ops = p.clone_ops();
                let ops_repr = if ops.is_empty() {
                    "(none)".to_string()
                } else {
                    ops.iter()
                        .map(|o| format!("{o:?}"))
                        .collect::<Vec<_>>()
                        .join(" → ")
                };
                ("i64", true, ops_repr)
            }
            QueryInner::U8 { ops, .. } => {
                let ops_repr = if ops.is_empty() {
                    "(none)".to_string()
                } else {
                    ops.iter()
                        .map(|o| format!("{o:?}"))
                        .collect::<Vec<_>>()
                        .join(" → ")
                };
                ("u8 (bool/uint8)", true, ops_repr)
            }
            QueryInner::LazyFloatList { ops, .. } => {
                let ops_repr = if ops.is_empty() {
                    "(none)".to_string()
                } else {
                    ops.iter()
                        .map(|o| format!("{o:?}"))
                        .collect::<Vec<_>>()
                        .join(" → ")
                };
                ("lazy_f64_list", true, ops_repr)
            }
            QueryInner::NumpyF64 { ops, .. } => {
                let ops_repr = if ops.is_empty() {
                    "(none)".to_string()
                } else {
                    ops.iter()
                        .map(|o| format!("{o:?}"))
                        .collect::<Vec<_>>()
                        .join(" → ")
                };
                ("numpy_f64 (zero-copy buffer)", true, ops_repr)
            }
            QueryInner::NumpyF32 { ops, .. } => {
                let ops_repr = if ops.is_empty() {
                    "(none)".to_string()
                } else {
                    ops.iter()
                        .map(|o| format!("{o:?}"))
                        .collect::<Vec<_>>()
                        .join(" → ")
                };
                ("numpy_f32 (zero-copy buffer, f32x8 SIMD)", true, ops_repr)
            }
            QueryInner::RustObj { ops, .. } => {
                let ops_repr = if ops.is_empty() {
                    "(none — dict→RustObj conversion already paid at construction)".to_string()
                } else {
                    ops.iter()
                        .map(|o| format!("{o:?}"))
                        .collect::<Vec<_>>()
                        .join(" → ")
                };
                ("rust_obj (GIL-free ops; cold path cost paid at preload/first field() filter)", true, ops_repr)
            }
            QueryInner::Obj { ops, .. } => {
                let n = ops.len();
                let ops_repr = if n == 0 {
                    "(none — will use obj_field path on first field() filter)".to_string()
                } else {
                    format!("{n} Python callable op(s)")
                };
                ("obj (Python path)", false, ops_repr)
            }
            QueryInner::ObjField {
                field_name, ops, ..
            } => {
                let ops_repr = if ops.is_empty() {
                    "(none)".to_string()
                } else {
                    ops.iter()
                        .map(|o| format!("{o:?}"))
                        .collect::<Vec<_>>()
                        .join(" → ")
                };
                let kind_str = format!("obj_field[{}] (extract+SIMD+reindex)", field_name);
                return format!(
                    "Query.explain()\n  kind:     {kind_str}\n  ops:      {ops_repr}\n  skip:     {}\n  take:     {}\n  parallel: {}\n  gil_free: partial (extract GIL, compare free, collect GIL)\n  alloc:    1 Vec<f64> + 1 Vec<usize> at terminal",
                    self.skip,
                    match self.take { Some(n) => format!("{n}"), None => "∞".to_string() },
                    self.parallel
                );
            }
            QueryInner::ObjFieldPy { ops, map_field, .. } => {
                let ops_repr = ops
                    .iter()
                    .map(|o| format!("{o:?}"))
                    .collect::<Vec<_>>()
                    .join(" → ");
                let map_repr = map_field
                    .as_deref()
                    .map(|f| format!(", map_field={f}"))
                    .unwrap_or_default();
                let kind_str =
                    format!("obj_field_py (C-API loop, no Python frame overhead{map_repr})");
                return format!(
                    "Query.explain()\n  kind:     {kind_str}\n  ops:      {ops_repr}\n  skip:     {}\n  take:     {}\n  parallel: {}\n  gil_free: false (GIL held, C-API only)\n  alloc:    1 Vec at terminal",
                    self.skip,
                    match self.take { Some(n) => format!("{n}"), None => "∞".to_string() },
                    self.parallel
                );
            }
            QueryInner::Py(items) => (
                "py (materialized)",
                false,
                format!("{} element(s)", items.len()),
            ),
            QueryInner::Empty => ("empty", true, "(none)".to_string()),
        };

        let take_str = match self.take {
            Some(n) => format!("{n}"),
            None => "∞".to_string(),
        };
        let alloc = if gil_free {
            "1 Vec at terminal"
        } else {
            "1 Vec at terminal (GIL held during ops)"
        };

        format!(
            "Query.explain()\n  kind:     {kind}\n  ops:      {ops_str}\n  skip:     {}\n  take:     {take_str}\n  parallel: {}\n  gil_free: {gil_free}\n  alloc:    {alloc}",
            self.skip, self.parallel
        )
    }

    /// Single-pass group + aggregate in Rust.
    ///
    /// Internal entrypoint — Python callers use `Query.group_agg(**specs)` which is
    /// attached in `zpyflow/__init__.py` and unpacks kwargs before delegating here.
    ///
    /// `names` and `specs` must be the same length.
    /// For the common Obj path with no pending filter/map ops, the source list is
    /// iterated directly — no intermediate Vec is built.
    /// Returns `list[dict]` where each dict has `"_key"` plus one entry per spec.
    fn _group_agg(
        &self,
        py: Python<'_>,
        key_fn: Bound<'_, PyAny>,
        names: Vec<String>,
        specs: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let kinds: Vec<AggSpecKind> = specs
            .iter()
            .map(|s| s.extract::<PyRef<PyAggSpec>>().map(|r| r.kind.clone()))
            .collect::<PyResult<_>>()?;

        if let Ok(fexpr) = key_fn.extract::<PyRef<PyFieldExpr>>() {
            if fexpr.op.is_some() {
                return Err(PyValueError::new_err(
                    "group_agg field key must be a bare field(\"name\") expression",
                ));
            }
            if kinds.iter().all(|k| matches!(k, AggSpecKind::Count)) {
                return group_agg_field_count(self, py, Arc::clone(&fexpr.name), &names, &kinds);
            }
            return Err(PyValueError::new_err(
                "group_agg(field(...), ...) currently supports agg_count() specs only; use a lambda key for callable aggregations",
            ));
        }

        let key_to_idx = PyDict::new_bound(py);
        let mut keys: Vec<PyObject> = Vec::new();
        let mut accs: Vec<Vec<f64>> = Vec::new();

        let mut process = |item: Bound<'_, PyAny>| -> PyResult<()> {
            let key = key_fn.call1((&item,))?;
            let idx = match key_to_idx.get_item(&key)? {
                Some(v) => v.extract::<usize>()?,
                None => {
                    let i = keys.len();
                    keys.push(key.clone().unbind());
                    accs.push(agg_init_acc(&kinds));
                    key_to_idx.set_item(&key, i)?;
                    i
                }
            };
            agg_update(py, &mut accs[idx], &kinds, &item)
        };

        // Direct path: Obj with no pending ops and no skip/take — iterate source once.
        let is_direct = matches!(&self.inner, QueryInner::Obj { ops, .. } if ops.is_empty())
            && self.skip == 0
            && self.take.is_none();

        if is_direct {
            if let QueryInner::Obj { source, .. } = &self.inner {
                for item_res in source.bind(py).iter()? {
                    process(item_res?)?;
                }
            }
        } else {
            let list = self.to_list(py)?;
            for item in list.bind(py).iter() {
                process(item)?;
            }
        }

        agg_build_result(py, keys, accs, &names, &kinds)
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
        QueryInner::Obj { .. } | QueryInner::ObjField { .. } | QueryInner::ObjFieldPy { .. } => {}
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
        let list = query.to_list(py)?;
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
                .execute_parallel_bounded(skip, take);
        }
        // skip and take are passed as explicit bounds — NOT inserted into ops.
        // Semantics: skip = source-level (before filters), take = output-level (after filters).
        // This avoids the O(n) ops.insert(0, ...) and clarifies execution order.
        execute_fused_f64_bounded(&data, &ops, skip, take)
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
                .execute_parallel_bounded(skip, take);
        }
        execute_fused_i64_bounded(&data, &ops, skip, take)
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
    py.allow_threads(move || execute_u8_bounded(&data, &ops, skip, take))
}

fn execute_u8_bounded(
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

fn count_u8_bounded(data: &[u8], ops: &[IntOp], out_skip: usize, out_take: Option<usize>) -> usize {
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

fn sum_u8_bounded(data: &[u8], ops: &[IntOp], out_skip: usize, out_take: Option<usize>) -> i64 {
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

/// Apply a Python filter predicate to f64 data (GIL held).
fn apply_py_filter_f64(
    py: Python<'_>,
    data: Vec<f64>,
    pred: &Bound<'_, PyAny>,
) -> PyResult<Vec<f64>> {
    let mut out = Vec::with_capacity(data.len() / 2);
    for val in data {
        let py_val = val.into_py(py);
        if pred.call1((py_val,))?.is_truthy()? {
            out.push(val);
        }
    }
    Ok(out)
}

fn apply_py_filter_i64(
    py: Python<'_>,
    data: Vec<i64>,
    pred: &Bound<'_, PyAny>,
) -> PyResult<Vec<i64>> {
    let mut out = Vec::with_capacity(data.len() / 2);
    for val in data {
        let py_val = val.into_py(py);
        if pred.call1((py_val,))?.is_truthy()? {
            out.push(val);
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
    let mut out = Vec::new();
    let cap = take.unwrap_or(items.len().saturating_sub(skip));
    out.reserve(cap);

    let mut skipped = 0;
    for item in items {
        if skipped < skip {
            skipped += 1;
            continue;
        }
        if pred.call1((item.clone_ref(py),))?.is_truthy()? {
            out.push(item.clone_ref(py));
        }
        if let Some(n) = take {
            if out.len() >= n {
                break;
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
