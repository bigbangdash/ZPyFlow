//! Constructor methods for [`PyQuery`]: `new()`, static constructors, and I/O source bridges.
#![allow(unused_imports)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::python::agg::{AggSpecKind, GroupKey, PyAggSpec};
use crate::python::conversion::{rust_row_to_py, try_convert_to_rust_obj};
use crate::python::expr::{ExprOp, PyExpr, PyFieldExpr};
use crate::python::fastpath::{
    count_by_field, count_by_field_py, count_lazy_float_list, count_numpy_f32, count_numpy_f64,
    execute_lazy_float_list, execute_numpy_f32, execute_numpy_f64, filter_by_field,
    filter_by_field_py, max_lazy_float_list, max_numpy_f64, mean_lazy_float_list,
    mean_numpy_f64, min_lazy_float_list, min_numpy_f64, objop_is_field_filter, objop_to_numeric,
    stats_lazy_float_list, stats_numpy_f64, sum_field_by_field, sum_lazy_float_list,
    sum_numpy_f64, var_lazy_float_list, var_numpy_f64,
};
use crate::python::io_bridge::{csv_col_spec, parse_csv, parse_jsonl};
use crate::io::ParsedOutput;
use crate::core::{
    count_fused_f64_with_skip_take, count_fused_i64_with_skip_take, count_obj_pipeline, eval_filter_f64,
    eval_filter_i64, execute_fused_f64_with_skip_take, execute_fused_i64_with_skip_take, execute_obj_pipeline,
    filter_multi_stat_f64, max_fused_f64_with_skip_take, mean_fused_f64_with_skip_take, min_fused_f64_with_skip_take,
    row_passes, stats_fused_f64_with_skip_take, sum_field_obj_pipeline, sum_fused_f64_with_skip_take,
    var_fused_f64_with_skip_take, IntOp, IntPipeline, NumericOp, NumericPipeline, ObjOp, RustRow,
    RustValue,
};
use super::{
    agg_build_result, agg_init_acc, agg_update, apply_py_filter, apply_py_filter_f64,
    apply_py_filter_i64, apply_py_map, apply_py_map_f64, apply_py_map_i64, apply_skip_take,
    bind_ops, branch_f64_pipeline, branch_i64_pipeline, clone_inner, collect_f64_pipeline,
    collect_i64_pipeline, collect_py_lazy, count_py_lazy, count_u8_with_skip_take, execute_f64,
    execute_i64, execute_u8, execute_u8_with_skip_take, expr_to_int_op, group_agg_field_count,
    group_by_collect as group_by_collect_fn, group_key_from_rust_value, group_key_to_py,
    group_keys_to_py, native_slice_as_bytes, parsed_to_query,
    push_numeric_op, sum_u8_with_skip_take, try_lambda_to_dsl_expr, BoundOp, collect_to_pylist, PyPipelineOp, PyQuery, QueryInner, take_query,
};
use ahash::AHashMap;
use std::sync::Arc;

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
        use super::super::fastpath::RawBuffer;
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
        use super::super::fastpath::RawBuffer;
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
        use super::super::fastpath::RawBuffer;
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
        use super::super::fastpath::RawBuffer;
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

}
