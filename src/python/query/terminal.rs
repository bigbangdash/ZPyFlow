//! Terminal operations that materialise the pipeline: `to_list()`, `count()`, `sum()`, aggregations, etc.
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
    count_fused_f64_bounded, count_fused_i64_bounded, count_obj_pipeline, eval_filter_f64,
    eval_filter_i64, execute_fused_f64_bounded, execute_fused_i64_bounded, execute_obj_pipeline,
    filter_multi_stat_f64, max_fused_f64_bounded, mean_fused_f64_bounded, min_fused_f64_bounded,
    row_passes, stats_fused_f64_bounded, sum_field_obj_pipeline, sum_fused_f64_bounded,
    var_fused_f64_bounded, IntOp, IntPipeline, NumericOp, NumericPipeline, ObjOp, RustRow,
    RustValue,
};
use super::{
    agg_build_result, agg_init_acc, agg_update, apply_py_filter, apply_py_filter_f64,
    apply_py_filter_i64, apply_py_map, apply_py_map_f64, apply_py_map_i64, apply_skip_take,
    bind_ops, branch_f64_pipeline, branch_i64_pipeline, clone_inner, collect_f64_pipeline,
    collect_i64_pipeline, collect_py_lazy, count_py_lazy, count_u8_bounded, execute_f64,
    execute_i64, execute_u8, execute_u8_bounded, expr_to_int_op, group_agg_field_count,
    group_by_collect as group_by_collect_fn, group_key_from_rust_value, group_key_to_py,
    group_keys_to_py, native_slice_as_bytes, native_vec_to_numpy, parsed_to_query,
    push_numeric_op, sum_u8_bounded, try_lambda_to_dsl_expr, BoundOp, collect_to_pylist, PyPipelineOp, PyQuery, QueryInner, take_query,
};
use ahash::AHashMap;
use std::sync::Arc;

#[pymethods]
impl PyQuery {
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
    /// Imports Python numpy only when this method is called. The Rust core does
    /// not link to the numpy crate. Numeric output is copied as raw bytes into
    /// numpy, avoiding per-element Python float/int boxing, then copied once more
    /// to return a writable, owned ndarray.
    ///
    /// Dtype mapping:
    ///   f64 pipeline → np.float64
    ///   i64 pipeline → np.int64
    ///   u8 pipeline  → np.int64 after pending numeric ops
    ///
    /// Raises ValueError for object/Py pipelines — use `to_list()` + `np.array()`.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                let result = execute_f64(py, pipeline, self.skip, self.take, self.parallel);
                native_vec_to_numpy(py, result, "float64")
            }
            QueryInner::I64(pipeline) => {
                let result = execute_i64(py, pipeline, self.skip, self.take, self.parallel);
                native_vec_to_numpy(py, result, "int64")
            }
            QueryInner::U8 { data, ops } => {
                let data = Arc::clone(data);
                let ops = ops.clone();
                let skip = self.skip;
                let take = self.take;
                let result = py.allow_threads(move || execute_u8_bounded(&data, &ops, skip, take));
                native_vec_to_numpy(py, result, "int64")
            }
            QueryInner::LazyFloatList { source, ops } => {
                let result =
                    execute_lazy_float_list(py, source.as_ptr(), ops, self.skip, self.take);
                native_vec_to_numpy(py, result, "float64")
            }
            QueryInner::NumpyF64 { source, ops } => {
                let result = execute_numpy_f64(py, source, ops, self.skip, self.take)?;
                native_vec_to_numpy(py, result, "float64")
            }
            QueryInner::NumpyF32 { source, ops } => {
                let result = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                native_vec_to_numpy(py, result, "float32")
            }
            QueryInner::Empty => {
                let empty: Vec<f64> = vec![];
                native_vec_to_numpy(py, empty, "float64")
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
    fn group_by_collect(&self, py: Python<'_>, key_fn: Bound<'_, PyAny>) -> PyResult<PyObject> {
        match &self.inner {
            QueryInner::Obj { source, ops } => {
                group_by_collect_fn(py, source, ops, &key_fn, self.skip, self.take)
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
        let list = collect_to_pylist(&take_query(self, py, 1), py)?;
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
                let data = pipeline.arc();
                let ops = pipeline.clone_ops();
                let skip = self.skip;
                let take = self.take;
                let s = py.allow_threads(|| sum_fused_f64_bounded(&data, &ops, skip, take));
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
                let s = sum_lazy_float_list(source.as_ptr(), ops, self.skip, self.take);
                Ok(s.into_py(py))
            }
            QueryInner::NumpyF64 { source, ops } => {
                let s = sum_numpy_f64(py, source, ops, self.skip, self.take)?;
                Ok(s.into_py(py))
            }
            QueryInner::NumpyF32 { source, ops } => {
                let v = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                let s = py.allow_threads(|| crate::core::simd_sum_f32(&v));
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
    fn mean(&self, py: Python<'_>) -> PyResult<PyObject> {
        match &self.inner {
            QueryInner::F64(pipeline) => {
                let data = pipeline.arc();
                let ops = pipeline.clone_ops();
                let skip = self.skip;
                let take = self.take;
                let mean = py.allow_threads(|| mean_fused_f64_bounded(&data, &ops, skip, take));
                Ok(mean.map_or_else(|| py.None(), |v| v.into_py(py)))
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
                let mean = mean_lazy_float_list(source.as_ptr(), ops, self.skip, self.take);
                Ok(mean.map_or_else(|| py.None(), |v| v.into_py(py)))
            }
            QueryInner::NumpyF64 { source, ops } => {
                let mean = mean_numpy_f64(py, source, ops, self.skip, self.take)?;
                Ok(mean.map_or_else(|| py.None(), |v| v.into_py(py)))
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
                let data = pipeline.arc();
                let ops = pipeline.clone_ops();
                let skip = self.skip;
                let take = self.take;
                let v = py.allow_threads(|| var_fused_f64_bounded(&data, &ops, skip, take));
                Ok(v.map_or_else(|| py.None(), |v| v.into_py(py)))
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
                let var = var_lazy_float_list(source.as_ptr(), ops, self.skip, self.take);
                Ok(var.map_or_else(|| py.None(), |v| v.into_py(py)))
            }
            QueryInner::NumpyF64 { source, ops } => {
                let var = var_numpy_f64(py, source, ops, self.skip, self.take)?;
                Ok(var.map_or_else(|| py.None(), |v| v.into_py(py)))
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
                let result = min_lazy_float_list(source.as_ptr(), ops, self.skip, self.take);
                Ok(result.map_or_else(|| py.None(), |v| v.into_py(py)))
            }
            QueryInner::NumpyF64 { source, ops } => {
                let result = min_numpy_f64(py, source, ops, self.skip, self.take)?;
                Ok(result.map_or_else(|| py.None(), |v| v.into_py(py)))
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
                let data = pipeline.arc();
                let ops = pipeline.clone_ops();
                let skip = self.skip;
                let take = self.take;
                let result = py.allow_threads(|| min_fused_f64_bounded(&data, &ops, skip, take));
                Ok(result.map_or_else(|| py.None(), |v| v.into_py(py)))
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
                let result = max_lazy_float_list(source.as_ptr(), ops, self.skip, self.take);
                Ok(result.map_or_else(|| py.None(), |v| v.into_py(py)))
            }
            QueryInner::NumpyF64 { source, ops } => {
                let result = max_numpy_f64(py, source, ops, self.skip, self.take)?;
                Ok(result.map_or_else(|| py.None(), |v| v.into_py(py)))
            }
            QueryInner::NumpyF32 { source, ops } => {
                let v = execute_numpy_f32(py, source, ops, self.skip, self.take)?;
                let result = py.allow_threads(|| crate::core::simd_max_f32(&v));
                match result {
                    Some(val) => Ok((val as f64).into_py(py)),
                    None => Ok(py.None()),
                }
            }
            QueryInner::F64(pipeline) => {
                let data = pipeline.arc();
                let ops = pipeline.clone_ops();
                let skip = self.skip;
                let take = self.take;
                let result = py.allow_threads(|| max_fused_f64_bounded(&data, &ops, skip, take));
                Ok(result.map_or_else(|| py.None(), |v| v.into_py(py)))
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
                let data = pipeline.arc();
                let ops = pipeline.clone_ops();
                let skip = self.skip;
                let take = self.take;
                let result = py.allow_threads(|| stats_fused_f64_bounded(&data, &ops, skip, take));
                to_dict(py, result)
            }
            QueryInner::LazyFloatList { source, ops } => {
                let result = stats_lazy_float_list(source.as_ptr(), ops, self.skip, self.take);
                to_dict(py, result)
            }
            QueryInner::NumpyF64 { source, ops } => {
                let result = stats_numpy_f64(py, source, ops, self.skip, self.take)?;
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
