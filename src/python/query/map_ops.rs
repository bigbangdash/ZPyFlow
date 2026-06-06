//! Transformation operations: `map()`, `map_field()`, `flat_map()`, `preload()`.
#![allow(unused_imports)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::python::agg::{AggSpecKind, GroupKey, PyAggSpec};
use crate::python::columnar::{
    apply_columnar_ops, columnar_indices_to_py_list, convert_to_columnar, ColumnarData,
};
use crate::python::conversion::{rust_row_to_py, try_convert_to_rust_obj};
use crate::python::schema::infer_schema_inner;
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
    push_numeric_op, sum_u8_with_skip_take, try_lambda_to_dsl_expr, BoundOp, collect_to_pylist, MappedResult, MappedResultI, PyPipelineOp, PyQuery, QueryInner, take_query,
};
use ahash::AHashMap;
use std::sync::Arc;

#[pymethods]
impl PyQuery {
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
            QueryInner::ColumnarObj { data, ops } => {
                // Materialize to dicts, then apply map as Obj pipeline
                let indices = apply_columnar_ops(data, ops, self.skip, self.take);
                let rows = columnar_indices_to_py_list(py, data, &indices)?;
                let mat = PyList::new_bound(py, rows.iter().map(|o| o.bind(py)));
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

    fn flat_map(&self, py: Python<'_>, f: Bound<'_, PyAny>) -> PyResult<PyQuery> {
        let list = collect_to_pylist(self, py)?;
        let mut out: Vec<PyObject> = Vec::new();
        for item in list.bind(py).iter() {
            let sub = f.call1((item,))?;
            for elem in sub.iter()? {
                out.push(elem?.unbind());
            }
        }
        Ok(PyQuery::from_inner(QueryInner::Py(out)))
    }

    /// Convert a dict-list pipeline to columnar layout (spec-082 T4).
    ///
    /// Pays the conversion cost once so repeated `field()` DSL queries run on
    /// typed column slices with no per-row dict lookup.
    ///
    ///   q = Query(logs).preload()            # converts once — GIL, O(N)
    ///   q.filter(field("score") > 5).count() # GIL-free column scan
    ///
    /// Falls back to `RustObj` if the source is not a list-of-dicts, and to
    /// a no-op clone if the pipeline already has pending ops (they are applied
    /// at terminal time as usual).
    fn preload(&self, py: Python<'_>) -> PyResult<PyQuery> {
        if let QueryInner::Obj { source, ops } = &self.inner {
            if ops.is_empty() {
                if let Ok(list) = source.bind(py).downcast::<PyList>() {
                    // Try columnar conversion first (spec-082 T4 primary path)
                    let schema = infer_schema_inner(list, 100)?;
                    if !schema.is_empty() {
                        let columnar = convert_to_columnar(py, list, &schema)?;
                        return Ok(PyQuery {
                            inner: QueryInner::ColumnarObj {
                                data: Arc::new(columnar),
                                ops: Vec::new(),
                            },
                            skip: self.skip,
                            take: self.take,
                            parallel: self.parallel,
                        });
                    }
                    // Non-dict list: fall back to RustObj
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

}
