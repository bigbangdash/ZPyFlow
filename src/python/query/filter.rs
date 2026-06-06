//! Filter and predicate operations: `filter()`, `take_while()`, `skip_while()`.
#![allow(unused_imports)]
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::python::agg::{AggSpecKind, GroupKey, PyAggSpec};
use crate::python::conversion::{rust_row_to_py, try_convert_to_rust_obj};
use crate::python::expr::{ExprOp, PyExpr, PyFieldExpr};
use crate::python::columnar::{apply_columnar_ops, columnar_indices_to_py_list};
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
            QueryInner::ColumnarObj { data, ops } => {
                // FieldExpr DSL: accumulate ObjOp without materializing (no skip/take pending)
                if self.skip == 0 && self.take.is_none() {
                    if let Ok(fexpr) = pred.extract::<PyRef<PyFieldExpr>>() {
                        if let Some(fop) = &fexpr.op {
                            let mut new_ops = ops.clone();
                            new_ops.push(fop.clone());
                            return Ok(PyQuery::from_inner(QueryInner::ColumnarObj {
                                data: Arc::clone(data),
                                ops: new_ops,
                            }));
                        }
                    }
                }
                // Lambda or skip/take pending: materialize to dicts, fall back to Obj
                let indices = apply_columnar_ops(data, ops, self.skip, self.take);
                let rows = columnar_indices_to_py_list(py, data, &indices)?;
                let mat = PyList::new_bound(py, rows.iter().map(|o| o.bind(py)));
                Ok(PyQuery::from_inner(QueryInner::Obj {
                    source: mat.unbind().into_py(py),
                    ops: vec![PyPipelineOp::Filter(pred.unbind())],
                }))
            }
            QueryInner::Empty => Ok(PyQuery::from_inner(QueryInner::Empty)),
        }
    }

    // ------------------------------------------------------------------
    // map(expr_or_callable)
    // ------------------------------------------------------------------

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
                let list = collect_to_pylist(self, py)?;
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
                let list = collect_to_pylist(self, py)?;
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

}
