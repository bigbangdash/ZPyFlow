//! Structural combinators: `take()`, `skip()`, `parallel()`, `chain()`, `enumerate()`, `zip()`.
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
                    let mut out = execute_fused_f64_with_skip_take(&a_data, &a_ops, a_skip, a_take);
                    out.extend_from_slice(&execute_fused_f64_with_skip_take(
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
                    let mut out = execute_fused_i64_with_skip_take(&a_data, &a_ops, a_skip, a_take);
                    out.extend_from_slice(&execute_fused_i64_with_skip_take(
                        &b_data, &b_ops, b_skip, b_take,
                    ));
                    out
                });
                Ok(PyQuery::from_inner(QueryInner::I64(IntPipeline::new(
                    result,
                ))))
            }
            _ => {
                let a_pylist = collect_to_pylist(self, py)?;
                let mut a_list = a_pylist.bind(py).iter().map(|x| x.unbind()).collect::<Vec<PyObject>>();
                let b_list = collect_to_pylist(&other, py)?;
                a_list.extend(b_list.bind(py).iter().map(|x| x.unbind()));
                Ok(PyQuery::from_inner(QueryInner::Py(a_list)))
            }
        }
    }

    /// Yield `(index, item)` tuples (zero-based). Always returns a Py pipeline.
    fn enumerate(&self, py: Python<'_>) -> PyResult<PyQuery> {
        use pyo3::types::PyTuple;
        let list = collect_to_pylist(self, py)?;
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
        let a = collect_to_pylist(self, py)?;
        let b = collect_to_pylist(&other, py)?;
        let n = a.bind(py).len().min(b.bind(py).len());
        let mut out: Vec<PyObject> = Vec::with_capacity(n);
        for (x, y) in a.bind(py).iter().zip(b.bind(py).iter()) {
            let t = PyTuple::new_bound(py, [x.unbind(), y.unbind()]);
            out.push(t.into());
        }
        Ok(PyQuery::from_inner(QueryInner::Py(out)))
    }

}
