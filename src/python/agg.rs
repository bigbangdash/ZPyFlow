use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) enum AggSpecKind {
    Count,
    Sum(PyObject),
    Mean(PyObject),
    Max(PyObject),
    Min(PyObject),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum GroupKey {
    Null,
    Bool(bool),
    Int(i64),
    Float(u64),
    Str(Arc<str>),
}

/// Python-facing aggregation specification for `Query.group_agg()`.
///
/// Use the static constructors::
///
///     from zpyflow import Query, AggSpec
///     result = Query(data).group_agg(
///         lambda p: p["category"],
///         ["count", "revenue"],
///         [AggSpec.count(), AggSpec.sum(lambda p: p["price"])],
///     )
#[pyclass(name = "AggSpec")]
pub struct PyAggSpec {
    pub(crate) kind: AggSpecKind,
}

#[pymethods]
impl PyAggSpec {
    #[staticmethod]
    fn count() -> Self {
        PyAggSpec {
            kind: AggSpecKind::Count,
        }
    }
    #[staticmethod]
    fn sum(field_fn: PyObject) -> Self {
        PyAggSpec {
            kind: AggSpecKind::Sum(field_fn),
        }
    }
    #[staticmethod]
    fn mean(field_fn: PyObject) -> Self {
        PyAggSpec {
            kind: AggSpecKind::Mean(field_fn),
        }
    }
    #[staticmethod]
    fn max(field_fn: PyObject) -> Self {
        PyAggSpec {
            kind: AggSpecKind::Max(field_fn),
        }
    }
    #[staticmethod]
    fn min(field_fn: PyObject) -> Self {
        PyAggSpec {
            kind: AggSpecKind::Min(field_fn),
        }
    }
}
