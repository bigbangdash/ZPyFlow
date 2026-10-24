use pyo3::prelude::*;
use std::sync::Arc;

pub(crate) enum AggSpecKind {
    Count,
    Sum(Py<PyAny>),
    Mean(Py<PyAny>),
    Max(Py<PyAny>),
    Min(Py<PyAny>),
}

impl AggSpecKind {
    pub(crate) fn clone_ref(&self, py: Python<'_>) -> Self {
        match self {
            AggSpecKind::Count => AggSpecKind::Count,
            AggSpecKind::Sum(f) => AggSpecKind::Sum(f.clone_ref(py)),
            AggSpecKind::Mean(f) => AggSpecKind::Mean(f.clone_ref(py)),
            AggSpecKind::Max(f) => AggSpecKind::Max(f.clone_ref(py)),
            AggSpecKind::Min(f) => AggSpecKind::Min(f.clone_ref(py)),
        }
    }
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
/// Use the static constructors:
/// ```python
/// from zpyflow import Query, AggSpec
/// result = Query(data).group_agg(
///     lambda p: p["category"],
///     ["count", "revenue"],
///     [AggSpec.count(), AggSpec.sum(lambda p: p["price"])],
/// )
/// ```
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
    fn sum(field_fn: Py<PyAny>) -> Self {
        PyAggSpec {
            kind: AggSpecKind::Sum(field_fn),
        }
    }
    #[staticmethod]
    fn mean(field_fn: Py<PyAny>) -> Self {
        PyAggSpec {
            kind: AggSpecKind::Mean(field_fn),
        }
    }
    #[staticmethod]
    fn max(field_fn: Py<PyAny>) -> Self {
        PyAggSpec {
            kind: AggSpecKind::Max(field_fn),
        }
    }
    #[staticmethod]
    fn min(field_fn: Py<PyAny>) -> Self {
        PyAggSpec {
            kind: AggSpecKind::Min(field_fn),
        }
    }
}
