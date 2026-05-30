use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use crate::pipeline::numeric::NumericOp;
use crate::pipeline::obj::{ObjOp, RustValue};

// ---------------------------------------------------------------------------
// FieldExpr DSL — object pipeline equivalent of col/Expr
// ---------------------------------------------------------------------------

/// `field("price") > 100` compiles to an ObjOp without any Python call.
#[pyclass(name = "FieldExpr")]
#[derive(Clone)]
pub struct PyFieldExpr {
    pub(crate) name: Arc<str>,
    pub(crate) op: Option<ObjOp>, // None = bare field access (for sum/group key)
}

#[pymethods]
impl PyFieldExpr {
    fn __gt__(&self, other: f64) -> Self {
        PyFieldExpr {
            name: Arc::clone(&self.name),
            op: Some(ObjOp::FilterFieldGt(Arc::clone(&self.name), other)),
        }
    }
    fn __ge__(&self, other: f64) -> Self {
        PyFieldExpr {
            name: Arc::clone(&self.name),
            op: Some(ObjOp::FilterFieldGe(Arc::clone(&self.name), other)),
        }
    }
    fn __lt__(&self, other: f64) -> Self {
        PyFieldExpr {
            name: Arc::clone(&self.name),
            op: Some(ObjOp::FilterFieldLt(Arc::clone(&self.name), other)),
        }
    }
    fn __le__(&self, other: f64) -> Self {
        PyFieldExpr {
            name: Arc::clone(&self.name),
            op: Some(ObjOp::FilterFieldLe(Arc::clone(&self.name), other)),
        }
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rv = py_to_rust_value(other)?;
        Ok(PyFieldExpr {
            name: Arc::clone(&self.name),
            op: Some(ObjOp::FilterFieldEq(Arc::clone(&self.name), rv)),
        })
    }
    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rv = py_to_rust_value(other)?;
        Ok(PyFieldExpr {
            name: Arc::clone(&self.name),
            op: Some(ObjOp::FilterFieldNe(Arc::clone(&self.name), rv)),
        })
    }
    /// `field("price").between(10, 100)`
    fn between(&self, lo: f64, hi: f64) -> Self {
        PyFieldExpr {
            name: Arc::clone(&self.name),
            op: Some(ObjOp::FilterFieldBetween(Arc::clone(&self.name), lo, hi)),
        }
    }
    fn __repr__(&self) -> String {
        format!("FieldExpr(\"{}\")", self.name)
    }

    /// Evaluate this field expression against a Python dict.
    /// Enables FieldExpr to be used anywhere a Python callable is expected
    /// (e.g. `.any(field("x") > 5)` on an Obj pipeline).
    fn __call__(&self, row: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Some(op) = &self.op else {
            return Ok(true);
        };
        if let Ok(dict) = row.downcast::<PyDict>() {
            obj_op_passes_py_dict(dict, op)
        } else {
            Ok(false)
        }
    }
}

// ---------------------------------------------------------------------------
// Expression DSL exposed to Python
// ---------------------------------------------------------------------------

/// Python-side predicate/transform descriptor.
/// When the user writes `col > 5.0` they get back an `Expr` that maps to a
/// Rust `NumericOp` — no GIL is needed to evaluate it.
#[pyclass(name = "Expr")]
#[derive(Clone)]
pub struct PyExpr {
    pub(crate) op: ExprOp,
}

#[derive(Clone)]
pub(crate) enum ExprOp {
    // Comparisons (filter)
    Gt(f64),
    Ge(f64),
    Lt(f64),
    Le(f64),
    Eq(f64),
    Ne(f64),
    Between(f64, f64),
    SelfEq, // col == col: passes when NOT NaN (arrow null-drop pattern)
    // Scalar transforms
    MulScalar(f64),
    AddScalar(f64),
    SubScalar(f64),
    DivScalar(f64),
    PowScalar(f64),
    // Unary maps
    Abs,
    Neg,
    Sqrt,
    Floor,
    Ceil,
    Round,
    Reciprocal,
}

impl ExprOp {
    pub fn to_f64_op(&self) -> Option<NumericOp> {
        Some(match self {
            ExprOp::Gt(t) => NumericOp::FilterGt(*t),
            ExprOp::Ge(t) => NumericOp::FilterGe(*t),
            ExprOp::Lt(t) => NumericOp::FilterLt(*t),
            ExprOp::Le(t) => NumericOp::FilterLe(*t),
            ExprOp::Eq(t) => NumericOp::FilterEq(*t),
            ExprOp::Ne(t) => NumericOp::FilterNe(*t),
            ExprOp::Between(lo, hi) => NumericOp::FilterBetween(*lo, *hi),
            ExprOp::SelfEq => NumericOp::FilterNotNan,
            ExprOp::MulScalar(s) => NumericOp::MapMulScalar(*s),
            ExprOp::AddScalar(s) => NumericOp::MapAddScalar(*s),
            ExprOp::SubScalar(s) => NumericOp::MapSubScalar(*s),
            ExprOp::DivScalar(s) => NumericOp::MapDivScalar(*s),
            ExprOp::PowScalar(s) => NumericOp::MapPowScalar(*s),
            ExprOp::Abs => NumericOp::MapAbs,
            ExprOp::Neg => NumericOp::MapNeg,
            ExprOp::Sqrt => NumericOp::MapSqrt,
            ExprOp::Floor => NumericOp::MapFloor,
            ExprOp::Ceil => NumericOp::MapCeil,
            ExprOp::Round => NumericOp::MapRound,
            ExprOp::Reciprocal => NumericOp::MapReciprocal,
        })
    }

    pub fn is_filter(&self) -> bool {
        matches!(
            self,
            ExprOp::Gt(_)
                | ExprOp::Ge(_)
                | ExprOp::Lt(_)
                | ExprOp::Le(_)
                | ExprOp::Eq(_)
                | ExprOp::Ne(_)
                | ExprOp::Between(_, _)
                | ExprOp::SelfEq
        )
    }
}

#[pymethods]
impl PyExpr {
    /// Make Expr callable from Python so it can be used as a map/filter function.
    fn __call__(&self, py: Python<'_>, val: f64) -> PyObject {
        match &self.op {
            ExprOp::Gt(t) => (val > *t).into_py(py),
            ExprOp::Ge(t) => (val >= *t).into_py(py),
            ExprOp::Lt(t) => (val < *t).into_py(py),
            ExprOp::Le(t) => (val <= *t).into_py(py),
            ExprOp::Eq(t) => (val == *t).into_py(py),
            ExprOp::Ne(t) => (val != *t).into_py(py),
            ExprOp::Between(lo, hi) => (val >= *lo && val <= *hi).into_py(py),
            ExprOp::SelfEq => (val == val).into_py(py),
            ExprOp::MulScalar(s) => (val * s).into_py(py),
            ExprOp::AddScalar(s) => (val + s).into_py(py),
            ExprOp::SubScalar(s) => (val - s).into_py(py),
            ExprOp::DivScalar(s) => (val / s).into_py(py),
            ExprOp::PowScalar(s) => val.powf(*s).into_py(py),
            ExprOp::Abs => val.abs().into_py(py),
            ExprOp::Neg => (-val).into_py(py),
            ExprOp::Sqrt => val.sqrt().into_py(py),
            ExprOp::Floor => val.floor().into_py(py),
            ExprOp::Ceil => val.ceil().into_py(py),
            ExprOp::Round => val.round().into_py(py),
            ExprOp::Reciprocal => (1.0 / val).into_py(py),
        }
    }
}

/// Factory: `field("price") > 100.0` → `FieldExpr` with an embedded `ObjOp`.
#[pyfunction]
pub fn field(name: &str) -> PyFieldExpr {
    PyFieldExpr {
        name: Arc::from(name),
        op: None,
    }
}

/// Sentinel object — `Query(data).filter(col > 5)`.
/// Returned by `col > 5` etc.
#[pyclass(name = "ColProxy")]
pub struct PyColProxy;

#[pymethods]
impl PyColProxy {
    fn __gt__(&self, other: f64) -> PyExpr {
        PyExpr {
            op: ExprOp::Gt(other),
        }
    }
    fn __ge__(&self, other: f64) -> PyExpr {
        PyExpr {
            op: ExprOp::Ge(other),
        }
    }
    fn __lt__(&self, other: f64) -> PyExpr {
        PyExpr {
            op: ExprOp::Lt(other),
        }
    }
    fn __le__(&self, other: f64) -> PyExpr {
        PyExpr {
            op: ExprOp::Le(other),
        }
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        // col == col  →  "not NaN" filter (Arrow null-drop pattern)
        if other.is_instance_of::<PyColProxy>() {
            return Ok(PyExpr { op: ExprOp::SelfEq });
        }
        let v: f64 = other.extract()?;
        Ok(PyExpr { op: ExprOp::Eq(v) })
    }
    fn __ne__(&self, other: f64) -> PyExpr {
        PyExpr {
            op: ExprOp::Ne(other),
        }
    }
    fn __mul__(&self, other: f64) -> PyExpr {
        PyExpr {
            op: ExprOp::MulScalar(other),
        }
    }
    fn __add__(&self, other: f64) -> PyExpr {
        PyExpr {
            op: ExprOp::AddScalar(other),
        }
    }
    fn __sub__(&self, other: f64) -> PyExpr {
        PyExpr {
            op: ExprOp::SubScalar(other),
        }
    }
    fn __truediv__(&self, other: f64) -> PyExpr {
        PyExpr {
            op: ExprOp::DivScalar(other),
        }
    }
    fn __pow__(&self, other: f64, _modulo: Option<f64>) -> PyExpr {
        PyExpr {
            op: ExprOp::PowScalar(other),
        }
    }
    fn __neg__(&self) -> PyExpr {
        PyExpr { op: ExprOp::Neg }
    }
    fn abs(&self) -> PyExpr {
        PyExpr { op: ExprOp::Abs }
    }
    fn sqrt(&self) -> PyExpr {
        PyExpr { op: ExprOp::Sqrt }
    }
    fn between(&self, lo: f64, hi: f64) -> PyExpr {
        PyExpr {
            op: ExprOp::Between(lo, hi),
        }
    }
    fn floor(&self) -> PyExpr {
        PyExpr { op: ExprOp::Floor }
    }
    fn ceil(&self) -> PyExpr {
        PyExpr { op: ExprOp::Ceil }
    }
    fn round(&self) -> PyExpr {
        PyExpr { op: ExprOp::Round }
    }
    fn reciprocal(&self) -> PyExpr {
        PyExpr {
            op: ExprOp::Reciprocal,
        }
    }
}

/// Python scalar → RustValue.  Nested lists/dicts become Null (unsupported).
pub(crate) fn py_to_rust_value(obj: &Bound<'_, PyAny>) -> PyResult<RustValue> {
    if obj.is_none() {
        return Ok(RustValue::Null);
    }
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(RustValue::Bool(b));
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(RustValue::Int(i));
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(RustValue::Float(f));
    }
    if let Ok(s) = obj.extract::<&str>() {
        return Ok(RustValue::Str(Arc::from(s)));
    }
    Ok(RustValue::Null)
}

/// Check an ObjOp against a Python dict without converting to a full RustRow.
/// Accesses only the field referenced by the op, so it's O(1) per element.
fn obj_op_passes_py_dict(dict: &Bound<'_, PyDict>, op: &ObjOp) -> PyResult<bool> {
    #[inline]
    fn get_f64(dict: &Bound<'_, PyDict>, field: &str) -> f64 {
        match dict.get_item(field) {
            Ok(Some(v)) => v
                .extract::<f64>()
                .or_else(|_| v.extract::<i64>().map(|i| i as f64))
                .or_else(|_| v.extract::<bool>().map(|b| if b { 1.0 } else { 0.0 }))
                .unwrap_or(f64::NAN),
            _ => f64::NAN,
        }
    }
    match op {
        ObjOp::FilterFieldGt(f, t) => Ok(get_f64(dict, f) > *t),
        ObjOp::FilterFieldGe(f, t) => Ok(get_f64(dict, f) >= *t),
        ObjOp::FilterFieldLt(f, t) => Ok(get_f64(dict, f) < *t),
        ObjOp::FilterFieldLe(f, t) => Ok(get_f64(dict, f) <= *t),
        ObjOp::FilterFieldBetween(f, lo, hi) => {
            let v = get_f64(dict, f);
            Ok(v >= *lo && v <= *hi)
        }
        ObjOp::FilterFieldEq(f, target) => {
            let Ok(Some(v)) = dict.get_item(f.as_ref()) else {
                return Ok(matches!(target, RustValue::Null));
            };
            Ok(match target {
                RustValue::Null => v.is_none(),
                RustValue::Bool(b) => v.extract::<bool>().map(|x| x == *b).unwrap_or(false),
                RustValue::Int(i) => v
                    .extract::<i64>()
                    .map(|x| x == *i)
                    .or_else(|_| v.extract::<f64>().map(|x| x == *i as f64))
                    .unwrap_or(false),
                RustValue::Float(f2) => v.extract::<f64>().map(|x| x == *f2).unwrap_or(false),
                RustValue::Str(s) => v
                    .extract::<&str>()
                    .map(|x| x == s.as_ref())
                    .unwrap_or(false),
            })
        }
        ObjOp::FilterFieldNe(f, target) => {
            let Ok(Some(v)) = dict.get_item(f.as_ref()) else {
                return Ok(!matches!(target, RustValue::Null));
            };
            Ok(match target {
                RustValue::Null => !v.is_none(),
                RustValue::Bool(b) => v.extract::<bool>().map(|x| x != *b).unwrap_or(true),
                RustValue::Int(i) => v
                    .extract::<i64>()
                    .map(|x| x != *i)
                    .or_else(|_| v.extract::<f64>().map(|x| x != *i as f64))
                    .unwrap_or(true),
                RustValue::Float(f2) => v.extract::<f64>().map(|x| x != *f2).unwrap_or(true),
                RustValue::Str(s) => v.extract::<&str>().map(|x| x != s.as_ref()).unwrap_or(true),
            })
        }
    }
}
