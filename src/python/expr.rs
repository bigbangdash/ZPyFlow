use pyo3::exceptions::{PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

use crate::core::NumericOp;
use crate::core::{ObjOp, RustValue};

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

    /// `field("name").startswith("prefix")`
    fn startswith(&self, prefix: &str) -> Self {
        PyFieldExpr {
            name: Arc::clone(&self.name),
            op: Some(ObjOp::StrStartsWith(Arc::clone(&self.name), Arc::from(prefix))),
        }
    }

    /// `field("name").endswith("suffix")`
    fn endswith(&self, suffix: &str) -> Self {
        PyFieldExpr {
            name: Arc::clone(&self.name),
            op: Some(ObjOp::StrEndsWith(Arc::clone(&self.name), Arc::from(suffix))),
        }
    }

    /// `field("name").contains("sub")`
    fn contains(&self, sub: &str) -> Self {
        PyFieldExpr {
            name: Arc::clone(&self.name),
            op: Some(ObjOp::StrContains(Arc::clone(&self.name), Arc::from(sub))),
        }
    }

    /// `field("name").matches(r"^\d+")`  — full regex match
    fn matches(&self, pattern: &str) -> PyResult<Self> {
        let re = regex::Regex::new(pattern)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyFieldExpr {
            name: Arc::clone(&self.name),
            op: Some(ObjOp::StrMatches(Arc::clone(&self.name), Arc::new(re))),
        })
    }
    fn __repr__(&self) -> String {
        format!("FieldExpr(\"{}\")", self.name)
    }

    /// Evaluate this field expression against a Python dict.
    /// - bare `field("x")`: extracts and returns the value at key "x"
    /// - `field("x") > 5`: returns bool (filter predicate)
    fn __call__(&self, py: Python<'_>, row: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let Some(op) = &self.op else {
            // bare field access — extract value, not a filter predicate
            let dict = row.downcast::<PyDict>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "FieldExpr('{}'): expected dict, got {}",
                    self.name,
                    row.get_type().name().unwrap_or_default()
                ))
            })?;
            return dict
                .get_item(self.name.as_ref())?
                .map(|v| v.unbind())
                .ok_or_else(|| PyKeyError::new_err(self.name.to_string()));
        };
        if let Ok(dict) = row.downcast::<PyDict>() {
            Ok(py_dict_satisfies_obj_predicate(dict, op)?.into_py(py))
        } else {
            Ok(false.into_py(py))
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
    SelfEq,    // col == col: passes when NOT NaN (arrow null-drop pattern)
    IsNan,     // only NaN passes
    IsFinite,  // only finite values pass
    IsInf,     // only ±infinity passes
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
    Log,
    Log2,
    Log10,
    Exp,
    Sigmoid,
    // Binary maps
    Clamp(f64, f64),
    Mod(f64),
    FloorDiv(f64),
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
            ExprOp::IsNan => NumericOp::FilterNan,
            ExprOp::IsFinite => NumericOp::FilterFinite,
            ExprOp::IsInf => NumericOp::FilterInf,
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
            ExprOp::Log => NumericOp::MapLog,
            ExprOp::Log2 => NumericOp::MapLog2,
            ExprOp::Log10 => NumericOp::MapLog10,
            ExprOp::Exp => NumericOp::MapExp,
            ExprOp::Sigmoid => NumericOp::MapSigmoid,
            ExprOp::Clamp(lo, hi) => NumericOp::MapClamp(*lo, *hi),
            ExprOp::Mod(s) => NumericOp::MapMod(*s),
            ExprOp::FloorDiv(s) => NumericOp::MapFloorDiv(*s),
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
            ExprOp::IsNan => val.is_nan().into_py(py),
            ExprOp::IsFinite => val.is_finite().into_py(py),
            ExprOp::IsInf => val.is_infinite().into_py(py),
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
            ExprOp::Log => val.ln().into_py(py),
            ExprOp::Log2 => val.log2().into_py(py),
            ExprOp::Log10 => val.log10().into_py(py),
            ExprOp::Exp => val.exp().into_py(py),
            ExprOp::Sigmoid => (1.0 / (1.0 + (-val).exp())).into_py(py),
            ExprOp::Clamp(lo, hi) => val.clamp(*lo, *hi).into_py(py),
            ExprOp::Mod(s) => (val % s).into_py(py),
            ExprOp::FloorDiv(s) => (val / s).floor().into_py(py),
        }
    }
}

// Helper macro: add scalar/unary methods to both PyExpr and PyColProxy
// (not used yet — methods are added manually for clarity)

impl PyExpr {
    fn unary(op: ExprOp) -> Self { PyExpr { op } }
}

#[pymethods]
impl PyExpr {
    fn __mod__(&self, other: f64) -> Self { PyExpr { op: ExprOp::Mod(other) } }
    fn __floordiv__(&self, other: f64) -> Self { PyExpr { op: ExprOp::FloorDiv(other) } }
    fn is_nan(&self) -> Self { PyExpr { op: ExprOp::IsNan } }
    fn not_nan(&self) -> Self { PyExpr { op: ExprOp::SelfEq } }
    fn is_finite(&self) -> Self { PyExpr { op: ExprOp::IsFinite } }
    fn is_inf(&self) -> Self { PyExpr { op: ExprOp::IsInf } }
    fn log(&self) -> Self { PyExpr::unary(ExprOp::Log) }
    fn log2(&self) -> Self { PyExpr::unary(ExprOp::Log2) }
    fn log10(&self) -> Self { PyExpr::unary(ExprOp::Log10) }
    fn exp(&self) -> Self { PyExpr::unary(ExprOp::Exp) }
    fn sigmoid(&self) -> Self { PyExpr::unary(ExprOp::Sigmoid) }
    fn clamp(&self, lo: f64, hi: f64) -> Self { PyExpr::unary(ExprOp::Clamp(lo, hi)) }
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
    fn reciprocal(&self) -> PyExpr { PyExpr { op: ExprOp::Reciprocal } }
    fn __mod__(&self, other: f64) -> PyExpr { PyExpr { op: ExprOp::Mod(other) } }
    fn __floordiv__(&self, other: f64) -> PyExpr { PyExpr { op: ExprOp::FloorDiv(other) } }
    fn is_nan(&self) -> PyExpr { PyExpr { op: ExprOp::IsNan } }
    fn not_nan(&self) -> PyExpr { PyExpr { op: ExprOp::SelfEq } }
    fn is_finite(&self) -> PyExpr { PyExpr { op: ExprOp::IsFinite } }
    fn is_inf(&self) -> PyExpr { PyExpr { op: ExprOp::IsInf } }
    fn log(&self) -> PyExpr { PyExpr { op: ExprOp::Log } }
    fn log2(&self) -> PyExpr { PyExpr { op: ExprOp::Log2 } }
    fn log10(&self) -> PyExpr { PyExpr { op: ExprOp::Log10 } }
    fn exp(&self) -> PyExpr { PyExpr { op: ExprOp::Exp } }
    fn sigmoid(&self) -> PyExpr { PyExpr { op: ExprOp::Sigmoid } }
    fn clamp(&self, lo: f64, hi: f64) -> PyExpr { PyExpr { op: ExprOp::Clamp(lo, hi) } }
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
fn py_dict_satisfies_obj_predicate(dict: &Bound<'_, PyDict>, op: &ObjOp) -> PyResult<bool> {
    #[inline]
    fn extract_f64_from_dict(dict: &Bound<'_, PyDict>, field: &str) -> f64 {
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
        ObjOp::FilterFieldGt(f, t) => Ok(extract_f64_from_dict(dict, f) > *t),
        ObjOp::FilterFieldGe(f, t) => Ok(extract_f64_from_dict(dict, f) >= *t),
        ObjOp::FilterFieldLt(f, t) => Ok(extract_f64_from_dict(dict, f) < *t),
        ObjOp::FilterFieldLe(f, t) => Ok(extract_f64_from_dict(dict, f) <= *t),
        ObjOp::FilterFieldBetween(f, lo, hi) => {
            let v = extract_f64_from_dict(dict, f);
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
        ObjOp::StrStartsWith(f, prefix) => Ok(dict
            .get_item(f.as_ref())?
            .and_then(|v| v.extract::<String>().ok())
            .map_or(false, |s| s.starts_with(prefix.as_ref()))),
        ObjOp::StrEndsWith(f, suffix) => Ok(dict
            .get_item(f.as_ref())?
            .and_then(|v| v.extract::<String>().ok())
            .map_or(false, |s| s.ends_with(suffix.as_ref()))),
        ObjOp::StrContains(f, sub) => Ok(dict
            .get_item(f.as_ref())?
            .and_then(|v| v.extract::<String>().ok())
            .map_or(false, |s| s.contains(sub.as_ref()))),
        ObjOp::StrMatches(f, re) => Ok(dict
            .get_item(f.as_ref())?
            .and_then(|v| v.extract::<String>().ok())
            .map_or(false, |s| re.is_match(&s))),
    }
}
