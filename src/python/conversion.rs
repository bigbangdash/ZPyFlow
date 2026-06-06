//! RustRow / RustValue ↔ Python dict conversion helpers.
//!
//! GIL must be held for every function in this module.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ahash::AHashMap;
use std::sync::Arc;

use super::expr::py_to_rust_value;
use crate::core::{RustRow, RustValue};

/// Python dict → RustRow. GIL must be held.
pub(super) fn py_dict_to_rust_row(dict: &Bound<'_, PyDict>) -> PyResult<RustRow> {
    let mut map = AHashMap::with_capacity(dict.len());
    for (k, v) in dict.iter() {
        let key: Arc<str> = Arc::from(k.extract::<&str>()?);
        let val = py_to_rust_value(&v)?;
        map.insert(key, val);
    }
    Ok(Arc::new(map))
}

/// RustRow → Python dict. GIL must be held.
pub(super) fn rust_row_to_py(py: Python<'_>, row: &RustRow) -> PyObject {
    let dict = PyDict::new_bound(py);
    for (k, v) in row.iter() {
        let _ = dict.set_item(k.as_ref(), rust_value_to_py(py, v));
    }
    dict.into()
}

/// RustValue → Python object. GIL must be held.
pub(super) fn rust_value_to_py(py: Python<'_>, v: &RustValue) -> PyObject {
    match v {
        RustValue::Null => py.None(),
        RustValue::Bool(b) => b.into_py(py),
        RustValue::Int(i) => i.into_py(py),
        RustValue::Float(f) => f.into_py(py),
        RustValue::Str(s) => s.as_ref().into_py(py),
    }
}

/// Convert a Python list-of-dicts to `Vec<RustRow>` (GIL held).
/// Returns `None` if the first item is not a dict (caller falls back to Obj path).
pub(crate) fn try_convert_to_rust_obj(
    py: Python<'_>,
    list: &Bound<'_, PyList>,
) -> PyResult<Option<Vec<RustRow>>> {
    if list.is_empty() {
        return Ok(Some(vec![]));
    }
    let first = list.get_item(0)?;
    if !first.is_instance_of::<PyDict>() {
        return Ok(None);
    }
    let mut rows = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyValueError::new_err("RustObj: all items must be dicts"))?;
        rows.push(py_dict_to_rust_row(dict)?);
    }
    let _ = py; // suppress unused warning
    Ok(Some(rows))
}
