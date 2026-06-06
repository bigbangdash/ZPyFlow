//! Hash join kernels — T3 fast path for string/integer field keys (spec 084).
//!
//! Python's `join()` routes here for inner joins when `on` is a string field name.
//! Keys are extracted as `JoinKey` (i64 or String), hashed in Rust's HashMap,
//! then probed. Dict merging is done with PyO3's `update()` — right wins on collision.

use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[derive(Hash, Eq, PartialEq)]
enum JoinKey {
    Int(i64),
    Str(String),
}

fn extract_join_key(row: &Bound<'_, PyAny>, field: &str) -> PyResult<Option<JoinKey>> {
    let Ok(d) = row.downcast::<PyDict>() else {
        return Ok(None);
    };
    let Some(val) = d.get_item(field)? else {
        return Ok(None);
    };
    if let Ok(i) = val.extract::<i64>() {
        return Ok(Some(JoinKey::Int(i)));
    }
    if let Ok(s) = val.extract::<String>() {
        return Ok(Some(JoinKey::Str(s)));
    }
    // Non-hashable key type — signal fallback to Python
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "join key must be int or str for the Rust fast path",
    ))
}

fn merge_dicts<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyAny>,
    right: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    if let Ok(ld) = left.downcast::<PyDict>() {
        out.update(ld.as_mapping())?;
    }
    if let Ok(rd) = right.downcast::<PyDict>() {
        out.update(rd.as_mapping())?;
    }
    Ok(out)
}

/// Inner hash join on a string field key — GIL held throughout.
///
/// Builds a right-side `HashMap<JoinKey, Vec<usize>>` (indices into *right*),
/// then probes left rows. Dict merging uses `PyDict::update` (right wins).
///
/// Returns `TypeError` if any key value is not int or str, prompting Python
/// to fall back to its own implementation.
#[pyfunction]
pub fn _hash_join_by_field<'py>(
    py: Python<'py>,
    left: &Bound<'py, PyList>,
    right: &Bound<'py, PyList>,
    field: &str,
) -> PyResult<Bound<'py, PyList>> {
    // Build right-side index: key → Vec<row index>
    let right_len = right.len();
    let mut index: HashMap<JoinKey, Vec<usize>> = HashMap::with_capacity(right_len);
    for i in 0..right_len {
        let row = right.get_item(i)?;
        if let Some(key) = extract_join_key(&row, field)? {
            index.entry(key).or_default().push(i);
        }
    }

    // Probe left side
    let result = PyList::empty_bound(py);
    for left_row in left {
        let Some(key) = extract_join_key(&left_row, field)? else {
            continue;
        };
        if let Some(indices) = index.get(&key) {
            for &i in indices {
                let right_row = right.get_item(i)?;
                let merged = merge_dicts(py, &left_row, &right_row)?;
                result.append(merged)?;
            }
        }
    }

    Ok(result)
}
