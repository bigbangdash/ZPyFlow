use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use std::collections::HashMap;

use crate::core::{ObjOp, RustValue};
use crate::python::schema::{infer_schema_inner, FieldDtype};

// ---------------------------------------------------------------------------
// Column storage types
// ---------------------------------------------------------------------------

/// Typed storage for a single column extracted from a list-of-dicts.
pub enum ColumnVec {
    F64(Vec<f64>),
    I64(Vec<i64>),
    Str(Vec<String>),
    PyObj(Vec<PyObject>),
}

/// A column with an associated null bitmap.
/// `nulls[i] == true` means row `i` was None or missing.
pub struct NullableColumn {
    pub data: ColumnVec,
    pub nulls: Vec<bool>,
}

/// All columns extracted from a list-of-dicts, with the original row count.
pub struct ColumnarData {
    pub columns: HashMap<String, NullableColumn>,
    pub len: usize,
}

// ---------------------------------------------------------------------------
// Value extraction helpers
// ---------------------------------------------------------------------------

fn extract_f64(val: &Bound<'_, PyAny>) -> f64 {
    if val.is_instance_of::<PyBool>() {
        return val.extract::<bool>().map(|b| b as i64 as f64).unwrap_or(f64::NAN);
    }
    if val.is_instance_of::<PyInt>() {
        return val.extract::<i64>().map(|i| i as f64).unwrap_or(f64::NAN);
    }
    if val.is_instance_of::<PyFloat>() {
        return val.extract::<f64>().unwrap_or(f64::NAN);
    }
    f64::NAN
}

fn extract_i64(val: &Bound<'_, PyAny>) -> i64 {
    if val.is_instance_of::<PyBool>() {
        return val.extract::<bool>().map(|b| b as i64).unwrap_or(i64::MIN);
    }
    val.extract::<i64>().unwrap_or(i64::MIN)
}

fn extract_str(val: &Bound<'_, PyAny>) -> String {
    if val.is_instance_of::<PyString>() {
        val.extract::<String>().unwrap_or_default()
    } else {
        String::new()
    }
}

// ---------------------------------------------------------------------------
// Core conversion kernel
// ---------------------------------------------------------------------------

/// Convert a Python list-of-dicts to columnar layout using a pre-inferred schema.
///
/// For each field in `schema`:
/// - `F64`   → `Vec<f64>`;  None/missing fills `f64::NAN`
/// - `I64`   → `Vec<i64>`;  None/missing fills `i64::MIN`
/// - `Str`   → `Vec<String>`; None/missing fills `""`
/// - `Mixed` → `Vec<PyObject>`; stores original Python objects (None included)
///
/// A parallel `Vec<bool>` tracks which rows are null (true = null).
pub fn convert_to_columnar(
    py: Python<'_>,
    data: &Bound<'_, PyList>,
    schema: &HashMap<String, FieldDtype>,
) -> PyResult<ColumnarData> {
    let n = data.len();

    let mut columns: HashMap<String, NullableColumn> = schema
        .iter()
        .map(|(name, dtype)| {
            let col = NullableColumn {
                data: match dtype {
                    FieldDtype::F64 => ColumnVec::F64(Vec::with_capacity(n)),
                    FieldDtype::I64 => ColumnVec::I64(Vec::with_capacity(n)),
                    FieldDtype::Str => ColumnVec::Str(Vec::with_capacity(n)),
                    FieldDtype::Mixed => ColumnVec::PyObj(Vec::with_capacity(n)),
                },
                nulls: Vec::with_capacity(n),
            };
            (name.clone(), col)
        })
        .collect();

    for i in 0..n {
        let item = data.get_item(i)?;
        let dict = item.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!("row {} is not a dict", i))
        })?;

        for (field_name, col) in columns.iter_mut() {
            match dict.get_item(field_name.as_str())? {
                None => {
                    // Missing field
                    col.nulls.push(true);
                    push_default(py, &mut col.data);
                }
                Some(val) if val.is_none() => {
                    // Explicit None
                    col.nulls.push(true);
                    push_default(py, &mut col.data);
                }
                Some(val) => {
                    col.nulls.push(false);
                    match &mut col.data {
                        ColumnVec::F64(v) => v.push(extract_f64(&val)),
                        ColumnVec::I64(v) => v.push(extract_i64(&val)),
                        ColumnVec::Str(v) => v.push(extract_str(&val)),
                        ColumnVec::PyObj(v) => v.push(val.unbind()),
                    }
                }
            }
        }
    }

    Ok(ColumnarData { columns, len: n })
}

fn push_default(py: Python<'_>, col: &mut ColumnVec) {
    match col {
        ColumnVec::F64(v) => v.push(f64::NAN),
        ColumnVec::I64(v) => v.push(i64::MIN),
        ColumnVec::Str(v) => v.push(String::new()),
        ColumnVec::PyObj(v) => v.push(py.None()),
    }
}

// ---------------------------------------------------------------------------
// Filter execution helpers — used by QueryInner::ColumnarObj
// ---------------------------------------------------------------------------

fn col_as_f64(data: &ColumnarData, field: &str, i: usize) -> Option<f64> {
    let col = data.columns.get(field)?;
    if col.nulls[i] { return None; }
    match &col.data {
        ColumnVec::F64(v) => Some(v[i]),
        ColumnVec::I64(v) => Some(v[i] as f64),
        _ => None,
    }
}

fn col_as_str<'a>(data: &'a ColumnarData, field: &str, i: usize) -> Option<&'a str> {
    let col = data.columns.get(field)?;
    if col.nulls[i] { return None; }
    match &col.data {
        ColumnVec::Str(v) => Some(v[i].as_str()),
        _ => None,
    }
}

fn col_matches_rust_value(
    data: &ColumnarData,
    field: &str,
    i: usize,
    target: &RustValue,
    eq: bool,
) -> bool {
    let matched = (|| -> Option<bool> {
        let col = data.columns.get(field)?;
        if col.nulls[i] {
            return Some(matches!(target, RustValue::Null));
        }
        let result = match (target, &col.data) {
            (RustValue::Null, _) => false,
            (RustValue::Float(f), ColumnVec::F64(v)) => v[i] == *f,
            (RustValue::Float(f), ColumnVec::I64(v)) => (v[i] as f64) == *f,
            (RustValue::Int(n), ColumnVec::I64(v)) => v[i] == *n,
            (RustValue::Int(n), ColumnVec::F64(v)) => v[i] == (*n as f64),
            (RustValue::Str(s), ColumnVec::Str(v)) => v[i].as_str() == s.as_ref(),
            (RustValue::Bool(b), ColumnVec::I64(v)) => (v[i] != 0) == *b,
            _ => false,
        };
        Some(result)
    })()
    .unwrap_or(false);
    if eq { matched } else { !matched }
}

fn eval_obj_op_at_row(data: &ColumnarData, op: &ObjOp, i: usize) -> bool {
    match op {
        ObjOp::FilterFieldGt(f, t)  => col_as_f64(data, f, i).map_or(false, |v| v > *t),
        ObjOp::FilterFieldGe(f, t)  => col_as_f64(data, f, i).map_or(false, |v| v >= *t),
        ObjOp::FilterFieldLt(f, t)  => col_as_f64(data, f, i).map_or(false, |v| v < *t),
        ObjOp::FilterFieldLe(f, t)  => col_as_f64(data, f, i).map_or(false, |v| v <= *t),
        ObjOp::FilterFieldBetween(f, lo, hi) => {
            col_as_f64(data, f, i).map_or(false, |v| v >= *lo && v <= *hi)
        }
        ObjOp::FilterFieldEq(f, target) => col_matches_rust_value(data, f, i, target, true),
        ObjOp::FilterFieldNe(f, target) => col_matches_rust_value(data, f, i, target, false),
        ObjOp::StrStartsWith(f, p) => {
            col_as_str(data, f, i).map_or(false, |s| s.starts_with(p.as_ref()))
        }
        ObjOp::StrEndsWith(f, p) => {
            col_as_str(data, f, i).map_or(false, |s| s.ends_with(p.as_ref()))
        }
        ObjOp::StrContains(f, p) => {
            col_as_str(data, f, i).map_or(false, |s| s.contains(p.as_ref()))
        }
        ObjOp::StrMatches(f, re) => col_as_str(data, f, i).map_or(false, |s| re.is_match(s)),
    }
}

/// Apply all filter ops and skip/take to columnar data; return surviving row indices.
///
/// All operations run on Rust-native column slices (no Python calls).
pub fn apply_columnar_ops(
    data: &ColumnarData,
    ops: &[ObjOp],
    skip: usize,
    take: Option<usize>,
) -> Vec<usize> {
    let mut out = Vec::new();
    let mut skipped = 0usize;
    'row: for i in 0..data.len {
        for op in ops {
            if !eval_obj_op_at_row(data, op, i) {
                continue 'row;
            }
        }
        if skipped < skip {
            skipped += 1;
            continue;
        }
        out.push(i);
        if take.is_some_and(|t| out.len() >= t) {
            break;
        }
    }
    out
}

/// Reconstruct Python dicts from a set of row indices.
pub fn columnar_indices_to_py_list(
    py: Python<'_>,
    data: &ColumnarData,
    indices: &[usize],
) -> PyResult<Vec<PyObject>> {
    let mut result = Vec::with_capacity(indices.len());
    for &i in indices {
        let row = PyDict::new_bound(py);
        for (name, col) in &data.columns {
            let val: PyObject = if col.nulls[i] {
                py.None()
            } else {
                match &col.data {
                    ColumnVec::F64(v) => v[i].into_py(py),
                    ColumnVec::I64(v) => v[i].into_py(py),
                    ColumnVec::Str(v) => v[i].as_str().into_py(py),
                    ColumnVec::PyObj(v) => v[i].clone_ref(py),
                }
            };
            row.set_item(name, val)?;
        }
        result.push(row.into_py(py));
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Python-exposed wrapper — for testing and introspection
// ---------------------------------------------------------------------------

/// Convert a list-of-dicts to a columnar representation (spec-082 T2).
///
/// Returns `dict[str, dict]` where each inner dict has:
///   `"dtype"` — `"f64"` | `"i64"` | `"str"` | `"mixed"`
///   `"data"`  — list of values (NaN/MIN/"" for null cells)
///   `"nulls"` — list of bools (True = null)
///
/// `sample_size` controls how many rows are used for schema inference (default 100).
#[pyfunction]
#[pyo3(signature = (data, sample_size=100))]
pub fn _convert_to_columnar(
    py: Python<'_>,
    data: &Bound<'_, PyList>,
    sample_size: usize,
) -> PyResult<PyObject> {
    let schema = infer_schema_inner(data, sample_size)?;
    let columnar = convert_to_columnar(py, data, &schema)?;

    let result = PyDict::new_bound(py);
    for (name, col) in &columnar.columns {
        let entry = PyDict::new_bound(py);
        let dtype_str = match &col.data {
            ColumnVec::F64(_) => "f64",
            ColumnVec::I64(_) => "i64",
            ColumnVec::Str(_) => "str",
            ColumnVec::PyObj(_) => "mixed",
        };
        entry.set_item("dtype", dtype_str)?;

        let data_list = PyList::empty_bound(py);
        match &col.data {
            ColumnVec::F64(v) => {
                for &x in v {
                    data_list.append(x)?;
                }
            }
            ColumnVec::I64(v) => {
                for &x in v {
                    data_list.append(x)?;
                }
            }
            ColumnVec::Str(v) => {
                for s in v {
                    data_list.append(s.as_str())?;
                }
            }
            ColumnVec::PyObj(v) => {
                for obj in v {
                    data_list.append(obj.bind(py))?;
                }
            }
        }
        entry.set_item("data", data_list)?;

        let nulls_list = PyList::empty_bound(py);
        for &is_null in &col.nulls {
            nulls_list.append(is_null)?;
        }
        entry.set_item("nulls", nulls_list)?;

        result.set_item(name, entry)?;
    }
    Ok(result.into())
}
