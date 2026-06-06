use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};
use std::collections::HashMap;

/// Inferred dtype for a single dict field.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldDtype {
    F64,   // all values are int or float (at least one float)
    I64,   // all values are int (including bool)
    Str,   // all values are str
    Mixed, // type conflict, None present, or unrecognised Python type
}

impl FieldDtype {
    fn merge(self, other: FieldDtype) -> FieldDtype {
        match (self, other) {
            (FieldDtype::I64, FieldDtype::I64) => FieldDtype::I64,
            (FieldDtype::I64, FieldDtype::F64) | (FieldDtype::F64, FieldDtype::I64) => {
                FieldDtype::F64
            }
            (FieldDtype::F64, FieldDtype::F64) => FieldDtype::F64,
            (FieldDtype::Str, FieldDtype::Str) => FieldDtype::Str,
            _ => FieldDtype::Mixed,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            FieldDtype::F64 => "f64",
            FieldDtype::I64 => "i64",
            FieldDtype::Str => "str",
            FieldDtype::Mixed => "mixed",
        }
    }
}

/// Classify a single Python value into a FieldDtype.
fn classify_value(val: &Bound<'_, PyAny>) -> FieldDtype {
    if val.is_none() {
        return FieldDtype::Mixed;
    }
    // PyBool is a subclass of PyInt — check bool first so we treat True/False as i64.
    if val.is_instance_of::<PyBool>() {
        return FieldDtype::I64;
    }
    if val.is_instance_of::<PyInt>() {
        return FieldDtype::I64;
    }
    if val.is_instance_of::<PyFloat>() {
        return FieldDtype::F64;
    }
    if val.is_instance_of::<PyString>() {
        return FieldDtype::Str;
    }
    FieldDtype::Mixed
}

/// Infer the schema of a Python list-of-dicts by sampling the first `sample_size` rows.
///
/// Returns a `HashMap<field_name, FieldDtype>`.
/// Only fields present in the first row are tracked; fields that first appear later
/// in the sample are ignored (they would need a Mixed fallback anyway).
///
/// Rules (from spec-082 T1):
/// - I64  — every sampled value is `int` (or `bool`)
/// - F64  — every sampled value is `int` or `float` (at least one `float`)
/// - Str  — every sampled value is `str`
/// - Mixed — any None, type conflict, or unrecognised type → Python fallback
pub fn infer_schema_inner(
    data: &Bound<'_, PyList>,
    sample_size: usize,
) -> PyResult<HashMap<String, FieldDtype>> {
    let n = data.len().min(sample_size);
    if n == 0 {
        return Ok(HashMap::new());
    }

    // Seed schema from the first row.
    let first = data.get_item(0)?;
    let first_dict = match first.downcast::<PyDict>() {
        Ok(d) => d,
        Err(_) => return Ok(HashMap::new()),
    };

    let mut schema: HashMap<String, FieldDtype> = HashMap::new();
    for (k, v) in first_dict.iter() {
        let key: String = k.extract()?;
        schema.insert(key, classify_value(&v));
    }

    // Merge subsequent rows.
    for i in 1..n {
        let item = data.get_item(i)?;
        let dict = match item.downcast::<PyDict>() {
            Ok(d) => d,
            Err(_) => {
                // Non-dict row: mark all fields as Mixed.
                for dtype in schema.values_mut() {
                    *dtype = FieldDtype::Mixed;
                }
                break;
            }
        };

        for (field_name, dtype) in schema.iter_mut() {
            if *dtype == FieldDtype::Mixed {
                continue; // already degraded, skip
            }
            match dict.get_item(field_name)? {
                Some(val) if val.is_none() => {
                    // None is compatible with any column type; will be stored as null
                    // via the nulls bitvec in convert_to_columnar — do not degrade.
                }
                Some(val) => {
                    let val_kind = classify_value(&val);
                    *dtype = dtype.clone().merge(val_kind);
                }
                None => {
                    // Field absent from this row → structurally variable schema → Mixed.
                    *dtype = FieldDtype::Mixed;
                }
            }
        }
    }

    Ok(schema)
}

/// Python-exposed wrapper: `_infer_schema(data, sample_size=100) -> dict[str, str]`
///
/// Returns `{"field": "f64" | "i64" | "str" | "mixed"}`.
/// Useful for testing and for `.preload()` (spec-082 T4).
#[pyfunction]
#[pyo3(signature = (data, sample_size=100))]
pub fn _infer_schema(
    py: Python<'_>,
    data: &Bound<'_, PyList>,
    sample_size: usize,
) -> PyResult<PyObject> {
    let schema = infer_schema_inner(data, sample_size)?;
    let result = PyDict::new_bound(py);
    for (k, v) in &schema {
        result.set_item(k, v.as_str())?;
    }
    Ok(result.into())
}
