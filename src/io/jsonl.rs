use ahash::AHashMap;
use std::io::{BufRead, Cursor};
use std::sync::Arc;

use crate::pipeline::obj::{RustRow, RustValue};

use super::{AutoMode, ParsedOutput};

/// Parse a JSONL byte blob into all rows (dict-per-row).
pub fn parse_jsonl_rows(bytes: Vec<u8>) -> Result<Vec<RustRow>, String> {
    let cursor = Cursor::new(bytes);
    let mut rows = Vec::new();
    for line_res in cursor.lines() {
        let line = line_res.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let val: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("JSON parse error: {}", e))?;
        if let serde_json::Value::Object(map) = val {
            let mut row = AHashMap::with_capacity(map.len());
            for (k, v) in map {
                row.insert(Arc::from(k.as_str()), json_to_rust(v));
            }
            rows.push(Arc::new(row));
        }
    }
    Ok(rows)
}

/// Parse a JSONL byte blob, extracting a single field.
pub fn parse_jsonl_field(
    bytes: Vec<u8>,
    field: String,
    dtype: &str,
) -> Result<ParsedOutput, String> {
    let cursor = Cursor::new(bytes);
    let mut mode = match dtype {
        "float" | "int" | "str" => AutoMode::explicit(dtype),
        _ => AutoMode::new(),
    };
    for line_res in cursor.lines() {
        let line = line_res.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let val: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("JSON parse error: {}", e))?;
        let fv = val.get(&field).cloned().unwrap_or(serde_json::Value::Null);
        match fv {
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    mode.push_i64(i);
                } else if let Some(f) = n.as_f64() {
                    mode.push_f64(f);
                } else {
                    mode.push("");
                }
            }
            serde_json::Value::String(s) => mode.push(&s),
            serde_json::Value::Bool(b) => mode.push_f64(if b { 1.0 } else { 0.0 }),
            _ => mode.push(""),
        }
    }
    mode.into_result(dtype)
}

/// Convert a serde_json Value to a RustValue (shallow; arrays/objects → Null).
fn json_to_rust(v: serde_json::Value) -> RustValue {
    match v {
        serde_json::Value::Null => RustValue::Null,
        serde_json::Value::Bool(b) => RustValue::Bool(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                RustValue::Int(i)
            } else if let Some(f) = n.as_f64() {
                RustValue::Float(f)
            } else {
                RustValue::Null
            }
        }
        serde_json::Value::String(s) => RustValue::Str(Arc::from(s.as_str())),
        _ => RustValue::Null,
    }
}
