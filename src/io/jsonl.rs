use ahash::AHashMap;
use std::io::{BufRead, Cursor};
use std::sync::Arc;

use crate::core::{RustRow, RustValue};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::RustValue;

    fn jsonl(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    // ── parse_jsonl_rows ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_rows_basic() {
        let rows = parse_jsonl_rows(jsonl("{\"a\":1,\"b\":2}\n{\"a\":3,\"b\":4}")).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["a"], RustValue::Int(1));
        assert_eq!(rows[1]["a"], RustValue::Int(3));
    }

    #[test]
    fn test_parse_rows_float_field() {
        let rows = parse_jsonl_rows(jsonl("{\"x\":1.5}")).unwrap();
        match &rows[0]["x"] {
            RustValue::Float(f) => assert!((f - 1.5).abs() < 1e-10),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_rows_skips_blank_lines() {
        let rows = parse_jsonl_rows(jsonl("{\"a\":1}\n\n{\"a\":2}")).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn test_parse_rows_empty_input() {
        let rows = parse_jsonl_rows(jsonl("")).unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn test_parse_rows_invalid_json_errors() {
        let result = parse_jsonl_rows(jsonl("not json"));
        assert!(result.is_err());
    }

    // ── parse_jsonl_field ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_field_float() {
        let result = parse_jsonl_field(
            jsonl("{\"score\":0.9}\n{\"score\":0.5}"),
            "score".to_string(),
            "float",
        )
        .unwrap();
        match result {
            ParsedOutput::F64(v) => {
                assert!((v[0] - 0.9).abs() < 1e-10);
                assert!((v[1] - 0.5).abs() < 1e-10);
            }
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_parse_field_int() {
        let result = parse_jsonl_field(
            jsonl("{\"n\":10}\n{\"n\":20}"),
            "n".to_string(),
            "int",
        )
        .unwrap();
        match result {
            ParsedOutput::I64(v) => assert_eq!(v, vec![10, 20]),
            _ => panic!("expected I64"),
        }
    }

    #[test]
    fn test_parse_field_missing_key_treated_as_null() {
        // missing field → RustValue::Null → parsed as Str (auto-mode fallback)
        let result = parse_jsonl_field(
            jsonl("{\"other\":1}"),
            "score".to_string(),
            "auto",
        );
        assert!(result.is_ok());
    }
}
