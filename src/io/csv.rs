use ahash::AHashMap;
use std::io::Cursor;
use std::sync::Arc;

use crate::core::{RustRow, RustValue};

use super::{AutoMode, ColSpec, ParsedOutput};

/// Parse a CSV byte blob into all rows (dict-per-row).
pub fn parse_csv_rows(
    bytes: Vec<u8>,
    delimiter: u8,
    has_header: bool,
) -> Result<Vec<RustRow>, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(has_header)
        .from_reader(Cursor::new(bytes));

    let mut rows = Vec::new();

    if has_header {
        let header_keys: Vec<Arc<str>> = rdr
            .headers()
            .map_err(|e| e.to_string())?
            .iter()
            .map(|h| Arc::from(h))
            .collect();
        for result in rdr.records() {
            let rec = result.map_err(|e| e.to_string())?;
            let mut map = AHashMap::with_capacity(header_keys.len());
            for (k, v) in header_keys.iter().zip(rec.iter()) {
                map.insert(Arc::clone(k), csv_auto(v));
            }
            rows.push(Arc::new(map));
        }
    } else {
        for result in rdr.records() {
            let rec = result.map_err(|e| e.to_string())?;
            let mut map = AHashMap::with_capacity(rec.len());
            for (i, v) in rec.iter().enumerate() {
                let k: Arc<str> = Arc::from(i.to_string().as_str());
                map.insert(k, csv_auto(v));
            }
            rows.push(Arc::new(map));
        }
    }
    Ok(rows)
}

/// Parse a CSV byte blob, extracting a single column.
pub fn parse_csv_column(
    bytes: Vec<u8>,
    col: ColSpec,
    dtype: &str,
    delimiter: u8,
    has_header: bool,
) -> Result<ParsedOutput, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(has_header)
        .from_reader(Cursor::new(bytes));

    let idx = match &col {
        ColSpec::Index(i) => *i,
        ColSpec::Name(name) => {
            let hdrs = rdr.headers().map_err(|e| e.to_string())?;
            hdrs.iter()
                .position(|h| h == name.as_str())
                .ok_or_else(|| format!("CSV column '{}' not found", name))?
        }
    };

    match dtype {
        "float" => {
            let mut out = Vec::new();
            for result in rdr.records() {
                let rec = result.map_err(|e| e.to_string())?;
                let s = rec.get(idx).unwrap_or("").trim();
                let v: f64 = s
                    .parse()
                    .map_err(|_| format!("cannot parse '{}' as float", s))?;
                out.push(v);
            }
            Ok(ParsedOutput::F64(out))
        }
        "int" => {
            let mut out = Vec::new();
            for result in rdr.records() {
                let rec = result.map_err(|e| e.to_string())?;
                let s = rec.get(idx).unwrap_or("").trim();
                let v: i64 = s
                    .parse()
                    .map_err(|_| format!("cannot parse '{}' as int", s))?;
                out.push(v);
            }
            Ok(ParsedOutput::I64(out))
        }
        "str" => {
            let mut out = Vec::new();
            for result in rdr.records() {
                let rec = result.map_err(|e| e.to_string())?;
                out.push(rec.get(idx).unwrap_or("").to_string());
            }
            Ok(ParsedOutput::Strs(out))
        }
        _ => {
            let mut mode = AutoMode::new();
            for result in rdr.records() {
                let rec = result.map_err(|e| e.to_string())?;
                mode.push(rec.get(idx).unwrap_or(""));
            }
            Ok(mode.finish())
        }
    }
}

/// Auto-detect a CSV string cell: int → float → string.
fn csv_auto(s: &str) -> RustValue {
    let s = s.trim();
    if let Ok(i) = s.parse::<i64>() {
        return RustValue::Int(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return RustValue::Float(f);
    }
    if s.eq_ignore_ascii_case("true") {
        return RustValue::Bool(true);
    }
    if s.eq_ignore_ascii_case("false") {
        return RustValue::Bool(false);
    }
    RustValue::Str(Arc::from(s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::RustValue;

    fn csv(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    // ── parse_csv_rows ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_rows_basic() {
        let rows = parse_csv_rows(csv("a,b\n1,2\n3,4"), b',', true).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["a"], RustValue::Int(1));
        assert_eq!(rows[0]["b"], RustValue::Int(2));
        assert_eq!(rows[1]["a"], RustValue::Int(3));
    }

    #[test]
    fn test_parse_rows_no_header() {
        let rows = parse_csv_rows(csv("1,2\n3,4"), b',', false).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["0"], RustValue::Int(1));
        assert_eq!(rows[0]["1"], RustValue::Int(2));
    }

    #[test]
    fn test_parse_rows_custom_delimiter() {
        let rows = parse_csv_rows(csv("a;b\n1;2"), b';', true).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["a"], RustValue::Int(1));
    }

    #[test]
    fn test_parse_rows_float_values() {
        let rows = parse_csv_rows(csv("x\n1.5\n2.7"), b',', true).unwrap();
        match &rows[0]["x"] {
            RustValue::Float(f) => assert!((f - 1.5).abs() < 1e-10),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_rows_empty_input() {
        let rows = parse_csv_rows(csv("a,b\n"), b',', true).unwrap();
        assert_eq!(rows.len(), 0);
    }

    // ── parse_csv_column ──────────────────────────────────────────────────────

    #[test]
    fn test_parse_column_by_name_float() {
        let result = parse_csv_column(
            csv("x,y\n1.0,2.0\n3.0,4.0"),
            ColSpec::Name("x".to_string()),
            "float",
            b',',
            true,
        )
        .unwrap();
        match result {
            ParsedOutput::F64(v) => assert_eq!(v, vec![1.0, 3.0]),
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn test_parse_column_by_index() {
        let result = parse_csv_column(
            csv("a,b\n10,20\n30,40"),
            ColSpec::Index(1),
            "int",
            b',',
            true,
        )
        .unwrap();
        match result {
            ParsedOutput::I64(v) => assert_eq!(v, vec![20, 40]),
            _ => panic!("expected I64"),
        }
    }

    #[test]
    fn test_parse_column_unknown_name_errors() {
        let result = parse_csv_column(
            csv("x,y\n1,2"),
            ColSpec::Name("z".to_string()),
            "auto",
            b',',
            true,
        );
        assert!(result.is_err());
    }
}
