/// GIL-free CSV and JSON Lines parsers.
///
/// All public functions take owned Rust data (`Vec<u8>`) and return owned results
/// so they can run inside `py.allow_threads()` without touching any Python objects.
use ahash::AHashMap;
use std::io::{BufRead, Cursor};
use std::sync::Arc;

use crate::pipeline::obj::{RustRow, RustValue};

// ---------------------------------------------------------------------------
// Parsed output — maps to a QueryInner variant after GIL is re-acquired
// ---------------------------------------------------------------------------

pub enum ParsedOutput {
    F64(Vec<f64>),
    I64(Vec<i64>),
    Strs(Vec<String>),
    Rows(Vec<RustRow>),
}

// ---------------------------------------------------------------------------
// Column specifier
// ---------------------------------------------------------------------------

pub enum ColSpec {
    Name(String),
    Index(usize),
}

// ---------------------------------------------------------------------------
// CSV parsing
// ---------------------------------------------------------------------------

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
            // "auto": int → float → str degradation
            let mut mode = AutoMode::new();
            for result in rdr.records() {
                let rec = result.map_err(|e| e.to_string())?;
                mode.push(rec.get(idx).unwrap_or(""));
            }
            Ok(mode.finish())
        }
    }
}

// ---------------------------------------------------------------------------
// JSON Lines parsing
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Value helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// AutoMode — streaming int→float→str type degradation
// ---------------------------------------------------------------------------

enum AutoMode {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Str(Vec<String>),
}

impl AutoMode {
    fn new() -> Self {
        AutoMode::Int(Vec::new())
    }

    fn explicit(dtype: &str) -> Self {
        match dtype {
            "float" => AutoMode::Float(Vec::new()),
            "int" => AutoMode::Int(Vec::new()),
            _ => AutoMode::Str(Vec::new()),
        }
    }

    fn push_i64(&mut self, i: i64) {
        match self {
            AutoMode::Int(v) => v.push(i),
            AutoMode::Float(v) => v.push(i as f64),
            AutoMode::Str(v) => v.push(i.to_string()),
        }
    }

    fn push_f64(&mut self, f: f64) {
        match self {
            AutoMode::Int(v) => {
                let mut fv: Vec<f64> = v.drain(..).map(|i| i as f64).collect();
                fv.push(f);
                *self = AutoMode::Float(fv);
            }
            AutoMode::Float(v) => v.push(f),
            AutoMode::Str(v) => v.push(f.to_string()),
        }
    }

    fn push_str_raw(&mut self, s: &str) {
        match self {
            AutoMode::Int(v) => {
                let mut sv: Vec<String> = v.drain(..).map(|i| i.to_string()).collect();
                sv.push(s.to_string());
                *self = AutoMode::Str(sv);
            }
            AutoMode::Float(v) => {
                let mut sv: Vec<String> = v.drain(..).map(|f| f.to_string()).collect();
                sv.push(s.to_string());
                *self = AutoMode::Str(sv);
            }
            AutoMode::Str(v) => v.push(s.to_string()),
        }
    }

    fn push(&mut self, s: &str) {
        let s = s.trim();
        match self {
            AutoMode::Int(_) => {
                if let Ok(i) = s.parse::<i64>() {
                    self.push_i64(i);
                } else if let Ok(f) = s.parse::<f64>() {
                    self.push_f64(f);
                } else {
                    self.push_str_raw(s);
                }
            }
            AutoMode::Float(_) => {
                if let Ok(f) = s.parse::<f64>() {
                    self.push_f64(f);
                } else {
                    self.push_str_raw(s);
                }
            }
            AutoMode::Str(v) => v.push(s.to_string()),
        }
    }

    fn finish(self) -> ParsedOutput {
        match self {
            AutoMode::Int(v) => ParsedOutput::I64(v),
            AutoMode::Float(v) => ParsedOutput::F64(v),
            AutoMode::Str(v) => ParsedOutput::Strs(v),
        }
    }

    /// For explicit dtypes, error if degraded to wrong type.
    fn into_result(self, dtype: &str) -> Result<ParsedOutput, String> {
        match (dtype, self) {
            ("float", AutoMode::Float(v)) => Ok(ParsedOutput::F64(v)),
            ("float", AutoMode::Int(v)) => {
                Ok(ParsedOutput::F64(v.into_iter().map(|i| i as f64).collect()))
            }
            ("float", AutoMode::Str(_)) => {
                Err("field contains non-numeric values; expected float".to_string())
            }
            ("int", AutoMode::Int(v)) => Ok(ParsedOutput::I64(v)),
            ("int", AutoMode::Float(_)) => {
                Err("field contains float values; expected int".to_string())
            }
            ("int", AutoMode::Str(_)) => {
                Err("field contains non-numeric values; expected int".to_string())
            }
            (_, mode) => Ok(mode.finish()),
        }
    }
}
