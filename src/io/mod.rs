/// GIL-free CSV and JSON Lines parsers.
///
/// All public functions take owned Rust data (`Vec<u8>`) and return owned results
/// so they can run inside `py.allow_threads()` without touching any Python objects.

pub mod csv;
pub mod jsonl;

pub use csv::{parse_csv_column, parse_csv_rows};
pub use jsonl::{parse_jsonl_field, parse_jsonl_rows};

use crate::pipeline::obj::RustRow;

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
// AutoMode — streaming int→float→str type degradation (shared by CSV + JSONL)
// ---------------------------------------------------------------------------

pub(super) enum AutoMode {
    Int(Vec<i64>),
    Float(Vec<f64>),
    Str(Vec<String>),
}

impl AutoMode {
    pub(super) fn new() -> Self {
        AutoMode::Int(Vec::new())
    }

    pub(super) fn explicit(dtype: &str) -> Self {
        match dtype {
            "float" => AutoMode::Float(Vec::new()),
            "int" => AutoMode::Int(Vec::new()),
            _ => AutoMode::Str(Vec::new()),
        }
    }

    pub(super) fn push_i64(&mut self, i: i64) {
        match self {
            AutoMode::Int(v) => v.push(i),
            AutoMode::Float(v) => v.push(i as f64),
            AutoMode::Str(v) => v.push(i.to_string()),
        }
    }

    pub(super) fn push_f64(&mut self, f: f64) {
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

    pub(super) fn push(&mut self, s: &str) {
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

    pub(super) fn finish(self) -> ParsedOutput {
        match self {
            AutoMode::Int(v) => ParsedOutput::I64(v),
            AutoMode::Float(v) => ParsedOutput::F64(v),
            AutoMode::Str(v) => ParsedOutput::Strs(v),
        }
    }

    pub(super) fn into_result(self, dtype: &str) -> Result<ParsedOutput, String> {
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
