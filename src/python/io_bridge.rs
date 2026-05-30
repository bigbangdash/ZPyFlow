//! CSV / JSONL bridge helpers — thin wrappers around `crate::io`.
//!
//! These functions are called from `PyQuery` static methods in `query.rs`.
//! `parsed_to_query` is intentionally kept in `query.rs` to avoid a circular
//! dependency on `QueryInner` / `PyQuery`.

use crate::io::{
    parse_csv_column, parse_csv_rows, parse_jsonl_field, parse_jsonl_rows, ColSpec, ParsedOutput,
};

pub(super) fn csv_col_spec(name: Option<String>, idx: Option<usize>) -> Option<ColSpec> {
    match (name, idx) {
        (Some(n), _) => Some(ColSpec::Name(n)),
        (_, Some(i)) => Some(ColSpec::Index(i)),
        _ => None,
    }
}

pub(super) fn parse_csv(
    bytes: Vec<u8>,
    col: Option<ColSpec>,
    dtype: &str,
    delim: u8,
    has_header: bool,
) -> Result<ParsedOutput, String> {
    match col {
        None => parse_csv_rows(bytes, delim, has_header).map(ParsedOutput::Rows),
        Some(col_spec) => parse_csv_column(bytes, col_spec, dtype, delim, has_header),
    }
}

pub(super) fn parse_jsonl(
    bytes: Vec<u8>,
    field: Option<String>,
    dtype: &str,
) -> Result<ParsedOutput, String> {
    match field {
        None => parse_jsonl_rows(bytes).map(ParsedOutput::Rows),
        Some(f) => parse_jsonl_field(bytes, f, dtype),
    }
}
