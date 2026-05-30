/// Rust-native object pipeline — GIL-free filter/count/sum over dict data.
///
/// Design:
///   Import (GIL)     : Python dict  →  Arc<RustRow>  (one bulk conversion)
///   Execute (no GIL) : filter/count/sum run on Arc<RustRow> without Python
///   Export (GIL)     : Arc<RustRow> →  Python dict  (one bulk conversion)
use ahash::AHashMap;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Value type
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum RustValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Arc<str>),
}

impl RustValue {
    pub fn as_f64(&self) -> f64 {
        match self {
            RustValue::Float(f) => *f,
            RustValue::Int(i) => *i as f64,
            RustValue::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
            _ => f64::NAN,
        }
    }
}

/// A row is a shared, immutable map of field name → value.
/// Arc means matching rows can be collected with O(1) clone per row.
/// AHashMap uses a non-cryptographic hash (ahash) — 3–5× faster than SipHash.
pub type RustRow = Arc<AHashMap<Arc<str>, RustValue>>;

// ---------------------------------------------------------------------------
// Operation type
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum ObjOp {
    FilterFieldGt(Arc<str>, f64),
    FilterFieldGe(Arc<str>, f64),
    FilterFieldLt(Arc<str>, f64),
    FilterFieldLe(Arc<str>, f64),
    FilterFieldEq(Arc<str>, RustValue),
    FilterFieldNe(Arc<str>, RustValue),
    FilterFieldBetween(Arc<str>, f64, f64),
}

// ---------------------------------------------------------------------------
// Execution kernels  (call inside py.allow_threads)
// ---------------------------------------------------------------------------

/// Collect matching rows. O(N * ops) time, O(matching) space (Arc clones only).
pub fn execute_obj_pipeline(
    data: &[RustRow],
    ops: &[ObjOp],
    skip: usize,
    take: Option<usize>,
) -> Vec<RustRow> {
    let cap = take.unwrap_or(data.len()).min(data.len());
    let mut out = Vec::with_capacity(cap);
    for_each_matching_row(data, ops, skip, take, |row| {
        out.push(Arc::clone(row));
    });
    out
}

/// Count matching rows without collecting them.
pub fn count_obj_pipeline(
    data: &[RustRow],
    ops: &[ObjOp],
    skip: usize,
    take: Option<usize>,
) -> usize {
    let mut count = 0usize;
    for_each_matching_row(data, ops, skip, take, |_| {
        count += 1;
    });
    count
}

/// Sum a numeric field over matching rows.
pub fn sum_field_obj_pipeline(
    data: &[RustRow],
    field: &str,
    ops: &[ObjOp],
    skip: usize,
    take: Option<usize>,
) -> f64 {
    let mut acc = 0.0f64;
    for_each_matching_row(data, ops, skip, take, |row| {
        if let Some(v) = row.get(field) {
            acc += v.as_f64();
        }
    });
    acc
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

pub fn row_passes(row: &RustRow, op: &ObjOp) -> bool {
    match op {
        ObjOp::FilterFieldGt(f, t) => cmp_num(row, f, |v| v > *t),
        ObjOp::FilterFieldGe(f, t) => cmp_num(row, f, |v| v >= *t),
        ObjOp::FilterFieldLt(f, t) => cmp_num(row, f, |v| v < *t),
        ObjOp::FilterFieldLe(f, t) => cmp_num(row, f, |v| v <= *t),
        ObjOp::FilterFieldEq(f, target) => row.get(f.as_ref()).map_or(false, |v| v == target),
        ObjOp::FilterFieldNe(f, target) => row.get(f.as_ref()).map_or(false, |v| v != target),
        ObjOp::FilterFieldBetween(f, lo, hi) => cmp_num(row, f, |v| v >= *lo && v <= *hi),
    }
}

fn for_each_matching_row(
    data: &[RustRow],
    ops: &[ObjOp],
    skip: usize,
    take: Option<usize>,
    mut visit: impl FnMut(&RustRow),
) {
    let mut skipped = 0usize;
    let mut emitted = 0usize;

    'outer: for row in data {
        for op in ops {
            if !row_passes(row, op) {
                continue 'outer;
            }
        }
        if skipped < skip {
            skipped += 1;
            continue;
        }
        visit(row);
        emitted += 1;
        if take.is_some_and(|n| emitted >= n) {
            break;
        }
    }
}

fn cmp_num(row: &RustRow, field: &str, cmp: impl Fn(f64) -> bool) -> bool {
    row.get(field).map_or(false, |v| cmp(v.as_f64()))
}

// ---------------------------------------------------------------------------
// Pure-Rust unit tests — runnable under Miri (no PyO3 FFI)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ahash::AHashMap;

    fn row(fields: &[(&str, RustValue)]) -> RustRow {
        let mut map = AHashMap::new();
        for (k, v) in fields {
            map.insert(Arc::from(*k), v.clone());
        }
        Arc::new(map)
    }

    fn f(v: f64) -> RustValue {
        RustValue::Float(v)
    }
    fn i(v: i64) -> RustValue {
        RustValue::Int(v)
    }
    fn s(v: &str) -> RustValue {
        RustValue::Str(Arc::from(v))
    }

    fn make_rows() -> Vec<RustRow> {
        vec![
            row(&[("score", f(10.0)), ("tag", s("a"))]),
            row(&[("score", f(20.0)), ("tag", s("b"))]),
            row(&[("score", f(30.0)), ("tag", s("a"))]),
            row(&[("score", f(40.0)), ("tag", s("c"))]),
        ]
    }

    #[test]
    fn filter_field_gt() {
        let data = make_rows();
        let ops = vec![ObjOp::FilterFieldGt(Arc::from("score"), 20.0)];
        let result = execute_obj_pipeline(&data, &ops, 0, None);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].get("score").unwrap().as_f64(), 30.0);
        assert_eq!(result[1].get("score").unwrap().as_f64(), 40.0);
    }

    #[test]
    fn filter_field_ge() {
        let data = make_rows();
        let ops = vec![ObjOp::FilterFieldGe(Arc::from("score"), 20.0)];
        let result = execute_obj_pipeline(&data, &ops, 0, None);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn filter_field_lt() {
        let data = make_rows();
        let ops = vec![ObjOp::FilterFieldLt(Arc::from("score"), 20.0)];
        let result = execute_obj_pipeline(&data, &ops, 0, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("score").unwrap().as_f64(), 10.0);
    }

    #[test]
    fn filter_field_le() {
        let data = make_rows();
        let ops = vec![ObjOp::FilterFieldLe(Arc::from("score"), 20.0)];
        let result = execute_obj_pipeline(&data, &ops, 0, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn filter_field_eq_float() {
        let data = make_rows();
        let ops = vec![ObjOp::FilterFieldEq(Arc::from("score"), f(20.0))];
        let result = execute_obj_pipeline(&data, &ops, 0, None);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].get("score").unwrap().as_f64(), 20.0);
    }

    #[test]
    fn filter_field_eq_str() {
        let data = make_rows();
        let ops = vec![ObjOp::FilterFieldEq(Arc::from("tag"), s("a"))];
        let result = execute_obj_pipeline(&data, &ops, 0, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn filter_field_ne() {
        let data = make_rows();
        let ops = vec![ObjOp::FilterFieldNe(Arc::from("tag"), s("a"))];
        let result = execute_obj_pipeline(&data, &ops, 0, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn filter_field_between() {
        let data = make_rows();
        let ops = vec![ObjOp::FilterFieldBetween(Arc::from("score"), 15.0, 35.0)];
        let result = execute_obj_pipeline(&data, &ops, 0, None);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].get("score").unwrap().as_f64(), 20.0);
        assert_eq!(result[1].get("score").unwrap().as_f64(), 30.0);
    }

    #[test]
    fn skip_and_take() {
        let data = make_rows();
        let ops = vec![];
        let result = execute_obj_pipeline(&data, &ops, 1, Some(2));
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].get("score").unwrap().as_f64(), 20.0);
        assert_eq!(result[1].get("score").unwrap().as_f64(), 30.0);
    }

    #[test]
    fn count_pipeline() {
        let data = make_rows();
        let ops = vec![ObjOp::FilterFieldGt(Arc::from("score"), 15.0)];
        let count = count_obj_pipeline(&data, &ops, 0, None);
        assert_eq!(count, 3);
    }

    #[test]
    fn sum_field_pipeline() {
        let data = make_rows();
        let ops = vec![];
        let total = sum_field_obj_pipeline(&data, "score", &ops, 0, None);
        assert!((total - 100.0).abs() < 1e-9);
    }

    #[test]
    fn sum_field_with_filter() {
        let data = make_rows();
        let ops = vec![ObjOp::FilterFieldGt(Arc::from("score"), 15.0)];
        let total = sum_field_obj_pipeline(&data, "score", &ops, 0, None);
        assert!((total - 90.0).abs() < 1e-9);
    }

    #[test]
    fn rust_value_as_f64_conversions() {
        assert_eq!(RustValue::Float(3.14).as_f64(), 3.14);
        assert_eq!(RustValue::Int(5).as_f64(), 5.0);
        assert_eq!(RustValue::Bool(true).as_f64(), 1.0);
        assert_eq!(RustValue::Bool(false).as_f64(), 0.0);
        assert!(RustValue::Null.as_f64().is_nan());
        assert!(RustValue::Str(Arc::from("x")).as_f64().is_nan());
    }

    #[test]
    fn row_passes_missing_field() {
        let r = row(&[("score", f(5.0))]);
        // Field "age" doesn't exist → cmp_num returns false → row does NOT pass
        assert!(!row_passes(
            &r,
            &ObjOp::FilterFieldGt(Arc::from("age"), 0.0)
        ));
    }

    #[test]
    fn int_value_as_f64() {
        let data = vec![row(&[("x", i(7))])];
        let ops = vec![ObjOp::FilterFieldGt(Arc::from("x"), 5.0)];
        let result = execute_obj_pipeline(&data, &ops, 0, None);
        assert_eq!(result.len(), 1);
    }
}
