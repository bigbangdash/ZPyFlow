use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use super::conversion::to_py;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;

use crate::core::{
    count_fused_f32_with_skip_take, count_fused_f64_with_skip_take, eval_filter_f64, execute_fused_f32_with_skip_take,
    execute_fused_f64_with_skip_take, filter_mean_fused_f64, filter_multi_stat_f64, filter_sum_fused_f64,
    ObjOp, RustValue, NumericOp,
};
use crate::core::numeric::pipeline::{apply_scalar_op, ScalarResult};

// ---------------------------------------------------------------------------
// abi3-compatible buffer protocol — wraps PyObject_GetBuffer / PyBuffer_Release.
// These functions are in Py_LIMITED_API (stable ABI) since Python 3.0.
// pyo3::buffer::PyBuffer is gated behind #[cfg(not(Py_LIMITED_API))] in pyo3 0.21,
// so we declare the necessary C types and externs directly.
// ---------------------------------------------------------------------------

/// Minimal layout of CPython's `Py_buffer` struct.
/// We only access `buf`, `len`, `itemsize`, and `ndim`; remaining fields are
/// opaque to us and must still be present so `PyBuffer_Release` can clean up.
#[repr(C)]
struct PyBufView {
    buf: *mut std::ffi::c_void,
    obj: *mut std::ffi::c_void, // Py<PyAny>* — we never dereference
    len: isize,
    itemsize: isize,
    readonly: std::ffi::c_int,
    ndim: std::ffi::c_int,
    format: *mut std::ffi::c_char,
    shape: *mut isize,
    strides: *mut isize,
    suboffsets: *mut isize,
    internal: *mut std::ffi::c_void,
}

/// `PyBUF_C_CONTIGUOUS | PyBUF_FORMAT` — request a C-contiguous view with
/// ndim/shape/strides and the struct-module format string filled in.
/// Computed as: `0x0020 | 0x0010 | 0x0008 | 0x0004`
/// (PyBUF_C_CONTIGUOUS | PyBUF_STRIDES | PyBUF_ND | PyBUF_FORMAT).
const PY_BUF_C_CONTIGUOUS: std::ffi::c_int = 0x003C;

/// Element types `RawBuffer` may be read as, keyed by struct-module format code.
/// `item_count::<T>()` rejects buffers whose format or itemsize doesn't match,
/// so a 7-byte `bytes` can never be reinterpreted as seven i64s.
pub(super) trait BufElem {
    const FORMATS: &'static [u8];
    const NAME: &'static str;
}
impl BufElem for f64 {
    const FORMATS: &'static [u8] = b"d";
    const NAME: &'static str = "float64";
}
impl BufElem for f32 {
    const FORMATS: &'static [u8] = b"f";
    const NAME: &'static str = "float32";
}
impl BufElem for i64 {
    // numpy int64 exports 'l' on LP64; Arrow/memoryview casts use 'q'.
    // The itemsize check rejects 'l' where long is 4 bytes (Windows).
    const FORMATS: &'static [u8] = b"qln";
    const NAME: &'static str = "int64";
}
impl BufElem for u8 {
    const FORMATS: &'static [u8] = b"B";
    const NAME: &'static str = "uint8";
}

extern "C" {
    fn PyObject_GetBuffer(
        obj: *mut pyo3::ffi::PyObject,
        view: *mut PyBufView,
        flags: std::ffi::c_int,
    ) -> std::ffi::c_int;
    fn PyBuffer_Release(view: *mut PyBufView);
}

/// RAII wrapper around a C-contiguous `PyBufView`.
/// Calls `PyBuffer_Release` on drop, keeping the buffer lock until we're done.
pub(super) struct RawBuffer {
    view: PyBufView,
}

impl RawBuffer {
    /// Acquire a C-contiguous read-only view on `obj`.
    /// Returns a Python error (BufferError) if the object does not implement the
    /// buffer protocol or is not C-contiguous (e.g. a Fortran-order numpy array).
    ///
    /// # Safety
    /// - `obj` must be a live Python object (GIL held).
    pub(super) unsafe fn get(py: Python<'_>, obj: *mut pyo3::ffi::PyObject) -> PyResult<Self> {
        let mut view = std::mem::MaybeUninit::<PyBufView>::uninit();
        let ret = PyObject_GetBuffer(obj, view.as_mut_ptr(), PY_BUF_C_CONTIGUOUS);
        if ret != 0 {
            return Err(pyo3::PyErr::fetch(py));
        }
        Ok(RawBuffer {
            view: view.assume_init(),
        })
    }

    /// Number of `T` elements (len_bytes / itemsize), after checking that the
    /// buffer's format code and itemsize actually describe `T`.
    pub(super) fn item_count<T: BufElem>(&self) -> PyResult<usize> {
        // A NULL format means unsigned bytes ("B") per the buffer protocol.
        let fmt: &[u8] = if self.view.format.is_null() {
            b"B"
        } else {
            unsafe { std::ffi::CStr::from_ptr(self.view.format) }.to_bytes()
        };
        // Accept native byte order markers; '<' only on little-endian hosts.
        let code = match fmt {
            [b'@' | b'=', c] => Some(*c),
            [b'<', c] if cfg!(target_endian = "little") => Some(*c),
            [c] => Some(*c),
            _ => None,
        };
        let matches = code.is_some_and(|c| T::FORMATS.contains(&c))
            && self.view.itemsize as usize == std::mem::size_of::<T>();
        if !matches {
            return Err(PyValueError::new_err(format!(
                "expected a {} buffer, got format '{}' with itemsize {}",
                T::NAME,
                String::from_utf8_lossy(fmt),
                self.view.itemsize
            )));
        }
        Ok((self.view.len as usize) / std::mem::size_of::<T>())
    }

    /// Number of dimensions (1 for a 1-D array).
    pub(super) fn ndim(&self) -> usize {
        self.view.ndim as usize
    }

    /// Raw pointer to the first element, cast to `*const T`.
    pub(super) fn buf_ptr<T>(&self) -> *const T {
        self.view.buf as *const T
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        unsafe { PyBuffer_Release(&mut self.view) };
    }
}

// ---------------------------------------------------------------------------
// LazyFloatList helpers — safe CPython float extraction
//
// We hold a Py reference to the list (refcount > 0) so neither the list nor
// its elements can be freed while we read them.  Python floats are immutable,
// so ob_fval never changes after allocation.
// ---------------------------------------------------------------------------

/// Extract f64 from a Python object, returning NaN for non-float elements.
///
/// Calls `PyFloat_AsDouble`; for non-float objects (e.g. None) it returns -1.0
/// and sets a TypeError. We call `PyErr_Clear()` both before (defensive) and
/// after detecting the error, ensuring the interpreter exception state is clean
/// regardless of platform or Python version.
///
/// Non-float → NaN is consistent with the Arrow null path (spec-027).
/// SIMD filter ops (col > x) naturally exclude NaN (IEEE 754).
#[inline]
unsafe fn pyfloat_as_f64(ptr: *mut pyo3::ffi::PyObject) -> f64 {
    pyo3::ffi::PyErr_Clear();
    let val = pyo3::ffi::PyFloat_AsDouble(ptr);
    if val == -1.0 && !pyo3::ffi::PyErr_Occurred().is_null() {
        pyo3::ffi::PyErr_Clear();
        f64::NAN
    } else {
        val
    }
}

const CHUNK_SIZE: usize = 4096; // 32 KB — fits in typical L1 cache

/// Materialize a Python list[float | None] into a fresh Vec<f64>.
/// Non-float elements (e.g. None) become NaN.
fn materialize_lazy_float_list(list_ptr: *mut pyo3::ffi::PyObject) -> Vec<f64> {
    unsafe {
        let n = pyo3::ffi::PyList_Size(list_ptr) as usize;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let elem = pyo3::ffi::PyList_GetItem(list_ptr, i as isize);
            out.push(pyfloat_as_f64(elem));
        }
        out
    }
}

/// Count elements passing all ops in a Python list[float], L1-cache-friendly.
///
/// Processes in CHUNK_SIZE (4096) batches. Each chunk fits in L1 cache, so the
/// SIMD count step runs faster than counting on a fully materialized 8 MB Vec.
/// GIL is held throughout (no allow_threads overhead); the SIMD count on each
/// 32 KB chunk is fast enough that GIL contention is negligible.
/// Falls back to full materialization when skip/take are active.
pub(super) fn count_lazy_float_list(
    list_ptr: *mut pyo3::ffi::PyObject,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> usize {
    if skip > 0 || take.is_some() {
        let v = materialize_lazy_float_list(list_ptr);
        return count_fused_f64_with_skip_take(&v, ops, skip, take);
    }
    let n = unsafe { pyo3::ffi::PyList_Size(list_ptr) as usize };
    let mut total = 0usize;
    let mut chunk_buf = vec![0f64; CHUNK_SIZE];
    let mut i = 0;
    while i < n {
        let end = (i + CHUNK_SIZE).min(n);
        let chunk_size = end - i;
        unsafe {
            for j in 0..chunk_size {
                let elem = pyo3::ffi::PyList_GetItem(list_ptr, (i + j) as isize);
                chunk_buf[j] = pyfloat_as_f64(elem);
            }
        }
        total += count_fused_f64_with_skip_take(&chunk_buf[..chunk_size], ops, 0, None);
        i = end;
    }
    total
}

pub(super) fn stats_lazy_float_list(
    list_ptr: *mut pyo3::ffi::PyObject,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> Option<(usize, f64, f64, f64)> {
    let n = unsafe { pyo3::ffi::PyList_Size(list_ptr) as usize };
    let mut skipped = 0usize;
    let mut count = 0usize;
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    'outer: for i in 0..n {
        let mut val = unsafe {
            let elem = pyo3::ffi::PyList_GetItem(list_ptr, i as isize);
            pyfloat_as_f64(elem)
        };
        for op in ops {
            match apply_scalar_op(val, op) {
                ScalarResult::Value(v) => val = v,
                ScalarResult::Filtered => continue 'outer,
            }
        }
        if skipped < skip {
            skipped += 1;
            continue;
        }
        count += 1;
        sum += val;
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
        if take.is_some_and(|n| count >= n) {
            break;
        }
    }

    if count == 0 {
        None
    } else {
        Some((count, sum, min, max))
    }
}

pub(super) fn sum_lazy_float_list(
    list_ptr: *mut pyo3::ffi::PyObject,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> f64 {
    stats_lazy_float_list(list_ptr, ops, skip, take)
        .map(|(_, sum, _, _)| sum)
        .unwrap_or(0.0)
}

pub(super) fn mean_lazy_float_list(
    list_ptr: *mut pyo3::ffi::PyObject,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> Option<f64> {
    stats_lazy_float_list(list_ptr, ops, skip, take).map(|(count, sum, _, _)| sum / count as f64)
}

pub(super) fn var_lazy_float_list(
    list_ptr: *mut pyo3::ffi::PyObject,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> Option<f64> {
    let (count, sum, _, _) = stats_lazy_float_list(list_ptr, ops, skip, take)?;
    let mean = sum / count as f64;
    let n = unsafe { pyo3::ffi::PyList_Size(list_ptr) as usize };
    let mut skipped = 0usize;
    let mut seen = 0usize;
    let mut ssq = 0.0f64;

    'outer: for i in 0..n {
        let mut val = unsafe {
            let elem = pyo3::ffi::PyList_GetItem(list_ptr, i as isize);
            pyfloat_as_f64(elem)
        };
        for op in ops {
            match apply_scalar_op(val, op) {
                ScalarResult::Value(v) => val = v,
                ScalarResult::Filtered => continue 'outer,
            }
        }
        if skipped < skip {
            skipped += 1;
            continue;
        }
        let d = val - mean;
        ssq += d * d;
        seen += 1;
        if take.is_some_and(|n| seen >= n) {
            break;
        }
    }

    Some(ssq / count as f64)
}

pub(super) fn min_lazy_float_list(
    list_ptr: *mut pyo3::ffi::PyObject,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> Option<f64> {
    stats_lazy_float_list(list_ptr, ops, skip, take).map(|(_, _, min, _)| min)
}

pub(super) fn max_lazy_float_list(
    list_ptr: *mut pyo3::ffi::PyObject,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> Option<f64> {
    stats_lazy_float_list(list_ptr, ops, skip, take).map(|(_, _, _, max)| max)
}

/// Execute a LazyFloatList pipeline.
///
/// Decision rule:
///   take * 4 < N  AND  N >= CHUNK_SIZE  → chunked SIMD lazy path (GIL released per chunk)
///   otherwise                           → eager path (materialize all, then SIMD)
pub(super) fn execute_lazy_float_list(
    py: Python<'_>,
    list_ptr: *mut pyo3::ffi::PyObject,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> Vec<f64> {
    let n = unsafe { pyo3::ffi::PyList_Size(list_ptr) as usize };
    let take_n = take.unwrap_or(n);

    let use_chunked = match take {
        Some(t) => t.saturating_mul(4) < n && n >= CHUNK_SIZE,
        None => false,
    };

    if !use_chunked {
        let v = materialize_lazy_float_list(list_ptr);
        return execute_fused_f64_with_skip_take(&v, ops, skip, take);
    }

    let mut out = Vec::with_capacity(take_n);
    let mut skipped = 0usize;
    let mut chunk_buf = vec![0f64; CHUNK_SIZE];
    let mut i = 0;

    while i < n && out.len() < take_n {
        let end = (i + CHUNK_SIZE).min(n);
        let chunk_size = end - i;

        unsafe {
            for j in 0..chunk_size {
                let elem = pyo3::ffi::PyList_GetItem(list_ptr, (i + j) as isize);
                chunk_buf[j] = pyfloat_as_f64(elem);
            }
        }

        let chunk_slice = &chunk_buf[..chunk_size];
        let filtered = py.detach(|| execute_fused_f64_with_skip_take(chunk_slice, ops, 0, None));

        for val in filtered {
            if skipped < skip {
                skipped += 1;
                continue;
            }
            out.push(val);
            if out.len() >= take_n {
                return out;
            }
        }

        i = end;
    }

    out
}

// ---------------------------------------------------------------------------
// ObjField helpers — fast field-filter path for list[dict]
// ---------------------------------------------------------------------------

/// Map a float-typed ObjOp to (field_name, NumericOp).
/// Returns None for string/bool equality ops (can't use SIMD for those).
pub(super) fn objop_to_numeric(op: &ObjOp) -> Option<(Arc<str>, NumericOp)> {
    match op {
        ObjOp::FilterFieldGt(f, v) => Some((Arc::clone(f), NumericOp::FilterGt(*v))),
        ObjOp::FilterFieldGe(f, v) => Some((Arc::clone(f), NumericOp::FilterGe(*v))),
        ObjOp::FilterFieldLt(f, v) => Some((Arc::clone(f), NumericOp::FilterLt(*v))),
        ObjOp::FilterFieldLe(f, v) => Some((Arc::clone(f), NumericOp::FilterLe(*v))),
        ObjOp::FilterFieldBetween(f, lo, hi) => {
            Some((Arc::clone(f), NumericOp::FilterBetween(*lo, *hi)))
        }
        _ => None,
    }
}

/// Extract one field from every dict in `source`, run SIMD filter, return matching original
/// Python dict references (no dict copy).
pub(super) fn filter_by_field(
    py: Python<'_>,
    source: &Py<PyAny>,
    field_name: &str,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<Vec<Py<PyAny>>> {
    use pyo3::ffi;

    let list = source
        .bind(py)
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err("ObjField source must be a list"))?;

    let n = list.len();

    let key = pyo3::types::PyString::new(py, field_name);
    let key_ptr = key.as_ptr();

    let mut values: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        // Owned ref: user __eq__ may run below and shrink the list.
        let held = list.get_item(i).ok();
        let item_ptr = held.as_ref().map_or(std::ptr::null_mut(), |h| h.as_ptr());
        let f = if !item_ptr.is_null() && unsafe { ffi::PyDict_Check(item_ptr) } != 0 {
            let val_ptr = unsafe { ffi::PyDict_GetItemWithError(item_ptr, key_ptr) };
            if val_ptr.is_null() {
                unsafe {
                    ffi::PyErr_Clear();
                }
                f64::NAN
            } else if unsafe { ffi::PyFloat_CheckExact(val_ptr) } != 0 {
                unsafe { ffi::PyFloat_AsDouble(val_ptr) }
            } else {
                let v = unsafe { ffi::PyLong_AsDouble(val_ptr) };
                if v == -1.0 && !unsafe { ffi::PyErr_Occurred() }.is_null() {
                    unsafe {
                        ffi::PyErr_Clear();
                    }
                    f64::NAN
                } else {
                    v
                }
            }
        } else {
            f64::NAN
        };
        values.push(f);
    }

    let ptr = values.as_ptr() as usize;
    let ops_c = ops.to_vec();
    // Safety: `values` lives in the enclosing frame, valid for the whole allow_threads call.
    let indices: Vec<usize> = py.detach(move || {
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const f64, n) };
        let mut out: Vec<usize> = Vec::new();
        let mut skipped = 0usize;
        'outer: for (i, &val) in slice.iter().enumerate() {
            for op in &ops_c {
                if !eval_filter_f64(val, op) {
                    continue 'outer;
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
    });

    let mut result: Vec<Py<PyAny>> = Vec::with_capacity(indices.len());
    for &i in &indices {
        // The list may have shrunk during the extraction pass (user __eq__);
        // indices past the new end are dropped rather than read as NULL.
        if let Ok(item) = list.get_item(i) {
            result.push(item.unbind());
        }
    }
    Ok(result)
}

/// Fused filter+sum for ObjField — single pass, no intermediate Vec of PyObjects.
///
/// Extracts `filter_field` values and (if different) `sum_field` values into f64 Vecs,
/// then filters and accumulates GIL-free.  No matching Python dicts are re-visited.
pub(super) fn sum_field_by_field(
    py: Python<'_>,
    source: &Py<PyAny>,
    filter_field: &str,
    ops: &[NumericOp],
    sum_field: &str,
    skip: usize,
    take: Option<usize>,
) -> PyResult<f64> {

    let list = source
        .bind(py)
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err("ObjField source must be a list"))?;

    let n = list.len();

    let same_field = filter_field == sum_field;

    let fkey = pyo3::types::PyString::new(py, filter_field);
    let fkey_ptr = fkey.as_ptr();

    let mut filter_vals: Vec<f64> = Vec::with_capacity(n);
    let mut sum_vals: Vec<f64> = if same_field {
        Vec::new()
    } else {
        Vec::with_capacity(n)
    };

    if same_field {
        // Single extraction pass — filter_vals doubles as sum_vals
        for i in 0..n {
            // Owned ref: user __eq__ may run below and shrink the list.
            let held = list.get_item(i).ok();
            let item_ptr = held.as_ref().map_or(std::ptr::null_mut(), |h| h.as_ptr());
            let f = unsafe { extract_f64_from_dict_item(item_ptr, fkey_ptr) };
            filter_vals.push(f);
        }
    } else {
        let skey = pyo3::types::PyString::new(py, sum_field);
        let skey_ptr = skey.as_ptr();
        for i in 0..n {
            // Owned ref: user __eq__ may run below and shrink the list.
            let held = list.get_item(i).ok();
            let item_ptr = held.as_ref().map_or(std::ptr::null_mut(), |h| h.as_ptr());
            filter_vals.push(unsafe { extract_f64_from_dict_item(item_ptr, fkey_ptr) });
            sum_vals.push(unsafe { extract_f64_from_dict_item(item_ptr, skey_ptr) });
        }
    }

    let fptr = filter_vals.as_ptr() as usize;
    let sptr = if same_field {
        fptr
    } else {
        sum_vals.as_ptr() as usize
    };
    let ops_c = ops.to_vec();

    // Safety: fptr/sptr point into filter_vals/sum_vals, which live in the
    // enclosing stack frame for the entire duration of allow_threads (which
    // blocks until the closure returns before those Vecs are dropped).
    Ok(py.detach(move || {
        let fslice = unsafe { std::slice::from_raw_parts(fptr as *const f64, n) };
        let sslice = unsafe { std::slice::from_raw_parts(sptr as *const f64, n) };
        let mut acc = 0.0f64;
        let mut skipped = 0usize;
        let mut taken = 0usize;
        'outer: for (i, &fval) in fslice.iter().enumerate() {
            for op in &ops_c {
                if !eval_filter_f64(fval, op) {
                    continue 'outer;
                }
            }
            if skipped < skip {
                skipped += 1;
                continue;
            }
            acc += sslice[i];
            taken += 1;
            if take.is_some_and(|t| taken >= t) {
                break;
            }
        }
        acc
    }))
}

/// Extract a f64 value from a dict item pointer using raw CPython FFI.
/// Returns NaN on missing key, non-numeric value, or type error.
///
/// # Safety
/// - GIL must be held throughout the call.
/// - `item_ptr` may be null (handled) but must not be dangling.
/// - `key_ptr` must be a non-null, live `PyString` object (borrowed ref from
///   `PyString::as_ptr()`); the caller must keep the `PyString` alive.
/// - `val_ptr` returned by `PyDict_GetItemWithError` is a borrowed ref — valid
///   only while the GIL is held and the dict is not modified.
#[inline]
unsafe fn extract_f64_from_dict_item(
    item_ptr: *mut pyo3::ffi::PyObject,
    key_ptr: *mut pyo3::ffi::PyObject,
) -> f64 {
    use pyo3::ffi;
    if item_ptr.is_null() || ffi::PyDict_Check(item_ptr) == 0 {
        return f64::NAN;
    }
    let val_ptr = ffi::PyDict_GetItemWithError(item_ptr, key_ptr);
    if val_ptr.is_null() {
        ffi::PyErr_Clear();
        return f64::NAN;
    }
    if ffi::PyFloat_CheckExact(val_ptr) != 0 {
        ffi::PyFloat_AsDouble(val_ptr)
    } else {
        let v = ffi::PyLong_AsDouble(val_ptr);
        if v == -1.0 && !ffi::PyErr_Occurred().is_null() {
            ffi::PyErr_Clear();
            f64::NAN
        } else {
            v
        }
    }
}

/// Count variant of filter_by_field — no output Vec, GIL-free comparison.
pub(super) fn count_by_field(
    py: Python<'_>,
    source: &Py<PyAny>,
    field_name: &str,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<usize> {
    let list = source
        .bind(py)
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err("ObjField source must be a list"))?;
    let n = list.len();

    let key = pyo3::types::PyString::new(py, field_name);
    let mut values: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let item = list.get_item(i)?;
        let v = if let Ok(dict) = item.cast::<PyDict>() {
            match dict.get_item(&key)? {
                Some(val) => val.extract::<f64>().unwrap_or(f64::NAN),
                None => f64::NAN,
            }
        } else {
            f64::NAN
        };
        values.push(v);
    }

    let ptr = values.as_ptr() as usize;
    let ops_c = ops.to_vec();
    // Safety: `values` lives in the enclosing frame, valid for the whole allow_threads call.
    Ok(py.detach(move || {
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const f64, n) };
        count_fused_f64_with_skip_take(slice, &ops_c, skip, take)
    }))
}

/// Returns true if `op` is a field-level filter op (Eq/Ne with any RustValue type).
/// Used to decide whether to route to ObjFieldPy vs fall through to Obj/RustObj.
pub(super) fn objop_is_field_filter(op: &ObjOp) -> bool {
    matches!(op, ObjOp::FilterFieldEq(..) | ObjOp::FilterFieldNe(..))
}

/// Prepared form of one ObjFieldPy filter op for the FFI hot loop.
/// Pre-builds Python key and target objects once — avoids per-element allocations.
struct PreparedFieldOp {
    key_py: Py<PyAny>,
    target: Py<PyAny>,
    is_eq: bool,
}

fn precompile_obj_field_ops(py: Python<'_>, ops: &[ObjOp]) -> PyResult<Vec<PreparedFieldOp>> {
    ops.iter()
        .filter_map(|op| {
            let (fname, rv, is_eq) = match op {
                ObjOp::FilterFieldEq(f, v) => (f, v, true),
                ObjOp::FilterFieldNe(f, v) => (f, v, false),
                _ => return None,
            };
            let py_key: Py<PyAny> = pyo3::types::PyString::new(py, fname.as_ref())
                .unbind()
                .into();
            let py_target: Py<PyAny> = match rv {
                RustValue::Str(s) => pyo3::types::PyString::new(py, s.as_ref())
                    .unbind()
                    .into(),
                RustValue::Bool(b) => to_py(py, b),
                RustValue::Int(i) => to_py(py, i),
                RustValue::Float(f) => to_py(py, f),
                RustValue::Null => py.None(),
            };
            Some(Ok(PreparedFieldOp {
                key_py: py_key,
                target: py_target,
                is_eq,
            }))
        })
        .collect()
}

/// Fast path for non-numeric field filtering — uses raw CPython FFI per element.
pub(super) fn filter_by_field_py(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[ObjOp],
    skip: usize,
    take: Option<usize>,
    map_field: Option<&str>,
) -> PyResult<Vec<Py<PyAny>>> {
    use pyo3::ffi;

    let list = source
        .bind(py)
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err("ObjFieldPy source must be a list"))?;

    let n = list.len();

    let prepared = precompile_obj_field_ops(py, ops)?;
    let map_key: Option<Py<PyAny>> =
        map_field.map(|s| pyo3::types::PyString::new(py, s).unbind().into());

    let mut result: Vec<Py<PyAny>> = Vec::new();
    let mut skipped = 0usize;

    'outer: for i in 0..n {
        // Owned ref: user __eq__ may run below and shrink the list.
        let held = list.get_item(i).ok();
        let item_ptr = held.as_ref().map_or(std::ptr::null_mut(), |h| h.as_ptr());
        if item_ptr.is_null() {
            continue;
        }
        if unsafe { ffi::PyDict_Check(item_ptr) } == 0 {
            continue;
        }

        let passes = prepared.iter().all(|op| {
            let val = unsafe { ffi::PyDict_GetItemWithError(item_ptr, op.key_py.as_ptr()) };
            if val.is_null() {
                unsafe {
                    ffi::PyErr_Clear();
                }
                return !op.is_eq;
            }
            // Hold `val` across the compare (a user __eq__ may mutate the dict and drop it)
            // and clear any exception it raised, matching the NaN-on-error policy above.
            let cmp = unsafe {
                ffi::Py_INCREF(val);
                let r = ffi::PyObject_RichCompareBool(val, op.target.as_ptr(), ffi::Py_EQ);
                ffi::Py_DECREF(val);
                if r < 0 {
                    ffi::PyErr_Clear();
                }
                r
            };
            if op.is_eq {
                cmp == 1
            } else {
                cmp != 1
            }
        });
        if !passes {
            continue;
        }

        if skipped < skip {
            skipped += 1;
            continue;
        }

        let output: Py<PyAny> = if let Some(ref mk) = map_key {
            let val = unsafe { ffi::PyDict_GetItemWithError(item_ptr, mk.as_ptr()) };
            if val.is_null() {
                unsafe {
                    ffi::PyErr_Clear();
                }
                py.None()
            } else {
                unsafe {
                    ffi::Py_INCREF(val);
                }
                unsafe { Bound::from_owned_ptr(py, val).unbind() }
            }
        } else {
            unsafe {
                ffi::Py_INCREF(item_ptr);
            }
            unsafe { Bound::from_owned_ptr(py, item_ptr).unbind() }
        };

        result.push(output);
        if take.is_some_and(|t| result.len() >= t) {
            break 'outer;
        }
    }
    Ok(result)
}

/// Count variant — no output Vec, just counts matching items.
pub(super) fn count_by_field_py(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[ObjOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<usize> {
    use pyo3::ffi;

    let list = source
        .bind(py)
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err("ObjFieldPy source must be a list"))?;

    let n = list.len();

    let prepared = precompile_obj_field_ops(py, ops)?;

    let mut count = 0usize;
    let mut skipped = 0usize;

    for i in 0..n {
        // Owned ref: user __eq__ may run below and shrink the list.
        let held = list.get_item(i).ok();
        let item_ptr = held.as_ref().map_or(std::ptr::null_mut(), |h| h.as_ptr());
        if item_ptr.is_null() {
            continue;
        }
        if unsafe { ffi::PyDict_Check(item_ptr) } == 0 {
            continue;
        }

        let passes = prepared.iter().all(|op| {
            let val = unsafe { ffi::PyDict_GetItemWithError(item_ptr, op.key_py.as_ptr()) };
            if val.is_null() {
                unsafe {
                    ffi::PyErr_Clear();
                }
                return !op.is_eq;
            }
            // Hold `val` across the compare (a user __eq__ may mutate the dict and drop it)
            // and clear any exception it raised, matching the NaN-on-error policy above.
            let cmp = unsafe {
                ffi::Py_INCREF(val);
                let r = ffi::PyObject_RichCompareBool(val, op.target.as_ptr(), ffi::Py_EQ);
                ffi::Py_DECREF(val);
                if r < 0 {
                    ffi::PyErr_Clear();
                }
                r
            };
            if op.is_eq {
                cmp == 1
            } else {
                cmp != 1
            }
        });
        if !passes {
            continue;
        }
        if skipped < skip {
            skipped += 1;
            continue;
        }
        count += 1;
        if take.is_some_and(|t| count >= t) {
            break;
        }
    }
    Ok(count)
}

/// Execute a NumpyF64 pipeline: get buffer-protocol slice, run SIMD — zero intermediate copy.
///
/// `buf` holds the buffer lock during `allow_threads`; the numpy array's data
/// remains valid until `buf` is dropped (after `allow_threads` returns, GIL re-acquired).
pub(super) fn execute_numpy_f64(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<Vec<f64>> {
    let buf = unsafe { RawBuffer::get(py, source.bind(py).as_ptr()) }?;
    let n = buf.item_count::<f64>()?;
    let ptr = buf.buf_ptr::<f64>() as usize;
    let result = py.detach(|| unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const f64, n);
        execute_fused_f64_with_skip_take(slice, ops, skip, take)
    });
    Ok(result)
}

pub(super) fn count_numpy_f64(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<usize> {
    let buf = unsafe { RawBuffer::get(py, source.bind(py).as_ptr()) }?;
    let n = buf.item_count::<f64>()?;
    let ptr = buf.buf_ptr::<f64>() as usize;
    let result = py.detach(|| unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const f64, n);
        count_fused_f64_with_skip_take(slice, ops, skip, take)
    });
    Ok(result)
}

#[inline]
fn compute_stats_f64_with_skip_take(
    data: &[f64],
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> Option<(usize, f64, f64, f64)> {
    if skip == 0
        && take.is_none()
        && ops.iter().all(|op| {
            matches!(
                op,
                NumericOp::FilterGt(_)
                    | NumericOp::FilterGe(_)
                    | NumericOp::FilterLt(_)
                    | NumericOp::FilterLe(_)
                    | NumericOp::FilterEq(_)
                    | NumericOp::FilterNe(_)
                    | NumericOp::FilterBetween(_, _)
                    | NumericOp::FilterNotNan
            )
        })
    {
        return filter_multi_stat_f64(data, ops);
    }

    let mut count = 0usize;
    let mut skipped = 0usize;
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    'outer: for &raw in data {
        let mut val = raw;
        for op in ops {
            match apply_scalar_op(val, op) {
                ScalarResult::Value(v) => val = v,
                ScalarResult::Filtered => continue 'outer,
            }
        }
        if skipped < skip {
            skipped += 1;
            continue;
        }
        count += 1;
        sum += val;
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
        if take.is_some_and(|n| count >= n) {
            break;
        }
    }

    if count == 0 {
        None
    } else {
        Some((count, sum, min, max))
    }
}

#[inline]
fn compute_variance_f64_with_skip_take(
    data: &[f64],
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> Option<f64> {
    let (count, sum, _, _) = compute_stats_f64_with_skip_take(data, ops, skip, take)?;
    let mean = sum / count as f64;
    let mut seen = 0usize;
    let mut skipped = 0usize;
    let mut ssq = 0.0f64;

    'outer: for &raw in data {
        let mut val = raw;
        for op in ops {
            match apply_scalar_op(val, op) {
                ScalarResult::Value(v) => val = v,
                ScalarResult::Filtered => continue 'outer,
            }
        }
        if skipped < skip {
            skipped += 1;
            continue;
        }
        let d = val - mean;
        ssq += d * d;
        seen += 1;
        if take.is_some_and(|n| seen >= n) {
            break;
        }
    }

    Some(ssq / count as f64)
}

pub(super) fn sum_numpy_f64(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<f64> {
    let buf = unsafe { RawBuffer::get(py, source.bind(py).as_ptr()) }?;
    let n = buf.item_count::<f64>()?;
    let ptr = buf.buf_ptr::<f64>() as usize;
    let result = py.detach(|| unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const f64, n);
        if skip == 0 && take.is_none() {
            if let Some(s) = filter_sum_fused_f64(slice, ops) {
                return s;
            }
        }
        compute_stats_f64_with_skip_take(slice, ops, skip, take)
            .map(|(_, sum, _, _)| sum)
            .unwrap_or(0.0)
    });
    Ok(result)
}

pub(super) fn mean_numpy_f64(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<Option<f64>> {
    let buf = unsafe { RawBuffer::get(py, source.bind(py).as_ptr()) }?;
    let n = buf.item_count::<f64>()?;
    let ptr = buf.buf_ptr::<f64>() as usize;
    let result = py.detach(|| unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const f64, n);
        if skip == 0 && take.is_none() {
            if let Some(mean) = filter_mean_fused_f64(slice, ops) {
                return mean;
            }
        }
        compute_stats_f64_with_skip_take(slice, ops, skip, take)
            .map(|(count, sum, _, _)| sum / count as f64)
    });
    Ok(result)
}

pub(super) fn var_numpy_f64(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<Option<f64>> {
    let buf = unsafe { RawBuffer::get(py, source.bind(py).as_ptr()) }?;
    let n = buf.item_count::<f64>()?;
    let ptr = buf.buf_ptr::<f64>() as usize;
    let result = py.detach(|| unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const f64, n);
        compute_variance_f64_with_skip_take(slice, ops, skip, take)
    });
    Ok(result)
}

pub(super) fn min_numpy_f64(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<Option<f64>> {
    let buf = unsafe { RawBuffer::get(py, source.bind(py).as_ptr()) }?;
    let n = buf.item_count::<f64>()?;
    let ptr = buf.buf_ptr::<f64>() as usize;
    let result = py.detach(|| unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const f64, n);
        compute_stats_f64_with_skip_take(slice, ops, skip, take).map(|(_, _, min, _)| min)
    });
    Ok(result)
}

pub(super) fn max_numpy_f64(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<Option<f64>> {
    let buf = unsafe { RawBuffer::get(py, source.bind(py).as_ptr()) }?;
    let n = buf.item_count::<f64>()?;
    let ptr = buf.buf_ptr::<f64>() as usize;
    let result = py.detach(|| unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const f64, n);
        compute_stats_f64_with_skip_take(slice, ops, skip, take).map(|(_, _, _, max)| max)
    });
    Ok(result)
}

pub(super) fn stats_numpy_f64(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<Option<(usize, f64, f64, f64)>> {
    let buf = unsafe { RawBuffer::get(py, source.bind(py).as_ptr()) }?;
    let n = buf.item_count::<f64>()?;
    let ptr = buf.buf_ptr::<f64>() as usize;
    let result = py.detach(|| unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const f64, n);
        if skip == 0 && take.is_none() {
            // Fast path for pure filter chains; falls through for maps.
            if ops.iter().all(|op| matches!(
                op,
                NumericOp::FilterGt(_)
                    | NumericOp::FilterGe(_)
                    | NumericOp::FilterLt(_)
                    | NumericOp::FilterLe(_)
                    | NumericOp::FilterEq(_)
                    | NumericOp::FilterNe(_)
                    | NumericOp::FilterBetween(_, _)
                    | NumericOp::FilterNotNan
            )) {
                return filter_multi_stat_f64(slice, ops);
            }
        }
        compute_stats_f64_with_skip_take(slice, ops, skip, take)
    });
    Ok(result)
}

/// Execute a NumpyF32 pipeline: read buffer as &[f32], run fused ops, return Vec<f32>.
pub(super) fn execute_numpy_f32(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<Vec<f32>> {
    let buf = unsafe { RawBuffer::get(py, source.bind(py).as_ptr()) }?;
    let n = buf.item_count::<f32>()?;
    let ptr = buf.buf_ptr::<f32>() as usize;
    let result = py.detach(|| unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const f32, n);
        execute_fused_f32_with_skip_take(slice, ops, skip, take)
    });
    Ok(result)
}

/// Count elements in a NumpyF32 pipeline without materialising the output.
pub(super) fn count_numpy_f32(
    py: Python<'_>,
    source: &Py<PyAny>,
    ops: &[NumericOp],
    skip: usize,
    take: Option<usize>,
) -> PyResult<usize> {
    let buf = unsafe { RawBuffer::get(py, source.bind(py).as_ptr()) }?;
    let n = buf.item_count::<f32>()?;
    let ptr = buf.buf_ptr::<f32>() as usize;
    let result = py.detach(|| unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const f32, n);
        count_fused_f32_with_skip_take(slice, ops, skip, take)
    });
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    // ── materialize_lazy_float_list ───────────────────────────────────────────

    #[test]
    fn test_materialize_normal() {
        Python::initialize();
        Python::attach(|py| {
            let list = PyList::new(py, [1.0f64, 2.0, 3.0]).unwrap();
            let result = materialize_lazy_float_list(list.as_ptr());
            assert_eq!(result, vec![1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn test_materialize_empty() {
        Python::initialize();
        Python::attach(|py| {
            let list = PyList::empty(py);
            let result = materialize_lazy_float_list(list.as_ptr());
            assert!(result.is_empty());
        });
    }

    #[test]
    fn test_materialize_integers_coerced() {
        Python::initialize();
        Python::attach(|py| {
            // Python list of ints → floats via PyLong_AsDouble
            let list = PyList::new(py, [1i64, 2, 3]).unwrap();
            let result = materialize_lazy_float_list(list.as_ptr());
            assert_eq!(result, vec![1.0, 2.0, 3.0]);
        });
    }

    // ── count_lazy_float_list ─────────────────────────────────────────────────

    #[test]
    fn test_count_matches_execute() {
        Python::initialize();
        Python::attach(|py| {
            let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
            let list = PyList::new(py, &data).unwrap();
            let ops = vec![NumericOp::FilterGt(50.0)];

            let count = count_lazy_float_list(list.as_ptr(), &ops, 0, None);
            let materialized = materialize_lazy_float_list(list.as_ptr());
            let expected: Vec<f64> = materialized.into_iter().filter(|&x| x > 50.0).collect();

            assert_eq!(count, expected.len());
        });
    }

    #[test]
    fn test_count_with_skip_take() {
        Python::initialize();
        Python::attach(|py| {
            let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
            let list = PyList::new(py, &data).unwrap();
            let ops = vec![NumericOp::FilterGt(5.0)];

            // count_lazy_float_list with skip/take falls back to count_fused_f64_with_skip_take,
            // which applies filter first, then skip/take on the filtered results.
            // Verify count equals execute path (materialize → filter → skip → take).
            let count = count_lazy_float_list(list.as_ptr(), &ops, 2, Some(5));
            let executed = execute_lazy_float_list(py, list.as_ptr(), &ops, 2, Some(5));
            assert_eq!(count, executed.len());
        });
    }

    #[test]
    fn test_count_empty_list() {
        Python::initialize();
        Python::attach(|py| {
            let list = PyList::empty(py);
            let count = count_lazy_float_list(list.as_ptr(), &[], 0, None);
            assert_eq!(count, 0);
        });
    }

    // ── extract_f64_from_dict_item ────────────────────────────────────────────

    #[test]
    fn test_extract_f64_from_float_value() {
        Python::initialize();
        Python::attach(|py| {
            use pyo3::types::{PyDict, PyString};
            let dict = PyDict::new(py);
            dict.set_item("score", 3.14f64).unwrap();
            let key = PyString::new(py, "score");
            let result = unsafe {
                extract_f64_from_dict_item(dict.as_ptr(), key.as_ptr())
            };
            assert!((result - 3.14).abs() < 1e-10);
        });
    }

    #[test]
    fn test_extract_f64_from_int_value() {
        Python::initialize();
        Python::attach(|py| {
            use pyo3::types::{PyDict, PyString};
            let dict = PyDict::new(py);
            dict.set_item("count", 42i64).unwrap();
            let key = PyString::new(py, "count");
            let result = unsafe {
                extract_f64_from_dict_item(dict.as_ptr(), key.as_ptr())
            };
            assert_eq!(result, 42.0);
        });
    }

    #[test]
    fn test_extract_f64_missing_key_returns_nan() {
        Python::initialize();
        Python::attach(|py| {
            use pyo3::types::{PyDict, PyString};
            let dict = PyDict::new(py);
            let key = PyString::new(py, "missing");
            let result = unsafe {
                extract_f64_from_dict_item(dict.as_ptr(), key.as_ptr())
            };
            assert!(result.is_nan());
        });
    }

    #[test]
    fn test_extract_f64_null_ptr_returns_nan() {
        Python::initialize();
        Python::attach(|_py| {
            use pyo3::types::PyString;
            Python::attach(|py| {
                let key = PyString::new(py, "k");
                let result = unsafe {
                    extract_f64_from_dict_item(std::ptr::null_mut(), key.as_ptr())
                };
                assert!(result.is_nan());
            });
        });
    }
}
