"""
Source adapters — convert external data formats into ZPyFlow Query objects.

Each adapter is a thin shim that normalizes data into a form the Rust core
understands (list[float], list[int], or list[Any]).  Zero-copy is used where
the format allows it (numpy f64 arrays go directly; others are converted once).

CSV / JSON Lines parsing routes through the Rust core (GIL-free for path
inputs; GIL-free parse after a one-time read for file-like inputs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, TypeVar

from ._zpyflow import Query

T = TypeVar("T")


def from_numpy(arr: Any) -> Query:
    """
    Create a Query from a numpy ndarray.

    For 1-D float64/int64/bool/uint8 C-contiguous arrays, reads via the buffer protocol
    (one bulk memcpy — no per-element boxing).  Other dtypes are converted to
    float64/int64 first, then read the same way.

    >>> import numpy as np
    >>> result = from_numpy(np.arange(1e6)).filter(col > 500_000).count()
    """
    import numpy as np

    if arr.ndim != 1:
        raise ValueError(f"from_numpy expects 1-D arrays, got shape {arr.shape}")

    # Ensure C-contiguous layout so the buffer protocol path works.
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    if arr.dtype == np.float64:
        return Query._from_buffer_f64(arr)        # 1 bulk memcpy, no boxing
    elif arr.dtype == np.int64:
        return Query._from_buffer_i64(arr)
    elif arr.dtype == np.bool_:
        # bool: keep compact 0/1 storage; maps promote to the i64 path.
        return Query._from_buffer_u8(arr.view(np.uint8))
    elif arr.dtype == np.uint8:
        return Query._from_buffer_u8(arr)
    elif arr.dtype == np.float32:
        return Query._from_buffer_f32(arr)
    elif arr.dtype == np.float16:
        return Query._from_buffer_f64(arr.astype(np.float64))
    elif arr.dtype in (np.int32, np.int16, np.int8, np.uint32, np.uint16):
        return Query._from_buffer_i64(arr.astype(np.int64))
    elif arr.dtype == np.uint64:
        # uint64 may overflow i64; fall back to tolist()
        return Query(arr.tolist())
    else:
        return Query(list(arr))


def from_arrow(table_or_array: Any) -> Query:
    """
    Create a Query from a PyArrow Array, ChunkedArray, or Table column.

    For null-free float64/int64 arrays, reads via the buffer protocol:
    numpy view (zero-copy for float64) → GIL-free memcpy into Rust Vec.

    Float64 arrays with nulls are supported: nulls become ``NaN`` (IEEE 754).
    Filter them out with ``filter(col == col)`` or ``filter(col.between(lo, hi))``.

    Int64 with nulls, bool, and string arrays fall back to ``to_pylist()``.
    """
    try:
        import pyarrow as pa
    except ImportError:
        raise ImportError("PyArrow is required for from_arrow(). pip install pyarrow")

    if isinstance(table_or_array, pa.Table):
        return Query(table_or_array.to_pylist())

    arr = table_or_array
    if hasattr(arr, "combine_chunks"):
        arr = arr.combine_chunks()  # ChunkedArray → Array

    py_type = arr.type

    # Fast path: null-free numeric → numpy view → buffer protocol + GIL-free memcpy
    if arr.null_count == 0:
        if pa.types.is_float64(py_type):
            # to_numpy() is zero-copy for null-free float64 arrays
            return Query._from_buffer_f64(arr.to_numpy())
        if pa.types.is_floating(py_type):
            return Query._from_buffer_f64(arr.cast(pa.float64()).to_numpy())
        if pa.types.is_integer(py_type):
            return Query._from_buffer_i64(arr.cast(pa.int64()).to_numpy())
    elif pa.types.is_float64(py_type):
        # Nulls become NaN; filter with col == col or col.between(...) as needed
        return Query._from_buffer_f64(arr.to_numpy(zero_copy_only=False))

    return Query(arr.to_pylist())


def from_csv(
    path_or_file: Any,
    column: "str | int | None" = None,
    dtype: Literal["auto", "float", "int", "str"] = "auto",
    delimiter: str = ",",
    has_header: bool = True,
) -> Query:
    """
    Parse a CSV file into a Query (GIL-free for path inputs).

    Parameters
    ----------
    path_or_file : str, Path, or file-like
        Input source.  When a path string or Path object is given, the entire
        read + parse happens in Rust with the GIL released.  For file-like
        objects the content is read once (GIL held), then parsed in Rust.
    column : str or int, optional
        Column name (requires has_header=True) or 0-based index to extract.
        If None, each row is returned as a dict.
    dtype : "auto" | "float" | "int" | "str"
        Value coercion for the extracted column ("auto" = int → float → str).
    delimiter : str
        Field separator (single character).
    has_header : bool
        Whether the first row contains column names.
    """
    column_name = column if isinstance(column, str) else None
    column_idx  = column if isinstance(column, int) else None

    if isinstance(path_or_file, (str, Path)):
        return Query._from_csv_path(
            str(path_or_file),
            column_name=column_name,
            column_idx=column_idx,
            dtype=dtype,
            delimiter=delimiter,
            has_header=has_header,
        )

    # File-like: read once (GIL), then parse in Rust (GIL released)
    raw = path_or_file.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return Query._from_csv_bytes(
        raw,
        column_name=column_name,
        column_idx=column_idx,
        dtype=dtype,
        delimiter=delimiter,
        has_header=has_header,
    )


def from_json_lines(
    path_or_file: Any,
    field: "str | None" = None,
    dtype: Literal["auto", "float", "int", "str"] = "auto",
) -> Query:
    """
    Parse a JSON Lines (NDJSON) file into a Query (GIL-free for path inputs).

    Each line must be a JSON object.  If `field` is given, that field's value
    is extracted from each line; otherwise each line becomes a dict row.
    """
    if isinstance(path_or_file, (str, Path)):
        return Query._from_jsonl_path(
            str(path_or_file),
            field_name=field,
            dtype=dtype,
        )

    raw = path_or_file.read()
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    return Query._from_jsonl_bytes(raw, field_name=field, dtype=dtype)


def from_generator(gen: Iterable[T]) -> Query:
    """
    Materialize a generator/iterable into a Query.

    Note: this eagerly consumes the generator.
    """
    return Query(list(gen))
