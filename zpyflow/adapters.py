"""
Source adapters — convert external data formats into ZPyFlow Query objects.

Each adapter is a thin shim that normalizes data into a form the Rust core
understands (list[float], list[int], or list[Any]).  Buffer-protocol fast paths
are used where the format allows it; other inputs are converted once.

CSV / JSON Lines parsing routes through the Rust core (GIL-free for path
inputs; GIL-free parse after a one-time read for file-like inputs).
"""

from __future__ import annotations

import csv as _csv
from pathlib import Path
from typing import Any, Generator, Iterable, Literal, TypeVar

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

    For null-free float64/int64 arrays, reads the Arrow data buffer via the
    buffer protocol: Arrow buffer → memoryview → GIL-free memcpy into Rust Vec.

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

    def data_view(array: Any, format_code: str, byte_width: int) -> memoryview:
        buf = array.buffers()[1]
        if buf is None:
            return memoryview(b"").cast(format_code)
        start = array.offset * byte_width
        end = start + len(array) * byte_width
        return memoryview(buf)[start:end].cast(format_code)

    arr = table_or_array
    if hasattr(arr, "combine_chunks"):
        arr = arr.combine_chunks()  # ChunkedArray → Array

    py_type = arr.type

    # Fast path: null-free numeric → Arrow data buffer → buffer protocol + GIL-free memcpy
    if arr.null_count == 0:
        if pa.types.is_float64(py_type):
            return Query._from_buffer_f64(data_view(arr, "d", 8))
        if pa.types.is_floating(py_type):
            arr = arr.cast(pa.float64())
            return Query._from_buffer_f64(data_view(arr, "d", 8))
        if pa.types.is_integer(py_type):
            arr = arr.cast(pa.int64())
            return Query._from_buffer_i64(data_view(arr, "q", 8))
    elif pa.types.is_float64(py_type):
        # Nulls become NaN; filter with col == col or col.between(...) as needed
        import pyarrow.compute as pc

        arr = pc.fill_null(arr, float("nan"))
        return Query._from_buffer_f64(data_view(arr, "d", 8))

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


def _csv_coerce(value: str, dtype: str) -> Any:
    """Convert a CSV string cell to a Python value matching the requested dtype."""
    if dtype == "float":
        return float(value)
    if dtype == "int":
        return int(value)
    if dtype == "str":
        return value
    # "auto": int → float → str (mirrors Rust csv_auto)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def from_csv_chunked(
    path_or_file: Any,
    chunk_size: int = 10_000,
    column: "str | int | None" = None,
    dtype: Literal["auto", "float", "int", "str"] = "auto",
    delimiter: str = ",",
    has_header: bool = True,
) -> Generator[Query, None, None]:
    """Stream a large CSV as an iterator of fixed-size ``Query`` objects.

    Unlike :func:`from_csv` (which loads the whole file at once), this
    generator yields one ``Query`` per chunk of ``chunk_size`` rows.  The
    file is read line-by-line so peak memory is bounded by chunk size rather
    than file size.

    Parameters
    ----------
    path_or_file : str, Path, or text-mode file-like
        Input source.
    chunk_size : int
        Number of data rows per yielded ``Query`` (last chunk may be smaller).
    column : str or int, optional
        Column name (requires ``has_header=True``) or 0-based index to
        extract.  When set, each ``Query`` contains scalar values rather
        than dicts.
    dtype : "auto" | "float" | "int" | "str"
        Value coercion for extracted column values.
    delimiter : str
        Field separator (single character).
    has_header : bool
        Whether the first row contains column names.

    Yields
    ------
    Query
        Each ``Query`` wraps ``chunk_size`` rows (or fewer for the last chunk).

    Example::

        total = 0
        for q in from_csv_chunked("large.csv", chunk_size=50_000):
            total += q.count()

        # Numeric column in streaming fashion
        total_revenue = 0.0
        for q in from_csv_chunked("sales.csv", column="amount", dtype="float"):
            total_revenue += q.filter(col > 0).sum()
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    opened_here = False
    if isinstance(path_or_file, (str, Path)):
        f = open(path_or_file, newline="", encoding="utf-8")
        opened_here = True
    else:
        f = path_or_file

    try:
        reader = _csv.reader(f, delimiter=delimiter)

        headers: list[str] | None = None
        col_idx: int | None = None
        if has_header:
            headers = next(reader)
            if isinstance(column, str):
                try:
                    col_idx = headers.index(column)
                except ValueError:
                    raise ValueError(
                        f"CSV column {column!r} not found; available: {headers}"
                    )
            elif isinstance(column, int):
                col_idx = column
        elif isinstance(column, int):
            col_idx = column

        chunk: list[Any] = []
        for row in reader:
            if col_idx is not None:
                chunk.append(_csv_coerce(row[col_idx], dtype))
            elif headers is not None:
                chunk.append({k: _csv_coerce(v, "auto") for k, v in zip(headers, row)})
            else:
                chunk.append({i: _csv_coerce(v, "auto") for i, v in enumerate(row)})

            if len(chunk) >= chunk_size:
                yield Query(chunk)
                chunk = []

        if chunk:
            yield Query(chunk)
    finally:
        if opened_here:
            f.close()


def from_arrow_ipc(
    path: Any,
    column: "str | int | None" = None,
) -> "Query":
    """Read an Arrow IPC file or stream and return a Query.

    Supports both the Arrow **file** format (random-access) and the Arrow
    **stream** format.  For single-column numeric data (or when *column* is
    given), values are extracted zero-copy via the buffer protocol — no
    per-element Python boxing.  Multi-column tables are returned as a Query
    of dicts.

    Parameters
    ----------
    path : str or Path
        Path to the ``.arrow`` / ``.arrows`` file.
    column : str or int, optional
        Column name or 0-based index to extract.  When given, the result is a
        numeric Query (same fast path as :func:`from_arrow`).  When *None* and
        the file has exactly one column, that column is extracted automatically.
        When *None* and the file has multiple columns, each row becomes a dict.

    Examples
    --------
    ::

        # Single numeric column — zero-copy float64 Query
        q = from_arrow_ipc("measurements.arrow")

        # Pick one column from a multi-column file
        q = from_arrow_ipc("events.arrow", column="latency_ms")

        # Multi-column → dict rows
        q = from_arrow_ipc("events.arrow")
    """
    try:
        import pyarrow as pa
        import pyarrow.ipc as _ipc
    except ImportError:
        raise ImportError("PyArrow is required for from_arrow_ipc(). pip install pyarrow")

    path_str = str(path)
    try:
        reader = _ipc.open_file(path_str)
    except pa.lib.ArrowInvalid:
        reader = _ipc.open_stream(path_str)

    table = reader.read_all()

    if column is not None:
        arr = table.column(column)
        return from_arrow(arr)

    if table.num_columns == 1:
        return from_arrow(table.column(0))

    return Query(table.to_pylist())


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
