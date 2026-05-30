"""
Type stubs for ZPyFlow — enables IDE autocomplete, mypy, and pyright checking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Iterator, Literal, TypeVar, overload

T = TypeVar("T")
U = TypeVar("U")
K = TypeVar("K")


# ---------------------------------------------------------------------------
# Numeric DSL
# ---------------------------------------------------------------------------

class Expr:
    """Encodes a Rust-side numeric operation — created by ``col > n``, etc."""

    def __gt__(self, other: float) -> Expr: ...
    def __ge__(self, other: float) -> Expr: ...
    def __lt__(self, other: float) -> Expr: ...
    def __le__(self, other: float) -> Expr: ...
    def __eq__(self, other: object) -> Expr: ...  # type: ignore[override]
    def __ne__(self, other: object) -> Expr: ...  # type: ignore[override]
    def __mul__(self, other: float) -> Expr: ...
    def __add__(self, other: float) -> Expr: ...
    def __sub__(self, other: float) -> Expr: ...
    def __truediv__(self, other: float) -> Expr: ...
    def __pow__(self, other: float, mod: None = None) -> Expr: ...
    def __neg__(self) -> Expr: ...
    def abs(self) -> Expr: ...
    def sqrt(self) -> Expr: ...
    def floor(self) -> Expr: ...
    def ceil(self) -> Expr: ...
    def round(self) -> Expr: ...
    def reciprocal(self) -> Expr: ...
    def between(self, lo: float, hi: float) -> Expr: ...


class ColProxy:
    """Sentinel — write ``col > 5`` to create an :class:`Expr`."""

    def __gt__(self, other: float) -> Expr: ...
    def __ge__(self, other: float) -> Expr: ...
    def __lt__(self, other: float) -> Expr: ...
    def __le__(self, other: float) -> Expr: ...
    def __eq__(self, other: object) -> Expr: ...  # type: ignore[override]
    def __ne__(self, other: object) -> Expr: ...  # type: ignore[override]
    def __mul__(self, other: float) -> Expr: ...
    def __add__(self, other: float) -> Expr: ...
    def __sub__(self, other: float) -> Expr: ...
    def __truediv__(self, other: float) -> Expr: ...
    def __pow__(self, other: float, mod: None = None) -> Expr: ...
    def __neg__(self) -> Expr: ...
    def abs(self) -> Expr: ...
    def sqrt(self) -> Expr: ...
    def floor(self) -> Expr: ...
    def ceil(self) -> Expr: ...
    def round(self) -> Expr: ...
    def reciprocal(self) -> Expr: ...
    def between(self, lo: float, hi: float) -> Expr: ...


col: ColProxy


# ---------------------------------------------------------------------------
# Object / dict DSL
# ---------------------------------------------------------------------------

class FieldExpr:
    """Encodes a field-based filter on dict records.

    Create with :func:`field`::

        field("price") > 100
        field("status") == 200
        field("latency_ms").between(10.0, 500.0)

    Also callable as a predicate::

        Query(records).filter(field("active") == True)
        list(filter(field("score") >= 0.9, records))
    """

    def __gt__(self, other: float) -> FieldExpr: ...
    def __ge__(self, other: float) -> FieldExpr: ...
    def __lt__(self, other: float) -> FieldExpr: ...
    def __le__(self, other: float) -> FieldExpr: ...
    def __eq__(self, other: object) -> FieldExpr: ...  # type: ignore[override]
    def __ne__(self, other: object) -> FieldExpr: ...  # type: ignore[override]
    def between(self, lo: float, hi: float) -> FieldExpr: ...
    def __call__(self, row: dict[str, Any]) -> bool: ...


def field(name: str) -> FieldExpr:
    """Create a :class:`FieldExpr` that accesses ``name`` on dict records.

    Example::

        from zpyflow import Query, field

        result = Query(records).filter(field("price") > 100).to_list()
    """
    ...


# ---------------------------------------------------------------------------
# Aggregation specs
# ---------------------------------------------------------------------------

class AggSpec:
    """Aggregation specification for :meth:`Query.group_agg`.

    Use the factory functions :func:`agg_count`, :func:`agg_sum`, etc.
    """

    @staticmethod
    def count() -> AggSpec: ...
    @staticmethod
    def sum(field_fn: Callable[[Any], float]) -> AggSpec: ...
    @staticmethod
    def mean(field_fn: Callable[[Any], float]) -> AggSpec: ...
    @staticmethod
    def max(field_fn: Callable[[Any], float]) -> AggSpec: ...
    @staticmethod
    def min(field_fn: Callable[[Any], float]) -> AggSpec: ...


def agg_count() -> AggSpec:
    """Count elements in each group."""
    ...


def agg_sum(field_fn: Callable[[Any], float]) -> AggSpec:
    """Sum ``field_fn(item)`` over each group."""
    ...


def agg_mean(field_fn: Callable[[Any], float]) -> AggSpec:
    """Mean of ``field_fn(item)`` over each group."""
    ...


def agg_max(field_fn: Callable[[Any], float]) -> AggSpec:
    """Maximum of ``field_fn(item)`` over each group."""
    ...


def agg_min(field_fn: Callable[[Any], float]) -> AggSpec:
    """Minimum of ``field_fn(item)`` over each group."""
    ...


# ---------------------------------------------------------------------------
# GroupBy
# ---------------------------------------------------------------------------

class GroupBy(Generic[K, T]):
    """Grouped query — obtain via :meth:`Query.group_by`.

    Example::

        from zpyflow import Query

        (
            Query(records)
                .group_by(lambda r: r["dept"])
                .agg(count=lambda g: g.count())
        )
    """

    def __init__(self, items: Iterable[T], key_fn: Callable[[T], K]) -> None: ...
    def keys(self) -> list[K]: ...
    def get_group(self, key: K) -> Query[T]: ...
    def agg(self, **reducers: Callable[[Query[T]], Any]) -> list[dict[str, Any]]: ...
    def map_groups(self, fn: Callable[[K, Query[T]], U]) -> list[U]: ...
    def count_per_group(self) -> dict[K, int]: ...
    def sum_per_group(
        self, field: Callable[[T], float] | None = None
    ) -> dict[K, float]: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class Query(Generic[T]):
    """Lazy query pipeline — operations are fused and deferred to the terminal call.

    Dispatch strategy
    -----------------
    * ``list[float]`` / numpy ``float64``  → f64 fast path (SIMD, GIL released)
    * ``list[int]``   / numpy ``int64``    → i64 fast path (GIL released)
    * ``list[dict]``                       → dict path; :func:`field` DSL converts
                                             to GIL-free RustObj on first DSL filter
    * Anything else                        → generic Python path (GIL held)

    Example::

        from zpyflow import Query, col, field

        # Numeric fast path
        total = Query(prices).filter(col > 0).sum()

        # Dict path with field DSL (GIL released after first filter)
        result = (
            Query(records)
                .filter(field("status") >= 500)
                .count()
        )

        # Pre-convert once, query many times
        q = Query(records).preload()
        count  = q.filter(field("latency_ms") > 100).count()
        errors = q.filter(field("status") >= 500).to_list()
    """

    def __init__(self, data: Iterable[T]) -> None: ...

    @staticmethod
    def f64(data: Iterable[Any]) -> "Query[float]":
        """Construct a Query with explicit f64 coercion (guarantees SIMD / GIL-free path).

        Use when your list contains mixed numeric types, e.g. ``[1, 2, 3.0]``.
        Raises ``ValueError`` if any element cannot be converted to float.
        """
        ...

    @staticmethod
    def i64(data: Iterable[Any]) -> "Query[int]":
        """Construct a Query with explicit i64 coercion (guarantees GIL-free path).

        Use when your list contains mixed integer types.
        Raises ``ValueError`` if any element cannot be converted to int.
        """
        ...

    # ------------------------------------------------------------------
    # Lazy combinators
    # ------------------------------------------------------------------

    @overload
    def filter(self, pred: Expr) -> Query[T]: ...
    @overload
    def filter(self, pred: FieldExpr) -> Query[T]: ...
    @overload
    def filter(self, pred: Callable[[T], bool]) -> Query[T]: ...

    @overload
    def map(self, f: Expr) -> Query[Any]: ...
    @overload
    def map(self, f: Callable[[T], U]) -> Query[U]: ...

    def take(self, n: int) -> Query[T]:
        """Stop after at most *n* elements (early exit)."""
        ...

    def skip(self, n: int) -> Query[T]:
        """Skip the first *n* elements."""
        ...

    def parallel(self) -> Query[T]:
        """Request parallel execution (numeric fast path only, uses rayon)."""
        ...

    def take_while(self, pred: Expr | FieldExpr | Callable[[T], bool]) -> Query[T]:
        """Yield elements while *pred* is true, then stop."""
        ...

    def skip_while(self, pred: Expr | FieldExpr | Callable[[T], bool]) -> Query[T]:
        """Skip elements while *pred* is true, then yield the rest."""
        ...

    def chain(self, other: Query[T]) -> Query[T]:
        """Concatenate *other* after this query (zero allocation)."""
        ...

    def enumerate(self) -> Query[tuple[int, T]]:
        """Pair each element with its 0-based index, yielding ``(int, element)`` tuples."""
        ...

    def zip(self, other: Query[U]) -> Query[tuple[T, U]]:
        """Pair elements with *other*, yielding ``(a, b)`` tuples (stops at shorter)."""
        ...

    def flat_map(self, f: Callable[[T], Iterable[U]]) -> Query[U]:
        """Apply *f* to each element and flatten the results one level."""
        ...

    def preload(self) -> Query[T]:
        """Convert dict records to RustObj eagerly (pay GIL cost once, query many times).

        Useful when the same dataset is queried repeatedly::

            q = Query(records).preload()
            for threshold in thresholds:
                counts[threshold] = q.filter(field("score") > threshold).count()
        """
        ...

    def group_by(self, key_fn: Callable[[T], K]) -> GroupBy[K, T]:
        """Group elements by *key_fn*, returning a :class:`GroupBy` object."""
        ...

    def group_agg(
        self,
        key_fn: FieldExpr | Callable[[T], Any],
        **specs: AggSpec,
    ) -> list[dict[str, Any]]:
        """Single-pass group + aggregate using the Rust kernel.

        Example::

            from zpyflow import Query, agg_count, agg_sum

            result = Query(products).group_agg(
                lambda p: p["category"],
                count   = agg_count(),
                revenue = agg_sum(lambda p: p["price"]),
            )
            # [{"_key": "books", "count": 42, "revenue": 1234.5}, ...]
        """
        ...

    # ------------------------------------------------------------------
    # Terminal operations
    # ------------------------------------------------------------------

    def to_list(self) -> list[T]:
        """Materialise the pipeline into a Python list (allocates once)."""
        ...

    def to_numpy(self) -> Any:
        """Materialise the pipeline into a numpy ndarray (no per-element boxing).

        Transfers the Rust ``Vec`` buffer directly to numpy — no extra copy
        and no Python float/int boxing.  Equivalent to ``to_bytes()`` +
        ``np.frombuffer()`` but returns a writable, owned array.

        Dtype mapping:

        =============  ==========
        Pipeline kind  numpy dtype
        =============  ==========
        f64            float64
        i64            int64
        u8             uint8
        =============  ==========

        Raises ``ValueError`` for object/Py pipelines.  Use
        ``np.array(q.to_list())`` for those.

        Example::

            import numpy as np
            arr = Query(data).filter(col > 0).map(col * 2).to_numpy()
            # arr is a writable np.ndarray[float64], zero extra copies
        """
        ...

    def to_dict(
        self,
        key: Callable[[T], K],
        value: Callable[[T], U],
    ) -> dict[K, U]:
        """Materialise as a dict using *key* and *value* callables."""
        ...

    def count(self) -> int:
        """Count matching elements without materialising them."""
        ...

    def first(self) -> T | None:
        """Return the first matching element, or ``None``."""
        ...

    def last(self) -> T | None:
        """Return the last matching element, or ``None``."""
        ...

    def sum(self) -> float | int:
        """Sum all (filtered) elements."""
        ...

    def sum_field(self, field_name: str) -> float:
        """Sum the numeric field *field_name* over matching dict records (GIL-free on RustObj)."""
        ...

    def mean(self) -> float | None:
        """Arithmetic mean of all (filtered) elements, or ``None`` if empty.

        For the f64 path with a single filter op the mean is computed in a
        single SIMD pass (sum and count accumulated simultaneously) — no
        intermediate Vec is allocated.

        Example::

            Query(data).filter(col > 0).mean()   # fast SIMD path for f64
            Query(data).skip(10).take(100).mean() # scalar fallback
        """
        ...

    def var(self) -> float | None:
        """Population variance (ddof=0, denominator N), or ``None`` if empty.

        Equivalent to ``numpy.var(arr, ddof=0)``.

        For f64 with a single filter op: single SIMD pass accumulating
        ``sum``, ``sum²``, and ``count`` simultaneously — no intermediate Vec.
        Variance is computed as ``E[X²] - E[X]²``.

        Example::

            Query(data).filter(col > 0).var()
        """
        ...

    def std(self) -> float | None:
        """Population standard deviation (ddof=0), or ``None`` if empty.

        Equivalent to ``numpy.std(arr, ddof=0)`` and ``sqrt(var())``.
        Returns ``None`` when the input is empty or all elements are filtered out.

        Example::

            Query(data).filter(col > 0).std()
        """
        ...

    def min(self) -> T | None:
        """Minimum of all (filtered) elements, or ``None`` if empty."""
        ...

    def max(self) -> T | None:
        """Maximum of all (filtered) elements, or ``None`` if empty."""
        ...

    def stats(self) -> dict[str, float | None]:
        """Compute count, sum, mean, min, and max in a single pass.

        Returns a dict with keys ``"count"`` (int), ``"sum"``, ``"mean"``,
        ``"min"``, ``"max"`` (float or ``None`` when empty).

        For the f64 fast path: single SIMD pass, GIL released, no intermediate Vec.

        Example::

            s = Query(data).filter(col > 0).stats()
            # {"count": 499_999, "sum": ..., "mean": ..., "min": ..., "max": ...}
        """
        ...

    def map_field(self, field_name: str) -> "Query[Any]":
        """Extract *field_name* from each dict record (equivalent to ``map(lambda r: r[field_name])``).

        On the ``ObjFieldPy`` path this fuses with a preceding field filter into a
        single Rust loop — no extra allocation.  On other paths it falls back to
        ``operator.itemgetter`` (C-level, faster than a Python lambda).

        Example::

            names = Query(records).filter(field("age") >= 18).map_field("name").to_list()
        """
        ...

    @overload
    def reduce(self, f: Callable[[T, T], T]) -> T: ...
    @overload
    def reduce(self, f: Callable[[U, T], U], initial: U) -> U: ...

    def for_each(self, f: Callable[[T], None]) -> None:
        """Call *f* on each element for side effects (no return value)."""
        ...

    def any(self, pred: Expr | FieldExpr | Callable[[T], bool]) -> bool:
        """Return ``True`` if any element satisfies *pred* (short-circuits)."""
        ...

    def all(self, pred: Expr | FieldExpr | Callable[[T], bool]) -> bool:
        """Return ``True`` if all elements satisfy *pred* (short-circuits)."""
        ...

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def explain(self) -> str:
        """Return a human-readable explanation of the query's execution path.

        Reports pipeline kind, queued operations, skip/take bounds, parallel flag,
        GIL classification, and allocation estimate.  Stable enough for tests.

        Example::

            q = Query(data).filter(col > 0).map(col * 2).take(1000)
            print(q.explain())
            # Query.explain()
            #   kind:     f64
            #   ops:      FilterGt(0.0) → MapMulScalar(2.0)
            #   skip:     0
            #   take:     1000
            #   parallel: false
            #   gil_free: true
            #   alloc:    1 Vec at terminal
        """
        ...

    # ------------------------------------------------------------------
    # Internal API — used by __init__.py monkey-patches
    # ------------------------------------------------------------------

    def _iter_parts(self) -> list[Any] | None:
        """Return ``[source, ops, skip, take]`` for Obj pipelines, else ``None``.

        Used by ``__init__.py`` to run filter/map loops in CPython's eval loop
        rather than crossing the PyO3 boundary per element.  Do not call from
        user code — this API may change without notice.
        """
        ...

    def __iter__(self) -> Iterator[T]: ...
    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------

def from_numpy(arr: Any) -> Query[Any]:
    """Create a :class:`Query` from a 1-D numpy ndarray.

    Uses the buffer protocol (one bulk memcpy, no per-element boxing) for
    ``float64``, ``int64``, ``bool`` and ``uint8`` arrays.  Other dtypes are
    cast first.
    """
    ...


def from_arrow(table_or_array: Any) -> Query[Any]:
    """Create a :class:`Query` from a PyArrow Array, ChunkedArray, or Table column.

    Null-free ``float64`` / ``int64`` arrays use the buffer protocol (GIL-free memcpy).
    Other types fall back to ``to_pylist()``.
    """
    ...


def from_csv(
    path_or_file: str | Path | Any,
    column: str | int | None = None,
    dtype: Literal["auto", "float", "int", "str"] = "auto",
    delimiter: str = ",",
    has_header: bool = True,
) -> Query[Any]:
    """Parse a CSV file into a :class:`Query` (GIL-free for path inputs).

    Parameters
    ----------
    path_or_file:
        ``str`` / ``Path`` → Rust reads + parses with GIL released.
        File-like → content read once (GIL), parsed in Rust (GIL released).
    column:
        Column name or 0-based index to extract.  ``None`` → dict-per-row.
    dtype:
        Value coercion for the extracted column (``"auto"`` = int → float → str).
    delimiter:
        Field separator (single character).
    has_header:
        Whether the first row contains column names.
    """
    ...


def from_json_lines(
    path_or_file: str | Path | Any,
    field: str | None = None,
    dtype: Literal["auto", "float", "int", "str"] = "auto",
) -> Query[Any]:
    """Parse a JSON Lines (NDJSON) file into a :class:`Query` (GIL-free for path inputs).

    Each line must be a JSON object.  If *field* is given, that field's value is
    extracted; otherwise each line becomes a dict row.
    """
    ...


def from_generator(gen: Iterable[T]) -> Query[T]:
    """Eagerly materialise a generator/iterable into a :class:`Query`."""
    ...


# ---------------------------------------------------------------------------
# Package version
# ---------------------------------------------------------------------------

__version__: str
