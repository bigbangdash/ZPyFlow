"""
Type stubs for ZPyFlow — enables IDE autocomplete, mypy, and pyright checking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Generator, Generic, Iterable, Iterator, Literal, TypeVar, overload

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
    def __mod__(self, other: float) -> Expr: ...
    def __floordiv__(self, other: float) -> Expr: ...
    def __neg__(self) -> Expr: ...
    def abs(self) -> Expr: ...
    def sqrt(self) -> Expr: ...
    def floor(self) -> Expr: ...
    def ceil(self) -> Expr: ...
    def round(self) -> Expr: ...
    def reciprocal(self) -> Expr: ...
    def between(self, lo: float, hi: float) -> Expr: ...
    def log(self) -> Expr: ...
    def log2(self) -> Expr: ...
    def log10(self) -> Expr: ...
    def exp(self) -> Expr: ...
    def sigmoid(self) -> Expr: ...
    def clamp(self, lo: float, hi: float) -> Expr: ...
    def is_nan(self) -> Expr: ...
    def not_nan(self) -> Expr: ...
    def is_finite(self) -> Expr: ...
    def is_inf(self) -> Expr: ...


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
    def __mod__(self, other: float) -> Expr: ...
    def __floordiv__(self, other: float) -> Expr: ...
    def __neg__(self) -> Expr: ...
    def abs(self) -> Expr: ...
    def sqrt(self) -> Expr: ...
    def floor(self) -> Expr: ...
    def ceil(self) -> Expr: ...
    def round(self) -> Expr: ...
    def reciprocal(self) -> Expr: ...
    def between(self, lo: float, hi: float) -> Expr: ...
    def log(self) -> Expr: ...
    def log2(self) -> Expr: ...
    def log10(self) -> Expr: ...
    def exp(self) -> Expr: ...
    def sigmoid(self) -> Expr: ...
    def clamp(self, lo: float, hi: float) -> Expr: ...
    def is_nan(self) -> Expr: ...
    def not_nan(self) -> Expr: ...
    def is_finite(self) -> Expr: ...
    def is_inf(self) -> Expr: ...


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
    def startswith(self, prefix: str) -> FieldExpr: ...
    def endswith(self, suffix: str) -> FieldExpr: ...
    def contains(self, sub: str) -> FieldExpr: ...
    def matches(self, pattern: str) -> FieldExpr: ...
    def __call__(self, row: dict[str, Any]) -> Any: ...


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


def agg_median(field_fn: Callable[[T], float]) -> Callable[["Query[T]"], float | None]:
    """GroupBy.agg reducer: median of ``field_fn(item)`` (average of middle two for even groups).

    Returns ``None`` for empty groups.

    Example::

        gb.agg(med=agg_median(lambda r: r["score"]))
    """
    ...


def agg_std(
    field_fn: Callable[[T], float], ddof: int = 0
) -> Callable[["Query[T]"], float | None]:
    """GroupBy.agg reducer: standard deviation (ddof=0 population, ddof=1 sample).

    Returns ``None`` for empty groups or groups smaller than ``ddof + 1``.
    """
    ...


def agg_first(
    field_fn: Callable[[T], Any] | None = None,
) -> Callable[["Query[T]"], Any]:
    """GroupBy.agg reducer: first element of the group (or ``field_fn(first)``).

    Returns ``None`` for empty groups.
    """
    ...


def agg_last(
    field_fn: Callable[[T], Any] | None = None,
) -> Callable[["Query[T]"], Any]:
    """GroupBy.agg reducer: last element of the group (or ``field_fn(last)``).

    Returns ``None`` for empty groups.
    """
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

    @staticmethod
    def iterate(fn: Callable[[T], T], seed: T) -> "Query[T]":
        """Infinite sequence: ``[seed, fn(seed), fn(fn(seed)), ...]``.

        Always combine with ``.take(n)`` or ``.take_while(pred)``.

        Example::

            Query.iterate(lambda x: x * 2, 1).take(6).to_list()
            # [1, 2, 4, 8, 16, 32]
        """
        ...

    @staticmethod
    def repeat(val: T, n: int | None = None) -> "Query[T]":
        """Repeat *val* exactly *n* times, or infinitely when ``n`` is ``None``.

        Example::

            Query.repeat(0.0, 5).to_list()          # [0.0, 0.0, 0.0, 0.0, 0.0]
            Query.repeat("x").take(3).to_list()     # ["x", "x", "x"]
        """
        ...

    @staticmethod
    def repeatedly(fn: Callable[[], T], n: int | None = None) -> "Query[T]":
        """Call ``fn()`` *n* times (or infinitely when ``n`` is ``None``).

        Example::

            import random
            Query.repeatedly(random.random, 5).to_list()  # 5 random floats
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

    def concat(self, other: Iterable[T]) -> Query[T]:
        """Concatenate *other* (any iterable) after this query.

        Unlike :meth:`chain`, *other* may be a plain list, generator, or any
        iterable — it is wrapped in a :class:`Query` automatically.

        Example::

            Query([1, 2]).concat([3, 4]).to_list()   # [1, 2, 3, 4]
            Query([1, 2]).concat(x for x in [3]).to_list()  # [1, 2, 3]
        """
        ...

    def chunk(self, n: int) -> "Query[list[T]]":
        """Split into fixed-size sublists of length *n* (last chunk may be shorter).

        Example::

            Query([1, 2, 3, 4, 5]).chunk(2).to_list()
            # [[1, 2], [3, 4], [5]]
        """
        ...

    def partition(
        self, pred: FieldExpr | Callable[[T], bool]
    ) -> tuple[list[T], list[T]]:
        """Split elements into ``(matching, non_matching)`` in a single pass.

        Returns a ``(yes_list, no_list)`` tuple.  *pred* must be a callable or
        :class:`FieldExpr`; numeric :class:`Expr` DSL is not supported.

        Example::

            evens, odds = Query(range(6)).partition(lambda x: x % 2 == 0)
            # evens=[0, 2, 4], odds=[1, 3, 5]
        """
        ...

    def sort(self, reverse: bool = False) -> "Query[T]":
        """Return a new Query with elements in sorted order.

        Example::

            Query([3, 1, 2]).sort().to_list()          # [1, 2, 3]
            Query([3, 1, 2]).sort(reverse=True).to_list()  # [3, 2, 1]
        """
        ...

    def sort_by(self, key_fn: Callable[[T], Any], reverse: bool = False) -> "Query[T]":
        """Return a new Query sorted by *key_fn*.

        Example::

            Query(records).sort_by(lambda r: r["score"]).to_list()
        """
        ...

    def distinct(self, key_fn: Callable[[T], Any] | None = None) -> "Query[T]":
        """Remove duplicates while preserving insertion order.

        *key_fn* extracts the comparison key; ``None`` compares elements directly.

        Example::

            Query([1, 2, 1, 3, 2]).distinct().to_list()          # [1, 2, 3]
            Query(records).distinct(lambda r: r["id"]).to_list()  # first occurrence per id
        """
        ...

    @overload
    def scan(self, f: Callable[[U, T], U], initial: U) -> "Query[U]": ...
    @overload
    def scan(self, f: Callable[[T, T], T], initial: T) -> "Query[T]": ...

    def scan(self, f: Callable[..., Any], initial: Any) -> "Query[Any]":
        """Return a Query of running accumulations (like reduce but yielding every step).

        The first yielded value is ``f(initial, items[0])``.

        Example::

            Query([1, 2, 3, 4]).scan(lambda acc, x: acc + x, 0).to_list()
            # [1, 3, 6, 10]  (cumulative sum)
        """
        ...

    def join(
        self,
        other: "Query[U]",
        on: "str | Callable[[Any], Any] | tuple[str | Callable[[Any], Any], str | Callable[[Any], Any]] | None" = None,
        how: Literal["inner", "left", "right", "cross"] = "inner",
    ) -> "Query[Any]":
        """SQL-style hash join between two Queries.

        Result rows are merged dicts (right wins on collision) when both sides
        are dicts; otherwise tuples ``(left, right)``.

        Parameters
        ----------
        other:
            Right-hand Query to join against.
        on:
            Key spec — string ``"field"``, callable, or 2-tuple for different keys
            per side.  Required for inner/left/right; omit for ``how="cross"``.
        how:
            ``"inner"`` (default), ``"left"``, ``"right"``, or ``"cross"``.

        Example::

            orders  = Query([{"id": 1, "item": "A"}, {"id": 2, "item": "B"}])
            details = Query([{"id": 1, "price": 9.9}])

            orders.join(details, on="id").to_list()
            # [{"id": 1, "item": "A", "price": 9.9}]

            orders.join(details, on="id", how="left").to_list()
            # [{"id": 1, "item": "A", "price": 9.9},
            #  {"id": 2, "item": "B", "price": None}]
        """
        ...

    def inner_join(
        self,
        other: "Query[U]",
        left_key: Callable[[T], Any],
        right_key: Callable[[U], Any] | None = None,
    ) -> "Query[tuple[T, U]]":
        """Hash inner-join — yield ``(left, right)`` for every matching key pair.

        Only rows whose key appears on both sides are included.

        Example::

            users  = Query([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
            orders = Query([{"user_id": 1, "item": "book"}, {"user_id": 1, "item": "pen"}])
            result = users.inner_join(orders, left_key=lambda u: u["id"],
                                              right_key=lambda o: o["user_id"]).to_list()
            # [({'id':1,'name':'Alice'}, {'user_id':1,'item':'book'}),
            #  ({'id':1,'name':'Alice'}, {'user_id':1,'item':'pen'})]
        """
        ...

    def left_join(
        self,
        other: "Query[U]",
        left_key: Callable[[T], Any],
        right_key: Callable[[U], Any] | None = None,
    ) -> "Query[tuple[T, U | None]]":
        """Hash left-join — yield ``(left, right)`` for matches, ``(left, None)`` otherwise.

        All rows from *self* are preserved. Rows with no matching key in *other*
        produce ``(left, None)``.

        Example::

            result = users.left_join(orders, left_key=lambda u: u["id"],
                                             right_key=lambda o: o["user_id"]).to_list()
        """
        ...

    def tee(self, n: int = 2) -> "tuple[Query[T], ...]":
        """Materialise once and return *n* independent Query copies.

        Example::

            q1, q2, q3 = Query(data).filter(col > 0).tee(3)
            total  = q1.sum()
            top10  = q2.sort(reverse=True).take(10).to_list()
            count  = q3.count()
        """
        ...

    def flatten(self) -> "Query[Any]":
        """Expand each element one level, yielding its items individually.

        Strings are treated as scalars (not character-expanded).
        Non-iterable elements are passed through unchanged.

        Example::

            Query([[1, 2], [3, 4]]).flatten().to_list()  # [1, 2, 3, 4]
        """
        ...

    def partition_by(
        self, key_fn: Callable[[T], Any] | None = None
    ) -> "Query[list[T]]":
        """Group consecutive elements with the same key value into sublists.

        Clojure analogue: ``(partition-by f coll)``.
        *key_fn* extracts the grouping key; ``None`` compares elements directly.

        Example::

            Query([1, 1, 2, 2, 3, 1, 1]).partition_by().to_list()
            # [[1, 1], [2, 2], [3], [1, 1]]
        """
        ...

    def dedupe(self, key_fn: Callable[[T], Any] | None = None) -> "Query[T]":
        """Remove *consecutive* duplicate elements (non-consecutive duplicates kept).

        Clojure analogue: ``(dedupe coll)``.
        *key_fn* extracts the comparison key; ``None`` compares elements directly.

        Example::

            Query([1, 1, 2, 2, 3, 1, 1]).dedupe().to_list()
            # [1, 2, 3, 1]
        """
        ...

    def cycle(self, n: int | None = None) -> "Query[T]":
        """Repeat the sequence *n* times (or infinitely when ``n`` is ``None``).

        Example::

            Query([1, 2, 3]).cycle(2).to_list()       # [1, 2, 3, 1, 2, 3]
            Query([1, 2]).cycle().take(5).to_list()   # [1, 2, 1, 2, 1]
        """
        ...

    def step_by(self, n: int) -> "Query[T]":
        """Return every *n*-th element (0, n, 2n, …).

        Example::

            Query(range(10)).step_by(3).to_list()  # [0, 3, 6, 9]
        """
        ...

    def interleave(self, other: "Query[U]") -> "Query[T | U]":
        """Interleave elements from *self* and *other*, stopping at the shorter one.

        Example::

            Query([1, 2, 3]).interleave(Query([10, 20, 30])).to_list()
            # [1, 10, 2, 20, 3, 30]
        """
        ...

    def sample(self, n: int, seed: int | None = None) -> "Query[T]":
        """Return *n* randomly chosen elements without replacement.

        Example::

            Query(range(100)).sample(5, seed=42).to_list()
        """
        ...

    def cache(self) -> "Query[T]":
        """Materialise the pipeline into an in-memory list and return a new Query.

        Use when the same dataset will be queried multiple times to avoid
        re-scanning the source on every terminal call.

        Example::

            q = Query(large_data).filter(col > 0).cache()
            count_above_1 = q.filter(col > 1).count()
            total         = q.sum()
        """
        ...

    def set_field(self, name: str, fn: Callable[[Any], Any]) -> "Query[T]":
        """Apply ``fn(old_value)`` to field *name* in each dict, returning a modified dict.

        Fields that do not exist receive ``fn(None)``.

        Example::

            Query(products).set_field("price", lambda v: round(v * 1.1, 2))
        """
        ...

    def add_field(self, name: str, fn: Callable[[T], Any]) -> "Query[T]":
        """Add a new field *name = fn(record)* to each dict.

        *fn* receives the whole record and returns the new field value.

        Example::

            Query(orders).add_field("total", lambda r: r["price"] * r["qty"])
        """
        ...

    def drop_field(self, *names: str) -> "Query[T]":
        """Remove the specified fields from each dict.

        Example::

            Query(users).drop_field("password", "token")
        """
        ...

    def select(self, *fields: str) -> "Query[T]":
        """Keep only the specified fields in each dict (others are dropped).

        Fields that do not exist are silently omitted. Field order follows *fields*.

        Example::

            Query(users).select("id", "name").to_list()
            # [{"id": 1, "name": "Alice"}, ...]
        """
        ...

    def rename_field(self, old: str, new: str) -> "Query[T]":
        """Rename field *old* to *new* in each dict.

        Records that do not contain *old* are passed through unchanged.

        Example::

            Query(records).rename_field("user_id", "id").to_list()
        """
        ...

    def value_counts(
        self, key_fn: Callable[[T], Any] | None = None
    ) -> dict[Any, int]:
        """Count occurrences of each element (or key) and return ``{value: count}``.

        *key_fn* extracts the counting key; ``None`` uses the element itself.

        Example::

            Query(["a", "b", "a"]).value_counts()         # {"a": 2, "b": 1}
            Query(records).value_counts(lambda r: r["status"])
        """
        ...

    def sliding_window(self, n: int) -> "Query[tuple[T, ...]]":
        """Yield overlapping tuples of *n* consecutive elements.

        Produces ``max(0, len - n + 1)`` windows.  Fewer than *n* elements yields
        an empty Query.

        Example::

            Query([1, 2, 3, 4]).sliding_window(2).to_list()
            # [(1, 2), (2, 3), (3, 4)]
        """
        ...

    def window(self, size: int, step: int = 1) -> "Query[list[T]]":
        """Sliding or tumbling window — generalised :meth:`sliding_window`.

        Returns lists (not tuples) and supports a *step* parameter.
        ``step=1`` is rolling; ``step=size`` is tumbling (non-overlapping).

        Example::

            Query([1, 2, 3, 4, 5]).window(3).to_list()
            # [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

            Query([1, 2, 3, 4]).window(2, step=2).to_list()
            # [[1, 2], [3, 4]]
        """
        ...

    def rolling_sum(self, window: int) -> "Query[float]":
        """Sliding-window sum — O(N) running-sum kernel.

        Produces ``len - window + 1`` values.  Rust SIMD path for F64 data.

        Example::

            Query([1.0, 2.0, 3.0, 4.0]).rolling_sum(2).to_list()
            # [3.0, 5.0, 7.0]
        """
        ...

    def rolling_mean(self, window: int) -> "Query[float]":
        """Sliding-window mean — O(N) running-sum kernel.

        Produces ``len - window + 1`` values.  Rust SIMD path for F64 data.

        Example::

            Query([1.0, 2.0, 3.0, 4.0]).rolling_mean(2).to_list()
            # [1.5, 2.5, 3.5]
        """
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
        """Convert dict records to columnar layout (pay GIL cost once, query many times).

        Transforms a list-of-dicts into typed column slices (``ColumnarObj``).
        Subsequent ``field()`` DSL filters scan column arrays directly — no
        per-row Python dict lookup.

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
    # Convenience methods (spec-077)
    # ------------------------------------------------------------------

    def filter_map(self, fn: Callable[[T], U | None]) -> "Query[U]":
        """Apply *fn* to each element; keep only non-``None`` results."""
        ...

    def tap(self, fn: Callable[[T], Any]) -> "Query[T]":
        """Call *fn* on each element for side effects; pass elements through."""
        ...

    def compact(self, falsy: bool = False) -> "Query[T]":
        """Remove ``None`` values (default) or all falsy values when *falsy=True*."""
        ...

    def min_by(self, key_fn: Callable[[T], Any]) -> T | None:
        """Return the element for which *key_fn* is smallest, or ``None`` if empty."""
        ...

    def max_by(self, key_fn: Callable[[T], Any]) -> T | None:
        """Return the element for which *key_fn* is largest, or ``None`` if empty."""
        ...

    def unzip(self) -> tuple[list[Any], list[Any]]:
        """Split a stream of ``(a, b)`` tuples into ``([a…], [b…])``."""
        ...

    def median(self) -> float | None:
        """Return the median value, or ``None`` if empty."""
        ...

    def product(self) -> float:
        """Return the product of all elements (1 for empty pipeline)."""
        ...

    def find(self, pred: Callable[[T], bool]) -> T | None:
        """Return the first element matching *pred*, or ``None`` if not found.

        Short-circuits — iteration stops as soon as a match is found.
        """
        ...

    def count_if(self, pred: Callable[[T], bool]) -> int:
        """Count elements satisfying *pred* in a single pass."""
        ...

    def sum_by(self, fn: Callable[[T], float]) -> float:
        """Sum of ``fn(item)`` over all elements in a single pass."""
        ...

    def mean_by(self, fn: Callable[[T], float]) -> float | None:
        """Arithmetic mean of ``fn(item)``; ``None`` if empty."""
        ...

    # ------------------------------------------------------------------
    # Terminal operations
    # ------------------------------------------------------------------

    def to_list(self) -> list[T]:
        """Materialise the pipeline into a Python list (allocates once)."""
        ...

    def to_arrow(self) -> Any:
        """Materialise as a PyArrow Array or RecordBatch.

        - **F64 path** — raw bytes via ``to_bytes()``, zero-copy into Arrow buffer.
          Returns ``pyarrow.Array<float64>``.
        - **ColumnarObj path** (after ``.preload()``) — typed column arrays →
          ``pyarrow.RecordBatch`` with inferred schema (no per-row dict reconstruction).
        - **Other paths** — ``to_list()`` with Arrow type inference.

        Requires ``pyarrow``.

        Example::

            arr = Query([1.0, 2.0, 3.0]).filter(col > 1.0).to_arrow()
            # pyarrow.Array<float64>: [2.0, 3.0]

            rb = Query(logs).preload().filter(field("score") > 0.5).to_arrow()
            # pyarrow.RecordBatch with schema {score: float64, ...}
        """
        ...

    def to_polars(self) -> Any:
        """Materialise as a Polars Series (numeric) or DataFrame (object).

        Delegates to ``to_arrow()`` and wraps via ``polars.from_arrow()``.
        Requires ``polars``.

        Example::

            s = Query([1.0, 2.0, 3.0]).filter(col > 1.0).to_polars()
            # polars.Series: [2.0, 3.0]
        """
        ...

    def to_pandas(self) -> Any:
        """Materialise as a pandas Series.

        Delegates to ``to_arrow().to_pandas()``.
        Requires ``pandas`` and ``pyarrow``.

        Example::

            s = Query([1.0, 2.0, 3.0]).filter(col > 1.0).to_pandas()
            # pandas.Series: [2.0, 3.0]
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


def from_csv_chunked(
    path_or_file: str | Path | Any,
    chunk_size: int = 10_000,
    column: str | int | None = None,
    dtype: Literal["auto", "float", "int", "str"] = "auto",
    delimiter: str = ",",
    has_header: bool = True,
) -> Generator[Query[Any], None, None]:
    """Stream a large CSV as an iterator of :class:`Query` objects (spec-081 T1).

    Yields one ``Query`` per ``chunk_size`` rows.  Memory is bounded by chunk
    size — the file is never fully loaded into RAM.

    Parameters
    ----------
    path_or_file:
        ``str`` / ``Path`` or text-mode file-like.
    chunk_size:
        Number of data rows per yielded ``Query`` (last chunk may be smaller).
    column:
        Column name or 0-based index to extract.  ``None`` → dict-per-row.
    dtype:
        Value coercion for the extracted column (``"auto"`` = int → float → str).
    delimiter:
        Field separator (single character).
    has_header:
        Whether the first row contains column names.

    Example::

        total = 0
        for q in from_csv_chunked("large.csv", chunk_size=50_000):
            total += q.count()
    """
    ...


def from_arrow_ipc(
    path: str | Path,
    column: str | int | None = None,
) -> Query[Any]:
    """Read an Arrow IPC file or stream and return a :class:`Query`.

    Supports both Arrow **file** format (random-access) and Arrow **stream**
    format.  Single-column numeric files are read zero-copy via the buffer
    protocol.  Multi-column tables become dict-row Queries.

    Parameters
    ----------
    path:
        Path to the ``.arrow`` / ``.arrows`` file.
    column:
        Column name or 0-based index to extract.  *None* auto-extracts when
        the file has exactly one column; otherwise returns dict rows.
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
