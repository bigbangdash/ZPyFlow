"""
ZPyFlow — Zero-allocation lazy query pipelines for Python, powered by Rust.

Quick start::

    from zpyflow import Query, col

    data = [float(i) for i in range(1_000_000)]

    result = (
        Query(data)
            .filter(col > 500_000)    # SIMD filter, GIL released
            .map(col * 2.0)           # SIMD map,    GIL released
            .take(10_000)
            .to_list()
    )

    # Or with Python lambdas (GIL held; throughput is comparable to pure Python list comps):
    result = (
        Query(data)
            .filter(lambda x: x > 0.5)
            .map(lambda x: x * 2.0)
            .take(10_000)
            .to_list()
    )

    # Parallel execution (numeric fast path only):
    result = Query(data).filter(col > 0).parallel().to_list()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from ._zpyflow import Query, col, Expr, ColProxy, AggSpec, FieldExpr, field, __version__
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "ZPyFlow native extension not found. "
        "Build it with: `maturin develop` or `pip install zpyflow`"
    ) from e

import itertools as _itertools

# Save the Rust implementations before any Python overrides.
_to_list_rust = Query.to_list
_count_rust = Query.count

from .adapters import from_numpy, from_arrow, from_csv, from_json_lines, from_generator
from .groupby import GroupBy, agg_count, agg_sum, agg_mean, agg_max, agg_min


def _query_to_list(self):
    """Override to_list() for Obj path: run hot loop in CPython, not through Rust boundary.

    For numeric/materialized paths (_iter_parts returns None) delegates to the Rust
    implementation.  For Obj paths with Python lambda ops, running the loop in pure
    Python is measurably faster because it avoids PyO3 boundary overhead per element.
    """
    parts = self._iter_parts()
    if parts is None:
        return _to_list_rust(self)

    source, ops, skip, take = parts

    # ── no ops, no bounds ────────────────────────────────────────────────────
    if not ops and skip == 0 and take is None:
        return list(source)

    # ── all-filter, no skip ──────────────────────────────────────────────────
    if skip == 0 and all(is_filter for is_filter, _ in ops):
        fns = [fn for _, fn in ops]
        if len(fns) == 1:
            fn = fns[0]
            if take is None:
                return [item for item in source if fn(item)]
            return list(_itertools.islice((item for item in source if fn(item)), take))
        if not fns:
            return list(source) if take is None else list(_itertools.islice(source, take))
        if take is None:
            return [item for item in source if all(f(item) for f in fns)]
        return list(_itertools.islice(
            (item for item in source if all(f(item) for f in fns)), take
        ))

    # ── 1-filter + 1-map, no skip — very common pattern ─────────────────────
    if len(ops) == 2 and ops[0][0] and not ops[1][0] and skip == 0:
        pred, mapper = ops[0][1], ops[1][1]
        if take is None:
            return [mapper(item) for item in source if pred(item)]
        return list(_itertools.islice(
            (mapper(item) for item in source if pred(item)), take
        ))

    # ── general case ─────────────────────────────────────────────────────────
    return list(_py_obj_gen(source, ops, skip, take))


def _query_count(self):
    """Override count() for Obj path: same rationale as _query_to_list."""
    parts = self._iter_parts()
    if parts is None:
        return _count_rust(self)

    source, ops, skip, take = parts

    if not ops and skip == 0:
        n = len(source) if hasattr(source, '__len__') else sum(1 for _ in source)
        return n if take is None else min(n, take)

    if skip == 0 and all(is_filter for is_filter, _ in ops):
        fns = [fn for _, fn in ops]
        if len(fns) == 1:
            fn = fns[0]
            gen = (1 for item in source if fn(item))
        else:
            gen = (1 for item in source if all(f(item) for f in fns))
        if take is None:
            return sum(gen)
        count = 0
        for _ in gen:
            count += 1
            if count >= take:
                return count
        return count

    return sum(1 for _ in _py_obj_gen(source, ops, skip, take))


def _py_obj_gen(source, ops, skip, take):
    """Generator for object-path with actual ops / skip / take.

    Runs entirely in CPython's eval loop — no Rust/PyO3 boundary per element.
    Only invoked when there is at least one op, a skip, or a take to apply.
    """
    if take is not None and take == 0:
        return
    count = skipped = 0
    for item in source:
        survived = True
        for is_filter, fn in ops:
            if is_filter:
                if not fn(item):
                    survived = False
                    break
            else:
                item = fn(item)
        if not survived:
            continue
        if skipped < skip:
            skipped += 1
            continue
        yield item
        count += 1
        if take is not None and count >= take:
            return


def _query_iter(self):
    """Return an iterator over this Query's elements.

    - No ops, no skip, no take  → ``iter(source)``  (direct list iterator)
    - All-filter, no skip/take  → native genexpr, avoids generator frame overhead
    - All-filter + take only    → ``itertools.islice(genexpr, take)``
    - Everything else           → ``_py_obj_gen``
    For numeric / materialized paths → ``iter(to_list())``.
    """
    parts = self._iter_parts()
    if parts is None:
        return iter(_to_list_rust(self))

    source, ops, skip, take = parts

    if not ops and skip == 0 and take is None:
        return iter(source)

    if skip == 0 and all(is_filter for is_filter, _ in ops):
        fns = [fn for _, fn in ops]
        if len(fns) == 1:
            fn = fns[0]
            gen = (item for item in source if fn(item))
        else:
            def _all_pass(item, _fns=fns):
                for f in _fns:
                    if not f(item):
                        return False
                return True
            gen = (item for item in source if _all_pass(item))
        return gen if take is None else _itertools.islice(gen, take)

    return _py_obj_gen(source, ops, skip, take)


def _query_group_by(self, key_fn):
    """Attach as Query.group_by for fused filter+group in a single Python loop.

    For the common case (Obj path, all-filter ops, no skip/take) this fuses
    filter and grouping into ONE loop — same structure as Python's
    ``Counter(key(x) for x in src if pred(x))``.

    Falls back to ``GroupBy(iter(self), key_fn)`` for other paths.
    """
    from collections import defaultdict

    parts = self._iter_parts()
    if parts is None:
        return GroupBy(self, key_fn)

    source, ops, skip, take = parts

    # ── Fused fast path: all-filter, no skip/take ─────────────────────────
    if skip == 0 and take is None and all(is_filter for is_filter, _ in ops):
        groups = defaultdict(list)
        fns = [fn for _, fn in ops]

        if not fns:
            for item in source:
                groups[key_fn(item)].append(item)
        elif len(fns) == 1:
            fn = fns[0]
            for item in source:
                if fn(item):
                    groups[key_fn(item)].append(item)
        else:
            for item in source:
                if all(f(item) for f in fns):
                    groups[key_fn(item)].append(item)

        return GroupBy._from_dict(groups)

    # ── General path: use __iter__ ────────────────────────────────────────
    return GroupBy(self, key_fn)


# Attach __iter__ so Python code can iterate any Query directly.
# This enables: list(query), for x in query, GroupBy(query, ...), etc.
Query.__iter__ = _query_iter

# Override to_list() and count() for Obj path: pure-Python hot loop is faster
# than Rust's collect_py_lazy for Python lambda callbacks (no PyO3 overhead/element).
Query.to_list = _query_to_list
Query.count   = _query_count

Query.group_by = _query_group_by


def _query_concat(self, other):
    """Concatenate this query with any iterable (list, generator, or Query).

    Unlike ``chain()``, *other* may be any iterable — it is materialised into
    a ``Query`` first when it is not already one.
    """
    if not isinstance(other, Query):
        other = Query(list(other))
    return self.chain(other)


Query.concat = _query_concat


def _query_chunk(self, n: int):
    """Split into fixed-size sublists of length *n* (last chunk may be shorter)."""
    if n < 1:
        raise ValueError(f"chunk size must be >= 1, got {n}")
    buf = []
    for item in self:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def _query_chunk_terminal(self, n: int):
    return Query(list(_query_chunk(self, n)))


Query.chunk = _query_chunk_terminal


def _query_partition(self, pred):
    """Split elements into (matching, non-matching) lists in a single pass.

    *pred* must be a callable (lambda/function) or :class:`FieldExpr`.
    Numeric :class:`Expr` DSL is not supported — use ``lambda x: x > n`` instead.
    """
    yes, no = [], []
    for item in self:
        (yes if pred(item) else no).append(item)
    return yes, no


Query.partition = _query_partition


def _query_sort(self, reverse: bool = False):
    """Return a new Query with elements in sorted order."""
    return Query(sorted(self, reverse=reverse))


def _query_sort_by(self, key_fn, reverse: bool = False):
    """Return a new Query with elements sorted by *key_fn*."""
    return Query(sorted(self, key=key_fn, reverse=reverse))


Query.sort = _query_sort
Query.sort_by = _query_sort_by


def _query_distinct(self, key_fn=None):
    """Return a new Query with duplicates removed (insertion order preserved).

    *key_fn* extracts the value used for equality comparison; ``None`` compares
    elements directly.
    """
    seen = set()
    result = []
    for item in self:
        key = key_fn(item) if key_fn is not None else item
        if key not in seen:
            seen.add(key)
            result.append(item)
    return Query(result)


Query.distinct = _query_distinct


def _query_scan(self, f, initial):
    """Return a Query of running accumulations (like reduce but yielding every step).

    The first element of the result is ``f(initial, first_item)``.
    """
    result = []
    acc = initial
    for item in self:
        acc = f(acc, item)
        result.append(acc)
    return Query(result)


Query.scan = _query_scan


def _query_sliding_window(self, n: int):
    """Yield overlapping tuples of *n* consecutive elements.

    Produces ``len(source) - n + 1`` windows; fewer than *n* elements yields nothing.
    """
    if n < 1:
        raise ValueError(f"sliding_window size must be >= 1, got {n}")
    from collections import deque
    buf: deque = deque()
    result = []
    for item in self:
        buf.append(item)
        if len(buf) == n:
            result.append(tuple(buf))
            buf.popleft()
    return Query(result)


Query.sliding_window = _query_sliding_window


def _query_set_field(self, name: str, fn):
    """Return a new Query where each dict has ``name`` replaced by ``fn(old_value)``.

    If the field does not exist, ``fn(None)`` is called.
    Non-dict elements are passed through unchanged.

    Example::

        Query(products).set_field("price", lambda v: round(v * 1.1, 2))
    """
    def _transform(item):
        if not isinstance(item, dict):
            return item
        old = item.get(name)
        return {**item, name: fn(old)}
    return Query(list(_transform(x) for x in self))


def _query_add_field(self, name: str, fn):
    """Return a new Query where each dict gains a new field ``name = fn(record)``.

    *fn* receives the whole record and returns the new field value.
    Non-dict elements are passed through unchanged.

    Example::

        Query(orders).add_field("total", lambda r: r["price"] * r["qty"])
    """
    def _transform(item):
        if not isinstance(item, dict):
            return item
        return {**item, name: fn(item)}
    return Query(list(_transform(x) for x in self))


def _query_drop_field(self, *names: str):
    """Return a new Query with the specified fields removed from each dict.

    Non-dict elements are passed through unchanged.

    Example::

        Query(users).drop_field("password", "token")
    """
    name_set = set(names)
    def _transform(item):
        if not isinstance(item, dict):
            return item
        return {k: v for k, v in item.items() if k not in name_set}
    return Query(list(_transform(x) for x in self))


def _query_cache(self):
    """Materialise the pipeline into a new Query backed by an in-memory list.

    Useful when the same dataset will be queried multiple times — the data is
    scanned once, and subsequent operations start from the cached list.

    The resulting Query preserves the typed path when possible:
    - f64/i64 pipelines stay on the numeric fast path
    - All other paths produce a Python-list-backed Query

    Example::

        q = Query(large_data).filter(col > 0).cache()
        count_above_1 = q.filter(col > 1).count()   # no re-scan of large_data
        total         = q.sum()
    """
    return Query(self.to_list())


Query.cache = _query_cache


def _query_tee(self, n: int = 2):
    """Materialise the pipeline once and return *n* independent copies as a tuple.

    Each copy is a separate ``Query`` backed by the same in-memory list.
    Useful when you need to apply different operations to the same data
    without scanning the source more than once.

    Example::

        q1, q2, q3 = Query(data).filter(col > 0).tee(3)
        total = q1.sum()
        top10 = q2.sort(reverse=True).take(10).to_list()
        count = q3.count()
    """
    if n < 1:
        raise ValueError(f"tee requires n >= 1, got {n}")
    materialized = self.to_list()
    return tuple(Query(list(materialized)) for _ in range(n))


Query.tee = _query_tee


def _query_flatten(self):
    """Expand each element one level, yielding its items as individual elements.

    Elements that are not iterable (or are strings) are yielded as-is.

    Example::

        Query([[1, 2], [3, 4], [5]]).flatten().to_list()  # [1, 2, 3, 4, 5]
        Query([range(3), range(2)]).flatten().to_list()   # [0, 1, 2, 0, 1]
    """
    result = []
    for item in self:
        if isinstance(item, str):
            result.append(item)
        else:
            try:
                result.extend(item)
            except TypeError:
                result.append(item)
    return Query(result)


Query.flatten = _query_flatten


_SENTINEL = object()  # unique sentinel for dedupe/partition_by first-element detection


def _query_partition_by(self, key_fn=None):
    """Group consecutive elements that share the same key value into sublists.

    Similar to Clojure's ``(partition-by f coll)``.

    *key_fn* extracts the grouping key; ``None`` uses the element itself.

    Example::

        Query([1, 1, 2, 2, 3, 1, 1]).partition_by().to_list()
        # [[1, 1], [2, 2], [3], [1, 1]]

        Query(logs).partition_by(lambda r: r["level"]).to_list()
        # groups consecutive log records by level
    """
    result = []
    current_key = _SENTINEL
    current_group = []
    for item in self:
        key = key_fn(item) if key_fn is not None else item
        if key != current_key:
            if current_group:
                result.append(current_group)
            current_group = [item]
            current_key = key
        else:
            current_group.append(item)
    if current_group:
        result.append(current_group)
    return Query(result)


def _query_dedupe(self, key_fn=None):
    """Remove consecutive duplicate elements (non-consecutive duplicates are kept).

    Similar to Clojure's ``(dedupe coll)``.

    *key_fn* extracts the comparison key; ``None`` compares elements directly.

    Example::

        Query([1, 1, 2, 2, 3, 1, 1]).dedupe().to_list()
        # [1, 2, 3, 1]  — the trailing 1s collapse, but middle 1 is kept

        Query(events).dedupe(lambda e: e["type"]).to_list()
        # deduplicate by event type for consecutive runs
    """
    last_key = _SENTINEL
    result = []
    for item in self:
        key = key_fn(item) if key_fn is not None else item
        if key != last_key:
            result.append(item)
            last_key = key
    return Query(result)


Query.partition_by = _query_partition_by
Query.dedupe = _query_dedupe


# ---------------------------------------------------------------------------
# Infinite sequence factories (Clojure-style)
# ---------------------------------------------------------------------------

def _query_iterate(fn, seed):
    """Create a Query that lazily generates ``[seed, fn(seed), fn(fn(seed)), ...]``.

    This is an *infinite* sequence — always combine with ``.take(n)`` or
    ``.take_while(pred)`` to terminate.

    Clojure analogue: ``(iterate inc 0)``

    Example::

        Query.iterate(lambda x: x * 2, 1).take(6).to_list()
        # [1, 2, 4, 8, 16, 32]
    """
    def _gen():
        val = seed
        while True:
            yield val
            val = fn(val)
    return Query(from_generator(_gen()))


def _query_repeat(val, n=None):
    """Create a Query of *val* repeated *n* times (or infinitely when ``n=None``).

    Clojure analogue: ``(repeat 5 42)``

    Example::

        Query.repeat(0.0, 5).to_list()    # [0.0, 0.0, 0.0, 0.0, 0.0]
        Query.repeat("x").take(3).to_list()  # ["x", "x", "x"]
    """
    if n is not None:
        return Query([val] * n)
    import itertools
    return Query(from_generator(itertools.repeat(val)))


def _query_repeatedly(fn, n=None):
    """Create a Query by calling ``fn()`` repeatedly.

    *n* limits the number of calls; ``None`` means infinite (combine with ``.take``).

    Clojure analogue: ``(repeatedly 3 rand)``

    Example::

        import random
        Query.repeatedly(random.random, 5).to_list()  # 5 random floats
    """
    if n is not None:
        return Query([fn() for _ in range(n)])
    def _gen():
        while True:
            yield fn()
    return Query(from_generator(_gen()))


def _query_cycle(self, n=None):
    """Repeat the pipeline's elements *n* times (or infinitely when ``n`` is ``None``).

    Clojure analogue: ``(cycle coll)``
    For infinite cycling, always combine with ``.take(k)``.

    Example::

        Query([1, 2, 3]).cycle(2).to_list()      # [1, 2, 3, 1, 2, 3]
        Query([1, 2]).cycle().take(7).to_list()  # [1, 2, 1, 2, 1, 2, 1]
    """
    import itertools
    materialized = self.to_list()
    if not materialized:
        return Query([])
    if n is not None:
        return Query(list(itertools.islice(itertools.cycle(materialized), len(materialized) * n)))
    return Query(from_generator(itertools.cycle(materialized)))


def _query_step_by(self, n: int):
    """Return every *n*-th element (0-based: elements at indices 0, n, 2n, …).

    Example::

        Query(range(10)).step_by(3).to_list()  # [0, 3, 6, 9]
    """
    if n < 1:
        raise ValueError(f"step_by requires n >= 1, got {n}")
    return Query([item for i, item in enumerate(self) if i % n == 0])


def _query_interleave(self, other):
    """Interleave elements from *self* and *other*, stopping at the shorter one.

    Example::

        Query([1, 2, 3]).interleave(Query([10, 20, 30])).to_list()
        # [1, 10, 2, 20, 3, 30]
    """
    result = []
    for a, b in zip(self, other):
        result.append(a)
        result.append(b)
    return Query(result)


def _query_sample(self, n: int, seed=None):
    """Return *n* elements chosen without replacement (random order).

    *seed* seeds the random number generator for reproducibility.

    Example::

        Query(range(100)).sample(5, seed=42).to_list()
    """
    import random
    items = self.to_list()
    if n > len(items):
        raise ValueError(f"sample size {n} exceeds population size {len(items)}")
    rng = random.Random(seed)
    return Query(rng.sample(items, n))


Query.cycle = _query_cycle
Query.step_by = _query_step_by
Query.interleave = _query_interleave
Query.sample = _query_sample

Query.iterate = staticmethod(_query_iterate)
Query.repeat = staticmethod(_query_repeat)
Query.repeatedly = staticmethod(_query_repeatedly)


Query.set_field = _query_set_field
Query.add_field = _query_add_field
Query.drop_field = _query_drop_field


def _query_select(self, *fields: str):
    """Return a new Query keeping only the specified fields in each dict.

    Fields that do not exist in a record are omitted silently.
    Non-dict elements are passed through unchanged.
    Preserves the order of *fields* as given.

    Example::

        Query(users).select("id", "name").to_list()
        # [{"id": 1, "name": "Alice"}, ...]
    """
    def _transform(item):
        if not isinstance(item, dict):
            return item
        return {k: item[k] for k in fields if k in item}
    return Query(list(_transform(x) for x in self))


def _query_rename_field(self, old: str, new: str):
    """Return a new Query with field *old* renamed to *new* in each dict.

    If *old* does not exist, the record is passed through unchanged.
    Non-dict elements are passed through unchanged.

    Example::

        Query(records).rename_field("user_id", "id").to_list()
    """
    def _transform(item):
        if not isinstance(item, dict) or old not in item:
            return item
        result = {k: v for k, v in item.items() if k != old}
        result[new] = item[old]
        return result
    return Query(list(_transform(x) for x in self))


Query.select = _query_select
Query.rename_field = _query_rename_field


def _query_value_counts(self, key_fn=None):
    """Count occurrences of each element (or key) and return ``{value: count}``.

    *key_fn* extracts the key used for counting; ``None`` uses the element itself.

    Example::

        Query(["a", "b", "a", "c", "a"]).value_counts()
        # {"a": 3, "b": 1, "c": 1}

        Query(records).value_counts(lambda r: r["status"])
        # {"active": 42, "inactive": 8}
    """
    from collections import Counter
    if key_fn is None:
        return dict(Counter(self))
    return dict(Counter(key_fn(x) for x in self))


Query.value_counts = _query_value_counts


def _query_inner_join(self, other, left_key, right_key=None):
    """Hash inner-join: yield (left, right) tuples for every matching key pair.

    *left_key* / *right_key* extract the join key from each side.
    If *right_key* is omitted, *left_key* is used for both.
    Only rows whose key appears on both sides are included.
    """
    from collections import defaultdict
    if right_key is None:
        right_key = left_key
    index = defaultdict(list)
    for item in other:
        index[right_key(item)].append(item)
    result = []
    for left in self:
        k = left_key(left)
        for right in index[k]:
            result.append((left, right))
    return Query(result)


def _query_left_join(self, other, left_key, right_key=None):
    """Hash left-join: yield (left, right) for every match, or (left, None) if no match.

    All rows from *self* are included. Rows with no matching key in *other*
    produce ``(left, None)``.
    """
    from collections import defaultdict
    if right_key is None:
        right_key = left_key
    index = defaultdict(list)
    for item in other:
        index[right_key(item)].append(item)
    result = []
    for left in self:
        k = left_key(left)
        matches = index.get(k)
        if matches:
            for right in matches:
                result.append((left, right))
        else:
            result.append((left, None))
    return Query(result)


Query.inner_join = _query_inner_join
Query.left_join = _query_left_join


def _query_group_agg(self, key_fn, **specs):
    """Single-pass group + aggregate using the Rust kernel.

    Each spec must be an `AggSpec` object (from `agg_count()`, `agg_sum(fn)`, etc.).
    Returns `list[dict]` with `"_key"` plus one entry per named spec.

    Example::

        from zpyflow import Query, agg_count, agg_sum

        result = (
            Query(products)
                .group_agg(
                    lambda p: p["category"],
                    count   = agg_count(),
                    revenue = agg_sum(lambda p: p["price"]),
                )
        )
    """
    names = list(specs.keys())
    spec_list = list(specs.values())
    return self._group_agg(key_fn, names, spec_list)


Query.group_agg = _query_group_agg


# ---------------------------------------------------------------------------
# Extended aggregation helpers — for use with GroupBy.agg()
# ---------------------------------------------------------------------------

def agg_median(field_fn):
    """Return a GroupBy.agg reducer that computes the median of ``field_fn(item)``.

    For even-length groups, returns the average of the two middle values.
    Returns ``None`` for empty groups.

    Example::

        gb = Query(records).group_by(lambda r: r["dept"])
        gb.agg(median_salary=agg_median(lambda r: r["salary"]))
    """
    def _agg(group):
        values = sorted(field_fn(x) for x in group)
        n = len(values)
        if n == 0:
            return None
        mid = n // 2
        return values[mid] if n % 2 == 1 else (values[mid - 1] + values[mid]) / 2
    return _agg


def agg_std(field_fn, ddof: int = 0):
    """Return a GroupBy.agg reducer that computes the standard deviation.

    Parameters
    ----------
    field_fn : callable
        Extracts the numeric value from each item.
    ddof : int
        Delta degrees of freedom. ``0`` = population std (default), ``1`` = sample std.

    Returns ``None`` for empty groups (or groups with fewer elements than ``ddof + 1``).
    """
    def _agg(group):
        values = [field_fn(x) for x in group]
        n = len(values)
        if n <= ddof:
            return None
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - ddof)
        return variance ** 0.5
    return _agg


def agg_first(field_fn=None):
    """Return a GroupBy.agg reducer that returns the first element (or extracted value).

    If ``field_fn`` is given, applies it to the first element.
    Returns ``None`` for empty groups.
    """
    def _agg(group):
        first = group.first()
        if first is None:
            return None
        return field_fn(first) if field_fn is not None else first
    return _agg


def agg_last(field_fn=None):
    """Return a GroupBy.agg reducer that returns the last element (or extracted value).

    If ``field_fn`` is given, applies it to the last element.
    Returns ``None`` for empty groups.
    """
    def _agg(group):
        last = group.last()
        if last is None:
            return None
        return field_fn(last) if field_fn is not None else last
    return _agg


__all__ = [
    "Query",
    "col",
    "field",
    "Expr",
    "ColProxy",
    "FieldExpr",
    "AggSpec",
    "GroupBy",
    "agg_count",
    "agg_sum",
    "agg_mean",
    "agg_max",
    "agg_min",
    "agg_median",
    "agg_std",
    "agg_first",
    "agg_last",
    "from_numpy",
    "from_arrow",
    "from_csv",
    "from_json_lines",
    "from_generator",
    "__version__",
]
