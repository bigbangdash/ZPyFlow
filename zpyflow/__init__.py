"""
ZPyFlow — Zero-allocation lazy query pipelines for Python, powered by Rust.

Quick start::

    from zpyflow import Query, col
    import numpy as np

    data = np.random.randn(1_000_000).tolist()

    result = (
        Query(data)
            .filter(col > 0.5)        # SIMD filter, GIL released
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
    "from_numpy",
    "from_arrow",
    "from_csv",
    "from_json_lines",
    "from_generator",
    "__version__",
]
