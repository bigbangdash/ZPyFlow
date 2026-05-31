"""
GroupBy — pure Python grouping layer over a materialized Query result.

The Rust core handles the per-element pipeline.  Grouping requires an index
structure (hash map) that doesn't benefit from SIMD, so it lives in Python
with `collections.defaultdict` — already very fast for this workload.

For grouped numeric aggregations (sum, mean, max per group), we delegate to
the Rust core via specialized entry points that process one group at a time.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Generic, Iterable, TypeVar

from ._zpyflow import Query, AggSpec

T = TypeVar("T")
K = TypeVar("K")
U = TypeVar("U")


# ---------------------------------------------------------------------------
# Structured aggregation spec factory functions
# ---------------------------------------------------------------------------

def agg_count() -> AggSpec:
    """Count elements in each group."""
    return AggSpec.count()


def agg_sum(field_fn: Callable[[Any], float]) -> AggSpec:
    """Sum `field_fn(item)` over each group."""
    return AggSpec.sum(field_fn)


def agg_mean(field_fn: Callable[[Any], float]) -> AggSpec:
    """Mean of `field_fn(item)` over each group."""
    return AggSpec.mean(field_fn)


def agg_max(field_fn: Callable[[Any], float]) -> AggSpec:
    """Maximum of `field_fn(item)` over each group."""
    return AggSpec.max(field_fn)


def agg_min(field_fn: Callable[[Any], float]) -> AggSpec:
    """Minimum of `field_fn(item)` over each group."""
    return AggSpec.min(field_fn)


class GroupBy(Generic[K, T]):
    """
    Grouped query.  Obtain via `Query(...).group_by(key_fn)`.

    Example::

        from zpyflow import Query

        data = [{"dept": "eng", "salary": 120_000}, ...]

        result = (
            Query(data)
                .group_by(lambda r: r["dept"])
                .agg(
                    count=lambda g: g.count(),
                    avg_salary=lambda g: g.map(lambda r: r["salary"]).sum() / g.count(),
                )
        )
    """

    def __init__(self, items: Iterable[T], key_fn: Callable[[T], K]) -> None:
        # `items` can be a list or any iterable (including a Query with __iter__).
        self._groups: dict[K, list[T]] = defaultdict(list)
        for item in items:
            self._groups[key_fn(item)].append(item)

    @classmethod
    def _from_dict(cls, groups: dict[K, list[T]]) -> "GroupBy[K, T]":
        """Wrap a pre-built {key: [items]} dict — skips the construction pass."""
        obj = cls.__new__(cls)
        obj._groups = groups
        return obj

    def keys(self) -> list[K]:
        return list(self._groups.keys())

    def get_group(self, key: K) -> Query:
        return Query(self._groups.get(key, []))

    def agg(self, **reducers: Callable[[Query], Any]) -> list[dict[str, Any]]:
        """
        Apply named aggregation functions to each group.

        Each reducer receives a `Query` over the group and returns a scalar.

        Returns a list of dicts: one per group with keys = group key + reducer names.
        """
        result: list[dict[str, Any]] = []
        for key, items in self._groups.items():
            group_query = Query(items)
            row: dict[str, Any] = {"_key": key}
            for name, fn in reducers.items():
                row[name] = fn(group_query)
            result.append(row)
        return result

    def map_groups(self, fn: Callable[[K, Query], U]) -> list[U]:
        """Apply `fn(key, Query(group))` for each group, return list of results."""
        return [fn(key, Query(items)) for key, items in self._groups.items()]

    def count_per_group(self) -> dict[K, int]:
        return {k: len(v) for k, v in self._groups.items()}

    def sum_per_group(self, field: Callable[[T], float] | None = None) -> dict[K, float]:
        result: dict[K, float] = {}
        for key, items in self._groups.items():
            item_query = Query(items)
            if field is not None:
                item_query = item_query.map(field)
            result[key] = float(item_query.sum())
        return result

    def __len__(self) -> int:
        return len(self._groups)

    def __repr__(self) -> str:
        return f"GroupBy({len(self._groups)} groups)"
