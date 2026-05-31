# GroupBy & group_agg

## group_agg — single-pass Rust kernel

`group_agg` runs all aggregations in one pass without materializing intermediate lists.

```python
from zpyflow import Query, field, agg_count, agg_sum, agg_mean

transactions = [
    {"user": "alice", "amount": 120.0, "category": "food"},
    {"user": "bob",   "amount":  45.0, "category": "transport"},
    {"user": "alice", "amount": 300.0, "category": "shopping"},
]

result = (
    Query(transactions)
        .group_agg(
            lambda t: t["user"],
            count   = agg_count(),
            total   = agg_sum(lambda t: t["amount"]),
        )
)
# → [{"_key": "alice", "count": 2, "total": 420.0},
#    {"_key": "bob",   "count": 1, "total":  45.0}]

# field() DSL key — Rust-side extraction
by_category = (
    Query(transactions)
        .group_agg(
            field("category"),
            count   = agg_count(),
            revenue = agg_sum(lambda t: t["amount"]),
            avg     = agg_mean(lambda t: t["amount"]),
        )
)
```

## GroupBy — multi-operation groups

Use `GroupBy` when you need per-group `Query` objects:

```python
from zpyflow import GroupBy, Query

gb = GroupBy(transactions, key_fn=lambda t: t["user"])

# Per-group queries
alice_txns = gb.get_group("alice").to_list()

# Aggregation via reducers
summary = gb.agg(
    count=lambda g: g.count(),
    total=lambda g: Query([t["amount"] for t in g.to_list()]).sum(),
)

# Quick stats
counts = gb.count_per_group()             # {"alice": 2, "bob": 1}
totals = gb.sum_per_group(field=lambda t: t["amount"])  # {"alice": 420.0, ...}
```

## When to prefer group_agg over GroupBy

| | `group_agg` | `GroupBy.agg` |
|---|---|---|
| Passes over data | 1 | 1 per group operation |
| Per-group queries | ❌ | ✅ |
| Custom multi-step logic | Limited | ✅ |
| Performance (count + sum) | Faster | Slower |
