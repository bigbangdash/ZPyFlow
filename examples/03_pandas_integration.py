"""
03_pandas_integration.py
------------------------
ZPyFlow as a pre-processing step inside a pandas workflow.

ZPyFlow is NOT a pandas replacement.  It accelerates specific steps:
  - Filtering / transforming a numeric column before writing it back
  - Row-level filtering on large DataFrames where boolean masks are slow
  - Computing aggregations on a column without materializing an intermediate Series
"""

import pandas as pd
import numpy as np
from zpyflow import Query, col, from_numpy

rng = np.random.default_rng(0)
N = 200_000

df = pd.DataFrame({
    "user_id":   np.arange(N),
    "score":     rng.uniform(0, 100, N).astype(float),
    "revenue":   rng.exponential(scale=50, size=N).astype(float),
    "active":    rng.integers(0, 2, N).astype(bool),
    "region":    rng.choice(["US", "JP", "EU", "APAC"], size=N),
    "plan":      rng.choice(["free", "pro", "enterprise"], size=N),
})

# ------------------------------------------------------------------
# Case 1: Filter + transform a numeric column, write back
# ------------------------------------------------------------------
scores = df["score"].tolist()

# Normalize scores above the median into [0, 1]
median = float(df["score"].median())
max_s  = float(df["score"].max())

normalized = (
    Query(scores)
        .filter(col > median)
        .map(col - median)
        .map(col / (max_s - median))
        .to_list()
)

df_high = df[df["score"] > median].copy()
df_high["score_normalized"] = normalized
print(f"Case 1 — normalized {len(df_high)} rows above median")
print(df_high[["user_id", "score", "score_normalized"]].head(3).to_string(index=False))

# ------------------------------------------------------------------
# Case 2: Row-level filtering via to_dict("records")
# ------------------------------------------------------------------
records = df.to_dict("records")

high_value_active = (
    Query(records)
        .filter(lambda r: r["active"] and r["revenue"] > 200 and r["plan"] != "free")
        .map(lambda r: {
            "user_id": r["user_id"],
            "revenue": round(r["revenue"], 2),
            "plan":    r["plan"],
            "region":  r["region"],
        })
        .take(10)
        .to_list()
)

print(f"\nCase 2 — top active high-revenue non-free users:")
for row in high_value_active[:3]:
    print(f"  {row}")

# ------------------------------------------------------------------
# Case 3: Aggregation on a column without an intermediate Series
# ------------------------------------------------------------------
revenues = df["revenue"].tolist()

# Compute statistics for enterprise users only
enterprise_mask = df["plan"] == "enterprise"
enterprise_rev  = df.loc[enterprise_mask, "revenue"].tolist()

rev_query = Query(enterprise_rev)
stats = {
    "count": rev_query.count(),
    "sum":   round(rev_query.sum(), 2),
    "max":   round(rev_query.max(), 2),
    "above_100": Query(enterprise_rev).filter(col > 100).count(),
}
print(f"\nCase 3 — enterprise revenue stats: {stats}")

# ------------------------------------------------------------------
# Case 4: ZPyFlow as a reusable pandas transform function
# ------------------------------------------------------------------
def clip_and_scale(series: pd.Series, lo: float, hi: float) -> list[float]:
    """Clip to [lo, hi] and rescale to [0, 1]."""
    span = hi - lo
    return (
        Query(series.tolist())
            .filter(col.between(lo, hi))
            .map(col - lo)
            .map(col / span)
            .to_list()
    )

clipped = clip_and_scale(df["score"], lo=20.0, hi=80.0)
print(f"\nCase 4 — clipped+scaled: {len(clipped)} rows kept (from {len(df)})")
print(f"  min={min(clipped):.4f}  max={max(clipped):.4f}")

# ------------------------------------------------------------------
# Case 5: Per-region revenue aggregation without groupby overhead
# ------------------------------------------------------------------
from collections import defaultdict

by_region: dict[str, list[float]] = defaultdict(list)
for r in records:
    by_region[r["region"]].append(r["revenue"])

region_stats = {
    region: {
        "count": Query(revs).count(),
        "avg":   round(Query(revs).sum() / len(revs), 2),
        "max":   round(Query(revs).max(), 2),
    }
    for region, revs in by_region.items()
}
print("\nCase 5 — revenue by region:")
for region, s in sorted(region_stats.items()):
    print(f"  {region}: {s}")
