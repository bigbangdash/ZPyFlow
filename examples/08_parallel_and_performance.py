"""
08_parallel_and_performance.py
------------------------------
Parallel execution and performance measurement.

Shows:
  - When to use .parallel() and when not to (overhead threshold)
  - Comparing single-thread vs parallel on different data sizes
  - Memory-efficient chaining vs naive Python
  - Profiling patterns
  - Materialise-once patterns with cache() and tee()
"""

from __future__ import annotations

import gc
import sys
import time

import numpy as np
from zpyflow import Query, col, from_numpy

rng = np.random.default_rng(42)

def timed(label: str, fn, warmup: bool = True):
    """Run fn() with optional warmup, return (result, elapsed_ms)."""
    if warmup:
        fn()   # warm up caches
    gc.collect()
    t0 = time.perf_counter()
    result = fn()
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {label:45s} {ms:7.2f}ms")
    return result, ms

# ------------------------------------------------------------------
# Case 1: Overhead threshold — parallel vs single-thread by size
# ------------------------------------------------------------------
print("Case 1 — Parallel overhead threshold\n         (parallel faster above ~500K elements)")
print(f"  {'approach':45s} {'time':>8s}")
print(f"  {'-'*53}")

for size in [10_000, 100_000, 500_000, 1_000_000, 5_000_000]:
    arr = rng.standard_normal(size).tolist()

    _, t_single = timed(
        f"  single   n={size:>9,}",
        lambda: Query(arr).filter(col > 0).map(col * 2).to_list(),
        warmup=False,
    )
    _, t_parallel = timed(
        f"  parallel n={size:>9,}",
        lambda: Query(arr).filter(col > 0).map(col * 2).parallel().to_list(),
        warmup=False,
    )
    speedup = t_single / t_parallel
    print(f"  {'→ speedup':>45s} {speedup:7.2f}x")

# ------------------------------------------------------------------
# Case 2: Chained pipeline — fused vs naive Python
# ------------------------------------------------------------------
print("\nCase 2 — Chained pipeline: filter+map+take (1M elements)")

SIZE = 1_000_000
arr = rng.standard_normal(SIZE)
lst = arr.tolist()

print(f"  {'approach':45s} {'time':>8s}")
print(f"  {'-'*53}")

timed("numpy eager (arr[arr>0]*2, 2 allocs)",
      lambda: (arr[arr > 0] * 2)[:10_000])

timed("list comprehension (2 passes, 2 allocs)",
      lambda: [x * 2 for x in lst if x > 0][:10_000])

timed("generator (lazy, 1 alloc at take)",
      lambda: list(x * 2 for x in lst if x > 0)[:10_000])

timed("ZPyFlow DSL single-thread (1 alloc, GIL released)",
      lambda: Query(lst).filter(col > 0).map(col * 2).take(10_000).to_list())

timed("ZPyFlow DSL parallel (1 alloc, GIL released)",
      lambda: Query(lst).filter(col > 0).map(col * 2).parallel().take(10_000).to_list())

# ------------------------------------------------------------------
# Case 3: Aggregation comparison (no output list — pure reduction)
# ------------------------------------------------------------------
print("\nCase 3 — Aggregation: sum of positives (1M elements)")
print(f"  {'approach':45s} {'time':>8s}")
print(f"  {'-'*53}")

timed("sum(x for x in lst if x > 0)  [Python generator]",
      lambda: sum(x for x in lst if x > 0))

timed("numpy: arr[arr>0].sum()  [eager mask]",
      lambda: arr[arr > 0].sum())

timed("ZPyFlow: filter(col>0).sum()  [fused, GIL released]",
      lambda: Query(lst).filter(col > 0).sum())

timed("ZPyFlow: parallel().filter(col>0).sum()",
      lambda: Query(lst).filter(col > 0).parallel().sum())

# ------------------------------------------------------------------
# Case 4: Memory footprint estimate
# ------------------------------------------------------------------
print("\nCase 4 — Rough memory cost")

SIZE_LARGE = 10_000_000
arr_large = rng.standard_normal(SIZE_LARGE)

# Python list of floats: each float object = 24 bytes on CPython
list_bytes = SIZE_LARGE * 24 + sys.getsizeof([])  # approx
# numpy array of float64: 8 bytes per element
np_bytes = SIZE_LARGE * 8

print(f"  Raw data ({SIZE_LARGE:,} elements):")
print(f"    Python list[float]:    ~{list_bytes/1e6:.0f} MB  (float objects on heap)")
print(f"    numpy float64 array:   ~{np_bytes/1e6:.0f} MB  (contiguous C buffer)")
print(f"    ZPyFlow result list:   ~{np_bytes/1e6:.0f} MB  (output only, no intermediates)")
print(f"  numpy filter: adds ~{np_bytes/1e6:.0f} MB for boolean mask + result array")
print(f"  ZPyFlow: adds 0 bytes for filter (writes directly to output Vec)")

# ------------------------------------------------------------------
# Case 5: Parallel multi-column processing
# ------------------------------------------------------------------
print("\nCase 5 — Parallel multi-column processing (5 columns × 2M rows)")

ROWS = 2_000_000
COLS = 5
columns = [rng.standard_normal(ROWS).tolist() for _ in range(COLS)]

t0 = time.perf_counter()
# Process each column sequentially
seq_results = [Query(c).filter(col > 0).map(col ** 2).to_list() for c in columns]
seq_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
# Process each column with parallel execution
par_results = [Query(c).filter(col > 0).map(col ** 2).parallel().to_list() for c in columns]
par_ms = (time.perf_counter() - t0) * 1000

print(f"  Sequential (SIMD, no parallel): {seq_ms:.1f}ms")
print(f"  Parallel per-column:            {par_ms:.1f}ms")
print(f"  Speedup: {seq_ms/par_ms:.2f}x")

# ------------------------------------------------------------------
# Case 6: Profile-friendly pattern — measure per operation
# ------------------------------------------------------------------
print("\nCase 6 — Per-operation timing breakdown (100K elements)")

data = rng.standard_normal(100_000).tolist()

ops = [
    ("filter only   (col > 0)",          lambda: Query(data).filter(col > 0).to_list()),
    ("map only      (col * 2)",           lambda: Query(data).map(col * 2).to_list()),
    ("filter + map",                       lambda: Query(data).filter(col > 0).map(col * 2).to_list()),
    ("filter + map + take(1000)",          lambda: Query(data).filter(col > 0).map(col * 2).take(1000).to_list()),
    ("filter + map + skip(100) + take",   lambda: Query(data).filter(col > 0).map(col * 2).skip(100).take(1000).to_list()),
    ("sum (filter + sum)",                 lambda: Query(data).filter(col > 0).sum()),
    ("count (filter + count)",             lambda: Query(data).filter(col > 0).count()),
]

print(f"  {'operation':45s} {'time':>8s}")
print(f"  {'-'*53}")
for label, fn in ops:
    timed(label, fn, warmup=True)

# ------------------------------------------------------------------
# Case 7: Materialise-once with cache() and tee()
# ------------------------------------------------------------------
print("\nCase 7 — Materialise-once: cache() vs tee()")

data_small = rng.standard_normal(50_000).tolist()

# Without cache: the filter runs three times
t0 = time.perf_counter()
q = Query(data_small).filter(col > 0).map(col * 2)
r1 = q.count()
r2 = q.sum()
r3 = q.mean()
ms_no_cache = (time.perf_counter() - t0) * 1000
print(f"  Without cache (3 passes):  {ms_no_cache:.2f}ms  count={r1}")

# With cache: filter+map runs once, results reused
t0 = time.perf_counter()
cached = Query(data_small).filter(col > 0).map(col * 2).cache()
c1 = cached.count()
c2 = cached.sum()
c3 = cached.mean()
ms_cache = (time.perf_counter() - t0) * 1000
print(f"  With cache    (1 pass):    {ms_cache:.2f}ms  count={c1}  ({ms_no_cache/ms_cache:.1f}x faster)")

# tee: fork into independent copies before the pipeline diverges
q_pos, q_neg = Query(data_small).tee()
pos_count = q_pos.filter(col > 0).count()
neg_count = q_neg.filter(col < 0).count()
print(f"\n  tee() — split into 2 independent queries from the same source:")
print(f"    positives: {pos_count:,}   negatives: {neg_count:,}   total: {pos_count+neg_count:,}")
