"""
02_numpy_integration.py
-----------------------
Using ZPyFlow with numpy arrays.

Key point: from_numpy() routes float64/int64 arrays to the Rust f64/i64
fast path.  The GIL is released during execution, and SIMD runs on the
data without creating any intermediate arrays.

Compare with numpy's own approach:
  numpy   : arr[arr > 0] * 2   →  allocates 2 new arrays (mask + result)
  ZPyFlow : filter + map        →  allocates 1 list (the final output only)
"""

import time
import numpy as np
from zpyflow import Query, col, from_numpy

rng = np.random.default_rng(42)

# ------------------------------------------------------------------
# Case 1: Basic float64 pipeline
# ------------------------------------------------------------------
arr = rng.standard_normal(1_000_000)   # 1M values, ~[-3, 3]

result = (
    from_numpy(arr)
        .filter(col > 0.5)      # keep top half of positives
        .map(col * 2.0)         # scale
        .take(10_000)           # first 10K
        .to_list()
)
print(f"Case 1 — filtered+scaled: {len(result)} elements, first 3: {result[:3]}")

# ------------------------------------------------------------------
# Case 2: Aggregations on a large array
# ------------------------------------------------------------------
# Extracting statistics without materializing the full filtered array
positive = from_numpy(arr).filter(col > 0)
n_pos    = positive.count()
sum_pos  = positive.sum()
max_pos  = positive.max()

print(f"Case 2 — positives: n={n_pos}, sum={sum_pos:.2f}, max={max_pos:.4f}")

# ------------------------------------------------------------------
# Case 3: Normalization pipeline
# ------------------------------------------------------------------
raw_scores = rng.uniform(0, 100, size=100_000).astype(np.float64)

mean = from_numpy(raw_scores).sum() / len(raw_scores)
std  = float(np.std(raw_scores))   # use numpy for std (not in DSL yet)

# Z-score normalize, keep values within 2 standard deviations
normalized = (
    from_numpy(raw_scores)
        .map(col - mean)
        .map(col / std)
        .filter(col.between(-2.0, 2.0))
        .to_list()
)
print(f"Case 3 — z-normalized: {len(normalized)} / {len(raw_scores)} kept")

# ------------------------------------------------------------------
# Case 4: int64 array
# ------------------------------------------------------------------
ids = np.arange(1_000_000, dtype=np.int64)

# Fast path: DSL on i64
large_ids = from_numpy(ids).filter(col > 900_000).to_list()
print(f"Case 4 — large IDs: {len(large_ids)} found (expected 99999)")

# ------------------------------------------------------------------
# Case 5: Speed comparison vs numpy
# ------------------------------------------------------------------
big = rng.standard_normal(5_000_000)

t0 = time.perf_counter()
_ = big[big > 0] * 2   # numpy: creates boolean mask array + result array
numpy_ms = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
_ = from_numpy(big).filter(col > 0).map(col * 2).to_list()
zpyflow_ms = (time.perf_counter() - t0) * 1000

print(f"Case 5 — 5M elements:")
print(f"  numpy:   {numpy_ms:.1f}ms  (2 intermediate arrays)")
print(f"  zpyflow: {zpyflow_ms:.1f}ms  (1 allocation, GIL released)")

# ------------------------------------------------------------------
# Case 6: Convert result back to numpy
# ------------------------------------------------------------------
filtered_arr = np.array(
    from_numpy(big).filter(col.between(-1.0, 1.0)).to_list(),
    dtype=np.float64,
)
print(f"Case 6 — back to numpy: shape={filtered_arr.shape}, dtype={filtered_arr.dtype}")
