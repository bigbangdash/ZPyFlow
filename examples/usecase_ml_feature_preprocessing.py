"""
10_ml_feature.py
----------------
ZPyFlow for ML feature preprocessing pipelines.

Common pattern before model training:
  1. Remove outliers (col.between clips extremes in a single SIMD pass)
  2. Normalize to a fixed range
  3. Optionally sub-sample (take N) for mini-batch or stratified split

ZPyFlow wins when you need filter + transform + take in a fused single pass.
Benchmarks: ~2x faster than numpy when take() is used; numpy wins on full output.
"""

from __future__ import annotations

import time
import numpy as np
from zpyflow import Query, col, from_numpy

rng = np.random.default_rng(0)

N = 1_000_000

# Raw feature values — uniform [-100, 100], simulating a model input column
raw = rng.uniform(-100, 100, N).astype(np.float64)

CLIP  = 90.0         # remove |x| > 90  (~10% of data)
SCALE = 1.0 / 90.0  # normalize to [-1, 1]

# ------------------------------------------------------------------
# Case 1: Remove outliers + normalize (full output)
# ------------------------------------------------------------------
t0 = time.perf_counter()
processed = (
    from_numpy(raw)
        .filter(col.between(-CLIP, CLIP))   # FilterBetween: single SIMD op
        .map(col * SCALE)                   # scale to [-1, 1]
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

print(f"Case 1 — Outlier removal + normalize (N={N:,}):")
print(f"  Input:  {N:,}")
print(f"  Output: {len(processed):,} ({len(processed)/N*100:.1f}% survived)")
print(f"  Time:   {ms:.1f}ms")
processed_query = Query(processed)
print(f"  Range:  [{processed_query.min():.4f}, {processed_query.max():.4f}]  (expected [-1, 1])")

# ------------------------------------------------------------------
# Case 2: Outlier removal + normalize + sub-sample (take N)
# early stopping makes this faster than numpy for large take fractions
# ------------------------------------------------------------------
SAMPLE_N = 50_000

t0 = time.perf_counter()
sample = (
    from_numpy(raw)
        .filter(col.between(-CLIP, CLIP))
        .map(col * SCALE)
        .take(SAMPLE_N)
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 2 — Same pipeline + take({SAMPLE_N:,}) sub-sample:")
print(f"  Output: {len(sample):,} items in {ms:.1f}ms")

# ------------------------------------------------------------------
# Case 3: Multi-column preprocessing (one column at a time)
# ------------------------------------------------------------------
FEATURES = {
    "age":           rng.integers(18, 80, N).astype(np.float64),
    "income":        rng.lognormal(10.5, 0.8, N),
    "credit_score":  rng.integers(300, 850, N).astype(np.float64),
    "loan_amount":   rng.lognormal(9.0, 1.0, N),
}

print(f"\nCase 3 — Multi-column preprocessing ({N:,} rows, {len(FEATURES)} features):")
t0 = time.perf_counter()

preprocessed = {}
for feat, values in FEATURES.items():
    arr = from_numpy(values)
    vmin, vmax = arr.min(), arr.max()
    vrange = vmax - vmin or 1.0
    # Two chained maps: shift then scale (PyExpr is single-op per .map())
    preprocessed[feat] = (
        from_numpy(values)
            .map(col - vmin)
            .map(col / vrange)
            .filter(col.between(0.0, 1.0))
            .to_list()
    )

ms = (time.perf_counter() - t0) * 1000
print(f"  Time: {ms:.1f}ms")
for feat, vals in preprocessed.items():
    feat_query = Query(vals)
    print(f"  {feat:16s}  n={len(vals):,}  min={feat_query.min():.4f}  max={feat_query.max():.4f}")

# ------------------------------------------------------------------
# Case 4: Clip + count (how many outliers?)
# ------------------------------------------------------------------
print(f"\nCase 4 — Outlier count at different clip thresholds:")
for clip in [50.0, 70.0, 90.0, 95.0]:
    n_clean = from_numpy(raw).filter(col.between(-clip, clip)).count()
    n_out   = N - n_clean
    print(f"  clip={clip:5.1f}  survivors={n_clean:,}  outliers={n_out:,} ({n_out/N*100:.1f}%)")
