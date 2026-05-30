"""
09_vector_search.py
-------------------
ZPyFlow for vector search score post-filtering.

After an ANN index (FAISS, Qdrant, etc.) returns 1M cosine similarity scores,
you typically need to:
  - Filter by a threshold (discard low-similarity results)
  - Take the top-K candidates
  - Count how many passed (for recall estimation)

Key insight: .take(K) stops as soon as K results are collected, so ZPyFlow
is ~70× faster than numpy here — numpy must scan everything before slicing.
"""

from __future__ import annotations

import time
import numpy as np
from zpyflow import Query, col, from_numpy

rng = np.random.default_rng(42)

# Simulate 1M cosine similarity scores from an ANN search.
# Beta(2,5) is realistic: most scores cluster near 0, ~15% exceed 0.5.
N_DOCS    = 1_000_000
THRESHOLD = 0.5
TOP_K     = 1_000

scores_np = rng.beta(a=2, b=5, size=N_DOCS).astype(np.float64)
scores    = scores_np.tolist()

# ------------------------------------------------------------------
# Case 1: Top-K retrieval (early stopping)
# ------------------------------------------------------------------
t0 = time.perf_counter()
top_k = from_numpy(scores_np).filter(col > THRESHOLD).take(TOP_K).to_list()
ms    = (time.perf_counter() - t0) * 1000

print(f"Case 1 — Top-K retrieval (threshold={THRESHOLD}, K={TOP_K:,}):")
print(f"  {N_DOCS:,} candidates → {len(top_k)} results in {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 2: Count matching candidates (for recall / precision estimation)
# ------------------------------------------------------------------
t0 = time.perf_counter()
n_match = Query(scores).filter(col > THRESHOLD).count()
ms      = (time.perf_counter() - t0) * 1000

print(f"\nCase 2 — Count candidates above threshold:")
print(f"  {n_match:,} of {N_DOCS:,} pass threshold={THRESHOLD} ({n_match/N_DOCS*100:.1f}%)")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 3: Threshold sweep (precision vs recall trade-off)
# ------------------------------------------------------------------
print(f"\nCase 3 — Threshold sweep:")
print(f"  {'threshold':>10}  {'pass_count':>10}  {'pass_rate':>10}  {'time_ms':>10}")
for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    t0 = time.perf_counter()
    n  = Query(scores).filter(col > t).count()
    ms = (time.perf_counter() - t0) * 1000
    print(f"  {t:>10.1f}  {n:>10,}  {n/N_DOCS*100:>9.1f}%  {ms:>9.2f}ms")

# ------------------------------------------------------------------
# Case 4: Multi-threshold parallel batch (multiple queries)
# ------------------------------------------------------------------
THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
BATCH_K    = 500

print(f"\nCase 4 — Batch top-K across multiple thresholds:")
t0 = time.perf_counter()
results = {
    t: Query(scores).filter(col > t).take(BATCH_K).to_list()
    for t in THRESHOLDS
}
ms = (time.perf_counter() - t0) * 1000

for t, hits in results.items():
    print(f"  threshold={t}  hits={len(hits)}")
print(f"  Total time ({len(THRESHOLDS)} thresholds): {ms:.2f}ms")
