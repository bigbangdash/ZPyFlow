"""
usecase_fraud_detection.py
--------------------------
Industry: Fintech / Insurance / Lending

ZPyFlow for real-time fraud scoring pipelines.

A risk model assigns a float score to each transaction or loan application.
Downstream logic needs to:
  - Count how many cases are flagged (score > threshold)
  - Pull the top-N highest-risk cases into a review queue (early stopping)
  - Compute total monetary exposure above a risk threshold

ZPyFlow's advantage: each of these is a single-pass Rust aggregation.
For multiple stats in one job, prefer Polars (see bench_etl).
"""

from __future__ import annotations

import time
import numpy as np
from zpyflow import Query, col

rng = np.random.default_rng(7)

# Simulate risk scores from a fraud model.
# Log-normal: most transactions are low-risk; a long tail is high-risk.
N_TRANSACTIONS   = 1_000_000
FRAUD_THRESHOLD  = 50.0   # flag if score > 50
REVIEW_QUEUE_MAX = 500    # human reviewers handle at most 500 cases/batch

raw_scores = rng.lognormal(mean=2.0, sigma=1.5, size=N_TRANSACTIONS)
risk_scores = raw_scores.tolist()

# Simulate transaction amounts correlated loosely with risk
amounts = (raw_scores * rng.uniform(10, 1000, N_TRANSACTIONS)).tolist()

# ------------------------------------------------------------------
# Case 1: Count flagged transactions
# ------------------------------------------------------------------
t0 = time.perf_counter()
n_flagged = Query(risk_scores).filter(col > FRAUD_THRESHOLD).count()
ms = (time.perf_counter() - t0) * 1000

flag_rate = n_flagged / N_TRANSACTIONS * 100
print(f"Case 1 — Flag count (threshold={FRAUD_THRESHOLD}):")
print(f"  {n_flagged:,} of {N_TRANSACTIONS:,} flagged ({flag_rate:.2f}%)")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 2: Fill the review queue (early stopping)
# ------------------------------------------------------------------
t0 = time.perf_counter()
review_queue = (
    Query(risk_scores)
        .filter(col > FRAUD_THRESHOLD)
        .take(REVIEW_QUEUE_MAX)   # stop scanning once queue is full
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 2 — Review queue (max={REVIEW_QUEUE_MAX}):")
print(f"  Queued: {len(review_queue)} cases in {ms:.2f}ms")
print(f"  Score range: [{min(review_queue):.1f}, {max(review_queue):.1f}]")

# ------------------------------------------------------------------
# Case 3: Total monetary exposure for high-risk transactions
# ------------------------------------------------------------------
HIGH_RISK = 100.0

t0 = time.perf_counter()
exposure = Query(amounts).filter(col > HIGH_RISK).sum()
n_high   = Query(amounts).filter(col > HIGH_RISK).count()
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 3 — Monetary exposure (amount > {HIGH_RISK}):")
print(f"  {n_high:,} high-risk transactions")
print(f"  Total exposure: ${exposure:,.0f}")
print(f"  Avg exposure:   ${exposure / max(n_high, 1):,.0f}")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 4: Threshold sensitivity (adjust threshold to control queue load)
# ------------------------------------------------------------------
print(f"\nCase 4 — Threshold sensitivity:")
print(f"  {'threshold':>10}  {'flagged':>10}  {'flag_rate':>10}")
for t in [20.0, 50.0, 100.0, 200.0, 500.0]:
    n = Query(risk_scores).filter(col > t).count()
    print(f"  {t:>10.0f}  {n:>10,}  {n/N_TRANSACTIONS*100:>9.2f}%")
