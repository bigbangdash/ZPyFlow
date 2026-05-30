"""
usecase_ab_testing.py
---------------------
Industry: Product analytics / Growth engineering

ZPyFlow for computing A/B test metrics from experiment event logs.

After an experiment runs, you have a large list of numeric outcomes
(revenue per session, conversion value, engagement score) split by variant.
Computing mean, total, and conversion rate per variant is a natural fit
for ZPyFlow's filter + aggregate pattern.

Note: for multi-metric analysis in a single pass, Polars is faster.
ZPyFlow wins when the input is already a Python list and you need one
metric at a time.
"""

from __future__ import annotations

import time
import random
import math
from zpyflow import Query, col

rng = random.Random(2025)

# ------------------------------------------------------------------
# Simulate experiment results
# ------------------------------------------------------------------
N_USERS        = 1_000_000
CONVERSION_RATE_CONTROL   = 0.08
CONVERSION_RATE_TREATMENT = 0.092  # +15% lift

def simulate_variant(n: int, conv_rate: float, avg_revenue: float) -> list[float]:
    """Revenue per user: 0 if no conversion, log-normal if converted."""
    return [
        round(math.exp(rng.gauss(math.log(avg_revenue), 0.6)), 2)
        if rng.random() < conv_rate else 0.0
        for _ in range(n)
    ]

print("Simulating experiment results...")
control   = simulate_variant(N_USERS // 2, CONVERSION_RATE_CONTROL,   35.0)
treatment = simulate_variant(N_USERS // 2, CONVERSION_RATE_TREATMENT,  37.0)
print(f"  {len(control):,} control users, {len(treatment):,} treatment users\n")

# ------------------------------------------------------------------
# Case 1: Conversion rate per variant
# ------------------------------------------------------------------
t0 = time.perf_counter()
conv_control   = Query(control).filter(col > 0).count()
conv_treatment = Query(treatment).filter(col > 0).count()
ms = (time.perf_counter() - t0) * 1000

cvr_c = conv_control   / len(control)   * 100
cvr_t = conv_treatment / len(treatment) * 100
lift  = (cvr_t - cvr_c) / cvr_c * 100

print(f"Case 1 — Conversion rate:")
print(f"  Control:   {cvr_c:.2f}%  ({conv_control:,} converters)")
print(f"  Treatment: {cvr_t:.2f}%  ({conv_treatment:,} converters)")
print(f"  Lift:      {lift:+.1f}%")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 2: Average order value (among converters)
# ------------------------------------------------------------------
t0 = time.perf_counter()
rev_c = Query(control).filter(col > 0)
rev_t = Query(treatment).filter(col > 0)

aov_c = rev_c.sum() / rev_c.count()
aov_t = rev_t.sum() / rev_t.count()
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 2 — Average order value (converters only):")
print(f"  Control:   ${aov_c:.2f}")
print(f"  Treatment: ${aov_t:.2f}  ({(aov_t-aov_c)/aov_c*100:+.1f}%)")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 3: Revenue per user (primary metric)
# ------------------------------------------------------------------
t0 = time.perf_counter()
rpu_c = Query(control).sum()   / len(control)
rpu_t = Query(treatment).sum() / len(treatment)
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 3 — Revenue per user (primary metric):")
print(f"  Control:   ${rpu_c:.4f}")
print(f"  Treatment: ${rpu_t:.4f}  ({(rpu_t-rpu_c)/rpu_c*100:+.1f}%)")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 4: High-value converter segment (revenue > $100)
# ------------------------------------------------------------------
HV_THRESHOLD = 100.0

t0 = time.perf_counter()
hv_c = Query(control).filter(col > HV_THRESHOLD).count()
hv_t = Query(treatment).filter(col > HV_THRESHOLD).count()
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 4 — High-value converters (>${HV_THRESHOLD}):")
print(f"  Control:   {hv_c:,} ({hv_c/len(control)*100:.2f}%)")
print(f"  Treatment: {hv_t:,} ({hv_t/len(treatment)*100:.2f}%)")
print(f"  Time: {ms:.2f}ms")
