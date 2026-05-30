"""
usecase_pricing_rules.py
------------------------
Industry: E-commerce / SaaS billing

ZPyFlow for applying pricing rules and catalog management.

Common operations on a product catalog:
  - Flag products below minimum margin for repricing
  - Apply promotional discounts to a price tier
  - Compute revenue impact of a proposed price change
  - Validate that catalog prices are within acceptable bounds
"""

from __future__ import annotations

import time
import random
import math
from zpyflow import Query, col

rng = random.Random(99)

N_PRODUCTS    = 500_000
MIN_MARGIN    = 0.15     # flag if margin < 15%
DISCOUNT_RATE = 0.10     # 10% promotional discount
MID_TIER_LO   = 20.0
MID_TIER_HI   = 200.0

# Simulate product catalog: cost, price, and derived margin
costs  = [round(math.exp(rng.gauss(3.5, 0.8)), 2) for _ in range(N_PRODUCTS)]
prices = [round(c * rng.uniform(1.05, 2.5), 2) for c in costs]
margins = [(p - c) / p for p, c in zip(prices, costs)]

print(f"Catalog: {N_PRODUCTS:,} products\n")

# ------------------------------------------------------------------
# Case 1: Count products below minimum margin (needs repricing)
# ------------------------------------------------------------------
t0 = time.perf_counter()
n_underpriced = Query(margins).filter(col < MIN_MARGIN).count()
ms = (time.perf_counter() - t0) * 1000

print(f"Case 1 — Margin health check (min margin={MIN_MARGIN:.0%}):")
print(f"  Underpriced: {n_underpriced:,} products ({n_underpriced/N_PRODUCTS*100:.1f}%)")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 2: Apply promotional discount to mid-tier products
# ------------------------------------------------------------------
t0 = time.perf_counter()
discounted_prices = (
    Query(prices)
        .filter(col.between(MID_TIER_LO, MID_TIER_HI))   # mid-tier only
        .map(col * (1.0 - DISCOUNT_RATE))                  # apply discount
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

q = Query(discounted_prices)
print(f"\nCase 2 — Promotional discount ({DISCOUNT_RATE:.0%} off mid-tier ${MID_TIER_LO}–${MID_TIER_HI}):")
print(f"  Affected products: {len(discounted_prices):,}")
print(f"  Price range after discount: [${q.min():.2f}, ${q.max():.2f}]")
print(f"  Total revenue at new prices: ${q.sum():,.0f}")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 3: Revenue impact of a proposed 5% price increase on premium tier
# ------------------------------------------------------------------
PREMIUM_LO   = 200.0
PRICE_HIKE   = 1.05

t0 = time.perf_counter()
q_current = Query(prices).filter(col > PREMIUM_LO)
q_new     = Query(prices).filter(col > PREMIUM_LO)

revenue_now  = q_current.sum()
revenue_new  = Query(
    Query(prices).filter(col > PREMIUM_LO).map(col * PRICE_HIKE).to_list()
).sum()
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 3 — Premium tier price hike +{PRICE_HIKE-1:.0%} (>${PREMIUM_LO}):")
print(f"  Products affected: {q_current.count():,}")
print(f"  Revenue now:  ${revenue_now:,.0f}")
print(f"  Revenue new:  ${revenue_new:,.0f}  (+${revenue_new-revenue_now:,.0f})")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 4: Catalog integrity — out-of-range prices
# ------------------------------------------------------------------
t0 = time.perf_counter()
n_invalid = Query(prices).filter(col < 0.01).count()
n_outlier = Query(prices).filter(col > 50_000).count()
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 4 — Catalog integrity check:")
print(f"  Prices < $0.01: {n_invalid}")
print(f"  Prices > $50,000: {n_outlier}")
print(f"  Time: {ms:.2f}ms")
