"""
usecase_api_response.py
-----------------------
Industry: Any backend service consuming third-party REST / streaming APIs.

ZPyFlow for processing numeric arrays returned by external APIs.

Common pattern:
  1. Call API → receive JSON with a large list of numeric values
  2. Extract the numeric column (list comprehension — Python's job)
  3. Validate range, aggregate, or take top-N (ZPyFlow's job)

Examples: pricing feeds, exchange rates, telemetry endpoints,
          recommendation scores, inventory levels.
"""

from __future__ import annotations

import json
import time
import random
import math
from zpyflow import Query, col

rng = random.Random(42)

# ------------------------------------------------------------------
# Simulate API responses
# ------------------------------------------------------------------

def mock_pricing_api(n: int) -> dict:
    """Simulates a pricing feed API returning N product prices."""
    return {
        "source":    "pricing-service-v2",
        "currency":  "USD",
        "prices": [
            round(math.exp(rng.gauss(4.5, 1.2)), 2)  # log-normal prices
            for _ in range(n)
        ],
    }

def mock_latency_api(n: int) -> dict:
    """Simulates a metrics endpoint returning request latencies."""
    return {
        "service":   "payment-gateway",
        "window_ms": 60_000,
        "latencies": [
            round(math.exp(rng.gauss(3.5, 1.0)), 2)  # log-normal latencies
            for _ in range(n)
        ],
    }

def mock_score_api(n: int) -> dict:
    """Simulates a recommendation API returning relevance scores."""
    return {
        "query_id": "q-8f3a2",
        "scores":   [rng.betavariate(2, 5) for _ in range(n)],
    }

print("Generating mock API responses...")
pricing_response  = mock_pricing_api(200_000)
latency_response  = mock_latency_api(100_000)
score_response    = mock_score_api(500_000)
print("  Done.\n")

# ------------------------------------------------------------------
# Case 1: Pricing feed — validate range, compute stats
# ------------------------------------------------------------------
MIN_PRICE, MAX_PRICE = 1.0, 10_000.0

t0 = time.perf_counter()
prices = pricing_response["prices"]                     # extract (Python)
valid  = Query(prices).filter(col.between(MIN_PRICE, MAX_PRICE))  # validate (Rust)

n_valid  = valid.count()
total    = valid.sum()
max_p    = valid.max()
ms       = (time.perf_counter() - t0) * 1000

print(f"Case 1 — Pricing feed ({len(prices):,} prices):")
print(f"  Valid range [{MIN_PRICE}, {MAX_PRICE}]: {n_valid:,} ({n_valid/len(prices)*100:.1f}%)")
print(f"  Total value: ${total:,.0f}  |  Max: ${max_p:,.2f}")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 2: Latency endpoint — P95-proxy and SLO check
# ------------------------------------------------------------------
SLO_MS = 500.0

t0 = time.perf_counter()
latencies = latency_response["latencies"]
q = Query(latencies)

total_requests  = q.count()
breaching       = q.filter(col > SLO_MS).count()
slo_compliance  = (1 - breaching / total_requests) * 100
ms              = (time.perf_counter() - t0) * 1000

print(f"\nCase 2 — Latency endpoint ({len(latencies):,} requests):")
print(f"  SLO ({SLO_MS}ms): {slo_compliance:.2f}% compliant")
print(f"  Breaching: {breaching:,} requests")
print(f"  Max latency: {q.max():.0f}ms")
print(f"  Time: {ms:.2f}ms")

# ------------------------------------------------------------------
# Case 3: Recommendation scores — top-K candidates (early stopping)
# ------------------------------------------------------------------
RELEVANCE_THRESHOLD = 0.5
TOP_K = 1_000

t0 = time.perf_counter()
scores     = score_response["scores"]
candidates = (
    Query(scores)
        .filter(col > RELEVANCE_THRESHOLD)
        .take(TOP_K)
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 3 — Recommendation scores ({len(scores):,} candidates):")
print(f"  Top-{TOP_K} above threshold={RELEVANCE_THRESHOLD}: {len(candidates)} found")
print(f"  Time: {ms:.2f}ms  (early stopping — did not scan all {len(scores):,})")
