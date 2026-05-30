"""
usecase_order_processing.py
---------------------------
Industry: E-commerce / Logistics

ZPyFlow for order management pipelines.

Operations on order records (Python dicts):
  - Filter by status, amount threshold, or customer tier
  - Map to notification/summary format (no intermediate list)
  - Group by status and compute per-group totals
  - take() for "most recent N orders" without scanning the full history

Note: dict/object operations use the Python path (GIL held).
ZPyFlow's advantage here is ergonomics and avoiding intermediate lists
when chaining filter → map → take.
"""

from __future__ import annotations

import time
import random
import math
from dataclasses import dataclass
from zpyflow import Query, GroupBy

rng = random.Random(42)

# ------------------------------------------------------------------
# Simulate order data
# ------------------------------------------------------------------

STATUSES = ["pending", "processing", "shipped", "delivered", "returned", "cancelled"]
WEIGHTS  = [0.10, 0.10, 0.25, 0.45, 0.05, 0.05]

def make_orders(n: int) -> list[dict]:
    return [
        {
            "order_id":    f"ORD-{i:07d}",
            "customer_id": rng.randint(1, 50_000),
            "status":      rng.choices(STATUSES, weights=WEIGHTS)[0],
            "amount":      round(math.exp(rng.gauss(4.0, 0.9)), 2),
            "items":       rng.randint(1, 10),
            "is_prime":    rng.random() < 0.35,
            "created_at":  1_700_000_000 + i * 60,   # 1 order/min
        }
        for i in range(n)
    ]

N = 500_000
print(f"Generating {N:,} orders...")
orders = make_orders(N)
print("Done.\n")

# ------------------------------------------------------------------
# Case 1: Filter orders that need action (pending + processing)
# ------------------------------------------------------------------
t0 = time.perf_counter()
actionable = (
    Query(orders)
        .filter(lambda o: o["status"] in {"pending", "processing"})
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

print(f"Case 1 — Actionable orders (pending + processing):")
print(f"  {len(actionable):,} of {N:,} orders need action")
print(f"  Time: {ms:.1f}ms")

# ------------------------------------------------------------------
# Case 2: High-value order notifications — filter + map + take
#         No intermediate list between filter and map
# ------------------------------------------------------------------
HIGH_VALUE = 500.0
NOTIFY_CAP = 200    # send at most 200 notifications per batch

t0 = time.perf_counter()
notifications = (
    Query(orders)
        .filter(lambda o: o["amount"] > HIGH_VALUE and o["status"] == "pending")
        .map(lambda o: {
            "order_id":   o["order_id"],
            "customer_id": o["customer_id"],
            "amount":     o["amount"],
            "message":    f"High-value order ${o['amount']:.2f} awaiting confirmation",
        })
        .take(NOTIFY_CAP)   # cap notifications — stop scanning once full
        .to_list()
)
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 2 — High-value notifications (>${HIGH_VALUE}, max {NOTIFY_CAP}):")
print(f"  Generated: {len(notifications)} notifications")
if notifications:
    print(f"  Sample: {notifications[0]['message']}")
print(f"  Time: {ms:.1f}ms")

# ------------------------------------------------------------------
# Case 3: Group by status — count and revenue per status
# ------------------------------------------------------------------
t0 = time.perf_counter()
by_status = (
    Query(orders)
        .group_by(lambda o: o["status"])
)
summary = by_status.agg(
    count   = lambda grp: grp.count(),
    revenue = lambda grp: round(grp.map(lambda o: o["amount"]).sum(), 2),
    avg_amt = lambda grp: round(grp.map(lambda o: o["amount"]).sum() / max(grp.count(), 1), 2),
)
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 3 — Orders by status:")
for row in sorted(summary, key=lambda r: r["revenue"], reverse=True):
    print(f"  {row['_key']:12s}  count={row['count']:6,}  "
          f"revenue=${row['revenue']:>12,.2f}  avg=${row['avg_amt']:>8.2f}")
print(f"  Time: {ms:.1f}ms")

# ------------------------------------------------------------------
# Case 4: Prime member return rate
# ------------------------------------------------------------------
t0 = time.perf_counter()
prime_orders   = Query(orders).filter(lambda o: o["is_prime"]).to_list()
prime_returned = Query(prime_orders).filter(lambda o: o["status"] == "returned").count()
std_orders     = Query(orders).filter(lambda o: not o["is_prime"]).to_list()
std_returned   = Query(std_orders).filter(lambda o: o["status"] == "returned").count()
ms = (time.perf_counter() - t0) * 1000

print(f"\nCase 4 — Return rate by membership:")
print(f"  Prime members: {prime_returned/len(prime_orders)*100:.2f}% return rate "
      f"({prime_returned:,}/{len(prime_orders):,})")
print(f"  Standard:      {std_returned/len(std_orders)*100:.2f}% return rate "
      f"({std_returned:,}/{len(std_orders):,})")
print(f"  Time: {ms:.1f}ms")
