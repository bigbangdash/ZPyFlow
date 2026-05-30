"""
06_etl_pipeline.py
------------------
ETL (Extract → Transform → Load) pipeline pattern.

Demonstrates:
  - Reading from multiple simulated sources (CSV-like, JSON-like)
  - Cleaning and validating records in a single fused pass
  - Enriching records with lookups
  - Writing to a target (in-memory dict as stand-in for a DB)
  - Measuring throughput
"""

from __future__ import annotations

import io
import csv
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Optional

from zpyflow import Query, col, from_csv, from_json_lines

random.seed(99)

# ------------------------------------------------------------------
# Simulate source data as in-memory file objects
# ------------------------------------------------------------------

def make_csv_source(n: int) -> io.StringIO:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["product_id", "name", "price", "category", "stock", "rating"])
    for i in range(n):
        w.writerow([
            f"P{i:06d}",
            f"Product {i}",
            round(random.uniform(0.5, 999.9), 2),
            random.choice(["electronics", "clothing", "food", "books", "toys"]),
            random.randint(0, 500),
            round(random.uniform(1.0, 5.0), 1),
        ])
    buf.seek(0)
    return buf

def make_jsonl_source(n: int) -> io.StringIO:
    buf = io.StringIO()
    for i in range(n):
        obj = {
            "event_id":   f"EVT{i:08d}",
            "user_id":    random.randint(1, 50_000),
            "product_id": f"P{random.randint(0, 9999):06d}",
            "action":     random.choice(["view", "view", "view", "add_to_cart", "purchase"]),
            "session_id": f"S{random.randint(1, 5000):05d}",
            "revenue":    round(random.uniform(5, 500), 2) if random.random() < 0.2 else None,
            "valid":      random.random() > 0.05,   # 5% invalid records
        }
        buf.write(json.dumps(obj) + "\n")
    buf.seek(0)
    return buf

print("Generating source data...")
csv_source  = make_csv_source(50_000)
jsonl_source = make_jsonl_source(200_000)
print("  50,000 product records (CSV)")
print("  200,000 event records (JSON Lines)\n")

# ------------------------------------------------------------------
# Case 1: Product ETL — clean → validate → transform
# ------------------------------------------------------------------

t0 = time.perf_counter()

# Step 1: Extract  (from_csv reads the StringIO object)
products_raw = from_csv(csv_source, has_header=True)   # list[dict]

# Step 2: Transform — filter invalid, enrich, reshape
products_clean = (
    products_raw
        .filter(lambda p: float(p["price"]) > 0 and int(p["stock"]) >= 0)
        .filter(lambda p: float(p["rating"]) >= 3.0)   # only quality products
        .map(lambda p: {
            "product_id":    p["product_id"],
            "name":          p["name"].strip(),
            "price_usd":     round(float(p["price"]), 2),
            "price_eur":     round(float(p["price"]) * 0.92, 2),  # FX conversion
            "category":      p["category"],
            "stock":         int(p["stock"]),
            "rating":        float(p["rating"]),
            "in_stock":      int(p["stock"]) > 0,
            "price_tier":    "budget" if float(p["price"]) < 20 else
                             "mid"    if float(p["price"]) < 100 else "premium",
        })
        .to_list()
)

etl_ms = (time.perf_counter() - t0) * 1000

print(f"Case 1 — Product ETL:")
print(f"  Input:  50,000 records")
print(f"  Output: {len(products_clean):,} valid records ({len(products_clean)/50_000*100:.1f}% pass rate)")
print(f"  Time:   {etl_ms:.1f}ms")

# Step 3: Load — store by category (stand-in for DB insert)
by_category: dict[str, list] = {}
for p in products_clean:
    by_category.setdefault(p["category"], []).append(p)

print("  Records per category:")
for cat, items in sorted(by_category.items()):
    avg_price = Query([i["price_usd"] for i in items]).sum() / len(items)
    print(f"    {cat:12s}  n={len(items):,}  avg_price=${avg_price:.2f}")

# ------------------------------------------------------------------
# Case 2: Event ETL — filter → aggregate → load
# ------------------------------------------------------------------

t0 = time.perf_counter()

events_raw = from_json_lines(jsonl_source)

# Only process valid purchase events with revenue
purchases = (
    events_raw
        .filter(lambda e: e["valid"] and e["action"] == "purchase" and e["revenue"] is not None)
        .map(lambda e: {
            "event_id":   e["event_id"],
            "user_id":    e["user_id"],
            "product_id": e["product_id"],
            "revenue":    float(e["revenue"]),
            "session_id": e["session_id"],
        })
        .to_list()
)

event_etl_ms = (time.perf_counter() - t0) * 1000

# Aggregate revenue — pull into f64 fast path
revenues = Query([p["revenue"] for p in purchases])

print(f"\nCase 2 — Event ETL (purchases):")
print(f"  Input:  200,000 event records")
print(f"  Output: {len(purchases):,} valid purchases")
print(f"  Time:   {event_etl_ms:.1f}ms")
print(f"  Total revenue: ${revenues.sum():,.2f}")
print(f"  Avg revenue:   ${revenues.sum() / revenues.count():.2f}")
print(f"  Max revenue:   ${revenues.max():.2f}")
print(f"  High-value (>$200): {revenues.filter(col > 200).count()} purchases")

# ------------------------------------------------------------------
# Case 3: Numeric column ETL — prices only, f64 fast path throughout
# ------------------------------------------------------------------

# Re-create CSV (already consumed above)
csv_source2 = make_csv_source(50_000)
t0 = time.perf_counter()

price_pipeline = (
    from_csv(csv_source2, column="price", dtype="float")
        .filter(col > 0)
        .filter(col.between(1.0, 500.0))   # exclude outliers
        .map(col * 1.15)                    # add 15% margin
        .to_list()
)

numeric_etl_ms = (time.perf_counter() - t0) * 1000
price_query = Query(price_pipeline)

print(f"\nCase 3 — Price column ETL (f64 fast path):")
print(f"  Time:     {numeric_etl_ms:.1f}ms")
print(f"  Records:  {len(price_pipeline):,}")
print(f"  Min:      ${price_query.min():.2f}")
print(f"  Max:      ${price_query.max():.2f}")
print(f"  Mean:     ${price_query.sum() / price_query.count():.2f}")

# ------------------------------------------------------------------
# Case 4: Throughput summary
# ------------------------------------------------------------------
total_records = 50_000 + 200_000 + 50_000
total_ms      = etl_ms + event_etl_ms + numeric_etl_ms
print(f"\nCase 4 — Total throughput:")
print(f"  {total_records:,} records in {total_ms:.1f}ms → "
      f"{total_records / total_ms * 1000:,.0f} records/sec")
