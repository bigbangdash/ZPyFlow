"""
05_log_processing.py
--------------------
Realistic log / event processing with ZPyFlow.

Simulates:
  - Web server access logs (dict records)
  - Extracting numeric fields for fast aggregation
  - Summarizing error patterns
  - Latency percentile approximation
"""

from __future__ import annotations

import random
import time
from collections import Counter, defaultdict

from zpyflow import Query, col, GroupBy

random.seed(7)

# ------------------------------------------------------------------
# Simulate access log records
# ------------------------------------------------------------------
PATHS    = ["/api/users", "/api/orders", "/api/products", "/health", "/auth/login"]
LEVELS   = ["INFO", "INFO", "INFO", "WARN", "ERROR"]   # weighted
STATUSES = [200, 200, 200, 200, 201, 204, 400, 429, 500, 503]

def make_logs(n: int) -> list[dict]:
    logs = []
    base_ts = 1_700_000_000
    for i in range(n):
        level  = random.choice(LEVELS)
        status = random.choice(STATUSES)
        if level == "ERROR":
            status = random.choice([500, 503])
        logs.append({
            "ts":         base_ts + i,
            "level":      level,
            "status":     status,
            "path":       random.choice(PATHS),
            "latency_ms": random.lognormvariate(3.5, 1.2),  # ~20–2000ms
            "user_id":    random.randint(1, 10_000) if status != 503 else None,
            "bytes":      random.randint(100, 50_000),
        })
    return logs

logs = make_logs(100_000)
print(f"Generated {len(logs):,} log records\n")

# ------------------------------------------------------------------
# Case 1: Count by level
# ------------------------------------------------------------------
level_counts = Counter(Query(logs).map(lambda l: l["level"]).to_list())
print("Case 1 — Log level distribution:")
for level, n in sorted(level_counts.items()):
    print(f"  {level:5s}: {n:,}")

# ------------------------------------------------------------------
# Case 2: Error rate by path
# ------------------------------------------------------------------
by_path = GroupBy(logs, key_fn=lambda l: l["path"])

print("\nCase 2 — Error rate by path:")
for path in sorted(PATHS):
    path_logs = by_path.get_group(path).to_list()
    if not path_logs:
        continue
    total  = len(path_logs)
    errors = Query(path_logs).filter(lambda l: l["status"] >= 500).count()
    print(f"  {path:20s}  total={total:,}  errors={errors:,}  rate={errors/total*100:.1f}%")

# ------------------------------------------------------------------
# Case 3: Latency statistics (f64 fast path for aggregation)
# ------------------------------------------------------------------
latencies = Query([l["latency_ms"] for l in logs])

p50_approx = sorted(latencies.to_list())[len(logs) // 2]
p95_approx = sorted(latencies.to_list())[int(len(logs) * 0.95)]
p99_approx = sorted(latencies.to_list())[int(len(logs) * 0.99)]

print(f"\nCase 3 — Latency (all requests):")
print(f"  avg={latencies.sum()/latencies.count():.1f}ms")
print(f"  max={latencies.max():.1f}ms")
print(f"  p50≈{p50_approx:.1f}ms  p95≈{p95_approx:.1f}ms  p99≈{p99_approx:.1f}ms")

# ------------------------------------------------------------------
# Case 4: Slow requests (> 500ms) — extract and summarize
# ------------------------------------------------------------------
slow = (
    Query(logs)
        .filter(lambda l: l["latency_ms"] > 500)
        .map(lambda l: {"path": l["path"], "latency_ms": round(l["latency_ms"], 1), "status": l["status"]})
        .to_list()
)
slow.sort(key=lambda l: l["latency_ms"], reverse=True)

print(f"\nCase 4 — Slow requests (> 500ms): {len(slow):,}")
print("  Top 5 slowest:")
for r in slow[:5]:
    print(f"    {r['path']:20s}  {r['latency_ms']:7.1f}ms  HTTP {r['status']}")

# ------------------------------------------------------------------
# Case 5: 5xx errors — group by path, count unique user_ids affected
# ------------------------------------------------------------------
errors_5xx = (
    Query(logs)
        .filter(lambda l: l["status"] >= 500 and l["user_id"] is not None)
        .to_list()
)

affected_users_by_path: dict[str, set] = defaultdict(set)
for log in errors_5xx:
    affected_users_by_path[log["path"]].add(log["user_id"])

print(f"\nCase 5 — 5xx errors: {len(errors_5xx):,} total")
print("  Unique users affected per path:")
for path, users in sorted(affected_users_by_path.items()):
    print(f"    {path:20s}  {len(users):,} users")

# ------------------------------------------------------------------
# Case 6: Byte throughput per path (f64 fast path per group)
# ------------------------------------------------------------------
print("\nCase 6 — Byte throughput by path (f64 fast path):")
for path in sorted(PATHS):
    path_bytes = [float(l["bytes"]) for l in logs if l["path"] == path]
    total_mb   = Query(path_bytes).sum() / 1_048_576
    print(f"  {path:20s}  {total_mb:7.2f} MB")

# ------------------------------------------------------------------
# Case 7: Real-time window simulation (rolling 1000-record window)
# ------------------------------------------------------------------
WINDOW = 1_000
print(f"\nCase 7 — Rolling {WINDOW}-record error rate (last 5 windows):")
for start in range(len(logs) - WINDOW * 5, len(logs), WINDOW):
    window = logs[start : start + WINDOW]
    error_rate = Query(window).filter(lambda l: l["status"] >= 500).count() / WINDOW * 100
    print(f"  records [{start:,}–{start+WINDOW:,}]  error_rate={error_rate:.1f}%")
