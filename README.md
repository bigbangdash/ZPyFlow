# ZPyFlow

> **⚠ Alpha — v0.1.0 is an early release for testing and feedback.**
> APIs may change before v1.0. Not recommended for production use without pinning the version.

Zero-allocation lazy query pipelines for Python, powered by Rust.

- **Lazy & fused** — filter + map + take run in a single pass with no intermediate lists
- **SIMD-accelerated** — float/int arrays execute in Rust with the GIL released, using `f64x4`
- **Expression DSL** — `col > 5` eliminates Python callbacks entirely
- **Python-friendly** — numpy, pandas, dataclasses, plain lists, and generators all work as input
- **Parallel execution** — `.parallel()` enables Rayon work-stealing

> **ZPyFlow is not a DataFrame engine.**
> It lets Python sequence hot paths — `list[float]`, numpy arrays, dict record streams —
> run as fused Rust pipelines without moving the surrounding codebase into a tabular data
> model.  See [§ 13 ZPyFlow vs Polars](#13-zpyflow-vs-polars) for the product-choice boundary.

---

## Installation

```bash
pip install zpyflow
```

Optional extras:

```bash
pip install zpyflow[numpy]   # NumPy integration
pip install zpyflow[arrow]   # PyArrow integration
```

To build from source, see [docs/contributing.md](docs/contributing.md).

---

## Contributing

See **[docs/contributing.md](docs/contributing.md)** for development setup,
Make commands, benchmarks, and how to submit changes.

---

## Contents

1. [Basic usage](#1-basic-usage)
2. [Expression DSL vs lambda](#2-expression-dsl-vs-lambda)
3. [numpy](#3-numpy)
4. [pandas](#4-pandas)
5. [Dataclasses and custom objects](#5-dataclasses-and-custom-objects)
6. [Dict records and log processing](#6-dict-records-and-log-processing)
7. [GroupBy and aggregation](#7-groupby-and-aggregation)
8. [CSV and JSON Lines streaming](#8-csv-and-json-lines-streaming)
9. [AI, embeddings, and LangChain / LangGraph](#9-ai-and-embedding-pipelines)
10. [Parallel execution](#10-parallel-execution)
11. [Full API reference](#11-full-api-reference)
12. [Performance](#12-performance)
13. [ZPyFlow vs Polars](#13-zpyflow-vs-polars)

---

## 1. Basic usage

```python
from zpyflow import Query, col

data = [1.5, -2.3, 0.7, 4.1, -0.5, 3.8, -1.1, 2.2]

result = (
    Query(data)
        .filter(col > 0)      # keep positive values
        .map(col * 2.0)       # double them
        .take(4)              # first 4
        .to_list()
)
# → [3.0, 1.4, 8.2, 7.6]

# Aggregations
total = Query(data).filter(col > 0).sum()    # 12.3
count = Query(data).filter(col > 0).count()  # 5
vmax  = Query(data).max()                    # 4.1
vmin  = Query(data).filter(col > 0).min()    # 0.7

first_positive = Query(data).filter(col > 0).first()  # 1.5
last_positive  = Query(data).filter(col > 0).last()   # 3.8
```

---

## 2. Expression DSL vs lambda

ZPyFlow supports two styles. Choose based on your data type.

### Expression DSL — recommended for numeric data (GIL released, SIMD)

```python
from zpyflow import col

(
    Query(data)
        .filter(col > 0)              # FilterGt     → Rust, SIMD
        .filter(col.between(0, 10))   # FilterBetween → Rust, SIMD
        .map(col * 2.0)               # MapMulScalar  → Rust, SIMD
        .map(col + 1.0)               # MapAddScalar  → Rust, SIMD
        .map(col.abs())               # MapAbs        → Rust
        .map(col.sqrt())              # MapSqrt       → Rust
        .map(col ** 2)                # MapPowScalar  → Rust
        .map(-col)                    # MapNeg        → Rust, SIMD
        .to_list()
)
```

**Supported DSL operators**

| Python expression  | Internal op        | SIMD | Description              |
|--------------------|--------------------|------|--------------------------|
| `col > x`          | FilterGt           | ✅   | greater than             |
| `col >= x`         | FilterGe           | ✅   | greater than or equal    |
| `col < x`          | FilterLt           | ✅   | less than                |
| `col <= x`         | FilterLe           | ✅   | less than or equal       |
| `col.between(a,b)` | FilterBetween      | ✅   | a ≤ v ≤ b               |
| `col * x`          | MapMulScalar       | ✅   | multiply by scalar       |
| `col + x`          | MapAddScalar       | ✅   | add scalar               |
| `col - x`          | MapSubScalar       | ✅   | subtract scalar          |
| `col / x`          | MapDivScalar       | ✅   | divide (via reciprocal)  |
| `col ** x`         | MapPowScalar       | —    | exponentiate             |
| `-col`             | MapNeg             | ✅   | negate                   |
| `col.abs()`        | MapAbs             | —    | absolute value           |
| `col.sqrt()`       | MapSqrt            | —    | square root              |
| `col.floor()`      | MapFloor           | —    | floor (round toward −∞)  |
| `col.ceil()`       | MapCeil            | —    | ceiling (round toward +∞)|
| `col.round()`      | MapRound           | —    | round to nearest integer |
| `col.reciprocal()` | MapReciprocal      | —    | 1 / x                    |

### Python lambda — for arbitrary Python objects (GIL held)

```python
(
    Query(records)
        .filter(lambda r: r["score"] > 80)
        .map(lambda r: r["name"].upper())
        .to_list()
)
```

> **Rule of thumb**: use the DSL for numeric arrays, use lambdas for arbitrary Python objects.
> The performance gap is roughly 20–40× at 1M elements.

---

## 3. numpy

```python
import numpy as np
from zpyflow import Query, col, from_numpy

rng = np.random.default_rng(42)
arr = rng.standard_normal(1_000_000)   # shape: (1M,), dtype: float64

# from_numpy() converts to Query using the f64 fast path
result = (
    from_numpy(arr)
        .filter(col > 0)        # SIMD filter, GIL released
        .map(col ** 2)          # square
        .take(10_000)
        .to_list()
)

# Aggregations on large arrays
positive = from_numpy(arr).filter(col > 0)
mean_positive = positive.sum() / positive.count()
max_val       = from_numpy(arr).max()

# Convert result back to numpy
result_arr = np.array(from_numpy(arr).filter(col.between(-1, 1)).to_list())
```

### Integer arrays

```python
ids = np.arange(1_000_000, dtype=np.int64)

# DSL path — fast, GIL released
big_ids = from_numpy(ids).filter(col > 500_000).to_list()

# Lambda fallback when DSL doesn't cover the operation
even_ids = from_numpy(ids).filter(lambda x: x % 2 == 0).take(100).to_list()
```

### Float32 arrays (ML / embedding workloads)

`from_numpy` routes `float32` arrays to a native **f32x8 SIMD path** (8 lanes per AVX2
register, twice the throughput of f64x4).  Results are promoted to Python `float` (f64)
at collection time; `to_numpy()` preserves the original `float32` dtype.

```python
# Typical ML scenario: embedding similarity post-filter
scores = np.random.rand(1_000_000).astype(np.float32)

# Zero-copy buffer read → f32x8 SIMD filter → 0 intermediate allocations
top_indices = from_numpy(scores).filter(col > 0.95).count()

# to_numpy() returns float32, not float64 — no precision loss
filtered = from_numpy(scores).filter(col > 0.9).to_numpy()
assert filtered.dtype == np.float32
```

### Explicit typed constructors

When a list contains mixed numeric types (e.g. `[1, 2, 3.0]`), `Query()` falls back to
the generic Python path.  Use the explicit constructors to force the fast path:

```python
# Mixed-type list → force f64 fast path (SIMD, GIL released)
Query.f64([1, 2, 3.0]).filter(col > 1).sum()   # → 5.0

# Force i64 fast path
Query.i64([1, 2, 3]).filter(col > 1).to_list() # → [2, 3]
```

### Speed comparison (1M float64)

```python
import time
import numpy as np
from zpyflow import from_numpy, col

arr = np.random.randn(1_000_000)

# numpy — eager, creates 2 intermediate arrays
t0 = time.perf_counter()
r_np = arr[arr > 0] * 2
print(f"numpy:   {(time.perf_counter()-t0)*1000:.1f}ms  (2 allocations)")

# ZPyFlow — single fused pass, 1 allocation, GIL released
t0 = time.perf_counter()
r_zpf = from_numpy(arr).filter(col > 0).map(col * 2).to_list()
print(f"zpyflow: {(time.perf_counter()-t0)*1000:.1f}ms  (1 allocation)")
```

---

## 4. pandas

ZPyFlow is not a pandas replacement. It accelerates **numeric column preprocessing** and
**row-level filtering** within existing pandas workflows.

### Processing a numeric column

```python
import pandas as pd
from zpyflow import Query, col

df = pd.DataFrame({
    "user_id": range(100_000),
    "score":   [float(i) / 100 for i in range(100_000)],
    "active":  [i % 3 != 0 for i in range(100_000)],
    "region":  ["JP" if i % 5 == 0 else "US" for i in range(100_000)],
})

# Extract, transform, and write back
scores = df["score"].tolist()

# Normalize scores above the median — single fused pass
median = df["score"].median()
normalized = (
    Query(scores)
        .filter(col > median)
        .map(col / df["score"].max())
        .to_list()
)
df_high = df[df["score"] > median].copy()
df_high["normalized"] = normalized
```

### Row-level processing via `to_dict("records")`

```python
records = df.to_dict("records")   # list[dict]

result = (
    Query(records)
        .filter(lambda r: r["active"] and r["region"] == "JP")
        .map(lambda r: {"user_id": r["user_id"], "score": r["score"] * 1.1})
        .take(1_000)
        .to_list()
)

result_df = pd.DataFrame(result)
```

### Wrapping ZPyFlow as a pandas transform step

```python
def fast_clip_and_scale(series: pd.Series, lo: float, hi: float) -> list[float]:
    """Filter to [lo, hi] and scale to [0, 1]."""
    span = hi - lo
    return (
        Query(series.tolist())
            .filter(col.between(lo, hi))
            .map((col - lo) / span)
            .to_list()
    )

df["score_scaled"] = fast_clip_and_scale(df["score"], lo=0.2, hi=0.8)
```

---

## 5. Dataclasses and custom objects

```python
from dataclasses import dataclass
from zpyflow import Query

@dataclass
class Employee:
    id: int
    name: str
    department: str
    salary: float
    years: int

employees = [
    Employee(1, "Alice",  "Engineering", 120_000, 5),
    Employee(2, "Bob",    "Marketing",    85_000, 3),
    Employee(3, "Carol",  "Engineering", 140_000, 8),
    Employee(4, "Dan",    "HR",           75_000, 2),
    Employee(5, "Eve",    "Engineering", 110_000, 4),
    Employee(6, "Frank",  "Marketing",   90_000,  6),
    Employee(7, "Grace",  "HR",          80_000,  7),
]

# Filter + project
senior_engineers = (
    Query(employees)
        .filter(lambda e: e.department == "Engineering" and e.years >= 5)
        .map(lambda e: e.name)
        .to_list()
)
# → ["Alice", "Carol"]

# Top earners as (name, salary) tuples
top_earners = (
    Query(employees)
        .filter(lambda e: e.salary > 100_000)
        .map(lambda e: (e.name, e.salary))
        .to_list()
)
# → [("Alice", 120000), ("Carol", 140000), ("Eve", 110000)]

# Any / all
has_hr = Query(employees).any(lambda e: e.department == "HR")       # True
all_ft = Query(employees).all(lambda e: e.years > 0)                 # True

# Reduce to compute total salary
total_salary = (
    Query(employees)
        .reduce(lambda acc, e: acc + e.salary, initial=0.0)
)

# Extract salaries as float list → use f64 fast path for aggregation
salaries = Query([e.salary for e in employees])
avg_salary = salaries.sum() / salaries.count()   # GIL released for sum
```

### Pydantic models

```python
from pydantic import BaseModel
from zpyflow import Query

class Order(BaseModel):
    order_id: str
    amount: float
    status: str
    customer_id: int

orders = [Order(**o) for o in raw_orders]

# Filter pending high-value orders
large_pending = (
    Query(orders)
        .filter(lambda o: o.status == "pending" and o.amount > 10_000)
        .map(lambda o: {"order_id": o.order_id, "amount": o.amount})
        .take(50)
        .to_list()
)

# Fast aggregation on the amount field alone
amounts = Query([o.amount for o in orders])
total_pending = (
    Query(orders)
        .filter(lambda o: o.status == "pending")
        .map(lambda o: o.amount)
        .reduce(lambda acc, x: acc + x, initial=0.0)
)
```

---

## 6. Dict records and log processing

```python
from zpyflow import Query, field

logs = [
    {"ts": "2024-01-15T10:23:11", "level": "ERROR", "status": 500, "path": "/api/users", "latency_ms": 312},
    {"ts": "2024-01-15T10:23:12", "level": "INFO",  "status": 200, "path": "/api/items", "latency_ms": 45},
    {"ts": "2024-01-15T10:23:13", "level": "WARN",  "status": 429, "path": "/api/users", "latency_ms": 8},
    {"ts": "2024-01-15T10:23:14", "level": "ERROR", "status": 500, "path": "/health",    "latency_ms": 520},
    # ... millions of records
]

# ✅ OK — field() DSL: GIL released after first filter, SIMD for numeric fields
slow_errors = (
    Query(logs)
        .filter(field("status") >= 500)
        .filter(field("latency_ms") > 100)
        .count()
)

# ✅ OK — field() DSL + map_field(): single fused Rust loop
slow_paths = (
    Query(logs)
        .filter(field("latency_ms") > 100)
        .map_field("path")
        .to_list()
)

# ✅ OK — map with dict construction: no DSL equivalent, lambda is the right choice
errors = (
    Query(logs)
        .filter(field("level") == "ERROR")
        .map(lambda l: {"path": l["path"], "latency_ms": l["latency_ms"], "ts": l["ts"]})
        .to_list()
)
errors.sort(key=lambda l: l["latency_ms"], reverse=True)

# NG — numeric filter via lambda: GIL held per element, same speed as Python
# Query(logs).filter(lambda l: l["latency_ms"] > 100).to_list()

# NG — single field extraction via lambda when map_field() exists
# Query(logs).filter(...).map(lambda l: l["path"]).to_list()

# ✅ OK — compound condition (AND of two fields): no DSL equivalent, lambda is correct
active_errors = (
    Query(logs)
        .filter(lambda l: l["level"] == "ERROR" and l["latency_ms"] > 100)
        .to_list()
)

# Extract latency as float list → f64 fast path for aggregation
latencies = Query([float(l["latency_ms"]) for l in logs])
avg_latency = latencies.sum() / latencies.count()
max_latency = latencies.max()
```

> **Rule of thumb for dict records**
> - Single-field numeric/equality filter → `field()` DSL (GIL released, faster at large N)
> - Single-field extraction → `map_field("key")` (fused with field filter in one Rust loop)
> - Multi-field compound condition (`and` / `or`) → lambda (no DSL equivalent)
> - Map that builds a new dict → lambda (no DSL equivalent)

### Multiple queries on the same dataset — use `.preload()`

`field()` pays a one-time dict→RustObj conversion cost on the first filter call (GIL held,
O(N)).  For a single query this cost amortizes across the filter pass itself.  When you
run **multiple queries over the same dataset**, call `.preload()` first to pay the
conversion once:

```python
# ✅ OK — preload() converts once, all subsequent filters run GIL-free
q = Query(logs).preload()
slow_count   = q.filter(field("latency_ms") > 100).count()
error_count  = q.filter(field("status") >= 500).count()

# Count by status code
from collections import Counter
status_counts = Counter(Query(logs).map(lambda l: l["status"]).to_list())
# → Counter({200: 1, 500: 2, 429: 1})
```

### Reading directly from JSON Lines

```python
from zpyflow import from_json_lines, col

# Extract a single numeric field — uses f64 fast path
latencies = from_json_lines("access.log.ndjson", field="latency_ms", dtype="float")

stats = {
    "count": latencies.count(),
    "total": latencies.sum(),
    "max":   latencies.max(),
    "above_200ms": latencies.filter(col > 200).count(),
}
```

---

## 7. GroupBy and aggregation

```python
from zpyflow import Query, GroupBy

transactions = [
    {"user": "alice", "amount": 120.0, "category": "food"},
    {"user": "bob",   "amount":  45.0, "category": "transport"},
    {"user": "alice", "amount": 300.0, "category": "shopping"},
    {"user": "carol", "amount":  80.0, "category": "food"},
    {"user": "bob",   "amount": 200.0, "category": "shopping"},
    {"user": "alice", "amount":  55.0, "category": "food"},
]

by_user = GroupBy(transactions, key_fn=lambda t: t["user"])

# Per-user count and total
summary = by_user.agg(
    count=lambda g: g.count(),
    total=lambda g: Query([t["amount"] for t in g.to_list()]).sum(),
)
# → [{"_key": "alice", "count": 3, "total": 475.0}, ...]

# Fetch a single group
alice_txns = by_user.get_group("alice").to_list()

# Sum per group using a field extractor
by_category = GroupBy(transactions, key_fn=lambda t: t["category"])
category_totals = by_category.sum_per_group(field=lambda t: t["amount"])
# → {"food": 255.0, "transport": 45.0, "shopping": 500.0}

# Count per group
counts = by_user.count_per_group()
# → {"alice": 3, "bob": 2, "carol": 1}
```

### Single-pass aggregation with `group_agg`

`group_agg` runs count, sum, mean, max, and min in **one Rust pass** over the data,
avoiding intermediate list materialization.

```python
from zpyflow import Query, field, agg_count, agg_sum, agg_mean

transactions = [
    {"user": "alice", "amount": 120.0, "category": "food"},
    {"user": "bob",   "amount":  45.0, "category": "transport"},
    {"user": "alice", "amount": 300.0, "category": "shopping"},
    {"user": "carol", "amount":  80.0, "category": "food"},
    {"user": "bob",   "amount": 200.0, "category": "shopping"},
    {"user": "alice", "amount":  55.0, "category": "food"},
]

# Lambda key
result = (
    Query(transactions)
        .group_agg(
            lambda t: t["user"],
            count   = agg_count(),
            total   = agg_sum(lambda t: t["amount"]),
        )
)
# → [{"_key": "alice", "count": 3, "total": 475.0},
#    {"_key": "bob",   "count": 2, "total": 245.0},
#    {"_key": "carol", "count": 1, "total":  80.0}]

# field() DSL key — Rust-side key extraction after dict→RustObj conversion
by_category = (
    Query(transactions)
        .group_agg(
            field("category"),
            count   = agg_count(),
            revenue = agg_sum(lambda t: t["amount"]),
            avg     = agg_mean(lambda t: t["amount"]),
        )
)
```

> **When to prefer `group_agg` over `GroupBy`**: `group_agg` is a single Rust pass and
> returns immediately.  Use `GroupBy` when you need per-group `Query` objects for
> multi-step operations (pagination, nested queries).

---

## 8. CSV and JSON Lines streaming

```python
from zpyflow import from_csv, from_json_lines, col

# Read a single numeric column from CSV — f64 fast path
prices = from_csv("products.csv", column="price", dtype="float")

discounted = (
    prices
        .filter(col > 10.0)
        .map(col * 0.9)
        .to_list()
)

# Read all rows as dicts
products = from_csv("products.csv")   # column=None → list[dict]

premium = (
    products
        .filter(lambda p: float(p["price"]) > 100 and p["in_stock"] == "true")
        .map(lambda p: {"name": p["name"], "price": float(p["price"])})
        .to_list()
)

# JSON Lines — filter at the source level
events = from_json_lines("events.ndjson")
errors = (
    events
        .filter(lambda e: e.get("level") == "error")
        .take(1_000)
        .to_list()
)
```

---

## 9. AI and embedding pipelines

Similarity score arrays are a natural fit for ZPyFlow's f64 fast path: large, homogeneous,
numeric, and the filtering threshold is known at query time.

```python
import numpy as np
from zpyflow import Query, col, from_numpy

# Pre-computed cosine similarity scores against a query vector (1M documents)
scores = np.random.uniform(0, 1, size=1_000_000).astype(np.float64)
doc_ids = np.arange(1_000_000, dtype=np.int64)

# Filter by threshold, retrieve top-K candidates — SIMD, GIL released
THRESHOLD = 0.85
TOP_K = 100

candidate_scores = from_numpy(scores).filter(col > THRESHOLD).to_list()
candidate_ids    = from_numpy(doc_ids).filter(col > THRESHOLD).to_list()

# Pair and rank
candidates = sorted(
    zip(candidate_scores, range(len(candidate_scores))),
    reverse=True,
)[:TOP_K]
```

### Batch inference scoring statistics

```python
def score_batch_stats(batch_scores: list[float]) -> dict:
    q = Query(batch_scores)
    n = q.count()
    return {
        "n":          n,
        "mean":       q.sum() / n,
        "high_conf":  q.filter(col > 0.9).count(),
        "low_conf":   q.filter(col < 0.5).count(),
        "max":        q.max(),
    }

# Apply to each inference batch (all Rust, no GIL for the numeric ops)
batches = [scores[i:i+10_000].tolist() for i in range(0, len(scores), 10_000)]
batch_stats = [score_batch_stats(b) for b in batches]
```

### Embedding norm validation

```python
# Detect embeddings that slipped through without L2 normalization
norms = [float(np.linalg.norm(emb)) for emb in embeddings]

unnormalized_count = (
    Query(norms)
        .filter(lambda n: abs(n - 1.0) > 0.01)
        .count()
)
print(f"{unnormalized_count} embeddings need re-normalization")

# Fast summary of norm distribution
norm_query = Query(norms)
print(f"min={norm_query.min():.4f}  max={norm_query.max():.4f}  "
      f"mean={norm_query.sum()/norm_query.count():.4f}")
```

### LangChain / LangGraph integration

ZPyFlow slots into LangChain and LangGraph wherever a node processes large numeric arrays.
No special integration is needed — just use it in your node functions or tools.

**RAG retrieval — filter similarity scores with early stopping**

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from zpyflow import from_numpy, col

class ZPyFlowRetriever(BaseRetriever):
    """Retriever that uses ZPyFlow for fast threshold filtering."""

    docs: list[Document]
    embeddings: object  # any embedding model

    def _get_relevant_documents(self, query: str) -> list[Document]:
        import numpy as np

        query_vec = self.embeddings.embed_query(query)
        scores = np.array([
            np.dot(query_vec, self.embeddings.embed_documents([d.page_content])[0])
            for d in self.docs
        ])

        # SIMD filter + early stopping — never scans beyond the K-th hit
        top_indices = (
            from_numpy(scores)
            .filter(col > 0.7)
            .take(20)
            .to_list()
        )
        return [self.docs[int(i)] for i in top_indices]
```

**LangGraph node — aggregate tool results without materializing a full list**

```python
from langgraph.graph import StateGraph, MessagesState

def score_filter_node(state: MessagesState) -> dict:
    """LangGraph node: filter and aggregate a large score array from a tool call."""
    scores: list[float] = state["tool_scores"]  # e.g. 500K candidates

    q = Query(scores)
    return {
        "candidate_count": q.filter(col > 0.8).count(),   # stays in Rust
        "top_score":       q.max(),
        "mean_score":      q.sum() / q.count(),
    }

graph = StateGraph(MessagesState)
graph.add_node("score_filter", score_filter_node)
```

**LangChain tool — return pre-aggregated stats to the LLM**

```python
from langchain_core.tools import tool
from zpyflow import Query, col

@tool
def analyze_search_results(scores: list[float]) -> dict:
    """Aggregate similarity scores from a vector search."""
    q = Query(scores)
    n = q.count()
    return {
        "total":        n,
        "high_quality": q.filter(col > 0.85).count(),
        "low_quality":  q.filter(col < 0.5).count(),
        "best_score":   q.max(),
        "mean_score":   round(q.sum() / n, 4) if n else 0.0,
    }
```

> **When ZPyFlow helps in AI pipelines**: large numeric score arrays (similarity, confidence,
> reward, logprobs) where you threshold-filter and aggregate. It does not speed up LLM calls
> themselves (those are I/O-bound) or small lists (< 10K elements).

---

## 10. Parallel execution

`.parallel()` applies to **numeric fast paths only** (f64 / i64). Python object pipelines
ignore this hint.

```python
from zpyflow import Query, col, from_numpy
import numpy as np

large = np.random.randn(10_000_000).tolist()

# Single-threaded: SIMD + GIL released
result_single   = Query(large).filter(col > 0).map(col * 2).to_list()

# Multi-threaded: Rayon work-stealing, GIL fully released
result_parallel = Query(large).filter(col > 0).map(col * 2).parallel().to_list()

# Aggregation also parallelizes
total = Query(large).filter(col > 0).parallel().sum()
```

| Data size | Single-thread (SIMD) | Parallel (8 cores) |
|-----------|---------------------|--------------------|
| 1M  f64   | ~3ms                | ~0.8ms             |
| 10M f64   | ~30ms               | ~5ms               |
| 100M f64  | ~300ms              | ~45ms              |

> **Note**: threading overhead (split + join) means parallel mode is slower than
> single-threaded for inputs under ~500K elements. Profile before enabling it.

---

## 11. Full API reference

### Constructors / source adapters

```python
from zpyflow import Query, from_numpy, from_arrow, from_csv, from_json_lines

Query([1.0, 2.0, 3.0])               # list[float] → f64 fast path
Query([1, 2, 3])                      # list[int]   → i64 fast path
Query(["a", "b"])                     # list[str]   → Python path
Query(x**2 for x in range(100))      # generator (consumed eagerly)
Query(dict_or_obj_list)               # any iterable

from_numpy(np_array)                  # 1-D numpy array; bool/uint8 stay compact
from_arrow(pa_array_or_table)         # PyArrow Array / ChunkedArray / Table
from_csv("data.csv", column="price") # CSV single-column
from_csv("data.csv")                  # CSV all rows as list[dict]
from_json_lines("log.ndjson")         # NDJSON all rows as list[dict]
from_json_lines("log.ndjson",
                field="value",
                dtype="float")        # NDJSON single numeric field
```

### Lazy combinators (deferred until terminal call)

```python
q.filter(col > 0)               # DSL predicate — Rust, SIMD where possible
q.filter(lambda x: x > 0)       # Python callable — GIL held
q.map(col * 2.0)                 # DSL transform — Rust, SIMD where possible
q.map(lambda x: x * 2)          # Python callable — GIL held
q.map_field("name")              # extract one field from dict records (fused with field() filter)
q.take(n)                        # keep first n elements
q.skip(n)                        # drop first n elements
q.take_while(pred)               # take while pred holds, then stop
q.skip_while(pred)               # skip while pred holds, then emit remainder
q.parallel()                     # request parallel execution (numeric only)
```

### Terminal operations (trigger execution)

```python
q.to_list()                      # collect to Python list
q.to_numpy()                     # collect to numpy ndarray (no per-element boxing)
q.to_dict(key=fn, value=fn)      # collect to Python dict
q.to_bytes()                     # raw f64 bytes (for zero-copy numpy.frombuffer)
q.count()                        # number of elements
q.sum()                          # sum (numeric paths use SIMD)
q.min()                          # minimum value
q.max()                          # maximum value (SIMD for f64)
q.mean()                         # arithmetic mean, or None if empty
q.var()                          # population variance (ddof=0), or None if empty
q.std()                          # standard deviation, or None if empty
q.stats()                        # count/sum/mean/min/max in one pass — {"count": N, ...}
q.first()                        # first element, or None
q.last()                         # last element, or None
q.reduce(fn, initial=val)        # general fold
q.for_each(fn)                   # consume with side effect, returns None
q.any(pred)                      # True if any element satisfies pred
q.all(pred)                      # True if all elements satisfy pred
```

---

## 12. Performance

### Benchmark: 1M float64 elements — `filter(x > 0) + map(x * 2) + take(10_000)`

| Approach                          | Time      | Allocations | GIL     |
|-----------------------------------|-----------|-------------|---------|
| Python list comprehension         | ~80ms     | 2 lists     | held    |
| Python generator + take           | ~40ms     | 0           | held    |
| numpy (`arr[arr > 0] * 2`)        | ~8ms      | 2 arrays    | released|
| ZPyFlow lambda (Python callback)  | ~70–80ms  | 1 list      | held    |
| ZPyFlow Expression DSL (SIMD)     | ~2–5ms    | 1 list      | released|
| ZPyFlow Expression DSL + parallel | ~0.5–1ms  | 1 list      | released|

> **Note:** The lambda path (Python callback) is GIL-bound and offers throughput
> comparable to a plain list comprehension — the benefit over raw Python is
> the unified pipeline API, not speed.  For maximum throughput, use the
> Expression DSL (e.g. `col > 0`, `col * 2`) which releases the GIL and uses SIMD.

### Running the benchmarks

```bash
# Rust (Criterion) — detailed per-operation breakdown
cargo bench --bench pipeline

# SIMD selectivity analysis (10% / 25% / 50% / 75% / 90% pass rate)
cargo bench --bench simd_filter

# Python (pytest-benchmark)
pip install pytest-benchmark
pytest tests/test_performance.py -v --benchmark-autosave
```

### Measuring memory

```bash
pip install memory-profiler
python -m memory_profiler your_script.py
```

Raw benchmark JSON results are saved in
[`sandbox/benchmark/results/`](sandbox/benchmark/results/) after each
`make dc-bench` run.  Use `make dc-bench-compare` to diff against a saved baseline.

---

## 13. ZPyFlow vs Polars

ZPyFlow is not a Polars replacement.

### When to use ZPyFlow

- Data already lives in Python as `list[float]`, numpy arrays, dict records, or generators
- The hot path is a **single fused pipeline**: `filter → take`, `filter → count`, `filter → sum`
- Moving to a DataFrame model would require re-architecting surrounding code
- You need the GIL released from a Python list without changing the calling code
- Early-stop semantics matter: `filter(col > t).take(K)` scans only until K results are found

### When to use Polars (or pandas)

- The workload involves **multi-column joins** or window functions
- You need stats **across multiple columns at once**, or complex GROUP BY analytics
- Data is table-shaped and loaded from a file from the start
- SQL-style GROUP BY over multiple columns with complex group logic

### Product-choice comparison

| Scenario | ZPyFlow | Polars |
|---|---|---|
| `filter(col > t).count()` on 1M floats | ✅ ~2ms, SIMD, GIL released | ✅ ~5ms (columnar) |
| `filter(col > t).take(K)` with small K | ✅ Early-stop, scans only until K found | ⚠️ Scans all N first |
| `filter + map + sum` in one pass | ✅ Fused, 1 allocation | ✅ Columnar, 2–3 allocations |
| count + sum + mean + min + max in one pass | ✅ `stats()` — 1 SIMD pass, GIL released | ✅ 1 columnar pass |
| Multi-column join | ❌ Not supported | ✅ Native |
| Loading CSV + analyzing as a table | ⚠️ Awkward | ✅ Natural |
| Arbitrary Python objects (dicts, dataclasses) | ✅ Lambda path | ⚠️ Requires schema |
| Embedding threshold + top-K (ANN post-filter) | ✅ Fast early-stop | ⚠️ No early-stop |

**Core rule**: if data is already Python sequences and the pipeline is simple, ZPyFlow
removes allocation and GIL cost with zero framework migration.  If the data is tabular
or the analysis spans multiple columns, use Polars.

### The API-response case — Polars is not in the picture

Polars is rarely considered for API response processing, and for good reason:

```python
# Typical API response: list[dict], schema unknown, nullable fields
resp = httpx.get("https://api.example.com/events")
events = resp.json()          # list[dict] — arbitrary structure

# Using Polars requires a round-trip through DataFrame
df = pl.DataFrame(events)     # schema inference + full materialization
active = df.filter(pl.col("status") == "active").to_dicts()  # back to list[dict]
```

That `list[dict] → DataFrame → list[dict]` round-trip pays schema inference and
columnar conversion costs — for simple filtering it is strictly slower and more
code than a list comprehension.

In practice, API response processing looks like one of these:

```python
# Before ZPyFlow — plain Python
active = [e for e in events if e["status"] == "active"]
top    = sorted(active, key=lambda e: e["score"], reverse=True)[:100]

# With ZPyFlow — same semantics, GIL released after first field() filter
active = Query(events).filter(field("status") == "active").to_list()
top    = Query(events).filter(field("score") >= threshold).take(100).to_list()
```

**In this space ZPyFlow's real competitor is plain Python, not Polars.**
The relevant comparison is not ZPyFlow vs Polars, but ZPyFlow vs list comprehensions
on `list[dict]` — and at N > 50K that gap is 3–8× in ZPyFlow's favour.

---

## Design background

ZPyFlow is inspired by [ZLinq](https://github.com/Cysharp/ZLinq) (zero-allocation LINQ for C# / .NET).

ZLinq fuses `Where().Select().Take()` into a single loop at JIT time using CLR generic
specialization.  ZPyFlow achieves the same fusion **inside the Rust core** using the
`ZStream` trait — every operator is a concrete generic type, LLVM inlines the full
chain at `-O3`.  The PyO3 boundary is the only place dynamic dispatch appears, and it
is crossed once per terminal call, not once per element.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design document.
