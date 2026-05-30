# ZPyFlow Benchmark Sandbox

Mirrors [ZLinq's `sandbox/Benchmark/`](https://github.com/Cysharp/ZLinq/tree/main/sandbox/Benchmark) in structure and intent.

```
sandbox/benchmark/
├── run.py              # entry point  (≈ Program.cs)
├── categories.py       # tag constants (≈ Categories.cs)
├── conftest.py         # pytest-benchmark config
├── models/
│   └── generators.py   # reproducible test data (≈ Models/ + TestData/)
├── benchmarks/
│   ├── bench_filter.py        # filter, selectivity sweep
│   ├── bench_chained.py       # filter+map+take, chain depth, small-N
│   ├── bench_aggregation.py   # sum / count / max — stay-in-Rust path
│   ├── bench_vs_numpy.py      # head-to-head vs numpy
│   ├── bench_objects.py       # dict / dataclass (Python object path)
│   ├── bench_vector_search.py # ANN post-filtering, top-K early stopping
│   ├── bench_ml_feature.py    # outlier removal + normalization pipeline
│   ├── bench_etl.py           # multi-stat aggregation, vs Polars/Pandas
│   ├── bench_fraud.py         # risk score thresholding, review queue, exposure sum
│   └── bench_groupby.py       # GroupBy vs Counter/defaultdict, pagination
└── results/            # saved benchmark JSON (gitignored except .gitkeep)
```

---

## Setup

```bash
# Build the extension first
maturin develop --release

# Install benchmark dependencies
pip install pytest pytest-benchmark numpy
```

---

## Running benchmarks

```bash
cd sandbox/benchmark

# All suites, default config
python run.py

# Single suite
python run.py --suite filter
python run.py --suite chained
python run.py --suite vs_numpy
python run.py --suite aggregation
python run.py --suite objects
python run.py --suite vector_search
python run.py --suite ml_feature
python run.py --suite etl
python run.py --suite fraud
python run.py --suite groupby

# Filter by data size (xs=100, s=1K, m=10K, l=100K, xl=1M, xxl=10M)
python run.py --suite filter --n xl

# Precise config (more iterations)
python run.py --suite chained --config precise

# Save baseline, then compare later
python run.py --suite filter --save
# ... make changes ...
python run.py --suite filter --compare   # fails if >10% regression

# Direct pytest (more control)
pytest benchmarks/bench_filter.py -v --benchmark-columns=mean,stddev,ops
pytest benchmarks/bench_chained.py -v -k "xl and not lambda"
pytest benchmarks/ -v --benchmark-compare   # compare all vs saved
```

---

## Benchmark suites

### `bench_filter.py`
Filter benchmarks across sizes and selectivities.

Key questions:
- At what N does ZPyFlow DSL beat Python list comprehension?
- How does SIMD perform at 10% / 50% / 90% selectivity?
- Does `count()` vs `to_list()` matter?

### `bench_chained.py`
The core value proposition: `filter + map + take` in one fused pass.

Key questions:
- Does adding more operators to the chain cost more time? (It shouldn't with fusion.)
- At what N does ZPyFlow overtake Python generators?
- How bad is the overhead for small N (10, 100, 1K)?

### `bench_aggregation.py`
Aggregations that stay inside Rust (`sum`, `count`, `max`, `min`).

Key insight: these are **always** faster than `to_list()` + Python aggregate,
even for moderate N, because no `PyObject` is created per element.

### `bench_vs_numpy.py`
Head-to-head with numpy.

Key questions:
- numpy wins on pure arithmetic (it's C + SIMD). By how much?
- ZPyFlow wins on `filter + map + take` with early termination. By how much?
- Does `from_numpy()` + ZPyFlow beat `arr[arr>0]*2` on memory efficiency?

### `bench_objects.py`
Python object path (dict / dataclass records, GIL held).

Key questions:
- Is ZPyFlow's object path at least as fast as Python list comprehension?
- Does avoiding intermediate lists help for `filter + map + take`?
- How does `count()` compare to `sum(1 for x in ...)`?

### `bench_vector_search.py`
Use case: ANN post-filtering (FAISS / Qdrant result re-ranking).

After retrieval, 1M cosine similarity scores are filtered by threshold and the
top-K candidates are returned.  `take(K)` stops as soon as K results are
collected — ZPyFlow never scans the full array, whereas numpy must.

Key question:
- How much does early stopping matter at different K and pass-rate combinations?

### `bench_ml_feature.py`
Use case: ML feature preprocessing before model training.

Pipeline: `filter(col.between(-CLIP, CLIP))` → `map(col * SCALE)` → optional `take(N)`.
Tests `FilterBetween` (not covered by other suites) combined with a map transform.

Key question:
- When does the single-pass fused pipeline beat numpy's two-allocation path?
  (Answer: whenever `take()` is involved; numpy wins on full output.)

### `bench_etl.py`
Use case: batch aggregation jobs (price/revenue stats).

Computes count + sum + max of filtered values in one job.
ZPyFlow needs 3 separate pipeline executions; Polars does it in one columnar pass.

Key question:
- What is the cost of ZPyFlow's per-stat overhead vs columnar libraries?
  (Answer: Polars ~4× faster for multi-stat; ZPyFlow wins for single-stat.)

### `bench_fraud.py`
Use case: financial transaction risk scoring (Fintech / Insurance).

Log-normal risk scores — most transactions are low-risk, a long tail is high-risk.
Tests three patterns from the fraud detection pipeline:
- `filter + count` (flag count for reporting)
- `filter + take` (fill review queue — early stopping)
- `filter + sum`  (total monetary exposure)

Key question:
- How does early stopping in `.take()` compare to numpy (which always scans all N)?

### `bench_groupby.py`
Use case: order status grouping, user cohort analytics, content category stats,
          paginated content feeds (E-commerce / SaaS / Media).

GroupBy is a pure-Python layer — speed is comparable to Python's `Counter` /
`defaultdict`.  The value is the chainable API and the `filter → group_by`
pipeline with no intermediate list.

Also benchmarks `skip + take` (pagination) on the object path.

Key questions:
- How does `group_by + count_per_group` compare to `Counter`?
- How does `group_by + agg` compare to manual `defaultdict` aggregation?
- How much overhead does ZPyFlow's pagination (skip + take) add vs Python islice?

---

## Reading results

```
Name (time in ms)                         Mean      StdDev      Ops/s
─────────────────────────────────────────────────────────────────────
bench_filter/test_python_listcomp_xl     82.31      1.23       12.1
bench_filter/test_numpy_xl               8.04       0.31      124.4
bench_filter/test_zpyflow_dsl_xl         4.21       0.18      237.5   ← winner
bench_filter/test_zpyflow_lambda_xl     79.44       1.87       12.6
```

- **Mean**: median of all rounds (more stable than average)
- **Ops/s**: inverse of mean — higher is better
- **Groups**: benchmarks with the same `benchmark.group` are compared together

---

## Measured results (Linux / aarch64 / Python 3.11, N=1M)

Numbers from actual benchmark runs.  Re-run on your machine with `make dc-bench`.

| Use case | ZPyFlow DSL | Best alternative | Speedup |
|----------|-------------|-----------------|---------|
| **filter + take(1K)** — vector search top-K | 40 µs | Python islice: 181 µs | **4.5×** |
| (vs numpy — must scan all N) | 40 µs | numpy: 2,877 µs | **72×** |
| **filter + map + take(10K)** — fused pipeline | 361 µs | Python generator: 1,190 µs | **3.3×** |
| (vs numpy eager) | 361 µs | numpy: 9,804 µs | **27×** |
| **filter(between) + map + take(50K)** — ML preprocess | 2.6 ms | numpy: 4.6 ms | **1.8×** |
| **filter N=1M** — plain filter to list | 16 ms | numpy: 6.5 ms | 0.4× (numpy wins) |
| **filter(between) + map N=1M** — full output | 26 ms | numpy: 19 ms | 0.7× (numpy wins) |
| **count + sum + max N=1M** — ETL multi-stat | 16 ms | Polars: 4 ms | 0.25× (Polars wins) |
| **single .sum() N=1M** — aggregation | 4.2 ms | numpy: 6.5 ms | **1.5×** |
| **fraud review queue take(500) N=1M** | 13 µs | Python islice: 71 µs | **5.5×** |
| (vs numpy — full scan then slice) | 13 µs | numpy: 6,610 µs | **500×** |
| **fraud exposure sum N=1M** | 4.0 ms | numpy: 6.1 ms | **1.5×** |
| **fraud flag count N=1M** | 3.7 ms | numpy: 0.9 ms | 0.25× (numpy wins) |
| **groupby count N=100K** — GroupBy vs Counter | 12 ms | Counter: 10 ms | 0.85× (Counter wins) |
| **groupby agg N=100K** — GroupBy vs defaultdict | 28 ms | defaultdict: 30 ms | **1.05×** (tie) |
| **pagination skip+take page=50 N=100K** | 223 µs | Python islice: 217 µs | **~1×** (tie) |

---

## Business use cases

Where ZPyFlow is the right tool — and where it isn't.
Numbers in parentheses are from actual benchmark runs (Linux / aarch64 / Python 3.11, N=1M).

---

### ✅ Search & Recommendation

**Who**: E-commerce, media platforms, RAG pipelines (LLM + vector DB).

After an ANN index (FAISS, Qdrant, Weaviate) returns similarity scores for 1M
documents, you need the top-K candidates above a relevance threshold.
`take(K)` stops the moment K results are collected — the rest of the list is
never touched.  numpy must scan all N scores first, then slice.

```python
# "Users who bought X also liked..." — top-1000 from 1M candidate scores
top_candidates = (
    Query(similarity_scores)        # 1M cosine sim scores from ANN index
        .filter(col > 0.5)          # relevance threshold
        .take(1_000)                # stop early — don't scan the tail
        .to_list()
)
```

Measured: **40 µs** vs numpy 2,877 µs (**72× faster**) — the gap grows with N.

---

### ✅ ML / AI Feature Preprocessing

**Who**: Data engineers, MLOps teams, model training pipelines.

Before feeding data to a model, a feature column typically needs outlier
clipping, normalization, and mini-batch sampling.  All three steps fuse into
a single Rust pass — no intermediate allocation between clip and normalize.

```python
# Prep one feature column for a training mini-batch
mini_batch = (
    Query(raw_feature_values)           # 1M raw floats from feature store
        .filter(col.between(-CLIP, CLIP))  # remove outliers (FilterBetween)
        .map(col * (1.0 / CLIP))           # normalize to [-1, 1]
        .take(BATCH_SIZE)               # sample without shuffle overhead
        .to_list()
)
```

Measured: **2.6 ms** vs numpy 4.6 ms (**1.8× faster**) when `take()` is used.
Without `take()`, numpy wins on full output — use numpy for that path.

---

### ✅ Fraud Detection & Risk Scoring

**Who**: Fintech, insurance, lending platforms.

A risk model produces a float score for each transaction or applicant.
Downstream logic needs to count flagged cases, retrieve the top-N highest-risk
records for review queues, or compute the total exposure above a threshold —
all on millions of scores per batch.

```python
# Count flagged transactions in a batch
n_flagged = Query(risk_scores).filter(col > FRAUD_THRESHOLD).count()

# Pull top-500 highest-risk cases for the review queue (early stopping)
review_queue = Query(risk_scores).filter(col > FRAUD_THRESHOLD).take(500).to_list()

# Total exposure: sum of transaction amounts where risk > threshold
exposure = Query(amounts).filter(col > HIGH_RISK_AMOUNT).sum()
```

Single aggregation (`.count()`, `.sum()`) stays entirely in Rust — no Python
list created.  For multiple stats in one job, prefer Polars (see ❌ below).

---

### ✅ API Response Processing

**Who**: Backend services consuming third-party REST / streaming APIs.

External APIs (pricing feeds, exchange rates, inventory levels, telemetry
streams) often return large arrays of numeric values.  ZPyFlow fits the
"extract numeric field → validate range → take first N valid" pattern
that appears in almost every API consumer.

```python
import httpx
from zpyflow import Query, col

response = httpx.get("https://api.example.com/prices").json()

# Validate and take the first 100 prices in the acceptable range
valid_prices = (
    Query([item["price"] for item in response["items"]])
        .filter(col.between(0.01, MAX_PRICE))   # reject negatives and spikes
        .take(100)
        .to_list()
)

# Summarize latency figures from a metrics endpoint
p95_approx = Query(response["latency_ms"]).filter(col > 0).max()
```

The pattern is: parse JSON once (Python), extract the numeric column (list
comprehension), then hand off to ZPyFlow for the numeric work.

---

### ✅ Real-time Monitoring & SLO Calculation

**Who**: Platform / SRE teams, observability pipelines.

Periodically, a monitoring job reads a rolling window of request latencies or
error codes and computes SLO breach counts.  The numeric field must be
extracted from log dicts first; after that, ZPyFlow's `.count()` and `.sum()`
complete entirely in Rust.

```python
# How many requests violated the 200ms SLO in the last minute?
latencies   = [r["latency_ms"] for r in recent_requests]   # extract once
breach_count = Query(latencies).filter(col > 200).count()
slo_rate     = 1.0 - breach_count / len(latencies)

# Total bytes sent by 5xx responses
error_bytes = (
    Query([r["bytes_sent"] for r in recent_requests if r["status"] >= 500])
        .sum()
)
```

Measured: **5–10× faster** than Python list comprehension for `.count()` / `.sum()`
on pre-extracted numeric lists.

---

### ✅ Pricing Rules & Catalog Management

**Who**: E-commerce backends, SaaS billing engines.

A product catalog or pricing engine needs to apply margin thresholds, flag
items for repricing, or compute a discount-adjusted price column.
The pipeline is naturally a filter + map on a large float list.

```python
# Find products below minimum margin — flag for repricing
n_underpriced = Query(margin_pcts).filter(col < MIN_MARGIN_PCT).count()

# Apply a 10% promotional discount to mid-tier products
discounted = (
    Query(prices)
        .filter(col.between(20.0, 200.0))   # mid-tier range
        .map(col * 0.90)                     # apply discount
        .to_list()
)
```

---

### ✅ IoT & Financial Tick Processing

**Who**: Industrial IoT platforms, algo-trading systems.

Sensor readings and market ticks arrive in high-frequency batches.
Processing a rolling window means: validate range → calibrate → keep the
most recent N — all in one pass, no intermediate buffer.

```python
# Sliding window: validate, calibrate, keep latest 10K readings
window = (
    Query(sensor_readings)
        .filter(col.between(SENSOR_MIN, SENSOR_MAX))  # reject out-of-range
        .map(col * CALIBRATION_FACTOR)
        .take(10_000)
        .to_list()
)
```

Measured: filter + map + take N=1M in **361 µs** (3.3× faster than Python
generators, 27× faster than numpy eager evaluation).

---

### ✅ A/B Test & Experiment Analytics

**Who**: Growth / product analytics teams.

Each experiment variant produces a large list of numeric outcomes (revenue,
session duration, click probability).  Computing a single metric per variant
is exactly ZPyFlow's sweet spot: filter by variant membership → aggregate.

```python
# Compute mean revenue for users in the treatment group
revenues_treatment = [r for r in events if r["variant"] == "treatment"]
total   = Query([e["revenue"] for e in revenues_treatment]).sum()
n_users = Query([e["revenue"] for e in revenues_treatment]).count()
mean_revenue = total / n_users
```

For a single metric, ZPyFlow's `.sum()` + `.count()` is faster than
`sum(...)` + `len(...)` in Python.  For multiple metrics simultaneously,
use Polars — it computes them all in one columnar pass.

---

### ✅ Game / Simulation — Spatial Filtering

**Who**: Game backends, physics simulations, agent-based models.

Every frame, entities outside interaction range or below an activity threshold
can be pruned.  With 100K+ entities, a fused DSL pass is measurably faster
than a Python loop with no intermediate list created.

```python
# Keep only entities within interaction radius, sorted by proximity
nearby = (
    Query(entity_distances)
        .filter(col < MAX_INTERACTION_RADIUS)
        .take(MAX_NEARBY_ENTITIES)
        .to_list()
)
```

---

### ✅ Fraud Detection & Risk Scoring (`bench_fraud`)

**Who**: Fintech, insurance, lending platforms.

A risk model assigns a float score to each transaction.  Downstream logic
fills a human review queue (take early stopping) and sums total exposure.

```python
# Fill 500-case review queue — stop scanning once full
queue    = Query(risk_scores).filter(col > THRESHOLD).take(500).to_list()
exposure = Query(amounts).filter(col > HIGH_RISK).sum()
```

Measured: review queue **13 µs** vs numpy 6,610 µs (**500× faster** — early stopping).
Exposure sum **4 ms** vs numpy 6 ms (**1.5× faster**).
Flag count: numpy wins (full SIMD scan, no early stopping needed).

---

### ✅ Order Management / Content Analytics — Object Path (`bench_groupby`)

**Who**: E-commerce, media platforms, SaaS products.

Grouping order records by status, content by category, users by segment.
ZPyFlow GroupBy speed matches Python `Counter` / `defaultdict`.
Value is the chainable API (`filter → group_by → agg`) and `skip + take`
pagination without building the full filtered list.

```python
# Order revenue by status — one pipeline, no intermediate list
by_status = (
    Query(orders)
        .filter(lambda o: o["amount"] > 0)
        .group_by(lambda o: o["status"])
        .agg(count=lambda g: g.count(),
             revenue=lambda g: g.map(lambda o: o["amount"]).sum())
)

# Content feed page 10 — filter + skip + take in one pass
page = (
    Query(articles)
        .filter(lambda a: a["status"] == "published")
        .skip(10 * 20).take(20)
        .to_list()
)
```

Measured: GroupBy agg ≈ manual `defaultdict` (tie), pagination ≈ Python islice (tie).
ZPyFlow's advantage here is ergonomics and the unified chainable API.

---

### ❌ When NOT to use ZPyFlow

| Business scenario | Why ZPyFlow is wrong here | Better choice |
|---|---|---|
| Daily revenue report: count + sum + max in one job | Needs 3 separate passes; Polars does all in one columnar scan | **Polars** (~4× faster) |
| Full filter → pass to numpy for arithmetic | numpy wins on pure throughput (~1.3×) when no `take()` involved | **numpy** throughout |
| Datasets under ~10K rows | Python→Rust round-trip overhead exceeds processing savings | **list comprehension** |
| Filtering on string fields or complex object attributes | GIL held per element — same speed as Python | Python generator |
| Multi-column joins or group-by aggregations | ZPyFlow is single-column only | **Polars** or **pandas** |
| One-off scripts, small utilities | No dependency worth it; list comprehension is readable and fast | Plain Python |

See [`docs/when_is_it_fast.md`](../../docs/when_is_it_fast.md) for the full analysis.
