# ZPyFlow Examples

Each file is self-contained and runnable.  Build the extension first:

```bash
maturin develop --release
pip install numpy pandas          # for API & integration examples
pip install pydantic              # optional, for 04_dataclasses.py
```

---

## API & Integration Examples

Demonstrates ZPyFlow's core API and how it connects to common libraries.

| File | What it shows | Key APIs |
|------|---------------|----------|
| [`01_basic_numeric.py`](01_basic_numeric.py) | Plain list pipelines, DSL vs lambda | `Query`, `col`, `.filter`, `.map`, `.reduce` |
| [`02_numpy_integration.py`](02_numpy_integration.py) | numpy float64/int64 arrays, speed comparison | `from_numpy`, SIMD fast path |
| [`03_pandas_integration.py`](03_pandas_integration.py) | Column preprocessing, row-level filtering | `from_numpy`, `to_dict("records")` |
| [`04_dataclasses.py`](04_dataclasses.py) | Dataclasses, Pydantic, domain objects | `GroupBy`, `.reduce`, `.any`, `.all` |
| [`05_log_processing.py`](05_log_processing.py) | Access logs, error rates, latency stats | `GroupBy`, Counter, rolling windows |
| [`06_etl_pipeline.py`](06_etl_pipeline.py) | Extract → Transform → Load with CSV/JSON | `from_csv`, `from_json_lines`, throughput |
| [`07_ai_embeddings.py`](07_ai_embeddings.py) | Vector search, batch inference, top-p sampling | `from_numpy`, f64 fast path for ML |
| [`08_parallel_and_performance.py`](08_parallel_and_performance.py) | Parallel vs single-thread, memory, profiling | `.parallel()`, timing patterns |

---

## Business Use Case Examples

End-to-end examples for specific industries and business scenarios.
Each file explains the business context and maps it to ZPyFlow operations.

### Numeric pipeline (f64 fast path — SIMD, GIL released)

| File | Business scenario | Industry |
|------|-------------------|----------|
| [`usecase_search_recommendation.py`](usecase_search_recommendation.py) | ANN post-filtering, top-K candidate retrieval | E-commerce, RAG / LLM, Media |
| [`usecase_ml_feature_preprocessing.py`](usecase_ml_feature_preprocessing.py) | Outlier removal + normalization + mini-batch sampling | MLOps, Data engineering |
| [`usecase_fraud_detection.py`](usecase_fraud_detection.py) | Risk score thresholding, review queue, exposure sum | Fintech, Insurance, Lending |
| [`usecase_api_response.py`](usecase_api_response.py) | Validate and aggregate numeric arrays from REST APIs | Any backend service |
| [`usecase_pricing_rules.py`](usecase_pricing_rules.py) | Margin checks, discount application, revenue impact | E-commerce, SaaS billing |
| [`usecase_ab_testing.py`](usecase_ab_testing.py) | Conversion rate, AOV, and RPU per experiment variant | Product analytics, Growth |
| [`usecase_game_simulation.py`](usecase_game_simulation.py) | Nearby entity detection, particle culling, radius sweep | Game backends, Simulation |

### Object pipeline (Python path — ergonomics + `take()` early stopping)

| File | Business scenario | Industry |
|------|-------------------|----------|
| [`usecase_order_processing.py`](usecase_order_processing.py) | Order filtering, status grouping, high-value notifications | E-commerce, Logistics |
| [`usecase_user_segmentation.py`](usecase_user_segmentation.py) | Churn risk detection, upsell targeting, cohort metrics | SaaS, CRM, Marketing |
| [`usecase_content_pipeline.py`](usecase_content_pipeline.py) | Feed generation, editorial queue, category analytics | Media, CMS, Newsletter |
| [`usecase_langchain_langgraph.py`](usecase_langchain_langgraph.py) | RAG retriever, LangGraph node, LangChain tool, batch scoring | LLM apps, AI agents, RAG |

---

## Running all examples

```bash
cd examples

# API & integration
for f in 0*.py; do
    echo "=== $f ===" && python "$f" && echo
done

# Business use cases
for f in usecase_*.py; do
    echo "=== $f ===" && python "$f" && echo
done
```

---

## Which fast path am I on?

```python
q = Query(data)
print(repr(q))
# Query<f64>(skip=0, take=None, parallel=False)   ← SIMD fast path
# Query<i64>(skip=0, take=None, parallel=False)   ← integer fast path
# Query<py>(skip=0,  take=None, parallel=False)   ← Python object path
```

---

## Decision guide

```
Input type          Predicate/transform        Recommended
──────────────────  ─────────────────────────  ───────────────────────────
list[float]         col > x, col * y, etc.     DSL  → f64 fast path (SIMD)
numpy float64       col > x, col * y, etc.     from_numpy() + DSL
list[int]           col > x, col + y, etc.     DSL  → i64 fast path
list[dict/object]   lambda                     Python path (still fused)
pandas column       col DSL                    .tolist() + DSL
pandas rows         lambda                     to_dict("records") + lambda
```

---

## When ZPyFlow beats numpy (benchmarked)

```
Use case                   Operation                          vs numpy
─────────────────────────  ─────────────────────────────────  ──────────
Vector search top-K        filter(col > t).take(K)            ~70×  faster
  (ANN post-filtering)     Early stopping: scans only until K
                           numpy must scan all N first

ML preprocessing+take      filter(col.between()).map().take() ~2×   faster
  (mini-batch sampling)    Single fused pass with early stop

Plain filter               filter(col > t).to_list()          ~1.5× faster
  (N > 100K)               SIMD + GIL released

Multi-stat aggregation     .count() + .sum() + .max()         ~4×   SLOWER
  (ETL, 3 stats at once)   ZPyFlow needs 3 passes;
                           use Polars for multi-stat jobs
```

---

## PyExpr is single-operation

Each `col` expression represents exactly one operation.
Chaining arithmetic does NOT compose — use multiple `.map()` calls:

```python
# WRONG: (col - 10) creates SubScalar(10), then / 90 replaces it with DivScalar(90)
Query(data).map((col - 10) / 90)          # silently computes col / 90 only

# CORRECT: two chained map calls
Query(data).map(col - 10).map(col / 90)   # shift then scale
```
