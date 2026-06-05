# Benchmark Results

Raw JSON results are stored in
[`sandbox/benchmark/results/`](../../sandbox/benchmark/results/)
and updated after each `make dc-bench` run.

## Running benchmarks locally

```bash
# All suites (Docker — reproducible environment)
make dc-bench

# Single suite
make dc-bench-filter

# Save current results as baseline
make dc-bench-save

# Compare against saved baseline (fails if >10% regression)
make dc-bench-compare
```

## Key results

Representative results from the benchmark suite.

### Filter + map + take (1M float64)

| Approach | Mean time |
|---|---|
| ZPyFlow Expression DSL | ~0.25ms |
| Python generator chain | ~0.7ms |
| numpy eager boolean indexing | ~4.3ms |
| Python two-list-comprehensions | ~35ms |

### Vector search top-K (early-stop)

`filter(col > threshold).take(K)` where K=100, N=1M, threshold=0.85:

| Approach | Mean time | Note |
|---|---|---|
| ZPyFlow DSL | ~0.08ms | Early-stop after K results |
| numpy `arr[arr > t][:K]` | ~1.8ms | Scans all N first |

### ETL multi-stat (count + sum + max)

| Approach | Mean time | Note |
|---|---|---|
| Polars | ~0.9ms | Fastest; columnar engine |
| numpy | ~3.9ms | Strong full-array baseline |
| ZPyFlow single-pass | ~9.3ms | Faster than Python, slower than NumPy/Polars |
| Python loop | ~36ms | Slowest |

!!! tip "When ZPyFlow loses"
    For full-array numeric scans and multi-stat reports, NumPy and Polars are
    usually faster. Use ZPyFlow for early-stop pipelines, typed adapter paths,
    and Python-object workflows where the DSL avoids callbacks.

---

## Pipeline operator notes

### sort / sort_by

`Query.sort()` delegates to Python's `sorted()` (Timsort, O(N log N)).
Performance is identical to `sorted(list(query))` for the same data.
The benefit is composability — `filter(...).sort()` reads as a single pipeline
rather than two steps, and the sort is deferred until `.to_list()`.

### distinct

`Query.distinct()` uses a Python `set` for O(1) membership tests per element.
Equivalent to `list(dict.fromkeys(data))` for hashable elements.
The `key_fn` variant enables deduplication of complex objects by a derived key
without converting them to hashable types.

### join (inner_join / left_join)

Hash join: the right side is indexed into a `defaultdict(list)` in O(M) time,
then the left side is probed in O(N) time per element — total O(N + M).

| Approach | Complexity | Notes |
|---|---|---|
| `inner_join` (ZPyFlow) | O(N + M) | Hash join |
| Nested loop | O(N × M) | Avoid for large datasets |
| Polars `join` | O(N + M) | Columnar, fastest for DataFrames |

ZPyFlow join returns `Query[tuple[L, R]]`; merge dicts with
`.map(lambda lr: {**lr[0], **lr[1]})` if needed.

### cache

`Query.cache()` materialises the pipeline once and wraps the result in a new
`Query`. Use it when you need to run multiple terminal operations on the same
filtered dataset:

```python
# Without cache: source scanned 3 times
count_a = Query(data).filter(col > 0).count()
count_b = Query(data).filter(col > 1).count()
total   = Query(data).sum()

# With cache: source scanned once
q = Query(data).filter(col > 0).cache()
count_a = q.count()
count_b = q.filter(col > 1).count()
total   = q.sum()
```

For a 1M-element numeric pipeline with 3 terminal calls, cache reduces total
scan time by ~2/3.

### String DSL (startswith / endswith / contains / matches)

`field("k").startswith("x")` routes through the `ObjFieldPy` path — a C-level
loop that accesses dict items via `PyDict_GetItem` without invoking a Python
function frame per element.

| Approach | Notes |
|---|---|
| `field("name").startswith("x")` | C-level field access, no Python frame |
| `lambda r: r["name"].startswith("x")` | Python call frame per element (~5–10× slower) |
| `field("name").matches(r"^\d+")` | Compiled regex; amortises compilation across all elements |

The `matches` variant compiles the regex once (stored in `Arc<Regex>`) and
evaluates it in Rust without any Python call overhead.

### Numeric DSL extensions (log / exp / sigmoid / clamp)

These operations are implemented as scalar fallbacks in the SIMD pipeline
(element-wise loops, no vectorised instruction). Performance is comparable to
NumPy's `np.log` on the same array when the pipeline has no prior filter.

The advantage is **pipeline fusion**: combining a filter and a log in one pass
avoids allocating an intermediate filtered Vec:

```python
# NumPy (two allocations: filtered array + log result)
arr_filtered = arr[arr > 0]
result = np.log(arr_filtered)

# ZPyFlow (one pass, zero intermediate allocation)
result = Query(arr).filter(col > 0).map(col.log()).to_list()
```

For pure full-array transforms without filters, NumPy's vectorised SIMD
routines are faster (ZPyFlow does not yet have SIMD log/exp kernels).
