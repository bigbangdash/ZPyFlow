# Performance Guide

## When ZPyFlow is fast

ZPyFlow wins when the pipeline is **single-stat, small output, or early-stop**:

| Use case | Why ZPyFlow wins |
|---|---|
| `filter(col > t).count()` on 1M floats | SIMD, GIL released, 1 pass |
| `filter(col > t).take(K)` with small K | Early-stop: scans only until K results found |
| `filter + map + sum` in one pass | Fused, 1 allocation |
| Embedding threshold + top-K | Fast early-stop vs NumPy full-scan |

## When to use Polars instead

| Use case | Polars wins because |
|---|---|
| 3+ stats in one pass (count + sum + max) | 1 columnar pass vs 3 ZPyFlow passes |
| Multi-column join | Not supported in ZPyFlow |
| Loading CSV → table analysis | Natural fit for Polars |
| SQL-style GROUP BY with multiple columns | Polars native |

## Benchmark results (1M float64)

| Approach | Time | Allocations | GIL |
|---|---|---|---|
| Python list comprehension | ~80ms | 2 lists | held |
| numpy (`arr[arr > 0] * 2`) | ~8ms | 2 arrays | released |
| ZPyFlow Expression DSL (SIMD) | ~2–5ms | 1 list | released |
| ZPyFlow DSL + parallel | ~0.5–1ms | 1 list | released |

Raw benchmark JSON files are in
[`sandbox/benchmark/results/`](../../sandbox/benchmark/results/).

## Execution path guide

Use `Query.explain()` to see which path a query takes:

```python
q = Query(data).filter(col > 0).map(col * 2).take(1000)
print(q.explain())
# Query.explain()
#   kind:     f64
#   ops:      FilterGt(0.0) → MapMulScalar(2.0)
#   skip:     0
#   take:     1000
#   parallel: false
#   gil_free: true
#   alloc:    1 Vec at terminal
```

## Parallel execution

`.parallel()` applies to **numeric fast paths only** (f64 / i64).

```python
# Multi-threaded: Rayon work-stealing, GIL fully released
result = Query(large).filter(col > 0).map(col * 2).parallel().to_list()
```

!!! warning "Threading overhead"
    Parallel mode adds split + join overhead.  It is slower than single-threaded
    for inputs under ~500K elements.  Profile before enabling.
