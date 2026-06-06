# Performance Guide

## When ZPyFlow is fast

ZPyFlow is strongest when the pipeline can stop early, avoid Python callbacks,
or stay on a typed buffer path. It is not a general replacement for NumPy or
Polars full-column kernels.

| Use case | Why it works well |
|---|---|
| `filter(col > t).take(K)` with small K | Early-stop: scans only until K results are found |
| Embedding threshold + top-K | Avoids NumPy's full boolean-index scan |
| `from_numpy(...).filter(...).to_bytes()` | Typed input and typed output avoid Python list boxing |
| Boolean flag count from NumPy | Compact `bool` / `uint8` path is faster than Python and often faster than NumPy |
| Null-free Arrow numeric arrays | Buffer path avoids `to_pylist()` conversion |
| `field()` filters on large dict records | Can beat plain Python when the field DSL covers the predicate |

## When ZPyFlow loses

| Use case | Better tool |
|---|---|
| Full-array `filter`, `map`, `sum`, or `count` on NumPy data | NumPy |
| Multi-stat ETL reports such as count + sum + max | Polars |
| Dense vectorized math | NumPy |
| Multi-column joins, windows, SQL-style analytics | Polars |
| Small data where readability dominates | Plain Python |

## Recent benchmark shape

These are representative results from the benchmark suite:

| Case | Result |
|---|---|
| `filter + map + take`, N=1M | ZPyFlow DSL around 0.25ms; NumPy around 4.3ms |
| Vector search top-K, N=1M | ZPyFlow DSL around 0.08ms; NumPy around 1.8ms |
| `filter → bytes`, N=1M | `from_numpy(...).to_bytes()` around 3.3ms; NumPy boolean index around 4.1ms |
| ETL 3-stat, N=1M | Polars around 0.9ms; ZPyFlow single-pass around 9.3ms |
| Full `filter + sum`, N=1M | NumPy/Polars beat ZPyFlow |
| Dict field filter/count, N=1M | `field()` DSL can beat Python list comprehensions; lambdas usually do not |

Raw benchmark JSON files are in
[`sandbox/benchmark/results/`](../../sandbox/benchmark/results/).

## Execution path guide

Use `Query.explain()` to see which path a query takes:

```python
q = Query(data).filter(col > 0).map(col * 2).take(1000)
print(q.explain())
# Query.explain()
#   kind:     f64
#   ops:      FilterGt(0.0) -> MapMulScalar(2.0)
#   skip:     0
#   take:     1000
#   parallel: false
#   gil_free: true
#   alloc:    1 Vec at terminal
```

## Parallel execution

`.parallel()` applies to numeric fast paths only.

```python
result = Query(large).filter(col > 0).map(col * 2).parallel().to_list()
```

!!! warning "Threading overhead"
    Parallel mode adds split + join overhead. It is slower than single-threaded
    for smaller inputs. Profile before enabling.
