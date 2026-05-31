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

## Key results (Linux, CPython 3.11, aarch64)

Results from the most recent CI run in
`sandbox/benchmark/results/Linux-CPython-3.11-64bit/`.

### Filter + take (1M float64)

| Approach | Mean time |
|---|---|
| Python list comprehension | ~80ms |
| numpy `arr[arr > 0]` | ~8ms |
| ZPyFlow Expression DSL | ~2–5ms |
| ZPyFlow DSL + parallel | ~0.5–1ms |

### Vector search top-K (early-stop)

`filter(col > threshold).take(K)` where K=100, N=1M, threshold=0.85:

| Approach | Mean time | Note |
|---|---|---|
| ZPyFlow DSL | ~70× faster than numpy | Early-stop after K results |
| numpy `arr[arr > t][:K]` | slower | Scans all N first |

### ETL multi-stat (count + sum + max)

| Approach | Mean time | Note |
|---|---|---|
| Polars | fastest | 1 columnar pass |
| ZPyFlow | ~4× slower | 3 separate passes |
| Python loop | slowest | — |

!!! tip "When ZPyFlow loses"
    For multi-stat aggregations (count + sum + max in one report), Polars is
    significantly faster.  Use ZPyFlow for single-stat or early-stop workloads.
