# ZPyFlow

> **Alpha - v0.1.x is an early release for testing and feedback.**

Lazy query pipelines for Python, powered by Rust.

ZPyFlow is for Python data that is already in memory: lists, NumPy arrays,
generators, dict records, dataclasses, and similar objects. It runs lazy
pipelines without building intermediate Python lists, and numeric DSL paths
execute in a SIMD-accelerated Rust kernel without constructing Python objects per element.

- **Lazy and fused**: chained operations run in one pass.
- **Numeric fast paths**: float/int/bool arrays execute in Rust.
- **Expression DSL**: `col > 5` and `col * 2` avoid Python callbacks.
- **Object-friendly**: lambdas still work for dicts, dataclasses, and models.
- **Parallel option**: `.parallel()` uses Rayon for large numeric workloads.

ZPyFlow is not trying to replace Polars or pandas. Use those when your data is
already table-shaped or you need joins, windows, or multi-column analytics. Use
ZPyFlow when you just want a fast pipeline inside ordinary Python code.

## Installation

```bash
pip install zpyflow
```

For source builds and contributor setup, see [docs/contributing.md](docs/contributing.md).

## Quick Start

```python
from zpyflow import Query, col

data = [1.5, -2.3, 0.7, 4.1, -0.5, 3.8, -1.1, 2.2]

result = (
    Query(data)
        .filter(col > 0)
        .map(col * 2.0)
        .take(4)
        .to_list()
)

assert result == [3.0, 1.4, 8.2, 7.6]
```

Aggregations stay inside the Rust numeric path:

```python
positive = Query(data).filter(col > 0)

assert positive.count() == 5
assert positive.sum() == 12.3
assert Query(data).max() == 4.1
```

## Expression DSL vs Lambdas

Use the expression DSL for numeric hot paths:

```python
from zpyflow import Query, col

scores = [0.2, 0.9, 0.4, 0.95, 0.7]

top = (
    Query(scores)
        .filter(col >= 0.7)
        .map(col * 100)
        .to_list()
)

assert top == [90.0, 95.0, 70.0]
```

Use lambdas when you need arbitrary Python logic:

```python
records = [
    {"name": "alice", "score": 91},
    {"name": "bob", "score": 64},
    {"name": "carol", "score": 88},
]

names = (
    Query(records)
        .filter(lambda r: r["score"] >= 80)
        .map(lambda r: r["name"])
        .to_list()
)

assert names == ["alice", "carol"]
```

Lambdas are flexible, but they run as Python callbacks. For speed, prefer `col`
or `field()` expressions where they fit.

## NumPy

```python
import numpy as np
from zpyflow import from_numpy, col

arr = np.random.default_rng(42).standard_normal(1_000_000)

count = (
    from_numpy(arr)
        .filter(col > 0)
        .take(10_000)
        .count()
)
```

`from_numpy()` supports common 1-D numeric dtypes, including `float64`,
`float32`, `int64`, `bool`, and `uint8`. See the full
[NumPy integration guide](site/docs/examples/numpy.md).

## Dict Records and Field DSL

For dict-like records, `field()` covers common filters and aggregations without
writing a Python callback for every element:

```python
from zpyflow import Query, field

logs = [
    {"path": "/api", "status": 200, "latency_ms": 42.0},
    {"path": "/api", "status": 500, "latency_ms": 310.0},
    {"path": "/health", "status": 200, "latency_ms": 8.0},
]

slow = (
    Query(logs)
        .filter(field("latency_ms") > 100)
        .to_list()
)

assert slow == [{"path": "/api", "status": 500, "latency_ms": 310.0}]
```

See [Object Field DSL](site/docs/examples/field_dsl.md) and
[Adapters](site/docs/examples/adapters.md) for JSON Lines, CSV, and Arrow input.

## Grouping

```python
from zpyflow import Query, agg_count, agg_sum, field

orders = [
    {"user": "alice", "amount": 120.0},
    {"user": "bob", "amount": 45.0},
    {"user": "alice", "amount": 80.0},
]

summary = Query(orders).group_agg(
    field("user"),
    count=agg_count(),
    total=agg_sum(field("amount")),
)

assert summary == [
    {"_key": "alice", "count": 2, "total": 200.0},
    {"_key": "bob", "count": 1, "total": 45.0},
]
```

See [GroupBy & group_agg](site/docs/examples/groupby.md) for more grouping
examples.

## When To Use It

Good fits:

- Data is already in Python sequences or NumPy arrays.
- Early stopping matters, for example `filter(col > threshold).take(k)`.
- You want a lazy pipeline API around ordinary Python data.
- You can use `col` or `field()` instead of a Python callback.
- You want Arrow or NumPy inputs to stay on a typed path.
- You have a sync web endpoint (FastAPI, Flask) filtering or aggregating a large
  numeric sequence — ZPyFlow's lower per-request CPU time translates directly to
  higher RPS under concurrent load.

Use another tool when:

- You need joins, window functions, or SQL-style analytics.
- The work spans many columns in a table.
- The main operation is dense vectorized math that NumPy already expresses well.
- You need full-array `filter`, `map`, `sum`, or `count` and NumPy already has
  the data.
- Your data is small enough that readability matters more than execution path.

More detail: [Performance Guide](site/docs/performance.md),
[Benchmark Results](site/docs/benchmarks.md), and
[When Is It Actually Fast?](docs/when_is_it_fast.md).

## API Overview

Common imports:

```python
from zpyflow import (
    Query, col, field,
    from_numpy, from_arrow, from_csv, from_json_lines, from_generator,
    agg_count, agg_sum, agg_mean, agg_max, agg_min,
)
```

Core pipeline methods:

```python
q.filter(pred).map(func).skip(n).take(n).parallel()
q.to_list()
q.count()
q.sum()
q.min()
q.max()
q.first()
q.last()
q.stats()
```

See [site/docs/api.md](site/docs/api.md) for the complete API reference.

## Documentation

- [API Reference](site/docs/api.md)
- [Examples](site/docs/examples/index.md)
- [Performance Guide](site/docs/performance.md)
- [Benchmark Results](site/docs/benchmarks.md)
- [When Is It Actually Fast?](docs/when_is_it_fast.md)
- [Design Notes](site/docs/design.md)
- [Architecture](ARCHITECTURE.md)
- [Contributing](docs/contributing.md)

## Examples

Runnable examples are in [examples/](examples/):

- `01_basic_numeric.py`
- `02_numpy_integration.py`
- `03_pandas_integration.py`
- `04_dataclasses.py`
- `05_log_processing.py`
- `06_etl_pipeline.py`
- `07_ai_embeddings.py`
- `08_parallel_and_performance.py`

See [examples/README.md](examples/README.md) for the full list.

## License

MIT. See [LICENSE](LICENSE).
