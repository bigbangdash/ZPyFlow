# Adapters

## from_numpy

```python
import numpy as np
from zpyflow import from_numpy, col

arr = np.arange(1_000_000, dtype=np.float64)
result = from_numpy(arr).filter(col > 500_000).to_list()
```

## from_arrow

```python
import pyarrow as pa
from zpyflow import from_arrow, col

arr = pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64())
result = from_arrow(arr).filter(col > 2.0).to_list()  # → [3.0, 4.0]

# Float64 with nulls — nulls become NaN
arr_null = pa.array([1.0, None, 3.0], type=pa.float64())
result = from_arrow(arr_null).filter(col == col).to_list()  # → [1.0, 3.0]
```

## from_csv

```python
from zpyflow import from_csv, col

# Single numeric column — f64 fast path
prices = from_csv("products.csv", column="price", dtype="float")
discounted = prices.filter(col > 10.0).map(col * 0.9).to_list()

# All rows as dicts
products = from_csv("products.csv")
premium = (
    products
        .filter(lambda p: float(p["price"]) > 100)
        .take(50)
        .to_list()
)
```

## from_json_lines

```python
from zpyflow import from_json_lines, col

# Single numeric field
latencies = from_json_lines("access.log.ndjson", field="latency_ms", dtype="float")
p99 = latencies.filter(col > 200).count()

# All rows as dicts
events = from_json_lines("events.ndjson")
errors = events.filter(lambda e: e.get("level") == "error").take(100).to_list()
```

## from_generator

```python
from zpyflow import from_generator

def generate_scores():
    for i in range(1_000_000):
        yield float(i) / 1_000_000

q = from_generator(generate_scores())   # eagerly materialises
```
