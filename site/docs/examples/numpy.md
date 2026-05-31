# NumPy Integration

`from_numpy()` reads the array via the buffer protocol — one bulk memcpy, no per-element boxing.

```python
import numpy as np
from zpyflow import from_numpy, col

arr = np.random.standard_normal(1_000_000)

# SIMD filter + map, GIL released, 1 allocation
result = (
    from_numpy(arr)
        .filter(col > 0)
        .map(col ** 2)
        .take(10_000)
        .to_list()
)

# Aggregations
positive = from_numpy(arr).filter(col > 0)
mean_pos = positive.sum() / positive.count()
```

## Supported dtypes

| NumPy dtype | ZPyFlow path |
|---|---|
| `float64` | f64 fast path (zero-copy numpy view) |
| `int64` | i64 fast path (zero-copy numpy view) |
| `bool` | u8 compact (compact storage, no boxing) |
| `uint8` | u8 compact |
| `float32`, `float16` | cast to float64 first |
| `int32`, `int16`, `int8`, `uint16`, `uint32` | cast to int64 first |
| `uint64` | Python fallback (`tolist()`) |
| Other | Python fallback (`list(arr)`) |
