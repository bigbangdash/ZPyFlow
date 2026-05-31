# Numeric DSL

The `col` proxy creates expressions that compile to Rust operations — no GIL, SIMD where applicable.

```python
from zpyflow import Query, col

data = [1.5, -2.3, 0.7, 4.1, -0.5, 3.8, -1.1, 2.2]

# Filter → map → take: single fused Rust pass
result = (
    Query(data)
        .filter(col > 0)
        .map(col * 2.0)
        .take(4)
        .to_list()
)
# → [3.0, 1.4, 8.2, 7.6]

# Aggregations
total = Query(data).filter(col > 0).sum()    # 12.3
count = Query(data).filter(col > 0).count()  # 5
vmax  = Query(data).max()                    # 4.1
vmin  = Query(data).filter(col > 0).min()    # 0.7
```

## Supported DSL operators

| Expression | Internal op | SIMD |
|---|---|---|
| `col > x` | FilterGt | ✅ |
| `col >= x` | FilterGe | ✅ |
| `col < x` | FilterLt | ✅ |
| `col <= x` | FilterLe | ✅ |
| `col.between(a, b)` | FilterBetween | ✅ |
| `col * x` | MapMulScalar | ✅ |
| `col + x` | MapAddScalar | ✅ |
| `col - x` | MapSubScalar | ✅ |
| `col / x` | MapDivScalar | ✅ |
| `-col` | MapNeg | ✅ |
| `col ** x` | MapPowScalar | — |
| `col.abs()` | MapAbs | — |
| `col.sqrt()` | MapSqrt | — |

!!! tip "Chaining arithmetic"
    Each `col` expression is **one operation**.  For multi-step transforms use
    multiple `.map()` calls:
    ```python
    # CORRECT
    Query(data).map(col - 10).map(col / 90)
    ```
