# Object Field DSL

`field("name")` creates expressions that run GIL-free after the first filter converts dicts to `RustObj`.

```python
from zpyflow import Query, field

logs = [
    {"status": 500, "latency_ms": 312, "path": "/api/users"},
    {"status": 200, "latency_ms":  45, "path": "/api/items"},
    {"status": 500, "latency_ms": 520, "path": "/health"},
]

# DSL filter — dict→RustObj conversion on first filter, then GIL-free
count = (
    Query(logs)
        .filter(field("status") >= 500)
        .filter(field("latency_ms") > 100)
        .count()
)

# Multiple queries on the same dataset — preload() pays conversion once
q = Query(logs).preload()
slow_errors = q.filter(field("status") >= 500).filter(field("latency_ms") > 100).count()
all_errors  = q.filter(field("status") >= 500).count()
```

## Supported field() operators

| Expression | Description |
|---|---|
| `field("x") > n` | Greater than |
| `field("x") >= n` | Greater than or equal |
| `field("x") < n` | Less than |
| `field("x") <= n` | Less than or equal |
| `field("x") == v` | Equals (numeric or string) |
| `field("x") != v` | Not equals |
| `field("x").between(a, b)` | Inclusive range |

!!! note "When to use field() vs lambda"
    Use `field()` when filtering dict records by a **known numeric field name** — it
    avoids calling Python per element after the first filter.  For complex logic
    (multiple fields, string operations), use a lambda.
