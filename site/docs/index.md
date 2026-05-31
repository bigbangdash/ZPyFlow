# ZPyFlow

Zero-allocation lazy query pipelines for Python, powered by Rust.

- **Lazy & fused** — filter + map + take run in a single pass with no intermediate lists
- **SIMD-accelerated** — float/int arrays execute in Rust with the GIL released
- **Expression DSL** — `col > 5` eliminates Python callbacks entirely
- **Python-friendly** — numpy, pandas, dataclasses, plain lists, and generators all work as input
- **Parallel execution** — `.parallel()` enables Rayon work-stealing

!!! note "ZPyFlow is not a DataFrame engine"
    It lets Python sequence hot paths run as fused Rust pipelines without moving the
    surrounding codebase into a tabular data model.
    See [Performance Guide](performance.md) for when to use ZPyFlow vs Polars.

## Quick start

```bash
pip install maturin
git clone https://github.com/bigbangdash/zpyflow
cd zpyflow
maturin develop --release
```

```python
from zpyflow import Query, col

data = [1.5, -2.3, 0.7, 4.1, -0.5, 3.8]

result = (
    Query(data)
        .filter(col > 0)   # SIMD filter, GIL released
        .map(col * 2.0)    # SIMD map
        .take(4)
        .to_list()
)
# → [3.0, 1.4, 8.2, 7.6]
```

## Navigation

- [API Reference](api.md) — complete public API
- [Performance Guide](performance.md) — when ZPyFlow is fast, when to use Polars
- [Examples](examples/index.md) — runnable examples
- [Benchmark Results](benchmarks.md) — measured performance data
