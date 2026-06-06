# ZPyFlow — Architecture

## Source Layout

```
src/
├── lib.rs                  # Crate root: PyO3 module registration
├── core/                   # Pure Rust domain (no Python dependency)
│   ├── mod.rs              # Re-exports + cross-module callable helpers
│   ├── numeric/
│   │   ├── mod.rs
│   │   ├── pipeline.rs     # NumericPipeline (f64/i64/u8), NumericOp, IntOp
│   │   └── simd.rs         # SIMD kernels (wide crate: f64x4, f32x8, i64x4)
│   ├── obj/
│   │   ├── mod.rs
│   │   └── pipeline.rs     # ObjOp, RustRow, RustValue, GIL-free obj execution
│   ├── parallel.rs          # Rayon bench utilities (production path is in numeric/pipeline.rs)
│   ├── sources.rs           # ZStream source types (SliceStream, VecStream, …)
│   └── traits.rs            # ZStream trait + combinators (Map, Filter, Take, …)
├── io/                     # I/O adapters (CSV, JSON Lines)
│   ├── mod.rs              # ParsedOutput enum, shared types
│   ├── csv.rs              # CSV parse (GIL-free for path inputs)
│   └── jsonl.rs            # JSON Lines parse (GIL-free for path inputs)
└── python/                 # PyO3 adapters — Python ↔ Rust boundary
    ├── mod.rs              # Re-exports of the public Python API
    ├── query/              # PyQuery — the main Query class
    │   ├── mod.rs          # PyQuery struct, QueryInner enum, shared helpers
    │   ├── construct.rs    # #[pymethods]: new(), f64(), i64(), _from_csv_*(), …
    │   ├── filter.rs       # #[pymethods]: filter(), take_while(), skip_while()
    │   ├── map_ops.rs      # #[pymethods]: map(), map_field(), flat_map(), preload()
    │   ├── transform.rs    # #[pymethods]: take(), skip(), parallel(), chain(), zip(), enumerate()
    │   └── terminal.rs     # #[pymethods]: to_list(), count(), sum(), mean(), stats(), …
    ├── agg.rs              # PyAggSpec, group_agg kernel
    ├── conversion.rs       # Python dict ↔ RustRow conversion
    ├── expr.rs             # PyExpr, PyColProxy, PyFieldExpr (numeric/field DSL)
    ├── fastpath.rs         # GIL-aware fast paths: LazyFloatList, NumpyF64/F32, ObjField
    └── io_bridge.rs        # Python-level I/O wrappers calling io/csv.rs + io/jsonl.rs
```

## Dispatch Strategy

When `Query(data)` is called, the data type determines the execution path:

| Input type              | QueryInner variant | GIL behaviour |
|-------------------------|-------------------|---------------|
| `list[float]`           | `LazyFloatList`   | GIL held (C API ob_fval reads) |
| `numpy float64`         | `NumpyF64`        | GIL released (buffer protocol) |
| `numpy float32`         | `NumpyF32`        | GIL released |
| `list[int]` / numpy int | `I64`             | GIL released |
| `list[bool/uint8]`      | `U8`              | GIL released |
| `list[dict]` (first DSL filter) | `ObjField` / `RustObj` | GIL released after conversion |
| everything else         | `Py` / `Obj`      | GIL held |

## Key Design Decisions

- **Lazy, fused pipeline**: operations accumulate into a `Vec<Op>`; a single pass runs at the terminal call. Analogous to ZLinq's struct chain.
- **Arc sharing**: `Arc<Vec<f64>>` lets `.filter()` / `.map()` clone the pipeline state without copying data.
- **multiple-pymethods**: PyO3's `multiple-pymethods` feature splits the `PyQuery` impl across 5 sub-modules while keeping a single Python class.
- **Cross-module helpers**: `collect_to_pylist` and `take_query` in `query/mod.rs` provide Rust-callable shims for methods that span `#[pymethods]` blocks.
- **abi3 wheel**: `pyo3/abi3-py310` + `Py_LIMITED_API` enables a single wheel for Python 3.10+.
