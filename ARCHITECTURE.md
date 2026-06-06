# ZPyFlow Architecture

**Zero-allocation lazy query pipelines for Python, powered by Rust.**

---

## 1. The Problem We Are Solving

Python's data pipeline libraries each solve part of the problem but none covers all of it:

| Library     | Strength                        | Weakness                                      |
|-------------|----------------------------------|-----------------------------------------------|
| `itertools` | Lazy, composable, stdlib         | Pure Python — GIL per element, no SIMD        |
| `pandas`    | Feature-rich, familiar API       | Eager, materializes intermediates, high RAM    |
| `polars`    | Fast, lazy, columnar             | Arrow-centric, poor for arbitrary Python types |
| `numpy`     | SIMD-accelerated arithmetic      | Not lazy, chaining creates temporaries         |
| generators  | Truly lazy, zero memory          | No parallelism, no type specialization         |

ZPyFlow's goal: **itertools ergonomics + numpy speed + polars memory discipline**, working with any Python objects.

---

## 2. ZLinq Philosophy, Translated to Python

ZLinq (C# / .NET) achieves zero-allocation LINQ by exploiting the CLR's generic specialization — every chain `Where().Select().Take()` is compiled to a unique generic type at JIT time, producing a single inlined loop.

In Python/Rust, we cannot do this end-to-end (Python is dynamically typed), but we can do it *within the Rust core* and minimize the number of Python ↔ Rust crossings.

### What we borrow from ZLinq

| ZLinq concept          | ZPyFlow equivalent                                                     |
|------------------------|------------------------------------------------------------------------|
| Generic value type chains | `ZStream` trait with monomorphic operator types (`Map<Filter<SliceStream>>`) |
| No heap allocation per op | Rust's type system composes operators into a single stack-allocated state machine |
| Iterator fusion        | LLVM inlines the entire chain into one loop at `-O3`                   |
| Specialization for primitives | `NumericPipeline` (f64/i64 paths) bypass Python entirely          |
| Span/Memory\<T\>       | `SliceStream<'a, T>` borrows data without copying                      |

### What we can't borrow

- **Full end-to-end fusion**: when a Python lambda is involved, the loop must call into Python per element. The GIL prevents us from hiding this cost.
- **Value type stack allocation in Python**: Python objects are always heap-allocated. We can minimize *new* allocations but not eliminate existing Python object costs.

---

## 3. Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Python user code                                           │
│  from zpyflow import Query, col                             │
│  Query(data).filter(col > 5).map(col * 2).to_list()        │
└────────────────────────┬────────────────────────────────────┘
                         │ Python method calls
┌────────────────────────▼────────────────────────────────────┐
│  zpyflow Python layer  (zpyflow/__init__.py, adapters.py)   │
│  - Data source normalization (numpy, arrow, generators)     │
│  - Lambda AST → DSL conversion (_lambda_parser.py)         │
│  - GroupBy (pure Python, hash-map based)                    │
│  - Type stubs (__init__.pyi) for IDE support                │
└────────────────────────┬────────────────────────────────────┘
                         │ PyO3 extension call
┌────────────────────────▼────────────────────────────────────┐
│  PyO3 binding layer    (src/python/query/)                  │
│  - PyQuery class — Python-visible, stores QueryInner        │
│  - Dispatch: F64 / I64 / U8 / NumpyF64 / NumpyF32 / Obj …  │
│  - PyExpr / FieldExpr / ColProxy — expression DSL           │
│  - GIL release for numeric and numpy paths                  │
└────────────┬───────────────────────────────┬────────────────┘
             │ NumericPipeline/IntPipeline   │ ZStream
┌────────────▼────────────┐  ┌──────────────▼───────────────┐
│  Numeric fast path      │  │  Generic ZStream pipeline     │
│  (src/core/numeric/)    │  │  (src/core/traits.rs)         │
│  - Op queue (no alloc)  │  │  - Map, Filter, Take, etc.    │
│  - Single-pass fusion   │  │  - SliceStream, VecStream     │
│  - SIMD dispatch        │  │  - Zero dynamic dispatch      │
└────────────┬────────────┘  └──────────────────────────────┘
             │
┌────────────▼────────────┐  ┌──────────────────────────────┐
│  SIMD layer             │  │  Parallel layer               │
│  (src/core/numeric/     │  │  (src/core/parallel.rs)       │
│   simd.rs)              │  │  - Rayon work-stealing        │
│  - f64x4 / f32x8        │  │  - par_execute_f64/i64        │
│  - Filter with masking  │  │  - Group-by parallelism       │
│  - In-place map ops     │  │  - GIL fully released         │
│  - Dot product, sum     │  │                               │
└─────────────────────────┘  └──────────────────────────────┘
```

---

## 4. Rust Core Design

### 4.1 The ZStream Trait

```rust
pub trait ZStream: Sized {
    type Item;
    fn next_item(&mut self) -> Option<Self::Item>;
    fn size_hint(&self) -> (usize, Option<usize>);

    // All combinators return CONCRETE types — no Box, no dyn
    fn zmap<B, F: FnMut(Self::Item) -> B>(self, f: F) -> Map<Self, F>;
    fn zfilter<F: FnMut(&Self::Item) -> bool>(self, pred: F) -> Filter<Self, F>;
    fn ztake(self, n: usize) -> Take<Self>;
    // ... etc
}
```

**Key invariant**: every method returns a concrete generic type. The chain:

```rust
SliceStream::new(&data)
    .zfilter(|x| *x > 0.0)
    .zmap(|x| x * 2.0)
    .ztake(1000)
```

...compiles to the type `Take<Map<Filter<SliceStream<f64>>, ...>, ...>`. LLVM can see the entire chain at compile time and inlines it into a single state machine loop — identical to hand-written code.

**Size comparison**: the entire pipeline above sits on the stack in ~3 words of state (pos, remaining, function pointers). Compare this to `Box<dyn Iterator>` which adds a vtable pointer and heap allocation per combinator.

### 4.2 The Type Erasure Boundary

The `ZStream` type chain cannot be stored in a Python object (PyO3 requires concrete known types). At the Python boundary we erase the type:

```rust
// Python-visible
#[pyclass]
pub struct PyQuery {
    inner: QueryInner,  // concrete enum, no Box<dyn> in hot paths
}

enum QueryInner {
    F64(NumericPipeline),          // typed, GIL-free
    I64(IntPipeline),              // typed, GIL-free
    U8(U8Pipeline),                // bool/u8, GIL-free
    Py(Vec<PyObject>),             // generic Python objects, GIL required
    Obj(Box<dyn Iterator<...>>),   // Python iterator fallback
    RustObj(Arc<Vec<RustValue>>),  // owned Rust values for ObjField path
    LazyFloatList(ChunkedF64),     // chunked f64 for partial computation
    NumpyF64(Arc<[f64]>),          // borrowed via buffer protocol (zero-copy)
    NumpyF32(Arc<[f32]>),          // borrowed via buffer protocol (zero-copy)
    ObjField { ... },              // field()-DSL over dict/dataclass objects
    Empty,
}
```

The cost: each `.filter()` / `.map()` call from Python must re-enter Rust and update `QueryInner`. This is acceptable because operations are deferred and executed once at the terminal call.

### 4.3 NumericPipeline — the Zero-GIL Path

For f64 / i64 data with expression DSL predicates, the entire pipeline runs in Rust with the GIL released:

```rust
// User writes (Python):
Query(data).filter(col > 5.0).map(col * 2.0).to_list()

// Rust sees:
NumericPipeline::new(data)
    .push_op(NumericOp::FilterGt(5.0))
    .push_op(NumericOp::MapMulScalar(2.0))
    .execute()  // called inside py.allow_threads(|| ...)
```

`execute()` performs a **single-pass fused scan** — filter and map are applied to each element in the same loop iteration, no intermediate `Vec` is created.

### 4.4 SIMD Strategy

We use the `wide` crate for stable (non-nightly) SIMD with `f64x4` (256-bit AVX2):

**Map operations** (`MapMulScalar`, `MapAddScalar`, etc.): applied in-place with 4-wide SIMD. These are pure throughput operations — no branching, maximum lane utilization.

**Filter operations** (`FilterGt`, etc.): SIMD comparison produces a 4-bit mask. We use that mask to selectively copy elements. The write rate is variable (depends on selectivity), but the *read rate* is always 4 elements per cycle.

Optimal selectivity for SIMD filter: **50%**. At 10% or 90% selectivity, the branch predictor handles scalar code equally well. The SIMD advantage is greatest at 40-60% selectivity where scalar code branch-mispredicts heavily.

**Dot product**: uses fused multiply-add with `f64x4`, critical for embedding similarity in AI pipelines.

### 4.5 Iterator Fusion — why it matters

Consider:

```python
# Without fusion (creates 2 intermediate lists):
filtered = [x for x in data if x > 0]    # alloc 1
mapped   = [x * 2 for x in filtered]      # alloc 2

# With fusion (one pass, one allocation):
result = Query(data).filter(col > 0).map(col * 2).to_list()
```

Memory saved for N=1M elements, f64:
- Without fusion: ~16 MB extra (2× the working set)
- With fusion: 0 bytes extra (writes directly to output Vec)

Latency saved: the fused path also has better cache behavior — the CPU never evicts the working set between passes.

---

## 5. PyO3 Integration and GIL Management

### 5.1 When the GIL is released

```rust
// Inside PyQuery::to_list()
py.allow_threads(move || {
    new_pipeline.execute()  // Pure Rust — GIL fully released
})
```

This means another Python thread (including async I/O) can run while our pipeline executes. For CPU-bound pipelines over large arrays, this removes ZPyFlow as a GIL bottleneck.

### 5.2 When the GIL must be held

Python callbacks (lambdas) require the GIL because calling a `PyObject` acquires it internally:

```rust
// In apply_py_filter_f64():
for val in data {
    let py_val = val.into_py(py);        // GIL: convert f64 → PyFloat
    if pred.call1(py, (py_val,))?         // GIL: invoke Python callable
           .is_truthy(py)? {             // GIL: read bool
        out.push(val);
    }
}
```

Each element costs: 1 Python float allocation + 1 Python call + 1 bool read. This is ~10-50× slower than the Expr DSL path, but still faster than a Python generator because:
- We avoid Python's frame creation overhead (no `yield`)
- The Rust loop itself has less overhead than Python bytecode dispatch

### 5.3 Zero-copy numpy inputs

For numpy arrays, `from_numpy()` uses PyO3's buffer protocol to borrow the numpy buffer directly — no `.tolist()` copy:

```rust
// Implemented via buffer protocol:
let view = data.call_method0(py, "__array_interface__")?;
// or: PyReadonlyArray1<f64> via pyo3-numpy — zero-copy borrow of numpy memory
// Pipeline operates on Arc<[f64]> pointing into the original array's memory.
```

The query is executed synchronously before the Python array can be released, so the borrow is safe. This eliminates any allocation for the input stage on the numpy path.

### 5.4 Python callback overhead mitigation strategies

| Strategy             | Mechanism                                      | Status      |
|----------------------|------------------------------------------------|-------------|
| Expression DSL       | `col > 5` → `NumericOp::FilterGt(5.0)` in Rust | ✅ Done    |
| Expression bytecode  | Parse simple lambda ASTs into NumericOps       | ✅ Done (`_lambda_parser.py`) |
| Batched callbacks    | Call Python with chunks, not single elements   | Spec 080    |
| Cython callbacks     | Accept C function pointers from Cython/ctypes  | Backlog     |

---

## 6. Performance Philosophy

### 6.1 Why intermediate allocations are expensive

Each `Vec::new()` is a `malloc()` call. On a modern CPU:
- `malloc` takes ~20-100 ns (depends on allocator state)
- `free` takes ~10-50 ns
- For 1M element pipeline with 3 intermediate Vecs: ~300-500μs just in allocator overhead
- Plus: cache pollution from touching fresh pages (TLB misses, cache line fills)

ZPyFlow's fused path allocates **once** — the output `Vec` with pre-sized capacity from `size_hint()`.

### 6.2 Branch prediction and SIMD

Modern CPUs can predict branches with >95% accuracy when patterns repeat. When a filter has ~50% selectivity, the branch predictor fails and each misprediction costs 10-20 cycles. SIMD avoids this by processing 4 elements with a comparison + mask — no branch, predictable throughput.

### 6.3 Cache friendliness

- `SliceStream` and `VecStream` access memory linearly — hardware prefetcher is very effective
- `f64x4` processes 32 bytes per cycle (L1 cache line = 64 bytes = 2 SIMD vectors)
- At 1GHz effective memory bandwidth, a 1M×f64 (8MB) dataset fits in L3 but not L1/L2 — throughput-bound, not compute-bound

### 6.4 Comparison table

For 1M f64 elements, `filter(x > 0.5) + map(x * 2) + take(10_000)`:

| Library / approach          | Approx time | Allocations | Notes                     |
|-----------------------------|-------------|-------------|---------------------------|
| Python list comprehension   | ~80ms       | 2 lists     | GIL, per-element objects  |
| Python generator + take     | ~40ms       | 0 lists     | GIL, lazy but Python loop |
| numpy (vectorized)          | ~8ms        | 2 arrays    | SIMD but eager            |
| polars                      | ~3ms        | 1 array     | Columnar, Arrow backend   |
| ZPyFlow (Expr DSL, SIMD)    | ~2-5ms      | 1 list      | GIL-free, fused           |
| ZPyFlow (parallel, 8 cores) | ~0.5-1ms    | 1 list      | Rayon work-stealing       |
| Rust std iterator (baseline)| ~1.5ms      | 1 Vec       | Theoretical minimum       |

*Times are illustrative; actual results depend on hardware and data patterns.*

---

## 7. Query Optimization

### 7.1 Predicate pushdown

When multiple filters are chained, each `.filter()` appends a `NumericOp` to the pipeline's op queue. All ops are evaluated in a single fused pass at terminal execution — no intermediate `Vec` is created. For the Python object path, predicates are applied element-by-element inside one Rust loop.

### 7.2 Operation collapsing ✅

Consecutive map scalars are collapsed at op-push time:

```
MapMulScalar(2.0) → MapMulScalar(3.0)  collapses to  MapMulScalar(6.0)
MapAddScalar(1.0) → MapAddScalar(2.0)  collapses to  MapAddScalar(3.0)
```

This reduces loop iterations in the SIMD path with zero runtime overhead.

### 7.3 Lambda AST → DSL conversion ✅

For simple lambdas like `lambda x: x > 5`, `_lambda_parser.py` inspects `f.__code__` bytecode and converts them to Expr DSL automatically before the pipeline reaches Rust:

```python
# User writes:
Query(data).filter(lambda x: x > 5)
# _lambda_parser detects LOAD_FAST → LOAD_CONST → COMPARE_OP pattern
# Converted to:  Query(data).filter(col > 5)   — GIL-free Rust path
```

Falls back to the Python callback path for complex lambdas that can't be decoded.

### 7.4 Batched Python callbacks (Spec 080)

For lambdas that can't be converted to DSL, call Python once per chunk rather than once per element. This reduces GIL acquisition overhead proportionally to chunk size:

```python
# Current: called N times
.filter(lambda x: x > threshold)

# Spec 080 target: called N/256 times
# chunk: numpy array of 256 elements passed at once
```

See [spec 080](specs/080-query-optimizer/tasks.md).

---

## 8. AI Era Fit

### 8.1 Embedding pipelines

Vector databases work with embedding vectors (f32 / f64 arrays). ZPyFlow's dot product SIMD primitive and f32x8 path are directly applicable:

```python
from zpyflow import Query, col, from_numpy

embeddings = load_embeddings()  # shape: (N, 1536) f32

# Filter embeddings by similarity threshold, take top-K
results = (
    Query(cosine_similarities)           # Vec<f64>
    .filter(col > 0.85)                  # SIMD filter, GIL-free
    .take(100)
    .to_list()
)
```

### 8.2 ETL pipelines

```python
(
    from_csv("events.csv", column="value", dtype="float")
        .filter(col > 0)
        .map(col.sqrt())
        .skip(1000)           # skip warm-up period
        .to_list()
)
```

### 8.3 Why "Python frontend + Rust core" matters

The ML/data ecosystem is already Python-first. Rewriting user code in Rust is not viable. But:

1. **Python ecosystems don't change** — users keep their notebooks, visualization tools, ML frameworks
2. **Rust handles the hot path** — the inner loop, SIMD, memory layout
3. **PyO3 GIL management** — the Rust core can release the GIL, enabling true concurrency alongside Python async I/O
4. **Memory safety without GC pressure** — Rust's ownership model prevents leaks and dangling references without a garbage collector that would stall the pipeline

---

## 9. Module Breakdown

```
src/
├── lib.rs                        # PyO3 module entry point, registers all classes
├── core/
│   ├── mod.rs
│   ├── traits.rs                 # ZStream trait + operator structs (Map, Filter, Take …)
│   ├── sources.rs                # SliceStream, VecStream, RangeStream, RepeatN …
│   ├── parallel.rs               # Rayon par_execute_f64/i64, group-by parallelism
│   ├── numeric/
│   │   ├── mod.rs
│   │   ├── pipeline.rs           # NumericPipeline (f64), IntPipeline (i64), U8Pipeline
│   │   └── simd.rs               # f64x4/f32x8: filter-mask, map, sum, dot product
│   └── obj/
│       ├── mod.rs
│       └── pipeline.rs           # RustValue, ObjField pipeline for dict/dataclass path
├── io/
│   ├── mod.rs
│   ├── csv.rs                    # from_csv — Rust CSV reader → Vec<RustValue>
│   └── jsonl.rs                  # from_json_lines — line-by-line JSON parser
└── python/
    ├── mod.rs                    # Registers all Python-visible types
    ├── expr.rs                   # PyExpr, ColProxy, FieldExpr (DSL objects)
    ├── agg.rs                    # Aggregation kernels: sum_by, mean_by, count_by …
    ├── conversion.rs             # Python ↔ Rust type bridges
    ├── fastpath.rs               # Fused terminal reductions (sum+count in one pass)
    ├── io_bridge.rs              # Python-facing I/O wrappers (from_csv, from_arrow)
    └── query/
        ├── mod.rs                # PyQuery class, QueryInner enum
        ├── construct.rs          # Query construction: from list/numpy/arrow/generator
        ├── filter.rs             # .filter(), .take_while(), .skip_while()
        ├── map_ops.rs            # .map(), .flat_map(), .enumerate()
        ├── transform.rs          # .skip(), .take(), .zip(), .chain(), .tee() …
        └── terminal.rs           # .to_list(), .sum(), .count(), .reduce() …

zpyflow/
├── __init__.py                   # Public API: Query, col, field, from_* functions
├── __init__.pyi                  # Type stubs for mypy / IDE
├── _lambda_parser.py             # Lambda AST → DSL conversion
├── adapters.py                   # from_numpy, from_arrow, from_csv, from_json_lines
└── groupby.py                    # GroupBy (pure Python, hash-map based)

benches/
├── pipeline.rs                   # Criterion benchmarks: filter, map, chain, SIMD
└── simd_filter.rs                # SIMD selectivity analysis (10%…90%)

tests/
├── test_numeric.py               # f64/i64/u8 pipeline correctness
├── test_objects.py               # dict/dataclass, field() DSL
├── test_transforms.py            # map, flat_map, zip, chain, enumerate …
├── test_groupby.py               # group_by, sum_by, mean_by, count_by
├── test_io.py                    # from_csv, from_json_lines
├── test_misc.py                  # infinite sequences, edge cases
├── test_api_exports.py           # public API surface stability
└── test_performance.py           # regression guards (N=10K, not benchmark)
```

---

## 10. Risks and Tradeoffs

### 10.1 The clone problem ✅

`NumericPipeline` uses `Arc<Vec<f64>>` for shared ownership. The Python-visible `Query` object is immutable; each chained operator returns a new `Query` that shares the input `Arc` without copying. A real copy only occurs when the data must be mutated in-place, which is rare (in-place scalar ops on borrowed data). Most linear pipelines have zero clones of the data.

### 10.2 Python lambda overhead

When users pass Python lambdas to `.filter()` or `.map()`, performance is constrained by GIL overhead. On a benchmark with N=1M:
- Expr DSL path: ~2ms
- Python lambda path: ~80ms (same as pure Python)

**Mitigation implemented**: `_lambda_parser.py` inspects `f.__code__` bytecode and converts simple patterns (`lambda x: x > 5`, `lambda x: x * 2`, etc.) to Expr DSL automatically before the call reaches Rust. Complex lambdas fall back to the Python callback path. Residual gap for unconvertible lambdas is tracked in Spec 080 (batched callbacks).

### 10.3 Type detection at construction

`Query(data)` inspects the first call to `.extract::<Vec<f64>>()` on the input. For heterogeneous Python lists (mixed int/float/str), this detection fails and falls back to the `Py` path, even if the user intended numeric processing.

**Mitigation**: detect numeric homogeneity by sampling. Add explicit `Query.f64()` and `Query.i64()` constructors for performance-critical paths.

### 10.4 Dynamic dispatch at PyO3 boundary

`QueryInner` enum dispatch is O(1) but not zero-cost — it's a branch. In tight loops (many small `Query` objects), this is visible. For large data with few Python calls, it's invisible.

---

## 11. Building and Running

```bash
# Prerequisites
pip install maturin

# Development build (debug, fast recompile)
maturin develop

# Release build (SIMD, LTO, full optimization)
maturin develop --release

# Run tests
pytest tests/

# Run Rust benchmarks
cargo bench --bench pipeline

# Run with parallel feature enabled
cargo bench --bench pipeline --features parallel

# Build wheel for distribution
maturin build --release
```

---

## 12. Comparison with ZLinq Philosophy

| Concept                  | ZLinq (C#)                          | ZPyFlow (Rust + Python)                    |
|--------------------------|--------------------------------------|--------------------------------------------|
| Allocation elimination   | Value type operator chains           | `ZStream` monomorphic chains in Rust       |
| Iterator fusion          | Generic specialization, JIT inlining | LLVM inlining at `-O3`                     |
| SIMD                     | Explicit via Intrinsics / Vector<T>  | `wide` crate (stable), `portable_simd` (nightly) |
| Zero-copy                | `Span<T>`, `Memory<T>`              | `SliceStream<'a, T>`, `Arc<[f64]>` via buffer protocol (numpy) |
| Lazy evaluation          | IEnumerable chain (deferred)         | `NumericOp` queue, fused at `.execute()`  |
| Python/user boundary     | N/A (all C#)                         | Expr DSL collapses user intent to Rust ops |
| Parallelism              | `AsParallel()` (PLINQ)              | `.parallel()` (Rayon)                      |

The key philosophical difference: ZLinq can achieve true zero-allocation because C# generics fully specialize at the JIT boundary. ZPyFlow cannot achieve this across the Python ↔ Rust boundary, but it *can* achieve zero-allocation *within* the Rust execution core, which is where 90%+ of the work happens for large datasets.
