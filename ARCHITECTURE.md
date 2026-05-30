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
│  - GroupBy (pure Python, hash-map based)                    │
│  - Type stubs (.pyi) for IDE support                        │
└────────────────────────┬────────────────────────────────────┘
                         │ PyO3 extension call
┌────────────────────────▼────────────────────────────────────┐
│  PyO3 binding layer    (src/python/query.rs)                │
│  - PyQuery class — Python-visible, stores QueryInner        │
│  - Dispatch: F64 path / I64 path / Py object path           │
│  - PyExpr / ColProxy — expression DSL                       │
│  - GIL release for numeric paths                            │
└────────────┬───────────────────────────────┬────────────────┘
             │ NumericPipeline/IntPipeline   │ ZStream
┌────────────▼────────────┐  ┌──────────────▼───────────────┐
│  Numeric fast path      │  │  Generic ZStream pipeline     │
│  (src/pipeline/numeric) │  │  (src/pipeline/traits.rs)     │
│  - Op queue (no alloc)  │  │  - Map, Filter, Take, etc.    │
│  - Single-pass fusion   │  │  - SliceStream, VecStream     │
│  - SIMD dispatch        │  │  - Zero dynamic dispatch      │
└────────────┬────────────┘  └──────────────────────────────┘
             │
┌────────────▼────────────┐  ┌──────────────────────────────┐
│  SIMD layer             │  │  Parallel layer               │
│  (src/simd/mod.rs)      │  │  (src/parallel/mod.rs)        │
│  - f64x4 / f32x8        │  │  - Rayon work-stealing        │
│  - Filter with masking  │  │  - par_execute_f64/i64        │
│  - In-place map ops     │  │  - Group-by parallelism       │
│  - Dot product, sum     │  │  - GIL fully released         │
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
    F64(NumericPipeline),   // typed, GIL-free
    I64(IntPipeline),       // typed, GIL-free
    Py(Vec<PyObject>),      // generic, GIL required
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

### 5.3 Zero-copy opportunities

For numpy arrays, `from_numpy()` currently calls `.tolist()` which copies once. A future version can use PyO3's `PyReadonlyArray<f64>` to borrow the numpy buffer directly:

```rust
// Future zero-copy numpy path:
let arr: PyReadonlyArray1<f64> = data.extract()?;
let slice = arr.as_slice()?;  // zero-copy borrow of numpy memory
// Process via SliceStream::new(slice) — no clone!
```

This eliminates the one remaining allocation. Safe because the query is executed synchronously before the array is released.

### 5.4 Python callback overhead mitigation strategies

| Strategy             | Mechanism                                      | Status  |
|----------------------|------------------------------------------------|---------|
| Expression DSL       | `col > 5` → `NumericOp::FilterGt(5.0)` in Rust | ✅ MVP  |
| Batched callbacks    | Call Python with chunks, not single elements   | Roadmap |
| Numba JIT            | Accept `@jit`-compiled functions as fast paths | Roadmap |
| Expression bytecode  | Parse simple lambda ASTs into NumericOps       | Roadmap |
| Cython callbacks     | Accept C function pointers from Cython/ctypes  | Roadmap |

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

## 7. Query Optimization Ideas (Roadmap)

### 7.1 Predicate pushdown (Phase 2)

When multiple filters are chained, collect them into a single compound predicate evaluated in one pass. Currently each `.filter()` creates a new `NumericPipeline` with an appended op — this already achieves the same effect. For the Py path, predicate pushdown would avoid re-scanning items.

### 7.2 Operation collapsing (Phase 2)

Consecutive map scalars can be collapsed:

```
MapMulScalar(2.0) + MapMulScalar(3.0) → MapMulScalar(6.0)
MapAddScalar(1.0) + MapAddScalar(2.0) → MapAddScalar(3.0)
```

This reduces loop iterations in the SIMD path.

### 7.3 Python lambda AST analysis (Phase 3)

For simple lambdas like `lambda x: x > 5`, we can inspect `f.__code__.co_consts` and `co_code` to detect patterns and convert them to Expr DSL automatically:

```python
def lambda_to_expr(f):
    import dis
    instructions = list(dis.get_instructions(f))
    # Pattern: LOAD_FAST → LOAD_CONST → COMPARE_OP → RETURN_VALUE
    if len(instructions) == 4:
        if instructions[2].opname == 'COMPARE_OP':
            threshold = instructions[1].argval
            op = instructions[2].argval
            return ExprOp from (op, threshold)
    return None  # Fall back to Python callback
```

### 7.4 JIT via Cranelift / LLVM (Phase 4)

For complex numeric pipelines, generate Cranelift IR at query-construction time and JIT-compile the entire pipeline to native code. This would achieve ZLinq-equivalent performance for arbitrary expressions.

### 7.5 Batched Python callbacks (Phase 3)

Instead of calling Python once per element, call it once per chunk (e.g., 256 elements). The Python function receives a list and returns a list. This reduces GIL acquisition overhead by 256×:

```python
# Current: called N times
.filter(lambda x: x > threshold)

# Future: called N/256 times
.filter_batch(lambda chunk: [x for x in chunk if x > threshold])
```

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

### 8.2 Streaming inference / event processing

Rust `async` streams (tokio) can feed `ZPyFlow` pipelines:

```
Kafka consumer → tokio channel → ZPyFlow pipeline → Python handler
```

The pipeline stage runs in Rust with zero GIL contention; only the Python handler at the end acquires the GIL.

### 8.3 ETL pipelines

```python
(
    from_csv("events.csv", column="value", dtype="float")
        .filter(col > 0)
        .map(col.sqrt())
        .skip(1000)           # skip warm-up period
        .to_list()
)
```

### 8.4 Why "Python frontend + Rust core" matters

The ML/data ecosystem is already Python-first. Rewriting user code in Rust is not viable. But:

1. **Python ecosystems don't change** — users keep their notebooks, visualization tools, ML frameworks
2. **Rust handles the hot path** — the inner loop, SIMD, memory layout
3. **PyO3 GIL management** — the Rust core can release the GIL, enabling true concurrency alongside Python async I/O
4. **Memory safety without GC pressure** — Rust's ownership model prevents leaks and dangling references without a garbage collector that would stall the pipeline

---

## 9. Module Breakdown

```
src/
├── lib.rs                  # PyO3 module entry point
├── pipeline/
│   ├── mod.rs              # Re-exports
│   ├── traits.rs           # ZStream trait + all operator structs
│   │                         Map, Filter, Take, Skip, Zip, Chain,
│   │                         FlatMap, Enumerate, TakeWhile, SkipWhile
│   ├── sources.rs          # SliceStream, SliceRefStream, VecStream,
│   │                         RangeStream, RepeatN, Once, ChunkedStream
│   └── numeric.rs          # NumericPipeline, NumericOp, IntPipeline, IntOp
├── simd/
│   └── mod.rs              # f64x4/f32x8 operations: filter, map, sum, dot
├── parallel/
│   └── mod.rs              # Rayon-based parallel execution
└── python/
    ├── mod.rs
    └── query.rs            # PyQuery, PyExpr, PyColProxy (PyO3 classes)

zpyflow/
├── __init__.py             # Public API surface
├── adapters.py             # from_numpy, from_arrow, from_csv, from_json_lines
├── groupby.py              # GroupBy (pure Python, hash-map)
└── _types.pyi              # Type stubs for mypy / IDE

benches/
├── pipeline.rs             # Criterion benchmarks: filter, map, chain, SIMD
└── simd_filter.rs          # SIMD selectivity analysis

tests/
├── test_basic.py           # Unit tests: correctness, edge cases
└── test_performance.py     # Performance regression tests
```

---

## 10. Risks and Tradeoffs

### 10.1 The clone problem

`NumericPipeline::execute()` currently clones `data` before executing because the Python API requires the `Query` object to be immutable (calling `.filter()` should not consume the original). This one clone (O(N)) is the dominant allocation cost.

**Mitigation**: use `Arc<Vec<f64>>` for shared ownership, clone only when a new branch is created. Most linear pipelines won't clone at all.

### 10.2 Python lambda overhead

When users pass Python lambdas to `.filter()` or `.map()`, performance is constrained by GIL overhead. On a benchmark with N=1M:
- Expr DSL path: ~2ms
- Python lambda path: ~80ms (same as pure Python)

The expression DSL is the answer, but it requires users to use `col > 5` syntax instead of `lambda x: x > 5`. This is a learning curve.

**Mitigation**: automatically detect simple lambdas (AST analysis) and convert to Expr. This is in the roadmap.

### 10.3 Type detection at construction

`Query(data)` inspects the first call to `.extract::<Vec<f64>>()` on the input. For heterogeneous Python lists (mixed int/float/str), this detection fails and falls back to the `Py` path, even if the user intended numeric processing.

**Mitigation**: detect numeric homogeneity by sampling. Add explicit `Query.f64()` and `Query.i64()` constructors for performance-critical paths.

### 10.4 Dynamic dispatch at PyO3 boundary

`QueryInner` enum dispatch is O(1) but not zero-cost — it's a branch. In tight loops (many small `Query` objects), this is visible. For large data with few Python calls, it's invisible.

---

## 11. Phased Implementation Roadmap

### Phase 1 — MVP (current)

- [x] `ZStream` trait + all basic operators
- [x] `SliceStream`, `VecStream`, `RangeStream` sources
- [x] `NumericPipeline` (f64) with `NumericOp` DSL
- [x] `IntPipeline` (i64)
- [x] SIMD: `f64x4` filter and map operations
- [x] SIMD: dot product, sum, max
- [x] Parallel execution via Rayon
- [x] PyO3 `PyQuery` class with GIL release
- [x] Expression DSL (`col > 5`, `col * 2`, etc.)
- [x] Python adapters: numpy, arrow, CSV, JSON Lines
- [x] `GroupBy` (pure Python)
- [x] Type stubs (`.pyi`)
- [x] Criterion benchmark suite
- [x] pytest test suite

### Phase 2 — Performance hardening

- [ ] Zero-copy numpy path via `PyReadonlyArray<f64>`
- [ ] Arc-based pipeline sharing (eliminate clone)
- [ ] Operation collapsing (MapMul + MapMul → MapMul)
- [ ] f32 fast path (ML embeddings are typically f32)
- [ ] Chunked/streaming CSV source (no full materialization)
- [ ] Async stream source (tokio channel bridge)

### Phase 3 — Optimizer

- [ ] Lambda AST analysis → automatic Expr conversion
- [ ] Batched Python callbacks (chunk-level GIL acquisition)
- [ ] Predicate reordering (cheapest filter first)
- [ ] Numba `@jit` integration
- [ ] Arrow IPC zero-copy reader

### Phase 4 — Advanced

- [ ] JIT compilation via Cranelift
- [ ] Query plan visualization (`Query.explain()`)
- [ ] Distributed execution (partitioned across processes)
- [ ] GPU backend (CUDA via cudarc)
- [ ] SQL-like syntax extension

---

## 12. Building and Running

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

## 13. Comparison with ZLinq Philosophy

| Concept                  | ZLinq (C#)                          | ZPyFlow (Rust + Python)                    |
|--------------------------|--------------------------------------|--------------------------------------------|
| Allocation elimination   | Value type operator chains           | `ZStream` monomorphic chains in Rust       |
| Iterator fusion          | Generic specialization, JIT inlining | LLVM inlining at `-O3`                     |
| SIMD                     | Explicit via Intrinsics / Vector<T>  | `wide` crate (stable), `portable_simd` (nightly) |
| Zero-copy                | `Span<T>`, `Memory<T>`              | `SliceStream<'a, T>`, future `PyReadonlyArray` |
| Lazy evaluation          | IEnumerable chain (deferred)         | `NumericOp` queue, fused at `.execute()`  |
| Python/user boundary     | N/A (all C#)                         | Expr DSL collapses user intent to Rust ops |
| Parallelism              | `AsParallel()` (PLINQ)              | `.parallel()` (Rayon)                      |

The key philosophical difference: ZLinq can achieve true zero-allocation because C# generics fully specialize at the JIT boundary. ZPyFlow cannot achieve this across the Python ↔ Rust boundary, but it *can* achieve zero-allocation *within* the Rust execution core, which is where 90%+ of the work happens for large datasets.
