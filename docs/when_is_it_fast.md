# When Is It Actually Fast?

An honest, data-driven answer for ZLinq and ZPyFlow.

---

## The short answer

Neither ZLinq nor ZPyFlow is fast for all cases.
Both are designed for **large data** processed with **simple, type-expressible operations**.

---

## What ZLinq's own README says about small data

> *"ValueEnumerable\<T\> is a struct, and its size increases slightly with each method
> chain. With many chained methods, copy costs can become significant.
> **When iterating over small collections, these copy costs can outweigh the benefits,
> causing performance to be worse than standard LINQ.**"*
>
> *"However, this is only an issue with extremely long method chains and small iteration
> counts, so it's **rarely a practical concern**."*

ZLinq acknowledges it explicitly and then moves on — because the library is built for
large-data workloads, where the tradeoff is invisible.

---

## ZLinq benchmark numbers (from the README)

```
Sum over N = 16,384 elements:

Method               Mean        Allocated
─────────────────── ─────────── ──────────
ForLoop             25,198 ns   –
System.Linq Sum      1,402 ns   –          ← standard LINQ baseline
ZLinq Sum            1,351 ns   –          ← +3.7% faster (barely)
ZLinq SumUnchecked     721 ns   –          ← +2x faster (SIMD path)
```

**Reading this correctly:**

- Zero-allocation alone buys only **3.7% speedup** at 16K elements.
- The **real 2× gain comes from SIMD** (`SumUnchecked`), not from removing allocations.
- At smaller N the struct-copy overhead would narrow or reverse that 3.7%.

Other benchmarks where SIMD dominates:

```
VectorizedUpdate N = 10,000:
  Plain for-loop        4,560 ns
  VectorizedUpdate        558 ns   ← 8× faster (SIMD)

VectorizableCount:
  System.Linq Count    10,909 ns
  ZLinq Count           1,048 ns   ← 10× faster (SIMD)
```

The pattern is clear: the wins are SIMD wins, not allocation wins.

---

## Why small data is slower — the structural reason

ZLinq chains operators as nested value-type structs:

```csharp
// After .Where().Select().Take() the type is:
ValueEnumerable<
  Take<
    Select<
      Where<FromArray<int>, int>,
    int, int>,
  int>,
int>
```

In C#, value types are copied when passed to methods.
The deeper the chain, the larger the struct, the more bytes copied on each call.
With small N, the copy cost dominates the iteration cost.

ZPyFlow has the same problem at its Python→Rust crossing:
every `Query(data)` call materializes Python objects into a Rust `Vec`,
and every `.to_list()` converts back.
For small data this round-trip cost exceeds any processing savings.

---

## The honest performance map

### By data size

| Elements | ZLinq vs std LINQ | ZPyFlow DSL to_list vs Python | ZPyFlow DSL aggregation vs Python |
|----------|-------------------|-------------------------------|-----------------------------------|
| < 1 K    | Equal or **slower** | Equal or **slower** | Equal or **slower** |
| 1 K–10 K | Marginally faster | **Slower** (PyO3 round-trip) | **~10× faster** (stays in Rust) |
| 10 K–100 K | ~1.5× faster | ~1.5× faster | **5–10× faster** |
| 100 K–1 M | 2–4× faster (SIMD) | 2–4× faster (SIMD) | **10–20× faster** |
| > 1 M    | 5–10× faster (SIMD) | ~2× faster | **10–20× faster** |

Measured on Docker Linux / CPython 3.11:
- N=10K filter+sum: Python 512µs / ZPyFlow 48µs / numpy 42µs
- N=10K filter+count: Python 415µs / ZPyFlow 40µs / numpy 11µs
- N=1M filter+to_list: Python 32ms / ZPyFlow 16ms / numpy+tolist 15ms

**Note**: crossover points differ between `to_list()` and aggregations.  
`count()` / `sum()` beat Python already at N=10K (result stays in Rust).  
`to_list()` crossover is around N=100K.

### By operation type

| Operation | ZPyFlow fast? | Why |
|-----------|--------------|-----|
| DSL filter/map on float list | ✅ at scale | Rust SIMD, GIL released |
| DSL filter/map on int list | ✅ at scale | Rust loop, GIL released |
| `.count()` / `.sum()` / `.max()` | ✅ always | No Python list created |
| Python lambda filter/map | ❌ | GIL held per element — same as Python |
| Small data (< 10 K) | ❌ | Round-trip cost dominates |
| Arbitrary Python objects | ❌ | GIL held, same speed as Python |

### By terminal operation

Not all terminal calls are equal.
Aggregations that stay inside Rust are much cheaper than collecting to Python:

```python
# Cheap: result stays in Rust as f64, one conversion at the end
Query(data).filter(col > 0).sum()    # fast
Query(data).filter(col > 0).count()  # fast
Query(data).filter(col > 0).max()    # fast

# More expensive: result is a Python list (O(N) PyObject creation)
Query(data).filter(col > 0).to_list()
```

---

## What "rarely a practical concern" actually means

ZLinq's authors dismiss small-data slowness because their target workload is:

- Game engines processing thousands of game objects per frame
- Server-side data transformation over database result sets
- Stream processing of large sequences

In all these cases N >> 1 K and chains are short (2–4 operators).

ZPyFlow's target workload is the same:

- ML/AI feature preprocessing (100 K–10 M rows)
- ETL pipelines over large CSV / JSON files
- Vector search score filtering (1 M+ candidates)
- Log analysis over millions of records

If your data fits comfortably in a Python list and you process it once, a list
comprehension is the right tool.  No library overhead, no Rust round-trip, no
explanation needed.

---

## None / null values

ZPyFlow infers the pipeline type from the **first element**. None handling depends on this.

| First element | Path | DSL (`col > 0`) | lambda (`x is not None and x > 0`) |
|--------------|------|-----------------|-------------------------------------|
| `float` | LazyFloatList | ❌ TypeError | ❌ TypeError (None cannot be converted to f64) |
| `None`  | Obj fallback  | ❌ not supported | ✅ lambda receives None as-is |
| `dict`  | Obj path      | ✅ field() DSL | ✅ lambda |

**Safe approaches (in order of preference):**

```python
# 1. from_arrow() — NaN fill path, fastest approach
from_arrow(pa_array_with_nulls).filter(col == col).filter(col > 0).count()

# 2. Pre-filter None before Query()
clean = [x for x in data if x is not None]
Query(clean).filter(col > 0).to_list()

# 3. Put None first to force Obj path
if data and data[0] is not None:
    data = [None] + data   # None first → Obj path
Query(data).filter(lambda x: x is not None and x > 0).to_list()
```

> **Note**: If None is the first element (e.g. 10% null list starting with None),
> ZPyFlow uses the Obj path and the lambda guard works correctly.
> If the list starts with a float, use approach 1 or 2.

---

## Decision guide

```
Data size < 10 K elements?
  → Use list comprehension or generator.  No library adds value here.

Paginating small in-memory lists?
  → Slice in Python before building a Query when you do not need DSL filtering.
  → ZPyFlow pagination is useful when it is part of a larger fused DSL pipeline
    or when pages are deep enough for Rust-side scanning to amortize overhead.

Data size 10 K – 100 K, simple numeric ops?
  → ZPyFlow DSL gives modest wins (~1.5×). Worth it if you chain 3+ ops.

Data size > 100 K, numeric, DSL-expressible?
  → ZPyFlow. SIMD + GIL release + single pass = 2–8× over Python.

Data is arbitrary Python objects (dicts, dataclasses)?
  → ZPyFlow still avoids intermediate lists, but speed is Python-equivalent.
  → The value is ergonomics (chainable API), not raw speed.

Need to aggregate without materializing a list?
  → ZPyFlow .sum() / .count() / .max() are always faster than to_list() + sum().

Using numpy already?
  → numpy is faster for pure array arithmetic (arr[arr>0]*2).
  → ZPyFlow wins on memory: one allocation vs numpy's two (mask + result).
  → For chained filter+map+take+sum, ZPyFlow can match or beat numpy.
```

---

## Summary

| Claim | Truth |
|-------|-------|
| "Zero-allocation = always faster" | ❌ False. Small data: copy overhead dominates |
| "ZLinq is faster than LINQ for everything" | ❌ False. README explicitly says otherwise |
| "ZPyFlow is faster than Python for everything" | ❌ False. Lambda path = same speed |
| "SIMD is the real win at scale" | ✅ True. ZLinq benchmarks prove this |
| "Large numeric data + DSL = significant speedup" | ✅ True for both ZLinq and ZPyFlow |
| "Aggregations (.sum, .count) are always worth it" | ✅ True. Stay in Rust, skip list creation |
