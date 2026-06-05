# API Reference

All public symbols are importable from `zpyflow` directly.

```python
from zpyflow import (
    Query, col, field,
    Expr, ColProxy, FieldExpr,
    AggSpec, GroupBy,
    agg_count, agg_sum, agg_mean, agg_max, agg_min,
    agg_median, agg_std, agg_first, agg_last,
    from_numpy, from_arrow, from_csv, from_json_lines, from_generator,
)
```

---

## Query

The core lazy pipeline class.

```python
Query(data: Iterable[T]) -> Query[T]
```

**Dispatch strategy**

| Input type | Fast path |
|---|---|
| `list[float]` / numpy `float64` | f64 (SIMD, GIL released) |
| `list[int]` / numpy `int64` | i64 (GIL released) |
| numpy `bool` / `uint8` | u8 compact (GIL released) |
| `list[dict]` | Obj; converts to RustObj on first `field()` filter |
| Anything else | Python path (GIL held) |

### Lazy combinators

| Method | Description |
|---|---|
| `.filter(pred)` | Keep elements matching `pred` (DSL or callable) |
| `.map(f)` | Transform each element (DSL or callable) |
| `.take(n)` | Stop after at most `n` elements |
| `.skip(n)` | Drop the first `n` elements |
| `.parallel()` | Request parallel execution (numeric paths only) |
| `.take_while(pred)` | Yield while `pred` is true, then stop |
| `.skip_while(pred)` | Skip while `pred` is true, then yield the rest |
| `.chain(other)` | Concatenate another `Query` after this one (typed fast path for f64/i64) |
| `.concat(other)` | Concatenate any iterable (list, generator, or `Query`) after this one |
| `.enumerate()` | Yield `(index, item)` tuples |
| `.zip(other)` | Pair with `other` (stops at shorter) |
| `.flat_map(f)` | Apply `f` and flatten one level |
| `.flatten()` | Expand each element one level, yielding its items individually |
| `.preload()` | Convert dict records to RustObj eagerly |
| `.group_by(key_fn)` | Group elements into a `GroupBy` object |
| `.group_agg(key_fn, **specs)` | Single-pass group + aggregate (Rust kernel) |
| `.cache()` | Materialise the pipeline into a list and return a reusable Query |
| `.tee(n=2)` | Materialise once and return `n` independent Query copies as a tuple |
| `.chunk(n)` | Split into fixed-size sublists of length `n` (last chunk may be shorter) |
| `.sort(reverse=False)` | Return a new `Query` with elements in sorted order |
| `.sort_by(key_fn, reverse=False)` | Return a new `Query` sorted by `key_fn` |
| `.distinct(key_fn=None)` | Remove duplicates (all occurrences), preserving insertion order |
| `.dedupe(key_fn=None)` | Remove **consecutive** duplicates (non-consecutive duplicates are kept) |
| `.scan(f, initial)` | Running accumulation — yield every intermediate value |
| `.sliding_window(n)` | Yield overlapping tuples of `n` consecutive elements |
| `.partition_by(key_fn=None)` | Group consecutive elements sharing the same key into sublists |

### Sequence tools

| Method | Description |
|---|---|
| `.cycle(n=None)` | Repeat the pipeline's elements `n` times; infinite when `n` is omitted |
| `.step_by(n)` | Return every `n`-th element (indices 0, n, 2n, …); `n` ≥ 1 |
| `.interleave(other)` | Alternate elements from `self` and `other`, stopping at the shorter |
| `.sample(n, seed=None)` | Return `n` elements chosen without replacement (random order) |
| `Query.iterate(fn, seed)` | `[seed, fn(seed), fn(fn(seed)), …]` — infinite lazy sequence |
| `Query.repeat(val, n=None)` | Repeat `val` exactly `n` times, or infinitely when `n` is omitted |
| `Query.repeatedly(fn, n=None)` | Call `fn()` repeatedly; infinite when `n` is omitted |

```python
# Clojure-style infinite sequence, take first 5 powers of 2
Query.iterate(lambda x: x * 2, 1).take(5).to_list()   # [1, 2, 4, 8, 16]

# Cycle a short list
Query([1, 2, 3]).cycle(2).to_list()   # [1, 2, 3, 1, 2, 3]

# Interleave two streams
Query([1, 2, 3]).interleave(Query([10, 20, 30])).to_list()  # [1, 10, 2, 20, 3, 30]

# Reproducible random sample
Query(range(100)).sample(10, seed=42).to_list()
```

### Object field operations

| Method | Description |
|---|---|
| `.map_field(name)` | Extract field `name` from each dict/object, yielding a scalar stream |
| `.set_field(name, fn)` | Replace field `name` with `fn(old_value)` in each dict |
| `.add_field(name, fn)` | Add a new field `name = fn(record)` to each dict |
| `.drop_field(*names)` | Remove the named fields from each dict |
| `.select(*fields)` | Keep only the specified fields in each dict (projection) |
| `.rename_field(old, new)` | Rename field `old` to `new` in each dict |
| `.value_counts(key_fn=None)` | Count occurrences of each element (or key) — returns `{value: count}` |
| `.inner_join(other, left_key, right_key=None)` | Hash inner-join; yield `(left, right)` tuples for matching keys |
| `.left_join(other, left_key, right_key=None)` | Hash left-join; yield `(left, right)` or `(left, None)` |

```python
# Field projection
records = [{"name": "Alice", "score": 90, "dept": "Eng"}]
Query(records).select("name", "score").to_list()
# [{"name": "Alice", "score": 90}]

# Derive a new field
Query(records).add_field("grade", lambda r: "A" if r["score"] >= 90 else "B").to_list()

# Join two record sets
users   = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
orders  = [{"uid": 1, "item": "book"}, {"uid": 1, "item": "pen"}]
results = Query(users).inner_join(orders, left_key="id", right_key="uid").to_list()
# [({'id': 1, 'name': 'Alice'}, {'uid': 1, 'item': 'book'}), ...]

# Frequency table
Query(["a", "b", "a", "c", "a"]).value_counts()
# {"a": 3, "b": 1, "c": 1}
```

### Terminal operations

| Method | Description |
|---|---|
| `.to_list()` | Collect to `list` |
| `.to_dict(key, value)` | Collect to `dict` using `key(item)` and `value(item)` callables |
| `.to_numpy()` | Collect to numpy `ndarray` (no per-element boxing; f64/i64/u8 only) |
| `.to_bytes()` | Collect numeric pipeline to raw `bytes` (little-endian) |
| `.count()` | Count matching elements |
| `.sum()` | Sum (SIMD for numeric) |
| `.mean()` | Arithmetic mean, or `None` if empty (single SIMD pass for f64 + filter) |
| `.var()` | Population variance (ddof=0), or `None` if empty |
| `.std()` | Population standard deviation (ddof=0), or `None` if empty |
| `.min()` | Minimum value |
| `.max()` | Maximum value (SIMD for f64) |
| `.stats()` | `{"count", "sum", "mean", "min", "max", "var", "std"}` in a single pass |
| `.first()` | First element or `None` |
| `.last()` | Last element or `None` |
| `.partition(pred)` | `(matching_list, non_matching_list)` in one pass (callable / `FieldExpr`) |
| `.reduce(fn, initial)` | General left fold |
| `.for_each(fn)` | Consume for side effects |
| `.any(pred)` | True if any element matches (short-circuits) |
| `.all(pred)` | True if all elements match (short-circuits) |
| `.explain()` | Human-readable pipeline description |

---

## col — numeric DSL sentinel

```python
from zpyflow import col

col > 5.0        # FilterGt
col >= 5.0       # FilterGe
col < 5.0        # FilterLt
col <= 5.0       # FilterLe
col * 2.0        # MapMulScalar
col + 1.0        # MapAddScalar
col - 1.0        # MapSubScalar
col / 2.0        # MapDivScalar
col ** 2         # MapPowScalar
col % 2          # MapMod (remainder)
col // 2         # MapFloorDiv
-col             # MapNeg
col.abs()        # MapAbs
col.sqrt()       # MapSqrt
col.floor()      # MapFloor
col.ceil()       # MapCeil
col.round()      # MapRound
col.reciprocal() # MapReciprocal
col.clamp(lo, hi)  # MapClamp — clamp to [lo, hi]
col.log()        # MapLog — natural log
col.log2()       # MapLog2
col.log10()      # MapLog10
col.exp()        # MapExp — e^x
col.sigmoid()    # MapSigmoid — 1/(1+e^-x)
col.between(a, b)  # FilterBetween (inclusive)
col.is_nan()     # FilterIsNan
col.not_nan()    # FilterNotNan
col.is_finite()  # FilterIsFinite
col.is_inf()     # FilterIsInf
```

---

## field() — object DSL

```python
from zpyflow import field

field("price") > 100           # FilterFieldGt
field("status") == 200         # FilterFieldEq
field("score").between(0, 1)   # FilterFieldBetween
field("name").startswith("A")  # FilterFieldStartswith
field("tag").contains("py")    # FilterFieldContains
field("code").matches(r"\d+")  # FilterFieldMatches (regex)

# Use as key in group_agg
Query(records).group_agg(field("category"), count=agg_count())
```

---

## Source adapters

| Function | Description |
|---|---|
| `from_numpy(arr)` | 1-D numpy array (buffer protocol, GIL-free memcpy) |
| `from_arrow(arr)` | PyArrow Array/ChunkedArray (buffer protocol for null-free numeric) |
| `from_csv(path, column, dtype, ...)` | CSV file or file-like |
| `from_json_lines(path, field, dtype)` | NDJSON file or file-like |
| `from_generator(gen)` | Eagerly materialise a generator |

---

## Aggregation specs

Used with `Query.group_agg()` and `GroupBy.agg()`:

```python
from zpyflow import (
    agg_count, agg_sum, agg_mean, agg_max, agg_min,
    agg_median, agg_std, agg_first, agg_last,
)

result = Query(records).group_agg(
    lambda r: r["category"],
    count   = agg_count(),
    revenue = agg_sum(lambda r: r["price"]),
    avg     = agg_mean(lambda r: r["price"]),
    median  = agg_median(lambda r: r["price"]),
    std     = agg_std(lambda r: r["price"]),
    first   = agg_first(lambda r: r["name"]),
    last    = agg_last(lambda r: r["name"]),
)
```

| Function | Description |
|---|---|
| `agg_count()` | Count elements in the group |
| `agg_sum(fn)` | Sum of `fn(item)` |
| `agg_mean(fn)` | Mean of `fn(item)` |
| `agg_max(fn)` | Maximum of `fn(item)` |
| `agg_min(fn)` | Minimum of `fn(item)` |
| `agg_median(fn)` | Median of `fn(item)` (materialises the group) |
| `agg_std(fn, ddof=0)` | Standard deviation of `fn(item)` |
| `agg_first(fn=None)` | First item or `fn(first_item)` |
| `agg_last(fn=None)` | Last item or `fn(last_item)` |

---

## GroupBy

```python
from zpyflow import GroupBy

gb = GroupBy(records, key_fn=lambda r: r["dept"])
gb.keys()                              # list of group keys
gb.get_group("Engineering")            # Query for one group
gb.count_per_group()                   # dict {key: count}
gb.sum_per_group(field=lambda r: r["salary"])  # dict {key: sum}
gb.agg(count=lambda g: g.count())      # list[dict] with "_key"
gb.map_groups(lambda k, g: (k, g.mean()))  # list of (key, value)
```
