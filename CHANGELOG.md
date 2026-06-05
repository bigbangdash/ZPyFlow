# Changelog

All notable changes to ZPyFlow will be documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.1] — planned

### Fixed

- **None in mixed float lists produces consistent NaN** (spec-048).
  `Query([1.0, None, 2.0]).filter(col > 0).to_list()` previously raised
  `TypeError` or returned inconsistent results depending on test-suite
  execution order. `None` is now always converted to `NaN` at the Rust
  boundary via `PyErr_Clear`-based recovery in `pyfloat_as_f64`.

---

## [0.1.0] — 2026-05-xx — Initial release

> **Alpha.** API may change without notice between 0.x releases.

### Added

#### Core pipeline

| Method | Description |
|--------|-------------|
| `Query(data)` | Wrap any iterable — list, range, generator, NumPy array |
| `filter(pred)` | DSL expr (`col > 5`) or Python lambda |
| `map(fn)` | Element-wise transform |
| `flat_map(fn)` | Flatten one level of nesting |
| `flatten()` | Flatten nested iterables |
| `take(n)` / `skip(n)` | Slice the stream |
| `take_while(fn)` / `skip_while(fn)` | Predicate-based slicing |
| `sort()` / `sort_by(key)` | Materialise and sort |
| `distinct()` / `dedupe()` | Uniqueness — hash-based / adjacent |
| `chunk(n)` | Split into fixed-size batches |
| `scan(fn, init)` | Running accumulate |
| `sliding_window(n)` | Overlapping windows |
| `partition(pred)` | Split stream into (true, false) pair |
| `partition_by(key)` | Consecutive-group partitioning |
| `enumerate()` | `(index, value)` pairs |
| `zip(other)` | Pair-wise zip with another `Query` |
| `chain(other)` / `concat(others)` | Concatenate streams |
| `for_each(fn)` | Side-effect iteration |
| `reduce(fn, init)` | Left fold |

#### Numeric DSL (zero-allocation fast path)

- `col > n`, `col >= n`, `col < n`, `col <= n`, `col == n`, `col != n`
- Arithmetic: `+ - * / ** % //`
- Math: `abs`, `sqrt`, `floor`, `ceil`, `round`, `reciprocal`, `log`, `log2`, `log10`, `exp`, `sigmoid`, `clamp`, `between`
- NaN/finite guards: `is_nan`, `not_nan`, `is_finite`, `is_inf`
- Numeric ops collapse into a single SIMD-accelerated Rust pass; no intermediate Python lists

#### Object field DSL

- `field("key") > value` — filter dicts/dataclasses by field
- String predicates: `startswith`, `endswith`, `contains`, `matches` (regex)
- `map_field(name, fn)` — transform a single field value
- `set_field(name, value_or_fn)` — set / update a field
- `add_field(name, fn)` — derive a new field
- `drop_field(name)` — remove a field
- `select(fields)` / `rename_field(old, new)` — projection and rename

#### Aggregation

| Terminal | Description |
|----------|-------------|
| `sum()` / `min()` / `max()` / `mean()` | Scalar aggregation |
| `count()` / `any()` / `all()` | Count and boolean reductions |
| `stats()` | `{count, sum, mean, min, max, var, std}` in one pass |
| `var()` / `std()` | Variance / standard deviation |
| `first()` / `last()` | First or last element |
| `to_list()` | Materialise to Python list |
| `to_dict()` | Object pipeline → list of dicts |
| `to_numpy()` | Numeric pipeline → NumPy array (zero-copy where possible) |
| `to_bytes()` | Numeric pipeline → raw bytes |
| `value_counts()` | Frequency table |

#### GroupBy

- `group_by(key_fn)` → `GroupedQuery`
- `.agg(**reducers)` — arbitrary per-group aggregation
- `.map_groups(fn)` — transform each group
- `.count_per_group()` / `.sum_per_group(field_fn)`
- Helper constructors: `agg_count()`, `agg_sum()`, `agg_mean()`, `agg_min()`, `agg_max()`, `agg_median()`, `agg_std()`, `agg_first()`, `agg_last()`
- `group_agg(key_fn, **reducers)` — shorthand for one-liner aggregation

#### Join

- `inner_join(other, key)` — inner join on shared key
- `left_join(other, key)` — left join preserving all left rows

#### Sequence factories

- `Query.iterate(fn, seed)` — Clojure-style iterate
- `Query.repeat(value, n=None)` — finite or infinite repeat
- `Query.repeatedly(fn, n=None)` — call fn repeatedly
- `cycle(n=None)` — repeat the source n times (∞ if omitted)
- `step_by(n)` — every n-th element
- `interleave(other)` — alternate elements from two streams
- `sample(n, seed=None)` — random sampling without replacement

#### Materialisation control

- `cache()` — materialise once, iterate many times
- `tee(n=2)` — fork into n independent copies
- `parallel()` — execute subsequent operations in parallel (Rayon)
- `preload()` — eager materialisation for repeated field-DSL queries
- `explain()` — print the Rust-side query plan

#### I/O

- `Query.from_numpy(arr)` — wrap a NumPy array
- `Query.from_arrow(table_or_chunked)` — wrap an Arrow table (column-level)
- `Query.from_csv(path, ...)` — streaming CSV ingestion
- `Query.from_jsonlines(path)` — streaming JSONL ingestion
- `Query.f64(data)` / `Query.i64(data)` — explicit typed constructors

### Requirements

- Python 3.10+ (abi3 wheel — runs on 3.10, 3.11, 3.12, 3.13)
- Platforms: Linux x86-64, Linux aarch64, macOS (Apple Silicon + Intel universal2)
- Optional: NumPy ≥ 1.23 (for `to_numpy` / `from_numpy`), PyArrow ≥ 13 (for `from_arrow`)

### Known limitations

- `None` in mixed float lists is state-dependent (fixed in 0.1.1, see spec-048).
- Windows wheels are not yet published to PyPI.
- `parallel()` is experimental; GIL behaviour with nested lambdas is untested.

---

[Unreleased]: https://github.com/bigbangdash/ZPyFlow/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/bigbangdash/ZPyFlow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bigbangdash/ZPyFlow/releases/tag/v0.1.0
