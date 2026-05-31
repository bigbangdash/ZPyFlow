# Design Notes

ZPyFlow is inspired by [ZLinq](https://github.com/Cysharp/ZLinq) (zero-allocation LINQ for C#/.NET).

ZLinq fuses `Where().Select().Take()` into a single loop at JIT time using CLR generic
specialization.  ZPyFlow achieves the same fusion **inside the Rust core** using the
`ZStream` trait — every operator is a concrete generic type, LLVM inlines the full
chain at `-O3`.  The PyO3 boundary is the only place dynamic dispatch appears, and it
is crossed once per terminal call, not once per element.

## Dispatch strategy

```
Python data → Query(data)
                │
                ├─ list[float]  → LazyFloatList → F64 (SIMD, GIL released)
                ├─ list[int]    → I64 (SIMD, GIL released)
                ├─ list[dict]   → Obj (lazy)
                │                  └─ field() filter → RustObj (GIL-free after conversion)
                ├─ numpy f64    → F64 (buffer protocol memcpy)
                ├─ numpy i64    → I64 (buffer protocol memcpy)
                ├─ numpy bool   → U8  (compact 0/1)
                └─ other        → Py (Python object path, GIL held)
```

## GIL management

| Path | GIL behavior |
|---|---|
| F64 / I64 / U8 | Released via `py.allow_threads()` for all numeric ops |
| RustObj | Released for filter/count/sum after dict→RustObj conversion |
| Obj / Py | Held for each callable invocation |
| Parallel | Fully released; rayon work-stealing; re-acquired at collection |

## Further reading

- [ARCHITECTURE.md](../../ARCHITECTURE.md) — detailed design document
- [docs/when_is_it_fast.md](../../docs/when_is_it_fast.md)
