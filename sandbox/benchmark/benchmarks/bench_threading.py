# bench_threading.py — GIL release under thread contention
#
# Point: Python list comp holds the GIL → N threads serialize.
#        ZPyFlow numeric DSL releases the GIL → N threads run in parallel.
#        Object/dict pipelines (Python lambda path) do NOT release the GIL.
#
# Run:
#   python sandbox/benchmark/benchmarks/bench_threading.py
#   python sandbox/benchmark/run.py --suite threading

import time
import concurrent.futures

try:
    from zpyflow import Query, col, field, agg_count, agg_sum
except ImportError:
    print("zpyflow not built — run: maturin develop --profile dev-release")
    raise SystemExit(1)

N          = 500_000
N_THREADS  = 4
REPEAT     = 5

# ---------------------------------------------------------------------------
# Data (module-level so workers share the same object in-process)
# ---------------------------------------------------------------------------

floats = [float(i % 1000) for i in range(N)]
dicts  = [{"status": "active" if i % 3 != 0 else "inactive",
           "value":  float(i % 1000),
           "cat":    str(i % 10)}
          for i in range(N)]


# ---------------------------------------------------------------------------
# Task pairs  (python_fn, zpyflow_fn, data)
# ---------------------------------------------------------------------------

# 1. filter + count (numeric)
def py_filter_count(d):   return sum(1 for x in d if x > 500)
def zp_filter_count(d):   return Query(d).filter(col > 500).count()

# 2. filter + sum (numeric)
def py_filter_sum(d):     return sum(x for x in d if x > 500)
def zp_filter_sum(d):     return Query(d).filter(col > 500).sum()

# 3. stats() — 7-stat aggregation in one pass
def py_stats(d):
    xs = [x for x in d if x > 0]
    n  = len(xs)
    if not n:
        return {}
    s  = sum(xs)
    mn = min(xs)
    mx = max(xs)
    mean = s / n
    var  = sum((x - mean) ** 2 for x in xs) / n
    return {"count": n, "sum": s, "mean": mean, "min": mn, "max": mx, "var": var}
def zp_stats(d):          return Query(d).filter(col > 0).stats()

# 4. object filter (dict / field DSL)
def py_obj_filter(d):     return [r for r in d if r["status"] == "active" and r["value"] > 300]
def zp_obj_filter(d):     return Query(d).filter(lambda r: r["status"] == "active" and r["value"] > 300).to_list()

# 5. chained: filter → map → sum
def py_chained(d):        return sum(x * 2 for x in d if x > 500)
def zp_chained(d):        return Query(d).filter(col > 500).map(col * 2).sum()

# 6. group_agg on dicts
def py_group_agg(d):
    from collections import defaultdict
    groups = defaultdict(lambda: {"count": 0, "sum": 0.0})
    for r in d:
        g = groups[r["cat"]]
        g["count"] += 1
        g["sum"]   += r["value"]
    return dict(groups)
def zp_group_agg(d):
    return Query(d).group_agg(lambda r: r["cat"], count=agg_count(), total=agg_sum(lambda r: r["value"]))


SCENARIOS = [
    ("filter+count (numeric)", py_filter_count, zp_filter_count, floats),
    ("filter+sum   (numeric)", py_filter_sum,   zp_filter_sum,   floats),
    ("stats()      (numeric)", py_stats,        zp_stats,        floats),
    ("object filter  (dict) ", py_obj_filter,   zp_obj_filter,   dicts),
    ("chained f→map→sum     ", py_chained,      zp_chained,      floats),
    ("group_agg     (dict)  ", py_group_agg,    zp_group_agg,    dicts),
]


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def time_single(fn, data, repeat=REPEAT):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(data)
        times.append(time.perf_counter() - t0)
    return min(times)


def time_parallel(fn, data, n_threads=N_THREADS, repeat=REPEAT):
    times = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
        for _ in range(repeat):
            t0 = time.perf_counter()
            list(ex.map(fn, [data] * n_threads))
            times.append(time.perf_counter() - t0)
    return min(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print(f"N={N:,}  N_THREADS={N_THREADS}  best-of-{REPEAT}\n")
print(f"{'scenario / impl':<38s} {'1 thread':>9s}  {f'{N_THREADS} threads':>9s}  {'speedup':>7s}")
print("─" * 72)

for label, py_fn, zp_fn, data in SCENARIOS:
    t_py1 = time_single(py_fn,   data)
    t_zp1 = time_single(zp_fn,   data)
    t_pyn = time_parallel(py_fn, data)
    t_zpn = time_parallel(zp_fn, data)
    print(f"  {label}")
    print(f"    {'Python native':<34s} {t_py1*1e3:>8.1f}ms  {t_pyn*1e3:>8.1f}ms  {t_py1/t_pyn:>6.2f}x")
    print(f"    {'ZPyFlow':<34s} {t_zp1*1e3:>8.1f}ms  {t_zpn*1e3:>8.1f}ms  {t_zp1/t_zpn:>6.2f}x")

print()
print("Note: speedup > 1.0 → GIL released during Rust execution.")
print("      speedup ≈ 1.0 → GIL held; threads serialize (Python lambda path).")
