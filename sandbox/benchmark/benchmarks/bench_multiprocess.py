# bench_multiprocess.py — scaling with ProcessPoolExecutor
#
# Each process has its own GIL, so both Python and ZPyFlow scale with cores.
# Key differences vs threading:
#   - ZPyFlow is faster per-process (SIMD), so absolute throughput is higher.
#   - Data must be pickled to worker processes — shown in the timing.
#
# Run:
#   python sandbox/benchmark/benchmarks/bench_multiprocess.py
#   python sandbox/benchmark/run.py --suite multiprocess

import time
import concurrent.futures

try:
    from zpyflow import Query, col, agg_count, agg_sum
    HAS_ZPYFLOW = True
except ImportError:
    print("zpyflow not built — run: maturin develop --profile dev-release")
    raise SystemExit(1)

N         = 500_000
N_WORKERS = 4
REPEAT    = 5

floats = [float(i % 1000) for i in range(N)]
dicts  = [{"status": "active" if i % 3 != 0 else "inactive",
           "value":  float(i % 1000),
           "cat":    str(i % 10)}
          for i in range(N)]


# ---------------------------------------------------------------------------
# Top-level task functions (required for pickling on macOS/Windows)
# ---------------------------------------------------------------------------

# 1. filter + count (numeric)
def py_filter_count(d):
    return sum(1 for x in d if x > 500)

def zp_filter_count(d):
    from zpyflow import Query, col
    return Query(d).filter(col > 500).count()


# 2. filter + sum (numeric)
def py_filter_sum(d):
    return sum(x for x in d if x > 500)

def zp_filter_sum(d):
    from zpyflow import Query, col
    return Query(d).filter(col > 500).sum()


# 3. stats() — 7-stat aggregation in one pass
def py_stats(d):
    xs   = [x for x in d if x > 0]
    n    = len(xs)
    if not n:
        return {}
    s    = sum(xs)
    mean = s / n
    var  = sum((x - mean) ** 2 for x in xs) / n
    return {"count": n, "sum": s, "mean": mean,
            "min": min(xs), "max": max(xs), "var": var}

def zp_stats(d):
    from zpyflow import Query, col
    return Query(d).filter(col > 0).stats()


# 4. object filter (dict / lambda)
def py_obj_filter(d):
    return [r for r in d if r["status"] == "active" and r["value"] > 300]

def zp_obj_filter(d):
    from zpyflow import Query
    return Query(d).filter(lambda r: r["status"] == "active" and r["value"] > 300).to_list()


# 5. chained: filter → map → sum
def py_chained(d):
    return sum(x * 2 for x in d if x > 500)

def zp_chained(d):
    from zpyflow import Query, col
    return Query(d).filter(col > 500).map(col * 2).sum()


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
    from zpyflow import Query, agg_count, agg_sum
    return Query(d).group_agg(lambda r: r["cat"],
                              count=agg_count(),
                              total=agg_sum(lambda r: r["value"]))


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


def time_parallel(fn, data, n_workers=N_WORKERS, repeat=REPEAT):
    times = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for _ in range(repeat):
            t0 = time.perf_counter()
            list(ex.map(fn, [data] * n_workers))
            times.append(time.perf_counter() - t0)
    return min(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"N={N:,}  N_WORKERS={N_WORKERS}  best-of-{REPEAT}")
    print("(times include pickle overhead for inter-process data transfer)\n")
    print(f"{'scenario / impl':<38s} {'1 proc':>9s}  {f'{N_WORKERS} procs':>9s}  {'speedup':>7s}")
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
    print("Compare with bench_threading.py to see the pickle overhead cost.")
