"""Per-case metrics for the zero-allocation goal.

Measures, for each fixed case in CASES:
  rust_allocs   – number of Rust heap allocations in ONE call   (deterministic)
  rust_bytes    – bytes requested by those allocations          (deterministic)
  rust_peak     – peak live Rust heap above the starting point  (deterministic)
  py_peak       – peak Python heap via tracemalloc              (deterministic)
  wall_us       – median wall time per call                     (noisy)
  cpu_us        – median CPU time per call, all threads         (noisy)

Requires the extension built with the counting allocator:
    maturin develop --profile dev-release --features alloc-stats

Usage:
    python scripts/metrics.py            # print table, write metrics/current.json
    python scripts/metrics.py --save     # also overwrite metrics/baseline.json (humans only)
    python scripts/metrics.py --check    # compare with baseline; exit 1 on regression
    python scripts/metrics.py --only filter   # substring filter on case names
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sandbox" / "benchmark"))

import zpyflow  # noqa: E402
from zpyflow import Query, col, field, from_numpy  # noqa: E402
from zpyflow import _zpyflow as _ext  # noqa: E402
from models import half_positive_float_list, nullable_float_list, log_dicts  # noqa: E402

OUT_DIR = ROOT / "metrics"

# Regression tolerances used by --check.
# Deterministic metrics: any increase fails. Timing: relative tolerance for noise.
TIME_TOL = 0.15  # 15 % (run-to-run noise on macOS is up to ~10 %)

N_F = 1_000_000  # float cases
N_D = 100_000    # dict cases


def build_cases():
    fl = half_positive_float_list(N_F)
    arr = np.array(fl)
    nl = nullable_float_list(N_F)
    dl = log_dicts(N_D)
    pre = Query(dl).preload()
    return {
        # numeric list → terminal aggregate (should be ~0 allocs)
        "f64.filter.count":        lambda: Query(fl).filter(col > 0).count(),
        "f64.filter.sum":          lambda: Query(fl).filter(col > 0).sum(),
        "f64.filter.max":          lambda: Query(fl).filter(col > 0).max(),
        "f64.filter.std":          lambda: Query(fl).filter(col > 0).std(),
        "f64.between.stats":       lambda: Query(fl).filter(col.between(-1.0, 1.0)).stats(),
        # numeric list → Python list (output list is unavoidable; intermediates aren't)
        "f64.filter.to_list":      lambda: Query(fl).filter(col > 0).to_list(),
        "f64.chain4.to_list":      lambda: Query(fl).filter(col > 0).map(col * 2).map(col + 1).filter(col < 100).to_list(),
        "f64.take10.to_list":      lambda: Query(fl).filter(col > 0).take(10).to_list(),
        # numpy source
        "np.filter.sum":           lambda: from_numpy(arr).filter(col > 0).sum(),
        # lambda / null (Python object path)
        "f64.lambda.count":        lambda: Query(fl).filter(lambda x: x > 0).count(),
        "null.lambda.count":       lambda: Query(nl).filter(lambda x: x is not None and x > 0).count(),
        # dict rows
        "dict.field.count":        lambda: Query(dl).filter(field("status") >= 500).count(),
        "dict.field.to_list":      lambda: Query(dl).filter(field("status") >= 500).to_list(),
        "dict.preloaded.count":    lambda: pre.filter(field("status") >= 500).count(),
        "dict.groupby.count":      lambda: Query(dl).group_by(lambda r: r["status"]).count_per_group(),
    }


def measure(fn, reps: int):
    fn()  # warm-up: thread pools, lazy statics, caches

    _ext._alloc_reset()
    fn()
    allocs, nbytes, peak = _ext._alloc_stats()

    tracemalloc.start()
    try:
        fn()
        _, py_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    walls, cpus = [], []
    for _ in range(reps):
        c0, w0 = time.process_time_ns(), time.perf_counter_ns()
        fn()
        w1, c1 = time.perf_counter_ns(), time.process_time_ns()
        walls.append(w1 - w0)
        cpus.append(c1 - c0)
    return {
        "rust_allocs": allocs,
        "rust_bytes": nbytes,
        "rust_peak": peak,
        "py_peak": py_peak,
        "wall_us": round(statistics.median(walls) / 1000, 1),
        "cpu_us": round(statistics.median(cpus) / 1000, 1),
    }


DET = ("rust_allocs", "rust_bytes", "rust_peak", "py_peak")
TIMED = ("wall_us", "cpu_us")


def check(cur: dict, base: dict) -> list[str]:
    bad = []
    for name, b in base.items():
        c = cur.get(name)
        if c is None:
            bad.append(f"{name}: case missing (cases must not be removed)")
            continue
        for k in DET:
            if c[k] > b[k]:
                bad.append(f"{name}: {k} {b[k]} -> {c[k]}")
        for k in TIMED:
            if c[k] > b[k] * (1 + TIME_TOL):
                bad.append(f"{name}: {k} {b[k]} -> {c[k]} (+{(c[k] / b[k] - 1) * 100:.0f}%)")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()

    if not hasattr(_ext, "_alloc_stats"):
        sys.exit("extension built without alloc-stats: "
                 "maturin develop --profile dev-release --features alloc-stats")

    cases = {k: v for k, v in build_cases().items() if a.only in k}
    res = {}
    hdr = f"{'case':24} {'allocs':>8} {'bytes':>12} {'peak':>12} {'py_peak':>12} {'wall_us':>10} {'cpu_us':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, fn in cases.items():
        m = measure(fn, a.reps)
        res[name] = m
        print(f"{name:24} {m['rust_allocs']:>8} {m['rust_bytes']:>12} {m['rust_peak']:>12} "
              f"{m['py_peak']:>12} {m['wall_us']:>10} {m['cpu_us']:>10}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "current.json").write_text(json.dumps(res, indent=1))
    if a.save:
        (OUT_DIR / "baseline.json").write_text(json.dumps(res, indent=1))
        print("saved metrics/baseline.json")
    if a.check:
        base = json.loads((OUT_DIR / "baseline.json").read_text())
        if a.only:
            base = {k: v for k, v in base.items() if a.only in k}
        bad = check(res, base)
        if bad:
            print("\nREGRESSION:\n  " + "\n  ".join(bad))
            sys.exit(1)
        print("\nOK: no regression vs baseline")


if __name__ == "__main__":
    main()
