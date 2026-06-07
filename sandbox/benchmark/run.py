#!/usr/bin/env python3
"""
run.py — benchmark runner for ZPyFlow sandbox.
Mirrors ZLinq's sandbox/Benchmark/Program.cs.

Usage:
    python run.py                         # all benchmarks, default config
    python run.py --suite filter          # single suite
    python run.py --suite chained --n xl  # specific suite + data size
    python run.py --suite threading       # concurrency benchmark (direct python)
    python run.py --compare               # compare with saved baseline
    python run.py --list                  # list available suites
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE    = Path(__file__).parent
ROOT    = HERE.parent.parent
RESULTS = HERE / "results"

# pytest-benchmark suites
SUITES = {
    "filter":         "benchmarks/bench_filter.py",
    "chained":        "benchmarks/bench_chained.py",
    "aggregation":    "benchmarks/bench_aggregation.py",
    "vs_numpy":       "benchmarks/bench_vs_numpy.py",
    "objects":        "benchmarks/bench_objects.py",
    "vector_search":  "benchmarks/bench_vector_search.py",
    "ml_feature":     "benchmarks/bench_ml_feature.py",
    "etl":            "benchmarks/bench_etl.py",
    "fraud":          "benchmarks/bench_fraud.py",
    "groupby":        "benchmarks/bench_groupby.py",
    "arrow":          "benchmarks/bench_arrow.py",
    "null":           "benchmarks/bench_null.py",
    "join_window":    "benchmarks/bench_join_window.py",
    "convenience":    "benchmarks/bench_convenience.py",
    "transform":      "benchmarks/bench_transform.py",
    "all":            "benchmarks/",
}

# Standalone scripts — run with `python` directly (not pytest-benchmark).
# These use concurrent.futures / HTTP servers that pytest cannot orchestrate.
DIRECT_SUITES = {
    "threading":    "benchmarks/bench_threading.py",
    "multiprocess": "benchmarks/bench_multiprocess.py",
    "fastapi":      "benchmarks/bench_fastapi.py",
}

ALL_SUITES = {**SUITES, **DIRECT_SUITES}

# bench_*.py files excluded from --suite all (direct-python scripts)
PYTEST_EXCLUDES = {Path(HERE / p).name for p in DIRECT_SUITES.values()}

CONFIGS = {
    "default": [
        "--benchmark-warmup-iterations=2",
        "--benchmark-min-rounds=5",
        "--benchmark-columns=mean,stddev,ops,rounds",
        "--benchmark-sort=mean",
    ],
    "quick": [
        "--benchmark-warmup-iterations=1",
        "--benchmark-min-rounds=3",
        "--benchmark-columns=mean,ops",
    ],
    "precise": [
        "--benchmark-warmup-iterations=5",
        "--benchmark-min-rounds=10",
        "--benchmark-columns=mean,stddev,median,iqr,ops,rounds",
        "--benchmark-sort=mean",
    ],
}


def main():
    parser = argparse.ArgumentParser(description="ZPyFlow benchmark runner")
    parser.add_argument("--suite",   default="all",     choices=list(ALL_SUITES), help="benchmark suite to run")
    parser.add_argument("--config",  default="default",  choices=list(CONFIGS), help="benchmark config")
    parser.add_argument("--n",       default=None,       help="filter by data size label (xs/s/m/l/xl/xxl)")
    parser.add_argument("--compare", action="store_true", help="compare against saved baseline")
    parser.add_argument("--save",    action="store_true", help="save results as new baseline")
    parser.add_argument("--list",    action="store_true", help="list available suites and exit")
    args = parser.parse_args()

    if args.list:
        print("pytest-benchmark suites:")
        for name, path in SUITES.items():
            print(f"  {name:15s}  {path}")
        print("\nDirect-python suites (no pytest-benchmark; --compare/--save ignored):")
        for name, path in DIRECT_SUITES.items():
            print(f"  {name:15s}  {path}")
        return 0

    # ── Direct-python suites ───────────────────────────────────────────────
    if args.suite in DIRECT_SUITES:
        script = HERE / DIRECT_SUITES[args.suite]
        print(f"Running: python {script}\n")
        result = subprocess.run([sys.executable, str(script)], cwd=str(HERE))
        return result.returncode

    # ── pytest-benchmark suites ────────────────────────────────────────────
    RESULTS.mkdir(exist_ok=True)
    target = str(HERE / SUITES[args.suite])
    cmd    = ["pytest", target, "-v",
              "--override-ini=filterwarnings=",
              "--override-ini=python_files=bench_*.py",
             ] + CONFIGS[args.config]

    # Exclude direct-python scripts when running the whole benchmarks/ dir
    if args.suite == "all":
        for name in PYTEST_EXCLUDES:
            cmd += ["--ignore", str(HERE / "benchmarks" / name)]

    if args.n:
        cmd += ["-k", args.n]

    if args.save:
        cmd += [f"--benchmark-save=baseline_{args.suite}",
                f"--benchmark-storage={RESULTS}"]

    if args.compare:
        cmd += [f"--benchmark-compare=baseline_{args.suite}",
                f"--benchmark-storage={RESULTS}",
                "--benchmark-compare-fail=mean:10%"]

    if not args.compare and not args.save:
        cmd += ["--benchmark-autosave",
                f"--benchmark-storage={RESULTS}"]

    print("Running:", " ".join(cmd))
    print()
    result = subprocess.run(cmd, cwd=str(HERE))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
