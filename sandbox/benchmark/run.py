#!/usr/bin/env python3
"""
run.py — benchmark runner for ZPyFlow sandbox.
Mirrors ZLinq's sandbox/Benchmark/Program.cs.

Usage:
    python run.py                         # all benchmarks, default config
    python run.py --suite filter          # single suite
    python run.py --suite chained --n xl  # specific suite + data size
    python run.py --compare               # compare with saved baseline
    python run.py --list                  # list available suites
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
from pathlib import Path

HERE    = Path(__file__).parent
ROOT    = HERE.parent.parent
RESULTS = HERE / "results"

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
    "all":            "benchmarks/",
}

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
    parser.add_argument("--suite",   default="all",     choices=list(SUITES), help="benchmark suite to run")
    parser.add_argument("--config",  default="default",  choices=list(CONFIGS), help="benchmark config")
    parser.add_argument("--n",       default=None,       help="filter by data size label (xs/s/m/l/xl/xxl)")
    parser.add_argument("--compare", action="store_true", help="compare against saved baseline")
    parser.add_argument("--save",    action="store_true", help="save results as new baseline")
    parser.add_argument("--list",    action="store_true", help="list available suites and exit")
    args = parser.parse_args()

    if args.list:
        print("Available suites:")
        for name, path in SUITES.items():
            print(f"  {name:15s}  {path}")
        return 0

    RESULTS.mkdir(exist_ok=True)
    target = str(HERE / SUITES[args.suite])
    cmd    = ["pytest", target, "-v",
              "--override-ini=filterwarnings=",        # don't treat warnings as errors in benchmarks
              "--override-ini=python_files=bench_*.py",  # collect bench_*.py, not just test_*.py
             ] + CONFIGS[args.config]

    if args.n:
        cmd += ["-k", args.n]

    if args.save:
        cmd += [f"--benchmark-save=baseline_{args.suite}",
                f"--benchmark-storage={RESULTS}"]

    if args.compare:
        cmd += [f"--benchmark-compare=baseline_{args.suite}",
                f"--benchmark-storage={RESULTS}",
                "--benchmark-compare-fail=mean:10%"]  # fail if >10% regression

    if not args.compare and not args.save:
        # Default: auto-save each run for history
        cmd += [f"--benchmark-autosave",
                f"--benchmark-storage={RESULTS}"]

    print("Running:", " ".join(cmd))
    print()
    result = subprocess.run(cmd, cwd=str(HERE))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
