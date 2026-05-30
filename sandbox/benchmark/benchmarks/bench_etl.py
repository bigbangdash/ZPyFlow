# bench_etl.py — ETL pipeline: filter + multiple aggregations in one job
# Use case: batch processing of price/revenue data — filter a range, compute stats.
#
# Key question: ZPyFlow needs one pass per aggregation (count, sum, max separately).
# How does that compare to a Python single-pass accumulator, NumPy, and Polars?
#
# I/O separation: all input data is generated in session-scoped fixtures (prices_xl,
# arr_xl, series_pl_xl, series_pd_xl).  The benchmark timer starts AFTER data is
# loaded into memory — only computation cost is measured, not data generation.
# This is the "warm" scenario: compare against bench_arrow.py for "cold" (I/O + parse).
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_etl.py -v --benchmark-columns=mean,ops

import pytest
import numpy as np

from models import skewed_float_list, SIZES

try:
    from zpyflow import Query, col, from_numpy
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")

# Log-normal prices: realistic revenue/price distribution
PRICE_LO = 10.0
PRICE_HI = 500.0


@pytest.fixture(scope="session")
def prices_xl():
    return skewed_float_list(SIZES["xl"])   # 1M log-normal


@pytest.fixture(scope="session")
def arr_xl(prices_xl):
    return np.array(prices_xl)


@pytest.fixture(scope="session")
def series_pl_xl(prices_xl):
    if not HAS_POLARS:
        pytest.skip("polars not installed")
    return pl.Series(prices_xl)


@pytest.fixture(scope="session")
def series_pd_xl(prices_xl):
    if not HAS_PANDAS:
        pytest.skip("pandas not installed")
    return pd.Series(prices_xl)


# ---------------------------------------------------------------------------
# Multi-stat: count + sum + max in one ETL job
# ---------------------------------------------------------------------------

class TestMultiStatXL:
    """
    Compute (count, sum, max) of prices in [LO, HI].

    ZPyFlow: 3 separate pipelines = 3 SIMD passes.
    Python single-pass: 1 loop, accumulates all stats.
    NumPy: 3 vectorized ops on pre-filtered array (2 passes: mask + 3×aggregate).
    Polars: single expression, columnar, 1 pass.

    Design trade-off: each ZPyFlow terminal (count/sum/mean/std/max) is one SIMD
    pass.  A 6-stat ETL job (count+sum+mean+std+min+max) costs 6 full scans.
    Python single-pass still costs 1 loop regardless of stat count.
    Workarounds: preload() + Python aggregation, or delegate to pandas/polars.
    """

    def test_python_three_pass(self, benchmark, prices_xl):
        """3 list comprehensions — readable but 3 full passes."""
        benchmark.group = "ETL 3-stat N=1M"
        lo, hi = PRICE_LO, PRICE_HI
        def run():
            filtered = [x for x in prices_xl if lo <= x <= hi]
            return len(filtered), sum(filtered), max(filtered)
        cnt, total, mx = benchmark(run)
        assert cnt > 0

    def test_python_single_pass(self, benchmark, prices_xl):
        """1 loop accumulator — efficient, no intermediate list."""
        benchmark.group = "ETL 3-stat N=1M"
        lo, hi = PRICE_LO, PRICE_HI
        def run():
            cnt, total, mx = 0, 0.0, 0.0
            for x in prices_xl:
                if lo <= x <= hi:
                    cnt  += 1
                    total += x
                    if x > mx:
                        mx = x
            return cnt, total, mx
        cnt, total, mx = benchmark(run)
        assert cnt > 0

    def test_numpy(self, benchmark, arr_xl):
        """Mask once, then 3 vectorized aggregations on the filtered array."""
        benchmark.group = "ETL 3-stat N=1M"
        lo, hi = PRICE_LO, PRICE_HI
        def run():
            mask     = (arr_xl >= lo) & (arr_xl <= hi)
            filtered = arr_xl[mask]
            return len(filtered), float(filtered.sum()), float(filtered.max())
        cnt, total, mx = benchmark(run)
        assert cnt > 0

    def test_zpyflow_three_pass(self, benchmark, prices_xl):
        """3 separate SIMD pipelines — each re-executes the filter."""
        benchmark.group = "ETL 3-stat N=1M"
        def run():
            q = Query(prices_xl).filter(col.between(PRICE_LO, PRICE_HI))
            return q.count(), q.sum(), q.max()
        cnt, total, mx = benchmark(run)
        assert cnt > 0

    def test_zpyflow_single_pass(self, benchmark, prices_xl):
        """stats() — fused single pass: count + sum + min + max in one scan."""
        benchmark.group = "ETL 3-stat N=1M"
        def run():
            s = Query(prices_xl).filter(col.between(PRICE_LO, PRICE_HI)).stats()
            return s["count"], s["sum"], s["max"]
        cnt, total, mx = benchmark(run)
        assert cnt > 0

    def test_zpyflow_from_numpy(self, benchmark, arr_xl):
        """from_numpy + stats() — buffer protocol input, single Rust pass."""
        benchmark.group = "ETL 3-stat N=1M (numpy input)"
        def run():
            s = from_numpy(arr_xl).filter(col.between(PRICE_LO, PRICE_HI)).stats()
            return s["count"], s["sum"], s["max"]
        cnt, total, mx = benchmark(run)
        assert cnt > 0

    def test_numpy_from_numpy(self, benchmark, arr_xl):
        """numpy baseline with the same numpy-array input."""
        benchmark.group = "ETL 3-stat N=1M (numpy input)"
        lo, hi = PRICE_LO, PRICE_HI
        def run():
            mask = (arr_xl >= lo) & (arr_xl <= hi)
            filtered = arr_xl[mask]
            return len(filtered), float(filtered.sum()), float(filtered.max())
        cnt, total, mx = benchmark(run)
        assert cnt > 0

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_polars_single_pass(self, benchmark, series_pl_xl):
        """Polars: filter then aggregate in one columnar expression."""
        benchmark.group = "ETL 3-stat N=1M"
        lo, hi = PRICE_LO, PRICE_HI
        def run():
            filtered = series_pl_xl.filter(
                (series_pl_xl >= lo) & (series_pl_xl <= hi)
            )
            return filtered.len(), filtered.sum(), filtered.max()
        cnt, total, mx = benchmark(run)
        assert cnt > 0

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_pandas_single_pass(self, benchmark, series_pd_xl):
        """Pandas: boolean mask then describe-style aggregation."""
        benchmark.group = "ETL 3-stat N=1M"
        lo, hi = PRICE_LO, PRICE_HI
        def run():
            filtered = series_pd_xl[(series_pd_xl >= lo) & (series_pd_xl <= hi)]
            return len(filtered), float(filtered.sum()), float(filtered.max())
        cnt, total, mx = benchmark(run)
        assert cnt > 0


# ---------------------------------------------------------------------------
# 4-stat ETL: count + sum + mean + max
# ZPyFlow needs 4 separate SIMD passes.
# Python single-pass accumulates all four in one loop.
# ---------------------------------------------------------------------------

class TestFourStatXL:
    """
    count + sum + mean + max of prices in [LO, HI].
    ZPyFlow: 4 SIMD passes (each terminal re-executes the filter).
    Python: 1 loop accumulator.
    NumPy: 2 passes (mask once, then 4 reductions on filtered array).
    """

    def test_python_single_pass(self, benchmark, prices_xl):
        """1 loop: cnt, total, mean, max simultaneously."""
        benchmark.group = "ETL 4-stat N=1M"
        lo, hi = PRICE_LO, PRICE_HI
        def run():
            cnt, total, mx = 0, 0.0, 0.0
            for x in prices_xl:
                if lo <= x <= hi:
                    cnt   += 1
                    total += x
                    if x > mx:
                        mx = x
            return cnt, total, (total / cnt if cnt else None), mx
        result = benchmark(run)
        assert result[0] > 0

    def test_numpy(self, benchmark, arr_xl):
        """Mask once, then 4 reductions (2 passes total)."""
        benchmark.group = "ETL 4-stat N=1M"
        lo, hi = PRICE_LO, PRICE_HI
        def run():
            mask     = (arr_xl >= lo) & (arr_xl <= hi)
            filtered = arr_xl[mask]
            return len(filtered), float(filtered.sum()), float(filtered.mean()), float(filtered.max())
        result = benchmark(run)
        assert result[0] > 0

    def test_zpyflow_four_pass(self, benchmark, prices_xl):
        """4 separate SIMD passes — ZPyFlow multi-stat limitation."""
        benchmark.group = "ETL 4-stat N=1M"
        def run():
            q = Query(prices_xl).filter(col.between(PRICE_LO, PRICE_HI))
            return q.count(), q.sum(), q.mean(), q.max()
        result = benchmark(run)
        assert result[0] > 0

    def test_zpyflow_single_pass(self, benchmark, prices_xl):
        """stats() — fused single pass: count + sum + mean + min + max."""
        benchmark.group = "ETL 4-stat N=1M"
        def run():
            s = Query(prices_xl).filter(col.between(PRICE_LO, PRICE_HI)).stats()
            return s["count"], s["sum"], s["mean"], s["max"]
        result = benchmark(run)
        assert result[0] > 0


