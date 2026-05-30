# bench_aggregation.py — terminal aggregation benchmarks
# sum / count / min / max stay inside Rust: no Python list created.
# These should always be fast, even for moderate N.
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_aggregation.py -v --benchmark-columns=mean,ops

import math
import pytest
import numpy as np

from models import half_positive_float_list, skewed_float_list, SIZES, measure_peak_kb

try:
    from zpyflow import Query, col, from_numpy
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")


@pytest.fixture(scope="session")
def data_xl():
    return half_positive_float_list(SIZES["xl"])  # 1M

@pytest.fixture(scope="session")
def arr_xl(data_xl):
    return np.array(data_xl)

@pytest.fixture(scope="session")
def series_pd_xl(data_xl):
    if not HAS_PANDAS:
        pytest.skip("pandas not installed")
    return pd.Series(data_xl)

@pytest.fixture(scope="session")
def series_pl_xl(data_xl):
    if not HAS_POLARS:
        pytest.skip("polars not installed")
    return pl.Series(data_xl)

@pytest.fixture(scope="session")
def data_l():
    return half_positive_float_list(SIZES["l"])   # 100K


# ---------------------------------------------------------------------------
# sum — stays in Rust, SIMD horizontal reduction
# ---------------------------------------------------------------------------

class TestSum:
    def test_python_builtin_sum(self, benchmark, data_xl):
        benchmark.group = "sum N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: sum(x for x in data_xl if x > 0)
        )
        result = benchmark(lambda: sum(x for x in data_xl if x > 0))
        assert result > 0

    def test_numpy_sum(self, benchmark, arr_xl):
        benchmark.group = "sum N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: arr_xl[arr_xl > 0].sum()
        )
        result = benchmark(lambda: arr_xl[arr_xl > 0].sum())
        assert result > 0

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_pandas_sum(self, benchmark, series_pd_xl):
        benchmark.group = "sum N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: series_pd_xl[series_pd_xl > 0].sum()
        )
        result = benchmark(lambda: series_pd_xl[series_pd_xl > 0].sum())
        assert result > 0

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_polars_sum(self, benchmark, series_pl_xl):
        benchmark.group = "sum N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: series_pl_xl.filter(series_pl_xl > 0).sum()
        )
        result = benchmark(lambda: series_pl_xl.filter(series_pl_xl > 0).sum())
        assert result > 0

    def test_zpyflow_lambda(self, benchmark, data_xl):
        benchmark.group = "sum N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(lambda x: x > 0).sum()
        )
        result = benchmark(lambda: Query(data_xl).filter(lambda x: x > 0).sum())
        assert math.isclose(result, sum(x for x in data_xl if x > 0), rel_tol=1e-6)

    def test_zpyflow_dsl(self, benchmark, data_xl):
        benchmark.group = "sum N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(col > 0).sum()
        )
        result = benchmark(lambda: Query(data_xl).filter(col > 0).sum())
        assert math.isclose(result, sum(x for x in data_xl if x > 0), rel_tol=1e-6)

    def test_zpyflow_from_numpy_sum(self, benchmark, arr_xl):
        """from_numpy → buffer protocol (bulk memcpy) → Rust sum."""
        benchmark.group = "sum N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: from_numpy(arr_xl).filter(col > 0).sum()
        )
        result = benchmark(lambda: from_numpy(arr_xl).filter(col > 0).sum())
        assert math.isclose(result, float(arr_xl[arr_xl > 0].sum()), rel_tol=1e-6)

    def test_zpyflow_tolist_then_sum(self, benchmark, data_xl):
        benchmark.group = "sum N=1M (antipattern: to_list+sum)"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: sum(Query(data_xl).filter(col > 0).to_list())
        )
        result = benchmark(lambda: sum(Query(data_xl).filter(col > 0).to_list()))
        assert math.isclose(result, sum(x for x in data_xl if x > 0), rel_tol=1e-6)


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

class TestCount:
    def test_python_sum_bool(self, benchmark, data_xl):
        benchmark.group = "count N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: sum(1 for x in data_xl if x > 0)
        )
        result = benchmark(lambda: sum(1 for x in data_xl if x > 0))
        assert result > 0

    def test_numpy_count(self, benchmark, arr_xl):
        benchmark.group = "count N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: int((arr_xl > 0).sum())
        )
        result = benchmark(lambda: int((arr_xl > 0).sum()))
        assert result > 0

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_pandas_count(self, benchmark, series_pd_xl):
        benchmark.group = "count N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: int((series_pd_xl > 0).sum())
        )
        result = benchmark(lambda: int((series_pd_xl > 0).sum()))
        assert result > 0

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_polars_count(self, benchmark, series_pl_xl):
        benchmark.group = "count N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: series_pl_xl.filter(series_pl_xl > 0).len()
        )
        result = benchmark(lambda: series_pl_xl.filter(series_pl_xl > 0).len())
        assert result > 0

    def test_zpyflow_lambda(self, benchmark, data_xl):
        benchmark.group = "count N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(lambda x: x > 0).count()
        )
        result = benchmark(lambda: Query(data_xl).filter(lambda x: x > 0).count())
        assert result == sum(1 for x in data_xl if x > 0)

    def test_zpyflow_dsl(self, benchmark, data_xl):
        benchmark.group = "count N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(col > 0).count()
        )
        result = benchmark(lambda: Query(data_xl).filter(col > 0).count())
        assert result == sum(1 for x in data_xl if x > 0)

    def test_zpyflow_from_numpy_count(self, benchmark, arr_xl):
        """from_numpy → buffer protocol path (bulk memcpy, no boxing)."""
        benchmark.group = "count N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: from_numpy(arr_xl).filter(col > 0).count()
        )
        result = benchmark(lambda: from_numpy(arr_xl).filter(col > 0).count())
        assert result == int((arr_xl > 0).sum())


# ---------------------------------------------------------------------------
# max / min
# ---------------------------------------------------------------------------

class TestMaxMin:
    def test_python_max(self, benchmark, data_xl):
        benchmark.group = "max N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: max(x for x in data_xl if x > 0)
        )
        result = benchmark(lambda: max(x for x in data_xl if x > 0))
        assert result > 0

    def test_numpy_max(self, benchmark, arr_xl):
        benchmark.group = "max N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: float(arr_xl[arr_xl > 0].max())
        )
        result = benchmark(lambda: float(arr_xl[arr_xl > 0].max()))
        assert result > 0

    def test_zpyflow_lambda(self, benchmark, data_xl):
        benchmark.group = "max N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(lambda x: x > 0).max()
        )
        result = benchmark(lambda: Query(data_xl).filter(lambda x: x > 0).max())
        assert result == max(x for x in data_xl if x > 0)

    def test_zpyflow_dsl(self, benchmark, data_xl):
        benchmark.group = "max N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(col > 0).max()
        )
        result = benchmark(lambda: Query(data_xl).filter(col > 0).max())
        assert result == max(x for x in data_xl if x > 0)

    def test_python_min(self, benchmark, data_xl):
        benchmark.group = "min N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: min(x for x in data_xl if x > 0)
        )
        result = benchmark(lambda: min(x for x in data_xl if x > 0))
        assert result > 0

    def test_numpy_min(self, benchmark, arr_xl):
        benchmark.group = "min N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: float(arr_xl[arr_xl > 0].min())
        )
        result = benchmark(lambda: float(arr_xl[arr_xl > 0].min()))
        assert result > 0

    def test_zpyflow_lambda_min(self, benchmark, data_xl):
        benchmark.group = "min N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(lambda x: x > 0).min()
        )
        result = benchmark(lambda: Query(data_xl).filter(lambda x: x > 0).min())
        assert result == min(x for x in data_xl if x > 0)

    def test_zpyflow_dsl_min(self, benchmark, data_xl):
        benchmark.group = "min N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(col > 0).min()
        )
        result = benchmark(lambda: Query(data_xl).filter(col > 0).min())
        assert result == min(x for x in data_xl if x > 0)


# ---------------------------------------------------------------------------
# Aggregation vs to_list — shows the cost difference
# ---------------------------------------------------------------------------

class TestAggVsToList:
    """filter → sum を Python / numpy / pandas / polars / zpyflow で比較。"""

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    def test_python_genexp(self, benchmark, n_label, n):
        data = half_positive_float_list(n)
        benchmark.group = f"filter+sum N={n_label}"
        result = benchmark(lambda: sum(x for x in data if x > 0))
        assert result > 0

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    def test_numpy(self, benchmark, n_label, n):
        arr = np.array(half_positive_float_list(n))
        benchmark.group = f"filter+sum N={n_label}"
        result = benchmark(lambda: arr[arr > 0].sum())
        assert result > 0

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_pandas(self, benchmark, n_label, n):
        s = pd.Series(half_positive_float_list(n))
        benchmark.group = f"filter+sum N={n_label}"
        result = benchmark(lambda: s[s > 0].sum())
        assert result > 0

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_polars(self, benchmark, n_label, n):
        s = pl.Series(half_positive_float_list(n))
        benchmark.group = f"filter+sum N={n_label}"
        result = benchmark(lambda: s.filter(s > 0).sum())
        assert result > 0

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    def test_zpyflow_sum(self, benchmark, n_label, n):
        data = half_positive_float_list(n)
        benchmark.group = f"filter+sum N={n_label}"
        result = benchmark(lambda: Query(data).filter(col > 0).sum())
        assert result > 0

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    def test_python_count(self, benchmark, n_label, n):
        data = half_positive_float_list(n)
        benchmark.group = f"filter+count N={n_label}"
        result = benchmark(lambda: sum(1 for x in data if x > 0))
        assert result > 0

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    def test_numpy_count(self, benchmark, n_label, n):
        arr = np.array(half_positive_float_list(n))
        benchmark.group = f"filter+count N={n_label}"
        result = benchmark(lambda: int((arr > 0).sum()))
        assert result > 0

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    def test_zpyflow_count(self, benchmark, n_label, n):
        data = half_positive_float_list(n)
        benchmark.group = f"filter+count N={n_label}"
        result = benchmark(lambda: Query(data).filter(col > 0).count())
        assert result > 0

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    def test_zpyflow_tolist_then_sum(self, benchmark, n_label, n):
        data = half_positive_float_list(n)
        benchmark.group = f"filter+sum N={n_label}"
        result = benchmark(lambda: sum(Query(data).filter(col > 0).to_list()))
        assert result > 0


# ---------------------------------------------------------------------------
# filter + map + sum — iterator fusion の最も純粋なデモ
#
# Python genexp : lazy, GIL per element, 0 intermediate alloc
# NumPy         : (arr[arr>0] * 2).sum() — 中間配列 2 本
# ZPyFlow DSL   : filter+map+sum を 1 fused pass — 中間 Vec ゼロ、GIL 解放
# ---------------------------------------------------------------------------

class TestFilterMapSum:
    """filter → map → sum: scalar terminal なので出力 Vec すら作らない。"""

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    def test_python_genexp(self, benchmark, n_label, n):
        data = half_positive_float_list(n)
        benchmark.group = f"filter+map+sum N={n_label}"
        result = benchmark(lambda: sum(x * 2 for x in data if x > 0))
        assert result > 0

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    def test_numpy(self, benchmark, n_label, n):
        arr = np.array(half_positive_float_list(n))
        benchmark.group = f"filter+map+sum N={n_label}"
        result = benchmark(lambda: float((arr[arr > 0] * 2).sum()))
        assert result > 0

    @pytest.mark.parametrize("n_label,n", [("10K", 10_000), ("100K", 100_000), ("1M", 1_000_000)])
    def test_zpyflow_dsl(self, benchmark, n_label, n):
        data = half_positive_float_list(n)
        benchmark.group = f"filter+map+sum N={n_label}"
        result = benchmark(lambda: Query(data).filter(col > 0).map(col * 2).sum())
        assert result > 0


# ---------------------------------------------------------------------------
# mean — SIMD fused filter+mean (sum+count in one pass) vs Python / NumPy
# ---------------------------------------------------------------------------

class TestMean:
    def test_python_genexp(self, benchmark, data_xl):
        benchmark.group = "filter+mean N=1M"
        data = data_xl
        def run():
            s, n = 0.0, 0
            for x in data:
                if x > 0:
                    s += x; n += 1
            return s / n if n else None
        result = benchmark(run)
        assert result > 0

    def test_numpy(self, benchmark, arr_xl):
        benchmark.group = "filter+mean N=1M"
        result = benchmark(lambda: float(arr_xl[arr_xl > 0].mean()))
        assert result > 0

    def test_zpyflow_dsl(self, benchmark, data_xl):
        benchmark.group = "filter+mean N=1M"
        result = benchmark(lambda: Query(data_xl).filter(col > 0).mean())
        assert result is not None and result > 0

    def test_zpyflow_from_numpy(self, benchmark, arr_xl):
        """from_numpy → buffer protocol → SIMD fused filter+mean."""
        benchmark.group = "filter+mean N=1M"
        result = benchmark(lambda: from_numpy(arr_xl).filter(col > 0).mean())
        assert result is not None and result > 0


# ---------------------------------------------------------------------------
# std — SIMD fused filter+std vs Python / NumPy
# ---------------------------------------------------------------------------

class TestStd:
    def test_python_genexp(self, benchmark, data_xl):
        """1-pass Welford-style population std (matches ZPyFlow ddof=0)."""
        benchmark.group = "filter+std N=1M"
        data = data_xl
        def run():
            s, ssq, n = 0.0, 0.0, 0
            for x in data:
                if x > 0:
                    s += x; ssq += x * x; n += 1
            if n == 0:
                return 0.0
            mean = s / n
            return ((ssq / n) - mean * mean) ** 0.5
        result = benchmark(run)
        assert result > 0

    def test_numpy(self, benchmark, arr_xl):
        """NumPy std(ddof=0) — population std, matches ZPyFlow."""
        benchmark.group = "filter+std N=1M"
        result = benchmark(lambda: float(arr_xl[arr_xl > 0].std(ddof=0)))
        assert result > 0

    def test_zpyflow_dsl(self, benchmark, data_xl):
        """ZPyFlow std() uses ddof=0 (population std)."""
        benchmark.group = "filter+std N=1M"
        result = benchmark(lambda: Query(data_xl).filter(col > 0).std())
        assert result is not None and result > 0

    def test_zpyflow_from_numpy(self, benchmark, arr_xl):
        """from_numpy → buffer protocol → SIMD fused filter+std."""
        benchmark.group = "filter+std N=1M"
        result = benchmark(lambda: from_numpy(arr_xl).filter(col > 0).std())
        assert result is not None and result > 0


# ---------------------------------------------------------------------------
# mean / std at N=100K — small-to-medium overhead characteristics
# ---------------------------------------------------------------------------

class TestMeanStdSmall:
    """filter+mean and filter+std at N=100K.

    Complements TestMean/TestStd (N=1M) with the smaller regime where
    per-call PyO3 overhead is more visible relative to the compute time.
    """

    @pytest.fixture(scope="class")
    def data(self):
        return half_positive_float_list(SIZES["l"])  # 100K

    @pytest.fixture(scope="class")
    def arr(self, data):
        return np.array(data)

    def test_python_mean(self, benchmark, data):
        benchmark.group = "filter+mean N=100K"
        def run():
            s, n = 0.0, 0
            for x in data:
                if x > 0:
                    s += x; n += 1
            return s / n if n else None
        result = benchmark(run)
        assert result > 0

    def test_numpy_mean(self, benchmark, arr):
        benchmark.group = "filter+mean N=100K"
        result = benchmark(lambda: float(arr[arr > 0].mean()))
        assert result > 0

    def test_zpyflow_mean(self, benchmark, data):
        benchmark.group = "filter+mean N=100K"
        result = benchmark(lambda: Query(data).filter(col > 0).mean())
        assert result is not None and result > 0

    def test_python_std(self, benchmark, data):
        """1-pass population std."""
        benchmark.group = "filter+std N=100K"
        def run():
            s, ssq, n = 0.0, 0.0, 0
            for x in data:
                if x > 0:
                    s += x; ssq += x * x; n += 1
            if n == 0:
                return 0.0
            mean = s / n
            return ((ssq / n) - mean * mean) ** 0.5
        result = benchmark(run)
        assert result > 0

    def test_numpy_std(self, benchmark, arr):
        benchmark.group = "filter+std N=100K"
        result = benchmark(lambda: float(arr[arr > 0].std(ddof=0)))
        assert result > 0

    def test_zpyflow_std(self, benchmark, data):
        benchmark.group = "filter+std N=100K"
        result = benchmark(lambda: Query(data).filter(col > 0).std())
        assert result is not None and result > 0


# ---------------------------------------------------------------------------
# mean / std at N=10K — crossover / PyO3 overhead regime
# ---------------------------------------------------------------------------

class TestMeanStdXS:
    """filter+mean and filter+std at N=10K.

    At this size PyO3 round-trip overhead is the dominant cost.
    Completes the N = 10K / 100K / 1M scaling curve for mean and std.
    """

    @pytest.fixture(scope="class")
    def data(self):
        return half_positive_float_list(SIZES["m"])  # 10K

    @pytest.fixture(scope="class")
    def arr(self, data):
        return np.array(data)

    def test_python_mean(self, benchmark, data):
        benchmark.group = "filter+mean N=10K"
        def run():
            s, n = 0.0, 0
            for x in data:
                if x > 0:
                    s += x; n += 1
            return s / n if n else None
        result = benchmark(run)
        assert result > 0

    def test_numpy_mean(self, benchmark, arr):
        benchmark.group = "filter+mean N=10K"
        result = benchmark(lambda: float(arr[arr > 0].mean()))
        assert result > 0

    def test_zpyflow_mean(self, benchmark, data):
        benchmark.group = "filter+mean N=10K"
        result = benchmark(lambda: Query(data).filter(col > 0).mean())
        assert result is not None and result > 0

    def test_python_std(self, benchmark, data):
        benchmark.group = "filter+std N=10K"
        def run():
            s, ssq, n = 0.0, 0.0, 0
            for x in data:
                if x > 0:
                    s += x; ssq += x * x; n += 1
            if n == 0:
                return 0.0
            mean = s / n
            return ((ssq / n) - mean * mean) ** 0.5
        result = benchmark(run)
        assert result > 0

    def test_numpy_std(self, benchmark, arr):
        benchmark.group = "filter+std N=10K"
        result = benchmark(lambda: float(arr[arr > 0].std(ddof=0)))
        assert result > 0

    def test_zpyflow_std(self, benchmark, data):
        benchmark.group = "filter+std N=10K"
        result = benchmark(lambda: Query(data).filter(col > 0).std())
        assert result is not None and result > 0
