# bench_filter.py — filter benchmarks
# Answers: "Is ZPyFlow filter faster than Python / numpy?"
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_filter.py -v --benchmark-columns=mean,stddev,ops
#   pytest sandbox/benchmark/benchmarks/bench_filter.py -v -k "xl"   # 1M only
#   pytest sandbox/benchmark/benchmarks/bench_filter.py --benchmark-compare

import pytest
import numpy as np

from models import float_list, float_array, half_positive_float_list, SIZES, measure_peak_kb

try:
    from zpyflow import Query, col, from_numpy
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")

THRESHOLD = 0.0

# ---------------------------------------------------------------------------
# Fixtures: pre-generate data once per session (not per benchmark iteration)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def data_m():
    return half_positive_float_list(SIZES["m"])   # 10K

@pytest.fixture(scope="session")
def data_l():
    return half_positive_float_list(SIZES["l"])   # 100K

@pytest.fixture(scope="session")
def data_xl():
    return half_positive_float_list(SIZES["xl"])  # 1M

@pytest.fixture(scope="session")
def arr_m():
    return np.array(half_positive_float_list(SIZES["m"]))

@pytest.fixture(scope="session")
def arr_xl():
    return np.array(half_positive_float_list(SIZES["xl"]))


# ---------------------------------------------------------------------------
# N = 10K
# ---------------------------------------------------------------------------

class TestFilterM:
    def test_python_listcomp_m(self, benchmark, data_m):
        benchmark.group = "filter N=10K"
        result = benchmark(lambda: [x for x in data_m if x > THRESHOLD])
        assert len(result) > 0

    def test_python_generator_m(self, benchmark, data_m):
        benchmark.group = "filter N=10K"
        result = benchmark(lambda: list(x for x in data_m if x > THRESHOLD))
        assert len(result) > 0

    def test_numpy_m(self, benchmark, arr_m):
        benchmark.group = "filter N=10K"
        result = benchmark(lambda: arr_m[arr_m > THRESHOLD].tolist())
        assert len(result) > 0

    def test_zpyflow_lambda_m(self, benchmark, data_m):
        benchmark.group = "filter N=10K"
        result = benchmark(lambda: Query(data_m).filter(lambda x: x > THRESHOLD).to_list())
        assert result == [x for x in data_m if x > THRESHOLD]

    def test_zpyflow_dsl_m(self, benchmark, data_m):
        benchmark.group = "filter N=10K"
        result = benchmark(lambda: Query(data_m).filter(col > THRESHOLD).to_list())
        assert result == [x for x in data_m if x > THRESHOLD]


# ---------------------------------------------------------------------------
# N = 100K
# ---------------------------------------------------------------------------

class TestFilterL:
    def test_python_listcomp_l(self, benchmark, data_l):
        benchmark.group = "filter N=100K"
        result = benchmark(lambda: [x for x in data_l if x > THRESHOLD])
        assert len(result) > 0

    def test_numpy_l(self, benchmark, data_l):
        benchmark.group = "filter N=100K"
        arr = np.array(data_l)
        result = benchmark(lambda: arr[arr > THRESHOLD].tolist())
        assert len(result) > 0

    def test_zpyflow_dsl_l(self, benchmark, data_l):
        benchmark.group = "filter N=100K"
        result = benchmark(lambda: Query(data_l).filter(col > THRESHOLD).to_list())
        assert result == [x for x in data_l if x > THRESHOLD]

    def test_zpyflow_lambda_l(self, benchmark, data_l):
        benchmark.group = "filter N=100K"
        result = benchmark(lambda: Query(data_l).filter(lambda x: x > THRESHOLD).to_list())
        assert result == [x for x in data_l if x > THRESHOLD]


# ---------------------------------------------------------------------------
# N = 1M
# ---------------------------------------------------------------------------

class TestFilterXL:
    def test_python_listcomp_xl(self, benchmark, data_xl):
        benchmark.group = "filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: [x for x in data_xl if x > THRESHOLD]
        )
        result = benchmark(lambda: [x for x in data_xl if x > THRESHOLD])
        assert len(result) > 0

    def test_python_generator_xl(self, benchmark, data_xl):
        benchmark.group = "filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: list(x for x in data_xl if x > THRESHOLD)
        )
        result = benchmark(lambda: list(x for x in data_xl if x > THRESHOLD))
        assert len(result) > 0

    def test_numpy_xl(self, benchmark, arr_xl):
        benchmark.group = "filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: arr_xl[arr_xl > THRESHOLD]
        )
        result = benchmark(lambda: arr_xl[arr_xl > THRESHOLD])
        assert len(result) > 0

    def test_numpy_tolist_xl(self, benchmark, arr_xl):
        """numpy filter then tolist — fair comparison with ZPyFlow's list output."""
        benchmark.group = "filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: arr_xl[arr_xl > THRESHOLD].tolist()
        )
        result = benchmark(lambda: arr_xl[arr_xl > THRESHOLD].tolist())
        assert len(result) > 0

    def test_zpyflow_dsl_xl(self, benchmark, data_xl):
        benchmark.group = "filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(col > THRESHOLD).to_list()
        )
        result = benchmark(lambda: Query(data_xl).filter(col > THRESHOLD).to_list())
        assert result == [x for x in data_xl if x > THRESHOLD]

    def test_zpyflow_dsl_count_xl(self, benchmark, data_xl):
        """count() avoids Python list creation entirely."""
        benchmark.group = "filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(col > THRESHOLD).count()
        )
        result = benchmark(lambda: Query(data_xl).filter(col > THRESHOLD).count())
        assert result == sum(1 for x in data_xl if x > THRESHOLD)

    def test_zpyflow_lambda_xl(self, benchmark, data_xl):
        benchmark.group = "filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(lambda x: x > THRESHOLD).to_list()
        )
        result = benchmark(lambda: Query(data_xl).filter(lambda x: x > THRESHOLD).to_list())
        assert result == [x for x in data_xl if x > THRESHOLD]

    def test_zpyflow_parallel_xl(self, benchmark, data_xl):
        benchmark.group = "filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(data_xl).filter(col > THRESHOLD).parallel().to_list()
        )
        result = benchmark(lambda: Query(data_xl).filter(col > THRESHOLD).parallel().to_list())
        assert sorted(result) == sorted(x for x in data_xl if x > THRESHOLD)

    def test_from_numpy_xl(self, benchmark, arr_xl):
        """from_numpy + DSL filter."""
        benchmark.group = "filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: from_numpy(arr_xl).filter(col > THRESHOLD).to_list()
        )
        result = benchmark(lambda: from_numpy(arr_xl).filter(col > THRESHOLD).to_list())
        assert result == arr_xl[arr_xl > THRESHOLD].tolist()


# ---------------------------------------------------------------------------
# Selectivity sweep — how does SIMD perform at different pass rates?
# (Mirrors ZLinq's N-parameter parametrize pattern)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pct", [10, 25, 50, 75, 90])
def test_filter_selectivity_zpyflow(benchmark, pct):
    """ZPyFlow filter at various selectivities (% of data that passes)."""
    rng = np.random.default_rng(0)
    data = rng.uniform(-100, 100, SIZES["xl"]).tolist()
    threshold = -100 + 200 * (1 - pct / 100)
    benchmark.group = f"selectivity"
    benchmark.name  = f"zpyflow DSL {pct}% pass"
    result = benchmark(lambda: Query(data).filter(col > threshold).count())
    assert result >= 0


@pytest.mark.parametrize("pct", [10, 25, 50, 75, 90])
def test_filter_selectivity_python(benchmark, pct):
    rng = np.random.default_rng(0)
    data = rng.uniform(-100, 100, SIZES["xl"]).tolist()
    threshold = -100 + 200 * (1 - pct / 100)
    benchmark.group = f"selectivity"
    benchmark.name  = f"python listcomp {pct}% pass"
    result = benchmark(lambda: sum(1 for x in data if x > threshold))
    assert result >= 0
