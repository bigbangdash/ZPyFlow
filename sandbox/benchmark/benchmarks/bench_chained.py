# bench_chained.py — chained pipeline benchmarks
# The core value proposition: filter + map + take in ONE fused pass.
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_chained.py -v --benchmark-columns=mean,ops

import pytest
import numpy as np
import itertools

from models import half_positive_float_list, SIZES

try:
    from zpyflow import Query, col, from_numpy
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")

TAKE_N = 10_000


@pytest.fixture(scope="session")
def data_xl():
    return half_positive_float_list(SIZES["xl"])  # 1M


@pytest.fixture(scope="session")
def arr_xl(data_xl):
    return np.array(data_xl)


# ---------------------------------------------------------------------------
# filter → map → take  (the "hello world" of lazy pipelines)
# ---------------------------------------------------------------------------

class TestChainedXL:
    """N=1M, filter+map+take.  Shows single-pass vs multi-pass allocation."""

    def test_python_two_listcomps(self, benchmark, data_xl):
        """Two separate list comprehensions — 2 full intermediate lists."""
        benchmark.group = "filter+map+take N=1M"
        def run():
            filtered = [x for x in data_xl if x > 0]
            mapped   = [x * 2 for x in filtered]
            return mapped[:TAKE_N]
        result = benchmark(run)
        assert len(result) <= TAKE_N

    def test_python_generator_chain(self, benchmark, data_xl):
        """Generator chain — lazy, 0 intermediate lists, but GIL per element."""
        benchmark.group = "filter+map+take N=1M"
        def run():
            return list(itertools.islice(
                (x * 2 for x in data_xl if x > 0),
                TAKE_N,
            ))
        result = benchmark(run)
        assert len(result) <= TAKE_N

    def test_numpy_eager(self, benchmark, arr_xl):
        """numpy — 2 intermediate arrays (mask + result), then slice."""
        benchmark.group = "filter+map+take N=1M"
        result = benchmark(lambda: (arr_xl[arr_xl > 0] * 2)[:TAKE_N].tolist())
        assert len(result) <= TAKE_N

    def test_zpyflow_dsl(self, benchmark, data_xl):
        """ZPyFlow DSL — 1 fused pass, 1 allocation, GIL released."""
        benchmark.group = "filter+map+take N=1M"
        result = benchmark(lambda: (
            Query(data_xl)
                .filter(col > 0)
                .map(col * 2)
                .take(TAKE_N)
                .to_list()
        ))
        assert len(result) <= TAKE_N

    def test_zpyflow_lambda(self, benchmark, data_xl):
        """ZPyFlow lambda — same structure but GIL held per element."""
        benchmark.group = "filter+map+take N=1M"
        result = benchmark(lambda: (
            Query(data_xl)
                .filter(lambda x: x > 0)
                .map(lambda x: x * 2)
                .take(TAKE_N)
                .to_list()
        ))
        assert len(result) <= TAKE_N

    def test_zpyflow_dsl_parallel(self, benchmark, data_xl):
        benchmark.group = "filter+map+take N=1M"
        result = benchmark(lambda: (
            Query(data_xl)
                .filter(col > 0)
                .map(col * 2)
                .parallel()
                .take(TAKE_N)
                .to_list()
        ))
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Chain depth — does adding more operators hurt?
# ---------------------------------------------------------------------------

class TestChainDepth:
    """Does a longer chain (more ops) cost more? It shouldn't with fusion.

    Each depth has both a Python equivalent and a ZPyFlow DSL version
    so the absolute performance gap is visible alongside the fusion benefit.
    """

    @pytest.fixture(scope="class")
    def data(self):
        return half_positive_float_list(SIZES["l"])  # 100K

    # depth 1: filter only
    def test_python_depth_1(self, benchmark, data):
        benchmark.group = "chain depth N=100K"
        result = benchmark(lambda: [x for x in data if x > 0])
        assert len(result) > 0

    def test_depth_1_filter(self, benchmark, data):
        benchmark.group = "chain depth N=100K"
        result = benchmark(lambda: Query(data).filter(col > 0).to_list())
        assert len(result) > 0

    # depth 2: filter + map
    def test_python_depth_2(self, benchmark, data):
        benchmark.group = "chain depth N=100K"
        result = benchmark(lambda: [x * 2 for x in data if x > 0])
        assert len(result) > 0

    def test_depth_2_filter_map(self, benchmark, data):
        benchmark.group = "chain depth N=100K"
        result = benchmark(lambda: Query(data).filter(col > 0).map(col * 2).to_list())
        assert len(result) > 0

    # depth 3: filter + map + map
    def test_python_depth_3(self, benchmark, data):
        benchmark.group = "chain depth N=100K"
        result = benchmark(lambda: [x * 2 + 1 for x in data if x > 0])
        assert len(result) > 0

    def test_depth_3_filter_map_map(self, benchmark, data):
        benchmark.group = "chain depth N=100K"
        result = benchmark(lambda: (
            Query(data).filter(col > 0).map(col * 2).map(col + 1).to_list()
        ))
        assert len(result) > 0

    # depth 4: filter + map + map + filter
    def test_python_depth_4(self, benchmark, data):
        benchmark.group = "chain depth N=100K"
        result = benchmark(lambda: [x * 2 + 1 for x in data if x > 0 and x * 2 + 1 < 100])
        assert len(result) >= 0

    def test_depth_4_filter_map_map_filter(self, benchmark, data):
        benchmark.group = "chain depth N=100K"
        result = benchmark(lambda: (
            Query(data).filter(col > 0).map(col * 2).map(col + 1).filter(col < 100).to_list()
        ))
        assert len(result) >= 0

    # depth 5: filter + map + filter + map + skip + take
    def test_python_depth_5(self, benchmark, data):
        benchmark.group = "chain depth N=100K"
        def run():
            gen = (x * 2 + 1 for x in data if x > 0 and x * 2 < 150)
            return list(itertools.islice(gen, 10, 1010))
        result = benchmark(run)
        assert len(result) > 0

    def test_depth_5_full_chain(self, benchmark, data):
        benchmark.group = "chain depth N=100K"
        result = benchmark(lambda: (
            Query(data)
                .filter(col > 0)
                .map(col * 2)
                .filter(col < 150)
                .map(col + 1)
                .skip(10)
                .take(1000)
                .to_list()
        ))
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Small N — verifies the "ZLinq overhead" claim from the docs
# ---------------------------------------------------------------------------

class TestSmallN:
    """Small data: does ZPyFlow add overhead vs plain Python?

    N=10 is excluded: at ~0.1µs operation the ~0.5-2µs PyO3 call overhead
    dominates and makes the comparison meaningless noise.
    N=100 is the PyO3 overhead regime — ZPyFlow is slower than Python here.
    N=1000 is the crossover point where SIMD starts to pay off.
    """

    @pytest.mark.parametrize("n", [100, 1_000])
    def test_python_listcomp(self, benchmark, n):
        data = half_positive_float_list(n)
        benchmark.group = f"small N={n} (PyO3 overhead regime)" if n <= 100 else f"small N={n}"
        result = benchmark(lambda: [x * 2 for x in data if x > 0])
        assert len(result) >= 0

    @pytest.mark.parametrize("n", [100, 1_000])
    def test_python_generator(self, benchmark, n):
        data = half_positive_float_list(n)
        benchmark.group = f"small N={n} (PyO3 overhead regime)" if n <= 100 else f"small N={n}"
        result = benchmark(lambda: list(x * 2 for x in data if x > 0))
        assert len(result) >= 0

    @pytest.mark.parametrize("n", [100, 1_000])
    def test_numpy(self, benchmark, n):
        data = half_positive_float_list(n)
        arr = np.array(data)
        benchmark.group = f"small N={n} (PyO3 overhead regime)" if n <= 100 else f"small N={n}"
        result = benchmark(lambda: (arr[arr > 0] * 2).tolist())
        assert len(result) >= 0

    @pytest.mark.parametrize("n", [100, 1_000])
    def test_zpyflow_lambda(self, benchmark, n):
        data = half_positive_float_list(n)
        benchmark.group = f"small N={n} (PyO3 overhead regime)" if n <= 100 else f"small N={n}"
        result = benchmark(lambda: Query(data).filter(lambda x: x > 0).map(lambda x: x * 2).to_list())
        assert len(result) >= 0

    @pytest.mark.parametrize("n", [100, 1_000])
    def test_zpyflow_dsl(self, benchmark, n):
        data = half_positive_float_list(n)
        benchmark.group = f"small N={n} (PyO3 overhead regime)" if n <= 100 else f"small N={n}"
        result = benchmark(lambda: Query(data).filter(col > 0).map(col * 2).to_list())
        assert len(result) >= 0
