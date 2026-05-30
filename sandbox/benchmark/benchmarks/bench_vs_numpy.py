# bench_vs_numpy.py — ZPyFlow vs numpy head-to-head
# numpy is heavily SIMD-optimized C code.  This shows where ZPyFlow can
# match it (memory efficiency) and where it can't (pure arithmetic speed).
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_vs_numpy.py -v --benchmark-columns=mean,ops

import pytest
import numpy as np

from models import half_positive_float_list, float_array, SIZES

try:
    from zpyflow import Query, col, from_numpy
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = [
    pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built"),
    pytest.mark.skipif(
        not True,  # numpy always available
        reason="numpy not installed",
    ),
]

N     = SIZES["xl"]  # 1M
_TAKE = 10_000       # module-level constant — safe to capture in lambdas


@pytest.fixture(scope="module")
def arr():
    return float_array(N)

@pytest.fixture(scope="module")
def lst(arr):
    return arr.tolist()


# ---------------------------------------------------------------------------
# Case 1: filter only
# ---------------------------------------------------------------------------

class TestFilterVsNumpy:
    def test_python_listcomp(self, benchmark, lst):
        benchmark.group = "filter N=1M"
        result = benchmark(lambda: [x for x in lst if x > 0])
        assert len(result) > 0

    def test_numpy_bool_index_tolist(self, benchmark, arr):
        benchmark.group = "filter N=1M"
        result = benchmark(lambda: arr[arr > 0].tolist())
        assert len(result) > 0

    def test_zpyflow_lambda(self, benchmark, lst):
        benchmark.group = "filter N=1M"
        result = benchmark(lambda: Query(lst).filter(lambda x: x > 0).to_list())
        assert len(result) > 0

    def test_zpyflow_dsl(self, benchmark, lst):
        """DSL: 1 allocation, GIL released, SIMD."""
        benchmark.group = "filter N=1M"
        result = benchmark(lambda: Query(lst).filter(col > 0).to_list())
        assert len(result) > 0

    def test_zpyflow_from_numpy(self, benchmark, arr):
        """from_numpy + DSL: buffer-protocol input + SIMD."""
        benchmark.group = "filter N=1M"
        result = benchmark(lambda: from_numpy(arr).filter(col > 0).to_list())
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Case 2: map only (pure arithmetic)
# ---------------------------------------------------------------------------

class TestMapVsNumpy:
    def test_python_listcomp(self, benchmark, lst):
        benchmark.group = "map (multiply) N=1M"
        result = benchmark(lambda: [x * 2.0 for x in lst])
        assert len(result) == N

    def test_numpy_multiply(self, benchmark, arr):
        """numpy returns ndarray — no Python boxing."""
        benchmark.group = "map (multiply) N=1M"
        result = benchmark(lambda: arr * 2.0)
        assert len(result) == N

    def test_numpy_multiply_tolist(self, benchmark, arr):
        """numpy + tolist — same output type as ZPyFlow (fair comparison)."""
        benchmark.group = "map (multiply) N=1M"
        result = benchmark(lambda: (arr * 2.0).tolist())
        assert len(result) == N

    def test_zpyflow_lambda(self, benchmark, lst):
        benchmark.group = "map (multiply) N=1M"
        result = benchmark(lambda: Query(lst).map(lambda x: x * 2.0).to_list())
        assert len(result) == N

    def test_zpyflow_dsl(self, benchmark, lst):
        benchmark.group = "map (multiply) N=1M"
        result = benchmark(lambda: Query(lst).map(col * 2.0).to_list())
        assert len(result) == N

    def test_zpyflow_from_numpy(self, benchmark, arr):
        benchmark.group = "map (multiply) N=1M"
        result = benchmark(lambda: from_numpy(arr).map(col * 2.0).to_list())
        assert len(result) == N

    def test_zpyflow_dsl_to_numpy(self, benchmark, lst):
        """to_numpy() output: Vec→ndarray zero-copy transfer."""
        benchmark.group = "map (multiply) N=1M → ndarray"
        result = benchmark(lambda: Query(lst).map(col * 2.0).to_numpy())
        assert len(result) == N

    def test_numpy_multiply_ndarray(self, benchmark, arr):
        """numpy: returns ndarray directly (no boxing) — fair comparison for ndarray output."""
        benchmark.group = "map (multiply) N=1M → ndarray"
        result = benchmark(lambda: arr * 2.0)
        assert len(result) == N


# ---------------------------------------------------------------------------
# Case 3: filter + map  (ZPyFlow's main advantage: 1 pass vs 2)
# ---------------------------------------------------------------------------

class TestFilterMapVsNumpy:
    def test_python_listcomp(self, benchmark, lst):
        benchmark.group = "filter+map N=1M"
        result = benchmark(lambda: [x * 2.0 for x in lst if x > 0])
        assert len(result) > 0

    def test_numpy_two_ops_tolist(self, benchmark, arr):
        benchmark.group = "filter+map N=1M"
        result = benchmark(lambda: (arr[arr > 0] * 2.0).tolist())
        assert len(result) > 0

    def test_zpyflow_lambda(self, benchmark, lst):
        benchmark.group = "filter+map N=1M"
        result = benchmark(lambda: Query(lst).filter(lambda x: x > 0).map(lambda x: x * 2.0).to_list())
        assert len(result) > 0

    def test_zpyflow_dsl(self, benchmark, lst):
        """DSL: filter+map in 1 fused SIMD pass."""
        benchmark.group = "filter+map N=1M"
        result = benchmark(lambda: Query(lst).filter(col > 0).map(col * 2.0).to_list())
        assert len(result) > 0

    def test_zpyflow_from_numpy(self, benchmark, arr):
        benchmark.group = "filter+map N=1M"
        result = benchmark(lambda: from_numpy(arr).filter(col > 0).map(col * 2.0).to_list())
        assert len(result) > 0

    def test_zpyflow_parallel(self, benchmark, lst):
        benchmark.group = "filter+map N=1M"
        result = benchmark(lambda: Query(lst).filter(col > 0).map(col * 2.0).parallel().to_list())
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Case 4: filter + map + take  (early termination — ZPyFlow wins clearly)
# ---------------------------------------------------------------------------

class TestFilterMapTakeVsNumpy:
    # Use a module-level constant instead of a class attribute accessed via
    # self.TAKE inside lambdas.  Three reasons:
    #   1. benchmark() calls the lambda thousands of times; self.TAKE performs
    #      an attribute lookup on every call, adding noise to timing results.
    #   2. Lambdas that close over `self` cannot be pickled, so they break
    #      pytest-xdist (parallel test execution).
    #   3. assert len(result) == self.TAKE is fragile: take(n) guarantees
    #      AT MOST n items, so if fewer pass the filter the assertion fails.
    #      A <= check is the correct invariant.

    def test_python_generator(self, benchmark, lst):
        benchmark.group = "filter+map+take N=1M take=10K"
        import itertools
        take = _TAKE
        result = benchmark(lambda: list(itertools.islice(
            (x * 2.0 for x in lst if x > 0), take
        )))
        assert len(result) <= take

    def test_numpy_no_early_stop(self, benchmark, arr):
        """numpy processes ALL 1M elements even though only 10K needed."""
        benchmark.group = "filter+map+take N=1M take=10K"
        take = _TAKE
        result = benchmark(lambda: (arr[arr > 0] * 2.0)[:take].tolist())
        assert len(result) <= take

    def test_zpyflow_lambda(self, benchmark, lst):
        benchmark.group = "filter+map+take N=1M take=10K"
        take = _TAKE
        result = benchmark(lambda: Query(lst)
            .filter(lambda x: x > 0).map(lambda x: x * 2.0).take(take).to_list())
        assert len(result) <= take

    def test_zpyflow_dsl(self, benchmark, lst):
        """DSL: stops after 10K — skips the rest of 1M."""
        benchmark.group = "filter+map+take N=1M take=10K"
        take = _TAKE
        result = benchmark(
            lambda: Query(lst).filter(col > 0).map(col * 2.0).take(take).to_list()
        )
        assert len(result) <= take


# ---------------------------------------------------------------------------
# Case 5: sum / aggregation
# ---------------------------------------------------------------------------

class TestSumVsNumpy:
    def test_python_sum(self, benchmark, lst):
        benchmark.group = "filter+sum N=1M"
        result = benchmark(lambda: sum(x for x in lst if x > 0))
        assert result > 0

    def test_numpy_filter_sum(self, benchmark, arr):
        benchmark.group = "filter+sum N=1M"
        result = benchmark(lambda: float(arr[arr > 0].sum()))
        assert result > 0

    def test_zpyflow_lambda(self, benchmark, lst):
        benchmark.group = "filter+sum N=1M"
        result = benchmark(lambda: Query(lst).filter(lambda x: x > 0).sum())
        assert result > 0

    def test_zpyflow_dsl(self, benchmark, lst):
        """DSL: fused SIMD filter+sum, no Vec allocation."""
        benchmark.group = "filter+sum N=1M"
        result = benchmark(lambda: Query(lst).filter(col > 0).sum())
        assert result > 0


# ---------------------------------------------------------------------------
# to_numpy() vs to_list() — output path comparison
# Shows the benefit of direct Vec→ndarray transfer over per-element boxing.
# ---------------------------------------------------------------------------

class TestToNumpyVsToList:
    """filter → output: compare to_numpy() vs to_list() + np.array()."""

    def test_numpy_boolean_index(self, benchmark, arr):
        """Baseline: numpy boolean indexing returns an ndarray directly."""
        benchmark.group = "filter→ndarray N=1M"
        result = benchmark(lambda: arr[arr > 0])
        assert len(result) > 0

    def test_zpyflow_to_list_then_array(self, benchmark, lst):
        """to_list() + np.array(): two allocations, per-element float boxing."""
        benchmark.group = "filter→ndarray N=1M"
        result = benchmark(lambda: np.array(Query(lst).filter(col > 0).to_list()))
        assert len(result) > 0

    def test_zpyflow_to_numpy(self, benchmark, lst):
        """to_numpy(): one allocation, zero per-element boxing."""
        benchmark.group = "filter→ndarray N=1M"
        result = benchmark(lambda: Query(lst).filter(col > 0).to_numpy())
        assert len(result) > 0

    def test_zpyflow_from_numpy_to_numpy(self, benchmark, arr):
        """from_numpy → SIMD filter → to_numpy: full zero-copy round-trip."""
        benchmark.group = "filter→ndarray N=1M"
        result = benchmark(lambda: from_numpy(arr).filter(col > 0).to_numpy())
        assert len(result) > 0
