# bench_arrow.py — from_arrow() fast paths vs to_pylist() fallback
#
# Benchmarks:
#   - float64 null-free: buffer protocol fast path
#   - float64 with nulls: NaN fast path (new)
#   - float64 via to_pylist(): baseline
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_arrow.py -v --benchmark-columns=mean,ops

import pytest
import numpy as np

try:
    import pyarrow as pa
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

try:
    from zpyflow import from_arrow, col
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = [
    pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built"),
    pytest.mark.skipif(not HAS_ARROW, reason="pyarrow not installed"),
]

N = 1_000_000


@pytest.fixture(scope="module")
def arr_f64_null_free():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(N)
    return pa.array(data, type=pa.float64())


@pytest.fixture(scope="module")
def arr_f64_with_nulls():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(N).tolist()
    # ~10% nulls scattered throughout
    for i in range(0, N, 10):
        data[i] = None
    return pa.array(data, type=pa.float64())


@pytest.fixture(scope="module")
def arr_i64_null_free():
    return pa.array(range(N), type=pa.int64())


class TestArrowF64NullFree:
    """Float64 null-free: buffer protocol path vs to_pylist() baseline."""

    def test_from_arrow_buffer(self, benchmark, arr_f64_null_free):
        benchmark.group = "arrow f64 N=1M (null-free)"
        result = benchmark(lambda: from_arrow(arr_f64_null_free).filter(col > 0).count())
        assert result > 0

    def test_to_pylist_baseline(self, benchmark, arr_f64_null_free):
        benchmark.group = "arrow f64 N=1M (null-free)"
        def run():
            lst = arr_f64_null_free.to_pylist()
            return sum(1 for x in lst if x > 0)
        result = benchmark(run)
        assert result > 0

    def test_numpy_via_to_numpy(self, benchmark, arr_f64_null_free):
        benchmark.group = "arrow f64 N=1M (null-free)"
        def run():
            arr = arr_f64_null_free.to_numpy()
            return int((arr > 0).sum())
        result = benchmark(run)
        assert result > 0


class TestArrowF64WithNulls:
    """Float64 with nulls: NaN fast path vs to_pylist() fallback."""

    def test_from_arrow_nan_path(self, benchmark, arr_f64_with_nulls):
        benchmark.group = "arrow f64 N=1M (10% nulls)"
        # NaN fast path: nulls become NaN; filter(col == col) drops them
        result = benchmark(
            lambda: from_arrow(arr_f64_with_nulls).filter(col == col).filter(col > 0).count()
        )
        assert result > 0

    def test_to_pylist_with_none_filter(self, benchmark, arr_f64_with_nulls):
        benchmark.group = "arrow f64 N=1M (10% nulls)"
        def run():
            lst = arr_f64_with_nulls.to_pylist()
            return sum(1 for x in lst if x is not None and x > 0)
        result = benchmark(run)
        assert result > 0

    def test_numpy_nan_filter(self, benchmark, arr_f64_with_nulls):
        """NumPy: nulls → NaN via to_numpy(zero_copy_only=False), then isnan filter."""
        benchmark.group = "arrow f64 N=1M (10% nulls)"
        def run():
            arr = arr_f64_with_nulls.to_numpy(zero_copy_only=False)
            return int((arr[~np.isnan(arr) & (arr > 0)]).sum() > 0)
        result = benchmark(run)
        assert result >= 0


class TestArrowI64:
    """Int64 null-free: buffer protocol vs to_pylist()."""

    def test_from_arrow_buffer(self, benchmark, arr_i64_null_free):
        benchmark.group = "arrow i64 N=1M (null-free)"
        result = benchmark(lambda: from_arrow(arr_i64_null_free).filter(col > 500_000).count())
        assert result > 0

    def test_to_pylist_baseline(self, benchmark, arr_i64_null_free):
        benchmark.group = "arrow i64 N=1M (null-free)"
        def run():
            lst = arr_i64_null_free.to_pylist()
            return sum(1 for x in lst if x > 500_000)
        result = benchmark(run)
        assert result > 0
