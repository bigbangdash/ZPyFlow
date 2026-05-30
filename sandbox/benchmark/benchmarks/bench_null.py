# bench_null.py — filter on null-mixed Python lists
#
# Answers: "How does ZPyFlow handle None values in a Python list?"
#
# Null handling semantics:
#   Python native  : explicit `x is not None` guard — skips None
#   ZPyFlow lambda : works ONLY when the list starts with None → Obj path.
#                    nullable_float_list(null_rate=0.1) sets data[0]=None,
#                    so Query(data) falls to the Obj fallback (not LazyFloatList).
#                    The lambda receives raw Python objects, including None.
#   ZPyFlow DSL    : NOT supported; `col > 0` on a None-first list falls to Obj
#                    path which doesn't support numeric DSL on plain values.
#
# If the list starts with a float (LazyFloatList path):
#   BOTH DSL and lambda fail when a None element is encountered.
#   → See tests/test_basic.py::TestNoneInList for the full behavior matrix.
#   → Use from_arrow() with NaN fill for nullable Arrow arrays (bench_arrow.py).
#
# This file benchmarks the Obj-path lambda approach (safe for None-first lists).
# See bench_arrow.py TestArrowF64WithNulls for the Arrow/NaN fast path.
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_null.py -v --benchmark-columns=mean,ops

import pytest

from models import nullable_float_list, SIZES

try:
    from zpyflow import Query
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")

NULL_RATE = 0.10   # 10% of elements are None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def data_m():
    return nullable_float_list(SIZES["m"], null_rate=NULL_RATE)   # 10K, 10% None


@pytest.fixture(scope="session")
def data_l():
    return nullable_float_list(SIZES["l"], null_rate=NULL_RATE)   # 100K, 10% None


@pytest.fixture(scope="session")
def data_xl():
    return nullable_float_list(SIZES["xl"], null_rate=NULL_RATE)  # 1M, 10% None


# ---------------------------------------------------------------------------
# N = 10K
# ---------------------------------------------------------------------------

class TestNullFilterM:
    """filter on 10% null list, N=10K — PyO3 overhead regime."""

    def test_python_listcomp_m(self, benchmark, data_m):
        benchmark.group = "null filter N=10K (10% None)"
        result = benchmark(lambda: [x for x in data_m if x is not None and x > 0])
        expected = [x for x in data_m if x is not None and x > 0]
        assert result == expected

    def test_zpyflow_lambda_m(self, benchmark, data_m):
        benchmark.group = "null filter N=10K (10% None)"
        result = benchmark(
            lambda: Query(data_m).filter(lambda x: x is not None and x > 0).to_list()
        )
        assert result == [x for x in data_m if x is not None and x > 0]


# ---------------------------------------------------------------------------
# N = 100K
# ---------------------------------------------------------------------------

class TestNullFilterL:
    """filter on 10% null list, N=100K."""

    def test_python_listcomp_l(self, benchmark, data_l):
        benchmark.group = "null filter N=100K (10% None)"
        result = benchmark(lambda: [x for x in data_l if x is not None and x > 0])
        assert len(result) > 0

    def test_zpyflow_lambda_l(self, benchmark, data_l):
        benchmark.group = "null filter N=100K (10% None)"
        result = benchmark(
            lambda: Query(data_l).filter(lambda x: x is not None and x > 0).to_list()
        )
        assert result == [x for x in data_l if x is not None and x > 0]


# ---------------------------------------------------------------------------
# N = 1M
# ---------------------------------------------------------------------------

class TestNullFilterXL:
    """filter on 10% null list, N=1M.

    nullable_float_list sets data[0]=None, so Query(data) uses the Obj path.
    Lambda with an explicit None guard works correctly on this path.

    DSL (`col > 0`) is not benchmarked: Obj path does not support numeric DSL
    on plain float values (only on dicts via field() DSL).
    For nullable Arrow data use from_arrow() — see bench_arrow.py.
    """

    def test_python_listcomp_xl(self, benchmark, data_xl):
        benchmark.group = "null filter N=1M (10% None)"
        result = benchmark(lambda: [x for x in data_xl if x is not None and x > 0])
        assert len(result) > 0

    def test_python_genexp_xl(self, benchmark, data_xl):
        """Generator with None guard — lazy, 0 intermediate list."""
        benchmark.group = "null filter N=1M (10% None)"
        result = benchmark(lambda: list(x for x in data_xl if x is not None and x > 0))
        assert len(result) > 0

    def test_zpyflow_lambda_xl(self, benchmark, data_xl):
        """Obj path (data starts with None): lambda receives Python objects, None guard works."""
        benchmark.group = "null filter N=1M (10% None)"
        result = benchmark(
            lambda: Query(data_xl).filter(lambda x: x is not None and x > 0).to_list()
        )
        assert result == [x for x in data_xl if x is not None and x > 0]


# ---------------------------------------------------------------------------
# count variant — avoids Python list creation
# ---------------------------------------------------------------------------

class TestNullCountXL:
    """filter + count on 10% null list, N=1M.

    count() keeps the result in Rust (no Python list), so only the per-element
    lambda overhead is paid — same as Python genexp but without list allocation.
    """

    def test_python_sum_bool_xl(self, benchmark, data_xl):
        benchmark.group = "null count N=1M (10% None)"
        result = benchmark(lambda: sum(1 for x in data_xl if x is not None and x > 0))
        assert result > 0

    def test_zpyflow_lambda_count_xl(self, benchmark, data_xl):
        benchmark.group = "null count N=1M (10% None)"
        result = benchmark(
            lambda: Query(data_xl).filter(lambda x: x is not None and x > 0).count()
        )
        assert result == sum(1 for x in data_xl if x is not None and x > 0)
