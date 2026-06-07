"""Benchmarks for spec-077 convenience methods.

Groups:
  "compact N=1M (10% None)"     — drop None vs listcomp
  "filter_map N=1M"             — filter+map one-pass vs two-pass Python
  "median N=1M"                 — median vs statistics.median / numpy
  "product N=1M"                — product vs math.prod / functools.reduce
  "min_by N=1M"                 — min_by(key) vs min(key=) builtin
  "max_by N=1M"                 — max_by(key) vs max(key=) builtin
  "unzip N=1M"                  — unzip vs zip(*pairs) pattern
"""

import math
import statistics
import functools
import operator
import pytest

from models import SIZES, nullable_float_list

try:
    from zpyflow import Query, col
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")

N = SIZES["xl"]  # 1M


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def floats_with_none():
    """1M floats, ~10% None."""
    return nullable_float_list(N, null_rate=0.10)

@pytest.fixture(scope="session")
def plain_floats():
    return [float(i % 1000) for i in range(N)]

@pytest.fixture(scope="session")
def pairs():
    """1M (int, float) tuples for unzip."""
    return [(i, float(i % 1000)) for i in range(N)]

@pytest.fixture(scope="session")
def dicts_1m():
    """1M dicts with 'v' field for min_by/max_by."""
    return [{"v": float(i % 1000), "id": i} for i in range(N)]


# ---------------------------------------------------------------------------
# compact — drop None
# ---------------------------------------------------------------------------

class TestCompact:
    def test_python_listcomp(self, benchmark, floats_with_none):
        benchmark.group = "compact N=1M (10% None)"
        result = benchmark(lambda: [x for x in floats_with_none if x is not None])
        assert len(result) > 0

    def test_python_filter(self, benchmark, floats_with_none):
        benchmark.group = "compact N=1M (10% None)"
        result = benchmark(lambda: list(filter(None.__ne__, floats_with_none)))
        assert len(result) > 0

    def test_zpyflow(self, benchmark, floats_with_none):
        benchmark.group = "compact N=1M (10% None)"
        result = benchmark(lambda: Query(floats_with_none).compact().to_list())
        assert len(result) > 0


# ---------------------------------------------------------------------------
# filter_map — filter + transform in one pass
# ---------------------------------------------------------------------------

class TestFilterMap:
    def _fn(self, x):
        return x * 2 if x > 500 else None

    def test_python_two_pass(self, benchmark, plain_floats):
        benchmark.group = "filter_map N=1M"
        fn = self._fn
        result = benchmark(lambda: [fn(x) for x in plain_floats if x > 500])
        assert len(result) > 0

    def test_python_walrus(self, benchmark, plain_floats):
        benchmark.group = "filter_map N=1M"
        result = benchmark(lambda: [y for x in plain_floats if (y := x * 2 if x > 500 else None) is not None])
        assert len(result) > 0

    def test_zpyflow(self, benchmark, plain_floats):
        benchmark.group = "filter_map N=1M"
        result = benchmark(lambda: Query(plain_floats).filter_map(lambda x: x * 2 if x > 500 else None).to_list())
        assert len(result) > 0


# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------

class TestMedian:
    def test_python_statistics(self, benchmark, plain_floats):
        benchmark.group = "median N=1M"
        result = benchmark(lambda: statistics.median(plain_floats))
        assert result is not None

    def test_numpy(self, benchmark, plain_floats):
        benchmark.group = "median N=1M"
        import numpy as np
        arr = [plain_floats]  # avoid re-creating array each iteration
        result = benchmark(lambda: float(np.median(plain_floats)))
        assert result is not None

    def test_zpyflow(self, benchmark, plain_floats):
        benchmark.group = "median N=1M"
        result = benchmark(lambda: Query(plain_floats).median())
        assert result is not None


# ---------------------------------------------------------------------------
# product
# ---------------------------------------------------------------------------

class TestProduct:
    def test_python_math_prod(self, benchmark):
        benchmark.group = "product N=100K"
        data = [1.0 + (i % 10) * 0.001 for i in range(100_000)]  # avoid overflow
        result = benchmark(lambda: math.prod(data))
        assert result > 0

    def test_python_reduce(self, benchmark):
        benchmark.group = "product N=100K"
        data = [1.0 + (i % 10) * 0.001 for i in range(100_000)]
        result = benchmark(lambda: functools.reduce(operator.mul, data, 1.0))
        assert result > 0

    def test_zpyflow(self, benchmark):
        benchmark.group = "product N=100K"
        data = [1.0 + (i % 10) * 0.001 for i in range(100_000)]
        result = benchmark(lambda: Query(data).product())
        assert result > 0


# ---------------------------------------------------------------------------
# min_by / max_by — compare with Python builtins
# ---------------------------------------------------------------------------

class TestMinBy:
    def test_python_builtin(self, benchmark, dicts_1m):
        benchmark.group = "min_by N=1M"
        result = benchmark(lambda: min(dicts_1m, key=lambda d: d["v"]))
        assert result is not None

    def test_zpyflow(self, benchmark, dicts_1m):
        benchmark.group = "min_by N=1M"
        result = benchmark(lambda: Query(dicts_1m).min_by(lambda d: d["v"]))
        assert result is not None


class TestMaxBy:
    def test_python_builtin(self, benchmark, dicts_1m):
        benchmark.group = "max_by N=1M"
        result = benchmark(lambda: max(dicts_1m, key=lambda d: d["v"]))
        assert result is not None

    def test_zpyflow(self, benchmark, dicts_1m):
        benchmark.group = "max_by N=1M"
        result = benchmark(lambda: Query(dicts_1m).max_by(lambda d: d["v"]))
        assert result is not None


# ---------------------------------------------------------------------------
# unzip — split (a, b) tuples
# ---------------------------------------------------------------------------

class TestUnzip:
    def test_python_zip_star(self, benchmark, pairs):
        benchmark.group = "unzip N=1M"
        result = benchmark(lambda: tuple(map(list, zip(*pairs))))
        assert len(result) == 2

    def test_python_listcomp(self, benchmark, pairs):
        benchmark.group = "unzip N=1M"
        result = benchmark(lambda: ([a for a, _ in pairs], [b for _, b in pairs]))
        assert len(result[0]) == N

    def test_zpyflow(self, benchmark, pairs):
        benchmark.group = "unzip N=1M"
        result = benchmark(lambda: Query(pairs).unzip())
        assert len(result[0]) == N
