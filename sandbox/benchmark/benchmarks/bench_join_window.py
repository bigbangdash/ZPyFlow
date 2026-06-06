"""Benchmarks for spec 084 T7: hash join (N=100K rows) and rolling aggregates (N=1M floats).

Groups:
  "hash join inner N=100K"    — inner join, string key
  "rolling mean N=1M"         — rolling mean window=50
"""
import pytest


@pytest.fixture(scope="session")
def orders_100k():
    return [{"order_id": i, "customer_id": str(i % 1000), "amount": float(i)} for i in range(100_000)]


@pytest.fixture(scope="session")
def customers_1k():
    return [{"customer_id": str(i), "name": f"cust_{i}"} for i in range(1000)]


@pytest.fixture(scope="session")
def floats_1m():
    import math
    return [math.sin(i * 0.001) for i in range(1_000_000)]


# ---------------------------------------------------------------------------
# Hash join: Python native vs ZPyFlow
# ---------------------------------------------------------------------------

class TestHashJoinInner:
    """inner join on string 'customer_id' — N=100K orders × 1K customers."""

    def test_python_native(self, benchmark, orders_100k, customers_1k):
        benchmark.group = "hash join inner N=100K"
        from collections import defaultdict

        def run():
            index = defaultdict(list)
            for c in customers_1k:
                index[c["customer_id"]].append(c)
            return [{**o, **c} for o in orders_100k for c in index.get(o["customer_id"], [])]

        result = benchmark(run)
        assert len(result) == 100_000

    def test_zpyflow_join(self, benchmark, orders_100k, customers_1k):
        benchmark.group = "hash join inner N=100K"
        from zpyflow import Query

        def run():
            return Query(orders_100k).join(Query(customers_1k), on="customer_id").to_list()

        result = benchmark(run)
        assert len(result) == 100_000


# ---------------------------------------------------------------------------
# Rolling mean: Python native vs ZPyFlow
# ---------------------------------------------------------------------------

class TestRollingMean:
    """rolling mean window=50 over N=1M floats."""

    def test_python_native(self, benchmark, floats_1m):
        benchmark.group = "rolling mean N=1M"

        def run():
            w = 50
            out = []
            s = sum(floats_1m[:w])
            out.append(s / w)
            for i in range(w, len(floats_1m)):
                s += floats_1m[i] - floats_1m[i - w]
                out.append(s / w)
            return out

        result = benchmark(run)
        assert len(result) == len(floats_1m) - 50 + 1

    def test_zpyflow_rolling_mean(self, benchmark, floats_1m):
        benchmark.group = "rolling mean N=1M"
        from zpyflow import Query

        def run():
            return Query(floats_1m).rolling_mean(50).to_list()

        result = benchmark(run)
        assert len(result) == len(floats_1m) - 50 + 1


# ---------------------------------------------------------------------------
# Rolling sum: Python native vs ZPyFlow
# ---------------------------------------------------------------------------

class TestRollingSum:
    """rolling sum window=100 over N=1M floats."""

    def test_python_native(self, benchmark, floats_1m):
        benchmark.group = "rolling sum N=1M"

        def run():
            w = 100
            out = []
            s = sum(floats_1m[:w])
            out.append(s)
            for i in range(w, len(floats_1m)):
                s += floats_1m[i] - floats_1m[i - w]
                out.append(s)
            return out

        result = benchmark(run)
        assert len(result) == len(floats_1m) - 100 + 1

    def test_zpyflow_rolling_sum(self, benchmark, floats_1m):
        benchmark.group = "rolling sum N=1M"
        from zpyflow import Query

        def run():
            return Query(floats_1m).rolling_sum(100).to_list()

        result = benchmark(run)
        assert len(result) == len(floats_1m) - 100 + 1
