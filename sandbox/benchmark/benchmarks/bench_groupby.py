# bench_groupby.py — GroupBy performance + object pagination
# Business use cases:
#   Order status grouping, user segment cohorts, content category analytics
#   Content feed pagination (skip + take on object path)
#
# Compares ZPyFlow GroupBy against Python Counter / defaultdict.
# Note: GroupBy is a pure-Python layer — speed ≈ Python.
# Value is the chainable API and no intermediate list from filter → group.
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_groupby.py -v --benchmark-columns=mean,ops
#
# ---------------------------------------------------------------------------
# Data shape (log_dicts)
#
#   Each record is a Python dict with 7 fields:
#
#     {
#       "ts":         int    — UNIX timestamp (sequential)
#       "level":      str    — "INFO" 70% / "WARN" 20% / "ERROR" 10%
#       "status":     int    — weighted choice of [200,200,200,201,400,429,500]
#       "path":       str    — one of 5 API paths
#       "latency_ms": float  — log-normal exp(gauss(3.5, 1.2)), median ≈ 33 ms
#       "user_id":    int|None — 2% are None
#       "bytes_sent": int    — uniform [200, 50000]
#     }
#
# Filter selectivity:
#   status >= 500   ≈ 14% pass    (groupby / filter tests)
#   status == 200   ≈ 43% pass    (pagination tests)
#
# GroupBy cardinality:
#   group by status  → 7 distinct values
#   group by path    → 5 distinct values
# ---------------------------------------------------------------------------

import pytest
import itertools
from collections import Counter, defaultdict

from models import log_dicts, products, SIZES

try:
    from zpyflow import Query, GroupBy, agg_count, agg_sum, field
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")

PAGE_SIZE = 20


@pytest.fixture(scope="session")
def logs_l():
    return log_dicts(SIZES["l"])    # 100K


@pytest.fixture(scope="session")
def logs_xl():
    return log_dicts(SIZES["xl"])   # 1M


@pytest.fixture(scope="session")
def prods_l():
    return [vars(p) for p in products(SIZES["l"])]  # 100K product dicts


@pytest.fixture(scope="session")
def rust_logs_l(logs_l):
    """Pre-converted RustObj for N=100K — GIL-free field ops after this."""
    return Query(logs_l).preload()


@pytest.fixture(scope="session")
def rust_logs_xl(logs_xl):
    """Pre-converted RustObj for N=1M."""
    return Query(logs_xl).preload()


@pytest.fixture(scope="session")
def rust_prods_l(prods_l):
    """Pre-converted RustObj for product dicts N=100K."""
    return Query(prods_l).preload()


# ---------------------------------------------------------------------------
# count_per_group — group by key and count
# Business: order count by status, user count by tier, article count by category
# ---------------------------------------------------------------------------

class TestCountPerGroup:
    """GroupBy + count_per_group vs Python Counter."""

    def test_python_counter_l(self, benchmark, logs_l):
        benchmark.group = "groupby count N=100K"
        result = benchmark(lambda: Counter(l["status"] for l in logs_l))
        assert len(result) > 0

    def test_python_defaultdict_l(self, benchmark, logs_l):
        benchmark.group = "groupby count N=100K"
        def run():
            d = defaultdict(int)
            for l in logs_l:
                d[l["status"]] += 1
            return d
        result = benchmark(run)
        assert len(result) > 0

    def test_zpyflow_l(self, benchmark, logs_l):
        benchmark.group = "groupby count N=100K"
        result = benchmark(lambda: (
            Query(logs_l).group_by(lambda l: l["status"]).count_per_group()
        ))
        assert len(result) > 0

    def test_python_counter_xl(self, benchmark, logs_xl):
        benchmark.group = "groupby count N=1M"
        result = benchmark(lambda: Counter(l["status"] for l in logs_xl))
        assert len(result) > 0

    def test_zpyflow_xl(self, benchmark, logs_xl):
        benchmark.group = "groupby count N=1M"
        result = benchmark(lambda: (
            Query(logs_xl).group_by(lambda l: l["status"]).count_per_group()
        ))
        assert len(result) > 0


# ---------------------------------------------------------------------------
# agg — group by key, compute count + sum per group
# Business: revenue by category, latency by path, conversion by segment
# ---------------------------------------------------------------------------

class TestAgg:
    """GroupBy + agg(count, sum) vs Python defaultdict."""

    def test_python_defaultdict_l(self, benchmark, prods_l):
        """Manual defaultdict: count + revenue per category."""
        benchmark.group = "groupby agg N=100K"
        def run():
            d = defaultdict(lambda: {"count": 0, "revenue": 0.0})
            for p in prods_l:
                d[p["category"]]["count"]   += 1
                d[p["category"]]["revenue"] += p["price"]
            return d
        result = benchmark(run)
        assert len(result) > 0

    def test_zpyflow_l(self, benchmark, prods_l):
        """ZPyFlow GroupBy.agg — generic 3-pass lambda API (not the fast path)."""
        benchmark.group = "groupby agg N=100K (generic 3-pass)"
        result = benchmark(lambda: (
            Query(prods_l)
                .group_by(lambda p: p["category"])
                .agg(
                    count   = lambda g: g.count(),
                    revenue = lambda g: g.map(lambda p: p["price"]).sum(),
                )
        ))
        assert len(result) > 0

    def test_zpyflow_fast_l(self, benchmark, prods_l):
        """ZPyFlow group_agg — single-pass Rust kernel (recommended for production)."""
        benchmark.group = "groupby agg N=100K (group_agg optimized)"
        result = benchmark(lambda: (
            Query(prods_l).group_agg(
                lambda p: p["category"],
                count   = agg_count(),
                revenue = agg_sum(lambda p: p["price"]),
            )
        ))
        assert len(result) > 0

    def test_zpyflow_count_lambda_key_l(self, benchmark, prods_l):
        """ZPyFlow group_agg count — lambda key extraction."""
        benchmark.group = "groupby count key N=100K"
        result = benchmark(lambda: (
            Query(prods_l).group_agg(
                lambda p: p["category"],
                count=agg_count(),
            )
        ))
        assert len(result) > 0

    def test_zpyflow_count_field_key_warm_l(self, benchmark, rust_prods_l):
        """ZPyFlow group_agg count — field() key, warm (preload outside benchmark)."""
        benchmark.group = "groupby count key N=100K"
        result = benchmark(lambda: rust_prods_l.group_agg(
            field("category"),
            count=agg_count(),
        ))
        assert len(result) > 0

    def test_zpyflow_count_field_key_cold_l(self, benchmark, prods_l):
        """ZPyFlow group_agg count — field() key, cold (preload + groupby measured)."""
        benchmark.group = "groupby count key N=100K (cold)"
        result = benchmark(lambda: (
            Query(prods_l).preload().group_agg(
                field("category"),
                count=agg_count(),
            )
        ))
        assert len(result) > 0


# ---------------------------------------------------------------------------
# filter → group — filter first, then group the survivors
# Business: group only error-status orders, group only active users
# ---------------------------------------------------------------------------

class TestFilterThenGroup:
    """filter(lambda) + group_by: one pipeline, no intermediate list."""

    def test_python_two_pass_l(self, benchmark, logs_l):
        """Python: filter to list, then Counter — 2 passes."""
        benchmark.group = "filter+groupby N=100K"
        def run():
            errors = [l for l in logs_l if l["status"] >= 500]
            return Counter(l["path"] for l in errors)
        result = benchmark(run)
        assert len(result) >= 0

    def test_python_one_pass_l(self, benchmark, logs_l):
        """Python: single-pass generator into Counter."""
        benchmark.group = "filter+groupby N=100K"
        result = benchmark(lambda: Counter(
            l["path"] for l in logs_l if l["status"] >= 500
        ))
        assert len(result) >= 0

    def test_zpyflow_l(self, benchmark, logs_l):
        """ZPyFlow: filter then group — no intermediate list.

        Note: GroupBy runs at CPython dispatch speed (same as Python Counter).
        ZPyFlow will be equal to or slightly slower than test_python_one_pass_l
        due to Query/GroupBy construction overhead.
        The advantage is API ergonomics and no intermediate filtered list,
        not raw throughput.
        """
        benchmark.group = "filter+groupby N=100K"
        result = benchmark(lambda: (
            Query(logs_l)
                .filter(lambda l: l["status"] >= 500)
                .group_by(lambda l: l["path"])
                .count_per_group()
        ))
        assert len(result) >= 0


# ---------------------------------------------------------------------------
# Pagination — skip + take on object path
# Business: content feed page, order history page, search results page
# ---------------------------------------------------------------------------

class TestPagination:
    """skip(offset) + take(page_size): paginated feed without building full list.

    cold: Query(data) constructed inside the lambda — includes dict→RustRow import.
    warm: pre-built RustObj — field DSL, GIL-free filter + skip/take.
    """

    @pytest.mark.parametrize("page", [0, 10, 50, 100])
    def test_python_islice(self, benchmark, logs_l, page):
        benchmark.group = "pagination N=100K"
        benchmark.extra_info["page"] = page
        offset = page * PAGE_SIZE
        result = benchmark(lambda: list(itertools.islice(
            (l for l in logs_l if l["status"] == 200),
            offset, offset + PAGE_SIZE,
        )))
        assert len(result) <= PAGE_SIZE

    @pytest.mark.parametrize("page", [0, 10, 50, 100])
    def test_zpyflow_cold(self, benchmark, logs_l, page):
        """Cold: Query construction + filter + skip/take every iteration."""
        benchmark.group = "pagination N=100K"
        benchmark.extra_info["page"] = page
        offset = page * PAGE_SIZE
        result = benchmark(lambda: (
            Query(logs_l)
                .filter(lambda l: l["status"] == 200)
                .skip(offset)
                .take(PAGE_SIZE)
                .to_list()
        ))
        assert len(result) <= PAGE_SIZE

    @pytest.mark.parametrize("page", [0, 10, 50, 100])
    def test_zpyflow_warm(self, benchmark, rust_logs_l, page):
        """Warm: pre-built RustObj, field DSL filter + skip/take, GIL-free."""
        benchmark.group = "pagination N=100K"
        benchmark.extra_info["page"] = page
        offset = page * PAGE_SIZE
        result = benchmark(lambda: (
            rust_logs_l
                .filter(field("status") == 200)
                .skip(offset)
                .take(PAGE_SIZE)
                .to_list()
        ))
        assert len(result) <= PAGE_SIZE
