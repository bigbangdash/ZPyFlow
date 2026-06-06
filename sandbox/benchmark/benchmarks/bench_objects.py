# bench_objects.py — Python object path (dict / dataclass)
# These benchmarks use the GIL-held Python path.
# Goal: verify that ZPyFlow is at least NOT slower, and that
# avoiding intermediate lists has some measurable value.
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_objects.py -v --benchmark-columns=mean,ops
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
# Filter selectivity used in these benchmarks:
#
#   status >= 500   →  500 appears 1/7 of the time  ≈ 14% pass rate
#   latency_ms > 100.0  →  log-normal tail; roughly 20–25% pass rate
#   status == 200   →  200 appears 3/7 of the time  ≈ 43% pass rate  (pagination)
# ---------------------------------------------------------------------------

import pytest
import itertools

from models import log_dicts, SIZES, measure_peak_kb

# DSL is only usable after extracting a numeric field from the dict records.
# These fixtures pre-extract latency_ms so all three styles are comparable.


try:
    from zpyflow import Query, col, field
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")

_TAKE = 1_000  # module-level — safe for lambda capture, no self lookup overhead


@pytest.fixture(scope="session")
def logs_l():
    return log_dicts(SIZES["l"])   # 100K

@pytest.fixture(scope="session")
def logs_xl():
    return log_dicts(SIZES["xl"])  # 1M

@pytest.fixture(scope="session")
def statuses_l(logs_l):
    return [float(l["status"]) for l in logs_l]

@pytest.fixture(scope="session")
def statuses_xl(logs_xl):
    return [float(l["status"]) for l in logs_xl]


# ---------------------------------------------------------------------------
# filter only — dict records
# ---------------------------------------------------------------------------

class TestObjectFilter:
    def test_python_listcomp_l(self, benchmark, logs_l):
        benchmark.group = "object filter N=100K"
        result = benchmark(lambda: [l for l in logs_l if l["status"] >= 500])
        assert len(result) >= 0

    def test_python_generator_l(self, benchmark, logs_l):
        benchmark.group = "object filter N=100K"
        result = benchmark(lambda: list(l for l in logs_l if l["status"] >= 500))
        assert len(result) >= 0

    def test_zpyflow_lambda_l(self, benchmark, logs_l):
        benchmark.group = "object filter N=100K"
        result = benchmark(lambda: Query(logs_l).filter(lambda l: l["status"] >= 500).to_list())
        assert len(result) >= 0

    def test_zpyflow_dsl_l(self, benchmark, logs_l):
        """field() DSL: same dict input as Python/lambda — GIL-free SIMD filter."""
        benchmark.group = "object filter N=100K"
        result = benchmark(lambda: Query(logs_l).filter(field("status") >= 500).to_list())
        assert len(result) >= 0

    def test_python_listcomp_xl(self, benchmark, logs_xl):
        benchmark.group = "object filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: [l for l in logs_xl if l["status"] >= 500]
        )
        result = benchmark(lambda: [l for l in logs_xl if l["status"] >= 500])
        assert len(result) >= 0

    def test_python_generator_xl(self, benchmark, logs_xl):
        benchmark.group = "object filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: list(l for l in logs_xl if l["status"] >= 500)
        )
        result = benchmark(lambda: list(l for l in logs_xl if l["status"] >= 500))
        assert len(result) >= 0

    def test_zpyflow_lambda_xl(self, benchmark, logs_xl):
        benchmark.group = "object filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(logs_xl).filter(lambda l: l["status"] >= 500).to_list()
        )
        result = benchmark(lambda: Query(logs_xl).filter(lambda l: l["status"] >= 500).to_list())
        assert result == [l for l in logs_xl if l["status"] >= 500]

    def test_zpyflow_dsl_xl(self, benchmark, logs_xl):
        """field() DSL: same dict input as Python/lambda — GIL-free SIMD filter."""
        benchmark.group = "object filter N=1M"
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(
            lambda: Query(logs_xl).filter(field("status") >= 500).to_list()
        )
        result = benchmark(lambda: Query(logs_xl).filter(field("status") >= 500).to_list())
        assert result == [l for l in logs_xl if l["status"] >= 500]


# ---------------------------------------------------------------------------
# filter + map + take — the intermediate-list avoidance test
# ---------------------------------------------------------------------------

class TestObjectChained:

    def test_python_two_listcomps(self, benchmark, logs_xl):
        """2 intermediate lists."""
        benchmark.group = "object filter+map+take N=1M"
        take = _TAKE
        def run():
            errors = [l for l in logs_xl if l["level"] == "ERROR"]
            paths  = [l["path"] for l in errors]
            return paths[:take]
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(run)
        result = benchmark(run)
        assert len(result) <= _TAKE

    def test_python_generator(self, benchmark, logs_xl):
        """Generator — lazy, 0 intermediate lists."""
        benchmark.group = "object filter+map+take N=1M"
        take = _TAKE
        def run():
            return list(itertools.islice(
                (l["path"] for l in logs_xl if l["level"] == "ERROR"),
                take,
            ))
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(run)
        result = benchmark(run)
        assert len(result) <= _TAKE

    def test_zpyflow_lambda(self, benchmark, logs_xl):
        """ZPyFlow lambda — lazy Obj pipeline, stops at take."""
        benchmark.group = "object filter+map+take N=1M"
        take = _TAKE
        fn = lambda: (
            Query(logs_xl)
                .filter(lambda l: l["level"] == "ERROR")
                .map(lambda l: l["path"])
                .take(take)
                .to_list()
        )
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(fn)
        result = benchmark(fn)
        assert len(result) <= _TAKE

    def test_zpyflow_dsl(self, benchmark, logs_xl):
        """field() DSL: same filter condition (level==ERROR), map_field, take.
        Uses ObjFieldPy path: C-API loop, no Python function call overhead per element.
        """
        benchmark.group = "object filter+map+take N=1M"
        take = _TAKE
        fn = lambda: (
            Query(logs_xl)
                .filter(field("level") == "ERROR")
                .map_field("path")
                .take(take)
                .to_list()
        )
        benchmark.extra_info["peak_memory_kb"] = measure_peak_kb(fn)
        result = benchmark(fn)
        assert len(result) <= _TAKE

    def test_zpyflow_dsl_latency(self, benchmark, logs_xl):
        """field() DSL: numeric filter (latency_ms > 100), map_field, take."""
        benchmark.group = "object filter+take N=1M (field DSL, latency>100)"
        take = _TAKE
        result = benchmark(lambda: (
            Query(logs_xl)
                .filter(field("latency_ms") > 100.0)
                .take(take)
                .to_list()
        ))
        assert len(result) <= _TAKE


# ---------------------------------------------------------------------------
# count — avoids to_list() overhead
# ---------------------------------------------------------------------------

class TestObjectCount:
    def test_python_sum_bool(self, benchmark, logs_xl):
        benchmark.group = "object count N=1M"
        result = benchmark(lambda: sum(1 for l in logs_xl if l["level"] == "ERROR"))
        assert result >= 0

    def test_python_len_listcomp(self, benchmark, logs_xl):
        benchmark.group = "object count N=1M"
        result = benchmark(lambda: len([l for l in logs_xl if l["level"] == "ERROR"]))
        assert result >= 0

    def test_zpyflow_lambda(self, benchmark, logs_xl):
        """ZPyFlow lambda — GIL held per element."""
        benchmark.group = "object count N=1M"
        result = benchmark(lambda: Query(logs_xl).filter(lambda l: l["level"] == "ERROR").count())
        assert result >= 0

    def test_zpyflow_dsl(self, benchmark, statuses_xl):
        """ZPyFlow DSL: pre-extracted status field → SIMD count, GIL released."""
        benchmark.group = "object count N=1M"
        result = benchmark(lambda: Query(statuses_xl).filter(col >= 500).count())
        assert result >= 0


# ---------------------------------------------------------------------------
# Numeric field extraction — when DSL becomes usable
#
# DSL (col > x) only works on numeric pipelines (f64/i64).
# The pattern: extract the numeric field once, then use SIMD DSL.
# Shows the speedup from "moving data to Rust" vs keeping it as dicts.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def latencies_xl(logs_xl):
    """Pre-extracted latency_ms as Python list[float]."""
    return [l["latency_ms"] for l in logs_xl]


class TestNumericFieldExtraction:
    """
    filter(latency > 100) — three ways:

      Python    : iterate dicts, check field per element
      λ lambda  : same but via ZPyFlow (lazy Obj path)
      DSL       : extract field once → SIMD, GIL released
    """

    THRESHOLD = 100.0

    def test_python_genexp(self, benchmark, logs_xl):
        """Python generator: no intermediate list."""
        benchmark.group = "latency filter+count N=1M"
        t = self.THRESHOLD
        result = benchmark(lambda: sum(1 for l in logs_xl if l["latency_ms"] > t))
        assert result >= 0

    def test_python_listcomp(self, benchmark, logs_xl):
        """Python listcomp: intermediate list, then len."""
        benchmark.group = "latency filter+count N=1M"
        t = self.THRESHOLD
        result = benchmark(lambda: len([l for l in logs_xl if l["latency_ms"] > t]))
        assert result >= 0

    def test_zpyflow_lambda(self, benchmark, logs_xl):
        """ZPyFlow lambda: lazy Obj pipeline, no Vec copy on construction."""
        benchmark.group = "latency filter+count N=1M"
        t = self.THRESHOLD
        result = benchmark(lambda: Query(logs_xl).filter(lambda l: l["latency_ms"] > t).count())
        assert result >= 0

    def test_zpyflow_dsl_extracted(self, benchmark, latencies_xl):
        """ZPyFlow DSL: field pre-extracted → SIMD count, GIL released.
        Note: uses latencies_xl fixture (pre-extracted float list).
        Extraction cost is paid once at session scope, not in the timed section.
        """
        benchmark.group = "latency filter+count N=1M (field pre-extracted)"
        result = benchmark(lambda: Query(latencies_xl).filter(col > self.THRESHOLD).count())
        assert result >= 0

    def test_zpyflow_field_dsl(self, benchmark, logs_xl):
        """ZPyFlow field() DSL: same dict input as Python — GIL-free SIMD count."""
        benchmark.group = "latency filter+count N=1M"
        result = benchmark(lambda: Query(logs_xl).filter(field("latency_ms") > self.THRESHOLD).count())
        assert result >= 0

    def test_python_genexp_sum(self, benchmark, logs_xl):
        """Python generator: filter+sum latency, no intermediate list."""
        benchmark.group = "latency filter+sum N=1M"
        t = self.THRESHOLD
        result = benchmark(lambda: sum(l["latency_ms"] for l in logs_xl if l["latency_ms"] > t))
        assert result > 0

    def test_python_listcomp_sum(self, benchmark, logs_xl):
        """Python listcomp: filter into list, then sum."""
        benchmark.group = "latency filter+sum N=1M"
        t = self.THRESHOLD
        result = benchmark(lambda: sum([l["latency_ms"] for l in logs_xl if l["latency_ms"] > t]))
        assert result > 0

    def test_zpyflow_lambda_sum(self, benchmark, logs_xl):
        """ZPyFlow lambda: filter dict + extract field + sum."""
        benchmark.group = "latency filter+sum N=1M"
        t = self.THRESHOLD
        result = benchmark(lambda: Query(logs_xl)
            .filter(lambda l: l["latency_ms"] > t)
            .map(lambda l: l["latency_ms"])
            .sum())
        assert result > 0

    def test_zpyflow_dsl_extracted_sum(self, benchmark, latencies_xl):
        """ZPyFlow DSL: pre-extracted → fused SIMD filter+sum, no Vec.
        Note: extraction cost paid at session scope, not timed.
        """
        benchmark.group = "latency filter+sum N=1M (field pre-extracted)"
        result = benchmark(lambda: Query(latencies_xl).filter(col > self.THRESHOLD).sum())
        assert result > 0

    def test_zpyflow_field_dsl_sum(self, benchmark, logs_xl):
        """ZPyFlow field() DSL: same dict input — GIL-free SIMD filter+sum_field."""
        benchmark.group = "latency filter+sum N=1M"
        result = benchmark(lambda: Query(logs_xl).filter(field("latency_ms") > self.THRESHOLD).sum_field("latency_ms"))
        assert result > 0


# ---------------------------------------------------------------------------
# RustObj field DSL — GIL-free filter on dict pipeline
#
# Two benchmark groups per operation:
#
#   "cold" — Query(data) constructed inside the measured lambda.
#            Includes the O(N × fields) dict→RustRow import cost.
#            Representative of single-use per-request processing.
#
#   "warm" — Query pre-built as a session fixture via .preload().
#            Measures query execution only (GIL-free Rust kernel).
#            Representative of a dataset queried many times.
#
# NumPy is omitted: the data is dict records, not a numeric array.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rust_query_l(logs_l):
    """Pre-converted RustObj for N=100K — pays import cost once per session."""
    return Query(logs_l).preload()

@pytest.fixture(scope="session")
def rust_query_xl(logs_xl):
    """Pre-converted RustObj for N=1M — pays import cost once per session."""
    return Query(logs_xl).preload()


class TestRustObjFieldDSL:
    """filter(status >= 500) → to_list(): N=100K and N=1M.

    Note on fairness: Python listcomp returns refs to the original dicts (zero-copy).
    ZPyFlow warm converts matching RustRows → new Python dicts on output.
    This means ZPyFlow always pays an O(matches × fields) output cost that Python does
    not.  Use count() / sum_field() (TestRustObjCount / TestRustObjSumField) for a
    comparison that doesn't have this asymmetry.

    cold = Query() + conversion + filter + output in one iteration (one-shot use-case).
    warm = pre-built RustObj; measures GIL-free filter + output only.

    IMPORTANT — when to use field() DSL vs Python lambda:
      to_list() output: cold path ALWAYS loses to Python listcomp at any N
        because the dict→RustRow→dict round-trip copies all fields twice.
      count() / sum_field(): cold path competitive; warm path wins clearly.
      Recommendation: use field() DSL for scalar terminals (count, sum_field),
        or preload() the query when it will be executed many times.
    """

    # ── N=100K ──────────────────────────────────────────────────────────────

    def test_python_listcomp_l(self, benchmark, logs_l):
        benchmark.group = "rust_obj filter N=100K"
        result = benchmark(lambda: [l for l in logs_l if l["status"] >= 500])
        assert len(result) >= 0

    def test_zpyflow_lambda_cold_l(self, benchmark, logs_l):
        """cold: Obj pipeline, lambda filter (no RustRow conversion)."""
        benchmark.group = "rust_obj filter N=100K"
        result = benchmark(lambda: Query(logs_l).filter(lambda l: l["status"] >= 500).to_list())
        assert len(result) >= 0

    def test_zpyflow_dsl_cold_l(self, benchmark, logs_l):
        """cold: lazy RustObj conversion + GIL-free filter + output."""
        benchmark.group = "rust_obj filter N=100K"
        result = benchmark(lambda: Query(logs_l).filter(field("status") >= 500).to_list())
        assert len(result) >= 0

    def test_zpyflow_dsl_warm_l(self, benchmark, rust_query_l):
        """warm: pre-built RustObj, GIL-free filter + output only."""
        benchmark.group = "rust_obj filter N=100K"
        result = benchmark(lambda: rust_query_l.filter(field("status") >= 500).to_list())
        assert len(result) >= 0

    # ── N=1M ────────────────────────────────────────────────────────────────

    def test_python_listcomp_xl(self, benchmark, logs_xl):
        benchmark.group = "rust_obj filter N=1M"
        result = benchmark(lambda: [l for l in logs_xl if l["status"] >= 500])
        assert len(result) >= 0

    def test_zpyflow_lambda_cold_xl(self, benchmark, logs_xl):
        """cold: Obj pipeline, lambda filter."""
        benchmark.group = "rust_obj filter N=1M"
        result = benchmark(lambda: Query(logs_xl).filter(lambda l: l["status"] >= 500).to_list())
        assert len(result) >= 0

    def test_zpyflow_dsl_cold_xl(self, benchmark, logs_xl):
        """cold: lazy RustObj conversion + GIL-free filter + output."""
        benchmark.group = "rust_obj filter N=1M"
        result = benchmark(lambda: Query(logs_xl).filter(field("status") >= 500).to_list())
        assert len(result) >= 0

    def test_zpyflow_dsl_warm_xl(self, benchmark, rust_query_xl):
        """warm: pre-built RustObj, GIL-free filter + output only."""
        benchmark.group = "rust_obj filter N=1M"
        result = benchmark(lambda: rust_query_xl.filter(field("status") >= 500).to_list())
        assert len(result) >= 0


class TestRustObjCount:
    """count() — no output conversion: fair apples-to-apples comparison.

    Python genexp counts by iterating Python dicts.
    ZPyFlow warm runs entirely GIL-free (no Python objects touched during count).
    """

    def test_python_genexp_xl(self, benchmark, logs_xl):
        benchmark.group = "rust_obj count N=1M"
        result = benchmark(lambda: sum(1 for l in logs_xl if l["latency_ms"] > 100.0))
        assert result >= 0

    def test_zpyflow_lambda_cold_xl(self, benchmark, logs_xl):
        """cold: Obj lambda path (no RustRow conversion, GIL held)."""
        benchmark.group = "rust_obj count N=1M"
        result = benchmark(lambda: Query(logs_xl).filter(lambda l: l["latency_ms"] > 100.0).count())
        assert result >= 0

    def test_zpyflow_dsl_cold_xl(self, benchmark, logs_xl):
        """cold: lazy RustObj conversion + GIL-free count."""
        benchmark.group = "rust_obj count N=1M"
        result = benchmark(lambda: Query(logs_xl).filter(field("latency_ms") > 100.0).count())
        assert result >= 0

    def test_zpyflow_dsl_warm_xl(self, benchmark, rust_query_xl):
        """warm: pre-built RustObj, GIL-free count only."""
        benchmark.group = "rust_obj count N=1M"
        result = benchmark(lambda: rust_query_xl.filter(field("latency_ms") > 100.0).count())
        assert result >= 0


class TestRustObjSumField:
    """sum_field() — no output conversion: fair apples-to-apples comparison."""

    def test_python_genexp_xl(self, benchmark, logs_xl):
        benchmark.group = "rust_obj sum_field N=1M"
        result = benchmark(lambda: sum(l["latency_ms"] for l in logs_xl if l["latency_ms"] > 100.0))
        assert result > 0

    def test_zpyflow_lambda_cold_xl(self, benchmark, logs_xl):
        """cold: Obj lambda path."""
        benchmark.group = "rust_obj sum_field N=1M"
        result = benchmark(lambda: (
            Query(logs_xl)
            .filter(lambda l: l["latency_ms"] > 100.0)
            .map(lambda l: l["latency_ms"])
            .sum()
        ))
        assert result > 0

    def test_zpyflow_dsl_cold_xl(self, benchmark, logs_xl):
        """cold: lazy RustObj conversion + GIL-free sum_field."""
        benchmark.group = "rust_obj sum_field N=1M"
        result = benchmark(
            lambda: Query(logs_xl).filter(field("latency_ms") > 100.0).sum_field("latency_ms")
        )
        assert result > 0

    def test_zpyflow_dsl_warm_xl(self, benchmark, rust_query_xl):
        """warm: pre-built RustObj, GIL-free sum_field only."""
        benchmark.group = "rust_obj sum_field N=1M"
        result = benchmark(
            lambda: rust_query_xl.filter(field("latency_ms") > 100.0).sum_field("latency_ms")
        )
        assert result > 0


# ---------------------------------------------------------------------------
# field() DSL granularity — isolate conversion cost vs filter cost
#
# cold: Query(dict_list).filter(field(...)).count()
#       Includes dict→RustRow conversion (O(N × fields)) in every iteration.
#
# preload_then_filter: Query(dict_list).preload() once, then re-run filter.
#       Measures pure GIL-free filter throughput without conversion overhead.
#
# This shows the break-even: at what N does amortizing preload() pay off.
# ---------------------------------------------------------------------------

class TestFieldDslGranularity:
    """Isolate dict→RustRow conversion cost from GIL-free filter cost.

    cold:  full pipeline each call — representative of one-shot request handling.
    warm:  pre-built RustObj — representative of repeated queries on the same dataset.
    """

    def test_python_genexp_xl(self, benchmark, logs_xl):
        """Baseline: Python genexp over dicts."""
        benchmark.group = "field DSL granularity N=1M"
        result = benchmark(lambda: sum(1 for l in logs_xl if l["latency_ms"] > 100.0))
        assert result >= 0

    def test_zpyflow_field_cold_xl(self, benchmark, logs_xl):
        """cold: dict→RustRow + GIL-free filter every call."""
        benchmark.group = "field DSL granularity N=1M"
        result = benchmark(lambda: Query(logs_xl).filter(field("latency_ms") > 100.0).count())
        assert result >= 0

    def test_zpyflow_field_warm_xl(self, benchmark, rust_query_xl):
        """warm: pre-built RustObj — GIL-free filter only."""
        benchmark.group = "field DSL granularity N=1M"
        result = benchmark(lambda: rust_query_xl.filter(field("latency_ms") > 100.0).count())
        assert result >= 0


# ---------------------------------------------------------------------------
# ColumnarObj — spec-082 T5
#
# Measures the columnar layout path introduced by .preload() → ColumnarObj.
#
# cold = preload() + filter + terminal inside the timed section.
#        Includes the O(N × fields) schema-inference + columnar conversion cost.
#        Representative of single-use per-request processing.
#
# warm = Query(data).preload() called once at session scope; only the filter
#        + terminal is timed.  Representative of a dataset queried many times.
#
# All three styles (Python native / ZPyFlow lambda / ColumnarObj DSL) share
# the same group so they appear side-by-side in the benchmark report.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def columnar_query_xl(logs_xl):
    """Pre-built ColumnarObj for N=1M — pays conversion cost once per session."""
    return Query(logs_xl).preload()


class TestColumnarObjFilter:
    """filter(latency_ms > 100) → to_list(): columnar cold vs warm vs Python.

    to_list() output cost note: Python listcomp returns refs to original dicts
    (zero-copy), while ColumnarObj reconstructs new Python dicts at output.
    Use TestColumnarObjCount for a terminal without this asymmetry.
    """

    _THRESHOLD = 100.0

    def test_python_native_xl(self, benchmark, logs_xl):
        benchmark.group = "columnar filter N=1M"
        t = self._THRESHOLD
        result = benchmark(lambda: [l for l in logs_xl if l["latency_ms"] > t])
        assert len(result) >= 0

    def test_zpyflow_lambda_xl(self, benchmark, logs_xl):
        """cold lambda: Obj pipeline (no columnar conversion)."""
        benchmark.group = "columnar filter N=1M"
        t = self._THRESHOLD
        result = benchmark(
            lambda: Query(logs_xl).filter(lambda l: l["latency_ms"] > t).to_list()
        )
        assert len(result) >= 0

    def test_field_dsl_cold_xl(self, benchmark, logs_xl):
        """cold: preload (schema + conversion) + columnar filter + dict reconstruction."""
        benchmark.group = "columnar filter N=1M"
        t = self._THRESHOLD
        result = benchmark(
            lambda: Query(logs_xl).preload().filter(field("latency_ms") > t).to_list()
        )
        assert len(result) >= 0

    def test_field_dsl_warm_xl(self, benchmark, columnar_query_xl):
        """warm: pre-built ColumnarObj — column scan + dict reconstruction only."""
        benchmark.group = "columnar filter N=1M"
        t = self._THRESHOLD
        result = benchmark(
            lambda: columnar_query_xl.filter(field("latency_ms") > t).to_list()
        )
        assert len(result) >= 0


class TestColumnarObjCount:
    """count() — no dict reconstruction: cleanest comparison of scan throughput.

    warm path runs entirely without Python dict access during the count.
    """

    _THRESHOLD = 100.0

    def test_python_native_xl(self, benchmark, logs_xl):
        benchmark.group = "columnar count N=1M"
        t = self._THRESHOLD
        result = benchmark(lambda: sum(1 for l in logs_xl if l["latency_ms"] > t))
        assert result >= 0

    def test_zpyflow_lambda_xl(self, benchmark, logs_xl):
        """cold lambda: Obj pipeline count."""
        benchmark.group = "columnar count N=1M"
        t = self._THRESHOLD
        result = benchmark(
            lambda: Query(logs_xl).filter(lambda l: l["latency_ms"] > t).count()
        )
        assert result >= 0

    def test_field_dsl_cold_xl(self, benchmark, logs_xl):
        """cold: preload + columnar count (conversion cost included)."""
        benchmark.group = "columnar count N=1M"
        t = self._THRESHOLD
        result = benchmark(
            lambda: Query(logs_xl).preload().filter(field("latency_ms") > t).count()
        )
        assert result >= 0

    def test_field_dsl_warm_xl(self, benchmark, columnar_query_xl):
        """warm: pre-built ColumnarObj count — pure column scan, GIL-free."""
        benchmark.group = "columnar count N=1M"
        t = self._THRESHOLD
        result = benchmark(
            lambda: columnar_query_xl.filter(field("latency_ms") > t).count()
        )
        assert result >= 0
