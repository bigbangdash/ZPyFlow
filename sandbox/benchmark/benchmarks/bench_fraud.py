# bench_fraud.py — fraud / risk scoring pipeline
# Business use case: financial transaction risk assessment
#
# Data: log-normal risk scores (skewed — most transactions are low risk,
#       a long tail is high risk). Mirrors bench_vector_search but uses
#       skewed_float_list (lognormal) instead of similarity_scores (beta).
#
# Key patterns:
#   filter + count  → flag count for reporting
#   filter + take   → review queue (early stopping)
#   filter + sum    → total exposure above threshold
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_fraud.py -v --benchmark-columns=mean,ops

import pytest
import numpy as np

from models import skewed_float_list, SIZES

try:
    from zpyflow import Query, col
    from zpyflow import from_numpy as zpyflow_from_numpy
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")

# skewed_float_list is lognormal(mean=3, sigma=1.5): median ~20, mean ~90
# Thresholds chosen so ~30% / ~10% of scores are flagged (realistic fraud rates)
THRESHOLD_MODERATE = 50.0    # ~30% flagged
THRESHOLD_STRICT   = 200.0   # ~10% flagged
REVIEW_CAP         = 500     # max cases per review batch


@pytest.fixture(scope="session")
def scores_xl():
    return skewed_float_list(SIZES["xl"])   # 1M log-normal scores


@pytest.fixture(scope="session")
def arr_xl(scores_xl):
    return np.array(scores_xl)


# ---------------------------------------------------------------------------
# Flag count — how many transactions exceed the risk threshold?
# ---------------------------------------------------------------------------

class TestFlagCount:
    """filter + count: stays in Rust, no Python list created."""

    def test_python_sum(self, benchmark, scores_xl):
        benchmark.group = "fraud flag count N=1M"
        t = THRESHOLD_MODERATE
        result = benchmark(lambda: sum(1 for s in scores_xl if s > t))
        assert result >= 0

    def test_numpy(self, benchmark, arr_xl):
        benchmark.group = "fraud flag count N=1M"
        t = THRESHOLD_MODERATE
        result = benchmark(lambda: int((arr_xl > t).sum()))
        assert result >= 0

    def test_zpyflow_dsl(self, benchmark, scores_xl):
        """Input: Python list[float] — bottleneck is PyFloat_AsDouble × N."""
        benchmark.group = "fraud flag count N=1M"
        result = benchmark(lambda: Query(scores_xl).filter(col > THRESHOLD_MODERATE).count())
        assert result >= 0

    def test_zpyflow_from_numpy(self, benchmark, arr_xl):
        """Input: numpy array — buffer protocol path, competitive with numpy."""
        benchmark.group = "fraud flag count N=1M"
        result = benchmark(lambda: zpyflow_from_numpy(arr_xl).filter(col > THRESHOLD_MODERATE).count())
        assert result >= 0


# ---------------------------------------------------------------------------
# Review queue — top-N cases for human review (early stopping)
# ---------------------------------------------------------------------------

class TestReviewQueue:
    """filter + take: stop once the queue is full — never scan the whole list."""

    def test_python_islice(self, benchmark, scores_xl):
        benchmark.group = "fraud review queue N=1M"
        import itertools
        t, k = THRESHOLD_MODERATE, REVIEW_CAP
        result = benchmark(lambda: list(itertools.islice(
            (s for s in scores_xl if s > t), k
        )))
        assert len(result) <= REVIEW_CAP

    def test_numpy(self, benchmark, arr_xl):
        """numpy: full scan + boolean index, then slice — no early stopping."""
        benchmark.group = "fraud review queue N=1M"
        t, k = THRESHOLD_MODERATE, REVIEW_CAP
        result = benchmark(lambda: arr_xl[arr_xl > t][:k].tolist())
        assert len(result) <= REVIEW_CAP

    def test_zpyflow_dsl(self, benchmark, scores_xl):
        benchmark.group = "fraud review queue N=1M"
        result = benchmark(lambda: (
            Query(scores_xl).filter(col > THRESHOLD_MODERATE).take(REVIEW_CAP).to_list()
        ))
        assert len(result) <= REVIEW_CAP


# ---------------------------------------------------------------------------
# Exposure sum — total risk amount above threshold
# ---------------------------------------------------------------------------

class TestExposureSum:
    """filter + sum: aggregate monetary exposure, no list materialized."""

    def test_python_sum(self, benchmark, scores_xl):
        benchmark.group = "fraud exposure sum N=1M"
        t = THRESHOLD_STRICT
        result = benchmark(lambda: sum(s for s in scores_xl if s > t))
        assert result > 0

    def test_numpy(self, benchmark, arr_xl):
        benchmark.group = "fraud exposure sum N=1M"
        t = THRESHOLD_STRICT
        result = benchmark(lambda: float(arr_xl[arr_xl > t].sum()))
        assert result > 0

    def test_zpyflow_dsl(self, benchmark, scores_xl):
        benchmark.group = "fraud exposure sum N=1M"
        result = benchmark(lambda: Query(scores_xl).filter(col > THRESHOLD_STRICT).sum())
        assert result > 0


# ---------------------------------------------------------------------------
# Threshold sensitivity — how does performance vary with pass rate?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("threshold", [20.0, 50.0, 100.0, 200.0, 500.0])
def test_threshold_sweep_zpyflow(benchmark, threshold):
    """ZPyFlow count at different thresholds — measures SIMD branch prediction."""
    scores = skewed_float_list(SIZES["xl"])
    benchmark.group = "fraud threshold sweep N=1M"
    benchmark.name  = f"zpyflow t={threshold:.0f}"
    result = benchmark(lambda: Query(scores).filter(col > threshold).count())
    assert result >= 0


@pytest.mark.parametrize("threshold", [20.0, 50.0, 100.0, 200.0, 500.0])
def test_threshold_sweep_python(benchmark, threshold):
    scores = skewed_float_list(SIZES["xl"])
    benchmark.group = "fraud threshold sweep N=1M"
    benchmark.name  = f"python t={threshold:.0f}"
    result = benchmark(lambda: sum(1 for s in scores if s > threshold))
    assert result >= 0


# ---------------------------------------------------------------------------
# Bool flag count — count True values in a boolean column
# Business: count transactions flagged by a pre-computed risk model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def bool_arr_xl(arr_xl):
    """Pre-computed bool flags: True where risk score > THRESHOLD_MODERATE (~30%)."""
    return arr_xl > THRESHOLD_MODERATE


@pytest.fixture(scope="session")
def bool_list_xl(bool_arr_xl):
    return bool_arr_xl.tolist()


class TestBoolFlagCount:
    """Count True values in a boolean column — 3-way comparison."""

    def test_python_sum(self, benchmark, bool_list_xl):
        """Python native: sum over Python list[bool]."""
        benchmark.group = "bool flag count N=1M"
        result = benchmark(lambda: sum(bool_list_xl))
        assert result >= 0

    def test_numpy(self, benchmark, bool_arr_xl):
        """NumPy: direct SIMD on contiguous bool array."""
        benchmark.group = "bool flag count N=1M"
        result = benchmark(lambda: int(bool_arr_xl.sum()))
        assert result >= 0

    def test_zpyflow_from_numpy(self, benchmark, bool_arr_xl):
        """ZPyFlow from_numpy: bool → compact uint8 buffer, fused count."""
        benchmark.group = "bool flag count N=1M"
        result = benchmark(lambda: zpyflow_from_numpy(bool_arr_xl).filter(col > 0).count())
        assert result >= 0
