# bench_ml_feature.py — ML feature preprocessing pipeline
# Use case: remove outliers then normalize feature values before model training.
#
# Pipeline: filter(col.between(-CLIP, CLIP)) → map(col * SCALE)
# This exercises col.between() (FilterBetween) which is untested by other suites.
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_ml_feature.py -v --benchmark-columns=mean,ops

import pytest
import numpy as np
import itertools

from models import float_list, float_array, SIZES

try:
    from zpyflow import Query, col
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")

CLIP  = 90.0          # keep |x| < 90 — removes ~10% of uniform [-100, 100]
SCALE = 1.0 / 90.0   # normalize filtered values to [-1, 1]
SAMPLE_N = 50_000     # sub-sample after filtering


@pytest.fixture(scope="session")
def data_xl():
    return float_list(SIZES["xl"])   # 1M uniform [-100, 100]


@pytest.fixture(scope="session")
def arr_xl(data_xl):
    return np.array(data_xl)


@pytest.fixture(scope="session")
def data_l():
    return float_list(SIZES["l"])    # 100K


# ---------------------------------------------------------------------------
# N=1M: filter outliers + scale (full output)
# ---------------------------------------------------------------------------

class TestPreprocessXL:
    """Remove outliers, scale to [-1, 1], return all surviving values."""

    def test_python_listcomp(self, benchmark, data_xl):
        benchmark.group = "feature preprocess N=1M"
        c, s = CLIP, SCALE
        result = benchmark(lambda: [x * s for x in data_xl if -c < x < c])
        assert len(result) > 0

    def test_numpy(self, benchmark, arr_xl):
        benchmark.group = "feature preprocess N=1M"
        c, s = CLIP, SCALE
        result = benchmark(lambda: (arr_xl[np.abs(arr_xl) < c] * s).tolist())
        assert len(result) > 0

    def test_zpyflow_dsl(self, benchmark, data_xl):
        benchmark.group = "feature preprocess N=1M"
        result = benchmark(lambda: (
            Query(data_xl)
                .filter(col.between(-CLIP, CLIP))
                .map(col * SCALE)
                .to_list()
        ))
        assert len(result) > 0

    def test_zpyflow_dsl_parallel(self, benchmark, data_xl):
        benchmark.group = "feature preprocess N=1M"
        result = benchmark(lambda: (
            Query(data_xl)
                .filter(col.between(-CLIP, CLIP))
                .map(col * SCALE)
                .parallel()
                .to_list()
        ))
        assert len(result) > 0


# ---------------------------------------------------------------------------
# N=1M: filter + scale + take (sub-sampled output)
# ---------------------------------------------------------------------------

class TestPreprocessSampleXL:
    """Same pipeline but stop after collecting SAMPLE_N results."""

    def test_python_islice(self, benchmark, data_xl):
        benchmark.group = "feature preprocess+take N=1M"
        c, s, k = CLIP, SCALE, SAMPLE_N
        result = benchmark(lambda: list(itertools.islice(
            (x * s for x in data_xl if -c < x < c), k
        )))
        assert len(result) <= SAMPLE_N

    def test_numpy(self, benchmark, arr_xl):
        """numpy: full scan then slice — no early stopping."""
        benchmark.group = "feature preprocess+take N=1M"
        c, s, k = CLIP, SCALE, SAMPLE_N
        result = benchmark(lambda: (arr_xl[np.abs(arr_xl) < c] * s)[:k].tolist())
        assert len(result) <= SAMPLE_N

    def test_zpyflow_dsl(self, benchmark, data_xl):
        benchmark.group = "feature preprocess+take N=1M"
        result = benchmark(lambda: (
            Query(data_xl)
                .filter(col.between(-CLIP, CLIP))
                .map(col * SCALE)
                .take(SAMPLE_N)
                .to_list()
        ))
        assert len(result) <= SAMPLE_N


