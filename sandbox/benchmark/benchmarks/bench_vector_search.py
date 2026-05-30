# bench_vector_search.py — vector search score post-filtering
# Use case: after ANN retrieval, filter 1M cosine similarity scores by threshold
# and take top-K candidates.
#
# similarity_scores uses Beta(2,5) — most values cluster near 0 (~15% pass 0.5).
# This skewed distribution makes early-stopping with take(K) meaningful.
#
# Run:
#   pytest sandbox/benchmark/benchmarks/bench_vector_search.py -v --benchmark-columns=mean,ops

import pytest
import numpy as np
import itertools

from models import similarity_scores, SIZES

try:
    from zpyflow import Query, col
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")

THRESHOLD = 0.5   # typical cosine sim cutoff — ~15% pass rate with Beta(2,5)
TOP_K     = 1_000


@pytest.fixture(scope="session")
def scores_xl():
    return similarity_scores(SIZES["xl"])  # 1M


@pytest.fixture(scope="session")
def arr_xl(scores_xl):
    return np.array(scores_xl)


# ---------------------------------------------------------------------------
# Top-K: filter then take K results
# ---------------------------------------------------------------------------

class TestTopK:
    """filter(col > threshold).take(K) — early stopping is the key advantage."""

    def test_python_listcomp(self, benchmark, scores_xl):
        benchmark.group = "vector search top-K N=1M"
        k = TOP_K
        result = benchmark(lambda: [s for s in scores_xl if s > THRESHOLD][:k])
        assert len(result) <= k

    def test_python_islice(self, benchmark, scores_xl):
        benchmark.group = "vector search top-K N=1M"
        k = TOP_K
        result = benchmark(lambda: list(itertools.islice(
            (s for s in scores_xl if s > THRESHOLD), k
        )))
        assert len(result) <= k

    def test_numpy(self, benchmark, arr_xl):
        """numpy: full scan + boolean index, then slice — no early stopping."""
        benchmark.group = "vector search top-K N=1M"
        k = TOP_K
        result = benchmark(lambda: arr_xl[arr_xl > THRESHOLD][:k].tolist())
        assert len(result) <= k

    def test_zpyflow_dsl(self, benchmark, scores_xl):
        benchmark.group = "vector search top-K N=1M"
        result = benchmark(lambda: Query(scores_xl).filter(col > THRESHOLD).take(TOP_K).to_list())
        assert len(result) <= TOP_K

    def test_zpyflow_dsl_parallel(self, benchmark, scores_xl):
        benchmark.group = "vector search top-K N=1M"
        result = benchmark(lambda: (
            Query(scores_xl).filter(col > THRESHOLD).parallel().take(TOP_K).to_list()
        ))
        assert len(result) > 0


