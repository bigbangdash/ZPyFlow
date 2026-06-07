"""Benchmarks for transform/sequence operations.

Groups:
  "sort N=1M"            — sort() vs sorted()
  "sort_by N=1M"         — sort_by(key) vs sorted(key=)
  "distinct N=1M"        — distinct() vs set-based dedup
  "dedupe N=1M"          — dedupe() vs adjacent dedup listcomp
  "flat_map N=100K"      — flat_map(fn) vs nested listcomp
  "chunk N=1M (size=100)" — chunk(n) vs manual slice loop
"""

import pytest

from models import SIZES

try:
    from zpyflow import Query, col
    HAS_ZPYFLOW = True
except ImportError:
    HAS_ZPYFLOW = False

pytestmark = pytest.mark.skipif(not HAS_ZPYFLOW, reason="zpyflow not built")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def floats_1m():
    import random
    rng = random.Random(42)
    return [rng.uniform(-1000, 1000) for _ in range(SIZES["xl"])]

@pytest.fixture(scope="session")
def ints_1m_low_cardinality():
    """1M ints, cardinality ~1K — realistic for distinct()."""
    return [i % 1000 for i in range(SIZES["xl"])]

@pytest.fixture(scope="session")
def ints_1m_runs():
    """1M ints with long runs of repeats — realistic for dedupe()."""
    # each value repeated 10 times: 0,0,...,1,1,...
    return [i // 10 for i in range(SIZES["xl"])]

@pytest.fixture(scope="session")
def nested_100k():
    """100K lists of 3 floats each — input for flat_map."""
    return [[float(j + i * 3) for j in range(3)] for i in range(SIZES["l"])]

@pytest.fixture(scope="session")
def floats_1m_dicts():
    """1M dicts with 'v' key for sort_by."""
    return [{"v": float(i % 10000), "id": i} for i in range(SIZES["xl"])]


# ---------------------------------------------------------------------------
# sort — numeric list
# ---------------------------------------------------------------------------

class TestSort:
    def test_python_sorted(self, benchmark, floats_1m):
        benchmark.group = "sort N=1M"
        result = benchmark(lambda: sorted(floats_1m))
        assert len(result) == SIZES["xl"]

    def test_python_list_sort(self, benchmark, floats_1m):
        benchmark.group = "sort N=1M"
        def run():
            lst = list(floats_1m)
            lst.sort()
            return lst
        result = benchmark(run)
        assert len(result) == SIZES["xl"]

    def test_zpyflow(self, benchmark, floats_1m):
        benchmark.group = "sort N=1M"
        result = benchmark(lambda: Query(floats_1m).sort().to_list())
        assert len(result) == SIZES["xl"]


# ---------------------------------------------------------------------------
# sort_by — sort dicts by field
# ---------------------------------------------------------------------------

class TestSortBy:
    def test_python_sorted_key(self, benchmark, floats_1m_dicts):
        benchmark.group = "sort_by N=1M"
        result = benchmark(lambda: sorted(floats_1m_dicts, key=lambda d: d["v"]))
        assert len(result) == SIZES["xl"]

    def test_zpyflow(self, benchmark, floats_1m_dicts):
        benchmark.group = "sort_by N=1M"
        result = benchmark(lambda: Query(floats_1m_dicts).sort_by(lambda d: d["v"]).to_list())
        assert len(result) == SIZES["xl"]


# ---------------------------------------------------------------------------
# distinct — remove duplicates (order not preserved in Python set)
# ---------------------------------------------------------------------------

class TestDistinct:
    def test_python_set_roundtrip(self, benchmark, ints_1m_low_cardinality):
        benchmark.group = "distinct N=1M"
        result = benchmark(lambda: list(set(ints_1m_low_cardinality)))
        assert len(result) == 1000

    def test_python_dict_keys(self, benchmark, ints_1m_low_cardinality):
        # dict.fromkeys preserves insertion order
        benchmark.group = "distinct N=1M"
        result = benchmark(lambda: list(dict.fromkeys(ints_1m_low_cardinality)))
        assert len(result) == 1000

    def test_zpyflow(self, benchmark, ints_1m_low_cardinality):
        benchmark.group = "distinct N=1M"
        result = benchmark(lambda: Query(ints_1m_low_cardinality).distinct().to_list())
        assert len(result) == 1000


# ---------------------------------------------------------------------------
# dedupe — collapse adjacent duplicates
# ---------------------------------------------------------------------------

class TestDedupe:
    def _python_dedupe(self, data):
        if not data:
            return []
        out = [data[0]]
        for x in data[1:]:
            if x != out[-1]:
                out.append(x)
        return out

    def test_python_listcomp(self, benchmark, ints_1m_runs):
        benchmark.group = "dedupe N=1M"
        fn = self._python_dedupe
        result = benchmark(lambda: fn(ints_1m_runs))
        assert len(result) == 100_000  # 1M // 10

    def test_python_itertools(self, benchmark, ints_1m_runs):
        benchmark.group = "dedupe N=1M"
        import itertools
        result = benchmark(lambda: [k for k, _ in itertools.groupby(ints_1m_runs)])
        assert len(result) == 100_000

    def test_zpyflow(self, benchmark, ints_1m_runs):
        benchmark.group = "dedupe N=1M"
        result = benchmark(lambda: Query(ints_1m_runs).dedupe().to_list())
        assert len(result) == 100_000


# ---------------------------------------------------------------------------
# flat_map — flatten one level
# ---------------------------------------------------------------------------

class TestFlatMap:
    def test_python_nested_listcomp(self, benchmark, nested_100k):
        benchmark.group = "flat_map N=100K×3"
        result = benchmark(lambda: [y for xs in nested_100k for y in xs])
        assert len(result) == SIZES["l"] * 3

    def test_python_chain_from_iterable(self, benchmark, nested_100k):
        benchmark.group = "flat_map N=100K×3"
        import itertools
        result = benchmark(lambda: list(itertools.chain.from_iterable(nested_100k)))
        assert len(result) == SIZES["l"] * 3

    def test_zpyflow_flat_map(self, benchmark, nested_100k):
        benchmark.group = "flat_map N=100K×3"
        result = benchmark(lambda: Query(nested_100k).flat_map(lambda xs: xs).to_list())
        assert len(result) == SIZES["l"] * 3

    def test_zpyflow_flatten(self, benchmark, nested_100k):
        benchmark.group = "flat_map N=100K×3"
        result = benchmark(lambda: Query(nested_100k).flatten().to_list())
        assert len(result) == SIZES["l"] * 3


# ---------------------------------------------------------------------------
# chunk — split into fixed-size batches
# ---------------------------------------------------------------------------

class TestChunk:
    CHUNK_SIZE = 100

    def test_python_slice_loop(self, benchmark, floats_1m):
        benchmark.group = f"chunk N=1M (size={self.CHUNK_SIZE})"
        n = self.CHUNK_SIZE
        result = benchmark(lambda: [floats_1m[i:i+n] for i in range(0, len(floats_1m), n)])
        assert len(result) == SIZES["xl"] // self.CHUNK_SIZE

    def test_python_iter_islice(self, benchmark, floats_1m):
        benchmark.group = f"chunk N=1M (size={self.CHUNK_SIZE})"
        import itertools
        n = self.CHUNK_SIZE
        def run():
            it = iter(floats_1m)
            return [list(batch) for batch in iter(lambda: list(itertools.islice(it, n)), [])]
        result = benchmark(run)
        assert len(result) == SIZES["xl"] // self.CHUNK_SIZE

    def test_zpyflow(self, benchmark, floats_1m):
        benchmark.group = f"chunk N=1M (size={self.CHUNK_SIZE})"
        n = self.CHUNK_SIZE
        result = benchmark(lambda: Query(floats_1m).chunk(n).to_list())
        assert len(result) == SIZES["xl"] // self.CHUNK_SIZE
