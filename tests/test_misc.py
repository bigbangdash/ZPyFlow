"""Miscellaneous tests: cache, tee, infinite sequences, cycle/step_by/interleave/sample."""

import pytest

try:
    from zpyflow import Query, col, field, AggSpec, agg_count, agg_sum, agg_mean, agg_max, agg_min
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False

pytestmark = pytest.mark.skipif(not HAS_EXTENSION, reason="Native extension not built")


class TestCache:
    """Query.cache() — materialise and reuse."""

    def test_cache_numeric_reuse(self):
        data = list(range(100))
        q = Query(data).filter(col > 50).cache()
        assert q.count() == 49
        assert q.filter(col > 75).count() == 24
        assert q.sum() == sum(x for x in range(51, 100))

    def test_cache_preserves_values(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        q = Query(data).cache()
        assert q.to_list() == data
        assert q.to_list() == data

    def test_cache_after_filter(self):
        data = list(range(20))
        q = Query(data).filter(col >= 10).cache()
        assert q.count() == 10
        assert q.min() == 10.0
        assert q.max() == 19.0

    def test_cache_dict_pipeline(self):
        records = [{"v": i} for i in range(10)]
        q = Query(records).filter(lambda r: r["v"] > 4).cache()
        assert q.count() == 5
        assert q.filter(lambda r: r["v"] > 7).count() == 2

    def test_cache_empty(self):
        q = Query([]).cache()
        assert q.to_list() == []
        assert q.count() == 0

    def test_cache_does_not_share_state(self):
        data = [1.0, 2.0, 3.0]
        q1 = Query(data).cache()
        q2 = Query(data).cache()
        assert q1.to_list() == q2.to_list()


class TestTee:
    """Query.tee(n) — materialise and return n independent copies."""

    def test_tee_default_2(self):
        q1, q2 = Query([1.0, 2.0, 3.0]).tee()
        assert q1.to_list() == [1.0, 2.0, 3.0]
        assert q2.to_list() == [1.0, 2.0, 3.0]

    def test_tee_n3(self):
        q1, q2, q3 = Query([1, 2, 3]).tee(3)
        assert q1.sum() == q2.sum() == q3.sum() == 6.0

    def test_tee_copies_are_independent(self):
        q1, q2 = Query([1.0, 2.0, 3.0, 4.0, 5.0]).tee()
        filtered = q1.filter(col > 3).to_list()
        assert filtered == [4.0, 5.0]
        assert q2.to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_tee_after_filter(self):
        q1, q2 = Query(range(10)).filter(lambda x: x % 2 == 0).tee()
        assert q1.count() == 5
        assert q2.to_list() == [0.0, 2.0, 4.0, 6.0, 8.0]

    def test_tee_empty(self):
        q1, q2 = Query([]).tee()
        assert q1.to_list() == []
        assert q2.to_list() == []

    def test_tee_n1(self):
        (q,) = Query([1, 2, 3]).tee(1)
        assert q.to_list() == [1.0, 2.0, 3.0]

    def test_tee_invalid_n(self):
        with pytest.raises(ValueError):
            Query([1, 2]).tee(0)


class TestInfiniteSequences:
    """Query.iterate / repeat / repeatedly — Clojure-style infinite factories."""

    def test_iterate_doubling(self):
        result = Query.iterate(lambda x: x * 2, 1).take(6).to_list()
        assert result == [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    def test_iterate_increment(self):
        result = Query.iterate(lambda x: x + 1, 0).take(5).to_list()
        assert result == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_iterate_with_filter(self):
        result = (
            Query.iterate(lambda x: x + 1, 0)
            .filter(lambda x: x % 2 == 0)
            .take(5)
            .to_list()
        )
        assert result == [0, 2, 4, 6, 8]

    def test_iterate_take_zero(self):
        assert Query.iterate(lambda x: x + 1, 0).take(0).to_list() == []

    def test_repeat_finite(self):
        result = Query.repeat(42, 4).to_list()
        assert result == [42, 42, 42, 42]

    def test_repeat_zero(self):
        assert Query.repeat(99, 0).to_list() == []

    def test_repeat_infinite_with_take(self):
        result = Query.repeat("x").take(3).to_list()
        assert result == ["x", "x", "x"]

    def test_repeat_string(self):
        result = Query.repeat("hello", 3).to_list()
        assert result == ["hello", "hello", "hello"]

    def test_repeatedly_finite(self):
        counter = [0]
        def inc():
            counter[0] += 1
            return counter[0]
        result = Query.repeatedly(inc, 5).to_list()
        assert result == [1, 2, 3, 4, 5]

    def test_repeatedly_infinite_with_take(self):
        result = Query.repeatedly(lambda: 7).take(4).to_list()
        assert result == [7, 7, 7, 7]

    def test_repeatedly_zero(self):
        assert Query.repeatedly(lambda: 1, 0).to_list() == []


class TestCycleStepInterleave:
    """cycle / step_by / interleave / sample"""

    def test_cycle_finite(self):
        result = Query([1, 2, 3]).cycle(2).to_list()
        assert result == [1, 2, 3, 1, 2, 3]

    def test_cycle_once(self):
        result = Query([1, 2, 3]).cycle(1).to_list()
        assert result == [1, 2, 3]

    def test_cycle_infinite_with_take(self):
        result = Query([1, 2]).cycle().take(5).to_list()
        assert result == [1.0, 2.0, 1.0, 2.0, 1.0]

    def test_cycle_empty(self):
        assert Query([]).cycle(3).to_list() == []

    def test_cycle_zero(self):
        assert Query([1, 2]).cycle(0).to_list() == []

    def test_step_by_3(self):
        result = Query(range(10)).step_by(3).to_list()
        assert result == [0, 3, 6, 9]

    def test_step_by_1(self):
        result = Query([1, 2, 3]).step_by(1).to_list()
        assert result == [1, 2, 3]

    def test_step_by_larger_than_data(self):
        result = Query([1, 2, 3]).step_by(10).to_list()
        assert result == [1]

    def test_step_by_empty(self):
        assert Query([]).step_by(2).to_list() == []

    def test_step_by_invalid(self):
        with pytest.raises(ValueError):
            Query([1, 2]).step_by(0)

    def test_interleave_equal_length(self):
        result = Query([1, 2, 3]).interleave(Query([10, 20, 30])).to_list()
        assert result == [1, 10, 2, 20, 3, 30]

    def test_interleave_shorter_left(self):
        result = Query([1, 2]).interleave(Query([10, 20, 30])).to_list()
        assert result == [1, 10, 2, 20]

    def test_interleave_shorter_right(self):
        result = Query([1, 2, 3]).interleave(Query([10])).to_list()
        assert result == [1, 10]

    def test_interleave_empty_left(self):
        assert Query([]).interleave(Query([1, 2])).to_list() == []

    def test_interleave_empty_right(self):
        assert Query([1, 2]).interleave(Query([])).to_list() == []

    def test_sample_count(self):
        result = Query(range(100)).sample(10, seed=42).to_list()
        assert len(result) == 10

    def test_sample_reproducible(self):
        r1 = Query(range(100)).sample(10, seed=42).to_list()
        r2 = Query(range(100)).sample(10, seed=42).to_list()
        assert r1 == r2

    def test_sample_different_seeds(self):
        r1 = Query(range(100)).sample(10, seed=1).to_list()
        r2 = Query(range(100)).sample(10, seed=2).to_list()
        assert r1 != r2

    def test_sample_no_duplicates(self):
        result = Query(range(10)).sample(10, seed=0).to_list()
        assert sorted(result) == list(range(10))

    def test_sample_too_large(self):
        with pytest.raises(ValueError):
            Query([1, 2, 3]).sample(5)
