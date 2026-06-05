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


class TestConvenienceMethods:
    """filter_map / tap / compact / min_by / max_by / unzip / median / product"""

    # filter_map
    def test_filter_map_drops_none(self):
        result = Query(["1", "x", "3", "y"]).filter_map(
            lambda s: int(s) if s.isdigit() else None
        ).to_list()
        assert result == [1, 3]

    def test_filter_map_all_none(self):
        assert Query([1, 2, 3]).filter_map(lambda _: None).to_list() == []

    def test_filter_map_none_kept(self):
        assert Query([1, 2, 3]).filter_map(lambda x: x * 2).to_list() == [2, 4, 6]

    def test_filter_map_empty(self):
        assert Query([]).filter_map(lambda x: x).to_list() == []

    # tap
    def test_tap_passes_through(self):
        seen = []
        result = Query([1, 2, 3]).tap(seen.append).to_list()
        assert result == [1, 2, 3]
        assert seen == [1, 2, 3]

    def test_tap_chained(self):
        log = []
        total = Query([1.0, 2.0, 3.0]).tap(log.append).filter(col > 1).sum()
        assert total == 5.0
        assert len(log) == 3  # tap sees all elements before filter

    # compact
    def test_compact_removes_none(self):
        assert Query([1, None, 2, None, 3]).compact().to_list() == [1, 2, 3]

    def test_compact_falsy_mode(self):
        assert Query([1, 0, 2, "", 3, False]).compact(falsy=True).to_list() == [1, 2, 3]

    def test_compact_empty(self):
        assert Query([]).compact().to_list() == []

    def test_compact_all_none(self):
        assert Query([None, None]).compact().to_list() == []

    def test_compact_keeps_zero_by_default(self):
        assert Query([0, None, 1]).compact().to_list() == [0, 1]

    # min_by / max_by
    def test_min_by_dict(self):
        records = [{"v": 3}, {"v": 1}, {"v": 2}]
        assert Query(records).min_by(lambda r: r["v"]) == {"v": 1}

    def test_max_by_dict(self):
        records = [{"v": 3}, {"v": 1}, {"v": 2}]
        assert Query(records).max_by(lambda r: r["v"]) == {"v": 3}

    def test_min_by_empty(self):
        assert Query([]).min_by(lambda x: x) is None

    def test_max_by_empty(self):
        assert Query([]).max_by(lambda x: x) is None

    def test_min_by_single(self):
        assert Query([{"v": 5}]).min_by(lambda r: r["v"]) == {"v": 5}

    # unzip
    def test_unzip_pairs(self):
        lefts, rights = Query([(1, "a"), (2, "b"), (3, "c")]).unzip()
        assert lefts == [1, 2, 3]
        assert rights == ["a", "b", "c"]

    def test_unzip_empty(self):
        lefts, rights = Query([]).unzip()
        assert lefts == [] and rights == []

    def test_unzip_after_zip(self):
        a = [1, 2, 3]
        b = [4, 5, 6]
        la, lb = Query(a).zip(Query(b)).unzip()
        assert la == a and lb == b

    # median
    def test_median_odd(self):
        assert Query([3.0, 1.0, 4.0, 1.0, 5.0]).median() == 3.0

    def test_median_even(self):
        assert Query([1.0, 2.0, 3.0, 4.0]).median() == 2.5

    def test_median_single(self):
        assert Query([7.0]).median() == 7.0

    def test_median_empty(self):
        assert Query([]).median() is None

    def test_median_sorted_already(self):
        assert Query([1, 2, 3, 4, 5]).median() == 3

    # product
    def test_product_integers(self):
        assert Query([1, 2, 3, 4]).product() == 24

    def test_product_floats(self):
        assert Query([2.0, 0.5, 4.0]).product() == 4.0

    def test_product_empty(self):
        assert Query([]).product() == 1

    def test_product_with_zero(self):
        assert Query([1, 2, 0, 3]).product() == 0
