"""Structural transform tests: take/skip_while, chain, chunk, sort, flatten, etc."""

import pytest

try:
    from zpyflow import Query, col, field, AggSpec, agg_count, agg_sum, agg_mean, agg_max, agg_min
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False

pytestmark = pytest.mark.skipif(not HAS_EXTENSION, reason="Native extension not built")


class TestTakeWhile:
    """take_while — emit elements while predicate is True."""

    def test_f64_dsl_fast_path(self):
        q = Query([1.0, 2.0, 3.0, 4.0, 5.0]).take_while(col < 4.0)
        assert q.to_list() == [1.0, 2.0, 3.0]

    def test_f64_dsl_all_pass(self):
        q = Query([1.0, 2.0, 3.0]).take_while(col < 100.0)
        assert q.to_list() == [1.0, 2.0, 3.0]

    def test_f64_dsl_none_pass(self):
        q = Query([5.0, 6.0, 7.0]).take_while(col < 1.0)
        assert q.to_list() == []

    def test_f64_lambda_fallback(self):
        q = Query([1.0, 2.0, 3.0, 4.0]).take_while(lambda x: x < 3.0)
        assert q.to_list() == [1.0, 2.0]

    def test_i64_dsl_fast_path(self):
        q = Query([1, 2, 3, 4, 5]).take_while(col <= 3)
        assert q.to_list() == [1, 2, 3]

    def test_i64_lambda_fallback(self):
        q = Query([10, 20, 5, 30]).take_while(lambda x: x < 25)
        assert q.to_list() == [10, 20, 5]

    def test_obj_field_expr_fast_path(self):
        data = [{"v": i} for i in range(5)]
        q = Query(data).take_while(field("v") < 3.0)
        result = q.to_list()
        assert [r["v"] for r in result] == [0, 1, 2]

    def test_obj_lambda_fallback(self):
        data = [{"v": i} for i in range(5)]
        q = Query(data).take_while(lambda r: r["v"] < 3)
        result = q.to_list()
        assert [r["v"] for r in result] == [0, 1, 2]

    def test_py_fallback(self):
        q = Query(["a", "bb", "ccc", "d"]).take_while(lambda s: len(s) < 3)
        assert q.to_list() == ["a", "bb"]

    def test_chained_with_filter(self):
        q = Query(list(range(10, 0, -1))).filter(col > 3).take_while(col > 6)
        assert q.to_list() == [10.0, 9.0, 8.0, 7.0]


class TestSkipWhile:
    """skip_while — drop elements while predicate is True, emit rest."""

    def test_f64_dsl_fast_path(self):
        q = Query([1.0, 2.0, 3.0, 4.0, 5.0]).skip_while(col < 3.0)
        assert q.to_list() == [3.0, 4.0, 5.0]

    def test_f64_dsl_skip_none(self):
        q = Query([5.0, 6.0]).skip_while(col < 1.0)
        assert q.to_list() == [5.0, 6.0]

    def test_f64_dsl_skip_all(self):
        q = Query([1.0, 2.0, 3.0]).skip_while(col < 100.0)
        assert q.to_list() == []

    def test_f64_lambda_fallback(self):
        q = Query([1.0, 2.0, 3.0, 1.0]).skip_while(lambda x: x < 3.0)
        assert q.to_list() == [3.0, 1.0]

    def test_i64_dsl_fast_path(self):
        q = Query([1, 2, 3, 4, 5]).skip_while(col < 4)
        assert q.to_list() == [4, 5]

    def test_i64_lambda_fallback(self):
        q = Query([1, 2, 10, 3]).skip_while(lambda x: x < 5)
        assert q.to_list() == [10, 3]

    def test_obj_field_expr_fast_path(self):
        data = [{"v": i} for i in range(6)]
        q = Query(data).skip_while(field("v") < 3.0)
        result = q.to_list()
        assert [r["v"] for r in result] == [3, 4, 5]

    def test_obj_lambda_fallback(self):
        data = [{"v": i} for i in range(5)]
        q = Query(data).skip_while(lambda r: r["v"] < 2)
        result = q.to_list()
        assert [r["v"] for r in result] == [2, 3, 4]

    def test_py_fallback(self):
        q = Query(["a", "bb", "ccc", "d"]).skip_while(lambda s: len(s) < 3)
        assert q.to_list() == ["ccc", "d"]

    def test_does_not_resume_skipping(self):
        q = Query([1.0, 2.0, 5.0, 1.0, 2.0]).skip_while(col < 4.0)
        assert q.to_list() == [5.0, 1.0, 2.0]


class TestChain:
    """chain — concatenate two queries."""

    def test_f64_plus_f64_gil_free(self):
        a = Query([1.0, 2.0, 3.0])
        b = Query([4.0, 5.0])
        assert a.chain(b).to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_f64_chain_preserves_type(self):
        a = Query([1.0, 2.0])
        b = Query([3.0, 4.0])
        result = a.chain(b)
        assert result.count() == 4

    def test_i64_plus_i64_gil_free(self):
        a = Query([1, 2, 3])
        b = Query([4, 5])
        assert a.chain(b).to_list() == [1, 2, 3, 4, 5]

    def test_f64_chain_with_filtered(self):
        a = Query(list(range(5))).filter(col >= 3)
        b = Query([10.0, 11.0])
        assert a.chain(b).to_list() == [3.0, 4.0, 10.0, 11.0]

    def test_mixed_types_fallback(self):
        a = Query([1.0, 2.0])
        b = Query(["x", "y"])
        result = a.chain(b).to_list()
        assert result == [1.0, 2.0, "x", "y"]

    def test_chain_empty_left(self):
        a = Query([])
        b = Query([1.0, 2.0])
        assert a.chain(b).to_list() == [1.0, 2.0]

    def test_chain_empty_right(self):
        a = Query([1.0, 2.0])
        b = Query([])
        assert a.chain(b).to_list() == [1.0, 2.0]

    def test_chain_count(self):
        a = Query(list(range(100)))
        b = Query(list(range(50)))
        assert a.chain(b).count() == 150


class TestConcat:
    """concat — concatenate with any iterable."""

    def test_concat_with_list(self):
        assert Query([1.0, 2.0]).concat([3.0, 4.0]).to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_concat_with_query(self):
        a = Query([1.0, 2.0])
        b = Query([3.0, 4.0])
        assert a.concat(b).to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_concat_with_generator(self):
        result = Query([1, 2]).concat(x * 10 for x in [3, 4]).to_list()
        assert result == [1, 2, 30, 40]

    def test_concat_empty_self(self):
        assert Query([]).concat([1.0, 2.0]).to_list() == [1.0, 2.0]

    def test_concat_empty_other(self):
        assert Query([1.0, 2.0]).concat([]).to_list() == [1.0, 2.0]

    def test_concat_both_empty(self):
        assert Query([]).concat([]).to_list() == []


class TestChunk:
    """chunk — split into fixed-size sublists."""

    def test_chunk_even(self):
        assert Query([1, 2, 3, 4]).chunk(2).to_list() == [[1, 2], [3, 4]]

    def test_chunk_with_remainder(self):
        assert Query([1, 2, 3, 4, 5]).chunk(2).to_list() == [[1, 2], [3, 4], [5]]

    def test_chunk_size_one(self):
        assert Query([1, 2, 3]).chunk(1).to_list() == [[1], [2], [3]]

    def test_chunk_larger_than_data(self):
        assert Query([1, 2]).chunk(10).to_list() == [[1, 2]]

    def test_chunk_empty(self):
        assert Query([]).chunk(3).to_list() == []

    def test_chunk_invalid_size(self):
        with pytest.raises(ValueError):
            Query([1, 2, 3]).chunk(0).to_list()

    def test_chunk_after_filter(self):
        result = Query([1, 2, 3, 4, 5, 6]).filter(lambda x: x % 2 == 0).chunk(2).to_list()
        assert result == [[2, 4], [6]]


class TestPartition:
    """partition — split into (matching, non_matching) tuple."""

    def test_partition_lambda(self):
        evens, odds = Query(range(6)).partition(lambda x: x % 2 == 0)
        assert evens == [0, 2, 4]
        assert odds == [1, 3, 5]

    def test_partition_all_match(self):
        yes, no = Query([1, 2, 3]).partition(lambda x: x > 0)
        assert yes == [1, 2, 3]
        assert no == []

    def test_partition_none_match(self):
        yes, no = Query([1, 2, 3]).partition(lambda x: x > 10)
        assert yes == []
        assert no == [1, 2, 3]

    def test_partition_empty(self):
        yes, no = Query([]).partition(lambda x: x > 0)
        assert yes == []
        assert no == []

    def test_partition_field_expr(self):
        records = [{"v": 1}, {"v": -1}, {"v": 2}, {"v": -2}]
        pos, neg = Query(records).partition(field("v") > 0)
        assert pos == [{"v": 1}, {"v": 2}]
        assert neg == [{"v": -1}, {"v": -2}]


class TestSort:
    """sort / sort_by — sorted Query."""

    def test_sort_ascending(self):
        assert Query([3, 1, 2]).sort().to_list() == [1, 2, 3]

    def test_sort_descending(self):
        assert Query([3, 1, 2]).sort(reverse=True).to_list() == [3, 2, 1]

    def test_sort_floats(self):
        assert Query([3.0, 1.0, 2.0]).sort().to_list() == [1.0, 2.0, 3.0]

    def test_sort_empty(self):
        assert Query([]).sort().to_list() == []

    def test_sort_by_key(self):
        records = [{"n": 3}, {"n": 1}, {"n": 2}]
        result = Query(records).sort_by(lambda r: r["n"]).to_list()
        assert result == [{"n": 1}, {"n": 2}, {"n": 3}]

    def test_sort_by_key_reverse(self):
        records = [{"n": 3}, {"n": 1}, {"n": 2}]
        result = Query(records).sort_by(lambda r: r["n"], reverse=True).to_list()
        assert result == [{"n": 3}, {"n": 2}, {"n": 1}]

    def test_sort_by_empty(self):
        assert Query([]).sort_by(lambda x: x).to_list() == []

    def test_sort_after_filter(self):
        result = Query([5, 3, 1, 4, 2]).filter(lambda x: x > 2).sort().to_list()
        assert result == [3, 4, 5]


class TestDistinct:
    """distinct — deduplicate while preserving insertion order."""

    def test_distinct_primitives(self):
        assert Query([1, 2, 1, 3, 2]).distinct().to_list() == [1, 2, 3]

    def test_distinct_preserves_order(self):
        assert Query([3, 1, 2, 1, 3]).distinct().to_list() == [3, 1, 2]

    def test_distinct_no_duplicates(self):
        assert Query([1, 2, 3]).distinct().to_list() == [1, 2, 3]

    def test_distinct_all_same(self):
        assert Query([5, 5, 5]).distinct().to_list() == [5]

    def test_distinct_empty(self):
        assert Query([]).distinct().to_list() == []

    def test_distinct_key_fn(self):
        records = [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}, {"id": 1, "v": "c"}]
        result = Query(records).distinct(lambda r: r["id"]).to_list()
        assert result == [{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]

    def test_distinct_strings(self):
        assert Query(["a", "b", "a", "c"]).distinct().to_list() == ["a", "b", "c"]


class TestScan:
    """scan — running accumulation yielding every intermediate value."""

    def test_scan_cumsum(self):
        assert Query([1, 2, 3, 4]).scan(lambda acc, x: acc + x, 0).to_list() == [1, 3, 6, 10]

    def test_scan_cummax(self):
        assert Query([1, 3, 2, 5, 4]).scan(lambda acc, x: max(acc, x), 0).to_list() == [1, 3, 3, 5, 5]

    def test_scan_cumprod(self):
        assert Query([1, 2, 3, 4]).scan(lambda acc, x: acc * x, 1).to_list() == [1, 2, 6, 24]

    def test_scan_single_element(self):
        assert Query([42]).scan(lambda acc, x: acc + x, 0).to_list() == [42]

    def test_scan_empty(self):
        assert Query([]).scan(lambda acc, x: acc + x, 0).to_list() == []

    def test_scan_string_concat(self):
        result = Query(["a", "b", "c"]).scan(lambda acc, x: acc + x, "").to_list()
        assert result == ["a", "ab", "abc"]


class TestSlidingWindow:
    """sliding_window — overlapping n-element windows."""

    def test_window_size_2(self):
        result = Query([1, 2, 3, 4]).sliding_window(2).to_list()
        assert result == [(1, 2), (2, 3), (3, 4)]

    def test_window_size_3(self):
        result = Query([1, 2, 3, 4, 5]).sliding_window(3).to_list()
        assert result == [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    def test_window_size_equals_length(self):
        result = Query([1, 2, 3]).sliding_window(3).to_list()
        assert result == [(1, 2, 3)]

    def test_window_larger_than_data(self):
        assert Query([1, 2]).sliding_window(5).to_list() == []

    def test_window_size_1(self):
        result = Query([1, 2, 3]).sliding_window(1).to_list()
        assert result == [(1,), (2,), (3,)]

    def test_window_empty(self):
        assert Query([]).sliding_window(2).to_list() == []

    def test_window_invalid_size(self):
        with pytest.raises(ValueError):
            Query([1, 2, 3]).sliding_window(0).to_list()

    def test_window_after_filter(self):
        result = Query([1, 2, 3, 4, 5]).filter(lambda x: x % 2 != 0).sliding_window(2).to_list()
        assert result == [(1, 3), (3, 5)]


class TestEnumerate:
    """enumerate — yield (index, item) tuples."""

    def test_basic(self):
        result = Query(["a", "b", "c"]).enumerate().to_list()
        assert result == [(0, "a"), (1, "b"), (2, "c")]

    def test_f64_values(self):
        result = Query([10.0, 20.0, 30.0]).enumerate().to_list()
        assert result == [(0, 10.0), (1, 20.0), (2, 30.0)]

    def test_i64_values(self):
        result = Query([7, 8, 9]).enumerate().to_list()
        assert result == [(0, 7), (1, 8), (2, 9)]

    def test_empty(self):
        assert Query([]).enumerate().to_list() == []

    def test_index_zero_based(self):
        pairs = Query(["x", "y", "z"]).enumerate().to_list()
        indices = [p[0] for p in pairs]
        assert indices == [0, 1, 2]

    def test_after_filter(self):
        result = Query([1.0, 2.0, 3.0, 4.0]).filter(col > 2.0).enumerate().to_list()
        assert result == [(0, 3.0), (1, 4.0)]

    def test_count_unchanged(self):
        assert Query(list(range(10))).enumerate().count() == 10


class TestZip:
    """zip — pair elements from two queries, stop at shorter."""

    def test_basic_same_length(self):
        a = Query([1.0, 2.0, 3.0])
        b = Query(["x", "y", "z"])
        result = a.zip(b).to_list()
        assert result == [(1.0, "x"), (2.0, "y"), (3.0, "z")]

    def test_stops_at_shorter_left(self):
        a = Query([1.0, 2.0])
        b = Query(["x", "y", "z"])
        result = a.zip(b).to_list()
        assert result == [(1.0, "x"), (2.0, "y")]

    def test_stops_at_shorter_right(self):
        a = Query([1.0, 2.0, 3.0])
        b = Query(["x"])
        result = a.zip(b).to_list()
        assert result == [(1.0, "x")]

    def test_both_empty(self):
        assert Query([]).zip(Query([])).to_list() == []

    def test_i64_values(self):
        a = Query([10, 20, 30])
        b = Query([1, 2, 3])
        result = a.zip(b).to_list()
        assert result == [(10, 1), (20, 2), (30, 3)]

    def test_zip_then_count(self):
        a = Query(list(range(5)))
        b = Query(list(range(5)))
        assert a.zip(b).count() == 5

    def test_zip_same_query(self):
        q = Query([1.0, 2.0, 3.0])
        r = q.zip(q).to_list()
        assert r == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]


class TestFlatMap:
    """flat_map — apply f to each element and flatten sub-iterables."""

    def test_basic_list_expansion(self):
        result = Query([1, 2, 3]).flat_map(lambda x: [x, x * 10]).to_list()
        assert result == [1, 10, 2, 20, 3, 30]

    def test_range_expansion(self):
        result = Query([1, 2, 3]).flat_map(lambda x: range(int(x))).to_list()
        assert result == [0, 0, 1, 0, 1, 2]

    def test_string_chars(self):
        result = Query(["ab", "cd"]).flat_map(list).to_list()
        assert result == ["a", "b", "c", "d"]

    def test_empty_sub_iterables(self):
        result = Query([1, 2, 3]).flat_map(lambda x: []).to_list()
        assert result == []

    def test_mixed_length_sub_iterables(self):
        result = Query([0, 1, 2]).flat_map(lambda x: list(range(int(x)))).to_list()
        assert result == [0, 0, 1]

    def test_count(self):
        assert Query([1, 2, 3, 4]).flat_map(lambda x: [x, x, x]).count() == 12

    def test_f64_source(self):
        result = Query([1.0, 2.0]).flat_map(lambda x: [x, -x]).to_list()
        assert result == [1.0, -1.0, 2.0, -2.0]


class TestExplain:
    """Stable output format for Query.explain()."""

    def test_f64_kind(self):
        out = Query([1.0, 2.0, 3.0]).explain()
        assert "f64" in out

    def test_i64_kind(self):
        out = Query([1, 2, 3]).explain()
        assert "i64" in out

    def test_ops_appear(self):
        out = Query([1.0, 2.0]).filter(col > 0).map(col * 2.0).explain()
        assert "FilterGt" in out
        assert "MapMulScalar" in out

    def test_take_appears(self):
        out = Query([1.0, 2.0]).take(5).explain()
        assert "5" in out

    def test_skip_appears(self):
        out = Query([1.0, 2.0]).skip(3).explain()
        assert "skip" in out
        assert "3" in out

    def test_parallel_flag(self):
        out = Query([1.0, 2.0]).parallel().explain()
        assert "parallel" in out
        assert "true" in out

    def test_obj_kind(self):
        out = Query([{"x": 1}]).explain()
        assert "obj" in out

    def test_rust_obj_kind(self):
        out = Query([{"x": 1}]).filter(field("x") > 0).explain()
        assert "obj_field" in out

    def test_unbounded_take_shows_infinity(self):
        out = Query([1.0]).explain()
        assert "∞" in out

    def test_output_has_required_fields(self):
        out = Query([1.0]).filter(col > 0).take(10).explain()
        for field_name in ("kind", "ops", "skip", "take", "parallel", "gil_free", "alloc"):
            assert field_name in out


class TestFlatten:
    """Query.flatten() — expand nested iterables one level."""

    def test_flatten_lists(self):
        result = Query([[1, 2], [3, 4], [5]]).flatten().to_list()
        assert result == [1, 2, 3, 4, 5]

    def test_flatten_ranges(self):
        result = Query([range(3), range(2)]).flatten().to_list()
        assert result == [0, 1, 2, 0, 1]

    def test_flatten_empty_inner(self):
        result = Query([[1, 2], [], [3]]).flatten().to_list()
        assert result == [1, 2, 3]

    def test_flatten_empty_outer(self):
        assert Query([]).flatten().to_list() == []

    def test_flatten_strings_not_expanded(self):
        result = Query(["abc", "de"]).flatten().to_list()
        assert result == ["abc", "de"]

    def test_flatten_mixed_iterable_and_scalar(self):
        result = Query([[1, 2], 3, [4, 5]]).flatten().to_list()
        assert result == [1, 2, 3, 4, 5]

    def test_flatten_after_filter(self):
        data = [[1, 2], [3, 4], [5, 6]]
        result = Query(data).filter(lambda lst: len(lst) == 2 and lst[0] > 2).flatten().to_list()
        assert result == [3, 4, 5, 6]

    def test_flatten_chunk_roundtrip(self):
        data = list(range(9))
        result = Query(data).chunk(3).flatten().to_list()
        assert result == data


class TestPartitionBy:
    """partition_by(key_fn) — Clojure-style consecutive grouping."""

    def test_basic_integers(self):
        result = Query([1, 1, 2, 2, 3, 1, 1]).partition_by().to_list()
        assert result == [[1, 1], [2, 2], [3], [1, 1]]

    def test_with_key_fn(self):
        result = Query([1, 3, 2, 4, 1, 5]).partition_by(lambda x: x % 2).to_list()
        assert result == [[1, 3], [2, 4], [1, 5]]

    def test_all_same(self):
        result = Query([5, 5, 5]).partition_by().to_list()
        assert result == [[5, 5, 5]]

    def test_all_different(self):
        result = Query([1, 2, 3]).partition_by().to_list()
        assert result == [[1], [2], [3]]

    def test_empty(self):
        assert Query([]).partition_by().to_list() == []

    def test_single_element(self):
        assert Query([42]).partition_by().to_list() == [[42]]

    def test_strings(self):
        result = Query(["a", "a", "b", "b", "a"]).partition_by().to_list()
        assert result == [["a", "a"], ["b", "b"], ["a"]]

    def test_dict_key_fn(self):
        logs = [
            {"level": "INFO", "msg": "a"},
            {"level": "INFO", "msg": "b"},
            {"level": "ERROR", "msg": "c"},
            {"level": "INFO", "msg": "d"},
        ]
        result = Query(logs).partition_by(lambda r: r["level"]).to_list()
        assert len(result) == 3
        assert len(result[0]) == 2
        assert result[1][0]["level"] == "ERROR"


class TestDedupe:
    """dedupe(key_fn) — Clojure-style consecutive deduplication."""

    def test_basic(self):
        result = Query([1, 1, 2, 2, 3, 1, 1]).dedupe().to_list()
        assert result == [1, 2, 3, 1]

    def test_no_consecutive_duplicates(self):
        result = Query([1, 2, 3]).dedupe().to_list()
        assert result == [1, 2, 3]

    def test_all_same(self):
        result = Query([5, 5, 5]).dedupe().to_list()
        assert result == [5]

    def test_empty(self):
        assert Query([]).dedupe().to_list() == []

    def test_with_key_fn(self):
        result = Query([1, 3, 2, 4, 1]).dedupe(lambda x: x % 2).to_list()
        assert result == [1, 2, 1]

    def test_single_element(self):
        assert Query([42]).dedupe().to_list() == [42]

    def test_non_consecutive_kept(self):
        result = Query([1, 2, 1, 2, 1]).dedupe().to_list()
        assert result == [1, 2, 1, 2, 1]

    def test_differs_from_distinct(self):
        data = [1, 2, 1, 2, 1]
        assert Query(data).dedupe().to_list() == [1, 2, 1, 2, 1]
        assert Query(data).distinct().to_list() == [1, 2]
