"""Miscellaneous tests: cache, tee, infinite sequences, cycle/step_by/interleave/sample."""

import pytest

try:
    from zpyflow import Query, col, field, AggSpec, agg_count, agg_sum, agg_mean, agg_max, agg_min
    from zpyflow._zpyflow import _infer_schema, _convert_to_columnar
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


class TestFind:
    """Query.find(pred) — first matching element, short-circuits."""

    def test_find_returns_first_match(self):
        records = [{"id": 1}, {"id": 2}, {"id": 3}]
        assert Query(records).find(lambda r: r["id"] == 2) == {"id": 2}

    def test_find_returns_none_when_no_match(self):
        assert Query([1, 2, 3]).find(lambda x: x > 10) is None

    def test_find_empty(self):
        assert Query([]).find(lambda x: x > 0) is None

    def test_find_returns_first_not_last(self):
        records = [{"v": 5}, {"v": 5}, {"v": 7}]
        result = Query(records).find(lambda r: r["v"] == 5)
        assert result is records[0]

    def test_find_short_circuits(self):
        seen = []
        Query([1, 2, 3, 4, 5]).find(lambda x: (seen.append(x), x == 3)[1])
        assert seen == [1, 2, 3]  # stops after finding 3

    def test_find_scalars(self):
        assert Query([10, 20, 30]).find(lambda x: x > 15) == 20

    def test_find_with_field_expr(self):
        records = [{"status": "ok"}, {"status": "error"}, {"status": "ok"}]
        result = Query(records).find(field("status") == "error")
        assert result == {"status": "error"}

    def test_find_first_element(self):
        assert Query([1, 2, 3]).find(lambda x: True) == 1

    def test_find_last_element_only(self):
        assert Query([1, 2, 3]).find(lambda x: x == 3) == 3


class TestAggregationShorthands:
    """count_if / sum_by / mean_by — single-pass aggregation shorthands."""

    # count_if
    def test_count_if_basic(self):
        records = [{"status": "ok"}, {"status": "error"}, {"status": "ok"}]
        assert Query(records).count_if(lambda r: r["status"] == "error") == 1

    def test_count_if_none_match(self):
        assert Query([1, 2, 3]).count_if(lambda x: x > 10) == 0

    def test_count_if_all_match(self):
        assert Query([1, 2, 3]).count_if(lambda x: x > 0) == 3

    def test_count_if_empty(self):
        assert Query([]).count_if(lambda x: True) == 0

    def test_count_if_field_expr(self):
        records = [{"active": True}, {"active": False}, {"active": True}]
        assert Query(records).count_if(field("active") == True) == 2

    def test_count_if_returns_int(self):
        result = Query([1, 2, 3]).count_if(lambda x: x > 1)
        assert isinstance(result, int) and result == 2

    # sum_by
    def test_sum_by_basic(self):
        records = [{"price": 10}, {"price": 20}, {"price": 30}]
        assert Query(records).sum_by(lambda r: r["price"]) == 60.0

    def test_sum_by_empty(self):
        assert Query([]).sum_by(lambda r: r["v"]) == 0.0

    def test_sum_by_floats(self):
        records = [{"v": 1.5}, {"v": 2.5}]
        assert Query(records).sum_by(lambda r: r["v"]) == 4.0

    def test_sum_by_field_expr(self):
        records = [{"score": 3}, {"score": 7}]
        assert Query(records).sum_by(field("score")) == 10.0

    def test_sum_by_scalars(self):
        assert Query([1, 2, 3, 4]).sum_by(lambda x: x * 2) == 20.0

    # mean_by
    def test_mean_by_basic(self):
        records = [{"score": 80}, {"score": 90}, {"score": 100}]
        assert Query(records).mean_by(lambda r: r["score"]) == 90.0

    def test_mean_by_empty_returns_none(self):
        assert Query([]).mean_by(lambda r: r["v"]) is None

    def test_mean_by_single(self):
        assert Query([{"v": 42}]).mean_by(lambda r: r["v"]) == 42.0

    def test_mean_by_floats(self):
        records = [{"v": 1.0}, {"v": 2.0}, {"v": 3.0}]
        assert Query(records).mean_by(lambda r: r["v"]) == 2.0

    def test_mean_by_field_expr(self):
        records = [{"rating": 4}, {"rating": 5}]
        assert Query(records).mean_by(field("rating")) == 4.5


class TestInferSchema:
    """_infer_schema — spec-082 T1: dtype inference from list-of-dicts sample."""

    def test_all_float(self):
        data = [{"score": 0.9}, {"score": 0.1}, {"score": 0.5}]
        schema = _infer_schema(data)
        assert schema["score"] == "f64"

    def test_all_int(self):
        data = [{"count": 1}, {"count": 2}, {"count": 3}]
        schema = _infer_schema(data)
        assert schema["count"] == "i64"

    def test_int_and_float_upgrades_to_f64(self):
        data = [{"x": 1}, {"x": 2.0}, {"x": 3}]
        schema = _infer_schema(data)
        assert schema["x"] == "f64"

    def test_all_str(self):
        data = [{"name": "alice"}, {"name": "bob"}]
        schema = _infer_schema(data)
        assert schema["name"] == "str"

    def test_none_value_keeps_numeric_dtype(self):
        # None is treated as nullable-of-the-inferred-type, not Mixed.
        # The null is recorded in the nulls bitvec; the schema stays F64.
        data = [{"v": 1.0}, {"v": None}, {"v": 3.0}]
        schema = _infer_schema(data)
        assert schema["v"] == "f64"

    def test_all_none_gives_mixed(self):
        # If the first sampled row has None, type cannot be inferred → Mixed.
        data = [{"v": None}, {"v": 1.0}]
        schema = _infer_schema(data)
        assert schema["v"] == "mixed"

    def test_missing_field_gives_mixed(self):
        data = [{"a": 1, "b": 2}, {"a": 3}]
        schema = _infer_schema(data)
        assert schema["a"] == "i64"
        assert schema["b"] == "mixed"

    def test_mixed_types_str_and_int(self):
        data = [{"v": "hello"}, {"v": 42}]
        schema = _infer_schema(data)
        assert schema["v"] == "mixed"

    def test_empty_list(self):
        assert _infer_schema([]) == {}

    def test_multi_field(self):
        data = [{"score": 0.9, "label": "ok", "count": 5},
                {"score": 0.2, "label": "bad", "count": 3}]
        schema = _infer_schema(data)
        assert schema["score"] == "f64"
        assert schema["label"] == "str"
        assert schema["count"] == "i64"

    def test_sample_size_limits_rows(self):
        # Only first 2 rows sampled; row 3 has None but is ignored.
        data = [{"v": 1.0}, {"v": 2.0}, {"v": None}]
        schema = _infer_schema(data, sample_size=2)
        assert schema["v"] == "f64"

    def test_bool_treated_as_i64(self):
        data = [{"flag": True}, {"flag": False}]
        schema = _infer_schema(data)
        assert schema["flag"] == "i64"


class TestConvertToColumnar:
    """_convert_to_columnar — spec-082 T2: dict list → typed column vectors."""

    def test_f64_column(self):
        data = [{"score": 0.9}, {"score": 0.2}, {"score": 0.5}]
        cols = _convert_to_columnar(data)
        assert cols["score"]["dtype"] == "f64"
        assert cols["score"]["data"] == [0.9, 0.2, 0.5]
        assert cols["score"]["nulls"] == [False, False, False]

    def test_i64_column(self):
        data = [{"n": 1}, {"n": 2}, {"n": 3}]
        cols = _convert_to_columnar(data)
        assert cols["n"]["dtype"] == "i64"
        assert cols["n"]["data"] == [1, 2, 3]

    def test_str_column(self):
        data = [{"name": "alice"}, {"name": "bob"}]
        cols = _convert_to_columnar(data)
        assert cols["name"]["dtype"] == "str"
        assert cols["name"]["data"] == ["alice", "bob"]

    def test_none_fills_nan_and_marks_null(self):
        import math
        data = [{"v": 1.0}, {"v": None}, {"v": 3.0}]
        cols = _convert_to_columnar(data)
        assert cols["v"]["nulls"] == [False, True, False]
        assert math.isnan(cols["v"]["data"][1])

    def test_missing_field_fills_default_and_marks_null(self):
        data = [{"a": 1, "b": 2}, {"a": 3}]
        cols = _convert_to_columnar(data)
        assert cols["b"]["nulls"] == [False, True]

    def test_mixed_column_stores_python_objects(self):
        data = [{"v": "hello"}, {"v": 42}]
        cols = _convert_to_columnar(data)
        assert cols["v"]["dtype"] == "mixed"
        assert cols["v"]["data"] == ["hello", 42]

    def test_multi_field(self):
        data = [
            {"score": 0.9, "label": "ok",  "count": 5},
            {"score": 0.2, "label": "bad", "count": 3},
        ]
        cols = _convert_to_columnar(data)
        assert cols["score"]["dtype"] == "f64"
        assert cols["label"]["dtype"] == "str"
        assert cols["count"]["dtype"] == "i64"

    def test_empty_list(self):
        cols = _convert_to_columnar([])
        assert cols == {}

    def test_int_in_f64_column_converts(self):
        # int mixed with float → schema infers f64; ints should become floats
        data = [{"x": 1}, {"x": 2.5}, {"x": 3}]
        cols = _convert_to_columnar(data)
        assert cols["x"]["dtype"] == "f64"
        assert cols["x"]["data"] == [1.0, 2.5, 3.0]


class TestColumnarObj:
    """spec-082 T3 — ColumnarObj filter hot path via .preload()."""

    LOGS = [
        {"score": 0.9, "status": "ok",  "count": 10},
        {"score": 0.3, "status": "bad", "count": 5},
        {"score": 0.7, "status": "ok",  "count": 8},
        {"score": 0.1, "status": "bad", "count": 2},
        {"score": 0.5, "status": "ok",  "count": 3},
    ]

    def test_preload_returns_columnar_repr(self):
        q = Query(self.LOGS).preload()
        assert "columnar_obj" in repr(q)

    def test_filter_gt_matches_python(self):
        logs = self.LOGS
        expected = [r for r in logs if r["score"] > 0.5]
        result = Query(logs).preload().filter(field("score") > 0.5).to_list()
        assert sorted(result, key=lambda r: r["score"]) == sorted(expected, key=lambda r: r["score"])

    def test_filter_lt(self):
        logs = self.LOGS
        expected = [r for r in logs if r["score"] < 0.5]
        result = Query(logs).preload().filter(field("score") < 0.5).to_list()
        assert len(result) == len(expected)

    def test_filter_eq_str(self):
        logs = self.LOGS
        expected = [r for r in logs if r["status"] == "ok"]
        result = Query(logs).preload().filter(field("status") == "ok").to_list()
        assert len(result) == len(expected)

    def test_filter_ne_str(self):
        logs = self.LOGS
        expected = [r for r in logs if r["status"] != "ok"]
        result = Query(logs).preload().filter(field("status") != "ok").to_list()
        assert len(result) == len(expected)

    def test_count_matches_python(self):
        logs = self.LOGS
        expected = sum(1 for r in logs if r["score"] > 0.4)
        assert Query(logs).preload().filter(field("score") > 0.4).count() == expected

    def test_chained_filters(self):
        logs = self.LOGS
        expected = [r for r in logs if r["score"] > 0.4 and r["count"] >= 8]
        result = (
            Query(logs)
            .preload()
            .filter(field("score") > 0.4)
            .filter(field("count") >= 8)
            .to_list()
        )
        assert len(result) == len(expected)

    def test_explain_shows_columnar(self):
        q = Query(self.LOGS).preload()
        expl = q.filter(field("score") > 0.5).explain()
        assert "columnar_obj" in expl

    def test_empty_result(self):
        result = Query(self.LOGS).preload().filter(field("score") > 99.0).to_list()
        assert result == []

    def test_all_rows_pass(self):
        result = Query(self.LOGS).preload().filter(field("score") > 0.0).to_list()
        assert len(result) == len(self.LOGS)

    def test_null_field_excluded_from_numeric_filter(self):
        data = [{"score": 1.0}, {"score": None}, {"score": 0.5}]
        result = Query(data).preload().filter(field("score") > 0.3).to_list()
        assert len(result) == 2  # None row excluded

    def test_str_startswith(self):
        logs = self.LOGS
        result = Query(logs).preload().filter(field("status").startswith("ok")).to_list()
        expected = [r for r in logs if r["status"].startswith("ok")]
        assert len(result) == len(expected)

    def test_str_contains(self):
        logs = self.LOGS
        result = Query(logs).preload().filter(field("status").contains("a")).to_list()
        expected = [r for r in logs if "a" in r["status"]]
        assert len(result) == len(expected)


class TestColumnarObjArrow:
    """spec-083 T2 — to_arrow() / to_polars() / to_pandas() for ColumnarObj."""

    pytest.importorskip("pyarrow", reason="pyarrow not installed")

    LOGS = [
        {"score": 0.9, "label": "ok",  "count": 10},
        {"score": 0.3, "label": "bad", "count": 5},
        {"score": 0.7, "label": "ok",  "count": 8},
        {"score": 0.1, "label": "bad", "count": 2},
    ]

    def test_to_arrow_returns_record_batch(self):
        pa = pytest.importorskip("pyarrow")
        rb = Query(self.LOGS).preload().to_arrow()
        assert isinstance(rb, pa.RecordBatch)
        assert rb.num_rows == len(self.LOGS)

    def test_to_arrow_column_types(self):
        pa = pytest.importorskip("pyarrow")
        rb = Query(self.LOGS).preload().to_arrow()
        assert rb.schema.field("score").type == pa.float64()
        assert rb.schema.field("count").type == pa.int64()

    def test_to_arrow_filtered(self):
        pa = pytest.importorskip("pyarrow")
        rb = Query(self.LOGS).preload().filter(field("score") > 0.5).to_arrow()
        assert rb.num_rows == 2
        scores = rb.column("score").to_pylist()
        assert all(s > 0.5 for s in scores)

    def test_to_arrow_nullable_column(self):
        pa = pytest.importorskip("pyarrow")
        data = [{"x": 1.0}, {"x": None}, {"x": 3.0}]
        rb = Query(data).preload().to_arrow()
        vals = rb.column("x").to_pylist()
        assert vals[0] == 1.0
        assert vals[1] is None
        assert vals[2] == 3.0

    def test_to_arrow_empty(self):
        pa = pytest.importorskip("pyarrow")
        rb = Query(self.LOGS).preload().filter(field("score") > 99.0).to_arrow()
        assert rb.num_rows == 0

    def test_to_polars_returns_dataframe(self):
        pl = pytest.importorskip("polars")
        pytest.importorskip("pyarrow")
        df = Query(self.LOGS).preload().filter(field("score") > 0.5).to_polars()
        assert hasattr(df, "shape")
        assert df.shape[0] == 2

    def test_to_pandas_returns_dataframe(self):
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")
        df = Query(self.LOGS).preload().filter(field("score") > 0.5).to_pandas()
        assert len(df) == 2
