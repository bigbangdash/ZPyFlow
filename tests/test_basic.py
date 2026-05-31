"""
Unit tests for ZPyFlow.

These tests run against the compiled extension.  Build first:
    maturin develop --release

Or run via pytest directly after build.
"""

import pytest

# ---------------------------------------------------------------------------
# Import guard — skip gracefully if extension not built
# ---------------------------------------------------------------------------
try:
    from zpyflow import Query, col, field, AggSpec, agg_count, agg_sum, agg_mean, agg_max, agg_min
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False

pytestmark = pytest.mark.skipif(not HAS_EXTENSION, reason="Native extension not built")


# ===========================================================================
# Numeric f64 fast path
# ===========================================================================

class TestF64Pipeline:
    def setup_method(self):
        self.data = list(range(-50, 51))  # -50..50 as floats
        self.fdata = [float(x) for x in self.data]

    def test_filter_gt_expr(self):
        result = Query(self.fdata).filter(col > 0).to_list()
        expected = [x for x in self.fdata if x > 0]
        assert result == expected

    def test_filter_ge_expr(self):
        result = Query(self.fdata).filter(col >= 0).to_list()
        expected = [x for x in self.fdata if x >= 0]
        assert result == expected

    def test_filter_lt_expr(self):
        result = Query(self.fdata).filter(col < 0).to_list()
        expected = [x for x in self.fdata if x < 0]
        assert result == expected

    def test_filter_between(self):
        result = Query(self.fdata).filter(col.between(-10, 10)).to_list()
        expected = [x for x in self.fdata if -10 <= x <= 10]
        assert result == expected

    def test_filter_between_then_map(self):
        """FilterBetween in SIMD path followed by a map op."""
        result = Query(self.fdata).filter(col.between(-10, 10)).map(col * 2).to_list()
        expected = [x * 2 for x in self.fdata if -10 <= x <= 10]
        assert pytest.approx(result) == expected

    def test_filter_between_count(self):
        result = Query(self.fdata).filter(col.between(-10, 10)).count()
        expected = sum(1 for x in self.fdata if -10 <= x <= 10)
        assert result == expected

    def test_filter_between_sum(self):
        result = Query(self.fdata).filter(col.between(-10, 10)).sum()
        expected = sum(x for x in self.fdata if -10 <= x <= 10)
        assert pytest.approx(result) == expected

    def test_filter_between_chained_with_gt(self):
        """FilterBetween followed by a second filter — both in SIMD pipeline."""
        result = (
            Query(self.fdata)
            .filter(col.between(-20, 20))
            .filter(col > 0)
            .to_list()
        )
        expected = [x for x in self.fdata if -20 <= x <= 20 and x > 0]
        assert pytest.approx(result) == expected

    def test_filter_between_large(self):
        """N=1M — SIMD 4-wide loop is the dominant path."""
        import numpy as np
        rng = np.random.default_rng(42)
        data = rng.uniform(-100, 100, 1_000_000).tolist()
        lo, hi = -25.0, 25.0
        result = Query(data).filter(col.between(lo, hi)).count()
        expected = sum(1 for x in data if lo <= x <= hi)
        assert result == expected

    def test_map_mul_expr(self):
        result = Query(self.fdata).map(col * 2.0).to_list()
        expected = [x * 2.0 for x in self.fdata]
        assert pytest.approx(result) == expected

    def test_map_add_expr(self):
        result = Query(self.fdata).map(col + 100.0).to_list()
        expected = [x + 100.0 for x in self.fdata]
        assert pytest.approx(result) == expected

    def test_map_neg_expr(self):
        result = Query(self.fdata).map(-col).to_list()
        expected = [-x for x in self.fdata]
        assert pytest.approx(result) == expected

    def test_map_abs_expr(self):
        result = Query(self.fdata).map(col.abs()).to_list()
        expected = [abs(x) for x in self.fdata]
        assert pytest.approx(result) == expected

    def test_map_sqrt_expr(self):
        pos = [x for x in self.fdata if x >= 0]
        result = Query(pos).map(col.sqrt()).to_list()
        expected = [x ** 0.5 for x in pos]
        assert pytest.approx(result, rel=1e-9) == expected

    def test_chained_filter_map(self):
        result = (
            Query(self.fdata)
            .filter(col > 0)
            .map(col * 2.0)
            .to_list()
        )
        expected = [x * 2.0 for x in self.fdata if x > 0]
        assert pytest.approx(result) == expected

    def test_take(self):
        result = Query(self.fdata).take(10).to_list()
        assert result == self.fdata[:10]

    def test_skip(self):
        result = Query(self.fdata).skip(10).to_list()
        assert pytest.approx(result) == self.fdata[10:]

    def test_skip_take_combined(self):
        result = Query(self.fdata).skip(5).take(10).to_list()
        assert pytest.approx(result) == self.fdata[5:15]

    def test_filter_skip_take_combined(self):
        result = Query(self.fdata).filter(col > 0).skip(5).take(10).to_list()
        expected = [x for x in self.fdata if x > 0][5:15]
        assert pytest.approx(result) == expected

    def test_parallel_filter_skip_take_combined(self):
        result = Query(self.fdata).filter(col > 0).parallel().skip(5).take(10).to_list()
        expected = [x for x in self.fdata if x > 0][5:15]
        assert pytest.approx(result) == expected

    def test_parallel_map_skip_take_combined(self):
        result = Query(self.fdata).map(col * 2.0).parallel().skip(5).take(10).to_list()
        expected = [x * 2.0 for x in self.fdata][5:15]
        assert pytest.approx(result) == expected

    def test_count(self):
        n = Query(self.fdata).filter(col > 0).count()
        assert n == len([x for x in self.fdata if x > 0])

    def test_sum(self):
        s = Query(self.fdata).filter(col >= 0).sum()
        expected = sum(x for x in self.fdata if x >= 0)
        assert pytest.approx(s) == expected

    def test_min(self):
        m = Query(self.fdata).min()
        assert m == min(self.fdata)

    def test_max(self):
        m = Query(self.fdata).max()
        assert m == max(self.fdata)

    def test_first(self):
        f = Query(self.fdata).filter(col > 0).first()
        assert f == 1.0

    def test_empty_pipeline(self):
        result = Query(self.fdata).filter(col > 999).to_list()
        assert result == []

    def test_all_filtered(self):
        n = Query(self.fdata).filter(col > 999).count()
        assert n == 0

    def test_lambda_filter_fallback(self):
        """Python lambdas fall back to GIL path but still work correctly."""
        result = Query(self.fdata).filter(lambda x: x > 0).to_list()
        expected = [x for x in self.fdata if x > 0]
        assert pytest.approx(result) == expected

    def test_lambda_map_fallback(self):
        result = Query(self.fdata).map(lambda x: x ** 2).to_list()
        expected = [x ** 2 for x in self.fdata]
        assert pytest.approx(result) == expected

    def test_large_data(self):
        """Smoke test on 1M elements."""
        data = [float(i % 1000) for i in range(1_000_000)]
        result = Query(data).filter(col > 500).take(100).to_list()
        assert len(result) == 100
        assert all(x > 500 for x in result)


# ===========================================================================
# Integer i64 fast path
# ===========================================================================

class TestI64Pipeline:
    def setup_method(self):
        self.data = list(range(-50, 51))

    def test_filter_gt(self):
        result = Query(self.data).filter(col > 0).to_list()
        expected = [x for x in self.data if x > 0]
        assert result == expected

    def test_map_mul(self):
        result = Query(self.data).map(col * 3).to_list()
        expected = [x * 3 for x in self.data]
        assert result == expected

    def test_chained(self):
        result = (
            Query(self.data)
            .filter(col >= 0)
            .map(col * 2)
            .take(5)
            .to_list()
        )
        expected = [x * 2 for x in self.data if x >= 0][:5]
        assert result == expected

    def test_count(self):
        n = Query(self.data).filter(col > 0).count()
        assert n == 50

    def test_sum(self):
        s = Query(self.data).filter(col > 0).sum()
        assert s == sum(x for x in self.data if x > 0)

    def test_any_dsl_fast_path(self):
        assert Query(self.data).any(col > 49)
        assert not Query(self.data).any(col > 100)

    def test_all_dsl_fast_path(self):
        assert Query(self.data).filter(col >= 0).all(col >= 0)
        assert not Query(self.data).all(col > 0)

    def test_map_only_skip_take_uses_correct_window(self):
        result = Query(self.data).map(col * 2).skip(10).take(5).to_list()
        expected = [x * 2 for x in self.data][10:15]
        assert result == expected

    def test_filter_skip_take_combined(self):
        result = Query(self.data).filter(col > 0).skip(5).take(10).to_list()
        expected = [x for x in self.data if x > 0][5:15]
        assert result == expected

    def test_parallel_filter_skip_take_combined(self):
        result = Query(self.data).filter(col > 0).parallel().skip(5).take(10).to_list()
        expected = [x for x in self.data if x > 0][5:15]
        assert result == expected

    def test_parallel_map_skip_take_combined(self):
        result = Query(self.data).map(col * 2).parallel().skip(10).take(5).to_list()
        expected = [x * 2 for x in self.data][10:15]
        assert result == expected


# ===========================================================================
# Generic Python object path
# ===========================================================================

class TestPyObjectPipeline:
    def setup_method(self):
        self.records = [
            {"name": "Alice", "age": 30, "dept": "eng"},
            {"name": "Bob",   "age": 25, "dept": "mkt"},
            {"name": "Carol", "age": 35, "dept": "eng"},
            {"name": "Dan",   "age": 22, "dept": "mkt"},
        ]

    def test_filter_lambda(self):
        result = (
            Query(self.records)
            .filter(lambda r: r["dept"] == "eng")
            .to_list()
        )
        assert len(result) == 2
        assert all(r["dept"] == "eng" for r in result)

    def test_map_lambda(self):
        result = (
            Query(self.records)
            .map(lambda r: r["name"])
            .to_list()
        )
        assert result == ["Alice", "Bob", "Carol", "Dan"]

    def test_chained_filter_map(self):
        result = (
            Query(self.records)
            .filter(lambda r: r["age"] >= 30)
            .map(lambda r: r["name"])
            .to_list()
        )
        assert result == ["Alice", "Carol"]

    def test_reduce(self):
        total = (
            Query(self.records)
            .map(lambda r: r["age"])
            .reduce(lambda acc, x: acc + x, initial=0)
        )
        assert total == 30 + 25 + 35 + 22

    def test_any(self):
        assert Query(self.records).any(lambda r: r["dept"] == "eng")
        assert not Query(self.records).any(lambda r: r["dept"] == "hr")

    def test_all(self):
        assert Query(self.records).all(lambda r: r["age"] > 20)
        assert not Query(self.records).all(lambda r: r["age"] > 25)

    def test_count(self):
        n = Query(self.records).filter(lambda r: r["age"] > 25).count()
        assert n == 2

    def test_first(self):
        f = Query(self.records).filter(lambda r: r["dept"] == "mkt").first()
        assert f == {"name": "Bob", "age": 25, "dept": "mkt"}

    def test_skip_take(self):
        result = Query(self.records).skip(1).take(2).to_list()
        assert result == self.records[1:3]

    def test_for_each(self):
        names = []
        Query(self.records).map(lambda r: r["name"]).for_each(names.append)
        assert names == ["Alice", "Bob", "Carol", "Dan"]

    def test_string_pipeline(self):
        words = ["hello", "world", "foo", "bar", "baz"]
        result = (
            Query(words)
            .filter(lambda w: len(w) > 3)
            .map(lambda w: w.upper())
            .to_list()
        )
        assert result == ["HELLO", "WORLD"]

    def test_generator_input(self):
        result = Query(x * 2 for x in range(10)).filter(lambda x: x > 10).to_list()
        assert result == [12, 14, 16, 18]

    def test_nested_pipeline(self):
        """Query over Query results."""
        inner = Query([1, 2, 3, 4, 5]).filter(col > 2).to_list()
        outer = Query(inner).map(col * 10).to_list()
        assert outer == [30.0, 40.0, 50.0]


# ===========================================================================
# Expression DSL
# ===========================================================================

class TestExprDSL:
    def test_col_proxy_operators(self):
        # Each operator returns an Expr, not a bool
        from zpyflow import col, Expr
        expr = col > 5
        assert isinstance(expr, Expr)

    def test_pow(self):
        data = [1.0, 2.0, 3.0, 4.0]
        result = Query(data).map(col ** 2).to_list()
        assert pytest.approx(result) == [1.0, 4.0, 9.0, 16.0]

    def test_reciprocal(self):
        data = [1.0, 2.0, 4.0, 8.0]
        result = Query(data).map(col.reciprocal()).to_list()
        assert pytest.approx(result) == [1.0, 0.5, 0.25, 0.125]

    def test_floor_ceil(self):
        data = [1.1, 2.7, -3.3, -0.5]
        floors = Query(data).map(col.floor()).to_list()
        assert floors == [1.0, 2.0, -4.0, -1.0]
        ceils = Query(data).map(col.ceil()).to_list()
        assert ceils == [2.0, 3.0, -3.0, 0.0]
        rounds = Query(data).map(col.round()).to_list()
        assert rounds == [1.0, 3.0, -3.0, -1.0]


# ===========================================================================
# Edge cases
# ===========================================================================

# ===========================================================================
# Regression: LazyFloatList → lambda fallback must not drop accumulated DSL ops
# ===========================================================================

class TestLazyFloatListFallback:
    """list[float] input → DSL op → lambda must apply both ops correctly."""

    def test_dsl_filter_then_lambda_filter(self):
        data = [1.0, -2.0, 3.0, -4.0, 5.0]
        result = (
            Query(data)
            .filter(col > 0)              # DSL: stays in LazyFloatList ops
            .filter(lambda x: x != 3.0)  # lambda: previously dropped col > 0
            .to_list()
        )
        assert result == [1.0, 5.0]

    def test_dsl_filter_then_lambda_map(self):
        data = [1.0, -2.0, 3.0, -4.0]
        result = (
            Query(data)
            .filter(col > 0)          # DSL filter
            .map(lambda x: x * 10)   # lambda: previously dropped the filter
            .to_list()
        )
        assert pytest.approx(result) == [10.0, 30.0]

    def test_multiple_dsl_then_lambda(self):
        data = [1.0, -2.0, 3.0, -4.0, 5.0]
        result = (
            Query(data)
            .filter(col > 0)
            .map(col * 2)             # second DSL op
            .filter(lambda x: x < 8) # lambda after two DSL ops
            .to_list()
        )
        assert pytest.approx(result) == [2.0, 6.0]

    def test_dsl_with_skip_take_then_lambda(self):
        data = [float(i) for i in range(10)]
        result = (
            Query(data)
            .filter(col > 2)
            .take(4)
            .filter(lambda x: x != 4.0)
            .to_list()
        )
        assert pytest.approx(result) == [3.0, 5.0, 6.0]


# ===========================================================================
# Regression: any() / all() must short-circuit (not fully materialize)
# ===========================================================================

class TestAnyAllShortCircuit:
    def test_any_f64_true(self):
        assert Query([1.0, 2.0, 3.0]).filter(col > 0).any(lambda x: x > 2)

    def test_any_f64_false(self):
        assert not Query([1.0, 2.0, 3.0]).any(lambda x: x > 10)

    def test_any_empty_is_false(self):
        assert not Query([]).any(lambda x: True)

    def test_all_f64_true(self):
        assert Query([1.0, 2.0, 3.0]).filter(col > 0).all(lambda x: x > 0)

    def test_all_f64_false(self):
        assert not Query([1.0, 2.0, 3.0]).all(lambda x: x > 2)

    def test_all_empty_is_true(self):
        assert Query([]).all(lambda x: False)

    def test_any_obj_with_filter(self):
        records = [{"x": 1}, {"x": 5}, {"x": 3}]
        assert Query(records).filter(lambda r: r["x"] > 2).any(lambda r: r["x"] > 4)
        assert not Query(records).filter(lambda r: r["x"] > 2).any(lambda r: r["x"] > 10)

    def test_all_obj_with_filter(self):
        records = [{"x": 3}, {"x": 5}, {"x": 7}]
        assert Query(records).filter(lambda r: r["x"] > 2).all(lambda r: r["x"] > 0)
        assert not Query(records).filter(lambda r: r["x"] > 2).all(lambda r: r["x"] > 4)

    def test_any_respects_take(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        # take(2) → [1.0, 2.0]; any(x > 3) should be False even though 4.0/5.0 exist
        assert not Query(data).take(2).any(lambda x: x > 3)

    def test_all_respects_take(self):
        data = [1.0, 2.0, 3.0, 4.0]
        # take(2) → [1.0, 2.0]; all(x < 3) should be True
        assert Query(data).take(2).all(lambda x: x < 3)

    def test_any_f64_dsl_respects_take(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert not Query(data).take(2).any(col > 3)

    def test_all_f64_dsl_respects_skip_take(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert Query(data).skip(2).take(2).all(col >= 3)

    def test_any_i64_dsl_respects_take(self):
        import numpy as np
        from zpyflow import from_numpy

        data = [1, 2, 3, 4, 5]
        assert not from_numpy(np.array(data, dtype=np.int64)).take(2).any(col > 3)

    def test_all_i64_dsl_respects_skip_take(self):
        import numpy as np
        from zpyflow import from_numpy

        data = [1, 2, 3, 4, 5]
        assert from_numpy(np.array(data, dtype=np.int64)).skip(2).take(2).all(col >= 3)


class TestEdgeCases:
    def test_empty_input(self):
        assert Query([]).to_list() == []
        assert Query([]).count() == 0
        assert Query([]).first() is None
        assert Query([]).last() is None

    def test_single_element(self):
        q = Query([42.0])
        assert q.to_list() == [42.0]
        assert q.count() == 1
        assert q.first() == 42.0
        assert q.last() == 42.0

    def test_take_more_than_available(self):
        result = Query([1.0, 2.0, 3.0]).take(100).to_list()
        assert result == [1.0, 2.0, 3.0]

    def test_skip_more_than_available(self):
        result = Query([1.0, 2.0, 3.0]).skip(100).to_list()
        assert result == []

    def test_mixed_int_float_list(self):
        """Python list with mixed types stays on Py path."""
        data = [1, 2.0, 3, "four"]
        result = Query(data).filter(lambda x: isinstance(x, (int, float))).to_list()
        assert result == [1, 2.0, 3]

    def test_repr(self):
        q = Query([1.0, 2.0]).filter(col > 0)
        r = repr(q)
        assert "Query" in r


# ===========================================================================
# GroupBy
# ===========================================================================

class TestFromNumpy:
    def test_from_numpy_f64(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = from_numpy(arr).filter(col > 2).to_list()
        assert result == pytest.approx([3.0, 4.0, 5.0])

    def test_from_numpy_i64(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        result = from_numpy(arr).filter(col > 2).to_list()
        assert result == [3, 4, 5]

    def test_from_numpy_large(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.arange(1_000_000, dtype=np.float64)
        count = from_numpy(arr).filter(col > 500_000).count()
        assert count == 499_999

    def test_from_numpy_non_contiguous(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])[::2]  # stride=2, non-contiguous
        result = from_numpy(arr).to_list()
        assert result == pytest.approx([1.0, 3.0, 5.0])

    def test_from_numpy_bool_compact_count_sum_to_list(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([True, False, True, True], dtype=np.bool_)
        q = from_numpy(arr)
        assert "Query<u8>" in repr(q)
        assert q.filter(col > 0).count() == 3
        assert q.sum() == 3
        assert q.to_list() == [1, 0, 1, 1]

    def test_from_numpy_uint8_compact_filter_map(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([0, 2, 5, 255], dtype=np.uint8)
        q = from_numpy(arr)
        assert "Query<u8>" in repr(q)
        assert q.filter(col >= 2).to_list() == [2, 5, 255]
        assert q.filter(col > 1).sum() == 262
        assert q.map(col + 1).to_list() == [1, 3, 6, 256]

    def test_from_numpy_f32_filter(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = from_numpy(arr).filter(col > 2).to_list()
        assert result == pytest.approx([3.0, 4.0, 5.0])

    def test_from_numpy_f32_repr(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert "numpy_f32" in repr(from_numpy(arr))

    def test_from_numpy_f32_count(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        assert from_numpy(arr).filter(col > 2).count() == 3

    def test_from_numpy_f32_sum(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        assert from_numpy(arr).filter(col > 0).sum() == pytest.approx(15.0, rel=1e-5)

    def test_from_numpy_f32_map(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = from_numpy(arr).map(col * 2).to_list()
        assert result == pytest.approx([2.0, 4.0, 6.0])

    def test_from_numpy_f32_to_numpy(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = from_numpy(arr).filter(col > 2).to_numpy()
        assert out.dtype == np.float32
        assert list(out) == pytest.approx([3.0, 4.0])

    def test_from_numpy_f32_large(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.arange(1_000_000, dtype=np.float32)
        count = from_numpy(arr).filter(col > 500_000).count()
        assert count == 499_999

    def test_from_numpy_f32_not_upcasted(self):
        # f32 arrays must NOT be silently upcast to f64; they stay on the f32 path
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        q = from_numpy(arr)
        assert "numpy_f32" in repr(q)


class TestToBytes:
    def test_to_bytes_len(self):
        import numpy as np
        arr = np.frombuffer(Query([1.0, 2.0, 3.0]).to_bytes(), dtype=np.float64)
        assert len(arr) == 3

    def test_to_bytes_values(self):
        import numpy as np
        data = [1.5, -2.3, 3.7]
        arr = np.frombuffer(Query(data).filter(col > 0).to_bytes(), dtype=np.float64)
        assert list(arr) == pytest.approx([1.5, 3.7])

    def test_to_bytes_filter_map(self):
        import numpy as np
        data = [float(i) for i in range(10)]
        arr = np.frombuffer(Query(data).filter(col > 4).map(col * 2).to_bytes(), dtype=np.float64)
        assert list(arr) == pytest.approx([10.0, 12.0, 14.0, 16.0, 18.0])

    def test_to_bytes_is_bytes(self):
        result = Query([1.0, 2.0, 3.0]).to_bytes()
        assert isinstance(result, bytes)
        assert len(result) == 3 * 8  # 3 f64 × 8 bytes

    def test_to_bytes_non_f64_raises(self):
        with pytest.raises(ValueError):
            Query(["a", "b"]).to_bytes()


class TestGroupBy:
    records = [
        {"dept": "eng",  "salary": 120_000},
        {"dept": "mkt",  "salary":  85_000},
        {"dept": "eng",  "salary": 140_000},
        {"dept": "hr",   "salary":  75_000},
        {"dept": "mkt",  "salary":  90_000},
    ]

    def test_group_by_keys(self):
        gb = Query(self.records).group_by(lambda r: r["dept"])
        assert sorted(gb.keys()) == ["eng", "hr", "mkt"]

    def test_group_by_count_per_group(self):
        counts = Query(self.records).group_by(lambda r: r["dept"]).count_per_group()
        assert counts == {"eng": 2, "mkt": 2, "hr": 1}

    def test_group_by_sum_per_group(self):
        totals = (
            Query(self.records)
            .group_by(lambda r: r["dept"])
            .sum_per_group(field=lambda r: r["salary"])
        )
        assert totals["eng"] == pytest.approx(260_000)
        assert totals["mkt"] == pytest.approx(175_000)
        assert totals["hr"]  == pytest.approx(75_000)

    def test_group_by_get_group(self):
        eng = (
            Query(self.records)
            .group_by(lambda r: r["dept"])
            .get_group("eng")
            .to_list()
        )
        assert len(eng) == 2
        assert all(r["dept"] == "eng" for r in eng)

    def test_group_by_agg(self):
        result = (
            Query(self.records)
            .group_by(lambda r: r["dept"])
            .agg(
                count=lambda g: g.count(),
                total=lambda g: Query([r["salary"] for r in g.to_list()]).sum(),
            )
        )
        by_key = {row["_key"]: row for row in result}
        assert by_key["eng"]["count"] == 2
        assert by_key["eng"]["total"] == pytest.approx(260_000)

    def test_group_by_after_filter(self):
        result = (
            Query(self.records)
            .filter(lambda r: r["salary"] > 80_000)
            .group_by(lambda r: r["dept"])
            .count_per_group()
        )
        assert result == {"eng": 2, "mkt": 2}


class TestGroupAgg:
    """group_agg — single-pass Rust kernel."""

    records = [
        {"dept": "eng",  "salary": 120_000},
        {"dept": "mkt",  "salary":  85_000},
        {"dept": "eng",  "salary": 140_000},
        {"dept": "hr",   "salary":  75_000},
        {"dept": "mkt",  "salary":  90_000},
    ]

    def _by_key(self, rows):
        return {r["_key"]: r for r in rows}

    def test_count(self):
        result = Query(self.records).group_agg(
            lambda r: r["dept"],
            n=agg_count(),
        )
        bk = self._by_key(result)
        assert bk["eng"]["n"] == 2
        assert bk["mkt"]["n"] == 2
        assert bk["hr"]["n"]  == 1

    def test_field_key_count(self):
        result = Query(self.records).group_agg(
            field("dept"),
            n=agg_count(),
        )
        bk = self._by_key(result)
        assert bk["eng"]["n"] == 2
        assert bk["mkt"]["n"] == 2
        assert bk["hr"]["n"] == 1

    def test_field_key_after_field_filter(self):
        result = (
            Query(self.records)
            .filter(field("salary") > 80_000)
            .group_agg(field("dept"), n=agg_count())
        )
        bk = self._by_key(result)
        assert bk["eng"]["n"] == 2
        assert bk["mkt"]["n"] == 2
        assert "hr" not in bk

    def test_field_key_missing_groups_as_none(self):
        rows = self.records + [{"salary": 10_000}]
        result = Query(rows).group_agg(field("dept"), n=agg_count())
        bk = self._by_key(result)
        assert bk[None]["n"] == 1

    def test_field_key_int_and_bool_groups(self):
        rows = [
            {"bucket": 1, "active": True},
            {"bucket": 2, "active": False},
            {"bucket": 1, "active": True},
            {"bucket": 2, "active": True},
        ]

        by_bucket = self._by_key(Query(rows).group_agg(field("bucket"), n=agg_count()))
        assert by_bucket[1]["n"] == 2
        assert by_bucket[2]["n"] == 2

        by_active = self._by_key(Query(rows).group_agg(field("active"), n=agg_count()))
        assert by_active[True]["n"] == 3
        assert by_active[False]["n"] == 1

    def test_field_key_preserves_lambda_key_path(self):
        field_result = self._by_key(Query(self.records).group_agg(field("dept"), n=agg_count()))
        lambda_result = self._by_key(Query(self.records).group_agg(lambda r: r["dept"], n=agg_count()))
        assert field_result == lambda_result

    def test_field_key_rejects_callable_aggregates(self):
        with pytest.raises(ValueError):
            Query(self.records).group_agg(
                field("dept"),
                total=agg_sum(lambda r: r["salary"]),
            )

    def test_sum(self):
        result = Query(self.records).group_agg(
            lambda r: r["dept"],
            total=agg_sum(lambda r: r["salary"]),
        )
        bk = self._by_key(result)
        assert bk["eng"]["total"] == pytest.approx(260_000)
        assert bk["mkt"]["total"] == pytest.approx(175_000)
        assert bk["hr"]["total"]  == pytest.approx(75_000)

    def test_mean(self):
        result = Query(self.records).group_agg(
            lambda r: r["dept"],
            avg=agg_mean(lambda r: r["salary"]),
        )
        bk = self._by_key(result)
        assert bk["eng"]["avg"] == pytest.approx(130_000)
        assert bk["hr"]["avg"]  == pytest.approx(75_000)

    def test_max(self):
        result = Query(self.records).group_agg(
            lambda r: r["dept"],
            top=agg_max(lambda r: r["salary"]),
        )
        bk = self._by_key(result)
        assert bk["eng"]["top"] == pytest.approx(140_000)
        assert bk["mkt"]["top"] == pytest.approx(90_000)

    def test_min(self):
        result = Query(self.records).group_agg(
            lambda r: r["dept"],
            bottom=agg_min(lambda r: r["salary"]),
        )
        bk = self._by_key(result)
        assert bk["eng"]["bottom"] == pytest.approx(120_000)
        assert bk["mkt"]["bottom"] == pytest.approx(85_000)

    def test_multiple_specs(self):
        result = Query(self.records).group_agg(
            lambda r: r["dept"],
            count=agg_count(),
            total=agg_sum(lambda r: r["salary"]),
            avg=agg_mean(lambda r: r["salary"]),
        )
        bk = self._by_key(result)
        assert bk["eng"]["count"] == 2
        assert bk["eng"]["total"] == pytest.approx(260_000)
        assert bk["eng"]["avg"]   == pytest.approx(130_000)

    def test_insertion_order_preserved(self):
        result = Query(self.records).group_agg(
            lambda r: r["dept"],
            n=agg_count(),
        )
        # First group encountered should be "eng" (first record)
        assert result[0]["_key"] == "eng"

    def test_agg_spec_static_methods(self):
        # Verify AggSpec static constructors are callable
        assert AggSpec.count() is not None
        assert AggSpec.sum(lambda x: x) is not None
        assert AggSpec.mean(lambda x: x) is not None
        assert AggSpec.max(lambda x: x) is not None
        assert AggSpec.min(lambda x: x) is not None

    def test_matches_group_by_agg(self):
        """group_agg result must agree with the general group_by().agg() API."""
        fast = self._by_key(Query(self.records).group_agg(
            lambda r: r["dept"],
            count=agg_count(),
            total=agg_sum(lambda r: r["salary"]),
        ))
        from zpyflow import GroupBy
        slow = {
            row["_key"]: row
            for row in Query(self.records)
                .group_by(lambda r: r["dept"])
                .agg(
                    count=lambda g: g.count(),
                    total=lambda g: sum(r["salary"] for r in g.to_list()),
                )
        }
        for key in slow:
            assert fast[key]["count"] == slow[key]["count"]
            assert fast[key]["total"] == pytest.approx(slow[key]["total"])


# ===========================================================================
# from_arrow — buffer protocol + GIL-free memcpy
# ===========================================================================

try:
    import pyarrow as pa
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

@pytest.mark.skipif(not HAS_ARROW, reason="pyarrow not installed")
class TestFromArrow:
    def test_float64_filter_sum(self):
        from zpyflow import from_arrow
        arr = pa.array([1.0, -2.0, 3.0, -4.0, 5.0], type=pa.float64())
        result = from_arrow(arr).filter(col > 0).sum()
        assert result == pytest.approx(9.0)

    def test_float64_to_list(self):
        from zpyflow import from_arrow
        arr = pa.array([1.0, 2.0, 3.0], type=pa.float64())
        assert from_arrow(arr).to_list() == [1.0, 2.0, 3.0]

    def test_int64_filter_count(self):
        from zpyflow import from_arrow
        arr = pa.array([1, -2, 3, -4, 5], type=pa.int64())
        result = from_arrow(arr).filter(col > 0).count()
        assert result == 3

    def test_float32_cast_to_float64(self):
        from zpyflow import from_arrow
        arr = pa.array([1.0, 2.0, 3.0], type=pa.float32())
        assert from_arrow(arr).sum() == pytest.approx(6.0)

    def test_int32_cast_to_int64(self):
        from zpyflow import from_arrow
        arr = pa.array([10, 20, 30], type=pa.int32())
        assert from_arrow(arr).sum() == pytest.approx(60.0)

    def test_chunked_array(self):
        from zpyflow import from_arrow
        chunked = pa.chunked_array([[1.0, 2.0], [3.0, 4.0]])
        assert from_arrow(chunked).sum() == pytest.approx(10.0)

    def test_with_nulls_drop_first(self):
        from zpyflow import from_arrow
        arr = pa.array([1.0, None, 3.0], type=pa.float64())
        # null あり → Arrow 側で drop_null() してから渡すのが推奨パターン
        result = from_arrow(arr.drop_null()).sum()
        assert result == pytest.approx(4.0)

    def test_large_array_values_correct(self):
        from zpyflow import from_arrow
        import numpy as np
        n = 100_000
        np_arr = np.arange(n, dtype=np.float64)
        arr = pa.array(np_arr)
        result = from_arrow(arr).filter(col >= n / 2).count()
        assert result == n // 2


# ===========================================================================
# TestRustObj — GIL-free object (dict) pipeline via RustValue + ObjOp
# ===========================================================================

@pytest.fixture
def products():
    return [
        {"name": "apple",  "price": 1.20, "qty": 50,  "active": True},
        {"name": "banana", "price": 0.50, "qty": 100, "active": True},
        {"name": "cherry", "price": 3.00, "qty": 20,  "active": False},
        {"name": "date",   "price": 5.00, "qty": 10,  "active": True},
        {"name": "elderberry", "price": 8.00, "qty": 5, "active": False},
    ]


class TestRustObj:
    def test_auto_detect_list_of_dicts(self, products):
        # Dict lists stay as Obj (lazy); upgrade to RustObj on first field() DSL op
        q = Query(products)
        assert "rust_obj" not in repr(q)
        assert "obj" in repr(q)
        # After a field() DSL filter, it becomes ObjField (lazy extract+SIMD path)
        q2 = q.filter(field("price") > 1.0)
        assert "obj_field" in repr(q2)

    def test_to_list_roundtrip(self, products):
        result = Query(products).to_list()
        assert len(result) == 5
        assert result[0]["name"] == "apple"

    def test_filter_field_gt(self, products):
        result = Query(products).filter(field("price") > 2.0).to_list()
        assert [r["name"] for r in result] == ["cherry", "date", "elderberry"]

    def test_filter_field_ge(self, products):
        result = Query(products).filter(field("price") >= 3.0).to_list()
        assert len(result) == 3

    def test_filter_field_lt(self, products):
        result = Query(products).filter(field("price") < 1.0).to_list()
        assert [r["name"] for r in result] == ["banana"]

    def test_filter_field_le(self, products):
        result = Query(products).filter(field("price") <= 1.20).to_list()
        assert len(result) == 2

    def test_filter_field_eq_bool(self, products):
        result = Query(products).filter(field("active") == True).to_list()
        assert len(result) == 3

    def test_filter_field_ne_bool(self, products):
        result = Query(products).filter(field("active") != True).to_list()
        assert len(result) == 2

    def test_filter_field_between(self, products):
        result = Query(products).filter(field("price").between(1.0, 4.0)).to_list()
        assert [r["name"] for r in result] == ["apple", "cherry"]

    def test_chained_filters(self, products):
        result = (
            Query(products)
            .filter(field("active") == True)
            .filter(field("price") > 1.0)
            .to_list()
        )
        assert [r["name"] for r in result] == ["apple", "date"]

    def test_count_gil_free(self, products):
        n = Query(products).filter(field("price") > 2.0).count()
        assert n == 3

    def test_sum_field(self, products):
        total = Query(products).filter(field("active") == True).sum_field("price")
        assert total == pytest.approx(1.20 + 0.50 + 5.00)

    def test_sum_field_no_filter(self, products):
        total = Query(products).sum_field("price")
        assert total == pytest.approx(1.20 + 0.50 + 3.00 + 5.00 + 8.00)

    def test_any_field_expr(self, products):
        assert Query(products).any(field("price") > 7.0) is True
        assert Query(products).any(field("price") > 100.0) is False

    def test_all_field_expr(self, products):
        assert Query(products).all(field("price") > 0.0) is True
        assert Query(products).all(field("price") > 1.0) is False

    def test_skip_take(self, products):
        result = Query(products).filter(field("active") == True).skip(1).take(2).to_list()
        assert [r["name"] for r in result] == ["banana", "date"]

    def test_repr_lazy(self, products):
        # filter(field()) on fresh Obj uses ObjField fast path
        q = Query(products).filter(field("price") > 1.0)
        assert "obj_field" in repr(q)

    def test_preload(self, products):
        # preload() explicitly converts to RustObj (for multi-query reuse pattern)
        q = Query(products).preload()
        assert "rust_obj" in repr(q)
        assert "lazy" not in repr(q)
        assert q.filter(field("active") == True).count() == 3

    def test_lambda_filter_fallback(self, products):
        # Lambda forces materialization fallback — result must still be correct
        result = Query(products).filter(lambda r: r["price"] > 2.0).to_list()
        assert [r["name"] for r in result] == ["cherry", "date", "elderberry"]

    def test_map_fallback(self, products):
        result = Query(products).map(lambda r: r["price"]).to_list()
        assert result == pytest.approx([1.20, 0.50, 3.00, 5.00, 8.00])


# ===========================================================================
# TestFromCsv — Rust CSV parser (GIL-free)
# ===========================================================================

class TestFromCsv:
    CSV_HEADER = "name,price,qty\napple,1.20,50\nbanana,0.50,100\ncherry,3.00,20\n"

    def test_all_rows_as_dicts(self):
        from zpyflow import from_csv
        import io
        q = from_csv(io.StringIO(self.CSV_HEADER))
        rows = q.to_list()
        assert len(rows) == 3
        assert rows[0]["name"] == "apple"

    def test_column_by_name_float(self):
        from zpyflow import from_csv
        import io
        q = from_csv(io.StringIO(self.CSV_HEADER), column="price", dtype="float")
        assert q.to_list() == pytest.approx([1.20, 0.50, 3.00])

    def test_column_by_name_auto(self):
        from zpyflow import from_csv
        import io
        # qty column contains ints → should auto-detect as I64
        q = from_csv(io.StringIO(self.CSV_HEADER), column="qty")
        assert q.to_list() == [50, 100, 20]

    def test_column_by_index(self):
        from zpyflow import from_csv
        import io
        q = from_csv(io.StringIO(self.CSV_HEADER), column=1, dtype="float")
        assert q.to_list() == pytest.approx([1.20, 0.50, 3.00])

    def test_filter_after_csv(self):
        from zpyflow import from_csv, field
        import io
        q = from_csv(io.StringIO(self.CSV_HEADER))
        result = q.filter(field("price") > 1.0).to_list()
        assert [r["name"] for r in result] == ["apple", "cherry"]

    def test_count_after_csv(self):
        from zpyflow import from_csv, field
        import io
        n = from_csv(io.StringIO(self.CSV_HEADER)).filter(field("qty") >= 50).count()
        assert n == 2

    def test_no_header(self):
        from zpyflow import from_csv
        import io
        data = "apple,1.20\nbanana,0.50\n"
        q = from_csv(io.StringIO(data), has_header=False, column=1, dtype="float")
        assert q.to_list() == pytest.approx([1.20, 0.50])

    def test_custom_delimiter(self):
        from zpyflow import from_csv
        import io
        data = "name|price\napple|1.20\nbanana|0.50\n"
        q = from_csv(io.StringIO(data), delimiter="|", column="price", dtype="float")
        assert q.to_list() == pytest.approx([1.20, 0.50])

    def test_from_path(self, tmp_path):
        from zpyflow import from_csv
        p = tmp_path / "test.csv"
        p.write_text(self.CSV_HEADER, encoding="utf-8")
        rows = from_csv(p).to_list()
        assert len(rows) == 3
        assert rows[1]["name"] == "banana"

    def test_from_path_column(self, tmp_path):
        from zpyflow import from_csv
        p = tmp_path / "prices.csv"
        p.write_text(self.CSV_HEADER, encoding="utf-8")
        total = from_csv(p, column="price", dtype="float").sum()
        assert total == pytest.approx(1.20 + 0.50 + 3.00)


# ===========================================================================
# TestFromJsonLines — Rust JSONL parser (GIL-free)
# ===========================================================================

class TestFromJsonLines:
    JSONL = '{"name":"apple","price":1.20,"qty":50}\n{"name":"banana","price":0.50,"qty":100}\n{"name":"cherry","price":3.00,"qty":20}\n'

    def test_all_rows_as_dicts(self):
        from zpyflow import from_json_lines
        import io
        rows = from_json_lines(io.StringIO(self.JSONL)).to_list()
        assert len(rows) == 3
        assert rows[0]["name"] == "apple"

    def test_field_extraction_float(self):
        from zpyflow import from_json_lines
        import io
        q = from_json_lines(io.StringIO(self.JSONL), field="price", dtype="float")
        assert q.to_list() == pytest.approx([1.20, 0.50, 3.00])

    def test_field_extraction_auto_int(self):
        from zpyflow import from_json_lines
        import io
        q = from_json_lines(io.StringIO(self.JSONL), field="qty")
        assert q.to_list() == [50, 100, 20]

    def test_filter_after_jsonl(self):
        from zpyflow import from_json_lines, field
        import io
        result = from_json_lines(io.StringIO(self.JSONL)).filter(field("price") > 1.0).to_list()
        assert [r["name"] for r in result] == ["apple", "cherry"]

    def test_sum_field(self):
        from zpyflow import from_json_lines, field
        import io
        total = from_json_lines(io.StringIO(self.JSONL)).sum_field("price")
        assert total == pytest.approx(1.20 + 0.50 + 3.00)

    def test_skips_blank_lines(self):
        from zpyflow import from_json_lines
        import io
        data = '{"x":1}\n\n{"x":2}\n'
        q = from_json_lines(io.StringIO(data))
        assert q.count() == 2

    def test_from_path(self, tmp_path):
        from zpyflow import from_json_lines
        p = tmp_path / "data.jsonl"
        p.write_text(self.JSONL, encoding="utf-8")
        rows = from_json_lines(p).to_list()
        assert len(rows) == 3
        assert rows[2]["name"] == "cherry"

    def test_from_path_field_sum(self, tmp_path):
        from zpyflow import from_json_lines
        p = tmp_path / "data.jsonl"
        p.write_text(self.JSONL, encoding="utf-8")
        total = from_json_lines(p, field="price", dtype="float").sum()
        assert total == pytest.approx(4.70)


# ===========================================================================
# ZStream combinators
# ===========================================================================

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
        # After first non-matching element, all subsequent items are emitted even
        # if they would match the predicate again.
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
        # Should stay as numeric pipeline, not Py
        assert result.count() == 4

    def test_i64_plus_i64_gil_free(self):
        a = Query([1, 2, 3])
        b = Query([4, 5])
        assert a.chain(b).to_list() == [1, 2, 3, 4, 5]

    def test_f64_chain_with_filtered(self):
        a = Query(list(range(5))).filter(col >= 3)     # [3, 4]
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
        # Query([1,2,3]) uses F64 pipeline → to_list() yields floats → need int cast
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
        # 0→[], 1→[0], 2→[0,1]
        assert result == [0, 0, 1]

    def test_count(self):
        # Each element expands to 3 items; 4 × 3 = 12
        assert Query([1, 2, 3, 4]).flat_map(lambda x: [x, x, x]).count() == 12

    def test_f64_source(self):
        result = Query([1.0, 2.0]).flat_map(lambda x: [x, -x]).to_list()
        assert result == [1.0, -1.0, 2.0, -2.0]


# ===========================================================================
# Query.explain() — spec 023
# ===========================================================================

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


# ===========================================================================
# Arrow null and string paths — spec 027
# ===========================================================================

class TestArrowNullPaths:
    """Arrow null handling via from_arrow()."""

    @pytest.fixture(autouse=True)
    def skip_without_arrow(self):
        pytest.importorskip("pyarrow")

    def test_float64_null_free_fast_path(self):
        import pyarrow as pa
        from zpyflow import from_arrow, col
        arr = pa.array([1.0, 2.0, 3.0, 4.0], type=pa.float64())
        result = from_arrow(arr).filter(col > 2.0).to_list()
        assert result == [3.0, 4.0]

    def test_float64_with_nulls_become_nan(self):
        import math
        import pyarrow as pa
        from zpyflow import from_arrow
        arr = pa.array([1.0, None, 3.0], type=pa.float64())
        result = from_arrow(arr).to_list()
        assert len(result) == 3
        assert result[0] == pytest.approx(1.0)
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(3.0)

    def test_float64_nulls_filtered_by_nan_check(self):
        import pyarrow as pa
        from zpyflow import from_arrow, col
        arr = pa.array([1.0, None, 3.0, None, 5.0], type=pa.float64())
        # NaN != NaN by IEEE 754; filter(col == col) drops NaN
        result = from_arrow(arr).filter(col == col).to_list()
        assert result == pytest.approx([1.0, 3.0, 5.0])

    def test_float64_nulls_filtered_by_between(self):
        import pyarrow as pa
        from zpyflow import from_arrow, col
        arr = pa.array([1.0, None, 3.0], type=pa.float64())
        # between() naturally excludes NaN
        result = from_arrow(arr).filter(col.between(0.0, 10.0)).to_list()
        assert result == pytest.approx([1.0, 3.0])

    def test_int64_with_nulls_falls_back_to_pylist(self):
        import pyarrow as pa
        from zpyflow import from_arrow
        arr = pa.array([1, None, 3], type=pa.int64())
        result = from_arrow(arr).to_list()
        assert result == [1, None, 3]

    def test_string_array_falls_back_to_pylist(self):
        import pyarrow as pa
        from zpyflow import from_arrow
        arr = pa.array(["a", "b", "c"], type=pa.string())
        result = from_arrow(arr).to_list()
        assert result == ["a", "b", "c"]

    def test_int64_null_free_fast_path(self):
        import pyarrow as pa
        from zpyflow import from_arrow, col
        arr = pa.array([10, 20, 30], type=pa.int64())
        result = from_arrow(arr).filter(col > 15).to_list()
        assert result == [20, 30]


class TestMean:
    """mean() terminal — f64/i64/empty/obj paths and SIMD fused fast path."""

    def setup_method(self):
        try:
            from zpyflow import Query, col
            self.Query = Query
            self.col = col
            self.available = True
        except ImportError:
            self.available = False

    def _q(self, data):
        return self.Query(data)

    def test_f64_basic(self):
        if not self.available: pytest.skip("extension not built")
        result = self._q([1.0, 2.0, 3.0, 4.0]).mean()
        assert abs(result - 2.5) < 1e-9

    def test_f64_filter_fused(self):
        if not self.available: pytest.skip("extension not built")
        # Single filter op → SIMD fused path
        result = self._q([1.0, 2.0, 3.0, 4.0]).filter(self.col > 2.0).mean()
        assert abs(result - 3.5) < 1e-9

    def test_f64_filter_ge_fused(self):
        if not self.available: pytest.skip("extension not built")
        result = self._q([1.0, 2.0, 3.0]).filter(self.col >= 2.0).mean()
        assert abs(result - 2.5) < 1e-9

    def test_f64_filter_lt_fused(self):
        if not self.available: pytest.skip("extension not built")
        result = self._q([1.0, 2.0, 3.0, 4.0]).filter(self.col < 3.0).mean()
        assert abs(result - 1.5) < 1e-9

    def test_f64_filter_le_fused(self):
        if not self.available: pytest.skip("extension not built")
        result = self._q([1.0, 2.0, 3.0]).filter(self.col <= 2.0).mean()
        assert abs(result - 1.5) < 1e-9

    def test_f64_filter_between_fused(self):
        if not self.available: pytest.skip("extension not built")
        result = self._q([0.0, 1.0, 2.0, 3.0, 4.0]).filter(self.col.between(1.0, 3.0)).mean()
        assert abs(result - 2.0) < 1e-9

    def test_f64_empty_returns_none(self):
        if not self.available: pytest.skip("extension not built")
        assert self._q([]).mean() is None

    def test_f64_all_filtered_out_returns_none(self):
        if not self.available: pytest.skip("extension not built")
        result = self._q([1.0, 2.0, 3.0]).filter(self.col > 100.0).mean()
        assert result is None

    def test_f64_skip_take_scalar_path(self):
        if not self.available: pytest.skip("extension not built")
        # skip/take disables the fused path → scalar fallback
        result = self._q([1.0, 2.0, 3.0, 4.0]).skip(1).take(2).mean()
        assert abs(result - 2.5) < 1e-9

    def test_i64_basic(self):
        if not self.available: pytest.skip("extension not built")
        result = self._q([1, 2, 3, 4]).mean()
        assert abs(result - 2.5) < 1e-9

    def test_i64_empty_returns_none(self):
        if not self.available: pytest.skip("extension not built")
        result = self._q([]).mean()
        assert result is None

    def test_large_f64_correctness(self):
        if not self.available: pytest.skip("extension not built")
        data = list(range(1, 1001))  # 1..1000
        result = self._q([float(x) for x in data]).mean()
        assert abs(result - 500.5) < 1e-6


class TestVarStd:
    """var() and std() — population statistics, SIMD fused fast path."""

    def setup_method(self):
        try:
            from zpyflow import Query, col
            self.Query = Query
            self.col = col
            self.available = True
        except ImportError:
            self.available = False

    def _q(self, data):
        return self.Query(data)

    # --- var() ---

    def test_var_uniform(self):
        if not self.available: pytest.skip("extension not built")
        # [2, 4, 4, 4, 5, 5, 7, 9]: population var = 4.0
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = self._q(data).var()
        assert abs(result - 4.0) < 1e-9

    def test_var_single_element(self):
        if not self.available: pytest.skip("extension not built")
        assert abs(self._q([5.0]).var() - 0.0) < 1e-9

    def test_var_empty_returns_none(self):
        if not self.available: pytest.skip("extension not built")
        assert self._q([]).var() is None

    def test_var_all_filtered_returns_none(self):
        if not self.available: pytest.skip("extension not built")
        assert self._q([1.0, 2.0]).filter(self.col > 100.0).var() is None

    def test_var_filter_gt_fused(self):
        if not self.available: pytest.skip("extension not built")
        import statistics
        data = [float(x) for x in range(1, 101)]
        expected = statistics.pvariance(x for x in data if x > 50)
        result = self._q(data).filter(self.col > 50.0).var()
        assert abs(result - expected) < 1e-6

    def test_var_filter_between_fused(self):
        if not self.available: pytest.skip("extension not built")
        import statistics
        data = [float(x) for x in range(1, 101)]
        expected = statistics.pvariance(x for x in data if 25 <= x <= 75)
        result = self._q(data).filter(self.col.between(25.0, 75.0)).var()
        assert abs(result - expected) < 1e-6

    def test_var_nonnegative(self):
        if not self.available: pytest.skip("extension not built")
        data = [float(x) for x in range(1000)]
        result = self._q(data).filter(self.col > 0.0).var()
        assert result is not None and result >= 0.0

    # --- std() ---

    def test_std_uniform(self):
        if not self.available: pytest.skip("extension not built")
        # [2, 4, 4, 4, 5, 5, 7, 9]: population std = 2.0
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = self._q(data).std()
        assert abs(result - 2.0) < 1e-9

    def test_std_empty_returns_none(self):
        if not self.available: pytest.skip("extension not built")
        assert self._q([]).std() is None

    def test_std_single_element(self):
        if not self.available: pytest.skip("extension not built")
        assert abs(self._q([42.0]).std() - 0.0) < 1e-9

    def test_std_is_sqrt_of_var(self):
        if not self.available: pytest.skip("extension not built")
        import math
        data = [float(x) for x in range(1, 1001)]
        var = self._q(data).var()
        std = self._q(data).std()
        assert abs(std - math.sqrt(var)) < 1e-9

    def test_std_filter_fused(self):
        if not self.available: pytest.skip("extension not built")
        import statistics, math
        data = [float(x) for x in range(1, 101)]
        expected = math.sqrt(statistics.pvariance(x for x in data if x > 50))
        result = self._q(data).filter(self.col > 50.0).std()
        assert abs(result - expected) < 1e-6


class TestToNumpy:
    """to_numpy() — direct Vec→ndarray transfer, no Python float boxing."""

    def setup_method(self):
        try:
            from zpyflow import Query, col
            import numpy as np
            self.Query = Query
            self.col = col
            self.np = np
            self.available = True
        except ImportError:
            self.available = False

    def _q(self, data):
        return self.Query(data)

    def test_f64_returns_float64_array(self):
        if not self.available: pytest.skip("extension not built")
        np = self.np
        result = self._q([1.0, 2.0, 3.0]).to_numpy()
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert list(result) == [1.0, 2.0, 3.0]

    def test_f64_filter_values_correct(self):
        if not self.available: pytest.skip("extension not built")
        np = self.np
        data = [float(x) for x in range(10)]
        result = self._q(data).filter(self.col > 5.0).to_numpy()
        expected = np.array([6.0, 7.0, 8.0, 9.0])
        np.testing.assert_array_equal(result, expected)

    def test_f64_map_values_correct(self):
        if not self.available: pytest.skip("extension not built")
        np = self.np
        result = self._q([1.0, 2.0, 3.0]).map(self.col * 10.0).to_numpy()
        np.testing.assert_array_equal(result, np.array([10.0, 20.0, 30.0]))

    def test_i64_returns_int64_array(self):
        if not self.available: pytest.skip("extension not built")
        np = self.np
        result = self._q([1, 2, 3]).to_numpy()
        assert result.dtype == np.int64
        assert list(result) == [1, 2, 3]

    def test_empty_f64_returns_empty_array(self):
        if not self.available: pytest.skip("extension not built")
        np = self.np
        result = self._q([]).to_numpy()
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_result_is_writable(self):
        if not self.available: pytest.skip("extension not built")
        result = self._q([1.0, 2.0, 3.0]).to_numpy()
        result[0] = 99.0  # should not raise
        assert result[0] == 99.0

    def test_object_pipeline_raises(self):
        if not self.available: pytest.skip("extension not built")
        with pytest.raises(Exception):
            self._q(["a", "b"]).to_numpy()

    def test_large_correctness(self):
        if not self.available: pytest.skip("extension not built")
        np = self.np
        data = [float(i) for i in range(10000)]
        arr = self._q(data).filter(self.col > 5000.0).to_numpy()
        expected = np.array([x for x in data if x > 5000.0])
        np.testing.assert_array_equal(arr, expected)


# ===========================================================================
# spec/034 — chunked SIMD lazy path for LazyFloatList
# CHUNK_SIZE = 4096; chunked path activates when take*4 < N and N >= CHUNK_SIZE
# ===========================================================================

@pytest.mark.skipif(not HAS_EXTENSION, reason="extension not built")
class TestLazyFloatListChunkedPath:
    """Correctness tests for the chunked SIMD lazy path (spec/034)."""

    def test_lazy_float_list_take_small(self):
        # take * 4 < N and N >= CHUNK_SIZE → chunked lazy path
        # N=10000, take=10: 10*4=40 < 10000 ✓, 10000 >= 4096 ✓
        N = 10_000
        data = [float(i) for i in range(N)]
        result = Query(data).filter(col > 0).take(10).to_list()
        expected = [x for x in data if x > 0][:10]
        assert result == expected

    def test_lazy_float_list_take_small_with_skip(self):
        N = 10_000
        data = [float(i) for i in range(N)]
        result = Query(data).filter(col > 0).skip(5).take(10).to_list()
        expected = [x for x in data if x > 0][5:15]
        assert result == expected

    def test_lazy_float_list_take_large(self):
        # take >= N/4 → eager path (correctness same, different internal path)
        # N=1000, take=500: 500*4=2000 >= 1000 → eager path
        N = 1_000
        data = [float(i) for i in range(N)]
        result = Query(data).filter(col > 100).take(500).to_list()
        expected = [x for x in data if x > 100][:500]
        assert result == expected

    def test_lazy_float_list_chunked_boundary(self):
        # N exactly at CHUNK_SIZE boundary: N=4096, take=10 (10*4=40 < 4096 ✓)
        N = 4096
        data = [float(i % 200) for i in range(N)]
        result = Query(data).filter(col > 100).take(10).to_list()
        expected = [x for x in data if x > 100][:10]
        assert result == expected

    def test_lazy_float_list_small_n_eager(self):
        # N < CHUNK_SIZE → always eager path
        N = 100
        data = [float(i) for i in range(N)]
        result = Query(data).filter(col > 50).to_list()
        expected = [x for x in data if x > 50]
        assert result == expected


# ===========================================================================
# spec/035 T2 — lambda AST parsing: simple comparison/arithmetic lambdas
# are auto-promoted to DSL Expr for GIL-free SIMD execution
# ===========================================================================

@pytest.mark.skipif(not HAS_EXTENSION, reason="extension not built")
class TestLambdaAstParsing:
    """Lambda AST auto-promotion to DSL Expr (spec/035 T2)."""

    def test_filter_gt_lambda_promoted(self):
        data = [float(i) for i in range(10)]
        result = Query(data).filter(lambda x: x > 5).to_list()
        assert result == [6.0, 7.0, 8.0, 9.0]

    def test_filter_ge_lambda_promoted(self):
        data = [float(i) for i in range(10)]
        result = Query(data).filter(lambda x: x >= 5).to_list()
        assert result == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_filter_lt_lambda_promoted(self):
        data = [float(i) for i in range(10)]
        result = Query(data).filter(lambda x: x < 3).to_list()
        assert result == [0.0, 1.0, 2.0]

    def test_filter_eq_lambda_promoted(self):
        data = [1.0, 2.0, 3.0, 2.0, 1.0]
        result = Query(data).filter(lambda x: x == 2.0).to_list()
        assert result == [2.0, 2.0]

    def test_map_mul_lambda_promoted(self):
        data = [1.0, 2.0, 3.0]
        result = Query(data).map(lambda x: x * 2).to_list()
        assert result == pytest.approx([2.0, 4.0, 6.0])

    def test_map_add_lambda_promoted(self):
        data = [1.0, 2.0, 3.0]
        result = Query(data).map(lambda x: x + 10).to_list()
        assert result == pytest.approx([11.0, 12.0, 13.0])

    def test_chained_lambda_promotions(self):
        data = [float(i) for i in range(10)]
        result = Query(data).filter(lambda x: x > 2).map(lambda x: x * 2).to_list()
        expected = [x * 2 for x in data if x > 2]
        assert result == pytest.approx(expected)

    def test_lambda_not_promoted_with_take(self):
        # take is set → lambda should NOT be promoted (correctness invariant)
        data = [float(i) for i in range(10)]
        result = (
            Query(data)
            .filter(col > 2)
            .take(4)
            .filter(lambda x: x != 4.0)
            .to_list()
        )
        assert result == pytest.approx([3.0, 5.0, 6.0])

    def test_complex_lambda_falls_back(self):
        # Multi-condition lambda cannot be promoted; falls back to Python callable
        data = [float(i) for i in range(10)]
        result = Query(data).filter(lambda x: x > 2 and x < 7).to_list()
        assert result == [3.0, 4.0, 5.0, 6.0]


# ===========================================================================
# spec/035 T3 — mid-pipeline skip/take correctness
# ===========================================================================

class TestMidPipelineSkip:
    """filter(col>0).skip(N).map(col*2) must skip AFTER filter, not before."""

    def test_filter_then_skip_then_map_f64(self):
        # [1,2,...,9] after filter(>0); skip(5) → [6,7,8,9]; map(*2) → [12,14,16,18]
        data = [float(i) for i in range(10)]
        result = Query(data).filter(col > 0).skip(5).map(col * 2).to_list()
        expected = [x * 2 for x in data if x > 0][5:]
        assert result == pytest.approx(expected)

    def test_filter_then_take_then_map_f64(self):
        # filter(>0) → [1..9]; take(3) → [1,2,3]; map(*2) → [2,4,6]
        data = [float(i) for i in range(10)]
        result = Query(data).filter(col > 0).take(3).map(col * 2).to_list()
        expected = [x * 2 for x in [x for x in data if x > 0][:3]]
        assert result == pytest.approx(expected)

    def test_filter_then_skip_then_filter_f64(self):
        # filter(>0) → [1..9]; skip(3) → [4..9]; filter(<7) → [4,5,6]
        data = [float(i) for i in range(10)]
        result = Query(data).filter(col > 0).skip(3).filter(col < 7).to_list()
        filtered = [x for x in data if x > 0]
        expected = [x for x in filtered[3:] if x < 7]
        assert result == pytest.approx(expected)

    def test_skip_then_map_dsl_f64(self):
        # Verify a plain skip() + map() DSL combination also applies map post-skip
        data = [float(i) for i in range(10)]
        result = Query(data).skip(5).map(col * 2).to_list()
        expected = [x * 2 for x in data[5:]]
        assert result == pytest.approx(expected)

    def test_filter_then_skip_then_map_lazy_float_list(self):
        # Same as above but exercising LazyFloatList path (list[float] input)
        data = [float(i) for i in range(20)]
        result = Query(data).filter(col > 5).skip(3).map(col * 3).to_list()
        filtered = [x for x in data if x > 5]
        expected = [x * 3 for x in filtered[3:]]
        assert result == pytest.approx(expected)

    def test_skip_mid_pipeline_i64(self):
        # I64 path: filter + skip + map (int data)
        data = list(range(10))
        result = Query(data).filter(col > 0).skip(4).map(col * 2).to_list()
        filtered = [x for x in data if x > 0]
        expected = [x * 2 for x in filtered[4:]]
        assert result == expected


class TestInternalApiProtection:
    """Guard internal APIs used by __init__.py monkey-patches from silent removal."""

    def test_iter_parts_exists(self):
        # _iter_parts() is used by __init__.py to_list/count/__iter__ overrides.
        # This test breaks loudly if a Rust refactor removes or renames it.
        assert hasattr(Query([1.0]), "_iter_parts")

    def test_iter_parts_obj_returns_list(self):
        # For an Obj (Python-object) pipeline with a lambda, _iter_parts() must
        # return a list of [source, ops, skip, take].
        q = Query(["a", "b", "c"]).filter(lambda x: x != "b")
        parts = q._iter_parts()
        assert parts is not None
        assert isinstance(parts, list)
        assert len(parts) == 4  # [source, ops, skip, take]

    def test_iter_parts_numeric_returns_none(self):
        # For numeric/materialized pipelines, _iter_parts() must return None
        # (the monkey-patch falls back to Rust's to_list/count).
        assert Query([1.0, 2.0])._iter_parts() is None
        assert Query([1, 2])._iter_parts() is None


class TestNumericOpCollapsing:
    """Consecutive scalar map ops of the same kind must be folded into one."""

    def test_mul_mul_collapsed_f64(self):
        # map(col * 2).map(col * 3) must behave as map(col * 6)
        data = [1.0, 2.0, 3.0]
        result = Query(data).map(col * 2).map(col * 3).to_list()
        assert result == pytest.approx([6.0, 12.0, 18.0])

    def test_add_add_collapsed_f64(self):
        data = [1.0, 2.0, 3.0]
        result = Query(data).map(col + 10).map(col + 5).to_list()
        assert result == pytest.approx([16.0, 17.0, 18.0])

    def test_no_collapse_across_filter(self):
        # A filter between two maps must NOT be collapsed
        data = [1.0, 2.0, 3.0, 4.0]
        result = Query(data).map(col * 2).filter(col > 4).map(col * 3).to_list()
        expected = [x * 2 * 3 for x in data if x * 2 > 4]
        assert result == pytest.approx(expected)

    def test_mixed_kinds_not_collapsed(self):
        # MapMulScalar + MapAddScalar are different kinds — must NOT collapse
        data = [1.0, 2.0, 3.0]
        result = Query(data).map(col * 2).map(col + 1).to_list()
        assert result == pytest.approx([3.0, 5.0, 7.0])


class TestExplicitConstructors:
    """Query.f64() and Query.i64() force-coerce mixed lists onto the fast path."""

    def test_f64_mixed_list(self):
        result = Query.f64([1, 2, 3.0]).filter(col > 1).to_list()
        assert result == pytest.approx([2.0, 3.0])

    def test_f64_repr_is_f64(self):
        assert "f64" in repr(Query.f64([1, 2, 3]))

    def test_f64_invalid_raises(self):
        with pytest.raises((ValueError, TypeError)):
            Query.f64(["a", "b"]).to_list()

    def test_i64_mixed_list(self):
        result = Query.i64([1, 2, 3]).filter(col > 1).to_list()
        assert result == [2, 3]

    def test_i64_repr_is_i64(self):
        assert "i64" in repr(Query.i64([1, 2, 3]))

    def test_i64_invalid_raises(self):
        with pytest.raises((ValueError, TypeError)):
            Query.i64([1.5, 2.5]).to_list()

    def test_f64_sum(self):
        assert Query.f64([1, 2, 3]).sum() == pytest.approx(6.0)


class TestStats:
    """stats() returns count/sum/mean/min/max in one pass."""

    def test_basic_f64(self):
        s = Query([1.0, 2.0, 3.0, 4.0, 5.0]).stats()
        assert s["count"] == 5
        assert s["sum"] == pytest.approx(15.0)
        assert s["mean"] == pytest.approx(3.0)
        assert s["min"] == pytest.approx(1.0)
        assert s["max"] == pytest.approx(5.0)

    def test_filter_then_stats(self):
        s = Query([1.0, 2.0, 3.0, 4.0, 5.0]).filter(col > 2).stats()
        assert s["count"] == 3
        assert s["sum"] == pytest.approx(12.0)
        assert s["mean"] == pytest.approx(4.0)
        assert s["min"] == pytest.approx(3.0)
        assert s["max"] == pytest.approx(5.0)

    def test_empty_returns_none_fields(self):
        s = Query([1.0, 2.0]).filter(col > 100).stats()
        assert s["count"] == 0
        assert s["mean"] is None
        assert s["min"] is None
        assert s["max"] is None

    def test_keys_present(self):
        s = Query([1.0]).stats()
        assert set(s.keys()) == {"count", "sum", "mean", "min", "max"}

    def test_large_f64(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.arange(1_000_000, dtype=np.float64)
        s = from_numpy(arr).filter(col >= 0).stats()
        assert s["count"] == 1_000_000
        assert s["min"] == pytest.approx(0.0)
        assert s["max"] == pytest.approx(999_999.0)


class TestMapField:
    """map_field() extracts a single field from dict records."""

    records = [
        {"name": "alice", "age": 30, "score": 90.0},
        {"name": "bob",   "age": 25, "score": 80.0},
        {"name": "carol", "age": 35, "score": 95.0},
    ]

    def test_extract_string_field(self):
        names = Query(self.records).map_field("name").to_list()
        assert names == ["alice", "bob", "carol"]

    def test_extract_numeric_field(self):
        ages = Query(self.records).map_field("age").to_list()
        assert ages == [30, 25, 35]

    def test_filter_then_map_field(self):
        names = Query(self.records).filter(lambda r: r["age"] >= 30).map_field("name").to_list()
        assert names == ["alice", "carol"]

    def test_map_field_count(self):
        assert Query(self.records).map_field("score").count() == 3

    def test_map_field_with_field_dsl(self):
        from zpyflow import field
        scores = Query(self.records).filter(field("age") >= 30).map_field("score").to_list()
        assert scores == pytest.approx([90.0, 95.0])


class TestToDict:
    """to_dict(key, value) materialises the pipeline into a Python dict."""

    def test_f64_basic(self):
        data = [1.0, 2.0, 3.0]
        result = Query(data).to_dict(key=lambda x: int(x), value=lambda x: x * 10)
        assert result == {1: 10.0, 2: 20.0, 3: 30.0}

    def test_f64_with_filter(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = Query(data).filter(col > 2).to_dict(
            key=lambda x: int(x), value=lambda x: x ** 2
        )
        assert result == {3: 9.0, 4: 16.0, 5: 25.0}

    def test_obj_records(self):
        records = [
            {"id": 1, "name": "alice"},
            {"id": 2, "name": "bob"},
            {"id": 3, "name": "carol"},
        ]
        result = Query(records).to_dict(
            key=lambda r: r["id"], value=lambda r: r["name"]
        )
        assert result == {1: "alice", 2: "bob", 3: "carol"}

    def test_obj_with_filter(self):
        records = [
            {"id": 1, "score": 40.0},
            {"id": 2, "score": 80.0},
            {"id": 3, "score": 90.0},
        ]
        result = Query(records).filter(lambda r: r["score"] >= 80).to_dict(
            key=lambda r: r["id"], value=lambda r: r["score"]
        )
        assert result == {2: 80.0, 3: 90.0}

    def test_duplicate_keys_last_wins(self):
        data = [1.0, 1.0, 2.0]
        result = Query(data).to_dict(key=lambda x: int(x), value=lambda x: x)
        assert 1 in result and 2 in result  # last duplicate survives
        assert result[2] == 2.0

    def test_empty_returns_empty_dict(self):
        result = Query([]).to_dict(key=lambda x: x, value=lambda x: x)
        assert result == {}


# ---------------------------------------------------------------------------
# None-in-list behavior (spec 047)
# ---------------------------------------------------------------------------

class TestNoneInList:
    """Behavior when a Python list contains None values.

    Fix (spec-048): pyfloat_ob_fval now uses PyFloat_Check before PyFloat_AsDouble.
    Non-float elements (None, int, str) return NaN without touching exception state.

    ZPyFlow infers the pipeline type from the FIRST element:
    - first element is float → LazyFloatList:
        None → NaN → filtered out by DSL (NaN > 0 is False)
        lambda path: None becomes NaN (a Python float), lambda receives float('nan')
    - first element is None  → Obj fallback:
        lambda works; None is passed as a Python object to the callable
    """

    # ── LazyFloatList path (starts with float): None → NaN → filtered out ──

    def test_dsl_filter_converts_none_to_nan(self):
        """LazyFloatList + col > 0: None becomes NaN, which fails the filter (NaN > 0 is False)."""
        data = [1.0, None, 2.0]
        result = Query(data).filter(col > 0).to_list()
        assert result == [1.0, 2.0]

    def test_dsl_count_excludes_none(self):
        """count() with None in LazyFloatList: NaN is filtered out."""
        data = [1.0, None, 2.0, None, -1.0]
        result = Query(data).filter(col > 0).count()
        assert result == 2

    def test_lambda_filter_on_float_first_list(self):
        """LazyFloatList + lambda: None becomes NaN; lambda receives float('nan').
        `x is not None` is True for NaN, but `x > 0` is False — so NaN is filtered out.
        """
        data = [1.0, None, -1.0, None, 2.0]
        result = Query(data).filter(lambda x: x is not None and x > 0).to_list()
        assert result == [1.0, 2.0]

    # ── Obj path (starts with None): lambda works ──────────────────────────

    def test_none_first_falls_to_obj_path_lambda_works(self):
        """list[None, float]: first element is None → Obj path → lambda handles None."""
        data = [None, 1.0, -1.0]
        result = Query(data).filter(lambda x: x is not None and x > 0).to_list()
        assert result == [1.0]

    def test_none_first_mixed_lambda_skips_none(self):
        """Obj path: lambda with None guard returns only non-None positives."""
        data = [None, 1.0, None, -1.0, None, 2.0]
        result = Query(data).filter(lambda x: x is not None and x > 0).to_list()
        assert result == [1.0, 2.0]

    def test_none_first_count_works(self):
        """count() on Obj path with lambda guard correctly skips None."""
        data = [None, 1.0, None, -1.0, None, 2.0]
        result = Query(data).filter(lambda x: x is not None and x > 0).count()
        assert result == 2

    def test_all_none_list_returns_empty(self):
        """All-None list → Obj path → filter passes nothing → empty."""
        data = [None, None, None]
        result = Query(data).filter(lambda x: x is not None).to_list()
        assert result == []
