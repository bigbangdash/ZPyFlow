"""Numeric fast-path tests: f64, i64, DSL, SIMD, stats, etc."""

import pytest

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
# Expression DSL
# ===========================================================================

class TestExprDSL:
    def test_col_proxy_operators(self):
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
# Regression: any() / all() must short-circuit
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
        assert not Query(data).take(2).any(lambda x: x > 3)

    def test_all_respects_take(self):
        data = [1.0, 2.0, 3.0, 4.0]
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
        data = list(range(1, 1001))
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

    def test_var_uniform(self):
        if not self.available: pytest.skip("extension not built")
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

    def test_std_uniform(self):
        if not self.available: pytest.skip("extension not built")
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
    """to_numpy() — runtime numpy import, raw-byte transfer, no Python float boxing."""

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
        result[0] = 99.0
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


@pytest.mark.skipif(not HAS_EXTENSION, reason="extension not built")
class TestLazyFloatListChunkedPath:
    """Correctness tests for the chunked SIMD lazy path (spec/034)."""

    def test_lazy_float_list_take_small(self):
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
        N = 1_000
        data = [float(i) for i in range(N)]
        result = Query(data).filter(col > 100).take(500).to_list()
        expected = [x for x in data if x > 100][:500]
        assert result == expected

    def test_lazy_float_list_chunked_boundary(self):
        N = 4096
        data = [float(i % 200) for i in range(N)]
        result = Query(data).filter(col > 100).take(10).to_list()
        expected = [x for x in data if x > 100][:10]
        assert result == expected

    def test_lazy_float_list_small_n_eager(self):
        N = 100
        data = [float(i) for i in range(N)]
        result = Query(data).filter(col > 50).to_list()
        expected = [x for x in data if x > 50]
        assert result == expected


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
        data = [float(i) for i in range(10)]
        result = Query(data).filter(lambda x: x > 2 and x < 7).to_list()
        assert result == [3.0, 4.0, 5.0, 6.0]


class TestMidPipelineSkip:
    """filter(col>0).skip(N).map(col*2) must skip AFTER filter, not before."""

    def test_filter_then_skip_then_map_f64(self):
        data = [float(i) for i in range(10)]
        result = Query(data).filter(col > 0).skip(5).map(col * 2).to_list()
        expected = [x * 2 for x in data if x > 0][5:]
        assert result == pytest.approx(expected)

    def test_filter_then_take_then_map_f64(self):
        data = [float(i) for i in range(10)]
        result = Query(data).filter(col > 0).take(3).map(col * 2).to_list()
        expected = [x * 2 for x in [x for x in data if x > 0][:3]]
        assert result == pytest.approx(expected)

    def test_filter_then_skip_then_filter_f64(self):
        data = [float(i) for i in range(10)]
        result = Query(data).filter(col > 0).skip(3).filter(col < 7).to_list()
        filtered = [x for x in data if x > 0]
        expected = [x for x in filtered[3:] if x < 7]
        assert result == pytest.approx(expected)

    def test_skip_then_map_dsl_f64(self):
        data = [float(i) for i in range(10)]
        result = Query(data).skip(5).map(col * 2).to_list()
        expected = [x * 2 for x in data[5:]]
        assert result == pytest.approx(expected)

    def test_filter_then_skip_then_map_lazy_float_list(self):
        data = [float(i) for i in range(20)]
        result = Query(data).filter(col > 5).skip(3).map(col * 3).to_list()
        filtered = [x for x in data if x > 5]
        expected = [x * 3 for x in filtered[3:]]
        assert result == pytest.approx(expected)

    def test_skip_mid_pipeline_i64(self):
        data = list(range(10))
        result = Query(data).filter(col > 0).skip(4).map(col * 2).to_list()
        filtered = [x for x in data if x > 0]
        expected = [x * 2 for x in filtered[4:]]
        assert result == expected


class TestInternalApiProtection:
    """Guard internal APIs used by __init__.py monkey-patches from silent removal."""

    def test_iter_parts_exists(self):
        assert hasattr(Query([1.0]), "_iter_parts")

    def test_iter_parts_obj_returns_list(self):
        q = Query(["a", "b", "c"]).filter(lambda x: x != "b")
        parts = q._iter_parts()
        assert parts is not None
        assert isinstance(parts, list)
        assert len(parts) == 4

    def test_iter_parts_numeric_returns_none(self):
        assert Query([1.0, 2.0])._iter_parts() is None
        assert Query([1, 2])._iter_parts() is None


class TestNumericOpCollapsing:
    """Consecutive scalar map ops of the same kind must be folded into one."""

    def test_mul_mul_collapsed_f64(self):
        data = [1.0, 2.0, 3.0]
        result = Query(data).map(col * 2).map(col * 3).to_list()
        assert result == pytest.approx([6.0, 12.0, 18.0])

    def test_add_add_collapsed_f64(self):
        data = [1.0, 2.0, 3.0]
        result = Query(data).map(col + 10).map(col + 5).to_list()
        assert result == pytest.approx([16.0, 17.0, 18.0])

    def test_no_collapse_across_filter(self):
        data = [1.0, 2.0, 3.0, 4.0]
        result = Query(data).map(col * 2).filter(col > 4).map(col * 3).to_list()
        expected = [x * 2 * 3 for x in data if x * 2 > 4]
        assert result == pytest.approx(expected)

    def test_mixed_kinds_not_collapsed(self):
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

    def test_f64_direct_terminal_aggregations_with_bounds(self):
        q = Query.f64([-2, -1, 0, 1, 2, 3, 4]).filter(col >= 0).map(col * 2).skip(1).take(3)
        expected = [2.0, 4.0, 6.0]

        assert q.sum() == pytest.approx(sum(expected))
        assert q.mean() == pytest.approx(sum(expected) / len(expected))
        assert q.var() == pytest.approx(8.0 / 3.0)
        assert q.std() == pytest.approx((8.0 / 3.0) ** 0.5)
        assert q.min() == pytest.approx(2.0)
        assert q.max() == pytest.approx(6.0)

        s = q.stats()
        assert s["count"] == 3
        assert s["sum"] == pytest.approx(12.0)
        assert s["mean"] == pytest.approx(4.0)
        assert s["min"] == pytest.approx(2.0)
        assert s["max"] == pytest.approx(6.0)

    def test_f64_direct_terminal_empty_with_bounds(self):
        q = Query.f64([1, 2, 3]).filter(col > 10).skip(1).take(2)
        assert q.sum() == pytest.approx(0.0)
        assert q.mean() is None
        assert q.var() is None
        assert q.std() is None
        assert q.min() is None
        assert q.max() is None
        assert q.stats() == {
            "count": 0,
            "sum": 0.0,
            "mean": None,
            "min": None,
            "max": None,
        }


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

    def test_lazy_float_direct_terminal_aggregations_with_bounds(self):
        q = Query([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]).filter(col >= 0).map(col * 2).skip(1).take(3)
        expected = [2.0, 4.0, 6.0]

        assert q.sum() == pytest.approx(sum(expected))
        assert q.mean() == pytest.approx(4.0)
        assert q.var() == pytest.approx(8.0 / 3.0)
        assert q.std() == pytest.approx((8.0 / 3.0) ** 0.5)
        assert q.min() == pytest.approx(2.0)
        assert q.max() == pytest.approx(6.0)

        s = q.stats()
        assert s["count"] == 3
        assert s["sum"] == pytest.approx(12.0)
        assert s["mean"] == pytest.approx(4.0)
        assert s["min"] == pytest.approx(2.0)
        assert s["max"] == pytest.approx(6.0)

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


class TestNumericDSLExtension:
    """col.clamp / log / log2 / log10 / exp / sigmoid"""

    def test_clamp_basic(self):
        data = [-2.0, 0.0, 1.5, 3.0, 5.0]
        result = Query(data).map(col.clamp(0.0, 3.0)).to_list()
        assert result == [0.0, 0.0, 1.5, 3.0, 3.0]

    def test_clamp_after_filter(self):
        data = list(range(-5, 6))
        result = Query(data).filter(col >= -3).map(col.clamp(-1.0, 1.0)).to_list()
        assert all(-1.0 <= v <= 1.0 for v in result)

    def test_log_basic(self):
        import math
        data = [1.0, math.e, math.e ** 2]
        result = Query(data).map(col.log()).to_list()
        assert abs(result[0] - 0.0) < 1e-10
        assert abs(result[1] - 1.0) < 1e-10
        assert abs(result[2] - 2.0) < 1e-10

    def test_log2(self):
        data = [1.0, 2.0, 4.0, 8.0]
        result = Query(data).map(col.log2()).to_list()
        assert result == [0.0, 1.0, 2.0, 3.0]

    def test_log10(self):
        data = [1.0, 10.0, 100.0]
        result = Query(data).map(col.log10()).to_list()
        assert result == [0.0, 1.0, 2.0]

    def test_exp_basic(self):
        import math
        data = [0.0, 1.0, 2.0]
        result = Query(data).map(col.exp()).to_list()
        assert abs(result[0] - 1.0) < 1e-10
        assert abs(result[1] - math.e) < 1e-10

    def test_sigmoid_bounds(self):
        data = [-100.0, 0.0, 100.0]
        result = Query(data).map(col.sigmoid()).to_list()
        assert result[0] < 0.01
        assert abs(result[1] - 0.5) < 1e-10
        assert result[2] > 0.99

    def test_sigmoid_monotone(self):
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        result = Query(data).map(col.sigmoid()).to_list()
        assert all(result[i] < result[i+1] for i in range(len(result)-1))

    def test_chained_log_exp_roundtrip(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = Query(data).map(col.log()).map(col.exp()).to_list()
        for orig, got in zip(data, result):
            assert abs(got - orig) < 1e-10

    def test_clamp_empty(self):
        assert Query([]).map(col.clamp(0.0, 1.0)).to_list() == []

    def test_expr_log_callable(self):
        """Expr should be callable as a Python function."""
        expr = col.log()
        import math
        assert abs(expr(math.e) - 1.0) < 1e-10


class TestModFloorDiv:
    """col % n and col // n — modulo and floor division DSL operators."""

    def test_mod_basic(self):
        data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        result = Query(data).map(col % 3).to_list()
        assert result == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]

    def test_mod_filter(self):
        data = list(range(10))
        result = Query(data).filter(lambda x: x % 2 == 0).to_list()
        assert result == [0, 2, 4, 6, 8]

    def test_floordiv_basic(self):
        data = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        result = Query(data).map(col // 3).to_list()
        assert result == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0]

    def test_floordiv_after_filter(self):
        data = list(range(12))
        result = Query(data).filter(col >= 6).map(col // 3).to_list()
        assert result == [2.0, 2.0, 2.0, 3.0, 3.0, 3.0]

    def test_mod_empty(self):
        assert Query([]).map(col % 2).to_list() == []

    def test_mod_callable(self):
        expr = col % 3
        assert expr(5.0) == 2.0
        assert expr(6.0) == 0.0

    def test_floordiv_callable(self):
        expr = col // 2
        assert expr(7.0) == 3.0
        assert expr(8.0) == 4.0

    def test_chained_mod_filter(self):
        data = list(range(10))
        evens = Query(data).filter(lambda x: x % 2 == 0).to_list()
        odds  = Query(data).filter(lambda x: x % 2 != 0).to_list()
        assert evens == [0, 2, 4, 6, 8]
        assert odds  == [1, 3, 5, 7, 9]


class TestNumericFilterDSL:
    """col.is_nan() / col.not_nan() / col.is_finite() / col.is_inf()"""

    def test_is_nan_filter(self):
        data = [1.0, float("nan"), 2.0, float("nan"), 3.0]
        result = Query(data).filter(col.is_nan()).to_list()
        assert len(result) == 2
        assert all(v != v for v in result)

    def test_not_nan_filter(self):
        data = [1.0, float("nan"), 2.0, float("nan"), 3.0]
        result = Query(data).filter(col.not_nan()).to_list()
        assert result == [1.0, 2.0, 3.0]

    def test_is_finite_excludes_nan_and_inf(self):
        data = [1.0, float("nan"), float("inf"), float("-inf"), 2.0]
        result = Query(data).filter(col.is_finite()).to_list()
        assert result == [1.0, 2.0]

    def test_is_inf_filter(self):
        data = [1.0, float("inf"), 2.0, float("-inf"), 3.0]
        result = Query(data).filter(col.is_inf()).to_list()
        assert len(result) == 2
        assert all(v == float("inf") or v == float("-inf") for v in result)

    def test_not_nan_empty(self):
        assert Query([]).filter(col.not_nan()).to_list() == []

    def test_is_finite_all_clean(self):
        data = [1.0, 2.0, 3.0]
        result = Query(data).filter(col.is_finite()).to_list()
        assert result == data

    def test_chain_not_nan_then_filter(self):
        data = [1.0, float("nan"), 5.0, float("nan"), 10.0]
        result = Query(data).filter(col.not_nan()).filter(col > 3).to_list()
        assert result == [5.0, 10.0]
