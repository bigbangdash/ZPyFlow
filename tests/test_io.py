"""I/O tests: from_numpy, from_arrow, from_csv, from_csv_chunked, from_json_lines."""

import pytest

try:
    from zpyflow import Query, col, field, AggSpec, agg_count, agg_sum, agg_mean, agg_max, agg_min
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False

try:
    import pyarrow as pa
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

pytestmark = pytest.mark.skipif(not HAS_EXTENSION, reason="Native extension not built")


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
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])[::2]
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

    def test_from_numpy_f64_direct_terminal_aggregations(self):
        import numpy as np
        from zpyflow import from_numpy

        arr = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        q = from_numpy(arr).filter(col >= 0).map(col * 2).skip(1).take(3)
        expected = np.array([2.0, 4.0, 6.0], dtype=np.float64)

        assert q.sum() == pytest.approx(float(expected.sum()))
        assert q.mean() == pytest.approx(float(expected.mean()))
        assert q.var() == pytest.approx(float(expected.var()))
        assert q.std() == pytest.approx(float(expected.std()))
        assert q.min() == pytest.approx(float(expected.min()))
        assert q.max() == pytest.approx(float(expected.max()))

        s = q.stats()
        assert s["count"] == len(expected)
        assert s["sum"] == pytest.approx(float(expected.sum()))
        assert s["mean"] == pytest.approx(float(expected.mean()))
        assert s["min"] == pytest.approx(float(expected.min()))
        assert s["max"] == pytest.approx(float(expected.max()))

    def test_from_numpy_f64_direct_terminal_empty(self):
        import numpy as np
        from zpyflow import from_numpy

        q = from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float64)).filter(col > 10)
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

    def test_from_numpy_f32_map(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = from_numpy(arr).map(col * 2).to_list()
        assert result == pytest.approx([2.0, 4.0, 6.0])

    def test_from_numpy_f32_filter_values(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = from_numpy(arr).filter(col > 2).to_list()
        assert out == pytest.approx([3.0, 4.0])

    def test_from_numpy_f32_large(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.arange(1_000_000, dtype=np.float32)
        count = from_numpy(arr).filter(col > 500_000).count()
        assert count == 499_999

    def test_from_numpy_f32_not_upcasted(self):
        import numpy as np
        from zpyflow import from_numpy
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        q = from_numpy(arr)
        assert "numpy_f32" in repr(q)


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
        result = from_arrow(arr).filter(col == col).to_list()
        assert result == pytest.approx([1.0, 3.0, 5.0])

    def test_float64_nulls_filtered_by_between(self):
        import pyarrow as pa
        from zpyflow import from_arrow, col
        arr = pa.array([1.0, None, 3.0], type=pa.float64())
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


@pytest.mark.skipif(not HAS_ARROW, reason="pyarrow not installed")
class TestFromArrowIpc:
    def _write_ipc_file(self, tmp_path, table, name="data.arrow"):
        import pyarrow.ipc as ipc
        p = tmp_path / name
        with ipc.new_file(str(p), table.schema) as w:
            for batch in table.to_batches():
                w.write_batch(batch)
        return p

    def _write_ipc_stream(self, tmp_path, table, name="data.arrows"):
        import pyarrow.ipc as ipc
        p = tmp_path / name
        with ipc.new_stream(str(p), table.schema) as w:
            for batch in table.to_batches():
                w.write_batch(batch)
        return p

    def test_single_column_f64_file_format(self, tmp_path):
        import pyarrow as pa
        from zpyflow import from_arrow_ipc
        table = pa.table({"x": pa.array([1.0, 2.0, 3.0, 4.0, 5.0])})
        p = self._write_ipc_file(tmp_path, table)
        result = from_arrow_ipc(p).to_list()
        assert result == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_single_column_f64_stream_format(self, tmp_path):
        import pyarrow as pa
        from zpyflow import from_arrow_ipc
        table = pa.table({"x": pa.array([10.0, 20.0, 30.0])})
        p = self._write_ipc_stream(tmp_path, table)
        result = from_arrow_ipc(p).to_list()
        assert result == pytest.approx([10.0, 20.0, 30.0])

    def test_single_column_auto_extract_filter(self, tmp_path):
        import pyarrow as pa
        from zpyflow import from_arrow_ipc, col
        table = pa.table({"v": pa.array([1.0, 2.0, 3.0, 4.0, 5.0])})
        p = self._write_ipc_file(tmp_path, table)
        count = from_arrow_ipc(p).filter(col > 3.0).count()
        assert count == 2

    def test_multi_column_dict_rows(self, tmp_path):
        import pyarrow as pa
        from zpyflow import from_arrow_ipc
        table = pa.table({"name": ["alice", "bob"], "score": [0.9, 0.4]})
        p = self._write_ipc_file(tmp_path, table)
        rows = from_arrow_ipc(p).to_list()
        assert len(rows) == 2
        assert rows[0]["name"] == "alice"
        assert rows[0]["score"] == pytest.approx(0.9)

    def test_column_by_name(self, tmp_path):
        import pyarrow as pa
        from zpyflow import from_arrow_ipc
        table = pa.table({"a": [1.0, 2.0], "b": [10.0, 20.0]})
        p = self._write_ipc_file(tmp_path, table)
        result = from_arrow_ipc(p, column="b").to_list()
        assert result == pytest.approx([10.0, 20.0])

    def test_column_by_index(self, tmp_path):
        import pyarrow as pa
        from zpyflow import from_arrow_ipc
        table = pa.table({"a": [1.0, 2.0], "b": [10.0, 20.0]})
        p = self._write_ipc_file(tmp_path, table)
        result = from_arrow_ipc(p, column=1).to_list()
        assert result == pytest.approx([10.0, 20.0])

    def test_i64_column_zero_copy(self, tmp_path):
        import pyarrow as pa
        from zpyflow import from_arrow_ipc, col
        table = pa.table({"n": pa.array([10, 20, 30, 40], type=pa.int64())})
        p = self._write_ipc_file(tmp_path, table)
        result = from_arrow_ipc(p).filter(col > 15).to_list()
        assert result == [20, 30, 40]

    def test_path_string_input(self, tmp_path):
        import pyarrow as pa
        from zpyflow import from_arrow_ipc
        table = pa.table({"x": [1.0, 2.0, 3.0]})
        p = self._write_ipc_file(tmp_path, table)
        result = from_arrow_ipc(str(p)).to_list()
        assert result == pytest.approx([1.0, 2.0, 3.0])

    def test_missing_pyarrow_raises(self, tmp_path, monkeypatch):
        import sys
        from zpyflow import from_arrow_ipc
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        monkeypatch.setitem(sys.modules, "pyarrow.ipc", None)
        with pytest.raises((ImportError, TypeError)):
            from_arrow_ipc(tmp_path / "nonexistent.arrow")


class TestFromCsvChunked:
    CSV_WITH_HEADER = "name,score,active\nalice,0.9,true\nbob,0.4,false\ncarol,0.7,true\ndan,0.2,false\neve,0.8,true\n"
    CSV_NO_HEADER = "alice,0.9\nbob,0.4\ncarol,0.7\n"

    def test_chunk_boundaries(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_WITH_HEADER, encoding="utf-8")
        chunks = list(from_csv_chunked(p, chunk_size=2))
        assert len(chunks) == 3
        assert chunks[0].count() == 2
        assert chunks[1].count() == 2
        assert chunks[2].count() == 1

    def test_total_row_count(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_WITH_HEADER, encoding="utf-8")
        total = sum(q.count() for q in from_csv_chunked(p, chunk_size=2))
        assert total == 5

    def test_dict_rows_default(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_WITH_HEADER, encoding="utf-8")
        chunks = list(from_csv_chunked(p, chunk_size=10))
        assert len(chunks) == 1
        rows = chunks[0].to_list()
        assert rows[0]["name"] == "alice"
        assert rows[0]["score"] == pytest.approx(0.9)

    def test_column_by_name(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_WITH_HEADER, encoding="utf-8")
        values = []
        for q in from_csv_chunked(p, chunk_size=2, column="score"):
            values.extend(q.to_list())
        assert values == pytest.approx([0.9, 0.4, 0.7, 0.2, 0.8])

    def test_column_by_index(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_WITH_HEADER, encoding="utf-8")
        values = []
        for q in from_csv_chunked(p, chunk_size=3, column=1):
            values.extend(q.to_list())
        assert values == pytest.approx([0.9, 0.4, 0.7, 0.2, 0.8])

    def test_dtype_float(self, tmp_path):
        from zpyflow import from_csv_chunked, col
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_WITH_HEADER, encoding="utf-8")
        total = sum(
            q.filter(col > 0.5).sum()
            for q in from_csv_chunked(p, chunk_size=2, column="score", dtype="float")
        )
        assert total == pytest.approx(0.9 + 0.7 + 0.8)

    def test_field_dsl_filter(self, tmp_path):
        from zpyflow import from_csv_chunked, field
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_WITH_HEADER, encoding="utf-8")
        results = []
        for q in from_csv_chunked(p, chunk_size=2):
            results.extend(q.filter(field("score") > 0.5).to_list())
        names = [r["name"] for r in results]
        assert names == ["alice", "carol", "eve"]

    def test_file_like_input(self):
        import io
        from zpyflow import from_csv_chunked
        f = io.StringIO(self.CSV_WITH_HEADER)
        chunks = list(from_csv_chunked(f, chunk_size=3))
        assert len(chunks) == 2
        assert sum(q.count() for q in chunks) == 5

    def test_no_header_mode(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_NO_HEADER, encoding="utf-8")
        chunks = list(from_csv_chunked(p, has_header=False))
        assert len(chunks) == 1
        rows = chunks[0].to_list()
        assert rows[0][0] == "alice"
        assert rows[0][1] == pytest.approx(0.9)

    def test_no_header_column_by_index(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_NO_HEADER, encoding="utf-8")
        values = []
        for q in from_csv_chunked(p, has_header=False, column=1, dtype="float"):
            values.extend(q.to_list())
        assert values == pytest.approx([0.9, 0.4, 0.7])

    def test_invalid_chunk_size(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_WITH_HEADER, encoding="utf-8")
        with pytest.raises(ValueError, match="chunk_size"):
            list(from_csv_chunked(p, chunk_size=0))

    def test_missing_column_name_raises(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_WITH_HEADER, encoding="utf-8")
        with pytest.raises(ValueError, match="not found"):
            list(from_csv_chunked(p, column="nonexistent"))

    def test_empty_csv_yields_nothing(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text("name,score\n", encoding="utf-8")
        chunks = list(from_csv_chunked(p))
        assert chunks == []

    def test_single_row_csv(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text("x\n42\n", encoding="utf-8")
        chunks = list(from_csv_chunked(p, chunk_size=100))
        assert len(chunks) == 1
        assert chunks[0].to_list() == [{"x": 42}]

    def test_custom_delimiter(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.tsv"
        p.write_text("name\tval\nalice\t1.5\nbob\t2.5\n", encoding="utf-8")
        chunks = list(from_csv_chunked(p, delimiter="\t"))
        rows = chunks[0].to_list()
        assert rows[0]["name"] == "alice"
        assert rows[1]["val"] == pytest.approx(2.5)

    def test_type_coercion_int(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text("id,count\n1,10\n2,20\n", encoding="utf-8")
        chunks = list(from_csv_chunked(p))
        rows = chunks[0].to_list()
        assert isinstance(rows[0]["id"], int)
        assert isinstance(rows[0]["count"], int)

    def test_from_path_string(self, tmp_path):
        from zpyflow import from_csv_chunked
        p = tmp_path / "data.csv"
        p.write_text(self.CSV_WITH_HEADER, encoding="utf-8")
        chunks = list(from_csv_chunked(str(p)))
        assert sum(q.count() for q in chunks) == 5
