"""Object/dict pipeline tests: RustObj, MapField, ToDict, StringDSL, Join, SetField, etc."""

import pytest

try:
    from zpyflow import Query, col, field, AggSpec, agg_count, agg_sum, agg_mean, agg_max, agg_min
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False

pytestmark = pytest.mark.skipif(not HAS_EXTENSION, reason="Native extension not built")


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


class TestRustObj:
    def test_auto_detect_list_of_dicts(self, products):
        q = Query(products)
        assert "rust_obj" not in repr(q)
        assert "obj" in repr(q)
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
        q = Query(products).filter(field("price") > 1.0)
        assert "obj_field" in repr(q)

    def test_preload(self, products):
        q = Query(products).preload()
        assert "rust_obj" in repr(q)
        assert "lazy" not in repr(q)
        assert q.filter(field("active") == True).count() == 3

    def test_lambda_filter_fallback(self, products):
        result = Query(products).filter(lambda r: r["price"] > 2.0).to_list()
        assert [r["name"] for r in result] == ["cherry", "date", "elderberry"]

    def test_map_fallback(self, products):
        result = Query(products).map(lambda r: r["price"]).to_list()
        assert result == pytest.approx([1.20, 0.50, 3.00, 5.00, 8.00])


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
        assert 1 in result and 2 in result
        assert result[2] == 2.0

    def test_empty_returns_empty_dict(self):
        result = Query([]).to_dict(key=lambda x: x, value=lambda x: x)
        assert result == {}


class TestNoneInList:
    """Behavior when a Python list contains None values (spec 047/048)."""

    def test_dsl_filter_converts_none_to_nan(self):
        data = [1.0, None, 2.0]
        result = Query(data).filter(col > 0).to_list()
        assert result == [1.0, 2.0]

    def test_dsl_count_excludes_none(self):
        data = [1.0, None, 2.0, None, -1.0]
        result = Query(data).filter(col > 0).count()
        assert result == 2

    def test_lambda_filter_on_float_first_list(self):
        data = [1.0, None, -1.0, None, 2.0]
        result = Query(data).filter(lambda x: x is not None and x > 0).to_list()
        assert result == [1.0, 2.0]

    def test_none_first_falls_to_obj_path_lambda_works(self):
        data = [None, 1.0, -1.0]
        result = Query(data).filter(lambda x: x is not None and x > 0).to_list()
        assert result == [1.0]

    def test_none_first_mixed_lambda_skips_none(self):
        data = [None, 1.0, None, -1.0, None, 2.0]
        result = Query(data).filter(lambda x: x is not None and x > 0).to_list()
        assert result == [1.0, 2.0]

    def test_none_first_count_works(self):
        data = [None, 1.0, None, -1.0, None, 2.0]
        result = Query(data).filter(lambda x: x is not None and x > 0).count()
        assert result == 2

    def test_all_none_list_returns_empty(self):
        data = [None, None, None]
        result = Query(data).filter(lambda x: x is not None).to_list()
        assert result == []


class TestStringDSL:
    """field().startswith / endswith / contains / matches"""

    @pytest.fixture
    def records(self):
        return [
            {"name": "alice", "role": "admin"},
            {"name": "bob", "role": "user"},
            {"name": "charlie", "role": "admin"},
            {"name": "dave", "role": "moderator"},
            {"name": "alice2", "role": "user"},
        ]

    def test_startswith(self, records):
        result = Query(records).filter(field("name").startswith("al")).to_list()
        assert [r["name"] for r in result] == ["alice", "alice2"]

    def test_startswith_no_match(self, records):
        result = Query(records).filter(field("name").startswith("z")).to_list()
        assert result == []

    def test_endswith(self, records):
        result = Query(records).filter(field("name").endswith("2")).to_list()
        assert [r["name"] for r in result] == ["alice2"]

    def test_endswith_empty(self):
        assert Query([]).filter(field("name").endswith("x")).to_list() == []

    def test_contains(self, records):
        result = Query(records).filter(field("name").contains("li")).to_list()
        assert [r["name"] for r in result] == ["alice", "charlie", "alice2"]

    def test_contains_role(self, records):
        result = Query(records).filter(field("role").contains("min")).to_list()
        assert all(r["role"] == "admin" for r in result)
        assert len(result) == 2

    def test_matches_digit(self):
        data = [{"id": "abc123"}, {"id": "xyz"}, {"id": "456def"}]
        result = Query(data).filter(field("id").matches(r"\d")).to_list()
        assert [r["id"] for r in result] == ["abc123", "456def"]

    def test_matches_anchored(self, records):
        result = Query(records).filter(field("name").matches(r"^alice")).to_list()
        assert [r["name"] for r in result] == ["alice", "alice2"]

    def test_matches_invalid_regex(self):
        with pytest.raises(ValueError):
            field("name").matches("[invalid")

    def test_missing_field_returns_no_match(self, records):
        result = Query(records).filter(field("missing").startswith("x")).to_list()
        assert result == []

    def test_chained_string_and_eq(self, records):
        result = (
            Query(records)
            .filter(field("name").startswith("a"))
            .filter(field("role") == "admin")
            .to_list()
        )
        assert [r["name"] for r in result] == ["alice"]


class TestJoin:
    """inner_join / left_join — hash join operations."""

    @pytest.fixture
    def users(self):
        return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}, {"id": 3, "name": "Carol"}]

    @pytest.fixture
    def orders(self):
        return [
            {"user_id": 1, "item": "book"},
            {"user_id": 1, "item": "pen"},
            {"user_id": 2, "item": "notebook"},
        ]

    def lk(self, u): return u["id"]
    def rk(self, o): return o["user_id"]

    def test_inner_join_basic(self, users, orders):
        result = Query(users).inner_join(Query(orders), left_key=self.lk, right_key=self.rk).to_list()
        assert len(result) == 3
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)
        names = [u["name"] for u, _ in result]
        assert names.count("Alice") == 2
        assert names.count("Bob") == 1

    def test_inner_join_no_match(self, users):
        other = [{"user_id": 99, "item": "x"}]
        result = Query(users).inner_join(Query(other), left_key=self.lk, right_key=self.rk).to_list()
        assert result == []

    def test_inner_join_empty_left(self, orders):
        result = Query([]).inner_join(Query(orders), left_key=self.lk, right_key=self.rk).to_list()
        assert result == []

    def test_inner_join_empty_right(self, users):
        result = Query(users).inner_join(Query([]), left_key=self.lk, right_key=self.rk).to_list()
        assert result == []

    def test_inner_join_same_key_fn(self):
        a = [1, 2, 3]
        b = [2, 3, 4]
        result = Query(a).inner_join(Query(b), left_key=lambda x: x).to_list()
        assert sorted(v for v, _ in result) == [2, 3]

    def test_inner_join_duplicate_keys(self):
        left = [{"k": 1, "v": "a"}, {"k": 1, "v": "b"}]
        right = [{"k": 1, "r": "x"}, {"k": 1, "r": "y"}]
        result = Query(left).inner_join(Query(right), left_key=lambda r: r["k"], right_key=lambda r: r["k"]).to_list()
        assert len(result) == 4

    def test_left_join_basic(self, users, orders):
        result = Query(users).left_join(Query(orders), left_key=self.lk, right_key=self.rk).to_list()
        assert len(result) == 4
        carol_pair = next(u for u, o in result if u["name"] == "Carol")
        assert carol_pair == {"id": 3, "name": "Carol"}

    def test_left_join_no_right_match(self, users):
        result = Query(users).left_join(Query([]), left_key=self.lk, right_key=self.rk).to_list()
        assert len(result) == 3
        assert all(right is None for _, right in result)

    def test_left_join_preserves_all_left(self, users, orders):
        result = Query(users).left_join(Query(orders), left_key=self.lk, right_key=self.rk).to_list()
        left_ids = {u["id"] for u, _ in result}
        assert left_ids == {1, 2, 3}

    def test_left_join_empty_left(self, orders):
        result = Query([]).left_join(Query(orders), left_key=self.lk, right_key=self.rk).to_list()
        assert result == []


class TestSetField:
    """set_field / add_field / drop_field — dict field operations."""

    @pytest.fixture
    def products(self):
        return [
            {"name": "apple", "price": 1.0, "qty": 3},
            {"name": "banana", "price": 0.5, "qty": 10},
            {"name": "cherry", "price": 3.0, "qty": 1},
        ]

    def test_set_field_transform_value(self, products):
        result = Query(products).set_field("price", lambda v: round(v * 2, 2)).to_list()
        assert [r["price"] for r in result] == [2.0, 1.0, 6.0]
        assert all("name" in r for r in result)

    def test_set_field_missing_key_gets_none(self):
        data = [{"a": 1}, {"a": 2, "b": 10}]
        result = Query(data).set_field("b", lambda v: (v or 0) + 1).to_list()
        assert result[0]["b"] == 1
        assert result[1]["b"] == 11

    def test_set_field_empty(self, products):
        assert Query([]).set_field("price", lambda v: v).to_list() == []

    def test_set_field_non_dict_passthrough(self):
        result = Query([1, 2, 3]).set_field("x", lambda v: v).to_list()
        assert result == [1, 2, 3]

    def test_set_field_after_filter(self, products):
        result = (
            Query(products)
            .filter(lambda r: r["qty"] > 1)
            .set_field("price", lambda v: v * 0.9)
            .to_list()
        )
        assert len(result) == 2
        assert abs(result[0]["price"] - 0.9) < 1e-10

    def test_add_field_computed(self, products):
        result = Query(products).add_field("total", lambda r: r["price"] * r["qty"]).to_list()
        assert result[0]["total"] == 3.0
        assert result[1]["total"] == 5.0
        assert result[2]["total"] == 3.0

    def test_add_field_string(self, products):
        result = Query(products).add_field("label", lambda r: r["name"].upper()).to_list()
        assert result[0]["label"] == "APPLE"

    def test_add_field_empty(self):
        assert Query([]).add_field("x", lambda r: 1).to_list() == []

    def test_add_field_preserves_existing(self, products):
        result = Query(products).add_field("extra", lambda r: 0).to_list()
        assert all(set(r.keys()) == {"name", "price", "qty", "extra"} for r in result)

    def test_drop_field_single(self, products):
        result = Query(products).drop_field("qty").to_list()
        assert all("qty" not in r for r in result)
        assert all("name" in r and "price" in r for r in result)

    def test_drop_field_multiple(self, products):
        result = Query(products).drop_field("price", "qty").to_list()
        assert all(set(r.keys()) == {"name"} for r in result)

    def test_drop_field_missing_key_no_error(self, products):
        result = Query(products).drop_field("nonexistent").to_list()
        assert result == products

    def test_drop_field_empty(self):
        assert Query([]).drop_field("x").to_list() == []

    def test_drop_field_non_dict_passthrough(self):
        result = Query([1, 2, 3]).drop_field("x").to_list()
        assert result == [1, 2, 3]


class TestSelectRename:
    """select(*fields) / rename_field(old, new) — dict field projection."""

    @pytest.fixture
    def users(self):
        return [
            {"id": 1, "name": "Alice", "password": "secret", "age": 30},
            {"id": 2, "name": "Bob",   "password": "hidden", "age": 25},
        ]

    def test_select_basic(self, users):
        result = Query(users).select("id", "name").to_list()
        assert result == [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

    def test_select_single_field(self, users):
        result = Query(users).select("name").to_list()
        assert result == [{"name": "Alice"}, {"name": "Bob"}]

    def test_select_missing_field_silently_omitted(self, users):
        result = Query(users).select("id", "nonexistent").to_list()
        assert result == [{"id": 1}, {"id": 2}]

    def test_select_all_missing(self, users):
        result = Query(users).select("x", "y").to_list()
        assert result == [{}, {}]

    def test_select_preserves_order(self, users):
        result = Query(users).select("name", "id").to_list()
        assert list(result[0].keys()) == ["name", "id"]

    def test_select_empty_input(self):
        assert Query([]).select("id").to_list() == []

    def test_select_non_dict_passthrough(self):
        result = Query([1, 2, 3]).select("x").to_list()
        assert result == [1, 2, 3]

    def test_select_after_filter(self, users):
        result = (
            Query(users)
            .filter(lambda r: r["age"] >= 28)
            .select("id", "name")
            .to_list()
        )
        assert result == [{"id": 1, "name": "Alice"}]

    def test_rename_basic(self, users):
        result = Query(users).rename_field("name", "full_name").to_list()
        assert all("full_name" in r and "name" not in r for r in result)
        assert result[0]["full_name"] == "Alice"

    def test_rename_missing_key_passthrough(self, users):
        result = Query(users).rename_field("nonexistent", "x").to_list()
        assert result == users

    def test_rename_preserves_other_fields(self, users):
        result = Query(users).rename_field("id", "user_id").to_list()
        assert "user_id" in result[0]
        assert "id" not in result[0]
        assert "name" in result[0]

    def test_rename_empty(self):
        assert Query([]).rename_field("a", "b").to_list() == []

    def test_select_then_rename(self, users):
        result = (
            Query(users)
            .select("id", "name")
            .rename_field("id", "user_id")
            .to_list()
        )
        assert result[0] == {"user_id": 1, "name": "Alice"}


class TestValueCounts:
    """value_counts(key_fn=None) — frequency histogram."""

    def test_basic_strings(self):
        result = Query(["a", "b", "a", "c", "a"]).value_counts()
        assert result == {"a": 3, "b": 1, "c": 1}

    def test_basic_integers(self):
        result = Query([1, 2, 1, 3, 2, 1]).value_counts()
        assert result == {1: 3, 2: 2, 3: 1}

    def test_with_key_fn(self):
        records = [{"status": "ok"}, {"status": "err"}, {"status": "ok"}, {"status": "ok"}]
        result = Query(records).value_counts(lambda r: r["status"])
        assert result == {"ok": 3, "err": 1}

    def test_after_filter(self):
        data = list(range(10))
        result = Query(data).filter(lambda x: x < 5).value_counts(lambda x: x % 2)
        assert result == {0: 3, 1: 2}

    def test_empty(self):
        assert Query([]).value_counts() == {}

    def test_all_same(self):
        result = Query([42, 42, 42]).value_counts()
        assert result == {42: 3}

    def test_no_duplicates(self):
        result = Query([1, 2, 3]).value_counts()
        assert result == {1: 1, 2: 1, 3: 1}
