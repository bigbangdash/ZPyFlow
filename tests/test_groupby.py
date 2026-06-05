"""GroupBy / group_agg tests."""

import pytest

try:
    from zpyflow import Query, col, field, AggSpec, agg_count, agg_sum, agg_mean, agg_max, agg_min
    HAS_EXTENSION = True
except ImportError:
    HAS_EXTENSION = False

pytestmark = pytest.mark.skipif(not HAS_EXTENSION, reason="Native extension not built")


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
        assert result[0]["_key"] == "eng"

    def test_agg_spec_static_methods(self):
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


class TestGroupAggExtensions:
    """agg_median / agg_std / agg_first / agg_last for GroupBy.agg()."""

    from zpyflow import agg_median, agg_std, agg_first, agg_last

    @pytest.fixture
    def salary_data(self):
        return [
            {"dept": "eng", "salary": 100.0, "name": "A"},
            {"dept": "eng", "salary": 200.0, "name": "B"},
            {"dept": "eng", "salary": 300.0, "name": "C"},
            {"dept": "hr",  "salary": 80.0,  "name": "D"},
            {"dept": "hr",  "salary": 120.0, "name": "E"},
        ]

    def test_median_odd_group(self, salary_data):
        from zpyflow import agg_median
        result = Query(salary_data).group_by(lambda r: r["dept"]).agg(
            med=agg_median(lambda r: r["salary"])
        )
        by_dept = {row["_key"]: row["med"] for row in result}
        assert by_dept["eng"] == 200.0
        assert by_dept["hr"] == 100.0

    def test_median_even_group(self):
        from zpyflow import agg_median
        data = [{"g": "a", "v": 1.0}, {"g": "a", "v": 3.0},
                {"g": "a", "v": 5.0}, {"g": "a", "v": 7.0}]
        result = Query(data).group_by(lambda r: r["g"]).agg(
            med=agg_median(lambda r: r["v"])
        )
        assert result[0]["med"] == 4.0

    def test_median_empty_group(self):
        from zpyflow import agg_median
        result = Query([]).group_by(lambda r: r["g"]).agg(
            med=agg_median(lambda r: r["v"])
        )
        assert result == []

    def test_std_population(self, salary_data):
        from zpyflow import agg_std
        result = Query(salary_data).group_by(lambda r: r["dept"]).agg(
            std=agg_std(lambda r: r["salary"])
        )
        by_dept = {row["_key"]: row["std"] for row in result}
        assert abs(by_dept["eng"] - (20000 / 3) ** 0.5) < 1e-6

    def test_std_sample(self, salary_data):
        from zpyflow import agg_std
        result = Query(salary_data).group_by(lambda r: r["dept"]).agg(
            std=agg_std(lambda r: r["salary"], ddof=1)
        )
        by_dept = {row["_key"]: row["std"] for row in result}
        assert by_dept["eng"] is not None

    def test_std_single_element_ddof1_returns_none(self):
        from zpyflow import agg_std
        data = [{"g": "x", "v": 42.0}]
        result = Query(data).group_by(lambda r: r["g"]).agg(
            std=agg_std(lambda r: r["v"], ddof=1)
        )
        assert result[0]["std"] is None

    def test_first_element(self, salary_data):
        from zpyflow import agg_first
        result = Query(salary_data).group_by(lambda r: r["dept"]).agg(
            first_name=agg_first(lambda r: r["name"])
        )
        by_dept = {row["_key"]: row["first_name"] for row in result}
        assert by_dept["eng"] == "A"
        assert by_dept["hr"] == "D"

    def test_last_element(self, salary_data):
        from zpyflow import agg_last
        result = Query(salary_data).group_by(lambda r: r["dept"]).agg(
            last_name=agg_last(lambda r: r["name"])
        )
        by_dept = {row["_key"]: row["last_name"] for row in result}
        assert by_dept["eng"] == "C"
        assert by_dept["hr"] == "E"

    def test_first_no_field_fn(self, salary_data):
        from zpyflow import agg_first
        result = Query(salary_data).group_by(lambda r: r["dept"]).agg(
            first=agg_first()
        )
        by_dept = {row["_key"]: row["first"] for row in result}
        assert by_dept["eng"]["name"] == "A"

    def test_first_empty_returns_none(self):
        from zpyflow import agg_first
        result = Query([]).group_by(lambda r: r["g"]).agg(first=agg_first())
        assert result == []

    def test_combined_agg(self, salary_data):
        from zpyflow import agg_median, agg_std, agg_first, agg_last
        result = Query(salary_data).group_by(lambda r: r["dept"]).agg(
            med=agg_median(lambda r: r["salary"]),
            std=agg_std(lambda r: r["salary"]),
            first=agg_first(lambda r: r["name"]),
            last=agg_last(lambda r: r["name"]),
        )
        eng = next(r for r in result if r["_key"] == "eng")
        assert eng["first"] == "A"
        assert eng["last"] == "C"
        assert eng["med"] == 200.0
        assert eng["std"] is not None
