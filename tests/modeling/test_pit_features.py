"""
Point-in-time features in a built dataset: filings joined by when they
became knowable, restatements read from the day they were restated, and a
provider that cannot say when refused before anything is fetched.

Planted around one earnings figure and one revenue restatement:

    AAA, fiscal 2023 Q2 (period end 2023-06-30)
        filed     2023-07-29   diluted EPS 1.20
        amended   2023-08-14   diluted EPS 1.05
    AAA, fiscal 2022 Q2 revenue 100, restated to 110 on 2023-09-01;
    fiscal 2023 Q2 revenue 120 throughout.

So a panel row on 14 July reads the Q1 figure, 1 August reads 1.20,
21 August reads 1.05; year-over-year growth reads 0.20 on 21 August and
120/110 - 1 on 11 September, because the prior year changed under it.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from standard_quant_tools.data.temporal import TemporalContract, price_contract
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.features.base import FeatureDefinition, FeatureScope
from standard_quant_tools.modeling.features.fundamental import (
    DILUTED_EPS,
    NET_INCOME,
    REVENUES,
    revenue_growth_yoy,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    FeatureSpec,
    MissingDataSpec,
    TargetSpec,
)

from .conftest import make_ohlcv, make_provider_mock

UNIVERSE = ["AAA", "BBB", "CCC"]
QUARTER_ENDS = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}


def _records() -> pd.DataFrame:
    rows = []
    for entity in UNIVERSE:
        for year in (2021, 2022, 2023):
            for q, (period, month_day) in enumerate(QUARTER_ENDS.items(), start=1):
                if year == 2023 and period == "Q4":
                    continue
                end = pd.Timestamp(f"{year}-{month_day}")
                revenue = {
                    "AAA": 100.0 if year < 2023 else 120.0,
                    "BBB": 200.0,
                    "CCC": 50.0 * q,
                }[entity]
                rows.append(
                    {
                        "entity": entity,
                        "event_time": end,
                        "available_time": end + pd.Timedelta(days=28),
                        "fiscal_year": year,
                        "fiscal_period": period,
                        DILUTED_EPS: float(f"{year}.{q}"),
                        NET_INCOME: 0.1 * revenue,
                        REVENUES: revenue,
                    }
                )
    frame = pd.DataFrame(rows)
    # The planted Q2 2023 filing for AAA: 1.20 on 29 July, amended to 1.05.
    q2 = (frame["entity"] == "AAA") & (frame["event_time"] == "2023-06-30")
    frame.loc[q2, ["available_time", DILUTED_EPS]] = [pd.Timestamp("2023-07-29"), 1.20]
    amended = frame[q2].copy()
    amended["available_time"] = pd.Timestamp("2023-08-14")
    amended[DILUTED_EPS] = 1.05
    # And the prior year's revenue restated a year later.
    restated = frame[
        (frame["entity"] == "AAA") & (frame["event_time"] == "2022-06-30")
    ].copy()
    restated["available_time"] = pd.Timestamp("2023-09-01")
    restated[REVENUES] = 110.0
    return pd.concat([frame, amended, restated], ignore_index=True)


def _provider(*, pit_safe: bool = True) -> MagicMock:
    provider = make_provider_mock(make_ohlcv)
    fundamentals = TemporalContract(
        source="mock",
        frame_kind="fundamentals",
        has_event_time=True,
        has_available_time=pit_safe,
        revisions="unknown",
    )
    provider.get_temporal_contract.side_effect = lambda kind="bars": (
        price_contract("mock") if kind == "bars" else fundamentals
    )
    provider.get_point_in_time_records.side_effect = (
        lambda symbols, kind, fields, start, end: _records()
    )
    return provider


def _spec(*features, **overrides) -> DatasetSpec:
    fields = dict(
        universe=UNIVERSE,
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), *features],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )
    fields.update(overrides)
    return DatasetSpec(**fields)


@pytest.fixture
def factory(monkeypatch):
    from standard_quant_tools.data.factory import DataFactory

    holder = {}
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: holder["p"])
    return holder


def _value(panel, date, entity, column):
    row = panel[(panel["date"] == pd.Timestamp(date)) & (panel["entity"] == entity)]
    assert len(row) == 1, f"no single row for {entity} on {date}"
    return float(row[column].iloc[0])


class TestTheJoin:
    def test_a_row_reads_the_version_current_on_its_date(self, factory):
        factory["p"] = _provider()
        built = build_dataset(_spec(FeatureSpec(id="fundamental.diluted_eps")))
        panel = built["panel"]
        eps = "fundamental.diluted_eps"
        assert eps in built["feature_ids"]
        # 14 July: the Q2 filing is a fortnight away; the Q1 figure stands.
        assert _value(panel, "2023-07-14", "AAA", eps) == 2023.1
        assert _value(panel, "2023-08-01", "AAA", eps) == 1.20
        assert _value(panel, "2023-08-21", "AAA", eps) == 1.05
        # Another entity is untouched by AAA's amendment.
        assert _value(panel, "2023-08-21", "BBB", eps) == 2023.2
        # The provider was asked once, for the union of fields, from before
        # the panel starts.
        provider = factory["p"]
        assert provider.get_point_in_time_records.call_count == 1
        symbols, kind, fields, start, end = (
            provider.get_point_in_time_records.call_args[0]
        )
        assert symbols == UNIVERSE and kind == "fundamentals"
        assert fields == [DILUTED_EPS]
        assert pd.Timestamp(start) < pd.Timestamp("2021-01-01") and end == "2023-12-31"

    def test_growth_reads_a_restated_prior_year_from_the_day_it_was_restated(
        self, factory
    ):
        factory["p"] = _provider()
        built = build_dataset(_spec(FeatureSpec(id="fundamental.revenue_growth_yoy")))
        panel = built["panel"]
        growth = "fundamental.revenue_growth_yoy"
        assert _value(panel, "2023-07-14", "AAA", growth) == pytest.approx(0.2)
        assert _value(panel, "2023-08-21", "AAA", growth) == pytest.approx(0.2)
        assert _value(panel, "2023-09-11", "AAA", growth) == pytest.approx(
            120 / 110 - 1
        )
        assert _value(panel, "2023-09-11", "BBB", growth) == pytest.approx(0.0)

    def test_rows_nobody_had_a_filing_for_are_dropped_and_attributed(self, factory):
        factory["p"] = _provider()
        built = build_dataset(_spec(FeatureSpec(id="fundamental.revenue_growth_yoy")))
        growth = "fundamental.revenue_growth_yoy"
        # Growth needs a prior fiscal year: the first 2022 rows have none
        # available until the Q1 2022 filing on 28 April 2022.
        assert built["panel"]["date"].min() >= pd.Timestamp("2022-04-28")
        attribution = built["drop_attribution"]["per_feature"][growth]
        assert attribution["n_missing"] > 0
        assert any(growth in w and "not yet available" in w for w in built["warnings"])
        assert any("more than one version" in w for w in built["warnings"])
        verdict = built["temporal_bundle"]
        assert "fundamentals" in verdict["kinds"] and verdict["usable"]

    def test_under_keep_the_rows_stay_with_a_hole(self, factory):
        factory["p"] = _provider()
        built = build_dataset(
            _spec(
                FeatureSpec(id="fundamental.revenue_growth_yoy"),
                missing=MissingDataSpec(policy="keep"),
            )
        )
        growth = built["panel"]["fundamental.revenue_growth_yoy"]
        assert growth.isna().any() and growth.notna().any()
        assert built["panel"]["date"].min() < pd.Timestamp("2022-04-28")
        attribution = built["drop_attribution"]
        assert attribution["policy"] == "keep"
        assert (
            attribution["per_feature"]["fundamental.revenue_growth_yoy"]["n_missing"]
            > 0
        )

    def test_staleness_bounds_how_old_a_filing_may_be(self, factory):
        factory["p"] = _provider()
        built = build_dataset(
            _spec(
                FeatureSpec(
                    id="fundamental.diluted_eps", params={"max_staleness_days": 40}
                ),
                missing=MissingDataSpec(policy="keep"),
            )
        )
        panel = built["panel"]
        eps = "fundamental.diluted_eps"
        # Q1 2023 filed 28 April: usable on 1 June (34 days), stale by 14 July.
        assert _value(panel, "2023-06-01", "AAA", eps) == 2023.1
        row = panel[(panel["date"] == "2023-07-14") & (panel["entity"] == "AAA")]
        assert row[eps].isna().all()


class TestTheGate:
    def test_a_provider_without_availability_times_is_refused_before_any_fetch(
        self, factory
    ):
        factory["p"] = _provider(pit_safe=False)
        with pytest.raises(ValidationError, match="point-in-time") as info:
            build_dataset(_spec(FeatureSpec(id="fundamental.diluted_eps")))
        assert "fundamental.diluted_eps" in str(info.value)
        provider = factory["p"]
        assert provider.get_point_in_time_records.call_count == 0
        assert provider.get_ohlcv_async.await_count == 0
        assert provider.get_ohlcv.call_count == 0

    def test_a_provider_with_no_contract_at_all_is_refused(self, factory):
        provider = make_provider_mock(make_ohlcv)
        del provider.get_temporal_contract
        factory["p"] = provider
        with pytest.raises(ValidationError, match="declares no temporal contract"):
            build_dataset(_spec(FeatureSpec(id="fundamental.net_margin")))

    def test_a_lag_on_a_point_in_time_feature_is_refused(self, factory):
        factory["p"] = _provider()
        with pytest.raises(ValidationError, match="cannot be lagged"):
            build_dataset(_spec(FeatureSpec(id="fundamental.diluted_eps", lags=[1])))

    def test_a_definition_must_say_what_it_reads(self):
        with pytest.raises(ValueError, match="frame_kind"):
            FeatureDefinition(
                id="x.pit",
                description="",
                fn=revenue_growth_yoy,
                temporal_support="pit_safe",
                scope=FeatureScope.POINT_IN_TIME,
                lookback=0,
                default_params={"max_staleness_days": 10},
            )
        with pytest.raises(ValueError, match="max_staleness_days"):
            FeatureDefinition(
                id="x.pit",
                description="",
                fn=revenue_growth_yoy,
                temporal_support="pit_safe",
                scope=FeatureScope.POINT_IN_TIME,
                lookback=0,
                frame_kind="fundamentals",
                fields=[REVENUES],
            )
        with pytest.raises(ValueError, match="POINT_IN_TIME only"):
            FeatureDefinition(
                id="x.bars",
                description="",
                fn=revenue_growth_yoy,
                temporal_support="pit_safe",
                lookback=0,
                frame_kind="fundamentals",
            )


class TestTheTransforms:
    def test_net_margin_is_nan_without_positive_revenue(self):
        from standard_quant_tools.modeling.features.fundamental import net_margin

        records = _records()
        records.loc[records.index[0], REVENUES] = 0.0
        out = net_margin(records, None)
        assert out.columns.tolist() == [
            "entity",
            "event_time",
            "available_time",
            "value",
        ]
        assert pd.isna(out["value"].iloc[0])
        assert out["value"].iloc[1] == pytest.approx(0.1)

    def test_growth_has_a_version_at_every_change_of_either_filing(self):
        out = revenue_growth_yoy(_records(), None)
        aaa_q2 = out[(out["entity"] == "AAA") & (out["event_time"] == "2023-06-30")]
        assert aaa_q2["available_time"].tolist() == [
            pd.Timestamp("2023-07-29"),
            pd.Timestamp("2023-08-14"),
            pd.Timestamp("2023-09-01"),
        ]
        assert aaa_q2["value"].tolist() == pytest.approx([0.2, 0.2, 120 / 110 - 1])
        # A prior year that was never restated: one version per filing.
        bbb = out[(out["entity"] == "BBB") & (out["event_time"] == "2023-06-30")]
        assert len(bbb) == 1 and bbb["value"].iloc[0] == 0.0
        # Nothing for a period with no prior year in the records.
        assert out[out["event_time"] < "2022-01-01"].empty

    def test_a_record_set_without_period_keys_is_refused_by_name(self):
        records = _records().drop(columns=["fiscal_year"])
        with pytest.raises(ValidationError, match="fiscal_year"):
            revenue_growth_yoy(records, None)
