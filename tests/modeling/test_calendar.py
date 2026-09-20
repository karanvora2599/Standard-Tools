"""
The exchange calendar: what turns bars per hour into bars per year.

An intraday volatility could not be annualized because bars per year at
"1h" depends on the venue -- 6.5 hours on NYSE, 8.5 on the LSE, 24 on a
crypto venue -- and the package had nothing to resolve that from. With a
calendar named on the dataset both numbers are read off it, and the
planted relation is that an hourly volatility on XNYS is the daily one
scaled by sqrt(bars per year at 1h / 252), to the last digit. Without a
calendar the intraday feature still refuses, and the library the calendar
needs is optional and refused by name when absent.
"""

import numpy as np
import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling import calendar as calendar_module
from standard_quant_tools.modeling.calendar import (
    bars_per_session,
    interval_minutes,
    periods_per_year,
    session_minutes,
    sessions_per_year,
)
from standard_quant_tools.modeling.capabilities import modeling_capabilities
from standard_quant_tools.modeling.dataset.builder import (
    build_dataset,
    dataset_spec_hash,
)
from standard_quant_tools.modeling.features.base import (
    FeatureContext,
    periods_per_year_for_interval,
)
from standard_quant_tools.modeling.features.registry import get_feature
from standard_quant_tools.modeling.specs import DatasetSpec, FeatureSpec, TargetSpec

from .conftest import make_ohlcv, make_provider_mock

requires_calendars = pytest.mark.skipif(
    not calendar_module.calendar_available(),
    reason="exchange_calendars is not installed",
)
REFUSED = (ValidationError, PydanticValidationError)


def _spec(**overrides) -> DatasetSpec:
    fields = dict(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi")],
        target=TargetSpec(horizon=5),
        benchmark="SPY",
    )
    fields.update(overrides)
    return DatasetSpec(**fields)


class TestIntervalParsing:
    @pytest.mark.parametrize(
        "interval, minutes",
        [
            ("1m", 1),
            ("5m", 5),
            ("15m", 15),
            ("30m", 30),
            ("60m", 60),
            ("90m", 90),
            ("1h", 60),
            ("2h", 120),
            ("1d", None),
            ("1wk", None),
            ("daily", None),
            ("0m", None),
        ],
    )
    def test_minutes_per_bar(self, interval, minutes):
        assert interval_minutes(interval) == minutes


@requires_calendars
class TestTheCalendar:
    def test_nyse_sessions_and_session_length_are_read_off_the_calendar(self):
        assert session_minutes("XNYS") == 390.0
        per_year = sessions_per_year("XNYS")
        assert 250 <= per_year <= 253
        # The partial last bar counts: a provider emits seven hourly bars
        # for a six-and-a-half-hour session.
        assert bars_per_session("1h", "XNYS") == 7
        assert bars_per_session("30m", "XNYS") == 13
        assert bars_per_session("5m", "XNYS") == 78
        assert periods_per_year("1h", "XNYS") == round(7 * per_year)
        assert periods_per_year("5m", "XNYS") == round(78 * per_year)

    def test_venues_differ_and_that_is_the_point(self):
        assert session_minutes("XLON") == 510.0
        assert bars_per_session("1h", "XLON") == 9
        assert bars_per_session("1h", "24/7") == 24
        assert sessions_per_year("24/7") == pytest.approx(365.25, abs=0.3)
        assert periods_per_year("1h", "24/7") > 3 * periods_per_year("1h", "XNYS")

    def test_the_feature_helper_reads_the_calendar_for_intraday_only(self):
        assert periods_per_year_for_interval("1d", "XNYS") == 252
        assert periods_per_year_for_interval("1wk", "XLON") == 52
        assert periods_per_year_for_interval("1h") is None
        assert periods_per_year_for_interval("1h", "XNYS") == periods_per_year(
            "1h", "XNYS"
        )
        assert periods_per_year_for_interval("fortnightly", "XNYS") is None

    def test_an_unknown_name_is_refused_with_the_known_ones(self):
        with pytest.raises(ValidationError, match="not an exchange_calendars name"):
            periods_per_year("1h", "NOPE")
        with pytest.raises(REFUSED, match="not an exchange_calendars name"):
            _spec(calendar="NOPE")

    def test_a_calendar_is_part_of_the_dataset_identity_and_absent_by_default(self):
        without = _spec()
        assert without.calendar is None
        assert dataset_spec_hash(without) == dataset_spec_hash(_spec(calendar=None))
        assert dataset_spec_hash(_spec(calendar="XNYS")) != dataset_spec_hash(without)
        assert _spec(calendar="XNYS").calendar == "XNYS"

    @pytest.mark.parametrize(
        "feature_id",
        [
            "risk.realized_volatility",
            "risk.parkinson_volatility",
            "risk.garman_klass_volatility",
            "risk.realized_semivariance",
            "risk.bipower_variation",
        ],
    )
    def test_an_intraday_feature_annualizes_with_a_calendar_and_refuses_without(
        self, feature_id
    ):
        fn = get_feature(feature_id).fn
        ohlcv = make_ohlcv("AAA")
        with pytest.raises(ValidationError, match="cannot annualize"):
            fn(ohlcv, FeatureContext(interval="1h"), period=20)
        hourly = fn(ohlcv, FeatureContext(interval="1h", calendar="XNYS"), period=20)
        daily = fn(ohlcv, FeatureContext(interval="1d"), period=20)
        ratio = np.sqrt(periods_per_year("1h", "XNYS") / 252.0)
        np.testing.assert_allclose(
            hourly.dropna().to_numpy(), daily.dropna().to_numpy() * ratio, rtol=1e-9
        )

    def test_the_builder_hands_the_calendar_to_the_features(self, monkeypatch):
        from standard_quant_tools.data.factory import DataFactory

        provider = make_provider_mock(make_ohlcv)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
        features = [FeatureSpec(id="risk.realized_volatility")]
        with pytest.raises(ValidationError, match="cannot annualize"):
            build_dataset(_spec(interval="1h", features=features))
        built = build_dataset(_spec(interval="1h", features=features, calendar="XNYS"))
        assert "risk.realized_volatility" in built["feature_ids"]
        assert not built["panel"].empty
        assert modeling_capabilities()["optional_dependencies"]["exchange_calendars"]


class TestWithoutTheLibrary:
    def test_a_named_calendar_is_refused_by_name(self, monkeypatch):
        monkeypatch.setattr(calendar_module, "calendar_available", lambda: False)
        with pytest.raises(REFUSED, match="exchange_calendars"):
            _spec(calendar="XNYS")
        with pytest.raises(ValidationError, match="exchange_calendars"):
            periods_per_year_for_interval("1h", "XNYS")
        # A daily interval never needed the library.
        assert periods_per_year_for_interval("1d", "XNYS") == 252
        assert _spec().calendar is None
        assert (
            modeling_capabilities()["optional_dependencies"]["exchange_calendars"]
            is False
        )
