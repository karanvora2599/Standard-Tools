"""Bars per year: how the number every annualized metric multiplies by is
decided (metrics/annualization.py)."""

import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.annualization import (
    DEFAULT_PERIODS_PER_YEAR,
    infer_periods_per_year,
    periods_per_year_for_interval,
    resolve_periods_per_year,
)


def _index(freq: str, periods: int = 120, start: str = "2015-01-02") -> pd.Index:
    return pd.date_range(start, periods=periods, freq=freq)


class TestTheSpacingIsBucketedNotDivided:
    @pytest.mark.parametrize(
        "freq, expected",
        [("B", 252), ("W-FRI", 52), ("ME", 12), ("QE", 4)],
    )
    def test_each_calendar_interval_reads_as_its_year(self, freq, expected):
        value, notes = infer_periods_per_year(_index(freq))
        assert value == expected
        assert notes == []

    def test_business_days_are_252_not_365(self):
        """The median gap between business days is one calendar day, so
        dividing 365.25 by it answers 365 for the most common input there
        is. Holidays and weekends only move the mean."""
        index = _index("B", periods=600).delete([10, 11, 50, 200])
        assert infer_periods_per_year(index)[0] == 252

    def test_daily_bars_with_no_weekends_say_252_is_probably_wrong(self):
        value, notes = infer_periods_per_year(_index("D", periods=400))
        assert value == 252
        assert any("every day" in note for note in notes)

    @pytest.mark.parametrize(
        "index",
        [
            pd.date_range("2024-01-02 09:30", periods=50, freq="h"),
            pd.RangeIndex(100),
            pd.DatetimeIndex(["2024-01-02"]),
            pd.DatetimeIndex(["2024-01-02", "2024-03-15", "2024-03-16", "2024-09-01"]),
        ],
        ids=["intraday", "not-dates", "one-bar", "irregular"],
    )
    def test_spacing_that_names_no_year_resolves_to_nothing(self, index):
        value, notes = infer_periods_per_year(index)
        assert value is None
        assert notes


class TestResolution:
    def test_explicit_wins_and_is_validated(self):
        assert resolve_periods_per_year(_index("B"), periods_per_year=12) == (
            12,
            "explicit",
            [],
        )
        for bad in (0, -252, 2.5, True, float("inf")):
            with pytest.raises(ValidationError, match="periods_per_year"):
                resolve_periods_per_year(_index("B"), periods_per_year=bad)

    def test_the_fetched_interval_decides_and_a_gapped_index_does_not_warn(self):
        gapped = _index("B", periods=300).delete(list(range(40, 60)))
        assert resolve_periods_per_year(gapped, interval="1d") == (252, "interval", [])
        assert resolve_periods_per_year(_index("W-FRI"), interval="1wk")[0] == 52

    def test_an_interval_the_bars_contradict_is_named(self):
        value, source, warnings = resolve_periods_per_year(_index("ME"), interval="1d")
        assert (value, source) == (252, "interval")
        assert "spaced like 12" in warnings[0]

    def test_inference_when_nothing_is_given(self):
        assert resolve_periods_per_year(_index("ME")) == (12, "inferred", [])

    @pytest.mark.parametrize(
        "kwargs, index",
        [
            ({}, pd.date_range("2024-01-02 09:30", periods=50, freq="h")),
            ({}, pd.RangeIndex(100)),
            ({"interval": "1h"}, _index("B")),
        ],
        ids=["intraday", "range-index", "intraday-interval"],
    )
    def test_the_fallback_is_252_and_says_how_to_settle_it(self, kwargs, index):
        value, source, warnings = resolve_periods_per_year(index, **kwargs)
        assert (value, source) == (DEFAULT_PERIODS_PER_YEAR, "default")
        (warning,) = warnings
        assert "periods_per_year=" in warning and "interval=" in warning


class TestOneTableForBacktestAndModeling:
    def test_the_modeling_import_is_the_same_function(self):
        from standard_quant_tools.modeling import portfolio_eval
        from standard_quant_tools.modeling.features import base

        assert base.periods_per_year_for_interval is periods_per_year_for_interval
        assert (
            portfolio_eval.periods_per_year_for_interval
            is periods_per_year_for_interval
        )

    def test_the_table(self):
        assert [
            periods_per_year_for_interval(i) for i in ("1d", "5d", "1wk", "1mo", "3mo")
        ] == [252, 52, 52, 12, 4]
        assert periods_per_year_for_interval("1h") is None
