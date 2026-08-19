"""Tests for backtest/constraints.py — liquidity/capacity diagnostics."""

import math

import pytest

from standard_quant_tools.backtest.constraints import (
    adv_participation,
    capacity_report,
    days_to_liquidate,
    sector_exposure,
)
from standard_quant_tools.error import ValidationError


class TestAdvParticipation:
    def test_basic(self):
        assert adv_participation(10_000.0, 100_000.0) == pytest.approx(0.1)

    @pytest.mark.parametrize("adv", [0.0, -5.0, float("nan"), float("inf")])
    def test_unusable_adv_is_not_estimable_not_zero(self, adv):
        """
        These used to assert 0.0 and call it conservative. It is the
        opposite: 0.0 participation is the score of a trade so small it
        barely touches the market, so a ticker with NO volume data ranked as
        the EASIEST in the universe to trade.

            adv_participation(1e9, adv=0)   -> 0.0    (looked frictionless)
            adv_participation(1e9, adv=1e7) -> 100.0  (honest: 100x ADV)

        NaN says "not estimable" and cannot be mistaken for a measurement.
        """
        assert math.isnan(adv_participation(10_000.0, adv))

    def test_a_real_baseline_still_measures(self):
        assert adv_participation(1e9, 1e7) == pytest.approx(100.0)


class TestDaysToLiquidate:
    def test_basic(self):
        # tradeable_per_day = 1,000,000 * 0.1 = 100,000; shares=500,000 -> 5 days
        assert days_to_liquidate(
            500_000, avg_daily_volume=1_000_000, max_participation=0.1
        ) == pytest.approx(5.0)

    def test_zero_avg_daily_volume_raises(self):
        with pytest.raises(ValidationError, match="avg_daily_volume"):
            days_to_liquidate(100, avg_daily_volume=0.0, max_participation=0.1)

    def test_zero_max_participation_raises(self):
        with pytest.raises(ValidationError, match="max_participation"):
            days_to_liquidate(100, avg_daily_volume=1_000_000, max_participation=0.0)

    def test_negative_shares_uses_absolute_value(self):
        assert days_to_liquidate(-500_000, 1_000_000, 0.1) == pytest.approx(5.0)


class TestSectorExposure:
    def test_aggregates_by_sector(self):
        weights = {"AAPL": 0.3, "MSFT": 0.2, "XOM": 0.5}
        sectors = {"AAPL": "Technology", "MSFT": "Technology", "XOM": "Energy"}
        result = sector_exposure(weights, sectors)
        assert result == {
            "Technology": pytest.approx(0.5),
            "Energy": pytest.approx(0.5),
        }

    def test_missing_sector_buckets_as_unknown(self):
        weights = {"AAPL": 0.6, "ZZZZ": 0.4}
        sectors = {"AAPL": "Technology"}
        result = sector_exposure(weights, sectors)
        assert result["Unknown"] == pytest.approx(0.4)

    def test_empty_weights_returns_empty(self):
        assert sector_exposure({}, {}) == {}


class TestCapacityReport:
    def test_binding_ticker_is_lowest_capacity(self):
        tickers = ["A", "B"]
        adv = {"A": 1_000_000.0, "B": 10_000_000.0}
        weights = {"A": 0.5, "B": 0.5}
        result = capacity_report(tickers, adv, weights, max_participation=0.1)
        # A: 0.1*1e6/0.5 = 200,000 ; B: 0.1*1e7/0.5 = 2,000,000
        assert result["per_ticker"]["A"] == pytest.approx(200_000.0)
        assert result["per_ticker"]["B"] == pytest.approx(2_000_000.0)
        assert result["binding_ticker"] == "A"
        assert result["max_account_size"] == pytest.approx(200_000.0)

    def test_zero_weight_ticker_has_infinite_capacity_and_is_excluded(self):
        tickers = ["A", "B"]
        adv = {"A": 1_000_000.0, "B": 10_000_000.0}
        weights = {"A": 0.0, "B": 1.0}
        result = capacity_report(tickers, adv, weights, max_participation=0.1)
        assert result["per_ticker"]["A"] == float("inf")
        assert result["binding_ticker"] == "B"

    def test_all_zero_weights_returns_infinite_capacity_and_no_binding_ticker(self):
        tickers = ["A"]
        result = capacity_report(
            tickers, {"A": 1_000_000.0}, {"A": 0.0}, max_participation=0.1
        )
        assert result["binding_ticker"] is None
        assert result["max_account_size"] == float("inf")

    def test_missing_adv_raises(self):
        with pytest.raises(ValidationError, match="avg_dollar_volumes"):
            capacity_report(
                ["A", "B"], {"A": 1.0}, {"A": 0.5, "B": 0.5}, max_participation=0.1
            )

    def test_missing_weight_raises(self):
        with pytest.raises(ValidationError, match="target_weights"):
            capacity_report(
                ["A", "B"], {"A": 1.0, "B": 1.0}, {"A": 0.5}, max_participation=0.1
            )

    def test_invalid_max_participation_raises(self):
        with pytest.raises(ValidationError, match="max_participation"):
            capacity_report(["A"], {"A": 1.0}, {"A": 0.5}, max_participation=0.0)
