"""Tests for backtest/costs.py — pluggable transaction-cost model building blocks."""

import pytest

from standard_quant_tools.backtest.costs import (
    fixed_bps_spread,
    impact_cost,
    margin_interest,
    pct_of_range_spread,
    per_share_commission,
    percentage_commission,
    short_borrow_cost,
    sqrt_impact_bps,
)
from standard_quant_tools.error import ValidationError


class TestPercentageCommission:
    def test_basic(self):
        assert percentage_commission(10_000.0, 0.001) == pytest.approx(10.0)

    def test_negative_notional_uses_absolute_value(self):
        assert percentage_commission(-10_000.0, 0.001) == pytest.approx(10.0)


class TestPerShareCommission:
    def test_basic(self):
        assert per_share_commission(100, 0.005) == pytest.approx(0.5)

    def test_minimum_floor_applies(self):
        assert per_share_commission(10, 0.005, minimum=1.0) == pytest.approx(1.0)

    def test_negative_shares_uses_absolute_value(self):
        assert per_share_commission(-100, 0.005) == pytest.approx(0.5)


class TestFixedBpsSpread:
    def test_basic(self):
        assert fixed_bps_spread(10_000.0, 5.0) == pytest.approx(5.0)


class TestPctOfRangeSpread:
    def test_basic(self):
        # range_frac = (102-98)/100 = 0.04; cost = 10000 * 0.04 * 0.5 = 200
        assert pct_of_range_spread(
            10_000.0, high=102.0, low=98.0, close=100.0, pct=0.5
        ) == pytest.approx(200.0)

    def test_zero_close_raises(self):
        with pytest.raises(ValidationError, match="close"):
            pct_of_range_spread(10_000.0, high=102.0, low=98.0, close=0.0, pct=0.5)


class TestSqrtImpactBps:
    def test_basic(self):
        # 1.0 * 0.02 * sqrt(0.25) * 10000 = 0.02*0.5*10000 = 100
        assert sqrt_impact_bps(
            participation=0.25, volatility=0.02, coefficient=1.0
        ) == pytest.approx(100.0)

    def test_zero_participation_is_zero_impact(self):
        assert sqrt_impact_bps(0.0, 0.02) == pytest.approx(0.0)

    def test_negative_participation_raises(self):
        with pytest.raises(ValidationError, match="participation"):
            sqrt_impact_bps(-0.1, 0.02)

    def test_higher_participation_gives_higher_impact(self):
        low = sqrt_impact_bps(0.01, 0.02)
        high = sqrt_impact_bps(0.5, 0.02)
        assert high > low


class TestImpactCost:
    def test_zero_adv_returns_zero(self):
        assert impact_cost(10_000.0, avg_dollar_volume=0.0, volatility=0.02) == 0.0

    def test_negative_adv_returns_zero(self):
        assert impact_cost(10_000.0, avg_dollar_volume=-1.0, volatility=0.02) == 0.0

    def test_basic_matches_manual_computation(self):
        notional = 10_000.0
        adv = 40_000.0  # participation = 0.25
        vol = 0.02
        expected_bps = sqrt_impact_bps(0.25, vol, 1.0)
        expected_cost = notional * (expected_bps / 10_000.0)
        assert impact_cost(notional, adv, vol) == pytest.approx(expected_cost)


class TestShortBorrowCost:
    def test_basic(self):
        # 10000 * (200/10000) * (365/365) = 200
        assert short_borrow_cost(
            10_000.0, annual_bps=200.0, days=365.0
        ) == pytest.approx(200.0)

    def test_one_day_accrual(self):
        one_day = short_borrow_cost(10_000.0, annual_bps=200.0, days=1.0)
        full_year = short_borrow_cost(10_000.0, annual_bps=200.0, days=365.0)
        assert one_day == pytest.approx(full_year / 365.0)


class TestMarginInterest:
    def test_positive_cash_is_zero(self):
        assert margin_interest(5_000.0, annual_rate=0.05, days=1.0) == 0.0

    def test_negative_cash_accrues(self):
        # |−5000| * 0.05 * (365/365) = 250
        assert margin_interest(-5_000.0, annual_rate=0.05, days=365.0) == pytest.approx(
            250.0
        )

    def test_zero_cash_is_zero(self):
        assert margin_interest(0.0, annual_rate=0.05, days=1.0) == 0.0
