"""
Hand-verified tests for the true portfolio simulation engine
(backtest/portfolio_engine.py). Expected numbers were computed by hand and
cross-checked with a standalone script (see the plan for the derivation)
before being encoded here as exact assertions.

Reference scenario: 2 tickers (AAPL, MSFT), 5 bars, rebalance at bar 0
({AAPL: 0.5, MSFT: 0.3}) and bar 3 ({AAPL: 0.2, MSFT: 0.6}).
"""

import pandas as pd
import pytest

from standard_quant_tools.backtest.portfolio_engine import run_portfolio_simulation
from standard_quant_tools.error import ValidationError


def _price_df(prices, dates):
    return pd.DataFrame({
        "Open": prices, "High": prices, "Low": prices, "Close": prices,
        "Volume": [1_000_000.0] * len(prices),
    }, index=dates)


@pytest.fixture
def two_ticker_price_data():
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    aapl = [100.0, 105.0, 102.0, 110.0, 108.0]
    msft = [50.0, 48.0, 52.0, 55.0, 57.0]
    return {
        "AAPL": _price_df(aapl, dates),
        "MSFT": _price_df(msft, dates),
    }, dates


class TestZeroCostReferenceCase:
    def test_hand_verified_equity_curve(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [0.5, None, None, 0.2, None], "MSFT": [0.3, None, None, 0.6, None]},
            index=dates,
        ).dropna()

        result = run_portfolio_simulation(
            price_data, target_weights,
            initial_capital=10_000.0, commission_pct=0.0, slippage_pct=0.0,
        )
        equity = result["equity_curve"]
        expected = [10000.0, 10130.0, 10220.0, 10800.0, 10996.363636363636]
        for actual, exp in zip(equity.tolist(), expected):
            assert actual == pytest.approx(exp, abs=1e-6)

    def test_equity_unchanged_immediately_across_zero_cost_rebalance(self, two_ticker_price_data):
        """A zero-cost rebalance shouldn't create or destroy equity — only
        redistribute it across positions/cash."""
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [0.5, 0.2], "MSFT": [0.3, 0.6]}, index=[dates[0], dates[3]],
        )
        result = run_portfolio_simulation(
            price_data, target_weights,
            initial_capital=10_000.0, commission_pct=0.0, slippage_pct=0.0,
        )
        equity = result["equity_curve"]
        assert equity.iloc[0] == pytest.approx(10000.0)
        # bar 3 equity == bar 2's mark-to-market equity (pre-rebalance level preserved)
        assert equity.iloc[3] == pytest.approx(10800.0)

    def test_rebalance_log_shape(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [0.5, 0.2], "MSFT": [0.3, 0.6]}, index=[dates[0], dates[3]],
        )
        result = run_portfolio_simulation(price_data, target_weights, commission_pct=0.0, slippage_pct=0.0)
        log = result["rebalance_log"]
        assert len(log) == 2
        assert set(log.columns) == {"date", "turnover_pct", "gross_leverage_after", "n_positions"}
        assert log.iloc[0]["n_positions"] == 2
        assert log.iloc[0]["gross_leverage_after"] == pytest.approx(0.8, abs=1e-4)


class TestWithCostsReferenceCase:
    def test_hand_verified_equity_curve_with_costs(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame(
            {"AAPL": [0.5, 0.2], "MSFT": [0.3, 0.6]}, index=[dates[0], dates[3]],
        )
        result = run_portfolio_simulation(
            price_data, target_weights,
            initial_capital=10_000.0, commission_pct=0.001, slippage_pct=0.0005,
        )
        equity = result["equity_curve"]
        expected = [9988.0, 10118.0, 10208.0, 10778.2272, 10974.372654545454]
        for actual, exp in zip(equity.tolist(), expected):
            assert actual == pytest.approx(exp, abs=1e-4)


class TestShortPosition:
    def test_short_entry_credits_cash_and_preserves_equity(self):
        dates = pd.date_range("2023-01-02", periods=2, freq="B")
        price_data = {"AAPL": _price_df([100.0, 102.0], dates)}
        target_weights = pd.DataFrame({"AAPL": [-0.5]}, index=[dates[0]])

        result = run_portfolio_simulation(
            price_data, target_weights,
            initial_capital=10_000.0, commission_pct=0.0, slippage_pct=0.0,
        )
        assert result["cash_curve"].iloc[0] == pytest.approx(15_000.0)
        assert result["equity_curve"].iloc[0] == pytest.approx(10_000.0)
        # price rose 100 -> 102 while short 50 shares: lose 50*2 = 100
        assert result["equity_curve"].iloc[1] == pytest.approx(9_900.0)


class TestValidation:
    def test_missing_ticker_in_price_data_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.5], "GOOGL": [0.3]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="GOOGL"):
            run_portfolio_simulation({"AAPL": price_data["AAPL"]}, target_weights)

    def test_gross_leverage_exceeded_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.8], "MSFT": [0.8]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="leverage"):
            run_portfolio_simulation(price_data, target_weights, max_gross_leverage=1.0)

    def test_position_bound_exceeded_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        target_weights = pd.DataFrame({"AAPL": [0.9], "MSFT": [0.05]}, index=[dates[0]])
        with pytest.raises(ValidationError, match="max_position_pct"):
            run_portfolio_simulation(price_data, target_weights, max_position_pct=0.5)

    def test_rebalance_date_outside_price_calendar_raises(self, two_ticker_price_data):
        price_data, dates = two_ticker_price_data
        bad_date = pd.Timestamp("2023-06-01")
        target_weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.3]}, index=[bad_date])
        with pytest.raises(ValidationError, match="no price data"):
            run_portfolio_simulation(price_data, target_weights)
