"""
Tests for backtest/pairs.py — synchronized two-leg pair-trade backtest.

The synthetic 20-bar scenario below has symbol_b held flat at 100 while
symbol_a dips to 60 (bars 5-9), reverts to 100 (bars 10-14), then spikes to
140 (bars 15-19), with hedge_ratio=1.0 so spread = Close_a - Close_b. With
entry_z=1.0/exit_z=0.3 (chosen so this small sample crosses both thresholds
without needing hundreds of bars), the full-sample static z-score
(spread_zscore's default) was independently computed as: 0.0 for bars 0-4,
-1.3784 for bars 5-9, 0.0 for bars 10-14, +1.3784 for bars 15-19 — verified
via a standalone script (spread_zscore(compute_spread(...), window=None)).
"""

import pandas as pd
import pytest

from standard_quant_tools.backtest.pairs import run_pair_backtest
from standard_quant_tools.error import ValidationError


def _pair_price_data():
    dates = pd.date_range("2023-01-02", periods=20, freq="B")
    close_a = [100.0] * 5 + [60.0] * 5 + [100.0] * 5 + [140.0] * 5
    close_b = [100.0] * 20

    def _df(close):
        return pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close,
             "Volume": [1_000_000.0] * len(close)},
            index=dates,
        )

    return {"A": _df(close_a), "B": _df(close_b)}, dates


class TestRunPairBacktest:
    def test_missing_symbol_raises(self):
        price_data, _ = _pair_price_data()
        with pytest.raises(ValidationError, match="C"):
            run_pair_backtest(
                {"A": price_data["A"]}, symbol_a="A", symbol_b="C", hedge_ratio=1.0,
            )

    def test_no_entry_crossing_raises(self):
        price_data, _ = _pair_price_data()
        with pytest.raises(ValidationError, match="never crossed"):
            # entry_z=100 is unreachable given the z-range of this scenario
            run_pair_backtest(
                price_data, symbol_a="A", symbol_b="B", hedge_ratio=1.0, entry_z=100.0,
            )

    def test_transitions_and_round_trips(self):
        price_data, dates = _pair_price_data()
        result = run_pair_backtest(
            price_data, symbol_a="A", symbol_b="B", hedge_ratio=1.0,
            entry_z=1.0, exit_z=0.3, commission_pct=0.0, slippage_pct=0.0,
            zscore_window=None,
        )
        # 3 transitions: enter long-spread (bar 5), exit to flat (bar 10),
        # enter short-spread (bar 15). n_round_trips counts only completed
        # entry->exit cycles (1: the bar-10 exit).
        assert len(result["rebalance_log"]) == 3
        assert result["n_round_trips"] == 1
        assert result["state"].loc[dates[5]] == 1.0
        assert result["state"].loc[dates[10]] == 0.0
        assert result["state"].loc[dates[15]] == -1.0

    def test_both_legs_move_together_at_each_rebalance(self):
        """
        The whole point of reusing run_portfolio_simulation: both legs are
        columns of the same target_weights row, so they rebalance on
        exactly the same dates — there's no way for one leg to enter/exit
        without the other.
        """
        price_data, dates = _pair_price_data()
        result = run_pair_backtest(
            price_data, symbol_a="A", symbol_b="B", hedge_ratio=1.0,
            entry_z=1.0, exit_z=0.3, commission_pct=0.0, slippage_pct=0.0,
            zscore_window=None,
        )
        rebalance_dates = [r["date"] for r in result["rebalance_log"].to_dict(orient="records")]
        assert rebalance_dates == [str(dates[5].date()), str(dates[10].date()), str(dates[15].date())]

    def test_dollar_neutral_sizing_matches_hedge_ratio(self):
        price_data, dates = _pair_price_data()
        result = run_pair_backtest(
            price_data, symbol_a="A", symbol_b="B", hedge_ratio=1.0,
            entry_z=1.0, exit_z=0.3, gross_leverage=1.0,
            commission_pct=0.0, slippage_pct=0.0, zscore_window=None,
        )
        # hedge_ratio=1.0 -> equal-magnitude legs: weight_a = weight_b = 0.5.
        first_rebalance = result["rebalance_log"].iloc[0]
        assert first_rebalance["gross_leverage_after"] == pytest.approx(1.0, abs=1e-3)

    def test_entry_and_current_spread_reported(self):
        price_data, _ = _pair_price_data()
        result = run_pair_backtest(
            price_data, symbol_a="A", symbol_b="B", hedge_ratio=1.0,
            entry_z=1.0, exit_z=0.3, zscore_window=None,
        )
        # Most recent entry is the bar-15 short-spread entry, spread = 40.0.
        assert result["entry_spread"] == pytest.approx(40.0)
        assert result["current_spread"] == pytest.approx(40.0)
        assert result["hedge_ratio"] == 1.0

    def test_result_includes_portfolio_simulation_fields(self):
        price_data, _ = _pair_price_data()
        result = run_pair_backtest(
            price_data, symbol_a="A", symbol_b="B", hedge_ratio=1.0,
            entry_z=1.0, exit_z=0.3, zscore_window=None,
        )
        for key in ("equity_curve", "cash_curve", "gross_exposure_curve",
                    "leverage_curve", "final_equity", "final_cash", "warnings"):
            assert key in result
