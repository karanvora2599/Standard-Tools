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


def _pair_price_data_unequal_prices():
    """
    Symbol B held flat at 50 throughout; symbol A at 200 (bars 0-4), dips
    to 40 for bars 5-9 (entry), back to 200 (bars 10-14, exit), spikes to
    360 (bars 15-19, short entry). Unlike _pair_price_data (equal price
    levels, hedge_ratio=1.0 -- exactly why the old sizing bug went
    uncaught), A and B trade at very different price levels here, so a
    hedge_ratio != 1.0 exercises the price-aware per-transition-date
    sizing fix (item 6).
    """
    dates = pd.date_range("2023-01-02", periods=20, freq="B")
    close_a = [200.0] * 5 + [40.0] * 5 + [200.0] * 5 + [360.0] * 5
    close_b = [50.0] * 20

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

    def test_high_leverage_low_hedge_ratio_does_not_falsely_reject(self):
        """
        weight_a = gross_leverage / (1 + |hedge_ratio|) can exceed 1.0 for a
        small hedge_ratio combined with gross_leverage > 1 (e.g. 2.0 / 1.3 ≈
        1.54) — this must not be rejected against the engine's default
        max_position_pct=1.0, since gross_leverage=2.0 is itself a valid,
        explicitly-requested input.
        """
        price_data, _ = _pair_price_data()
        result = run_pair_backtest(
            price_data, symbol_a="A", symbol_b="B", hedge_ratio=0.3,
            entry_z=1.0, exit_z=0.3, gross_leverage=2.0,
            commission_pct=0.0, slippage_pct=0.0, zscore_window=None,
        )
        first_rebalance = result["rebalance_log"].iloc[0]
        assert first_rebalance["gross_leverage_after"] == pytest.approx(2.0, abs=1e-2)

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

    def test_hedge_ratio_sizing_is_share_ratio_not_dollar_ratio(self):
        """
        Regression test (P0 item 6): hedge_ratio is a SHARE ratio (spread
        = Close_a - hedge_ratio * Close_b), so hedging 1 share of A means
        holding hedge_ratio shares of B -- shares_b/shares_a must equal
        hedge_ratio, using each transition date's own prices. The old
        formula (weight_a = gross_leverage/(1+|hedge_ratio|)) ignored
        price entirely and only "happened" to work when Close_a ~=
        Close_b, exactly the case the pre-existing hedge_ratio=1.0
        fixture (_pair_price_data, equal price levels) could never catch.

        gross_exposure_curve/net_exposure_curve (both already-public
        outputs) are used to back out each leg's dollar allocation without
        needing a new per-ticker-shares output: for a long-spread position
        (state=+1, long A / short B), dollar_a = (gross+net)/2 and
        dollar_b = (gross-net)/2.
        """
        price_data, dates = _pair_price_data_unequal_prices()
        result = run_pair_backtest(
            price_data, symbol_a="A", symbol_b="B", hedge_ratio=2.0,
            entry_z=1.0, exit_z=0.3, gross_leverage=1.0,
            commission_pct=0.0, slippage_pct=0.0, zscore_window=None,
            fill_price="close",
        )
        entry_date = dates[5]
        assert result["state"].loc[entry_date] == 1.0  # long A, short B
        gross = float(result["gross_exposure_curve"].loc[entry_date])
        net = float(result["net_exposure_curve"].loc[entry_date])
        dollar_a = (gross + net) / 2.0
        dollar_b = (gross - net) / 2.0
        price_a, price_b = 40.0, 50.0  # Close at bar 5 (the entry bar)
        shares_a = dollar_a / price_a
        shares_b = dollar_b / price_b
        assert shares_b / shares_a == pytest.approx(2.0, rel=1e-3)

    def test_default_fill_price_defers_execution_to_next_bar(self):
        """
        Regression test (P0 item 7): the z-score signal at a transition
        date is computed from that date's own Close, so executing at that
        same Close (the old default) is look-ahead. run_pair_backtest's
        default must defer execution to the following bar's Open.
        """
        price_data, dates = _pair_price_data()
        result = run_pair_backtest(
            price_data, symbol_a="A", symbol_b="B", hedge_ratio=1.0,
            entry_z=1.0, exit_z=0.3, commission_pct=0.0, slippage_pct=0.0,
            zscore_window=None,
            # fill_price intentionally omitted -- this proves the new default.
        )
        entry_date, day_after = dates[5], dates[6]
        assert result["cash_curve"].loc[entry_date] == pytest.approx(10_000.0)
        assert result["gross_exposure_curve"].loc[entry_date] == pytest.approx(0.0)
        assert result["gross_exposure_curve"].loc[day_after] > 0.0

    def test_result_includes_portfolio_simulation_fields(self):
        price_data, _ = _pair_price_data()
        result = run_pair_backtest(
            price_data, symbol_a="A", symbol_b="B", hedge_ratio=1.0,
            entry_z=1.0, exit_z=0.3, zscore_window=None,
        )
        for key in ("equity_curve", "cash_curve", "gross_exposure_curve",
                    "leverage_curve", "final_equity", "final_cash", "warnings"):
            assert key in result
