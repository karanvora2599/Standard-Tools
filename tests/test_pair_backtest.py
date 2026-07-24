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
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": [1_000_000.0] * len(close),
            },
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
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": [1_000_000.0] * len(close),
            },
            index=dates,
        )

    return {"A": _df(close_a), "B": _df(close_b)}, dates


def _pair_price_data_unequal_overnight_gaps():
    """
    Same Close path as _pair_price_data_unequal_prices (A dips 200->40 at
    bar 5, entry trigger), but Open != Close at the execution bar (bar 6,
    the bar AFTER the trigger, where fill_price="next_open" actually
    executes the trade) -- and, crucially, A and B gap by DIFFERENT
    percentages overnight: A gaps up from 40 to 45 (+12.5%), B does not gap
    at all (50 -> 50). This is exactly the scenario P0-1 identifies: sizing
    weights from the trigger date's own Close (bar 5) and then executing at
    a DIFFERENT bar's Open (bar 6) breaks the hedge share ratio unless the
    two legs' overnight gaps happen to be identical -- which every other
    fixture in this file (Open == Close everywhere) can never expose.
    """
    dates = pd.date_range("2023-01-02", periods=20, freq="B")
    close_a = [200.0] * 5 + [40.0] * 5 + [200.0] * 5 + [360.0] * 5
    close_b = [50.0] * 20
    open_a = list(close_a)
    open_b = list(close_b)
    open_a[6] = 45.0  # +12.5% overnight gap from bar 5's Close (40) into bar 6

    def _df(close, open_):
        return pd.DataFrame(
            {
                "Open": open_,
                "High": [max(c, o) for c, o in zip(close, open_)],
                "Low": [min(c, o) for c, o in zip(close, open_)],
                "Close": close,
                "Volume": [1_000_000.0] * len(close),
            },
            index=dates,
        )

    return {"A": _df(close_a, open_a), "B": _df(close_b, open_b)}, dates


class TestRunPairBacktest:
    def test_missing_symbol_raises(self):
        price_data, _ = _pair_price_data()
        with pytest.raises(ValidationError, match="C"):
            run_pair_backtest(
                {"A": price_data["A"]},
                symbol_a="A",
                symbol_b="C",
                hedge_ratio=1.0,
            )

    def test_no_entry_crossing_raises(self):
        price_data, _ = _pair_price_data()
        with pytest.raises(ValidationError, match="never crossed"):
            # entry_z=100 is unreachable given the z-range of this scenario
            run_pair_backtest(
                price_data,
                symbol_a="A",
                symbol_b="B",
                hedge_ratio=1.0,
                entry_z=100.0,
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
            price_data,
            symbol_a="A",
            symbol_b="B",
            hedge_ratio=0.3,
            entry_z=1.0,
            exit_z=0.3,
            gross_leverage=2.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            zscore_window=None,
        )
        first_rebalance = result["rebalance_log"].iloc[0]
        assert first_rebalance["gross_leverage_after"] == pytest.approx(2.0, abs=1e-2)

    def test_transitions_and_round_trips(self):
        price_data, dates = _pair_price_data()
        result = run_pair_backtest(
            price_data,
            symbol_a="A",
            symbol_b="B",
            hedge_ratio=1.0,
            entry_z=1.0,
            exit_z=0.3,
            commission_pct=0.0,
            slippage_pct=0.0,
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
            price_data,
            symbol_a="A",
            symbol_b="B",
            hedge_ratio=1.0,
            entry_z=1.0,
            exit_z=0.3,
            commission_pct=0.0,
            slippage_pct=0.0,
            zscore_window=None,
        )
        rebalance_dates = [
            r["date"] for r in result["rebalance_log"].to_dict(orient="records")
        ]
        assert rebalance_dates == [
            str(dates[5].date()),
            str(dates[10].date()),
            str(dates[15].date()),
        ]

    def test_dollar_neutral_sizing_matches_hedge_ratio(self):
        price_data, dates = _pair_price_data()
        result = run_pair_backtest(
            price_data,
            symbol_a="A",
            symbol_b="B",
            hedge_ratio=1.0,
            entry_z=1.0,
            exit_z=0.3,
            gross_leverage=1.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            zscore_window=None,
        )
        # hedge_ratio=1.0 -> equal-magnitude legs: weight_a = weight_b = 0.5.
        first_rebalance = result["rebalance_log"].iloc[0]
        assert first_rebalance["gross_leverage_after"] == pytest.approx(1.0, abs=1e-3)

    def test_entry_and_current_spread_reported(self):
        price_data, _ = _pair_price_data()
        result = run_pair_backtest(
            price_data,
            symbol_a="A",
            symbol_b="B",
            hedge_ratio=1.0,
            entry_z=1.0,
            exit_z=0.3,
            zscore_window=None,
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
            price_data,
            symbol_a="A",
            symbol_b="B",
            hedge_ratio=2.0,
            entry_z=1.0,
            exit_z=0.3,
            gross_leverage=1.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            zscore_window=None,
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

    def test_hedge_ratio_sizing_correct_under_default_next_open_with_unequal_gaps(self):
        """
        Regression test (P0-1): with the default fill_price="next_open",
        the trade EXECUTES at the bar after the trigger date's Open, not
        the trigger date's own Close. Sizing weights from the trigger
        date's Close (bar 5) and then executing at a DIFFERENT bar's Open
        (bar 6) breaks shares_b/shares_a == hedge_ratio unless A and B
        happen to have identical overnight gaps -- which the "close"-mode
        test above, and every fixture with Open == Close, can never expose.
        This fixture's bar 6 has A gap +12.5% (40 -> 45) while B does not
        gap at all (50 -> 50): the fix must size off bar 6's own Open
        (45, 50), not bar 5's Close (40, 50).

        gross_exposure_curve/net_exposure_curve (used by the sibling
        "close"-mode test above) are marked at each bar's own Close, so
        they can't be used here: bar 6's Close (40, unchanged from bar 5 --
        this fixture only moves Open) differs from its Open (45, the
        actual execution price), so Close-marked exposure would reflect an
        intraday price move, not the trade's actual execution size.
        cash_curve instead moves by exactly delta*execution_price at the
        moment of execution, unaffected by later marking, so it's used to
        back out each leg's dollar allocation at the true execution price:
        cash_after = initial_capital - dollar_a + dollar_b (long A, short
        B), and dollar_a + dollar_b = gross_leverage * initial_capital
        always (weights sum to gross_leverage regardless of execution
        price) -- solving those two equations gives dollar_b = cash_after / 2,
        dollar_a = initial_capital - dollar_b.
        """
        price_data, dates = _pair_price_data_unequal_overnight_gaps()
        initial_capital = 10_000.0
        result = run_pair_backtest(
            price_data,
            symbol_a="A",
            symbol_b="B",
            hedge_ratio=2.0,
            entry_z=1.0,
            exit_z=0.3,
            gross_leverage=1.0,
            initial_capital=initial_capital,
            commission_pct=0.0,
            slippage_pct=0.0,
            zscore_window=None,
            # fill_price intentionally omitted -- exercises the new default.
        )
        entry_date = dates[5]
        exec_date = dates[6]
        assert result["state"].loc[entry_date] == 1.0  # long A, short B
        # No trade has executed yet at the trigger bar itself.
        assert result["cash_curve"].loc[entry_date] == pytest.approx(initial_capital)

        cash_after = float(result["cash_curve"].loc[exec_date])
        dollar_b = cash_after / 2.0
        dollar_a = initial_capital - dollar_b
        price_a, price_b = 45.0, 50.0  # Open at bar 6 (the actual execution bar)
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
            price_data,
            symbol_a="A",
            symbol_b="B",
            hedge_ratio=1.0,
            entry_z=1.0,
            exit_z=0.3,
            commission_pct=0.0,
            slippage_pct=0.0,
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
            price_data,
            symbol_a="A",
            symbol_b="B",
            hedge_ratio=1.0,
            entry_z=1.0,
            exit_z=0.3,
            zscore_window=None,
        )
        for key in (
            "equity_curve",
            "cash_curve",
            "gross_exposure_curve",
            "leverage_curve",
            "final_equity",
            "final_cash",
            "warnings",
        ):
            assert key in result
