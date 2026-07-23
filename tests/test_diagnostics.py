"""Tests for extended backtest diagnostics (metrics/diagnostics.py)."""

import pandas as pd
import pytest

from standard_quant_tools.metrics.diagnostics import (
    drawdown_periods,
    top_n_drawdowns,
    trade_expectancy,
    trade_excursions,
    exposure_stats,
)


# ── drawdown_periods / top_n_drawdowns ──────────────────────────────────────

class TestDrawdownPeriods:
    def test_empty_series(self):
        result = drawdown_periods(pd.Series(dtype=float))
        assert result.empty

    def test_no_drawdown_monotonic_series(self):
        dates = pd.date_range("2022-01-01", periods=5, freq="B")
        equity = pd.Series([100, 105, 110, 115, 120], index=dates)
        result = drawdown_periods(equity)
        assert result.empty

    def test_two_episodes_hand_verified(self):
        """
        equity = [100, 110, 105, 95, 100, 115, 110, 90]
        Episode 1: peak@t1(110) -> trough@t3(95) -> recovered@t5(115).
                   depth = (95-110)/110 = -0.13636..., duration=4, recovery=2.
        Episode 2 (unrecovered at series end): peak@t5(115) -> trough@t7(90).
                   depth = (90-115)/115 = -0.21739..., duration=2, recovery=None.
        """
        dates = pd.date_range("2022-01-01", periods=8, freq="B")
        equity = pd.Series([100, 110, 105, 95, 100, 115, 110, 90], index=dates)
        result = drawdown_periods(equity)

        assert len(result) == 2

        ep1 = result.iloc[0]
        assert ep1["start"] == dates[1]
        assert ep1["trough"] == dates[3]
        assert ep1["end"] == dates[5]
        assert ep1["depth"] == pytest.approx((95 - 110) / 110, abs=1e-6)
        assert ep1["duration_bars"] == 4
        assert ep1["recovery_bars"] == 2

        ep2 = result.iloc[1]
        assert ep2["start"] == dates[5]
        assert ep2["trough"] == dates[7]
        # pandas homogenizes the "end" column to datetime64 dtype (since
        # episode 1 has a real Timestamp) so an unrecovered episode's None
        # comes back as NaT, not None — expected pandas behavior.
        assert pd.isna(ep2["end"])
        assert ep2["depth"] == pytest.approx((90 - 115) / 115, abs=1e-6)
        assert ep2["duration_bars"] == 2
        # Same column-homogenization behavior as "end": mixing an int (from
        # episode 1) with None promotes the column to float64 with NaN.
        assert pd.isna(ep2["recovery_bars"])

    def test_top_n_sorted_worst_first(self):
        dates = pd.date_range("2022-01-01", periods=8, freq="B")
        equity = pd.Series([100, 110, 105, 95, 100, 115, 110, 90], index=dates)
        top = top_n_drawdowns(equity, n=1)
        assert len(top) == 1
        # Episode 2 (-0.217) is deeper than episode 1 (-0.136).
        assert top.iloc[0]["depth"] == pytest.approx((90 - 115) / 115, abs=1e-6)

    def test_top_n_on_empty_periods(self):
        dates = pd.date_range("2022-01-01", periods=3, freq="B")
        equity = pd.Series([100, 105, 110], index=dates)
        assert top_n_drawdowns(equity).empty


# ── trade_expectancy ─────────────────────────────────────────────────────

class TestTradeExpectancy:
    def test_empty_trade_log(self):
        result = trade_expectancy(pd.DataFrame(columns=["return_pct"]))
        assert result["expectancy_pct"] == 0.0
        assert result["max_consecutive_wins"] == 0

    def test_hand_verified_expectancy(self):
        """
        Trades: +10, +10, -5, +10, -5, -5 (percent).
        win_rate = 3/6 = 0.5, avg_winner = 10, avg_loser = -5.
        expectancy = 0.5*10 + 0.5*(-5) = 2.5. payoff_ratio = 10/5 = 2.0.
        Win/loss sequence: W W L W L L -> max_consecutive_wins=2, max_consecutive_losses=2.
        """
        trade_log = pd.DataFrame({"return_pct": [10.0, 10.0, -5.0, 10.0, -5.0, -5.0]})
        result = trade_expectancy(trade_log)
        assert result["expectancy_pct"] == pytest.approx(2.5)
        assert result["avg_winner_pct"] == pytest.approx(10.0)
        assert result["avg_loser_pct"] == pytest.approx(-5.0)
        assert result["payoff_ratio"] == pytest.approx(2.0)
        assert result["max_consecutive_wins"] == 2
        assert result["max_consecutive_losses"] == 2

    def test_all_winners_payoff_ratio_is_inf(self):
        trade_log = pd.DataFrame({"return_pct": [5.0, 3.0, 8.0]})
        result = trade_expectancy(trade_log)
        assert result["payoff_ratio"] == float("inf")
        assert result["max_consecutive_wins"] == 3
        assert result["max_consecutive_losses"] == 0


# ── trade_excursions (MAE/MFE) ───────────────────────────────────────────

class TestTradeExcursions:
    def test_empty_trade_log(self):
        result = trade_excursions(
            pd.DataFrame(columns=["entry_date", "exit_date", "direction", "entry_price"]),
            pd.DataFrame(columns=["High", "Low"]),
        )
        assert result.empty
        assert "mae_pct" in result.columns
        assert "mfe_pct" in result.columns

    def test_long_trade_hand_verified(self):
        dates = pd.date_range("2022-01-01", periods=5, freq="B")
        price_data = pd.DataFrame({
            "High": [101, 108, 112, 95, 100],
            "Low":  [99, 102, 90, 92, 98],
        }, index=dates)
        trade_log = pd.DataFrame({
            "entry_date": [dates[0]],
            "exit_date": [dates[2]],
            "direction": ["long"],
            "entry_price": [100.0],
        })
        result = trade_excursions(trade_log, price_data)
        # Window is bars 0-2: High max=112, Low min=90.
        assert result.iloc[0]["mfe_pct"] == pytest.approx((112 - 100) / 100 * 100)
        assert result.iloc[0]["mae_pct"] == pytest.approx((90 - 100) / 100 * 100)

    def test_short_trade_hand_verified(self):
        dates = pd.date_range("2022-01-01", periods=5, freq="B")
        price_data = pd.DataFrame({
            "High": [101, 108, 112, 95, 100],
            "Low":  [99, 102, 90, 92, 98],
        }, index=dates)
        trade_log = pd.DataFrame({
            "entry_date": [dates[0]],
            "exit_date": [dates[2]],
            "direction": ["short"],
            "entry_price": [100.0],
        })
        result = trade_excursions(trade_log, price_data)
        # Short: favorable move is price falling (Low min=90), adverse is price rising (High max=112).
        assert result.iloc[0]["mfe_pct"] == pytest.approx((100 - 90) / 100 * 100)
        assert result.iloc[0]["mae_pct"] == pytest.approx((100 - 112) / 100 * 100)


# ── exposure_stats ────────────────────────────────────────────────────────

class TestExposureStats:
    def test_all_flat(self):
        dates = pd.date_range("2022-01-01", periods=4, freq="B")
        executed = pd.Series([0.0, 0.0, 0.0, 0.0], index=dates)
        result = exposure_stats(executed)
        assert result["time_in_market"] == 0.0
        assert result["avg_holding_period_bars"] is None

    def test_hand_verified_long_short_mix(self):
        dates = pd.date_range("2022-01-01", periods=5, freq="B")
        executed = pd.Series([1.0, 1.0, 0.0, -1.0, 0.0], index=dates)
        result = exposure_stats(executed)
        assert result["time_in_market"] == pytest.approx(3 / 5)
        assert result["pct_long"] == pytest.approx(2 / 5)
        assert result["pct_short"] == pytest.approx(1 / 5)
        assert result["avg_gross_exposure"] == pytest.approx((1 + 1 + 0 + 1 + 0) / 5)
        assert result["avg_net_exposure"] == pytest.approx((1 + 1 + 0 - 1 + 0) / 5)

    def test_avg_holding_period_from_trade_log(self):
        dates = pd.date_range("2022-01-01", periods=6, freq="B")
        executed = pd.Series([1.0, 1.0, 1.0, 0.0, -1.0, -1.0], index=dates)
        trade_log = pd.DataFrame({
            "entry_date": [dates[0], dates[4]],
            "exit_date": [dates[2], dates[5]],
        })
        result = exposure_stats(executed, trade_log)
        # Trade 1: bars 0->2 = 2 bars held. Trade 2: bars 4->5 = 1 bar held. avg = 1.5.
        assert result["avg_holding_period_bars"] == pytest.approx(1.5)
