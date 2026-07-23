"""
Hand-verified tests for run_strategy's fill_price="next_open" mode
(backtest/engine.py). The expected numbers below were computed by hand
(and cross-checked with an independent script) for a small 5-bar example
with one entry (bar 2) and one exit (bar 4), so both legs of the
overnight/intraday decomposition are exercised.
"""

import pandas as pd
import pytest

from standard_quant_tools.backtest.engine import run_strategy, backtest_grid


@pytest.fixture
def small_ohlcv() -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "Open":  [100.0, 101.0, 103.0, 102.0, 106.0],
            "High":  [101.0, 103.0, 104.0, 106.0, 107.0],
            "Low":   [99.0, 100.0, 100.0, 101.0, 102.0],
            "Close": [100.0, 102.0, 101.0, 105.0, 103.0],
            "Volume": [1_000_000.0] * 5,
        },
        index=dates,
    )


@pytest.fixture
def small_signals(small_ohlcv) -> pd.Series:
    # signals[i] known using data through bar i; executed[i] = signals[i-1].
    # executed = [0, 0, 1, 1, 0] -> enter at bar 2, hold bar 3, exit going into bar 4.
    return pd.Series([0.0, 1.0, 1.0, 0.0, 0.0], index=small_ohlcv.index)


class TestFillPriceClose:
    def test_default_fill_price_matches_explicit_close(self, small_ohlcv, small_signals):
        default = run_strategy(small_ohlcv, small_signals, 10_000.0, commission_pct=0.0, slippage_pct=0.0)
        explicit = run_strategy(
            small_ohlcv, small_signals, 10_000.0, commission_pct=0.0, slippage_pct=0.0, fill_price="close",
        )
        assert default["total_return"] == explicit["total_return"]
        assert default["final_equity"] == explicit["final_equity"]

    def test_close_mode_hand_verified_equity(self, small_ohlcv, small_signals):
        """
        executed = [0,0,1,1,0]; close-to-close returns applied only on bars
        2 and 3 (held bars): -0.009804 then +0.039604.
        equity: 10000 -> 9901.96 -> 10294.16 (bars 0,1,4 contribute 0).
        """
        result = run_strategy(small_ohlcv, small_signals, 10_000.0, commission_pct=0.0, slippage_pct=0.0)
        assert result["final_equity"] == pytest.approx(10294.16, abs=0.05)


class TestFillPriceNextOpen:
    def test_hand_verified_equity_curve(self, small_ohlcv, small_signals):
        """
        Two-leg decomposition (see test module docstring for the setup):
        gross returns per bar = [0, 0, -0.019417, 0.039313, 0.009524]
        (bar 2 = entry, open-to-close only; bar 3 = held, overnight+intraday
        sum; bar 4 = exit, overnight-gap only — position was flat all of
        bar 4 itself). Independently verified via a standalone script.
        """
        result = run_strategy(
            small_ohlcv, small_signals, 10_000.0,
            commission_pct=0.0, slippage_pct=0.0, fill_price="next_open",
        )
        equity = result["equity_curve"]
        expected = [10000.0, 10000.0, 9805.83, 10191.32, 10288.38]
        for actual, exp in zip(equity.tolist(), expected):
            assert actual == pytest.approx(exp, abs=0.05)
        assert result["final_equity"] == pytest.approx(10288.38, abs=0.05)

    def test_next_open_differs_from_close(self, small_ohlcv, small_signals):
        close_result = run_strategy(small_ohlcv, small_signals, 10_000.0, commission_pct=0.0, slippage_pct=0.0)
        next_open_result = run_strategy(
            small_ohlcv, small_signals, 10_000.0, commission_pct=0.0, slippage_pct=0.0, fill_price="next_open",
        )
        assert close_result["final_equity"] != pytest.approx(next_open_result["final_equity"], abs=1e-6)

    def test_exit_day_overnight_gap_is_not_dropped(self, small_ohlcv, small_signals):
        """
        Regression test for the specific bug caught during development: an
        earlier draft zeroed out the exit bar's contribution entirely
        (multiplying by today's now-flat executed value), silently dropping
        the overnight gap the position was still exposed to before selling
        at the open. Bar 4 (the exit bar) must contribute a nonzero return
        here since the overnight gap (Close[3]->Open[4]) was positive.
        """
        result = run_strategy(
            small_ohlcv, small_signals, 10_000.0,
            commission_pct=0.0, slippage_pct=0.0, fill_price="next_open",
        )
        equity = result["equity_curve"]
        bar4_return = equity.iloc[4] / equity.iloc[3] - 1
        assert bar4_return != pytest.approx(0.0, abs=1e-9)
        assert bar4_return == pytest.approx(0.009524, abs=1e-4)

    def test_flat_series_all_zero_returns(self, small_ohlcv):
        flat_signals = pd.Series(0.0, index=small_ohlcv.index)
        result = run_strategy(
            small_ohlcv, flat_signals, 10_000.0,
            commission_pct=0.0, slippage_pct=0.0, fill_price="next_open",
        )
        assert result["final_equity"] == pytest.approx(10_000.0)


class TestFillPriceMidpoint:
    def test_hand_verified_equity_curve(self, small_ohlcv, small_signals):
        """
        Same two-leg decomposition as next_open, but the reference price
        each bar is (High + Low) / 2 instead of Open: ref = [100.0, 101.5,
        102.0, 103.5, 104.5]. Independently verified via a standalone script.
        """
        result = run_strategy(
            small_ohlcv, small_signals, 10_000.0,
            commission_pct=0.0, slippage_pct=0.0, fill_price="midpoint",
        )
        equity = result["equity_curve"]
        expected = [10000.0, 10000.0, 9901.9608, 10290.5655, 10241.5628]
        for actual, exp in zip(equity.tolist(), expected):
            assert actual == pytest.approx(exp, abs=0.05)
        assert result["final_equity"] == pytest.approx(10241.5628, abs=0.05)

    def test_midpoint_differs_from_close_and_next_open(self, small_ohlcv, small_signals):
        close_result = run_strategy(small_ohlcv, small_signals, 10_000.0, commission_pct=0.0, slippage_pct=0.0)
        next_open_result = run_strategy(
            small_ohlcv, small_signals, 10_000.0, commission_pct=0.0, slippage_pct=0.0, fill_price="next_open",
        )
        midpoint_result = run_strategy(
            small_ohlcv, small_signals, 10_000.0, commission_pct=0.0, slippage_pct=0.0, fill_price="midpoint",
        )
        assert midpoint_result["final_equity"] != pytest.approx(close_result["final_equity"], abs=1e-6)
        assert midpoint_result["final_equity"] != pytest.approx(next_open_result["final_equity"], abs=1e-6)

    def test_flat_series_all_zero_returns(self, small_ohlcv):
        flat_signals = pd.Series(0.0, index=small_ohlcv.index)
        result = run_strategy(
            small_ohlcv, flat_signals, 10_000.0,
            commission_pct=0.0, slippage_pct=0.0, fill_price="midpoint",
        )
        assert result["final_equity"] == pytest.approx(10_000.0)


class TestFillPriceBacktestGrid:
    def test_backtest_grid_threads_fill_price(self, small_ohlcv):
        def my_signal(df: pd.DataFrame, level: float) -> pd.Series:
            return (df["Close"] > level).astype(float)

        grid_close = backtest_grid(
            small_ohlcv, strategy=my_signal, param_grid={"level": [100.0]},
            n_workers=1, fill_price="close",
        )
        grid_next_open = backtest_grid(
            small_ohlcv, strategy=my_signal, param_grid={"level": [100.0]},
            n_workers=1, fill_price="next_open",
        )
        assert grid_close.iloc[0]["final_equity"] != pytest.approx(
            grid_next_open.iloc[0]["final_equity"], abs=1e-6
        )

    def test_backtest_grid_threads_midpoint_fill_price(self, small_ohlcv):
        def my_signal(df: pd.DataFrame, level: float) -> pd.Series:
            return (df["Close"] > level).astype(float)

        grid_close = backtest_grid(
            small_ohlcv, strategy=my_signal, param_grid={"level": [100.0]},
            n_workers=1, fill_price="close",
        )
        grid_midpoint = backtest_grid(
            small_ohlcv, strategy=my_signal, param_grid={"level": [100.0]},
            n_workers=1, fill_price="midpoint",
        )
        assert grid_close.iloc[0]["final_equity"] != pytest.approx(
            grid_midpoint.iloc[0]["final_equity"], abs=1e-6
        )

    def test_backtest_grid_default_fill_price_is_close(self, small_ohlcv):
        def my_signal(df: pd.DataFrame, level: float) -> pd.Series:
            return (df["Close"] > level).astype(float)

        default = backtest_grid(small_ohlcv, strategy=my_signal, param_grid={"level": [100.0]}, n_workers=1)
        explicit = backtest_grid(
            small_ohlcv, strategy=my_signal, param_grid={"level": [100.0]}, n_workers=1, fill_price="close",
        )
        assert default.iloc[0]["final_equity"] == explicit.iloc[0]["final_equity"]
