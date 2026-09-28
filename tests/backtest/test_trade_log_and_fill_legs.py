"""
The fill-aware equity curve, and how exactly the trade log reconciles with it.

Under fill_price="next_open" (and "hl2_exploratory") each bar is an
overnight leg at yesterday's position and an intraday leg at today's. The
two used to be ADDED, on the C++ and the Python path alike, so a held bar
lost the product term: over five years of daily bars the equity curve fell
0.2-0.5 points below the fill-to-fill trade log at zero cost. They compound
now.

The trade log reconciles EXACTLY, costs included: each lot's return_pct is
its share of the equity curve, so the lots multiply back to the curve's
total return on both engines. It used to charge costs as a simple fraction
of notional, which agreed with the curve to first order only -- the log
overstated cumulative P&L by 0.07-0.15 points at 15 bps.
"""

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.engine as engine
from standard_quant_tools.backtest.engine import run_strategy
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY


@pytest.fixture(params=["native", "python"])
def engine_path(request, monkeypatch):
    if request.param == "native":
        if not engine.HAS_CPP:
            pytest.skip("the C++ extension is not built")
    else:
        monkeypatch.setattr(engine, "HAS_CPP", False)
    return request.param


@pytest.fixture(scope="module")
def gapping() -> pd.DataFrame:
    """A random walk whose every bar opens away from the prior close."""
    rng = np.random.default_rng(21)
    n = 400
    gaps = rng.normal(0.0, 0.006, n)
    days = rng.normal(0.0004, 0.01, n)
    opens, closes = np.empty(n), np.empty(n)
    prev = 100.0
    for i in range(n):
        opens[i] = prev * (1.0 + gaps[i])
        closes[i] = opens[i] * (1.0 + days[i])
        prev = closes[i]
    return pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(opens, closes) * 1.002,
            "Low": np.minimum(opens, closes) * 0.998,
            "Close": closes,
            "Volume": 1e6,
        },
        index=pd.date_range("2020-01-02", periods=n, freq="B"),
    )


def _hand_built(frame: pd.DataFrame, signal: np.ndarray, capital: float) -> np.ndarray:
    """Carry yesterday's position from the prior close to today's open,
    re-size at the open, carry today's position to the close."""
    opens, closes = frame["Open"].to_numpy(), frame["Close"].to_numpy()
    equity = np.full(len(frame), capital)
    for i in range(1, len(frame)):
        overnight_pos = signal[i - 2] if i >= 2 else 0.0
        intraday_pos = signal[i - 1]
        at_open = equity[i - 1] * (
            1.0 + overnight_pos * (opens[i] / closes[i - 1] - 1.0)
        )
        equity[i] = at_open * (1.0 + intraday_pos * (closes[i] / opens[i] - 1.0))
    return equity


class TestTheLegsCompound:
    def test_zero_cost_curve_is_the_hand_built_account(self, engine_path, gapping):
        rng = np.random.default_rng(8)
        signal = rng.choice([-1.0, 0.0, 0.5, 1.0], size=len(gapping))
        result = run_strategy(
            gapping,
            pd.Series(signal, index=gapping.index),
            commission_pct=0.0,
            slippage_pct=0.0,
            fill_price="next_open",
        )
        np.testing.assert_allclose(
            result["equity_curve"].to_numpy(),
            _hand_built(gapping, signal, 10_000.0),
            rtol=1e-12,
        )

    def test_a_held_position_earns_the_whole_move(self, engine_path, gapping):
        """Long throughout: filled at the first open, marked to the last close."""
        result = run_strategy(
            gapping,
            pd.Series(1.0, index=gapping.index),
            commission_pct=0.0,
            slippage_pct=0.0,
            fill_price="next_open",
        )
        expected = gapping["Close"].iloc[-1] / gapping["Open"].iloc[1] - 1.0
        assert result["total_return"] == pytest.approx(expected, abs=1e-6)


class TestHowExactlyTheLogReconciles:
    @pytest.mark.parametrize("fill", ["close", "next_open"])
    def test_at_zero_cost_every_unit_lot_is_its_share_of_the_curve(
        self, engine_path, gapping, fill
    ):
        """Exact, lot by lot: the equity curve's growth from the bar before
        entry to the exit bar is the lot's fill-to-fill return. Under
        next_open this failed by the dropped product term on every held
        bar."""
        signal = STRATEGY_REGISTRY["sma_crossover"](
            gapping, fast_period=5, slow_period=20
        )
        result = run_strategy(
            gapping,
            signal,
            commission_pct=0.0,
            slippage_pct=0.0,
            fill_price=fill,
            include_trade_log=True,
        )
        equity = result["equity_curve"]
        log = result["trade_log"]
        assert len(log) > 3
        for _, lot in log.iterrows():
            before_entry = equity.index.get_loc(lot["entry_date"]) - 1
            growth = equity.loc[lot["exit_date"]] / equity.iloc[before_entry]
            assert lot["return_pct"] == pytest.approx((growth - 1.0) * 100, abs=1e-4)

    def test_with_costs_a_lot_is_exactly_its_share_of_the_curve(self, engine_path):
        """One long lot 100 -> 120 at 15 bps a side. The curve charges each
        cost against the equity of the bar it is paid on and compounds:
        (1 - c) * 1.2 * (1 - c) - 1 = 19.64%. The log now reads the lot off
        that curve. It used to charge both costs as a simple fraction,
        0.20 - 2c = 19.70%, a gap of about 2 * c * |r| + c^2 per lot that
        its docstring had to disclose."""
        c = 0.0015
        dates = pd.date_range("2023-01-02", periods=6, freq="B")
        close = [100.0, 100.0, 110.0, 120.0, 120.0, 120.0]
        frame = pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close},
            index=dates,
        )
        signal = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=dates)
        result = run_strategy(
            frame, signal, commission_pct=c, slippage_pct=0.0, include_trade_log=True
        )
        (lot_return,) = result["trade_log"]["return_pct"] / 100
        curve_return = result["total_return"]
        assert curve_return == pytest.approx((1 - c) * 1.2 * (1 - c) - 1, abs=1e-6)
        assert lot_return == pytest.approx(curve_return, abs=5e-7)
        assert result["avg_trade_return_pct"] / 100 == pytest.approx(
            curve_return, abs=5e-7
        )


class TestTheLogCompoundsToTheCurve:
    """
    The whole log, not one lot: the product of (1 + return_pct / 100) over
    every lot is the curve's final equity over its initial capital, under
    every fill and at every size, with long, short, resized and flipped
    lots. Flat bars contribute exactly 1, and a flip bar is split where the
    closing lot has earned the leg it still held and paid its exit cost.

    The tolerance is the log's display rounding (return_pct to 4 decimals,
    5e-7 per lot), not an allowance for a modelling gap: at 15 bps the old
    log sat 0.07-0.15 points (7e-4 to 1.5e-3) away from the curve.
    """

    @staticmethod
    def _compounded(log: pd.DataFrame) -> float:
        return float(np.prod(1.0 + log["return_pct"].to_numpy(dtype=float) / 100.0))

    @pytest.mark.parametrize("fill", ["close", "next_open", "hl2_exploratory"])
    @pytest.mark.parametrize(
        "levels", [[-1.0, 0.0, 0.5, 1.0], [0.0, 1.0], [-2.0, -1.0, 1.0, 2.0]]
    )
    def test_the_lots_multiply_back_to_the_curve(
        self, engine_path, gapping, fill, levels
    ):
        rng = np.random.default_rng(len(levels) * 7 + len(fill))
        # Runs of several bars, so lots last more than one bar.
        blocks = rng.choice(levels, size=len(gapping) // 5 + 1)
        signal = pd.Series(np.repeat(blocks, 5)[: len(gapping)], index=gapping.index)
        result = run_strategy(
            gapping,
            signal,
            commission_pct=0.001,
            slippage_pct=0.0005,
            fill_price=fill,
            include_trade_log=True,
        )
        log = result["trade_log"]
        assert len(log) > 5
        growth = float(result["equity_curve"].iloc[-1] / result["equity_curve"].iloc[0])
        assert self._compounded(log) == pytest.approx(growth, rel=len(log) * 1e-6)
        # The native kernel's own trade stats read the same lots.
        assert result["num_trades"] == len(log)
        assert result["avg_trade_return_pct"] == pytest.approx(
            float(log["return_pct"].mean()), abs=5e-5
        )

    def test_null_case_no_trade_no_log(self, engine_path, gapping):
        result = run_strategy(
            gapping,
            pd.Series(0.0, index=gapping.index),
            commission_pct=0.001,
            slippage_pct=0.0005,
            include_trade_log=True,
        )
        assert result["trade_log"].empty
        assert result["total_return"] == 0.0
