"""
A profit factor where every trade returned exactly zero is 0/0: NaN.

Gross profit over gross loss. With a gross profit and no gross loss it is
x/0 with x > 0, +inf -- the "never lost" reading. With neither, every trade
returned exactly 0.0 and the ratio is undefined, like a Sharpe with no
dispersion or a Sortino of a book that never moved. It used to be +inf on
both engines, which ranked a run of do-nothing trades first under
sort_by="profit_factor".
"""

import math

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.engine as engine
from standard_quant_tools.backtest.engine import _undefined_ratios_as_nan, run_strategy


@pytest.fixture(params=["native", "python"])
def engine_path(request, monkeypatch):
    if request.param == "native":
        if not engine.HAS_CPP:
            pytest.skip("the C++ extension is not built")
    else:
        monkeypatch.setattr(engine, "HAS_CPP", False)
    return request.param


def _frame(close):
    dates = pd.date_range("2024-01-01", periods=len(close), freq="B")
    return pd.DataFrame({"Close": close}, index=dates)


class TestZeroOverZero:
    def test_every_trade_exactly_zero_is_nan(self, engine_path):
        frame = _frame([100.0] * 8)
        signal = pd.Series([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], index=frame.index)
        result = run_strategy(frame, signal, commission_pct=0.0, slippage_pct=0.0)
        assert result["num_trades"] == 2
        assert result["avg_trade_return_pct"] == 0.0
        assert math.isnan(result["profit_factor"])

    def test_null_case_a_winner_and_no_loser_is_still_inf(self, engine_path):
        frame = _frame([100.0, 110.0, 110.0, 110.0, 110.0])
        signal = pd.Series([1.0, 0.0, 1.0, 0.0, 0.0], index=frame.index)
        result = run_strategy(frame, signal, commission_pct=0.0, slippage_pct=0.0)
        assert result["num_trades"] == 2
        assert math.isinf(result["profit_factor"]) and result["profit_factor"] > 0

    def test_null_case_a_real_ratio_is_unchanged(self, engine_path):
        frame = _frame([100.0, 110.0, 110.0, 99.0, 99.0])
        signal = pd.Series([1.0, 0.0, 1.0, 0.0, 0.0], index=frame.index)
        result = run_strategy(frame, signal, commission_pct=0.0, slippage_pct=0.0)
        assert result["profit_factor"] == pytest.approx(10.0 / 10.0, abs=1e-4)


class TestTheBatchKernelAgrees:
    def test_batch_row_is_nan(self):
        cpp = pytest.importorskip("standard_quant_tools._sqt_core")
        flat = np.full(8, 100.0)
        rows = np.array([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])
        out = cpp.batch_run_strategy(flat, rows, 10_000.0, 0.0, 0.0)
        assert out[0, 9] == 2  # num_trades
        assert math.isnan(out[0, 8])  # profit_factor


class TestTheBoundaryRuleForAStaleBuild:
    """An extension compiled before this rule returns +inf for 0/0; the
    boundary helper maps it to NaN when no trade won, and to nothing else."""

    def test_inf_with_no_winner_is_nan(self):
        _, _, pf = _undefined_ratios_as_nan(
            np.nan, np.nan, np.inf, 0.0, 0.0, 3, 0.0, win_rate=0.0
        )
        assert math.isnan(float(pf))

    def test_inf_with_a_winner_stays_inf(self):
        _, _, pf = _undefined_ratios_as_nan(
            np.nan, np.nan, np.inf, 0.1, 0.1, 3, 0.0, win_rate=1 / 3
        )
        assert math.isinf(float(pf))

    def test_grid_columns(self):
        pf = pd.Series([np.inf, np.inf, 2.0, 0.0])
        wins = pd.Series([0.0, 0.5, 0.5, 0.0])
        trades = pd.Series([2, 2, 2, 2])
        zeros = pd.Series([0.0] * 4)
        _, _, out = _undefined_ratios_as_nan(
            zeros, zeros, pf, zeros, zeros, trades, 0.0, win_rate=wins
        )
        assert math.isnan(out[0])
        assert math.isinf(out[1])
        assert out[2] == 2.0 and out[3] == 0.0
