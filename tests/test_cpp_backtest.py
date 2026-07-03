"""
Python integration tests for the backtest C++ extension and Python wrapper.

Two execution modes:
  1. _sqt_core NOT built → cpp_* tests are skipped; wrapper tests verify the
     pure-Python fallback produces correct results.
  2. _sqt_core IS built  → all tests run; cross-validates C++ vs pure Python.

Run:
    pytest tests/test_cpp_backtest.py -v
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

# ── Extension availability ────────────────────────────────────────────────────

_cpp: Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

requires_cpp = pytest.mark.skipif(not HAS_CPP, reason="_sqt_core not built")

from standard_quant_tools.backtest.engine import (
    HAS_CPP as ENGINE_HAS_CPP,
    run_strategy,
)

# ── Fixtures and helpers ──────────────────────────────────────────────────────

DATE_IDX = pd.date_range("2020-01-01", periods=500, freq="B")


def _make_df(prices: np.ndarray) -> pd.DataFrame:
    idx = DATE_IDX[: len(prices)]
    return pd.DataFrame({"Open": prices, "High": prices, "Low": prices,
                         "Close": prices, "Volume": 1_000_000},
                        index=idx)


def _make_signals(values: np.ndarray, n: int) -> pd.Series:
    idx = DATE_IDX[:n]
    return pd.Series(values, index=idx)


def _py_run_strategy(prices_arr, signals_arr, initial_capital=10_000.0,
                     commission_pct=0.001, slippage_pct=0.0005):
    """Pure-Python reference implementation of the vectorized engine."""
    import pandas as pd
    n = len(prices_arr)
    idx = pd.RangeIndex(n)
    prices   = pd.Series(prices_arr, index=idx)
    executed = pd.Series(np.concatenate([[0.0], signals_arr[:-1]]), index=idx)
    returns  = prices.pct_change().fillna(0.0)
    pos_diff = executed.diff().fillna(executed.iloc[0])
    tcosts   = pos_diff.abs() * (commission_pct + slippage_pct)
    strat    = executed * returns - tcosts
    equity   = initial_capital * (1 + strat).cumprod()
    total_ret = (equity.iloc[-1] - initial_capital) / initial_capital
    return {"total_return": total_ret, "equity_curve": equity.to_numpy()}


# ── Wrapper tests (always run) ────────────────────────────────────────────────

class TestRunStrategyReturnSchema:
    REQUIRED_KEYS = {
        "final_equity", "total_return", "annualized_volatility",
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
        "win_rate", "profit_factor", "num_trades", "avg_trade_return_pct",
        "equity_curve",
    }

    def _make_inputs(self, n: int = 100):
        rng = np.random.default_rng(1)
        prices = 100.0 * np.exp(rng.standard_normal(n).cumsum() * 0.01)
        df = _make_df(prices)
        sig = _make_signals(np.ones(n), n)
        return df, sig

    def test_keys_present(self):
        df, sig = self._make_inputs()
        r = run_strategy(df, sig)
        assert set(r.keys()) >= self.REQUIRED_KEYS

    def test_types(self):
        df, sig = self._make_inputs()
        r = run_strategy(df, sig)
        assert isinstance(r["final_equity"], float)
        assert isinstance(r["total_return"], float)
        assert isinstance(r["num_trades"], int)
        assert isinstance(r["equity_curve"], pd.Series)

    def test_equity_curve_length_matches_input(self):
        n = 120
        rng = np.random.default_rng(3)
        prices = 100.0 + np.cumsum(rng.standard_normal(n))
        df = _make_df(prices)
        sig = _make_signals(np.ones(n), n)
        r = run_strategy(df, sig)
        assert len(r["equity_curve"]) == n

    def test_flat_signal_zero_return(self):
        prices = np.linspace(100.0, 120.0, 50)
        df = _make_df(prices)
        sig = _make_signals(np.zeros(50), 50)
        r = run_strategy(df, sig, commission_pct=0.0, slippage_pct=0.0)
        assert abs(r["total_return"]) < 1e-10
        assert abs(r["final_equity"] - 10_000.0) < 1e-6

    def test_positive_returns_for_long_rising_prices(self):
        prices = np.linspace(100.0, 130.0, 60)
        df = _make_df(prices)
        sig = _make_signals(np.ones(60), 60)
        r = run_strategy(df, sig, commission_pct=0.0, slippage_pct=0.0)
        assert r["total_return"] > 0.0
        assert r["final_equity"] > 10_000.0

    def test_max_drawdown_nonpositive(self):
        rng = np.random.default_rng(7)
        prices = 100.0 + np.cumsum(rng.standard_normal(200))
        df = _make_df(prices)
        sig = _make_signals(np.ones(200), 200)
        r = run_strategy(df, sig)
        assert r["max_drawdown"] <= 0.0

    def test_win_rate_in_unit_interval(self):
        rng = np.random.default_rng(11)
        prices = 100.0 + np.cumsum(rng.standard_normal(100))
        signals = np.sign(rng.standard_normal(100))
        df = _make_df(prices)
        sig = _make_signals(signals, 100)
        r = run_strategy(df, sig)
        assert 0.0 <= r["win_rate"] <= 1.0

    def test_include_trade_log(self):
        prices = np.linspace(100.0, 110.0, 20)
        df = _make_df(prices)
        sig_vals = np.zeros(20)
        sig_vals[5:12] = 1.0
        sig = _make_signals(sig_vals, 20)
        r = run_strategy(df, sig, include_trade_log=True)
        assert "trade_log" in r
        assert isinstance(r["trade_log"], pd.DataFrame)

    def test_no_trade_log_by_default(self):
        prices = np.linspace(100.0, 110.0, 20)
        df = _make_df(prices)
        sig = _make_signals(np.ones(20), 20)
        r = run_strategy(df, sig)
        assert "trade_log" not in r


# ── C++ binding tests (skipped unless _sqt_core is built) ────────────────────

class TestCppRunStrategyDirect:
    """Direct calls to _sqt_core.run_strategy — bypasses the Python wrapper."""

    @requires_cpp
    def test_returns_dict_with_required_keys(self):
        prices  = np.array([100.0, 105.0, 102.0, 108.0])
        signals = np.array([1.0,   1.0,   0.0,   1.0])
        r = _cpp.run_strategy(prices, signals)
        expected = {
            "final_equity", "total_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "num_trades", "win_rate", "profit_factor",
            "avg_trade_return_pct", "equity_curve",
        }
        assert set(r.keys()) == expected

    @requires_cpp
    def test_flat_signal_zero_return(self):
        prices  = np.array([100.0, 105.0, 110.0, 95.0, 100.0])
        signals = np.zeros(5)
        r = _cpp.run_strategy(prices, signals, 10_000.0, 0.0, 0.0)
        assert abs(r["total_return"]) < 1e-10
        assert abs(r["final_equity"] - 10_000.0) < 1e-8

    @requires_cpp
    def test_long_monotone_up_no_costs_known_return(self):
        # prices=[100,110,121], signals=[1,1,1], no costs
        # executed=[0,1,1]; strat_ret=[0, 0.10, 0.10]
        # equity=[10000, 11000, 12100]; total_return=0.21
        prices  = np.array([100.0, 110.0, 121.0])
        signals = np.ones(3)
        r = _cpp.run_strategy(prices, signals, 10_000.0, 0.0, 0.0)
        assert abs(r["total_return"] - 0.21) < 1e-8
        assert abs(r["final_equity"] - 12_100.0) < 1e-3

    @requires_cpp
    def test_equity_curve_length(self):
        n = 50
        rng = np.random.default_rng(5)
        prices  = 100.0 + np.cumsum(rng.standard_normal(n))
        signals = np.sign(rng.standard_normal(n))
        r = _cpp.run_strategy(prices, signals)
        eq = np.asarray(r["equity_curve"])
        assert len(eq) == n
        assert abs(eq[0] - 10_000.0) < 1e-8

    @requires_cpp
    def test_costs_reduce_return(self):
        prices  = np.linspace(100.0, 130.0, 100)
        signals = np.ones(100)
        r_free = _cpp.run_strategy(prices, signals, 10_000.0, 0.0,   0.0)
        r_cost = _cpp.run_strategy(prices, signals, 10_000.0, 0.001, 0.0005)
        assert r_cost["total_return"] < r_free["total_return"]

    @requires_cpp
    def test_max_drawdown_negative_for_volatile_series(self):
        prices  = np.array([100.0, 120.0, 80.0, 110.0, 90.0])
        signals = np.ones(5)
        r = _cpp.run_strategy(prices, signals, 10_000.0, 0.0, 0.0)
        assert r["max_drawdown"] < 0.0

    @requires_cpp
    def test_max_drawdown_zero_for_monotone_up(self):
        prices  = np.linspace(100.0, 130.0, 20)
        signals = np.ones(20)
        r = _cpp.run_strategy(prices, signals, 10_000.0, 0.0, 0.0)
        assert abs(r["max_drawdown"]) < 1e-10

    @requires_cpp
    def test_short_profits_when_prices_fall(self):
        prices  = np.array([100.0, 90.0, 80.0, 70.0])
        signals = np.full(4, -1.0)
        r = _cpp.run_strategy(prices, signals, 10_000.0, 0.0, 0.0)
        assert r["total_return"] > 0.0

    @requires_cpp
    def test_mismatched_length_raises(self):
        with pytest.raises(Exception):
            _cpp.run_strategy(np.ones(10), np.ones(9))

    @requires_cpp
    def test_matches_python_fallback_total_return(self):
        rng     = np.random.default_rng(42)
        prices  = 100.0 + np.cumsum(rng.standard_normal(300))
        signals = np.sign(rng.standard_normal(300))
        r_cpp = _cpp.run_strategy(prices, signals, 10_000.0, 0.001, 0.0005)
        r_py  = _py_run_strategy(prices, signals, 10_000.0, 0.001, 0.0005)
        assert abs(r_cpp["total_return"] - r_py["total_return"]) < 1e-8

    @requires_cpp
    def test_matches_python_fallback_equity_curve(self):
        rng     = np.random.default_rng(77)
        prices  = 100.0 + np.cumsum(rng.standard_normal(100))
        signals = np.sign(rng.standard_normal(100))
        r_cpp = _cpp.run_strategy(prices, signals, 10_000.0, 0.001, 0.0005)
        r_py  = _py_run_strategy(prices, signals, 10_000.0, 0.001, 0.0005)
        eq_cpp = np.asarray(r_cpp["equity_curve"])
        eq_py  = r_py["equity_curve"]
        assert np.allclose(eq_cpp, eq_py, atol=1e-8)

    @requires_cpp
    def test_one_winning_trade_stats(self):
        # Enter long at bar 1, exit at bar 2: gain
        prices  = np.array([100.0, 110.0, 120.0])
        signals = np.array([1.0,   0.0,   0.0])
        r = _cpp.run_strategy(prices, signals, 10_000.0, 0.0, 0.0)
        assert r["num_trades"] == 1
        assert abs(r["win_rate"] - 1.0) < 1e-10
        assert r["avg_trade_return_pct"] > 0.0

    @requires_cpp
    def test_sortino_inf_no_negative_returns(self):
        prices  = np.linspace(100.0, 110.0, 20)
        signals = np.ones(20)
        r = _cpp.run_strategy(prices, signals, 10_000.0, 0.0, 0.0)
        import math
        assert math.isinf(r["sortino_ratio"])

    @requires_cpp
    def test_wrapper_routes_to_cpp(self):
        assert ENGINE_HAS_CPP is True


# ══════════════════════════════════════════════════════════════════════════════
# batch_run_strategy — C++ extension tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCppBatchRunStrategy:
    """Direct calls to _sqt_core.batch_run_strategy."""

    def _signals_matrix(self, n: int, num_tests: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        rows = [np.sign(rng.standard_normal(n)).astype(np.float64)
                for _ in range(num_tests)]
        return np.vstack(rows)

    @requires_cpp
    def test_returns_list_of_dicts(self):
        prices  = np.linspace(100.0, 110.0, 50)
        signals = np.vstack([np.ones(50), np.zeros(50)])
        results = _cpp.batch_run_strategy(prices, signals)
        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, dict)

    @requires_cpp
    def test_result_count_equals_num_tests(self):
        prices = np.linspace(100.0, 120.0, 100)
        for num_tests in (1, 3, 10):
            sig_mat = np.tile(np.ones(100), (num_tests, 1))
            results = _cpp.batch_run_strategy(prices, sig_mat)
            assert len(results) == num_tests

    @requires_cpp
    def test_each_result_has_required_keys(self):
        prices   = np.linspace(100.0, 110.0, 60)
        sig_mat  = np.vstack([np.ones(60), np.zeros(60)])
        required = {
            "final_equity", "total_return", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
            "num_trades", "win_rate", "profit_factor", "avg_trade_return_pct",
        }
        for r in _cpp.batch_run_strategy(prices, sig_mat):
            assert required <= set(r.keys())

    @requires_cpp
    def test_equity_curve_stripped(self):
        """batch_run_strategy strips the equity_curve to save memory."""
        prices  = np.linspace(100.0, 110.0, 50)
        sig_mat = np.vstack([np.ones(50)])
        r = _cpp.batch_run_strategy(prices, sig_mat)[0]
        assert "equity_curve" not in r

    @requires_cpp
    def test_flat_signal_zero_return(self):
        prices  = np.linspace(100.0, 130.0, 80)
        sig_mat = np.zeros((1, 80))
        r = _cpp.batch_run_strategy(prices, sig_mat, 10_000.0, 0.0, 0.0)[0]
        assert abs(r["total_return"]) < 1e-10

    @requires_cpp
    def test_all_long_rising_prices_positive_return(self):
        prices  = np.linspace(100.0, 130.0, 100)
        sig_mat = np.ones((1, 100))
        r = _cpp.batch_run_strategy(prices, sig_mat, 10_000.0, 0.0, 0.0)[0]
        assert r["total_return"] > 0.0

    @requires_cpp
    def test_matches_single_run_strategy(self):
        """Each row must match an individual run_strategy call exactly."""
        rng     = np.random.default_rng(99)
        n       = 150
        prices  = 100.0 + np.cumsum(rng.standard_normal(n))
        sig_mat = self._signals_matrix(n, 4, seed=99)

        batch = _cpp.batch_run_strategy(prices, sig_mat, 10_000.0, 0.001, 0.0005)
        for i, br in enumerate(batch):
            sr = _cpp.run_strategy(prices, sig_mat[i], 10_000.0, 0.001, 0.0005)
            assert abs(br["total_return"]  - sr["total_return"])  < 1e-10, f"row {i}"
            assert abs(br["sharpe_ratio"]  - sr["sharpe_ratio"])  < 1e-10, f"row {i}"
            assert abs(br["max_drawdown"]  - sr["max_drawdown"])  < 1e-10, f"row {i}"

    @requires_cpp
    def test_different_signals_produce_different_results(self):
        n       = 120
        prices  = np.linspace(100.0, 115.0, n)
        sig_mat = np.vstack([np.ones(n), -np.ones(n)])
        results = _cpp.batch_run_strategy(prices, sig_mat, 10_000.0, 0.0, 0.0)
        assert results[0]["total_return"] != results[1]["total_return"]

    @requires_cpp
    def test_shape_mismatch_raises(self):
        prices  = np.linspace(100.0, 110.0, 50)
        sig_mat = np.ones((2, 60))  # wrong n_cols
        with pytest.raises(Exception):
            _cpp.batch_run_strategy(prices, sig_mat)

    @requires_cpp
    def test_non_2d_signals_raises(self):
        prices = np.linspace(100.0, 110.0, 50)
        with pytest.raises(Exception):
            _cpp.batch_run_strategy(prices, np.ones(50))  # 1D, not 2D


# ══════════════════════════════════════════════════════════════════════════════
# backtest_grid C++ batch path (wrapper-level)
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktestGridCppPath:
    """When _sqt_core is built, backtest_grid should use the C++ batch kernel."""

    @requires_cpp
    def test_cpp_grid_matches_python_fallback(self):
        """Force both paths and compare sharpe_ratios (allow small float tolerance)."""
        from unittest.mock import patch
        from standard_quant_tools.backtest import backtest_grid
        from standard_quant_tools.backtest import engine as eng

        rng    = np.random.default_rng(1)
        n      = 300
        dates  = pd.date_range("2020-01-01", periods=n, freq="B")
        prices = pd.DataFrame({
            "Open":   100.0 + np.cumsum(rng.standard_normal(n)),
            "High":   100.0 + np.cumsum(rng.standard_normal(n)) + 0.5,
            "Low":    100.0 + np.cumsum(rng.standard_normal(n)) - 0.5,
            "Close":  100.0 + np.cumsum(rng.standard_normal(n)),
            "Volume": 1_000_000.0,
        }, index=dates)
        prices["High"]  = prices[["Open", "High", "Close"]].max(axis=1)
        prices["Low"]   = prices[["Open", "Low",  "Close"]].min(axis=1)

        grid = {"fast_period": [5, 10], "slow_period": [20, 30]}

        # C++ path (default when built)
        cpp_result = backtest_grid(prices, strategy="sma_crossover",
                                   param_grid=grid, n_workers=1)

        # Python fallback: patch HAS_CPP to False inside engine module
        with patch.object(eng, "HAS_CPP", False), \
             patch.object(eng, "_cpp_core", None):
            py_result = backtest_grid(prices, strategy="sma_crossover",
                                      param_grid=grid, n_workers=1)

        assert len(cpp_result) == len(py_result)
        cpp_sorted = cpp_result.sort_values(
            ["fast_period", "slow_period"]).reset_index(drop=True)
        py_sorted  = py_result.sort_values(
            ["fast_period", "slow_period"]).reset_index(drop=True)
        for col in ["total_return", "sharpe_ratio", "max_drawdown"]:
            np.testing.assert_allclose(
                cpp_sorted[col].to_numpy(),
                py_sorted[col].to_numpy(),
                atol=1e-6,
                err_msg=f"C++ vs Python mismatch in column '{col}'",
            )
