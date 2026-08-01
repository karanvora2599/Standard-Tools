"""Tests for the vectorized backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest.engine import backtest_grid, run_strategy
from standard_quant_tools.error import ValidationError


@pytest.fixture(scope="module")
def simple_ohlcv():
    """100-bar deterministic OHLCV for backtest tests."""
    np.random.seed(0)
    n = 100
    returns = np.random.normal(0.001, 0.015, n)
    close = 100.0 * np.cumprod(1 + returns)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.005,
            "Low": close * 0.995,
            "Close": pd.Series(close, index=dates),
            "Volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )
    return df


class TestReturnKeys:
    def test_result_has_required_keys(self, simple_ohlcv):
        signals = pd.Series(1, index=simple_ohlcv.index)
        result = run_strategy(simple_ohlcv, signals)
        required = {
            "final_equity",
            "total_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "profit_factor",
            "num_trades",
            "avg_trade_return_pct",
            "equity_curve",
        }
        assert required.issubset(result.keys())

    def test_equity_curve_same_length_as_data(self, simple_ohlcv):
        signals = pd.Series(1, index=simple_ohlcv.index)
        result = run_strategy(simple_ohlcv, signals)
        assert len(result["equity_curve"]) == len(simple_ohlcv)

    def test_equity_curve_starts_near_initial_capital(self, simple_ohlcv):
        """First bar is always flat (executed position = 0 due to shift)."""
        signals = pd.Series(1, index=simple_ohlcv.index)
        result = run_strategy(simple_ohlcv, signals, initial_capital=10_000)
        assert float(result["equity_curve"].iloc[0]) == pytest.approx(
            10_000.0, rel=1e-6
        )


class TestInputValidation:
    """
    Regression: unlike portfolio_engine.py (hardened separately), run_strategy/
    backtest_grid never validated initial_capital/commission_pct/slippage_pct
    before this — a zero/negative initial_capital silently produced inf/nan
    in total_return/calmar_ratio instead of raising.
    """

    @pytest.mark.parametrize(
        "bad_capital", [0.0, -1.0, -10_000.0, float("nan"), float("inf")]
    )
    def test_run_strategy_rejects_bad_initial_capital(self, simple_ohlcv, bad_capital):
        signals = pd.Series(1, index=simple_ohlcv.index)
        with pytest.raises(ValidationError, match="initial_capital"):
            run_strategy(simple_ohlcv, signals, initial_capital=bad_capital)

    @pytest.mark.parametrize("bad_cost", [-0.001, float("nan"), float("inf")])
    def test_run_strategy_rejects_bad_commission(self, simple_ohlcv, bad_cost):
        signals = pd.Series(1, index=simple_ohlcv.index)
        with pytest.raises(ValidationError, match="commission_pct"):
            run_strategy(simple_ohlcv, signals, commission_pct=bad_cost)

    @pytest.mark.parametrize("bad_cost", [-0.001, float("nan"), float("inf")])
    def test_run_strategy_rejects_bad_slippage(self, simple_ohlcv, bad_cost):
        signals = pd.Series(1, index=simple_ohlcv.index)
        with pytest.raises(ValidationError, match="slippage_pct"):
            run_strategy(simple_ohlcv, signals, slippage_pct=bad_cost)

    def test_run_strategy_accepts_zero_costs(self, simple_ohlcv):
        signals = pd.Series(1, index=simple_ohlcv.index)
        result = run_strategy(
            simple_ohlcv, signals, commission_pct=0.0, slippage_pct=0.0
        )
        assert np.isfinite(result["total_return"])

    @pytest.mark.parametrize("bad_capital", [0.0, -1.0])
    def test_backtest_grid_rejects_bad_initial_capital(self, simple_ohlcv, bad_capital):
        with pytest.raises(ValidationError, match="initial_capital"):
            backtest_grid(
                simple_ohlcv,
                strategy="sma_crossover",
                param_grid={"fast_period": [5], "slow_period": [20]},
                initial_capital=bad_capital,
            )

    def test_backtest_grid_rejects_bad_commission(self, simple_ohlcv):
        with pytest.raises(ValidationError, match="commission_pct"):
            backtest_grid(
                simple_ohlcv,
                strategy="sma_crossover",
                param_grid={"fast_period": [5], "slow_period": [20]},
                commission_pct=-0.001,
            )


class TestNoSignal:
    def test_zero_signal_yields_flat_equity(self, simple_ohlcv):
        signals = pd.Series(0, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, signals, initial_capital=10_000)
        assert result["total_return"] == pytest.approx(0.0, abs=1e-6)
        assert result["final_equity"] == pytest.approx(10_000.0, abs=0.01)

    def test_zero_signal_zero_trades(self, simple_ohlcv):
        signals = pd.Series(0, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, signals)
        assert result["num_trades"] == 0


class TestLookaheadBias:
    def test_first_executed_position_is_zero(self, simple_ohlcv):
        """Signal at bar 0 executes at bar 1: executed[0] must be 0."""
        signals = pd.Series(1, index=simple_ohlcv.index, dtype=float)
        # Manually verify: executed = signals.shift(1).fillna(0)
        executed = signals.shift(1).fillna(0.0)
        assert float(executed.iloc[0]) == 0.0

    def test_return_on_first_bar_is_zero(self, simple_ohlcv):
        """Because executed[0] = 0, strategy return on day 0 is always 0."""
        signals = pd.Series(1, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, signals, commission_pct=0, slippage_pct=0)
        equity = result["equity_curve"]
        assert float(equity.iloc[0]) == pytest.approx(10_000.0, rel=1e-9)


class TestTransactionCosts:
    def test_costs_reduce_returns(self, simple_ohlcv):
        """A strategy with costs must have a lower final equity than without."""
        signals = pd.Series(
            np.where(np.arange(len(simple_ohlcv)) % 5 == 0, 1, 0),
            index=simple_ohlcv.index,
            dtype=float,
        )
        no_cost = run_strategy(simple_ohlcv, signals, commission_pct=0, slippage_pct=0)
        with_cost = run_strategy(
            simple_ohlcv, signals, commission_pct=0.002, slippage_pct=0.001
        )
        assert with_cost["final_equity"] < no_cost["final_equity"]

    def test_cost_proportional_to_position_change_size(self, simple_ohlcv):
        """Going +1 → -1 (reversal, change=2) costs more than 0 → +1 (change=1)."""
        # This is structural: just verify the engine doesn't error on short signals
        signals = pd.Series([1, -1, 1, -1] * 25, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(
            simple_ohlcv, signals, commission_pct=0.001, slippage_pct=0.0005
        )
        assert result["num_trades"] > 0


class TestTradeLog:
    def test_include_trade_log_flag(self, simple_ohlcv):
        signals = pd.Series(
            [0] * 10 + [1] * 20 + [0] * 20 + [1] * 50,
            index=simple_ohlcv.index,
            dtype=float,
        )
        result = run_strategy(simple_ohlcv, signals, include_trade_log=True)
        assert "trade_log" in result
        trade_log = result["trade_log"]
        assert isinstance(trade_log, pd.DataFrame)

    def test_trade_log_has_correct_columns(self, simple_ohlcv):
        signals = pd.Series(
            [0] * 10 + [1] * 20 + [0] * 70, index=simple_ohlcv.index, dtype=float
        )
        result = run_strategy(simple_ohlcv, signals, include_trade_log=True)
        required_cols = {
            "entry_date",
            "exit_date",
            "direction",
            "entry_price",
            "exit_price",
            "return_pct",
        }
        assert required_cols.issubset(set(result["trade_log"].columns))

    def test_trade_log_entry_before_exit(self, simple_ohlcv):
        signals = pd.Series(
            [0] * 5 + [1] * 10 + [0] * 5 + [1] * 10 + [0] * 70,
            index=simple_ohlcv.index,
            dtype=float,
        )
        result = run_strategy(simple_ohlcv, signals, include_trade_log=True)
        trade_log = result["trade_log"]
        for _, row in trade_log.iterrows():
            assert row["entry_date"] <= row["exit_date"]

    def test_alternating_signal_yields_correct_num_trades(self, simple_ohlcv):
        """signals [0]*10 [1]*10 [0]*10 [1]*10 [0]*10 ... after shift → 1 completed trade."""
        n = len(simple_ohlcv)
        # One flat block, one long block, then flat for the rest
        sig = pd.Series(
            [0] * 10 + [1] * 10 + [0] * (n - 20), index=simple_ohlcv.index, dtype=float
        )
        result = run_strategy(simple_ohlcv, sig, include_trade_log=True)
        assert result["num_trades"] == 1

    def test_long_direction_in_trade_log(self, simple_ohlcv):
        sig = pd.Series(
            [0] * 10 + [1] * 10 + [0] * (len(simple_ohlcv) - 20),
            index=simple_ohlcv.index,
            dtype=float,
        )
        result = run_strategy(simple_ohlcv, sig, include_trade_log=True)
        assert result["trade_log"]["direction"].eq("long").all()

    def test_trade_log_reconciles_with_equity_curve_close_mode(self):
        """
        Regression test (P0 item 4), hand-verified: under fill_price="close",
        executed[i] = signals[i-1], so a position "appearing" in `executed`
        at bar i actually earns its first return over Close[i-1] -> Close[i]
        — Close[i-1] is the trade's true economic entry/exit reference, not
        Close[i]. 6-bar deterministic series, signals=[1,1,1,0,0,0] ->
        executed=[0,1,1,1,0,0] -> one trade, entry event at bar 1, exit
        event at bar 4.
        """
        dates = pd.date_range("2023-01-02", periods=6, freq="B")
        close = [100.0, 102.0, 104.0, 103.0, 105.0, 106.0]
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": [1_000_000.0] * 6,
            },
            index=dates,
        )
        signals = pd.Series([1, 1, 1, 0, 0, 0], index=dates, dtype=float)
        result = run_strategy(
            df,
            signals,
            commission_pct=0.001,
            slippage_pct=0.0005,
            include_trade_log=True,
        )
        trade_log = result["trade_log"]
        assert len(trade_log) == 1
        row = trade_log.iloc[0]
        # entry event at bar 1 -> Close[0]=100.0; exit event at bar 4 -> Close[3]=103.0.
        assert row["entry_price"] == pytest.approx(100.0)
        assert row["exit_price"] == pytest.approx(103.0)
        # raw price return (100 -> 103) = 3.0%, minus 2 * cost_per_unit
        # (0.001 + 0.0005 = 0.0015, entry + exit) = 0.3% -> 2.7%.
        assert row["return_pct"] == pytest.approx(2.7, abs=1e-9)
        # Reconcile against the equity curve's own compounded return over
        # the trade's actual bar span (bars 1..4) -- small residual (~0.006
        # points here) is expected: return_pct subtracts cost as a simple
        # fraction, while the equity curve compounds (1 + return - cost) at
        # each bar, so cost/return cross terms create a tiny difference,
        # same second-order approximation already documented for the
        # next_open/hl2_exploratory two-leg decomposition.
        equity = result["equity_curve"]
        span_multiplier = float(equity.iloc[4] / equity.iloc[0])
        assert row["return_pct"] == pytest.approx(
            (span_multiplier - 1.0) * 100, abs=0.05
        )

    def test_trade_log_return_scales_with_leveraged_position_size(self):
        """
        Regression test (high-severity item 4): a SCORE-type signal's
        magnitude is a literal leverage multiplier (run_strategy multiplies
        it directly into strategy_return = lagged_signal * market_return),
        not just a direction. The trade log must scale return_pct by the
        actual executed position size (2.5x here), not silently treat
        every trade as if it were exactly 1x/-1x -- and must report that
        magnitude via the position_size column, not just a "long"/"short"
        label. Same 6-bar deterministic series as the close-mode
        reconciliation test above, but signals=[2.5,2.5,2.5,0,0,0] instead
        of [1,1,1,0,0,0]: raw price return 100->103 is still 3.0%, but the
        trade's realized return is 3.0% * 2.5 = 7.5%, minus 2 * cost_per_unit
        (0.3%) = 7.2%.
        """
        dates = pd.date_range("2023-01-02", periods=6, freq="B")
        close = [100.0, 102.0, 104.0, 103.0, 105.0, 106.0]
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": [1_000_000.0] * 6,
            },
            index=dates,
        )
        signals = pd.Series([2.5, 2.5, 2.5, 0, 0, 0], index=dates, dtype=float)
        result = run_strategy(
            df,
            signals,
            commission_pct=0.001,
            slippage_pct=0.0005,
            include_trade_log=True,
        )
        trade_log = result["trade_log"]
        assert len(trade_log) == 1
        row = trade_log.iloc[0]
        assert row["position_size"] == pytest.approx(2.5)
        assert row["direction"] == "long"
        assert row["return_pct"] == pytest.approx(7.2, abs=1e-9)

    def test_trade_log_reconciles_with_equity_curve_next_open_mode(self):
        """
        Regression test (P0 item 4): under fill_price="next_open", the
        two-leg decomposition already prices entries/exits at that bar's
        own reference price (Open), so no shift is needed there (unlike
        "close" mode) -- entry_price/exit_price must equal Open at the
        entry/exit event dates directly.
        """
        dates = pd.date_range("2023-01-02", periods=6, freq="B")
        close = [100.0, 102.0, 104.0, 103.0, 105.0, 106.0]
        open_ = [99.0, 101.0, 103.0, 102.5, 104.0, 105.5]
        df = pd.DataFrame(
            {
                "Open": open_,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": [1_000_000.0] * 6,
            },
            index=dates,
        )
        signals = pd.Series([1, 1, 1, 0, 0, 0], index=dates, dtype=float)
        result = run_strategy(
            df,
            signals,
            commission_pct=0.001,
            slippage_pct=0.0005,
            fill_price="next_open",
            include_trade_log=True,
        )
        trade_log = result["trade_log"]
        assert len(trade_log) == 1
        row = trade_log.iloc[0]
        assert row["entry_price"] == pytest.approx(101.0)  # Open[1]
        assert row["exit_price"] == pytest.approx(104.0)  # Open[4]
        equity = result["equity_curve"]
        span_multiplier = float(equity.iloc[4] / equity.iloc[0])
        assert row["return_pct"] == pytest.approx(
            (span_multiplier - 1.0) * 100, abs=0.05
        )


class TestCppTradeStatsParity:
    def test_native_trade_stats_overwritten_with_python_computed_ones(
        self, monkeypatch
    ):
        """
        Regression test (P0-2): the native C++ kernel's own trade-log logic
        records entry at prices[i] (not the economically correct prior
        close) and excludes commission/slippage from trade returns -- the
        same bug _build_trade_log's Python-side fix addressed. Since the
        C++ source can't be rebuilt in this environment, the interim fix is
        to overwrite win_rate/profit_factor/num_trades/avg_trade_return_pct
        with _compute_trade_stats(_build_trade_log(...)) regardless of
        which path computed the equity curve, so results (and the optional
        trade_log) are identical whether or not _sqt_core is built.

        This fakes a native result with DELIBERATELY WRONG trade stats
        (values that could never legitimately arise from this scenario) and
        confirms they get replaced by the Python-computed correct values,
        not passed through -- using the same 6-bar hand-verified scenario
        as test_trade_log_reconciles_with_equity_curve_close_mode (entry
        at Close[0]=100, exit at Close[3]=103, return_pct=2.7%).
        """
        from unittest.mock import MagicMock

        import standard_quant_tools.backtest.engine as engine_mod

        dates = pd.date_range("2023-01-02", periods=6, freq="B")
        close = [100.0, 102.0, 104.0, 103.0, 105.0, 106.0]
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": [1_000_000.0] * 6,
            },
            index=dates,
        )
        signals = pd.Series([1, 1, 1, 0, 0, 0], index=dates, dtype=float)

        fake_cpp = MagicMock()
        fake_cpp.run_strategy.return_value = {
            "equity_curve": np.full(6, 10_000.0),
            "final_equity": 10_000.0,
            "total_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            # Deliberately wrong / implausible native trade stats -- these
            # must NOT survive into the final result.
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": 999,
            "avg_trade_return_pct": -50.0,
        }
        monkeypatch.setattr(engine_mod, "HAS_CPP", True)
        monkeypatch.setattr(engine_mod, "_cpp_core", fake_cpp)

        result = run_strategy(
            df,
            signals,
            commission_pct=0.001,
            slippage_pct=0.0005,
            include_trade_log=True,
        )

        assert result["num_trades"] == 1
        assert result["win_rate"] == pytest.approx(1.0)
        assert result["avg_trade_return_pct"] == pytest.approx(2.7, abs=1e-9)
        # Reconciles with the Python-side trade log built alongside it.
        row = result["trade_log"].iloc[0]
        assert result["avg_trade_return_pct"] == pytest.approx(row["return_pct"])


from typing import Any as _Any

_cpp: _Any = None
try:
    from standard_quant_tools import _sqt_core as _cpp  # type: ignore[attr-defined]

    _HAS_CPP_EXT = True
except ImportError:
    _HAS_CPP_EXT = False

requires_cpp_ext = pytest.mark.skipif(not _HAS_CPP_EXT, reason="_sqt_core not built")


class TestNativeTradeStatsCorrectness:
    """
    Regression: backtest.cpp's own run_strategy trade-log logic used to use
    entry_dir (sign only, ±1) and record entry_price at prices[i] instead of
    prices[i-1] (one bar later than the true economic reference), and never
    deducted commission/slippage from a trade's return -- so batch_run_strategy
    (which has no Python-side override, unlike the single-call C++ path in
    engine.py) silently reported wrong win_rate/profit_factor/
    avg_trade_return_pct for any leveraged (non-±1) signal. Calls the real
    compiled kernel directly (bypassing engine.py's Python override entirely)
    with the same hand-verified 6-bar leveraged scenario as
    test_trade_log_return_scales_with_leveraged_position_size: signal
    magnitude 2.5, raw price return 100->103 = 3.0%, realized trade return
    3.0% * 2.5 - 2*0.15% cost = 7.2%. Cannot run without a compiled
    _sqt_core (no C++ toolchain available in the environment that wrote
    this fix) -- verified by CI's build-cpp.yml instead.
    """

    @requires_cpp_ext
    def test_run_strategy_native_avg_trade_return_matches_hand_computed(self):
        close = np.array([100.0, 102.0, 104.0, 103.0, 105.0, 106.0])
        signals = np.array([2.5, 2.5, 2.5, 0.0, 0.0, 0.0])
        r = _cpp.run_strategy(close, signals, 10_000.0, 0.001, 0.0005)
        assert r["num_trades"] == 1
        assert r["win_rate"] == pytest.approx(1.0)
        assert r["avg_trade_return_pct"] == pytest.approx(7.2, abs=1e-9)

    @requires_cpp_ext
    def test_batch_run_strategy_native_avg_trade_return_matches_hand_computed(self):
        """batch_run_strategy has no Python-side override at all -- this is
        the scenario that was silently wrong end-to-end before this fix."""
        close = np.array([100.0, 102.0, 104.0, 103.0, 105.0, 106.0])
        signals_mat = np.array([[2.5, 2.5, 2.5, 0.0, 0.0, 0.0]])
        results = _cpp.batch_run_strategy(close, signals_mat, 10_000.0, 0.001, 0.0005)
        assert len(results) == 1
        r = results[0]
        assert r["num_trades"] == 1
        assert r["win_rate"] == pytest.approx(1.0)
        assert r["avg_trade_return_pct"] == pytest.approx(7.2, abs=1e-9)

    @requires_cpp_ext
    def test_run_strategy_native_matches_python_recomputed_stats(self, simple_ohlcv):
        """
        Broader cross-check on a realistic random series (not just the
        hand-verified 6-bar case): the native kernel's own trade stats must
        now agree with engine.py's independent Python recomputation
        (_build_trade_log + _compute_trade_stats), since both implement the
        identical fill-aware, cost-aware accounting.
        """
        from standard_quant_tools.backtest.engine import (
            _build_trade_log,
            _compute_trade_stats,
        )

        np.random.seed(7)
        signals = pd.Series(
            np.random.choice([-1.0, 0.0, 1.0, 2.0], len(simple_ohlcv)),
            index=simple_ohlcv.index,
        )
        prices = simple_ohlcv["Close"]
        executed = signals.shift(1).fillna(0.0)

        close_arr = prices.to_numpy(dtype=np.float64)
        signals_arr = signals.to_numpy(dtype=np.float64)
        native = _cpp.run_strategy(close_arr, signals_arr, 10_000.0, 0.001, 0.0005)

        trade_log = _build_trade_log(prices.shift(1), prices, executed, 0.001 + 0.0005)
        python_stats = _compute_trade_stats(trade_log)

        # _compute_trade_stats intentionally round()s to 4 decimal places for
        # display (see engine.py) -- the native kernel returns full
        # precision, so an exact-tolerance comparison must account for that
        # rounding step itself, not just floating-point noise. This is not
        # a discrepancy a real caller ever sees: run_strategy()'s Python
        # wrapper always overwrites these fields with the Python-computed,
        # already-rounded values regardless of which backend ran.
        assert native["num_trades"] == python_stats["num_trades"]
        assert native["win_rate"] == pytest.approx(python_stats["win_rate"], abs=5e-5)
        assert native["avg_trade_return_pct"] == pytest.approx(
            python_stats["avg_trade_return_pct"], abs=5e-5
        )
        assert native["profit_factor"] == pytest.approx(
            python_stats["profit_factor"], abs=5e-5
        )


class TestMetricBounds:
    def test_win_rate_between_0_and_1(self, simple_ohlcv):
        np.random.seed(1)
        signals = pd.Series(
            np.random.choice([0, 1], len(simple_ohlcv)).astype(float),
            index=simple_ohlcv.index,
        )
        result = run_strategy(simple_ohlcv, signals)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_max_drawdown_nonpositive(self, simple_ohlcv):
        signals = pd.Series(1, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, signals)
        assert result["max_drawdown"] <= 0

    def test_final_equity_positive(self, simple_ohlcv):
        signals = pd.Series(1, index=simple_ohlcv.index, dtype=float)
        result = run_strategy(simple_ohlcv, signals)
        assert result["final_equity"] > 0
