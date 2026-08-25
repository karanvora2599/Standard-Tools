"""Tests for the advanced agentic tools (mocked data provider)."""

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.models import (
    BacktestDiagnosticsInput,
    BuyAndHoldInput,
    CapacityReportInput,
    CompareStrategiesInput,
    DataQualityReportInput,
    ImpliedVolatilityInput,
    OptionPricingInput,
    PairScannerInput,
    PairTradeBacktestInput,
    PCAInput,
    PortfolioInput,
    PortfolioOptimizationInput,
    PortfolioSimulationInput,
    PositionSizerInput,
    RegimeAdaptiveInput,
    RegimeAdaptiveWalkForwardInput,
    RiskAttributionInput,
    RobustnessDiagnosticsInput,
    WalkForwardInput,
)
from standard_quant_tools.agent.runtimes.backtest import tools as backtest_tools
from standard_quant_tools.agent.runtimes.research import tools as research_tools
from standard_quant_tools.agent.tools import (
    _TOOL_DISPATCH,
    compare_strategies,
    dispatch,
    get_agent_tools,
    get_backtest_diagnostics,
    get_capacity_report,
    get_data_quality_report,
    get_implied_volatility,
    get_option_pricing,
    get_portfolio_risk_attribution,
    get_position_size,
    get_robustness_diagnostics,
    run_buy_and_hold,
    run_pair_trade_backtest,
    run_portfolio_optimization,
    run_portfolio_simulation,
    run_regime_adaptive_backtest,
    run_regime_adaptive_walkforward_backtest,
    run_walk_forward_backtest,
    scan_pairs,
)

START, END = "2022-01-01", "2024-01-01"


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def long_ohlcv() -> pd.DataFrame:
    """700-bar OHLCV — enough for walk-forward (252 train + 63 test × 2 windows)."""
    np.random.seed(99)
    n = 700
    returns = np.random.normal(0.0003, 0.013, n)
    close = 100.0 * np.cumprod(1 + returns)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    spread = np.random.uniform(0.2, 1.2, n)
    return pd.DataFrame(
        {
            "Open": pd.Series(close * 0.999, index=dates),
            "High": pd.Series(close + spread, index=dates),
            "Low": pd.Series(close - spread, index=dates),
            "Close": pd.Series(close, index=dates),
            "Volume": pd.Series(
                np.random.randint(1_000_000, 5_000_000, n).astype(float), index=dates
            ),
        }
    )


@pytest.fixture
def patched_long(long_ohlcv, monkeypatch):
    """Patch DataFactory to return the long OHLCV for every symbol."""
    from unittest.mock import AsyncMock, MagicMock

    from standard_quant_tools.data.factory import DataFactory

    provider = MagicMock()
    provider.get_ohlcv.return_value = long_ohlcv
    provider.get_ohlcv_async = AsyncMock(return_value=long_ohlcv)
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return provider


# ── Tool registry ──────────────────────────────────────────────────────────────


class TestToolRegistry:
    def test_tool_defs_and_dispatch_stay_in_sync(self):
        assert len(get_agent_tools()) == len(_TOOL_DISPATCH)

    def test_new_tool_names_present(self):
        names = {t["function"]["name"] for t in get_agent_tools()}
        assert "run_regime_adaptive_backtest" in names
        assert "scan_pairs" in names
        assert "run_walk_forward_backtest" in names
        assert "get_portfolio_risk_attribution" in names
        assert "get_position_size" in names
        assert "run_buy_and_hold" in names
        assert "compare_strategies" in names
        assert "get_backtest_diagnostics" in names

    def test_all_new_tools_have_valid_schema(self):
        new_tools = [
            t
            for t in get_agent_tools()
            if t["function"]["name"]
            in {
                "run_regime_adaptive_backtest",
                "scan_pairs",
                "run_walk_forward_backtest",
                "get_portfolio_risk_attribution",
                "get_position_size",
            }
        ]
        for tool in new_tools:
            schema = tool["function"]["parameters"]
            assert schema.get("type") == "object"
            assert "properties" in schema


# ── Feature 1: Regime-Adaptive Strategy Selector ──────────────────────────────


class TestRegimeAdaptiveBacktest:
    def test_returns_result(self, patched_long):
        inp = RegimeAdaptiveInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            n_workers=1,
        )
        result = run_regime_adaptive_backtest(inp)
        assert result.symbol == "AAPL"

    def test_regime_is_valid(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert result.regime in ("trending", "mean_reverting", "random_walk", "unknown")

    def test_selected_strategy_is_valid(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert result.selected_strategy in (
            "sma_crossover",
            "rsi_mean_reversion",
            "macd_crossover",
            "bollinger_reversion",
        )

    def test_hurst_bounded(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert 0.0 <= result.hurst <= 1.0 or result.hurst == 0.0  # 0.0 on NaN fallback

    def test_backtest_fields_populated(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert isinstance(result.backtest.sharpe_ratio, float)
        assert result.backtest.final_equity > 0
        assert 0.0 <= result.backtest.win_rate <= 1.0

    def test_grid_combinations_positive(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert result.grid_combinations > 0

    def test_best_parameters_nonempty(self, patched_long):
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert len(result.best_parameters) > 0

    def test_custom_param_grid_respected(self, patched_long):
        inp = RegimeAdaptiveInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            sma_param_grid={"fast_period": [5], "slow_period": [20]},
        )
        result = run_regime_adaptive_backtest(inp)
        if result.selected_strategy == "sma_crossover":
            assert result.grid_combinations == 1


# ── Feature 1b: Regime-Adaptive Walk-Forward Backtest (leakage-free) ──────────

_RAWF_GRID = {
    "sma_crossover": {"fast_period": [5, 10], "slow_period": [30, 50]},
    "rsi_mean_reversion": {
        "period": [7, 14],
        "oversold": [25, 30],
        "overbought": [65, 70],
    },
    "macd_crossover": {"fast": [8, 12], "slow": [21, 26], "signal": [7, 9]},
    "bollinger_reversion": {"period": [15, 20], "num_std": [1.5, 2.0]},
}


class TestRegimeAdaptiveWalkForwardBacktest:
    def test_returns_result(self, patched_long):
        inp = RegimeAdaptiveWalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            train_bars=252,
            test_bars=63,
            sma_param_grid=_RAWF_GRID["sma_crossover"],
            rsi_param_grid=_RAWF_GRID["rsi_mean_reversion"],
            macd_param_grid=_RAWF_GRID["macd_crossover"],
            bollinger_param_grid=_RAWF_GRID["bollinger_reversion"],
        )
        result = run_regime_adaptive_walkforward_backtest(inp)
        assert result.symbol == "AAPL"
        assert result.n_windows >= 1
        assert len(result.windows) == result.n_windows

    def test_window_dates_sequential(self, patched_long):
        inp = RegimeAdaptiveWalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            train_bars=252,
            test_bars=63,
            sma_param_grid=_RAWF_GRID["sma_crossover"],
            rsi_param_grid=_RAWF_GRID["rsi_mean_reversion"],
            macd_param_grid=_RAWF_GRID["macd_crossover"],
            bollinger_param_grid=_RAWF_GRID["bollinger_reversion"],
        )
        result = run_regime_adaptive_walkforward_backtest(inp)
        for i, win in enumerate(result.windows):
            assert win.window_index == i
            assert win.train_start <= win.train_end
            assert win.test_start <= win.test_end
            assert win.train_end < win.test_start

    def test_selected_strategy_is_valid_and_not_hardcoded_by_regime(self, patched_long):
        """
        The whole point of this tool vs. run_regime_adaptive_backtest: the
        regime is diagnostic context, not a hard selector. Every window's
        selected_strategy must be a real registry name, and — unlike the
        old tool's fixed regime->strategy map — a "trending" window is not
        required to pick sma_crossover specifically.
        """
        inp = RegimeAdaptiveWalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            train_bars=252,
            test_bars=63,
            sma_param_grid=_RAWF_GRID["sma_crossover"],
            rsi_param_grid=_RAWF_GRID["rsi_mean_reversion"],
            macd_param_grid=_RAWF_GRID["macd_crossover"],
            bollinger_param_grid=_RAWF_GRID["bollinger_reversion"],
        )
        result = run_regime_adaptive_walkforward_backtest(inp)
        from standard_quant_tools.backtest import strategies as strategies_mod

        # Every registered strategy is eligible here (the walk-forward
        # selector tries all of STRATEGY_REGISTRY per window, not just the
        # 3 the single-shot run_regime_adaptive_backtest maps by regime) --
        # derived dynamically so this assertion doesn't go stale as new
        # strategies are registered.
        valid_strategies = set(strategies_mod.STRATEGY_REGISTRY)
        for win in result.windows:
            assert win.selected_strategy in valid_strategies
            assert win.regime in (
                "trending",
                "mean_reverting",
                "random_walk",
                "unknown",
            )
            assert 0.0 <= win.hurst <= 1.0 or win.hurst == 0.0

    def test_stitched_and_stability_fields_present(self, patched_long):
        inp = RegimeAdaptiveWalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            train_bars=252,
            test_bars=63,
            sma_param_grid=_RAWF_GRID["sma_crossover"],
            rsi_param_grid=_RAWF_GRID["rsi_mean_reversion"],
            macd_param_grid=_RAWF_GRID["macd_crossover"],
            bollinger_param_grid=_RAWF_GRID["bollinger_reversion"],
        )
        result = run_regime_adaptive_walkforward_backtest(inp)
        for field in (
            "stitched_oos_return",
            "stitched_oos_sharpe",
            "stitched_oos_sortino",
            "stitched_oos_max_drawdown",
            "stitched_oos_calmar",
        ):
            assert isinstance(getattr(result, field), float)
        assert 0 <= result.worst_oos_window < result.n_windows
        assert 0 <= result.longest_losing_window_streak <= result.n_windows
        assert "most_common" in result.strategy_stability
        assert "frequency" in result.strategy_stability
        assert 0.0 <= result.strategy_stability["frequency"] <= 1.0

    def test_insufficient_data_raises(self, patched_long, long_ohlcv, monkeypatch):
        from unittest.mock import MagicMock

        from standard_quant_tools.data.factory import DataFactory

        tiny_df = long_ohlcv.iloc[:50]
        prov = MagicMock()
        prov.get_ohlcv.return_value = tiny_df
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: prov)

        inp = RegimeAdaptiveWalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            train_bars=252,
            test_bars=63,
        )
        with pytest.raises(ValueError, match="Not enough data"):
            run_regime_adaptive_walkforward_backtest(inp)

    def test_fill_price_threads_into_oos_leg(self, patched_long):
        """Same fix as WalkForwardInput: fill_price used to be hardcoded to
        "close" for the OOS leg (04_backtesting.md's own documented gap)."""
        base_kwargs = dict(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            train_bars=252,
            test_bars=63,
            sma_param_grid=_RAWF_GRID["sma_crossover"],
            rsi_param_grid=_RAWF_GRID["rsi_mean_reversion"],
            macd_param_grid=_RAWF_GRID["macd_crossover"],
            bollinger_param_grid=_RAWF_GRID["bollinger_reversion"],
        )
        default_result = run_regime_adaptive_walkforward_backtest(
            RegimeAdaptiveWalkForwardInput(**base_kwargs)
        )
        next_open_result = run_regime_adaptive_walkforward_backtest(
            RegimeAdaptiveWalkForwardInput(**base_kwargs, fill_price="next_open")
        )
        assert next_open_result.stitched_oos_return != pytest.approx(
            default_result.stitched_oos_return, abs=1e-9
        )

    def test_no_lookahead_window0_unaffected_by_future_mutation(
        self, long_ohlcv, monkeypatch
    ):
        """
        Same regression pattern as run_walk_forward_backtest's no-lookahead
        test: window 0's regime/hurst/selected_strategy/best_params are all
        derived from bars [0, train_bars) only. Replacing every bar from
        train_bars onward with a different synthetic path must leave
        window 0's in-sample selection untouched, while window 0's own
        out-of-sample result (which lives inside the mutated region) does
        change — proving the mutation is actually visible to the tool.
        """
        from unittest.mock import MagicMock

        from standard_quant_tools.data.factory import DataFactory

        train_bars, test_bars = 252, 63

        mutated = long_ohlcv.copy()
        rng = np.random.default_rng(4242)
        n_mutate = len(mutated) - train_bars
        mutated_returns = rng.normal(-0.001, 0.03, n_mutate)
        last_train_close = float(mutated["Close"].iloc[train_bars - 1])
        mutated_close = last_train_close * np.cumprod(1 + mutated_returns)
        spread = rng.uniform(0.2, 1.2, n_mutate)
        close_col, open_col = mutated.columns.get_loc("Close"), mutated.columns.get_loc(
            "Open"
        )
        high_col, low_col = mutated.columns.get_loc("High"), mutated.columns.get_loc(
            "Low"
        )
        mutated.iloc[train_bars:, close_col] = mutated_close
        mutated.iloc[train_bars:, open_col] = mutated_close * 0.999
        mutated.iloc[train_bars:, high_col] = mutated_close + spread
        mutated.iloc[train_bars:, low_col] = mutated_close - spread

        def run_with(df):
            provider = MagicMock()
            provider.get_ohlcv.return_value = df
            monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
            inp = RegimeAdaptiveWalkForwardInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                train_bars=train_bars,
                test_bars=test_bars,
                sma_param_grid=_RAWF_GRID["sma_crossover"],
                rsi_param_grid=_RAWF_GRID["rsi_mean_reversion"],
                macd_param_grid=_RAWF_GRID["macd_crossover"],
                bollinger_param_grid=_RAWF_GRID["bollinger_reversion"],
            )
            return run_regime_adaptive_walkforward_backtest(inp)

        baseline = run_with(long_ohlcv)
        mutated_result = run_with(mutated)

        assert baseline.windows[0].regime == mutated_result.windows[0].regime
        assert baseline.windows[0].hurst == pytest.approx(
            mutated_result.windows[0].hurst
        )
        assert (
            baseline.windows[0].selected_strategy
            == mutated_result.windows[0].selected_strategy
        )
        assert baseline.windows[0].best_params == mutated_result.windows[0].best_params
        assert baseline.windows[0].in_sample_sharpe == pytest.approx(
            mutated_result.windows[0].in_sample_sharpe
        )

        assert baseline.windows[0].out_of_sample_return != pytest.approx(
            mutated_result.windows[0].out_of_sample_return, abs=1e-9
        )

    def test_oos_signal_generation_receives_train_plus_test_warm_up(
        self, patched_long, monkeypatch
    ):
        """
        Regression test (P0 item 5), regime-adaptive walk-forward: same
        warm-up guarantee as run_walk_forward_backtest -- OOS signals must
        be generated from train_df + test_df together, never test_bars
        alone, regardless of which of the four strategies wins a given
        window. Patches every STRATEGY_REGISTRY entry (not just one),
        since this tool grid-searches all four per window.
        """
        import standard_quant_tools.backtest.strategies as strategies_mod

        train_bars, test_bars = 252, 63
        call_lengths: list = []

        def make_spy(real_fn):
            def spy(df, **kwargs):
                call_lengths.append(len(df))
                return real_fn(df, **kwargs)

            return spy

        for name, fn in list(strategies_mod.STRATEGY_REGISTRY.items()):
            monkeypatch.setitem(strategies_mod.STRATEGY_REGISTRY, name, make_spy(fn))

        inp = RegimeAdaptiveWalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            train_bars=train_bars,
            test_bars=test_bars,
            sma_param_grid=_RAWF_GRID["sma_crossover"],
            rsi_param_grid=_RAWF_GRID["rsi_mean_reversion"],
            macd_param_grid=_RAWF_GRID["macd_crossover"],
            bollinger_param_grid=_RAWF_GRID["bollinger_reversion"],
        )
        run_regime_adaptive_walkforward_backtest(inp)

        # test_bars alone must never appear -- every OOS signal call used
        # train_bars + test_bars of history. (train_bars alone also appears,
        # legitimately, from backtest_grid's own in-sample search.)
        assert test_bars not in call_lengths
        assert (train_bars + test_bars) in call_lengths

    def test_stitched_oos_is_one_continuous_backtest_not_reset_per_window(
        self, patched_long, monkeypatch
    ):
        """Same continuity guarantee as run_walk_forward_backtest's version:
        the stitched OOS aggregate comes from one continuous run_strategy
        call spanning the whole OOS region, not a per-window reset."""
        import standard_quant_tools.agent.tools as tools_mod

        train_bars, test_bars = 252, 63
        call_lengths: list = []
        real_run_strategy = backtest_tools.run_strategy

        def spy(price_data, signal_series, **kwargs):
            call_lengths.append(len(price_data))
            return real_run_strategy(price_data, signal_series, **kwargs)

        monkeypatch.setattr(backtest_tools, "run_strategy", spy)

        inp = RegimeAdaptiveWalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            train_bars=train_bars,
            test_bars=test_bars,
            sma_param_grid=_RAWF_GRID["sma_crossover"],
            rsi_param_grid=_RAWF_GRID["rsi_mean_reversion"],
            macd_param_grid=_RAWF_GRID["macd_crossover"],
            bollinger_param_grid=_RAWF_GRID["bollinger_reversion"],
        )
        result = run_regime_adaptive_walkforward_backtest(inp)

        assert call_lengths.count(test_bars) == result.n_windows
        assert call_lengths.count(result.n_windows * test_bars) == 1

    def test_dispatched_through_dispatch(self, patched_long):
        result = dispatch(
            "run_regime_adaptive_walkforward_backtest",
            {
                "symbol": "AAPL",
                "start_date": START,
                "end_date": END,
                "train_bars": 252,
                "test_bars": 63,
                "sma_param_grid": _RAWF_GRID["sma_crossover"],
                "rsi_param_grid": _RAWF_GRID["rsi_mean_reversion"],
                "macd_param_grid": _RAWF_GRID["macd_crossover"],
                "bollinger_param_grid": _RAWF_GRID["bollinger_reversion"],
            },
        )
        assert result["symbol"] == "AAPL"
        assert "strategy_stability" in result
        assert "stitched_oos_sharpe" in result


# ── Feature 2: Pair Scanner ────────────────────────────────────────────────────


class TestScanPairs:
    def test_returns_result(self, patched_long):
        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START,
            end_date=END,
        )
        result = scan_pairs(inp)
        assert result.n_pairs_tested >= 0

    def test_result_structure(self, patched_long):
        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START,
            end_date=END,
        )
        result = scan_pairs(inp)
        assert isinstance(result.n_pairs_tested, int)
        assert isinstance(result.n_pairs_cointegrated, int)
        assert isinstance(result.pairs, list)

    def test_pairs_respect_max_pairs(self, patched_long):
        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL", "TSLA"],
            start_date=START,
            end_date=END,
            max_pairs=2,
        )
        result = scan_pairs(inp)
        assert result.n_pairs_returned <= 2
        assert len(result.pairs) <= 2

    def test_pair_fields_populated(self, patched_long):
        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START,
            end_date=END,
        )
        result = scan_pairs(inp)
        for pair in result.pairs:
            assert pair.symbol_a != pair.symbol_b
            assert 0.0 <= pair.p_value <= 1.0
            assert pair.half_life_days > 0
            assert pair.signal in ("long_a_short_b", "short_a_long_b", "neutral")

    def test_n_pairs_tested_equals_combinations(self, patched_long):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        inp = PairScannerInput(tickers=tickers, start_date=START, end_date=END)
        result = scan_pairs(inp)
        # C(3,2) = 3 combinations
        assert result.n_pairs_tested == 3

    def test_pairs_sorted_by_half_life(self, patched_long):
        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL", "TSLA"],
            start_date=START,
            end_date=END,
        )
        result = scan_pairs(inp)
        hls = [p.half_life_days for p in result.pairs]
        assert hls == sorted(hls)

    def test_single_ticker_returns_zero_pairs(self, patched_long):
        inp = PairScannerInput(tickers=["AAPL"], start_date=START, end_date=END)
        result = scan_pairs(inp)
        assert result.n_pairs_tested == 0
        assert len(result.pairs) == 0

    def test_ticker_fetch_failure_is_reported_not_swallowed(
        self, long_ohlcv, monkeypatch
    ):
        from unittest.mock import MagicMock

        from standard_quant_tools.data.factory import DataFactory

        def flaky_get_ohlcv(symbol, start, end):
            if symbol == "MSFT":
                raise ValueError("no data for MSFT")
            return long_ohlcv

        provider = MagicMock()
        provider.get_ohlcv.side_effect = flaky_get_ohlcv
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL"], start_date=START, end_date=END
        )
        result = scan_pairs(inp)

        assert result.failed_tickers == {"MSFT": "no data for MSFT"}
        # Only AAPL/GOOGL remain -> exactly one pair, MSFT excluded entirely.
        assert result.n_pairs_tested == 1

    def test_pair_test_failure_is_reported_not_swallowed(
        self, patched_long, monkeypatch
    ):
        """The per-pair fallback path reports failures rather than hiding them.

        scan_pairs has two paths: a batch native call when every series shares
        an index, and this per-pair loop otherwise. Disabling the batch entry
        point is what puts the loop under test -- without it this monkeypatch
        no longer reaches the code it is trying to exercise, because the batch
        path never calls cointegration_test.
        """
        import standard_quant_tools.analysis.cointegration as cointegration_module

        def failing_coint(*args, **kwargs):
            raise RuntimeError("degenerate series")

        def unavailable_batch(*args, **kwargs):
            raise RuntimeError("batch path disabled for this test")

        monkeypatch.setattr(cointegration_module, "cointegration_test", failing_coint)
        monkeypatch.setattr(
            cointegration_module, "scan_cointegrated_pairs", unavailable_batch
        )

        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START,
            end_date=END,
        )
        result = scan_pairs(inp)

        assert len(result.pairs) == 0
        assert len(result.failed_pairs) == 3  # C(3,2) combinations, all erroring
        assert all(f.reason == "degenerate series" for f in result.failed_pairs)
        tested_symbols = {(f.symbol_a, f.symbol_b) for f in result.failed_pairs}
        assert tested_symbols == {
            ("AAPL", "MSFT"),
            ("AAPL", "GOOGL"),
            ("MSFT", "GOOGL"),
        }

    def test_batch_path_produces_the_same_answer_as_the_loop(
        self, patched_long, monkeypatch
    ):
        """The two paths must agree; the batch one exists only to be faster.

        Runs the same universe twice -- once with the batch entry point
        available, once with it disabled so the per-pair loop runs -- and
        requires identical output.
        """
        import standard_quant_tools.analysis.cointegration as cointegration_module

        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START,
            end_date=END,
        )
        batched = scan_pairs(inp)

        real_batch = cointegration_module.scan_cointegrated_pairs

        def unavailable_batch(*args, **kwargs):
            raise RuntimeError("batch path disabled for this test")

        monkeypatch.setattr(
            cointegration_module, "scan_cointegrated_pairs", unavailable_batch
        )
        looped = scan_pairs(inp)
        monkeypatch.setattr(cointegration_module, "scan_cointegrated_pairs", real_batch)

        assert batched.n_pairs_tested == looped.n_pairs_tested
        assert batched.n_pairs_cointegrated == looped.n_pairs_cointegrated
        assert [
            (p.symbol_a, p.symbol_b, p.p_value, p.hedge_ratio, p.half_life_days)
            for p in batched.pairs
        ] == [
            (p.symbol_a, p.symbol_b, p.p_value, p.hedge_ratio, p.half_life_days)
            for p in looped.pairs
        ]

    def test_batch_path_falls_back_when_indexes_differ(self, patched_long, monkeypatch):
        """A ragged universe must not silently change which bars a pair sees.

        The batch path aligns the whole universe onto one common sample; the
        loop aligns each pair against only its partner. Those differ when the
        indexes differ, so scan_pairs is required to take the loop instead of
        quietly re-testing every pair on a shorter sample.
        """
        import standard_quant_tools.agent.tools as tools_module

        provider = research_tools.DataFactory.get_provider()
        original = provider.get_ohlcv
        calls = {"n": 0}

        def ragged(symbol, start, end, *args, **kwargs):
            df = original(symbol, start, end, *args, **kwargs)
            calls["n"] += 1
            # Drop a leading bar from one symbol so the indexes disagree.
            return df.iloc[1:] if symbol == "GOOGL" else df

        monkeypatch.setattr(provider, "get_ohlcv", ragged)

        inp = PairScannerInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=START,
            end_date=END,
        )
        result = scan_pairs(inp)

        # The point is that it completes correctly on the loop path, testing
        # every pair on its own pairwise-aligned sample.
        assert calls["n"] == 3
        assert result.n_pairs_tested == 3


# ── Feature 3: Walk-Forward Backtest ──────────────────────────────────────────


class TestWalkForwardBacktest:
    def test_returns_result(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        assert result.symbol == "AAPL"
        assert result.strategy == "sma_crossover"

    def test_windows_populated(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        assert result.n_windows >= 1
        assert len(result.windows) == result.n_windows

    def test_window_dates_sequential(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        for i, win in enumerate(result.windows):
            assert win.window_index == i
            assert win.train_start <= win.train_end
            assert win.test_start <= win.test_end
            assert win.train_end < win.test_start

    def test_aggregate_metrics_are_floats(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        assert isinstance(result.avg_oos_sharpe, float)
        assert isinstance(result.avg_oos_return, float)
        assert isinstance(result.avg_oos_max_drawdown, float)
        assert 0.0 <= result.pct_windows_profitable <= 1.0

    def test_param_stability_has_correct_keys(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        assert "fast_period" in result.param_stability
        assert "slow_period" in result.param_stability
        for key, info in result.param_stability.items():
            assert "most_common" in info
            assert "frequency" in info
            assert 0.0 <= info["frequency"] <= 1.0

    def test_invalid_strategy_raises(self, patched_long):
        """
        `strategy` is a Literal over STRATEGY_REGISTRY now, so an
        unregistered name is refused when the INPUT is constructed rather
        than part-way through the tool. That is strictly earlier: the old
        path fetched data and began a walk-forward before discovering the
        strategy did not exist.

        Pydantic's ValidationError subclasses ValueError, so the type this
        test always asserted is unchanged; only the message and the moment
        moved.
        """
        with pytest.raises(ValueError, match="strategy"):
            WalkForwardInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                strategy="nonexistent_strategy",
                param_grid={"fast_period": [5]},
                train_bars=252,
                test_bars=63,
            )

    def test_every_registered_strategy_is_accepted(self):
        """The Literal must not be narrower than the registry it mirrors."""
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        for name in STRATEGY_REGISTRY:
            WalkForwardInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                strategy=name,
                param_grid={"fast_period": [5]},
                train_bars=252,
                test_bars=63,
            )

    def test_insufficient_data_raises(self, patched_long, long_ohlcv, monkeypatch):
        from unittest.mock import MagicMock

        from standard_quant_tools.data.factory import DataFactory

        tiny_df = long_ohlcv.iloc[:50]
        prov = MagicMock()
        prov.get_ohlcv.return_value = tiny_df
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: prov)

        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5], "slow_period": [30]},
            train_bars=252,
            test_bars=63,
        )
        with pytest.raises(ValueError, match="Not enough data"):
            run_walk_forward_backtest(inp)

    def test_stitched_fields_present_and_typed(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        for field in (
            "stitched_oos_return",
            "stitched_oos_sharpe",
            "stitched_oos_sortino",
            "stitched_oos_max_drawdown",
            "stitched_oos_calmar",
            "is_to_oos_sharpe_decay",
            "is_to_oos_return_decay",
        ):
            assert isinstance(getattr(result, field), float)
        assert isinstance(result.worst_oos_window, int)
        assert isinstance(result.longest_losing_window_streak, int)
        assert isinstance(result.parameter_turnover, float)

    def test_worst_oos_window_is_a_valid_index(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        assert 0 <= result.worst_oos_window < result.n_windows
        worst = next(
            w for w in result.windows if w.window_index == result.worst_oos_window
        )
        assert worst.out_of_sample_return == min(
            w.out_of_sample_return for w in result.windows
        )

    def test_longest_losing_streak_is_bounded(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        assert 0 <= result.longest_losing_window_streak <= result.n_windows

    def test_parameter_turnover_is_a_fraction(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        assert 0.0 <= result.parameter_turnover <= 1.0

    def test_stitched_return_is_not_the_naive_average(self, patched_long):
        """
        With >=2 windows and non-trivial per-window returns, the stitched
        (compounded) total return should generally differ from the simple
        average of per-window returns — proving the aggregate is actually
        computed from one chronological equity curve, not re-deriving the
        old avg_oos_return computation under a new name.
        """
        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        if result.n_windows >= 2:
            assert result.stitched_oos_return != pytest.approx(
                result.avg_oos_return, abs=1e-9
            )

    def test_in_sample_return_populated_per_window(self, patched_long):
        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        result = run_walk_forward_backtest(inp)
        for window in result.windows:
            assert isinstance(window.in_sample_return, float)

    def test_fill_price_threads_into_oos_leg(self, patched_long):
        """
        fill_price used to be silently hardcoded to "close" for every
        window's OOS evaluation (04_backtesting.md's own documented
        limitation). Now it threads through — next_open should change the
        stitched OOS result relative to the default.
        """
        kwargs = dict(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=252,
            test_bars=63,
        )
        default_result = run_walk_forward_backtest(WalkForwardInput(**kwargs))
        next_open_result = run_walk_forward_backtest(
            WalkForwardInput(**kwargs, fill_price="next_open")
        )
        assert next_open_result.stitched_oos_return != pytest.approx(
            default_result.stitched_oos_return, abs=1e-9
        )

    def test_no_lookahead_window0_params_unaffected_by_future_mutation(
        self, long_ohlcv, monkeypatch
    ):
        """
        Regression test proving run_walk_forward_backtest has no look-ahead
        bias: window 0's best_params are selected using backtest_grid on
        train_df = bars [0, train_bars) only. Replacing every bar from
        train_bars onward with a completely different synthetic path must
        leave window 0's in-sample selection untouched, even though it does
        change window 0's own out-of-sample result (test_df sits entirely
        inside the mutated region, which is the point — it proves the
        mutation is actually visible to the tool, so the first assertion
        isn't passing vacuously).
        """
        from unittest.mock import MagicMock

        from standard_quant_tools.data.factory import DataFactory

        train_bars, test_bars = 252, 63

        mutated = long_ohlcv.copy()
        rng = np.random.default_rng(4242)
        n_mutate = len(mutated) - train_bars
        # Deliberately different regime (negative drift, higher vol) from the
        # baseline fixture's bars [train_bars:], so if the tool leaked future
        # data into window 0's parameter search, the result would very
        # likely change.
        mutated_returns = rng.normal(-0.001, 0.03, n_mutate)
        last_train_close = float(mutated["Close"].iloc[train_bars - 1])
        mutated_close = last_train_close * np.cumprod(1 + mutated_returns)
        spread = rng.uniform(0.2, 1.2, n_mutate)
        close_col, open_col = mutated.columns.get_loc("Close"), mutated.columns.get_loc(
            "Open"
        )
        high_col, low_col = mutated.columns.get_loc("High"), mutated.columns.get_loc(
            "Low"
        )
        mutated.iloc[train_bars:, close_col] = mutated_close
        mutated.iloc[train_bars:, open_col] = mutated_close * 0.999
        mutated.iloc[train_bars:, high_col] = mutated_close + spread
        mutated.iloc[train_bars:, low_col] = mutated_close - spread

        def run_with(df):
            provider = MagicMock()
            provider.get_ohlcv.return_value = df
            monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
            inp = WalkForwardInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                strategy="sma_crossover",
                param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
                train_bars=train_bars,
                test_bars=test_bars,
            )
            return run_walk_forward_backtest(inp)

        baseline = run_with(long_ohlcv)
        mutated_result = run_with(mutated)

        # In-sample selection for window 0 depends only on bars [0, train_bars)
        # — untouched by the mutation — so it must be bit-for-bit identical.
        assert baseline.windows[0].best_params == mutated_result.windows[0].best_params
        assert baseline.windows[0].in_sample_sharpe == pytest.approx(
            mutated_result.windows[0].in_sample_sharpe
        )
        assert baseline.windows[0].in_sample_return == pytest.approx(
            mutated_result.windows[0].in_sample_return
        )

        # Sanity check the mutation is actually meaningful: window 0's
        # out-of-sample result lives entirely inside the mutated region and
        # should differ — otherwise the assertions above would be vacuous.
        assert baseline.windows[0].out_of_sample_return != pytest.approx(
            mutated_result.windows[0].out_of_sample_return, abs=1e-9
        )

    def test_stitched_oos_is_one_continuous_backtest_not_reset_per_window(
        self,
        patched_long,
        monkeypatch,
    ):
        """
        Regression test (P0 item 5): the stitched OOS aggregate must come
        from ONE continuous run_strategy call spanning the entire OOS
        region -- not from resetting capital and re-running independently
        per window. Spies on every run_strategy call's price_data length:
        expects exactly one call per window at length test_bars (the
        window-scoped diagnostic, unchanged), PLUS exactly one additional
        call at length n_windows * test_bars (the single continuous
        backtest the stitched_oos_* fields are computed from).
        """
        import standard_quant_tools.agent.tools as tools_mod

        train_bars, test_bars = 252, 63
        call_lengths: list = []
        real_run_strategy = backtest_tools.run_strategy

        def spy(price_data, signal_series, **kwargs):
            call_lengths.append(len(price_data))
            return real_run_strategy(price_data, signal_series, **kwargs)

        monkeypatch.setattr(backtest_tools, "run_strategy", spy)

        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=train_bars,
            test_bars=test_bars,
        )
        result = run_walk_forward_backtest(inp)

        assert call_lengths.count(test_bars) == result.n_windows
        assert call_lengths.count(result.n_windows * test_bars) == 1

    def test_oos_signal_generation_receives_train_plus_test_warm_up(
        self,
        patched_long,
        monkeypatch,
    ):
        """
        Regression test (P0 item 5): OOS signals must be generated from
        train_df + test_df together (giving indicators train_bars of
        warm-up before the OOS region starts), not test_df alone -- the
        original bug meant e.g. a 50-bar SMA was wrong for its first 49
        bars of every OOS window. Spies on the strategy registry's signal
        function (STRATEGY_REGISTRY is a single shared dict object --
        mutating the entry in place, via monkeypatch.setitem, is visible
        to agent.tools's own reference to the same dict) to confirm every
        call receives train_bars + test_bars of data, never test_bars alone.
        """
        import standard_quant_tools.backtest.strategies as strategies_mod

        train_bars, test_bars = 252, 63
        call_lengths: list = []
        real_sma = strategies_mod.STRATEGY_REGISTRY["sma_crossover"]

        def spy(df, **kwargs):
            call_lengths.append(len(df))
            return real_sma(df, **kwargs)

        monkeypatch.setitem(strategies_mod.STRATEGY_REGISTRY, "sma_crossover", spy)

        inp = WalkForwardInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            train_bars=train_bars,
            test_bars=test_bars,
        )
        result = run_walk_forward_backtest(inp)

        # One call per window for the OOS signal generation (train+test
        # slice) -- never called with test_bars alone.
        assert call_lengths.count(train_bars + test_bars) == result.n_windows
        assert test_bars not in call_lengths


# ── Feature 4: Portfolio Risk Attribution ─────────────────────────────────────


class TestPortfolioRiskAttribution:
    def test_returns_result(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.35, 0.25],
            start_date=START,
            end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        assert result.tickers == ["AAPL", "MSFT", "GOOGL"]

    def test_portfolio_metrics_are_floats(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.35, 0.25],
            start_date=START,
            end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        for field in (
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "var_95",
            "cvar_95",
            "information_ratio",
        ):
            assert isinstance(getattr(result, field), float), f"{field} not float"

    def test_risk_contributions_sum_to_one(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.35, 0.25],
            start_date=START,
            end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        total = sum(result.asset_risk_contributions.values())
        assert abs(total - 1.0) < 1e-4, f"MCR sum {total} != 1.0"

    def test_risk_contributions_have_correct_keys(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.35, 0.25],
            start_date=START,
            end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        assert set(result.asset_risk_contributions.keys()) == {"AAPL", "MSFT", "GOOGL"}

    def test_pca_keys_correct(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT", "GOOGL"],
            weights=[0.4, 0.35, 0.25],
            start_date=START,
            end_date=END,
            n_components=2,
        )
        result = get_portfolio_risk_attribution(inp)
        assert set(result.pca_variance_explained.keys()) == {"PC1", "PC2"}
        assert set(result.portfolio_pc_exposures.keys()) == {"PC1", "PC2"}

    def test_factor_fields_none_without_input(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT"],
            weights=[0.6, 0.4],
            start_date=START,
            end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        assert result.factor_loadings is None
        assert result.factor_r_squared is None
        assert result.factor_alpha is None

    def test_factor_fields_populated_with_input(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT"],
            weights=[0.6, 0.4],
            start_date=START,
            end_date=END,
            factor_tickers=["SPY"],
            factor_names=["mkt"],
        )
        result = get_portfolio_risk_attribution(inp)
        assert result.factor_loadings is not None
        assert "mkt" in result.factor_loadings
        assert result.factor_r_squared is not None
        assert result.factor_alpha is not None

    def test_max_drawdown_non_positive(self, patched_long):
        inp = RiskAttributionInput(
            tickers=["AAPL", "MSFT"],
            weights=[0.6, 0.4],
            start_date=START,
            end_date=END,
        )
        result = get_portfolio_risk_attribution(inp)
        assert result.max_drawdown <= 0.0


# ── Feature 5: Position Sizer ──────────────────────────────────────────────────


class TestPositionSizer:
    def test_returns_result(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=100_000.0,
        )
        result = get_position_size(inp)
        assert result.symbol == "AAPL"

    def test_last_close_positive(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=100_000.0,
        )
        result = get_position_size(inp)
        assert result.last_close > 0

    def test_atr_positive(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=100_000.0,
        )
        result = get_position_size(inp)
        assert result.atr > 0

    def test_fixed_risk_sizing_math(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=100_000.0,
            risk_per_trade_pct=0.01,
            atr_multiplier=2.0,
        )
        result = get_position_size(inp)
        expected_dollar_risk = 100_000.0 * 0.01
        expected_stop = result.atr * 2.0
        expected_shares = int(expected_dollar_risk / expected_stop)
        assert result.shares_fixed_risk == expected_shares

    def test_max_loss_bounded_by_risk_pct(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=100_000.0,
            risk_per_trade_pct=0.01,
        )
        result = get_position_size(inp)
        assert result.max_loss_fixed_risk <= 100_000.0 * 0.01 + 1  # rounding tolerance

    def test_without_kelly_inputs_recommendation_is_fixed_risk(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=100_000.0,
        )
        result = get_position_size(inp)
        assert result.recommended_sizing == "fixed_risk"
        assert result.kelly_fraction is None

    def test_with_kelly_inputs_populated(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=100_000.0,
            win_rate=0.55,
            avg_win_pct=0.05,
            avg_loss_pct=0.03,
        )
        result = get_position_size(inp)
        assert result.kelly_fraction is not None
        assert result.kelly_fraction >= 0.0
        assert result.shares_half_kelly is not None
        assert result.recommended_sizing in ("half_kelly", "fixed_risk")

    def test_negative_edge_falls_back_to_fixed_risk(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=100_000.0,
            win_rate=0.30,
            avg_win_pct=0.02,
            avg_loss_pct=0.05,
        )
        result = get_position_size(inp)
        assert result.kelly_fraction == 0.0
        assert result.recommended_sizing == "fixed_risk"

    def test_recommended_shares_is_int(self, patched_long):
        inp = PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=50_000.0,
            win_rate=0.6,
            avg_win_pct=0.06,
            avg_loss_pct=0.03,
        )
        result = get_position_size(inp)
        assert isinstance(result.recommended_shares, int)
        assert result.recommended_shares >= 0


class TestPositionSizerKellyInputValidation:
    """Regression coverage: win_rate/avg_win_pct/avg_loss_pct used to have no
    validation at all (unlike the sibling risk_per_trade_pct field), so an
    impossible win_rate like 2.0 could reach the Kelly formula and produce a
    silently over-100%-of-equity sizing recommendation."""

    def test_win_rate_above_one_rejected(self):
        with pytest.raises(pydantic.ValidationError, match="win_rate"):
            PositionSizerInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                account_equity=100_000.0,
                win_rate=2.0,
                avg_win_pct=0.05,
                avg_loss_pct=0.03,
            )

    def test_win_rate_below_zero_rejected(self):
        with pytest.raises(pydantic.ValidationError, match="win_rate"):
            PositionSizerInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                account_equity=100_000.0,
                win_rate=-0.1,
            )

    def test_win_rate_zero_and_one_are_valid_boundaries(self):
        PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=100_000.0,
            win_rate=0.0,
        )
        PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=100_000.0,
            win_rate=1.0,
        )

    def test_negative_avg_win_pct_rejected(self):
        with pytest.raises(pydantic.ValidationError, match="avg_win_pct"):
            PositionSizerInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                account_equity=100_000.0,
                avg_win_pct=-0.05,
            )

    def test_non_positive_avg_loss_pct_rejected(self):
        with pytest.raises(pydantic.ValidationError, match="avg_loss_pct"):
            PositionSizerInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                account_equity=100_000.0,
                avg_loss_pct=0.0,
            )
        with pytest.raises(pydantic.ValidationError, match="avg_loss_pct"):
            PositionSizerInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                account_equity=100_000.0,
                avg_loss_pct=-0.02,
            )


# ── Feature: "unknown" regime fallback ────────────────────────────────────────


class TestRegimeUnknownFallback:
    """When hurst_exponent returns 'unknown', the tool should not raise and
    must fall back to macd_crossover."""

    def test_unknown_regime_returns_valid_result(self, patched_long, monkeypatch):
        import standard_quant_tools.analysis.hurst as hurst_module

        # run_regime_adaptive_backtest imports hurst_exponent locally, so patch the source module.
        monkeypatch.setattr(
            hurst_module,
            "hurst_exponent",
            lambda *a, **kw: {
                "hurst": float("nan"),
                "regime": "unknown",
                "fit_r_squared": 0.0,
            },
        )
        inp = RegimeAdaptiveInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_regime_adaptive_backtest(inp)
        assert result.regime == "unknown"
        assert result.selected_strategy == "macd_crossover"
        assert result.backtest.final_equity > 0


# ── Feature 6: Buy-and-Hold Baseline ──────────────────────────────────────────


class TestBuyAndHold:
    """run_buy_and_hold returns a BacktestResult (same as active strategies)."""

    def test_returns_backtest_result(self, patched_long):
        inp = BuyAndHoldInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_buy_and_hold(inp)
        assert result.final_equity > 0
        assert isinstance(result.total_return, float)

    def test_total_return_is_float(self, patched_long):
        inp = BuyAndHoldInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_buy_and_hold(inp)
        assert isinstance(result.total_return, float)

    def test_max_drawdown_non_positive(self, patched_long):
        inp = BuyAndHoldInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_buy_and_hold(inp)
        assert result.max_drawdown <= 0.0

    def test_win_rate_bounded(self, patched_long):
        inp = BuyAndHoldInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_buy_and_hold(inp)
        assert 0.0 <= result.win_rate <= 1.0

    def test_equity_curve_nonempty(self, patched_long):
        inp = BuyAndHoldInput(symbol="AAPL", start_date=START, end_date=END)
        result = run_buy_and_hold(inp)
        assert len(result.equity_curve) > 0

    def test_custom_capital_reflected(self, patched_long):
        # With 50k capital, final equity should scale proportionally from the 50k base.
        # Even a 50% loss leaves 25k — much larger than the 10k default.
        inp = BuyAndHoldInput(
            symbol="AAPL", start_date=START, end_date=END, initial_capital=50_000.0
        )
        result = run_buy_and_hold(inp)
        assert result.final_equity > 25_000.0


# ── Feature 7: Compare Strategies ─────────────────────────────────────────────


class TestCompareStrategies:
    def test_returns_result(self, patched_long):
        inp = CompareStrategiesInput(symbol="AAPL", start_date=START, end_date=END)
        result = compare_strategies(inp)
        assert result.symbol == "AAPL"

    def test_four_strategies_returned(self, patched_long):
        inp = CompareStrategiesInput(symbol="AAPL", start_date=START, end_date=END)
        result = compare_strategies(inp)
        assert len(result.strategies) == 4

    def test_all_strategy_names_present(self, patched_long):
        inp = CompareStrategiesInput(symbol="AAPL", start_date=START, end_date=END)
        result = compare_strategies(inp)
        names = {s.strategy for s in result.strategies}
        assert names == {
            "sma_crossover",
            "rsi_mean_reversion",
            "macd_crossover",
            "bollinger_reversion",
        }

    def test_best_strategy_is_first(self, patched_long):
        inp = CompareStrategiesInput(
            symbol="AAPL", start_date=START, end_date=END, sort_by="sharpe_ratio"
        )
        result = compare_strategies(inp)
        assert result.best_strategy == result.strategies[0].strategy

    def test_sorted_descending_by_sharpe(self, patched_long):
        inp = CompareStrategiesInput(
            symbol="AAPL", start_date=START, end_date=END, sort_by="sharpe_ratio"
        )
        result = compare_strategies(inp)
        sharpes = [s.sharpe_ratio for s in result.strategies]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_sorted_descending_by_total_return(self, patched_long):
        inp = CompareStrategiesInput(
            symbol="AAPL", start_date=START, end_date=END, sort_by="total_return"
        )
        result = compare_strategies(inp)
        returns = [s.total_return for s in result.strategies]
        assert returns == sorted(returns, reverse=True)

    def test_buy_and_hold_return_is_float(self, patched_long):
        inp = CompareStrategiesInput(symbol="AAPL", start_date=START, end_date=END)
        result = compare_strategies(inp)
        assert isinstance(result.buy_and_hold_return, float)

    def test_strategy_fields_are_floats(self, patched_long):
        inp = CompareStrategiesInput(symbol="AAPL", start_date=START, end_date=END)
        result = compare_strategies(inp)
        for s in result.strategies:
            assert isinstance(s.total_return, float)
            assert isinstance(s.sharpe_ratio, float)
            assert isinstance(s.max_drawdown, float)
            assert s.max_drawdown <= 0.0
            assert 0.0 <= s.win_rate <= 1.0

    def test_sort_by_in_result(self, patched_long):
        inp = CompareStrategiesInput(
            symbol="AAPL", start_date=START, end_date=END, sort_by="calmar_ratio"
        )
        result = compare_strategies(inp)
        assert result.sort_by == "calmar_ratio"


# ── Feature 8: Dispatch ────────────────────────────────────────────────────────


class TestDispatch:
    def test_routes_buy_and_hold(self, patched_long):
        result = dispatch(
            "run_buy_and_hold",
            {
                "symbol": "AAPL",
                "start_date": START,
                "end_date": END,
            },
        )
        assert "total_return" in result
        assert "final_equity" in result

    def test_routes_analyze_stock_risk(self, patched_long):
        result = dispatch(
            "analyze_stock_risk",
            {
                "symbol": "AAPL",
                "benchmark": "SPY",
                "period": "1y",
            },
        )
        assert "sharpe_ratio" in result

    def test_unknown_tool_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            dispatch("does_not_exist", {})

    def test_invalid_arguments_raise_validation_error(self, patched_long):
        with pytest.raises(pydantic.ValidationError):
            dispatch("run_buy_and_hold", {"bad_field": 99})

    def test_dispatch_covers_all_registry_tools(self):
        from standard_quant_tools.agent.tools import _TOOL_DISPATCH

        registry_names = {t["function"]["name"] for t in get_agent_tools()}
        assert registry_names == set(_TOOL_DISPATCH.keys())


# ── Feature 9: Model Validators ───────────────────────────────────────────────


class TestModelValidators:
    def test_portfolio_weights_must_sum_to_one(self):
        with pytest.raises(pydantic.ValidationError, match="sum"):
            PortfolioInput(
                tickers=["AAPL", "MSFT"],
                weights=[0.6, 0.6],
                start_date=START,
                end_date=END,
            )

    def test_portfolio_weights_length_must_match_tickers(self):
        with pytest.raises(pydantic.ValidationError):
            PortfolioInput(
                tickers=["AAPL", "MSFT", "GOOGL"],
                weights=[0.5, 0.5],
                start_date=START,
                end_date=END,
            )

    def test_portfolio_valid_weights_accepted(self):
        inp = PortfolioInput(
            tickers=["AAPL", "MSFT"],
            weights=[0.5, 0.5],
            start_date=START,
            end_date=END,
        )
        assert inp.tickers == ["AAPL", "MSFT"]

    def test_risk_attribution_weights_must_sum_to_one(self):
        with pytest.raises(pydantic.ValidationError, match="sum"):
            RiskAttributionInput(
                tickers=["AAPL", "MSFT"],
                weights=[0.7, 0.7],
                start_date=START,
                end_date=END,
            )

    def test_risk_attribution_weights_length_mismatch(self):
        with pytest.raises(pydantic.ValidationError):
            RiskAttributionInput(
                tickers=["AAPL", "MSFT", "GOOGL"],
                weights=[0.5, 0.5],
                start_date=START,
                end_date=END,
            )

    def test_pca_n_components_must_be_positive(self):
        with pytest.raises(pydantic.ValidationError):
            PCAInput(
                tickers=["AAPL", "MSFT"],
                start_date=START,
                end_date=END,
                n_components=0,
            )

    def test_pca_negative_n_components_rejected(self):
        with pytest.raises(pydantic.ValidationError):
            PCAInput(
                tickers=["AAPL", "MSFT"],
                start_date=START,
                end_date=END,
                n_components=-1,
            )

    def test_position_sizer_risk_pct_must_be_positive(self):
        with pytest.raises(pydantic.ValidationError):
            PositionSizerInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                account_equity=100_000.0,
                risk_per_trade_pct=0.0,
            )

    def test_position_sizer_risk_pct_cannot_exceed_one(self):
        with pytest.raises(pydantic.ValidationError):
            PositionSizerInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                account_equity=100_000.0,
                risk_per_trade_pct=1.5,
            )

    def test_position_sizer_valid_risk_pct_accepted(self):
        inp = PositionSizerInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            account_equity=100_000.0,
            risk_per_trade_pct=0.01,
        )


# ── Extended Backtest Diagnostics ─────────────────────────────────────────────


class TestBacktestDiagnostics:
    def test_returns_result(self, patched_long):
        inp = BacktestDiagnosticsInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="sma_crossover",
            parameters={"fast_period": 10, "slow_period": 50},
        )
        result = get_backtest_diagnostics(inp)
        assert result.symbol == "AAPL"
        assert result.strategy_type == "sma_crossover"

    def test_unknown_strategy_raises(self, patched_long):
        inp = BacktestDiagnosticsInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="nonexistent_strategy",
        )
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_backtest_diagnostics(inp)

    def test_top_drawdowns_respects_top_n(self, patched_long):
        inp = BacktestDiagnosticsInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="sma_crossover",
            parameters={"fast_period": 5, "slow_period": 20},
            top_n_drawdowns=2,
        )
        result = get_backtest_diagnostics(inp)
        assert len(result.top_drawdowns) <= 2

    def test_drawdown_episode_fields_populated(self, patched_long):
        inp = BacktestDiagnosticsInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="sma_crossover",
            parameters={"fast_period": 5, "slow_period": 20},
        )
        result = get_backtest_diagnostics(inp)
        for ep in result.top_drawdowns:
            assert ep.depth <= 0.0
            assert ep.duration_bars >= 0
            if ep.end is None:
                assert ep.recovery_bars is None

    def test_trade_diagnostics_fields_populated(self, patched_long):
        inp = BacktestDiagnosticsInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="sma_crossover",
            parameters={"fast_period": 5, "slow_period": 20},
        )
        result = get_backtest_diagnostics(inp)
        td = result.trade_diagnostics
        assert isinstance(td.expectancy_pct, float)
        assert td.max_consecutive_wins >= 0
        assert td.max_consecutive_losses >= 0

    def test_exposure_fields_bounded(self, patched_long):
        inp = BacktestDiagnosticsInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy_type="sma_crossover",
            parameters={"fast_period": 5, "slow_period": 20},
        )
        result = get_backtest_diagnostics(inp)
        exp = result.exposure
        assert 0.0 <= exp.time_in_market <= 1.0
        assert 0.0 <= exp.pct_long <= 1.0
        assert 0.0 <= exp.pct_short <= 1.0

    def test_dispatched_through_dispatch(self, patched_long):
        result = dispatch(
            "get_backtest_diagnostics",
            {
                "symbol": "AAPL",
                "start_date": START,
                "end_date": END,
                "strategy_type": "sma_crossover",
                "parameters": {"fast_period": 10, "slow_period": 50},
            },
        )
        assert result["symbol"] == "AAPL"
        assert "top_drawdowns" in result
        assert "trade_diagnostics" in result
        assert "exposure" in result

    def test_next_open_fill_price_differs_from_default(self, patched_long):
        base = get_backtest_diagnostics(
            BacktestDiagnosticsInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                strategy_type="sma_crossover",
                parameters={"fast_period": 10, "slow_period": 50},
            )
        )
        next_open = get_backtest_diagnostics(
            BacktestDiagnosticsInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                strategy_type="sma_crossover",
                parameters={"fast_period": 10, "slow_period": 50},
                fill_price="next_open",
            )
        )
        assert base.total_return != pytest.approx(next_open.total_return, abs=1e-9)


# ── True Portfolio Simulation (shared cash, rebalancing) ──────────────────────


def _rebalance_weights(long_ohlcv, tickers, weight_per_ticker, rebalance_indices):
    dates = [str(long_ohlcv.index[i].date()) for i in rebalance_indices]
    return {t: {d: weight_per_ticker for d in dates} for t in tickers}


class TestPortfolioSimulation:
    def test_returns_result(self, patched_long, long_ohlcv):
        tickers = ["AAPL", "MSFT"]
        weights = _rebalance_weights(long_ohlcv, tickers, 0.4, [0, 100, 300])
        inp = PortfolioSimulationInput(
            tickers=tickers,
            start_date=START,
            end_date=END,
            target_weights=weights,
        )
        result = run_portfolio_simulation(inp)
        assert result.tickers == tickers
        assert result.n_rebalances == 3
        assert len(result.rebalance_log) == 3

    def test_summary_fields_typed(self, patched_long, long_ohlcv):
        tickers = ["AAPL", "MSFT"]
        weights = _rebalance_weights(long_ohlcv, tickers, 0.4, [0, 150])
        inp = PortfolioSimulationInput(
            tickers=tickers,
            start_date=START,
            end_date=END,
            target_weights=weights,
        )
        result = run_portfolio_simulation(inp)
        for field in (
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "var_95",
            "cvar_95",
            "avg_gross_leverage",
            "max_gross_leverage_used",
        ):
            assert isinstance(getattr(result, field), float)
        assert result.final_equity > 0
        assert len(result.equity_curve) > 0

    def test_rebalance_log_reports_gross_leverage_near_target(
        self, patched_long, long_ohlcv
    ):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        weights = _rebalance_weights(long_ohlcv, tickers, 0.25, [0])  # gross = 0.75
        inp = PortfolioSimulationInput(
            tickers=tickers,
            start_date=START,
            end_date=END,
            target_weights=weights,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        result = run_portfolio_simulation(inp)
        assert result.rebalance_log[0].gross_leverage_after == pytest.approx(
            0.75, abs=1e-6
        )
        assert result.rebalance_log[0].n_positions == 3

    def test_gross_leverage_exceeded_raises(self, patched_long, long_ohlcv):
        tickers = ["AAPL", "MSFT"]
        weights = _rebalance_weights(long_ohlcv, tickers, 0.8, [0])  # gross = 1.6
        inp_kwargs = dict(
            tickers=tickers, start_date=START, end_date=END, target_weights=weights
        )
        with pytest.raises(pydantic.ValidationError, match="leverage"):
            PortfolioSimulationInput(**inp_kwargs)

    def test_mismatched_rebalance_calendars_raise(self, long_ohlcv):
        dates_a = [str(long_ohlcv.index[0].date())]
        dates_b = [str(long_ohlcv.index[10].date())]
        with pytest.raises(pydantic.ValidationError, match="rebalance calendar"):
            PortfolioSimulationInput(
                tickers=["AAPL", "MSFT"],
                start_date=START,
                end_date=END,
                target_weights={
                    "AAPL": {dates_a[0]: 0.4},
                    "MSFT": {dates_b[0]: 0.4},
                },
            )

    def test_dispatched_through_dispatch(self, patched_long, long_ohlcv):
        tickers = ["AAPL", "MSFT"]
        weights = _rebalance_weights(long_ohlcv, tickers, 0.4, [0])
        result = dispatch(
            "run_portfolio_simulation",
            {
                "tickers": tickers,
                "start_date": START,
                "end_date": END,
                "target_weights": weights,
            },
        )
        assert result["tickers"] == tickers
        assert "rebalance_log" in result
        assert "sharpe_ratio" in result

    def test_per_share_commission_model_accepted(self, patched_long, long_ohlcv):
        tickers = ["AAPL", "MSFT"]
        weights = _rebalance_weights(long_ohlcv, tickers, 0.4, [0])
        inp = PortfolioSimulationInput(
            tickers=tickers,
            start_date=START,
            end_date=END,
            target_weights=weights,
            commission_model="per_share",
            per_share_rate=0.01,
        )
        result = run_portfolio_simulation(inp)
        assert result.n_rebalances == 1

    def test_next_open_fill_price_accepted(self, patched_long, long_ohlcv):
        tickers = ["AAPL", "MSFT"]
        weights = _rebalance_weights(long_ohlcv, tickers, 0.4, [0, 100])
        inp = PortfolioSimulationInput(
            tickers=tickers,
            start_date=START,
            end_date=END,
            target_weights=weights,
            fill_price="next_open",
        )
        result = run_portfolio_simulation(inp)
        assert result.n_rebalances == 2

    def test_total_return_matches_final_over_initial_capital(
        self, patched_long, long_ohlcv
    ):
        """
        Regression test (P0 item 8): a bar-0 rebalance under the default
        nonzero commission/slippage means equity_curve.iloc[0] is already
        net of that rebalance's cost. total_return must still be computed
        against the true starting capital, not against the already-net
        bar-0 value -- so it must reconcile exactly with
        final_equity/initial_capital - 1.
        """
        tickers = ["AAPL", "MSFT"]
        weights = _rebalance_weights(long_ohlcv, tickers, 0.4, [0])
        inp = PortfolioSimulationInput(
            tickers=tickers,
            start_date=START,
            end_date=END,
            target_weights=weights,
        )
        result = run_portfolio_simulation(inp)
        expected = result.final_equity / inp.initial_capital - 1.0
        assert result.total_return == pytest.approx(expected, abs=1e-6)

    def test_day0_cost_reflected_in_risk_metrics(self, patched_long, long_ohlcv):
        """
        Regression test (P0 item 8): the day-0 rebalance cost must show up
        in the return series feeding Sharpe/VaR/CVaR, not be silently
        replaced by a 0.0 first-day return. Comparing a costed run against
        an otherwise-identical zero-cost run, the costed run's day-0 loss
        must make VaR/CVaR at least as large (var_95/cvar_95 are reported
        as positive loss magnitudes -- bigger means worse).
        """
        tickers = ["AAPL", "MSFT"]
        weights = _rebalance_weights(long_ohlcv, tickers, 0.4, [0])
        costed = run_portfolio_simulation(
            PortfolioSimulationInput(
                tickers=tickers,
                start_date=START,
                end_date=END,
                target_weights=weights,
                commission_pct=0.05,
                slippage_pct=0.0,
            )
        )
        free = run_portfolio_simulation(
            PortfolioSimulationInput(
                tickers=tickers,
                start_date=START,
                end_date=END,
                target_weights=weights,
                commission_pct=0.0,
                slippage_pct=0.0,
            )
        )
        assert costed.var_95 >= free.var_95
        assert costed.cvar_95 >= free.cvar_95
        assert costed.total_return < free.total_return

    def test_day0_cost_reflected_in_max_drawdown_and_calmar(self, monkeypatch):
        """
        Regression test (P0-3): a portfolio that loses 5% entirely to a
        day-0 rebalance cost and then stays perfectly flat must report
        max_drawdown == -0.05, not 0.0. _metrics_with_day0_cost's
        equity_with_start (initial_capital prepended before the day-0
        rebalance) was already computed for total_return/Sharpe/VaR (item
        8) but max_drawdown/calmar_ratio were still computed from the bare
        equity_curve, whose iloc[0] is already net of the day-0 cost --
        so the "dip" from initial_capital to the post-cost value was
        invisible to a peak-to-trough calculation that never saw the peak.

        Uses a dedicated flat-price, single-ticker fixture (not the
        long_ohlcv random walk) so the expected max_drawdown is known
        exactly: weight=1.0, commission_pct=0.05, slippage_pct=0.0 ->
        exactly a 5% one-time cost on the full-notional day-0 entry, flat
        thereafter.
        """
        from unittest.mock import AsyncMock, MagicMock

        from standard_quant_tools.data.factory import DataFactory

        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        flat_close = pd.Series(100.0, index=dates)
        flat_df = pd.DataFrame(
            {
                "Open": flat_close,
                "High": flat_close,
                "Low": flat_close,
                "Close": flat_close,
                "Volume": pd.Series(1_000_000.0, index=dates),
            }
        )
        provider = MagicMock()
        provider.get_ohlcv.return_value = flat_df
        provider.get_ohlcv_async = AsyncMock(return_value=flat_df)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)

        weights = {"AAPL": {str(dates[0].date()): 1.0}}
        result = run_portfolio_simulation(
            PortfolioSimulationInput(
                tickers=["AAPL"],
                start_date=START,
                end_date=END,
                target_weights=weights,
                commission_pct=0.05,
                slippage_pct=0.0,
            )
        )
        assert result.total_return == pytest.approx(-0.05, abs=1e-6)
        assert result.max_drawdown == pytest.approx(-0.05, abs=1e-6)
        import math as _math

        assert _math.isfinite(result.calmar_ratio)


def _score_values(long_ohlcv, tickers, scores_by_ticker, rebalance_indices):
    dates = [str(long_ohlcv.index[i].date()) for i in rebalance_indices]
    return {t: {d: scores_by_ticker[t] for d in dates} for t in tickers}


class TestPortfolioSimulationScoreSignals:
    def test_zscore_normalized_gross_leverage_matches(self, patched_long, long_ohlcv):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        scores = _score_values(
            long_ohlcv, tickers, {"AAPL": 2.0, "MSFT": -1.0, "GOOGL": 0.5}, [0]
        )
        inp = PortfolioSimulationInput(
            tickers=tickers,
            start_date=START,
            end_date=END,
            target_weights=scores,
            signal_type="score",
            construction_method="zscore_normalized",
            gross_leverage=1.0,
            commission_pct=0.0,
            slippage_pct=0.0,
        )
        result = run_portfolio_simulation(inp)
        assert result.rebalance_log[0].gross_leverage_after == pytest.approx(
            1.0, abs=1e-3
        )

    def test_equal_weight_top_bottom_requires_n_long_n_short(self, long_ohlcv):
        tickers = ["AAPL", "MSFT", "GOOGL"]
        scores = _score_values(
            long_ohlcv, tickers, {"AAPL": 2.0, "MSFT": -1.0, "GOOGL": 0.5}, [0]
        )
        with pytest.raises(pydantic.ValidationError, match="n_long"):
            PortfolioSimulationInput(
                tickers=tickers,
                start_date=START,
                end_date=END,
                target_weights=scores,
                signal_type="score",
                construction_method="equal_weight_top_bottom",
            )

    def test_missing_construction_method_raises(self, long_ohlcv):
        tickers = ["AAPL", "MSFT"]
        scores = _score_values(long_ohlcv, tickers, {"AAPL": 2.0, "MSFT": -1.0}, [0])
        with pytest.raises(pydantic.ValidationError, match="construction_method"):
            PortfolioSimulationInput(
                tickers=tickers,
                start_date=START,
                end_date=END,
                target_weights=scores,
                signal_type="score",
            )

    def test_dollar_neutral_flag_runs_end_to_end(self, patched_long, long_ohlcv):
        # rank_weighted's weight-invariant unit tests live in test_sizing.py;
        # this only checks the make_dollar_neutral wiring runs end-to-end
        # through the agent tool without erroring.
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        scores = _score_values(
            long_ohlcv,
            tickers,
            {"AAPL": 2.0, "MSFT": 1.0, "GOOGL": 0.5, "AMZN": -1.5},
            [0],
        )
        inp = PortfolioSimulationInput(
            tickers=tickers,
            start_date=START,
            end_date=END,
            target_weights=scores,
            signal_type="score",
            construction_method="rank_weighted",
            make_dollar_neutral=True,
        )
        result = run_portfolio_simulation(inp)
        assert result.rebalance_log[0].gross_leverage_after > 0
        assert result.rebalance_log[0].n_positions == 4


# ── Pair Trade Backtest (synchronized two-leg execution) ─────────────────────


def _diverging_pair_ohlcv():
    """symbol_a dips then spikes relative to a flat symbol_b — same
    scenario shape as test_pair_backtest.py's unit-level fixture, sized to
    cross entry_z=1.0/exit_z=0.3 with hedge_ratio=1.0."""
    dates = pd.date_range(START, periods=20, freq="B")
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

    return {"A": _df(close_a), "B": _df(close_b)}


@pytest.fixture
def patched_pair(monkeypatch):
    from unittest.mock import MagicMock

    from standard_quant_tools.data.factory import DataFactory

    price_data = _diverging_pair_ohlcv()
    provider = MagicMock()
    provider.get_ohlcv.side_effect = lambda symbol, *a, **kw: price_data[symbol]
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return provider


class TestPairTradeBacktest:
    def test_returns_result(self, patched_pair):
        inp = PairTradeBacktestInput(
            symbol_a="A",
            symbol_b="B",
            start_date=START,
            end_date=END,
            hedge_ratio=1.0,
            entry_z=1.0,
            exit_z=0.3,
            commission_pct=0.0,
            slippage_pct=0.0,
            zscore_window=None,
        )
        result = run_pair_trade_backtest(inp)
        assert result.symbol_a == "A"
        assert result.symbol_b == "B"
        assert result.n_rebalances == 3
        assert result.n_round_trips == 1

    def test_summary_fields_typed(self, patched_pair):
        inp = PairTradeBacktestInput(
            symbol_a="A",
            symbol_b="B",
            start_date=START,
            end_date=END,
            hedge_ratio=1.0,
            entry_z=1.0,
            exit_z=0.3,
            zscore_window=None,
        )
        result = run_pair_trade_backtest(inp)
        for field in (
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
        ):
            assert isinstance(getattr(result, field), float)
        assert result.current_spread == pytest.approx(40.0)
        assert result.entry_spread == pytest.approx(40.0)

    def test_total_return_matches_final_over_initial_capital(self, patched_pair):
        """Regression test (P0 item 8), pair-backtest side: same invariant
        as TestPortfolioSimulation's version — total_return must reconcile
        exactly with final_equity/initial_capital, not silently absorb the
        entry rebalance's cost into a lost bar-0 baseline."""
        inp = PairTradeBacktestInput(
            symbol_a="A",
            symbol_b="B",
            start_date=START,
            end_date=END,
            hedge_ratio=1.0,
            entry_z=1.0,
            exit_z=0.3,
            zscore_window=None,
        )
        result = run_pair_trade_backtest(inp)
        expected = result.final_equity / inp.initial_capital - 1.0
        assert result.total_return == pytest.approx(expected, abs=1e-6)

    def test_no_entry_crossing_raises(self, patched_pair):
        from standard_quant_tools.error import ValidationError

        inp = PairTradeBacktestInput(
            symbol_a="A",
            symbol_b="B",
            start_date=START,
            end_date=END,
            hedge_ratio=1.0,
            entry_z=100.0,
        )
        with pytest.raises(ValidationError, match="never crossed"):
            run_pair_trade_backtest(inp)

    def test_dispatched_through_dispatch(self, patched_pair):
        result = dispatch(
            "run_pair_trade_backtest",
            {
                "symbol_a": "A",
                "symbol_b": "B",
                "start_date": START,
                "end_date": END,
                "hedge_ratio": 1.0,
                "entry_z": 1.0,
                "exit_z": 0.3,
                "zscore_window": None,
            },
        )
        assert result["symbol_a"] == "A"
        assert result["n_round_trips"] == 1


# ── Robustness Diagnostics (parameter sensitivity, DSR, block-bootstrap CI) ──


class TestRobustnessDiagnostics:
    def test_returns_result(self, patched_long):
        inp = RobustnessDiagnosticsInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            n_bootstrap_iterations=50,
            random_seed=0,
        )
        result = get_robustness_diagnostics(inp)
        assert result.symbol == "AAPL"
        assert result.strategy == "sma_crossover"
        assert "fast_period" in result.best_params
        assert "slow_period" in result.best_params

    def test_parameter_sensitivity_fields_present(self, patched_long):
        inp = RobustnessDiagnosticsInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50]},
            n_bootstrap_iterations=50,
            random_seed=0,
        )
        result = get_robustness_diagnostics(inp)
        for key in (
            "n_trials",
            "best",
            "median",
            "best_minus_median",
            "best_minus_rank2",
            "best_minus_top5_mean",
        ):
            assert key in result.parameter_sensitivity
        assert result.parameter_sensitivity["n_trials"] == 6  # 3 * 2 combos

    def test_dsr_in_unit_interval(self, patched_long):
        inp = RobustnessDiagnosticsInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            n_bootstrap_iterations=50,
            random_seed=0,
        )
        result = get_robustness_diagnostics(inp)
        assert 0.0 <= result.deflated_sharpe_ratio <= 1.0

    def test_bootstrap_ci_contains_point_estimate(self, patched_long):
        inp = RobustnessDiagnosticsInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            n_bootstrap_iterations=100,
            bootstrap_block_size=20,
            random_seed=0,
        )
        result = get_robustness_diagnostics(inp)
        assert (
            result.bootstrap_ci_lower
            <= result.bootstrap_point_estimate
            <= result.bootstrap_ci_upper
        )

    def test_reproducible_with_same_seed(self, patched_long):
        kwargs = dict(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            n_bootstrap_iterations=50,
            random_seed=7,
        )
        r1 = get_robustness_diagnostics(RobustnessDiagnosticsInput(**kwargs))
        r2 = get_robustness_diagnostics(RobustnessDiagnosticsInput(**kwargs))
        assert r1.bootstrap_ci_lower == r2.bootstrap_ci_lower
        assert r1.bootstrap_ci_upper == r2.bootstrap_ci_upper

    def test_few_trials_warns(self, patched_long):
        inp = RobustnessDiagnosticsInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            strategy="sma_crossover",
            param_grid={"fast_period": [5], "slow_period": [30]},
            n_bootstrap_iterations=20,
            random_seed=0,
        )
        result = get_robustness_diagnostics(inp)
        assert len(result.warnings) == 1

    def test_invalid_strategy_raises(self, patched_long):
        """Rejected at construction now — see the walk-forward equivalent for
        why that is the better boundary."""
        with pytest.raises(ValueError, match="strategy"):
            RobustnessDiagnosticsInput(
                symbol="AAPL",
                start_date=START,
                end_date=END,
                strategy="nonexistent_strategy",
                param_grid={"fast_period": [5]},
            )

    def test_dispatched_through_dispatch(self, patched_long):
        result = dispatch(
            "get_robustness_diagnostics",
            {
                "symbol": "AAPL",
                "start_date": START,
                "end_date": END,
                "strategy": "sma_crossover",
                "param_grid": {"fast_period": [5, 10], "slow_period": [30, 50]},
                "n_bootstrap_iterations": 30,
                "random_seed": 0,
            },
        )
        assert result["symbol"] == "AAPL"
        assert "deflated_sharpe_ratio" in result


# ── Capacity Report (liquidity/ADV-based capacity) ────────────────────────────


class TestCapacityReport:
    def test_returns_result(self, patched_factory):
        inp = CapacityReportInput(
            tickers=["AAPL", "MSFT"],
            start_date=START,
            end_date=END,
            target_weights={"AAPL": 0.6, "MSFT": 0.4},
        )
        result = get_capacity_report(inp)
        assert result.tickers == ["AAPL", "MSFT"]
        assert "AAPL" in result.per_ticker_max_account_size
        assert "MSFT" in result.per_ticker_max_account_size

    def test_binding_ticker_and_max_account_size_populated(self, patched_factory):
        inp = CapacityReportInput(
            tickers=["AAPL", "MSFT"],
            start_date=START,
            end_date=END,
            target_weights={"AAPL": 0.6, "MSFT": 0.4},
            max_participation=0.1,
        )
        result = get_capacity_report(inp)
        assert result.binding_ticker in ("AAPL", "MSFT")
        assert result.max_account_size is not None
        assert result.max_account_size > 0

    def test_zero_weight_ticker_has_none_max_account_size(self, patched_factory):
        inp = CapacityReportInput(
            tickers=["AAPL", "MSFT"],
            start_date=START,
            end_date=END,
            target_weights={"AAPL": 0.0, "MSFT": 1.0},
        )
        result = get_capacity_report(inp)
        assert result.per_ticker_max_account_size["AAPL"] is None
        assert result.days_to_liquidate_at_capacity["AAPL"] == 0.0

    def test_sector_exposure_included_by_default(self, patched_factory):
        inp = CapacityReportInput(
            tickers=["AAPL", "MSFT"],
            start_date=START,
            end_date=END,
            target_weights={"AAPL": 0.6, "MSFT": 0.4},
        )
        result = get_capacity_report(inp)
        assert result.sector_exposure is not None
        assert sum(result.sector_exposure.values()) == pytest.approx(1.0)

    def test_sector_exposure_excluded_when_disabled(self, patched_factory):
        inp = CapacityReportInput(
            tickers=["AAPL", "MSFT"],
            start_date=START,
            end_date=END,
            target_weights={"AAPL": 0.6, "MSFT": 0.4},
            include_sector_exposure=False,
        )
        result = get_capacity_report(inp)
        assert result.sector_exposure is None

    def test_missing_target_weight_raises(self):
        with pytest.raises(pydantic.ValidationError, match="target_weights"):
            CapacityReportInput(
                tickers=["AAPL", "MSFT"],
                start_date=START,
                end_date=END,
                target_weights={"AAPL": 0.6},
            )

    def test_dispatched_through_dispatch(self, patched_factory):
        result = dispatch(
            "get_capacity_report",
            {
                "tickers": ["AAPL", "MSFT"],
                "start_date": START,
                "end_date": END,
                "target_weights": {"AAPL": 0.6, "MSFT": 0.4},
            },
        )
        assert result["tickers"] == ["AAPL", "MSFT"]
        assert "max_account_size" in result


# ── Data Quality Report ────────────────────────────────────────────────────────


class TestDataQualityReport:
    def test_returns_result(self, patched_factory):
        inp = DataQualityReportInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_data_quality_report(inp)
        assert result.symbol == "AAPL"
        assert result.metadata["provider"] == "yfinance"
        assert result.metadata["survivorship_free"] is False
        assert result.metadata["point_in_time"] is False

    def test_findings_are_lists(self, patched_factory):
        inp = DataQualityReportInput(symbol="AAPL", start_date=START, end_date=END)
        result = get_data_quality_report(inp)
        assert isinstance(result.missing_bars, list)
        assert isinstance(result.stale_price_runs, list)
        assert isinstance(result.price_jumps, list)

    def test_custom_thresholds_accepted(self, patched_factory):
        inp = DataQualityReportInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            stale_run_length=5,
            jump_threshold=0.5,
        )
        result = get_data_quality_report(inp)
        assert result.symbol == "AAPL"

    def test_dispatched_through_dispatch(self, patched_factory):
        result = dispatch(
            "get_data_quality_report",
            {
                "symbol": "AAPL",
                "start_date": START,
                "end_date": END,
            },
        )
        assert result["symbol"] == "AAPL"
        assert "metadata" in result

    def test_non_empty_findings_map_correctly_to_pydantic_fields(self, patched_factory):
        """
        The three detect_* functions in data/quality.py return plain dicts
        that get_data_quality_report converts to MissingBar/StalePriceRun/
        PriceJump via **kwargs unpacking — this only exercises that mapping
        if the fixture data actually produces at least one finding of each
        kind (the default sample_ohlcv fixture is dense/noisy and produces
        none, so the dict->model field-name mapping was previously untested
        at the tool-integration layer).
        """
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        # Drop the 4th business day to create a weekday gap; hold Close flat
        # for the remaining bars (a stale run) except a >15% jump at the end.
        gapped_dates = dates.delete(3)  # 9 dates remain
        close = [100.0] * (len(gapped_dates) - 1) + [140.0]
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": [1_000_000.0] * len(close),
            },
            index=gapped_dates,
        )
        patched_factory.get_ohlcv.return_value = df

        inp = DataQualityReportInput(
            symbol="AAPL",
            start_date=START,
            end_date=END,
            stale_run_length=3,
            jump_threshold=0.15,
        )
        result = get_data_quality_report(inp)

        assert len(result.missing_bars) == 1
        assert result.missing_bars[0].date == str(dates[3].date())
        assert result.missing_bars[0].weekday == dates[3].strftime("%A")

        assert len(result.stale_price_runs) == 1
        stale = result.stale_price_runs[0]
        assert stale.price == pytest.approx(100.0)
        assert stale.run_length == len(gapped_dates) - 1  # all bars but the final jump

        assert len(result.price_jumps) == 1
        jump = result.price_jumps[0]
        assert jump.date == str(gapped_dates[-1].date())
        assert jump.pct_change == pytest.approx(0.4, abs=1e-6)


# ── Portfolio Optimization (Markowitz / risk parity / Black-Litterman) ──────


def _three_asset_ohlcv():
    """3 tickers with distinct (non-degenerate) return series -- a
    singular/perfectly-collinear covariance would break the unconstrained
    closed-form path, so each needs genuinely different price action."""
    np.random.seed(11)
    n = 300
    dates = pd.date_range(START, periods=n, freq="B")

    def _series(mu, sigma, seed):
        rng = np.random.default_rng(seed)
        rets = rng.normal(mu, sigma, n)
        close = 100.0 * np.cumprod(1 + rets)
        return pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": [1_000_000.0] * n,
            },
            index=dates,
        )

    return {
        "AAA": _series(0.0006, 0.012, 1),
        "BBB": _series(0.0004, 0.020, 2),
        "CCC": _series(0.0003, 0.008, 3),
    }


@pytest.fixture
def patched_three_asset(monkeypatch):
    from unittest.mock import AsyncMock, MagicMock

    from standard_quant_tools.data.factory import DataFactory

    price_data = _three_asset_ohlcv()
    provider = MagicMock()
    provider.get_ohlcv.side_effect = lambda symbol, *a, **kw: price_data[symbol]
    provider.get_ohlcv_async = AsyncMock(
        side_effect=lambda symbol, *a, **kw: price_data[symbol]
    )
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return provider


class TestPortfolioOptimization:
    def _input(self, **overrides):
        base = dict(
            tickers=["AAA", "BBB", "CCC"],
            start_date=START,
            end_date=END,
            method="max_sharpe",
        )
        base.update(overrides)
        return PortfolioOptimizationInput(**base)

    def test_max_sharpe_weights_sum_to_one(self, patched_three_asset):
        result = run_portfolio_optimization(
            self._input(method="max_sharpe", allow_short=True)
        )
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)
        assert result.converged is True
        assert set(result.weights) == {"AAA", "BBB", "CCC"}

    def test_min_volatility_long_only(self, patched_three_asset):
        result = run_portfolio_optimization(
            self._input(method="min_volatility", allow_short=False)
        )
        assert result.converged is True
        for w in result.weights.values():
            assert w >= -1e-6
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_target_return_achieves_target(self, patched_three_asset):
        result = run_portfolio_optimization(
            self._input(method="target_return", target_return=0.05, allow_short=True)
        )
        assert result.expected_return == pytest.approx(0.05, abs=1e-3)

    def test_risk_parity_reports_risk_contributions(self, patched_three_asset):
        result = run_portfolio_optimization(self._input(method="risk_parity"))
        assert result.converged is True
        assert result.risk_contributions is not None
        assert sum(result.risk_contributions.values()) == pytest.approx(1.0, abs=1e-3)
        for c in result.risk_contributions.values():
            assert c == pytest.approx(1 / 3, abs=0.05)

    def test_black_litterman_with_bullish_view(self, patched_three_asset):
        result = run_portfolio_optimization(
            self._input(
                method="black_litterman",
                views=[
                    {"assets": {"AAA": 1.0}, "view_return": 0.30, "confidence": 1.0}
                ],
            )
        )
        assert result.converged is True
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_dispatch_routes_correctly(self, patched_three_asset):
        result = dispatch(
            "run_portfolio_optimization",
            {
                "tickers": ["AAA", "BBB", "CCC"],
                "start_date": START,
                "end_date": END,
                "method": "min_volatility",
            },
        )
        assert "weights" in result
        assert result["method"] == "min_volatility"

    def test_black_litterman_requires_views(self):
        with pytest.raises(pydantic.ValidationError, match="at least one view"):
            self._input(method="black_litterman")


# ── Options Pricing, Greeks & Implied Volatility ────────────────────────────


class TestOptionPricingTool:
    def test_call_pricing_and_greeks(self):
        result = get_option_pricing(
            OptionPricingInput(
                spot=42,
                strike=40,
                time_to_expiry=0.5,
                risk_free_rate=0.10,
                volatility=0.20,
                option_type="call",
            )
        )
        assert result.price == pytest.approx(4.76, abs=0.01)
        assert 0.0 < result.greeks.delta < 1.0
        assert result.greeks.gamma > 0
        assert result.greeks.vega > 0

    def test_put_pricing(self):
        result = get_option_pricing(
            OptionPricingInput(
                spot=42,
                strike=40,
                time_to_expiry=0.5,
                risk_free_rate=0.10,
                volatility=0.20,
                option_type="put",
            )
        )
        assert result.price == pytest.approx(0.81, abs=0.01)
        assert -1.0 < result.greeks.delta < 0.0

    def test_dispatch_routes_correctly(self):
        result = dispatch(
            "get_option_pricing",
            {
                "spot": 42,
                "strike": 40,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.10,
                "volatility": 0.20,
                "option_type": "call",
            },
        )
        assert result["price"] == pytest.approx(4.76, abs=0.01)

    def test_invalid_option_type_rejected_by_pydantic(self):
        with pytest.raises(pydantic.ValidationError):
            OptionPricingInput(
                spot=42,
                strike=40,
                time_to_expiry=0.5,
                risk_free_rate=0.10,
                volatility=0.20,
                option_type="straddle",
            )


class TestImpliedVolatilityTool:
    def test_recovers_known_volatility(self):
        result = get_implied_volatility(
            ImpliedVolatilityInput(
                option_price=4.759422392871528,
                spot=42,
                strike=40,
                time_to_expiry=0.5,
                risk_free_rate=0.10,
                option_type="call",
            )
        )
        assert result.implied_volatility == pytest.approx(0.20, abs=1e-3)
        assert result.converged is True

    def test_dispatch_routes_correctly(self):
        result = dispatch(
            "get_implied_volatility",
            {
                "option_price": 4.759422392871528,
                "spot": 42,
                "strike": 40,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.10,
                "option_type": "call",
            },
        )
        assert result["implied_volatility"] == pytest.approx(0.20, abs=1e-3)
