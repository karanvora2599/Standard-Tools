"""
Every agent door that ranks backtests obeys one rule and says what it left out.

Only a row that traded and has a finite metric can win. The optimization
tool, the regime-adaptive walk-forward's choice between strategies and the
strategy matrix each used their own sort: a bare `metric > best` that
accepted +inf and could never displace a NaN, and a `list.sort` over NaN
keys, whose order is undefined. compare_strategies is covered beside its
sort-direction tests in test_state_curves.py.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.agent.runtimes.backtest.tools as backtest_tools
from standard_quant_tools.agent.models import (
    BacktestOptInput,
    RegimeAdaptiveWalkForwardInput,
    StrategyMatrixInput,
)
from standard_quant_tools.agent.runtimes.backtest.tools import (
    run_backtest_optimization,
    run_regime_adaptive_walkforward_backtest,
    run_strategy_matrix,
)
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY
from standard_quant_tools.data.factory import DataFactory

DATES = pd.date_range("2021-01-04", periods=400, freq="B")


def _ohlcv(seed: int, drift: float = 0.0006) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(drift, 0.012, len(DATES)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.004,
            "Low": close * 0.996,
            "Close": close,
            "Volume": 1e6,
        },
        index=DATES,
    )


@pytest.fixture
def served(monkeypatch, tmp_path):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    frames = {"AAA": _ohlcv(1), "BBB": _ohlcv(2)}
    provider = MagicMock()
    provider.get_ohlcv.side_effect = lambda symbol, *a, **kw: frames[symbol]
    provider.get_ohlcv_async = AsyncMock(
        side_effect=lambda symbol, *a, **kw: frames[symbol]
    )
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return frames


class TestTheOptimizationTool:
    def test_repeated_values_are_one_combination(self, served):
        result = run_backtest_optimization(
            BacktestOptInput(
                symbol="AAA",
                strategy="sma_crossover",
                start_date="2021-01-04",
                end_date="2022-07-01",
                param_grid={"fast_period": [10, 10, 20], "slow_period": [50, 50]},
            )
        )
        assert result.n_combinations == 2
        assert len(result.top_results) == 2
        assert any("repeated" in w for w in result.warnings)

    def test_a_no_trade_combination_cannot_win_calmar(self, served):
        """slow_period=500 on 400 bars never trades: its Calmar used to be
        +inf, and it was best_params."""
        result = run_backtest_optimization(
            BacktestOptInput(
                symbol="AAA",
                strategy="sma_crossover",
                start_date="2021-01-04",
                end_date="2022-07-01",
                param_grid={"fast_period": [10], "slow_period": [500, 50]},
                sort_by="calmar_ratio",
            )
        )
        assert result.best_params["slow_period"] == 50
        assert result.top_results[-1].num_trades == 0
        assert result.n_unrankable == 1
        assert any("never traded" in w for w in result.warnings)
        assert list(result.best_params) == ["fast_period", "slow_period"]

    def test_a_clean_grid_says_nothing(self, served):
        """Null case."""
        result = run_backtest_optimization(
            BacktestOptInput(
                symbol="AAA",
                strategy="sma_crossover",
                start_date="2021-01-04",
                end_date="2022-07-01",
                param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            )
        )
        assert result.n_combinations == 4
        assert result.n_unrankable == 0
        assert result.warnings == []


class TestTheRegimeAdaptiveChoiceBetweenStrategies:
    def _grid_with(self, planted):
        """A backtest_grid stand-in: each strategy's single row, with the
        metric and trade count the test plants for it."""

        def fake_grid(price_data, strategy, param_grid, **kwargs):
            metric, trades = planted.get(strategy, (0.5, 3))
            row = {k: v[0] for k, v in param_grid.items()}
            row.update(
                {
                    kwargs.get("sort_by", "sharpe_ratio"): metric,
                    "sharpe_ratio": 0.1 if metric is None else metric,
                    "total_return": 0.01,
                    "num_trades": trades,
                }
            )
            frame = pd.DataFrame([row])
            frame.attrs["n_unrankable"] = int(
                trades == 0 or metric is None or not np.isfinite(metric)
            )
            return frame

        return fake_grid

    def _run(self, served, monkeypatch, planted, sort_by="sortino_ratio"):
        monkeypatch.setattr(backtest_tools, "backtest_grid", self._grid_with(planted))
        return run_regime_adaptive_walkforward_backtest(
            RegimeAdaptiveWalkForwardInput(
                symbol="AAA",
                start_date="2021-01-04",
                end_date="2022-07-01",
                train_bars=250,
                test_bars=150,
                sort_by=sort_by,
            )
        )

    def test_an_infinite_metric_does_not_win_the_window(self, served, monkeypatch):
        names = list(STRATEGY_REGISTRY)
        planted = {name: (0.2, 3) for name in names}
        planted[names[0]] = (float("inf"), 0)  # a do-nothing row, +inf Sortino
        planted[names[2]] = (1.4, 5)
        result = self._run(served, monkeypatch, planted)
        assert result.windows[0].selected_strategy == names[2]
        assert result.n_unrankable == 1
        assert any("candidate" in w for w in result.warnings)

    def test_a_nan_first_candidate_is_displaced(self, served, monkeypatch):
        """`metric > NaN` is always False, so a NaN in first place used to
        keep it whatever came after."""
        names = list(STRATEGY_REGISTRY)
        planted = {name: (0.2, 3) for name in names}
        planted[names[0]] = (float("nan"), 4)
        planted[names[1]] = (0.9, 4)
        result = self._run(served, monkeypatch, planted)
        assert result.windows[0].selected_strategy == names[1]

    def test_a_window_with_no_rankable_strategy_says_so(self, served, monkeypatch):
        planted = {name: (float("inf"), 0) for name in STRATEGY_REGISTRY}
        result = self._run(served, monkeypatch, planted)
        assert result.n_unrankable == len(STRATEGY_REGISTRY)
        assert any("Window(s) [0]" in w for w in result.warnings)

    def test_ordinary_candidates_pick_the_best(self, served, monkeypatch):
        """Null case: every candidate rankable, the highest wins."""
        names = list(STRATEGY_REGISTRY)
        planted = {name: (0.1 * i, 2) for i, name in enumerate(names)}
        result = self._run(served, monkeypatch, planted)
        assert result.windows[0].selected_strategy == names[-1]
        assert result.n_unrankable == 0
        assert result.warnings == []


class TestTheStrategyMatrix:
    def test_a_cell_that_never_traded_is_not_best(self, served):
        """Under max_drawdown a cell that never traded has 0.0, the best
        value possible, and it was best_overall."""
        result = run_strategy_matrix(
            StrategyMatrixInput(
                tickers=["AAA", "BBB"],
                strategies=["sma_crossover", "momentum_timeseries"],
                start_date="2021-01-04",
                end_date="2022-07-01",
                parameters={"sma_crossover": {"fast_period": 10, "slow_period": 500}},
                sort_by="max_drawdown",
            )
        )
        idle = [c for c in result.cells if c.num_trades == 0]
        assert {c.strategy for c in idle} == {"sma_crossover"}
        assert result.cells[-len(idle) :] == idle
        assert result.best_overall.num_trades > 0
        assert set(result.best_per_ticker.values()) == {"momentum_timeseries"}
        assert result.n_unrankable == len(idle)
        assert any("never traded" in n for n in result.notes)

    def test_an_ordinary_matrix_is_ranked_as_before(self, served):
        """Null case."""
        result = run_strategy_matrix(
            StrategyMatrixInput(
                tickers=["AAA", "BBB"],
                strategies=["sma_crossover", "momentum_timeseries"],
                start_date="2021-01-04",
                end_date="2022-07-01",
            )
        )
        sharpes = [c.sharpe_ratio for c in result.cells]
        assert sharpes == sorted(sharpes, reverse=True)
        assert result.n_unrankable == 0
        assert result.best_overall == result.cells[0]
