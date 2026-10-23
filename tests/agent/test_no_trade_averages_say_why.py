"""
A run that never traded reports no win rate and no average trade, and says why.

Every agent door that returns a backtest's `win_rate` or
`avg_trade_return_pct` reported 0.0 for a strategy that never traded -- a 0%
win rate and a flat average trade. They are null on the wire now, with the
reason in the result's warnings (or notes), the way `0683c50` did for the
trade diagnostics; the doors that rank or aggregate them leave the run out
and say how many they left.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent import models as M
from standard_quant_tools.agent.runtimes import _shared
from standard_quant_tools.agent.runtimes.backtest.tools import (
    compare_strategies,
    get_robustness_diagnostics,
    run_backtest_optimization,
    run_strategy_matrix,
)
from standard_quant_tools.data.factory import DataFactory

DATES = pd.date_range("2021-01-04", periods=400, freq="B")

WIN_RATE_LINE = (
    "win_rate is null: no trade closed, so there is no trade to count a win among."
)
AVG_LINE = (
    "avg_trade_return_pct is null: no trade closed, so there is no trade to average."
)


def _ohlcv(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0006, 0.012, len(DATES)))
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


def _lines(notes, field):
    return [line for line in notes if field in line and " null: " in line]


class TestTheBacktestResult:
    def _run(self, frame, signals):
        inp = M.BacktestInput(
            symbol="AAA",
            start_date="2021-01-04",
            end_date="2022-07-01",
            strategy_type="custom_signal",
        )
        return _shared._run_backtest(inp, frame, signals)

    def test_a_run_that_never_traded_is_null_with_a_reason(self, served):
        frame = served["AAA"]
        result = self._run(frame, pd.Series(0.0, index=frame.index))
        assert result.num_trades == 0
        assert result.win_rate is None
        assert result.avg_trade_return_pct is None
        assert WIN_RATE_LINE in result.warnings
        assert AVG_LINE in result.warnings
        wire = result.model_dump(mode="json")
        assert wire["win_rate"] is None and wire["avg_trade_return_pct"] is None

    def test_null_case_a_run_that_traded_has_both_and_no_line(self, served):
        frame = served["AAA"]
        signals = pd.Series(
            np.where(np.arange(len(frame)) % 40 < 20, 1.0, 0.0), index=frame.index
        )
        result = self._run(frame, signals)
        assert result.num_trades > 0
        assert 0.0 <= result.win_rate <= 1.0
        assert result.avg_trade_return_pct is not None
        assert _lines(result.warnings, "win_rate") == []
        assert _lines(result.warnings, "avg_trade_return_pct") == []


class TestTheRankedDoors:
    def test_the_optimization_ranks_the_idle_combination_last(self, served):
        """slow_period=500 on 400 bars never trades. Under sort_by=win_rate
        it ranked by its 0.0; now it has none, and says so."""
        result = run_backtest_optimization(
            M.BacktestOptInput(
                symbol="AAA",
                strategy="sma_crossover",
                start_date="2021-01-04",
                end_date="2022-07-01",
                param_grid={"fast_period": [10], "slow_period": [500, 50]},
                sort_by="win_rate",
            )
        )
        assert result.best_params["slow_period"] == 50
        idle = result.top_results[-1]
        assert idle.num_trades == 0
        assert idle.win_rate is None and idle.avg_trade_return_pct is None
        assert result.top_results[0].win_rate is not None
        assert result.n_unrankable == 1
        assert _lines(result.warnings, "top_results[1].win_rate")
        assert _lines(result.warnings, "top_results[1].avg_trade_return_pct")

    def test_the_matrix_cell_that_never_traded(self, served):
        result = run_strategy_matrix(
            M.StrategyMatrixInput(
                tickers=["AAA", "BBB"],
                strategies=["sma_crossover", "momentum_timeseries"],
                start_date="2021-01-04",
                end_date="2022-07-01",
                parameters={"sma_crossover": {"fast_period": 10, "slow_period": 500}},
                sort_by="win_rate",
            )
        )
        idle = [c for c in result.cells if c.num_trades == 0]
        assert idle and all(c.win_rate is None for c in idle)
        assert result.cells[-len(idle) :] == idle
        assert result.best_overall.win_rate is not None
        assert _lines(result.notes, "win_rate")
        assert result.model_dump(mode="json")["cells"][-1]["win_rate"] is None

    def test_the_comparison_names_the_null(self, served):
        """A comparison row of a strategy that never traded -- a slow
        average longer than the data -- has a null win rate and average
        trade, and ranks last under either."""
        for sort_by in ("win_rate", "avg_trade_return_pct"):
            result = compare_strategies(
                M.CompareStrategiesInput(
                    symbol="AAA",
                    start_date="2021-01-04",
                    end_date="2022-07-01",
                    sort_by=sort_by,
                    sma_parameters={"fast_period": 10, "slow_period": 500},
                )
            )
            last = len(result.strategies) - 1
            idle = result.strategies[last]
            assert idle.strategy == "sma_crossover"
            assert idle.num_trades == 0
            assert idle.win_rate is None and idle.avg_trade_return_pct is None
            assert result.n_unrankable == 1
            assert _lines(result.warnings, f"strategies[{last}].win_rate")
            assert _lines(result.warnings, f"strategies[{last}].avg_trade_return_pct")


class TestTheAggregate:
    def test_robustness_leaves_the_idle_trial_out_and_says_so(self, served):
        """Under max_drawdown a combination that never traded has 0.0, the
        best value there is; it was parameter_sensitivity's `best`."""
        result = get_robustness_diagnostics(
            M.RobustnessDiagnosticsInput(
                symbol="AAA",
                start_date="2021-01-04",
                end_date="2022-07-01",
                strategy="sma_crossover",
                param_grid={"fast_period": [10, 20], "slow_period": [50, 500]},
                sort_by="max_drawdown",
                n_bootstrap_iterations=50,
                random_seed=0,
            )
        )
        sensitivity = result.parameter_sensitivity
        assert sensitivity["n_excluded"] == 2
        assert sensitivity["n_trials"] == 2
        assert sensitivity["best"] < 0.0
        assert any(
            "parameter_sensitivity ranks 2 of 4" in w and "never traded" in w
            for w in result.warnings
        )

    def test_null_case_a_grid_that_always_traded_says_nothing(self, served):
        result = get_robustness_diagnostics(
            M.RobustnessDiagnosticsInput(
                symbol="AAA",
                start_date="2021-01-04",
                end_date="2022-07-01",
                strategy="sma_crossover",
                param_grid={"fast_period": [10, 20], "slow_period": [50, 100]},
                n_bootstrap_iterations=50,
                random_seed=0,
            )
        )
        assert result.parameter_sensitivity["n_excluded"] == 0
        assert not any("parameter_sensitivity ranks" in w for w in result.warnings)
