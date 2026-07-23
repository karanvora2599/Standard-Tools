"""
Agent-tool level tests for run_backtest_compact / BacktestResultV2
(backtest/artifacts.py's URI round-trip, in the context of a real backtest).
"""

import pandas as pd
import pytest

from standard_quant_tools.agent.models import BacktestCompactInput
from standard_quant_tools.agent.tools import run_backtest_compact, dispatch
from standard_quant_tools.backtest.artifacts import load_artifact

START, END = "2022-01-01", "2024-01-01"


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path / "runs"


@pytest.fixture
def long_ohlcv() -> pd.DataFrame:
    import numpy as np
    np.random.seed(11)
    n = 500
    returns = np.random.normal(0.0004, 0.012, n)
    close = 100.0 * np.cumprod(1 + returns)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    spread = np.random.uniform(0.2, 1.2, n)
    return pd.DataFrame({
        "Open": pd.Series(close * 0.999, index=dates),
        "High": pd.Series(close + spread, index=dates),
        "Low": pd.Series(close - spread, index=dates),
        "Close": pd.Series(close, index=dates),
        "Volume": pd.Series(np.random.randint(1_000_000, 5_000_000, n).astype(float), index=dates),
    })


@pytest.fixture
def patched_long(long_ohlcv, monkeypatch):
    from unittest.mock import MagicMock
    from standard_quant_tools.data.factory import DataFactory
    provider = MagicMock()
    provider.get_ohlcv.return_value = long_ohlcv
    monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **kw: provider)
    return provider


class TestRunBacktestCompact:
    def test_returns_result_shape(self, patched_long, runs_dir):
        inp = BacktestCompactInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy_type="sma_crossover", parameters={"fast_period": 10, "slow_period": 30},
        )
        result = run_backtest_compact(inp)
        assert result.strategy_name == "sma_crossover"
        assert result.run_id
        assert result.equity_curve_uri.endswith("equity_curve.parquet")

    def test_equity_curve_uri_loads_back_matching_data(self, patched_long, runs_dir):
        inp = BacktestCompactInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy_type="sma_crossover", parameters={"fast_period": 10, "slow_period": 30},
        )
        result = run_backtest_compact(inp)
        loaded = load_artifact(result.equity_curve_uri).squeeze("columns")
        assert len(loaded) > 0
        assert loaded.iloc[-1] > 0

    def test_trades_uri_none_when_no_trades(self, patched_long, runs_dir):
        # A threshold combo unlikely to ever trigger a crossover on this series.
        inp = BacktestCompactInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy_type="sma_crossover", parameters={"fast_period": 400, "slow_period": 450},
        )
        result = run_backtest_compact(inp)
        if result.costs.num_trades == 0:
            assert result.trades_uri is None

    def test_summary_risk_exposure_cost_fields_present(self, patched_long, runs_dir):
        inp = BacktestCompactInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy_type="sma_crossover", parameters={"fast_period": 10, "slow_period": 30},
        )
        result = run_backtest_compact(inp)
        assert isinstance(result.summary.sharpe_ratio, float)
        assert isinstance(result.risk.max_drawdown, float)
        assert 0.0 <= result.exposure.time_in_market <= 1.0
        assert result.costs.total_cost_pct >= 0.0

    def test_few_trades_sets_warning_status(self, patched_long, runs_dir):
        inp = BacktestCompactInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy_type="sma_crossover", parameters={"fast_period": 400, "slow_period": 450},
        )
        result = run_backtest_compact(inp)
        if result.costs.num_trades < 5:
            assert result.validation_status == "warning"
            assert len(result.warnings) >= 1

    def test_custom_run_id_used_in_artifact_path(self, patched_long, runs_dir):
        inp = BacktestCompactInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy_type="sma_crossover", parameters={"fast_period": 10, "slow_period": 30},
            run_id="my-custom-run",
        )
        result = run_backtest_compact(inp)
        assert result.run_id == "my-custom-run"
        assert "my-custom-run" in result.equity_curve_uri

    def test_invalid_strategy_raises(self, patched_long, runs_dir):
        inp = BacktestCompactInput(
            symbol="AAPL", start_date=START, end_date=END,
            strategy_type="nonexistent_strategy",
        )
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_backtest_compact(inp)

    def test_dispatched_through_dispatch(self, patched_long, runs_dir):
        result = dispatch("run_backtest_compact", {
            "symbol": "AAPL", "start_date": START, "end_date": END,
            "strategy_type": "sma_crossover", "parameters": {"fast_period": 10, "slow_period": 30},
        })
        assert result["strategy_name"] == "sma_crossover"
        assert "equity_curve_uri" in result
