"""
Tests for backtest_grid: parallel parameter sweep over built-in strategies.
"""

import pandas as pd
import pytest

from standard_quant_tools.backtest import backtest_grid
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def price_df(sample_ohlcv) -> pd.DataFrame:
    return sample_ohlcv


# ── Strategy registry ─────────────────────────────────────────────────────────


class TestStrategyRegistry:
    def test_all_strategies_present(self):
        assert set(STRATEGY_REGISTRY) == {
            "sma_crossover",
            "rsi_mean_reversion",
            "macd_crossover",
            "bollinger_reversion",
            "donchian_breakout",
            "momentum_timeseries",
            "vwap_reversion",
            "adx_trend",
        }

    def test_all_strategies_callable(self):
        for fn in STRATEGY_REGISTRY.values():
            assert callable(fn)

    @pytest.mark.parametrize("name", list(STRATEGY_REGISTRY))
    def test_strategy_returns_series(self, name, price_df):
        fn = STRATEGY_REGISTRY[name]
        result = fn(price_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_df)
        assert set(result.unique()).issubset({0.0, 1.0, -1.0})


# ── backtest_grid output structure ────────────────────────────────────────────


class TestBacktestGridOutput:
    def test_returns_dataframe(self, price_df):
        result = backtest_grid(
            price_df,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            n_workers=1,
        )
        assert isinstance(result, pd.DataFrame)

    def test_row_count_equals_combinations(self, price_df):
        fast = [5, 10, 20]
        slow = [30, 50]
        result = backtest_grid(
            price_df,
            strategy="sma_crossover",
            param_grid={"fast_period": fast, "slow_period": slow},
            n_workers=1,
        )
        assert len(result) == len(fast) * len(slow)

    def test_param_columns_present(self, price_df):
        result = backtest_grid(
            price_df,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30]},
            n_workers=1,
        )
        assert "fast_period" in result.columns
        assert "slow_period" in result.columns

    def test_metric_columns_present(self, price_df):
        result = backtest_grid(
            price_df,
            strategy="sma_crossover",
            param_grid={"fast_period": [10], "slow_period": [30]},
            n_workers=1,
        )
        for col in [
            "sharpe_ratio",
            "total_return",
            "max_drawdown",
            "win_rate",
            "num_trades",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_by_sharpe_descending_by_default(self, price_df):
        result = backtest_grid(
            price_df,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50]},
            n_workers=1,
        )
        sharpes = result["sharpe_ratio"].tolist()
        assert sharpes == sorted(sharpes, reverse=True)

    def test_sort_ascending(self, price_df):
        result = backtest_grid(
            price_df,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10, 20], "slow_period": [30, 50]},
            sort_by="sharpe_ratio",
            ascending=True,
            n_workers=1,
        )
        sharpes = result["sharpe_ratio"].tolist()
        assert sharpes == sorted(sharpes)

    def test_max_drawdown_nonpositive(self, price_df):
        result = backtest_grid(
            price_df,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30]},
            n_workers=1,
        )
        assert (result["max_drawdown"] <= 0).all()

    def test_win_rate_bounded(self, price_df):
        result = backtest_grid(
            price_df,
            strategy="rsi_mean_reversion",
            param_grid={"period": [14], "oversold": [30], "overbought": [70]},
            n_workers=1,
        )
        assert (result["win_rate"] >= 0).all()
        assert (result["win_rate"] <= 1).all()


# ── All four strategies ───────────────────────────────────────────────────────


class TestAllStrategies:
    @pytest.mark.parametrize(
        "strategy,grid",
        [
            ("sma_crossover", {"fast_period": [5, 10], "slow_period": [30]}),
            (
                "rsi_mean_reversion",
                {"period": [14], "oversold": [30], "overbought": [70]},
            ),
            ("macd_crossover", {"fast": [12], "slow": [26], "signal": [9]}),
            ("bollinger_reversion", {"period": [20], "num_std": [2.0]}),
        ],
    )
    def test_strategy_runs(self, strategy, grid, price_df):
        result = backtest_grid(
            price_df, strategy=strategy, param_grid=grid, n_workers=1
        )
        assert not result.empty
        assert "sharpe_ratio" in result.columns


# ── Single-combo edge case ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_combination(self, price_df):
        result = backtest_grid(
            price_df,
            strategy="sma_crossover",
            param_grid={"fast_period": [10], "slow_period": [30]},
            n_workers=1,
        )
        assert len(result) == 1

    def test_invalid_strategy_raises(self, price_df):
        with pytest.raises(ValueError, match="Unknown strategy"):
            backtest_grid(price_df, strategy="nonexistent", param_grid={"x": [1]})

    def test_parallel_same_as_sequential(self, price_df):
        """n_workers=2 must produce the same rows as n_workers=1 (order may differ)."""
        grid = {"fast_period": [5, 10, 20], "slow_period": [30, 50]}
        seq = (
            backtest_grid(
                price_df, strategy="sma_crossover", param_grid=grid, n_workers=1
            )
            .sort_values(["fast_period", "slow_period"])
            .reset_index(drop=True)
        )

        par = (
            backtest_grid(
                price_df, strategy="sma_crossover", param_grid=grid, n_workers=2
            )
            .sort_values(["fast_period", "slow_period"])
            .reset_index(drop=True)
        )

        assert len(seq) == len(par)
        pd.testing.assert_frame_equal(
            seq[["fast_period", "slow_period", "sharpe_ratio"]],
            par[["fast_period", "slow_period", "sharpe_ratio"]],
            check_exact=False,
            atol=1e-6,
        )
