"""
Tests for backtest_grid's user-supplied signal-callable path — grid search
over a caller's own alpha logic, not just the built-in strategy registry.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.backtest import backtest_grid, run_strategy


def _momentum_signal(price_data: pd.DataFrame, threshold: float) -> pd.Series:
    ret = price_data["Close"].pct_change(5)
    return (ret > threshold).astype(int)


@pytest.fixture(scope="module")
def price_df(sample_ohlcv) -> pd.DataFrame:
    return sample_ohlcv


class TestCustomCallableStrategy:
    def test_returns_one_row_per_combo(self, price_df):
        result = backtest_grid(
            price_df,
            strategy=_momentum_signal,
            param_grid={"threshold": [0.0, 0.01, 0.02]},
            n_workers=1,
        )
        assert len(result) == 3
        assert set(result["threshold"]) == {0.0, 0.01, 0.02}

    def test_matches_manual_run_strategy(self, price_df):
        result = backtest_grid(
            price_df,
            strategy=_momentum_signal,
            param_grid={"threshold": [0.01]},
            n_workers=1,
        )
        row = result.iloc[0]
        manual = run_strategy(price_df, _momentum_signal(price_df, threshold=0.01))
        assert row["sharpe_ratio"] == pytest.approx(manual["sharpe_ratio"], abs=1e-9)
        assert row["total_return"] == pytest.approx(manual["total_return"], abs=1e-9)
        assert row["num_trades"] == manual["num_trades"]

    def test_multi_worker_request_does_not_crash_and_matches_sequential(self, price_df):
        sequential = backtest_grid(
            price_df,
            strategy=_momentum_signal,
            param_grid={"threshold": [0.0, 0.01, 0.02]},
            n_workers=1,
        )
        # A custom callable can't safely go through ProcessPoolExecutor (it may
        # be unpicklable) — requesting >1 workers must be silently downgraded
        # to sequential execution rather than crash or hang.
        requested_parallel = backtest_grid(
            price_df,
            strategy=_momentum_signal,
            param_grid={"threshold": [0.0, 0.01, 0.02]},
            n_workers=4,
        )
        pd.testing.assert_frame_equal(
            sequential.reset_index(drop=True),
            requested_parallel.reset_index(drop=True),
        )

    def test_lambda_strategy_works_sequentially(self, price_df):
        # Lambdas are the canonical unpicklable case — must still work when
        # never sent through a subprocess (n_workers=1 / sequential path).
        signal_fn = lambda df, threshold: (
            df["Close"].pct_change(5) > threshold
        ).astype(int)
        result = backtest_grid(
            price_df,
            strategy=signal_fn,
            param_grid={"threshold": [0.0, 0.01]},
            n_workers=1,
        )
        assert len(result) == 2

    def test_builtin_string_strategy_unaffected(self, price_df):
        result = backtest_grid(
            price_df,
            strategy="sma_crossover",
            param_grid={"fast_period": [5, 10], "slow_period": [30, 50]},
            n_workers=1,
        )
        assert len(result) == 4

    def test_unknown_string_strategy_still_raises(self, price_df):
        with pytest.raises(ValueError, match="Unknown strategy"):
            backtest_grid(
                price_df,
                strategy="not_a_real_strategy",
                param_grid={"x": [1]},
                n_workers=1,
            )

    def test_result_ranked_by_sort_by(self, price_df):
        result = backtest_grid(
            price_df,
            strategy=_momentum_signal,
            param_grid={"threshold": [0.0, 0.01, 0.02]},
            sort_by="sharpe_ratio",
            n_workers=1,
        )
        sharpes = result["sharpe_ratio"].to_numpy()
        assert np.all(sharpes[:-1] >= sharpes[1:])
