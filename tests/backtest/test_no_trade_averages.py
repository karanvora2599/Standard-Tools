"""
A strategy that never traded has no win rate and no average trade.

`win_rate` and `avg_trade_return_pct` are taken over the closed trades. With
none, both engines reported 0.0 -- a 0% win rate and a flat average trade,
statements about trades that did not happen -- and those numbers fed every
ranking, comparison and aggregate that reads a backtest. They are NaN now on
the Python path and in the native kernel alike (run_strategy,
run_strategy_summary and the two batch kernels), so the two backends still
agree, and every consumer that ranks or aggregates them excludes the run
rather than counting it at 0.0.
"""

import math

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.engine as engine
from standard_quant_tools.backtest.engine import (
    _compute_trade_stats,
    backtest_grid,
    run_strategy,
)
from standard_quant_tools.backtest.robustness import parameter_sensitivity
from standard_quant_tools.error import ValidationError

AVERAGES = ("win_rate", "avg_trade_return_pct")


@pytest.fixture(params=["native", "python"])
def engine_path(request, monkeypatch):
    if request.param == "native":
        if not engine.HAS_CPP:
            pytest.skip("the C++ extension is not built")
    else:
        monkeypatch.setattr(engine, "HAS_CPP", False)
    return request.param


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-04", periods=300, freq="B")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, len(dates)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.004,
            "Low": close * 0.996,
            "Close": close,
            "Volume": 1e6,
        },
        index=dates,
    )


def _every_twenty(price_data: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(np.arange(len(price_data)) % 40 < 20, 1.0, 0.0),
        index=price_data.index,
    )


def by_mode(price_data: pd.DataFrame, mode: str) -> pd.Series:
    """'none' never trades; 'cycle' trades every twenty bars."""
    if mode == "none":
        return pd.Series(0.0, index=price_data.index)
    return _every_twenty(price_data)


class TestRunStrategy:
    @pytest.mark.parametrize("fill_price", ["close", "next_open"])
    def test_no_trade_is_nan_not_zero(self, engine_path, bars, fill_price):
        flat = pd.Series(0.0, index=bars.index)
        result = run_strategy(bars, flat, fill_price=fill_price)
        assert result["num_trades"] == 0
        for key in AVERAGES:
            assert math.isnan(result[key]), key

    def test_the_two_backends_agree_on_the_null(self, bars, monkeypatch):
        if not engine.HAS_CPP:
            pytest.skip("the C++ extension is not built")
        flat = pd.Series(0.0, index=bars.index)
        native = run_strategy(bars, flat)
        monkeypatch.setattr(engine, "HAS_CPP", False)
        python = run_strategy(bars, flat)
        for key in AVERAGES:
            assert math.isnan(native[key]) and math.isnan(python[key]), key

    def test_null_case_a_run_that_traded_has_both(self, engine_path, bars):
        result = run_strategy(bars, _every_twenty(bars))
        assert result["num_trades"] > 0
        assert 0.0 <= result["win_rate"] <= 1.0
        assert math.isfinite(result["avg_trade_return_pct"])

    def test_the_python_trade_stats_of_an_empty_log(self):
        stats = _compute_trade_stats(pd.DataFrame())
        assert stats["num_trades"] == 0
        for key in (*AVERAGES, "profit_factor"):
            assert math.isnan(stats[key]), key


class TestTheNativeKernels:
    """The kernel itself, not the wrapper, so a C++ regression cannot hide
    behind a Python boundary."""

    @pytest.fixture
    def cpp(self):
        return pytest.importorskip("standard_quant_tools._sqt_core")

    def test_run_strategy(self, cpp, bars):
        close = bars["Close"].to_numpy(dtype=np.float64)
        out = cpp.run_strategy(close, np.zeros_like(close))
        assert out["num_trades"] == 0
        for key in AVERAGES:
            assert math.isnan(out[key]), key

    def test_batch_run_strategy(self, cpp, bars):
        close = bars["Close"].to_numpy(dtype=np.float64)
        rows = np.vstack(
            [np.zeros_like(close), _every_twenty(bars).to_numpy(dtype=np.float64)]
        )
        out = cpp.batch_run_strategy(close, rows)
        idle, traded = out
        assert idle[9] == 0 and math.isnan(idle[7]) and math.isnan(idle[10])
        assert traded[9] > 0 and math.isfinite(traded[7])
        assert math.isfinite(traded[10])

    def test_batch_backtest_crossover(self, cpp, bars):
        close = bars["Close"].to_numpy(dtype=np.float64)
        # Row 0 never rises above row 1: a crossover that never fires.
        indicators = np.vstack([close - 1e6, close])
        out = cpp.batch_backtest_crossover(
            close, indicators, np.array([[0, 1]], dtype=np.int32)
        )
        assert out[0, 9] == 0
        assert math.isnan(out[0, 7]) and math.isnan(out[0, 10])


class TestTheGrid:
    @pytest.mark.parametrize("sort_by", AVERAGES)
    def test_a_no_trade_row_is_nan_and_ranks_last(self, engine_path, bars, sort_by):
        out = backtest_grid(
            bars,
            strategy=by_mode,
            param_grid={"mode": ["none", "cycle"]},
            sort_by=sort_by,
            n_workers=1,
        )
        assert list(out["mode"]) == ["cycle", "none"]
        idle = out.iloc[-1]
        assert idle["num_trades"] == 0
        for key in AVERAGES:
            assert np.isnan(idle[key]), key
        assert out.attrs["n_unrankable"] == 1
        assert any("never traded" in w for w in out.attrs["warnings"])

    def test_the_fused_crossover_path_reports_the_null(self, engine_path, bars):
        """A slow average longer than the data never crosses: the native
        grid takes its fused crossover kernel here."""
        out = backtest_grid(
            bars,
            strategy="sma_crossover",
            param_grid={"fast_period": [5], "slow_period": [50, 400]},
            sort_by="win_rate",
            n_workers=1,
        )
        idle = out.set_index("slow_period").loc[400]
        assert idle["num_trades"] == 0
        for key in AVERAGES:
            assert np.isnan(idle[key]), key
        assert out.iloc[0]["slow_period"] == 50


class TestParameterSensitivity:
    """The aggregate a grid feeds: a trial that never traded is excluded
    and counted, under the rule the grid's own ranking follows."""

    def _grid(self):
        return pd.DataFrame(
            {
                "max_drawdown": [0.0, -0.10, -0.20, -0.30],
                "win_rate": [np.nan, 0.6, 0.5, 0.4],
                "num_trades": [0, 12, 9, 7],
            }
        )

    def test_a_no_trade_drawdown_is_not_the_best_trial(self):
        """Its 0.0 drawdown -- no position, no loss -- was `best`."""
        out = parameter_sensitivity(self._grid(), metric_col="max_drawdown")
        assert out["best"] == pytest.approx(-0.10)
        assert out["n_trials"] == 3
        assert out["n_excluded"] == 1
        assert out["median"] == pytest.approx(-0.20)

    def test_a_no_trade_win_rate_does_not_pull_the_median(self):
        out = parameter_sensitivity(self._grid(), metric_col="win_rate")
        assert out["n_trials"] == 3
        assert out["n_excluded"] == 1
        assert out["median"] == pytest.approx(0.5)

    def test_null_case_without_a_trade_count_only_non_finite_is_excluded(self):
        grid = self._grid().drop(columns="num_trades")
        out = parameter_sensitivity(grid, metric_col="max_drawdown")
        assert out["n_trials"] == 4 and out["n_excluded"] == 0
        assert out["best"] == 0.0

    def test_a_grid_that_never_traded_has_nothing_to_rank(self):
        grid = pd.DataFrame({"max_drawdown": [0.0, 0.0], "num_trades": [0, 0]})
        with pytest.raises(ValidationError, match="never traded"):
            parameter_sensitivity(grid, metric_col="max_drawdown")
