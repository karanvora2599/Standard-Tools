"""
What every backtest entry point refuses before it computes anything.

A reversed frame was backtested backwards (+9.9% became -19.8%, with no
warning) on both engine paths and in backtest_grid. A repeated bar raised a
bare pandas ValueError on the native path, and on the Python path and in
backtest_grid it was silently counted twice. A NaN risk-free rate was
refused by one path and turned into a NaN Sharpe beside a +inf Sortino by an
older build of the other. Each is now a ValidationError naming the remedy,
on both paths.
"""

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.engine as engine
from standard_quant_tools.backtest.engine import backtest_grid, run_strategy
from standard_quant_tools.backtest.panel import run_signal_panel_backtest
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY
from standard_quant_tools.error import ValidationError


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
    rng = np.random.default_rng(5)
    dates = pd.date_range("2022-01-03", periods=300, freq="B")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.012, len(dates)))
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
        index=dates,
    )


def _sma(frame):
    return STRATEGY_REGISTRY["sma_crossover"](frame, fast_period=10, slow_period=50)


GRID = {"fast_period": [10], "slow_period": [50]}


class TestAnUnsortedFrameIsRefused:
    def test_run_strategy(self, engine_path, bars):
        reversed_bars = bars.iloc[::-1]
        with pytest.raises(ValidationError, match="not sorted"):
            run_strategy(reversed_bars, _sma(reversed_bars))

    def test_backtest_grid(self, engine_path, bars):
        with pytest.raises(ValidationError, match="not sorted"):
            backtest_grid(bars.iloc[::-1], "sma_crossover", GRID, n_workers=1)

    def test_a_sorted_frame_runs_as_before(self, engine_path, bars):
        """Null case."""
        result = run_strategy(bars, _sma(bars))
        grid = backtest_grid(bars, "sma_crossover", GRID, n_workers=1)
        assert grid.iloc[0]["total_return"] == pytest.approx(
            result["total_return"], abs=1e-6
        )

    def test_a_signal_is_read_onto_the_bars_by_date(self, engine_path, bars):
        """Only the PRICE index sets bar order. A signal whose rows are out of
        order is aligned by label and gives the same answer, so it is not
        refused."""
        signal = _sma(bars)
        shuffled = signal.sample(frac=1.0, random_state=0)
        assert run_strategy(bars, shuffled)["total_return"] == pytest.approx(
            run_strategy(bars, signal)["total_return"], abs=1e-12
        )


class TestARepeatedBarIsRefused:
    def _duplicated(self, bars):
        return pd.concat([bars.iloc[:100], bars.iloc[99:100], bars.iloc[100:]])

    def test_run_strategy_names_the_date(self, engine_path, bars):
        doubled = self._duplicated(bars)
        with pytest.raises(ValidationError, match=str(bars.index[99].date())):
            run_strategy(doubled, pd.Series(1.0, index=doubled.index))

    def test_a_repeated_signal_date(self, engine_path, bars):
        signal = _sma(bars)
        doubled = pd.concat([signal.iloc[:100], signal.iloc[99:100], signal.iloc[100:]])
        with pytest.raises(ValidationError, match="signal_series.index has duplicate"):
            run_strategy(bars, doubled)

    def test_backtest_grid(self, engine_path, bars):
        with pytest.raises(ValidationError, match="duplicate"):
            backtest_grid(self._duplicated(bars), "sma_crossover", GRID, n_workers=1)

    def test_the_signal_panel(self, engine_path, bars):
        doubled = self._duplicated(bars)
        with pytest.raises(ValidationError, match="price_data\\['A'\\]"):
            run_signal_panel_backtest(
                {"A": doubled}, pd.DataFrame({"A": 1.0}, index=bars.index)
            )
        panel = pd.DataFrame({"A": 1.0}, index=doubled.index)
        with pytest.raises(ValidationError, match="signal_panel.index"):
            run_signal_panel_backtest({"A": bars}, panel)


class TestTheRateIsCheckedUpFront:
    @pytest.mark.parametrize("rate", [float("nan"), float("inf"), -float("inf")])
    def test_a_non_finite_rate_is_refused_on_both_paths(self, engine_path, bars, rate):
        with pytest.raises(ValidationError, match="risk_free_rate"):
            run_strategy(bars, _sma(bars), risk_free_rate=rate)
        with pytest.raises(ValidationError, match="risk_free_rate"):
            backtest_grid(bars, "sma_crossover", GRID, risk_free_rate=rate)

    def test_a_negative_rate_is_allowed(self, engine_path, bars):
        """Null case: negative policy rates are real."""
        result = run_strategy(bars, _sma(bars), risk_free_rate=-0.005)
        assert np.isfinite(result["sharpe_ratio"])
