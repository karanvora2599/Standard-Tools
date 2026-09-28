"""
Every backtest annualizes by the bars' own year.

run_strategy, backtest_grid and run_signal_panel_backtest used 252 always:
the native kernel took the number as a parameter and was handed 252.0, and
the Python path used the metric functions' defaults. Weekly bars reported a
volatility sqrt(252/52) too high and monthly bars a CAGR of 38%/yr against a
true 1.6%, with no warning. Each test runs on both engine paths.
"""

import numpy as np
import pandas as pd
import pytest

import standard_quant_tools.backtest.engine as engine
from standard_quant_tools.backtest.engine import backtest_grid, run_strategy
from standard_quant_tools.backtest.panel import run_signal_panel_backtest
from standard_quant_tools.error import ValidationError


@pytest.fixture(params=["native", "python"])
def engine_path(request, monkeypatch):
    if request.param == "native":
        if not engine.HAS_CPP:
            pytest.skip("the C++ extension is not built")
    else:
        monkeypatch.setattr(engine, "HAS_CPP", False)
    return request.param


def _frame(index) -> pd.DataFrame:
    n = len(index)
    returns = np.where(np.arange(n) % 2 == 0, 0.02, -0.01)
    returns[0] = 0.0
    close = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
        index=index,
    )


@pytest.fixture(scope="module")
def monthly() -> pd.DataFrame:
    return _frame(pd.date_range("2010-01-31", periods=121, freq="ME"))


def _long(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=frame.index)


def _per_bar(result) -> pd.Series:
    return result["equity_curve"].pct_change(fill_method=None).fillna(0.0)


class TestMonthlyBarsAreAnnualizedByTwelve:
    def test_run_strategy(self, engine_path, monthly):
        result = run_strategy(
            monthly, _long(monthly), commission_pct=0.0, slippage_pct=0.0
        )
        per_bar = _per_bar(result)
        assert result["periods_per_year"] == 12
        assert result["periods_per_year_source"] == "inferred"
        assert result["annualized_volatility"] == pytest.approx(
            per_bar.std() * np.sqrt(12), abs=1e-6
        )
        assert result["sharpe_ratio"] == pytest.approx(
            per_bar.mean() / per_bar.std() * np.sqrt(12), abs=1e-4
        )
        assert not any("annualized with" in w for w in result["warnings"])

    def test_an_explicit_value_on_a_positional_index(self, engine_path, monthly):
        """The same bars with no dates: nothing to infer from, so the caller
        says it, and gets the same numbers."""
        dated = run_strategy(
            monthly, _long(monthly), commission_pct=0.0, slippage_pct=0.0
        )
        bare = monthly.reset_index(drop=True)
        explicit = run_strategy(
            bare,
            _long(bare),
            commission_pct=0.0,
            slippage_pct=0.0,
            periods_per_year=12,
        )
        assert explicit["periods_per_year_source"] == "explicit"
        for key in ("annualized_volatility", "sharpe_ratio", "calmar_ratio"):
            assert explicit[key] == pytest.approx(dated[key], rel=1e-12), key

    @pytest.mark.parametrize(
        "strategy, grid",
        [
            ("sma_crossover", {"fast_period": [3], "slow_period": [12]}),
            ("momentum_timeseries", {"lookback": [6], "threshold": [0.0]}),
        ],
        ids=["fused-crossover", "batch"],
    )
    def test_backtest_grid_agrees_with_run_strategy(
        self, engine_path, monthly, strategy, grid
    ):
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        out = backtest_grid(monthly, strategy, grid, n_workers=1)
        params = {k: v[0] for k, v in grid.items()}
        single = run_strategy(monthly, STRATEGY_REGISTRY[strategy](monthly, **params))
        assert out.attrs["periods_per_year"] == 12
        for key in ("annualized_volatility", "sharpe_ratio"):
            assert out.iloc[0][key] == pytest.approx(single[key], abs=1e-4), key


class TestDailyBarsAreUnchanged:
    def test_business_days_give_the_old_numbers_and_no_warning(self, engine_path):
        """Null case: 252 is still the answer for daily bars, and a
        holiday-gapped index still reads as daily."""
        daily = _frame(pd.date_range("2020-01-02", periods=300, freq="B").delete(40))
        signal = _long(daily)
        inferred = run_strategy(daily, signal)
        stated = run_strategy(daily, signal, periods_per_year=252)
        assert inferred["periods_per_year"] == 252
        for key in ("annualized_volatility", "sharpe_ratio", "sortino_ratio"):
            assert inferred[key] == stated[key]
        assert not any("annualized with" in w for w in inferred["warnings"])


class TestWhenTheSpacingCannotSay:
    def test_hourly_bars_warn_and_name_the_parameters(self, engine_path):
        hourly = _frame(pd.date_range("2024-01-02 09:30", periods=200, freq="h"))
        result = run_strategy(hourly, _long(hourly))
        assert result["periods_per_year"] == 252
        assert result["periods_per_year_source"] == "default"
        (warning,) = [w for w in result["warnings"] if "annualized with" in w]
        assert "periods_per_year=" in warning and "interval=" in warning

    def test_the_interval_the_bars_were_fetched_at_settles_it(self, engine_path):
        weekly = _frame(pd.date_range("2018-01-05", periods=150, freq="W-FRI"))
        result = run_strategy(weekly, _long(weekly), interval="1wk")
        assert (result["periods_per_year"], result["periods_per_year_source"]) == (
            52,
            "interval",
        )

    def test_a_bad_value_is_refused(self, engine_path, monthly):
        for bad in (0, -12, 12.5, True):
            with pytest.raises(ValidationError, match="periods_per_year"):
                run_strategy(monthly, _long(monthly), periods_per_year=bad)


class TestThePanelUsesOneYearForEveryRow:
    def test_weekly_panel(self, engine_path):
        index = pd.date_range("2018-01-05", periods=150, freq="W-FRI")
        prices = {"A": _frame(index), "B": _frame(index) * 1.5}
        signals = pd.DataFrame({"A": 1.0, "B": 1.0}, index=index)
        out = run_signal_panel_backtest(prices, signals)
        assert out["periods_per_year"] == 52
        assert all(r["periods_per_year"] == 52 for r in out["per_ticker"].values())
        port = out["portfolio_returns"]
        assert out["portfolio_metrics"]["sharpe_ratio"] == pytest.approx(
            port.mean() / port.std() * np.sqrt(52), abs=1e-4
        )
