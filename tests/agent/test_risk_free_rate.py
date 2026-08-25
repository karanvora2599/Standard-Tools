"""
The risk-free rate, end to end.

Seventeen tools report a Sharpe ratio and exactly one took a rate, so the
rest measured total return per unit of risk and called it Sharpe. At a 4-5%
short rate that is not a rounding difference — for a low-volatility
strategy it is most of the number.

It is threaded now: through the Python engine, through the C++ kernel, and
through all three of `backtest_grid`'s execution paths. That creates the
failure mode these tests exist to prevent — the two execution paths
disagreeing only when a rate is set, which no zero-rate test would ever
catch. So the parity checks below run at SEVERAL rates, and they compare
the native path against the pure-Python fallback rather than against
recorded numbers.

The subtle one is Sortino. Python clips EXCESS returns, so the rate moves
the denominator as well as the numerator, and in the kernel's
allocation-free summary path bar 0's implicit strat_ret of 0.0 has an
excess of -rf/ppy that contributes to the downside sum. That term is
invisible at rf = 0 and is exactly what would make a grid disagree with a
single run.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.agent.models import (
    AnalysisInput,
    BacktestInput,
    PortfolioInput,
    RiskAttributionInput,
)
from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.tools import _TOOL_DISPATCH
from standard_quant_tools.backtest import engine
from standard_quant_tools.metrics.risk_metrics import sharpe_ratio, sortino_ratio

RATES = [0.0, 0.01, 0.02, 0.045, 0.10]


@pytest.fixture
def prices():
    """Drift plus real downside, so Sortino is finite and the rate has
    somewhere to move both ratios."""
    rng = np.random.default_rng(5)
    n = 400
    close = 100.0 * np.exp(np.linspace(0, 0.30, n) + rng.normal(0, 0.006, n).cumsum())
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.004,
            "Low": close * 0.996,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=pd.bdate_range("2022-01-03", periods=n),
    )


@pytest.fixture
def signals(prices):
    rng = np.random.default_rng(9)
    return pd.Series((rng.random(len(prices)) > 0.5).astype(float), index=prices.index)


def _python_path(*args, **kwargs):
    """run_strategy with the native path forced off."""
    real, engine.HAS_CPP = engine.HAS_CPP, False
    try:
        return engine.run_strategy(*args, **kwargs)
    finally:
        engine.HAS_CPP = real


class TestTheEngineHonoursTheRate:
    def test_a_higher_rate_lowers_the_sharpe(self, prices, signals):
        ratios = [
            engine.run_strategy(prices, signals, risk_free_rate=rf)["sharpe_ratio"]
            for rf in RATES
        ]
        assert ratios == sorted(ratios, reverse=True)
        assert ratios[0] != ratios[-1]

    def test_a_higher_rate_lowers_the_sortino(self, prices, signals):
        ratios = [
            engine.run_strategy(prices, signals, risk_free_rate=rf)["sortino_ratio"]
            for rf in RATES
        ]
        assert ratios == sorted(ratios, reverse=True)

    def test_the_rate_does_not_touch_return_or_drawdown(self, prices, signals):
        """It is a scoring convention, not a cash flow. A rate that changed
        the equity curve would be modelling a cash balance nobody asked
        for."""
        base = engine.run_strategy(prices, signals, risk_free_rate=0.0)
        high = engine.run_strategy(prices, signals, risk_free_rate=0.10)
        for field in ("total_return", "max_drawdown", "final_equity", "num_trades"):
            assert base[field] == high[field], field

    def test_zero_is_exactly_what_was_reported_before(self, prices, signals):
        implicit = engine.run_strategy(prices, signals)
        explicit = engine.run_strategy(prices, signals, risk_free_rate=0.0)
        assert implicit["sharpe_ratio"] == explicit["sharpe_ratio"]
        assert implicit["sortino_ratio"] == explicit["sortino_ratio"]


class TestTheTwoExecutionPathsAgree:
    """The failure this whole change could have introduced: a rate honoured
    in one path and not the other, so the answer depends on whether the
    extension happens to be built."""

    @pytest.mark.parametrize("rf", RATES)
    def test_sharpe_matches_across_paths(self, prices, signals, rf):
        native = engine.run_strategy(prices, signals, risk_free_rate=rf)
        python = _python_path(prices, signals, risk_free_rate=rf)
        assert native["sharpe_ratio"] == pytest.approx(python["sharpe_ratio"], abs=1e-9)

    @pytest.mark.parametrize("rf", RATES)
    def test_sortino_matches_across_paths(self, prices, signals, rf):
        """The one with a denominator that moves. Python clips EXCESS
        returns, so a kernel that clipped raw ones would agree at rf=0 and
        diverge everywhere else."""
        native = engine.run_strategy(prices, signals, risk_free_rate=rf)
        python = _python_path(prices, signals, risk_free_rate=rf)
        assert native["sortino_ratio"] == pytest.approx(
            python["sortino_ratio"], abs=1e-9
        )

    @pytest.mark.parametrize("rf", RATES)
    def test_both_paths_match_the_metric_functions(self, prices, signals, rf):
        """Neither path is allowed to invent its own definition: both must
        equal metrics/risk_metrics.py applied to the equity curve."""
        result = engine.run_strategy(prices, signals, risk_free_rate=rf)
        curve = result["equity_curve"]
        returns = curve.pct_change().fillna(0.0)
        returns.iloc[0] = 0.0
        assert result["sharpe_ratio"] == pytest.approx(
            round(sharpe_ratio(returns, rf), 4), abs=1e-4
        )
        assert result["sortino_ratio"] == pytest.approx(
            round(sortino_ratio(returns, rf), 4), abs=1e-4
        )


class TestTheGridHonoursTheRate:
    """A grid that ranked on a zero-rate Sharpe while the single run it is
    compared against used a real one would pick a different winner, and
    nothing in either result would say why."""

    @pytest.mark.parametrize("rf", [0.0, 0.05])
    def test_the_fused_crossover_path_agrees_with_a_single_run(self, prices, rf):
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        params = {"fast_period": 5, "slow_period": 20}
        grid = engine.backtest_grid(
            prices,
            "sma_crossover",
            {"fast_period": [5], "slow_period": [20]},
            risk_free_rate=rf,
            n_workers=1,
        )
        single = engine.run_strategy(
            prices,
            STRATEGY_REGISTRY["sma_crossover"](prices, **params),
            risk_free_rate=rf,
        )
        assert grid.iloc[0]["sharpe_ratio"] == pytest.approx(
            single["sharpe_ratio"], abs=1e-4
        )

    @pytest.mark.parametrize("rf", [0.0, 0.05])
    def test_the_batch_path_agrees_with_a_single_run(self, prices, rf):
        """A non-crossover strategy skips the fused path and goes through
        batch_run_strategy, which is a different kernel function."""
        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        params = {"period": 14, "oversold": 30, "overbought": 70}
        grid = engine.backtest_grid(
            prices,
            "rsi_mean_reversion",
            {"period": [14], "oversold": [30], "overbought": [70]},
            risk_free_rate=rf,
            n_workers=1,
        )
        single = engine.run_strategy(
            prices,
            STRATEGY_REGISTRY["rsi_mean_reversion"](prices, **params),
            risk_free_rate=rf,
        )
        assert grid.iloc[0]["sharpe_ratio"] == pytest.approx(
            single["sharpe_ratio"], abs=1e-4
        )

    def test_a_rate_can_reorder_the_grid(self, prices):
        """Not cosmetic: the rate penalizes low-return strategies hardest,
        so the winner genuinely can change — which is why a grid ranking on
        the wrong rate was a real problem and not a display issue."""
        grid = {"fast_period": [3, 10, 30], "slow_period": [40, 80]}
        at_zero = engine.backtest_grid(
            prices, "sma_crossover", grid, risk_free_rate=0.0, n_workers=1
        )
        at_high = engine.backtest_grid(
            prices, "sma_crossover", grid, risk_free_rate=0.20, n_workers=1
        )
        assert at_zero.iloc[0]["sharpe_ratio"] > at_high.iloc[0]["sharpe_ratio"]


class TestTheToolSurfaceExposesIt:
    def test_every_sharpe_reporting_tool_takes_a_rate(self):
        """The invariant that replaced the old one. Previously fourteen
        tools reported a Sharpe they could not qualify; a tool that reports
        one now must let a caller say what it is measured against."""
        import inspect

        offenders = []
        for name, (fn, model) in _TOOL_DISPATCH.items():
            annotation = inspect.signature(fn).return_annotation
            fields = set(getattr(annotation, "model_fields", {}) or {})
            if any("sharpe" in f for f in fields):
                if "risk_free_rate" not in model.model_fields:
                    offenders.append(name)
        assert not offenders, (
            f"these tools report a Sharpe with no way to set the rate it is "
            f"measured against: {offenders}"
        )

    def test_every_sharpe_rate_field_defaults_to_zero(self):
        """Zero is what every one of these reported before, so a different
        default would silently restate published numbers.

        Scoped to the Sharpe-reporting tools on purpose. The option-pricing
        tools carry a field of the same NAME that is a different quantity —
        the Black-Scholes discount rate — and it is deliberately REQUIRED,
        because there is no defensible default for discounting a cash flow.
        A blanket assertion over the name would force one of the two to be
        wrong."""
        import inspect

        for _name, (fn, model) in _TOOL_DISPATCH.items():
            annotation = inspect.signature(fn).return_annotation
            fields = set(getattr(annotation, "model_fields", {}) or {})
            if not any("sharpe" in f for f in fields):
                continue
            field = model.model_fields["risk_free_rate"]
            assert field.default == 0.0, model.__name__

    def test_the_option_pricing_rate_stays_required(self):
        """Guarding the distinction above: if this ever gains a default,
        the test that scopes by Sharpe would silently start covering a
        field it does not describe."""
        from standard_quant_tools.agent.models import OptionPricingInput

        assert OptionPricingInput.model_fields["risk_free_rate"].is_required()

    def test_a_backtest_tool_actually_honours_it(self, monkeypatch, prices):
        class _Stub:
            def get_ohlcv(self, *a, **k):
                return prices

        monkeypatch.setattr(
            "standard_quant_tools.agent.runtimes.backtest.tools."
            "DataFactory.get_provider",
            staticmethod(lambda *a, **k: _Stub()),
        )
        backtest = resolve("backtest")
        base = {
            "symbol": "TEST",
            "start_date": "2022-01-01",
            "end_date": "2023-08-01",
            "strategy_type": "sma_crossover",
        }
        at_zero = backtest.dispatch("run_sma_backtest", base)
        at_five = backtest.dispatch(
            "run_sma_backtest", {**base, "risk_free_rate": 0.05}
        )
        assert at_five["sharpe_ratio"] < at_zero["sharpe_ratio"]
        # ...and the trading itself is unchanged.
        assert at_five["total_return"] == at_zero["total_return"]

    def test_the_three_analysis_inputs_still_carry_it(self):
        for model in (AnalysisInput, PortfolioInput, RiskAttributionInput):
            assert "risk_free_rate" in model.model_fields, model.__name__

    def test_a_negative_or_absurd_rate_is_rejected(self):
        for bad in (-0.01, 1.5):
            with pytest.raises(Exception):
                BacktestInput(
                    symbol="X",
                    start_date="2022-01-01",
                    end_date="2023-01-01",
                    strategy_type="sma_crossover",
                    risk_free_rate=bad,
                )
