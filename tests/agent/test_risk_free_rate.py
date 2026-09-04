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

from typing import Any, Optional, get_args

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

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


def _reports_a_sharpe(model: Any, _depth: int = 0, _seen: Optional[set] = None) -> bool:
    """Does this result model expose a Sharpe ANYWHERE, nesting included?

    The predicate this replaces looked only at a result's own field names,
    and five tools carry their Sharpe one level down -- inside a nested
    `BacktestResult`, a list of per-strategy rows, or a per-ticker mapping.
    `RegimeAdaptiveResult`'s own fields are symbol/regime/hurst/.../backtest;
    not one of them contains "sharpe", so the sweep that fixed seventeen
    tools never saw it, and `run_regime_adaptive_backtest` went on reporting
    a Sharpe measured against 0% while carrying a `risk_free_rate` field
    that the structural test was satisfied to find.

    Two of the five had no rate field at all.
    """
    if _seen is None:
        _seen = set()
    if model in _seen or _depth > 4:
        return False
    _seen.add(model)
    for field_name, field in (getattr(model, "model_fields", {}) or {}).items():
        if "sharpe" in field_name:
            return True
        annotation = field.annotation
        # Unwrap one level of Optional/List/Dict to reach the model inside.
        candidates = (annotation,) + tuple(get_args(annotation) or ())
        for candidate in candidates:
            inner = candidate
            for sub in get_args(candidate) or ():
                if isinstance(sub, type) and issubclass(sub, BaseModel):
                    inner = sub
            if isinstance(inner, type) and issubclass(inner, BaseModel):
                if _reports_a_sharpe(inner, _depth + 1, _seen):
                    return True
    return False


def _sharpe_reporting_tools():
    """(tool name, input model) for every tool that reports a Sharpe."""
    import inspect

    for name, (fn, model) in _TOOL_DISPATCH.items():
        annotation = inspect.signature(fn).return_annotation
        if _reports_a_sharpe(annotation):
            yield name, model


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
        one now must let a caller say what it is measured against.

        Scoped by the RECURSIVE predicate, which is the point: the shallow
        one missed five tools whose Sharpe sits one level down, and two of
        those five had no rate field at all."""
        offenders = [
            name
            for name, model in _sharpe_reporting_tools()
            if "risk_free_rate" not in model.model_fields
        ]
        assert not offenders, (
            f"these tools report a Sharpe with no way to set the rate it is "
            f"measured against: {offenders}"
        )

    def test_the_predicate_actually_sees_through_nesting(self):
        """Guarding the guard. If this regresses to a shallow field scan the
        test above passes vacuously for exactly the tools that broke."""
        from standard_quant_tools.agent.models import (
            CompareStrategiesResult,
            RegimeAdaptiveResult,
        )

        # No field of its own contains "sharpe" -- it is inside `backtest`.
        assert not any("sharpe" in f for f in RegimeAdaptiveResult.model_fields)
        assert _reports_a_sharpe(RegimeAdaptiveResult)
        # And through a List[StrategyComparison].
        assert not any("sharpe" in f for f in CompareStrategiesResult.model_fields)
        assert _reports_a_sharpe(CompareStrategiesResult)

    def test_nesting_does_not_make_everything_a_sharpe_reporter(self):
        """The recursion must not be so eager that the invariant becomes
        trivially true for every tool on the surface."""
        names = {name for name, _ in _sharpe_reporting_tools()}
        assert len(names) < len(_TOOL_DISPATCH) / 2, (
            f"{len(names)} of {len(_TOOL_DISPATCH)} tools counted as Sharpe "
            f"reporters -- the predicate is over-matching"
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
        for _name, model in _sharpe_reporting_tools():
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


class TestTheNestedSharpeToolsHonourItInFact:
    """Structure is not behaviour.

    `test_every_sharpe_reporting_tool_takes_a_rate` asserts the FIELD
    exists. `RegimeAdaptiveInput` had it, and the tool dropped it anyway --
    rebuilding `BacktestInput` field by field sixteen lines after handing
    the same rate to `backtest_grid`, so one call SELECTED parameters on the
    caller's rate and REPORTED a Sharpe measured against zero.

    These tools are the ones whose Sharpe is nested, which is why the
    original sweep never reached them. The assertion is the general one the
    plan names: vary one input, assert the output moves.
    """

    #: Keys that identify a row, tried in order. A list POSITION is not an
    #: identity here: `compare_strategies` sorts by `sort_by`, which defaults
    #: to sharpe_ratio, so raising the rate reorders the list and
    #: `strategies[1]` is a different strategy in the two runs. Comparing
    #: positionally reports that reordering as a leak into the trading.
    _IDENTITY_KEYS = ("strategy", "strategy_type", "name", "ticker", "symbol")

    @classmethod
    def _walk(cls, obj, leaf, path=""):
        """(stable path, value) for every `leaf` field, keyed by identity."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == leaf and isinstance(value, (int, float)):
                    yield f"{path}.{key}", value
                else:
                    yield from cls._walk(value, leaf, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                tag = f"[{i}]"
                if isinstance(value, dict):
                    for id_key in cls._IDENTITY_KEYS:
                        if isinstance(value.get(id_key), str):
                            tag = f"[{value[id_key]}]"
                            break
                yield from cls._walk(value, leaf, f"{path}{tag}")

    @classmethod
    def _sharpes(cls, obj, path=""):
        return cls._walk(obj, "sharpe_ratio", path)

    @pytest.fixture
    def backtest_runtime(self, monkeypatch, prices):
        class _Stub:
            def get_ohlcv(self, *a, **k):
                return prices

            async def get_ohlcv_async(self, *a, **k):
                return prices

        monkeypatch.setattr(
            "standard_quant_tools.agent.runtimes.backtest.tools."
            "DataFactory.get_provider",
            staticmethod(lambda *a, **k: _Stub()),
        )
        return resolve("backtest")

    @pytest.fixture
    def panel_signal(self, prices):
        rng = np.random.default_rng(11)
        values = (rng.random(len(prices)) > 0.5).astype(float)
        return {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(prices.index, values)}

    def _cases(self, panel_signal):
        window = {"start_date": "2022-01-01", "end_date": "2023-08-01"}
        return {
            "run_regime_adaptive_backtest": {"symbol": "TEST", **window},
            "run_backtest_compact": {
                "symbol": "TEST",
                "strategy_type": "sma_crossover",
                **window,
            },
            "run_strategy_matrix": {
                "tickers": ["TEST"],
                "strategies": ["sma_crossover"],
                **window,
            },
            "compare_strategies": {"symbol": "TEST", **window},
            "run_signal_panel_backtest": {
                "tickers": ["TEST"],
                "signal_panel": {"TEST": panel_signal},
                **window,
            },
        }

    def test_every_reported_sharpe_moves_with_the_rate(
        self, backtest_runtime, panel_signal
    ):
        inert = []
        for tool, kwargs in self._cases(panel_signal).items():
            at_zero = dict(self._sharpes(backtest_runtime.dispatch(tool, dict(kwargs))))
            at_five = dict(
                self._sharpes(
                    backtest_runtime.dispatch(tool, {**kwargs, "risk_free_rate": 0.05})
                )
            )
            assert at_zero, f"{tool} reported no Sharpe at all"
            for field, value in at_zero.items():
                if abs(value - at_five[field]) <= 1e-9:
                    inert.append(f"{tool}{field}")
        assert not inert, (
            f"these reported Sharpes did not move when the rate went from 0% "
            f"to 5%, so the rate is being dropped somewhere: {inert}"
        )

    def test_the_rate_does_not_change_the_trading(self, backtest_runtime, panel_signal):
        """The counterpart assertion. A rate that moved total_return would
        mean it had leaked into the fills rather than the ratio."""
        for tool, kwargs in self._cases(panel_signal).items():
            at_zero = backtest_runtime.dispatch(tool, dict(kwargs))
            at_five = backtest_runtime.dispatch(
                tool, {**kwargs, "risk_free_rate": 0.05}
            )

            before = dict(self._walk(at_zero, "total_return"))
            after = dict(self._walk(at_five, "total_return"))
            assert set(before) == set(
                after
            ), f"{tool} reported a different set of rows at the two rates"
            for field, value in before.items():
                assert value == pytest.approx(after[field], abs=1e-9), (
                    f"{tool}{field} changed with the risk-free rate -- it has "
                    f"leaked into the trading, not just the ratio"
                )
