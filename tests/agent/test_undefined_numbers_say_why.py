"""
A number a tool cannot compute is null, and the result says why.

JSON has no NaN or infinity, and a result field typed as a plain float
carried one anyway: the typed result held the NaN, the audit record hashed
it, and only on the way out did `sanitize_for_json` turn it into a null with
nothing beside it -- so "this ratio is 0/0 on a range of one bar" and "this
was never computed" looked the same to a caller. The fields below are typed
`Stat` (or a finite-or-None element type) now, and a result built on
`ExplainsNulls` writes one line per null into its own `warnings` (or
`notes`) naming the field and the reason.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest
from pydantic import ConfigDict
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent import models as M
from standard_quant_tools.agent.runtimes._json_safe import (
    ExplainsNulls,
    finite_or_none,
)
from standard_quant_tools.agent.runtimes.delta_one import results as D
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.modeling.agent import models as MM
from standard_quant_tools.modeling.agent import portfolio_models as PM

from ..surface import synth

NAN, INF = float("nan"), float("inf")


def _valid(model: type, **overrides: Any) -> Any:
    """A valid instance of `model` with `overrides` applied at construction."""
    built = synth.build(model, length=20)
    return model(**{**built.model_dump(), **overrides})


def _notes(result: Any) -> List[str]:
    return list(getattr(result, "warnings", None) or getattr(result, "notes", []))


def _null_lines(result: Any) -> List[str]:
    return [line for line in _notes(result) if " is null" in line]


# ── the mechanism ───────────────────────────────────────────────────────


class _Row(ExplainsNulls):
    null_reasons = {"ratio": ("it is 0/0 here", "it divides by zero here")}
    ratio: M.Stat
    other: M.Stat = None


class _Reporting(ExplainsNulls):
    null_reasons = {
        "total": "the total is undefined",
        "curve": "the curve is undefined",
    }
    total: M.Stat = None
    metrics: Dict[str, M.Stat] = {}
    curve: List[M.Stat] = []
    rows: List[_Row] = []
    warnings: List[str] = []


class _Open(ExplainsNulls):
    model_config = ConfigDict(extra="allow")
    warnings: List[str] = []


class TestTheMechanism:
    def test_a_nan_is_null_with_its_reason(self):
        result = _Reporting(total=NAN)
        assert result.total is None
        assert result.warnings == ["total is null: the total is undefined."]

    def test_an_infinity_takes_the_unbounded_reason(self):
        result = _Reporting(rows=[{"ratio": INF}])
        assert result.rows[0].ratio is None
        assert result.warnings == ["rows[0].ratio is null: it divides by zero here."]

    def test_a_field_without_a_reason_still_says_it_is_null(self):
        result = _Reporting(rows=[{"ratio": 1.0, "other": NAN}])
        assert result.warnings == [
            "rows[0].other is null: it is not a number for this input."
        ]

    def test_a_mapping_entry_is_named_and_many_are_counted(self):
        result = _Reporting(metrics={"a": NAN, "b": 1.0}, curve=[NAN] * 5 + [1.0])
        assert result.metrics == {"a": None, "b": 1.0}
        assert result.curve[:5] == [None] * 5
        assert (
            "metrics.a is null: it is not a number for this input." in result.warnings
        )
        assert (
            "curve[*] is null in 5 places: the curve is undefined." in result.warnings
        )

    def test_a_required_number_that_arrives_null_is_explained(self):
        """A row copied from a result that already nulled it: the origin is
        unknown, so both reasons are given."""
        result = _Reporting(rows=[_Row(ratio=None)])
        assert result.warnings == [
            "rows[0].ratio is null: it is 0/0 here; or it divides by zero here."
        ]

    def test_an_undeclared_key_is_nulled_and_said_to_be(self):
        result = _Open(surprise=INF, fine=2.0)
        assert result.model_dump()["surprise"] is None
        assert result.model_dump()["fine"] == 2.0
        assert result.warnings == ["surprise is null: it is infinite for this input."]

    def test_rebuilding_from_its_own_dump_adds_nothing(self):
        result = _Reporting(total=NAN, rows=[{"ratio": NAN}])
        again = _Reporting.model_validate(result.model_dump())
        assert again.warnings == result.warnings

    def test_finite_numbers_say_nothing(self):
        """The null case."""
        result = _Reporting(
            total=1.0, metrics={"a": 2.0}, curve=[1.0], rows=[{"ratio": 0.5}]
        )
        assert result.warnings == []

    def test_a_numpy_float32_nan_is_caught(self):
        """It is not a `float` subclass, so the old isinstance test let it
        through; integers and booleans still pass untouched."""
        assert finite_or_none(np.float32("nan")) is None
        assert finite_or_none(np.float64("inf")) is None
        assert finite_or_none(np.int64(3)) == 3
        assert finite_or_none(True) is True
        assert finite_or_none("NaN") == "NaN"


# ── backtest ratios ─────────────────────────────────────────────────────


def _bars(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.001,
            "Low": close * 0.999,
            "Close": close,
            "Volume": 1_000_000.0,
        }
    )


class TestBacktestRatios:
    def _run(self, close: pd.Series, signals: pd.Series) -> M.BacktestResult:
        from standard_quant_tools.agent.runtimes import _shared

        inp = M.BacktestInput(
            symbol="AAA",
            start_date="2023-01-02",
            end_date="2023-12-29",
            strategy_type="sma_crossover",
            fill_price="next_open",
        )
        return _shared._run_backtest(inp, _bars(close), signals)

    def test_a_strategy_that_never_moved_has_four_null_ratios(self):
        """A flat market and no position: every ratio is 0/0. They were NaN
        in the typed result and in the audit record's output hash."""
        index = pd.bdate_range("2023-01-02", periods=120)
        result = self._run(pd.Series(100.0, index=index), pd.Series(0.0, index=index))
        assert result.sharpe_ratio is None
        assert result.sortino_ratio is None
        assert result.calmar_ratio is None
        assert result.profit_factor is None
        lines = _null_lines(result)
        assert any(
            line.startswith("sharpe_ratio is null: the returns have no dispersion")
            for line in lines
        )
        assert (
            "sortino_ratio is null: no return differed from the risk-free rate, so the ratio is 0/0."
            in lines
        )
        assert (
            "calmar_ratio is null: the equity curve neither grew nor drew down, so the ratio is 0/0."
            in lines
        )
        assert (
            "profit_factor is null: no trade closed, so there is no gross profit or loss to divide."
            in lines
        )

    def test_a_strategy_that_traded_has_numbers_and_no_null_lines(self):
        """The null case."""
        index = pd.bdate_range("2023-01-02", periods=120)
        rng = np.random.default_rng(3)
        close = pd.Series(
            100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index)))), index=index
        )
        signals = pd.Series(
            np.where(np.arange(len(index)) % 20 < 10, 1.0, 0.0), index=index
        )
        result = self._run(close, signals)
        assert all(
            isinstance(v, float) and math.isfinite(v)
            for v in (
                result.sharpe_ratio,
                result.sortino_ratio,
                result.calmar_ratio,
                result.profit_factor,
            )
        )
        assert _null_lines(result) == []

    def test_a_regime_adaptive_walk_forward_on_a_flat_market_is_null_not_an_error(
        self, monkeypatch
    ):
        """Every out-of-sample window is 0/0, so their mean is undefined too:
        null with a reason, where the NaN once reached the payload."""
        from standard_quant_tools.agent.runtimes.backtest import tools as T

        index = pd.bdate_range("2020-01-02", periods=500)
        flat = _bars(pd.Series(100.0, index=index))
        provider = MagicMock()
        provider.get_ohlcv.side_effect = lambda *a, **k: flat
        provider.get_ohlcv_async = AsyncMock(side_effect=lambda *a, **k: flat)
        monkeypatch.setattr(DataFactory, "get_provider", lambda *a, **k: provider)
        result = T.run_regime_adaptive_walkforward_backtest(
            M.RegimeAdaptiveWalkForwardInput(
                symbol="AAA",
                start_date="2020-01-02",
                end_date="2021-12-01",
                train_bars=200,
                test_bars=50,
            )
        )
        assert result.windows and all(
            w.out_of_sample_sharpe is None for w in result.windows
        )
        assert result.avg_oos_sharpe is None
        assert any(
            line.startswith("avg_oos_sharpe is null: at least one window")
            for line in result.warnings
        )


# ── every other result model in scope ───────────────────────────────────

#: (model, field, value, a phrase the warning must contain)
SCALARS = [
    (
        M.CorrelationAnalysisResult,
        "avg_pairwise_correlation",
        NAN,
        "at least one pairwise correlation",
    ),
    (
        M.PartialCorrelationResult,
        "partial_correlation",
        NAN,
        "nothing of x or y is left",
    ),
    (M.RallyDetectionResult, "adx", NAN, "never finished its warm-up"),
    (M.RallyDetectionResult, "adx_threshold_used", INF, "infinite for this input"),
    (M.VolatilityEstimatorsResult, "parkinson_annualized", NAN, "no complete window"),
    (M.CapacityReportResult, "max_account_size", NAN, "NOT unbounded"),
    (M.EstimateTradeCostResult, "breakeven_move_bps", INF, "infinite for this input"),
    (M.PositionSizerResult, "stop_distance", INF, "infinite for this input"),
    (M.PortfolioSimulationResult, "final_equity", NAN, "stopped being a finite number"),
    (
        M.RobustnessDiagnosticsResult,
        "deflated_sharpe_ratio",
        NAN,
        "not a number for this input",
    ),
    (M.CompareCostModelsResult, "gross_sharpe_ratio", NAN, "no dispersion"),
    (M.PairTradeBacktestResult, "calmar_ratio", INF, "never drew down"),
    (M.BacktestDiagnosticsResult, "sortino_ratio", NAN, "0/0"),
    (M.BacktestOptResult, "best_sharpe", NAN, "no defined Sharpe ratio"),
    (M.RegimeAdaptiveWalkForwardResult, "avg_oos_sharpe", NAN, "at least one window"),
    (M.PlanRebalanceResult, "total_turnover", INF, "no schedule toward it is defined"),
]


@pytest.mark.parametrize(
    "model,field,value,phrase",
    SCALARS,
    ids=[f"{m.__name__}.{f}" for m, f, _v, _p in SCALARS],
)
def test_a_non_finite_field_is_null_and_explained(model, field, value, phrase):
    result = _valid(model, **{field: value})
    assert getattr(result, field) is None
    assert any(
        line.startswith(f"{field} is null") and phrase in line
        for line in _notes(result)
    ), _notes(result)


@pytest.mark.parametrize(
    "model,field,value,phrase",
    SCALARS[:4],
    ids=[f"{m.__name__}.{f}" for m, f, _v, _p in SCALARS[:4]],
)
def test_a_finite_field_is_kept_and_unexplained(model, field, value, phrase):
    """The null case."""
    result = _valid(model, **{field: 0.25})
    assert getattr(result, field) == 0.25
    assert not any(line.startswith(f"{field} is null") for line in _notes(result))


class TestNumbersInsideMappingsAndRows:
    def test_signal_panel_metrics(self):
        result = _valid(
            M.SignalPanelBacktestResult,
            portfolio_metrics={"sortino_ratio": NAN, "n": 3},
        )
        assert result.portfolio_metrics == {"sortino_ratio": None, "n": 3}
        assert (
            "portfolio_metrics.sortino_ratio is null: no return differed"
            in " ".join(result.warnings)
        )

    def test_a_correlation_matrix(self):
        matrix = {a: {b: (1.0 if a == b else NAN) for b in "XYZ"} for a in "XYZ"}
        result = _valid(M.CorrelationAnalysisResult, correlation_matrix=matrix)
        assert result.correlation_matrix["X"]["Y"] is None
        assert any(
            line.startswith(
                "correlation_matrix.*.* is null in 6 places: a series in the pair did not move"
            )
            for line in result.warnings
        )

    def test_an_equity_curve(self):
        result = _valid(M.PortfolioSimulationResult, equity_curve=[1.0, NAN, NAN])
        assert result.equity_curve == [1.0, None, None]
        assert any(
            line.startswith("equity_curve[1], equity_curve[2] are null")
            for line in result.warnings
        )

    def test_frontier_weights_are_reported_by_the_frontier(self):
        point = _valid(M.FrontierPoint, weights={"AAA": NAN, "BBB": 1.0})
        assert point.weights["AAA"] is None
        result = _valid(M.EfficientFrontierResult, tangency=point)
        assert any(
            line.startswith("tangency.weights.AAA is null") for line in result.warnings
        )

    def test_rebalance_rows_are_reported_by_the_plan(self):
        step = _valid(M.RebalanceStep, turnover=INF, weights={"AAA": NAN})
        result = _valid(M.PlanRebalanceResult, schedule=[step.model_dump(), step])
        text = " ".join(result.warnings)
        assert "schedule[0].turnover" in text and "schedule[1].weights.AAA" in text

    def test_a_comparison_row_copied_from_a_nulled_backtest(self):
        """compare_strategies copies each backtest's Sortino; a null one is
        explained by the comparison, which is the result the caller reads."""
        row = _valid(M.StrategyComparison, sortino_ratio=None)
        result = _valid(M.CompareStrategiesResult, strategies=[row])
        assert any(
            line.startswith("strategies[0].sortino_ratio is null")
            for line in result.warnings
        )

    def test_a_cost_scenario_row(self):
        scenario = _valid(M.CostScenarioResult, sharpe_ratio=NAN)
        result = _valid(M.CompareCostModelsResult, scenarios=[scenario])
        assert any(
            line.startswith("scenarios[0].sharpe_ratio is null")
            for line in result.notes
        )


# ── delta one: every key declared ───────────────────────────────────────


class TestDeltaOneKeysAreDeclared:
    def test_the_half_life_statistics_are_declared_and_explained(self):
        result = D.BasisHistoryResult(
            half_life_t_statistic=NAN, half_life_critical_value=NAN, basis_flat=True
        )
        assert result.half_life_t_statistic is None
        assert result.basis_flat is True
        assert any(
            line.startswith("half_life_t_statistic is null: the basis did not vary")
            for line in result.warnings
        )

    @pytest.mark.parametrize(
        "model,field",
        [
            (D.CashFuturesBasisResult, "carry_spread_rate"),
            (D.EtfFairValueResult, "premium_vs_reference_bps"),
        ],
    )
    def test_an_infinite_declared_field_is_null(self, model, field):
        result = model(**{field: INF})
        assert getattr(result, field) is None
        assert any(line.startswith(f"{field} is null") for line in result.warnings)

    @pytest.mark.parametrize(
        "model",
        [D.CashFuturesBasisResult, D.BasisHistoryResult, D.SpreadAlert, D.BasisBreak],
    )
    def test_an_undeclared_key_is_refused(self, model):
        """It used to be accepted and passed through unconverted."""
        with pytest.raises(PydanticValidationError, match="extra"):
            model(not_a_field=1.0)

    def test_every_key_a_real_call_returns_is_declared(self):
        """The null case: the library's own dicts construct cleanly."""
        from standard_quant_tools.agent.runtimes.delta_one import tools as T
        from standard_quant_tools.agent.runtimes.delta_one.models import (
            BasisHistoryInput,
            CashFuturesBasisInput,
        )

        basis = T.analyze_cash_futures_basis(
            CashFuturesBasisInput(
                spot=100.0, future_price=101.0, time_to_expiry=0.5, risk_free_rate=0.03
            )
        )
        assert math.isfinite(basis.carry_spread_rate)
        spot = list(100 + np.cumsum(np.random.default_rng(4).normal(0, 1, 120)))
        history = T.analyze_basis_history(
            BasisHistoryInput(spot_prices=spot, futures_prices=[s * 1.01 for s in spot])
        )
        assert history.basis_flat is True
        assert history.half_life_t_statistic is None

    def test_a_monitor_state_carries_no_infinity(self):
        result = D.SpreadMonitorResult(state={"m2": INF, "n": 3})
        assert result.state == {"m2": None, "n": 3}


# ── the other runtimes ──────────────────────────────────────────────────


class TestCompareArtifacts:
    def _differences(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        from standard_quant_tools.agent.runtimes import resolve

        fn, model = resolve("meta").dispatch_table["compare_artifacts"]
        return fn(model(a=a, b=b)).model_dump()["differences"]

    def test_a_non_finite_side_is_its_token_not_null(self):
        """Null already means "absent on this side" here; a NaN reported as
        null read as a field missing from one artifact."""
        diffs = {
            d["path"]: d
            for d in self._differences({"x": NAN, "y": INF}, {"x": 1.0, "y": 2.0})
        }
        assert diffs["x"]["a"] == "NaN" and diffs["x"]["b"] == 1.0
        assert diffs["y"]["a"] == "Infinity"

    def test_an_absent_side_is_still_null(self):
        """The null case."""
        diffs = {
            d["path"]: d for d in self._differences({"x": 1.0, "z": 1.0}, {"x": 1.0})
        }
        assert diffs["z"]["kind"] == "only_in_a" and diffs["z"]["b"] is None


class TestCompareDistributions:
    def test_a_moment_too_large_to_represent_is_null_and_explained(self):
        from standard_quant_tools.agent.runtimes import resolve

        fn, model = resolve("research").dispatch_table["compare_distributions"]
        rng = np.random.default_rng(1)
        with np.errstate(over="ignore"):
            result = fn(
                model(
                    sample_a=list(rng.normal(0, 1, 300) * 1e300),
                    sample_b=list(rng.normal(0, 1, 300)),
                )
            )
        std = next(s for s in result.moment_shifts if s.moment == "std")
        assert std.model_dump()["a"] is None
        assert any("overflowed" in line for line in result.warnings)


class TestFinancialRatios:
    def test_a_vendor_nan_is_null_and_named(self):
        from standard_quant_tools.agent.runtimes import resolve

        fn, model = resolve("data").dispatch_table["validate_financial_ratios"]
        result = fn(
            model(
                ratios={"symbol": "AAA", "price_to_earnings": NAN, "price_to_book": 3.0}
            )
        )
        assert result.ratios["price_to_earnings"] is None
        assert result.ratios["price_to_book"] == 3.0
        assert any(
            line.startswith("ratios.price_to_earnings is null")
            for line in result.warnings
        )


class TestModelingMetrics:
    @pytest.mark.parametrize(
        "model",
        [MM.EvaluateModelPortfolioResult, PM.EvaluatePredictionsPortfolioResult],
    )
    def test_an_unbounded_sortino_is_null_and_explained(self, model):
        result = _valid(model, metrics={"sortino_ratio": INF, "cagr": 0.1})
        assert result.metrics == {"sortino_ratio": None, "cagr": 0.1}
        assert any(
            line.startswith(
                "metrics.sortino_ratio is null: no return fell below the risk-free rate"
            )
            for line in result.warnings
        )

    def test_a_baseline_metric_that_overflowed_is_null(self):
        result = _valid(
            MM.ScorePredictionsResult,
            baseline={"baseline_r2": -INF, "baseline_mae": 0.1},
        )
        assert result.baseline["baseline_r2"] is None
        assert any(
            line.startswith("baseline.baseline_r2 is null") for line in result.warnings
        )
