"""
Regression tests for Pass 4 of the full-codebase audit: the classic agent
schemas.

The modeling schemas had gained Literals, numeric bounds, universe caps and
seed ranges. The classic quant schemas kept bare `str` / `float` / `int` /
`Dict`, so the LLM-facing contract was materially weaker on exactly the
surface an agent drives most — and a bad value was discovered part-way
through a tool, after data had been fetched, rather than at construction.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from standard_quant_tools.agent.models import (
    BacktestInput,
    PairScannerInput,
    TechnicalInput,
    WalkForwardInput,
)
from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

BASE = dict(symbol="AAPL", start_date="2023-01-02", end_date="2024-01-01")
WF_BASE = dict(
    **BASE,
    strategy="sma_crossover",
    param_grid={"fast_period": [5]},
    train_bars=252,
    test_bars=63,
)


class TestFinancialScalarsAreBounded:
    """The engine rejected these later; the agent contract should refuse
    before dispatch, since by then a fetch has already happened."""

    @pytest.mark.parametrize("capital", [0.0, -1000.0])
    def test_non_positive_capital_rejected(self, capital):
        with pytest.raises(PydanticValidationError):
            BacktestInput(
                **BASE, strategy_type="sma_crossover", initial_capital=capital
            )

    @pytest.mark.parametrize("field", ["commission_pct", "slippage_pct"])
    @pytest.mark.parametrize("value", [-0.001, 1.5])
    def test_cost_fractions_bounded_to_zero_one(self, field, value):
        """Above 1.0 means paying more than the trade's whole notional."""
        with pytest.raises(PydanticValidationError):
            BacktestInput(**BASE, strategy_type="sma_crossover", **{field: value})

    def test_ordinary_values_still_accepted(self):
        spec = BacktestInput(
            **BASE,
            strategy_type="sma_crossover",
            initial_capital=25_000.0,
            commission_pct=0.001,
            slippage_pct=0.0005,
        )
        assert spec.initial_capital == 25_000.0


class TestStrategyAndFillPriceAreEnumerated:
    def test_unregistered_strategy_rejected_at_construction(self):
        """
        Previously a bare `str`, so an unregistered name was discovered
        part-way through the tool — after fetching data and beginning a
        walk-forward.
        """
        with pytest.raises(PydanticValidationError):
            WalkForwardInput(**{**WF_BASE, "strategy": "nonexistent_strategy"})

    def test_the_literal_matches_the_registry_exactly(self):
        """A Literal narrower than the registry silently makes a working
        strategy unreachable from the agent surface."""
        for name in STRATEGY_REGISTRY:
            WalkForwardInput(**{**WF_BASE, "strategy": name})

    def test_unsupported_fill_price_rejected(self):
        with pytest.raises(PydanticValidationError):
            BacktestInput(**BASE, strategy_type="sma_crossover", fill_price="magic")

    def test_unknown_sort_metric_rejected(self):
        """
        `sort_by` was a raw string, and the code did
        `if sort_by and sort_by in df.columns` — so an unrecognized metric
        was SILENTLY IGNORED and the caller got unsorted results with no
        indication the request had not been honoured.
        """
        with pytest.raises(PydanticValidationError):
            WalkForwardInput(**{**WF_BASE, "sort_by": "not_a_metric"})

    def test_real_sort_metrics_accepted(self):
        for metric in ("sharpe_ratio", "calmar_ratio", "total_return"):
            WalkForwardInput(**{**WF_BASE, "sort_by": metric})


class TestIndicatorNamesAreEnumerated:
    def test_hallucinated_indicator_rejected(self):
        with pytest.raises(PydanticValidationError):
            TechnicalInput(**BASE, indicators=["rsi", "quantum_oscillator"])

    def test_supported_indicators_accepted(self):
        spec = TechnicalInput(**BASE, indicators=["rsi", "macd", "adx"])
        assert spec.indicators == ["rsi", "macd", "adx"]

    def test_empty_indicator_list_rejected(self):
        with pytest.raises(PydanticValidationError):
            TechnicalInput(**BASE, indicators=[])


class TestParameterGridBudget:
    """
    A grid is evaluated as the CARTESIAN PRODUCT of its axes, so cost is
    multiplicative and a reasonable-looking request can ask for an
    unreasonable amount of work: four axes of ten values is 10,000 full
    backtests from a dict that fits on one line. Estimator complexity was
    bounded; the NUMBER of estimator invocations was not.
    """

    def test_oversized_grid_rejected(self):
        huge = {name: list(range(20)) for name in ("a", "b", "c", "d")}
        with pytest.raises(PydanticValidationError, match="param_grid"):
            WalkForwardInput(**{**WF_BASE, "param_grid": huge})

    def test_empty_axis_rejected(self):
        with pytest.raises(PydanticValidationError, match="param_grid"):
            WalkForwardInput(**{**WF_BASE, "param_grid": {"fast_period": []}})

    def test_empty_grid_rejected(self):
        with pytest.raises(PydanticValidationError, match="param_grid"):
            WalkForwardInput(**{**WF_BASE, "param_grid": {}})

    def test_a_realistic_grid_is_accepted(self):
        spec = WalkForwardInput(
            **{
                **WF_BASE,
                "param_grid": {
                    "fast_period": [5, 10, 20],
                    "slow_period": [30, 50, 100],
                },
            }
        )
        assert len(spec.param_grid) == 2


class TestPairScannerBounds:
    def test_invalid_p_value_threshold_rejected(self):
        base = dict(tickers=["A", "B"], start_date="2023-01-02", end_date="2024-01-01")
        for bad in (0.0, 1.5, -0.1):
            with pytest.raises(PydanticValidationError):
                PairScannerInput(**base, p_value_threshold=bad)

    def test_non_positive_max_pairs_rejected(self):
        base = dict(tickers=["A", "B"], start_date="2023-01-02", end_date="2024-01-01")
        with pytest.raises(PydanticValidationError):
            PairScannerInput(**base, max_pairs=0)


class TestPeriodStringsAreParsedStrictly:
    """
    An unrecognized unit used to fall through to `now - 365 days`, so a
    malformed request did not fail — it silently became a DIFFERENT valid
    request, and a wrong window is not detectable from the result.
    """

    @pytest.mark.parametrize("period", ["1y", "6mo", "30d", "4w"])
    def test_supported_forms_parse(self, period):
        from standard_quant_tools.agent.tools import _parse_period

        assert _parse_period(period) is not None

    @pytest.mark.parametrize("period", ["6m", "1yr", "ytd", "", "abc", "0d"])
    def test_malformed_period_no_longer_becomes_one_year(self, period):
        from standard_quant_tools.agent.tools import _parse_period
        from standard_quant_tools.error import ValidationError

        with pytest.raises(ValidationError):
            _parse_period(period)

    def test_the_units_actually_differ(self):
        """Guards against a parser that accepts the forms and ignores them."""
        import datetime

        from standard_quant_tools.agent.tools import _parse_period

        now = datetime.datetime.now()
        days_30 = (now - _parse_period("30d")).days
        days_1y = (now - _parse_period("1y")).days
        assert 29 <= days_30 <= 31
        assert 364 <= days_1y <= 366
