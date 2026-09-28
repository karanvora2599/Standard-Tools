"""
What the derivatives surface does with the numbers it is handed.

Two maps on this surface are keyed by a NUMBER written as a string, because
JSON has no numeric keys. Two strings that are the same number -- '0.25' and
'0.250' -- used to collapse into one key, and whichever quote the JSON object
listed last silently replaced the other. The volatility cone then truncated
its keys with int(), so '21.7' became 21 and collided with '21' the same
way.

And no rate field on the surface had a bound: a NaN rate priced to a null
price with no warning, and a huge one returned a number or a bare
OverflowError depending on the model.

Every refusal below has its null: distinct keys, whole horizons and an
ordinary negative rate are all answered.
"""

import numpy as np
import pytest
from pydantic import ValidationError as SchemaError

from standard_quant_tools.agent.models import (
    ImpliedVolatilityInput,
    OptionPricingInput,
)
from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes.derivatives.models import (
    DeltaHedgeInput,
    ImpliedForwardInput,
    OptionGreeksInput,
    OptionScenariosInput,
    OptionStrategyInput,
    PutCallParityInput,
)
from standard_quant_tools.error import ValidationError


def _dispatch(name, arguments):
    return resolve("derivatives").dispatch(name, arguments)


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return list(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))))


class TestKeysThatAreTheSameNumber:
    def test_two_spellings_of_one_expiry_are_refused_naming_both(self):
        with pytest.raises(ValidationError, match=r"'0\.25' and '0\.250'"):
            _dispatch(
                "analyze_vol_term_structure",
                {"implied_by_expiry": {"0.25": 0.20, "0.250": 0.90, "0.5": 0.25}},
            )

    def test_distinct_expiries_are_answered(self):
        """The null case."""
        result = _dispatch(
            "analyze_vol_term_structure",
            {"implied_by_expiry": {"0.25": 0.20, "0.5": 0.25}},
        )
        assert result["n_expiries"] == 2

    def test_two_spellings_of_one_horizon_are_refused(self):
        with pytest.raises(ValidationError, match=r"'21' and '21\.0'"):
            _dispatch(
                "get_volatility_cone",
                {"prices": _prices(), "current_implied": {"21": 0.2, "21.0": 0.9}},
            )

    def test_a_fractional_horizon_is_refused_rather_than_truncated(self):
        """int('21.7') would have been 21, replacing the real 21-day quote."""
        with pytest.raises(ValidationError, match="21.7"):
            _dispatch(
                "get_volatility_cone",
                {"prices": _prices(), "current_implied": {"21": 0.2, "21.7": 0.9}},
            )

    def test_a_whole_horizon_is_placed_on_the_cone(self):
        """The null case: '21' reaches the 21-day row."""
        result = _dispatch(
            "get_volatility_cone",
            {"prices": _prices(), "current_implied": {"21": 0.2}},
        )
        row = next(r for r in result["cone"] if r["horizon_days"] == 21)
        assert row["implied_vol"] == 0.2


_ATM = dict(spot=100.0, strike=100.0, time_to_expiry=1.0)


class TestEveryRateFieldIsBounded:
    MODELS = [
        (OptionPricingInput, {"volatility": 0.2}),
        (ImpliedVolatilityInput, {"option_price": 12.0}),
        (OptionGreeksInput, {"volatility": 0.2}),
        (OptionScenariosInput, {"volatility": 0.2}),
        (
            PutCallParityInput,
            {"call_price": 10.0, "put_price": 5.0},
        ),
        (
            DeltaHedgeInput,
            {"implied_vol": 0.2, "realized_vol": 0.2},
        ),
    ]

    @pytest.mark.parametrize("model_cls, extra", MODELS)
    @pytest.mark.parametrize("rate", [float("nan"), 11.0, -11.0, 1e300])
    def test_a_rate_past_the_bound_is_refused_at_the_schema(
        self, model_cls, extra, rate
    ):
        with pytest.raises(SchemaError, match="risk_free_rate"):
            model_cls(risk_free_rate=rate, **extra, **_ATM)

    @pytest.mark.parametrize("model_cls, extra", MODELS)
    def test_a_negative_rate_is_accepted(self, model_cls, extra):
        """The null case: the bound is on magnitude, never on sign."""
        model_cls(risk_free_rate=-0.5, **extra, **_ATM)

    def test_the_forward_and_strategy_inputs_are_bounded_too(self):
        with pytest.raises(SchemaError, match="risk_free_rate"):
            ImpliedForwardInput(spot=100.0, time_to_expiry=1.0, risk_free_rate=11.0)
        with pytest.raises(SchemaError, match="risk_free_rate"):
            OptionStrategyInput(
                legs=[{"option_type": "stock", "quantity": 1}],
                spot=100.0,
                risk_free_rate=float("nan"),
            )


class TestTheStrategyResultSaysHowItWasValued:
    LEG = {"option_type": "call", "strike": 100.0, "volatility": 0.2}

    def test_a_calendar_reports_its_evaluation_date(self):
        result = _dispatch(
            "analyze_option_strategy",
            {
                "legs": [
                    dict(self.LEG, quantity=1, time_to_expiry=0.75),
                    dict(self.LEG, quantity=-1, time_to_expiry=1 / 12),
                ],
                "spot": 100.0,
            },
        )
        assert result["payoff_basis"] == "first_expiry_marked"
        assert result["evaluated_at_years"] == pytest.approx(1 / 12)
        assert len(result["breakevens"]) == 2

    def test_a_single_expiry_reports_expiry(self):
        """The null case."""
        result = _dispatch(
            "analyze_option_strategy",
            {"legs": [dict(self.LEG, quantity=1, time_to_expiry=0.25)], "spot": 100.0},
        )
        assert result["payoff_basis"] == "expiry"
        assert result["evaluated_at_years"] == pytest.approx(0.25)
