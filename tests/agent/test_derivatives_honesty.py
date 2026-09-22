"""
The three answers the derivatives surface gave that looked confident.

`get_implied_volatility` solved a deep in-the-money quote priced at
intrinsic, where no volatility is identifiable, and returned seven times
the true one with `converged: True` -- the library had said `at_bound` and
the boundary dropped it. `get_option_risk_scenarios` had no
`dividend_yield` at all, so every cell of every grid priced a non-payer.
And every yield and borrow field on the surface was pinned at `ge=0`,
which made an FX option (whose foreign rate is the yield) and a commodity
whose convenience yield exceeds its storage cost unpriceable through tools
whose own formulas handle either sign.

Each case below is planted, and each detector has its null: a price away
from the bound raises no warning, and a zero yield reproduces today's
number bit for bit.

See the CHANGELOG entry of 2026-09-22.
"""

import pytest
from pydantic import ValidationError as SchemaError

from standard_quant_tools.agent.models import (
    ImpliedVolatilityInput,
    OptionPricingInput,
)
from standard_quant_tools.agent.runtimes.derivatives.models import (
    OptionGreeksInput,
    OptionScenariosInput,
)
from standard_quant_tools.agent.runtimes.derivatives.tools import (
    get_implied_volatility,
    get_option_greeks,
    get_option_pricing,
    get_option_risk_scenarios,
)
from standard_quant_tools.analysis.options import black_scholes_price

#: A quarter-year call struck at 40 on a spot of 100. At a 5% rate its
#: Black-Scholes price is bit-for-bit its intrinsic value, so every
#: volatility at or below the true 0.05 reproduces it.
_DEEP_ITM = dict(spot=100.0, strike=40.0, time_to_expiry=0.25, risk_free_rate=0.05)
_TRUE_SIGMA = 0.05

#: One year, at the money, 20% vol, 5% rates.
_ATM = dict(spot=100.0, strike=100.0, time_to_expiry=1.0, risk_free_rate=0.05)


class TestAVolatilityAtTheBoundIsLabelledACeiling:
    def test_a_call_priced_at_intrinsic_reports_a_ceiling_with_a_warning(self):
        price = black_scholes_price(
            _DEEP_ITM["spot"],
            _DEEP_ITM["strike"],
            _DEEP_ITM["time_to_expiry"],
            _DEEP_ITM["risk_free_rate"],
            _TRUE_SIGMA,
            "call",
        )
        result = get_implied_volatility(
            ImpliedVolatilityInput(option_price=price, **_DEEP_ITM)
        )

        assert result.at_bound is True
        # Seven times the volatility that produced the price, and the
        # solver is not wrong to report it: every sigma at or below 0.05
        # prices this option at the same number.
        assert result.implied_volatility > 5 * _TRUE_SIGMA
        # `converged` stays TRUE. The bisection did converge; what it
        # converged to is a bound, and that is a separate fact.
        assert result.converged is True
        assert result.warnings, "an unidentifiable volatility with no warning"
        warning = " ".join(result.warnings)
        assert "CEILING" in warning
        assert "lower bound" in warning

    def test_a_price_away_from_the_bound_says_nothing(self):
        """The null case. A warning on every quote teaches agents to skip it."""
        price = black_scholes_price(
            _ATM["spot"],
            _ATM["strike"],
            _ATM["time_to_expiry"],
            _ATM["risk_free_rate"],
            0.2,
            "call",
        )
        result = get_implied_volatility(
            ImpliedVolatilityInput(option_price=price, **_ATM)
        )

        assert result.at_bound is False
        assert result.warnings == []
        assert result.implied_volatility == pytest.approx(0.2, abs=1e-6)
        # The tolerance the solver reports `price_error` against.
        assert result.price_error <= 1e-6

    def test_the_price_error_is_the_residual_the_solver_stopped_on(self):
        price = black_scholes_price(
            _ATM["spot"],
            _ATM["strike"],
            _ATM["time_to_expiry"],
            _ATM["risk_free_rate"],
            0.35,
            "put",
        )
        result = get_implied_volatility(
            ImpliedVolatilityInput(option_price=price, option_type="put", **_ATM)
        )
        reproduced = black_scholes_price(
            _ATM["spot"],
            _ATM["strike"],
            _ATM["time_to_expiry"],
            _ATM["risk_free_rate"],
            result.implied_volatility,
            "put",
        )
        assert result.price_error == pytest.approx(abs(reproduced - price), abs=1e-6)


class TestTheScenarioGridPricesTheDividend:
    def test_a_four_percent_yield_moves_the_whole_grid(self):
        flat = get_option_risk_scenarios(OptionScenariosInput(volatility=0.2, **_ATM))
        payer = get_option_risk_scenarios(
            OptionScenariosInput(volatility=0.2, dividend_yield=0.04, **_ATM)
        )

        # A one-year at-the-money call on a 4% yielder is worth about 23%
        # less than the same call on a non-payer, and before this field
        # existed there was no argument that could say so.
        assert payer.base_value / flat.base_value - 1 == pytest.approx(
            -0.2247, abs=5e-4
        )
        assert payer.dividend_yield == 0.04
        # Every cell moves with it, not only the base.
        for row_flat, row_payer in zip(flat.grid, payer.grid):
            for cell_flat, cell_payer in zip(row_flat.cells, row_payer.cells):
                assert cell_payer.value < cell_flat.value

    def test_a_zero_yield_reproduces_todays_grid_bit_for_bit(self):
        """The null case, and the compatibility pin: the default path is the

        old path."""
        flat = get_option_risk_scenarios(OptionScenariosInput(volatility=0.2, **_ATM))
        explicit = get_option_risk_scenarios(
            OptionScenariosInput(volatility=0.2, dividend_yield=0.0, **_ATM)
        )

        assert flat.base_value == 10.450583572185565
        assert flat.grid[0].cells[0].value == 0.14757028598521282
        assert flat.worst_case.pnl == -10.303013286200352
        assert flat.dividend_yield == 0.0
        assert explicit.model_dump() == flat.model_dump()


class TestANegativeYieldIsPriced:
    def test_the_pricer_takes_a_negative_foreign_rate_and_the_price_moves(self):
        flat = get_option_pricing(OptionPricingInput(volatility=0.2, **_ATM))
        foreign = get_option_pricing(
            OptionPricingInput(volatility=0.2, dividend_yield=-0.005, **_ATM)
        )

        assert foreign.price == pytest.approx(10.772142, abs=1e-6)
        # A negative yield adds carry to the forward, so the call is worth
        # MORE. Accepting the field is not the point; the number moving is.
        assert foreign.price > flat.price
        assert foreign.greeks.delta > flat.greeks.delta

    def test_the_second_order_greeks_take_it_too(self):
        flat = get_option_greeks(OptionGreeksInput(volatility=0.2, **_ATM))
        foreign = get_option_greeks(
            OptionGreeksInput(volatility=0.2, dividend_yield=-0.005, **_ATM)
        )

        assert foreign.vanna != flat.vanna
        assert foreign.charm != flat.charm

    def test_a_zero_yield_is_unchanged(self):
        """The null case on the relaxed bound: widening it moved nothing."""
        flat = get_option_pricing(OptionPricingInput(volatility=0.2, **_ATM))
        explicit = get_option_pricing(
            OptionPricingInput(volatility=0.2, dividend_yield=0.0, **_ATM)
        )

        assert flat.price == 10.450584
        assert explicit.model_dump() == flat.model_dump()

    @pytest.mark.parametrize(
        "model_cls, extra",
        [
            (OptionPricingInput, {"volatility": 0.2}),
            (OptionGreeksInput, {"volatility": 0.2}),
            (OptionScenariosInput, {"volatility": 0.2}),
            (ImpliedVolatilityInput, {"option_price": 12.0}),
        ],
    )
    def test_a_yield_past_the_magnitude_bound_is_refused_by_range(
        self, model_cls, extra
    ):
        """The bound moved from a SIGN to a MAGNITUDE, and it is still a

        bound: -11 is a unit error, not an FX option."""
        with pytest.raises(SchemaError, match="-10"):
            model_cls(dividend_yield=-11.0, **extra, **_ATM)

        with pytest.raises(SchemaError, match="10"):
            model_cls(dividend_yield=11.0, **extra, **_ATM)
