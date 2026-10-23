"""
A cost of capital from the caller's inputs: CAPM, after-tax cost of debt,
WACC on market values, and Hamada's unlevered and relevered beta.

THE PLANTED FIRM, by hand. Equity 800 and debt 200 at market, so weights
0.8 / 0.2 and D/E 0.25; risk-free 4%, beta 1.2, premium 5.5%, a pre-tax
cost of debt of 6% and a 21% tax rate:

    cost of equity    0.04 + 1.2 x 0.055                  = 0.106
    after-tax debt    0.06 x 0.79                         = 0.0474
    WACC              0.8 x 0.106 + 0.2 x 0.0474          = 0.09428

HAMADA'S ROUND TRIP: a levered beta of 1.3 at D/E 0.5 and t = 21%
unlevers to 1.3 / (1 + 0.79 x 0.5) = 0.931900, and relevered at the same
D/E returns 1.3.
"""

from __future__ import annotations

import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.error import ValidationError
from standard_quant_tools.finance import (
    estimate_cost_of_capital,
    relever_beta,
    unlever_beta,
)

FIRM = {
    "risk_free_rate": 0.04,
    "beta": 1.2,
    "equity_risk_premium": 0.055,
    "equity_value": 800.0,
    "debt_value": 200.0,
    "cost_of_debt": 0.06,
    "tax_rate": 0.21,
}


@pytest.fixture(scope="module")
def finance():
    return resolve("finance")


class TestPlantedFirm:
    def test_capm_after_tax_debt_and_wacc(self, finance):
        """Every term of the module docstring's arithmetic, to 1e-15."""
        out = finance.dispatch("estimate_cost_of_capital", FIRM)
        assert out["cost_of_equity"] == pytest.approx(0.106, rel=1e-14)
        assert out["after_tax_cost_of_debt"] == pytest.approx(0.0474, rel=1e-14)
        assert out["equity_weight"] == pytest.approx(0.8, rel=1e-15)
        assert out["debt_to_equity"] == pytest.approx(0.25, rel=1e-15)
        assert out["wacc"] == pytest.approx(0.09428, rel=1e-14)

    def test_the_hamada_round_trip_returns_the_beta(self, finance):
        """1.3 -> 0.931900 -> 1.3 at D/E 0.5 and t = 21%."""
        out = finance.dispatch(
            "estimate_cost_of_capital",
            {
                **FIRM,
                "beta": 1.3,
                "equity_value": 100.0,
                "debt_value": 50.0,
                "target_debt_to_equity": 0.5,
            },
        )
        assert out["unlevered_beta"] == pytest.approx(1.3 / 1.395, rel=1e-15)
        assert round(out["unlevered_beta"], 6) == 0.931900
        assert out["at_target"]["beta"] == pytest.approx(1.3, rel=1e-15)
        assert out["at_target"]["wacc"] == pytest.approx(out["wacc"], rel=1e-15)

    def test_the_library_helpers_invert_each_other(self):
        for beta, leverage, tax in (
            (1.3, 0.5, 0.21),
            (0.8, 2.0, 0.35),
            (1.1, 0.0, 0.0),
        ):
            assert relever_beta(
                unlever_beta(beta, leverage, tax), leverage, tax
            ) == pytest.approx(beta, rel=1e-15)

    def test_with_no_debt_wacc_is_the_cost_of_equity_exactly(self, finance):
        """Weight 1 on equity and 0 on debt, to the bit -- and no cost of
        debt is needed."""
        out = finance.dispatch(
            "estimate_cost_of_capital",
            {
                key: value
                for key, value in FIRM.items()
                if key not in ("debt_value", "cost_of_debt")
            },
        )
        assert out["equity_weight"] == 1.0
        assert out["wacc"] == out["cost_of_equity"]
        assert out["unlevered_beta"] == 1.2
        assert out["after_tax_cost_of_debt"] is None

    def test_relevering_at_zero_debt_is_the_unlevered_firm(self, finance):
        out = finance.dispatch(
            "estimate_cost_of_capital", {**FIRM, "target_debt_to_equity": 0.0}
        )
        assert out["at_target"]["beta"] == pytest.approx(out["unlevered_beta"])
        assert out["at_target"]["wacc"] == pytest.approx(
            out["unlevered_cost_of_equity"], rel=1e-15
        )

    def test_the_sensitivity_is_one_standard_error_of_premium(self, finance):
        """k_e moves by ERP x SE each way, and WACC by that times E / V."""
        out = finance.dispatch(
            "estimate_cost_of_capital", {**FIRM, "beta_standard_error": 0.2}
        )
        band = out["beta_sensitivity"]
        assert band["beta_low"] == pytest.approx(1.0)
        assert band["beta_high"] == pytest.approx(1.4)
        assert band["cost_of_equity_high"] - band["cost_of_equity_low"] == (
            pytest.approx(2 * 0.2 * 0.055, rel=1e-12)
        )
        assert band["wacc_high"] - band["wacc_low"] == pytest.approx(
            2 * 0.2 * 0.055 * 0.8, rel=1e-12
        )

    def test_without_a_target_or_an_error_those_blocks_are_null(self, finance):
        out = finance.dispatch("estimate_cost_of_capital", FIRM)
        assert out["at_target"] is None and out["beta_sensitivity"] is None

    def test_a_target_that_adds_unpriced_debt_has_no_wacc(self, finance):
        """No debt today and no cost of debt: relevering to D/E 0.5 adds
        debt nobody priced, so that WACC is null with the reason."""
        out = finance.dispatch(
            "estimate_cost_of_capital",
            {
                "risk_free_rate": 0.04,
                "beta": 1.0,
                "equity_risk_premium": 0.05,
                "equity_value": 100.0,
                "tax_rate": 0.25,
                "target_debt_to_equity": 0.5,
            },
        )
        assert out["at_target"]["beta"] == pytest.approx(1.375)
        assert out["at_target"]["wacc"] is None
        assert any("no cost_of_debt" in w for w in out["warnings"])


class TestRefusals:
    @pytest.mark.parametrize(
        "change, named",
        [
            ({"equity_value": -5.0}, "equity_value"),
            ({"equity_value": 0.0}, "equity_value"),
            ({"tax_rate": 1.0}, "tax_rate"),
            ({"tax_rate": -0.1}, "tax_rate"),
            ({"tax_rate": 21.0}, "tax_rate"),
            ({"cost_of_debt": None}, "cost_of_debt"),
            ({"risk_free_rate": 4.0}, "risk_free_rate"),
            ({"debt_value": -1.0}, "debt_value"),
        ],
    )
    def test_through_the_tool(self, finance, change, named):
        arguments = {k: v for k, v in {**FIRM, **change}.items() if v is not None}
        with pytest.raises((ValidationError, pydantic.ValidationError)) as exc:
            finance.dispatch("estimate_cost_of_capital", arguments)
        assert named in str(exc.value)

    @pytest.mark.parametrize(
        "change, fix",
        [
            ({"equity_value": -5.0}, "market capitalization"),
            ({"tax_rate": 1.0}, "21% is 0.21"),
            ({"debt_value": 10.0, "cost_of_debt": None}, "cost_of_debt"),
            ({"risk_free_rate": 4.0}, "percentage divided by 100"),
        ],
    )
    def test_in_the_library_with_the_fix(self, change, fix):
        with pytest.raises(ValidationError, match=fix):
            estimate_cost_of_capital(**{**FIRM, **change})
