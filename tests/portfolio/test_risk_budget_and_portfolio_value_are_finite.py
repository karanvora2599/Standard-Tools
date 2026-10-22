"""
A risk budget and a portfolio value are finite numbers, at every door.

Risk parity checked that each budget was positive, and NaN compares False
against `<= 0`, so a NaN budget passed; an infinite one passed too and made
every normalized target NaN. Both came back as NaN weights. `plan_rebalance`
checked `portfolio_value <= 0`, which NaN passes the same way: the plan came
back with a NaN cost and converged=True, and an infinite value -- which the
tool's `gt=0` admits -- gave a NaN cost. Each is now refused, naming the
argument and the value, at the library, at the tool's input model and
through dispatch. See the CHANGELOG entry of 2026-10-22.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.models import PlanRebalanceInput
from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes.portfolio.construction_tools import (
    RiskParityInput,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.construction import risk_parity
from standard_quant_tools.portfolio.optimize import risk_parity_weights
from standard_quant_tools.portfolio.rebalance import plan_rebalance

NAN, INF = math.nan, math.inf
ASSETS = ["A", "B", "C"]
COV = [[0.04, 0.01, 0.0], [0.01, 0.09, 0.02], [0.0, 0.02, 0.16]]

#: A refusal at any of the three doors. Dispatch builds the input model
#: first, so a schema refusal arrives there as pydantic's error.
REFUSED = (ValidationError, pydantic.ValidationError)


@pytest.mark.parametrize("bad", [NAN, INF], ids=["nan", "inf"])
class TestARiskBudgetIsFinite:
    def test_the_library_refuses_it_by_asset(self, bad):

        frame = pd.DataFrame(COV, index=ASSETS, columns=ASSETS)
        with pytest.raises(ValidationError, match=r"not finite for \{'A'"):
            risk_parity(frame, budget=[bad, 0.5, 0.5])

    def test_the_mean_variance_module_refuses_it(self, bad):
        with pytest.raises(ValidationError, match="non-finite"):
            risk_parity_weights(np.array(COV), risk_budget=np.array([bad, 0.5, 0.5]))

    def test_the_tool_schema_and_dispatch_refuse_it(self, bad):
        with pytest.raises(pydantic.ValidationError, match="budget is not finite"):
            RiskParityInput(assets=ASSETS, covariance=COV, budget=[bad, 0.5, 0.5])
        with pytest.raises(REFUSED, match="budget"):
            resolve("portfolio").dispatch(
                "optimize_risk_parity",
                {"assets": ASSETS, "covariance": COV, "budget": [bad, 0.5, 0.5]},
            )


class TestATolerancePastAnySweepIsRefused:
    def test_an_infinite_tolerance_is_refused_at_both_doors(self):
        """`abs(change) < inf` holds after one sweep, so the inverse-volatility
        start came back as a converged risk parity portfolio."""

        frame = pd.DataFrame(COV, index=ASSETS, columns=ASSETS)
        with pytest.raises(ValidationError, match="tolerance"):
            risk_parity(frame, tolerance=INF)
        with pytest.raises(pydantic.ValidationError, match="tolerance"):
            RiskParityInput(assets=ASSETS, covariance=COV, tolerance=INF)


class TestAFiniteBudgetIsAnsweredAsBefore:
    def test_a_budget_is_met(self):

        frame = pd.DataFrame(COV, index=ASSETS, columns=ASSETS)
        result = risk_parity(frame, budget=[0.5, 0.3, 0.2])
        assert result["converged"] is True
        for asset, target in zip(ASSETS, (0.5, 0.3, 0.2)):
            assert result["risk_shares"][asset] == pytest.approx(target, abs=1e-8)


WEIGHTS_NOW = {"A": 0.5, "B": 0.5}
WEIGHTS_WANTED = {"A": 0.2, "B": 0.8}


@pytest.mark.parametrize("bad", [NAN, INF], ids=["nan", "inf"])
class TestAPortfolioValueIsFinite:
    def test_the_library_refuses_it(self, bad):
        with pytest.raises(ValidationError, match="portfolio_value must be a finite"):
            plan_rebalance(
                WEIGHTS_NOW,
                WEIGHTS_WANTED,
                portfolio_value=bad,
                adv={"A": 1e6, "B": 1e6},
            )

    def test_the_tool_schema_and_dispatch_refuse_it(self, bad):
        arguments = dict(
            current_weights=WEIGHTS_NOW,
            target_weights=WEIGHTS_WANTED,
            portfolio_value=bad,
            adv={"A": 1e6, "B": 1e6},
        )
        with pytest.raises(pydantic.ValidationError, match="portfolio_value"):
            PlanRebalanceInput(**arguments)
        with pytest.raises(REFUSED, match="portfolio_value"):
            resolve("portfolio").dispatch("plan_rebalance", arguments)


class TestTheRestOfThePlanIsFiniteToo:
    def test_a_nan_holding_is_refused_by_the_library(self):
        """The tool's schema refused it already; a direct caller got a NaN
        turnover and a NaN residual distance."""
        with pytest.raises(ValidationError, match=r"current_weights is not finite"):
            plan_rebalance({"A": NAN, "B": 0.5}, WEIGHTS_WANTED, portfolio_value=1e6)

    @pytest.mark.parametrize("bad", [NAN, INF, -1.0])
    def test_an_impact_coefficient_that_is_no_cost_is_refused(self, bad):
        with pytest.raises(ValidationError, match="impact_coefficient"):
            plan_rebalance(
                WEIGHTS_NOW,
                WEIGHTS_WANTED,
                portfolio_value=1e6,
                adv={"A": 1e6, "B": 1e6},
                impact_coefficient=bad,
            )

    def test_a_finite_plan_is_answered_as_before(self):
        result = plan_rebalance(
            WEIGHTS_NOW,
            WEIGHTS_WANTED,
            portfolio_value=1e6,
            adv={"A": 1e9, "B": 1e9},
            urgency=1.0,
        )
        assert result["converged"] is True
        assert result["total_turnover"] == pytest.approx(0.6)
        # 300,000 a side at 0.03% of a 1e9 ADV: 10 bps * sqrt(3e-4).
        assert result["total_cost_bps"] == pytest.approx(10.0 * math.sqrt(3e-4))
