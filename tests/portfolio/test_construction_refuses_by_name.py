"""
Portfolio construction refuses, by name, what its tool doors do.

`analyze_concentration` and `get_factor_exposure_budget` check their inline
weights and loadings in their schemas. The library functions behind them
did not, so a direct caller was answered anyway: a NaN weight was dropped
from the concentration book without a word, an infinite one -- or finite
ones whose gross sum passed the float range -- divided by zero, a NaN or
infinite weight or loading came out as a NaN or infinite exposure, a
repeated factor was reported once, and a repeated asset broke the
alignment with a matmul shape error naming no input.

The covariance-taking functions had the same hole at both levels: a
covariance whose rows repeat a name collapsed two assets into one key while
both rows were used -- risk parity over ["A", "A"] answered {"A": 0.4} --
and `marginal_risk_contribution` and `portfolio_scenarios` turned a NaN
weight into NaN rows or dropped it. Each is now a ValidationError naming
the entry and the remedy, with an ordinary book beside it as the null case.
See the CHANGELOG entry of 2026-09-28.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.construction import (
    concentration_analysis,
    factor_exposure_budget,
    hierarchical_risk_parity,
    marginal_risk_contribution,
    max_diversification,
    portfolio_scenarios,
    risk_parity,
)

NAN, INF = float("nan"), float("inf")

LOADINGS = pd.DataFrame(
    {"mkt": [1.0, 0.8, 1.2], "size": [0.2, -0.4, 0.0]}, index=["A", "B", "C"]
)
WEIGHTS = {"A": 0.5, "B": 0.3, "C": 0.2}


class TestFactorExposureBudget:
    @pytest.mark.parametrize("bad", [NAN, INF, -INF])
    def test_a_non_finite_weight_is_refused_by_name(self, bad):
        with pytest.raises(ValidationError, match=r"weights is not finite at \['A'\]"):
            factor_exposure_budget({**WEIGHTS, "A": bad}, LOADINGS)

    @pytest.mark.parametrize("bad", [NAN, INF])
    def test_a_non_finite_loading_of_a_held_asset_is_refused_by_pair(self, bad):
        loadings = LOADINGS.copy()
        loadings.loc["B", "size"] = bad
        with pytest.raises(ValidationError, match=r"\['B/size'\]"):
            factor_exposure_budget(WEIGHTS, loadings)

    def test_a_gap_in_an_asset_not_held_is_left_alone(self):
        """The null case for loadings: it never enters the sum."""
        loadings = pd.concat(
            [LOADINGS, pd.DataFrame({"mkt": [0.9], "size": [NAN]}, index=["D"])]
        )
        result = factor_exposure_budget(WEIGHTS, loadings)
        assert result["exposures"]["size"] == pytest.approx(0.1 - 0.12)

    def test_a_repeated_factor_is_refused(self):
        """It was reported once and counted twice in the variance."""
        loadings = LOADINGS.copy()
        loadings.columns = ["mkt", "mkt"]
        with pytest.raises(ValidationError, match=r"\['mkt'\] appear more than once"):
            factor_exposure_budget(WEIGHTS, loadings, factor_covariance=np.eye(2))

    def test_a_repeated_asset_in_the_loadings_is_refused(self):
        loadings = pd.concat([LOADINGS, LOADINGS.loc[["A"]]])
        with pytest.raises(ValidationError, match=r"\['A'\] appear more than once"):
            factor_exposure_budget(WEIGHTS, loadings)

    def test_a_repeated_weight_is_refused(self):
        weights = pd.Series([0.5, 0.5], index=["A", "A"])
        with pytest.raises(
            ValidationError, match=r"\['A'\] appear more than once in weights"
        ):
            factor_exposure_budget(weights, LOADINGS)

    def test_an_exposure_past_the_float_range_is_refused(self):
        with pytest.raises(ValidationError, match="float range"):
            factor_exposure_budget({"A": 1e300, "B": 1e300}, LOADINGS * 1e10)

    def test_a_gross_exposure_past_the_float_range_is_refused(self):
        """The exposures cancel to zero; the gross does not."""
        loadings = pd.DataFrame({"mkt": [1.0, 1.0]}, index=["A", "B"])
        with pytest.raises(ValidationError, match="gross exposure"):
            factor_exposure_budget({"A": 1.7e308, "B": -1.7e308}, loadings)

    def test_an_ordinary_book_is_the_weighted_loadings(self):
        """The null case."""
        result = factor_exposure_budget(WEIGHTS, LOADINGS)
        assert result["exposures"]["mkt"] == pytest.approx(0.5 + 0.24 + 0.24)
        assert result["exposures"]["size"] == pytest.approx(0.1 - 0.12)
        assert result["gross_exposure"] == pytest.approx(1.0)
        assert all(math.isfinite(v) for v in result["exposures"].values())


class TestConcentrationAnalysis:
    def test_a_nan_weight_is_refused_not_dropped(self):
        """It used to vanish, so the book described was not the book given."""
        with pytest.raises(ValidationError, match=r"weights is not finite at \['A'\]"):
            concentration_analysis({"A": NAN, "B": 0.5, "C": 0.5})

    @pytest.mark.parametrize("bad", [INF, -INF])
    def test_an_infinite_weight_is_refused_not_divided_by(self, bad):
        with pytest.raises(ValidationError, match=r"not finite at \['A'\]"):
            concentration_analysis({"A": bad, "B": 0.5})

    def test_a_gross_exposure_past_the_float_range_is_refused(self):
        """Each weight finite, their sum not."""
        with pytest.raises(ValidationError, match="float range"):
            concentration_analysis({"A": 1.5e308, "B": 1.5e308})

    def test_a_repeated_name_is_refused(self):
        with pytest.raises(ValidationError, match=r"\['A'\] appear more than once"):
            concentration_analysis(pd.Series([0.5, 0.5], index=["A", "A"]))

    def test_an_ordinary_book_is_measured(self):
        """The null case."""
        result = concentration_analysis({"A": 0.5, "B": 0.3, "C": -0.2})
        assert result["gross_exposure"] == pytest.approx(1.0)
        assert result["herfindahl"] == pytest.approx(0.25 + 0.09 + 0.04)
        assert result["is_long_short"] is True


COV = [[0.04, 0.01], [0.01, 0.09]]
TWICE = pd.DataFrame(COV, index=["A", "A"], columns=["A", "A"])
NAMED = pd.DataFrame(COV, index=["A", "B"], columns=["A", "B"])


class TestACovarianceNamesEachAssetOnce:
    @pytest.mark.parametrize(
        "call",
        [
            lambda cov: risk_parity(cov),
            lambda cov: max_diversification(cov),
            lambda cov: marginal_risk_contribution({"A": 0.5}, cov),
            lambda cov: portfolio_scenarios(
                {"A": 0.5}, {"crash": {"A": -0.2}}, covariance=cov
            ),
            lambda cov: factor_exposure_budget(
                {"A": 1.0},
                pd.DataFrame({"A": [1.0], "B": [0.5]}, index=["A"]),
                factor_covariance=cov,
            ),
        ],
        ids=[
            "risk_parity",
            "max_diversification",
            "marginal_risk_contribution",
            "portfolio_scenarios",
            "factor_exposure_budget",
        ],
    )
    def test_a_repeated_asset_is_refused_by_name(self, call):
        """Risk parity over ["A", "A"] used to answer {"A": 0.4}: two rows
        under one key, a book whose weights summed to 0.4."""
        with pytest.raises(
            ValidationError, match=r"\['A'\] appear more than once in the covariance"
        ):
            call(TWICE)

    def test_hrp_refuses_a_repeated_return_column(self):
        """The name sort then selected both columns twice."""
        returns = pd.DataFrame(
            [[0.01, 0.02], [-0.01, 0.0], [0.02, -0.01]], columns=["A", "A"]
        )
        with pytest.raises(ValidationError, match=r"\['A'\] appear more than once"):
            hierarchical_risk_parity(returns)

    def test_distinct_names_give_a_whole_book(self):
        """The null case: the weights sum to one, one per asset."""
        weights = risk_parity(NAMED)["weights"]
        assert set(weights) == {"A", "B"}
        assert sum(weights.values()) == pytest.approx(1.0)


class TestMarginalRiskContribution:
    @pytest.mark.parametrize("bad", [NAN, INF, -INF])
    def test_a_non_finite_weight_is_refused_by_name(self, bad):
        """NaN made the volatility and every row NaN; inf an infinite
        volatility over NaN contributions."""
        with pytest.raises(ValidationError, match=r"weights is not finite at \['A'\]"):
            marginal_risk_contribution({"A": bad, "B": 0.5}, NAMED)

    def test_a_variance_past_the_float_range_is_refused(self):
        """Every marginal used to come back 0.0 under an infinite volatility."""
        with pytest.raises(ValidationError, match="portfolio variance"):
            marginal_risk_contribution({"A": 1e200, "B": 0.5}, NAMED)

    def test_contributions_sum_to_the_volatility(self):
        """The null case."""
        result = marginal_risk_contribution({"A": 0.6, "B": 0.4}, NAMED)
        assert result["sum_of_contributions"] == pytest.approx(
            result["portfolio_volatility"]
        )


class TestPortfolioScenarios:
    SCENARIOS = {"crash": {"A": -0.2, "B": -0.1}}

    def test_a_nan_weight_is_refused_not_dropped(self):
        """It used to vanish, so the scenario described a smaller book."""
        with pytest.raises(ValidationError, match=r"weights is not finite at \['A'\]"):
            portfolio_scenarios({"A": NAN, "B": 0.5}, self.SCENARIOS)

    @pytest.mark.parametrize("bad", [INF, -INF])
    def test_an_infinite_weight_is_refused(self, bad):
        with pytest.raises(ValidationError, match=r"not finite at \['A'\]"):
            portfolio_scenarios({"A": bad, "B": 0.5}, self.SCENARIOS)

    def test_a_non_finite_shock_is_refused_naming_the_scenario(self):
        """A NaN shock made its scenario's return NaN, sorted anywhere."""
        with pytest.raises(ValidationError, match=r"scenario 'crash' is not finite"):
            portfolio_scenarios({"A": 0.5, "B": 0.5}, {"crash": {"A": NAN}})

    def test_a_return_past_the_float_range_is_refused(self):
        with pytest.raises(ValidationError, match="more than the float range"):
            portfolio_scenarios(
                {"A": 1e300, "B": 1e300}, {"up": {"A": 1e10, "B": 1e10}}
            )

    def test_a_repeated_weight_is_refused(self):
        with pytest.raises(ValidationError, match=r"\['A'\] appear more than once"):
            portfolio_scenarios(pd.Series([0.5, 0.5], index=["A", "A"]), self.SCENARIOS)

    def test_an_ordinary_book_is_shocked(self):
        """The null case."""
        result = portfolio_scenarios(
            {"A": 0.5, "B": 0.5}, self.SCENARIOS, covariance=NAMED
        )
        assert result["worst_scenario"]["portfolio_return"] == pytest.approx(-0.15)
        assert math.isfinite(result["worst_scenario"]["sigma_move"])
