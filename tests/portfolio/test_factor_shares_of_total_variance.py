"""
Factor risk is a share of the portfolio's TOTAL variance, or says it is not.

`get_factor_exposure_budget` computed each factor's share of the FACTOR
variance x'Fx and described it as each factor's share of PORTFOLIO
variance. It took no specific risk, so those shares sum to 1 whatever the
names' own risk is, and the warning built on them called any book whose top
factor passed 60% "one bet". Its docstring said the residual was reported;
it was not. See the CHANGELOG entry of 2026-10-22.

THE PLANTED BOOK. Two names, 50/50, each with a beta of 1 to one market
factor at 1% volatility and 40% specific volatility each:

    factor variance    (0.5 + 0.5)^2 * 0.01^2            = 0.0001
    specific variance  0.5^2 * 0.16 + 0.5^2 * 0.16        = 0.08
    total              0.0801
    market's share of the total  0.0001 / 0.0801         = 0.12484%

The old result said 100% and "it is one bet". It is a stock-picking book.
"""

from __future__ import annotations

import pandas as pd
import pydantic
import pytest

from standard_quant_tools.agent.runtimes import resolve
from standard_quant_tools.agent.runtimes.portfolio.construction_tools import (
    FactorExposureInput,
    get_factor_exposure_budget,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio.construction import factor_exposure_budget

WEIGHTS = {"X": 0.5, "Y": 0.5}
LOADINGS = {"X": {"market": 1.0}, "Y": {"market": 1.0}}
FACTOR_COV = [[0.01**2]]
SPECIFIC = {"X": 0.40**2, "Y": 0.40**2}

FACTOR_VARIANCE = 0.0001
SPECIFIC_VARIANCE = 0.08
TOTAL_VARIANCE = 0.0801
MARKET_SHARE = FACTOR_VARIANCE / TOTAL_VARIANCE


def _library(**kwargs):
    loadings = pd.DataFrame(LOADINGS).T
    return factor_exposure_budget(
        WEIGHTS, loadings, factor_covariance=FACTOR_COV, **kwargs
    )


class TestSharesOfTheTotal:
    def test_the_market_is_a_tenth_of_a_percent_of_the_total(self):
        result = _library(specific_variance=SPECIFIC)
        assert result["factor_variance"] == pytest.approx(FACTOR_VARIANCE, rel=1e-12)
        assert result["specific_variance"] == pytest.approx(
            SPECIFIC_VARIANCE, rel=1e-12
        )
        assert result["total_variance"] == pytest.approx(TOTAL_VARIANCE, rel=1e-12)
        assert result["total_variance_shares"]["market"] == pytest.approx(
            0.0012484394506866417, rel=1e-12
        )
        assert result["total_variance_shares"]["market"] == pytest.approx(
            MARKET_SHARE, rel=1e-12
        )
        assert result["specific_variance_share"] == pytest.approx(
            1 - MARKET_SHARE, rel=1e-12
        )
        # Shares of the factor variance are still what they are, and named so.
        assert result["factor_variance_shares"] == {"market": pytest.approx(1.0)}
        assert not any("one bet" in w for w in result["warnings"])

    def test_without_specific_risk_the_factor_shares_are_not_called_the_total(self):
        """The planted book without specific_variance: the factor shares say
        100% market, and the result says that is a share of the factor part
        only rather than calling the book one bet."""
        result = _library()
        assert result["factor_variance_shares"] == {"market": pytest.approx(1.0)}
        for field in (
            "specific_variance",
            "total_variance",
            "total_variance_shares",
            "specific_variance_share",
        ):
            assert result[field] is None, field
        assert not any("one bet" in w for w in result["warnings"])
        (caveat,) = [w for w in result["warnings"] if "not of the portfolio's" in w]
        assert "FACTOR variance" in caveat
        assert "specific_variance" in caveat

    def test_a_factor_dominated_book_is_still_called_one_bet(self):
        """The null case of the warning: at 1% specific volatility the market
        carries 1e-4 / (1e-4 + 2 * 0.25 * 1e-4) = 2/3 of the total."""
        result = _library(specific_variance={"X": 0.01**2, "Y": 0.01**2})
        assert result["total_variance_shares"]["market"] == pytest.approx(2 / 3)
        (bet,) = [w for w in result["warnings"] if "one bet" in w]
        assert "67% of the portfolio's total variance" in bet

    def test_a_held_name_without_specific_variance_is_refused_by_name(self):
        with pytest.raises(ValidationError, match=r"\['Y'\]"):
            _library(specific_variance={"X": 0.16})

    def test_a_name_at_zero_weight_may_be_left_out(self):
        loadings = pd.DataFrame({**LOADINGS, "Z": {"market": 2.0}}).T
        result = factor_exposure_budget(
            {**WEIGHTS, "Z": 0.0},
            loadings,
            factor_covariance=FACTOR_COV,
            specific_variance=SPECIFIC,
        )
        assert result["total_variance"] == pytest.approx(TOTAL_VARIANCE, rel=1e-12)

    @pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
    def test_a_specific_variance_that_is_not_one_is_refused(self, bad):
        with pytest.raises(ValidationError, match="non-negative variance"):
            _library(specific_variance={"X": 0.16, "Y": bad})

    def test_specific_variance_needs_the_factor_covariance(self):
        with pytest.raises(ValidationError, match="without factor_covariance"):
            factor_exposure_budget(
                WEIGHTS, pd.DataFrame(LOADINGS).T, specific_variance=SPECIFIC
            )


class TestTheToolDoor:
    def test_the_tool_reports_the_share_of_the_total(self):
        result = get_factor_exposure_budget(
            FactorExposureInput(
                weights=WEIGHTS,
                factor_loadings=LOADINGS,
                factor_covariance=FACTOR_COV,
                specific_variance=SPECIFIC,
            )
        )
        assert result.total_variance_shares["market"] == pytest.approx(
            MARKET_SHARE, rel=1e-12
        )
        assert result.specific_variance_share == pytest.approx(1 - MARKET_SHARE)

    def test_through_dispatch_without_specific_variance(self):
        out = resolve("portfolio").dispatch(
            "get_factor_exposure_budget",
            {
                "weights": WEIGHTS,
                "factor_loadings": LOADINGS,
                "factor_covariance": FACTOR_COV,
            },
        )
        assert out["factor_variance_shares"] == {"market": pytest.approx(1.0)}
        assert out["total_variance_shares"] is None
        assert not any("one bet" in w for w in out["warnings"])
        assert any("total_variance is null" in w for w in out["warnings"])

    def test_a_non_finite_specific_variance_is_a_schema_refusal(self):
        with pytest.raises(pydantic.ValidationError, match="specific_variance"):
            FactorExposureInput(
                weights=WEIGHTS,
                factor_loadings=LOADINGS,
                factor_covariance=FACTOR_COV,
                specific_variance={"X": 0.16, "Y": float("nan")},
            )
