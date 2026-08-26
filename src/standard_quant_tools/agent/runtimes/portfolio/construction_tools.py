"""
Allocation that does not lean on expected returns, and what a portfolio is
actually exposed to.

`run_portfolio_optimization` maximizes a mean-variance objective, which is
the right tool when you genuinely have return forecasts and the wrong one
otherwise -- mean-variance puts weight exactly where estimation error is
most likely to have put it. Everything here either avoids the noisiest input
entirely (risk parity, HRP) or answers a question the optimizer never asks:
what is this portfolio a bet ON, how concentrated is it really, and what
does the risk look like once you admit you cannot exit at the mark.

INPUTS ARE INLINE. A covariance matrix arrives as a nested list with names,
returns as a map of asset to series. Nothing here fetches prices -- these
consume a matrix the caller already estimated, which matters because the
estimator IS a choice (`estimate_covariance` offers shrinkage for a reason)
and burying it inside an allocator would hide it.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated, Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from standard_quant_tools.error import ValidationError
from standard_quant_tools.portfolio import construction as lib

logger = logging.getLogger(__name__)


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


class _Result(BaseModel):
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(default_factory=list)


def _square_frame(
    matrix: List[List[float]], assets: List[str], who: str
) -> pd.DataFrame:
    """A named covariance matrix, refusing a shape mismatch by name."""
    if len(matrix) != len(assets):
        raise ValidationError(
            f"{who}: {len(matrix)} covariance rows against {len(assets)} "
            "asset names. Every row needs a name -- an unnamed weight cannot "
            "be acted on."
        )
    for i, row in enumerate(matrix):
        if len(row) != len(assets):
            raise ValidationError(
                f"{who}: covariance row {i} has {len(row)} entries for "
                f"{len(assets)} assets."
            )
    return pd.DataFrame(matrix, index=assets, columns=assets)


# ── inputs ──────────────────────────────────────────────────────────────


class RiskParityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assets: List[str] = Field(..., min_length=2, description="Asset names.")
    covariance: List[List[float]] = Field(
        ...,
        description="Square covariance matrix, rows parallel to `assets`. "
        "Annualized or not -- the weights are scale-invariant either way.",
    )
    budget: Optional[List[float]] = Field(
        None,
        description="RISK budget per asset, e.g. [0.5, 0.3, 0.2]. Normalized "
        "internally. Omit for equal contribution. Most real mandates are "
        "stated this way rather than as equal risk.",
    )
    max_iterations: int = Field(5000, ge=1, le=100000)
    tolerance: float = Field(1e-10, gt=0)


class HRPInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: Dict[str, List[float]] = Field(
        ...,
        description="Asset name -> return series. All the same length. "
        "Returns rather than a covariance matrix, because HRP needs the "
        "correlation structure to build its tree.",
    )


class FactorExposureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: Dict[str, float] = Field(..., description="Asset -> portfolio weight.")
    factor_loadings: Dict[str, Dict[str, float]] = Field(
        ..., description="Asset -> {factor: loading}."
    )
    factor_covariance: Optional[List[List[float]]] = Field(
        None,
        description="Factor covariance matrix, in the order the factors "
        "first appear in factor_loadings. WITHOUT it only exposures are "
        "reported, not risk -- a large loading on a quiet factor is not a "
        "large risk.",
    )
    factors: Optional[List[str]] = Field(
        None, description="Explicit factor order for factor_covariance."
    )


class ConcentrationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: Dict[str, float] = Field(
        ...,
        description="Asset -> weight. Signed: a long-short book is measured "
        "on GROSS weights, since net would make the denominator near-zero.",
    )


class LiquidityVarInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positions: Dict[str, float] = Field(
        ..., description="Asset -> position value in currency."
    )
    volatilities: Dict[str, float] = Field(
        ..., description="Asset -> ANNUALIZED volatility as a decimal."
    )
    daily_volumes: Dict[str, float] = Field(
        ...,
        description="Asset -> average daily volume in currency. Required: "
        "without it the liquidation horizon is unknown, which is the "
        "entire question.",
    )
    confidence: float = Field(0.95, gt=0, lt=1)
    participation_rate: float = Field(
        0.15,
        gt=0,
        le=1,
        description="Fraction of daily volume you are willing to be. Lower "
        "means a longer horizon and more risk.",
    )
    correlation: float = Field(
        0.0,
        ge=-1,
        le=1,
        description="Assumed correlation between positions. At 0 risks add "
        "in quadrature; at 1 they add linearly, which is the crisis case -- "
        "and crisis is when liquidation horizons matter.",
    )


# ── results ─────────────────────────────────────────────────────────────


class RiskParityResult(_Result):
    n_assets: int = 0
    assets: List[str] = Field(default_factory=list)
    weights: Dict[str, float] = Field(default_factory=dict)
    risk_contributions: Dict[str, float] = Field(default_factory=dict)
    risk_shares: Dict[str, float] = Field(
        default_factory=dict,
        description="Each asset's share of total portfolio volatility. These "
        "are what is being equalized, NOT the weights.",
    )
    target_shares: Dict[str, float] = Field(default_factory=dict)
    portfolio_volatility: Stat = None
    converged: bool = False
    iterations: int = 0
    max_share_error: Stat = None


class HRPResult(_Result):
    n_assets: int = 0
    n_observations: int = 0
    weights: Dict[str, float] = Field(default_factory=dict)
    cluster_order: List[str] = Field(
        default_factory=list,
        description="Assets reordered so similar ones sit adjacent. The "
        "grouping the tree found.",
    )
    risk_contributions: Dict[str, float] = Field(default_factory=dict)
    portfolio_volatility: Stat = None
    effective_n: Stat = None


class NamedExposure(BaseModel):
    model_config = ConfigDict(extra="allow")

    factor: str = ""
    exposure: Stat = None


class FactorExposureResult(_Result):
    n_positions: int = 0
    n_unmapped: int = 0
    unmapped: List[str] = Field(default_factory=list)
    n_factors: int = 0
    exposures: Dict[str, float] = Field(default_factory=dict)
    largest_exposures: List[NamedExposure] = Field(default_factory=list)
    factor_variance: Stat = None
    factor_variance_shares: Optional[Dict[str, float]] = Field(
        None,
        description="Each factor's share of portfolio variance. This is the "
        "number that answers 'what am I taking risk on'. Null when no factor "
        "covariance was supplied.",
    )
    gross_exposure: Stat = None
    net_exposure: Stat = None


class LargestPosition(BaseModel):
    model_config = ConfigDict(extra="allow")

    asset: str = ""
    weight: Stat = None
    gross_share: Stat = None


class ConcentrationResult(_Result):
    n_positions: int = 0
    gross_exposure: Stat = None
    net_exposure: Stat = None
    is_long_short: bool = False
    herfindahl: Stat = None
    effective_n: Stat = Field(
        None,
        description="The count of EQUALLY WEIGHTED positions with the same "
        "concentration. A 100-name book with an effective N of 12 has the "
        "concentration of 12. This is the number to quote.",
    )
    concentration_ratio: Stat = None
    top_1_share: Stat = None
    top_3_share: Stat = None
    top_5_share: Stat = None
    top_10_share: Stat = None
    largest_positions: List[LargestPosition] = Field(default_factory=list)


class PositionVar(BaseModel):
    model_config = ConfigDict(extra="allow")

    asset: str = ""
    position_value: Stat = None
    annual_volatility: Stat = None
    liquidation_days: Stat = None
    naive_1d_var: Stat = None
    liquidity_adjusted_var: Stat = None
    adjustment_multiple: Stat = None
    expected_liquidation_cost: Stat = None


class LiquidityVarResult(_Result):
    n_positions: int = 0
    confidence: Stat = None
    participation_rate: Stat = None
    assumed_correlation: Stat = None
    naive_var: Stat = None
    liquidity_adjusted_var: Stat = None
    adjustment_multiple: Stat = None
    expected_liquidation_cost: Stat = Field(
        None,
        description="Reported SEPARATELY from the VaR. Cost is an "
        "expectation and VaR is a quantile; adding them gives neither.",
    )
    worst_position: Optional[PositionVar] = None
    by_position: List[PositionVar] = Field(default_factory=list)


# ── tools ───────────────────────────────────────────────────────────────


def optimize_risk_parity(input_data: RiskParityInput) -> RiskParityResult:
    return RiskParityResult(
        **lib.risk_parity(
            _square_frame(
                input_data.covariance, input_data.assets, "optimize_risk_parity"
            ),
            max_iterations=input_data.max_iterations,
            tolerance=input_data.tolerance,
            budget=input_data.budget,
        )
    )


def optimize_hierarchical_risk_parity(input_data: HRPInput) -> HRPResult:
    lengths = {name: len(values) for name, values in input_data.returns.items()}
    if len(set(lengths.values())) > 1:
        raise ValidationError(
            "optimize_hierarchical_risk_parity: the return series have "
            f"different lengths ({lengths}). They must be aligned in time."
        )
    return HRPResult(**lib.hierarchical_risk_parity(pd.DataFrame(input_data.returns)))


def get_factor_exposure_budget(
    input_data: FactorExposureInput,
) -> FactorExposureResult:
    loadings = pd.DataFrame(input_data.factor_loadings).T
    if input_data.factors:
        missing = [f for f in input_data.factors if f not in loadings.columns]
        if missing:
            raise ValidationError(
                f"get_factor_exposure_budget: factors {missing} do not appear "
                f"in the loadings. Available: {list(loadings.columns)}."
            )
        loadings = loadings[input_data.factors]
    covariance = None
    if input_data.factor_covariance is not None:
        covariance = _square_frame(
            input_data.factor_covariance,
            [str(c) for c in loadings.columns],
            "get_factor_exposure_budget",
        )
    return FactorExposureResult(
        **lib.factor_exposure_budget(
            input_data.weights, loadings, factor_covariance=covariance
        )
    )


def analyze_concentration(input_data: ConcentrationInput) -> ConcentrationResult:
    return ConcentrationResult(**lib.concentration_analysis(input_data.weights))


def get_liquidity_adjusted_var(input_data: LiquidityVarInput) -> LiquidityVarResult:
    return LiquidityVarResult(
        **lib.liquidity_adjusted_var(
            input_data.positions,
            input_data.volatilities,
            input_data.daily_volumes,
            confidence=input_data.confidence,
            participation_rate=input_data.participation_rate,
            correlation=input_data.correlation,
        )
    )


CONSTRUCTION_TOOL_DEFS = [
    (
        "optimize_risk_parity",
        "Weights at which every asset contributes the SAME AMOUNT OF RISK -- "
        "not the same weight. An equally weighted portfolio of a bond fund "
        "and a biotech stock is a biotech portfolio; the equity contributes "
        "nearly all the variance. Uses NO expected returns, which is the "
        "reason to prefer it over mean-variance in most real situations: the "
        "standard error on a mean return from two years of daily data is "
        "about the size of the estimate, and mean-variance is maximally "
        "sensitive to exactly that input. Pass `budget` for an unequal risk "
        "budget, which is how most mandates are actually written.",
        RiskParityInput,
    ),
    (
        "optimize_hierarchical_risk_parity",
        "Allocation that never INVERTS the covariance matrix. Inversion is "
        "where an ill-conditioned estimate does its damage -- the smallest "
        "eigenvalue becomes the largest, so the direction the data says least "
        "about becomes the one the portfolio bets most on, and with 50 assets "
        "on 500 observations that eigenvalue is noise. HRP clusters by "
        "correlation, orders the assets so similar ones sit adjacent, and "
        "splits capital down the tree. It has NO optimality property and does "
        "not maximize anything; it buys robustness by giving that up.",
        HRPInput,
    ),
    (
        "get_factor_exposure_budget",
        "What the portfolio is actually betting on, once the names collapse "
        "into factors. Answers the failure that sinks more portfolios than "
        "any optimizer: 'I hold 40 names so I am diversified' -- 40 names "
        "with the same loading are one position with extra transaction costs. "
        "Supply factor_covariance to get each factor's share of portfolio "
        "VARIANCE, which is the number that answers what you are taking risk "
        "on; without it only exposures can be reported, and a large loading "
        "on a quiet factor is not a large risk.",
        FactorExposureInput,
    ),
    (
        "analyze_concentration",
        "How concentrated a portfolio is, in numbers with known "
        "interpretations. Effective N is the count of equally weighted "
        "positions that would give the same concentration -- a 100-position "
        "book with an effective N of 12 holds 100 names and has the "
        "concentration of 12, and that is the number to quote. Long-short "
        "books are measured on GROSS weights, because net makes the "
        "denominator near-zero and the shares meaningless.",
        ConcentrationInput,
    ),
    (
        "get_liquidity_adjusted_var",
        "VaR that accounts for not being able to exit at the mark. A 1-day "
        "95% VaR describes a position you could close today; one that takes "
        "15 days to liquidate at a sane participation rate is exposed for 15 "
        "days and carries roughly sqrt(15) times the risk -- a factor of "
        "four, and the part usually missed. The liquidation COST is reported "
        "separately from the quantile on purpose: cost is an expectation and "
        "VaR is a quantile, and adding them produces a number that is "
        "neither.",
        LiquidityVarInput,
    ),
]

CONSTRUCTION_TOOL_DISPATCH = {
    "optimize_risk_parity": (optimize_risk_parity, RiskParityInput),
    "optimize_hierarchical_risk_parity": (
        optimize_hierarchical_risk_parity,
        HRPInput,
    ),
    "get_factor_exposure_budget": (get_factor_exposure_budget, FactorExposureInput),
    "analyze_concentration": (analyze_concentration, ConcentrationInput),
    "get_liquidity_adjusted_var": (get_liquidity_adjusted_var, LiquidityVarInput),
}

__all__ = [
    "CONSTRUCTION_TOOL_DEFS",
    "CONSTRUCTION_TOOL_DISPATCH",
    "analyze_concentration",
    "get_factor_exposure_budget",
    "get_liquidity_adjusted_var",
    "optimize_hierarchical_risk_parity",
    "optimize_risk_parity",
]
