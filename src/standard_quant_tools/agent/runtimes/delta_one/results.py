"""
Typed results for the delta_one runtime.

WHY THESE EXIST rather than returning the library's dicts directly. The MCP
server declares a structured-output schema per tool, built from the return
annotation, and a tool without one silently stops being able to describe
its own output -- a client then receives JSON it has no schema for and an
agent has to guess at key names. On this surface the schema is carrying
real information: an agent reading `basis` needs to know it is in POINTS of
the underlying while `annualized_basis_spread_bps` is a RATE, and that the
two are not convertible without the time to expiry.

EVERY NUMERIC FIELD IS `Stat`, which maps a non-finite value to null. JSON
has no NaN or Infinity literal, so emitting one produces a document a
strict parser rejects -- and several quantities here are legitimately
undefined rather than zero: a half-life on a basis that does not revert, a
basis against an index level nobody supplied, a break-even on a position
with no notional. `None` says "not defined"; 0.0 would say something false
and tradeable.

WARNINGS ARE A DECLARED FIELD, not an afterthought. Most of what these
tools know that a caller does not is in there -- that a wide basis is
usually a stale print rather than an arbitrage, that roll yield is not a
return, that an omitted expense ratio makes an ETF look free -- and a
schema that omitted it would train agents to ignore it.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

__all__ = [
    "BasisHistoryResult",
    "CashFuturesBasisResult",
    "CompareExpressionsResult",
    "FuturesCurveResult",
    "FuturesHedgeResult",
    "HedgeEffectivenessResult",
    "IndexBasketResult",
    "RollAnalysisResult",
    "SolveForwardCarryResult",
    "Stat",
]


def _finite_or_none(value: Any) -> Any:
    """
    Non-finite in, null out.

    Applied before validation so a NaN never reaches the serializer. The
    alternative -- letting it through and relying on the JSON encoder --
    produces `NaN` in the payload, which is not valid JSON and which
    several MCP clients reject at the transport layer rather than at the
    tool.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


#: Copied rather than imported. Five other modules carry their own; a
#: shared one would become the place cross-runtime coupling accumulates.
Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


class _Result(BaseModel):
    """Shared base: every result carries its own caveats."""

    # extra="allow" because every wrapper does Result(**lib.fn(...)) and
    # the library layer owns the shape of that dict.
    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(
        default_factory=list,
        description="What this result knows that the numbers do not say. "
        "Not decorative -- these carry the conditions under which the "
        "figures above are wrong.",
    )


class CashFuturesBasisResult(_Result):
    spot: Stat = None
    future: Stat = None
    fair_future: Stat = Field(None, description="Carry-fair forward, S*exp((r-q-b)T).")
    observed_basis_points: Stat = Field(
        None, description="future - spot, in POINTS of the underlying."
    )
    fair_basis_points: Stat = None
    basis_spread_points: Stat = Field(
        None, description="future - fair_future. The mispricing, in points."
    )
    annualized_basis_spread_bps: Stat = Field(
        None,
        description="The same mispricing as an annualized RATE. This is the "
        "figure comparable across expiries; points are not.",
    )
    observed_carry_rate: Stat = None
    fair_carry_rate: Stat = None
    classification: str = Field(
        "", description="'future_rich', 'future_cheap' or 'fair'."
    )
    tolerance_bps: Stat = None
    implied_financing_rate: Stat = Field(
        None,
        description="The funding rate that would make the quote fair, given "
        "your dividend and borrow. Compare to SOFR before calling it edge.",
    )
    implied_financing_bps: Stat = None
    components: Dict[str, Stat] = Field(
        default_factory=dict,
        description="Financing, dividend and borrow contributions to the "
        "fair forward, in POINTS.",
    )
    time_to_expiry: Stat = None


class SolveForwardCarryResult(_Result):
    solved_for: str = ""
    solved_rate: Stat = Field(None, description="As a decimal (0.043 = 4.3%).")
    solved_rate_bps: Stat = None
    net_carry_rate: Stat = Field(None, description="ln(F/S)/T, the whole left side.")
    spot: Stat = None
    forward: Stat = None
    time_to_expiry: Stat = None
    assumed: Dict[str, Stat] = Field(
        default_factory=dict,
        description="The two rates supplied. The answer is CONDITIONAL on "
        "them and absorbs every error in both.",
    )
    meaning: str = ""


class BasisHistoryResult(_Result):
    n_observations: int = 0
    current_basis_bps: Stat = None
    current_basis_points: Stat = None
    mean_bps: Stat = None
    std_bps: Stat = None
    min_bps: Stat = None
    max_bps: Stat = None
    zscore: Stat = Field(
        None, description="Full-sample unless `window` was given; see warnings."
    )
    percentile: Stat = Field(None, description="0-100, where the latest sits.")
    half_life_observations: Stat = Field(
        None,
        description="In OBSERVATIONS, not days. Null when the series shows "
        "no mean reversion, which is a refusal rather than a slow one.",
    )
    annualized: bool = False
    window: Optional[int] = None


class CurvePoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = ""
    time_to_expiry: Stat = None
    price: Stat = None
    basis_points: Stat = None
    implied_carry_rate: Stat = None
    annualized_basis_bps: Stat = None


class CalendarSpread(BaseModel):
    model_config = ConfigDict(extra="allow")

    near: str = ""
    far: str = ""
    calendar_spread_points: Stat = None
    years_between: Stat = None
    forward_carry_rate: Stat = None
    forward_carry_bps: Stat = Field(
        None,
        description="What the calendar spread actually prices -- NOT the far "
        "contract's own carry.",
    )


class FuturesCurveResult(_Result):
    n_contracts: int = 0
    shape: str = Field(
        "", description="'contango', 'backwardation' or 'mixed'. PRICE curve."
    )
    spot: Stat = None
    curve: List[CurvePoint] = Field(default_factory=list)
    calendar_spreads: List[CalendarSpread] = Field(default_factory=list)
    curve_slope_rate: Stat = None
    curve_curvature: Stat = Field(
        None, description="Null with fewer than three contracts -- no bend to measure."
    )
    front_label: Optional[str] = None
    back_label: Optional[str] = None


class RollAnalysisResult(_Result):
    front_price: Stat = None
    next_price: Stat = None
    roll_spread_points: Stat = None
    contracts_held: Stat = None
    next_contracts_exact: Stat = Field(
        None, description="Sized to hold the same MONEY, not the same count."
    )
    next_contracts_rounded: Stat = None
    front_notional: Stat = None
    cash_impact: Stat = Field(
        None, description="Currency. Negative is a cost to this position."
    )
    execution_cost: Stat = None
    net_roll_cost: Stat = None
    net_roll_cost_bps: Stat = None
    roll_yield_rate: Stat = None
    roll_yield_bps: Stat = Field(
        None,
        description="A price STEP as a rate, not a return. What the position "
        "must overcome, not what it earns.",
    )
    days_to_front_expiry: Stat = None
    breakeven_annualized_rate: Stat = None


class FuturesHedgeResult(_Result):
    objective: str = ""
    objective_meaning: str = ""
    portfolio_value: Stat = None
    portfolio_beta: Stat = None
    dollar_beta: Stat = Field(None, description="portfolio_value * portfolio_beta.")
    future_price: Stat = None
    multiplier: Stat = None
    future_beta: Stat = None
    contract_notional: Stat = None
    exposure_per_contract: Stat = None
    contracts_exact: Stat = Field(None, description="Unrounded. Negative is short.")
    contracts_rounded: Stat = None
    existing_contracts: Stat = None
    trade_contracts_exact: Stat = None
    trade_contracts_rounded: Stat = Field(
        None, description="What to actually send, net of what is already held."
    )
    hedge_notional: Stat = None
    pre_hedge_beta: Stat = None
    post_hedge_beta: Stat = None
    pre_hedge_dollar_beta: Stat = None
    post_hedge_dollar_beta: Stat = None
    residual_dollar_beta: Stat = Field(
        None,
        description="What rounding leaves unhedged, in currency. The number "
        "that decides whether the hedge is finished.",
    )


class HedgeEffectivenessResult(_Result):
    n_observations: int = 0
    hedge_ratio: Stat = None
    volatility_before: Stat = Field(None, description="Annualized.")
    volatility_after: Stat = None
    volatility_reduction_pct: Stat = None
    beta_before: Stat = None
    beta_after: Stat = None
    r_squared_before: Stat = None
    correlation: Stat = None
    tracking_error: Stat = Field(None, description="Annualized active-return sigma.")
    max_drawdown_before: Stat = None
    max_drawdown_after: Stat = None
    drawdown_reduction_pct: Stat = None
    rolling_hedge_ratio: Optional[Dict[str, Stat]] = Field(
        None,
        description="Mean, std, min, max, range and sign flips of the "
        "rolling ratio. A wide range means the static ratio averaged two "
        "regimes rather than describing either.",
    )
    window: Optional[int] = None


class BasketConstituent(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str = ""
    price: Stat = None
    weight: Stat = None
    value: Stat = None
    index_points: Stat = None
    is_stale: bool = False


class IndexBasketResult(_Result):
    construction: str = Field("", description="'divisor' or 'weighted'.")
    n_constituents: int = 0
    basket_value: Stat = None
    index_level: Stat = None
    spread_points: Stat = None
    spread_bps: Stat = None
    divisor: Stat = None
    constituents: List[BasketConstituent] = Field(
        default_factory=list, description="Sorted by absolute contribution."
    )
    largest_contributor: Optional[str] = None
    stale_symbols: List[str] = Field(default_factory=list)
    missing_symbols: List[str] = Field(default_factory=list)


class PricedExpression(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = ""
    kind: str = ""
    carry_bps: Stat = None
    financing_bps: Stat = None
    borrow_bps: Stat = None
    fee_bps: Stat = None
    dividend_bps: Stat = None
    execution_bps: Stat = Field(
        None, description="Round trip AMORTIZED over the horizon."
    )
    execution_round_trip_bps: Stat = None
    roll_bps: Stat = None
    total_annualized_bps: Stat = None
    cost_over_horizon_bps: Stat = None
    cost_over_horizon_currency: Stat = None
    capital_requirement_pct: Stat = None
    capital_required: Stat = None
    terms_supplied: List[str] = Field(
        default_factory=list,
        description="Which cost terms were actually given. Everything else "
        "defaulted to zero -- check this on any surprisingly cheap row.",
    )


class CompareExpressionsResult(_Result):
    direction: str = ""
    notional: Stat = None
    horizon_years: Stat = None
    n_expressions: int = 0
    expressions: List[PricedExpression] = Field(
        default_factory=list, description="Ranked cheapest first."
    )
    cheapest: Optional[str] = None
    cheapest_total_bps: Stat = None
    dearest: Optional[str] = None
    dearest_total_bps: Stat = None
    spread_bps: Stat = None
    spread_currency_over_horizon: Stat = Field(
        None, description="What choosing wrong costs over the stated horizon."
    )
