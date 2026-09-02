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
    "BasisDislocationResult",
    "SpreadMonitorResult",
    "BasisHistoryResult",
    "CashFuturesBasisResult",
    "CompareExpressionsResult",
    "DividendPointsResult",
    "EtfFairValueResult",
    "FuturesCurveResult",
    "FuturesHedgeResult",
    "HedgeEffectivenessResult",
    "IndexBasketResult",
    "IndexRebalanceResult",
    "ReplicationBasketResult",
    "RollAnalysisResult",
    "SolveForwardCarryResult",
    "Stat",
    "TotalReturnFutureResult",
    "TotalReturnSwapResult",
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


# ── Phase II: the desk instruments ──────────────────────────────────────


class ReplicationBasketResult(_Result):
    n_candidates: int = 0
    n_observations: int = 0
    n_selected: int = 0
    max_names: Optional[int] = None
    long_only: bool = True
    covariance_method: str = ""
    weights: Dict[str, Stat] = Field(
        default_factory=dict, description="Only the names actually held."
    )
    selected_names: List[str] = Field(default_factory=list)
    predicted_tracking_error: Stat = Field(
        None, description="Annualized, from the covariance."
    )
    realized_tracking_error: Stat = Field(
        None,
        description="Annualized, from the actual series. IN SAMPLE -- the "
        "weights were fitted on the window they are scored on.",
    )
    correlation: Stat = None
    beta: Stat = None
    gross_weight: Stat = None
    net_weight: Stat = Field(
        None, description="Should be 1.0; a deviation is reported."
    )
    largest_weight: Stat = None
    periods_per_year: int = 252


class EtfFairValueResult(_Result):
    etf_price: Stat = None
    nav: Stat = None
    nav_is_intraday: bool = False
    premium_discount: Stat = Field(None, description="Currency, per share.")
    premium_discount_pct: Stat = None
    premium_discount_bps: Stat = None
    classification: str = Field("", description="'premium', 'discount' or 'fair'.")
    tolerance_bps: Stat = None
    basket_value_per_share: Stat = None
    basket_vs_nav_bps: Stat = Field(
        None,
        description="The creation basket against the fund's own stated value. "
        "Null when no basket was supplied.",
    )
    gross_arbitrage_bps: Stat = None
    execution_bps: Stat = Field(None, description="ROUND TRIP, both legs.")
    creation_fee_bps: Stat = None
    net_arbitrage_bps: Stat = Field(
        None, description="What survives costs. The number that decides anything."
    )
    arbitrage_survives: bool = False
    action: Optional[str] = Field(
        None, description="'create' on a premium, 'redeem' on a discount."
    )


class TotalReturnSwapResult(_Result):
    direction: str = ""
    notional: Stat = None
    initial_price: Stat = None
    current_price: Stat = None
    time_elapsed_years: Stat = None
    day_count: str = ""
    price_return: Stat = None
    dividend_return: Stat = None
    total_return: Stat = None
    financing_rate: Stat = None
    spread_bps: Stat = None
    all_in_financing_rate: Stat = None
    financing_accrued: Stat = Field(None, description="As a rate over the period.")
    equity_leg: Stat = Field(None, description="Currency.")
    financing_leg: Stat = Field(None, description="Currency. Negative to a receiver.")
    net_pnl: Stat = None
    net_return_on_notional: Stat = None


class TotalReturnFutureResult(_Result):
    quote: Stat = None
    quote_convention: str = ""
    convention_meaning: str = ""
    underlying_price: Stat = None
    time_to_expiry: Stat = None
    reference_rate: Stat = None
    implied_spread_bps: Stat = Field(
        None,
        description="The financing spread the quote implies. CONDITIONAL on "
        "the reference rate given, which it absorbs one-for-one.",
    )
    implied_level: Stat = None
    all_in_financing_rate: Stat = None
    net_carry_rate: Stat = None
    comparison_spread_bps: Stat = None
    difference_bps: Stat = Field(
        None, description="Null unless a comparison spread was supplied."
    )


class DividendContribution(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str = ""
    ex_date: str = ""
    dividend_per_share: Stat = None
    index_points: Stat = None


class DividendPointsResult(_Result):
    as_of: str = ""
    expiry: str = ""
    divisor: Stat = None
    n_constituents: int = 0
    n_included: int = 0
    n_excluded_already_ex: int = 0
    n_excluded_after_expiry: int = 0
    total_index_points: Stat = Field(
        None, description="INDEX POINTS, not a yield. That is the whole point."
    )
    points_by_date: Dict[str, Stat] = Field(default_factory=dict)
    points_by_constituent: List[DividendContribution] = Field(default_factory=list)
    largest_contributor: Optional[str] = None
    implied_index_points: Stat = Field(
        None, description="What the quoted future implies. Null without one."
    )
    forecast_minus_implied: Stat = Field(
        None,
        description="The tradeable quantity. A gap is usually a position "
        "rather than an error on either side.",
    )


class RebalanceChange(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str = ""
    event: str = Field("", description="addition, deletion, increase or decrease.")
    old_weight: Stat = None
    new_weight: Stat = None
    weight_change: Stat = None
    side: str = ""
    notional: Stat = Field(None, description="Signed currency.")
    adv: Stat = None
    days_of_adv: Stat = Field(
        None, description="Null when no ADV was supplied for this name."
    )
    auction_participation: Stat = Field(
        None, description="Flow as a multiple of the auction's own volume."
    )


class IndexRebalanceResult(_Result):
    indexed_assets: Stat = None
    auction_fraction: Stat = None
    n_changes: int = 0
    n_additions: int = 0
    n_deletions: int = 0
    buy_notional: Stat = None
    sell_notional: Stat = None
    net_notional: Stat = None
    gross_notional: Stat = None
    turnover_pct: Stat = Field(
        None, description="One-way, as index providers quote it."
    )
    changes: List[RebalanceChange] = Field(
        default_factory=list, description="Sorted by absolute notional."
    )
    hardest_to_trade: List[str] = Field(
        default_factory=list, description="By days of ADV, worst first."
    )
    largest_flow: Optional[str] = None


class BasisBreak(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: Optional[int] = None
    gain: Stat = None
    mean_before: Stat = None
    mean_after: Stat = None
    std_before: Stat = None
    std_after: Stat = None


class BasisDislocationResult(_Result):
    n_observations: int = 0
    units: str = Field("", description="'bps of spot' or 'annualized bps'.")
    current: Stat = None
    triggered: bool = False
    first_crossing: Optional[int] = Field(
        None, description="Observation index, or null if it never crossed."
    )
    peak_statistic: Stat = Field(
        None, description="Accumulated standardized deviations, at its highest."
    )
    peak_at: Optional[int] = None
    direction: Optional[str] = None
    severity: Optional[str] = None
    baseline_mean: Stat = None
    baseline_std: Stat = None
    degenerate_baseline: bool = Field(
        False,
        description="The warm-up saw no variation, so any statistic here "
        "is arithmetic rather than evidence -- it divides by almost "
        "nothing. A stale or halted feed through the warm-up does this.",
    )
    mean_after_reference: Stat = Field(
        None,
        description="The before-and-after a reader can check. A peak "
        "statistic is accumulated and unbounded; this is not.",
    )
    shift: Stat = None
    shift_in_reference_sd: Stat = None
    n_reference: Optional[int] = None
    threshold: Stat = None
    n_breaks: int = 0
    breaks: List[BasisBreak] = Field(default_factory=list)


class SpreadAlert(BaseModel):
    model_config = ConfigDict(extra="allow")

    observation: Optional[int] = None
    value: Stat = None
    statistic: Stat = None
    direction: Optional[str] = None
    baseline_mean: Stat = None
    baseline_std: Stat = None
    shift_in_baseline_sd: Stat = None
    channel: str = ""
    degenerate_baseline: bool = False
    message: str = ""


class SpreadMonitorResult(_Result):
    state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pass THIS back on the next call. JSON-safe, so a "
        "monitor can be paused, moved between processes and resumed without "
        "losing its baseline.",
    )
    channel: str = Field("", description="Which formula combined the two legs.")
    alert: Optional[SpreadAlert] = Field(
        None,
        description="Set once, on the update that crosses. It does not "
        "re-fire while a dislocation persists -- a monitor that alerts every "
        "tick is a monitor somebody turns off.",
    )
    triggered: bool = False
    warming_up: bool = False
    n_observations: int = 0
    current_value: Stat = None
    statistic: Stat = None
    peak_statistic: Stat = None
    threshold: Stat = None
    baseline_mean: Stat = Field(None, description="Null until the warm-up completes.")
    baseline_std: Stat = None
