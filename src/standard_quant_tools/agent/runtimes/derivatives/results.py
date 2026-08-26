"""
Typed results for the derivatives runtime.

WHY THESE EXIST rather than returning the library's dicts directly. The MCP
server declares a structured-output schema per tool, built from the return
annotation, and a tool without one silently stops being able to describe its
own output -- a client then receives JSON it has no schema for and an agent
has to guess at key names. `test_every_tool_has_an_output_schema` pins that,
and it is the right invariant: on this surface especially, an agent reading
`vega` needs to know it is per volatility POINT and not per unit.

EVERY NUMERIC FIELD IS `Stat`, which maps a non-finite value to null. JSON
has no NaN or Infinity literal, so emitting one produces a document that a
strict parser rejects outright -- and several quantities here are legitimately
undefined rather than zero: a forward vol across an arbitrage-violating
interval, a Roll spread on a trending series, an adjustment multiple when the
base VaR is zero. `None` says "not defined"; 0.0 would say something false.

WARNINGS ARE A DECLARED FIELD, not an afterthought. Most of what these tools
know that a caller does not is in there -- that an expected move is not a
bound, that a parity break is usually a stale quote, that spot and vol do not
move independently -- and a schema that omitted it would train agents to
ignore it.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


def _finite_or_none(value: Any) -> Any:
    """
    Non-finite in, null out.

    Applied before validation so a NaN never reaches the serializer. The
    alternative -- letting it through and relying on the JSON encoder --
    produces `NaN` in the payload, which is not valid JSON and which several
    MCP clients reject at the transport layer rather than at the tool.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]


class _Result(BaseModel):
    """Shared base: every result carries its own caveats."""

    model_config = ConfigDict(extra="allow")

    warnings: List[str] = Field(
        default_factory=list,
        description="What this result knows that the numbers do not say. "
        "Not decorative -- these carry the conditions under which the "
        "figures above are wrong.",
    )


class OptionGreeksResult(_Result):
    price: Stat = None
    delta: Stat = None
    gamma: Stat = None
    vega: Stat = Field(None, description="Per 1 volatility POINT (0.01), not per unit.")
    theta: Stat = Field(None, description="Per CALENDAR day.")
    rho: Stat = None
    vanna: Stat = Field(None, description="Change in delta per 1 volatility point.")
    volga: Stat = Field(None, description="Change in vega per 1 volatility point.")
    charm: Stat = Field(None, description="Change in delta per calendar day.")
    speed: Stat = Field(None, description="Change in gamma per $1 of spot.")
    d1: Stat = None
    d2: Stat = None
    moneyness: Stat = None
    units: Dict[str, str] = Field(
        default_factory=dict,
        description="The unit of each greek, stated because there is no "
        "convention and the mismatch is a real source of error.",
    )


class StrategyGreeks(BaseModel):
    model_config = ConfigDict(extra="allow")

    delta: Stat = None
    gamma: Stat = None
    vega: Stat = None
    theta: Stat = None


class PayoffPoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    spot: Stat = None
    profit: Stat = None


class OptionStrategyResult(_Result):
    n_legs: int = 0
    net_premium: Stat = None
    position: str = Field("", description="'debit' if it costs money, else 'credit'.")
    breakevens: List[float] = Field(default_factory=list)
    max_profit: Stat = None
    max_profit_at_spot: Stat = None
    max_profit_unbounded: bool = False
    max_loss: Stat = None
    max_loss_at_spot: Stat = None
    max_loss_unbounded: bool = Field(
        False,
        description="True when the extreme sits at the edge of the scanned "
        "range, which is what an unbounded payoff looks like from inside a "
        "numerical scan. Read the number as unbounded, not as itself.",
    )
    greeks: StrategyGreeks = Field(default_factory=StrategyGreeks)
    payoff_curve: List[PayoffPoint] = Field(default_factory=list)


class SmilePoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    strike: Stat = None
    observed_vol: Stat = None
    fitted_vol: Stat = None


class ArbitrageViolation(BaseModel):
    model_config = ConfigDict(extra="allow")

    log_moneyness: Stat = None
    strike: Stat = None
    durrleman_g: Stat = None
    reason: str = ""


class VolatilitySmileResult(_Result):
    n_strikes: int = 0
    forward: Stat = None
    time_to_expiry: Stat = None
    atm_vol: Stat = Field(None, description="The level: fitted vol at the forward.")
    skew: Stat = Field(
        None,
        description="Slope in LOG-MONEYNESS, so it is comparable across "
        "expiries and underlyings. Negative for a typical equity smile.",
    )
    curvature: Stat = Field(None, description="Convexity; what a butterfly prices.")
    r_squared: Stat = None
    residual_std: Stat = None
    strike_range: List[float] = Field(
        default_factory=list,
        description="The fit does not extrapolate beyond this.",
    )
    arbitrage_violations: List[ArbitrageViolation] = Field(default_factory=list)
    fitted: List[SmilePoint] = Field(default_factory=list)


class ConeRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    horizon_days: int = 0
    n_windows: int = 0
    independent_windows: int = Field(
        0,
        description="Rolling windows overlap; this is the count that "
        "actually supports the percentiles.",
    )
    min: Stat = None
    p10: Stat = None
    p25: Stat = None
    median: Stat = None
    p75: Stat = None
    p90: Stat = None
    max: Stat = None
    current: Stat = None
    implied_vol: Stat = None
    implied_percentile: Stat = None
    implied_vs_median: Stat = None


class VolatilityConeResult(_Result):
    n_returns: int = 0
    cone: List[ConeRow] = Field(default_factory=list)


class ForwardVol(BaseModel):
    model_config = ConfigDict(extra="allow")

    from_expiry: Stat = None
    to_expiry: Stat = None
    spot_vol_near: Stat = None
    spot_vol_far: Stat = None
    forward_variance: Stat = None
    forward_vol: Stat = Field(
        None,
        description="What a calendar spread actually prices. Null when the "
        "forward variance is negative, which is a calendar arbitrage.",
    )


class TermStructurePoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    expiry_years: Stat = None
    implied_vol: Stat = None


class VolTermStructureResult(_Result):
    n_expiries: int = 0
    shape: str = Field("", description="'contango', 'backwardation' or 'flat'.")
    slope: Stat = None
    term_structure: List[TermStructurePoint] = Field(default_factory=list)
    forward_vols: List[ForwardVol] = Field(default_factory=list)
    arbitrage_violations: List[ForwardVol] = Field(default_factory=list)


class PutCallParityResult(_Result):
    call_minus_put: Stat = None
    forward_minus_strike_pv: Stat = None
    violation: Stat = None
    violation_bps_of_strike: Stat = None
    within_tolerance: bool = True
    tolerance_bps: Stat = None
    implied_dividend_yield: Stat = Field(
        None,
        description="The dividend at which parity would hold exactly. An "
        "implausible value identifies the cause of the break.",
    )
    implied_forward: Stat = None


class CarryComponents(BaseModel):
    model_config = ConfigDict(extra="allow")

    financing: Stat = None
    dividend: Stat = None
    borrow: Stat = None


class ImpliedForwardResult(_Result):
    spot: Stat = None
    forward: Stat = None
    time_to_expiry: Stat = None
    net_carry_rate: Stat = None
    basis: Stat = None
    basis_pct: Stat = None
    components: CarryComponents = Field(default_factory=CarryComponents)


class RealizedMoves(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_observations: int = 0
    median_move_pct: Stat = None
    mean_move_pct: Stat = None
    max_move_pct: Stat = None
    exceeded_implied_pct: Stat = Field(
        None,
        description="How often the implied move was actually exceeded "
        "historically. Compare against theoretical_exceedance_pct.",
    )


class ExpectedMoveResult(_Result):
    spot: Stat = None
    implied_vol: Stat = None
    days: Stat = None
    one_sd_move: Stat = Field(None, description="ONE STANDARD DEVIATION, not a bound.")
    one_sd_move_pct: Stat = None
    straddle_approximation: Stat = Field(
        None, description="0.8 x the 1-sd move: what the ATM straddle costs."
    )
    straddle_approximation_pct: Stat = None
    upper_1sd: Stat = None
    lower_1sd: Stat = None
    theoretical_exceedance_pct: Stat = Field(
        None, description="31.7 under the model's own assumptions."
    )
    realized: Optional[RealizedMoves] = None


class DeltaHedgeResult(_Result):
    option_premium: Stat = None
    implied_vol: Stat = None
    realized_vol: Stat = None
    n_hedges: int = 0
    n_paths: int = 0
    mean_pnl: Stat = None
    median_pnl: Stat = None
    std_pnl: Stat = Field(
        None,
        description="The dispersion, which is what decides the size. Scales "
        "as 1/sqrt(n_hedges), not 1/n_hedges.",
    )
    p05_pnl: Stat = None
    p95_pnl: Stat = None
    worst_pnl: Stat = None
    best_pnl: Stat = None
    win_rate: Stat = None
    mean_transaction_cost: Stat = None
    theoretical_continuous_pnl: Stat = Field(
        None,
        description="Evaluated at the INITIAL dollar gamma. A reference "
        "point, not a prediction.",
    )
    pnl_as_pct_of_premium: Stat = None


class ScenarioCell(BaseModel):
    model_config = ConfigDict(extra="allow")

    vol_shock: Stat = None
    value: Stat = None
    pnl: Stat = None
    pnl_pct_of_base: Stat = None


class ScenarioRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    spot_shock_pct: Stat = None
    cells: List[ScenarioCell] = Field(default_factory=list)


class WorstCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    pnl: Stat = None
    spot_shock_pct: Stat = None
    vol_shock: Stat = None


class OptionScenariosResult(_Result):
    base_value: Stat = None
    quantity: Stat = None
    days_forward: Stat = None
    grid: List[ScenarioRow] = Field(default_factory=list)
    worst_case: Optional[WorstCase] = None


__all__ = [
    "DeltaHedgeResult",
    "ExpectedMoveResult",
    "ImpliedForwardResult",
    "OptionGreeksResult",
    "OptionScenariosResult",
    "OptionStrategyResult",
    "PutCallParityResult",
    "Stat",
    "VolTermStructureResult",
    "VolatilityConeResult",
    "VolatilitySmileResult",
]
