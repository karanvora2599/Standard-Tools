"""
Typed inputs for the delta_one runtime.

EVERY MODEL FORBIDS EXTRA FIELDS. Pydantic's default silently drops an
unknown argument, so a hallucinated or misspelled name runs on defaults
while the caller believes it configured something. On this surface that is
expensive in a specific way: a dropped `multiplier` prices a futures
position off by the contract's point value -- a factor of 50 on ES -- and
every number downstream stays perfectly plausible. A dropped `fee_rate`
makes an ETF look free.

INPUTS ARE INLINE, not fetched. Nothing here takes a ticker. A futures
curve arrives as a list of contracts, a basket as a list of constituents,
a financing rate as a number, because this library has no futures data
provider, no index-constituent source and no dividend calendar -- and a
tool that pretended otherwise would compute a curve that does not exist.
It is the call the derivatives runtime already made about option chains,
with the same side benefit: these work on a hypothetical curve, which is
most of what they are used for.

RATES ARE DECIMALS, TIME IS YEARS, SPREADS ARE BASIS POINTS. Three
conventions, stated on every field that uses them, because mixing them is
the error this surface is most exposed to: passing 43 for a 4.3% rate, or
90 days for 0.25 years, produces a number rather than a refusal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "BasisHistoryInput",
    "CashFuturesBasisInput",
    "BasisDislocationInput",
    "SpreadMonitorInput",
    "CompareExpressionsInput",
    "DeltaOneExpression",
    "DividendConstituent",
    "DividendPointsInput",
    "EtfFairValueInput",
    "FuturesContractQuote",
    "FuturesCurveInput",
    "FuturesHedgeInput",
    "HedgeEffectivenessInput",
    "IndexBasketInput",
    "IndexConstituent",
    "IndexRebalanceInput",
    "ReplicationBasketInput",
    "RollAnalysisInput",
    "SolveForwardCarryInput",
    "TotalReturnFutureInput",
    "TotalReturnSwapInput",
]


class CashFuturesBasisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spot: float = Field(..., gt=0, description="Current cash index or share price.")
    future_price: float = Field(..., gt=0, description="The QUOTED future.")
    time_to_expiry: float = Field(
        ..., gt=0, le=100, description="Years to expiry (0.25 = three months)."
    )
    risk_free_rate: float = Field(
        ...,
        ge=-10,
        le=10,
        description="Financing, as a decimal (0.043 = 4.3%), continuously compounded.",
    )
    dividend_yield: float = Field(
        0.0, ge=-10, le=10, description="Continuous dividend yield as a decimal."
    )
    borrow_rate: float = Field(
        0.0,
        ge=-10,
        le=10,
        description="Stock-loan rate as a decimal. Kept apart from the "
        "dividend because it floats and can move hundreds of bps in a day.",
    )
    tolerance_bps: float = Field(
        25.0,
        ge=0,
        le=10_000,
        description="Annualized bps within which the basis reads as fair.",
    )


class SolveForwardCarryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spot: float = Field(..., gt=0, description="Current cash price.")
    forward: float = Field(
        ..., gt=0, description="The quoted forward or future to solve against."
    )
    time_to_expiry: float = Field(..., gt=0, le=100, description="Years to expiry.")
    solve_for: Literal["financing_rate", "dividend_yield", "borrow_rate"] = Field(
        ..., description="Which of the three unknowns to recover."
    )
    risk_free_rate: Optional[float] = Field(
        None,
        ge=-10,
        le=10,
        description="Required unless solving for it. Decimal, not percent.",
    )
    dividend_yield: Optional[float] = Field(
        None, ge=-10, le=10, description="Required unless solving for it."
    )
    borrow_rate: Optional[float] = Field(
        None, ge=-10, le=10, description="Required unless solving for it."
    )


class BasisHistoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spot_prices: List[float] = Field(
        ..., min_length=3, description="Cash closes, oldest first."
    )
    futures_prices: List[float] = Field(
        ...,
        min_length=3,
        description="Futures closes on the SAME dates, oldest first.",
    )
    window: Optional[int] = Field(
        None,
        ge=2,
        le=5000,
        description="Rolling z-score window. Omit for a full-sample z-score, "
        "which describes history but looks ahead.",
    )
    time_to_expiry: Optional[List[float]] = Field(
        None,
        description="Years to expiry on each date. Supply it to annualize "
        "the basis; without it the series steps at every roll.",
    )


class FuturesContractQuote(BaseModel):
    """One contract on a curve."""

    model_config = ConfigDict(extra="forbid")

    price: float = Field(..., gt=0, description="Quoted futures price.")
    time_to_expiry: float = Field(
        ..., gt=0, le=100, description="Years to this contract's expiry."
    )
    label: Optional[str] = Field(
        None, description="Contract name, e.g. 'ESZ5'. Used in the output."
    )


class FuturesCurveInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contracts: List[FuturesContractQuote] = Field(
        ..., min_length=2, description="The contracts. Sorted by expiry here."
    )
    spot: Optional[float] = Field(
        None,
        gt=0,
        description="Cash level. Without it only the relationships BETWEEN "
        "contracts are computable, not richness to cash.",
    )


class RollAnalysisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    front_price: float = Field(..., gt=0, description="The expiring contract.")
    next_price: float = Field(..., gt=0, description="The contract rolled into.")
    contracts_held: float = Field(
        ...,
        description="SIGNED position. Negative is short, whose roll "
        "economics are the opposite sign of a long's.",
    )
    multiplier: float = Field(
        ..., gt=0, description="Front contract's point value, e.g. 50 for ES."
    )
    days_to_front_expiry: float = Field(
        ..., gt=0, le=10_000, description="Calendar days until the front expires."
    )
    days_between_expiries: Optional[float] = Field(
        None,
        gt=0,
        le=10_000,
        description="Calendar days between the two contracts' expiries, e.g. "
        "about 91 for a quarterly roll. This is what roll_yield is "
        "annualized over, because the step between the contracts is earned "
        "across the gap between them. Omitted, roll_yield_bps is null "
        "rather than annualized by the front's remaining life, which made "
        "the answer depend on the roll date.",
    )
    next_multiplier: Optional[float] = Field(
        None, gt=0, description="Only when the two differ, e.g. micro to full-size."
    )
    cost_per_contract: float = Field(
        0.0, ge=0, description="Commission per contract, per side, in currency."
    )
    spread_ticks: float = Field(0.0, ge=0, description="Ticks crossed per leg.")
    tick_value: float = Field(0.0, ge=0, description="Currency per tick.")


class FuturesHedgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio_value: float = Field(
        ..., description="Market value of what is being hedged, in currency."
    )
    portfolio_beta: float = Field(
        ..., ge=-100, le=100, description="Its beta to the hedge's benchmark."
    )
    future_price: float = Field(..., gt=0, description="Quoted future.")
    multiplier: float = Field(
        ..., gt=0, description="Contract point value, e.g. 50 for ES."
    )
    future_beta: float = Field(
        1.0,
        ge=-100,
        le=100,
        description="The contract's beta to that same benchmark. 1.0 is "
        "right only when they are the same index -- hedging an S&P book "
        "with NQ at 1.0 under-hedges.",
    )
    objective: Literal["beta_neutral", "dollar_neutral"] = Field(
        "beta_neutral", description="beta_neutral is what 'hedged' usually means."
    )
    existing_contracts: float = Field(
        0.0, description="Contracts already held; makes the answer a TRADE."
    )


class HedgeEffectivenessInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    portfolio_returns: List[float] = Field(
        ..., min_length=3, description="Periodic returns as decimals, oldest first."
    )
    hedge_returns: List[float] = Field(
        ..., min_length=3, description="The hedge instrument's returns, same dates."
    )
    hedge_ratio: float = Field(
        ...,
        ge=-1000,
        le=1000,
        description="SIGNED units of hedge per unit of portfolio. A short "
        "hedge is negative; a positive one doubles the exposure.",
    )
    window: Optional[int] = Field(
        60, ge=2, le=5000, description="Rolling window for hedge-ratio stability."
    )
    periods_per_year: int = Field(
        252, ge=1, le=31_536_000, description="252 for daily, 52 weekly, 12 monthly."
    )


class IndexConstituent(BaseModel):
    """One name in a basket."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="Ticker, used to attribute the spread.")
    price: float = Field(..., gt=0, description="Current price.")
    weight: Optional[float] = Field(
        None, description="Index weight as a decimal. Use this OR shares."
    )
    shares: Optional[float] = Field(
        None, gt=0, description="Index shares. Requires a divisor."
    )
    reference_price: Optional[float] = Field(
        None,
        gt=0,
        description="Last known traded price. An identical current price "
        "flags the name as possibly stale.",
    )


class IndexBasketInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constituents: List[IndexConstituent] = Field(
        ..., min_length=1, description="The basket."
    )
    index_level: Optional[float] = Field(
        None, gt=0, description="The published level to compare against."
    )
    divisor: Optional[float] = Field(
        None,
        gt=0,
        description="Index divisor. With it the basket is share-based and "
        "reproduces the LEVEL; without it, weight-based, reproducing returns.",
    )


class DeltaOneExpression(BaseModel):
    """One way of holding the exposure, with whatever it costs.

    Every rate is a COST to the holder except `dividend_yield`, which is a
    receipt. Omitted terms are zero, which is why the result reports what
    was actually supplied.
    """

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., description="How this appears in the ranking.")
    kind: Literal["cash", "etf", "future", "forward", "synthetic", "swap"] = Field(
        ..., description="Instrument family."
    )
    financing_rate: float = Field(
        0.0, ge=-10, le=10, description="Funding cost as a decimal."
    )
    dividend_yield: float = Field(
        0.0, ge=-10, le=10, description="Received on a long, as a decimal."
    )
    borrow_rate: float = Field(0.0, ge=-10, le=10, description="Stock loan, decimal.")
    fee_rate: float = Field(
        0.0,
        ge=-10,
        le=10,
        description="Expense ratio or swap spread, as a decimal. The term "
        "most often forgotten on an ETF, and the whole difference over a "
        "long hold.",
    )
    spread_bps: float = Field(0.0, ge=0, description="Half-spread crossed, ONE way.")
    commission_bps: float = Field(0.0, ge=0, description="Commission, one way.")
    impact_bps: float = Field(0.0, ge=0, description="Expected impact, one way.")
    rolls_per_year: float = Field(0.0, ge=0, le=365, description="Rolls per year.")
    roll_cost_bps: float = Field(0.0, description="Cost of ONE roll, in bps.")
    capital_requirement_pct: float = Field(
        1.0,
        ge=0,
        le=100,
        description="Fraction of notional tied up. 1.0 fully funded, 0.06 "
        "for futures margin.",
    )


class CompareExpressionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expressions: List[DeltaOneExpression] = Field(
        ..., min_length=1, description="The candidates to rank."
    )
    notional: float = Field(..., gt=0, description="Exposure wanted, in currency.")
    horizon_years: float = Field(
        ...,
        gt=0,
        le=100,
        description="How long it will be held. This CHANGES THE RANKING: "
        "execution is paid once and carry accrues, so 0.08 and 2.0 reorder "
        "the same candidates.",
    )
    direction: Literal["long", "short"] = Field(
        "long", description="Flips every carry sign."
    )


# ── Phase II: the desk instruments ──────────────────────────────────────


class ReplicationBasketInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returns: Dict[str, List[float]] = Field(
        ...,
        min_length=2,
        description="Candidate return series keyed by symbol, oldest first. "
        "Decimals, not percent.",
    )
    benchmark_returns: List[float] = Field(
        ..., min_length=3, description="The series to track, on the same dates."
    )
    max_names: Optional[int] = Field(
        None,
        ge=1,
        le=5000,
        description="Cardinality ceiling. Enforced by thresholding, not by "
        "an integer solver -- the result is a good basket of that size, not "
        "provably the best one.",
    )
    long_only: bool = Field(
        True,
        description="A basket allowed to short is a long-short position "
        "benchmarked to an index, not a replication basket.",
    )
    max_weight: Optional[float] = Field(
        None, gt=0, le=1, description="Ceiling on any single name."
    )
    weight_caps: Optional[Dict[str, float]] = Field(
        None,
        description="Per-name ceiling. The natural home for an ADV-derived "
        "limit, so a name that cannot be traded in size is not selected in size.",
    )
    covariance_method: Literal["ledoit_wolf", "sample", "ewma"] = Field(
        "ledoit_wolf",
        description="Shrinkage matters here: a replication universe usually "
        "has more candidates than a sample covariance can support.",
    )
    periods_per_year: int = Field(
        252, ge=1, le=31_536_000, description="252 daily, 52 weekly, 12 monthly."
    )


class EtfFairValueInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    etf_price: float = Field(..., gt=0, description="Traded price of the fund.")
    nav: float = Field(..., gt=0, description="Net asset value PER SHARE.")
    nav_is_intraday: bool = Field(
        False,
        description="False means a struck end-of-day NAV, in which case an "
        "intraday premium is mostly the market's move since the strike.",
    )
    basket_value: Optional[float] = Field(
        None,
        gt=0,
        description="Independent value of one share's worth of creation "
        "basket. Supply it to separate a fund away from its holdings from a "
        "NAV that disagrees with them.",
    )
    cash_component: float = Field(
        0.0, description="Per-share cash in the creation basket."
    )
    creation_unit_shares: Optional[float] = Field(
        None,
        gt=0,
        description="Shares per creation unit. Needed to express a fee in bps.",
    )
    creation_fee: float = Field(0.0, ge=0, description="Currency, per unit.")
    etf_spread_bps: float = Field(0.0, ge=0, description="Half-spread, ONE way.")
    basket_spread_bps: float = Field(
        0.0, ge=0, description="Blended basket half-spread, one way."
    )
    tolerance_bps: float = Field(
        25.0, ge=0, le=10_000, description="Within this the premium reads as fair."
    )


class TotalReturnSwapInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notional: float = Field(..., description="Swap notional, in currency.")
    initial_price: float = Field(..., gt=0, description="Underlying at inception.")
    current_price: float = Field(..., gt=0, description="Underlying now.")
    financing_rate: float = Field(
        ..., ge=-10, le=10, description="Reference rate as a decimal."
    )
    spread_bps: float = Field(
        0.0,
        ge=-10_000,
        le=10_000,
        description="Spread over the reference. This is the negotiated part "
        "and where the product's economics are.",
    )
    dividends: float = Field(
        0.0,
        description="Cash dividends per unit of underlying over the period. "
        "Zero makes this a PRICE-return swap in everything but name.",
    )
    start_date: Optional[str] = Field(None, description="ISO date, e.g. '2026-01-02'.")
    valuation_date: Optional[str] = Field(None, description="ISO date.")
    time_elapsed: Optional[float] = Field(
        None, description="Years, as an alternative to the two dates."
    )
    day_count: Literal["ACT/365F", "ACT/360", "30/360", "ACT/ACT"] = Field(
        "ACT/365F",
        description="ACT/360 accrues about 1.4% more financing than ACT/365F.",
    )
    direction: Literal["receive", "pay"] = Field(
        "receive", description="Receive the total return, or pay it."
    )


class TotalReturnFutureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quote: float = Field(
        ...,
        ge=-100_000,
        le=1_000_000,
        description="The TRF's quoted number. Bounded because it is read as "
        "either a spread in bps or a price level, and an unbounded value "
        "under the first reading overflows the exponential.",
    )
    quote_convention: Literal["spread_bps", "index_level"] = Field(
        ...,
        description="What that number IS. The two are not convertible without "
        "knowing which you have, and guessing misprices by the whole "
        "financing leg.",
    )
    underlying_price: float = Field(..., gt=0, description="Index or share price.")
    time_to_expiry: float = Field(..., gt=0, le=100, description="Years.")
    reference_rate: float = Field(
        ..., ge=-10, le=10, description="The rate the spread is quoted over."
    )
    dividend_yield: float = Field(0.0, ge=-10, le=10, description="Decimal.")
    comparison_spread_bps: Optional[float] = Field(
        None,
        description="What the same exposure costs elsewhere, typically the "
        "TRS quote. Supplying it turns a measurement into a decision.",
    )


class DividendConstituent(BaseModel):
    """One name's dividend."""

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., description="Ticker, used to attribute the total.")
    shares: float = Field(
        ..., gt=0, description="INDEX shares, not shares outstanding."
    )
    dividend_per_share: float = Field(..., description="Cash amount per share.")
    ex_date: str = Field(..., description="ISO ex-date, e.g. '2026-04-15'.")


class DividendPointsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    constituents: List[DividendConstituent] = Field(..., min_length=1)
    divisor: float = Field(
        ...,
        gt=0,
        description="The index divisor. Getting it wrong scales every point "
        "figure by a constant.",
    )
    as_of: str = Field(..., description="ISO date. Dividends already ex are excluded.")
    expiry: str = Field(..., description="ISO date of the contract.")
    spot: Optional[float] = Field(None, gt=0, description="Index level.")
    future_price: Optional[float] = Field(
        None,
        gt=0,
        description="Supply with spot, financing_rate and time_to_expiry to "
        "get the market's own dividend number alongside your forecast.",
    )
    financing_rate: Optional[float] = Field(None, ge=-10, le=10)
    time_to_expiry: Optional[float] = Field(None, gt=0, le=100)


class IndexRebalanceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    old_weights: Dict[str, float] = Field(
        ..., min_length=1, description="Index weights before the change."
    )
    new_weights: Dict[str, float] = Field(
        ..., min_length=1, description="Index weights after it."
    )
    indexed_assets: float = Field(
        ...,
        gt=0,
        description="Money tracking the index, in currency. The least "
        "knowable input and the one everything scales by -- published "
        "figures exclude closet indexers, so treat the output as a floor.",
    )
    adv: Optional[Dict[str, float]] = Field(
        None,
        description="Average daily dollar volume per name. Without it flow is "
        "sized in currency but not in days, and currency alone does not say "
        "whether a trade is difficult.",
    )
    auction_fraction: float = Field(
        1.0,
        gt=0,
        le=1,
        description="Share of daily volume printing in the closing auction, "
        "where index trades execute. Typically 0.10-0.20; leaving it at 1.0 "
        "understates participation by five to ten times.",
    )


class BasisDislocationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spot_prices: List[float] = Field(
        ..., min_length=20, description="Cash closes, oldest first."
    )
    futures_prices: List[float] = Field(
        ..., min_length=20, description="Futures closes on the SAME dates."
    )
    time_to_expiry: Optional[List[float]] = Field(
        None,
        description="Years to expiry per date. Without it the channel is in "
        "bps of spot and STEPS at a roll, which both detectors read as a "
        "structural shift because in that channel it is one.",
    )
    reference_fraction: float = Field(
        0.3,
        gt=0,
        lt=1,
        description="Fraction of the START used to learn normal. A baseline "
        "drawn from the whole series hides the shift inside it.",
    )
    threshold: float = Field(
        9.0,
        gt=0,
        le=1000,
        description="CUSUM decision threshold. 9.0 is calibrated, not the "
        "textbook 5.0, which measures 51% false alarms over 300 "
        "observations when asking whether anything happened anywhere.",
    )
    slack: float = Field(
        0.5, ge=0, le=100, description="Standardized deviations absorbed per step."
    )
    max_breaks: int = Field(
        3, ge=1, le=20, description="Most segment boundaries to look for."
    )


class SpreadMonitorInput(BaseModel):
    """One stateful call rather than five tools.

    Omit `state` to open a monitor; pass back the `state` from the previous
    call to advance it. The caller holds the state between calls, because a
    tool call returns and there is nowhere for a subscription to live.

    ONE MONITOR COVERS FIVE JOBS through `channel`. Live basis, ETF NAV,
    index arbitrage and a generic cross-instrument spread are the same
    arithmetic and differ only in what the two legs are CALLED; the roll
    spread is a difference in points. Three formulas, so three channels.
    """

    model_config = ConfigDict(extra="forbid")

    primary_prices: List[float] = Field(
        ...,
        min_length=1,
        description="The leg that is DEAR when the spread is positive: the "
        "future against spot, the ETF against NAV, the basket against the "
        "index, the next contract against the front.",
    )
    reference_prices: List[float] = Field(
        ..., min_length=1, description="The other leg, on the same ticks."
    )
    channel: Literal["relative_bps", "annualized_bps", "absolute_points"] = Field(
        "relative_bps",
        description="How the legs combine. relative_bps is (primary/"
        "reference - 1) x 10,000 and serves basis, ETF premium, index "
        "arbitrage and any cross-instrument spread. annualized_bps is "
        "ln(primary/reference)/T x 10,000, comparable across expiries and "
        "needing time_to_expiry. absolute_points is primary - reference, "
        "for a roll spread and anything else quoted in points.",
    )
    state: Optional[Dict[str, Any]] = Field(
        None,
        description="The `state` returned by the previous call. Omit to open "
        "a new monitor. Pass the state, not the whole result.",
    )
    time_to_expiry: Optional[List[float]] = Field(
        None, description="Years to expiry per tick. Required iff annualized_bps."
    )
    reset: bool = Field(
        False,
        description="Clear the accumulators before applying this update. Use "
        "after acknowledging an alert.",
    )
    keep_baseline_on_reset: bool = Field(
        True,
        description="On reset, keep the same idea of normal (a spike that "
        "passed) or relearn it (a regime change). Opposite conclusions that "
        "look identical in the accumulators, so this is not guessed.",
    )
    label: str = Field(
        "spread",
        description="Names the monitor in its alerts. The channel says how "
        "the legs combine; this says what they are.",
    )
    warmup: int = Field(
        60,
        ge=10,
        le=100_000,
        description="Observations before the baseline is fixed and anything "
        "can trigger. A baseline from fewer than about ten is estimation "
        "error, and a detector standardized against it fires on that.",
    )
    threshold: float = Field(
        9.0,
        gt=0,
        le=1000,
        description="CUSUM decision threshold. 9.0 is calibrated; the "
        "textbook 5.0 measures 51% false alarms over 300 observations.",
    )
    slack: float = Field(
        0.5, ge=0, le=100, description="Standardized deviations absorbed per step."
    )
