"""
Typed inputs for the derivatives runtime.

EVERY MODEL FORBIDS EXTRA FIELDS, for the reason recorded on
`OptionPricingInput`: Pydantic's default silently drops an unknown argument,
so a hallucinated or misspelled name runs on defaults while the caller
believes it configured something. On this surface that failure is especially
expensive -- `volatility` means a RELATIVE vol to Black-Scholes and an
ABSOLUTE one to Bachelier, and a dropped `model` field would price the wrong
one without a word.

INPUTS ARE INLINE, not fetched. Nothing here takes a ticker: an option chain
is passed as parallel lists of strikes and implied vols, a term structure as
a mapping of expiry to vol. That is deliberate. The library has no options
data provider, and a tool that pretended to have one would fetch equity
prices and compute a "chain" that does not exist. Passing the quotes in
means the caller can also feed the same tool a hypothetical surface, which
is most of what these are used for.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class OptionGreeksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spot: float = Field(..., gt=0, description="Current underlying price.")
    strike: float = Field(..., gt=0, description="Option strike.")
    time_to_expiry: float = Field(
        ..., gt=0, description="Years to expiry (0.25 = three months)."
    )
    volatility: float = Field(
        ..., gt=0, description="Annualized volatility as a decimal (0.20 = 20%)."
    )
    risk_free_rate: float = Field(
        0.0, description="Continuously-compounded annual risk-free rate."
    )
    option_type: Literal["call", "put"] = Field("call")
    dividend_yield: float = Field(0.0, ge=0, description="Continuous dividend yield.")


class OptionStrategyLeg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_type: Literal["call", "put", "stock"] = Field(
        ..., description="'stock' legs carry delta 1 and ignore strike/vol."
    )
    quantity: float = Field(
        ...,
        description="Signed size. Negative is short. Zero is refused rather "
        "than ignored -- it is not a position.",
    )
    strike: Optional[float] = Field(
        None, gt=0, description="Required for option legs, ignored for stock."
    )
    volatility: Optional[float] = Field(
        None, gt=0, description="Required for option legs."
    )
    time_to_expiry: Optional[float] = Field(
        None, gt=0, description="Required for option legs, in years."
    )


class OptionStrategyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    legs: List[OptionStrategyLeg] = Field(
        ..., min_length=1, description="The legs of the structure."
    )
    spot: float = Field(..., gt=0, description="Current underlying price.")
    risk_free_rate: float = Field(0.0)
    dividend_yield: float = Field(0.0, ge=0)
    spot_range: Optional[List[float]] = Field(
        None,
        description="Prices to evaluate the payoff at. Omit for an "
        "automatic range around the strikes.",
    )


class VolatilitySmileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strikes: List[float] = Field(
        ..., min_length=5, description="Strikes, parallel to implied_vols."
    )
    implied_vols: List[float] = Field(
        ..., min_length=5, description="Implied vols as decimals, one per strike."
    )
    forward: float = Field(
        ...,
        gt=0,
        description="Forward price for this expiry. The fit is in "
        "log(K/F), not in strike -- see the tool description.",
    )
    time_to_expiry: float = Field(..., gt=0, description="Years to expiry.")


class VolatilityConeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prices: List[float] = Field(
        ..., min_length=60, description="Historical closing prices, oldest first."
    )
    horizons: Optional[List[int]] = Field(
        None, description="Horizons in trading days. Defaults to 5/10/21/42/63/126."
    )
    current_implied: Optional[Dict[str, float]] = Field(
        None,
        description="Optional map of horizon (as a string, e.g. '21') to "
        "today's implied vol, to place it as a percentile of realized.",
    )


class VolTermStructureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    implied_by_expiry: Dict[str, float] = Field(
        ...,
        description="Map of years-to-expiry (as a string, e.g. '0.0833') to "
        "implied vol. At least two entries.",
    )


class PutCallParityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_price: float = Field(..., ge=0)
    put_price: float = Field(..., ge=0)
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    time_to_expiry: float = Field(..., gt=0, description="Years.")
    risk_free_rate: float = Field(...)
    dividend_yield: float = Field(0.0, ge=0)
    tolerance_bps: float = Field(
        25.0,
        gt=0,
        description="Violation size, in basis points of strike, above which "
        "the result is flagged. Set it near the bid-ask spread.",
    )


class ImpliedForwardInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spot: float = Field(..., gt=0)
    time_to_expiry: float = Field(..., gt=0, description="Years.")
    risk_free_rate: float = Field(...)
    dividend_yield: float = Field(0.0, ge=0)
    borrow_rate: float = Field(
        0.0,
        ge=0,
        description="Stock borrow cost, kept SEPARATE from the dividend "
        "because the two behave differently -- borrow floats and can move "
        "hundreds of basis points in a day.",
    )


class ExpectedMoveInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spot: float = Field(..., gt=0)
    implied_vol: float = Field(..., gt=0, description="Annualized, as a decimal.")
    days: float = Field(..., gt=0, description="Calendar days to the event or expiry.")
    realized_moves: Optional[List[float]] = Field(
        None,
        description="Past absolute moves over the same horizon, as decimals. "
        "Supplying them gives the HISTORICAL exceedance rate instead of only "
        "the lognormal one.",
    )


class DeltaHedgeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    time_to_expiry: float = Field(..., gt=0, description="Years.")
    implied_vol: float = Field(..., gt=0, description="The vol the option was sold at.")
    realized_vol: float = Field(
        ..., gt=0, description="The vol the underlying actually realizes."
    )
    risk_free_rate: float = Field(0.0)
    option_type: Literal["call", "put"] = Field("call")
    n_hedges: int = Field(
        21, ge=1, le=2000, description="Rehedges over the option's life."
    )
    n_paths: int = Field(
        500, ge=1, le=5000, description="Simulated paths. The DISPERSION is the point."
    )
    transaction_cost_bps: float = Field(
        0.0, ge=0, description="Cost per unit of notional traded, in bps."
    )
    seed: int = Field(0, description="Seed, so the result is reproducible.")


class OptionScenariosInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    time_to_expiry: float = Field(..., gt=0, description="Years.")
    volatility: float = Field(..., gt=0)
    risk_free_rate: float = Field(0.0)
    option_type: Literal["call", "put"] = Field("call")
    quantity: float = Field(1.0, description="Signed position size. Negative is short.")
    spot_shocks: Optional[List[float]] = Field(
        None, description="Fractional spot moves, e.g. [-0.2, 0, 0.2]."
    )
    vol_shocks: Optional[List[float]] = Field(
        None, description="Absolute vol changes, e.g. [-0.05, 0, 0.05]."
    )
    days_forward: float = Field(
        0.0,
        ge=0,
        description="Decay the position this many days before revaluing, "
        "which is how a weekend or an overnight gap should be stressed.",
    )


__all__ = [
    "DeltaHedgeInput",
    "ExpectedMoveInput",
    "ImpliedForwardInput",
    "OptionGreeksInput",
    "OptionScenariosInput",
    "OptionStrategyInput",
    "OptionStrategyLeg",
    "PutCallParityInput",
    "VolTermStructureInput",
    "VolatilityConeInput",
    "VolatilitySmileInput",
]
