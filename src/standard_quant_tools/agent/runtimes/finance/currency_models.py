"""
Inputs and results of the currency tools: the exposure of a multi-currency
set of positions, its carry, and the FX forward that hedges it.

THE POSITIONS ARE `Holdings`. Both position tools subclass it, so the lines,
the base currency and the FX rates arrive and are checked exactly as every
finance tool takes them (`agent/runtimes/_positions.py`); each tool's line
model is `Position` plus the one or few numbers its question needs on each
line -- a local return, or carry terms -- rather than a second mapping keyed
by instrument that could name a line that is not there. The pair tool takes
no positions: a pair, a spot, two rates and their day counts.

EVERY RATE IS THE CALLER'S, a decimal within 1,000% a year; every refusal
names the argument, the value and the fix. The schemas are kept short on
purpose: the finance runtime's advertised tools share one budget.
"""

from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from standard_quant_tools.agent.runtimes._json_safe import ExplainsNulls, finite_or_none
from standard_quant_tools.agent.runtimes._positions import Holdings, Position
from standard_quant_tools.backtest.financing import _compact_schema
from standard_quant_tools.finance.carry import carry_kind
from standard_quant_tools.finance.currency_exposure import hedge_legs
from standard_quant_tools.finance.fx_forward import currency_pair
from standard_quant_tools.finance.positions import MAX_POSITIONS

Stat = Annotated[Optional[float], BeforeValidator(finite_or_none)]

DayCount = Literal["ACT/360", "ACT/365F", "ACT/ACT", "30/360"]

__all__ = [
    "BookCarryInput",
    "BookCarryResult",
    "CarryLine",
    "CarryLineResult",
    "CurrencyExposureInput",
    "CurrencyExposureResult",
    "CurrencyExposureRow",
    "CurrencyReturnLine",
    "CurrencyReturnSplit",
    "ExposureLine",
    "FxForwardHedgeInput",
    "FxForwardHedgeResult",
    "HedgedPeriod",
]

_POSITIONS = Holdings.model_fields["positions"].description

#: Every input model here drops its per-field titles and its own docstring
#: from the schema, as the financing terms do: a title restates the field's
#: name, and each byte is paid once per tool in a runtime that shares one
#: budget. The docstrings below are for readers of the code.
_COMPACT = ConfigDict(extra="forbid", json_schema_extra=_compact_schema)


# ── get_currency_exposure ────────────────────────────────────────────────


class ExposureLine(Position):
    """A position, with its price return over the period for the split."""

    model_config = _COMPACT

    local_return: Optional[float] = Field(
        None, ge=-10, le=10, description="Its own-currency price return."
    )


class CurrencyExposureInput(Holdings):
    model_config = _COMPACT

    positions: List[ExposureLine] = Field(
        ..., min_length=1, max_length=MAX_POSITIONS, description=_POSITIONS
    )
    hedges: Dict[str, float] = Field(
        default_factory=dict,
        description="FX forwards or spot: pair -> units of its first currency "
        "bought (+) or sold (-), e.g. {'EURUSD': -1e7}.",
    )
    fx_returns: Optional[Dict[str, float]] = Field(
        None,
        description="Currency -> its fx_rate's return (0.02 = +2%), with every "
        "local_return.",
    )

    @field_validator("hedges")
    @classmethod
    def _hedges(cls, value: Dict[str, float], info: ValidationInfo) -> Dict:
        # Pairs, notionals and rate coverage, once the base and the rates
        # are known to be valid; when either was not, its own refusal is the
        # one to read.
        base, rates = info.data.get("base_currency"), info.data.get("fx_rates")
        if base is not None and rates is not None:
            hedge_legs(value, base, rates)
        return value


_NO_EXPOSURE = (
    "it is the base currency, which carries no currency risk, or its "
    "positions net to nothing, so there is no exposure for a hedge to offset"
)


class CurrencyExposureRow(ExplainsNulls):
    """One currency, in the base currency unless said otherwise. Its nulls
    are reported by the result that holds it."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {"hedge_ratio": _NO_EXPOSURE}

    currency: str
    fx_rate: Stat = Field(None, description="Units of base per one of it.")
    positions: int = 0
    hedge_legs: int = 0
    local_net: Stat = Field(
        None, description="Positions plus hedge legs, in this currency."
    )
    position_net: Stat = Field(None, description="Its positions' net exposure.")
    hedge_net: Stat = Field(None, description="Its hedge legs' net exposure.")
    net: Stat = Field(None, description="position_net + hedge_net.")
    gross: Stat = Field(
        None, description="Sum of every line's and hedge leg's absolute value."
    )
    hedge_ratio: Stat = Field(
        None,
        description="-hedge_net / position_net: 1 fully hedged, above 1 "
        "over-hedged, negative a hedge adding to the exposure.",
    )


class CurrencyReturnLine(ExplainsNulls):
    """One position's base-currency return, split. Reported by its result."""

    model_config = ConfigDict(extra="forbid")

    instrument: str
    currency: str
    base_notional: Stat = None
    local_return: Stat = None
    fx_return: Stat = Field(None, description="Exactly 0 in the base currency.")
    cross_return: Stat = Field(None, description="local_return x fx_return.")
    base_return: Stat = Field(
        None,
        description="(1 + local)(1 + fx) - 1, the exact sum of the three terms.",
    )
    local_pnl: Stat = Field(None, description="base_notional x local_return.")
    fx_pnl: Stat = Field(None, description="base_notional x fx_return.")
    cross_pnl: Stat = Field(None, description="base_notional x cross_return.")
    pnl: Stat = Field(None, description="The three P&L terms' exact sum.")


class CurrencyReturnSplit(ExplainsNulls):
    """The return split of every position, with the hedges' FX P&L."""

    model_config = ConfigDict(extra="forbid")

    lines: List[CurrencyReturnLine] = Field(default_factory=list)
    local_pnl: Stat = None
    fx_pnl: Stat = Field(None, description="The positions' FX P&L.")
    cross_pnl: Stat = None
    hedge_pnl: Stat = Field(
        None,
        description="The hedge legs' FX P&L: each leg's value x its "
        "currency's fx_return.",
    )
    pnl: Stat = Field(None, description="All of the above, in the base currency.")
    fx_pnl_by_currency: Dict[str, Stat] = Field(
        default_factory=dict,
        description="Positions' and hedges' FX P&L per currency, cross excluded.",
    )
    cross_pnl_by_currency: Dict[str, Stat] = Field(default_factory=dict)
    fx_returns: Dict[str, Stat] = Field(
        default_factory=dict, description="The FX return used per currency."
    )


class CurrencyExposureResult(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")

    base_currency: str
    positions: int = 0
    hedges: int = 0
    gross: Stat = Field(None, description="The positions' gross, in base.")
    net: Stat = Field(None, description="The positions' net, in base.")
    long: Stat = None
    short: Stat = None
    foreign_net: Stat = Field(
        None, description="Net exposure outside the base currency, after hedges."
    )
    foreign_gross: Stat = Field(
        None, description="Sum over foreign currencies of |net|, after hedges."
    )
    unhedged_foreign_gross: Stat = Field(
        None, description="The same before hedges: sum of |position_net|."
    )
    currencies: List[CurrencyExposureRow] = Field(default_factory=list)
    returns: Optional[CurrencyReturnSplit] = Field(
        None,
        description="The return split; null unless each position has a "
        "local_return.",
    )
    unused_fx_rates: List[str] = Field(
        default_factory=list,
        description="Rates given for currencies no position or hedge holds.",
    )
    warnings: List[str] = Field(default_factory=list)


# ── analyze_book_carry ───────────────────────────────────────────────────


class CarryLine(Position):
    """A position with its carry terms: spot and years for a future or
    forward, income and financing for a funded holding."""

    model_config = _COMPACT

    spot: Optional[float] = Field(
        None, gt=0, description="A future's spot; price is the future's."
    )
    years_to_expiry: Optional[float] = Field(None, gt=0, le=100)
    income_rate: Optional[float] = Field(
        None, ge=-10, le=10, description="Funded: dividend, coupon or deposit rate."
    )
    financing_rate: Optional[float] = Field(
        None, ge=-10, le=10, description="Paid by a long, earned by a short."
    )

    @model_validator(mode="after")
    def _one_kind(self) -> "CarryLine":
        carry_kind(self, f"line {self.instrument!r}")
        return self


class BookCarryInput(Holdings):
    model_config = _COMPACT

    positions: List[CarryLine] = Field(
        ..., min_length=1, max_length=MAX_POSITIONS, description=_POSITIONS
    )


_ZERO_GROSS = "every line's notional is zero, so the set has no gross to weight by"


class CarryLineResult(ExplainsNulls):
    """One line's carry. Reported by its result."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "carry_log": "the line is funded: its carry is already a simple annual "
        "rate, and carry_log is a future's continuously compounded one",
        "weight": _ZERO_GROSS,
    }

    instrument: str
    currency: str
    kind: Literal["forward", "funded"]
    side: Literal["long", "short"]
    base_notional: Stat = None
    weight: Stat = Field(None, description="|base_notional| / gross.")
    carry: Stat = Field(
        None,
        description="Annual rate earned on |notional|: (S - F)/(F T) for a "
        "future, income - financing (- borrow if short) when funded; a "
        "short's is the opposite of a long's.",
    )
    carry_log: Stat = Field(
        None, description="A future's ln(S / F) / T, signed as carry is."
    )
    annual_carry: Stat = Field(
        None, description="carry x |base_notional|, in the base currency."
    )


class BookCarryResult(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")
    null_reasons = {"carry": _ZERO_GROSS}

    base_currency: str
    positions: int = 0
    gross: Stat = None
    net: Stat = None
    carry: Stat = Field(
        None,
        description="The set's carry: annual_carry / gross, the "
        "notional-weighted carry of its lines.",
    )
    annual_carry: Stat = Field(
        None, description="Sum of the lines' annual_carry, in the base currency."
    )
    annual_carry_by_currency: Dict[str, Stat] = Field(default_factory=dict)
    annual_carry_by_kind: Dict[str, Stat] = Field(default_factory=dict)
    lines: List[CarryLineResult] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ── price_fx_forward_hedge ───────────────────────────────────────────────


class FxForwardHedgeInput(BaseModel):
    model_config = _COMPACT

    pair: str = Field(
        ...,
        max_length=7,
        description="Market quote, e.g. 'EURUSD' or 'USDJPY': spot is units of "
        "the second per one of the first.",
    )
    spot: float = Field(..., gt=0, le=1e12)
    days: int = Field(..., ge=1, le=36_600, description="Days from spot to delivery.")
    base_currency: str = Field(
        ...,
        max_length=16,
        description="The hedger's currency, one of the pair's; the other is "
        "foreign.",
    )
    base_rate: float = Field(
        ..., ge=-10, le=10, description="base_currency's money-market rate, 0.053."
    )
    foreign_rate: float = Field(..., ge=-10, le=10)
    base_day_count: DayCount
    foreign_day_count: DayCount
    basis_bps: float = Field(
        0.0,
        ge=-10_000,
        le=10_000,
        description="Cross-currency basis added to foreign_rate; negative "
        "raises the hedge's pickup.",
    )
    pip_size: Optional[float] = Field(
        None, gt=0, le=1, description="Default 0.0001; 0.01 for a JPY quote."
    )
    start_date: Optional[str] = Field(
        None, description="Spot date, ISO; ACT/ACT and 30/360 need it."
    )
    asset_return: Optional[float] = Field(
        None,
        ge=-1,
        le=10,
        description="A foreign asset's return in its own currency to delivery.",
    )
    spot_at_maturity: Optional[float] = Field(
        None,
        gt=0,
        le=1e12,
        description="The pair's spot at delivery; with asset_return, the hedged "
        "and unhedged returns.",
    )
    hedge_ratio: float = Field(
        1.0, ge=0, le=2, description="Share of the asset's value sold forward."
    )

    @field_validator("pair")
    @classmethod
    def _pair(cls, value: str) -> str:
        first, second = currency_pair(value)
        return first + second

    @field_validator("start_date")
    @classmethod
    def _start(cls, value: Optional[str]) -> Optional[str]:
        # The surface's one ISO reader, so a bad date is refused in the
        # same words as every other dated argument. Imported here: the
        # shared module imports the agent models, which load the runtimes.
        from standard_quant_tools.agent.runtimes._shared import parse_iso_date

        if value is None:
            return value
        stamp = parse_iso_date(value, "start_date", "price_fx_forward_hedge")
        return stamp.date().isoformat()

    @model_validator(mode="after")
    def _base_in_pair(self) -> "FxForwardHedgeInput":
        from standard_quant_tools.finance.currency import currency_code

        base = currency_code(self.base_currency, argument="base_currency")
        if base not in (self.pair[:3], self.pair[3:]):
            raise ValueError(
                f"base_currency={self.base_currency!r} is neither currency of "
                f"{self.pair}. The hedger's currency is one of the pair's: for "
                f"a {base}-based holder of a {self.pair[:3]} asset, price "
                f"{self.pair[:3]}{base}."
            )
        self.base_currency = base
        return self


class HedgedPeriod(ExplainsNulls):
    """A foreign asset's return to delivery, hedged and not."""

    model_config = ConfigDict(extra="forbid")

    local_return: Stat = None
    fx_return: Stat = Field(
        None, description="The change in units of base per one foreign."
    )
    cross_return: Stat = Field(None, description="local_return x fx_return.")
    hedge_return: Stat = Field(
        None,
        description="hedge_ratio x (forward_premium - fx_return): the "
        "forward's gain per unit of starting value.",
    )
    unhedged_return: Stat = Field(None, description="local + fx + cross.")
    hedged_return: Stat = Field(None, description="unhedged_return + hedge_return.")
    hedge_ratio: Stat = None


class FxForwardHedgeResult(ExplainsNulls):
    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "hedge_carry": "the base leg's year fraction is zero (30/360 counts no "
        "days from a 30th to a 31st), so the premium has no annual rate",
    }

    pair: str
    base_currency: str
    foreign_currency: str
    spot: Stat = None
    forward: Stat = Field(
        None,
        description="S (1 + r_second t_second) / (1 + r_first t_first), each "
        "leg on its own day count, the basis in the foreign rate.",
    )
    forward_points: Stat = Field(None, description="(forward - spot) / pip_size.")
    pip_size: Stat = None
    pip_size_rule: str = ""
    days: int = 0
    start_date: Optional[str] = None
    maturity_date: Optional[str] = None
    day_counts: Dict[str, str] = Field(default_factory=dict)
    year_fractions: Dict[str, Stat] = Field(default_factory=dict)
    rates: Dict[str, Stat] = Field(
        default_factory=dict, description="Each leg's rate, the basis included."
    )
    basis_bps: Stat = None
    rate_differential: Stat = Field(
        None, description="Base rate less the foreign rate with its basis."
    )
    forward_premium: Stat = Field(
        None,
        description="Units of base per one foreign at the forward over those "
        "at spot, less 1: the hedge's gain to delivery.",
    )
    hedge_carry: Stat = Field(
        None,
        description="forward_premium / the base leg's year fraction: positive "
        "a pickup, negative a cost, a year.",
    )
    period: Optional[HedgedPeriod] = Field(
        None, description="Null unless asset_return and spot_at_maturity are given."
    )
    warnings: List[str] = Field(default_factory=list)
