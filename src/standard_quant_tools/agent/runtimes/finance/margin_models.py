"""
The inputs and results of the financing, margin and margin-runway tools.

POSITIONS ARE THE SHARED MODEL. `FinancePosition` is `_positions.Position`
-- the same fields and the same checks -- advertised without the titles
and the model description, and with the same statements in fewer words
(`_SHORTER`), since a nested schema is carried whole by every tool that
takes it. `MarginPosition` adds what margin needs and financing
does not: the line's kind (a stock, an ETF with its leverage factor, or
an equity option with its strike and underlying price). Nothing in the
shared model changes meaning: a `Position` passed to the margin library
is a stock.

THE SCHEMAS ARE TERSE ON PURPOSE. The runtime's tools share one context
budget, and a nested type is paid for once per tool that carries it.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from standard_quant_tools.agent.runtimes._financing import (
    FinancingBenchmarkReport,
    FinancingSpecInput,
)
from standard_quant_tools.agent.runtimes._json_safe import (
    ExplainsNulls,
    finite_or_none,
)
from standard_quant_tools.agent.runtimes._positions import Holdings, Position
from standard_quant_tools.finance.financing_cost import MAX_EQUITY, MAX_HORIZON_DAYS
from standard_quant_tools.finance.margin import (
    DEFAULT_CURE_DAYS,
    MAX_LEVERAGE,
    MAX_PATH_DAYS,
    check_margin_terms,
)
from standard_quant_tools.finance.positions import MAX_POSITIONS

Stat = Annotated[Optional[float], BeforeValidator(finite_or_none)]
#: A day number that is null with a reason when there is no such day.
Day = Annotated[Optional[int], BeforeValidator(finite_or_none)]

_POSITIONS_DESCRIPTION = (
    "One line per instrument and currency; a repeat is refused, so net lots "
    "yourself."
)


#: The shared fields' descriptions as these tools advertise them: the same
#: statements, shorter, since each tool carries every line's schema.
_SHORTER = {
    "instrument": "Your label for it; nothing is fetched.",
    "quantity": "Signed units; negative is short, zero refused.",
    "price": "Price of one unit in currency; a premium for an option.",
    "multiplier": "Currency per point of price per unit: 50 for an E-mini, 1 "
    "for a share.",
    "currency": "ISO 4217 code of price, e.g. 'USD'.",
    "borrow_fee_bps": "Annual borrow bps while short; overrides the terms'.",
    "base_currency": "ISO 4217 code of every total.",
    "fx_rates": "base_currency per ONE unit of each other currency held: "
    "{'EUR': 1.10}.",
}


def _compact(schema: Dict[str, Any]) -> None:
    """Drop the model's title and description and every property's title,
    each a restatement of a name already in the schema, and shorten the
    shared fields' descriptions."""
    schema.pop("title", None)
    _compact_top(schema)


def _compact_top(schema: Dict[str, Any]) -> None:
    """The same for a tool's own input, keeping the model's title."""
    schema.pop("description", None)
    for name, prop in schema.get("properties", {}).items():
        prop.pop("title", None)
        if name in _SHORTER and "description" in prop:
            prop["description"] = _SHORTER[name]


# ── positions ───────────────────────────────────────────────────────────


class FinancePosition(Position):
    """A `Position`, advertised compactly."""

    model_config = ConfigDict(extra="forbid", json_schema_extra=_compact)


class MarginPosition(FinancePosition):
    """A `Position` with what margin reads: its kind and an option's terms."""

    model_config = ConfigDict(extra="forbid", json_schema_extra=_compact)

    kind: Literal["equity", "etf", "call", "put"] = Field(
        "equity",
        description="call/put: an equity option, multiplier its contract size.",
    )
    leverage: float = Field(
        1.0, ge=1.0, le=MAX_LEVERAGE, description="An ETF's factor: 3 for 3x or -3x."
    )
    strike: Optional[float] = Field(None, gt=0)
    underlying_price: Optional[float] = Field(None, gt=0)

    @model_validator(mode="after")
    def _terms(self) -> "MarginPosition":
        check_margin_terms(
            self.kind,
            self.leverage,
            self.strike,
            self.underlying_price,
            multiplier=self.multiplier,
            price=self.price,
            at=f"position {self.instrument!r}",
        )
        return self


class FinanceHoldings(Holdings):
    """`Holdings` of compact positions."""

    model_config = ConfigDict(extra="forbid", json_schema_extra=_compact_top)

    positions: List[FinancePosition] = Field(
        ...,
        min_length=1,
        max_length=MAX_POSITIONS,
        description=_POSITIONS_DESCRIPTION,
    )


class MarginHoldings(Holdings):
    """`Holdings` of margin positions, with any house maintenance rates."""

    model_config = ConfigDict(extra="forbid", json_schema_extra=_compact)

    positions: List[MarginPosition] = Field(
        ...,
        min_length=1,
        max_length=MAX_POSITIONS,
        description=_POSITIONS_DESCRIPTION,
    )
    house_rates: Dict[str, float] = Field(
        default_factory=dict,
        description="{instrument: house maintenance rate}, at or above the rule's.",
    )


# ── inputs ──────────────────────────────────────────────────────────────


class FinancingCostInput(FinanceHoldings):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_compact_top)

    equity: float = Field(
        ...,
        gt=0,
        le=MAX_EQUITY,
        description="The account's own capital; cash is equity less the lines' net.",
    )
    financing: FinancingSpecInput
    days: Optional[int] = Field(
        None,
        ge=1,
        le=MAX_HORIZON_DAYS,
        description="Calendar days, for a scalar benchmark_rate.",
    )
    start_date: Optional[str] = Field(
        None, description="ISO date; with end_date, a dated horizon."
    )
    end_date: Optional[str] = None


class MarginRequirementInput(MarginHoldings):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_compact_top)

    equity: Optional[float] = Field(
        None,
        ge=-MAX_EQUITY,
        le=MAX_EQUITY,
        description="Equity today, for the excess over each requirement.",
    )


class MarginRunwayInput(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra=_compact_top)

    equity: float = Field(
        ...,
        ge=-MAX_EQUITY,
        le=MAX_EQUITY,
        description="Equity today: cash + longs - shorts at market.",
    )
    equity_changes: Optional[List[float]] = Field(
        None,
        min_length=1,
        max_length=MAX_PATH_DAYS,
        description="The path: equity's change each day.",
    )
    initial_margin: Optional[float] = Field(None, ge=0, le=MAX_EQUITY)
    maintenance_margin: Optional[float] = Field(None, ge=0, le=MAX_EQUITY)
    holdings: Optional[MarginHoldings] = Field(
        None, description="Positions, to compute the requirements."
    )
    returns: Optional[Dict[str, List[float]]] = Field(
        None,
        description="With holdings, the path as {instrument: daily returns}.",
    )
    restore_to: Literal["initial", "maintenance"] = Field(
        "initial", description="What a call restores."
    )
    cure_days: int = Field(
        DEFAULT_CURE_DAYS,
        ge=0,
        le=MAX_PATH_DAYS,
        description="Days a call may go unmet; 15 is FINRA 4210(f)(6)'s most.",
    )


# ── results ─────────────────────────────────────────────────────────────


class ShortBorrowRow(BaseModel):
    """One short line's borrow over the horizon."""

    model_config = ConfigDict(extra="forbid")

    instrument: str
    currency: str
    market_value: float = Field(..., description="In base_currency, positive.")
    borrow_bps: float
    source: str = Field(
        ...,
        description="'position' (its own borrow_fee_bps), 'borrow_bps' or "
        "'default_borrow_bps'.",
    )
    cost: float


class FinancingCostResult(ExplainsNulls):
    """What the holdings cost to finance over the horizon, in base_currency.
    Costs are positive; credit_interest and rebate are income."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "effective_rate_on_gross": "every position's notional is zero, so the "
        "holdings have no gross exposure to take a rate on",
    }

    base_currency: str
    day_count: str
    basis: float
    start_date: Optional[str] = Field(
        None, description="Null when the horizon was given in days."
    )
    end_date: Optional[str] = None
    days: int
    year_fraction: float = Field(..., description="days / basis.")
    benchmark: FinancingBenchmarkReport
    equity: float
    long: float
    short: float
    net: float
    gross: float
    cash: float = Field(..., description="equity - net, short proceeds included.")
    interest_bearing_cash: float = Field(
        ..., description="cash less segregated proceeds: what accrues interest."
    )
    segregated_proceeds: float
    credit_rate: float = Field(
        ..., description="The mean annual rate credit accrued at, day-weighted."
    )
    debit_rate: float
    rebate_rate: float
    debit_interest: float = Field(..., description="On negative cash.")
    credit_interest: float = Field(..., description="On positive cash.")
    borrow_cost: float
    rebate: float = Field(..., description="On segregated short proceeds.")
    net_cost: float = Field(
        ..., description="debit_interest + borrow_cost - credit_interest - rebate."
    )
    cash_drag: float = Field(
        ...,
        description="What the idle cash and segregated proceeds earn below the "
        "benchmark: their credit and rebate spreads over the horizon.",
    )
    effective_rate_on_gross: Stat = Field(
        None, description="net_cost / gross / year_fraction."
    )
    effective_rate_on_equity: Stat = Field(
        None, description="net_cost / equity / year_fraction."
    )
    borrows: List[ShortBorrowRow] = Field(default_factory=list)
    unused_borrow_bps: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class MarginLineRow(BaseModel):
    """One position's requirements, in USD."""

    model_config = ConfigDict(extra="forbid")

    instrument: str
    currency: str
    kind: str
    side: str
    quantity: float
    price: float
    leverage: float
    market_value: float
    reg_t: float = Field(..., description="Reg T initial margin, as equity.")
    reg_t_rule: str
    maintenance: float
    maintenance_rule: str
    maintenance_rate: Optional[float] = Field(
        None,
        description="The rate on market value applied; null on an option, "
        "whose rule is not one rate.",
    )
    house_rate: Optional[float] = Field(
        None, description="The house rate given for it; null when none was."
    )
    underlying_value: Optional[float] = Field(
        None, description="An option's underlying value; null otherwise."
    )
    out_of_the_money: Optional[float] = Field(
        None, description="An option's out-of-the-money amount; null otherwise."
    )


class MarginRequirementResult(ExplainsNulls):
    """Reg T initial and FINRA 4210 maintenance margin, in USD equity."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "excess_over_initial": "no equity was given to compare with the "
        "requirements",
        "excess_over_maintenance": "no equity was given to compare with the "
        "requirements",
    }

    base_currency: str
    lines: List[MarginLineRow] = Field(default_factory=list)
    long_market_value: float
    short_market_value: float
    reg_t: float = Field(..., description="The lines' Reg T amounts, summed.")
    maintenance: float = Field(..., description="The lines' maintenance, summed.")
    minimum_equity: float = Field(
        ..., description="FINRA 4210(b)(4)'s $2,000, or a purchase's cost if less."
    )
    initial: float = Field(
        ..., description="The greater of reg_t, maintenance and minimum_equity."
    )
    initial_binding: str
    initial_rule: str
    equity: Stat = None
    excess_over_initial: Stat = Field(None, description="equity - initial.")
    excess_over_maintenance: Stat = Field(None, description="equity - maintenance.")
    warnings: List[str] = Field(default_factory=list)


class MarginCallRow(BaseModel):
    """One maintenance call along the path."""

    model_config = ConfigDict(extra="forbid")

    day: int
    equity: float
    maintenance: float
    target: float = Field(..., description="The equity the call restores.")
    amount: float = Field(..., description="target - equity on the call's day.")
    deadline: int = Field(..., description="day + cure_days.")
    outcome: str = Field(
        ...,
        description="'cured' (equity reached target by the deadline), "
        "'deleveraged' or 'outstanding' (the path ended first).",
    )
    resolved_day: Optional[int] = Field(
        None,
        description="The day it was cured or deleveraging starts; null while "
        "outstanding.",
    )


_NO_CALL = (
    "the equity stays at or above the maintenance requirement on every day "
    "of the path, so no call is made"
)


class MarginRunwayResult(ExplainsNulls):
    """A stress path walked day by day. Day 0 is today."""

    model_config = ConfigDict(extra="forbid")
    null_reasons = {
        "first_call_day": _NO_CALL,
        "first_call_amount": _NO_CALL,
        "deleveraging_day": "no call goes unmet through its cure period inside "
        "the path: each is cured by the equity's recovery, or the path ends "
        "first (see calls)",
        "first_deficit_day": "the equity stays at or above zero on every day "
        "of the path",
    }

    source: str = Field(
        ...,
        description="'requirements' (given), 'positions' (today's, held) or "
        "'revalued' (recomputed each day).",
    )
    days: int
    restore_to: str
    cure_days: int
    equity_today: float
    initial_today: float
    maintenance_today: float
    first_call_day: Day = Field(
        None, description="The first day equity is below maintenance."
    )
    first_call_amount: Stat = Field(
        None, description="What that call asks for: target - equity (calls[0])."
    )
    deleveraging_day: Day = Field(
        None,
        description="The day forced deleveraging starts: a call's deadline "
        "passed unmet.",
    )
    calls: List[MarginCallRow] = Field(default_factory=list)
    lowest_equity: float
    lowest_equity_day: int
    first_deficit_day: Day = Field(
        None, description="The first day equity is below zero."
    )
    tightest_cushion: float = Field(
        ..., description="The least equity - maintenance over the path."
    )
    tightest_day: int
    buffer: float = Field(
        ...,
        description="The least cash added today with no call on any day; one "
        "unit less calls on tightest_day.",
    )
    final_equity: float
    final_maintenance: float
    warnings: List[str] = Field(default_factory=list)


__all__ = [
    "FinanceHoldings",
    "FinancePosition",
    "FinancingCostInput",
    "FinancingCostResult",
    "MarginCallRow",
    "MarginHoldings",
    "MarginLineRow",
    "MarginPosition",
    "MarginRequirementInput",
    "MarginRequirementResult",
    "MarginRunwayInput",
    "MarginRunwayResult",
    "ShortBorrowRow",
]
