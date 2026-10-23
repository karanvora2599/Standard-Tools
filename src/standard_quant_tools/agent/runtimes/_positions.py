"""
Positions as a tool input: a `Position` per line, and `Holdings` -- the
lines, the base currency and the FX rates -- for every tool that values a
set of holdings.

HOW A TOOL TAKES THEM. Subclass `Holdings` and the three fields sit at the
top level of the tool's input, beside its own; or give the tool a field of
type `Holdings`. Either way the checks below run when the input is built,
and `position_values()` / `exposure_summary()` hand back the library's
valuation (`standard_quant_tools.finance`). Never name that field `book`:
the surface fuzzer synthesizes anything called that as an order book.

WHY `instrument` AND NOT `symbol`. A field named `symbol`, `ticker` or
`universe` anywhere in an input marks the tool as fetching market data: the
MCP layer flags it open-world and the determinism layer skips it. Nothing
here is fetched -- the prices and rates are the caller's -- so the
identifier is `instrument`, read as a label only.

THE DECISIONS, each refused by argument and value:
- `quantity` is signed, finite and non-zero. Zero is refused, as an option
  leg of zero is elsewhere: a flat line is not a position, and kept it
  would sit in every per-line report and count.
- `price` is any finite number. Zero (a worthless line still held) and
  negative (crude oil, power and calendar spreads have printed below zero)
  are kept, and the notional takes the product's sign. A calculation that
  cannot read a negative price, such as equity margin, refuses it itself.
- `multiplier` is finite and above zero; a short is a negative quantity.
- `currency` is an active ISO 4217 code, upper-cased; an unknown one is
  refused with the nearest valid code, and pence ('GBp', 'GBX') with the
  division that fixes them.
- One instrument in one currency is one line. A repeat is refused, not
  added up or kept as lots: two reports concatenated, two lots and a long
  held against a short look the same here, and margin and financing differ
  on which it is, so netting is the caller's decision.
- `fx_rates` needs every non-base currency held; the base's own rate, if
  given, is exactly 1. Rates for currencies not held are accepted -- a rate
  table usually covers more than one set of holdings, and a misspelt key is
  refused as a code anyway -- and `exposure_summary` lists them as unused.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from standard_quant_tools.finance.currency import currency_code
from standard_quant_tools.finance.positions import (
    MAX_BORROW_FEE_BPS,
    MAX_INSTRUMENT_LENGTH,
    MAX_MULTIPLIER,
    MAX_POSITIONS,
    ExposureSummary,
    PositionValue,
    check_borrow_fee_bps,
    check_distinct_lines,
    check_fx_rates,
    check_instrument,
    check_multiplier,
    check_position_count,
    check_price,
    check_quantity,
    check_rate_coverage,
    exposure_summary,
    position_values,
)

__all__ = ["Holdings", "Position"]


class Position(BaseModel):
    """One line of holdings, valued as quantity x price x multiplier in
    `currency`."""

    model_config = ConfigDict(extra="forbid")

    instrument: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INSTRUMENT_LENGTH,
        description="Your identifier for the instrument (ticker, ISIN, "
        "contract code). A label only: nothing is fetched for it.",
    )
    quantity: float = Field(
        ...,
        description="Signed units held: shares, contracts or face amount. "
        "Negative is short; zero is refused.",
    )
    price: float = Field(
        ...,
        description="Price of one unit in `currency`. Zero and negative "
        "prices are kept.",
    )
    multiplier: float = Field(
        1.0,
        gt=0,
        le=MAX_MULTIPLIER,
        description="Currency per point of price per unit: 50 for an E-mini "
        "S&P 500 future, 1 for a share.",
    )
    currency: str = Field(
        ...,
        max_length=16,
        description="ISO 4217 code `price` is in, e.g. 'USD'. Pence are "
        "refused: give GBP.",
    )
    borrow_fee_bps: Optional[float] = Field(
        None,
        ge=0,
        le=MAX_BORROW_FEE_BPS,
        description="Annual borrow fee in bps, read only while short. It "
        "overrides the account's per-instrument and default borrow; omitted, "
        "those apply.",
    )

    @field_validator("instrument")
    @classmethod
    def _instrument(cls, value: str) -> str:
        return check_instrument(value)

    @field_validator("quantity")
    @classmethod
    def _quantity(cls, value: float) -> float:
        return check_quantity(value)

    @field_validator("price")
    @classmethod
    def _price(cls, value: float) -> float:
        return check_price(value)

    @field_validator("multiplier", mode="before")
    @classmethod
    def _multiplier(cls, value: Any) -> Any:
        # Before the bounds, so a NaN, an infinity or a negative is refused
        # with its fix rather than as "greater than 0". A value that is not
        # a number is left to the type check.
        if _is_number(value):
            check_multiplier(value)
        return value

    @field_validator("currency")
    @classmethod
    def _currency(cls, value: str) -> str:
        return currency_code(value, argument="currency")

    @field_validator("borrow_fee_bps", mode="before")
    @classmethod
    def _borrow_fee(cls, value: Any) -> Any:
        if _is_number(value):
            check_borrow_fee_bps(value)
        return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class Holdings(BaseModel):
    """
    The holdings a finance tool values: positions, the base currency every
    total is reported in, and the rates that convert to it.
    """

    model_config = ConfigDict(extra="forbid")

    positions: List[Position] = Field(
        ...,
        min_length=1,
        max_length=MAX_POSITIONS,
        description="One line per instrument and currency; a repeat is "
        "refused, so net lots yourself.",
    )
    base_currency: str = Field(
        ...,
        max_length=16,
        description="ISO 4217 code every total is reported in, e.g. 'USD'.",
    )
    fx_rates: Dict[str, float] = Field(
        default_factory=dict,
        validate_default=True,
        description="Units of base_currency per ONE unit of each other "
        "currency held: base USD and 1 EUR = 1.10 USD is {'EUR': 1.10}. The "
        "base's own rate, if given, is 1.",
    )

    @field_validator("positions", mode="before")
    @classmethod
    def _count(cls, value: Any) -> Any:
        # Before the items are built, so 50,000 lines are refused at once
        # and with the fix, not after 50,000 validations as "too long".
        if isinstance(value, (list, tuple)):
            check_position_count(len(value))
        return value

    @field_validator("positions")
    @classmethod
    def _distinct(cls, value: List[Position]) -> List[Position]:
        check_distinct_lines(
            [(i, p.instrument, p.currency) for i, p in enumerate(value)]
        )
        return value

    @field_validator("base_currency")
    @classmethod
    def _base(cls, value: str) -> str:
        return currency_code(value, argument="base_currency")

    @field_validator("fx_rates")
    @classmethod
    def _rates(cls, value: Dict[str, float], info: ValidationInfo) -> Dict[str, float]:
        # Fields are validated in order, so the positions and the base are
        # in `info.data` when they were valid; when one was not, its own
        # refusal is the one to read, and coverage is not judged.
        base = info.data.get("base_currency")
        rates = check_fx_rates(value, base_currency=base)
        positions = info.data.get("positions")
        if base is not None and positions is not None:
            check_rate_coverage(
                [(i, p.instrument, p.currency) for i, p in enumerate(positions)],
                base,
                rates,
            )
        return rates

    def position_values(self) -> List[PositionValue]:
        """Each position valued in the base currency, in input order."""
        return position_values(self.positions, self.base_currency, self.fx_rates)

    def exposure_summary(self) -> ExposureSummary:
        """Gross, net, long, short and the exposure per currency."""
        return exposure_summary(self.positions, self.base_currency, self.fx_rates)
