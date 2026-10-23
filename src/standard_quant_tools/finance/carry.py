"""
The carry of each position and of a set of them: what each would earn over
a year if prices did not move (Koijen, Moskowitz, Pedersen & Vrugt, "Carry",
Journal of Financial Economics 2018), as an annual rate on the position's
notional and as an amount in the base currency.

TWO KINDS OF LINE, told apart by the terms each carries.

- A FUTURE OR FORWARD -- an index or commodity future, an FX forward --
  carries `spot` and `years_to_expiry`, and its `price` is the future's F.
  Its carry is KMPV's (S - F) / (F x T): the return a long earns if the spot
  does not move and the future converges to it, over the years to expiry.
  `spot` may be the nearer contract's price, with T the years between the
  two expiries, which is KMPV's own construction where the spot does not
  trade. For a future priced off the carry identity F = S exp((r - q) T)
  this is

      (exp(-(r - q) T) - 1) / T = (q - r) + (r - q)^2 T / 2 + O(T^2),

  q - r to first order in T, with an error that shrinks in proportion to T;
  `carry_log`, ln(S / F) / T (`delta_one.carry.observed_carry_rate` negated),
  is q - r exactly. An FX forward is the same line: F and S in units of its
  currency per one of the currency bought, and the carry the interest
  differential the forward implies -- under covered interest parity with
  one day count, (r_foreign - r_domestic) / (1 + r_domestic T), the
  differential discounted one period (`fx_forward.covered_interest_forward`).
- A FUNDED HOLDING -- a share, a bond, a currency deposit -- carries
  `income_rate` (its dividend yield, coupon or deposit rate) and
  `financing_rate` (what funding a long costs, or what a short's proceeds
  earn). A long's carry is income - financing; a short pays the income,
  earns the financing and pays its borrow, financing - income - borrow, the
  borrow being the line's `borrow_fee_bps`. A deposit in a foreign currency
  funded in the base currency carries exactly the interest differential.

A SHORT EARNS THE OPPOSITE OF A LONG: the carry of a short future is minus
the long's, and a funded short's is minus the long's less its borrow. Every
carry is a rate the position EARNS on its absolute notional, so a negative
one is a cost.

THE BOOK. Each line's annual amount is its carry times its absolute
notional in the base currency; the book's carry is their sum over the
gross notional -- the notional-weighted carry -- and `annual_carry` the sum
itself. Every total is a `math.fsum`, so the order of the lines changes no
bit. Carry is what holding earns if nothing moves; it is not a forecast of
the return.
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Dict, List, Mapping, Optional, Sequence

from standard_quant_tools.delta_one.carry import observed_carry_rate
from standard_quant_tools.error import ValidationError

from .currency import currency_code
from .positions import MAX_ABS_PRICE, check_borrow_fee_bps, position_values

__all__ = ["MAX_CARRY_RATE", "MAX_YEARS_TO_EXPIRY", "book_carry", "carry_kind"]

#: 1,000% a year as a decimal, the bound the library puts on an annual rate.
MAX_CARRY_RATE = 10.0
#: A future or forward a century out is a unit error (days given as years).
MAX_YEARS_TO_EXPIRY = 100.0

_FORWARD_TERMS = ("spot", "years_to_expiry")
_FUNDED_TERMS = ("income_rate", "financing_rate")


def _attribute(position: Any, name: str) -> Any:
    if isinstance(position, Mapping):
        return position.get(name)
    return getattr(position, name, None)


def _number(value: Any, argument: str, fix: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{argument}={value!r} is not a number. {fix}")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{argument}={value!r} is not finite. {fix}")
    return number


def carry_kind(position: Any, at: str = "positions[0]") -> str:
    """
    'forward' for a line carrying `spot` and `years_to_expiry`, 'funded' for
    one carrying `income_rate` and `financing_rate`; a line with both sets,
    neither, or half of one is refused, naming the line and the fix. So is a
    funded short with no `borrow_fee_bps` -- a short pays to borrow, and the
    rate is the caller's -- and a short future with one, which borrows
    nothing.
    """
    forward = [n for n in _FORWARD_TERMS if _attribute(position, n) is not None]
    funded = [n for n in _FUNDED_TERMS if _attribute(position, n) is not None]
    if forward and funded:
        raise ValidationError(
            f"{at} carries {', '.join(forward + funded)}: the terms of a "
            "future or forward (spot, years_to_expiry) and of a funded "
            "holding (income_rate, financing_rate) at once. A line is one or "
            "the other; drop the set that does not apply."
        )
    if not forward and not funded:
        raise ValidationError(
            f"{at} has no carry terms. Give spot and years_to_expiry for a "
            "future or forward (its price is the future's), or income_rate "
            "and financing_rate for a holding that is funded."
        )
    present, terms, kind = (
        (forward, _FORWARD_TERMS, "forward")
        if forward
        else (funded, _FUNDED_TERMS, "funded")
    )
    if len(present) < len(terms):
        missing = [n for n in terms if n not in present]
        raise ValidationError(
            f"{at} gives {present[0]} without {missing[0]}. "
            + (
                "A future's carry needs its spot and its years to expiry."
                if kind == "forward"
                else "A funded line's carry is its income less its financing, "
                "and both are rates you supply (0 for none)."
            )
        )
    quantity = _attribute(position, "quantity")
    short = isinstance(quantity, numbers.Real) and quantity < 0
    borrow = _attribute(position, "borrow_fee_bps")
    if short and kind == "forward" and borrow is not None:
        raise ValidationError(
            f"{at} is a short future or forward with borrow_fee_bps={borrow!r}, "
            "and a future is not borrowed. Drop borrow_fee_bps from the line."
        )
    if short and kind == "funded" and borrow is None:
        raise ValidationError(
            f"{at} is short and funded, and has no borrow_fee_bps. A short "
            "pays to borrow what it sold: give the line's borrow_fee_bps (0 if "
            "it borrows free, as a currency borrowed at income_rate does)."
        )
    return kind


def book_carry(
    positions: Sequence[Any],
    base_currency: str,
    fx_rates: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """
    The carry of each line and of the set, in the base currency. Positions,
    the base currency and the rates are those of `positions.position_values`;
    each line also carries its terms (see the module docstring and
    `carry_kind`). Refusals are `ValidationError`s naming the line, the
    value and the fix.
    """
    positions = list(positions)
    values = position_values(positions, base_currency, fx_rates)
    lines: List[Dict[str, Any]] = []
    for index, (position, value) in enumerate(zip(positions, values)):
        at = f"positions[{index}] ({value.instrument!r})"
        kind = carry_kind(position, at)
        sign = 1.0 if value.side == "long" else -1.0
        borrow_bps = _attribute(position, "borrow_fee_bps")
        carry_log = math.nan
        if kind == "forward":
            price = value.price
            if not price > 0:
                raise ValidationError(
                    f"{at} has price {price!r}, and a future's carry is a rate "
                    "on its price, undefined at or below zero."
                )
            spot = _number(
                _attribute(position, "spot"),
                f"{at}.spot",
                "Give the spot, in the line's currency.",
            )
            if not 0 < spot <= MAX_ABS_PRICE:
                raise ValidationError(
                    f"{at}.spot={spot!r} is outside (0, {MAX_ABS_PRICE:g}]. "
                    "Give the spot (or the nearer contract's price) in the "
                    "line's currency."
                )
            years = _number(
                _attribute(position, "years_to_expiry"),
                f"{at}.years_to_expiry",
                "Give the years to expiry: 0.25 for three months.",
            )
            if not 0 < years <= MAX_YEARS_TO_EXPIRY:
                raise ValidationError(
                    f"{at}.years_to_expiry={years!r} is outside (0, "
                    f"{MAX_YEARS_TO_EXPIRY:g}]. It is in YEARS: 0.25 is three "
                    "months, and 90 is ninety years."
                )
            instrument_carry = (spot - price) / (price * years)
            carry_log = (
                sign
                * -observed_carry_rate(spot=spot, forward=price, time_to_expiry=years)
                + 0.0
            )
            carry = sign * instrument_carry
        else:
            if value.price < 0:
                raise ValidationError(
                    f"{at} has price {value.price!r}. A funded line's carry is "
                    "a rate on its value, undefined below zero."
                )
            income = _number(
                _attribute(position, "income_rate"),
                f"{at}.income_rate",
                "Give the annual income yield as a decimal: 0.025 is 2.5%.",
            )
            financing = _number(
                _attribute(position, "financing_rate"),
                f"{at}.financing_rate",
                "Give the annual financing rate as a decimal: 0.053 is 5.3%.",
            )
            for name, rate in (("income_rate", income), ("financing_rate", financing)):
                if abs(rate) > MAX_CARRY_RATE:
                    raise ValidationError(
                        f"{at}.{name}={rate!r} is beyond {MAX_CARRY_RATE:g} in "
                        "magnitude. Rates are decimals: 0.053 is 5.3% a year."
                    )
            if sign > 0:
                carry = income - financing
            else:
                # carry_kind has refused a funded short with no borrow fee.
                borrow = check_borrow_fee_bps(borrow_bps, f"{at}.borrow_fee_bps")
                borrow = borrow / 10_000.0
                carry = financing - income - borrow
        carry = carry + 0.0
        notional = value.base_notional
        lines.append(
            {
                "instrument": value.instrument,
                "currency": value.currency,
                "kind": kind,
                "side": value.side,
                "base_notional": notional,
                "carry": carry,
                "carry_log": carry_log,
                "annual_carry": carry * abs(notional) + 0.0,
            }
        )
    gross = math.fsum(abs(line["base_notional"]) for line in lines)
    annual = math.fsum(line["annual_carry"] for line in lines)
    for line in lines:
        line["weight"] = abs(line["base_notional"]) / gross if gross > 0 else math.nan
    by_currency: Dict[str, float] = {}
    by_kind: Dict[str, float] = {}
    for key, table in (("currency", by_currency), ("kind", by_kind)):
        for group in sorted({line[key] for line in lines}):
            table[group] = math.fsum(
                line["annual_carry"] for line in lines if line[key] == group
            )
    return {
        "base_currency": currency_code(base_currency, argument="base_currency"),
        "positions": len(lines),
        "gross": gross,
        "net": math.fsum(line["base_notional"] for line in lines),
        "carry": annual / gross if gross > 0 else math.nan,
        "annual_carry": annual,
        "annual_carry_by_currency": by_currency,
        "annual_carry_by_kind": by_kind,
        "lines": lines,
        "warnings": [
            "Carry is what each position earns over a year if prices do not "
            "move -- the future converging to an unchanged spot, the income "
            "less the financing. It is not a forecast of the return."
        ],
    }
