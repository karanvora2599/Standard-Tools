"""
Positions valued in one base currency: the arithmetic that financing,
margin, currency exposure and carry on a set of holdings all start from.

A position is a signed quantity of an instrument at a price in a currency,
scaled by a contract multiplier. Its local notional is quantity x price x
multiplier, in its own currency; its base notional is that times the
caller's rate for the currency, given as units of base currency per ONE
unit of it. The gross, net, long and short of the holdings, and their
exposure per currency, are sums of base notionals. Nothing is fetched: the
prices and the rates are the caller's.

EXACT, SO NOTHING DEPENDS ON ORDER. Each notional is the correctly rounded
product of its factors -- taken on the floats' exact rational values and
rounded once -- so 3 contracts x 50 is the same bits as 150 x 1, and a
base-currency line's base notional is its local notional exactly. Every
total is `math.fsum`, the correctly rounded sum, so reordering positions
cannot move a total by a bit. Long minus short equals net to the rounding
of the three sums, each correctly rounded on its own.

SIDE IS THE SIGN OF THE QUANTITY. A price may be zero (a worthless line
still held) or negative (crude oil, power and calendar spreads have printed
below zero), and the notional then carries the product's sign: a long at
-37.63 has a negative notional. Long is the sum of the long lines'
notionals and short is minus the sum of the short lines', so long - short =
net whatever the prices; long + short is gross only when no price is
negative.

REFUSED, naming the argument and the value: a zero or non-finite quantity,
a non-finite price, a multiplier that is not finite and positive, an
unknown currency, a missing FX rate, a base-currency rate other than
exactly 1, one instrument on two lines in one currency, and more than
MAX_POSITIONS lines. A zero quantity is not a position, as an option leg
of zero is not a leg elsewhere in the library. A repeated instrument is
refused rather than added up or kept as lots, because which of those the
caller means -- two reports concatenated, two lots, or a long held against
a short -- is the caller's decision, and margin and financing differ on it.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from standard_quant_tools.error import ValidationError

from .currency import currency_code

__all__ = [
    "MAX_ABS_PRICE",
    "MAX_ABS_QUANTITY",
    "MAX_BORROW_FEE_BPS",
    "MAX_FX_RATE",
    "MAX_INSTRUMENT_LENGTH",
    "MAX_MULTIPLIER",
    "MAX_POSITIONS",
    "MIN_FX_RATE",
    "CurrencyExposure",
    "ExposureSummary",
    "PositionValue",
    "check_borrow_fee_bps",
    "check_distinct_lines",
    "check_fx_rates",
    "check_instrument",
    "check_multiplier",
    "check_position_count",
    "check_price",
    "check_quantity",
    "check_rate_coverage",
    "exact_product",
    "exposure_summary",
    "position_values",
]

#: The most lines one call takes. Payloads of this size are about half a
#: megabyte of JSON; a larger set of holdings is aggregated first.
MAX_POSITIONS = 5_000
#: Magnitude bounds. They are not a judgement of what is plausible -- a
#: face amount in a currency of a million to the dollar is large -- but
#: they keep every product and sum finite: 1e18 x 1e15 x 1e9 x 1e12 x 5,000
#: is far inside a float.
MAX_ABS_QUANTITY = 1e18
MAX_ABS_PRICE = 1e15
MAX_MULTIPLIER = 1e9
#: Units of base currency per one unit of another. Real rates span about
#: 1e-10 (a rial in ounces of gold) to 1e10; outside this range the rate is
#: a unit error.
MIN_FX_RATE = 1e-12
MAX_FX_RATE = 1e12
#: 1,000% a year, the bound the library puts on every annual rate (10 as a
#: decimal). Hard-to-borrow names do pay several hundred percent.
MAX_BORROW_FEE_BPS = 100_000.0
MAX_INSTRUMENT_LENGTH = 100

Side = Literal["long", "short"]

#: How many offending lines a refusal names before counting the rest.
_SHOWN = 5


@dataclass(frozen=True)
class PositionValue:
    """One position valued: its inputs, its side and both notionals."""

    instrument: str
    currency: str
    side: Side
    quantity: float
    price: float
    multiplier: float
    #: Units of base currency per one unit of `currency`; exactly 1.0 for
    #: a line in the base currency.
    fx_rate: float
    #: quantity x price x multiplier, in `currency`.
    local_notional: float
    #: quantity x price x multiplier x fx_rate, in the base currency.
    base_notional: float


@dataclass(frozen=True)
class CurrencyExposure:
    """The lines quoted in one currency, summed. Amounts are in the base
    currency except `local_net`, which is in this one -- the amount a
    hedge of this currency is sized on."""

    currency: str
    fx_rate: float
    positions: int
    local_net: float
    net: float
    gross: float
    long: float
    short: float
    #: This currency's gross over the holdings' gross; None when the
    #: holdings' gross is zero (see ExposureSummary).
    share_of_gross: Optional[float]


@dataclass(frozen=True)
class ExposureSummary:
    """Gross, net, long and short of the holdings in the base currency, and
    their exposure per currency (sorted by code)."""

    base_currency: str
    positions: int
    long_positions: int
    short_positions: int
    gross: float
    net: float
    long: float
    short: float
    currencies: Tuple[CurrencyExposure, ...]
    #: Currencies given a rate in fx_rates that no line is quoted in. Not an
    #: error -- a caller's rate table may cover more than these holdings --
    #: but listed, so a rate meant for a line that is missing is visible.
    unused_fx_rates: Tuple[str, ...]
    #: Why every share_of_gross is None, when it is.
    share_of_gross_null_reason: Optional[str]


def _listed(items: Sequence[str]) -> str:
    shown = ", ".join(items[:_SHOWN])
    more = len(items) - _SHOWN
    return shown + (f" and {more} more" if more > 0 else "")


def _real(value: Any, argument: str) -> float:
    """`value` as a finite float, or a refusal naming `argument`."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{argument}={value!r} is not a number. Give a number.")
    try:
        number = float(value)
    except OverflowError:
        number = math.inf
    if not math.isfinite(number):
        raise ValidationError(
            f"{argument}={value!r} is not finite. Every notional and total is "
            "a product or sum of it, so one NaN or infinity makes them all "
            "non-finite. Give the number."
        )
    return number


def check_quantity(value: Any, argument: str = "quantity") -> float:
    """A signed, finite, non-zero quantity within MAX_ABS_QUANTITY."""
    quantity = _real(value, argument)
    if quantity == 0:
        raise ValidationError(
            f"{argument}={value!r} is zero, and a zero quantity is not a "
            "position. Drop the line; a short is a negative quantity."
        )
    if abs(quantity) > MAX_ABS_QUANTITY:
        raise ValidationError(
            f"{argument}={value!r} is beyond {MAX_ABS_QUANTITY:g} in "
            "magnitude, the most this accepts. Give the quantity in units, "
            "shares, contracts or face amount."
        )
    return quantity


def check_price(value: Any, argument: str = "price") -> float:
    """A finite price within MAX_ABS_PRICE; zero and negative are kept."""
    price = _real(value, argument)
    if abs(price) > MAX_ABS_PRICE:
        raise ValidationError(
            f"{argument}={value!r} is beyond {MAX_ABS_PRICE:g} in magnitude, "
            "the most this accepts. Give the price of one unit."
        )
    return price


def check_multiplier(value: Any, argument: str = "multiplier") -> float:
    """A finite multiplier in (0, MAX_MULTIPLIER]."""
    multiplier = _real(value, argument)
    if multiplier <= 0:
        raise ValidationError(
            f"{argument}={value!r} must be greater than 0: it is the currency "
            "one point of price is worth per unit (50 for an E-mini S&P 500 "
            "future, 1 for a share). A short is a negative quantity, not a "
            "negative multiplier."
        )
    if multiplier > MAX_MULTIPLIER:
        raise ValidationError(
            f"{argument}={value!r} is above {MAX_MULTIPLIER:g}, the most this "
            "accepts. Give the point value of one contract."
        )
    return multiplier


def check_borrow_fee_bps(
    value: Any, argument: str = "borrow_fee_bps"
) -> Optional[float]:
    """None, or a finite annual fee in [0, MAX_BORROW_FEE_BPS] bps."""
    if value is None:
        return None
    fee = _real(value, argument)
    if fee < 0:
        raise ValidationError(
            f"{argument}={value!r} is negative. It is the annual fee paid to "
            "borrow a short, in basis points, and is 0 or more."
        )
    if fee > MAX_BORROW_FEE_BPS:
        raise ValidationError(
            f"{argument}={value!r} is above {MAX_BORROW_FEE_BPS:,.0f} bps "
            "(1,000% a year), the most this accepts. It is in basis points: "
            "300 is 3% a year."
        )
    return fee


def check_instrument(value: Any, argument: str = "instrument") -> str:
    """A non-empty identifier, stripped, of at most MAX_INSTRUMENT_LENGTH."""
    if not isinstance(value, str):
        raise ValidationError(
            f"{argument}={value!r} is not text. Give the instrument's "
            "identifier as a string, e.g. 'AAPL' or an ISIN."
        )
    text = value.strip()
    if not text:
        raise ValidationError(
            f"{argument} is empty. Give the instrument's identifier, e.g. "
            "'AAPL' or an ISIN."
        )
    if len(text) > MAX_INSTRUMENT_LENGTH:
        raise ValidationError(
            f"{argument} is {len(text)} characters, above the "
            f"{MAX_INSTRUMENT_LENGTH} this accepts. Give a shorter identifier."
        )
    return text


def check_position_count(count: int, argument: str = "positions") -> None:
    """At least one line and at most MAX_POSITIONS."""
    if count < 1:
        raise ValidationError(f"{argument} is empty. Give at least one position.")
    if count > MAX_POSITIONS:
        raise ValidationError(
            f"{argument} has {count:,} lines, above the {MAX_POSITIONS:,} one "
            "call takes. Aggregate the lines first -- e.g. one line per "
            "instrument and currency."
        )


def check_fx_rates(
    fx_rates: Optional[Mapping[Any, Any]],
    *,
    base_currency: Optional[str] = None,
    argument: str = "fx_rates",
) -> Dict[str, float]:
    """
    The rate table with its codes upper-cased and every rate checked: a
    finite number in [MIN_FX_RATE, MAX_FX_RATE], one per currency, and the
    base currency's own rate, when one is given, exactly 1.
    """
    if fx_rates is None:
        return {}
    if not isinstance(fx_rates, Mapping):
        raise ValidationError(
            f"{argument}={fx_rates!r} is not a mapping. Give "
            "{currency: units of base_currency per one unit of it}."
        )
    rates: Dict[str, float] = {}
    spelled: Dict[str, Any] = {}
    for key, value in fx_rates.items():
        code = currency_code(key, argument=f"{argument} key")
        if code in rates:
            raise ValidationError(
                f"{argument} names {code} twice ({spelled[code]!r} and "
                f"{key!r}). Give one rate per currency."
            )
        rate = _real(value, f"{argument}[{key!r}]")
        if rate <= 0:
            raise ValidationError(
                f"{argument}[{key!r}]={value!r} must be greater than 0: it is "
                f"how many units of the base currency one {code} buys."
            )
        if not MIN_FX_RATE <= rate <= MAX_FX_RATE:
            raise ValidationError(
                f"{argument}[{key!r}]={value!r} is outside "
                f"[{MIN_FX_RATE:g}, {MAX_FX_RATE:g}]; a rate that far from 1 "
                "is a unit error. Give units of the base currency per one "
                f"{code}."
            )
        rates[code] = rate
        spelled[code] = key
    if base_currency is not None and base_currency in rates:
        if rates[base_currency] != 1.0:
            raise ValidationError(
                f"{argument}[{spelled[base_currency]!r}]="
                f"{fx_rates[spelled[base_currency]]!r} is the base currency's "
                f"own rate, which is exactly 1: one {base_currency} is one "
                f"{base_currency}. Drop it or give 1. To report in another "
                "currency, change base_currency."
            )
    return rates


def check_rate_coverage(
    lines: Sequence[Tuple[int, str, str]],
    base_currency: str,
    rates: Mapping[str, float],
    argument: str = "fx_rates",
) -> None:
    """
    Refuse unless every currency a line is quoted in, other than the base,
    has a rate. `lines` holds (position index, instrument, currency).
    """
    missing: Dict[str, List[str]] = {}
    for index, instrument, currency in lines:
        if currency != base_currency and currency not in rates:
            missing.setdefault(currency, []).append(
                f"positions[{index}] {instrument!r}"
            )
    if not missing:
        return
    held = "; ".join(
        f"{code}, held by {_listed(where)}" for code, where in sorted(missing.items())
    )
    first = min(missing)
    raise ValidationError(
        f"{argument} has no rate for {held}. {argument} gives units of "
        f"base_currency ({base_currency}) per ONE unit of every other "
        f"currency held: add {{'{first}': <{base_currency} per 1 {first}>}}, "
        f"e.g. 1.10 for EUR when base_currency is USD and 1 EUR = 1.10 USD."
    )


def check_distinct_lines(
    lines: Sequence[Tuple[int, str, str]], argument: str = "positions"
) -> None:
    """
    Refuse an instrument on two lines in one currency. `lines` holds
    (position index, instrument, currency); instruments are compared
    without regard to case, so 'aapl' and 'AAPL' are one instrument.
    """
    where: Dict[Tuple[str, str], List[int]] = {}
    names: Dict[Tuple[str, str], str] = {}
    for index, instrument, currency in lines:
        key = (instrument.casefold(), currency)
        where.setdefault(key, []).append(index)
        names.setdefault(key, instrument)
    repeated = [
        f"{names[key]!r} in {key[1]} at "
        + " and ".join(f"{argument}[{i}]" for i in indices)
        for key, indices in where.items()
        if len(indices) > 1
    ]
    if repeated:
        raise ValidationError(
            f"{argument} repeats {_listed(repeated)}. One instrument in one "
            "currency is one line here, and netting is yours to decide: add "
            "the quantities into one line (lots on one side with their own "
            "borrow fees net to one line at the quantity-weighted mean fee, "
            "which charges the same), or, if they are distinct holdings, "
            "give each its own identifier."
        )


def exact_product(*factors: float) -> float:
    """
    The product of `factors`, correctly rounded: computed on their exact
    rational values and rounded once, so it does not depend on the order
    they are multiplied in. Raises OverflowError beyond the float range.
    """
    numerator, denominator = 1, 1
    for factor in factors:
        top, bottom = float(factor).as_integer_ratio()
        numerator *= top
        denominator *= bottom
    # int / int is correctly rounded in CPython; the denominators are powers
    # of two, so this is one rounding of the exact product.
    return numerator / denominator


_MISSING = object()


def _attribute(position: Any, name: str, index: int, default: Any = _MISSING) -> Any:
    if isinstance(position, Mapping):
        value = position.get(name, default)
    else:
        value = getattr(position, name, default)
    if value is _MISSING:
        raise ValidationError(
            f"positions[{index}] has no {name}. A position needs instrument, "
            "quantity, price and currency; multiplier defaults to 1."
        )
    return value


@dataclass(frozen=True)
class _Line:
    index: int
    instrument: str
    quantity: float
    price: float
    multiplier: float
    currency: str


def _lines(positions: Sequence[Any]) -> List[_Line]:
    lines = []
    for index, position in enumerate(positions):
        at = f"positions[{index}]"
        lines.append(
            _Line(
                index=index,
                instrument=check_instrument(
                    _attribute(position, "instrument", index), f"{at}.instrument"
                ),
                quantity=check_quantity(
                    _attribute(position, "quantity", index), f"{at}.quantity"
                ),
                price=check_price(_attribute(position, "price", index), f"{at}.price"),
                multiplier=check_multiplier(
                    _attribute(position, "multiplier", index, 1.0),
                    f"{at}.multiplier",
                ),
                currency=currency_code(
                    _attribute(position, "currency", index), argument=f"{at}.currency"
                ),
            )
        )
    return lines


def _valued(
    positions: Sequence[Any],
    base_currency: str,
    fx_rates: Optional[Mapping[Any, Any]],
) -> Tuple[List[PositionValue], str, Dict[str, float]]:
    positions = list(positions)
    check_position_count(len(positions))
    base = currency_code(base_currency, argument="base_currency")
    rates = check_fx_rates(fx_rates, base_currency=base)
    lines = _lines(positions)
    keys = [(line.index, line.instrument, line.currency) for line in lines]
    check_distinct_lines(keys)
    check_rate_coverage(keys, base, rates)
    values = []
    for line in lines:
        fx_rate = 1.0 if line.currency == base else rates[line.currency]
        try:
            local = exact_product(line.quantity, line.price, line.multiplier)
            in_base = exact_product(line.quantity, line.price, line.multiplier, fx_rate)
        except OverflowError:
            raise ValidationError(
                f"positions[{line.index}] ({line.instrument!r}): quantity x "
                "price x multiplier x rate overflows a float. One of them is "
                "a unit error."
            ) from None
        values.append(
            PositionValue(
                instrument=line.instrument,
                currency=line.currency,
                side="long" if line.quantity > 0 else "short",
                quantity=line.quantity,
                price=line.price,
                multiplier=line.multiplier,
                fx_rate=fx_rate,
                local_notional=local,
                base_notional=in_base,
            )
        )
    return values, base, rates


def position_values(
    positions: Sequence[Any],
    base_currency: str,
    fx_rates: Optional[Mapping[Any, Any]] = None,
) -> List[PositionValue]:
    """
    Each position valued, in input order.

    `positions` is a sequence of objects or mappings carrying `instrument`,
    `quantity`, `price`, `currency` and optionally `multiplier` (default 1)
    -- the agent's `Position` model is one. `fx_rates` maps a currency to
    units of `base_currency` per ONE unit of it (base USD, 1 EUR = 1.10 USD:
    {'EUR': 1.10}); it needs every non-base currency held, the base's own
    rate is exactly 1 when given, and rates for currencies not held are
    allowed. Refusals are `ValidationError`s naming the argument and value.
    """
    values, _base, _rates = _valued(positions, base_currency, fx_rates)
    return values


_ZERO_GROSS = (
    "every position's notional is zero -- every price is zero -- so the "
    "holdings have no gross to take a share of"
)


def _side_sums(values: Sequence[PositionValue]) -> Tuple[float, float, float, float]:
    """(net, gross, long, short) of `values`, each correctly rounded."""
    net = math.fsum(v.base_notional for v in values)
    gross = math.fsum(abs(v.base_notional) for v in values)
    long = math.fsum(v.base_notional for v in values if v.side == "long")
    # 0.0 - x rather than -x, so no shorts reads 0.0 rather than -0.0.
    short = 0.0 - math.fsum(v.base_notional for v in values if v.side == "short")
    return net, gross, long, short


def exposure_summary(
    positions: Sequence[Any],
    base_currency: str,
    fx_rates: Optional[Mapping[Any, Any]] = None,
) -> ExposureSummary:
    """
    Gross, net, long and short in the base currency, and the exposure per
    currency with its share of gross. Arguments and refusals are those of
    `position_values`.

    net is the sum of signed base notionals and gross the sum of their
    absolute values; long and short sum the long and the short lines (see
    the module docstring for negative prices). Per currency, net, gross,
    long and short sum that currency's lines' base notionals, and local_net
    their local notionals. Every sum is `math.fsum`, so no total depends on
    the order of the positions.
    """
    values, base, rates = _valued(positions, base_currency, fx_rates)
    net, gross, long, short = _side_sums(values)
    by_currency: Dict[str, List[PositionValue]] = {}
    for value in values:
        by_currency.setdefault(value.currency, []).append(value)
    null_reason = None if gross > 0 else _ZERO_GROSS
    currencies = []
    for code in sorted(by_currency):
        group = by_currency[code]
        c_net, c_gross, c_long, c_short = _side_sums(group)
        currencies.append(
            CurrencyExposure(
                currency=code,
                fx_rate=group[0].fx_rate,
                positions=len(group),
                local_net=math.fsum(v.local_notional for v in group),
                net=c_net,
                gross=c_gross,
                long=c_long,
                short=c_short,
                share_of_gross=c_gross / gross if gross > 0 else None,
            )
        )
    return ExposureSummary(
        base_currency=base,
        positions=len(values),
        long_positions=sum(1 for v in values if v.side == "long"),
        short_positions=sum(1 for v in values if v.side == "short"),
        gross=gross,
        net=net,
        long=long,
        short=short,
        currencies=tuple(currencies),
        unused_fx_rates=tuple(
            sorted(code for code in rates if code != base and code not in by_currency)
        ),
        share_of_gross_null_reason=null_reason,
    )
