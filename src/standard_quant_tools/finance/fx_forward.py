"""
The FX forward that hedges a currency exposure, priced by covered interest
parity, and what hedging with it does to a foreign asset's return.

A PAIR IS QUOTED AS UNITS OF ITS SECOND CURRENCY PER ONE OF ITS FIRST:
EURUSD 1.1000 is 1.10 USD per EUR, USDJPY 150.00 is 150 JPY per USD. The
hedger's currency (`base_currency`) is one of the two and the other is the
foreign one, whichever order the market quotes them in.

COVERED INTEREST PARITY, EACH LEG ON ITS OWN DAY COUNT. Borrowing one unit
of the first currency for the period and lending its spot value in the
second leaves no profit only at the forward

    F = S x (1 + r_second x t_second) / (1 + r_first x t_first)

with simple money-market interest, each t that leg's year fraction under its
own convention: USD and EUR money markets count ACT/360 and GBP ACT/365F,
so the same 90 days are 0.25 of a year in one and 0.2466 in the other. The
conventions are the caller's, as the rates are: nothing here knows a
market's convention, and two equal rates on different day counts do not
give F = S. The growth ratio is formed before it multiplies the spot, so
equal growth on both legs gives F = S exactly.

THE BASIS IS ADDED TO THE FOREIGN LEG. Since 2008 forwards have not sat at
parity with money-market rates; the cross-currency basis is the spread that
closes the gap, and `basis_bps` here is added to the foreign currency's rate
(the one that is not `base_currency`). EUR and JPY have traded at a negative
basis against USD for most of that time: the forward then prices as if the
foreign rate were lower, and a USD-based hedger selling the foreign currency
forward picks the basis up on top of the rate differential. A basis of zero
is parity, the same expression with nothing added.

FORWARD POINTS are F - S in pips. A pip is 0.0001 of the quote for most
pairs and 0.01 when the quote currency (the second) is JPY; any other
convention is the caller's `pip_size`.

THE HEDGE. With X units of base per one foreign at spot and X_F at the
forward, the forward premium over the period is f = X_F / X - 1, and
`hedge_carry` is f over the base leg's year fraction: positive is a pickup,
negative a cost. Under parity with one day count, hedge_carry is
(r_base - r_foreign) / (1 + r_foreign x t): the rate differential,
discounted. A holder of a foreign asset with local return r_L who sells a
share h of its starting value forward, and sees the rate move by r_FX, earns

    unhedged = r_L + r_FX + r_L x r_FX
    hedged   = unhedged + h x (f - r_FX)

so a full hedge (h = 1) leaves r_L + f + r_L x r_FX: the cross term stays,
because the hedge is sized on the starting value and the asset's gain over
the period is not hedged. A forward's own value before delivery is not
modelled; the returns run to delivery.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, List, Optional, Tuple

from standard_quant_tools.delta_one.daycount import _convention as _canonical_convention
from standard_quant_tools.delta_one.daycount import to_date, year_fraction
from standard_quant_tools.delta_one.futures import DAY_COUNT_DENOMINATORS
from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import require_finite_scalar

from .currency import currency_code
from .positions import MAX_FX_RATE

__all__ = [
    "DEFAULT_PIP_SIZE",
    "JPY_PIP_SIZE",
    "MAX_BASIS_BPS",
    "MAX_FORWARD_DAYS",
    "MAX_RATE",
    "covered_interest_forward",
    "currency_pair",
    "fx_forward_hedge",
    "leg_year_fraction",
    "pip_size_for",
]

#: The pip of most quotes, and of a quote in JPY.
DEFAULT_PIP_SIZE = 0.0001
JPY_PIP_SIZE = 0.01
#: A hundred years of calendar days; beyond it a "forward" is a unit error.
MAX_FORWARD_DAYS = 36_600
#: 1,000% a year as a decimal, the bound the library puts on an annual rate.
MAX_RATE = 10.0
#: 100% a year in basis points: no basis has come within a tenth of it.
MAX_BASIS_BPS = 10_000.0
#: Past a year, forwards are priced off cross-currency swap curves with
#: compounding rather than one simple-interest period, and this says so.
_ONE_YEAR_DAYS = 366

_FUNC = "price_fx_forward_hedge"

#: Day counts that need only the number of days; the others count dates.
_DAYS_ONLY = ("ACT/360", "ACT/365F")


def _number(
    value: Any,
    name: str,
    *,
    low: Optional[float] = None,
    high: Optional[float] = None,
    low_open: bool = False,
) -> float:
    """A finite number inside [low, high] (or (low, high] with `low_open`),
    or a refusal naming `name`, the value and the range."""
    number = require_finite_scalar(value, name, _FUNC)
    below = low is not None and (number <= low if low_open else number < low)
    above = high is not None and number > high
    if below or above:
        left = "(" if low_open else "["
        lo = "-inf" if low is None else f"{low:g}"
        hi = "inf" if high is None else f"{high:g}"
        raise ValidationError(
            f"{name}={value!r} is outside {left}{lo}, {hi}]. "
            + _RANGE_HINTS.get(name, "Check the units.")
        )
    return number


_RANGE_HINTS = {
    "spot": "It is units of the pair's second currency per one of its first.",
    "spot_at_maturity": "It is the pair's spot at delivery, quoted as spot is.",
    "base_rate": "Rates are decimals: 0.053 is 5.3% a year.",
    "foreign_rate": "Rates are decimals: 0.053 is 5.3% a year.",
    "basis_bps": "The basis is in basis points: -25 is -0.25% a year.",
    "pip_size": "A pip is 0.0001 of most quotes and 0.01 of a JPY quote.",
    "asset_return": "It is a return over the period: 0.04 is +4%, and the "
    "least an asset can return is -1 (-100%).",
    "hedge_ratio": "It is the share of the asset's starting value sold "
    "forward: 1 hedges all of it, 0 none.",
}


def currency_pair(value: Any, argument: str = "pair") -> Tuple[str, str]:
    """
    A market quote's two ISO 4217 codes, in order: 'EURUSD' or 'EUR/USD' is
    ('EUR', 'USD'), one EUR priced in USD. Each code is read by
    `currency_code`, so pence and withdrawn codes are refused with the fix.
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"{argument}={value!r} is not a currency pair. Write the two ISO "
            "4217 codes together, the quoted one first: 'EURUSD' or "
            "'EUR/USD'."
        )
    text = value.strip()
    compact = text.replace("/", "", 1) if text.count("/") == 1 else text
    if len(compact) != 6 or not compact.isalpha():
        raise ValidationError(
            f"{argument}={value!r} is not a currency pair. Write the two ISO "
            "4217 codes together, the quoted one first: 'EURUSD' (or "
            "'EUR/USD') is units of USD per one EUR."
        )
    first = currency_code(compact[:3], argument=f"{argument} {value!r}, first code")
    second = currency_code(compact[3:], argument=f"{argument} {value!r}, second code")
    if first == second:
        raise ValidationError(
            f"{argument}={value!r} names {first} twice. A pair is two "
            "different currencies, e.g. 'EURUSD'."
        )
    return first, second


def pip_size_for(second: str) -> float:
    """The pip of a quote in `second`: 0.01 for JPY, otherwise 0.0001."""
    return JPY_PIP_SIZE if second == "JPY" else DEFAULT_PIP_SIZE


def leg_year_fraction(
    days: int,
    day_count: str,
    start_date: Optional[Any] = None,
    *,
    argument: str = "day_count",
) -> float:
    """
    The year fraction of `days` calendar days under `day_count`. ACT/360
    and ACT/365F need only the count; ACT/ACT and 30/360 count from dates,
    so they need `start_date`, and the period then ends `days` later.
    """
    try:
        convention = _canonical_convention(day_count)
    except ValidationError as exc:
        raise ValidationError(f"{argument}: {exc}") from None
    if start_date is None:
        if convention not in _DAYS_ONLY:
            raise ValidationError(
                f"{argument}={day_count!r} counts from dates -- ACT/ACT splits "
                "the period at a year end and 30/360 clamps a 31st -- and no "
                "start_date was given. Give start_date (the spot date, ISO), "
                "or use ACT/360 or ACT/365F, which need only the days."
            )
        return days / DAY_COUNT_DENOMINATORS[convention]
    start = to_date(start_date, "start_date")
    return year_fraction(start, start + _dt.timedelta(days=days), convention=convention)


def covered_interest_forward(
    spot: float,
    *,
    first_rate: float,
    first_years: float,
    second_rate: float,
    second_years: float,
) -> float:
    """
    F = S x (1 + r_second t_second) / (1 + r_first t_first): the forward of a
    pair quoted as units of its second currency per one of its first. The
    ratio is formed first, so equal growth on both legs returns S exactly.
    """
    growth_first = 1.0 + first_rate * first_years
    growth_second = 1.0 + second_rate * second_years
    for leg, growth, rate, years in (
        ("first", growth_first, first_rate, first_years),
        ("second", growth_second, second_rate, second_years),
    ):
        if not growth > 0:
            raise ValidationError(
                f"the {leg} currency's leg grows by 1 + {rate:g} x {years:g} = "
                f"{growth:g} over the period, which is not above zero, so no "
                "forward exists. A rate that negative over that long is a "
                "unit error: rates are decimals, -0.005 is -0.5% a year."
            )
    return spot * (growth_second / growth_first)


def fx_forward_hedge(
    *,
    pair: str,
    spot: float,
    days: int,
    base_currency: str,
    base_rate: float,
    foreign_rate: float,
    base_day_count: str,
    foreign_day_count: str,
    basis_bps: float = 0.0,
    pip_size: Optional[float] = None,
    start_date: Optional[Any] = None,
    asset_return: Optional[float] = None,
    spot_at_maturity: Optional[float] = None,
    hedge_ratio: float = 1.0,
) -> Dict[str, Any]:
    """
    The parity forward of `pair` for delivery `days` after spot, its points,
    the annualized pickup or cost of hedging the foreign currency into
    `base_currency`, and, given `asset_return` and `spot_at_maturity`, the
    hedged and unhedged returns of a foreign asset over the period. See the
    module docstring for every convention. Refusals are `ValidationError`s
    naming the argument, the value and the fix.
    """
    first, second = currency_pair(pair)
    base = currency_code(base_currency, argument="base_currency")
    if base not in (first, second):
        raise ValidationError(
            f"base_currency={base_currency!r} is neither currency of "
            f"{first}{second}. The hedger's currency is one of the pair's: "
            f"for a {base}-based holder of a {first} asset, price "
            f"{first}{base}."
        )
    foreign = second if base == first else first
    s = _number(spot, "spot", low=0.0, high=MAX_FX_RATE, low_open=True)
    if isinstance(days, bool) or not isinstance(days, int):
        raise ValidationError(
            f"days={days!r} is not a whole number. Give the calendar days from "
            "the spot date to delivery, e.g. 90."
        )
    if not 1 <= days <= MAX_FORWARD_DAYS:
        raise ValidationError(
            f"days={days} is outside [1, {MAX_FORWARD_DAYS:,}]. Give the "
            "calendar days from the spot date to delivery, e.g. 90."
        )
    r_base = _number(base_rate, "base_rate", low=-MAX_RATE, high=MAX_RATE)
    r_foreign = _number(foreign_rate, "foreign_rate", low=-MAX_RATE, high=MAX_RATE)
    basis = _number(basis_bps, "basis_bps", low=-MAX_BASIS_BPS, high=MAX_BASIS_BPS)
    if pip_size is None:
        pip, pip_rule = pip_size_for(second), (
            "0.01, the pip of a JPY quote"
            if second == "JPY"
            else "0.0001, the pip of a quote outside JPY"
        )
    else:
        pip = _number(pip_size, "pip_size", low=0.0, high=1.0, low_open=True)
        pip_rule = "given"
    h = _number(hedge_ratio, "hedge_ratio", low=0.0, high=2.0)
    if (asset_return is None) != (spot_at_maturity is None):
        given, missing = (
            ("asset_return", "spot_at_maturity")
            if spot_at_maturity is None
            else ("spot_at_maturity", "asset_return")
        )
        raise ValidationError(
            f"{given} was given without {missing}. The hedged and unhedged "
            "returns need both: the asset's return in its own currency and "
            "the pair's spot at delivery. Give both, or neither for the "
            "forward alone."
        )

    t_base = leg_year_fraction(
        days, base_day_count, start_date, argument="base_day_count"
    )
    t_foreign = leg_year_fraction(
        days, foreign_day_count, start_date, argument="foreign_day_count"
    )
    # The basis is added to the foreign leg; with zero it is that leg's rate.
    r_foreign_used = r_foreign + basis / 10_000.0
    legs = {
        base: (r_base, t_base),
        foreign: (r_foreign_used, t_foreign),
    }
    forward = covered_interest_forward(
        s,
        first_rate=legs[first][0],
        first_years=legs[first][1],
        second_rate=legs[second][0],
        second_years=legs[second][1],
    )
    foreign_first = first == foreign
    # Units of base per one foreign rise when the pair does if the foreign
    # currency is quoted first, and fall when it does if it is second.
    premium = (forward / s - 1.0) if foreign_first else (s / forward - 1.0)

    warnings: List[str] = []
    conventions = {
        base: _canonical_convention(base_day_count),
        foreign: _canonical_convention(foreign_day_count),
    }
    if conventions[base] != conventions[foreign]:
        warnings.append(
            f"The legs count days differently ({base} {conventions[base]}, "
            f"{foreign} {conventions[foreign]}), so equal rates do not give a "
            "forward equal to spot: each leg accrues over its own year fraction."
        )
    if days > _ONE_YEAR_DAYS:
        warnings.append(
            f"{days} days is more than a year. This prices the forward with "
            "one period of simple money-market interest per leg; forwards "
            "that long are priced off cross-currency swap curves with "
            "compounding, so treat the result as parity at these rates, not "
            "as a market quote."
        )

    period = None
    if asset_return is not None:
        r_local = _number(asset_return, "asset_return", low=-1.0, high=10.0)
        s1 = _number(
            spot_at_maturity,
            "spot_at_maturity",
            low=0.0,
            high=MAX_FX_RATE,
            low_open=True,
        )
        r_fx = (s1 / s - 1.0) if foreign_first else (s / s1 - 1.0)
        cross = r_local * r_fx
        hedge = h * (premium - r_fx)
        period = {
            "local_return": r_local,
            "fx_return": r_fx,
            "cross_return": cross,
            "hedge_return": hedge,
            "unhedged_return": math.fsum([r_local, r_fx, cross]),
            "hedged_return": math.fsum([r_local, r_fx, cross, hedge]),
            "hedge_ratio": h,
        }

    maturity = None
    if start_date is not None:
        maturity = (
            to_date(start_date, "start_date") + _dt.timedelta(days=days)
        ).isoformat()

    return {
        "pair": f"{first}{second}",
        "base_currency": base,
        "foreign_currency": foreign,
        "spot": s,
        "forward": forward,
        "forward_points": (forward - s) / pip,
        "pip_size": pip,
        "pip_size_rule": pip_rule,
        "days": days,
        "start_date": (
            to_date(start_date, "start_date").isoformat()
            if start_date is not None
            else None
        ),
        "maturity_date": maturity,
        "day_counts": conventions,
        "year_fractions": {base: t_base, foreign: t_foreign},
        "rates": {base: r_base, foreign: r_foreign_used},
        "basis_bps": basis,
        "rate_differential": r_base - r_foreign_used,
        "forward_premium": premium,
        # A 30/360 leg from a 30th to a 31st counts no days, and a premium
        # over no time has no annual rate: NaN, nulled with its reason.
        "hedge_carry": premium / t_base if t_base > 0 else math.nan,
        "period": period,
        "warnings": warnings,
    }
