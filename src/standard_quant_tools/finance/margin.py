"""
The margin a US broker must hold against a set of holdings, and how long a
stress path leaves before the first maintenance call.

THE RULES, each line naming the one it applies:

- Initial, Reg T (12 CFR 220.12). A long stock or ETF: 50% of its market
  value (220.12(a)). A short: 150% of its market value held in the account,
  its own proceeds included (220.12(c)) -- which is 50% of it as equity,
  the basis every requirement here is stated on. Options take the
  requirement of the options rules (220.12(f)), below.
- Maintenance, FINRA Rule 4210(c). A long: 25% of its market value
  ((c)(1)). A short under $5 a share: the greater of $2.50 a share and 100%
  of its value ((c)(2)); at $5 or more, the greater of $5.00 a share and 30%
  ((c)(3)).
- A leveraged or inverse ETF: the 25% or 30% times its leverage factor,
  never above 100% of its value -- the higher requirement FINRA set under
  4210(f)(8)(A) in Regulatory Notice 09-53. A 3x fund held long is 75%.
- An uncovered short equity option, FINRA 4210(f)(2)(E)(i): 100% of the
  option's value plus 20% of the underlying's, less any out-of-the-money
  amount, and at least the option's value plus 10% of the underlying's (a
  call) or of the strike (a put). A long option is paid in full: 100% of
  its value, the requirement for one expiring within nine months (one
  listed with longer to run may be margined at 75%; that is not applied).
- The account's initial requirement, FINRA 4210(b): the greater of the Reg
  T amount, the maintenance amount, and $2,000 of equity ((b)(4); a book
  of purchases alone needs no more than their cost).

EVERY REQUIREMENT IS EQUITY. Account equity is cash plus the longs less
the shorts at market, and each requirement is the equity the account must
hold against its line, so the lines add up and the totals compare with
the account's equity directly. A short stock's 150% is its proceeds plus
50%; on this basis it is the 50%.

STRATEGY-BASED, WITHOUT OFFSETS. Every line is margined on its own: a
short option is uncovered even when the stock is held, and a spread or a
hedge earns no reduction. Portfolio margin and a broker's house terms
beyond a rate on market value are not modelled.

A HOUSE RATE (per instrument, a maintenance rate on market value) only
raises a requirement: one below the regulatory rate of its line is
refused, naming the minimum, and the per-share floors still apply above
it. An option line takes none, since its rule has two rates.

THE DOLLAR AMOUNTS ($2.50, $5.00, $2,000) are dollars, so the base
currency is USD; a line quoted in another currency is read in dollars at
its FX rate, its price included when the price tiers are read.

THE RUNWAY walks a stress path the caller supplies, one step a day:
equity changes against fixed requirements, or daily returns that revalue
the positions and recompute their requirements. The first day the equity
falls below maintenance is a call, for the amount that restores it to the
initial requirement (the default) or to maintenance. A call the market
does not cure within `cure_days` -- the equity back at the call's target
-- starts forced deleveraging on its last day. The buffer is the smallest
cash added today that leaves no day below maintenance.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from standard_quant_tools.error import ValidationError

from .currency import currency_code
from .positions import (
    MAX_ABS_PRICE,
    PositionValue,
    _attribute,
    exact_product,
    position_values,
)

__all__ = [
    "DEFAULT_CURE_DAYS",
    "INSTRUMENT_KINDS",
    "MAX_LEVERAGE",
    "MAX_PATH_CELLS",
    "MAX_PATH_DAYS",
    "MarginCall",
    "MarginLine",
    "MarginTerms",
    "MarginRequirement",
    "MarginRunway",
    "check_margin_terms",
    "margin_requirement",
    "margin_runway",
]

INSTRUMENT_KINDS = ("equity", "etf", "call", "put")
RESTORE_TO = ("initial", "maintenance")

REG_T_RATE = 0.50
LONG_RATE = 0.25
SHORT_RATE = 0.30
LOW_PRICE = 5.00
LOW_PRICE_PER_SHARE = 2.50
PER_SHARE = 5.00
OPTION_UNDERLYING_RATE = 0.20
OPTION_MINIMUM_RATE = 0.10
MINIMUM_EQUITY = 2_000.0
#: The most leverage a fund's factor may state. A 3x fund is 3; the bound
#: catches a percentage given for a factor (300 for 3x).
MAX_LEVERAGE = 10.0
#: FINRA 4210(f)(6): a deficiency is to be met within 15 business days.
DEFAULT_CURE_DAYS = 15
#: Forty years of daily steps.
MAX_PATH_DAYS = 10_080
#: Positions x days a revalued path computes; above it, aggregate lines.
MAX_PATH_CELLS = 5_000_000
#: The most a house rate may state: ten times market value.
MAX_HOUSE_RATE = 10.0

RULE_REG_T_LONG = "Reg T 12 CFR 220.12(a): 50% of a long's market value"
RULE_REG_T_SHORT = (
    "Reg T 12 CFR 220.12(c): 150% of a short's market value, its proceeds "
    "included -- 50% of it as equity"
)
RULE_REG_T_OPTION = (
    "Reg T 12 CFR 220.12(f): the options rules' requirement, as maintenance"
)
RULE_LONG = "FINRA 4210(c)(1): 25% of a long's market value"
RULE_SHORT_UNDER_5 = (
    "FINRA 4210(c)(2): a short under $5, the greater of $2.50 a share and "
    "100% of its market value"
)
RULE_SHORT = (
    "FINRA 4210(c)(3): a short at $5 or more, the greater of $5.00 a share "
    "and 30% of its market value"
)
RULE_LONG_OPTION = (
    "FINRA 4210(f)(2): a long option is paid in full, 100% of its value "
    "(the rule to nine months' expiry; 75% past it is not applied)"
)
RULE_SHORT_CALL = (
    "FINRA 4210(f)(2)(E)(i): an uncovered short call, 100% of its value + 20% "
    "of the underlying less the out-of-the-money amount, at least its value "
    "+ 10% of the underlying"
)
RULE_SHORT_PUT = (
    "FINRA 4210(f)(2)(E)(i): an uncovered short put, 100% of its value + 20% "
    "of the underlying less the out-of-the-money amount, at least its value "
    "+ 10% of the strike"
)
RULE_INITIAL = (
    "FINRA 4210(b): the greater of the Reg T amount, the maintenance amount "
    "and $2,000 of equity"
)


@dataclass(frozen=True)
class MarginLine:
    """One position's requirements, in the base currency (USD)."""

    instrument: str
    currency: str
    kind: str
    side: str
    quantity: float
    price: float
    leverage: float
    #: The line's value at market in USD, a positive amount on either side.
    market_value: float
    #: Reg T initial margin, as equity.
    reg_t: float
    reg_t_rule: str
    maintenance: float
    maintenance_rule: str
    #: The maintenance rate on market value the rule applied: 0.25, 0.30,
    #: 1.0, a leveraged rate or a house rate; None for an option line.
    maintenance_rate: Optional[float]
    house_rate: Optional[float]
    #: Options only: the underlying's value per the position, and the
    #: out-of-the-money amount, both in USD.
    underlying_value: Optional[float]
    out_of_the_money: Optional[float]


@dataclass(frozen=True)
class MarginRequirement:
    """A set of holdings' margin requirements, totalled."""

    base_currency: str
    lines: Tuple[MarginLine, ...]
    long_market_value: float
    short_market_value: float
    #: The sum of the lines' Reg T amounts.
    reg_t: float
    #: The sum of the lines' maintenance amounts.
    maintenance: float
    #: FINRA 4210(b)(4): $2,000 of equity, or the purchases' cost when the
    #: book is purchases alone and costs less.
    minimum_equity: float
    #: The greater of reg_t, maintenance and minimum_equity.
    initial: float
    #: Which of the three set it: "reg_t", "maintenance" or "minimum_equity".
    initial_binding: str
    initial_rule: str
    notes: Tuple[str, ...]


# ── reading a margin line ───────────────────────────────────────────────


def _number(value: Any, argument: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{argument}={value!r} is not a number. Give a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{argument}={value!r} is not finite. Give the number.")
    return number


@dataclass(frozen=True)
class MarginTerms:
    """What a line is for margin: its kind, an ETF's leverage factor, and
    an option's strike and underlying price per share."""

    kind: str
    leverage: float
    strike: Optional[float]
    underlying_price: Optional[float]


def check_margin_terms(
    kind: Any,
    leverage: Any,
    strike: Any,
    underlying_price: Any,
    *,
    multiplier: float,
    price: float,
    at: str = "position",
) -> MarginTerms:
    """
    A line's margin terms checked, or a refusal naming `at`: a kind in
    INSTRUMENT_KINDS; a leverage factor in [1, MAX_LEVERAGE] on an ETF and
    1 on anything else; a strike and an underlying price above zero on an
    option and on nothing else; a multiplier of 1 on a stock or an ETF,
    whose quantity is its shares; and a price of zero or more.
    """
    if kind not in INSTRUMENT_KINDS:
        raise ValidationError(
            f"{at}.kind={kind!r} must be one of {INSTRUMENT_KINDS}: 'equity' "
            "(a stock), 'etf' (with its leverage factor), or 'call'/'put' (an "
            "equity option, with its strike and underlying_price)."
        )
    leverage = _number(leverage, f"{at}.leverage")
    if price < 0:
        raise ValidationError(
            f"{at}.price={price!r} is negative. Margin is a share of market "
            "value and an amount per share, and neither reads a negative "
            "price; a line priced below zero is not a margin security."
        )
    if kind == "etf":
        if not 1.0 <= leverage <= MAX_LEVERAGE:
            raise ValidationError(
                f"{at}.leverage={leverage!r} must be in [1, {MAX_LEVERAGE:g}]: "
                "the size of the fund's daily multiple, 3 for a 3x or a -3x "
                "fund, 1 for an unleveraged one."
            )
    elif leverage != 1.0:
        raise ValidationError(
            f"{at}.leverage={leverage!r} on a {kind!r} line. The leverage "
            "factor is an ETF's; set kind='etf'. An option on a leveraged ETF "
            "takes a leveraged share of its underlying, which is not "
            "modelled here."
        )
    if kind in ("call", "put"):
        if strike is None or underlying_price is None:
            raise ValidationError(
                f"{at} is a {kind} and needs strike and underlying_price: the "
                "option rule is a share of the underlying's value less the "
                "out-of-the-money amount."
            )
        strike = _number(strike, f"{at}.strike")
        underlying_price = _number(underlying_price, f"{at}.underlying_price")
        for name, number in (
            ("strike", strike),
            ("underlying_price", underlying_price),
        ):
            if not 0 < number <= MAX_ABS_PRICE:
                raise ValidationError(
                    f"{at}.{name}={number!r} must be above 0 and at most "
                    f"{MAX_ABS_PRICE:g}: it is a price per share of the "
                    "underlying."
                )
        return MarginTerms(kind, 1.0, strike, underlying_price)
    if strike is not None or underlying_price is not None:
        raise ValidationError(
            f"{at} is a {kind!r} line with strike or underlying_price set. "
            "Those are an option's; set kind='call' or 'put', or drop them."
        )
    if multiplier != 1.0:
        raise ValidationError(
            f"{at}.multiplier={multiplier!r} on a {kind!r} line. A stock or "
            "ETF line is shares, and the per-share amounts of FINRA 4210(c) "
            "are read on them: give the share count as quantity and leave "
            "multiplier at 1."
        )
    return MarginTerms(kind, leverage, None, None)


def _terms(position: Any, value: PositionValue, index: int) -> MarginTerms:
    return check_margin_terms(
        _attribute(position, "kind", index, "equity"),
        _attribute(position, "leverage", index, 1.0),
        _attribute(position, "strike", index, None),
        _attribute(position, "underlying_price", index, None),
        multiplier=value.multiplier,
        price=value.price,
        at=f"positions[{index}]",
    )


def _regulatory_rate(kind: str, side: str, usd_price: float, leverage: float) -> float:
    """The maintenance rate on market value the rules set for a stock or
    ETF line, before any per-share floor."""
    if side == "long":
        return min(1.0, LONG_RATE * leverage)
    if usd_price < LOW_PRICE:
        return 1.0
    return min(1.0, SHORT_RATE * leverage)


def _rate_rule(kind: str, side: str, usd_price: float, leverage: float) -> str:
    if side == "long":
        base = RULE_LONG
    else:
        base = RULE_SHORT_UNDER_5 if usd_price < LOW_PRICE else RULE_SHORT
    rate = _regulatory_rate(kind, side, usd_price, leverage)
    if (
        kind == "etf"
        and leverage != 1.0
        and not (side == "short" and usd_price < LOW_PRICE)
    ):
        return (
            f"{base}, times the fund's leverage {leverage:g} = {rate:.0%}, at "
            "most 100% (FINRA 4210(f)(8)(A), Regulatory Notice 09-53)"
        )
    return base


def _house_rate(
    house: Mapping[str, float], value: PositionValue, index: int
) -> Optional[float]:
    if value.instrument not in house:
        return None
    return _number(house[value.instrument], f"house_rates[{value.instrument!r}]")


def _line(
    position: Any,
    value: PositionValue,
    index: int,
    house: Mapping[str, float],
) -> MarginLine:
    terms = _terms(position, value, index)
    shares = abs(value.quantity)
    market_value = abs(value.base_notional)
    house_rate = _house_rate(house, value, index)
    if terms.kind in ("call", "put"):
        if house_rate is not None:
            raise ValidationError(
                f"house_rates[{value.instrument!r}] is set on an option line "
                f"(positions[{index}]). The option rule has two rates -- 20% "
                "of the underlying and a 10% minimum -- so one house rate on "
                "market value does not say which it raises. Drop it."
            )
        return _option_line(value, terms, shares, market_value)
    usd_price = exact_product(value.price, value.fx_rate)
    reg_t_rule = RULE_REG_T_LONG if value.side == "long" else RULE_REG_T_SHORT
    reg_t = REG_T_RATE * market_value
    rate = _regulatory_rate(terms.kind, value.side, usd_price, terms.leverage)
    rule = _rate_rule(terms.kind, value.side, usd_price, terms.leverage)
    regulatory = rate * market_value
    if value.side == "short":
        per_share = LOW_PRICE_PER_SHARE if usd_price < LOW_PRICE else PER_SHARE
        regulatory = max(regulatory, per_share * shares)
    maintenance = regulatory
    applied_rate = rate
    if house_rate is not None:
        if house_rate < rate:
            raise ValidationError(
                f"house_rates[{value.instrument!r}]={house_rate!r} is below the "
                f"{rate:.0%} minimum for positions[{index}] ({rule}). A house "
                f"rate can only raise a requirement: give at least {rate:g}, "
                "or drop it to apply the rule's own rate."
            )
        if house_rate > MAX_HOUSE_RATE:
            raise ValidationError(
                f"house_rates[{value.instrument!r}]={house_rate!r} is above "
                f"{MAX_HOUSE_RATE:g}, ten times market value. It is a decimal: "
                "0.35 is 35%."
            )
        maintenance = max(house_rate * market_value, regulatory)
        applied_rate = house_rate
        rule = f"house rate {house_rate:.2%} of market value, at or above {rule}"
    return MarginLine(
        instrument=value.instrument,
        currency=value.currency,
        kind=terms.kind,
        side=value.side,
        quantity=value.quantity,
        price=value.price,
        leverage=terms.leverage,
        market_value=market_value,
        reg_t=reg_t,
        reg_t_rule=reg_t_rule,
        maintenance=maintenance,
        maintenance_rule=rule,
        maintenance_rate=applied_rate,
        house_rate=house_rate,
        underlying_value=None,
        out_of_the_money=None,
    )


def _option_line(
    value: PositionValue, terms: MarginTerms, contracts: float, market_value: float
) -> MarginLine:
    assert terms.strike is not None and terms.underlying_price is not None
    # Everything per share of the underlying, in the line's currency, then
    # scaled by contracts x multiplier x FX once.
    scale = exact_product(contracts, value.multiplier, value.fx_rate)
    s, k, premium = terms.underlying_price, terms.strike, value.price
    if terms.kind == "call":
        otm = max(k - s, 0.0)
        floor_base = s
    else:
        otm = max(s - k, 0.0)
        floor_base = k
    if value.side == "long":
        requirement = market_value
        rule = RULE_LONG_OPTION
    else:
        per_share = premium + max(
            OPTION_UNDERLYING_RATE * s - otm, OPTION_MINIMUM_RATE * floor_base
        )
        requirement = per_share * scale
        rule = RULE_SHORT_CALL if terms.kind == "call" else RULE_SHORT_PUT
    return MarginLine(
        instrument=value.instrument,
        currency=value.currency,
        kind=terms.kind,
        side=value.side,
        quantity=value.quantity,
        price=value.price,
        leverage=1.0,
        market_value=market_value,
        reg_t=requirement,
        reg_t_rule=RULE_REG_T_OPTION,
        maintenance=requirement,
        maintenance_rule=rule,
        maintenance_rate=None,
        house_rate=None,
        underlying_value=s * scale,
        out_of_the_money=otm * scale,
    )


def _usd(base_currency: Any) -> str:
    base = currency_code(base_currency, argument="base_currency")
    if base != "USD":
        raise ValidationError(
            f"base_currency={base_currency!r}: Reg T and FINRA 4210 are US "
            "rules stated in dollars -- $2.50 and $5.00 a share, $2,000 of "
            "equity -- so the base currency is USD. Give base_currency='USD' "
            "and fx_rates with each other currency's rate to the dollar."
        )
    return base


def _check_house(
    house_rates: Optional[Mapping[str, Any]], values: Sequence[PositionValue]
) -> Dict[str, float]:
    if not house_rates:
        return {}
    if not isinstance(house_rates, Mapping):
        raise ValidationError(
            f"house_rates={house_rates!r} is not a mapping. Give {{instrument: "
            "maintenance rate on market value}}, e.g. {'XYZ': 0.40}."
        )
    held = {v.instrument for v in values}
    unknown = [k for k in house_rates if k not in held]
    if unknown:
        raise ValidationError(
            f"house_rates names {unknown[:5]}, which no position holds "
            "(instruments match exactly, case included). Every key is the "
            "instrument of a line it raises."
        )
    return {k: _number(v, f"house_rates[{k!r}]") for k, v in house_rates.items()}


def margin_requirement(
    positions: Sequence[Any],
    base_currency: str = "USD",
    fx_rates: Optional[Mapping[Any, Any]] = None,
    house_rates: Optional[Mapping[str, Any]] = None,
) -> MarginRequirement:
    """
    Reg T initial and FINRA 4210 maintenance margin, line by line and for
    the account. See the module docstring for the rules.

    `positions` are what `position_values` reads, plus, on each, `kind`
    ('equity' by default, 'etf', 'call' or 'put'), `leverage` (an ETF's
    factor, default 1) and, on an option, `strike` and `underlying_price`
    per share; an option's `price` is its premium per share and its
    `multiplier` the contract size. `house_rates` maps an instrument to a
    maintenance rate on market value at or above the rule's own.
    """
    positions = list(positions)
    base = _usd(base_currency)
    values = position_values(positions, base, fx_rates)
    house = _check_house(house_rates, values)
    lines = tuple(
        _line(position, value, i, house)
        for i, (position, value) in enumerate(zip(positions, values))
    )
    return _totals(lines, base)


def _minimum_equity(long_value: float, any_short: bool) -> float:
    if any_short:
        return MINIMUM_EQUITY
    return min(MINIMUM_EQUITY, long_value)


def _totals(lines: Tuple[MarginLine, ...], base: str) -> MarginRequirement:
    long_value = math.fsum(x.market_value for x in lines if x.side == "long")
    short_value = math.fsum(x.market_value for x in lines if x.side == "short")
    reg_t = math.fsum(x.reg_t for x in lines)
    maintenance = math.fsum(x.maintenance for x in lines)
    minimum = _minimum_equity(long_value, any(x.side == "short" for x in lines))
    candidates = (
        ("reg_t", reg_t),
        ("maintenance", maintenance),
        ("minimum_equity", minimum),
    )
    binding, initial = max(candidates, key=lambda item: item[1])
    notes: List[str] = []
    if any(x.kind in ("call", "put") and x.side == "short" for x in lines) and any(
        x.kind in ("equity", "etf") for x in lines
    ):
        notes.append(
            "every short option is margined as uncovered and every stock line "
            "on its own: a covered call, a spread or a hedge earns no offset "
            "here, so a hedged book's requirement is overstated."
        )
    if any(x.kind in ("call", "put") and x.side == "long" for x in lines):
        notes.append(
            "a long option is taken as paid in full; a listed one with more "
            "than nine months to expiry may be margined at 75% of its value."
        )
    return MarginRequirement(
        base_currency=base,
        lines=lines,
        long_market_value=long_value,
        short_market_value=short_value,
        reg_t=reg_t,
        maintenance=maintenance,
        minimum_equity=minimum,
        initial=initial,
        initial_binding=binding,
        initial_rule=RULE_INITIAL,
        notes=tuple(notes),
    )


# ── the runway ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarginCall:
    """One maintenance call along the path. Days count from 0, today."""

    day: int
    equity: float
    maintenance: float
    #: The equity the call restores: the initial or the maintenance
    #: requirement on the call's day.
    target: float
    amount: float
    #: The last day the call may be met: day + cure_days.
    deadline: int
    #: "cured" (the equity reached the target by the deadline),
    #: "deleveraged" (it did not, and the deadline is inside the path) or
    #: "outstanding" (the path ends first).
    outcome: str
    #: The day it was cured, or the day deleveraging starts; None when
    #: outstanding.
    resolved_day: Optional[int]


@dataclass(frozen=True)
class MarginRunway:
    """A stress path walked day by day against the margin requirements."""

    #: "requirements" (given), "positions" (today's, held fixed) or
    #: "revalued" (recomputed each day from the revalued positions).
    source: str
    days: int
    restore_to: str
    cure_days: int
    equity_today: float
    initial_today: float
    maintenance_today: float
    first_call_day: Optional[int]
    first_call_amount: Optional[float]
    deleveraging_day: Optional[int]
    calls: Tuple[MarginCall, ...]
    lowest_equity: float
    lowest_equity_day: int
    #: The first day the equity is below zero: the account owes the broker.
    first_deficit_day: Optional[int]
    #: The smallest of equity - maintenance over the path, and its day.
    tightest_cushion: float
    tightest_day: int
    #: The smallest cash added today with no day below maintenance: 0 when
    #: the path never breaches. One unit less calls on `tightest_day`.
    buffer: float
    final_equity: float
    final_maintenance: float
    notes: Tuple[str, ...]


def _amount(value: Any, argument: str, *, positive: bool = False) -> float:
    number = _number(value, argument)
    if abs(number) > 1e18:
        raise ValidationError(f"{argument}={value!r} is beyond 1e18 in magnitude.")
    if positive and number < 0:
        raise ValidationError(
            f"{argument}={value!r} is negative; a margin requirement is an "
            "amount of equity, 0 or more."
        )
    return number


def _path_days(count: int, argument: str) -> None:
    if count < 1:
        raise ValidationError(f"{argument} is empty. Give at least one day.")
    if count > MAX_PATH_DAYS:
        raise ValidationError(
            f"{argument} has {count:,} days, above the {MAX_PATH_DAYS:,} one "
            "call walks. Shorten the path."
        )


def _equity_path(equity: float, changes: Sequence[Any]) -> np.ndarray:
    """equity, then equity plus each running total of the changes, each
    the exact sum rounded once: no day's equity depends on how the float
    additions before it happened to round."""
    _path_days(len(changes), "equity_changes")
    total = Fraction(equity)
    out = [equity]
    for i, change in enumerate(changes):
        total += Fraction(_amount(change, f"equity_changes[{i}]"))
        out.append(float(total))
    return np.asarray(out, dtype=np.float64)


def _revalued(
    positions: Sequence[Any],
    values: Sequence[PositionValue],
    house: Mapping[str, float],
    returns: Mapping[str, Sequence[Any]],
    equity: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equity, initial and maintenance on each day of a path of daily
    returns applied to the positions' prices."""
    if not isinstance(returns, Mapping) or not returns:
        raise ValidationError(
            "returns is empty. Give {instrument: [daily return, ...]} for every "
            "position's instrument."
        )
    held = {v.instrument for v in values}
    missing = sorted(held - set(returns))
    unknown = sorted(set(returns) - held)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"no returns for {missing[:5]}")
        if unknown:
            parts.append(f"returns for {unknown[:5]}, which no position holds")
        raise ValidationError(
            f"returns and positions must name the same instruments: "
            f"{' and '.join(parts)}. Day t of every list is the same day."
        )
    lengths = {k: len(v) for k, v in returns.items()}
    if len(set(lengths.values())) != 1:
        shown = dict(list(lengths.items())[:5])
        raise ValidationError(
            f"the returns lists have different lengths ({shown}). Day t is "
            "position t of every list, so each needs one return a day."
        )
    days = next(iter(lengths.values()))
    _path_days(days, "returns")
    if days * len(values) > MAX_PATH_CELLS:
        raise ValidationError(
            f"{len(values):,} positions x {days:,} days is above the "
            f"{MAX_PATH_CELLS:,} a revalued path computes. Aggregate lines or "
            "shorten the path."
        )
    terms = [_terms(p, v, i) for i, (p, v) in enumerate(zip(positions, values))]
    options = [i for i, t in enumerate(terms) if t.kind in ("call", "put")]
    if options:
        raise ValidationError(
            f"positions[{options[0]}] is an option. A daily return revalues a "
            "price, and an option's premium and its underlying move together "
            "by a model, not by one return: walk a book with options on "
            "equity_changes instead."
        )
    paths = {}
    for name, series in returns.items():
        r = np.asarray(
            [_number(x, f"returns[{name!r}][{t}]") for t, x in enumerate(series)],
            dtype=np.float64,
        )
        low = np.flatnonzero(r < -1.0)
        if low.size:
            raise ValidationError(
                f"returns[{name!r}][{int(low[0])}]={r[low[0]]!r} is below -1: "
                "a price cannot fall more than 100%."
            )
        paths[name] = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    n_days = days + 1
    value_change = np.zeros(n_days)
    reg_t = np.zeros(n_days)
    maintenance = np.zeros(n_days)
    long_value = np.zeros(n_days)
    any_short = False
    # Summed in one order whatever order the lines came in, so reordering
    # the positions moves no day's total.
    order = sorted(
        range(len(values)),
        key=lambda i: (values[i].instrument.casefold(), values[i].currency),
    )
    for i in order:
        value, term = values[i], terms[i]
        growth = paths[value.instrument]
        notional = value.base_notional * growth
        value_change += notional - value.base_notional
        mv = np.abs(notional)
        reg_t += REG_T_RATE * mv
        if value.side == "long":
            long_value += mv
            req = min(1.0, LONG_RATE * term.leverage) * mv
        else:
            any_short = True
            low_price = value.price * value.fx_rate * growth < LOW_PRICE
            shares = abs(value.quantity)
            req = np.where(
                low_price,
                np.maximum(1.0 * mv, LOW_PRICE_PER_SHARE * shares),
                np.maximum(
                    min(1.0, SHORT_RATE * term.leverage) * mv, PER_SHARE * shares
                ),
            )
        house_rate = house.get(value.instrument)
        if house_rate is not None:
            # Checked against today's tier by the caller; on a later day
            # whose tier is higher, the rule's own requirement binds.
            req = np.maximum(house_rate * mv, req)
        maintenance += req
    minimum = (
        np.full(n_days, MINIMUM_EQUITY)
        if any_short
        else np.minimum(MINIMUM_EQUITY, long_value)
    )
    initial = np.maximum(np.maximum(reg_t, maintenance), minimum)
    equity_path = equity + value_change
    return equity_path, initial, maintenance


def _walk(
    equity: np.ndarray,
    initial: np.ndarray,
    maintenance: np.ndarray,
    restore_to: str,
    cure_days: int,
) -> List[MarginCall]:
    target = initial if restore_to == "initial" else maintenance
    last = len(equity) - 1
    calls: List[MarginCall] = []
    day = 0
    while day <= last:
        below = np.flatnonzero(equity[day:] < maintenance[day:])
        if not below.size:
            break
        start = day + int(below[0])
        deadline = start + cure_days
        aim = float(target[start])
        cured = None
        window = np.flatnonzero(
            equity[start + 1 : min(deadline, last) + 1]
            >= target[start + 1 : min(deadline, last) + 1]
        )
        if window.size:
            cured = start + 1 + int(window[0])
        if cured is not None:
            outcome, resolved = "cured", cured
        elif deadline <= last:
            outcome, resolved = "deleveraged", deadline
        else:
            outcome, resolved = "outstanding", None
        calls.append(
            MarginCall(
                day=start,
                equity=float(equity[start]),
                maintenance=float(maintenance[start]),
                target=aim,
                amount=aim - float(equity[start]),
                deadline=deadline,
                outcome=outcome,
                resolved_day=resolved,
            )
        )
        if outcome != "cured":
            break
        day = cured + 1
    return calls


def _buffer(equity: np.ndarray, maintenance: np.ndarray) -> float:
    """The smallest cash b >= 0 with equity + b >= maintenance on every
    day, in the float arithmetic the walk compares in."""
    need = float(np.max(maintenance - equity))
    if need <= 0:
        return 0.0
    # The difference is rounded, and so is the sum the walk compares: step
    # to the least float whose sum clears every day, a few ulps at most.
    while np.any(equity + need < maintenance):
        need = math.nextafter(need, math.inf)
    while need > 0 and not np.any(equity + math.nextafter(need, 0.0) < maintenance):
        need = math.nextafter(need, 0.0)
    return need


def margin_runway(
    equity: float,
    *,
    equity_changes: Optional[Sequence[Any]] = None,
    initial_margin: Optional[float] = None,
    maintenance_margin: Optional[float] = None,
    positions: Optional[Sequence[Any]] = None,
    base_currency: str = "USD",
    fx_rates: Optional[Mapping[Any, Any]] = None,
    house_rates: Optional[Mapping[str, Any]] = None,
    returns: Optional[Mapping[str, Sequence[Any]]] = None,
    restore_to: str = "initial",
    cure_days: int = DEFAULT_CURE_DAYS,
) -> MarginRunway:
    """
    Walk a stress path against the margin requirements. Three shapes:

    - `initial_margin`, `maintenance_margin` and `equity_changes`: the
      requirements as given, held for the whole path;
    - `positions` and `equity_changes`: the requirements of
      `margin_requirement(positions, ...)` today, held for the path;
    - `positions` and `returns` ({instrument: daily returns}): each day's
      prices are the last day's times 1 + its return, the equity moves by
      the change in the positions' value, and the requirements are those
      rules on that day's prices (stocks and ETFs only; FX held).

    `equity` is the account's equity today; day 0 is today and day t the
    end of the t-th step. See the module docstring for calls, deleveraging
    and the buffer.
    """
    if restore_to not in RESTORE_TO:
        raise ValidationError(
            f"restore_to={restore_to!r} must be 'initial' (a call restores the "
            "equity to the initial requirement) or 'maintenance'."
        )
    if (
        isinstance(cure_days, bool)
        or not isinstance(cure_days, numbers.Integral)
        or not 0 <= cure_days <= MAX_PATH_DAYS
    ):
        raise ValidationError(
            f"cure_days={cure_days!r} must be a whole number of days, 0 or more: "
            "the days a call may stay unmet before forced deleveraging starts."
        )
    today = _amount(equity, "equity")
    given = initial_margin is not None or maintenance_margin is not None
    if positions is not None and given:
        raise ValidationError(
            "positions and initial_margin/maintenance_margin together state "
            "the requirements twice. Give the positions (their requirements "
            "are computed) or the two amounts."
        )
    if returns is not None and positions is None:
        raise ValidationError(
            "returns revalue positions, and none were given. Give positions, "
            "or walk equity_changes against initial_margin and "
            "maintenance_margin."
        )
    if returns is not None and equity_changes is not None:
        raise ValidationError(
            "equity_changes and returns are two stress paths. Give one: the "
            "equity's daily change, or the positions' daily returns."
        )
    notes: List[str] = []
    if positions is not None and returns is not None:
        positions = list(positions)
        base = _usd(base_currency)
        values = position_values(positions, base, fx_rates)
        house = _check_house(house_rates, values)
        for i, (p, v) in enumerate(zip(positions, values)):
            _line(p, v, i, house)  # every line's own refusals, today
        equity_path, initial, maintenance = _revalued(
            positions, values, house, returns, today
        )
        source = "revalued"
        if any(v.currency != base for v in values):
            notes.append("FX rates are held at today's for the whole path.")
    else:
        if equity_changes is None:
            raise ValidationError(
                "no stress path was given. Give equity_changes (the account "
                "equity's change each day) or, with positions, returns."
            )
        equity_path = _equity_path(today, equity_changes)
        if positions is not None:
            req = margin_requirement(positions, base_currency, fx_rates, house_rates)
            im, mm, source = req.initial, req.maintenance, "positions"
        else:
            if initial_margin is None or maintenance_margin is None:
                raise ValidationError(
                    "initial_margin and maintenance_margin are both needed with "
                    "equity_changes and no positions (or give the positions)."
                )
            im = _amount(initial_margin, "initial_margin", positive=True)
            mm = _amount(maintenance_margin, "maintenance_margin", positive=True)
            if mm > im:
                raise ValidationError(
                    f"maintenance_margin={mm:g} is above initial_margin={im:g}. "
                    "The initial requirement is at least the maintenance one "
                    "(FINRA 4210(b)): a book that opened at initial would be "
                    "in call at once."
                )
            source = "requirements"
        n = len(equity_path)
        initial = np.full(n, im)
        maintenance = np.full(n, mm)
        notes.append(
            "the requirements are held at today's for the whole path; give "
            "positions and returns to recompute them as prices move."
        )

    calls = _walk(equity_path, initial, maintenance, restore_to, cure_days)
    first = calls[0] if calls else None
    deleveraging = next(
        (c.resolved_day for c in calls if c.outcome == "deleveraged"), None
    )
    cushion = equity_path - maintenance
    tightest = int(np.argmin(cushion))
    lowest = int(np.argmin(equity_path))
    deficit = np.flatnonzero(equity_path < 0)
    if first is not None and first.day == 0:
        notes.append("the account is below maintenance today, on day 0.")
    if calls and calls[-1].outcome == "outstanding":
        notes.append(
            f"the call of day {calls[-1].day} is unmet when the path ends on day "
            f"{len(equity_path) - 1}, before its cure period does on day "
            f"{calls[-1].deadline}."
        )
    return MarginRunway(
        source=source,
        days=len(equity_path) - 1,
        restore_to=restore_to,
        cure_days=int(cure_days),
        equity_today=float(equity_path[0]),
        initial_today=float(initial[0]),
        maintenance_today=float(maintenance[0]),
        first_call_day=None if first is None else first.day,
        first_call_amount=None if first is None else first.amount,
        deleveraging_day=deleveraging,
        calls=tuple(calls),
        lowest_equity=float(equity_path[lowest]),
        lowest_equity_day=lowest,
        first_deficit_day=int(deficit[0]) if deficit.size else None,
        tightest_cushion=float(cushion[tightest]),
        tightest_day=tightest,
        buffer=_buffer(equity_path, maintenance),
        final_equity=float(equity_path[-1]),
        final_maintenance=float(maintenance[-1]),
        notes=tuple(notes),
    )
