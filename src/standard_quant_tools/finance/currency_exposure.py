"""
The currency exposure of a set of positions, after any FX hedges, and the
split of each position's base-currency return into its local, FX and cross
terms.

EXPOSURE IS THE BASE-CURRENCY VALUE HELD IN EACH CURRENCY. A position's
base notional is quantity x price x multiplier x the caller's rate
(`positions.position_values`), and a currency's exposure from positions is
the sum of its lines' base notionals (`positions.exposure_summary`). Each
is the correctly rounded product of its factors and each total a
`math.fsum`, so no total depends on the order the lines or the currencies
arrive in, and relabelling the currencies -- a consistent change of codes
and rates -- changes no number.

A HEDGE IS A PAIR AND A NOTIONAL. `{'EURUSD': -10_000_000}` sells ten
million EUR against USD, forward or spot: a positive notional buys the
pair's first currency, a negative one sells it, in units of that currency.
Its other leg is the equal and opposite base-currency value in the second
currency at the caller's rates, so a hedge has no value of its own here and
moves exposure from one currency to another -- a forward's points are a
return over the period (`fx_forward`), not exposure. A cross hedge
('EURGBP' in a USD book) needs both rates. A currency's net exposure is its
positions plus its hedge legs, and its hedge ratio is the share of its
positions' exposure the hedges offset, -hedge / position: 1 is fully
hedged, above 1 over-hedged, negative a hedge that adds to the exposure. It
is null in the base currency, which carries no currency risk, and where the
positions net to nothing.

THE RETURN SPLIT. A position whose price returns r_L in its own currency,
in a currency whose rate (units of base per one) returns r_FX, returns in
the base currency

    (1 + r_L)(1 + r_FX) - 1 = r_L + r_FX + r_L x r_FX,

the local, FX and cross terms. `base_return` is their `math.fsum`, so the
three add up to it exactly, and each P&L term is the base notional times its
return. A position in the base currency has an FX return of exactly zero
and no FX or cross term. A hedge leg returns only its currency's r_FX: the
interest a forward accrues is its premium, priced by `fx_forward`, and is
not part of this split.
"""

from __future__ import annotations

import math
import numbers
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from standard_quant_tools.error import ValidationError

from .currency import currency_code
from .fx_forward import currency_pair
from .positions import (
    MAX_ABS_QUANTITY,
    MAX_POSITIONS,
    check_fx_rates,
    exact_product,
    exposure_summary,
    position_values,
)

__all__ = ["MAX_ABS_RETURN", "currency_exposure", "hedge_legs"]

#: A return over the period, as a decimal, within 1,000%: the bound the
#: library puts on every rate. An FX return is also above -1 (a rate cannot
#: fall to zero); a local return is not, because a price can go negative.
MAX_ABS_RETURN = 10.0

#: How many offending items a refusal names before counting the rest.
_SHOWN = 5


def _listed(items: Sequence[str]) -> str:
    shown = ", ".join(items[:_SHOWN])
    more = len(items) - _SHOWN
    return shown + (f" and {more} more" if more > 0 else "")


def _finite(value: Any, argument: str, fix: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValidationError(f"{argument}={value!r} is not a number. {fix}")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{argument}={value!r} is not finite. {fix}")
    return number


def _attribute(position: Any, name: str) -> Any:
    if isinstance(position, Mapping):
        return position.get(name)
    return getattr(position, name, None)


# A hedge leg: (hedge key, currency, local amount, base amount).
_Leg = Tuple[str, str, float, float]


def hedge_legs(
    hedges: Optional[Mapping[Any, Any]],
    base_currency: str,
    rates: Mapping[str, float],
    argument: str = "hedges",
) -> List[_Leg]:
    """
    Each hedge's two legs, in input order: the first currency's notional and
    its base value, then the second currency's equal and opposite base value
    and that value in the second currency. `rates` is a checked rate table
    (`positions.check_fx_rates`); every non-base currency a hedge names
    needs a rate.
    """
    if not hedges:
        return []
    if not isinstance(hedges, Mapping):
        raise ValidationError(
            f"{argument}={hedges!r} is not a mapping. Give {{pair: notional}}, "
            "e.g. {'EURUSD': -10000000} to sell ten million EUR against USD."
        )
    if len(hedges) > MAX_POSITIONS:
        raise ValidationError(
            f"{argument} has {len(hedges):,} pairs, above the {MAX_POSITIONS:,} "
            "one call takes. Net the hedges in each pair first."
        )
    seen: Dict[Tuple[str, str], Any] = {}
    missing: Dict[str, List[str]] = {}
    parsed = []
    for key, value in hedges.items():
        first, second = currency_pair(key, argument=f"{argument} key")
        if (first, second) in seen:
            raise ValidationError(
                f"{argument} names {first}{second} twice ({seen[(first, second)]!r} "
                f"and {key!r}). Give one notional per pair: add them."
            )
        seen[(first, second)] = key
        notional = _finite(
            value,
            f"{argument}[{key!r}]",
            f"Give the units of {first} bought (positive) or sold (negative).",
        )
        if notional == 0:
            raise ValidationError(
                f"{argument}[{key!r}] is zero, and a zero hedge is not a hedge. "
                "Drop the pair."
            )
        if abs(notional) > MAX_ABS_QUANTITY:
            raise ValidationError(
                f"{argument}[{key!r}]={value!r} is beyond {MAX_ABS_QUANTITY:g} in "
                f"magnitude. Give the notional in units of {first}."
            )
        for code in (first, second):
            if code != base_currency and code not in rates:
                missing.setdefault(code, []).append(f"{argument}[{key!r}]")
        parsed.append((str(key), first, second, notional))
    if missing:
        held = "; ".join(
            f"{code}, in {_listed(where)}" for code, where in sorted(missing.items())
        )
        first_missing = min(missing)
        raise ValidationError(
            f"fx_rates has no rate for {held}. A hedge's legs are valued at "
            f"fx_rates, units of base_currency ({base_currency}) per ONE unit "
            f"of each other currency: add {{'{first_missing}': <{base_currency} "
            f"per 1 {first_missing}>}}."
        )
    legs: List[_Leg] = []
    for key, first, second, notional in parsed:
        x_first = 1.0 if first == base_currency else rates[first]
        x_second = 1.0 if second == base_currency else rates[second]
        try:
            base_first = exact_product(notional, x_first)
        except OverflowError:
            raise ValidationError(
                f"{argument}[{key!r}]: the notional times its rate overflows a "
                "float. One of them is a unit error."
            ) from None
        # The second leg is the same base value, opposite, so the hedge nets
        # to nothing in the base currency bit for bit; its amount in the
        # second currency is that value over the second rate, rounded once.
        local_second = float(
            -Fraction(notional) * Fraction(x_first) / Fraction(x_second)
        )
        legs.append((key, first, notional, base_first))
        legs.append((key, second, local_second, -base_first))
    return legs


def _fx_returns(
    fx_returns: Optional[Mapping[Any, Any]],
    base: str,
    needed: Sequence[str],
) -> Dict[str, float]:
    """The FX return table checked: codes read, values finite, above -1 and
    within MAX_ABS_RETURN, the base's own exactly 0, and every currency in
    `needed` present."""
    table: Dict[str, float] = {}
    spelled: Dict[str, Any] = {}
    for key, value in (fx_returns or {}).items():
        code = currency_code(key, argument="fx_returns key")
        if code in table:
            raise ValidationError(
                f"fx_returns names {code} twice ({spelled[code]!r} and {key!r}). "
                "Give one return per currency."
            )
        number = _finite(
            value,
            f"fx_returns[{key!r}]",
            f"Give the change in {code}'s rate over the period: 0.02 is +2%.",
        )
        if not -1.0 < number <= MAX_ABS_RETURN:
            raise ValidationError(
                f"fx_returns[{key!r}]={value!r} is outside (-1, "
                f"{MAX_ABS_RETURN:g}]. It is the change in units of the base "
                f"currency per one {code} over the period, a decimal: 0.02 is "
                "+2%, and a rate cannot fall to zero."
            )
        if code == base and number != 0.0:
            raise ValidationError(
                f"fx_returns[{key!r}]={value!r} is the base currency's own "
                f"return, which is exactly 0: one {base} is always one {base}. "
                "Drop it or give 0."
            )
        table[code] = number
        spelled[code] = key
    absent = sorted(c for c in set(needed) if c != base and c not in table)
    if absent:
        raise ValidationError(
            f"fx_returns has no return for {', '.join(absent)}, which the "
            "positions or hedges hold. Give each held currency's change in "
            f"its rate (units of {base} per one) over the period, e.g. "
            f"{{'{absent[0]}': 0.02}} for +2%."
        )
    return table


def currency_exposure(
    positions: Sequence[Any],
    base_currency: str,
    fx_rates: Optional[Mapping[Any, Any]] = None,
    *,
    hedges: Optional[Mapping[Any, Any]] = None,
    fx_returns: Optional[Mapping[Any, Any]] = None,
) -> Dict[str, Any]:
    """
    Exposure per currency in the base currency from `positions` and
    `hedges`, each currency's hedge ratio, and -- when each position carries
    a `local_return` and `fx_returns` covers the currencies held -- the
    local, FX and cross split of every position's base-currency return.

    `positions` and `fx_rates` are those of `positions.position_values`;
    a position may carry `local_return`, its price return in its own
    currency over the period. `hedges` maps a pair ('EURUSD' or 'EUR/USD')
    to a notional in units of its first currency, positive bought.
    `fx_returns` maps a currency to the change in its rate over the period.
    See the module docstring for the definitions.
    """
    positions = list(positions)
    summary = exposure_summary(positions, base_currency, fx_rates)
    values = position_values(positions, base_currency, fx_rates)
    base = summary.base_currency
    rates = check_fx_rates(fx_rates, base_currency=base)
    legs = hedge_legs(hedges, base, rates)

    held = sorted({v.currency for v in values} | {leg[1] for leg in legs})
    by_summary = {row.currency: row for row in summary.currencies}
    rows: List[Dict[str, Any]] = []
    for code in held:
        lines = [v for v in values if v.currency == code]
        code_legs = [leg for leg in legs if leg[1] == code]
        position_net = by_summary[code].net if code in by_summary else 0.0
        hedge_net = math.fsum(leg[3] for leg in code_legs)
        net = math.fsum(
            [v.base_notional for v in lines] + [leg[3] for leg in code_legs]
        )
        if code == base or position_net == 0:
            ratio = math.nan
        else:
            ratio = -hedge_net / position_net + 0.0
        rows.append(
            {
                "currency": code,
                "fx_rate": 1.0 if code == base else rates[code],
                "positions": len(lines),
                "hedge_legs": len(code_legs),
                "local_net": math.fsum(
                    [v.local_notional for v in lines] + [leg[2] for leg in code_legs]
                ),
                "position_net": position_net,
                "hedge_net": hedge_net,
                "net": net,
                "gross": math.fsum(
                    [abs(v.base_notional) for v in lines]
                    + [abs(leg[3]) for leg in code_legs]
                ),
                "hedge_ratio": ratio,
            }
        )
    foreign = [row for row in rows if row["currency"] != base]
    result: Dict[str, Any] = {
        "base_currency": base,
        "positions": summary.positions,
        "hedges": len(legs) // 2,
        "gross": summary.gross,
        "net": summary.net,
        "long": summary.long,
        "short": summary.short,
        "foreign_net": math.fsum(row["net"] for row in foreign),
        "foreign_gross": math.fsum(abs(row["net"]) for row in foreign),
        "unhedged_foreign_gross": math.fsum(
            abs(row["position_net"]) for row in foreign
        ),
        "currencies": rows,
        "returns": _return_split(positions, values, legs, base, fx_returns),
        "unused_fx_rates": sorted(c for c in rates if c != base and c not in held),
        "warnings": [],
    }
    if legs and all(abs(row["position_net"]) == 0 for row in foreign):
        result["warnings"].append(
            "The hedges are the only foreign exposure: no position is held in "
            "a currency they hedge, so they are an outright currency position."
        )
    return result


def _return_split(
    positions: Sequence[Any],
    values: Sequence[Any],
    legs: Sequence[_Leg],
    base: str,
    fx_returns: Optional[Mapping[Any, Any]],
) -> Optional[Dict[str, Any]]:
    """The local / FX / cross split, or None when no return was given."""
    local = [_attribute(p, "local_return") for p in positions]
    given = [i for i, r in enumerate(local) if r is not None]
    if not given:
        if fx_returns is not None:
            raise ValidationError(
                "fx_returns was given and no position has a local_return. The "
                "split needs both: give each position's local_return (its "
                "price return in its own currency over the period), or drop "
                "fx_returns."
            )
        return None
    if len(given) < len(local):
        absent = [
            f"positions[{i}] {values[i].instrument!r}"
            for i, r in enumerate(local)
            if r is None
        ]
        raise ValidationError(
            f"local_return is missing on {_listed(absent)}. The split is of "
            "every position or none: give each its price return in its own "
            "currency over the period (0 if it did not move)."
        )
    needed = [v.currency for v in values] + [leg[1] for leg in legs]
    table = _fx_returns(fx_returns, base, needed)

    lines = []
    for index, (value, r_local) in enumerate(zip(values, local)):
        r_local = _finite(
            r_local,
            f"positions[{index}].local_return",
            "Give its return over the period as a decimal: 0.04 is +4%.",
        )
        if abs(r_local) > MAX_ABS_RETURN:
            raise ValidationError(
                f"positions[{index}].local_return={r_local!r} is beyond "
                f"{MAX_ABS_RETURN:g} in magnitude. It is a decimal: 0.04 is +4%."
            )
        notional = value.base_notional
        if value.currency == base:
            # No FX term and no cross term, exactly.
            r_fx = cross = pnl_fx = pnl_cross = 0.0
        else:
            r_fx = table[value.currency]
            cross = r_local * r_fx + 0.0
            pnl_fx = notional * r_fx + 0.0
            pnl_cross = notional * cross + 0.0
        pnl_local = notional * r_local + 0.0
        lines.append(
            {
                "instrument": value.instrument,
                "currency": value.currency,
                "base_notional": notional,
                "local_return": r_local,
                "fx_return": r_fx,
                "cross_return": cross,
                "base_return": math.fsum([r_local, r_fx, cross]),
                "local_pnl": pnl_local,
                "fx_pnl": pnl_fx,
                "cross_pnl": pnl_cross,
                "pnl": math.fsum([pnl_local, pnl_fx, pnl_cross]),
            }
        )
    leg_pnl = [
        (leg[1], 0.0 if leg[1] == base else leg[3] * table[leg[1]] + 0.0)
        for leg in legs
    ]
    codes = sorted({line["currency"] for line in lines} | {c for c, _ in leg_pnl})
    terms = [
        line[k] for line in lines for k in ("local_pnl", "fx_pnl", "cross_pnl")
    ] + [pnl for _, pnl in leg_pnl]
    return {
        "lines": lines,
        "local_pnl": math.fsum(line["local_pnl"] for line in lines),
        "fx_pnl": math.fsum(line["fx_pnl"] for line in lines),
        "cross_pnl": math.fsum(line["cross_pnl"] for line in lines),
        "hedge_pnl": math.fsum(pnl for _, pnl in leg_pnl),
        "pnl": math.fsum(terms),
        "fx_pnl_by_currency": {
            code: math.fsum(
                [line["fx_pnl"] for line in lines if line["currency"] == code]
                + [pnl for c, pnl in leg_pnl if c == code]
            )
            for code in codes
        },
        "cross_pnl_by_currency": {
            code: math.fsum(
                line["cross_pnl"] for line in lines if line["currency"] == code
            )
            for code in codes
        },
        "fx_returns": {code: table.get(code, 0.0) for code in codes},
    }
