"""
Dollar greeks across a book of options and underlying positions on several
underlyings, and how far delta-gamma is from revaluing it.

Each option is priced and differentiated by `options_batch.
black_scholes_greeks_batch` -- the library's Black-Scholes-Merton, every
line in one call -- and the greeks of one option on one unit of the
underlying are turned into a POSITION's by its contract multiplier and
quantity, n = multiplier x quantity:

    dollar delta   delta x S x n                 $ per 100% move, i.e. the
                                                 delta in underlying dollars
    dollar gamma   gamma x S^2 / 100 x n         the change in dollar delta
                                                 for a 1% move
    dollar vega    vega x n                      $ per volatility point
    dollar theta   theta x n                     $ per CALENDAR day (the
                                                 annual theta / 365)
    dollar rho     rho x n                       $ per 1% (100 bp) of rate

An underlying position (shares, units) has delta 1 and nothing else:
dollar delta units x S.

THE MULTIPLIER IS REQUIRED. 100 is the convention for US equity options
and nothing more: index options, mini contracts, futures options and
contracts adjusted after a corporate action all differ, and a book summed
on an assumed 100 is wrong by the ratio on every line it got wrong. A line
without one is refused.

BETA-WEIGHTED DELTA is sum_i beta_i x dollar delta_i, in index-equivalent
dollars: the dollar exposure to an index that moves the book as its
underlyings' betas say they move with it. The betas are the caller's.

DELTA-GAMMA AGAINST REVALUATION. On a grid of spot shocks (default +/-1,
2, 5 and 10%), every underlying moves by the same percentage, instantly,
with volatility, rates and time held, and each line is revalued in full
with the same pricer. The approximation is delta dS + gamma dS^2 / 2 per
line, dS the spot move; its error is the Taylor remainder, which starts
at speed x dS^3 / 6 and so shrinks like the cube of the shock. At a 1%
move it is noise; at 10% on short-dated options it is not, and the full
revaluation is the number to read.

AGGREGATING ACROSS UNDERLYINGS adds dollars to dollars, and two of the
sums carry an assumption the totals cannot state on their own: the book's
dollar gamma is the change in its dollar delta when EVERY underlying moves
1% together, and its vega the P&L when every volatility rises one point
together. A smile, or a dispersion book, moves neither way.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from standard_quant_tools.analysis.options_batch import black_scholes_greeks_batch
from standard_quant_tools.error import ValidationError

__all__ = ["DEFAULT_SPOT_SHOCKS", "aggregate_option_book"]

#: The spot shocks a book is revalued at when none are given.
DEFAULT_SPOT_SHOCKS = (-0.10, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.10)

#: The largest rate or yield magnitude a model input may have, as a
#: decimal: 10 is 1,000%, past which the number is a unit error.
_MAX_RATE = 10.0

#: The largest spot a shocked underlying may reach (the pricer's own bound).
_MAX_PRICE = 1e12

_WHO = "aggregate_option_book"

#: What every dollar greek is, for the result.
UNITS = {
    "dollar_delta": "delta x spot x multiplier x quantity: the position's "
    "exposure in underlying dollars",
    "dollar_gamma": "gamma x spot^2 / 100 x multiplier x quantity: the change "
    "in dollar delta for a 1% move in the underlying (the P&L of that move "
    "from gamma is half of it times 0.01)",
    "dollar_vega": "$ per 1 volatility point (0.01), holding everything else",
    "dollar_theta": "$ per calendar day: the annual theta / 365",
    "dollar_rho": "$ per 1% (0.01) move in the rate",
    "beta_weighted_delta": "sum of beta x dollar delta: index-equivalent dollars",
    "market_value": "price x multiplier x quantity (units x spot for an "
    "underlying position)",
}


def _number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{_WHO}: {field} must be a number, got {value!r:.80}."
        ) from None
    if not math.isfinite(number):
        raise ValidationError(f"{_WHO}: {field} must be finite, got {value!r}.")
    return number


def _map(values: Any, field: str) -> Dict[str, float]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValidationError(
            f"{_WHO}: {field} is a {type(values).__name__}; give it as a map of "
            "underlying -> number."
        )
    return {str(k): _number(v, f"{field}[{str(k)!r}]") for k, v in values.items()}


def _years(line: Mapping[str, Any], i: int, valuation_date: Any) -> float:
    """A line's time to expiry in years, from `time_to_expiry` or from its
    `expiry` date and the valuation date (ACT/365F)."""
    from standard_quant_tools.delta_one.daycount import to_date, year_fraction

    t = line.get("time_to_expiry")
    expiry = line.get("expiry")
    if (t is None) == (expiry is None):
        raise ValidationError(
            f"{_WHO}: options[{i}] needs exactly one of time_to_expiry (years) "
            f"and expiry (an ISO date with valuation_date); got "
            f"{'both' if t is not None else 'neither'}."
        )
    if t is not None:
        years = _number(t, f"options[{i}].time_to_expiry")
    else:
        if valuation_date is None:
            raise ValidationError(
                f"{_WHO}: options[{i}] gives an expiry date ({expiry!r}) and no "
                "valuation_date was given, so its time to expiry is unknown. "
                "Pass valuation_date (ISO), or time_to_expiry in years."
            )
        years = year_fraction(
            to_date(valuation_date, "valuation_date"),
            to_date(expiry, f"options[{i}].expiry"),
            convention="ACT/365F",
        )
    if not years > 0:
        raise ValidationError(
            f"{_WHO}: options[{i}] has {years:.6g} years to expiry. An option "
            "at or past expiry has no time value to differentiate; drop it, or "
            "book its exercise as an underlying position."
        )
    return float(years)


def _option_lines(
    options: Sequence[Mapping[str, Any]], spots: Dict[str, float], valuation_date: Any
) -> List[Dict[str, Any]]:
    lines = []
    for i, line in enumerate(options):
        if not isinstance(line, Mapping):
            raise ValidationError(
                f"{_WHO}: options[{i}] is a {type(line).__name__}; each option "
                "is a map of underlying, option_type, strike, volatility, "
                "quantity, multiplier and time_to_expiry or expiry."
            )
        underlying = line.get("underlying")
        if not isinstance(underlying, str) or not underlying:
            raise ValidationError(
                f"{_WHO}: options[{i}].underlying must name the underlying, got "
                f"{underlying!r}."
            )
        if underlying not in spots:
            raise ValidationError(
                f"{_WHO}: options[{i}] is on {underlying!r}, which spots does not "
                f"price (it names {sorted(spots)[:8]}). Give every underlying a "
                "spot."
            )
        kind = line.get("option_type")
        if kind not in ("call", "put"):
            raise ValidationError(
                f"{_WHO}: options[{i}].option_type must be 'call' or 'put', got "
                f"{kind!r}. An underlying position goes in underlying_units."
            )
        if line.get("multiplier") is None:
            raise ValidationError(
                f"{_WHO}: options[{i}] has no multiplier. It is required: 100 "
                "shares a contract is the convention of US equity options, not "
                "a fact -- index, mini, futures and adjusted contracts differ, "
                "and every dollar greek scales with it. Give the contract's "
                "multiplier (100 for a standard US equity option)."
            )
        multiplier = _number(line["multiplier"], f"options[{i}].multiplier")
        quantity = _number(line.get("quantity"), f"options[{i}].quantity")
        strike = _number(line.get("strike"), f"options[{i}].strike")
        volatility = _number(line.get("volatility"), f"options[{i}].volatility")
        if not multiplier > 0:
            raise ValidationError(
                f"{_WHO}: options[{i}].multiplier must be positive, got "
                f"{multiplier!r}; a short position is a negative quantity."
            )
        if quantity == 0:
            raise ValidationError(
                f"{_WHO}: options[{i}].quantity is 0, which is not a position. "
                "Drop the line, or give the signed number of contracts."
            )
        for name, value in (("strike", strike), ("volatility", volatility)):
            if not value > 0:
                raise ValidationError(
                    f"{_WHO}: options[{i}].{name} must be positive, got {value!r}."
                )
        lines.append(
            {
                "underlying": underlying,
                "option_type": kind,
                "strike": strike,
                "time_to_expiry": _years(line, i, valuation_date),
                "volatility": volatility,
                "quantity": quantity,
                "multiplier": multiplier,
            }
        )
    return lines


def _shocks(spot_shocks: Optional[Sequence[float]]) -> List[float]:
    if spot_shocks is None:
        return list(DEFAULT_SPOT_SHOCKS)
    shocks = [_number(s, f"spot_shocks[{i}]") for i, s in enumerate(spot_shocks)]
    if not shocks:
        raise ValidationError(
            f"{_WHO}: spot_shocks is empty; omit it for the default grid "
            f"{list(DEFAULT_SPOT_SHOCKS)}."
        )
    low = [s for s in shocks if not s > -1.0]
    if low:
        raise ValidationError(
            f"{_WHO}: spot_shocks {low[:5]} take spot to zero or below; a shock "
            "is a fractional move greater than -1 (-0.10 is a 10% fall)."
        )
    if len(set(shocks)) != len(shocks):
        repeated = sorted({s for s in shocks if shocks.count(s) > 1})
        raise ValidationError(
            f"{_WHO}: spot_shocks repeats {repeated[:5]}; each shock is one row."
        )
    return shocks


def _batch(fn_args: Dict[str, np.ndarray], greeks: Sequence[str]) -> Dict[str, Any]:
    """The batch kernel, with a refusal naming the option line it broke on."""
    try:
        return black_scholes_greeks_batch(**fn_args, greeks=tuple(greeks))
    except ValidationError as exc:
        text = re.sub(
            r"^black_scholes_greeks_batch: contract \(?(\d+)",
            lambda m: f"{_WHO}: options[{m.group(1)}]",
            str(exc),
        )
        raise ValidationError(text) from None


def _fsum(values: Any) -> float:
    return math.fsum(float(v) for v in values)


def aggregate_option_book(
    options: Sequence[Mapping[str, Any]],
    spots: Mapping[str, float],
    rates: Mapping[str, float],
    *,
    dividend_yields: Optional[Mapping[str, float]] = None,
    underlying_units: Optional[Mapping[str, float]] = None,
    betas: Optional[Mapping[str, float]] = None,
    valuation_date: Any = None,
    spot_shocks: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Dollar greeks per line, per underlying and for the whole book, the
    beta-weighted delta, and delta-gamma P&L against full Black-Scholes
    revaluation on a grid of spot shocks.

    Args:
        options: one map per option line -- `underlying`, `option_type`
            ('call' / 'put'), `strike`, `volatility` (decimal), `quantity`
            (signed contracts), `multiplier` (REQUIRED) and either
            `time_to_expiry` (years) or `expiry` (a date, with
            `valuation_date`; converted ACT/365F).
        spots: underlying -> spot. Every underlying held needs one.
        rates: underlying -> continuously compounded rate (decimal), for
            every underlying an option is on.
        dividend_yields: underlying -> continuous yield (decimal); an
            underlying not named is priced as a non-payer, and the
            warnings say which.
        underlying_units: underlying -> signed units held (shares).
        betas: underlying -> beta to an index, for every underlying held,
            for the beta-weighted delta.
        valuation_date: the date `expiry` dates are measured from.
        spot_shocks: fractional spot moves; default +/-1, 2, 5, 10%.

    Returns a dict: `lines` (per option: its per-unit greeks in the batch
    kernel's units and the position's dollar greeks), `n_options`,
    `n_underlyings`, `by_underlying` (the sums, beta, beta-weighted delta
    and scenarios per underlying), `totals`, `scenarios` (the book's
    delta-gamma P&L, full P&L, error and relative error per shock),
    `largest_error` and the shock it is at, `spot_shocks`, `units`,
    `theta_basis` and `warnings`.
    """
    spot_map = _map(spots, "spots")
    bad = [u for u, s in spot_map.items() if not 0 < s <= _MAX_PRICE]
    if bad:
        raise ValidationError(
            f"{_WHO}: spots must be positive (and at most {_MAX_PRICE:g}); "
            f"{bad[:5]} are not."
        )
    if options is None:
        options = []
    if isinstance(options, Mapping) or not isinstance(options, Sequence):
        raise ValidationError(
            f"{_WHO}: options is a {type(options).__name__}; give a list of "
            "option lines."
        )
    lines = _option_lines(options, spot_map, valuation_date)
    units = _map(underlying_units, "underlying_units")
    for u, q in units.items():
        if u not in spot_map:
            raise ValidationError(
                f"{_WHO}: underlying_units holds {u!r}, which spots does not "
                "price. Give it a spot."
            )
        if q == 0:
            raise ValidationError(
                f"{_WHO}: underlying_units[{u!r}] is 0, which is not a position; "
                "drop it."
            )
    if not lines and not units:
        raise ValidationError(
            f"{_WHO}: there is no position: options and underlying_units are "
            "both empty."
        )
    held: List[str] = []
    for u in [line["underlying"] for line in lines] + list(units):
        if u not in held:
            held.append(u)
    optioned = {line["underlying"] for line in lines}
    rate_map = _map(rates, "rates")
    yield_map = _map(dividend_yields, "dividend_yields")
    beta_map = _map(betas, "betas") if betas is not None else None
    missing_rates = [u for u in held if u in optioned and u not in rate_map]
    if missing_rates:
        raise ValidationError(
            f"{_WHO}: rates has no rate for {missing_rates[:5]}, which options "
            "are on. Every rate is the caller's: give each underlying's "
            "continuously compounded rate as a decimal (0.05 is 5%)."
        )
    for field, values in (("rates", rate_map), ("dividend_yields", yield_map)):
        big = [u for u, v in values.items() if abs(v) > _MAX_RATE]
        if big:
            raise ValidationError(
                f"{_WHO}: {field} for {big[:5]} exceed {_MAX_RATE:g} (1,000%) in "
                "magnitude. A rate is a decimal: 0.05, not 5."
            )
    if beta_map is not None:
        absent = [u for u in held if u not in beta_map]
        if absent:
            raise ValidationError(
                f"{_WHO}: betas has none for {absent[:5]}. A beta-weighted delta "
                "of the book needs every underlying's beta (0.0 for one that "
                "does not move with the index); give them all or omit betas."
            )
    shocks = _shocks(spot_shocks)
    shock_array = np.array(shocks, dtype=float)
    top = max(spot_map[u] for u in held) * (1.0 + float(shock_array.max()))
    if top > _MAX_PRICE:
        raise ValidationError(
            f"{_WHO}: the largest spot shock takes a spot to {top:.6g}, past the "
            f"pricer's {_MAX_PRICE:g}. Use smaller shocks."
        )

    n_lines = len(lines)
    S = np.array([spot_map[x["underlying"]] for x in lines], dtype=float)
    K = np.array([x["strike"] for x in lines], dtype=float)
    T = np.array([x["time_to_expiry"] for x in lines], dtype=float)
    V = np.array([x["volatility"] for x in lines], dtype=float)
    R = np.array([rate_map.get(x["underlying"], 0.0) for x in lines], dtype=float)
    Q = np.array([yield_map.get(x["underlying"], 0.0) for x in lines], dtype=float)
    C = np.array([x["option_type"] == "call" for x in lines], dtype=bool)
    size = np.array([x["quantity"] * x["multiplier"] for x in lines], dtype=float)

    line_rows: List[Dict[str, Any]] = []
    full_by_line = approx_by_line = np.zeros((0, len(shocks)))
    if n_lines:
        g = _batch(
            dict(
                spot=S,
                strike=K,
                time_to_expiry=T,
                volatility=V,
                risk_free_rate=R,
                dividend_yield=Q,
                is_call=C,
            ),
            ("price", "delta", "gamma", "vega", "theta", "rho"),
        )
        shocked = S[:, None] * (1.0 + shock_array[None, :])
        revalued = _batch(
            dict(
                spot=shocked,
                strike=K[:, None],
                time_to_expiry=T[:, None],
                volatility=V[:, None],
                risk_free_rate=R[:, None],
                dividend_yield=Q[:, None],
                is_call=C[:, None],
            ),
            ("price",),
        )["price"]
        dS = shocked - S[:, None]
        full_by_line = (revalued - g["price"][:, None]) * size[:, None]
        approx_by_line = (
            g["delta"][:, None] * dS + 0.5 * g["gamma"][:, None] * dS * dS
        ) * size[:, None]
        for i, x in enumerate(lines):
            n = size[i]
            line_rows.append(
                {
                    **x,
                    "spot": float(S[i]),
                    "price": float(g["price"][i]),
                    "delta": float(g["delta"][i]),
                    "gamma": float(g["gamma"][i]),
                    "vega": float(g["vega"][i]),
                    "theta": float(g["theta"][i]),
                    "rho": float(g["rho"][i]),
                    "delta_units": float(g["delta"][i] * n),
                    "dollar_delta": float(g["delta"][i] * S[i] * n),
                    "dollar_gamma": float(g["gamma"][i] * S[i] * S[i] / 100.0 * n),
                    "dollar_vega": float(g["vega"][i] * n),
                    "dollar_theta": float(g["theta"][i] * n),
                    "dollar_rho": float(g["rho"][i] * n),
                    "market_value": float(g["price"][i] * n),
                }
            )

    def scenario_rows(full: np.ndarray, approx: np.ndarray) -> List[Dict[str, Any]]:
        rows = []
        for j, shock in enumerate(shocks):
            f, a = float(full[j]), float(approx[j])
            rows.append(
                {
                    "shock": shock,
                    "delta_gamma_pnl": a,
                    "full_pnl": f,
                    "error": a - f,
                    "relative_error": (a - f) / abs(f) if f != 0 else float("nan"),
                }
            )
        return rows

    by_underlying: List[Dict[str, Any]] = []
    for u in held:
        rows = [r for r in line_rows if r["underlying"] == u]
        idx = [i for i, x in enumerate(lines) if x["underlying"] == u]
        spot = spot_map[u]
        held_units = units.get(u, 0.0)
        stock_pnl = held_units * (spot * (1.0 + shock_array) - spot)
        full = np.array(
            [_fsum([*full_by_line[idx, j], stock_pnl[j]]) for j in range(len(shocks))]
        )
        approx = np.array(
            [_fsum([*approx_by_line[idx, j], stock_pnl[j]]) for j in range(len(shocks))]
        )
        dollar_delta = _fsum([r["dollar_delta"] for r in rows] + [held_units * spot])
        beta = beta_map[u] if beta_map is not None else float("nan")
        by_underlying.append(
            {
                "underlying": u,
                "spot": spot,
                "rate": rate_map.get(u, float("nan")) if u in optioned else None,
                "dividend_yield": yield_map.get(u, 0.0),
                "beta": beta,
                "n_options": len(rows),
                "underlying_units": held_units,
                "delta_units": _fsum([r["delta_units"] for r in rows] + [held_units]),
                "dollar_delta": dollar_delta,
                "dollar_gamma": _fsum(r["dollar_gamma"] for r in rows),
                "dollar_vega": _fsum(r["dollar_vega"] for r in rows),
                "dollar_theta": _fsum(r["dollar_theta"] for r in rows),
                "dollar_rho": _fsum(r["dollar_rho"] for r in rows),
                "market_value": _fsum(
                    [r["market_value"] for r in rows] + [held_units * spot]
                ),
                "beta_weighted_delta": beta * dollar_delta,
                "scenarios": scenario_rows(full, approx),
            }
        )
    # The book's scenario P&L as the exact sum of its underlyings'.
    book_full = np.array(
        [
            _fsum(r["scenarios"][j]["full_pnl"] for r in by_underlying)
            for j in range(len(shocks))
        ]
    )
    book_approx = np.array(
        [
            _fsum(r["scenarios"][j]["delta_gamma_pnl"] for r in by_underlying)
            for j in range(len(shocks))
        ]
    )
    totals = {
        name: _fsum(r[name] for r in by_underlying)
        for name in (
            "dollar_delta",
            "dollar_gamma",
            "dollar_vega",
            "dollar_theta",
            "dollar_rho",
            "market_value",
        )
    }
    totals["beta_weighted_delta"] = (
        _fsum(r["beta_weighted_delta"] for r in by_underlying)
        if beta_map is not None
        else float("nan")
    )
    scenarios = scenario_rows(book_full, book_approx)
    errors = np.abs(book_approx - book_full)
    worst = int(np.argmax(errors))

    warnings: List[str] = [
        "Every greek is Black-Scholes-Merton at each line's own volatility. "
        "The book's dollar gamma assumes every underlying moves the same "
        "percentage together, and its vega that every volatility moves one "
        "point together; a smile or a dispersion position moves neither way.",
        "The scenarios move every underlying by the same percentage at once, "
        "with volatility, rates and time held. The delta-gamma error grows "
        "with the cube of the shock; for a large move read full_pnl.",
    ]
    no_yield = [u for u in held if u in optioned and u not in yield_map]
    if no_yield:
        warnings.append(
            f"No dividend yield was given for {no_yield[:5]}, so their options "
            "are priced as non-payers; on a dividend payer that overstates "
            "calls and understates puts."
        )
    unused = [u for u in spot_map if u not in held]
    if unused:
        warnings.append(
            f"spots names {unused[:5]}, which no position holds; they are not "
            "in the result."
        )
    return {
        "n_options": n_lines,
        "n_underlyings": len(held),
        "lines": line_rows,
        "by_underlying": by_underlying,
        "totals": totals,
        "scenarios": scenarios,
        "largest_error": float(book_approx[worst] - book_full[worst]),
        "largest_error_shock": shocks[worst],
        "spot_shocks": shocks,
        "units": dict(UNITS),
        "theta_basis": "per calendar day: the annual theta / 365",
        "warnings": warnings,
    }
