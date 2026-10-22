"""
Index dividends as POINTS, which is what a futures price is made of.

WHY A CONTINUOUS YIELD IS NOT ENOUGH. `F = S exp((r - q - b) T)` treats
dividends as a smooth yield, and for a broad index over a year that is a
fair approximation. Over a quarter it is not. Index dividends arrive in
dense seasonal clusters -- a European index pays most of its year in April
and May -- so a June contract and a September contract that look adjacent
on the curve straddle the entire dividend season, and pricing both off the
same `q` puts one of them badly wrong. Points are the honest unit: a
contract is worth the index less the dividends that go ex before it
expires, and that is a sum of dated cash amounts, not a rate.

THE CONVERSION IS THE DIVISOR. A constituent's cash dividend becomes index
points as `shares * dividend / divisor`, the same divisor the index level
is defined by. That is the only place the index construction enters, and
getting it wrong scales every number here by a constant -- which is
detectable only by comparing the total against what the futures imply,
which is why this tool does that comparison rather than stopping at a sum.

THE COMPARISON IS OF PRESENT VALUES. What a quoted future implies,
`S - F exp(-rT)`, is the PRESENT VALUE of the dividends in the window: the
cash-and-carry that pins `F` holds the stock, receives each dividend when
it is paid, and invests it to expiry, so `F = S exp(rT) - sum d_i
exp(r (T - t_i))` and `S - F exp(-rT) = sum d_i exp(-r t_i)`. The forecast
total is a NOMINAL sum. Comparing the two read pure discounting as a view:
a future priced exactly off a 60-point dividend 91 days out at 5% implied
59.26 points, and the 0.74-point gap was reported as "a position". Each
forecast dividend is now discounted at the financing rate before the
comparison, and both totals are reported.

TO THE PAY DATE WHEN GIVEN, ELSE THE EX-DATE. The ex-date decides
entitlement -- which window a dividend belongs to -- and the pay date is
when the cash arrives, so the pay date is the right discount date. A
constituent with no `pay_date` is discounted to its ex-date and the result
says how many were: the pay date trails the ex-date by days to weeks, so
that overstates each present value by about rate x lag (0.3% of the
dividend at 5% and a three-week lag).

THE IMPLIED NUMBER IS A MARKET VIEW, NOT A CHECK ON YOURS. When a forecast
and the futures-implied total disagree, the usual cause is neither side
being wrong: the market is pricing a probability distribution over cuts and
special dividends, and a forecast is a point estimate of the most likely
path. A gap is a position, not an error -- which is the whole basis of
dividend trading, and is why the difference is reported rather than
reconciled away.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

from standard_quant_tools.analysis.derivatives import _bounded_exponent
from standard_quant_tools.error import ValidationError

from ._numbers import finite, positive
from .daycount import DEFAULT_CONVENTION, to_date, year_fraction

__all__ = ["dividend_points"]

#: A `time_to_expiry` further than this from the dates' own year fraction
#: (half a day) is said to disagree with them.
_TIME_TOLERANCE_YEARS = 0.5 / 365.0


def dividend_points(
    constituents: Sequence[Mapping[str, Any]],
    *,
    divisor: float,
    as_of: Any,
    expiry: Any,
    spot: Optional[float] = None,
    future_price: Optional[float] = None,
    financing_rate: Optional[float] = None,
    time_to_expiry: Optional[float] = None,
    day_count: str = DEFAULT_CONVENTION,
) -> Dict[str, Any]:
    """
    Total index points of dividend between two dates, attributed and dated.

    Each constituent is a mapping with `symbol`, `shares` (index shares, not
    shares outstanding), `dividend_per_share`, `ex_date`, and optionally
    `pay_date`. Only ex-dates strictly after `as_of` and on or before
    `expiry` count -- a dividend that has already gone ex is in the spot
    price, and one going ex after expiry is somebody else's contract.

    With `financing_rate`, each counted dividend is discounted from its pay
    date (its ex-date when no pay date is given) back to `as_of`, at that
    rate continuously compounded over the `day_count` year fraction, and
    `forecast_pv_points` is their sum; `total_index_points` stays the
    nominal sum.

    Supply `spot`, `future_price`, `financing_rate` and `time_to_expiry`
    together to get the market's own number alongside yours: the present
    value of the dividends the quoted future implies, given that financing.
    `forecast_minus_implied` compares it with `forecast_pv_points` -- a
    present value against a present value -- and that difference is the
    tradeable quantity.
    """
    d = positive(divisor, "divisor")
    start = to_date(as_of, "as_of")
    end = to_date(expiry, "expiry")
    if end < start:
        raise ValidationError(
            f"expiry {end} precedes as_of {start}. A dividend total over a "
            "negative period is not a quantity."
        )
    # Validates the convention by name before anything is computed with it.
    window_years = year_fraction(start, end, convention=day_count)

    rate: Optional[float] = None
    if financing_rate is not None:
        rate = finite(financing_rate, "financing_rate", "dividend_points")

    rows = _parse(constituents)
    included: List[Dict[str, Any]] = []
    excluded_before = 0
    excluded_after = 0
    to_ex_date = 0

    for row in rows:
        ex = row["ex_date"]
        if ex <= start:
            excluded_before += 1
            continue
        if ex > end:
            excluded_after += 1
            continue
        points = float(row["shares"] * row["dividend_per_share"] / d)
        paid = row["pay_date"]
        discount_date = paid if paid is not None else ex
        if paid is None:
            to_ex_date += 1
        years = year_fraction(start, discount_date, convention=day_count)
        present_value: Optional[float] = None
        if rate is not None:
            exponent = _bounded_exponent(
                -rate * years,
                f"-financing_rate * year_fraction(as_of, {discount_date})",
                "dividend_points",
            )
            present_value = points * math.exp(exponent)
        included.append(
            {
                "symbol": row["symbol"],
                "ex_date": ex.isoformat(),
                "pay_date": paid.isoformat() if paid is not None else None,
                "dividend_per_share": row["dividend_per_share"],
                "index_points": points,
                "discounted_to": "pay_date" if paid is not None else "ex_date",
                "discount_years": float(years),
                "present_value_points": present_value,
            }
        )

    total_points = float(sum(item["index_points"] for item in included))
    forecast_pv: Optional[float] = None
    if rate is not None:
        forecast_pv = float(
            sum(item["present_value_points"] for item in included)  # type: ignore[misc]
        )

    by_date: Dict[str, float] = {}
    for item in included:
        by_date[item["ex_date"]] = (
            by_date.get(item["ex_date"], 0.0) + item["index_points"]
        )

    by_constituent = sorted(
        included, key=lambda item: item["index_points"], reverse=True
    )

    implied_points: Optional[float] = None
    difference: Optional[float] = None
    if future_price is not None:
        missing = [
            name
            for name, value in (
                ("spot", spot),
                ("financing_rate", financing_rate),
                ("time_to_expiry", time_to_expiry),
            )
            if value is None
        ]
        if missing:
            raise ValidationError(
                f"implying dividends from a quoted future also needs {missing}. "
                "The future embeds financing and dividends together and there "
                "is no way to separate them without knowing the financing."
            )
        s = positive(spot, "spot")
        f = positive(future_price, "future_price")
        t = positive(time_to_expiry, "time_to_expiry")
        # F = S exp(rT) - sum d_i exp(r (T - t_i))  ->  S - F exp(-rT) =
        # sum d_i exp(-r t_i): the PRESENT VALUE of the window's dividends.
        # The discrete form, deliberately: the whole point of this module is
        # that the continuous-yield form is the approximation being avoided.
        # No bound at all here -- only the pydantic layer constrained it,
        # so the library call overflowed outright. And a percent-for-
        # fraction typo (4.3 for 4.3%) produced 5,308 index points of
        # implied dividends on a 6,000 index -- 88% of the index paid out
        # in six months -- reported with the ordinary "It is a position"
        # warning. `solve_carry` guards exactly this with `abs(...) > 1.0`.
        exponent = _bounded_exponent(
            -float(financing_rate) * t,
            "-financing_rate * time_to_expiry",
            "dividend_points",
        )
        implied_points = float(s - f * math.exp(exponent))
        # Present value against present value. The nominal total against
        # this read the discounting itself as a view (see the module notes).
        difference = float(forecast_pv) - implied_points  # type: ignore[arg-type]

    warnings: List[str] = []

    # A plausibility floor on the IMPLIED total, which `solve_carry` has and
    # this did not. Passing 4.3 for a 4.3% financing rate -- the exact
    # mistake models.py's own header names -- returns 5,308 index points on
    # a 6,000 index: 88% of the index paid out as dividends in six months,
    # reported with the ordinary "It is a position" line below and nothing
    # else. This does not refuse, because a genuine special can be large;
    # it says which input is the likely cause.
    if spot and implied_points is not None:
        implied_share = abs(implied_points) / float(spot)
        if implied_share > 0.25:
            warnings.append(
                f"The future implies {implied_points:.2f} index points of "
                f"dividends, which is {implied_share * 100:.0f}% of spot over "
                f"this window. No index pays that. The usual cause is a "
                f"financing_rate given as a percent rather than a fraction "
                f"({financing_rate!r} would be "
                f"{float(financing_rate) * 100:.0f}%) -- check that before "
                "reading the gap below as a position."
            )

    if time_to_expiry is not None:
        given = float(time_to_expiry)
        if abs(given - window_years) > _TIME_TOLERANCE_YEARS:
            warnings.append(
                f"time_to_expiry={given:.6f} years, but as_of {start} to "
                f"expiry {end} is {window_years:.6f} years under {day_count}. "
                "The future's financing is taken over time_to_expiry and each "
                f"dividend is discounted over the dates under {day_count}, so "
                "the two clocks disagree; pass the year fraction of the dates "
                "(or the day_count it was computed under)."
            )

    if not included:
        warnings.append(
            f"No constituent goes ex between {start} and {end}, so the total "
            f"is zero. {excluded_before} had already gone ex and "
            f"{excluded_after} go ex after expiry -- a zero here is a "
            "statement about the window, not about the index."
        )
    if excluded_before:
        warnings.append(
            f"{excluded_before} dividend(s) had already gone ex on or before "
            f"{start} and are excluded. They are already in the spot price; "
            "counting them again would double-count them into the forward."
        )
    if rate is not None and to_ex_date:
        warnings.append(
            f"{to_ex_date} of {len(included)} dividend(s) were discounted to "
            "their ex-date because no pay_date was given. The cash arrives on "
            "the pay date, days to weeks later, so each present value is "
            "overstated by about financing_rate x the lag (0.3% of the "
            "dividend at 5% and three weeks). Pass pay_date for the exact "
            "figure."
        )
    if rate is None and included:
        warnings.append(
            "No financing_rate, so the dividends are not discounted: "
            "forecast_pv_points is null and total_index_points is a nominal "
            "sum, which is NOT comparable with a futures-implied number (a "
            "present value)."
        )
    if difference is not None and abs(difference) > 0.5:
        richer = "above" if difference > 0 else "below"
        warnings.append(
            f"Your forecast is {abs(difference):.2f} index points {richer} "
            f"what the future implies, both as present values "
            f"({forecast_pv:.2f} against {implied_points:.2f}; "
            f"{total_points:.2f} nominal). That gap is usually not an error "
            "on either side: the market prices a distribution over cuts and "
            "specials while a forecast is a point estimate. It is a position."
        )
    if spot:
        warnings.append(
            f"{total_points:.2f} points is "
            f"{total_points / float(spot) * 100.0:.2f}% of spot over this "
            "window. Compare that to the annualized yield you would "
            "otherwise have used -- the two agree only when the window "
            "happens to contain a representative slice of the season."
        )

    return {
        "as_of": start.isoformat(),
        "expiry": end.isoformat(),
        "divisor": d,
        "day_count": day_count,
        "n_constituents": len(rows),
        "n_included": len(included),
        "n_excluded_already_ex": excluded_before,
        "n_excluded_after_expiry": excluded_after,
        "n_discounted_to_ex_date": to_ex_date if rate is not None else 0,
        "total_index_points": total_points,
        "forecast_pv_points": forecast_pv,
        "points_by_date": dict(sorted(by_date.items())),
        "points_by_constituent": by_constituent,
        "largest_contributor": (
            by_constituent[0]["symbol"] if by_constituent else None
        ),
        "implied_index_points": implied_points,
        "forecast_minus_implied": difference,
        "warnings": warnings,
    }


# ── internals ───────────────────────────────────────────────────────────


def _parse(constituents: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not constituents:
        raise ValidationError("constituents is empty; there are no dividends to sum.")
    rows: List[Dict[str, Any]] = []
    for index, item in enumerate(constituents):
        if not isinstance(item, Mapping):
            raise ValidationError(
                f"constituents[{index}] is {type(item).__name__}, not a mapping."
            )
        symbol = str(item.get("symbol") or f"constituent_{index}")
        for field in ("shares", "dividend_per_share", "ex_date"):
            if field not in item:
                raise ValidationError(f"constituent {symbol!r} has no `{field}`.")
        ex = to_date(item["ex_date"], f"{symbol}.ex_date")
        paid = None
        if item.get("pay_date") is not None:
            paid = to_date(item["pay_date"], f"{symbol}.pay_date")
            if paid < ex:
                raise ValidationError(
                    f"constituent {symbol!r}: pay_date {paid} precedes ex_date "
                    f"{ex}. A dividend is paid on or after the date that "
                    "decides who receives it; check the two are not swapped."
                )
        rows.append(
            {
                "symbol": symbol,
                "shares": positive(item["shares"], f"{symbol}.shares"),
                "dividend_per_share": finite(
                    item["dividend_per_share"], f"{symbol}.dividend_per_share"
                ),
                "ex_date": ex,
                "pay_date": paid,
            }
        )
    return rows
