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

from standard_quant_tools.error import ValidationError

from ._numbers import bounded, finite, non_negative, positive
from .daycount import to_date

__all__ = ["dividend_points"]


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
) -> Dict[str, Any]:
    """
    Total index points of dividend between two dates, attributed and dated.

    Each constituent is a mapping with `symbol`, `shares` (index shares, not
    shares outstanding), `dividend_per_share`, and `ex_date`. Only ex-dates
    strictly after `as_of` and on or before `expiry` count -- a dividend
    that has already gone ex is in the spot price, and one going ex after
    expiry is somebody else's contract.

    Supply `spot`, `future_price`, `financing_rate` and `time_to_expiry`
    together to get the market's own number alongside yours: the dividend
    total the quoted future implies, given that financing. The difference is
    the tradeable quantity.
    """
    d = positive(divisor, "divisor")
    start = to_date(as_of, "as_of")
    end = to_date(expiry, "expiry")
    if end < start:
        raise ValidationError(
            f"expiry {end} precedes as_of {start}. A dividend total over a "
            "negative period is not a quantity."
        )

    rows = _parse(constituents)
    included: List[Dict[str, Any]] = []
    excluded_before = 0
    excluded_after = 0

    for row in rows:
        ex = row["ex_date"]
        if ex <= start:
            excluded_before += 1
            continue
        if ex > end:
            excluded_after += 1
            continue
        points = row["shares"] * row["dividend_per_share"] / d
        included.append(
            {
                "symbol": row["symbol"],
                "ex_date": ex.isoformat(),
                "dividend_per_share": row["dividend_per_share"],
                "index_points": float(points),
            }
        )

    total_points = float(sum(item["index_points"] for item in included))

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
        # F = (S - D) * exp(r*T)  ->  D = S - F * exp(-r*T). The discrete
        # form, deliberately: the whole point of this module is that the
        # continuous-yield form is the approximation being avoided.
        implied_points = float(s - f * math.exp(-float(financing_rate) * t))
        difference = total_points - implied_points

    warnings: List[str] = []
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
    if difference is not None and abs(difference) > 0.5:
        richer = "above" if difference > 0 else "below"
        warnings.append(
            f"Your forecast is {abs(difference):.2f} index points {richer} "
            f"what the future implies ({total_points:.2f} against "
            f"{implied_points:.2f}). That gap is usually not an error on "
            "either side: the market prices a distribution over cuts and "
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
        "n_constituents": len(rows),
        "n_included": len(included),
        "n_excluded_already_ex": excluded_before,
        "n_excluded_after_expiry": excluded_after,
        "total_index_points": total_points,
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
        rows.append(
            {
                "symbol": symbol,
                "shares": positive(item["shares"], f"{symbol}.shares"),
                "dividend_per_share": finite(
                    item["dividend_per_share"], f"{symbol}.dividend_per_share"
                ),
                "ex_date": to_date(item["ex_date"], f"{symbol}.ex_date"),
            }
        )
    return rows
