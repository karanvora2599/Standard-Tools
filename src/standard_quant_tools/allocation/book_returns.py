"""
What a portfolio with deposits and withdrawals returned, three ways (GIPS
2020): the true time-weighted return, Modified Dietz linked across
sub-periods, and the money-weighted return.

THE THREE ANSWER DIFFERENT QUESTIONS. A manager who makes +20% and then
-10% on a portfolio whose owner doubled it at the top has a time-weighted
return of 1.2 x 0.9 - 1 = +8.000%: the manager's result, independent of
when the money arrived. The owner lost money: 100 grew to 120, 120 more
went in, and 240 became 216, so the rate that equates what went in with
what came out -- the money-weighted return, an IRR -- is -1.2549% a year.
The gap is the owner's timing, not the manager's skill.

    sub-period (true TWR)   r_j = V_{j+1} / (V_j + F_j) - 1
    Modified Dietz          R_j = (V_{j+1} - V_j - sum F) / (V_j + sum w_i F_i),
                            w_i = (CD - D_i) / CD (calendar days)
    linked                  prod (1 + r_j) - 1
    money-weighted          the IRR of -V_0, -F_i and +V_end, by
                            `finance.cash_flows.rate_roots`

A VALUATION ON A FLOW'S DATE is taken BEFORE the flow by default -- the
flow arrives after the close and the next sub-period starts from V + F --
or, with `flows_in_valuation`, already includes it. Under the first a flow
on the last valuation date is after the period, and under the second a
flow on the first is before it; both are refused by name.

THE TRUE TWR NEEDS A VALUATION AT EVERY FLOW. Without one the sub-periods
it multiplies cannot be formed, so it is NaN and the dates missing a
valuation are listed -- never approximated. Modified Dietz is the
approximation, and is reported beside it: between two valuations it
weights each flow by the share of the sub-period it was invested for, and
with every flow on a valuation date it IS the true TWR.

NOTHING UNDER A YEAR IS ANNUALIZED (GIPS 2020): the annualized fields are
NaN for a period shorter than one year under the day count, and the
cumulative ones are always reported. The money-weighted return over the
period is (1 + IRR)^T - 1, which does not depend on the day count's year
for the ACT conventions; its annual rate does.

AN MWR IS AN IRR AND CAN HAVE SEVERAL. Withdrawals between deposits can
give the flows more than one root; then the MWR is NaN with every root
listed as a return over the period.

A LARGE FLOW is one above `large_flow_share` of the portfolio's value
(default 10%), measured against the valuation before it; each is listed,
with whether it has its own valuation -- GIPS 2020 asks for one at a
large cash flow.

Refused, by name: fewer than two valuations, a negative or non-finite
value, a non-finite flow, two keys naming one date, a flow outside the
period, and more than MAX_VALUATIONS valuations or MAX_FLOWS flows.
"""

from __future__ import annotations

import bisect
import datetime as _dt
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from standard_quant_tools.delta_one.daycount import _convention as _day_count_name
from standard_quant_tools.delta_one.daycount import to_date, year_fraction
from standard_quant_tools.error import ValidationError
from standard_quant_tools.finance.cash_flows import rate_roots, root_status

__all__ = [
    "MAX_FLOWS",
    "MAX_PERIOD_ROWS",
    "MAX_VALUATIONS",
    "measure_book_returns",
]

#: The most valuations one call takes: eighty years of daily values.
MAX_VALUATIONS = 20_000
#: The most external flows one call takes.
MAX_FLOWS = 5_000
#: Sub-periods are listed one row each up to this many, and summarized
#: beyond it: a daily-valued decade is 2,500 rows nobody reads.
MAX_PERIOD_ROWS = 400
_MAX_ABS_AMOUNT = 1e15


def _dated(
    values: Optional[Mapping[Any, float]], field: str, most: int
) -> Dict[_dt.date, float]:
    out: Dict[_dt.date, float] = {}
    if not values:
        return out
    if len(values) > most:
        raise ValidationError(
            f"{field} has {len(values)} entries, more than the {most} one call "
            "takes."
        )
    for key, value in values.items():
        day = to_date(key, f"{field} date {key!r}")
        if day in out:
            raise ValidationError(
                f"two {field} keys fall on {day.isoformat()} (one is {key!r}); "
                "give one entry per date."
            )
        number = float(value)
        if not math.isfinite(number) or abs(number) > _MAX_ABS_AMOUNT:
            raise ValidationError(
                f"{field}[{key!r}]={value!r} must be a finite amount within "
                f"{_MAX_ABS_AMOUNT:g}."
            )
        out[day] = number
    return dict(sorted(out.items()))


def _listed(days: List[_dt.date], most: int = 5) -> str:
    names = [d.isoformat() for d in days[:most]]
    more = f" and {len(days) - most} more" if len(days) > most else ""
    return f"{names}{more}"


def _annualized(total: float, years: float) -> Optional[float]:
    """None under a year (GIPS 2020 does not annualize it); NaN when the
    return over the period is not defined."""
    if years < 1.0:
        return None
    if not math.isfinite(total) or total <= -1.0:
        return float("nan")
    return math.expm1(math.log1p(total) / years)


def _linked(returns: List[float]) -> float:
    growth = 1.0
    for r in returns:
        if not math.isfinite(r):
            return float("nan")
        growth *= 1.0 + r
    return growth - 1.0


def measure_book_returns(
    valuations: Mapping[Any, float],
    flows: Optional[Mapping[Any, float]] = None,
    *,
    day_count: str = "ACT/365F",
    flows_in_valuation: bool = False,
    large_flow_share: float = 0.10,
) -> Dict[str, Any]:
    """
    The true time-weighted, linked Modified Dietz and money-weighted
    returns of a portfolio valued on `valuations` ({date: market value})
    with external `flows` ({date: amount}, + a contribution, - a
    withdrawal). A number that is not defined is NaN: the true TWR without
    a valuation at every flow, a return across a sub-period that opens
    with no capital, an MWR the flows give no single IRR for. An
    annualized figure for a period under a year is None, as is the timing
    gap when either return is missing.
    """
    values = _dated(valuations, "valuations", MAX_VALUATIONS)
    if len(values) < 2:
        raise ValidationError(
            f"valuations has {len(values)} date(s); a return needs at least "
            "the starting and the ending market value."
        )
    negative = [d for d, v in values.items() if v < 0.0]
    if negative:
        raise ValidationError(
            f"valuations are negative on {_listed(negative)}. A portfolio's "
            "market value is 0 or more; a negative one has no return."
        )
    external = _dated(flows, "flows", MAX_FLOWS)
    share = float(large_flow_share)
    if not math.isfinite(share) or not 0.0 < share <= 1.0:
        raise ValidationError(
            f"large_flow_share={large_flow_share!r} must be in (0, 1]; GIPS "
            "firms commonly use 0.10 (10% of the portfolio's value)."
        )
    convention = _day_count_name(day_count)

    days = list(values)
    first, last = days[0], days[-1]
    if flows_in_valuation:
        outside = [d for d in external if not first < d <= last]
        rule = (
            "with flows_in_valuation=True a flow must fall after the first "
            "valuation date (whose value already includes it) and on or "
            "before the last"
        )
    else:
        outside = [d for d in external if not first <= d < last]
        rule = (
            "with flows_in_valuation=False a valuation is taken before a "
            "flow on its date, so a flow must fall on or after the first "
            "valuation date and before the last"
        )
    if outside:
        raise ValidationError(
            f"flows on {_listed(outside)} fall outside the period "
            f"{first.isoformat()}..{last.isoformat()}: {rule}. Add the "
            "valuation that closes the period after them, or drop them."
        )

    notes: List[str] = []
    valued = set(days)
    missing = [d for d in external if d not in valued]
    flow_days = list(external)

    # Each flow's sub-period: the valuations either side of it.
    by_period: Dict[int, List[Tuple[_dt.date, float]]] = {}
    for d, amount in external.items():
        if flows_in_valuation:
            j = bisect.bisect_left(days, d) - 1
        else:
            j = bisect.bisect_right(days, d) - 1
        by_period.setdefault(j, []).append((d, amount))

    rows: List[Dict[str, Any]] = []
    dietz: List[float] = []
    true_twr: List[float] = []
    for j in range(len(days) - 1):
        start, end = days[j], days[j + 1]
        v0, v1 = values[start], values[end]
        inside = by_period.get(j, [])
        calendar = (end - start).days
        net = math.fsum(a for _d, a in inside)
        weighted = math.fsum(
            a * (calendar - (d - start).days) / calendar for d, a in inside
        )
        capital = v0 + weighted
        r_dietz = (v1 - v0 - net) / capital if capital > 0.0 else float("nan")
        boundary = end if flows_in_valuation else start
        on_boundary = all(d == boundary for d, _a in inside)
        if not on_boundary:
            r_true = float("nan")
        elif flows_in_valuation:
            r_true = (v1 - net) / v0 - 1.0 if v0 > 0.0 else float("nan")
        else:
            opening = v0 + net
            r_true = v1 / opening - 1.0 if opening > 0.0 else float("nan")
        dietz.append(r_dietz)
        true_twr.append(r_true)
        rows.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "start_value": v0,
                "end_value": v1,
                "net_flow": net,
                "modified_dietz": r_dietz,
                "time_weighted": r_true,
            }
        )

    years = year_fraction(first, last, convention=convention)
    # Never approximated: without a valuation at every flow it is NaN, and
    # `missing_valuation_dates` names the dates.
    twr = float("nan") if missing else _linked(true_twr)
    linked_dietz = _linked(dietz)

    # The owner's cash: in at the start and with each deposit, out with each
    # withdrawal and at the end.
    cash: Dict[_dt.date, float] = {first: -values[first]}
    for d, amount in external.items():
        cash[d] = cash.get(d, 0.0) - amount
    cash[last] = cash.get(last, 0.0) + values[last]
    cash_days = sorted(cash)
    times = [year_fraction(first, d, convention=convention) for d in cash_days]
    roots = rate_roots(times, [cash[d] for d in cash_days])
    status, _why = root_status(roots)
    with np.errstate(over="ignore"):
        period_roots = [float(np.expm1(u * years)) for u in roots.log_rates]
    if status == "unique":
        u = roots.log_rates[0]
        with np.errstate(over="ignore"):
            mwr = float(np.expm1(u * years))
            mwr_annual = float(np.expm1(u)) if years >= 1.0 else None
    else:
        mwr = float("nan")
        mwr_annual = float("nan") if years >= 1.0 else None
    if years < 1.0:
        notes.append(
            f"The period is {years:.4f} years ({convention}), under one, so "
            "no return is annualized (GIPS 2020); the cumulative returns are "
            "for the whole period."
        )

    large: List[Dict[str, Any]] = []
    for d, amount in external.items():
        if d in valued:
            reference = values[d] - amount if flows_in_valuation else values[d]
        else:
            reference = values[days[bisect.bisect_right(days, d) - 1]]
        fraction = abs(amount) / reference if reference > 0.0 else float("inf")
        if fraction > share:
            large.append(
                {
                    "date": d.isoformat(),
                    "amount": amount,
                    "share_of_value": fraction,
                    "valued": d in valued,
                }
            )
    if large:
        unvalued = [row["date"] for row in large if not row["valued"]]
        notes.append(
            f"{len(large)} flow(s) exceed {share:.0%} of the portfolio's value "
            "(large_flows); GIPS 2020 asks for a valuation at a large cash "
            "flow"
            + (
                f", and {unvalued[:5]} have none, so the Modified Dietz return "
                "across them is an approximation."
                if unvalued
                else ", and every one has its own."
            )
        )
    listed = rows if len(rows) <= MAX_PERIOD_ROWS else []
    if len(rows) > MAX_PERIOD_ROWS:
        notes.append(
            f"periods is empty: {len(rows)} sub-periods are more than the "
            f"{MAX_PERIOD_ROWS} listed one by one."
        )
    gap = mwr - twr if math.isfinite(mwr) and math.isfinite(twr) else None
    return {
        "start_date": first.isoformat(),
        "end_date": last.isoformat(),
        "years": years,
        "day_count": convention,
        "flows_in_valuation": bool(flows_in_valuation),
        "n_valuations": len(days),
        "n_flows": len(flow_days),
        "n_periods": len(rows),
        "start_value": values[first],
        "end_value": values[last],
        "net_flows": math.fsum(external.values()),
        "time_weighted_return": twr,
        "time_weighted_annualized": _annualized(twr, years),
        "modified_dietz_return": linked_dietz,
        "modified_dietz_annualized": _annualized(linked_dietz, years),
        "money_weighted_return": mwr,
        "money_weighted_annualized": mwr_annual,
        "mwr_status": status,
        "mwr_roots": period_roots,
        "timing_gap": gap,
        "missing_valuation_dates": [d.isoformat() for d in missing[:50]],
        "n_missing_valuation_dates": len(missing),
        "large_flows": large,
        "periods": listed,
        "warnings": notes,
    }
