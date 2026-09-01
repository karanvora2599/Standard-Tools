"""
Year fractions, because every carry number is a rate times one.

WHY THIS EXISTS AT ALL. Before this module the library had no year
fraction anywhere -- no `year_fraction`, no day-count convention, no
`ACT/360`. What it had instead was five inline `/ 365.0` sites in three
files, each correct for its own caller and none reusable, and
`TRADING_DAYS = 252` written out as an independent literal in seven
modules. Carry is a rate applied over a period, so a Delta One surface
without a year fraction is a surface that cannot price anything.

THE CONVENTION IS NOT A DETAIL. USD financing accrues ACT/360 and GBP
accrues ACT/365F, and the same six-month period is 0.502778 years under
one and 0.495890 under the other. On $100m financed at 4% that gap is
about $27,500 -- not a rounding error, and not visible anywhere in the
answer unless the convention is named. So it is an explicit argument with
a stated default rather than a hidden constant, and the functions that use
it report which one they used.

WHAT THIS DELIBERATELY DOES NOT DO. There is no holiday calendar in this
library and no exchange calendar dependency, so there are no business-day
conventions here -- no Following, no Modified Following, no good-business-
day adjustment of a maturity date. Every convention below counts calendar
days. A settlement date that lands on a Sunday is used as given, and if
that matters for your instrument the adjustment belongs upstream, done by
something that knows the holidays. Saying so is better than shipping a
`Modified Following` that silently assumes weekends are the only holidays.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Union

import pandas as pd

from standard_quant_tools.error import ValidationError

__all__ = [
    "CONVENTIONS",
    "DEFAULT_CONVENTION",
    "TRADING_DAYS_PER_YEAR",
    "day_count",
    "to_date",
    "year_fraction",
]

DateLike = Union[str, _dt.date, _dt.datetime, "pd.Timestamp"]

#: Trading days in a year. This library counts 252 in seven other modules,
#: each as its own literal; this is the name new code should import so the
#: eighth copy is not written. It is NOT interchangeable with a day count
#: below -- 252 annualizes a series of daily observations, while a year
#: fraction prices an accrual over calendar time, and using either for the
#: other's job is a real and quiet error.
TRADING_DAYS_PER_YEAR = 252

#: What each convention counts, in one line, for the error message and for
#: the `convention` field every caller gets back.
CONVENTIONS: Dict[str, str] = {
    "ACT/365F": (
        "Actual calendar days over a fixed 365. The equity and GBP default, "
        "and the one this library's existing inline `/ 365.0` sites use."
    ),
    "ACT/360": (
        "Actual calendar days over 360. USD and EUR money-market financing, "
        "so it is the right one for a repo or margin-loan leg -- it accrues "
        "about 1.4% more than ACT/365F over the same period."
    ),
    "30/360": (
        "Every month is 30 days and every year 360 (the US/Bond basis). "
        "Used by many total return swap financing legs. Counts days that do "
        "not exist, which is the point: it makes every period regular."
    ),
    "ACT/ACT": (
        "Actual days over the actual length of the year(s) spanned, "
        "splitting the period at year boundaries so leap years count 366. "
        "The most faithful to the calendar and the least common in equity "
        "financing."
    ),
}

DEFAULT_CONVENTION = "ACT/365F"


def to_date(value: DateLike, name: str) -> _dt.date:
    """
    Anything date-shaped in, a `datetime.date` out.

    Accepts what a JSON payload actually carries (an ISO string) as well as
    the three Python types, because refusing a string here would push the
    parsing into every caller and they would not all do it the same way.
    """
    if value is None:
        raise ValidationError(f"{name} is required and was not given")
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{name}={value!r} is not a date. Pass an ISO string "
            "('2026-03-20'), a datetime.date, or a pandas Timestamp."
        ) from exc
    if pd.isna(stamp):
        raise ValidationError(f"{name}={value!r} parsed as NaT, not a date.")
    return stamp.date()


def day_count(start: DateLike, end: DateLike, *, convention: str = DEFAULT_CONVENTION):
    """
    The numerator and denominator a year fraction is built from, separately.

    Returned as a pair rather than a quotient because the two carry
    different information when a result looks wrong: a surprising numerator
    is a date problem and a surprising denominator is a convention problem,
    and one number cannot tell you which you have.
    """
    convention = _convention(convention)
    d0, d1 = to_date(start, "start"), to_date(end, "end")

    if convention == "30/360":
        # US/Bond basis. The two clamps are the convention, not a tidy-up:
        # a 31st is treated as the 30th, and only after the start day has
        # been clamped -- doing them in the other order changes the count
        # for a period that starts on the 31st.
        y0, m0, dd0 = d0.year, d0.month, min(d0.day, 30)
        y1, m1, dd1 = d1.year, d1.month, d1.day
        if dd1 == 31 and dd0 == 30:
            dd1 = 30
        days = 360 * (y1 - y0) + 30 * (m1 - m0) + (dd1 - dd0)
        return float(days), 360.0

    days = float((d1 - d0).days)

    if convention == "ACT/365F":
        return days, 365.0
    if convention == "ACT/360":
        return days, 360.0

    # ACT/ACT: the denominator is not a constant, so report the effective
    # one -- days divided by the fraction -- rather than a number that
    # would not reproduce the quotient.
    fraction = _act_act(d0, d1)
    denominator = days / fraction if fraction else 365.0
    return days, denominator


def year_fraction(
    start: DateLike,
    end: DateLike,
    *,
    convention: str = DEFAULT_CONVENTION,
) -> float:
    """
    The fraction of a year between two dates, under a named convention.

    Negative when `end` precedes `start`, deliberately. A carry calculation
    run on a contract that already expired should produce a negative
    accrual and let the caller notice, rather than an absolute value that
    quietly prices the position as though time ran the other way.
    """
    convention = _convention(convention)
    d0, d1 = to_date(start, "start"), to_date(end, "end")

    if convention == "ACT/ACT":
        return _act_act(d0, d1)

    numerator, denominator = day_count(d0, d1, convention=convention)
    return numerator / denominator


# ── internals ───────────────────────────────────────────────────────────


def _convention(value: Any) -> str:
    """Normalize and check a convention name, naming the alternatives."""
    if not isinstance(value, str):
        raise ValidationError(
            f"convention must be a string, got {value!r}. "
            f"One of: {sorted(CONVENTIONS)}."
        )
    key = value.strip().upper().replace("ACT/365", "ACT/365F").replace("FF", "F")
    # "ACT/365" is the same thing as "ACT/365F" often enough that refusing
    # it would be pedantry; "30E/360" is a DIFFERENT convention and is
    # refused rather than silently treated as "30/360".
    if key not in CONVENTIONS:
        raise ValidationError(
            f"convention={value!r} is not one this library counts. "
            f"Supported: {sorted(CONVENTIONS)}. Note that 30E/360 (the "
            "Eurobond basis) differs from 30/360 in how it clamps a 31st "
            "and is not supported rather than approximated."
        )
    return key


def _act_act(d0: _dt.date, d1: _dt.date) -> float:
    """
    ACT/ACT (ISDA): each calendar year contributes its own actual length.

    Split at year boundaries rather than dividing by 365.25, because the
    average-year shortcut is wrong for any period shorter than several
    years -- which is every period a Delta One carry calculation uses.
    """
    if d1 == d0:
        return 0.0
    if d1 < d0:
        return -_act_act(d1, d0)

    total = 0.0
    year = d0.year
    cursor = d0
    while cursor < d1:
        year_end = _dt.date(year + 1, 1, 1)
        segment_end = min(year_end, d1)
        days_in_year = 366.0 if _is_leap(year) else 365.0
        total += (segment_end - cursor).days / days_in_year
        cursor = segment_end
        year += 1
    return total


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
