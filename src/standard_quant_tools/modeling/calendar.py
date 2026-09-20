"""
Exchange calendars: the one thing that turns "bars per hour" into "bars
per year".

WHY THIS COULD NOT BE A CONSTANT. Daily, weekly and monthly bars annualize
by calendar arithmetic and need no assumption about a venue. An intraday
bar does not: bars per year at "1h" is bars per SESSION times sessions per
year, and the session is 6.5 hours on NYSE, 8.5 on the LSE and 24 on a
crypto venue. A constant chosen for one of them is wrong by a fixed factor
on the others while still looking precise, which is why `features/risk.py`
has refused to annualize an intraday interval at all rather than guess.

The calendar is a property of the DATASET, named on `DatasetSpec.calendar`
as an `exchange_calendars` code ("XNYS", "XLON", "24/7"), so it is hashed
into the dataset's identity, bundled into the model, and reused by the
portfolio evaluator -- the same rule as `provider` and `interval`. The
library is an optional dependency, guarded like optuna and lightgbm: a
spec that names a calendar on a machine without it is refused by name.

TWO NUMBERS, BOTH READ OFF THE CALENDAR RATHER THAN ASSUMED. Sessions per
year is the count of sessions over the calendar's whole years divided by
the years, so holidays are counted and not estimated. Session length is
the MEDIAN over the most recent sessions, so an early close before a
holiday does not shorten every day. Bars per session is the session
rounded UP to whole bars, because a provider emits the partial last bar
(seven hourly bars for a 6.5-hour NYSE session) and the annualization
should count the bars that are actually there.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import List, Optional

from standard_quant_tools.error import ValidationError

#: Sessions examined for the session length: about a year, so the median
#: is a full session and not an early close.
_SESSIONS_FOR_LENGTH = 250

_INTERVAL = re.compile(r"^\s*(\d+)\s*(m|min|h|hr|hour)\s*$", re.IGNORECASE)


def calendar_available() -> bool:
    """Whether `exchange_calendars` can be imported, without importing it."""
    from importlib.util import find_spec

    try:
        return find_spec("exchange_calendars") is not None
    except (ImportError, ValueError):
        return False


def require_calendar_library(where: str) -> None:
    """Refuse, by name, a calendar on a machine without the library."""
    if calendar_available():
        return
    raise ValidationError(
        f"{where}: resolving an exchange calendar needs the optional "
        "`exchange_calendars` package, which is not installed in this "
        "environment (pip install exchange_calendars). Without it an intraday "
        "interval cannot be annualized, and a daily-or-coarser interval does "
        "not need it."
    )


def interval_minutes(interval: str) -> Optional[int]:
    """Minutes per bar for an intraday interval ('5m', '90m', '1h'), or
    None for anything else -- a daily-or-coarser interval, or one this
    library does not recognise."""
    match = _INTERVAL.match(str(interval))
    if not match:
        return None
    count, unit = int(match.group(1)), match.group(2).lower()
    minutes = count * (60 if unit in ("h", "hr", "hour") else 1)
    return minutes if minutes > 0 else None


@lru_cache(maxsize=None)
def calendar_names() -> List[str]:
    require_calendar_library("calendar_names")
    import exchange_calendars as xcals

    return sorted(xcals.get_calendar_names())


def validate_calendar_name(name: str, where: str = "DatasetSpec.calendar") -> str:
    """The name, or a refusal that lists what would have been accepted."""
    require_calendar_library(where)
    text = str(name).strip()
    names = calendar_names()
    if text not in names:
        sample = ", ".join(names[:8])
        raise ValidationError(
            f"{where}: {name!r} is not an exchange_calendars name. Known names "
            f"include {sample}, ... ({len(names)} in all); "
            "exchange_calendars.get_calendar_names() lists them."
        )
    return text


@lru_cache(maxsize=None)
def sessions_per_year(calendar: str) -> float:
    """Sessions per year, counted over the calendar's complete years."""
    import pandas as pd

    cal = _calendar(calendar)
    sessions = pd.DatetimeIndex(cal.sessions)
    first_year = sessions[0].year + (0 if sessions[0].is_year_start else 1)
    last_year = sessions[-1].year - (0 if sessions[-1].is_year_end else 1)
    if last_year < first_year:
        raise ValidationError(
            f"calendar {calendar!r} spans no complete year, so sessions per "
            "year cannot be counted from it."
        )
    within = sessions[(sessions.year >= first_year) & (sessions.year <= last_year)]
    return float(len(within)) / float(last_year - first_year + 1)


@lru_cache(maxsize=None)
def session_minutes(calendar: str) -> float:
    """Length of a full session in minutes: the median over recent sessions."""
    import pandas as pd

    cal = _calendar(calendar)
    sessions = pd.DatetimeIndex(cal.sessions)[-_SESSIONS_FOR_LENGTH:]
    lengths = [
        (cal.session_close(s) - cal.session_open(s)).total_seconds() / 60.0
        for s in sessions
    ]
    return float(pd.Series(lengths).median())


def bars_per_session(interval: str, calendar: str) -> int:
    """Whole bars in a session, the partial last bar counted."""
    minutes = interval_minutes(interval)
    if minutes is None:
        raise ValidationError(
            f"interval {interval!r} is not an intraday interval this library "
            "can place in a session; daily-or-coarser intervals annualize by "
            "calendar arithmetic and need no exchange calendar."
        )
    return int(math.ceil(session_minutes(calendar) / minutes))


def periods_per_year(interval: str, calendar: str) -> int:
    """Bars per year for an intraday interval on a named calendar."""
    name = validate_calendar_name(calendar, "periods_per_year")
    return int(round(bars_per_session(interval, name) * sessions_per_year(name)))


def _calendar(name: str):
    require_calendar_library("calendar")
    import exchange_calendars as xcals

    return xcals.get_calendar(name)


__all__ = [
    "bars_per_session",
    "calendar_available",
    "calendar_names",
    "interval_minutes",
    "periods_per_year",
    "require_calendar_library",
    "session_minutes",
    "sessions_per_year",
    "validate_calendar_name",
]
