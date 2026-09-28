"""
Bars per year: the one number every annualized metric multiplies by.

A Sharpe ratio, a volatility, a CAGR and a Calmar ratio are all computed per
bar and scaled by how many bars make a year. Get that number wrong and every
one of them is wrong by a fixed factor while still looking precise: weekly
bars annualized as though they were days report a volatility sqrt(252/52) =
2.2x too high, and monthly bars read as days put ten years of history into
half a year, so the CAGR (and every Calmar built on it) is off by far more.

This module is where that number is decided, for the backtest engine and
the modeling layer alike. It lives under `metrics` because both of them sit
on top of it; `modeling.features.base` re-exports the interval table so its
existing importers are untouched.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.numeric_contract import require_periods_per_year

#: What every annualized metric assumed before it could be told otherwise:
#: US equity trading days. Used only when nothing better can be resolved,
#: and never silently.
DEFAULT_PERIODS_PER_YEAR = 252

# Bars per year, by interval, for annualizing a per-bar quantity.
#
# Only intervals whose constant is unambiguous are listed. Daily, weekly and
# monthly are calendar-derived and need no assumption about session length.
# INTRADAY IS DELIBERATELY ABSENT: bars-per-year at "1h" depends on how many
# trading hours the venue is open (6.5 for US equities, 8.5 for the LSE,
# ~24 for crypto), and only an exchange calendar can say which. Picking one
# silently would make an "annualized" volatility wrong by a fixed
# multiplicative factor for every other market -- a number that looks
# precise and is not. With a calendar named, `modeling.calendar` reads the
# session length and the sessions per year off it.
_PERIODS_PER_YEAR = {
    "1d": 252,
    "5d": 52,
    "1wk": 52,
    "1mo": 12,
    "3mo": 4,
}

# Median bar spacing, in days, to bars per year. BUCKETED, not divided:
# 365.25 / median spacing gives 365 for business-day bars (their median gap
# is one calendar day; the weekend only moves the mean), which is the wrong
# answer for the most common input there is.
_SPACING_BUCKETS = (
    (0.9, 4.0, 252),
    (5.0, 8.0, 52),
    (26.0, 33.0, 12),
    (85.0, 95.0, 4),
)

# Daily bars with no weekend gaps run about 365 to a calendar year; trading
# days run 252-261. Above this, daily spacing is a 24/7 market, and 252 is
# probably not its year.
_CONTINUOUS_BARS_PER_YEAR = 300.0


def periods_per_year_for_interval(
    interval: str, calendar: Optional[str] = None
) -> Optional[int]:
    """
    Bars per year for `interval`.

    A daily-or-coarser interval is a constant. An intraday interval is
    bars per session times sessions per year, both read off the named
    exchange calendar, and None without one -- the caller then refuses or
    warns rather than assuming a venue.
    """
    known = _PERIODS_PER_YEAR.get(str(interval).strip())
    if known is not None:
        return known
    if calendar is None:
        return None
    # Imported here, and only for an intraday interval with a calendar: the
    # calendar module is the modeling layer's, and nothing on the daily path
    # should have to load it.
    from standard_quant_tools.modeling.calendar import (
        interval_minutes,
        periods_per_year,
    )

    if interval_minutes(interval) is None:
        return None
    return periods_per_year(interval, calendar)


def infer_periods_per_year(index: pd.Index) -> Tuple[Optional[int], List[str]]:
    """
    Bars per year read off a sorted, unique date index's median spacing.

    Returns (value, notes). The value is None when the spacing does not
    identify a year -- intraday bars (which need a venue's session length),
    irregular spacing, too few bars, or an index that is not dates. The
    notes say why, or qualify an answer that was found.
    """
    if not isinstance(index, pd.DatetimeIndex):
        return None, [
            "the index is not a DatetimeIndex, so its bar spacing says nothing "
            "about how many bars make a year"
        ]
    if len(index) < 2:
        return None, ["fewer than two bars, so there is no bar spacing to read"]
    # Through Timedelta division rather than the raw integers, which are in
    # whatever unit (s, ms, us, ns) the index happens to be stored in.
    gaps = np.asarray((index[1:] - index[:-1]) / pd.Timedelta(days=1), dtype=float)
    median_days = float(np.median(gaps))
    if not math.isfinite(median_days) or median_days <= 0:
        return None, ["the bar spacing is not positive"]
    if median_days < 0.9:
        return None, [
            f"the bars are intraday (median spacing {median_days * 24:.2f} "
            "hours), and bars per year at an intraday interval depends on the "
            "venue's session length, which the bars do not carry"
        ]
    for low, high, value in _SPACING_BUCKETS:
        if low <= median_days <= high:
            notes: List[str] = []
            span_years = (index[-1] - index[0]).days / 365.25
            if value == 252 and span_years >= 28 / 365.25:
                bars_per_calendar_year = (len(index) - 1) / span_years
                if bars_per_calendar_year > _CONTINUOUS_BARS_PER_YEAR:
                    notes.append(
                        f"the daily bars have no weekend gaps (about "
                        f"{bars_per_calendar_year:.0f} a calendar year), which "
                        "is a market that trades every day; 252 trading days "
                        "is probably not its year -- pass periods_per_year="
                        "365 if it is"
                    )
            return value, notes
    return None, [
        f"the median bar spacing ({median_days:.2f} days) is not daily, "
        "weekly, monthly or quarterly"
    ]


def resolve_periods_per_year(
    index: pd.Index,
    *,
    periods_per_year: Any = None,
    interval: Optional[str] = None,
    where: str = "run_strategy",
) -> Tuple[int, str, List[str]]:
    """
    Decide bars per year for a backtest over `index`, and say how.

    Order: an explicit `periods_per_year` (validated: a positive whole
    number, not a bool); else the `interval` the bars were fetched at, from
    the table above; else the spacing of `index`, which must already be
    sorted and unique; else 252, with a warning naming the two parameters
    that would settle it.

    Returns (value, source, warnings), source one of "explicit",
    "interval", "inferred" or "default". A caller that passes the interval
    it fetched gets no warning on a holiday-gapped daily index; one that
    passes nothing gets the spacing's answer, and a warning whenever that
    answer is a guess.
    """
    if periods_per_year is not None:
        return require_periods_per_year(periods_per_year, where), "explicit", []

    warnings: List[str] = []
    if interval is not None:
        from_interval = periods_per_year_for_interval(interval)
        if from_interval is not None:
            inferred, _ = infer_periods_per_year(index)
            if inferred is not None and inferred != from_interval:
                warnings.append(
                    f"{where}: interval={interval!r} means {from_interval} bars "
                    f"a year, but the bars are spaced like {inferred} a year. "
                    f"Annualized with {from_interval}, as asked; check that the "
                    "interval matches the data."
                )
            return from_interval, "interval", warnings
        reason = (
            f"interval={interval!r} has no fixed number of bars per year "
            "without an exchange calendar"
        )
    else:
        inferred, notes = infer_periods_per_year(index)
        if inferred is not None:
            warnings.extend(f"{where}: {note}." for note in notes)
            return inferred, "inferred", warnings
        reason = notes[0] if notes else "the bar spacing could not be read"

    warnings.append(
        f"{where}: annualized with {DEFAULT_PERIODS_PER_YEAR} bars a year "
        f"because {reason}. Volatility, Sharpe, Sortino, CAGR and Calmar are "
        "wrong by a fixed factor if that is not this data's year; pass "
        "periods_per_year= (or the interval= the bars were fetched at)."
    )
    return DEFAULT_PERIODS_PER_YEAR, "default", warnings
