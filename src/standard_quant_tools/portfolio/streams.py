"""
Dated return streams, put on one calendar.

WHY DATES ARE THE INPUT. Two return series of the same length look aligned
and need not be: one a fortnight later than the other, one with another
market's holidays, one a monthly fund beside a daily strategy. Read row by
row they pair each day with some other day, and every covariance,
correlation, Sharpe of a difference or bootstrap built on them is computed
on a calendar that does not exist. A stream here is a return series keyed by
DATE, and streams meet only on the dates they share. Undated values are not
a stream: there is nothing to check their alignment against.

THE RULES, one function for every caller. `align_return_streams` keeps the
dates every stream has, refuses what cannot be returns or cannot be placed
in time, and reports per stream what it supplied and what the common
calendar cost it -- before the window they all cover, after it, and inside
it on a date some other stream lacks. Refused, each by name:

- an overlap shorter than `min_overlap`, naming the streams that limit it
  (the one whose removal would share the most dates, and the streams that
  set the window's start and end);
- a date that appears twice in one stream;
- a missing or non-finite value;
- a stream whose typical |value| is that of a price or an equity curve
  rather than of a return (`refuse_if_not_returns`);
- undated values, and dates that mix time zones.

`streams_from_panel` reads a wide returns frame as streams, a blank cell
being a date that column was not observed on. `combine_return_streams`
resolves streams from wherever they are, aligns them and publishes the
aligned panel as a `returns_panel` reference.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

__all__ = [
    "MAX_PLAUSIBLE_TYPICAL_RETURN",
    "align_return_streams",
    "alignment_notes",
    "combine_return_streams",
    "refuse_if_not_returns",
    "streams_from_panel",
]

#: Typical |value| above this is not a fractional return. An equity curve
#: at any base -- 10 000, 100, or normalized to 1.0 -- clears it; a daily
#: return series sits five or six orders of magnitude below it. Median
#: rather than max, so one genuine +200% day does not trip it.
MAX_PLAUSIBLE_TYPICAL_RETURN = 1.0

#: How many offending names or dates a refusal lists before counting the rest.
_SHOWN = 5


def refuse_if_not_returns(series: Any, what: str) -> None:
    """Refuse a series that cannot be returns, naming the likely cause.

    Deliberately ONE rule, and a blunt one. The tempting second rule --
    refuse a series with no negative observation -- would refuse a
    money-market or T-bill return series, which really is all-positive and
    really does have an enormous Sharpe against a 0% rate. Being unable to
    measure a real series is worse than the narrower guard, so the softer
    signal lives where there is a warnings channel to carry it.
    """
    finite = series[series.notna()]
    if finite.empty:
        return
    typical = float(finite.abs().median())
    if typical > MAX_PLAUSIBLE_TYPICAL_RETURN:
        raise ValidationError(
            f"{what}: this is a RETURN series, and the typical |value| here "
            f"is {typical:.4g} -- a {typical * 100:.0f}% move per period. "
            "That is a level series (an equity curve, or prices), not "
            "returns. A Sharpe taken on levels is scale-invariant and comes "
            "back two to three orders of magnitude too high with nothing "
            "looking wrong. If you passed values, convert them first with "
            "`.pct_change().dropna()`. If you passed a REFERENCE -- which "
            "you cannot call pandas on -- run convert_reference from "
            "`equity_curve` to `returns_panel`, which drops the first bar "
            "rather than carrying it as a fabricated flat period, and hand "
            "the reference it returns to this tool instead."
        )


def _listed(items: Sequence[Any]) -> str:
    shown = ", ".join(repr(i) for i in list(items)[:_SHOWN])
    more = len(items) - _SHOWN
    return f"[{shown}{f' and {more} more' if more > 0 else ''}]"


def _iso(stamp: pd.Timestamp) -> str:
    """A date as a caller writes it: 'YYYY-MM-DD' unless it has a time."""
    return str(stamp.date()) if stamp == stamp.normalize() else stamp.isoformat()


def _checked_stream(name: str, series: Any, what: str) -> pd.Series:
    """One stream as float64 on a sorted DatetimeIndex, or a refusal."""
    label = f"{what}: stream {name!r}"
    if not isinstance(series, pd.Series):
        raise ValidationError(
            f"{label} is a {type(series).__name__}, not a dated series. "
            "Undated values are not a stream: two equal-length lists from "
            "different calendars look aligned and are not, and nothing here "
            "could tell. Give a pandas Series on a DatetimeIndex, or "
            "{ISO date: return}."
        )
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValidationError(
            f"{label} is indexed by {type(series.index).__name__}, not by "
            f"date (first labels {_listed(list(series.index[:3]))}). Undated "
            "values are not a stream: positions cannot be checked against "
            "another stream's calendar. Index it by date."
        )
    if series.empty:
        raise ValidationError(
            f"{label} has no observations; a stream needs at least one dated " "return."
        )
    if series.index.hasnans:
        raise ValidationError(f"{label} has a missing date (NaT) in its index.")
    if series.index.has_duplicates:
        repeated = series.index[series.index.duplicated()].unique()
        raise ValidationError(
            f"{label} names the same date twice: {_listed([_iso(d) for d in repeated])}. "
            "Only one return per date can be used, and keeping either would "
            "be a guess. Keep the one you mean."
        )
    try:
        values = series.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{label} holds values that are not numbers ({exc})."
        ) from exc
    bad = ~np.isfinite(values.to_numpy())
    if bad.any():
        dates = [_iso(d) for d in values.index[bad]]
        raise ValidationError(
            f"{label} has a missing or non-finite value on {int(bad.sum())} "
            f"date(s): {_listed(dates)}. A gap is a date the stream does not "
            "have, not a NaN on it, and an infinity is not a return. Drop "
            "those dates or supply the value."
        )
    refuse_if_not_returns(values, label)
    return values.sort_index()


def align_return_streams(
    streams: Mapping[str, pd.Series],
    *,
    min_overlap: int,
    what: str = "align_return_streams",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    The streams on the dates they all share, and what that cost each.

    `streams` maps a name to a dated return Series. The frame has one column
    per stream, in the mapping's order, and one row per shared date, in
    date order; its values are the streams' own, bit for bit.

    The report:

    - `n_streams`, `min_overlap`, `n_dates` (rows kept), `n_dates_any`
      (dates at least one stream has), `first_date` and `last_date` of the
      rows kept;
    - `window_start` / `window_end`: the window every stream covers (the
      latest first date, the earliest last date), and `start_set_by` /
      `end_set_by`, the streams that set them;
    - `by_stream`: per stream, `n_supplied`, `first_date`, `last_date`,
      `n_kept`, and `n_dropped_before`, `n_dropped_after` and
      `n_dropped_inside` -- returns before the window, after it, and inside
      it on a date some other stream lacks. They sum to `n_supplied`.

    Refused by name: no streams; `min_overlap` below 1; a stream that is
    not a Series on a DatetimeIndex (undated values are not a stream);
    streams mixing time zones; an empty stream; a repeated date; a missing
    or non-finite value; a level series passed as returns; and fewer shared
    dates than `min_overlap`, naming the streams that limit it.
    """
    if isinstance(min_overlap, bool) or not isinstance(min_overlap, (int, np.integer)):
        raise ValidationError(
            f"{what}: min_overlap must be a whole number of shared dates, got "
            f"{min_overlap!r}."
        )
    if int(min_overlap) < 1:
        raise ValidationError(
            f"{what}: min_overlap={min_overlap} asks for no shared date at all; "
            "it is the fewest dates every stream must share, at least 1."
        )
    if not isinstance(streams, Mapping) or not streams:
        raise ValidationError(
            f"{what}: no streams were given. Pass a mapping of name to dated "
            "return series."
        )
    names = [str(name) for name in streams]
    if len(set(names)) != len(names):
        repeated = sorted({n for n in names if names.count(n) > 1})
        raise ValidationError(
            f"{what}: two streams are named {_listed(repeated)}; each name is a "
            "column of the aligned panel, so each must be distinct."
        )
    checked: Dict[str, pd.Series] = {
        str(name): _checked_stream(str(name), series, what)
        for name, series in streams.items()
    }

    zones = {
        name: str(s.index.tz) if s.index.tz else None for name, s in checked.items()
    }
    if len(set(zones.values())) > 1:
        raise ValidationError(
            f"{what}: the streams' dates are in different time zones "
            f"({zones}), so a date in one is not the same date in another. "
            "Give every stream's dates the same way, e.g. all as plain "
            "'YYYY-MM-DD' dates."
        )

    firsts = {name: s.index[0] for name, s in checked.items()}
    lasts = {name: s.index[-1] for name, s in checked.items()}
    window_start = max(firsts.values())
    window_end = min(lasts.values())

    common = None
    seen = None
    for series in checked.values():
        common = series.index if common is None else common.intersection(series.index)
        seen = series.index if seen is None else seen.union(series.index)
    common = common.sort_values()

    if len(common) < int(min_overlap):
        raise ValidationError(
            _short_overlap(
                checked, common, int(min_overlap), window_start, window_end, what
            )
        )

    frame = pd.DataFrame(
        {name: series.reindex(common) for name, series in checked.items()},
        index=common,
    )
    frame.index.name = "date"

    by_stream: Dict[str, Dict[str, Any]] = {}
    n_kept = int(len(common))
    for name, series in checked.items():
        index = series.index
        before = int((index < window_start).sum())
        after = int((index > window_end).sum())
        by_stream[name] = {
            "n_supplied": int(len(series)),
            "first_date": _iso(index[0]),
            "last_date": _iso(index[-1]),
            "n_kept": n_kept,
            "n_dropped_before": before,
            "n_dropped_after": after,
            "n_dropped_inside": int(len(series)) - n_kept - before - after,
        }
    report = {
        "n_streams": len(checked),
        "min_overlap": int(min_overlap),
        "n_dates": n_kept,
        "n_dates_any": int(len(seen)),
        "first_date": _iso(common[0]),
        "last_date": _iso(common[-1]),
        "window_start": _iso(window_start),
        "window_end": _iso(window_end),
        "start_set_by": [n for n, d in firsts.items() if d == window_start],
        "end_set_by": [n for n, d in lasts.items() if d == window_end],
        "by_stream": by_stream,
    }
    return frame, report


def _short_overlap(
    checked: Dict[str, pd.Series],
    common: pd.Index,
    min_overlap: int,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    what: str,
) -> str:
    """The refusal for too few shared dates, naming what limits them."""
    starters = [n for n, s in checked.items() if s.index[0] == window_start]
    enders = [n for n, s in checked.items() if s.index[-1] == window_end]
    if len(checked) == 1:
        (name,) = checked
        return (
            f"{what}: stream {name!r} has {len(common)} dated return(s), fewer "
            f"than min_overlap={min_overlap}. Supply a longer stream."
        )
    # Which single stream, left out, would leave the most dates shared: that
    # is the one holding the overlap down.
    without: Dict[str, int] = {}
    for left_out in checked:
        shared = None
        for name, series in checked.items():
            if name == left_out:
                continue
            shared = (
                series.index if shared is None else shared.intersection(series.index)
            )
        without[left_out] = int(len(shared))
    best = max(without.values())
    limiting = [
        n for n, count in without.items() if count == best and count > len(common)
    ]
    if window_start > window_end:
        span = (
            f"They cover no common window at all: {_listed(enders)} end(s) on "
            f"{_iso(window_end)}, before {_listed(starters)} start(s) on "
            f"{_iso(window_start)}."
        )
    else:
        span = (
            f"The window they all cover runs {_iso(window_start)} to "
            f"{_iso(window_end)}, its start set by {_listed(starters)} and its "
            f"end by {_listed(enders)}."
        )
    if limiting:
        ranges = "; ".join(
            f"{n!r} has {len(checked[n])} returns, {_iso(checked[n].index[0])} to "
            f"{_iso(checked[n].index[-1])}"
            for n in limiting[:_SHOWN]
        )
        culprit = (
            f" {_listed(limiting)} limit(s) it most: without it {best} dates "
            f"would be shared ({ranges})."
        )
    else:
        culprit = (
            " No single stream limits it; the streams' dates barely coincide "
            "however one is left out."
        )
    return (
        f"{what}: the streams share {len(common)} date(s), fewer than "
        f"min_overlap={min_overlap}. {span}{culprit} Give streams that cover a "
        "common period, leave the limiting one out, or lower min_overlap if "
        f"{len(common)} observations are enough."
    )


def alignment_notes(report: Mapping[str, Any], what: str) -> List[str]:
    """One sentence naming what the common calendar dropped, or nothing."""
    dropped = {
        name: row["n_supplied"] - row["n_kept"]
        for name, row in report["by_stream"].items()
        if row["n_supplied"] != row["n_kept"]
    }
    if not dropped:
        return []
    return [
        f"{what}: the streams were aligned on the {report['n_dates']} dates "
        f"all {report['n_streams']} share ({report['first_date']} to "
        f"{report['last_date']}), which dropped {dropped} return(s) from the "
        "streams named -- before, after or inside that window as the "
        "alignment report says. Nothing was filled."
    ]


def streams_from_panel(
    frame: pd.DataFrame,
    *,
    what: str,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, pd.Series]:
    """
    The columns of a wide returns frame as dated streams.

    A blank cell is a date that column was not observed on, so it is left
    out of that column's stream rather than carried as a missing return --
    a panel is rectangular, and a blank is how it says a name had no
    return there. `columns` picks and orders the streams; by default every
    column, in the frame's order.

    The index must be dates: a DatetimeIndex, or labels that parse as ISO
    dates (or date objects). A positional index is refused, because
    undated values are not a stream.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError(
            f"{what}: expected a returns panel (a date-indexed frame), got a "
            f"{type(frame).__name__}."
        )
    labels = [str(c) for c in frame.columns]
    if len(set(labels)) != len(labels):
        repeated = sorted({c for c in labels if labels.count(c) > 1})
        raise ValidationError(
            f"{what}: the panel names column(s) {_listed(repeated)} twice, so "
            "a column name does not pick one stream."
        )
    index = _dated_index(frame.index, what)
    if columns is None:
        chosen = labels
    else:
        chosen = [str(c) for c in columns]
        missing = [c for c in chosen if c not in labels]
        if missing:
            raise ValidationError(
                f"{what}: the panel has no column {_listed(missing)}; its "
                f"columns are {_listed(labels)}."
            )
    streams: Dict[str, pd.Series] = {}
    for name in chosen:
        column = frame.iloc[:, labels.index(name)]
        series = pd.Series(column.to_numpy(), index=index, name=name)
        try:
            series = series.astype("float64")
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"{what}: column {name!r} holds values that are not numbers ({exc})."
            ) from exc
        streams[name] = series[series.notna()]
    return streams


def _dated_index(index: pd.Index, what: str) -> pd.DatetimeIndex:
    if isinstance(index, pd.DatetimeIndex):
        return index
    undated = (
        f"{what}: the panel is indexed by {type(index).__name__} "
        f"(first labels {_listed(list(index[:3]))}), not by date. Undated "
        "values are not a stream: rows cannot be checked against another "
        "stream's calendar. Publish the panel with its dates as the index."
    )
    # Only labels that ARE dates -- ISO strings, or date and datetime
    # objects -- are read as dates. A number would convert too, to an
    # instant counted from 1970, which is a position wearing a date.
    labels = list(index)
    try:
        if labels and all(isinstance(label, str) for label in labels):
            parsed = pd.to_datetime(index, format="ISO8601")
        elif labels and all(
            isinstance(label, (_dt.date, np.datetime64)) for label in labels
        ):
            parsed = pd.to_datetime(index)
        else:
            raise ValidationError(undated)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValidationError):
            raise
        raise ValidationError(undated) from exc
    if not isinstance(parsed, pd.DatetimeIndex):
        raise ValidationError(undated)
    return parsed


def combine_return_streams(
    streams: Mapping[str, Any],
    *,
    run_id: str,
    min_overlap: int,
    name: str = "streams",
    what: str = "combine_return_streams",
) -> Dict[str, Any]:
    """
    Resolve streams, align them on their shared dates, and publish the panel.

    Each value of `streams` is a dated return Series, or a stream spec --
    a `ReturnStream` (or the dict of one) from the agent layer: a
    `returns_panel` reference with its `column`, `{ISO date: return}`, or a
    symbol with an explicit window. The aligned frame is published as a
    `returns_panel` at (`run_id`, `name`), which refuses a pair already
    published as every publisher does.

    Returns the reference (`returns_ref`), the column names in order, the
    alignment report of `align_return_streams` and any notes on what the
    alignment dropped.
    """
    from standard_quant_tools.agent.runtimes import handoff

    if not isinstance(streams, Mapping) or not streams:
        raise ValidationError(
            f"{what}: no streams were given. Pass a mapping of name to a dated "
            "return series or a stream spec."
        )
    series: Dict[str, pd.Series] = {}
    for key, value in streams.items():
        series[str(key)] = (
            value if isinstance(value, pd.Series) else _resolved(str(key), value, what)
        )
    frame, report = align_return_streams(series, min_overlap=min_overlap, what=what)
    ref = handoff.publish(
        frame, kind="returns_panel", run_id=run_id, name=name, producer=what
    )
    return {
        "returns_ref": ref,
        "columns": [str(c) for c in frame.columns],
        "alignment": report,
        "warnings": alignment_notes(report, what),
    }


def _resolved(name: str, spec: Any, what: str) -> pd.Series:
    """One stream spec resolved through the agent layer's input model."""
    from standard_quant_tools.agent.runtimes.streams import (
        ReturnStream,
        resolve_return_streams,
    )

    stream = (
        spec if isinstance(spec, ReturnStream) else ReturnStream.model_validate(spec)
    )
    return resolve_return_streams({name: stream}, what=what)[name]
