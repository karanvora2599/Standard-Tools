"""
Data-quality checks on an already-fetched OHLCV frame: missing bars, stale
(frozen) prices, large single-bar jumps that may indicate an unadjusted
split/dividend or a data error, zero or thin volume, duplicate and
out-of-order timestamps, bars whose Open, High, Low and Close contradict
each other, and a frame whose provenance names a sample feed. All pure
functions operating on data the caller already has — no new data source or
provider required.

WHAT A STATISTIC ON THE FRAME CANNOT SEE. Every volume test here compares a
bar with the frame's own history, so it is blind to scale: a feed carrying
3.6% of the tape on every bar reads exactly like the tape. Whether a frame
came from a sample feed is answered by where it came from, not by its
numbers -- `detect_sample_feed` reads the dataset the provider stamped on
the frame.

Which sessions SHOULD have a bar is answered by an exchange calendar, not
by a weekday rule: `detect_missing_bars` takes a calendar code and asks
`exchange_calendars` for that exchange's sessions, so a market holiday is
not a gap. The calendar is an argument because it changes the answer — the
same 2024 US frame has no gaps under XNYS and nine under XCME, since the
two exchanges do not trade on the same days.

The weekday heuristic survives only as the fallback for an environment
without `exchange_calendars` installed, or for a code it does not
recognize. It flags every holiday, so every entry carries `basis`, which
says which rule judged it. Treat findings as leads to investigate, not
proven defects.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.data.databento import DATASET_CONSOLIDATED

logger = logging.getLogger(__name__)

#: Feeds known to carry a SAMPLE of the consolidated tape rather than all of
#: it, by the dataset name a provider stamps on its frames
#: (`attrs["dataset"]`), with what that means for the numbers.
SAMPLE_FEEDS: Dict[str, str] = {
    DATASET_CONSOLIDATED: (
        "a sample of the consolidated tape: its volume is 2-4% of "
        "consolidated volume, and its daily close is the last print of the "
        "UTC day, which is often an after-hours trade"
    ),
}


def _stamp(value: Any) -> str:
    """A bar label as text: the date for a midnight label, the full
    timestamp otherwise, so two intraday bars on one day stay distinct."""
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(ts):
        return "NaT"
    if ts == ts.normalize():
        return str(ts.date())
    return ts.isoformat()


def _calendar_sessions(name: str, start, end) -> "pd.DatetimeIndex | None":
    """The exchange's sessions in [start, end], or None without the library."""
    try:
        import exchange_calendars as xcals
    except ImportError:
        return None
    try:
        calendar = xcals.get_calendar(name)
        sessions = calendar.sessions_in_range(
            pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        )
    except Exception:  # noqa: BLE001 - an unknown code is a weekday fallback
        return None
    return pd.DatetimeIndex(sessions).tz_localize(None)


def detect_missing_bars(
    df: pd.DataFrame, calendar: str = "XNYS"
) -> List[Dict[str, Any]]:
    """
    Flag sessions between the first and last bar that have no row.

    Against the exchange calendar when `exchange_calendars` is present
    (`calendar` is its code; XNYS for US equities), so a holiday is not a
    gap: measured live, the weekday heuristic this used alone flagged 21
    'gaps' over 500 sessions, every one of them a holiday. Without the
    library it falls back to weekdays and each entry says so.

    Returns:
        List of {"date": iso date string, "weekday": name, "basis": "calendar" |
        "weekday"} for each gap, chronological order. Empty list if df has
        fewer than 2 rows.

    The span is the earliest bar to the latest, not the first row to the
    last: on an index out of order those differ, and the span between the
    first and last rows could be empty or cover the wrong dates.
    """
    if len(df) < 2:
        return []
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.dropna()
    if len(index) < 2:
        return []
    first, last = index.min(), index.max()
    expected = _calendar_sessions(calendar, first, last)
    basis = "calendar"
    if expected is None:
        expected = pd.bdate_range(first, last)
        basis = "weekday"
    actual = set(index.normalize())
    gaps = [d for d in expected if d not in actual]
    return [
        {"date": str(d.date()), "weekday": d.strftime("%A"), "basis": basis}
        for d in gaps
    ]


def detect_volume_anomalies(
    df: pd.DataFrame, window: int = 20, thin_fraction: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Bars whose volume is zero, or below `thin_fraction` of the trailing
    `window`-bar median.

    A bar is judged against the frame's OWN recent past, so this finds a
    bar that is thin next to its neighbours -- a halted or half session, a
    dropped print -- and cannot find a feed that is thin throughout. The
    test is scale-invariant: multiplying every volume by 0.036 gives the
    same answer, so a sample feed carrying 3.6% of the tape on every bar
    reads exactly like the tape. That question is answered by provenance,
    `detect_sample_feed`, not by any statistic on the frame.
    """
    if "Volume" not in df.columns or len(df) == 0:
        return []
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    trailing = volume.shift(1).rolling(window, min_periods=max(3, window // 2)).median()
    out: List[Dict[str, Any]] = []
    for stamp, value, median in zip(df.index, volume, trailing):
        if pd.isna(value):
            continue
        if value == 0:
            kind = "zero"
        elif pd.notna(median) and median > 0 and value < thin_fraction * median:
            kind = "thin"
        else:
            continue
        out.append(
            {
                "date": str(pd.Timestamp(stamp).date()),
                "volume": float(value),
                "trailing_median": float(median) if pd.notna(median) else None,
                "kind": kind,
            }
        )
    return out


def detect_stale_prices(df: pd.DataFrame, n: int = 3) -> List[Dict[str, Any]]:
    """
    Flag runs of n or more consecutive identical Close values — a likely
    stale/frozen quote (a real market rarely closes at the exact same price
    for multiple consecutive sessions).

    Args:
        df: OHLCV frame with a 'Close' column.
        n: Minimum run length to flag (default 3).

    Returns:
        List of {"start": iso date, "end": iso date, "price": float,
        "run_length": int}, one entry per qualifying run.
    """
    close = df["Close"]
    if len(close) == 0:
        return []

    runs: List[Dict[str, Any]] = []
    run_start = 0
    for i in range(1, len(close) + 1):
        changed = i == len(close) or close.iloc[i] != close.iloc[run_start]
        if changed:
            run_length = i - run_start
            if run_length >= n:
                runs.append(
                    {
                        "start": str(close.index[run_start].date()),
                        "end": str(close.index[i - 1].date()),
                        "price": float(close.iloc[run_start]),
                        "run_length": run_length,
                    }
                )
            run_start = i
    return runs


def detect_price_jumps(
    df: pd.DataFrame, threshold: float = 0.15
) -> List[Dict[str, Any]]:
    """
    Flag single-bar Close-to-Close moves exceeding threshold — a proxy for
    an unadjusted split/dividend or a data error, not a proven one (a
    genuinely volatile session produces the same signature).

    Args:
        df: OHLCV frame with a 'Close' column.
        threshold: Fractional move to flag (default 0.15 = 15%).

    Returns:
        List of {"date": iso date, "pct_change": float}, chronological
        order.
    """
    close = df["Close"]
    if len(close) < 2:
        return []
    pct_change = close.pct_change(fill_method=None)
    flagged = pct_change[pct_change.abs() > threshold]
    return [
        {"date": str(idx.date()), "pct_change": round(float(val), 4)}
        for idx, val in flagged.items()
    ]


def detect_sample_feed(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    The frame's provenance, when it names a feed that carries a sample of
    the tape rather than all of it; None otherwise.

    Reads `df.attrs["dataset"]`, which a provider that chooses among
    datasets stamps on every frame it returns (Databento does, including on
    a disk-cache hit), and looks it up in `SAMPLE_FEEDS`. This is the check
    `detect_volume_anomalies` cannot make: a feed thin on every bar is thin
    against nothing in its own history. A frame with no dataset stamp -- a
    provider with one feed, or bars built by hand -- gives None, which means
    "not known to be a sample", not "known to be the tape".

    Returns:
        {"dataset": name as stamped, "provider": attrs["provider"] or None,
        "note": what the feed is} or None.
    """
    dataset = getattr(df, "attrs", {}).get("dataset")
    if not isinstance(dataset, str):
        return None
    note = SAMPLE_FEEDS.get(dataset.strip().upper())
    if note is None:
        return None
    provider = df.attrs.get("provider")
    return {
        "dataset": dataset,
        "provider": provider if isinstance(provider, str) else None,
        "note": note,
    }


def detect_duplicate_timestamps(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Bar labels that occur more than once.

    Two rows under one timestamp are two answers to one question: a join or
    a `.loc` lookup silently returns both, a resample counts the bar twice,
    and every detector above that walks the rows in order sees a zero-length
    step. Nothing downstream refuses them, so they are reported here.

    Returns:
        List of {"timestamp": label, "count": occurrences, "positions":
        row positions}, one entry per repeated label, in order of first
        occurrence.
    """
    if len(df) < 2:
        return []
    index = pd.Index(df.index)
    repeated = index.duplicated(keep=False)
    if not repeated.any():
        return []
    positions: Dict[Any, List[int]] = {}
    for position in np.flatnonzero(repeated):
        positions.setdefault(index[position], []).append(int(position))
    return [
        {"timestamp": _stamp(label), "count": len(where), "positions": where}
        for label, where in positions.items()
    ]


def detect_out_of_order_timestamps(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Rows whose label is EARLIER than the row before it.

    Every rolling window, return and fill in this library reads the rows in
    order and assumes that order is time: two swapped bars turn two returns
    into their negatives' neighbours, and a gap check that reads the first
    and last rows as the span measures the wrong span. A repeated label is
    not out of order -- `detect_duplicate_timestamps` reports those.

    Returns:
        List of {"position": row, "timestamp": its label, "previous": the
        label of the row before it}, in row order.
    """
    if len(df) < 2:
        return []
    try:
        index = pd.DatetimeIndex(df.index)
    except (TypeError, ValueError):
        return []
    values = index.asi8
    valid = ~np.asarray(index.isna())
    backwards = (values[1:] < values[:-1]) & valid[1:] & valid[:-1]
    return [
        {
            "position": int(i) + 1,
            "timestamp": _stamp(index[int(i) + 1]),
            "previous": _stamp(index[int(i)]),
        }
        for i in np.flatnonzero(backwards)
    ]


def _column(df: pd.DataFrame, name: str) -> Optional[np.ndarray]:
    if name not in df.columns:
        return None
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype="float64")


def detect_ohlc_inconsistencies(
    df: pd.DataFrame, rel_tolerance: float = 1e-9
) -> List[Dict[str, Any]]:
    """
    Bars whose prices contradict each other: `Low` above `High`, or `Open`
    or `Close` outside `[Low, High]`.

    A bar's high and low bound every trade in it, the open and the close
    included, so any of these is a data error rather than a market event --
    and it is invisible to every Close-only check above. `rel_tolerance`
    (relative to the bar's price) absorbs floating-point noise in a price
    that was scaled or adjusted, so a bar where all four prices are equal is
    never flagged. A bar with Low above High is reported once, as that: its
    range is empty, so every open and close would also be "outside" it.
    Rows missing High or Low are skipped; a frame without both columns
    gives no findings.

    Returns:
        List of {"date": bar label, "position": row, "kind":
        "low_above_high" | "open_outside_range" | "close_outside_range",
        "open", "high", "low", "close"}, in row order.
    """
    high = _column(df, "High")
    low = _column(df, "Low")
    if high is None or low is None or len(df) == 0:
        return []
    open_ = _column(df, "Open")
    close = _column(df, "Close")
    scale = np.maximum(np.maximum(np.abs(high), np.abs(low)), 1.0)
    tol = rel_tolerance * scale
    known = ~(np.isnan(high) | np.isnan(low))
    inverted = known & (low > high + tol)

    def _outside(values: Optional[np.ndarray]) -> np.ndarray:
        if values is None:
            return np.zeros(len(df), dtype=bool)
        present = known & ~inverted & ~np.isnan(values)
        return present & ((values > high + tol) | (values < low - tol))

    kinds = (
        ("low_above_high", inverted),
        ("open_outside_range", _outside(open_)),
        ("close_outside_range", _outside(close)),
    )

    def _price(values: Optional[np.ndarray], i: int) -> Optional[float]:
        if values is None or np.isnan(values[i]):
            return None
        return float(values[i])

    found: List[Dict[str, Any]] = []
    for i in np.flatnonzero(inverted | kinds[1][1] | kinds[2][1]):
        i = int(i)
        for kind, mask in kinds:
            if mask[i]:
                found.append(
                    {
                        "date": _stamp(df.index[i]),
                        "position": i,
                        "kind": kind,
                        "open": _price(open_, i),
                        "high": float(high[i]),
                        "low": float(low[i]),
                        "close": _price(close, i),
                    }
                )
    return found
