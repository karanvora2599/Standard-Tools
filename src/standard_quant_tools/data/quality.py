"""
Data-quality checks on an already-fetched OHLCV frame: missing bars, stale
(frozen) prices, and large single-bar jumps that may indicate an
unadjusted split/dividend or a data error. All pure functions operating on
data the caller already has — no new data source or provider required.

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
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


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
    """
    if len(df) < 2:
        return []
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    expected = _calendar_sessions(calendar, index[0], index[-1])
    basis = "calendar"
    if expected is None:
        expected = pd.bdate_range(index[0], index[-1])
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

    This module never read `Volume`, so it gave identical verdicts on a
    frame carrying 3% of the tape and on the real tape (findings §4). A
    thin bar is not proof of a sample feed, but a run of them is the
    signature, and it is the caller's to read.
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
