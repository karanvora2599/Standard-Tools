"""
Data-quality checks on an already-fetched OHLCV frame: missing bars, stale
(frozen) prices, and large single-bar jumps that may indicate an
unadjusted split/dividend or a data error. All pure functions operating on
data the caller already has — no new data source or provider required.

Calendar-free heuristic, stated explicitly: detect_missing_bars infers
expected trading days from the data's own weekday pattern. It does not add
a market-holiday-calendar dependency, so U.S. market holidays will show up
as false-positive "gaps" — a documented limitation, not a silently hidden
one. Treat findings as leads to investigate, not proven defects.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def detect_missing_bars(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Flag weekday gaps in df's DatetimeIndex — dates between the first and
    last bar that fall on a weekday but have no row. Includes false
    positives for U.S. market holidays (Thanksgiving, Christmas, etc.) —
    this function has no holiday calendar, only a weekday heuristic.

    Returns:
        List of {"date": iso date string, "weekday": name} for each gap,
        chronological order. Empty list if df has fewer than 2 rows.
    """
    if len(df) < 2:
        return []
    expected = pd.bdate_range(df.index[0], df.index[-1])
    actual = set(df.index.normalize())
    gaps = [d for d in expected if d not in actual]
    return [{"date": str(d.date()), "weekday": d.strftime("%A")} for d in gaps]


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
