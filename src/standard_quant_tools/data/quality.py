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
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Which moves are named is one rule, `_split_screen.screen_moves`, read
# here and by the backtest's screen (`backtest.screens`). The tolerance and
# the ratio label live with it and are re-exported here under their names.
from standard_quant_tools._split_screen import (  # noqa: F401
    RATIOS_NAMED_BELOW_THRESHOLD,
    SPLIT_RATIO_TOLERANCE,
    screen_moves,
    split_ratio_label,
)
from standard_quant_tools.constants import SPLIT_SCREEN_THRESHOLD
from standard_quant_tools.data.databento import DATASET_CONSOLIDATED
from standard_quant_tools.error import ValidationError

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

    dtype = getattr(close, "dtype", None)
    if isinstance(dtype, np.dtype) and dtype.kind in "biuf":
        return _stale_runs(close, n)

    # A Close column numpy cannot compare as one array -- object, or a
    # nullable extension type -- keeps the comparison it always had.
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


def _stale_runs(close: pd.Series, n: int) -> List[Dict[str, Any]]:
    """
    `detect_stale_prices` for a numeric Close, as array passes.

    The run lengths come from the start of the run each bar belongs to,
    carried forward with `np.maximum.accumulate` -- the pattern
    `flat_window_mask` uses in `indicators.volatility`. A bar starts a run
    when it differs from the bar before it. The loop this replaces compared
    each bar with the FIRST bar of its run instead, which is the same test:
    equality between numbers that are not NaN is transitive, and NaN
    differs from everything, so every NaN is still a run of one. Only the
    qualifying runs touch pandas. The per-bar `iloc` loop measured 11.8 ms
    on a 2,115-bar frame.
    """
    values = close.to_numpy()
    position = np.arange(len(values))
    starts_run = np.ones(len(values), dtype=bool)
    starts_run[1:] = values[1:] != values[:-1]
    run_start = np.maximum.accumulate(np.where(starts_run, position, 0))
    ends_run = np.append(starts_run[1:], True)
    last = position[ends_run]
    first = run_start[ends_run]
    length = last - first + 1
    qualifying = length >= n
    return [
        {
            "start": str(close.index[start].date()),
            "end": str(close.index[end].date()),
            "price": float(values[start]),
            "run_length": run_length,
        }
        for start, end, run_length in zip(
            first[qualifying].tolist(),
            last[qualifying].tolist(),
            length[qualifying].tolist(),
        )
    ]


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


#: The split ratios a split-sized move is named after, as new shares per old
#: share: 3:2, 2:1, ... 50:1. A reverse split is the reciprocal (1:10 is
#: 0.1), so each of these is also tried the other way up.
SPLIT_RATIOS: Tuple[float, ...] = (
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    8.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    40.0,
    50.0,
)

# SPLIT_RATIO_TOLERANCE (0.10 on a log scale) and `split_ratio_label` are
# imported above from `_split_screen`, where the rule that reads them lives.


def nearest_split_ratio(factor: float) -> Tuple[float, float]:
    """
    The listed split ratio nearest `factor`, and how far it is.

    `factor` is the close before the bar over the close on it, so a 10:1
    split gives 10 and a 1:10 reverse split 0.1. Distance is
    |ln(factor / ratio)|, a proportion rather than a difference: 9.5 and
    10.5 are both about 5% from 10, as 1.9 and 2.1 are from 2.
    """
    if not (factor > 0 and math.isfinite(factor)):
        raise ValidationError(
            f"nearest_split_ratio: factor must be a positive finite number, "
            f"got {factor!r}."
        )
    candidates = SPLIT_RATIOS if factor >= 1.0 else tuple(1.0 / k for k in SPLIT_RATIOS)
    log_factor = math.log(factor)
    ratio = min(candidates, key=lambda k: abs(log_factor - math.log(k)))
    return float(ratio), abs(log_factor - math.log(ratio))


def split_like_moves_at(
    close: pd.Series, threshold: float = SPLIT_SCREEN_THRESHOLD
) -> List[Tuple[int, float, Optional[float], Optional[float]]]:
    """
    The bars `detect_split_like_moves` reports, by POSITION in `close`:
    (position, close_move, split_ratio or None, ratio_error or None).

    The form a caller needs to count what reads each bar -- a dataset
    build counts the labels and feature rows within reach of each one --
    and the single definition of which bars are screened, so the reported
    form cannot drift from it.
    """
    if not (isinstance(threshold, (int, float)) and math.isfinite(threshold)):
        raise ValidationError(
            f"detect_split_like_moves: threshold must be a finite number, got "
            f"{threshold!r}."
        )
    if threshold <= 0:
        raise ValidationError(
            f"detect_split_like_moves: threshold must be positive, got "
            f"{threshold!r}; it is the size of move (0.35 = 35%) to screen."
        )
    values = pd.to_numeric(pd.Series(close), errors="coerce").to_numpy(dtype=float)
    if len(values) < 2:
        return []
    before = values[:-1]
    after = values[1:]
    screened = screen_moves(values, threshold)
    moves = screened.moves
    flagged = np.flatnonzero(screened.flagged)
    out: List[Tuple[int, float, Optional[float], Optional[float]]] = []
    for i in flagged:
        move = float(moves[i])
        prior, current = float(before[i]), float(after[i])
        ratio: Optional[float] = None
        error: Optional[float] = None
        if prior > 0 and current > 0:
            nearest, distance = nearest_split_ratio(prior / current)
            error = distance
            if distance <= SPLIT_RATIO_TOLERANCE:
                ratio = nearest
        out.append((int(i) + 1, move, ratio, error))
    return out


def detect_split_like_moves(
    close: "pd.Series | pd.DataFrame", threshold: float = SPLIT_SCREEN_THRESHOLD
) -> List[Dict[str, Any]]:
    """
    Close-to-close moves large enough to be an unadjusted split, each named
    after the split ratio it is consistent with.

    The screen the backtest runs (`backtest.screens`, one rule in
    `_split_screen`), in the form a dataset build records: a move beyond
    `threshold` is listed, and so is a fall within 10% (on a log scale) of
    a 3:2 split however small -- 26.3% to 39.7%, so a 3:2 split's -33% is
    listed under the default 35%. When the price ratio across a listed bar
    is within 10% of a listed split ratio -- 3:2, 2:1, 3:1, 4:1, 5:1, 8:1,
    10:1, 15:1, 20:1, 25:1, 30:1, 40:1, 50:1, or any of them reversed --
    that ratio is named. Being named is consistency, not proof: a genuine
    -90% day reads exactly like a 10:1 split, and a genuine -30% day like a
    3:2 split.

    Args:
        close: Close prices in bar order (a frame's 'Close' column is used
            when a frame is passed).
        threshold: The fractional move to screen (default 0.35 = 35%).

    Returns:
        [{"date", "close_move", "split_ratio", "ratio_error"}] in bar order.
        `split_ratio` is new shares per old share (10.0 for a 10:1 split,
        0.1 for a 1:10 reverse split) or None when no listed ratio is near;
        `ratio_error` is |ln(implied / nearest listed ratio)|, None only for
        a non-positive price.
    """
    if isinstance(close, pd.DataFrame):
        close = close["Close"]
    index = close.index
    return [
        {
            "date": _stamp(index[position]),
            "close_move": round(move, 4),
            "split_ratio": ratio,
            "ratio_error": None if error is None else round(error, 4),
        }
        for position, move, ratio, error in split_like_moves_at(close, threshold)
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
