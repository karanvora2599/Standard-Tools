"""
What an ORDER-level feed says that a depth book cannot.

THE DIFFERENCE THIS MODULE EXISTS FOR. `order_book.py` reads snapshots: at
this instant, what is resting and where. That is aggregated size per price
level, and aggregation destroys exactly the facts below. A book showing
5,000 shares at the bid cannot say whether that is one order or two hundred,
which of them arrived first, or how many were cancelled in the second before
you looked. An order feed can, because every add, cancel, modify and fill is
its own record with its own identifier.

WHAT THAT BUYS, and why each was out of reach before:

  QUEUE POSITION. How much size sits ahead of an order at its own price
  level when it arrives -- the single number that decides whether a passive
  order fills. Aggregated depth gives the total at that level and cannot
  say how much of it is in front of you.

  CANCELLATION RATE. Cancels per add, and cancels per trade. A book
  snapshot shows size appearing and disappearing; it cannot distinguish a
  cancel from a fill, and those mean opposite things about who wanted to
  trade.

  EVENT INTENSITY. How fast the book is being worked, per action. A
  snapshot stream measures the SAMPLING rate when it is sampled and the
  update rate when it is not, and nothing in the frame says which.

  ORDER LIFETIME. How long an order rests before it is cancelled or
  filled. There is no snapshot equivalent at all.

WHAT IS COUNTED AND WHAT IS NOT. An order resting before the window opened
has no ADD in it, so its cancel or fill has no measurable lifetime and no
measurable queue position. Those are counted SEPARATELY rather than folded
in as zero -- a left-censored order treated as instantaneous would make
every average lifetime shorter than the truth, and the bias is largest for
exactly the resting orders a queue study is about. The same orders sit
ahead of every early arrival, so a window with no snapshot reports its queue
figures as LOWER BOUNDS and counts the size it never saw.

A CLEAR (`R`) wipes the book. The accumulators reset on one rather than
carrying a stale level into the next session, which would report queue
depth accumulated across a boundary where none existed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

# Optional native fast path for the two per-event loops below, the queue
# pass and the lifetime pass: 94% of `order_event_metrics` on a
# 2,000,000-event session in Python. The loops stay as the reference and
# the fallback.
_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = hasattr(_cpp_core, "order_queue_ahead") and hasattr(
        _cpp_core, "order_lifetimes"
    )
except ImportError:
    pass

#: The canonical order-event columns. Named here because this module reads
#: them and `DataProvider.get_order_events` declares them, the same split
#: `order_book.py` and `get_order_book` already use.
ORDER_EVENT_COLUMNS = ("timestamp", "order_id", "action", "side", "price", "size")

#: Databento's action vocabulary, which is the venue's. Spelled out because
#: the letters are not self-evident and the difference between C and F is
#: the whole point of this module.
ADD = "A"
CANCEL = "C"
MODIFY = "M"
FILL = "F"
TRADE = "T"
CLEAR = "R"

ACTION_MEANINGS: Dict[str, str] = {
    ADD: "a new order joins the book",
    CANCEL: "an order is withdrawn by whoever posted it",
    MODIFY: "an order's price or size changes",
    FILL: "a resting order is executed against",
    TRADE: "a trade print, which may not correspond to a resting order",
    CLEAR: "the book is wiped, e.g. at a session boundary",
}

BID = "B"
ASK = "A"


def _require(frame: pd.DataFrame, name: str = "order events") -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValidationError(f"{name}: expected a non-empty DataFrame of events.")
    missing = [c for c in ORDER_EVENT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"{name}: missing column(s) {missing}. An order-event frame needs "
            f"{list(ORDER_EVENT_COLUMNS)} -- `action` and `order_id` are the "
            "two that make it an order feed rather than a book, and without "
            "them every measure here is a depth measure wearing a different "
            "name."
        )
    return frame


#: Databento's snapshot bit: the record repeats state (the book at the
#: window's open) rather than reporting a change.
F_SNAPSHOT = 32


def _snapshot_mask(frame: pd.DataFrame) -> np.ndarray:
    """
    Which rows are snapshot records: a `snapshot` column when the caller
    supplied one, else the vendor's flag bit, else none. A snapshot row
    is the book as it stood, not an event: it must seed the queue and
    explain a later cancel, and it must not count as an arrival, a rate
    or a clock tick (findings: 54.5% of 'terminated without an add' were
    snapshot orders; events_per_second was off by 16,000x).
    """
    if "snapshot" in frame.columns:
        return frame["snapshot"].fillna(False).astype(bool).to_numpy()
    if "flags" in frame.columns:
        flags = pd.to_numeric(frame["flags"], errors="coerce").fillna(0).astype("int64")
        return ((flags & F_SNAPSHOT) != 0).to_numpy()
    return np.zeros(len(frame), dtype=bool)


def _is_datetime(dtype: Any) -> bool:
    return isinstance(dtype, pd.DatetimeTZDtype) or (
        isinstance(dtype, np.dtype) and dtype.kind == "M"
    )


def _as_datetimes(stamps: pd.Series) -> pd.Series:
    """
    `pd.to_datetime(stamps, errors="coerce")`, without the pass it makes
    over a column that already holds datetimes: it returns the same dtype
    and the same values (checked on pandas 2.3 and 3.0, every unit, naive
    and zoned), after 0.4 s on 2,000,000 zoned nanosecond stamps.
    """
    if _is_datetime(stamps.dtype):
        return stamps
    return pd.to_datetime(stamps, errors="coerce")


def _elapsed_seconds(stamps: pd.Series) -> Optional[float]:
    valid = _as_datetimes(stamps).dropna()
    if len(valid) < 2:
        return None
    span = (valid.max() - valid.min()).total_seconds()
    return float(span) if span > 0 else None


def queue_positions(events: pd.DataFrame) -> Dict[str, Any]:
    """
    Size resting ahead of each new order, at its own price level.

    ONE PASS, NOT A BOOK REBUILD. A running total per (side, price) is all
    the queue-ahead figure needs: when an order is added, whatever the
    accumulator holds for its level is what is in front of it. Adds
    increase that level, cancels and fills decrease it, a clear resets
    everything.

    An order whose price level has no history in this window still gets a
    position of zero, and that is honest: as far as this window can see, it
    joined an empty queue. What is NOT honest is treating a cancel or fill
    with no matching add as a zero-lifetime order, and that is why the
    left-censored ones are counted apart.

    NOR IS IT MARKET STRUCTURE, and the result now says which it is. A
    window that opens mid-session with no snapshot starts every level
    EMPTY, so each queue-ahead figure counts only the size added inside the
    window: with 1,000 shares resting before it opened, five 100-share
    arrivals read 0, 100, 200, 300 and 400 where the queue was 1,000 to
    1,400, and the one that "joined an empty level" is warm-up rather than
    a fact about the market. Until a snapshot
    or a CLEAR tells the window what the book held, every reading is a
    LOWER BOUND: `n_unseeded_adds` counts them and `queue_is_lower_bound`
    says whether there are any.

    THE SIZE AN UNSEEN ORDER TAKES WITH IT IS NOT TAKEN FROM A SEEN ONE.
    Resting size is tracked per ORDER, so a cancel or fill of an order this
    window never saw added (and no snapshot showed) removes nothing from
    the level -- that size was never in the running total -- and is
    counted instead: `n_unseen_decrements` events, `unseen_size` shares of
    resting size the queue figures never included. Level-only bookkeeping
    subtracted it from the orders the window HAD seen, pushing an
    already-short queue further toward zero without a trace. See the
    CHANGELOG entry of 2026-09-27.

    THE PASS IS NATIVE when the extension carries it and the frame holds
    64 events or more: the same state machine over coded columns, giving
    the same numbers (see the CHANGELOG entry of 2026-10-04). A frame whose
    order ids or sides cannot be coded the way the loop compares them -- a
    missing id on an add, cancel or fill, a missing side on an add -- runs
    the loop.
    """
    codes = _native_codes(events, sides=True) if _sized_for_native(events) else None
    return _queue_positions(events, codes)


def _queue_positions(
    events: pd.DataFrame, codes: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if codes is not None and codes["sides"] is not None:
        ahead, warm_up = _queue_pass_native(events, codes)
    else:
        ahead, warm_up = _queue_pass_python(events)
    return _queue_summary(ahead, warm_up)


def _queue_pass_python(events: pd.DataFrame):
    """The queue pass as a loop over events: the fallback for
    `_sqt_core.order_queue_ahead` and the reference it is tested against."""
    resting: Dict[Any, float] = {}
    # order_id -> [level key, size still resting], for every order this
    # window saw added (a snapshot add included).
    orders: Dict[Any, List[Any]] = {}
    ahead: List[float] = []
    n_snapshot_orders = 0
    # The book the window opened on is known once a snapshot seeds it or a
    # CLEAR empties it; an add before either is measured against a book
    # whose earlier contents were never seen.
    seeded = False
    n_unseeded_adds = 0
    n_unseen_decrements = 0
    unseen_size = 0.0
    for order_id, action, side, price, size, is_snapshot in zip(
        events["order_id"].to_numpy(),
        events["action"].to_numpy(),
        events["side"].to_numpy(),
        pd.to_numeric(events["price"], errors="coerce").to_numpy(dtype="float64"),
        pd.to_numeric(events["size"], errors="coerce").to_numpy(dtype="float64"),
        _snapshot_mask(events),
    ):
        if action == CLEAR:
            resting.clear()
            orders.clear()
            seeded = True
            continue
        if not np.isfinite(price) or not np.isfinite(size):
            continue
        key = (side, price)
        if action == ADD:
            # A snapshot add is the book as it stood: it seeds the queue
            # every later arrival waits behind, and is not itself an
            # arrival. Dropping it understated real queues by 33-79%
            # against the mbp-10 book for the same sequence numbers.
            if is_snapshot:
                n_snapshot_orders += 1
                seeded = True
            else:
                ahead.append(resting.get(key, 0.0))
                if not seeded:
                    n_unseeded_adds += 1
            resting[key] = resting.get(key, 0.0) + size
            orders[order_id] = [key, size]
        elif action in (CANCEL, FILL):
            known = orders.get(order_id)
            if known is None:
                n_unseen_decrements += 1
                unseen_size += size
                continue
            level, remaining = known
            taken = min(size, remaining)
            resting[level] = max(0.0, resting.get(level, 0.0) - taken)
            if remaining - taken > 0:
                known[1] = remaining - taken
            else:
                del orders[order_id]
    warm_up = {
        "n_snapshot_orders": int(n_snapshot_orders),
        "n_unseeded_adds": int(n_unseeded_adds),
        "queue_is_lower_bound": bool(n_unseeded_adds > 0),
        "n_unseen_decrements": int(n_unseen_decrements),
        "unseen_size": float(unseen_size),
    }
    return ahead, warm_up


def _queue_pass_native(events: pd.DataFrame, codes: Dict[str, Any]):
    """The queue pass in `_sqt_core.order_queue_ahead`, over the codes."""
    ahead, n_snapshot_orders, n_unseeded_adds, n_unseen_decrements, unseen_size = (
        _cpp_core.order_queue_ahead(
            codes["orders"],
            codes["n_orders"],
            codes["actions"],
            codes["sides"],
            pd.to_numeric(events["price"], errors="coerce").to_numpy(dtype="float64"),
            pd.to_numeric(events["size"], errors="coerce").to_numpy(dtype="float64"),
            codes["snapshot"],
        )
    )
    warm_up = {
        "n_snapshot_orders": int(n_snapshot_orders),
        "n_unseeded_adds": int(n_unseeded_adds),
        "queue_is_lower_bound": bool(n_unseeded_adds > 0),
        "n_unseen_decrements": int(n_unseen_decrements),
        "unseen_size": float(unseen_size),
    }
    return ahead, warm_up


def _queue_summary(ahead, warm_up: Dict[str, Any]) -> Dict[str, Any]:
    if len(ahead) == 0:
        return {
            "n_adds": 0,
            "mean_queue_ahead": None,
            "median_queue_ahead": None,
            "share_joining_empty": None,
            **warm_up,
        }
    values = np.asarray(ahead, dtype="float64")
    return {
        "n_adds": int(values.size),
        "mean_queue_ahead": float(values.mean()),
        "median_queue_ahead": float(np.median(values)),
        # A high share means the level is usually empty when you arrive,
        # which is a different market from one where you always queue --
        # unless `queue_is_lower_bound`, when it is partly the window's own
        # warm-up.
        "share_joining_empty": float((values == 0.0).mean()),
        **warm_up,
    }


def order_lifetimes(events: pd.DataFrame) -> Dict[str, Any]:
    """
    How long an order rests before it is cancelled or filled.

    LEFT-CENSORED ORDERS ARE COUNTED, NOT MEASURED. An order that was
    already resting when the window opened has no ADD here, so its
    lifetime is longer than anything observable and folding it in as the
    time since the window started would bias every average downward --
    worst for exactly the long-resting orders a queue study cares about.

    THE PASS IS NATIVE when the extension carries it and the frame holds
    64 events or more, with each lifetime turned into seconds as
    Timedelta.total_seconds() turns it -- whole microseconds, so a lifetime
    under one reads 0.0 -- and the same numbers (see the CHANGELOG entry of
    2026-10-04). A frame the codes cannot represent, or whose timestamps do
    not parse to one datetime column, runs the loop.
    """
    codes = _native_codes(events, stamps=True) if _sized_for_native(events) else None
    return _order_lifetimes(events, codes)


def _order_lifetimes(
    events: pd.DataFrame, codes: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    passed = None
    if codes is not None and codes["stamps"] is not None:
        passed = _lifetime_pass_native(codes)
    if passed is None:
        passed = _lifetime_pass_python(events)
    filled, cancelled, still_resting, censored, from_snapshot, known_at_open = passed
    return _lifetime_summary(
        filled, cancelled, still_resting, censored, from_snapshot, known_at_open
    )


def _lifetime_pass_python(events: pd.DataFrame):
    """The lifetime pass as a loop over events: the fallback for
    `_sqt_core.order_lifetimes` and the reference it is tested against."""
    added: Dict[Any, Any] = {}
    # Orders the window's snapshot showed already resting: known, but
    # with no observable start, so a later cancel or fill is neither a
    # measured lifetime nor a mystery.
    known_at_open: set = set()
    filled: List[float] = []
    cancelled: List[float] = []
    censored = 0
    from_snapshot = 0
    stamps = pd.to_datetime(events["timestamp"], errors="coerce")
    for order_id, action, when, is_snapshot in zip(
        events["order_id"].to_numpy(),
        events["action"].to_numpy(),
        stamps,
        _snapshot_mask(events),
    ):
        if pd.isna(when):
            continue
        if action == ADD:
            if is_snapshot:
                known_at_open.add(order_id)
            else:
                added[order_id] = when
        elif action in (CANCEL, FILL):
            start = added.pop(order_id, None)
            if start is None:
                if order_id in known_at_open:
                    known_at_open.discard(order_id)
                    from_snapshot += 1
                else:
                    censored += 1
                continue
            seconds = (when - start).total_seconds()
            (cancelled if action == CANCEL else filled).append(float(seconds))
    return filled, cancelled, len(added), censored, from_snapshot, len(known_at_open)


def _lifetime_pass_native(codes: Dict[str, Any]):
    """The lifetime pass in `_sqt_core.order_lifetimes`, or None when a
    lifetime cannot be held exactly (the loop then runs, and raises where
    pandas does)."""
    stamps, unit = codes["stamps"]
    filled, cancelled, still_resting, censored, from_snapshot, known_at_open, over = (
        _cpp_core.order_lifetimes(
            codes["orders"],
            codes["n_orders"],
            codes["actions"],
            codes["snapshot"],
            stamps,
        )
    )
    if over:
        return None
    filled_seconds = _total_seconds(filled, unit)
    cancelled_seconds = _total_seconds(cancelled, unit)
    if filled_seconds is None or cancelled_seconds is None:
        return None
    return (
        filled_seconds,
        cancelled_seconds,
        int(still_resting),
        int(censored),
        int(from_snapshot),
        int(known_at_open),
    )


#: Microseconds per unit of each datetime64 resolution pandas uses, as a
#: (multiplier, divisor) pair: whole microseconds are value * m // d.
_MICROSECONDS = {"s": (1_000_000, 1), "ms": (1_000, 1), "us": (1, 1), "ns": (1, 1_000)}

#: Past this many microseconds an int64 product or a float64 conversion is
#: no longer exact; such a lifetime (more than 285 years) runs the loop.
_EXACT_MICROSECONDS = 2**53


def _total_seconds(durations: np.ndarray, unit: str) -> Optional[np.ndarray]:
    """
    Durations in `unit` as Timedelta.total_seconds() returns them, or None.

    pandas computes total_seconds() from the timedelta's components:
    days * 86400 + seconds, an integer, plus microseconds / 1e6 -- with the
    nanoseconds dropped (floored, so -1 ns is -1 us) and the integer part
    added to the fraction in floating point. So 1.234567891 s is 1.234567,
    and 21.472982 s is 21.472982000000002 where a single division gives
    21.472982. This forms the same three numbers and the same sum.
    """
    if unit not in _MICROSECONDS:
        return None
    durations = np.asarray(durations, dtype=np.int64)
    multiplier, divisor = _MICROSECONDS[unit]
    # Compared from both sides rather than through abs(), which wraps at
    # INT64_MIN.
    limit = _EXACT_MICROSECONDS // multiplier
    if multiplier > 1 and ((durations > limit) | (durations < -limit)).any():
        return None
    micro = np.floor_divide(durations, divisor) * multiplier
    whole = np.floor_divide(micro, 1_000_000)
    if ((whole > _EXACT_MICROSECONDS) | (whole < -_EXACT_MICROSECONDS)).any():
        return None
    fraction = micro - whole * 1_000_000
    return whole.astype(np.float64) + fraction / 1_000_000


def _lifetime_summary(
    filled, cancelled, still_resting, censored, from_snapshot, known_at_open
) -> Dict[str, Any]:
    def _summary(values) -> Dict[str, Any]:
        # QUANTILES, NOT JUST A MEAN AND A MEDIAN. Order lifetimes are one
        # of the most skewed distributions this library measures -- a
        # cancelled-order median of 10.1 ms against a mean of 475.7 ms, a
        # 47x ratio -- so the two central numbers describe two different
        # populations and neither describes the tail that decides whether a
        # passive order ever rests long enough to fill. The quartiles say
        # how wide the bulk is and p90/p99 say how long the long ones live.
        keys = ("p25_seconds", "p75_seconds", "p90_seconds", "p99_seconds")
        if len(values) == 0:
            empty: Dict[str, Any] = {
                "n": 0,
                "mean_seconds": None,
                "median_seconds": None,
            }
            empty.update({key: None for key in keys})
            return empty
        arr = np.asarray(values, dtype="float64")
        quantiles = np.percentile(arr, [25, 75, 90, 99])
        out: Dict[str, Any] = {
            "n": int(arr.size),
            "mean_seconds": float(arr.mean()),
            "median_seconds": float(np.median(arr)),
        }
        out.update(
            {key: float(value) for key, value in zip(keys, quantiles, strict=False)}
        )
        return out

    return {
        "filled": _summary(filled),
        "cancelled": _summary(cancelled),
        # Still open when the window closed: right-censored, and reported
        # for the same reason the left-censored count is.
        "still_resting": int(still_resting),
        "terminated_without_an_add": censored,
        # Terminated orders the snapshot had shown resting: explained,
        # not censored -- they were 54.5% of the censored count on a CME
        # reopen before the snapshot was read.
        "terminated_from_snapshot": int(from_snapshot),
        "resting_at_open": int(known_at_open + from_snapshot),
    }


#: Below this many events a standalone queue or lifetime pass runs the loop.
#: Coding the columns costs three pd.factorize calls, and on their own the
#: two passes measured 0.82-0.92x of the loop at 25-50 events and 1.1-1.5x
#: from 75. `order_event_metrics` codes the columns once for both passes,
#: and is faster natively at every size measured (1.10x at 25 events), so
#: it is not gated.
_NATIVE_MIN_EVENTS = 64


def _sized_for_native(events: pd.DataFrame) -> bool:
    return len(events) >= _NATIVE_MIN_EVENTS


#: The action codes the kernels read, by the comparisons the loops make.
_CODE_ADD, _CODE_CANCEL, _CODE_FILL, _CODE_CLEAR, _CODE_OTHER = 0, 1, 2, 3, 4


def _action_code(value: Any) -> int:
    """One action value's code: the branch each loop takes for it."""
    if value == CLEAR:
        return _CODE_CLEAR
    if value == ADD:
        return _CODE_ADD
    if value in (CANCEL, FILL):
        return _CODE_CANCEL if value == CANCEL else _CODE_FILL
    return _CODE_OTHER


def _factorized(values: np.ndarray) -> Optional[tuple]:
    """pd.factorize, or None for a column it cannot hash (the loops can
    still compare such values, so they run instead)."""
    try:
        codes, uniques = pd.factorize(values, sort=False)
    except TypeError:
        return None
    return np.asarray(codes, dtype=np.int64), uniques


def _native_codes(
    events: pd.DataFrame, *, sides: bool = False, stamps: bool = False
) -> Optional[Dict[str, Any]]:
    """
    The columns as the kernels read them, or None to run the loops.

    Two values are one order (or one side) to the loops exactly when they
    are one dict key, which is what pd.factorize groups by: equal and of
    equal hash, -0.0 with 0.0, 1 with 1.0. A missing value is where the two
    part: factorize codes NaN and None as -1, while a dict holds None as a
    key and never finds a NaN it was handed from a float column. So an
    order id missing on an add, cancel or fill, or a side missing on an
    add, sends the frame to the loops (`sides` is then None for the queue,
    and the whole result None). The timestamps are their integers in the
    column's own unit, or None when they do not parse to one datetime
    column.
    """
    if not HAS_CPP:
        return None
    action_values = _factorized(events["action"].to_numpy())
    order_values = _factorized(events["order_id"].to_numpy())
    if action_values is None or order_values is None:
        return None
    codes, uniques = action_values
    table = np.array(
        [_action_code(value) for value in uniques] + [_CODE_OTHER], dtype=np.int64
    )
    actions = table[codes]
    order_codes, order_uniques = order_values
    if (order_codes[actions <= _CODE_FILL] < 0).any():
        return None
    out: Dict[str, Any] = {
        "actions": actions,
        "orders": order_codes,
        "n_orders": int(len(order_uniques)),
        "snapshot": _snapshot_mask(events),
        "sides": None,
        "stamps": None,
    }
    if sides:
        side_values = _factorized(events["side"].to_numpy())
        if (
            side_values is not None
            and not (side_values[0][actions == _CODE_ADD] < 0).any()
        ):
            out["sides"] = side_values[0]
    if stamps:
        parsed = _as_datetimes(events["timestamp"])
        if _is_datetime(parsed.dtype):
            out["stamps"] = (
                np.asarray(parsed.array.asi8, dtype=np.int64),
                str(getattr(parsed.dt, "unit", "ns")),
            )
    return out


def event_rates(events: pd.DataFrame) -> Dict[str, Any]:
    """
    Events per second, in total and by action.

    A rate needs a real clock, so this returns None rather than zero when
    the window has no duration -- one event, or every event on the same
    timestamp. Zero would read as a quiet market.
    """
    # Snapshot records are the book, not events: they neither count nor
    # tick the clock (a snapshot-bearing window read 16,000x too many
    # events per second before this).
    snapshot = _snapshot_mask(events)
    # The two columns read, not the whole frame: the same rows in the same
    # order as `events.loc[~snapshot]`, without copying the other columns.
    stamps, actions = events["timestamp"], events["action"]
    if snapshot.any():
        stamps, actions = stamps[~snapshot], actions[~snapshot]
    seconds = _elapsed_seconds(stamps)
    counts = actions.value_counts().to_dict()
    total = int(len(actions))
    per_action = {str(k): int(v) for k, v in counts.items()}
    rates = (
        {str(k): float(v) / seconds for k, v in per_action.items()} if seconds else {}
    )
    adds = per_action.get(ADD, 0)
    cancels = per_action.get(CANCEL, 0)
    # ONE execution is a T (the print) and an F (the resting order it
    # hit): counting both counted every trade twice. Trades are the T
    # prints when the feed carries them, else the fills.
    trades = per_action.get(TRADE, 0) or per_action.get(FILL, 0)
    return {
        "n_events": total,
        "n_snapshot_events": int(snapshot.sum()),
        "elapsed_seconds": seconds,
        "events_per_second": (total / seconds) if seconds else None,
        "counts_by_action": per_action,
        "rates_by_action": rates,
        # The two ratios a maker cares about. Both are None rather than 0
        # when the denominator is absent, because "no adds in the window"
        # and "nothing was ever cancelled" are different statements.
        "cancel_to_add": (cancels / adds) if adds else None,
        "cancel_to_trade": (cancels / trades) if trades else None,
    }


def order_event_metrics(
    events: pd.DataFrame, *, name: str = "order events"
) -> Dict[str, Any]:
    """Every order-level measure, over one window of events."""
    frame = _require(events, name)
    warnings: List[str] = []

    # The distinct actions, read once for the three notes below.
    present = set(frame["action"].dropna().unique())
    unknown = sorted(present - set(ACTION_MEANINGS))
    if unknown:
        warnings.append(
            f"NOTE: action code(s) {unknown} are not in this feed's known "
            f"vocabulary {sorted(ACTION_MEANINGS)}; their events are counted "
            "in the totals and ignored by the queue and lifetime measures."
        )
    if CLEAR in present:
        warnings.append(
            "NOTE: the window contains a CLEAR, so the book was wiped inside "
            "it. Queue depth resets there rather than carrying across, which "
            "is right, but means the measures span a discontinuity."
        )
    if MODIFY in present:
        warnings.append(
            "NOTE: MODIFY events are counted but do not adjust queue depth "
            "here. A modify that raises size or changes price loses queue "
            "priority at the venue, and treating it as a cancel-plus-add "
            "would require the venue's own rule, which differs by venue."
        )

    rates = event_rates(frame)
    if rates.get("n_snapshot_events"):
        warnings.append(
            f"NOTE: {rates['n_snapshot_events']:,} snapshot record(s) seed the "
            "queue and explain later cancels; they are excluded from the "
            "event counts, the rates and the clock."
        )
    if rates["elapsed_seconds"] is None:
        warnings.append(
            "WARNING: every event carries the same timestamp, or there is "
            "only one, so no rate is defined. Reported as null rather than "
            "zero, which would read as a quiet market."
        )

    codes = _native_codes(frame, sides=True, stamps=True)
    lifetimes = _order_lifetimes(frame, codes)
    if lifetimes["terminated_without_an_add"]:
        warnings.append(
            f"NOTE: {lifetimes['terminated_without_an_add']:,} order(s) were "
            "cancelled or filled without an add in this window -- they were "
            "already resting when it opened. They are counted here and "
            "EXCLUDED from the lifetime averages, because their true "
            "lifetime is longer than anything this window can see."
        )

    queue = _queue_positions(frame, codes)
    if queue["queue_is_lower_bound"]:
        removed = (
            f" {queue['n_unseen_decrements']:,} cancel(s) or fill(s) removed "
            f"{queue['unseen_size']:,.0f} shares of orders this window never "
            "saw added -- size that was resting, possibly ahead, and is in "
            "none of the queue figures."
            if queue["n_unseen_decrements"]
            else ""
        )
        warnings.append(
            "WARNING: no snapshot or CLEAR opened this window, so every price "
            "level starts EMPTY and "
            f"{queue['n_unseeded_adds']:,} of {queue['n_adds']:,} queue-ahead "
            "readings are LOWER BOUNDS: they count only size added inside "
            "the window, and `share_joining_empty` is partly the window's "
            f"own warm-up rather than the market.{removed} Read a window that "
            "opens with the venue's snapshot for queue positions that "
            "describe the book."
        )
    elif queue["n_unseen_decrements"]:
        warnings.append(
            f"NOTE: {queue['n_unseen_decrements']:,} cancel(s) or fill(s) "
            f"removed {queue['unseen_size']:,.0f} shares of orders neither "
            "the snapshot nor this window showed resting. That size is in "
            "none of the queue figures; a snapshot that does not cover the "
            "whole book is the usual cause."
        )

    return {
        "n_events": rates["n_events"],
        "queue": queue,
        "lifetimes": lifetimes,
        "rates": rates,
        "warnings": warnings,
    }


__all__ = [
    "ACTION_MEANINGS",
    "ADD",
    "ASK",
    "BID",
    "CANCEL",
    "CLEAR",
    "FILL",
    "MODIFY",
    "ORDER_EVENT_COLUMNS",
    "TRADE",
    "event_rates",
    "order_event_metrics",
    "order_lifetimes",
    "queue_positions",
]
