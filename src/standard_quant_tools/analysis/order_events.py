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
exactly the resting orders a queue study is about.

A CLEAR (`R`) wipes the book. The accumulators reset on one rather than
carrying a stale level into the next session, which would report queue
depth accumulated across a boundary where none existed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

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


def _elapsed_seconds(stamps: pd.Series) -> Optional[float]:
    valid = pd.to_datetime(stamps, errors="coerce").dropna()
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
    """
    resting: Dict[Any, float] = {}
    ahead: List[float] = []
    for action, side, price, size in zip(
        events["action"].to_numpy(),
        events["side"].to_numpy(),
        pd.to_numeric(events["price"], errors="coerce").to_numpy(dtype="float64"),
        pd.to_numeric(events["size"], errors="coerce").to_numpy(dtype="float64"),
    ):
        if action == CLEAR:
            resting.clear()
            continue
        if not np.isfinite(price) or not np.isfinite(size):
            continue
        key = (side, price)
        if action == ADD:
            ahead.append(resting.get(key, 0.0))
            resting[key] = resting.get(key, 0.0) + size
        elif action in (CANCEL, FILL):
            resting[key] = max(0.0, resting.get(key, 0.0) - size)
    if not ahead:
        return {
            "n_adds": 0,
            "mean_queue_ahead": None,
            "median_queue_ahead": None,
            "share_joining_empty": None,
        }
    values = np.asarray(ahead, dtype="float64")
    return {
        "n_adds": int(values.size),
        "mean_queue_ahead": float(values.mean()),
        "median_queue_ahead": float(np.median(values)),
        # A high share means the level is usually empty when you arrive,
        # which is a different market from one where you always queue.
        "share_joining_empty": float((values == 0.0).mean()),
    }


def order_lifetimes(events: pd.DataFrame) -> Dict[str, Any]:
    """
    How long an order rests before it is cancelled or filled.

    LEFT-CENSORED ORDERS ARE COUNTED, NOT MEASURED. An order that was
    already resting when the window opened has no ADD here, so its
    lifetime is longer than anything observable and folding it in as the
    time since the window started would bias every average downward --
    worst for exactly the long-resting orders a queue study cares about.
    """
    added: Dict[Any, Any] = {}
    filled: List[float] = []
    cancelled: List[float] = []
    censored = 0
    stamps = pd.to_datetime(events["timestamp"], errors="coerce")
    for order_id, action, when in zip(
        events["order_id"].to_numpy(), events["action"].to_numpy(), stamps
    ):
        if pd.isna(when):
            continue
        if action == ADD:
            added[order_id] = when
        elif action in (CANCEL, FILL):
            start = added.pop(order_id, None)
            if start is None:
                censored += 1
                continue
            seconds = (when - start).total_seconds()
            (cancelled if action == CANCEL else filled).append(float(seconds))

    def _summary(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {"n": 0, "mean_seconds": None, "median_seconds": None}
        arr = np.asarray(values, dtype="float64")
        return {
            "n": int(arr.size),
            "mean_seconds": float(arr.mean()),
            "median_seconds": float(np.median(arr)),
        }

    return {
        "filled": _summary(filled),
        "cancelled": _summary(cancelled),
        # Still open when the window closed: right-censored, and reported
        # for the same reason the left-censored count is.
        "still_resting": int(len(added)),
        "terminated_without_an_add": censored,
    }


def event_rates(events: pd.DataFrame) -> Dict[str, Any]:
    """
    Events per second, in total and by action.

    A rate needs a real clock, so this returns None rather than zero when
    the window has no duration -- one event, or every event on the same
    timestamp. Zero would read as a quiet market.
    """
    seconds = _elapsed_seconds(events["timestamp"])
    counts = events["action"].value_counts().to_dict()
    total = int(len(events))
    per_action = {str(k): int(v) for k, v in counts.items()}
    rates = (
        {str(k): float(v) / seconds for k, v in per_action.items()} if seconds else {}
    )
    adds = per_action.get(ADD, 0)
    cancels = per_action.get(CANCEL, 0)
    trades = per_action.get(TRADE, 0) + per_action.get(FILL, 0)
    return {
        "n_events": total,
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

    unknown = sorted(set(frame["action"].dropna().unique()) - set(ACTION_MEANINGS))
    if unknown:
        warnings.append(
            f"NOTE: action code(s) {unknown} are not in this feed's known "
            f"vocabulary {sorted(ACTION_MEANINGS)}; their events are counted "
            "in the totals and ignored by the queue and lifetime measures."
        )
    if CLEAR in set(frame["action"].dropna().unique()):
        warnings.append(
            "NOTE: the window contains a CLEAR, so the book was wiped inside "
            "it. Queue depth resets there rather than carrying across, which "
            "is right, but means the measures span a discontinuity."
        )
    if MODIFY in set(frame["action"].dropna().unique()):
        warnings.append(
            "NOTE: MODIFY events are counted but do not adjust queue depth "
            "here. A modify that raises size or changes price loses queue "
            "priority at the venue, and treating it as a cancel-plus-add "
            "would require the venue's own rule, which differs by venue."
        )

    rates = event_rates(frame)
    if rates["elapsed_seconds"] is None:
        warnings.append(
            "WARNING: every event carries the same timestamp, or there is "
            "only one, so no rate is defined. Reported as null rather than "
            "zero, which would read as a quiet market."
        )

    lifetimes = order_lifetimes(frame)
    if lifetimes["terminated_without_an_add"]:
        warnings.append(
            f"NOTE: {lifetimes['terminated_without_an_add']:,} order(s) were "
            "cancelled or filled without an add in this window -- they were "
            "already resting when it opened. They are counted here and "
            "EXCLUDED from the lifetime averages, because their true "
            "lifetime is longer than anything this window can see."
        )

    return {
        "n_events": rates["n_events"],
        "queue": queue_positions(frame),
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
