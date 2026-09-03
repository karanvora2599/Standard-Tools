"""
Order-level analytics, against `DataProvider.get_order_events`.

WHY THIS IS NOT IN `book_tools.py`. A book and an order feed are different
objects, not two depths of the same one. `book_tools` reads snapshots of
aggregated size per price level; this reads individual orders with their own
identifiers. The four measures below exist BECAUSE aggregation destroys
them: a book showing 5,000 shares at the bid cannot say whether that is one
order or two hundred, which arrived first, or whether the size that
disappeared was cancelled or filled — and cancelled and filled mean opposite
things about who wanted to trade.

THE EVENTS ARRIVE AS AN ARGUMENT OR A REFERENCE, the same way a book does
and for the same reason. `DatabentoProvider` serves them, and so does a
vendor extract, an ITCH replay or another system; a tool that fetched its
own would serve one source and refuse every other. An MBO session is large
enough that the reference path is the normal one — one active name produces
millions of records where mbp-10 produces thousands.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from standard_quant_tools.analysis import order_events as lib
from standard_quant_tools.error import ValidationError

__all__ = [
    "EVENT_TOOL_CATEGORY",
    "EVENT_TOOL_DEFS",
    "EVENT_TOOL_DISPATCH",
    "OrderEventInput",
    "OrderEventResult",
    "get_order_event_metrics",
]


class OrderEventInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: Optional[List[Dict[str, Any]]] = Field(
        None,
        min_length=1,
        description=(
            "One dict per order event, following "
            "DataProvider.get_order_events: timestamp, order_id, action "
            "(A add, C cancel, M modify, F fill, T trade, R clear), side "
            "(B/A), price, size. For a window small enough to pass inline."
        ),
    )
    ref: Optional[str] = Field(
        None,
        description=(
            "An `sqt://order_event_panel/...` reference from "
            "register_external_dataset, read in batches off disk. This is "
            "the normal path: a session of market-by-order is millions of "
            "records and cannot travel through a tool argument."
        ),
    )
    max_events: int = Field(
        2_000_000,
        gt=0,
        le=200_000_000,
        description=(
            "Cap on events read from a `ref`. Order lifetimes and queue "
            "positions are measured WITHIN the window read, so a cap makes "
            "them a statement about a prefix of the session -- and the open "
            "is its least typical part. The result says when the cap bound."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "OrderEventInput":
        given = [
            name
            for name, value in (("events", self.events), ("ref", self.ref))
            if value is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "order events need exactly one of `events` (inline) or `ref` "
                f"(a registered order_event_panel); got {given or 'neither'}. "
                "Two sources would make the precedence rule part of the "
                "contract, and a caller who passed both meant something this "
                "tool cannot infer."
            )
        return self


class QueueSummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    n_adds: int = 0
    mean_queue_ahead: Optional[float] = Field(
        None,
        description="Resting size already at an order's own price level when "
        "it arrived. The number that decides whether a passive order fills.",
    )
    median_queue_ahead: Optional[float] = None
    share_joining_empty: Optional[float] = Field(
        None,
        description="Fraction of adds that joined an EMPTY level. A high "
        "share is a different market from one where you always queue.",
    )


class LifetimeSummary(BaseModel):
    model_config = ConfigDict(extra="allow")
    n: int = 0
    mean_seconds: Optional[float] = None
    median_seconds: Optional[float] = None


class OrderEventResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_events: int = 0
    elapsed_seconds: Optional[float] = None
    events_per_second: Optional[float] = None
    counts_by_action: Dict[str, int] = Field(default_factory=dict)
    rates_by_action: Dict[str, float] = Field(default_factory=dict)
    cancel_to_add: Optional[float] = Field(
        None,
        description="Cancels per add. Null rather than 0 when there were no "
        "adds -- 'nothing was added' and 'nothing was cancelled' differ.",
    )
    cancel_to_trade: Optional[float] = None
    queue: QueueSummary = Field(default_factory=QueueSummary)
    filled: LifetimeSummary = Field(default_factory=LifetimeSummary)
    cancelled: LifetimeSummary = Field(default_factory=LifetimeSummary)
    still_resting: int = Field(
        0, description="Open when the window closed: right-censored."
    )
    terminated_without_an_add: int = Field(
        0,
        description="Already resting when the window opened. Counted, and "
        "EXCLUDED from the lifetime averages, because their true lifetime is "
        "longer than anything this window can see.",
    )
    truncated: bool = False
    warnings: List[str] = Field(default_factory=list)


def _events_from_reference(ref: str, cap: int):
    """Read a registered order-event panel off disk, bounded."""
    from ..handoff import resolve as _resolve

    handle = _resolve(ref, expect="order_event_panel")
    if isinstance(handle, pd.DataFrame):
        raise ValidationError(
            f"{ref!r} resolves to an in-memory frame rather than a registered "
            "external dataset. Only a panel registered with "
            "register_external_dataset can be streamed here."
        )
    wanted = [c for c in lib.ORDER_EVENT_COLUMNS if c in handle.columns]
    chunks: List[pd.DataFrame] = []
    read = 0
    truncated = False
    for chunk in handle.batches(columns=wanted or None):
        room = cap - read
        if room <= 0:
            truncated = True
            break
        if len(chunk) > room:
            chunk = chunk.iloc[:room]
            truncated = True
        chunks.append(chunk)
        read += len(chunk)
        if read >= cap:
            truncated = True
            break
    if not chunks:
        raise ValidationError(
            f"{ref!r} resolved but held no rows. A registered dataset can "
            "become empty if the file it points at was replaced."
        )
    return pd.concat(chunks, ignore_index=True), truncated, handle


def get_order_event_metrics(input_data: OrderEventInput) -> OrderEventResult:
    notes: List[str] = []
    truncated = False
    if input_data.ref is not None:
        events, truncated, handle = _events_from_reference(
            input_data.ref, input_data.max_events
        )
        if truncated:
            total = f"{handle.rows:,}" if handle.rows else "an unknown number of"
            notes.append(
                f"WARNING: read {len(events):,} of {total} events, stopping "
                f"at max_events={input_data.max_events:,}. Lifetimes and "
                "queue positions are measured within what was READ, so an "
                "order resting past the cut looks unterminated and one "
                "resting before it looks unadded."
            )
    else:
        events = pd.DataFrame(input_data.events)

    metrics = lib.order_event_metrics(events)
    rates = metrics["rates"]
    lifetimes = metrics["lifetimes"]
    return OrderEventResult(
        n_events=metrics["n_events"],
        elapsed_seconds=rates["elapsed_seconds"],
        events_per_second=rates["events_per_second"],
        counts_by_action=rates["counts_by_action"],
        rates_by_action=rates["rates_by_action"],
        cancel_to_add=rates["cancel_to_add"],
        cancel_to_trade=rates["cancel_to_trade"],
        queue=QueueSummary(**metrics["queue"]),
        filled=LifetimeSummary(**lifetimes["filled"]),
        cancelled=LifetimeSummary(**lifetimes["cancelled"]),
        still_resting=lifetimes["still_resting"],
        terminated_without_an_add=lifetimes["terminated_without_an_add"],
        truncated=truncated,
        warnings=notes + list(metrics["warnings"]),
    )


EVENT_TOOL_DEFS = [
    (
        "get_order_event_metrics",
        "Order-level statistics a depth book cannot produce: how much size "
        "rests AHEAD of an order at its own price level when it arrives, how "
        "long orders live before they are cancelled or filled, cancels per "
        "add and per trade, and event intensity by action. Aggregated depth "
        "destroys all four -- 5,000 shares at the bid may be one order or two "
        "hundred, and size that disappears may have been cancelled or filled, "
        "which mean opposite things about who wanted to trade. Orders already "
        "resting when the window opened are counted and EXCLUDED from the "
        "lifetime averages rather than folded in as instantaneous, which "
        "would bias every average toward impatience precisely where the "
        "long-resting orders are. Takes events inline or as an "
        "`sqt://order_event_panel` reference.",
        OrderEventInput,
    ),
]

EVENT_TOOL_DISPATCH = {
    "get_order_event_metrics": (get_order_event_metrics, OrderEventInput),
}

EVENT_TOOL_CATEGORY = {"get_order_event_metrics": "microstructure"}
