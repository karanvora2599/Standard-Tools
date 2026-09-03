"""
Depth-book analytics, against the contract `DataProvider.get_order_book`
declared before any provider implemented it.

That declaration said why: "the analysis that consumes a book (microprice,
order-flow imbalance, depth slope) can be written and tested against
synthetic books now, so that when a source arrives the correctness-critical
part already exists rather than being invented under deadline." These read
that column contract and nothing else, so any feed shaped to it works --
including one this library has no provider for.

THE BOOK ARRIVES AS AN ARGUMENT for the same reason option chains do in the
derivatives runtime, and it stays that way now that a provider does serve
depth. `DatabentoProvider.get_order_book` returns exactly this column
contract, so a fetched book is passed in like any other -- and a book from a
vendor extract, a replay or another system works identically, which is the
case that actually matters. A tool that fetched its own book would serve one
source and refuse every other.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from standard_quant_tools.analysis import order_book as lib
from standard_quant_tools.data.external import book_levels as lib_book_levels
from standard_quant_tools.error import ValidationError

__all__ = [
    "BOOK_TOOL_CATEGORY",
    "BOOK_TOOL_DEFS",
    "BOOK_TOOL_DISPATCH",
    "OrderBookInput",
    "OrderBookResult",
    "get_order_book_metrics",
]


class OrderBookInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshots: Optional[List[Dict[str, float]]] = Field(
        None,
        min_length=1,
        description="One dict per book update, following "
        "DataProvider.get_order_book: bid_price_0/bid_size_0/ask_price_0/"
        "ask_size_0 and upward, level 0 being the touch. For a book small "
        "enough to pass inline; use `ref` for one that is not.",
    )
    ref: Optional[str] = Field(
        None,
        description=(
            "An `sqt://order_book_panel/...` reference from "
            "register_external_dataset, read in batches off disk. This is "
            "the path for a real depth feed -- a session of L2 is millions "
            "of snapshots and cannot travel through a tool argument."
        ),
    )
    max_snapshots: int = Field(
        1_000_000,
        gt=0,
        le=50_000_000,
        description=(
            "Cap on snapshots read from a `ref`. The statistics here are "
            "MEANS, so a cap makes them a mean over a PREFIX of the session "
            "rather than over all of it -- and the start of a session is its "
            "least typical part. The result says when the cap bound."
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "OrderBookInput":
        given = [
            name
            for name, value in (("snapshots", self.snapshots), ("ref", self.ref))
            if value is not None
        ]
        if len(given) != 1:
            raise ValueError(
                "a book needs exactly one of `snapshots` (inline) or `ref` "
                f"(a registered order_book_panel); got {given or 'neither'}. "
                "Two sources would make the precedence rule part of the "
                "contract, and a caller who passed both meant something this "
                "tool cannot infer."
            )
        return self

    levels: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="How deep to read. Omitted, every complete level present.",
    )
    include_dynamics: bool = Field(
        False,
        description="Also return what changes BETWEEN snapshots: order-flow "
        "imbalance at the touch (the Cont-Kukanov-Stoikov definition, not "
        "the bar-based signed-volume proxy `get_order_flow_imbalance` "
        "computes), update and mid-change rates. Needs at least two "
        "snapshots, because every measure is a comparison.",
    )
    include_profile: bool = Field(
        False,
        description="Also return size and distance PER LEVEL. A sum is what "
        "makes a thin book look deep.",
    )


class DepthLevel(BaseModel):
    model_config = ConfigDict(extra="allow")

    level: int = 0
    mean_bid_size: Optional[float] = None
    mean_ask_size: Optional[float] = None
    mean_bid_distance_bps: Optional[float] = None
    mean_ask_distance_bps: Optional[float] = None


class OrderBookResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    n_snapshots: int = 0
    levels_available: int = 0
    levels_read: int = 0
    n_crossed: int = Field(
        0, description="Snapshots with bid >= ask, excluded from price stats."
    )
    mean_spread: Optional[float] = None
    mean_spread_bps: Optional[float] = None
    mean_mid: Optional[float] = None
    mean_microprice: Optional[float] = Field(
        None,
        description="Size-weighted touch price. Each side weighted by the "
        "OPPOSITE side's size, because the heavy side absorbs.",
    )
    mean_microprice_lean: Optional[float] = Field(
        None, description="Where it sits in the spread: 0 bid, 1 ask, 0.5 mid."
    )
    mean_touch_imbalance: Optional[float] = Field(
        None, description="Level 0 only. Predicts the next tick."
    )
    mean_cumulative_imbalance: Optional[float] = Field(
        None, description="All levels read. Predicts where SIZE ends up."
    )
    mean_touch_size: Optional[float] = None
    mean_cumulative_size: Optional[float] = None
    depth_slope: Optional[float] = Field(
        None,
        description="Resting size per basis point from the mid. Null on a "
        "one-level book, which has no depth to slope.",
    )
    ofi: Optional[float] = Field(
        None,
        description="Cont-Kukanov-Stoikov order-flow imbalance at the touch, "
        "summed over the window. NOT signed volume: a bid price that rose "
        "contributes its whole new size, one that fell removes the whole old "
        "size, and only an unchanged price reduces to the size difference.",
    )
    ofi_per_update: Optional[float] = None
    ofi_per_second: Optional[float] = None
    updates_per_second: Optional[float] = None
    mid_changes: Optional[int] = None
    mid_changes_per_second: Optional[float] = None
    spread_changes: Optional[int] = None
    n_pairs_dropped: Optional[int] = Field(
        None,
        description="Consecutive pairs skipped because one side was "
        "non-finite or the book was crossed. Interpolating across one would "
        "invent a transition nobody observed.",
    )
    profile: List[DepthLevel] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def _book_from_reference(ref: str, cap: int, levels: Optional[int]):
    """
    Read a registered depth panel off disk, bounded, into one frame.

    Columns are PROJECTED to the ones the statistics actually read. A real
    mbp-10 export carries sixty-odd columns -- order counts, flags, actions,
    sequence numbers -- and pulling all of them to compute a microprice buys
    nothing. What is kept is exactly the four columns per level that
    `book_metrics` reads.
    """
    from ..handoff import resolve as _resolve

    handle = _resolve(ref, expect="order_book_panel")
    if isinstance(handle, pd.DataFrame):
        raise ValidationError(
            f"{ref!r} resolves to an in-memory frame rather than a registered "
            "external dataset. Only a book registered with "
            "register_external_dataset can be streamed here."
        )

    wanted: Optional[List[str]] = None
    depth = lib_book_levels(handle.columns)
    if depth:
        deep = depth if levels is None else min(int(levels), depth)
        wanted = [
            f"{side}_{field}_{index}"
            for index in range(deep)
            for side in ("bid", "ask")
            for field in ("price", "size")
        ]
        if "timestamp" in handle.columns:
            wanted.insert(0, "timestamp")

    chunks: List[pd.DataFrame] = []
    read = 0
    truncated = False
    for chunk in handle.batches(columns=wanted):
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


def get_order_book_metrics(input_data: OrderBookInput) -> OrderBookResult:
    notes: List[str] = []
    if input_data.ref is not None:
        book, truncated, handle = _book_from_reference(
            input_data.ref, input_data.max_snapshots, input_data.levels
        )
        if truncated:
            total = f"{handle.rows:,}" if handle.rows else "an unknown number of"
            notes.append(
                f"WARNING: read {len(book):,} of {total} snapshots, stopping "
                f"at max_snapshots={input_data.max_snapshots:,}. Every mean "
                "below is over that PREFIX of the session, not over all of "
                "it -- and the start of a session is its least typical part."
            )
    else:
        book = pd.DataFrame(input_data.snapshots)

    metrics = lib.book_metrics(book, levels=input_data.levels)
    if input_data.include_dynamics:
        dynamics = lib.book_dynamics(book)
        notes = notes + list(dynamics.pop("warnings", []))
        dynamics.pop("n_snapshots", None)
        dynamics.pop("n_pairs", None)
        dynamics.pop("elapsed_seconds", None)
        metrics.update(dynamics)
    profile: List[Dict[str, Any]] = []
    if input_data.include_profile:
        detail = lib.depth_profile(book, levels=input_data.levels)
        profile = detail["profile"]
        metrics["warnings"] = list(metrics["warnings"]) + list(detail["warnings"])
    metrics["warnings"] = notes + list(metrics.get("warnings", []))
    return OrderBookResult(profile=profile, **metrics)


BOOK_TOOL_DEFS = [
    (
        "get_order_book_metrics",
        "Depth-book statistics a top-of-book quote cannot give: the "
        "microprice, imbalance at the touch AND cumulatively, and how fast "
        "liquidity thins with distance. The midpoint ignores size, so a book "
        "with 5,000 bid and 100 offered reads the same as its mirror and the "
        "second is about to trade higher. Touch and cumulative imbalance "
        "routinely disagree, and a book bid at the touch with weight behind "
        "the offer is exactly the one that ticks up and fills badly. Takes "
        "the book inline for a small one, or an `sqt://order_book_panel` "
        "reference for a real session, which is read off disk in batches "
        "because millions of snapshots cannot travel through a tool "
        "argument.",
        OrderBookInput,
    ),
]

BOOK_TOOL_DISPATCH = {
    "get_order_book_metrics": (get_order_book_metrics, OrderBookInput),
}

BOOK_TOOL_CATEGORY = {"get_order_book_metrics": "microstructure"}
