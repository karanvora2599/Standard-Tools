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
derivatives runtime. No shipped provider serves depth; a tool that fetched
one would compute a book that does not exist. Passing it in also means these
work on a book from anywhere, which is the case that actually matters here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from standard_quant_tools.analysis import order_book as lib

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

    snapshots: List[Dict[str, float]] = Field(
        ...,
        min_length=1,
        description="One dict per book update, following "
        "DataProvider.get_order_book: bid_price_0/bid_size_0/ask_price_0/"
        "ask_size_0 and upward, level 0 being the touch.",
    )
    levels: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="How deep to read. Omitted, every complete level present.",
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
    profile: List[DepthLevel] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


def get_order_book_metrics(input_data: OrderBookInput) -> OrderBookResult:
    book = pd.DataFrame(input_data.snapshots)
    metrics = lib.book_metrics(book, levels=input_data.levels)
    profile: List[Dict[str, Any]] = []
    if input_data.include_profile:
        detail = lib.depth_profile(book, levels=input_data.levels)
        profile = detail["profile"]
        metrics["warnings"] = list(metrics["warnings"]) + list(detail["warnings"])
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
        "the book as an argument; no provider here serves depth.",
        OrderBookInput,
    ),
]

BOOK_TOOL_DISPATCH = {
    "get_order_book_metrics": (get_order_book_metrics, OrderBookInput),
}

BOOK_TOOL_CATEGORY = {"get_order_book_metrics": "microstructure"}
