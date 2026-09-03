"""
What a depth book says that a top-of-book quote cannot.

WRITTEN AGAINST THE DECLARED CONTRACT, NOT AGAINST A FEED. `DataProvider.
get_order_book` has specified its columns since before any provider
implemented it -- `timestamp`, then `bid_price_{i}` / `bid_size_{i}` /
`ask_price_{i}` / `ask_size_{i}` for level 0 upward, level 0 being the touch
-- and said explicitly why: the analysis that consumes a book can be written
and tested against synthetic books now, so that when a source arrives the
correctness-critical part already exists rather than being invented under
deadline. This is that analysis. It reads the contract and nothing else, so
any feed shaped to it works.

THE MID IS THE WRONG FAIR VALUE AND EVERYONE USES IT. The midpoint ignores
size, so a book with 5,000 bid and 100 offered says the same thing as one
with 100 and 5,000 -- and the second is about to trade higher. The
microprice weights each side by the OPPOSITE side's size, which is the
right way round and reads as backwards until you see why: the side with
more resting size is the side that will absorb, so price is pinned nearer
the thin side. On a balanced book it equals the mid exactly, so nothing is
lost by using it everywhere.

DEPTH IS NOT A NUMBER, IT IS A SLOPE. Two books can show identical size at
the touch and behave completely differently a hundred shares in. What
matters to anyone sizing a trade is how fast liquidity thins with distance,
which is a slope, and reporting only level-0 size is why an order that
looked liquid at the touch prints three ticks through it.

WHAT THIS REFUSES. A book with one level is top-of-book wearing a costume:
its imbalance is computable but its slope is not, and every depth question
below the touch has no data behind it. That is refused by name rather than
answered with a slope of zero, which would read as a perfectly flat and
infinitely deep book.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

__all__ = ["book_dynamics", "book_metrics", "depth_profile", "microprice"]


def _levels_present(frame: pd.DataFrame) -> int:
    """
    How many levels the frame actually carries DATA for.

    It used to count columns. A two-level book whose level-1 prices are all
    NaN -- the most ordinary malformed shape a vendor produces -- reported
    `levels_available: 2`, emitted no "only one level" warning, and returned
    a depth_slope computed from the single real level. The refusal this
    module's docstring advertises was defeated by the commonest bad input.

    A level counts when its four columns exist AND at least one snapshot
    carries a finite price on each side. Size is allowed to be zero or
    missing -- an empty level is a real state of the book -- but a level
    with no price anywhere is not a level, it is a column.
    """
    count = 0
    while all(
        f"{side}_{field}_{count}" in frame.columns
        for side in ("bid", "ask")
        for field in ("price", "size")
    ):
        has_data = all(
            bool(
                np.isfinite(
                    pd.to_numeric(frame[f"{side}_price_{count}"], errors="coerce")
                ).any()
            )
            for side in ("bid", "ask")
        )
        if not has_data:
            break
        count += 1
    return count


def microprice(bid_price: Any, bid_size: Any, ask_price: Any, ask_size: Any) -> Any:
    """
    The size-weighted touch price:  (Pb*Sa + Pa*Sb) / (Sb + Sa).

    Each side is weighted by the OPPOSITE side's size. That reads backwards
    and is right: the heavy side is the side that absorbs, so the price sits
    nearer the thin one. With equal sizes it collapses to the midpoint
    exactly, which is what makes it safe to use unconditionally.

    Works on scalars or arrays. Returns NaN where both sizes are zero --
    a book with nothing resting on either side has no size-weighted price,
    and the midpoint would be an answer to a question nobody could trade.
    """
    pb = np.asarray(bid_price, dtype=float)
    sb = np.asarray(bid_size, dtype=float)
    pa = np.asarray(ask_price, dtype=float)
    sa = np.asarray(ask_size, dtype=float)
    total = sb + sa
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(total > 0, (pb * sa + pa * sb) / total, np.nan)
    return out if out.ndim else float(out)


def book_metrics(book: pd.DataFrame, *, levels: Optional[int] = None) -> Dict[str, Any]:
    """
    Per-snapshot book statistics, averaged over the window.

    `book` follows `DataProvider.get_order_book`'s column contract. `levels`
    caps how deep to read; omitted, it reads every complete level present.

    The imbalance is reported at the TOUCH and CUMULATIVELY, because they
    answer different questions and routinely disagree. Touch imbalance
    predicts the next tick; cumulative imbalance predicts where a size order
    ends up. A book that is bid at level 0 and heavily offered behind it is
    about to tick up and is a bad place to buy size.
    """
    if not isinstance(book, pd.DataFrame) or book.empty:
        raise ValidationError("book must be a non-empty DataFrame of snapshots.")

    available = _levels_present(book)
    if available == 0:
        raise ValidationError(
            "no complete level found. This function reads the "
            "`DataProvider.get_order_book` contract: bid_price_0, bid_size_0, "
            "ask_price_0, ask_size_0 and upward. A top-of-book quote panel "
            "has bid_price/ask_price and is a different shape."
        )
    depth = available if levels is None else min(int(levels), available)
    if depth < 1:
        raise ValidationError(f"levels={levels} must be at least 1.")

    bid_prices = np.column_stack(
        [book[f"bid_price_{i}"].to_numpy(dtype=float) for i in range(depth)]
    )
    bid_sizes = np.column_stack(
        [book[f"bid_size_{i}"].to_numpy(dtype=float) for i in range(depth)]
    )
    ask_prices = np.column_stack(
        [book[f"ask_price_{i}"].to_numpy(dtype=float) for i in range(depth)]
    )
    ask_sizes = np.column_stack(
        [book[f"ask_size_{i}"].to_numpy(dtype=float) for i in range(depth)]
    )

    touch_bid, touch_ask = bid_prices[:, 0], ask_prices[:, 0]
    touch_bid_size, touch_ask_size = bid_sizes[:, 0], ask_sizes[:, 0]

    warnings: List[str] = []
    crossed = touch_bid >= touch_ask
    if crossed.any():
        warnings.append(
            f"{int(crossed.sum())} of {len(book)} snapshots are crossed or "
            "locked (bid >= ask). Those are excluded from the spread and "
            "microprice statistics rather than averaged in -- a crossed book "
            "is a stale side or a feed artefact, and its negative spread "
            "drags a mean toward zero."
        )
    usable = ~crossed

    mid = (touch_bid + touch_ask) / 2.0
    micro = microprice(touch_bid, touch_bid_size, touch_ask, touch_ask_size)
    spread = touch_ask - touch_bid
    with np.errstate(invalid="ignore", divide="ignore"):
        spread_bps = np.where(mid > 0, spread / mid * 10_000.0, np.nan)
        # Where the microprice sits inside the spread: 0 is the bid, 1 the
        # ask, 0.5 the mid. A pure position, so it is comparable across
        # names and tick sizes in a way the microprice itself is not.
        lean = np.where(spread > 0, (micro - touch_bid) / spread, np.nan)

    touch_total = touch_bid_size + touch_ask_size
    with np.errstate(invalid="ignore", divide="ignore"):
        touch_imbalance = np.where(
            touch_total > 0,
            (touch_bid_size - touch_ask_size) / touch_total,
            np.nan,
        )
    cumulative_bid = bid_sizes.sum(axis=1)
    cumulative_ask = ask_sizes.sum(axis=1)
    cumulative_total = cumulative_bid + cumulative_ask
    with np.errstate(invalid="ignore", divide="ignore"):
        cumulative_imbalance = np.where(
            cumulative_total > 0,
            (cumulative_bid - cumulative_ask) / cumulative_total,
            np.nan,
        )

    slope = _depth_slope(bid_prices, bid_sizes, ask_prices, ask_sizes, mid, depth)

    if depth == 1:
        warnings.append(
            "Only one level was read, so the depth slope is undefined and is "
            "reported as null. A one-level book is top-of-book: its "
            "imbalance is real, but every question about what sits behind "
            "the touch has no data behind it."
        )
    # COMPARED ONLY WHERE BOTH ARE A DIRECTION.
    #
    # `np.sign(a) != np.sign(b)` yields a BOOL array, and `nanmean` never
    # skips a bool -- there is no NaN in one. So every snapshot where an
    # imbalance was undefined counted as a disagreement, because
    # `sign(nan) != sign(nan)` is True. On an all-zero-size book, where
    # both imbalances are None, this reported "Touch and cumulative
    # imbalance point opposite ways in 100% of snapshots ... fills badly"
    # -- a confident directional claim about a book with no data in it.
    #
    # A sign of exactly 0 is an evenly balanced side, not a direction, so
    # those are excluded too rather than counted as pointing the other way.
    touch_sign = np.sign(touch_imbalance)
    cumulative_sign = np.sign(cumulative_imbalance)
    comparable = (
        np.isfinite(touch_imbalance)
        & np.isfinite(cumulative_imbalance)
        & (touch_sign != 0)
        & (cumulative_sign != 0)
    )
    n_comparable = int(comparable.sum())
    disagree = (
        float(np.mean(touch_sign[comparable] != cumulative_sign[comparable]))
        if n_comparable
        else 0.0
    )
    if disagree > 0.2 and n_comparable:
        warnings.append(
            f"Touch and cumulative imbalance point opposite ways in "
            f"{disagree:.0%} of the {n_comparable} snapshots where both are "
            f"defined and neither is exactly balanced. They answer different "
            f"questions -- "
            "the touch predicts the next tick, the cumulative predicts where "
            "size ends up -- and a book bid at the touch with weight behind "
            "the offer is exactly the one that ticks up and fills badly."
        )
    warnings.append(
        "The microprice weights each side by the OPPOSITE side's resting "
        "size, because the heavy side is the side that absorbs. On a "
        "balanced book it equals the mid exactly."
    )

    return {
        "n_snapshots": int(len(book)),
        "levels_available": int(available),
        "levels_read": int(depth),
        "n_crossed": int(crossed.sum()),
        "mean_spread": _mean(spread[usable]),
        "mean_spread_bps": _mean(spread_bps[usable]),
        "mean_mid": _mean(mid[usable]),
        "mean_microprice": _mean(micro[usable]),
        "mean_microprice_lean": _mean(lean[usable]),
        "mean_touch_imbalance": _mean(touch_imbalance),
        "mean_cumulative_imbalance": _mean(cumulative_imbalance),
        "mean_touch_size": _mean(touch_total),
        "mean_cumulative_size": _mean(cumulative_total),
        "depth_slope": slope,
        "warnings": warnings,
    }


def book_dynamics(book: pd.DataFrame) -> Dict[str, Any]:
    """
    Order-flow imbalance and update rates, BETWEEN consecutive snapshots.

    `book_metrics` answers what a book holds at an instant. This answers
    what changed, which is a different question and the one that predicts.

    OFI IS NOT SIGNED VOLUME. `get_order_flow_imbalance` in this library
    computes signed return times volume from BARS, which is a proxy for
    this and a different quantity. What is computed here is the
    Cont-Kukanov-Stoikov definition, from the book itself:

        e_n = 1{P_b(n) >= P_b(n-1)} q_b(n) - 1{P_b(n) <= P_b(n-1)} q_b(n-1)
            - 1{P_a(n) <= P_a(n-1)} q_a(n) + 1{P_a(n) >= P_a(n-1)} q_a(n-1)

    Read it a term at a time. If the bid price ROSE, someone posted a more
    aggressive bid and the whole new size is demand: +q_b(n). If it FELL,
    the old bid was pulled or hit and the whole old size left: -q_b(n-1).
    If it is unchanged both fire and the result is the size CHANGE, which
    is the intuitive case and the only one a naive implementation gets
    right. The ask terms are the mirror, negated because ask-side pressure
    pushes the other way.

    AT THE TOUCH, DELIBERATELY. This is the level-0 definition, which is
    the one with published evidence behind it. A multi-level OFI is a
    different quantity with a different decay, not a better version of
    this one, so it is not silently substituted here.

    A CROSSED OR INCOMPLETE SNAPSHOT breaks the pairwise comparison rather
    than merely being odd, so the pairs it participates in are dropped and
    counted. Interpolating across a gap would invent a book transition
    nobody observed.
    """
    if not isinstance(book, pd.DataFrame) or len(book) < 2:
        raise ValidationError(
            "book dynamics need at least two snapshots: every measure here "
            "is a comparison between consecutive states, and one snapshot "
            "has nothing to compare with."
        )
    for column in ("bid_price_0", "bid_size_0", "ask_price_0", "ask_size_0"):
        if column not in book.columns:
            raise ValidationError(
                f"book dynamics need {column!r}. This reads the "
                "`DataProvider.get_order_book` contract at the touch."
            )

    bid_p = pd.to_numeric(book["bid_price_0"], errors="coerce").to_numpy(float)
    bid_q = pd.to_numeric(book["bid_size_0"], errors="coerce").to_numpy(float)
    ask_p = pd.to_numeric(book["ask_price_0"], errors="coerce").to_numpy(float)
    ask_q = pd.to_numeric(book["ask_size_0"], errors="coerce").to_numpy(float)

    usable = (
        np.isfinite(bid_p)
        & np.isfinite(bid_q)
        & np.isfinite(ask_p)
        & np.isfinite(ask_q)
        & (bid_p < ask_p)
    )
    pair = usable[1:] & usable[:-1]
    n_pairs = int(pair.sum())
    warnings: List[str] = []
    dropped = int(len(book) - 1 - n_pairs)
    if dropped:
        warnings.append(
            f"NOTE: {dropped} of {len(book) - 1} consecutive pairs were "
            "dropped -- one side was non-finite or the book was crossed. A "
            "pairwise comparison across such a snapshot describes a "
            "transition nobody observed."
        )
    if n_pairs == 0:
        raise ValidationError(
            "no usable consecutive pair: every snapshot has a non-finite "
            "touch or a crossed book, so no transition can be measured."
        )

    prev_bid_p, curr_bid_p = bid_p[:-1][pair], bid_p[1:][pair]
    prev_bid_q, curr_bid_q = bid_q[:-1][pair], bid_q[1:][pair]
    prev_ask_p, curr_ask_p = ask_p[:-1][pair], ask_p[1:][pair]
    prev_ask_q, curr_ask_q = ask_q[:-1][pair], ask_q[1:][pair]

    contribution = (
        (curr_bid_p >= prev_bid_p) * curr_bid_q
        - (curr_bid_p <= prev_bid_p) * prev_bid_q
        - (curr_ask_p <= prev_ask_p) * curr_ask_q
        + (curr_ask_p >= prev_ask_p) * prev_ask_q
    )

    mid = (bid_p + ask_p) / 2.0
    mid_moves = int((np.diff(mid[usable]) != 0).sum()) if usable.sum() > 1 else 0
    spread_changes = (
        int((np.diff((ask_p - bid_p)[usable]) != 0).sum()) if usable.sum() > 1 else 0
    )

    seconds = None
    if "timestamp" in book.columns:
        stamps = pd.to_datetime(book["timestamp"], errors="coerce").dropna()
        if len(stamps) > 1:
            span = (stamps.max() - stamps.min()).total_seconds()
            seconds = float(span) if span > 0 else None
    if seconds is None:
        warnings.append(
            "NOTE: no usable timestamp span, so the per-second rates are "
            "null rather than zero. Zero would read as a still book."
        )

    total = float(contribution.sum())
    return {
        "n_snapshots": int(len(book)),
        "n_pairs": n_pairs,
        "n_pairs_dropped": dropped,
        "elapsed_seconds": seconds,
        # The window total, which is the quantity the literature regresses
        # a price change on. The mean is reported beside it because the
        # total scales with how long you looked.
        "ofi": total,
        "ofi_per_update": float(contribution.mean()),
        "ofi_per_second": (total / seconds) if seconds else None,
        # How hard the book is being worked. Counting SNAPSHOTS measures
        # the sampling rate if the feed is sampled, which is why the count
        # is reported beside the rate rather than only the rate.
        "updates_per_second": (len(book) / seconds) if seconds else None,
        "mid_changes": mid_moves,
        "mid_changes_per_second": (mid_moves / seconds) if seconds else None,
        "spread_changes": spread_changes,
        "warnings": warnings,
    }


def depth_profile(
    book: pd.DataFrame, *, levels: Optional[int] = None
) -> Dict[str, Any]:
    """
    Resting size and distance from the mid, level by level.

    This is the shape a size order actually walks. Reported per level rather
    than summed, because the sum is what makes an illiquid book look deep:
    ten levels of a hundred shares each is a thousand shares and is not the
    same market as one level of a thousand.
    """
    available = _levels_present(book)
    if available == 0:
        raise ValidationError(
            "no complete level found; see `book_metrics` for the expected "
            "column contract."
        )
    depth = available if levels is None else min(int(levels), available)

    mid = (
        book["bid_price_0"].to_numpy(dtype=float)
        + book["ask_price_0"].to_numpy(dtype=float)
    ) / 2.0

    rows: List[Dict[str, Any]] = []
    for i in range(depth):
        bid_price = book[f"bid_price_{i}"].to_numpy(dtype=float)
        ask_price = book[f"ask_price_{i}"].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            bid_distance = np.where(mid > 0, (mid - bid_price) / mid * 10_000.0, np.nan)
            ask_distance = np.where(mid > 0, (ask_price - mid) / mid * 10_000.0, np.nan)
        rows.append(
            {
                "level": i,
                "mean_bid_size": _mean(book[f"bid_size_{i}"].to_numpy(dtype=float)),
                "mean_ask_size": _mean(book[f"ask_size_{i}"].to_numpy(dtype=float)),
                "mean_bid_distance_bps": _mean(bid_distance),
                "mean_ask_distance_bps": _mean(ask_distance),
            }
        )

    return {
        "n_snapshots": int(len(book)),
        "levels_read": int(depth),
        "profile": rows,
        "warnings": [
            "Size is reported PER LEVEL rather than summed. A sum is what "
            "makes a thin book look deep: ten levels of a hundred is not the "
            "same market as one level of a thousand, and only the first can "
            "be lifted in one trade."
        ],
    }


# ── internals ───────────────────────────────────────────────────────────


def _depth_slope(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    mid: np.ndarray,
    depth: int,
) -> Optional[float]:
    """
    Cumulative size per basis point of distance from the mid, both sides.

    A regression through the origin of cumulative size on distance: the
    slope IS the liquidity density, in shares per bp. Through the origin
    rather than with an intercept, because at zero distance the cumulative
    size is zero by construction and fitting an intercept lets it come back
    non-zero, which would claim size resting at a price that is not a price.
    """
    if depth < 2:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        bid_distance = np.where(
            mid[:, None] > 0,
            (mid[:, None] - bid_prices) / mid[:, None] * 10_000.0,
            np.nan,
        )
        ask_distance = np.where(
            mid[:, None] > 0,
            (ask_prices - mid[:, None]) / mid[:, None] * 10_000.0,
            np.nan,
        )
    distance = np.concatenate([bid_distance, ask_distance], axis=1).ravel()
    size = np.concatenate(
        [np.cumsum(bid_sizes, axis=1), np.cumsum(ask_sizes, axis=1)], axis=1
    ).ravel()

    ok = np.isfinite(distance) & np.isfinite(size) & (distance > 0)
    if ok.sum() < 2:
        return None
    x, y = distance[ok], size[ok]
    denominator = float(x @ x)
    if denominator <= 0:
        return None
    return float(x @ y / denominator)


def _mean(values: np.ndarray) -> Optional[float]:
    """A nan-safe mean that returns None rather than NaN for an empty slice."""
    if values.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        out = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return None if not math.isfinite(out) else out
