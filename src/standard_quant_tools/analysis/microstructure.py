"""
Tick-level microstructure estimators: spreads, trade signing, and impact.

WHY THIS MODULE EXISTS. `DataProvider.get_trades` and `get_quotes` have
been part of the data interface since the tick capability was added, and
nothing in this library consumed them. Meanwhile `backtest/liquidity.py`
estimates the bid/ask spread from OHLCV bars via Corwin-Schultz, and
`get_liquidity_metrics` says so in as many words: those are proxies,
present precisely because the real data is usually absent.

This module is what the real data buys. When a provider actually serves
trades and quotes, the spread stops being estimated and starts being
measured -- and the proxies become checkable against it, which is the one
thing a proxy can never do for itself.

WHAT IT DOES NOT DO. Nothing here synthesizes ticks from bars. A "trade"
derived from an OHLCV row is a fiction that every measure below would then
treat as fact, so a caller without a tick feed gets an error naming the
missing capability rather than a number. Quotes are TOP OF BOOK, because
only provider='databento' offers depth; queue position and resting size at a
level are out of reach, not approximated.

THE SIGNING PROBLEM. Every spread decomposition needs to know whether a
trade was buyer- or seller-initiated, and no feed says. `sign_trades`
implements Lee-Ready (1991): compare the price to the prevailing midpoint,
and fall back to the tick test only at the midpoint itself, where the quote
comparison is uninformative. The prevailing quote is the last one strictly
BEFORE the trade -- using a quote stamped at or after it lets information
from the trade leak into its own classification, which biases every
downstream estimate toward looking more informed than it was.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

#: Columns each input frame must carry. Checked up front so a missing
#: column fails naming itself rather than as a KeyError several estimators
#: deep.
_TRADE_COLUMNS = ("price", "size")
_QUOTE_COLUMNS = ("bid_price", "ask_price")


def _require_frame(frame: pd.DataFrame, columns: tuple, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValidationError(f"{label} is empty; there is nothing to measure")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"{label} is missing column(s) {missing}; expected {list(columns)}"
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValidationError(
            f"{label} must be indexed by timestamp — every estimator here "
            "matches trades to the quote that preceded them, which needs a "
            "real time index rather than a positional one."
        )
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    return frame


def _classified(signs: pd.Series) -> pd.Series:
    """Keep only the trades a rule actually decided.

    A NaN sign means no rule applied -- the first trade of a window has no
    prior price for the tick test, and a trade at the midpoint with no
    prevailing quote has nothing from either. Zero means the tests
    cancelled. Both are dropped rather than defaulted: a side assigned by
    coin flip puts noise into every size-weighted mean while leaving the
    output looking complete.
    """
    usable = signs.notna() & (signs != 0)
    return signs[usable].astype(int)


def quoted_spread(quotes: pd.DataFrame) -> pd.DataFrame:
    """
    Per-quote spread and midpoint.

    The quoted spread is what a taker would pay to cross at that instant,
    before any of it is recovered. It is an upper bound on realized cost
    for a passive strategy and a lower bound for an aggressive one that
    moves the market -- which is why it is reported alongside, not instead
    of, the effective spread.

    Returns a frame with `mid`, `spread`, `spread_bps` and `imbalance`,
    indexed like `quotes`. Rows with a non-positive or crossed quote are
    dropped: a negative spread is a feed artifact (a stale side, a
    cross-venue crossed book), and averaging one in would silently pull
    every summary toward zero.
    """
    quotes = _require_frame(quotes, _QUOTE_COLUMNS, "quotes")
    bid = quotes["bid_price"].astype(float)
    ask = quotes["ask_price"].astype(float)

    mid = (bid + ask) / 2.0
    spread = ask - bid
    usable = (spread > 0) & (mid > 0) & np.isfinite(mid) & np.isfinite(spread)

    out = pd.DataFrame(
        {
            "mid": mid.where(usable),
            "spread": spread.where(usable),
            "spread_bps": (spread / mid * 10_000.0).where(usable),
        }
    )
    if "bid_size" in quotes.columns and "ask_size" in quotes.columns:
        bid_size = quotes["bid_size"].astype(float)
        ask_size = quotes["ask_size"].astype(float)
        total = bid_size + ask_size
        # Positive = more size resting on the bid than the ask.
        out["imbalance"] = ((bid_size - ask_size) / total).where(total > 0)
    return out.dropna(subset=["mid"])


def sign_trades(
    trades: pd.DataFrame,
    quotes: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Classify each trade as buyer-initiated (+1) or seller-initiated (-1).

    Lee-Ready (1991): a trade above the prevailing midpoint is
    buyer-initiated, below it seller-initiated, and exactly at the midpoint
    is decided by the tick test (higher than the last different price =
    buy). With no quotes at all, the tick test carries every trade, which
    is materially less accurate and is why `quotes` is optional but
    strongly preferred.

    The prevailing quote is the last one STRICTLY BEFORE the trade.
    Matching a quote stamped at the same nanosecond would let the trade's
    own effect on the book classify it, biasing every downstream spread
    toward zero and every impact measure toward zero as well.

    Returns a Series of +1/-1 aligned to `trades.index`. Trades that
    neither test can classify -- an opening trade at the midpoint with no
    prior price -- are dropped rather than defaulted, because assigning
    them a side would put a coin flip into a mean.
    """
    return _classified(_sign_series(trades, quotes))


def _signs_positional(
    trades: pd.DataFrame, quotes: Optional[pd.DataFrame] = None
) -> np.ndarray:
    """
    Every trade's sign in the trades' OWN ROW ORDER: NaN where no rule
    decided, 0 where the rules cancelled, +1/-1 otherwise.

    Real tapes repeat timestamps -- 32% of prints in a live AAPL minute --
    and a label-based `.loc[signs.index]` on such an index either raises
    ('cannot reindex on an axis with duplicate labels') or fans rows out
    by label: `_signed_volume` reported a net imbalance of -229,340 on a
    tape whose whole volume was 64,780 (findings D7). Positions never
    collide, so every consumer here aligns by position.
    """
    return _sign_series(trades, quotes).to_numpy(dtype=float)


def _sign_series(
    trades: pd.DataFrame, quotes: Optional[pd.DataFrame] = None
) -> pd.Series:
    """The full-length sign series behind `sign_trades`, one entry per
    trade row in order, undecided trades left as NaN."""
    trades = _require_frame(trades, _TRADE_COLUMNS, "trades")
    price = trades["price"].astype(float)

    # Tick test, with the zero-tick rule: a print at the same price as the
    # one before it inherits the sign of the last non-zero tick. Dropping
    # zero ticks instead would discard every repeat in a run, which on a
    # real tape is most of it -- and would bias the surviving sample toward
    # the prints that moved the price, i.e. toward looking informed.
    tick_sign = np.sign(price.diff()).replace(0.0, np.nan).ffill()

    if quotes is None:
        return pd.Series(tick_sign.to_numpy(), index=price.index, dtype="float64")

    q = quoted_spread(quotes)
    if q.empty:
        return pd.Series(tick_sign.to_numpy(), index=price.index, dtype="float64")

    # The prevailing mid per trade, matched BY POSITION: the trades are
    # sorted stably with their row number carried along, matched to the
    # last quote strictly before them, and the mids scattered back by row
    # number. A repeated timestamp is then just two rows.
    n = len(price)
    order = np.argsort(price.index.to_numpy(), kind="stable")
    left = pd.DataFrame({"_pos": np.arange(n)[order]}, index=price.index[order])
    matched = pd.merge_asof(
        left,
        q[["mid"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
        allow_exact_matches=False,
    )
    mid = np.full(n, np.nan)
    mid[matched["_pos"].to_numpy()] = matched["mid"].to_numpy(dtype=float)
    quote_sign = np.sign(price.to_numpy(dtype=float) - mid)
    has_mid = np.isfinite(mid)
    # At the midpoint (or with no prevailing quote) the quote comparison
    # says nothing, so the tick test decides.
    signs = np.where((quote_sign != 0) & has_mid, quote_sign, tick_sign.to_numpy())
    return pd.Series(signs, index=price.index, dtype="float64")


def effective_spread(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    realized_horizon: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """
    Per-trade effective spread, and optionally its decomposition.

    The effective spread is 2 * side * (price - mid) / mid: what the trade
    actually paid relative to the midpoint that stood before it, doubled to
    put it on the same footing as a quoted (full) spread. It differs from
    the quoted spread whenever trades execute inside the quotes or sweep
    through them, which is most of the time -- so a backtest that charges
    the quoted spread is not charging what trading costs.

    With `realized_horizon`, two more columns appear:

      realized_spread_bps  2 * side * (price - mid_after) / mid, where
                           mid_after is the midpoint one horizon later.
                           This is what the liquidity provider KEPT.
      price_impact_bps     effective - realized: what the trade moved the
                           market, i.e. what it paid for being informed.

    The split matters because the two halves have opposite implications. A
    wide effective spread that is mostly realized means the venue is
    expensive; one that is mostly impact means the strategy is trading
    against people who notice it, and trading smaller will help while
    switching venue will not.
    """
    trades = _require_frame(trades, _TRADE_COLUMNS, "trades")
    q = quoted_spread(quotes)
    if q.empty:
        raise ValidationError(
            "no usable quotes: every row had a non-positive or crossed "
            "spread, so there is no midpoint to measure trades against."
        )

    # By position, not by label: `.loc[side.index]` raised on any tape
    # with a repeated timestamp (findings D7).
    signs = _signs_positional(trades, quotes)
    keep = np.isfinite(signs) & (signs != 0)
    frame = pd.DataFrame(
        {
            "price": trades["price"].astype(float).to_numpy()[keep],
            "size": trades["size"].astype(float).to_numpy()[keep],
            "side": signs[keep].astype(int),
        },
        index=trades.index[keep],
    )

    matched = pd.merge_asof(
        frame.sort_index(),
        q[["mid", "spread_bps"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
        allow_exact_matches=False,
    ).dropna(subset=["mid"])

    matched["effective_spread_bps"] = (
        2.0 * matched["side"] * (matched["price"] - matched["mid"]) / matched["mid"]
    ) * 10_000.0

    if realized_horizon is not None:
        if realized_horizon <= pd.Timedelta(0):
            raise ValidationError(
                f"realized_horizon must be positive, got {realized_horizon}. "
                "The realized spread compares against a midpoint AFTER the "
                "trade; a zero or negative horizon compares it with itself."
            )
        future = q[["mid"]].rename(columns={"mid": "mid_after"})
        # Shift the future quotes back by the horizon so a backward asof
        # from the trade lands on the first quote at or after t+horizon.
        future.index = future.index - realized_horizon
        matched = pd.merge_asof(
            matched.sort_index(),
            future.sort_index(),
            left_index=True,
            right_index=True,
            direction="forward",
            allow_exact_matches=True,
        )
        matched["realized_spread_bps"] = (
            2.0
            * matched["side"]
            * (matched["price"] - matched["mid_after"])
            / matched["mid"]
        ) * 10_000.0
        matched["price_impact_bps"] = (
            matched["effective_spread_bps"] - matched["realized_spread_bps"]
        )

    return matched


def microstructure_summary(
    trades: pd.DataFrame,
    quotes: Optional[pd.DataFrame] = None,
    realized_horizon: Optional[pd.Timedelta] = None,
) -> Dict[str, Any]:
    """
    One symbol's tick-derived liquidity profile, size-weighted.

    Averages here are weighted by trade size, not by trade count. An equal
    weight over prints answers "what did a typical print cost", which is
    dominated by the odd-lot tail; weighting by size answers "what did a
    typical SHARE cost", which is the question a strategy sizing a position
    is actually asking. Both are reported so the difference is visible.
    """
    trades = _require_frame(trades, _TRADE_COLUMNS, "trades")
    size = trades["size"].astype(float)
    summary: Dict[str, Any] = {
        "n_trades": int(len(trades)),
        "total_volume": float(size.sum()),
        "vwap": (
            float((trades["price"].astype(float) * size).sum() / size.sum())
            if size.sum() > 0
            else float("nan")
        ),
        "first_trade": str(trades.index[0]),
        "last_trade": str(trades.index[-1]),
    }

    if quotes is None:
        summary["notes"] = [
            "No quotes supplied: trades were signed by the tick test alone, "
            "which is materially less accurate than Lee-Ready, and no "
            "spread could be measured."
        ]
        signs = _signs_positional(trades)
        keep = np.isfinite(signs) & (signs != 0)
        sized = size.to_numpy(dtype=float)[keep]
        summary["n_signed"] = int(keep.sum())
        summary["buy_volume_fraction"] = (
            float(sized[signs[keep] > 0].sum() / sized.sum())
            if keep.any() and sized.sum() > 0
            else float("nan")
        )
        return summary

    q = quoted_spread(quotes)
    summary["n_quotes"] = int(len(q))
    summary["quoted_spread_bps_mean"] = float(q["spread_bps"].mean())
    summary["quoted_spread_bps_median"] = float(q["spread_bps"].median())
    if "imbalance" in q.columns:
        summary["quote_imbalance_mean"] = float(q["imbalance"].mean())

    per_trade = effective_spread(trades, quotes, realized_horizon)
    weights = per_trade["size"]
    summary["n_signed"] = int(len(per_trade))
    summary["buy_volume_fraction"] = (
        float(weights[per_trade["side"] > 0].sum() / weights.sum())
        if weights.sum() > 0
        else float("nan")
    )

    def _weighted(column: str) -> float:
        values = per_trade[column]
        usable = values.notna() & (weights > 0)
        if not usable.any():
            return float("nan")
        return float(
            np.average(values[usable].to_numpy(), weights=weights[usable].to_numpy())
        )

    summary["effective_spread_bps_mean"] = float(
        per_trade["effective_spread_bps"].mean()
    )
    summary["effective_spread_bps_size_weighted"] = _weighted("effective_spread_bps")
    if realized_horizon is not None:
        summary["realized_spread_bps_size_weighted"] = _weighted("realized_spread_bps")
        summary["price_impact_bps_size_weighted"] = _weighted("price_impact_bps")
    return summary


def trade_size_profile(trades: pd.DataFrame, buckets: int = 5) -> Dict[str, Any]:
    """
    How this symbol's volume is distributed across trade sizes.

    Reported as size QUANTILES rather than fixed share-count buckets,
    because a fixed grid that suits one symbol misreads another by orders
    of magnitude. The number that usually matters is the share of volume in
    the largest bucket: a book where most volume arrives in a few large
    prints behaves nothing like one where the same volume arrives as
    thousands of small ones, even at an identical daily total.
    """
    trades = _require_frame(trades, _TRADE_COLUMNS, "trades")
    if buckets < 2:
        raise ValidationError(f"buckets must be at least 2, got {buckets}")

    size = trades["size"].astype(float)
    size = size[size > 0]
    if size.empty:
        raise ValidationError("no trades with a positive size")

    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(size.quantile(quantiles).to_numpy())
    if len(edges) < 2:
        # Every trade is the same size; one bucket is the honest answer.
        return {
            "n_trades": int(len(size)),
            "total_volume": float(size.sum()),
            "median_size": float(size.median()),
            "buckets": [
                {
                    "lower": float(edges[0]),
                    "upper": float(edges[0]),
                    "n_trades": int(len(size)),
                    "volume_fraction": 1.0,
                }
            ],
            "largest_bucket_volume_fraction": 1.0,
        }

    labels = pd.cut(size, bins=edges, include_lowest=True)
    grouped = size.groupby(labels, observed=True)
    total = size.sum()
    rows = [
        {
            "lower": float(interval.left),
            "upper": float(interval.right),
            "n_trades": int(group.count()),
            "volume_fraction": round(float(group.sum() / total), 6),
        }
        for interval, group in grouped
    ]
    return {
        "n_trades": int(len(size)),
        "total_volume": float(total),
        "median_size": float(size.median()),
        "buckets": rows,
        "largest_bucket_volume_fraction": rows[-1]["volume_fraction"] if rows else 0.0,
    }


def _session_minutes(session) -> "tuple[int, int]":
    def _minutes(label: str) -> int:
        hours, minutes = str(label).split(":")
        return int(hours) * 60 + int(minutes)

    lo, hi = _minutes(session[0]), _minutes(session[1])
    if hi <= lo:
        raise ValidationError(
            f"session={tuple(session)!r} must end after it starts (HH:MM, HH:MM)."
        )
    return lo, hi


def intraday_volume_profile(
    trades: pd.DataFrame,
    freq: str = "30min",
    *,
    session: "tuple[str, str]" = ("09:30", "16:00"),
    exchange_timezone: str = "America/New_York",
) -> Dict[str, Any]:
    """
    Volume by time of day, as a fraction of the session.

    A strategy that trades a fixed share of ADV at a fixed time is not
    taking a fixed share of the liquidity available then: US equity volume
    is famously U-shaped, so the same order is a far larger share of the
    book at midday than at the close. This reports the actual shape rather
    than assuming one.
    """
    trades = _require_frame(trades, _TRADE_COLUMNS, "trades")
    size = trades["size"].astype(float)
    # A tape stamped with a timezone is bucketed over the REGULAR SESSION
    # in the exchange's clock, and the extended-hours share is reported
    # beside it. Bucketing over the observed range of a live feed put the
    # open at 4am and the close at 8pm, reported a 0% open and close, and
    # warned the caller that THEIR data was unusual (findings). A naive
    # index is taken as already in session time and profiled as given.
    extended_share: Optional[float] = None
    index = trades.index
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        local = index.tz_convert(exchange_timezone)
        minutes = local.hour * 60 + local.minute
        lo_m, hi_m = _session_minutes(session)
        regular = (minutes >= lo_m) & (minutes < hi_m)
        total_all = float(size.sum())
        extended_share = (
            float(size.to_numpy()[~regular].sum() / total_all)
            if total_all > 0
            else None
        )
        if not regular.any():
            raise ValidationError(
                f"no trades inside the {session[0]}-{session[1]} session in "
                f"{exchange_timezone}; every print is extended hours."
            )
        size = pd.Series(
            size.to_numpy()[regular], index=local.tz_localize(None)[regular]
        )
        index = size.index
    by_bucket = size.groupby(index.floor(freq).time).sum()
    total = float(by_bucket.sum())
    if total <= 0:
        raise ValidationError("no positive volume to profile")
    return {
        "freq": freq,
        "session": list(session),
        "extended_hours_share": extended_share,
        "buckets": [
            {"time": str(when), "volume_fraction": round(float(volume / total), 6)}
            for when, volume in by_bucket.items()
        ],
        "peak_time": str(by_bucket.idxmax()),
        "peak_volume_fraction": round(float(by_bucket.max() / total), 6),
    }
