"""
Which part of the market changed, not merely that it did.

"NVDA moved 1.4 sigma" is a summary of one channel — price — and it is the
channel that changes last. A liquidity event usually shows up first in the
spread, in the imbalance of arriving orders, or in depth disappearing from
one side, and by the time the mid price has moved 1.4 sigma the interesting
part is over. So this runs a change detector over SEVERAL channels and
reports which of them broke:

    NVDA
        price shock             low
        spread shock            high
        signed-volume shock     very high
        trade-intensity shock   high

ONE TOOL OVER A DECLARED CHANNEL SET, not one tool per channel. The channel
list is data; adding `depth_slope` is a table entry, not a thirteenth tool.
That is the same rule that keeps `STRATEGY_REGISTRY` from becoming twelve
backtest tools.

WHAT IS AVAILABLE TODAY. Six channels need only trades and quotes, which
providers in this library can actually serve. Seven need an order book, and
nothing here serves L2 yet — those are DECLARED with their requirement and
refuse by name rather than being omitted. An agent asking for `ofi` should
learn that the channel exists and what it needs, not that it was never
heard of.

WHY CUSUM AND NOT A Z-SCORE. A z-score asks "is this observation unusual",
which fires on a single tick and misses a sustained shift of half a sigma.
CUSUM accumulates, so it detects a persistent change in level that no single
observation would flag — which is what a liquidity event is. The cost is a
detection lag, and that is the right trade for a diagnostic rather than a
trigger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import has_no_dispersion

logger = logging.getLogger(__name__)

#: CUSUM slack, in standard deviations of the reference window. Drifts
#: smaller than this accumulate nothing, which is what stops the statistic
#: wandering off on noise alone. 0.5 is the textbook default and is a
#: deliberate choice rather than a tuned one -- a slack fitted to the sample
#: is a detector fitted to the sample.
DEFAULT_SLACK = 0.5

#: Decision threshold, calibrated rather than taken from a textbook.
#:
#: The textbook operating point of 5.0 is stated as an average RUN LENGTH --
#: one false alarm every so many observations -- and this tool asks a
#: different question: "did anything happen anywhere in this window". Any
#: fixed threshold alarms eventually, so a run-length figure becomes a
#: near-certainty over a long enough window. Measured on iid noise with no
#: change in it, at 5.0:
#:
#:     n=120    36% false alarms
#:     n=300    51%
#:     n=1000   82%
#:
#: Worse with more data, which is the tell. Calibrating instead for a ~5%
#: rate over the WHOLE window gives 8.6-9.3 across 42 to 1050 tested
#: observations -- essentially flat, because the maximum of the reflected
#: random walk has a steep exponential tail. So this is a better constant
#: rather than a formula, which is what the measurement supports.
#:
#: Lower it deliberately to trade false alarms for sensitivity; a diagnostic
#: that is read by a person can afford a looser threshold than one that
#: fires an alert.
DEFAULT_THRESHOLD = 9.0

#: Fraction of the series used to learn "normal". The baseline must come
#: from a REFERENCE window rather than the whole series: a shock included in
#: its own baseline inflates the standard deviation it is measured against
#: and hides itself, which is the single easiest way to build a detector
#: that finds nothing.
DEFAULT_REFERENCE_FRACTION = 0.3

#: Severity ladder, in multiples of the decision threshold. Reported as
#: words as well as a number because the number's units -- accumulated
#: standardized deviations -- mean nothing to a reader who has not just
#: read the CUSUM definition.
#: Below this coefficient of variation, the reference window did not observe
#: normal variation and every standardized quantity measured against it is
#: arithmetic rather than evidence. Measured: a frozen-spread series
#: produced a CUSUM peak of 286,431 while moving from 1.00 bps to 1.02 bps.
#: The detection is still reported -- a calm period before a real shock is
#: the case this tool is for -- but it is labelled.
DEGENERATE_BASELINE_CV = 1e-3

_SEVERITY = ((1.0, "low"), (2.0, "moderate"), (4.0, "high"))


@dataclass(frozen=True)
class Channel:
    """One measurable thing about the market, and what it needs to compute."""

    name: str
    requires: tuple
    description: str
    compute: Optional[Callable[..., pd.Series]] = None
    refused_because: Optional[str] = None

    @property
    def available(self) -> bool:
        """Whether this library can compute the channel at all today."""
        return self.compute is not None

    def why_unavailable(self) -> str:
        if self.refused_because:
            return self.refused_because
        return (
            f"{self.name!r} needs {' and '.join(self.requires)}, and no "
            "provider in this library serves an order book yet. The channel "
            "is declared rather than omitted so it is clear it exists and "
            "what it would take."
        )


# ── the channels themselves ─────────────────────────────────────────────


def _resample(frame: pd.DataFrame, freq: str) -> pd.core.resample.Resampler:
    """Resample by time, accepting either a timestamp COLUMN or a time index.

    Both shapes are in circulation: the provider methods return a time
    index, and a frame that has been through a JSON boundary comes back with
    a column. Guessing wrong raises deep inside pandas with a message about
    parsing 'size' as a date, which is not a clue anybody can act on."""
    indexed = frame.set_index("timestamp") if "timestamp" in frame.columns else frame
    return indexed.resample(freq)


def _mid_return(quotes: pd.DataFrame, freq: str, **_) -> pd.Series:
    """
    Log return of the mid per bucket, NOT the mid itself.

    A CUSUM over a price level fires on drift alone: the level is not
    stationary, so the statistic accumulates the random walk and triggers
    whether or not anything happened. Measured before this was fixed --
    eight of eight pure random walks with no shock triggered, which is a
    detector that has learned to say yes.

    The return series is stationary, so a trigger means the DISTRIBUTION of
    returns changed -- a volatility or drift regime shift -- which is the
    question worth asking about price in a liquidity context anyway.
    """
    mid = (quotes["bid_price"] + quotes["ask_price"]) / 2.0
    last = _resample(quotes.assign(_v=mid), freq)["_v"].last().dropna()
    return np.log(last).diff().dropna()


def _spread(quotes: pd.DataFrame, freq: str, **_) -> pd.Series:
    """
    Spread in BASIS POINTS of the mid, not in currency.

    A one-cent spread is enormous on a $3 stock and invisible on a $300 one,
    so a detector on the raw spread would fire on price level rather than on
    liquidity -- and would fire hardest on whichever name happens to be
    cheapest.
    """
    mid = (quotes["bid_price"] + quotes["ask_price"]) / 2.0
    bps = (quotes["ask_price"] - quotes["bid_price"]) / mid.replace(0.0, np.nan) * 1e4
    return _resample(quotes.assign(_v=bps), freq)["_v"].mean().dropna()


def _trade_intensity(trades: pd.DataFrame, freq: str, **_) -> pd.Series:
    return _resample(trades.assign(_v=1.0), freq)["_v"].sum()


def _signed_volume(
    trades: pd.DataFrame, quotes=None, *, freq: str = "1min", **_
) -> pd.Series:
    """
    Buy volume minus sell volume, signed by the Lee-Ready rule the library
    already implements rather than by a fresh one.
    """
    from standard_quant_tools.analysis.microstructure import sign_trades

    # sign_trades matches each trade to the quote that preceded it, so it
    # needs a real time INDEX rather than a timestamp column -- and it
    # returns a Series of signs, aligned to the trades it could classify.
    # Trades it could NOT classify are absent from that Series rather than
    # signed zero, so the reindex below drops them instead of counting them
    # as balanced flow.
    indexed = trades.set_index("timestamp") if "timestamp" in trades.columns else trades
    quoted = (
        quotes.set_index("timestamp")
        if quotes is not None and "timestamp" in quotes.columns
        else quotes
    )
    signs = sign_trades(indexed, quoted)
    sized = indexed.loc[signs.index, "size"] * signs
    return sized.resample(freq).sum()


def _realized_vol(trades: pd.DataFrame, freq: str, **_) -> pd.Series:
    """
    Realized volatility of trade prices within each bucket.

    Sample standard deviation of log returns, NOT annualized: the channel is
    compared against its own recent history, so a scaling constant common to
    every observation cancels and quoting one would only invite the reader
    to compare it against an annualized number from somewhere else.
    """
    prices = _resample(trades.assign(_v=trades["price"]), freq)["_v"]
    return prices.apply(
        lambda window: (
            float(np.std(np.diff(np.log(window)), ddof=1))
            if len(window) > 2
            else np.nan
        )
    ).dropna()


def _effective_spread(trades: pd.DataFrame, quotes: pd.DataFrame, freq: str, **_):
    """Twice the signed distance from the mid, in bps -- what a taker paid."""
    from standard_quant_tools.analysis.microstructure import effective_spread

    ti = trades.set_index("timestamp") if "timestamp" in trades.columns else trades
    qi = quotes.set_index("timestamp") if "timestamp" in quotes.columns else quotes
    frame = effective_spread(ti, qi)
    return frame["effective_spread_bps"].resample(freq).mean().dropna()


#: Every channel this tool knows about. Ones with `compute=None` are
#: declared but not computable here -- an agent asking for `ofi` should
#: learn that the channel exists and needs an order book, rather than that
#: the name was never heard of.
CHANNELS: Dict[str, Channel] = {
    c.name: c
    for c in (
        Channel(
            "mid_return",
            ("quotes",),
            "Log return of the mid per bucket. The channel that moves LAST -- "
            "by the time it has, the liquidity event is usually over.",
            _mid_return,
        ),
        Channel(
            "mid_price",
            ("quotes",),
            "The mid PRICE LEVEL. Declared so the trap is visible, and "
            "refused because a level is not stationary.",
            None,
            refused_because=(
                "a CUSUM over a price LEVEL fires on drift alone -- measured, "
                "eight of eight pure random walks with no shock triggered. "
                "Use 'mid_return', which is the same question asked of a "
                "stationary series."
            ),
        ),
        Channel(
            "spread",
            ("quotes",),
            "Quoted spread in basis points of the mid. Widens before price "
            "moves when makers step back.",
            _spread,
        ),
        Channel(
            "trade_intensity",
            ("trades",),
            "Trades per bucket. Rises on news and on forced flow alike.",
            _trade_intensity,
        ),
        Channel(
            "signed_volume",
            ("trades", "quotes"),
            "Buy volume minus sell volume, Lee-Ready signed. One-sided flow "
            "is what moves a book.",
            _signed_volume,
        ),
        Channel(
            "realized_vol",
            ("trades",),
            "Standard deviation of log trade-price changes within each "
            "bucket. Not annualized -- it is compared against its own past.",
            _realized_vol,
        ),
        Channel(
            "effective_spread",
            ("trades", "quotes"),
            "Twice the signed distance from the mid: what a taker actually "
            "paid, as opposed to what was quoted.",
            _effective_spread,
        ),
        # ── declared, not yet computable: every one needs L2 depth ──
        Channel(
            "microprice",
            ("orderbook",),
            "Depth-weighted mid. Leads the mid when the book is imbalanced.",
        ),
        Channel(
            "book_imbalance",
            ("orderbook",),
            "Top-of-book size imbalance between bid and ask.",
        ),
        Channel(
            "l5_imbalance",
            ("orderbook",),
            "Imbalance across the top five levels -- less noisy than L1 and "
            "harder to spoof.",
        ),
        Channel(
            "ofi",
            ("orderbook",),
            "Order flow imbalance: net change in bid depth minus ask depth "
            "across updates. The most direct measure of pressure.",
        ),
        Channel(
            "bid_depth",
            ("orderbook",),
            "Total size resting on the bid. Collapses first in a liquidity " "event.",
        ),
        Channel("ask_depth", ("orderbook",), "Total size resting on the ask."),
        Channel(
            "depth_slope",
            ("orderbook",),
            "How fast size accumulates away from the touch -- a shallow book "
            "is one that moves on small orders.",
        ),
        Channel(
            "cancel_rate",
            ("orderbook",),
            "Cancellations per unit time. Spikes when makers withdraw.",
        ),
    )
}


def available_channels() -> List[str]:
    return sorted(name for name, c in CHANNELS.items() if c.available)


def declared_channels() -> List[str]:
    return sorted(CHANNELS)


# ── the detector ────────────────────────────────────────────────────────


def cusum(
    series: pd.Series,
    *,
    slack: float = DEFAULT_SLACK,
    threshold: float = DEFAULT_THRESHOLD,
    reference_fraction: float = DEFAULT_REFERENCE_FRACTION,
) -> Dict[str, Any]:
    """
    Two-sided CUSUM over one channel.

    The baseline mean and standard deviation come from the FIRST
    `reference_fraction` of the series and nothing later. Standardizing
    against the whole series would put the shock into its own baseline,
    inflating the denominator it is measured against -- a detector that
    reliably fails to find exactly the events it was built for.

    Returns the running statistics, the first crossing, and the peak. The
    peak is what the severity ladder reads: the first crossing says when,
    and the peak says how badly.
    """
    values = pd.Series(series).astype(float).dropna()
    if len(values) < 10:
        raise ValidationError(
            f"CUSUM needs at least 10 observations, got {len(values)}. A "
            "shorter series cannot separate a level change from its own noise."
        )
    if not 0.05 <= reference_fraction <= 0.9:
        raise ValidationError("reference_fraction must be between 0.05 and 0.9")

    n_reference = max(5, int(len(values) * reference_fraction))
    reference = values.iloc[:n_reference]
    mean = float(reference.mean())
    std = float(reference.std(ddof=1))
    # `std == 0.0` is the absolute test. A reference window of a constant
    # 0.07 has std = 2.79e-17, which is not equal to zero, so the guard was
    # skipped and the statistic came back at 87.8 -- a confident detection
    # on a window with nothing in it. The same flat window answered
    # differently depending on the constant's binary representation.
    if has_no_dispersion(reference.to_numpy(), std):
        return {
            "triggered": False,
            "reason": (
                "the reference window is constant, so there is no scale to "
                "measure a change against. This is a statement about the "
                "window, not evidence that nothing happened."
            ),
            "n_observations": int(len(values)),
            "n_reference": int(n_reference),
        }

    standardized = (values - mean) / std
    # The recursion is genuinely sequential -- each step reads the previous
    # one -- so there is no numpy escape from the loop itself. What there IS
    # an escape from is `standardized.iloc[i]`, which was a pandas scalar
    # lookup executed twice per observation and dominated the runtime.
    # Reading the buffer once turns each step into array indexing.
    standardized_values = standardized.to_numpy(dtype=float)
    up = np.zeros(len(values))
    down = np.zeros(len(values))
    for i in range(1, len(values)):
        z = standardized_values[i]
        up[i] = max(0.0, up[i - 1] + z - slack)
        down[i] = max(0.0, down[i - 1] - z - slack)

    statistic = np.maximum(up, down)
    # Only the out-of-reference part can trigger: the reference window
    # defines normal, so a crossing inside it would be the detector firing
    # on the data that taught it what normal is.
    testable = np.arange(len(values)) >= n_reference
    crossings = np.flatnonzero(testable & (statistic >= threshold))
    peak_index = int(np.argmax(np.where(testable, statistic, -np.inf)))
    peak = float(statistic[peak_index])

    after = values.iloc[n_reference:]
    after_mean = float(after.mean()) if len(after) else float("nan")
    # The shift a reader can actually check, in the channel's own units and
    # in reference sigma. A peak statistic is accumulated and unbounded; a
    # before-and-after mean is not.
    shift = after_mean - mean
    coefficient_of_variation = abs(std / mean) if mean not in (0.0,) else float("inf")
    degenerate = coefficient_of_variation < DEGENERATE_BASELINE_CV

    notes = []
    if degenerate:
        notes.append(
            f"the reference window is nearly constant (mean {mean:.6g}, sd "
            f"{std:.3g}), so it never observed normal variation. The level "
            f"moved from {mean:.4g} to {after_mean:.4g} -- judge the shock "
            "from THAT, not from the statistic, which is a ratio to a "
            "denominator near zero and can be arbitrarily large."
        )

    return {
        "triggered": bool(crossings.size),
        "first_crossing": (
            str(values.index[int(crossings[0])]) if crossings.size else None
        ),
        "peak_statistic": peak,
        "peak_at": str(values.index[peak_index]),
        "direction": "up" if up[peak_index] >= down[peak_index] else "down",
        "severity": _severity(peak, threshold),
        "baseline_mean": mean,
        "baseline_std": std,
        "mean_after_reference": after_mean,
        "shift": float(shift),
        "shift_in_reference_sd": float(shift / std) if std else float("nan"),
        "degenerate_baseline": bool(degenerate),
        "notes": notes,
        "n_observations": int(len(values)),
        "n_reference": int(n_reference),
        "value_at_peak": float(values.iloc[peak_index]),
    }


def _severity(peak: float, threshold: float) -> str:
    if peak < threshold:
        return "none"
    ratio = peak / threshold
    for bound, label in _SEVERITY:
        if ratio < bound:
            return label
    return "very high"


def detect_liquidity_events(
    *,
    channels: Sequence[str],
    trades: Optional[pd.DataFrame] = None,
    quotes: Optional[pd.DataFrame] = None,
    freq: str = "1min",
    slack: float = DEFAULT_SLACK,
    threshold: float = DEFAULT_THRESHOLD,
    reference_fraction: float = DEFAULT_REFERENCE_FRACTION,
) -> Dict[str, Any]:
    """
    Run the detector over several channels and report which ones broke.

    A channel that cannot be computed is reported as `unavailable` with what
    it needs, never silently dropped. Dropping it would let a caller ask for
    order-flow imbalance, receive a clean report with no OFI row, and
    conclude the flow was balanced.
    """
    requested = list(dict.fromkeys(channels))
    if not requested:
        raise ValidationError("detect_liquidity_events: no channels requested")

    unknown = [c for c in requested if c not in CHANNELS]
    if unknown:
        from difflib import get_close_matches

        near = get_close_matches(unknown[0], declared_channels(), n=3)
        raise ValidationError(
            f"unknown channel(s) {unknown}. Declared channels are "
            f"{declared_channels()}." + (f" Did you mean: {near}?" if near else "")
        )

    frames = {"trades": trades, "quotes": quotes}
    results: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, str]] = []

    for name in requested:
        channel = CHANNELS[name]
        if not channel.available:
            unavailable.append(
                {
                    "channel": name,
                    "requires": ", ".join(channel.requires),
                    "reason": channel.why_unavailable(),
                }
            )
            continue
        missing = [f for f in channel.requires if frames.get(f) is None]
        if missing:
            unavailable.append(
                {
                    "channel": name,
                    "requires": ", ".join(channel.requires),
                    "reason": f"{name!r} needs {missing}, which was not supplied.",
                }
            )
            continue

        try:
            series = channel.compute(
                **{f: frames[f] for f in channel.requires}, freq=freq
            )
            outcome = cusum(
                series,
                slack=slack,
                threshold=threshold,
                reference_fraction=reference_fraction,
            )
        except ValidationError as exc:
            unavailable.append(
                {
                    "channel": name,
                    "requires": ", ".join(channel.requires),
                    "reason": str(exc),
                }
            )
            continue
        outcome["channel"] = name
        results.append(outcome)

    triggered = [r for r in results if r.get("triggered")]
    results.sort(key=lambda r: -r.get("peak_statistic", 0.0))

    return {
        "channels_run": [r["channel"] for r in results],
        "unavailable": unavailable,
        "results": results,
        "n_triggered": len(triggered),
        "worst_channel": results[0]["channel"] if results else None,
        "summary": [
            f"{r['channel']} shock: {r['severity']}"
            for r in results
            if r.get("triggered")
        ],
        "warnings": _warnings(results, unavailable),
    }


def _warnings(results, unavailable) -> List[str]:
    out = []
    degenerate = [r["channel"] for r in results if r.get("degenerate_baseline")]
    if degenerate:
        out.append(
            f"{degenerate}: the reference window barely varied, so the "
            "statistic is a ratio to a denominator near zero and can be "
            "arbitrarily large. Read `shift` and `mean_after_reference` for "
            "these channels rather than the severity."
        )
    if unavailable:
        names = sorted(u["channel"] for u in unavailable)
        out.append(
            f"{len(unavailable)} channel(s) could not be run: {names}. They "
            "are reported rather than dropped -- a clean report with a "
            "channel silently missing reads as that channel being quiet."
        )
    price = next((r for r in results if r["channel"] == "mid_return"), None)
    others = [r for r in results if r["channel"] != "mid_return" and r.get("triggered")]
    if others and price is not None and not price.get("triggered"):
        out.append(
            "liquidity channels fired while the mid price did not. That is "
            "the ordinary sequence rather than a contradiction -- the book "
            "thins before the price moves -- and it is the case a "
            "price-only monitor misses entirely."
        )
    return out


__all__ = [
    "CHANNELS",
    "DEGENERATE_BASELINE_CV",
    "DEFAULT_REFERENCE_FRACTION",
    "DEFAULT_SLACK",
    "DEFAULT_THRESHOLD",
    "Channel",
    "available_channels",
    "cusum",
    "declared_channels",
    "detect_liquidity_events",
]
