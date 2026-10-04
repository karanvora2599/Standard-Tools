"""
Which close-to-close moves the split screens name.

One rule, read by the backtest's screen (`backtest.screens`) and the
dataset build's (`data.quality.detect_split_like_moves`), so a backtest and
a dataset built from the same bars name the same bars. Here, beside
`constants`, because each screen importing it from the other's package
would load that whole package to read one function.

A move is named when it is larger than `SPLIT_SCREEN_THRESHOLD` (35%)
either way, or when it is a fall within `SPLIT_RATIO_TOLERANCE` (10% on a
log scale) of a 3:2 split: the close before the bar over the close on it is
between 1.357 and 1.658, a fall of 26.3% to 39.7%. A 3:2 split moves
-33.3%, below the threshold, and both screens used to miss it.

WHY ONLY 3:2 BELOW THE THRESHOLD. Every other listed ratio's band lies
beyond 35% (2:1 is a fall of 44.7% to 54.8%; the 2:3 reverse split a rise
of 35.7% to 65.8%). 4:3 (-25%) and 5:4 (-20%) splits exist, but their
bands reach down to falls of 17.1% and 11.6%, where ordinary moves live.
On 2,000 simulated series of 10,000 bars each -- GARCH(1,1) with
Student-t(4) innovations at a long-run daily volatility of 2%, 3% and 4%
-- the 3:2 band below 35% named an ordinary fall once per 75, 22 and 9
name-years, fewer than the 35% threshold already names (once per 57, 16
and 6); a 4:3 band would have named one once per 14, 4 and 2 name-years,
and a 5:4 band once per 4, 1.2 and 0.5.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import NamedTuple, Tuple

import numpy as np

from standard_quant_tools.constants import SPLIT_SCREEN_THRESHOLD

__all__ = [
    "RATIOS_NAMED_BELOW_THRESHOLD",
    "SPLIT_RATIO_TOLERANCE",
    "ScreenedMoves",
    "below_threshold_band",
    "screen_moves",
    "split_ratio_label",
]

#: How close, on a log scale, a move's implied ratio must sit to a listed
#: ratio to be named after it: |ln(implied / ratio)| <= 0.10, about 10%.
#: The six splits in a live 2022-2026 Databento window all sat within 3%
#: (the split day's own move is the rest). Where a move is within reach of
#: two listed ratios -- 25 and 30 are 0.18 apart -- the nearer is named.
SPLIT_RATIO_TOLERANCE = 0.10

#: The split ratios (new shares per old share) whose move is named even
#: below the threshold: a fall within the tolerance of one of these is a
#: split-sized move however small.
RATIOS_NAMED_BELOW_THRESHOLD: Tuple[float, ...] = (1.5,)


def split_ratio_label(ratio: float) -> str:
    """'10:1' for 10, '3:2' for 1.5, '1:10' for a 0.1 reverse split; a
    ratio with no small fraction is printed as a number."""
    fraction = Fraction(float(ratio)).limit_denominator(1000)
    if abs(float(fraction) - float(ratio)) > 1e-9 * max(1.0, abs(float(ratio))):
        return f"{float(ratio):g}:1"
    return f"{fraction.numerator}:{fraction.denominator}"


class ScreenedMoves(NamedTuple):
    """`moves[i]` is the close-to-close move into bar i+1 (NaN where either
    close is missing); `beyond` marks the moves larger than the threshold,
    and `by_ratio` the others named because they are the size of a listed
    split. `flagged` is the bars the screens name."""

    moves: np.ndarray
    beyond: np.ndarray
    by_ratio: np.ndarray

    @property
    def flagged(self) -> np.ndarray:
        return self.beyond | self.by_ratio


def _near_a_listed_ratio(prior: float, current: float) -> bool:
    """Whether the bar's implied ratio is within the tolerance of a ratio
    named below the threshold. The distance is computed exactly as
    `data.quality.nearest_split_ratio` computes it, so a bar named here is
    named after the same ratio there."""
    log_factor = math.log(prior / current)
    return any(
        abs(log_factor - math.log(ratio)) <= SPLIT_RATIO_TOLERANCE
        for ratio in RATIOS_NAMED_BELOW_THRESHOLD
    )


def screen_moves(
    close: np.ndarray, threshold: float = SPLIT_SCREEN_THRESHOLD
) -> ScreenedMoves:
    """The moves of `close` (prices in bar order, as floats) and which of
    them the split screens name."""
    values = np.asarray(close, dtype=float)
    if len(values) < 2:
        empty = np.zeros(0, dtype=bool)
        return ScreenedMoves(np.zeros(0, dtype=float), empty, empty.copy())
    before = values[:-1]
    after = values[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        moves = after / before - 1.0
        beyond = np.abs(moves) > threshold
        # Only a fall can be a forward split; a coarse band first, then the
        # exact distance on the few bars inside it.
        lowest = min(RATIOS_NAMED_BELOW_THRESHOLD) * math.exp(-SPLIT_RATIO_TOLERANCE)
        highest = max(RATIOS_NAMED_BELOW_THRESHOLD) * math.exp(SPLIT_RATIO_TOLERANCE)
        candidate = (
            ~beyond
            & (before > 0)
            & (after > 0)
            & (before >= after * lowest * (1 - 1e-9))
            & (before <= after * highest * (1 + 1e-9))
        )
    by_ratio = np.zeros(len(moves), dtype=bool)
    for i in np.flatnonzero(candidate):
        by_ratio[i] = _near_a_listed_ratio(float(before[i]), float(after[i]))
    return ScreenedMoves(moves, beyond, by_ratio)


def below_threshold_band(
    threshold: float = SPLIT_SCREEN_THRESHOLD,
) -> Tuple[float, float]:
    """The smallest and largest fall named by ratio below `threshold`, as
    positive fractions: (0.263, 0.35) at the default."""
    smallest = min(
        1.0 - 1.0 / (ratio * math.exp(-SPLIT_RATIO_TOLERANCE))
        for ratio in RATIOS_NAMED_BELOW_THRESHOLD
    )
    largest = max(
        1.0 - 1.0 / (ratio * math.exp(SPLIT_RATIO_TOLERANCE))
        for ratio in RATIOS_NAMED_BELOW_THRESHOLD
    )
    return smallest, min(largest, threshold)
