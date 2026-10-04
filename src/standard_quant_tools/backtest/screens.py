"""
Checks every backtest entry point runs on its bars before trusting them.

Two of them, shared so the single-asset engine and the portfolio engine
cannot drift apart again:

- the date index must be sorted and unique, because every engine here
  reads bar order as time order; and
- a close-to-close move large enough to be an unadjusted split is named,
  because every engine here compounds through it as a real return: one
  beyond 35%, or a fall the size of a 3:2 split (26% to 35%).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

# Which moves are named is one rule, `_split_screen.screen_moves` (with
# SPLIT_SCREEN_THRESHOLD, 35%, from `constants`), read here and by the
# dataset build's screen (`data.quality.detect_split_like_moves`), so a
# backtest and a dataset built from the same bars name the same moves.
from standard_quant_tools._split_screen import (
    RATIOS_NAMED_BELOW_THRESHOLD,
    below_threshold_band,
    screen_moves,
    split_ratio_label,
)
from standard_quant_tools.constants import SPLIT_SCREEN_THRESHOLD
from standard_quant_tools.error import ValidationError


def require_sorted_unique_index(
    index: pd.Index, name: str, func: str, order_matters: bool = True
) -> None:
    """
    Refuse a date index that is duplicated or out of order, by name.

    Refused rather than sorted: a silently sorted frame changes which bar a
    signal applies to. Unchecked, a reversed frame backtested the series
    backwards (+9.9% became -19.8% with no warning), and a repeated bar was
    counted twice on one path and raised a bare pandas ValueError on the
    other -- whether the caller saw an error or a number depended on
    whether the C++ extension was built.

    `order_matters=False` checks uniqueness only, for a series that is
    aligned BY LABEL onto a calendar whose order has already been checked
    (a signal read onto its price bars): its own row order never decides
    which bar comes first, but a repeated label still multiplies rows.
    """
    if index.has_duplicates:
        dupes = index[index.duplicated()].unique()[:5]
        raise ValidationError(
            f"{func}: {name}.index has duplicate date(s) "
            f"{[str(d) for d in dupes]} -- every bar must be unique. Drop or "
            "aggregate the repeated rows before backtesting."
        )
    if order_matters and not index.is_monotonic_increasing:
        raise ValidationError(
            f"{func}: {name}.index is not sorted in increasing order. Sort it "
            "(df.sort_index()) first: this engine reads row order as time "
            "order, so an unsorted frame is backtested out of sequence."
        )


def split_screen_warnings(prices: pd.Series, adjusted: Optional[bool]) -> List[str]:
    """
    One warning naming every bar whose |return| exceeds the split
    threshold, or that falls by about a 3:2 split's -33%, phrased by what
    is known about the bars' adjustment.
    """
    values = pd.to_numeric(pd.Series(prices), errors="coerce").to_numpy(dtype=float)
    screened = screen_moves(values, SPLIT_SCREEN_THRESHOLD)
    flagged = np.flatnonzero(screened.flagged)
    if len(flagged) == 0:
        return []
    labels = " or ".join(split_ratio_label(r) for r in RATIOS_NAMED_BELOW_THRESHOLD)

    def _listed(i: int) -> str:
        at = pd.Timestamp(prices.index[i + 1]).date()
        near = f", near a {labels} split" if screened.by_ratio[i] else ""
        return f"{at} ({float(screened.moves[i]):+.1%}{near})"

    listed = ", ".join(_listed(int(i)) for i in flagged[:5])
    more = f" and {len(flagged) - 5} more" if len(flagged) > 5 else ""
    size = f"move more than {SPLIT_SCREEN_THRESHOLD:.0%} close to close"
    if screened.by_ratio.any():
        smallest, largest = below_threshold_band(SPLIT_SCREEN_THRESHOLD)
        size += (
            f", or fall {smallest:.0%} to {largest:.0%} as a {labels} split does "
            f"({1 / RATIOS_NAMED_BELOW_THRESHOLD[0] - 1:+.0%})"
        )
    if adjusted is False:
        provenance = (
            "The provider reports adjusted=False, so a split is a real bar "
            "here and every metric that compounds through it is wrong; "
            "adjust the prices or fetch adjusted bars"
        )
    elif adjusted is True:
        provenance = (
            "The provider reports adjusted=True, so this is either a genuine "
            "move or a bad print; check the bar before trusting the result"
        )
    else:
        provenance = (
            "Whether these bars are split-adjusted is not known here; if "
            "they are not, every metric that compounds through such a bar "
            "is wrong (a 10:1 split read as -90%)"
        )
    return [
        f"SPLIT SCREEN: {len(flagged)} bar(s) {size}: {listed}{more}. {provenance}."
    ]
