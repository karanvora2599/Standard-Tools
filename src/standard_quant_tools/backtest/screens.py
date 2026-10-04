"""
Checks every backtest entry point runs on its bars before trusting them.

Two of them, shared so the single-asset engine and the portfolio engine
cannot drift apart again:

- the date index must be sorted and unique, because every engine here
  reads bar order as time order; and
- a close-to-close move large enough to be an unadjusted split is named,
  because every engine here compounds through it as a real return.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

# SPLIT_SCREEN_THRESHOLD (35%) is defined once, in `constants`, and read
# here and by the dataset build's screen
# (`data.quality.detect_split_like_moves`), so a backtest and a dataset
# built from the same bars name the same moves.
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
    threshold, phrased by what is known about the bars' adjustment.
    """
    moves = prices.pct_change(fill_method=None)
    jumps = moves[moves.abs() > SPLIT_SCREEN_THRESHOLD]
    if jumps.empty:
        return []
    listed = ", ".join(
        f"{pd.Timestamp(at).date()} ({float(move):+.1%})"
        for at, move in list(jumps.items())[:5]
    )
    more = f" and {len(jumps) - 5} more" if len(jumps) > 5 else ""
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
        f"SPLIT SCREEN: {len(jumps)} bar(s) move more than "
        f"{SPLIT_SCREEN_THRESHOLD:.0%} close to close: {listed}{more}. "
        f"{provenance}."
    ]
