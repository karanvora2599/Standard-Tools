"""
The dataset layer's half of the missing-data policy: a bounded forward
fill, per entity, before the panel is stacked.

WHY HERE AND NOT IN THE PIPELINE. A forward fill reaches an entity's
EARLIER bars, which is only meaningful on that entity's own frame, on its
own bar index -- exactly the argument `dataset/lags.py` makes for shifting
before stacking. After `stack_long` a fill would run down the long panel
and hand one entity another's last value, producing a plausible panel and
no visible symptom. And unlike an `impute` step, which fits a statistic on
a training fold, a fill has no fold: the value carried onto bar t is bar
t-1's, known at t, so nothing about it depends on which fold t lands in.

WHY BOUNDED, AND WHY AN ALLOWLIST. A value carried forever is a series
that stopped being a measurement and kept being a feature. `max_staleness`
is the most bars a value may stand in for, and the fill is applied only to
the features the caller named, because whether a stale value is a fair
stand-in is a fact about the feature -- true of a slowly-updating level,
false of a bar's volume -- that only the caller can assert.

WARM-UP IS NEVER FABRICATED. A leading NaN has no prior value to carry, so
pandas leaves it, and a feature's lookback still costs exactly the rows it
always did.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import pandas as pd


def forward_fill_bounded(
    frame: pd.DataFrame, features: Sequence[str], max_staleness: int
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Carry each named column's last value forward for at most
    `max_staleness` consecutive bars, on ONE entity's frame.

    Returns the filled frame and, per column, how many values were filled
    -- the number the build reports, because a fill that recovered nothing
    and one that rewrote a tenth of the column are different datasets.
    """
    if max_staleness < 1:
        return frame, {}
    out = frame.copy()
    filled: Dict[str, int] = {}
    for name in features:
        if name not in out.columns:
            continue
        before = int(out[name].isna().sum())
        out[name] = out[name].ffill(limit=int(max_staleness))
        n_filled = before - int(out[name].isna().sum())
        if n_filled:
            filled[name] = n_filled
    return out, filled


__all__ = ["forward_fill_bounded"]
