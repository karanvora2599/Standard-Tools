"""
The refit grid that universe-scope features share.

A rolling refit that is anchored on the frame's FIRST bar makes a
feature's value at date t depend on where the fetched frame begins:
`for end in range(window, n + 1, refit_every)` refits on bars
window-1, window-1+refit_every, ... counted from bar zero, so two frames
that start k bars apart refit on different bars unless k is a multiple of
`refit_every`. That is what the live findings measured (D17): recomputing
`pca_loading(252, 21)` after dropping one leading bar changed every value,
the worst by 33.6%, and the deployed estimator -- which scores on a frame
rebuilt from `as_of - lookback_days`, a different anchor than the training
build -- was fed a different variable under the same column name.

The grid here is a function of each bar's TIMESTAMP alone. Daily and
slower bars are numbered by weekdays since 1970-01-01; a bar refits when
its number is a multiple of `refit_every` and a full window precedes it.
Two frames that both contain a date and its window therefore refit on the
same bars up to that date, whatever either frame's first or last bar is:
dropping leading bars leaves every value the shorter frame can still
compute bit-identical, and truncating trailing bars leaves every value at
or before the cut unchanged, which is the point-in-time property the
network features already pinned. Holidays shift every later weekday count
by the same amount, so they do not break the agreement; an exchange
calendar would number sessions more tightly and is not required.

Faster bars are numbered by their own median spacing since the epoch, and
a frame with no datetime index is numbered by position, which is the old
behaviour and the only one available for it.

The price is warm-up: the first refit is the first grid bar with a full
window behind it, up to `refit_every - 1` bars later than the window
itself, and the feature is NaN until then rather than fitted on a bar the
grid does not name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPOCH_DAY = np.datetime64("1970-01-01", "D")
_EPOCH_NS = np.datetime64("1970-01-01", "ns")
_ONE_DAY_NS = 86_400_000_000_000


def bar_ordinals(index) -> np.ndarray:
    """
    A global bar number per row of `index`, a function of the row's own
    timestamp: weekdays since 1970-01-01 for daily-or-slower bars, median
    spacings since the epoch for faster ones, the position for an index
    that is not datetimes.
    """
    if not pd.api.types.is_datetime64_any_dtype(index):
        return np.arange(len(index), dtype=np.int64)
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    values = idx.values.astype("datetime64[ns]")
    if len(values) < 2:
        return np.busday_count(_EPOCH_DAY, values.astype("datetime64[D]")).astype(
            np.int64
        )
    spacing_ns = int(
        np.median(np.diff(values).astype("timedelta64[ns]").astype(np.int64))
    )
    if spacing_ns >= _ONE_DAY_NS:
        return np.busday_count(_EPOCH_DAY, values.astype("datetime64[D]")).astype(
            np.int64
        )
    spacing_ns = max(spacing_ns, 1)
    return ((values - _EPOCH_NS).astype(np.int64) // spacing_ns).astype(np.int64)


def refit_mask(index, window: int, refit_every: int) -> np.ndarray:
    """
    Which bars of `index` a rolling estimator refits on: those whose bar
    number is a multiple of `refit_every` and that have `window` bars at or
    before them. Boolean, one entry per row.
    """
    n = len(index)
    has_window = np.arange(n) + 1 >= int(window)
    return has_window & (bar_ordinals(index) % int(refit_every) == 0)


__all__ = ["bar_ordinals", "refit_mask"]
