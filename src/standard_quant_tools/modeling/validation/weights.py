"""
Training-row weights, for the two reasons every row is not equally
informative.

`effective_sample_size` has always been reported next to the out-of-sample
metrics: with a `horizon`-bar forward return generated every bar,
consecutive labels overlap on `horizon - 1` of their bars, so 2,000 daily
rows of a 20-day forward return carry roughly 100 independent observations
per entity rather than 2,000. That number was computed and then ignored --
every row still entered the fit at weight 1. These are the weights that act
on it.

LABEL UNIQUENESS. The overlap is not uniform, which is the part a single
scalar cannot express. Interior rows of a training window sit under
`horizon` concurrent labels; rows near the start of the window, and rows
near the end after the target-overlap purge has removed their neighbours,
sit under fewer, and are correspondingly MORE informative per row. Weighting
by average uniqueness -- the mean of 1/concurrency over the bars a row's own
label spans (Lopez de Prado, Advances in Financial Machine Learning, ch. 4)
-- is the standard correction, and it is computed per entity because two
entities' labels are different series that do not make each other redundant.

TIME DECAY. Separately from redundancy, older evidence is less relevant when
the relationship being estimated drifts. An exponential half-life is the
usual expression of that, and it is deliberately kept as its own option
rather than folded into uniqueness: they correct for different things, and a
caller should be able to say which one they believe in.

Both are OPT-IN. The default remains unweighted, because turning either on
changes what the model fits and therefore what it predicts -- that is a
modelling decision and not a default this module gets to make.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

# Optional native fast path. This was the worst per-row cost in the module --
# 5.7 microseconds per row at 2,000,000 rows, two orders of magnitude off
# what the arithmetic costs -- because the Python below loops once per
# entity. The Python stays as the reference and the test oracle.
_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = hasattr(_cpp_core, "label_uniqueness")
except ImportError:
    pass

# THE GUARD THAT USED TO BE HERE IS GONE, and the reason it existed is
# worth keeping. The kernel once LOST on small panels -- 0.4x on 12,600
# rows -- because the argument conversion went through pandas. "A fast path
# that is slower is a bug", so it was gated at 50,000 rows.
#
# `_as_int64_ns` removed that conversion and the loss with it, and the
# threshold never moved. Re-measured, outputs matching to 1e-12 at every
# size:
#
#       10 rows   9.1x        2,520 rows   5.1x
#      100 rows   7.5x       12,600 rows   3.0x
#      500 rows   7.9x       25,200 rows   4.2x
#
# There is no crossover left to gate at: it wins everywhere, and it won by
# MORE below the old threshold than above it. The guard's cost was not
# hypothetical either -- per-fold rows are `train_window * entities`, so a
# 200-name universe on a 252-day window is 49,392 rows and landed 608 short,
# which meant the kernel written for the fold loop fired once per run, on
# the final whole-panel refit.


def _as_int64_ns(values: np.ndarray) -> np.ndarray:
    """
    Datetime values as int64 nanoseconds, without a pandas round trip when
    one is not needed.

    `datetime64[ns]` reinterprets as int64 for free, and NaT arrives as
    INT64_MIN -- which is exactly the sentinel the kernel tests for. Anything
    else (object dtype, a coarser unit) goes through pandas to normalize it
    first. The distinction matters: the round trip was most of why the
    kernel lost on small panels.
    """
    array = np.asarray(values)
    if array.dtype == np.dtype("datetime64[ns]"):
        return np.ascontiguousarray(array).view(np.int64)
    converted = pd.to_datetime(pd.Series(array)).to_numpy("datetime64[ns]")
    return np.ascontiguousarray(converted).view(np.int64)


def label_uniqueness_weights(
    dates: np.ndarray,
    label_end_dates: np.ndarray,
    entities: np.ndarray,
) -> np.ndarray:
    """
    Average uniqueness of each row's label, computed within its entity.

    Returns weights normalized to mean 1.0, so switching weighting on does
    not also rescale the effective regularization strength -- an unweighted
    fit and a uniqueness-weighted fit see the same total weight, just
    distributed differently.
    """
    dates = np.asarray(dates)
    label_end_dates = np.asarray(label_end_dates)
    entities = np.asarray(entities)
    n = dates.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if label_end_dates.size != n or entities.size != n:
        raise ValidationError(
            "label_uniqueness_weights: dates, label_end_dates and entities must "
            f"be the same length, got {n}, {label_end_dates.size}, {entities.size}"
        )

    entity_codes = pd.factorize(entities, sort=False)[0]

    if HAS_CPP:
        # Timestamps rather than integer offsets because `horizon` counts
        # each ENTITY's own bars: with entities on different calendars,
        # t+horizon of one entity is not t+horizon of the global date axis.
        native = _cpp_core.label_uniqueness(
            _as_int64_ns(dates),
            _as_int64_ns(label_end_dates),
            np.ascontiguousarray(entity_codes, dtype=np.int64),
            int(entity_codes.max()) + 1 if entity_codes.size else 0,
        )
        return np.asarray(native, dtype=np.float64)

    weights = np.ones(n, dtype=np.float64)
    for code in np.unique(entity_codes):
        rows = np.flatnonzero(entity_codes == code)
        if rows.size == 0:
            continue
        row_dates = dates[rows]
        row_ends = label_end_dates[rows]
        order = np.argsort(row_dates, kind="stable")
        rows = rows[order]
        row_dates = row_dates[order]
        row_ends = row_ends[order]

        # Each label spans the entity's own bars [start, end]. Bars are
        # indexed by position in this entity's date axis, so an entity on a
        # different calendar is handled without assuming a shared grid.
        axis = row_dates
        start_pos = np.arange(row_dates.size)
        finite_end = pd.notna(row_ends)
        end_pos = np.where(
            finite_end,
            np.searchsorted(axis, row_ends, side="right") - 1,
            start_pos,
        )
        end_pos = np.maximum(end_pos, start_pos)

        # Concurrency by difference array: +1 where a label starts, -1 just
        # past where it ends, then a running sum.
        delta = np.zeros(axis.size + 1, dtype=np.float64)
        np.add.at(delta, start_pos, 1.0)
        np.add.at(delta, end_pos + 1, -1.0)
        concurrency = np.cumsum(delta)[: axis.size]
        # A bar covered by no label cannot be inside any label's span, so
        # the guard only protects the division, never a real value.
        inverse = 1.0 / np.maximum(concurrency, 1.0)
        cumulative = np.concatenate(([0.0], np.cumsum(inverse)))
        spans = (end_pos - start_pos + 1).astype(np.float64)
        weights[rows] = (cumulative[end_pos + 1] - cumulative[start_pos]) / spans

    mean = float(weights.mean())
    return weights / mean if mean > 0 else weights


def time_decay_weights(dates: np.ndarray, half_life: float) -> np.ndarray:
    """
    Exponential decay in calendar time, normalized to mean 1.0.

    `half_life` is in DAYS, not bars: a caller thinking about how fast a
    relationship goes stale is thinking in calendar time, and bar counts
    differ between a daily and an hourly dataset for the same intent.
    """
    if half_life <= 0:
        raise ValidationError(
            f"time_decay_weights: half_life must be > 0 days, got {half_life}"
        )
    stamps = pd.to_datetime(pd.Series(np.asarray(dates)))
    if stamps.empty:
        return np.empty(0, dtype=np.float64)
    age_days = (stamps.max() - stamps).dt.total_seconds().to_numpy() / 86400.0
    weights = np.power(0.5, age_days / float(half_life))
    mean = float(weights.mean())
    return weights / mean if mean > 0 else weights


def build_sample_weights(
    method: str,
    dates: np.ndarray,
    label_end_dates: Optional[np.ndarray],
    entities: np.ndarray,
    half_life: float,
) -> Optional[np.ndarray]:
    """
    Weights for one training fold, or None when the fit should be
    unweighted (which is also what a caller gets for method='none').
    """
    if method == "none":
        return None
    if method in ("label_uniqueness", "uniqueness_and_time_decay"):
        if label_end_dates is None:
            raise ValidationError(
                "sample weighting by label uniqueness needs each row's label end "
                "date, which this dataset does not carry. Rebuild it with a "
                "current version of build_model_dataset, or use "
                "weighting='time_decay', which needs only the dates."
            )
        weights = label_uniqueness_weights(dates, label_end_dates, entities)
    else:
        weights = np.ones(np.asarray(dates).size, dtype=np.float64)

    if method in ("time_decay", "uniqueness_and_time_decay"):
        weights = weights * time_decay_weights(dates, half_life)

    mean = float(weights.mean()) if weights.size else 0.0
    return weights / mean if mean > 0 else weights
