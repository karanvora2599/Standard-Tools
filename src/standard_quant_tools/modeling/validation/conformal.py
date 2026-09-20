"""
Split conformal intervals: a prediction plus or minus a residual quantile
learned on rows the estimator did not fit.

WHY NOT THE TRAINING RESIDUALS. An estimator's residuals on the rows it
was fitted to are optimistic in exactly the way that matters here -- a
flexible model fits its training rows closely and would report a narrow
interval that the next fold does not honour. Split conformal reads the
residuals on HELD-OUT rows instead: the training window is cut into
contiguous date blocks, the estimator is refit without each block, and
the absolute residuals on that block are collected. The (1 - alpha)
quantile of those, with the finite-sample correction, is the radius, and
for exchangeable rows an interval of that radius covers a new outcome
with probability at least 1 - alpha.

THE SAME DISCIPLINE AS EVERY OTHER SPLIT HERE. Blocks are contiguous in
time, the rows within `embargo` dates of a block are left out of the
refit, and a training row whose label reaches into the block is purged by
the same `label_overlap_mask` the outer folds and the inner search use.
A residual read on a row whose label the refit had already seen would be
too small, and the interval too narrow, in the direction that flatters.

WHAT IS NOT CLAIMED. Exchangeability is the assumption, and a return panel
is not exchangeable across regimes: coverage is honest on average over the
window and can fail in a regime the calibration window did not contain.
The reported OOS coverage is the check, and it is why the metric is
computed on the test rows rather than taken from the theory.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np

from standard_quant_tools.error import ValidationError

from .walk_forward import label_overlap_mask

#: Fewer held-out residuals than this cannot place a quantile at any
#: level a caller would ask for.
MIN_CALIBRATION_ROWS = 10

FitPredict = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def held_out_residuals(
    fit_predict: FitPredict,
    row_dates: np.ndarray,
    label_end: Optional[np.ndarray],
    *,
    n_folds: int,
    embargo: int = 0,
) -> np.ndarray:
    """
    Absolute residuals on rows held out in `n_folds` contiguous date
    blocks, each block predicted by a refit that excluded it, its embargo
    band and every row whose label reaches into it.

    `fit_predict(train_mask, test_mask)` fits on the first rows and returns
    `(y_true, y_pred)` for the second; it is supplied by the engine so the
    refit is the engine's own -- same estimator, same parameters, same
    weights.
    """
    dates = np.array(sorted(set(row_dates)))
    if len(dates) < n_folds:
        raise ValidationError(
            f"conformal calibration needs at least {n_folds} distinct dates in "
            f"the training window for {n_folds} calibration folds; this window "
            f"has {len(dates)}."
        )
    date_code = np.searchsorted(dates, row_dates)
    residuals = []
    for block in np.array_split(np.arange(len(dates)), n_folds):
        if block.size == 0:
            continue
        first, last = int(block[0]), int(block[-1])
        in_test = np.zeros(len(dates), dtype=bool)
        in_test[block] = True
        banned = np.zeros(len(dates), dtype=bool)
        banned[max(0, first - embargo) : min(len(dates), last + embargo + 1)] = True
        test_mask = in_test[date_code]
        train_mask = ~banned[date_code]
        overlaps = label_overlap_mask(
            train_mask, row_dates, label_end, dates[first], dates[last]
        )
        train_mask &= ~overlaps
        if not train_mask.any() or not test_mask.any():
            continue
        y_true, y_pred = fit_predict(train_mask, test_mask)
        residuals.append(
            np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
        )
    if not residuals:
        raise ValidationError(
            "conformal calibration held out no rows: every block left an empty "
            "refit or an empty block after the embargo and the purge."
        )
    out = np.concatenate(residuals)
    out = out[np.isfinite(out)]
    if out.size < MIN_CALIBRATION_ROWS:
        raise ValidationError(
            f"conformal calibration has {out.size} held-out residual(s), fewer "
            f"than the {MIN_CALIBRATION_ROWS} a quantile can be read from. Widen "
            "train_window or lower intervals.calibration_folds."
        )
    return out


def conformal_radius(residuals: np.ndarray, alpha: float) -> float:
    """
    The split-conformal radius: the ceil((n + 1)(1 - alpha))-th smallest
    absolute residual. When that rank exceeds n -- too few residuals for
    the level asked -- the largest residual stands, which is the honest
    (wide) answer rather than a narrower one the sample cannot support.
    """
    values = np.sort(np.asarray(residuals, dtype=float))
    n = values.size
    if n == 0:
        raise ValidationError("conformal_radius: no residuals")
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(values[rank - 1])


__all__ = ["MIN_CALIBRATION_ROWS", "conformal_radius", "held_out_residuals"]
