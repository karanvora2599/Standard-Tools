"""
What each feature is actually worth to a fitted model.

Every other tool in this package scores a feature ON ITS OWN: its IC, its
stability, its distribution. That is the cheap question and usually the
right one, but it cannot answer the question that decides whether to keep a
feature in a *model* -- how much worse the model gets without it.

Those differ, and they differ in the direction that costs money. A feature
with a strong standalone IC that duplicates another contributes nothing
marginal: drop it and the model does not move. A feature with a mediocre IC
that is the only source of some information can be the one holding the model
up. Neither shows in a per-feature report, and tree importance does not
answer it either -- importance is a statement about how one fitted estimator
happened to split, in units that differ per estimator, and a feature can
score high on it while being freely substitutable.

THE COST IS THE DESIGN CONSTRAINT. This refits the whole walk-forward
validation once per feature. A 40-feature panel with 8 folds is 328 fits,
which is minutes to hours, and an agent that calls it casually because the
name sounded informative will lose an afternoon. So the fit count is
computed and returned BEFORE anything is fit, and a run past the ceiling is
refused rather than started -- the same contract `validate_model_spec`
already offers for a search grid.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import numpy as np

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

#: Fits allowed without an explicit acknowledgement. 200 is roughly a
#: 24-feature panel at 8 folds -- large enough that the common case never
#: sees this, small enough that the runaway case stops. Deliberately a
#: number a caller can raise ON PURPOSE rather than a hard limit: the tool
#: refuses and says what to pass, which turns an accidental afternoon into
#: a decision.
DEFAULT_MAX_FITS = 200

#: Metrics where a LOWER value is better, so the sign of "how much worse
#: without this feature" flips. Getting this wrong would rank the most
#: important feature as the least, which is the kind of error that reads as
#: a plausible result.
_LOWER_IS_BETTER = ("mae", "mse", "rmse", "log_loss", "brier")


def estimate_ablation_fits(n_features: int, n_folds: int) -> int:
    """
    Fits an ablation will run: one baseline plus one per feature, each
    across every fold.

    Exposed separately so a caller can ask before committing, and so the
    number in the refusal message and the number actually run come from one
    place.
    """
    if n_features < 1:
        raise ValidationError("ablation needs at least one feature")
    if n_folds < 1:
        raise ValidationError("ablation needs at least one fold")
    return (n_features + 1) * n_folds


def _lower_is_better(metric: str) -> bool:
    name = metric.lower()
    return any(name == m or name.endswith(f"_{m}") for m in _LOWER_IS_BETTER)


def _metric_value(metrics: Dict[str, Any], metric: str) -> float:
    if metric not in metrics:
        raise ValidationError(
            f"metric {metric!r} is not in the experiment's OOS metrics. "
            f"Available: {sorted(k for k, v in metrics.items() if isinstance(v, (int, float)))}"
        )
    value = metrics[metric]
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise ValidationError(
            f"metric {metric!r} came back as {value!r}, which cannot be "
            "compared across ablations"
        )
    return float(value)


def ablation_contributions(
    baseline: float,
    without: Dict[str, float],
    metric: str,
) -> List[Dict[str, Any]]:
    """
    Turn baseline-and-leave-one-out scores into a ranked contribution table.

    `contribution` is always "how much the model LOSES without this
    feature", regardless of whether the metric is one where high or low is
    better. A positive contribution means the feature earns its place; a
    negative one means the model was better off without it, which happens
    and is worth seeing rather than clipping to zero.
    """
    flip = _lower_is_better(metric)
    rows = []
    for feature, score in without.items():
        # For a higher-is-better metric, losing the feature should lower the
        # score, so contribution = baseline - without. For lower-is-better
        # the ablated score should RISE, so the subtraction reverses.
        contribution = (score - baseline) if flip else (baseline - score)
        rows.append(
            {
                "feature": feature,
                "metric_with": baseline,
                "metric_without": score,
                "contribution": float(contribution),
                "relative_contribution": (
                    float(contribution / abs(baseline))
                    if baseline not in (0.0,) and np.isfinite(baseline)
                    else float("nan")
                ),
            }
        )
    rows.sort(key=lambda r: (-r["contribution"], r["feature"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def summarize_ablation(rows: Sequence[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    """The findings worth surfacing without being asked."""
    contributions = np.array([r["contribution"] for r in rows], dtype=float)
    negative = [r["feature"] for r in rows if r["contribution"] < 0]
    warnings: List[str] = []
    if negative:
        warnings.append(
            f"{len(negative)} feature(s) made the model WORSE by {metric}: "
            f"{sorted(negative)}. Removing them is the cheapest improvement "
            "available, but confirm on a second period first -- a negative "
            "contribution on one sample is also what noise looks like."
        )
    if contributions.size and float(np.max(contributions)) <= 0.0:
        warnings.append(
            "no feature made a positive contribution. Either the model has "
            "no signal to attribute, or every feature is substitutable by "
            "the others -- get_feature_redundancy separates those two."
        )
    return {
        "n_features": len(rows),
        "best_feature": rows[0]["feature"] if rows else None,
        "worst_feature": rows[-1]["feature"] if rows else None,
        "mean_contribution": (
            float(np.mean(contributions)) if contributions.size else float("nan")
        ),
        "n_negative_contributions": len(negative),
        "warnings": warnings,
    }


__all__ = [
    "DEFAULT_MAX_FITS",
    "ablation_contributions",
    "estimate_ablation_fits",
    "summarize_ablation",
]
