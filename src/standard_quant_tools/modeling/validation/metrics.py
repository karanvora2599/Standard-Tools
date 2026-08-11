"""
Out-of-sample evaluation metrics, computed once per fold by engine.py and
averaged across folds. No new dependency: rank-IC uses pandas' built-in
Spearman correlation (Series.corr(method="spearman")) rather than scipy
(not a declared dependency of this package); R2/MAE/accuracy/AUC reuse
sklearn.metrics (scikit-learn is already a core dependency).
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score


def positive_class_proba(estimator: Any, X: np.ndarray) -> np.ndarray:
    """
    estimator.classes_ is not guaranteed to place the positive class (1)
    at column index 1 of predict_proba's output -- look it up explicitly
    rather than hardcoding [:, 1], which would silently score the WRONG
    class's probability if sklearn ever orders classes differently, and
    would raise a raw IndexError if the estimator only ever saw one
    class. Shared by engine.py (per-fold evaluation) and scoring.py
    (scoring a registered classifier), so both compute classifier
    probabilities the same, correct way.
    """
    proba = estimator.predict_proba(X)
    classes = list(estimator.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]  # single-class estimator; see callers for why
    # this should be unreachable in practice.


def _safe_corr(true_s: pd.Series, pred_s: pd.Series, method: str) -> float:
    value = float(true_s.corr(pred_s, method=method))
    return 0.0 if np.isnan(value) else value


def cross_sectional_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
    method: str = "spearman",
) -> pd.Series:
    """
    Information coefficient computed WITHIN each date's cross-section,
    returned as a per-date series.

    This is the quantity a cross-sectional model is actually judged on.
    Pooling every (entity, date) row into one correlation — what
    `regression_metrics` reported as `ic`/`rank_ic` — mixes two different
    effects: whether the model ranks names correctly against each other on
    a given day, and whether it times the market's overall level across
    days. A model with no cross-sectional skill can still show a strong
    pooled IC purely because it tracks the market factor, since on
    up-market days both predictions and realized returns are high for
    nearly every name.

    Dates with fewer than 2 entities are dropped: a correlation over a
    single point is undefined, not zero.
    """
    frame = pd.DataFrame({"date": dates, "y": y_true, "p": y_pred})
    per_date = {}
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 2:
            continue
        per_date[date] = _safe_corr(group["y"], group["p"], method)
    return pd.Series(per_date, dtype=float)


def summarize_cross_sectional_ic(ic_series: pd.Series, prefix: str) -> Dict[str, float]:
    """
    Time-series summary of a per-date IC series: mean, volatility, ICIR,
    and the fraction of dates with positive IC.

    ICIR (mean / std) is the metric that says whether an IC is dependable
    rather than merely large on average — a 0.03 IC that is positive on 60%
    of days is a very different model from a 0.03 IC driven by a handful of
    extreme days, and a mean alone cannot distinguish them.
    """
    if ic_series.empty:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_icir": float("nan"),
            f"{prefix}_hit_rate": float("nan"),
            f"{prefix}_n_dates": 0.0,
        }
    mean = float(ic_series.mean())
    # ddof=1 needs >= 2 dates; a single date has no dispersion to measure.
    std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else float("nan")
    icir = mean / std if std and np.isfinite(std) and std > 0 else float("nan")
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_icir": icir,
        f"{prefix}_hit_rate": float((ic_series > 0).mean()),
        f"{prefix}_n_dates": float(len(ic_series)),
    }


def effective_sample_size(n_obs: int, horizon: int, n_entities: int = 1) -> float:
    """
    Observation count discounted for target overlap.

    A `horizon`-bar forward return generated every bar produces labels that
    overlap on `horizon - 1` of their bars, so consecutive rows are far from
    independent. Reporting a raw row count materially overstates how much
    evidence a metric rests on: 2,000 daily rows of a 20-day forward return
    carry roughly 100 independent observations per entity, not 2,000.

    This is the standard first-order correction (divide by the overlap
    factor), not a full Newey-West style adjustment — enough to stop the
    headline count being misleading, and labelled as an estimate.
    """
    if horizon <= 0:
        return float(n_obs)
    per_entity = max(n_obs / max(n_entities, 1), 0.0)
    return float(max(per_entity / horizon, 0.0) * max(n_entities, 1))


def baseline_regression_metrics(y_true: np.ndarray) -> Dict[str, float]:
    """
    Metrics for the trivial "predict the mean" model, so a reported R2/MAE
    has something to be compared against. Without this a result can look
    informative while being no better than a constant.
    """
    if len(y_true) == 0:
        return {"baseline_mae": float("nan"), "baseline_r2": 0.0}
    constant = np.full_like(y_true, float(np.mean(y_true)), dtype=float)
    return {
        "baseline_mae": float(mean_absolute_error(y_true, constant)),
        # R2 of the mean predictor is 0.0 by construction; stated
        # explicitly so the comparison is legible in the output.
        "baseline_r2": 0.0,
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: "np.ndarray | None" = None,
) -> Dict[str, float]:
    """
    `ic`/`rank_ic` are POOLED across every (entity, date) row — kept for
    continuity, but see cross_sectional_ic for why they conflate
    cross-sectional skill with market timing. When `dates` is supplied the
    per-date cross-sectional versions are added alongside them, and those
    are the ones to judge a cross-sectional model on.
    """
    true_s = pd.Series(y_true)
    pred_s = pd.Series(y_pred)
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "ic": _safe_corr(true_s, pred_s, "pearson"),
        "rank_ic": _safe_corr(true_s, pred_s, "spearman"),
    }
    metrics.update(baseline_regression_metrics(y_true))
    if dates is not None:
        metrics.update(
            summarize_cross_sectional_ic(
                cross_sectional_ic(y_true, y_pred, dates, "pearson"), "cs_ic"
            )
        )
        metrics.update(
            summarize_cross_sectional_ic(
                cross_sectional_ic(y_true, y_pred, dates, "spearman"), "cs_rank_ic"
            )
        )
    return metrics


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    metrics: Dict[str, float] = {"accuracy": float(accuracy_score(y_true, y_pred))}
    # roc_auc_score needs both classes present in the fold — a short or
    # unlucky test fold can be single-class, which isn't a bug in the
    # model, so report NaN rather than raising. (sanitize_for_json turns
    # that into null at the agent boundary; NaN is kept internally so
    # nan-aware fold aggregation still works.)
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics["auc"] = float("nan")

    # Class balance, reported rather than assumed. Accuracy is close to
    # meaningless without it: a 95/5 split scores 0.95 by always predicting
    # the majority class, and TargetSpec.threshold makes exactly that kind
    # of imbalance easy to request (a 2% up-move threshold labels most bars
    # 0). base_rate is the accuracy of always guessing the majority class —
    # the number `accuracy` has to beat to mean anything.
    positive_rate = float(np.mean(np.asarray(y_true, dtype=float) == 1.0))
    metrics["positive_rate"] = positive_rate
    metrics["majority_class_accuracy"] = max(positive_rate, 1.0 - positive_rate)
    return metrics


def average_fold_metrics(
    fold_metrics: List[Dict[str, float]],
    fold_weights: "List[float] | None" = None,
) -> Dict[str, float]:
    """
    Weighted mean of each metric across folds, ignoring NaN (e.g. a
    single-class AUC fold) rather than propagating it into the summary.

    `fold_weights` should be each fold's out-of-sample prediction count.
    An equal-weighted mean — the previous behavior — gives a fold covering
    30 predictions exactly as much influence on the headline number as one
    covering 3,000, which is wrong whenever entity coverage varies across
    the sample (a universe with mid-history IPOs, say). Falls back to equal
    weights when none are supplied.

    Metrics whose names end in `_n_dates` are SUMMED rather than averaged:
    a count of dates observed is not a per-fold rate.
    """
    keys = list(fold_metrics[0].keys())
    if fold_weights is None:
        fold_weights = [1.0] * len(fold_metrics)
    weights = np.asarray(fold_weights, dtype=float)

    result: Dict[str, float] = {}
    for k in keys:
        values = np.array([fm.get(k, np.nan) for fm in fold_metrics], dtype=float)
        finite = ~np.isnan(values)
        if not finite.any():
            # NaN is still the correct answer (e.g. every surviving fold's
            # test set was single-class, so AUC was NaN everywhere) --
            # computed without numpy's "Mean of empty slice" warning.
            result[k] = float("nan")
            continue
        if k.endswith("_n_dates"):
            result[k] = float(values[finite].sum())
            continue
        w = weights[finite]
        total = w.sum()
        result[k] = (
            float(np.average(values[finite], weights=w))
            if total > 0
            else float(np.mean(values[finite]))
        )
    return result
