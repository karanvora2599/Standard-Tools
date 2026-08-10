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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    true_s = pd.Series(y_true)
    pred_s = pd.Series(y_pred)
    ic = float(true_s.corr(pred_s, method="pearson"))
    rank_ic = float(true_s.corr(pred_s, method="spearman"))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "ic": 0.0 if np.isnan(ic) else ic,
        "rank_ic": 0.0 if np.isnan(rank_ic) else rank_ic,
    }


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    metrics: Dict[str, float] = {"accuracy": float(accuracy_score(y_true, y_pred))}
    # roc_auc_score needs both classes present in the fold — a short or
    # unlucky test fold can be single-class, which isn't a bug in the
    # model, so report NaN rather than raising.
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def average_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of each metric key across folds, ignoring NaN (e.g. a
    single-class AUC fold) rather than propagating NaN into the summary."""
    keys = fold_metrics[0].keys()
    result: Dict[str, float] = {}
    for k in keys:
        values = np.array([fm[k] for fm in fold_metrics])
        # np.nanmean warns "Mean of empty slice" on an all-NaN array
        # (e.g. every surviving fold's test set happened to be
        # single-class, so AUC was NaN everywhere) -- NaN is still the
        # correct answer here, just without numpy's warning about it.
        result[k] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
    return result
