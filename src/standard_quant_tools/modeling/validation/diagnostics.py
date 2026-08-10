"""Per-fold diagnostics: feature-importance stability across walk-forward
folds. For linear estimators, importance is |coefficient|; for tree
estimators, the model's own feature_importances_. Averaged and reported
alongside its cross-fold standard deviation, so inspect_model can show
whether a feature's importance is stable or an artifact of one fold."""

from typing import Any, Dict, List

import numpy as np


def fold_feature_importance(estimator: Any, feature_ids: List[str]) -> Dict[str, float]:
    if hasattr(estimator, "coef_"):
        coef = np.atleast_1d(np.asarray(estimator.coef_)).ravel()
        values = np.abs(coef)
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_)
    else:
        return {fid: float("nan") for fid in feature_ids}
    return {fid: float(v) for fid, v in zip(feature_ids, values)}


def summarize_importance(
    per_fold_importance: List[Dict[str, float]], feature_ids: List[str]
) -> Dict[str, Dict[str, float]]:
    summary = {}
    for fid in feature_ids:
        values = np.array([fold[fid] for fold in per_fold_importance if fid in fold])
        # np.nanmean/nanstd warn "Mean of empty slice" on an all-NaN
        # array (e.g. every fold used HistGradientBoosting, which has no
        # coef_/feature_importances_) -- NaN is still the correct answer
        # here, just without numpy's warning about how it got there.
        has_data = len(values) > 0 and not np.all(np.isnan(values))
        summary[fid] = {
            "mean": float(np.nanmean(values)) if has_data else float("nan"),
            "std": float(np.nanstd(values)) if has_data else float("nan"),
        }
    return summary
