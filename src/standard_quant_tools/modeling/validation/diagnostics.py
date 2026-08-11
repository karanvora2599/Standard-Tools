"""Per-fold diagnostics: feature-importance stability across walk-forward
folds, reported alongside its cross-fold spread so inspect_model can show
whether a feature's importance is stable or an artifact of one fold.

For linear estimators importance comes from `coef_`, for tree estimators
from `feature_importances_`. The distinction matters more than it looks:
a coefficient has a DIRECTION and a tree importance does not, and folding
the direction away silently broke the stability report this module exists
to produce -- see summarize_importance.
"""

from typing import Any, Dict, List, NamedTuple

import numpy as np


class FoldImportance(NamedTuple):
    """One fold's per-feature importance, plus whether those numbers carry
    a direction.

    `signed` is not cosmetic bookkeeping: it decides whether the sign
    statistics below mean anything. Tree `feature_importances_` are
    non-negative by construction, so computing "sign consistency" over them
    would report perfect directional agreement for a quantity that has no
    direction at all -- a confident answer to a question that was never
    asked.
    """

    values: Dict[str, float]
    signed: bool


def fold_feature_importance(estimator: Any, feature_ids: List[str]) -> FoldImportance:
    """
    Returns the SIGNED coefficient for linear estimators (magnitude is
    derived later), the model's own non-negative importances for trees, and
    NaN for an estimator exposing neither.
    """
    if hasattr(estimator, "coef_"):
        coef = np.atleast_1d(np.asarray(estimator.coef_, dtype=float)).ravel()
        if coef.size != len(feature_ids):
            # Multiclass `coef_` is (n_classes, n_features), so ravel gives
            # n_classes * n_features values and zip() silently kept the
            # first n_features of them -- i.e. reported class 0's
            # coefficients as though they were THE importances and dropped
            # every other class without a word. Not reachable through the
            # tool surface today (forward_direction is binary), but
            # register_estimator accepts custom estimators, and a wrong
            # attribution is worse than an absent one: there is no single
            # per-feature coefficient in a multiclass fit, so NaN is the
            # honest answer.
            return FoldImportance({fid: float("nan") for fid in feature_ids}, True)
        return FoldImportance(
            {fid: float(v) for fid, v in zip(feature_ids, coef)}, True
        )

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float).ravel()
        if values.size != len(feature_ids):  # pragma: no cover — defensive
            return FoldImportance({fid: float("nan") for fid in feature_ids}, False)
        return FoldImportance(
            {fid: float(v) for fid, v in zip(feature_ids, values)}, False
        )

    return FoldImportance({fid: float("nan") for fid in feature_ids}, False)


def _nan() -> float:
    return float("nan")


def summarize_importance(
    per_fold_importance: List[FoldImportance], feature_ids: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Per feature: magnitude, spread, and — for linear estimators — direction.

    `mean`/`std` are magnitude statistics, computed over |value|, and keep
    the meaning they have always had so existing manifests stay comparable.
    They are not sufficient on their own, which is the point of the added
    keys:

      signed_mean       average coefficient WITH its sign. A stably negative
                        feature is a working contrarian signal; reporting it
                        as "importance 0.4" identical to a positive one
                        throws away the most actionable thing the model
                        learned.
      signed_std        spread of the signed coefficient. This is the real
                        stability metric. Taking the absolute value FIRST
                        made a feature whose coefficient alternates
                        +0.5, -0.5, +0.5, -0.5 across folds -- the
                        maximally unstable case, and a textbook sign of
                        fitting noise -- come out as |0.5| every fold, i.e.
                        std exactly 0.0: reported as PERFECTLY stable by the
                        very number whose stated job was catching that.
      sign_consistency  fraction of folds agreeing with the majority sign,
                        in [0.5, 1.0]. 1.0 is one direction throughout, 0.5
                        is a coin flip. A separate number from signed_std
                        because a feature can be directionally consistent
                        while varying a lot in size, and the two failure
                        modes call for different responses.

    All three are NaN for tree estimators, which have no direction to
    report — deliberately NaN rather than a plausible-looking default, so
    "no sign information exists" cannot be misread as "the sign was
    stable".
    """
    summary: Dict[str, Dict[str, float]] = {}
    any_signed = any(fold.signed for fold in per_fold_importance)

    for fid in feature_ids:
        raw = np.array(
            [fold.values[fid] for fold in per_fold_importance if fid in fold.values],
            dtype=float,
        )
        # np.nanmean/nanstd warn "Mean of empty slice" on an all-NaN array
        # (e.g. every fold used HistGradientBoosting, which exposes neither
        # coef_ nor feature_importances_) -- NaN is still the correct
        # answer, just without numpy's warning about how it got there.
        has_data = raw.size > 0 and not np.all(np.isnan(raw))
        magnitude = np.abs(raw)

        entry = {
            "mean": float(np.nanmean(magnitude)) if has_data else _nan(),
            "std": float(np.nanstd(magnitude)) if has_data else _nan(),
        }

        if has_data and any_signed:
            finite = raw[~np.isnan(raw)]
            entry["signed_mean"] = float(np.mean(finite))
            entry["signed_std"] = float(np.std(finite))
            entry["sign_consistency"] = _sign_consistency(finite)
        else:
            entry["signed_mean"] = _nan()
            entry["signed_std"] = _nan()
            entry["sign_consistency"] = _nan()

        summary[fid] = entry
    return summary


def _sign_consistency(values: np.ndarray) -> float:
    """
    Fraction of folds sharing the majority sign, over the folds that have
    one.

    Exact zeros are excluded rather than counted as agreeing with either
    side: a coefficient driven to exactly 0.0 (routine under L1) expresses
    no direction, and letting it vote would let a feature the model
    DISCARDED in most folds be reported as directionally consistent. If
    every fold is zero there is no direction to be consistent about, hence
    NaN.
    """
    non_zero = values[values != 0.0]
    if non_zero.size == 0:
        return float("nan")
    positive = int(np.sum(non_zero > 0))
    return max(positive, non_zero.size - positive) / non_zero.size
