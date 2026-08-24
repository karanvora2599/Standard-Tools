"""
Turning a continuous target into something a ranker can learn from, and
scoring the result.

WHY THIS EXISTS AT ALL. The pipeline judges a cross-sectional model on rank
IC — did it order the names correctly today — while every estimator in the
registry optimizes squared error or log loss. That mismatch is the reason to
add rankers: a learning-to-rank objective trains directly on the pairwise
ordering the scorecard measures, instead of fitting magnitudes and hoping the
ordering follows.

THE CONVERSION IS NOT OPTIONAL. Measured against LightGBM 4.5 and XGBoost
2.0, both rankers REJECT a continuous label outright:

    LightGBMError: label should be int type (met 0.810529) for ranking task
    XGBoostError:  ... (label must be graded relevance)

Shifting the returns to be non-negative does not help; the requirement is
integer relevance grades, not positivity. So a forward return has to become
a grade before a ranker can see it, and doing that badly is the easiest way
to make a ranker worse than the regression it replaced.

WHERE THE GRADES COME FROM. Within each date, independently: rank the
entities by the target and cut the ranking into `n_grades` equal buckets, 0
(worst) to n_grades-1 (best). Per date, because that is what the query group
IS — a ranker learns "AAPL should rank above MSFT *today*", and a grade
pooled across dates would silently be asking it to rank today's names against
last year's.

No leakage is introduced by this. The grades are a monotone transform of the
target within a date, and the target's own forward-looking-ness is already
handled by the embargo and the label-overlap purge.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .metrics import cross_sectional_ic, summarize_cross_sectional_ic


def relevance_grades(
    target: np.ndarray, dates: np.ndarray, n_grades: int = 8
) -> np.ndarray:
    """
    Integer relevance grades in [0, n_grades), assigned within each date.

    Ties take the same grade wherever the bucket boundary allows, because
    `rank(method="first")` would otherwise split genuinely equal targets
    across a boundary and teach the ranker an ordering that is not in the
    data. `method="average"` then flooring does that.

    A date with fewer entities than grades cannot fill the buckets. Rather
    than collapsing it to a couple of levels — which quietly changes what
    the objective is optimizing on those dates — its rows are graded across
    however many levels the cross-section supports.
    """
    if n_grades < 2:
        raise ValidationError(f"n_grades must be >= 2, got {n_grades}")
    target = np.asarray(target, dtype=float)
    dates = np.asarray(dates)
    if target.size != dates.size:
        raise ValidationError(
            f"target and dates must be the same length, got {target.size} and {dates.size}"
        )
    if target.size == 0:
        return np.empty(0, dtype=np.int32)

    frame = pd.DataFrame({"date": dates, "target": target})
    grouped = frame.groupby("date")["target"]
    counts = grouped.transform("count")
    ranks = grouped.rank(method="average")

    # Buckets are cut on the RANK, not on the value: a fat-tailed return
    # distribution would put almost every name in one value-bucket, which is
    # exactly the case a ranker is supposed to be robust to.
    levels = np.minimum(counts.to_numpy(dtype=float), float(n_grades))
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = (
            (ranks.to_numpy(dtype=float) - 1.0) * levels / counts.to_numpy(dtype=float)
        )
    grades = np.floor(np.nan_to_num(scaled, nan=0.0)).astype(np.int32)
    return np.clip(grades, 0, n_grades - 1)


def group_sizes(dates: np.ndarray) -> np.ndarray:
    """
    Query-group sizes for a DATE-SORTED array, as both rankers want them.

    Both libraries take `group` as consecutive counts and assume the rows
    are already ordered by group — they do not check. Handing them counts
    computed from unsorted rows produces a model that silently trained on
    the wrong groupings, so the caller sorts first and this raises rather
    than trusting.
    """
    dates = np.asarray(dates)
    if dates.size == 0:
        return np.empty(0, dtype=np.int64)
    codes = pd.factorize(dates, sort=False)[0]
    boundaries = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    counts = np.diff(np.r_[boundaries, codes.size])
    if len(np.unique(codes[boundaries])) != len(boundaries):
        raise ValidationError(
            "group_sizes: rows are not sorted by date — a date appears in more "
            "than one run. Both rankers assume consecutive query groups and do "
            "not verify it, so this would train on the wrong groupings."
        )
    return counts.astype(np.int64)


def ndcg_at_k(
    scores: np.ndarray, grades: np.ndarray, dates: np.ndarray, k: int
) -> float:
    """
    Mean NDCG@k across dates — the metric the ranking objective optimizes.

    Reported alongside rank IC rather than instead of it, because they say
    different things. Rank IC weighs the whole cross-section equally; NDCG's
    logarithmic discount weighs the top of the ranking far more heavily,
    which is closer to how a long-only or concentrated book actually uses a
    score. A model can improve one and not the other, and knowing which is
    the point of reporting both.

    Dates whose ideal DCG is zero — every name at grade 0, so no ordering is
    better than any other — are skipped rather than scored 0.0, which would
    be indistinguishable from ranking them backwards.
    """
    if k < 1:
        raise ValidationError(f"k must be >= 1, got {k}")
    frame = pd.DataFrame(
        {
            "date": np.asarray(dates),
            "score": np.asarray(scores, dtype=float),
            "grade": np.asarray(grades, dtype=float),
        }
    ).dropna()
    if frame.empty:
        return float("nan")

    values: List[float] = []
    for _, group in frame.groupby("date", sort=False):
        if len(group) < 2:
            continue
        by_score = group.sort_values("score", ascending=False)["grade"].to_numpy()[:k]
        by_grade = group.sort_values("grade", ascending=False)["grade"].to_numpy()[:k]
        discount = 1.0 / np.log2(np.arange(2, len(by_score) + 2))
        gain = np.power(2.0, by_score) - 1.0
        ideal_gain = np.power(2.0, by_grade) - 1.0
        ideal = float(np.sum(ideal_gain * discount[: len(by_grade)]))
        if ideal <= 0:
            continue
        values.append(float(np.sum(gain * discount)) / ideal)
    return float(np.mean(values)) if values else float("nan")


def ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    dates: np.ndarray,
    n_grades: int = 8,
    ks: "tuple[int, ...]" = (5, 10),
) -> Dict[str, float]:
    """
    Out-of-sample metrics for a ranking model.

    Deliberately NOT `regression_metrics`. A ranker's output is an ordering
    score on an arbitrary scale — LambdaRank is invariant to any monotone
    transform of it — so R2 and MAE against a return would be measuring the
    scale of a quantity that has none. Reporting them would invite exactly
    the comparison they cannot support.

    What is reported instead is what survives that invariance: the
    cross-sectional ICs, which depend only on the ordering, and NDCG, which
    is what the objective optimized.
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    dates = np.asarray(dates)

    metrics: Dict[str, float] = {}
    for method, prefix in (("pearson", "cs_ic"), ("spearman", "cs_rank_ic")):
        series = cross_sectional_ic(y_true, scores, dates, method)
        metrics.update(summarize_cross_sectional_ic(series, prefix))

    grades = relevance_grades(y_true, dates, n_grades)
    for k in ks:
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(scores, grades, dates, k)
    return metrics


def fold_ic_series(
    y_true: np.ndarray, scores: np.ndarray, dates: np.ndarray
) -> Dict[str, pd.Series]:
    """The per-date IC series a ranking fold contributes to the pooled OOS
    dispersion statistics, in the same shape the regression path returns."""
    return {
        "cs_ic": cross_sectional_ic(y_true, scores, dates, "pearson"),
        "cs_rank_ic": cross_sectional_ic(y_true, scores, dates, "spearman"),
    }
