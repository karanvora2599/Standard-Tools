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


def _segment_bounds(codes: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Start offset and length of each run in an already-grouped code array."""
    if codes.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    return starts, np.diff(np.r_[starts, codes.size])


def _rank_segments(
    values: np.ndarray,
    codes: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """
    Average ranks within each date, for a ragged (variable-width) panel.

    Matches Series.rank()'s default "average" tie handling, which is what
    pandas' spearman uses internally. Ties are the part that is easy to get
    wrong: every member of a tied run must receive the run's MEAN ordinal,
    not its first or last, or the resulting correlation quietly disagrees
    with the implementation this replaced.
    """
    n = values.size
    # Primary key `codes`, secondary key `values`: orders each date's
    # cross-section without disturbing the date grouping.
    order = np.lexsort((values, codes))
    sorted_values = values[order]
    sorted_codes = codes[order]
    # 1-based position within the row's own date.
    ordinal = np.arange(n, dtype=np.float64) - np.repeat(starts, counts) + 1.0
    # A tie run is consecutive equal values inside a single date.
    new_run = np.r_[
        True,
        (sorted_codes[1:] != sorted_codes[:-1])
        | (sorted_values[1:] != sorted_values[:-1]),
    ]
    run_starts = np.flatnonzero(new_run)
    run_counts = np.diff(np.r_[run_starts, n])
    run_means = np.add.reduceat(ordinal, run_starts) / run_counts
    out = np.empty(n, dtype=np.float64)
    out[order] = np.repeat(run_means, run_counts)
    return out


def _rank_rows(block: np.ndarray) -> np.ndarray:
    """Average ranks along axis 1 of a balanced (n_dates, n_entities) block."""
    n_rows, width = block.shape
    order = np.argsort(block, axis=1, kind="stable")
    sorted_block = np.take_along_axis(block, order, axis=1)
    new_run = np.empty((n_rows, width), dtype=bool)
    new_run[:, 0] = True
    np.not_equal(sorted_block[:, 1:], sorted_block[:, :-1], out=new_run[:, 1:])
    # Tie-run id within each row, offset per row so one bincount covers all.
    group = np.cumsum(new_run, axis=1) - 1
    flat = (group + np.arange(n_rows, dtype=np.int64)[:, None] * width).ravel()
    ordinal = np.broadcast_to(np.arange(1.0, width + 1.0), (n_rows, width)).ravel()
    sums = np.bincount(flat, weights=ordinal, minlength=n_rows * width)
    counts = np.bincount(flat, minlength=n_rows * width)
    means = (sums / np.maximum(counts, 1)).reshape(n_rows, width)
    out = np.empty_like(block)
    np.put_along_axis(out, order, np.take_along_axis(means, group, axis=1), axis=1)
    return out


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

    IMPLEMENTATION. This was measured at 72% of a ridge walk-forward run,
    because the obvious version — groupby("date") then Series.corr per date
    — pays full pandas construction and dispatch overhead thousands of
    times per run. Every date's correlation is an independent reduction
    over that date's rows, so the whole panel is a handful of array passes
    instead. Two layouts, because they are not equally good:

      * balanced panel (every date has the same entity count): reshape to
        (n_dates, n_entities) and reduce along axis 1 — contiguous 2-D work
      * ragged panel (entities enter and leave): np.add.reduceat over the
        segment boundaries

    The balanced path measured faster than the ragged one at every size
    tested, so it is preferred whenever it applies; the ragged path is a
    correctness fallback, not a slow mode for unusual data. Both agree with
    the previous per-date implementation to floating point (worst observed
    |diff| 1.1e-16 spearman, 2.2e-14 pearson, ties and ragged panels
    included).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    dates = np.asarray(dates)

    if y_true.size == 0:
        return pd.Series(dtype=float)

    all_codes, uniques = pd.factorize(dates, sort=True)
    n_dates = len(uniques)
    # Which dates appear in the output is decided by the RAW row count, not
    # the count of usable rows. That is the rule the per-date version used
    # (`if len(group) < 2: continue` ran before Series.corr saw any NaN), so
    # a date with two all-NaN rows was emitted as exactly 0.0 rather than
    # dropped. Preserved deliberately: this replacement is a speed change
    # and must not quietly move a reported metric. It is arguably the wrong
    # rule — a date with no usable data is not a date with zero IC, and it
    # drags cs_ic_mean and cs_ic_hit_rate toward zero — but changing it is a
    # separate decision from making it fast.
    emit = np.bincount(all_codes, minlength=n_dates) >= 2
    if not emit.any():
        return pd.Series(dtype=float)

    # Series.corr drops NaN PAIRWISE before correlating, and for spearman
    # ranks only what survives — so removing incomplete rows here is the
    # same computation, not an approximation. Infinities are NOT dropped:
    # pandas treats only NaN as missing, so inf is left to flow through,
    # where it ranks as an extreme for spearman and pushes pearson into the
    # non-finite -> 0.0 branch below. Both match pandas.
    complete = ~(np.isnan(y_true) | np.isnan(y_pred))
    codes = all_codes[complete]
    y_true = y_true[complete]
    y_pred = y_pred[complete]

    ic_by_date = np.zeros(n_dates, dtype=np.float64)
    if codes.size == 0:
        # Every usable row was NaN; every emitted date correlates to 0.0.
        return pd.Series(ic_by_date[emit], index=uniques[emit], dtype=float)

    order = np.argsort(codes, kind="stable")
    codes = codes[order]
    y_true = y_true[order]
    y_pred = y_pred[order]
    starts, counts = _segment_bounds(codes)

    width = int(counts[0])
    # An infinite input makes the centering step evaluate inf - inf, and a
    # constant cross-section divides 0 by 0. Both are how an undefined
    # correlation is SUPPOSED to arrive here — it becomes 0.0 below, which
    # is what pandas produced too — so the resulting warnings are noise.
    with np.errstate(invalid="ignore", divide="ignore"):
        if bool(np.all(counts == width)) and width >= 2:
            left = y_true.reshape(-1, width)
            right = y_pred.reshape(-1, width)
            if method == "spearman":
                left = _rank_rows(left)
                right = _rank_rows(right)
            # Centered (two-pass) form, not the n*Sxy - Sx*Sy shortcut. On
            # return-scale data the shortcut differences two nearly equal
            # large numbers and loses most of its significant digits; this
            # is what np.corrcoef does and it costs one extra pass.
            left = left - left.mean(axis=1, keepdims=True)
            right = right - right.mean(axis=1, keepdims=True)
            cov = np.einsum("ij,ij->i", left, right)
            var_left = np.einsum("ij,ij->i", left, left)
            var_right = np.einsum("ij,ij->i", right, right)
        else:
            if method == "spearman":
                y_true = _rank_segments(y_true, codes, starts, counts)
                y_pred = _rank_segments(y_pred, codes, starts, counts)
            widths = counts.astype(np.float64)
            y_true = y_true - np.repeat(
                np.add.reduceat(y_true, starts) / widths, counts
            )
            y_pred = y_pred - np.repeat(
                np.add.reduceat(y_pred, starts) / widths, counts
            )
            cov = np.add.reduceat(y_true * y_pred, starts)
            var_left = np.add.reduceat(y_true * y_true, starts)
            var_right = np.add.reduceat(y_pred * y_pred, starts)

        ic = cov / np.sqrt(var_left * var_right)
    # Undefined correlations land here and become 0.0, matching _safe_corr:
    # a constant cross-section (zero variance), and a date left with fewer
    # than two usable rows after the NaN drop (zero or one point).
    ic_by_date[codes[starts]] = np.where(np.isfinite(ic), ic, 0.0)
    return pd.Series(ic_by_date[emit], index=uniques[emit], dtype=float)


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


def baseline_regression_metrics(
    y_true: np.ndarray, train_y: "np.ndarray | None" = None
) -> Dict[str, float]:
    """
    Metrics for the trivial "predict the mean" model, so a reported R2/MAE
    has something to be compared against. Without this a result can look
    informative while being no better than a constant.

    The constant comes from `train_y` — the TRAINING fold's mean — not from
    y_true. Using the test fold's own mean made this an ORACLE baseline: at
    prediction time nobody knows the future window's average realized
    return, so a model compared against it was being held to a standard no
    real forecaster could meet, and `model MAE vs baseline MAE` was not a
    valid comparison. It never contaminated the trained model itself, only
    the number it was judged against.

    `train_y=None` falls back to the old in-sample constant and is reported
    as such via `baseline_is_oracle`, so a caller can tell which of the two
    they are looking at rather than having to infer it.
    """
    if len(y_true) == 0:
        return {
            "baseline_mae": float("nan"),
            "baseline_r2": 0.0,
            "baseline_is_oracle": 0.0,
        }
    if train_y is not None and len(train_y) > 0:
        constant_value = float(np.mean(np.asarray(train_y, dtype=float)))
        is_oracle = 0.0
    else:
        constant_value = float(np.mean(y_true))
        is_oracle = 1.0
    constant = np.full_like(np.asarray(y_true, dtype=float), constant_value)
    # R2 against a TRAINING-derived constant is no longer 0.0 by
    # construction -- that identity only held for the in-sample mean -- so
    # it is computed rather than asserted.
    return {
        "baseline_mae": float(mean_absolute_error(y_true, constant)),
        "baseline_r2": float(r2_score(y_true, constant)) if len(y_true) > 1 else 0.0,
        "baseline_is_oracle": is_oracle,
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: "np.ndarray | None" = None,
    train_y: "np.ndarray | None" = None,
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
    metrics.update(baseline_regression_metrics(y_true, train_y))
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


def aggregate_cross_sectional_ic(
    fold_ic_series: "List[pd.Series]", prefix: str
) -> Dict[str, float]:
    """
    Dispersion statistics over the POOLED out-of-sample daily IC series.

    average_fold_metrics computes a weighted mean across folds, which is
    correct for a mean but not for the statistics built on top of one:

        mean(fold standard deviations)  !=  std(all OOS daily ICs)
        mean(fold ICIRs)                !=  mean(all ICs) / std(all ICs)

    A fold's std measures dispersion WITHIN that fold's dates only, so
    averaging folds' stds discards the between-fold variation entirely --
    exactly the variation that says whether an IC is dependable across
    regimes, which is the question ICIR exists to answer. Concatenating
    every fold's dates and computing once gives the actual OOS quantity.

    The per-fold versions are still reported in validation_report, where
    they answer a different and also useful question ("did this fold work").
    """
    usable = [s for s in fold_ic_series if s is not None and not s.empty]
    if not usable:
        return summarize_cross_sectional_ic(pd.Series(dtype=float), prefix)
    # Folds are disjoint in time by construction (walk-forward), so a plain
    # concat is the full OOS date series with no double counting.
    return summarize_cross_sectional_ic(pd.concat(usable).sort_index(), prefix)


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
