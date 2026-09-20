"""
Judging a survival model: did it order the durations the right way.

A survival label is two numbers per row -- how long until the event, and
whether the event was observed at all -- and a model of it emits a RISK
score: higher means sooner. The proper question is not how far the score
is from the duration (it is not a duration) but whether, of two rows where
the earlier one's event was seen, the model gave that one the higher risk.
That is Harrell's concordance index: the fraction of such comparable
pairs the model orders correctly, ties in risk counting half. A censored
row can only ever be the LATER member of a pair, because nobody knows
when its event came; that is where the censoring enters the metric and
why a regression's R2 on the same label means nothing.

The cross-sectional version asks the same of each date's rows alone --
did the model order today's names right -- which is the question a
cross-sectional model is built to answer and the one the rest of this
runtime leads with.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

#: The panel column carrying the event indicator beside `target`.
EVENT_COL = "event"

#: Rows per block when the pairwise comparison is vectorized; keeps the
#: (block x n) intermediate at a few megabytes.
_BLOCK = 512


def survival_labels(frame: pd.DataFrame) -> np.ndarray:
    """
    The (n, 2) label a survival estimator fits: [duration, event].

    Read off the panel's `target` and `event` columns; refused, by name,
    when the event indicator is missing or is not 0/1, because a duration
    without one would be fitted as though every row's event had been seen.
    """
    if EVENT_COL not in frame.columns:
        raise ValidationError(
            "a survival label needs an `event` column beside `target`: 1 where "
            "the event was observed, 0 where the window ended first. An "
            "external panel declares it as `event_column` on the target."
        )
    duration = frame["target"].to_numpy(dtype=float)
    event = frame[EVENT_COL].to_numpy(dtype=float)
    if not np.isin(event[~np.isnan(event)], (0.0, 1.0)).all():
        raise ValidationError(
            "the `event` column must be 0 or 1 on every row; it says whether "
            "the event was observed, not when."
        )
    return np.column_stack([duration, event])


def concordance_index(
    durations: np.ndarray, events: np.ndarray, risk: np.ndarray
) -> Tuple[float, int]:
    """
    Harrell's C and the number of comparable pairs it was read from.

    A pair (i, j) is comparable when `durations[i] < durations[j]` and
    row i's event was observed; it is concordant when `risk[i] > risk[j]`,
    tied when the risks are equal (counted half). Pairs tied in duration
    are not comparable. NaN with zero pairs, which is the honest answer
    for a window with no observed event.
    """
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=float)
    r = np.asarray(risk, dtype=float)
    n = t.size
    if n < 2:
        return float("nan"), 0
    concordant = 0.0
    comparable = 0
    for start in range(0, n, _BLOCK):
        stop = min(n, start + _BLOCK)
        earlier = (t[start:stop, None] < t[None, :]) & (e[start:stop, None] == 1.0)
        comparable += int(earlier.sum())
        if not earlier.any():
            continue
        diff = r[start:stop, None] - r[None, :]
        concordant += float((earlier & (diff > 0)).sum())
        concordant += 0.5 * float((earlier & (diff == 0)).sum())
    if comparable == 0:
        return float("nan"), 0
    return float(concordant / comparable), int(comparable)


def censoring_distribution(
    durations: np.ndarray, events: np.ndarray
) -> "tuple[np.ndarray, np.ndarray]":
    """
    The Kaplan-Meier estimate of the CENSORING survival G(t) = P(C > t),
    with events and censorings swapped and a tie between the two resolved
    with the event first -- the convention the inverse-probability weights
    below need, and the one scikit-survival uses, so the two agree to the
    last digit. Returned as the distinct times and G at each.
    """
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=float)
    order = np.argsort(t, kind="stable")
    ts, es = t[order], e[order]
    unique_t, first = np.unique(ts, return_index=True)
    n_at_risk = ts.size - first
    group = np.searchsorted(unique_t, ts)
    n_events = np.bincount(group, weights=es, minlength=unique_t.size)
    n_censored = np.bincount(group, weights=1.0 - es, minlength=unique_t.size)
    denominator = n_at_risk - n_events
    ratio = np.divide(
        n_censored,
        denominator,
        out=np.zeros_like(n_censored),
        where=(n_censored != 0) & (denominator > 0),
    )
    return unique_t, np.cumprod(1.0 - ratio)


def _step_at(times: np.ndarray, knots: np.ndarray, values: np.ndarray) -> np.ndarray:
    """A right-continuous step function, 1 before its first knot."""
    index = np.searchsorted(knots, np.asarray(times, dtype=float), side="right") - 1
    return np.where(index >= 0, values[np.clip(index, 0, None)], 1.0)


def brier_time_grid(
    train_y: np.ndarray, test_y: np.ndarray, *, max_points: int = 64
) -> np.ndarray:
    """
    Where the Brier score is read: the distinct test event times inside
    BOTH follow-ups -- at or after the later start, strictly before the
    earlier end -- thinned evenly to `max_points`. Empty when fewer than
    two remain, which is the honest answer for a fold too short to score.
    """
    train = np.asarray(train_y, dtype=float)
    test = np.asarray(test_y, dtype=float)
    low = max(train[:, 0].min(), test[:, 0].min())
    high = min(train[:, 0].max(), test[:, 0].max())
    candidates = np.unique(test[test[:, 1] == 1.0, 0])
    candidates = candidates[(candidates >= low) & (candidates < high)]
    if candidates.size < 2:
        return np.empty(0, dtype=float)
    if candidates.size > max_points:
        picks = np.linspace(0, candidates.size - 1, max_points).round().astype(int)
        candidates = candidates[np.unique(picks)]
    return candidates


def brier_scores(
    train_y: np.ndarray,
    test_y: np.ndarray,
    survival_probs: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """
    The time-dependent Brier score of Graf et al. (1999) at each of
    `times`: the squared error of S(t | x) against "still going at t",
    with each row weighted by the inverse probability of NOT having been
    censored by the time its contribution is decided -- so a censored row
    counts fully while it is still observed and not at all after, and the
    rows that are observed stand in for those that are not. G is estimated
    on the TRAINING labels, the same reference the model was fitted on.
    """
    train = np.asarray(train_y, dtype=float)
    test = np.asarray(test_y, dtype=float)
    grid = np.asarray(times, dtype=float)
    S = np.asarray(survival_probs, dtype=float)
    if S.shape != (test.shape[0], grid.size):
        raise ValidationError(
            f"survival probabilities must be (n_test, n_times) = "
            f"({test.shape[0]}, {grid.size}); got {S.shape}."
        )
    knots, G = censoring_distribution(train[:, 0], train[:, 1])
    g_at_times = _step_at(grid, knots, G)
    g_at_rows = _step_at(test[:, 0], knots, G)
    g_at_times = np.where(g_at_times == 0, np.inf, g_at_times)
    g_at_rows = np.where(g_at_rows == 0, np.inf, g_at_rows)
    scores = np.empty(grid.size, dtype=float)
    for i, t in enumerate(grid):
        est = S[:, i]
        is_case = ((test[:, 0] <= t) & (test[:, 1] == 1.0)).astype(float)
        is_control = (test[:, 0] > t).astype(float)
        scores[i] = np.mean(
            np.square(est) * is_case / g_at_rows
            + np.square(1.0 - est) * is_control / g_at_times[i]
        )
    return scores


def integrated_brier_score(scores: np.ndarray, times: np.ndarray) -> float:
    """The trapezoid integral of the Brier curve over its grid, divided by
    the grid's span: one number in [0, 1], lower is better, 0.25 is a
    coin flip at every horizon."""
    bs = np.asarray(scores, dtype=float)
    grid = np.asarray(times, dtype=float)
    if grid.size < 2:
        return float("nan")
    area = np.sum((bs[1:] + bs[:-1]) / 2.0 * np.diff(grid))
    return float(area / (grid[-1] - grid[0]))


def survival_metrics(
    y_true: np.ndarray,
    risk: np.ndarray,
    dates: Optional[np.ndarray] = None,
    *,
    train_y: Optional[np.ndarray] = None,
    survival_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Concordance pooled over the fold, the per-date concordance summarized
    across dates, how much of the label was actually observed, and --
    when the estimator can say how likely each row is to have gone by a
    given time -- the integrated Brier score of those probabilities.

    `survival_function(times)` returns S(t | x) for the fold's test rows
    at `times`; `train_y` is the training label the censoring weights
    are estimated on. Without both, the Brier score is not reported
    rather than reported as something else.
    """
    y = np.asarray(y_true, dtype=float)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValidationError(
            "survival_metrics expects a (n, 2) label of [duration, event]."
        )
    durations, events = y[:, 0], y[:, 1]
    score = np.asarray(risk, dtype=float)
    c, pairs = concordance_index(durations, events, score)
    out: Dict[str, float] = {
        "concordance": c,
        "n_comparable_pairs": float(pairs),
        "event_rate": float(np.nanmean(events)) if events.size else float("nan"),
        "n_events": float(np.nansum(events)),
    }
    if dates is not None:
        per_date = []
        frame = pd.DataFrame(
            {"date": np.asarray(dates), "t": durations, "e": events, "r": score}
        )
        for _date, group in frame.groupby("date", sort=True):
            if len(group) < 2 or group["e"].sum() == 0:
                continue
            value, _n = concordance_index(
                group["t"].to_numpy(), group["e"].to_numpy(), group["r"].to_numpy()
            )
            if np.isfinite(value):
                per_date.append(value)
        series = np.asarray(per_date, dtype=float)
        out["cs_concordance_mean"] = (
            float(series.mean()) if series.size else float("nan")
        )
        out["cs_concordance_std"] = (
            float(series.std(ddof=1)) if series.size > 1 else float("nan")
        )
        out["cs_concordance_n_dates"] = float(series.size)
    if train_y is not None and survival_function is not None:
        times = brier_time_grid(train_y, y)
        if times.size >= 2:
            probabilities = np.asarray(survival_function(times), dtype=float)
            curve = brier_scores(train_y, y, probabilities, times)
            out["integrated_brier"] = integrated_brier_score(curve, times)
            out["brier_n_times"] = float(times.size)
            out["brier_horizon_min"] = float(times[0])
            out["brier_horizon_max"] = float(times[-1])
        else:
            # Too few event times inside both follow-ups to integrate
            # over; NaN keeps the key and says so.
            out["integrated_brier"] = float("nan")
    return out


__all__ = [
    "EVENT_COL",
    "brier_scores",
    "brier_time_grid",
    "censoring_distribution",
    "concordance_index",
    "integrated_brier_score",
    "survival_labels",
    "survival_metrics",
]
