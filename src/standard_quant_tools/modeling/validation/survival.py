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

from typing import Dict, Optional, Tuple

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


def survival_metrics(
    y_true: np.ndarray,
    risk: np.ndarray,
    dates: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Concordance pooled over the fold, the per-date concordance summarized
    across dates, and how much of the label was actually observed.
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
    return out


__all__ = ["EVENT_COL", "concordance_index", "survival_labels", "survival_metrics"]
