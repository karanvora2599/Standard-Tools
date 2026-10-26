"""
Judging a distribution, not a point.

A point prediction is scored by how far it lands from the outcome. A
quantile is scored by whether the outcome fell on the right side of it
the right fraction of the time, and the pinball loss is the proper score
for that: a 95th percentile that is exceeded 5% of the time minimizes it,
one that is exceeded 20% of the time does not, however tight it looks. An
interval is scored by two numbers that must be read together, coverage
and width -- an interval covering 90% of outcomes by being wide enough to
cover everything has told nobody anything.

Quantile crossing is reported because it is the failure a set of
separately fitted quantiles can have and a single fit cannot: the 5th
percentile predicted above the 95th on some row is not a distribution, and
a caller building an interval from the pair needs to know how often that
happened rather than discover a negative width.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


def quantile_column(q: float) -> str:
    """The OOS column for quantile `q`: `q05`, `q50`, `q95`; `q02.5` for a
    level that is not a whole percent."""
    percent = float(q) * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return f"q{int(round(percent)):02d}"
    return f"q{percent:04.1f}"


#: The columns a model writes for an interval, beside its quantile columns.
INTERVAL_COLUMNS = ("lower", "upper")


def quantile_columns(columns: Iterable[object]) -> Dict[str, float]:
    """
    Which of `columns` are quantile columns, and at what level, by the round
    trip that wrote them.

    A column is a quantile column when parsing its name back to a level
    and re-deriving the name yields the same string -- so `q05` is 0.05
    and `q02.5` is 0.025, while `q5`, `quantile` and `q120` are not
    quantile columns at all. Recognising them by the writer's own
    function rather than a second regex is what keeps a rename on one
    side from silently producing an empty result on the other.
    """
    found: Dict[str, float] = {}
    for column in columns:
        name = str(column)
        if not name.startswith("q"):
            continue
        try:
            percent = float(name[1:])
        except ValueError:
            continue
        level = percent / 100.0
        if not 0.0 < level < 1.0:
            continue
        if quantile_column(level) == name:
            found[name] = level
    return found


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss at level `q`; at 0.5 it is half the MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    loss = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
    return float(np.mean(loss)) if loss.size else float("nan")


def distributional_metrics(
    y_true: np.ndarray,
    quantile_predictions: Dict[float, np.ndarray],
    *,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    alpha: Optional[float] = None,
) -> Dict[str, float]:
    """
    Per-quantile pinball loss, the crossing rate, coverage and width of
    every symmetric pair of requested quantiles, and coverage and width of
    a conformal interval when one is supplied.

    Reported beside the point metrics under names that say which level
    they describe: `pinball_q05`, `quantile_coverage_90`,
    `interval_coverage`.
    """
    y = np.asarray(y_true, dtype=float)
    out: Dict[str, float] = {}
    levels = sorted(quantile_predictions)
    for q in levels:
        out[f"pinball_{quantile_column(q)}"] = pinball_loss(
            y, quantile_predictions[q], q
        )
    if len(levels) >= 2:
        stacked = np.column_stack([np.asarray(quantile_predictions[q]) for q in levels])
        crossed = (np.diff(stacked, axis=1) < 0).any(axis=1)
        out["quantile_crossing_rate"] = float(crossed.mean()) if crossed.size else 0.0
        for low in levels:
            if low >= 0.5:
                continue
            high = next((q for q in levels if abs(q - (1.0 - low)) < 1e-9), None)
            if high is None:
                continue
            level = int(round((high - low) * 100))
            lo = np.asarray(quantile_predictions[low], dtype=float)
            hi = np.asarray(quantile_predictions[high], dtype=float)
            out[f"quantile_coverage_{level}"] = float(np.mean((y >= lo) & (y <= hi)))
            out[f"quantile_width_{level}"] = float(np.mean(hi - lo))
    if lower is not None and upper is not None:
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        out["interval_coverage"] = float(np.mean((y >= lo) & (y <= hi)))
        out["interval_width"] = float(np.mean(hi - lo))
        if alpha is not None:
            out["interval_nominal_coverage"] = float(1.0 - alpha)
    return out


__all__ = ["distributional_metrics", "pinball_loss", "quantile_column"]
