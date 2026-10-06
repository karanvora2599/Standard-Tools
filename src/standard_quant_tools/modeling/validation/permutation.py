"""
Permutation importance: what the model loses when a feature is scrambled.

WHY THIS AND NOT ABLATION. `run_feature_ablation` was the only
model-relative importance here, and the library calls it "EXPENSIVE -- 40
features at 8 folds is 328 fits" and tells agents to narrow away from it,
which inverts the question it answers: you narrow using the thing you
wanted the answer in order to narrow. Permutation costs one PREDICT per
feature per repeat per fold instead of one FIT, so the same 40 features at
8 folds is 320 predicts -- the same shape of loop against a cost two or
three orders of magnitude smaller.

They do not measure the same thing and both are worth having. Ablation
REFITS without the feature, so it answers "would a model built without this
have been worse" -- it lets the remaining features take over the dropped
one's job. Permutation keeps the fitted model and destroys the feature's
information, so it answers "does THIS model use it". A feature with a
perfect substitute scores near zero under ablation and can score high under
permutation; that difference is information, not a discrepancy.

SHUFFLED WITHIN THE DATE, not across the panel. A global shuffle moves a
value from one date's cross-section into another's, which destroys the
feature's cross-sectional ordering AND breaks its time structure at once --
so the drop confounds "the model used this feature" with "the panel stopped
being a panel". Within a date, every cross-section keeps its own values and
only who holds them changes, which is exactly the null the feature lab's
own screens use.

MEASURED IN THE FOLD'S OWN METRIC. The caller supplies `score`, and the
engine supplies the fold's own `_predict_fold` behind it, so a drop is
reported in the number that fold already reports and not in a second metric
invented here.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

#: Repeats a caller gets when it does not say. One repeat is a single draw
#: of a random permutation and its drop carries the variance of that draw,
#: which on a short test window is most of the number; five is enough for
#: the spread to mean something without the cost mattering.
DEFAULT_REPEATS = 5


def shuffle_within_date(
    values: np.ndarray, codes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """One column permuted inside each date, every other row left alone.

    `codes` is the per-row date code; rows sharing a code are one
    cross-section. A date with one row is unchanged by construction, which
    is right: there is nothing to permute it with, and leaving it alone
    keeps the baseline and the permuted score comparable on those rows.
    """
    out = np.array(values, copy=True)
    order = np.argsort(codes, kind="stable")
    boundaries = np.flatnonzero(np.diff(codes[order])) + 1
    for block in np.split(order, boundaries):
        if block.size > 1:
            out[block] = out[rng.permutation(block)]
    return out


def permutation_importance(
    score: Callable[[pd.DataFrame], float],
    X: pd.DataFrame,
    dates: np.ndarray,
    *,
    columns: Optional[Sequence[str]] = None,
    n_repeats: int = DEFAULT_REPEATS,
    seed: int = 0,
) -> Dict[str, object]:
    """
    How much `score` falls when each column is scrambled within its date.

    `score(frame) -> float` is the caller's own metric on those rows, so
    the drop is in the units the caller already reports. Higher is better
    is assumed -- the engine passes its headline, which is -- and a
    NEGATIVE drop means the model scored BETTER without the feature's real
    values, which happens and is reported rather than clipped to zero: it
    is evidence the feature is noise this model is fitting.

    Returns `{"baseline": float, "columns": {name: {...}}}`, with each
    column carrying the mean drop, its standard deviation across repeats,
    and the repeats it rests on.
    """
    if n_repeats < 1:
        raise ValidationError(
            f"permutation_importance: n_repeats must be at least 1, got {n_repeats}."
        )
    if len(dates) != len(X):
        raise ValidationError(
            "permutation_importance needs one date per row: got "
            f"{len(dates)} dates for {len(X)} rows."
        )
    names = list(X.columns if columns is None else columns)
    unknown = [name for name in names if name not in X.columns]
    if unknown:
        raise ValidationError(
            f"permutation_importance: no such column(s) {unknown}."
        )

    baseline = float(score(X))
    if not np.isfinite(baseline):
        raise ValidationError(
            "permutation_importance: the baseline score is not finite "
            f"({baseline}), so no drop from it would mean anything."
        )

    # One code per distinct date, so the shuffle groups rows without
    # sorting the frame or depending on it arriving sorted.
    _, codes = np.unique(np.asarray(dates), return_inverse=True)
    rng = np.random.default_rng(seed)

    out: Dict[str, Dict[str, float]] = {}
    for name in names:
        values = X[name].to_numpy()
        drops = []
        for _ in range(int(n_repeats)):
            permuted = X.copy()
            permuted[name] = shuffle_within_date(values, codes, rng)
            drops.append(baseline - float(score(permuted)))
        array = np.asarray(drops, dtype=float)
        out[name] = {
            "mean_drop": float(np.mean(array)),
            "std_drop": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
            "n_repeats": int(array.size),
        }
    return {"baseline": baseline, "columns": out}


def summarize_permutation(per_fold: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    """
    One importance per feature across the folds that measured it.

    The mean of the per-fold mean drops, and the spread ACROSS FOLDS beside
    it -- which is the number worth reading. A feature that matters in one
    fold and not the others is a feature that mattered in one regime, and a
    mean alone hides that as effectively as a mean of fold metrics hides an
    unstable model. Folds that did not measure a feature do not count
    against it; `n_folds` says how many did.
    """
    gathered: Dict[str, list] = {}
    for fold in per_fold:
        for name, record in (fold or {}).get("columns", {}).items():  # type: ignore[union-attr]
            gathered.setdefault(name, []).append(float(record["mean_drop"]))
    summary: Dict[str, Dict[str, float]] = {}
    for name, drops in gathered.items():
        array = np.asarray(drops, dtype=float)
        summary[name] = {
            "mean_drop": float(np.mean(array)),
            "std_drop_across_folds": (
                float(np.std(array, ddof=1)) if array.size > 1 else 0.0
            ),
            "n_folds": int(array.size),
        }
    return summary


__all__ = [
    "DEFAULT_REPEATS",
    "permutation_importance",
    "shuffle_within_date",
    "summarize_permutation",
]
