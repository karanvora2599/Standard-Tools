"""
Hyperparameter search inside a fold's training window.

WHY NOT GridSearchCV. sklearn's search helpers cross-validate by splitting
ROWS. A modeling panel is stacked (entity, date) rows, so an ordinary
K-fold puts the same date in both the training and the scoring half of an
inner split — every entity on that date is a near-duplicate of the others,
and the search then selects whichever hyperparameter best memorizes them.
The selection is leaked even though the outer walk-forward split is clean,
and the damage shows up as hyperparameters that look excellent in-search
and disappoint out-of-sample. Splitting on DATES, forward in time, is the
only version of this that means anything here.

WHAT IT COSTS. Roughly (grid size x inner_splits) extra fits per outer
fold, on top of the one fit that fold already did. A 12-point grid with 3
inner splits over 20 outer folds is 720 fits where there was 20. That is
the honest price of not hand-picking `alpha`, and it is why the search is
opt-in.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .metrics import cross_sectional_ic
from .walk_forward import WalkForwardSplit

logger = logging.getLogger(__name__)


def _candidates(search_spec: Any, random_seed: int) -> List[Dict[str, Any]]:
    """The parameter combinations to try, in a deterministic order."""
    names = sorted(search_spec.param_grid)
    grid = [
        dict(zip(names, combo))
        for combo in itertools.product(*(search_spec.param_grid[n] for n in names))
    ]
    if search_spec.method == "grid" or len(grid) <= search_spec.n_iter:
        return grid
    # Sampled WITHOUT replacement from the enumerated grid rather than by
    # drawing from each axis independently: the same combination twice
    # would spend budget re-measuring a candidate already scored.
    rng = np.random.default_rng(random_seed)
    picks = rng.choice(len(grid), size=search_spec.n_iter, replace=False)
    return [grid[int(i)] for i in sorted(picks)]


def _score(
    task: str,
    scoring: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray],
    dates: np.ndarray,
) -> float:
    """
    One inner fold's score, always oriented so that HIGHER IS BETTER.

    Returns NaN for a fold the metric cannot be computed on (a single-class
    AUC window, say); the caller averages with NaN ignored so one awkward
    inner fold does not disqualify an otherwise good candidate.
    """
    from sklearn.metrics import (
        accuracy_score,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
    )

    if scoring in ("cs_rank_ic", "cs_ic"):
        method = "spearman" if scoring == "cs_rank_ic" else "pearson"
        series = cross_sectional_ic(y_true, predictions, dates, method)
        return float(series.mean()) if len(series) else float("nan")
    if scoring == "r2":
        return float(r2_score(y_true, predictions))
    if scoring == "neg_mae":
        return -float(mean_absolute_error(y_true, predictions))
    if scoring == "accuracy":
        return float(accuracy_score(y_true, predictions))
    if scoring == "auc":
        if probabilities is None:
            return float("nan")
        try:
            return float(roc_auc_score(y_true, probabilities))
        except ValueError:
            return float("nan")
    raise ValidationError(f"unknown scoring metric {scoring!r}")


def _inner_splitter(n_dates: int, inner_splits: int) -> Optional[WalkForwardSplit]:
    """
    Size an inner walk-forward so it yields exactly `inner_splits` folds.

    Returns None when the training window is too short to be split that
    many times — the caller then skips the search for that fold rather
    than silently searching on one or two dates, which would select on
    noise and be worse than not searching at all.
    """
    test_window = n_dates // (inner_splits + 1)
    if test_window < 1:
        return None
    train_window = n_dates - inner_splits * test_window
    if train_window < 1:
        return None
    return WalkForwardSplit(
        train_window=train_window, test_window=test_window, embargo=0
    )


def search_best_params(
    *,
    task: str,
    search_spec: Any,
    base_params: Dict[str, Any],
    train_frame: pd.DataFrame,
    feature_ids: List[str],
    random_seed: int,
    fit_predict: Callable[
        [Dict[str, Any], pd.DataFrame, pd.DataFrame],
        Tuple[np.ndarray, Optional[np.ndarray]],
    ],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Choose estimator parameters using only `train_frame`.

    `fit_predict(params, inner_train, inner_test)` is supplied by the
    engine so the search reuses the engine's own preprocessing and
    weighting rather than reimplementing them — a search that normalized
    its data differently from the final fit would select for the wrong
    thing.

    Returns (best_params, report). The report carries every candidate's
    score, because "which alpha won" is much less informative than "the
    top four alphas were within 0.001 of each other", and only the second
    tells a reader the search did not actually find anything.
    """
    dates = pd.Index(sorted(train_frame["date"].unique()))
    splitter = _inner_splitter(len(dates), search_spec.inner_splits)
    if splitter is None:
        return dict(base_params), {
            "searched": False,
            "reason": (
                f"training window has {len(dates)} dates, too few for "
                f"{search_spec.inner_splits} inner folds"
            ),
        }

    date_code = np.searchsorted(dates.to_numpy(), train_frame["date"].to_numpy())
    folds = list(splitter.split(dates))
    candidates = _candidates(search_spec, random_seed)

    results: List[Dict[str, Any]] = []
    for params in candidates:
        merged = {**base_params, **params}
        fold_scores: List[float] = []
        for train_pos, test_pos in folds:
            in_train = np.zeros(len(dates), dtype=bool)
            in_train[train_pos] = True
            in_test = np.zeros(len(dates), dtype=bool)
            in_test[test_pos] = True
            inner_train = train_frame[in_train[date_code]]
            inner_test = train_frame[in_test[date_code]]
            if inner_train.empty or inner_test.empty:
                continue
            if task == "classification" and len(np.unique(inner_train["target"])) < 2:
                continue
            try:
                predictions, probabilities = fit_predict(
                    merged, inner_train, inner_test
                )
            except Exception as exc:  # noqa: BLE001
                # One candidate failing to fit (an invalid combination, a
                # degenerate window) must not abort the whole search.
                logger.debug("[modeling] search candidate %s failed: %s", params, exc)
                fold_scores.append(float("nan"))
                continue
            fold_scores.append(
                _score(
                    task,
                    search_spec.scoring,
                    inner_test["target"].to_numpy(),
                    predictions,
                    probabilities,
                    inner_test["date"].to_numpy(),
                )
            )
        finite = [s for s in fold_scores if np.isfinite(s)]
        results.append(
            {
                "params": params,
                "score": float(np.mean(finite)) if finite else float("nan"),
                "n_folds_scored": len(finite),
            }
        )

    scored = [r for r in results if np.isfinite(r["score"])]
    if not scored:
        return dict(base_params), {
            "searched": False,
            "reason": "no candidate could be scored on any inner fold",
            "candidates": results,
        }
    best = max(scored, key=lambda r: r["score"])
    return {**base_params, **best["params"]}, {
        "searched": True,
        "scoring": search_spec.scoring,
        "n_candidates": len(results),
        "n_inner_folds": len(folds),
        "best_params": best["params"],
        "best_score": best["score"],
        # Sorted best-first and kept whole: a caller can see how flat the
        # surface was, which is the difference between a real choice and a
        # coin flip dressed up as one.
        "candidates": sorted(
            results,
            key=lambda r: (-r["score"] if np.isfinite(r["score"]) else float("inf")),
        ),
    }
