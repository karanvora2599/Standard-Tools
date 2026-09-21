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

THREE BACKENDS, ONE DISCIPLINE. `grid` scores every combination, `random`
a seeded sample of them, and `tpe` asks optuna's Tree-structured Parzen
Estimator for each next candidate from what the previous ones scored --
which is the one that makes a CONTINUOUS axis (`param_ranges`) worth
declaring, since a grid over a log-spaced regularization strength is
either coarse or enormous. All three cut the same purged, embargoed inner
folds and score them with the same closure; the sampler decides which
parameters to try and nothing else.
"""

from __future__ import annotations

import itertools
import logging
from math import prod
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .metrics import cross_sectional_ic
from .walk_forward import WalkForwardSplit, label_overlap_mask

logger = logging.getLogger(__name__)


def optuna_available() -> bool:
    """Whether optuna can be imported, without importing it."""
    from importlib.util import find_spec

    try:
        return find_spec("optuna") is not None
    except (ImportError, ValueError):
        return False


def require_optuna(where: str) -> None:
    """Refuse, by name, a tpe search on a machine without optuna."""
    if optuna_available():
        return
    raise ValidationError(
        f"{where}: search.method='tpe' needs optuna, which is not installed "
        "in this environment (pip install optuna). 'grid' and 'random' need "
        "nothing; list_modeling_capabilities reports which backends are "
        "available."
    )


def search_candidates(search_spec: Any, random_seed: int) -> List[Dict[str, Any]]:
    """
    The parameter combinations to try, in a deterministic order.

    Public because the experiment plan counts them before anything is
    fitted, and the count the plan reports has to be the list the search
    walks -- same enumeration, same sampling, same seed. A tpe search has
    no list: its candidates are chosen one at a time from the scores of
    the previous ones, so it is counted by `n_search_candidates` and not
    enumerated here.
    """
    if search_spec.method == "tpe":
        raise ValidationError(
            "a tpe search samples its candidates rather than enumerating "
            "them; ask n_search_candidates for how many it will run."
        )
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


def n_search_candidates(search_spec: Any) -> int:
    """
    How many candidates the search will score per outer fold, without
    enumerating them: the grid's size, the random sample's size, or the
    tpe trial budget. What the plan multiplies through the inner folds.
    """
    if search_spec.method == "tpe":
        return int(search_spec.max_trials)
    size = prod(max(1, len(values)) for values in search_spec.param_grid.values())
    if search_spec.method == "random":
        return int(min(size, search_spec.n_iter))
    return int(size)


def rank_turnover(
    predictions: np.ndarray, dates: np.ndarray, entities: np.ndarray
) -> float:
    """
    How much a signal's ordering moves from one date to the next: the
    mean absolute change in each entity's percentile rank between
    consecutive dates, averaged over dates, in [0, 1]. An entity absent
    on either date of a pair contributes nothing to that pair. Zero for
    a signal whose ordering never changes; a signal reshuffled at random
    every date sits near one third.
    """
    frame = pd.DataFrame(
        {
            "date": np.asarray(dates),
            "entity": np.asarray(entities),
            "p": np.asarray(predictions, dtype=float),
        }
    )
    frame["rank"] = frame.groupby("date")["p"].rank(pct=True)
    wide = frame.pivot_table(index="date", columns="entity", values="rank")
    if len(wide) < 2:
        return 0.0
    per_date = wide.sort_index().diff().abs().mean(axis=1).iloc[1:]
    return float(per_date.mean()) if per_date.notna().any() else 0.0


def _score(
    task: str,
    scoring: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray],
    dates: np.ndarray,
    *,
    entities: Optional[np.ndarray] = None,
    turnover_penalty: float = 0.0,
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

    if scoring == "concordance":
        from .survival import concordance_index

        labels = np.asarray(y_true, dtype=float)
        value, _pairs = concordance_index(labels[:, 0], labels[:, 1], predictions)
        return value
    if scoring == "cs_rank_ic_net_of_turnover":
        if entities is None:
            raise ValidationError(
                "cs_rank_ic_net_of_turnover needs the rows' entities to measure "
                "turnover on."
            )
        series = cross_sectional_ic(y_true, predictions, dates, "spearman")
        ic = float(series.mean()) if len(series) else float("nan")
        return ic - float(turnover_penalty) * rank_turnover(
            predictions, dates, entities
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


def _inner_splitter(
    n_dates: int, inner_splits: int, embargo: int = 0
) -> Optional[WalkForwardSplit]:
    """
    Size an inner walk-forward so it yields exactly `inner_splits` folds.

    Returns None when the training window is too short to be split that
    many times — the caller then skips the search for that fold rather
    than silently searching on one or two dates, which would select on
    noise and be worse than not searching at all.

    `embargo` is the outer spec's, applied between every inner train and
    test window. It was hardwired to zero, so a spec that asked for a
    five-bar gap on its outer folds got none on the folds that CHOSE its
    parameters. The embargo is taken off the axis before it is divided, so
    the fold count is still exactly `inner_splits`: the last fold ends on
    the last date and every earlier one steps back by one test window.
    """
    usable = n_dates - int(embargo)
    test_window = usable // (inner_splits + 1)
    if test_window < 1:
        return None
    train_window = usable - inner_splits * test_window
    if train_window < 1:
        return None
    return WalkForwardSplit(
        train_window=train_window, test_window=test_window, embargo=int(embargo)
    )


def _labels_for(task: str, frame: pd.DataFrame) -> np.ndarray:
    """The label a candidate is scored against: `target`, or for survival
    the (duration, event) pair the concordance needs."""
    if task == "survival":
        from .survival import survival_labels

        return survival_labels(frame)
    return frame["target"].to_numpy()


def inner_fold_count(n_dates: int, inner_splits: int, embargo: int = 0) -> int:
    """
    Inner folds a training window of `n_dates` supports: `inner_splits`,
    or zero when the window is too short and the search will not run.

    The plan's question, answered by the splitter's own sizing rule rather
    than a restatement of it.
    """
    return int(inner_splits) if _inner_splitter(n_dates, inner_splits, embargo) else 0


FitPredict = Callable[
    [Dict[str, Any], pd.DataFrame, pd.DataFrame, int],
    Tuple[np.ndarray, Optional[np.ndarray]],
]


def search_best_params(
    *,
    task: str,
    search_spec: Any,
    base_params: Dict[str, Any],
    train_frame: pd.DataFrame,
    feature_ids: List[str],
    random_seed: int,
    fit_predict: FitPredict,
    embargo: int = 0,
    label_end: Optional[np.ndarray] = None,
    max_parallelism: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Choose estimator parameters using only `train_frame`.

    `fit_predict(params, inner_train, inner_test, fold_index)` is supplied
    by the engine so the search reuses the engine's own preprocessing and
    weighting rather than reimplementing them — a search that normalized
    its data differently from the final fit would select for the wrong
    thing. `fold_index` names the inner fold, which is the same frames for
    every candidate, so the engine can preprocess it once.

    `embargo` and `label_end` are the outer loop's leakage discipline,
    applied to the inner folds. `label_end` is one entry per row of
    `train_frame`, the date its label finishes observing; a training row
    whose label reaches into an inner test window is purged from that
    fold, by the same `label_overlap_mask` the engine applies. Without it
    the inner folds were cut on dates alone, and the candidate that won was
    the one that scored best on rows whose labels had already seen the
    window they were scored against.

    Returns (best_params, report). The report carries every candidate's
    score, because "which alpha won" is much less informative than "the
    top four alphas were within 0.001 of each other", and only the second
    tells a reader the search did not actually find anything. It also
    records the embargo and how many rows each inner fold purged, so a
    reader can see the selection ran under the discipline it claims.
    """
    if label_end is not None and len(label_end) != len(train_frame):
        raise ValidationError(
            f"search_best_params: label_end has {len(label_end)} entries for "
            f"{len(train_frame)} training rows; it must be one per row."
        )
    dates = pd.Index(sorted(train_frame["date"].unique()))
    splitter = _inner_splitter(len(dates), search_spec.inner_splits, embargo)
    if splitter is None:
        return dict(base_params), {
            "searched": False,
            "method": search_spec.method,
            "reason": (
                f"training window has {len(dates)} dates, too few for "
                f"{search_spec.inner_splits} inner folds with embargo={embargo}"
            ),
        }

    row_dates = train_frame["date"].to_numpy()
    date_code = np.searchsorted(dates.to_numpy(), row_dates)

    # The inner folds' row masks, cut ONCE: they do not depend on the
    # candidate, and the purge is the same for every one of them.
    fold_masks: List[Tuple[np.ndarray, np.ndarray]] = []
    purged_per_fold: List[int] = []
    for train_pos, test_pos in splitter.split(dates):
        in_train = np.zeros(len(dates), dtype=bool)
        in_train[train_pos] = True
        in_test = np.zeros(len(dates), dtype=bool)
        in_test[test_pos] = True
        train_mask = in_train[date_code]
        test_mask = in_test[date_code]
        test_axis = dates[test_pos]
        overlaps = label_overlap_mask(
            train_mask, row_dates, label_end, test_axis[0], test_axis[-1]
        )
        purged_per_fold.append(int(overlaps.sum()))
        fold_masks.append((train_mask & ~overlaps, test_mask))

    # The inner frames, sliced ONCE: they do not depend on the candidate
    # either, and each was re-sliced from the training frame once per
    # candidate per fold.
    inner_frames = [
        (train_frame[train_mask], train_frame[test_mask])
        for train_mask, test_mask in fold_masks
    ]

    def score_fold(params: Dict[str, Any], fold_index: int) -> Optional[float]:
        """One candidate on one inner fold: its score, NaN when it could
        not be scored, None when the fold itself cannot score anything."""
        inner_train, inner_test = inner_frames[fold_index]
        if inner_train.empty or inner_test.empty:
            return None
        if task == "classification" and len(np.unique(inner_train["target"])) < 2:
            return None
        if task == "survival" and not (inner_train["event"].to_numpy() == 1).any():
            return None
        try:
            predictions, probabilities = fit_predict(
                params, inner_train, inner_test, fold_index
            )
        except Exception as exc:  # noqa: BLE001
            # One candidate failing to fit (an invalid combination, a
            # degenerate window) must not abort the whole search.
            logger.debug("[modeling] search candidate %s failed: %s", params, exc)
            return float("nan")
        return _score(
            task,
            search_spec.scoring,
            _labels_for(task, inner_test),
            predictions,
            probabilities,
            inner_test["date"].to_numpy(),
            entities=inner_test["entity"].to_numpy(),
            turnover_penalty=float(getattr(search_spec, "turnover_penalty", 0.0)),
        )

    if search_spec.method == "tpe":
        results = _tpe_trials(
            search_spec, base_params, random_seed, score_fold, len(inner_frames)
        )
    else:
        candidates = list(search_candidates(search_spec, random_seed))
        merged_candidates = [{**base_params, **params} for params in candidates]
        n_folds = len(inner_frames)
        if int(max_parallelism) > 1 and len(merged_candidates) > 1:
            # The first candidate runs alone, so each inner fold's
            # preprocessing is fitted once and cached before anything
            # reads it; every remaining (candidate, fold) pair is then
            # scored side by side. Scores are gathered in spec order
            # whatever order the threads finish in, so the result is the
            # sequential one.
            from concurrent.futures import ThreadPoolExecutor

            first = [score_fold(merged_candidates[0], i) for i in range(n_folds)]
            jobs = [
                (c, i) for c in range(1, len(merged_candidates)) for i in range(n_folds)
            ]
            with ThreadPoolExecutor(max_workers=int(max_parallelism)) as pool:
                rest = list(
                    pool.map(
                        lambda job: score_fold(merged_candidates[job[0]], job[1]), jobs
                    )
                )
            per_candidate = [first] + [
                rest[k * n_folds : (k + 1) * n_folds]
                for k in range(len(merged_candidates) - 1)
            ]
        else:
            per_candidate = [
                [score_fold(merged, i) for i in range(n_folds)]
                for merged in merged_candidates
            ]
        results = []
        for params, fold_scores in zip(candidates, per_candidate):
            finite = [s for s in fold_scores if s is not None and np.isfinite(s)]
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
            "method": search_spec.method,
            "reason": "no candidate could be scored on any inner fold",
            "candidates": results,
        }
    best = max(scored, key=lambda r: r["score"])
    report = {
        "searched": True,
        "method": search_spec.method,
        "scoring": search_spec.scoring,
        "n_candidates": len(results),
        "n_inner_folds": len(fold_masks),
        "max_parallelism": int(max_parallelism),
        "turnover_penalty": float(getattr(search_spec, "turnover_penalty", 0.0)),
        "embargo": int(embargo),
        # Per inner fold. Zero everywhere means the training window
        # carried no label ends, not that nothing overlapped.
        "n_train_rows_purged_overlap": purged_per_fold,
        "purged_on_label_end": label_end is not None,
        "purge": "label_end" if label_end is not None else "not_applicable",
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
    if search_spec.method == "tpe":
        report["n_trials_pruned"] = sum(1 for r in results if r.get("pruned"))
    return {**base_params, **best["params"]}, report


def _tpe_trials(
    search_spec: Any,
    base_params: Dict[str, Any],
    random_seed: int,
    score_fold: Callable[[Dict[str, Any], int], Optional[float]],
    n_folds: int,
) -> List[Dict[str, Any]]:
    """
    `max_trials` candidates chosen by optuna's TPE sampler, each scored on
    the same inner folds a grid candidate is.

    The sampler is seeded from the spec's `random_seed`, so a fold's search
    is reproducible; the study is in memory and single-threaded, because
    an inner search that ran in parallel would fit the same fold's
    pipeline in several processes at once for no gain. With
    `early_pruning`, a trial reports its running mean after each inner
    fold and is stopped when that falls below the median of the completed
    trials at the same point; its score is what it had when stopped, and
    the report says it was pruned rather than letting a half-scored trial
    pass as a whole one.
    """
    require_optuna("search_best_params")
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def suggest(trial: Any) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for name in sorted(search_spec.param_grid):
            params[name] = trial.suggest_categorical(
                name, list(search_spec.param_grid[name])
            )
        for name in sorted(search_spec.param_ranges):
            axis = search_spec.param_ranges[name]
            if axis.integer:
                params[name] = trial.suggest_int(
                    name, int(axis.low), int(axis.high), log=bool(axis.log)
                )
            else:
                params[name] = trial.suggest_float(
                    name, float(axis.low), float(axis.high), log=bool(axis.log)
                )
        return params

    results: List[Dict[str, Any]] = []

    def objective(trial: Any) -> float:
        params = suggest(trial)
        merged = {**base_params, **params}
        finite: List[float] = []
        pruned = False
        for fold_index in range(n_folds):
            score = score_fold(merged, fold_index)
            if score is not None and np.isfinite(score):
                finite.append(float(score))
            if search_spec.early_pruning and finite:
                trial.report(float(np.mean(finite)), step=fold_index)
                if trial.should_prune():
                    pruned = True
                    break
        value = float(np.mean(finite)) if finite else float("nan")
        results.append(
            {
                "params": params,
                "score": value,
                "n_folds_scored": len(finite),
                "pruned": pruned,
            }
        )
        if pruned:
            raise optuna.TrialPruned()
        if not np.isfinite(value):
            # optuna treats a NaN objective as a failed trial and moves on,
            # which is the right reading of "could not be scored".
            return float("nan")
        return value

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
        if search_spec.early_pruning
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(random_seed)),
        pruner=pruner,
    )
    study.optimize(objective, n_trials=int(search_spec.max_trials), n_jobs=1)
    return results


__all__ = [
    "inner_fold_count",
    "n_search_candidates",
    "optuna_available",
    "rank_turnover",
    "require_optuna",
    "search_best_params",
    "search_candidates",
]
