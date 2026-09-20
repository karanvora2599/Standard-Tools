"""
The experiment plan: what `run_experiment` will do, decided before it does
any of it.

WHY A PLAN AND NOT A LOOP. The engine cut its folds, purged them and
counted its fits AS it ran, so the only way to learn what a spec cost was
to pay it. `validate_model_spec` estimated the fit count from the spec
alone -- folds times grid times inner splits -- and the estimate could not
see that a fold's training window was too short for its inner search, that
the purge had emptied a fold, or that a full-panel refit follows the folds.
The cost of a search-heavy spec was discovered by running it.

A plan is a pure function of the spec and the date axis, plus the panel
when one is present: the folds with their date ranges, the rows the purge
removes from each, the inner fold count each training window supports, the
candidate list the search will score, and the fit count all of that
implies. `run_experiment` executes the plan rather than re-deriving it, so
the number the plan reports is the number that runs.

THE BUDGET IS REFUSED, NEVER TRUNCATED. `ModelSpec.budget.max_fits` is a
ceiling the plan is checked against before the first fit. A plan over it
is refused by name with the count and what would bring it under; nothing
is quietly shortened, because a search that ran half its grid is not the
search the spec described.

ONE HASH PER NODE. Every fold carries a content hash of what determines
its fitted estimator -- the dataset, the fold's rows, the resolved
preprocessing pipeline, the estimator and its parameters, the seed -- and a
narrower one of what determines its PREPROCESSED matrices, which the fold
cache keys on. The hashes make a persistent cache possible; this library
keeps its cache in-process, because orchestration is the layer above it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .dataset.alignment import LABEL_END_COL
from .specs import ModelSpec
from .validation.search import inner_fold_count, search_candidates
from .validation.walk_forward import (
    build_splitter,
    contiguous_runs,
    label_overlap_mask,
)


def fits_per_estimator(model_spec: ModelSpec) -> int:
    """
    How many estimator fits one "fit" of this spec costs.

    One, unless the classifier is calibrated: `CalibratedClassifierCV`
    fits the estimator once per calibration fold, and those are real fits
    of the real estimator on most of the window each. The inner search
    does not calibrate its candidates, so this multiplies the outer fit
    and the refit only.
    """
    if getattr(model_spec.estimator, "calibration", "none") == "none":
        return 1
    return int(getattr(model_spec.estimator, "calibration_folds", 3))


def fit_count(
    model_spec: ModelSpec,
    n_folds: int,
    *,
    n_inner_folds: Optional[Sequence[int]] = None,
) -> int:
    """
    Fits a spec implies over `n_folds` outer folds, refit included.

    `n_inner_folds` is per outer fold, the number of inner folds that
    fold's training window supports; omitted, every fold is assumed to
    support `search.inner_splits`, which is what a spec-only estimate can
    say. The one place this arithmetic lives, so the plan, the tool and the
    refusal message cannot disagree about it.
    """
    per_fit = fits_per_estimator(model_spec)
    total = per_fit  # the full-panel refit
    candidates = 0
    if model_spec.search is not None:
        candidates = len(search_candidates(model_spec.search, model_spec.random_seed))
    for i in range(int(n_folds)):
        inner = (
            int(n_inner_folds[i])
            if n_inner_folds is not None
            else (int(model_spec.search.inner_splits) if model_spec.search else 0)
        )
        total += per_fit + candidates * inner
    return int(total)


def _content_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _date_label(value: Any) -> str:
    """A date as 'YYYY-MM-DD'; a positional axis (a RangeIndex, which the
    spec validator plans over when it has only a date COUNT) as the
    position itself rather than as the epoch it would parse to."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return str(pd.Timestamp(value).date())
    except (TypeError, ValueError):
        return str(value)


@dataclass
class FoldPlan:
    """One outer fold: which dates, which rows, what it will cost."""

    index: int
    train_positions: np.ndarray = field(repr=False)
    test_positions: np.ndarray = field(repr=False)
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_dates: int
    n_test_dates: int
    #: Inner folds the training window supports for the search: 0 when it
    #: is too short, in which case the engine's search returns the base
    #: parameters and says so, and nothing is fitted for it.
    n_inner_folds: int
    n_candidates: int
    n_fits: int
    #: What determines the fitted estimator.
    node_hash: str
    #: What determines the preprocessed matrices: the node hash without the
    #: estimator, its parameters and the seed.
    preprocessing_hash: str
    #: Panel-level detail, present when the plan was built with a panel.
    purged_rows: Optional[np.ndarray] = field(default=None, repr=False)
    n_train_rows: Optional[int] = None
    n_test_rows: Optional[int] = None
    n_purged: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_train_dates": self.n_train_dates,
            "n_test_dates": self.n_test_dates,
            "n_inner_folds": self.n_inner_folds,
            "n_candidates": self.n_candidates,
            "n_fits": self.n_fits,
            "node_hash": self.node_hash,
            "preprocessing_hash": self.preprocessing_hash,
            "n_train_rows": self.n_train_rows,
            "n_test_rows": self.n_test_rows,
            "n_purged": self.n_purged,
        }


@dataclass
class ExperimentPlan:
    """The schedule `run_experiment` executes, and what it costs."""

    method: str
    n_dates: int
    folds: List[FoldPlan]
    n_candidates: int
    fits_per_fit: int
    n_fits_folds: int
    n_fits_refit: int
    n_fits: int
    max_fits: int
    dataset_hash: Optional[str]
    has_panel: bool
    n_purged: Optional[int]

    @property
    def within_budget(self) -> bool:
        return self.n_fits <= self.max_fits

    def refuse_over_budget(self, where: str) -> None:
        """Raise, by name, when the plan costs more than the spec allows."""
        if self.within_budget:
            return
        raise ValidationError(
            f"{where}: this spec implies {self.n_fits:,} estimator fits "
            f"({len(self.folds)} fold(s) x ({self.fits_per_fit} + "
            f"{self.n_candidates} candidate(s) x inner folds) + "
            f"{self.n_fits_refit} for the refit), over budget.max_fits="
            f"{self.max_fits:,}. Nothing was fitted. Shrink the search grid "
            "or its inner_splits, use fewer folds, or pass "
            f"budget.max_fits={self.n_fits} to accept the cost on purpose."
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_dates": self.n_dates,
            "n_folds": len(self.folds),
            "n_candidates": self.n_candidates,
            "fits_per_fit": self.fits_per_fit,
            "n_fits_folds": self.n_fits_folds,
            "n_fits_refit": self.n_fits_refit,
            "n_fits": self.n_fits,
            "max_fits": self.max_fits,
            "within_budget": self.within_budget,
            "dataset_hash": self.dataset_hash,
            "n_purged": self.n_purged,
            "folds": [f.to_dict() for f in self.folds],
        }


def plan_experiment(
    model_spec: ModelSpec,
    dates: pd.Index,
    *,
    panel: Optional[pd.DataFrame] = None,
    dataset_hash: Optional[str] = None,
    feature_ids: Optional[Sequence[str]] = None,
) -> ExperimentPlan:
    """
    Plan an experiment over `dates` -- the sorted unique date axis -- and,
    when `panel` is given, over its rows.

    Without a panel the plan is what the spec and the axis alone determine:
    the folds, their date ranges, the inner fold count each supports and
    the fit count. With one it also carries the rows the label-overlap
    purge removes from each fold, which the engine applies as given, so the
    purge count the plan reports is the purge count that ran.

    Raises ValidationError when the axis yields no folds.
    """
    splitter = build_splitter(model_spec.validation)
    n_dates = int(len(dates))
    if splitter.n_splits(dates) < 1:
        raise ValidationError(
            f"plan_experiment: {n_dates} dates yield no fold under "
            f"validation.method={model_spec.validation.method!r}."
        )

    per_fit = fits_per_estimator(model_spec)
    search = model_spec.search
    candidates = search_candidates(search, model_spec.random_seed) if search else []
    n_candidates = len(candidates)
    embargo = int(model_spec.validation.embargo)
    steps = [s.model_dump() for s in model_spec.preprocessing.resolved_steps]
    estimator = {
        "type": model_spec.estimator.type,
        "params": dict(model_spec.estimator.params),
        "calibration": getattr(model_spec.estimator, "calibration", "none"),
    }
    features = list(feature_ids) if feature_ids is not None else None

    date_values = dates.to_numpy()
    panel_dates = panel_label_end = date_code = None
    if panel is not None:
        panel_dates = panel["date"].to_numpy()
        date_code = np.searchsorted(date_values, panel_dates)
        if LABEL_END_COL in panel.columns:
            panel_label_end = panel[LABEL_END_COL].to_numpy()

    folds: List[FoldPlan] = []
    total_purged = 0
    for index, (train_pos, test_pos) in enumerate(splitter.split(dates)):
        train_pos = np.asarray(train_pos)
        test_pos = np.asarray(test_pos)
        n_train_dates = int(len(train_pos))
        purged_rows = n_train_rows = n_test_rows = n_purged = None
        if panel is not None:
            in_train = np.zeros(n_dates, dtype=bool)
            in_train[train_pos] = True
            in_test = np.zeros(n_dates, dtype=bool)
            in_test[test_pos] = True
            train_mask = in_train[date_code]
            test_mask = in_test[date_code]
            overlaps = np.zeros(train_mask.shape, dtype=bool)
            if panel_label_end is not None and train_mask.any() and len(test_pos):
                # Per contiguous test block, ORed: a training row between
                # two blocks is purged only when its own label reaches the
                # later one. See engine.run_experiment for the history.
                for first, last in contiguous_runs(test_pos):
                    overlaps |= label_overlap_mask(
                        train_mask,
                        panel_dates,
                        panel_label_end,
                        date_values[first],
                        date_values[last],
                    )
            purged_rows = np.flatnonzero(overlaps)
            n_purged = int(purged_rows.size)
            total_purged += n_purged
            surviving = train_mask & ~overlaps
            n_train_rows = int(surviving.sum())
            n_test_rows = int(test_mask.sum())
            # The inner search runs on the rows that SURVIVED the purge,
            # whose date axis can be shorter than the scheduled window.
            n_train_dates = int(np.unique(panel_dates[surviving]).size)
        n_inner = (
            inner_fold_count(n_train_dates, int(search.inner_splits), embargo)
            if search is not None
            else 0
        )
        n_fits = per_fit + n_candidates * n_inner
        span = {
            "train_start": _date_label(date_values[train_pos[0]]),
            "train_end": _date_label(date_values[train_pos[-1]]),
            "test_start": _date_label(date_values[test_pos[0]]),
            "test_end": _date_label(date_values[test_pos[-1]]),
        }
        preprocessing_payload = {
            "dataset_hash": dataset_hash,
            "fold": {**span, "n_purged": n_purged},
            "steps": steps,
        }
        node_payload = {
            **preprocessing_payload,
            "features": features,
            "estimator": estimator,
            "random_seed": int(model_spec.random_seed),
        }
        folds.append(
            FoldPlan(
                index=index,
                train_positions=train_pos,
                test_positions=test_pos,
                n_train_dates=n_train_dates,
                n_test_dates=int(len(test_pos)),
                n_inner_folds=int(n_inner),
                n_candidates=n_candidates,
                n_fits=int(n_fits),
                node_hash=_content_hash(node_payload),
                preprocessing_hash=_content_hash(preprocessing_payload),
                purged_rows=purged_rows,
                n_train_rows=n_train_rows,
                n_test_rows=n_test_rows,
                n_purged=n_purged,
                **span,
            )
        )

    n_fits_folds = int(sum(f.n_fits for f in folds))
    return ExperimentPlan(
        method=model_spec.validation.method,
        n_dates=n_dates,
        folds=folds,
        n_candidates=n_candidates,
        fits_per_fit=per_fit,
        n_fits_folds=n_fits_folds,
        n_fits_refit=per_fit,
        n_fits=n_fits_folds + per_fit,
        max_fits=int(model_spec.budget.max_fits),
        dataset_hash=dataset_hash,
        has_panel=panel is not None,
        n_purged=total_purged if panel is not None else None,
    )


__all__ = [
    "ExperimentPlan",
    "FoldPlan",
    "fit_count",
    "fits_per_estimator",
    "plan_experiment",
]
