"""
The fold cache: preprocessed matrices, kept for the next fit that needs them.

WHAT WAS RECOMPUTED. The inner hyperparameter search preprocessed each
inner fold once per CANDIDATE, although the inner folds do not depend on
the candidate and neither does the pipeline's output for them: a 12-point
grid fitted the same winsorize-and-zscore twelve times per inner fold. And
the feature ablation ran the whole walk-forward once per feature, refitting
every fold's pipeline on the same rows with one column fewer.

WHAT A COLUMN-WISE PIPELINE MAKES EXACT. Every default step transforms a
column from that column alone -- a winsorize quantile, a z-score, a
cross-sectional rank, a median for imputation are each fitted per column --
so the preprocessed matrix for a SUBSET of the features is the full
matrix's columns, exactly, to the last bit. `Preprocessor.column_wise`
(phase 1) is the flag that says so, and this cache projects only when
every step in the pipeline carries it AND the pipeline's output columns
are its input columns. A PCA whitening is not column-wise; a missingness
indicator adds columns; both miss and refit, which is the honest answer
rather than an approximate one.

IN-PROCESS, KEYED BY THE PLAN. Entries are keyed by the experiment plan's
`preprocessing_hash` -- dataset hash, fold rows, resolved steps -- so two
runs over the same dataset and folds share matrices whatever estimator
they fit, and two runs over different data cannot collide. A persistent,
cross-process cache is deliberately not built: the hashes make one
possible, and orchestration is the layer above this library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .preprocessing.registry import get_preprocessor


def column_wise_pipeline(steps: Sequence[Any]) -> bool:
    """Whether every step transforms each column from that column alone,
    read off the registry so it cannot disagree with the step."""
    return all(get_preprocessor(_step_type(step)).column_wise for step in steps)


def _step_type(step: Any) -> str:
    if isinstance(step, str):
        return step
    if isinstance(step, dict):
        return str(step["type"])
    return str(getattr(step, "type"))


@dataclass
class _Entry:
    columns: List[str]
    train: pd.DataFrame
    test: pd.DataFrame
    #: Whether a subset of `columns` may be read off these frames directly.
    projectable: bool


class FoldCache:
    """
    Preprocessed (train, test) matrices by key, with exact projection.

    `lookup` returns the stored frames on an exact column match, a column
    projection of the key's WIDEST projectable entry when the wanted
    columns are a subset of it, and None otherwise. `store` keeps every
    exact entry -- two feature sets under a pipeline that cannot project
    coexist rather than evict each other -- except a projectable one that
    a wider projectable entry already covers, which would be redundant.

    Frames are returned by reference on a hit. Nothing in the engine
    mutates a preprocessed matrix -- the adapters read it into arrays --
    and a caller that wants to must copy.
    """

    def __init__(self) -> None:
        self._exact: Dict[Tuple[str, Tuple[str, ...]], _Entry] = {}
        self._widest: Dict[str, _Entry] = {}
        self.hits = 0
        self.misses = 0
        self.projections = 0

    def __len__(self) -> int:
        return len(self._exact)

    def stats(self) -> Dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "projections": self.projections,
            "entries": len(self._exact),
        }

    def lookup(
        self, key: str, feature_ids: Sequence[str]
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        wanted = list(feature_ids)
        entry = self._exact.get((key, tuple(wanted)))
        if entry is not None:
            self.hits += 1
            return entry.train, entry.test
        widest = self._widest.get(key)
        if widest is not None and set(wanted) <= set(widest.columns):
            self.projections += 1
            return widest.train[wanted], widest.test[wanted]
        self.misses += 1
        return None

    def store(
        self,
        key: str,
        feature_ids: Sequence[str],
        train: pd.DataFrame,
        test: pd.DataFrame,
        *,
        projectable: bool,
    ) -> None:
        wanted = list(feature_ids)
        # Projectable only when the pipeline is column-wise AND kept the
        # column set: a step that adds or replaces columns has an output
        # a feature subset cannot be read off by name.
        same_columns = list(train.columns) == wanted and list(test.columns) == wanted
        projectable = bool(projectable and same_columns)
        widest = self._widest.get(key)
        if projectable and widest is not None and set(wanted) < set(widest.columns):
            return  # a projection of what is already held
        entry = _Entry(columns=wanted, train=train, test=test, projectable=projectable)
        self._exact[(key, tuple(wanted))] = entry
        if projectable and (widest is None or set(widest.columns) < set(wanted)):
            self._widest[key] = entry


__all__ = ["FoldCache", "column_wise_pipeline"]
