"""
The contract every preprocessing step satisfies.

WHY A CONTRACT AND NOT TWO FUNCTIONS. Preprocessing was `normalization`,
a Literal with two values, and the two values were two code paths --
`fit_preprocessing`/`apply_preprocessing` for one and
`standardize_cross_sectional` for the other -- consulted in three places:
the fold loop, the search closure and the full-panel refit. The refit
consulted neither and fitted the pooled statistics whatever the spec said,
which is how a model validated cross-sectionally was deployed pooled
(CHANGELOG, "The refit fitted the pooled statistics whatever the spec
said"). A branch that has to be repeated in every consumer is a branch
that will be forgotten in one of them.

So a step is an object with two methods and a state:

    fit(X, ctx)               -> state, JSON-serializable, from TRAINING rows
    transform(X, state, ctx)  -> X, the same state applied to any rows

and a pipeline is a list of them. The engine fits the pipeline on each
fold's training rows and applies the fitted state to that fold's test rows;
the refit fits it once on the whole panel and persists the state; scoring
applies the persisted state. One implementation of "fit on train, apply to
test", and the deployed transform IS the validated one because there is no
second place for it to be written.

TWO FLAGS A CONSUMER READS. `stateless` says the step fits nothing --
cross-sectional standardization uses only each date's own cross-section,
which is contemporaneous information a live model also has, so nothing
crosses the fold boundary and there is nothing to persist. `column_wise`
says the transform of one column depends only on that column, which is
what lets the feature ablation drop a column from a fitted matrix rather
than refit the pipeline without it; a PCA whitening is not column-wise and
the ablation must refit.

WHAT `ctx` CARRIES, AND WHAT IT NEVER CARRIES. The rows' dates and
entities, because a cross-sectional step needs to know which rows share a
date. Never the target. A step that could read the label would be a step
that could leak it, and the contract is what makes that impossible to
write by accident rather than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from standard_quant_tools.modeling.estimators.bounds import EstimatorParamSchema


@dataclass(frozen=True)
class FoldContext:
    """The row metadata a step may read: which rows share a date, and which
    entity each belongs to. Deliberately not the target."""

    dates: np.ndarray
    entities: Optional[np.ndarray] = None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "FoldContext":
        """From a long panel carrying `date` and, optionally, `entity`."""
        return cls(
            dates=frame["date"].to_numpy(),
            entities=frame["entity"].to_numpy() if "entity" in frame.columns else None,
        )


class Preprocessor:
    """
    Base step: subclasses set `id`, the two flags, and the two methods.

    `params` are the caller's overrides, already validated against the
    step's schema at the spec boundary, merged onto the definition's
    defaults by the pipeline before construction.
    """

    id: ClassVar[str] = ""
    stateless: ClassVar[bool] = False
    column_wise: ClassVar[bool] = True

    def __init__(self, **params: Any) -> None:
        self.params: Dict[str, Any] = dict(params)

    def fit(self, X: pd.DataFrame, ctx: FoldContext) -> Dict[str, Any]:
        raise NotImplementedError

    def transform(
        self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext
    ) -> pd.DataFrame:
        raise NotImplementedError


class PreprocessorDefinition(BaseModel):
    """One registry entry: the step class, its bounded parameters and its
    two flags, read off the class so they cannot disagree with it."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    description: str
    cls: type
    schema_: EstimatorParamSchema = Field(alias="schema")
    default_params: Dict[str, Any] = Field(default_factory=dict)

    @property
    def stateless(self) -> bool:
        return bool(getattr(self.cls, "stateless", False))

    @property
    def column_wise(self) -> bool:
        return bool(getattr(self.cls, "column_wise", True))


__all__ = ["FoldContext", "Preprocessor", "PreprocessorDefinition"]
