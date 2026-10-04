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

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.estimators.bounds import EstimatorParamSchema


def datetime_values(column: Any) -> np.ndarray:
    """
    A date column's values as a numpy array, without one Python object per
    row.

    `to_numpy()` on a timezone-aware column builds a `pd.Timestamp` for
    every row. One ridge walk-forward run on a 31,680-row panel made 58
    such calls, 0.85 s of its 1.52 s under the profiler (pandas 3.0), and
    the object arrays they returned made every sort and search on the
    dates slow as well. A timezone-aware column is returned here as its
    UTC instants instead -- `datetime64` of the column's own unit, naive,
    in a fresh array -- and any other column exactly as `to_numpy()`
    returns it, which for a naive date column was already `datetime64`.

    Two values of one column are equal, ordered and apart by the same
    amount as instants as they were as zoned timestamps, so everything read
    off these arrays -- the purge, the sample weights, the calibration
    blocks, the ranking groups, a step's per-date groups -- comes out the
    same. What does depend on the zone is a calendar label: the UTC date of
    a row stamped at midnight in Tokyo is the day before. Labels are read
    off the frame, never off these arrays.
    """
    values = getattr(column, "array", column)
    if isinstance(getattr(values, "dtype", None), pd.DatetimeTZDtype):
        return np.array(values.tz_convert(None).to_numpy(), copy=True)
    if hasattr(column, "to_numpy"):
        return column.to_numpy()
    return np.asarray(column)


def _zone_kind(column: Any) -> Optional[str]:
    """'aware' or 'naive' for a datetime64 column, None for anything else."""
    dtype = getattr(column, "dtype", None)
    if isinstance(dtype, pd.DatetimeTZDtype):
        return "aware"
    if getattr(dtype, "kind", None) == "M":
        return "naive"
    return None


def refuse_mixed_time_zones(dates: Any, label_end: Any, where: str) -> None:
    """
    Refuse a date column and a label-end column of which one carries a time
    zone and the other does not.

    The purge compares each row's label end with a test window's dates. A
    naive time and a timezone-aware one name no common instant, and pandas
    raised `TypeError: Cannot compare tz-naive and tz-aware timestamps`
    from inside the purge. Compared as `datetime_values` instants they
    would compare -- the naive one read as UTC, which for a label end
    recorded in New York local time is four or five hours early, and a row
    whose label ends on the first test date would no longer be purged. So
    the pair is refused by name instead. Two aware columns in different
    zones are fine: they compare as instants, as they always did.
    """
    kinds = (_zone_kind(dates), _zone_kind(label_end))
    if None in kinds or kinds[0] == kinds[1]:
        return
    raise ValidationError(
        f"{where}: the panel's 'date' column is timezone-{kinds[0]} "
        f"({dates.dtype}) and its 'label_end_date' column is "
        f"timezone-{kinds[1]} ({label_end.dtype}). The label-overlap purge "
        "compares each row's label end with the test window's dates, and a "
        "naive time and a timezone-aware one do not name a common instant. "
        "Give both columns a time zone, or neither, and register or build "
        "the panel again."
    )


@dataclass(frozen=True)
class FoldContext:
    """The row metadata a step may read: which rows share a date, and which
    entity each belongs to. Deliberately not the target.

    `dates` is `datetime64` for a panel with a date column, a
    timezone-aware one as its UTC instants (see `datetime_values`)."""

    dates: np.ndarray
    entities: Optional[np.ndarray] = None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "FoldContext":
        """From a long panel carrying `date` and, optionally, `entity`."""
        return cls(
            dates=datetime_values(frame["date"]),
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


__all__ = [
    "FoldContext",
    "Preprocessor",
    "PreprocessorDefinition",
    "datetime_values",
    "refuse_mixed_time_zones",
]
