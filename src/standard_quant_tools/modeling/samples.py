"""
The sample index: what a row IS, beside what it contains.

WHAT THE ENGINE WAS DOING IMPLICITLY. The purge reads each row's date and
label end; the weights read its date, its label end and its entity; the
ranking adapter reorders by date and entity and cuts its query groups on
the dates; the metrics read the dates. Every one of those was reading
columns off whichever DataFrame slice happened to be in hand, and the
contract between the engine and an adapter was "a frame with some
columns" -- which is a contract on the shape of the data rather than on
the meaning of a sample.

`SampleIndex` is that meaning, written down once: the dates, the entities
and the label ends of the rows an estimator sees, in the row order the
feature matrix is in. It travels beside `X` inside `FitArrays`, is taken
and reordered with the same masks and permutations, and is what the
weights, the conformal calibration and the adapters are defined on. The
matrix itself is typed by the adapter's declared `input_kind`; the one
kind today is `tabular`, and a sequence kind later builds `(n, T, F)` per
entity from the same index without any other consumer changing.

NO SPEC FIELD. A `RepresentationSpec(kind="tabular")` with one allowed
value would churn every persisted `ModelSpec` for no behaviour, and the
repository's own spike measured a shared representation at +0.0014 R2 on
the most favourable panel it was given. The contract gets the engine
ready; the field waits for a measured case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .dataset.alignment import LABEL_END_COL
from .preprocessing.base import FoldContext


@dataclass(frozen=True)
class SampleIndex:
    """Dates, entities and label ends of a set of rows, in row order."""

    dates: np.ndarray
    entities: np.ndarray
    label_end: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.entities):
            raise ValidationError(
                f"SampleIndex: {len(self.dates)} dates and {len(self.entities)} "
                "entities; one of each per row."
            )
        if self.label_end is not None and len(self.label_end) != len(self.dates):
            raise ValidationError(
                f"SampleIndex: {len(self.label_end)} label ends for "
                f"{len(self.dates)} rows; one per row or none."
            )

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "SampleIndex":
        """From a long panel slice carrying `date`, `entity` and optionally
        `label_end_date`."""
        if "date" not in frame.columns or "entity" not in frame.columns:
            raise ValidationError(
                "SampleIndex.from_frame needs `date` and `entity` columns."
            )
        return cls(
            dates=frame["date"].to_numpy(),
            entities=frame["entity"].to_numpy(),
            label_end=(
                frame[LABEL_END_COL].to_numpy()
                if LABEL_END_COL in frame.columns
                else None
            ),
        )

    def __len__(self) -> int:
        return int(len(self.dates))

    @property
    def n(self) -> int:
        return len(self)

    def take(self, selection: Any) -> "SampleIndex":
        """The same rows a boolean mask or an index array selects from `X`,
        in the order it selects them."""
        return SampleIndex(
            dates=self.dates[selection],
            entities=self.entities[selection],
            label_end=None if self.label_end is None else self.label_end[selection],
        )

    def context(self) -> FoldContext:
        """The preprocessing pipeline's view of the same rows."""
        return FoldContext(dates=self.dates, entities=self.entities)


__all__ = ["SampleIndex"]
