"""
Which feature actually cost the training rows.

WHY "YOU LOST 44% OF THE DATA" IS NOT AN ANSWER. `build_dataset` already
reports `drop_attribution`, and `attribute_drops` already computes the one
number that makes it actionable -- but only as a field inside a build. A
caller staring at a shrunken panel could not ask the question directly,
and the question is the whole point:

    n_missing       rows where this column is NaN. Several columns are
                    usually NaN in the SAME row -- every feature's warm-up
                    overlaps -- so these sum to far more than the rows
                    actually lost and cannot be read as a cost per feature.
    n_sole_missing  rows where this column is the ONLY thing missing. This
                    is exactly what dropping that one feature would give
                    back.

A 252-day momentum feature sitting behind a 500-day PCA loading has
`n_missing` in the hundreds of thousands and `n_sole_missing` of zero.
Removing it gains NOTHING, and only the second number says so. Reading the
first and dropping the feature is a decision that feels informed and
changes nothing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict, Field

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.alignment import attribute_drops

logger = logging.getLogger(__name__)


class ExplainRowLossInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="A dataset built by build_model_dataset.")


class ColumnLoss(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    n_missing: int = 0
    n_sole_missing: int = Field(
        0,
        description=(
            "Rows this column ALONE cost -- what removing it would give "
            "back. The actionable number."
        ),
    )


class ExplainRowLossResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    rows_before: int = 0
    rows_after: int = 0
    rows_lost: int = 0
    columns: List[ColumnLoss] = Field(default_factory=list)
    free_to_drop: List[str] = Field(
        default_factory=list,
        description=(
            "Columns whose removal would give back NO rows, because "
            "everything they are missing is already missing for another "
            "reason. Dropping these to recover data does nothing."
        ),
    )
    warnings: List[str] = Field(default_factory=list)


def explain_dataset_row_loss(
    input_data: ExplainRowLossInput,
) -> ExplainRowLossResult:
    """Which column cost which rows, and which are free to drop."""
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

    # `_load_dataset_panel` and not a bare artifact read: it verifies the
    # panel against the hash recorded when the dataset was built. An edited
    # panel.parquet would otherwise be described here as though it were the
    # dataset, which is a quieter failure than a wrong model and the same
    # kind.
    panel, meta, _directory = _load_dataset_panel(input_data.dataset_id)

    feature_cols = [c for c in meta.get("feature_ids", []) if c in panel.columns]
    if not feature_cols:
        raise ValidationError(
            f"dataset {input_data.dataset_id!r} records no feature columns "
            "that are present in its panel, so there is no attribution to "
            "compute."
        )
    target_col = "target" if "target" in panel.columns else None

    report: Dict[str, Any] = attribute_drops(panel, feature_cols, target_col)
    per_column: Dict[str, Any] = report.get("per_feature", {})

    rows: List[ColumnLoss] = [
        ColumnLoss(
            column=str(column),
            n_missing=int(counts.get("n_missing", 0)),
            n_sole_missing=int(counts.get("n_sole_missing", 0)),
        )
        for column, counts in per_column.items()
    ]
    rows.sort(key=lambda r: (-r.n_sole_missing, -r.n_missing))

    before = int(report.get("rows_before_alignment", len(panel)))
    after = int(report.get("rows_after_alignment", len(panel)))

    free = [r.column for r in rows if r.n_missing > 0 and r.n_sole_missing == 0]
    warnings: List[str] = []
    if free:
        warnings.append(
            f"{len(free)} column(s) are missing rows that are ALREADY "
            f"missing for another reason: {free}. Dropping any of them to "
            "recover data would give back nothing -- read n_sole_missing, "
            "not n_missing."
        )
    if before and after / before < 0.6:
        warnings.append(
            f"the panel lost {before - after:,} of {before:,} rows "
            f"({1 - after / before:.0%}). Check whether one long lookback is "
            "responsible before assuming the universe is the problem."
        )

    return ExplainRowLossResult(
        dataset_id=input_data.dataset_id,
        rows_before=before,
        rows_after=after,
        rows_lost=int(report.get("rows_dropped", before - after)),
        columns=rows,
        free_to_drop=free,
        warnings=warnings,
    )


__all__ = [
    "ColumnLoss",
    "ExplainRowLossInput",
    "ExplainRowLossResult",
    "explain_dataset_row_loss",
]
