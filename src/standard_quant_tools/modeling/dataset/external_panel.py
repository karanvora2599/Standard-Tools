"""
A model-ready panel this library did not build.

WHY THIS EXISTS. `build_model_dataset` fetches OHLCV, computes registry
features and writes a panel. That is the right path when the features are
this library's own. It is the wrong path when the features were computed
somewhere else -- in a C++ pipeline over an L2 feed, say -- because there is
nothing to fetch and nothing to compute, and the only thing standing between
a finished feature matrix and `run_model_experiment` is the dataset record.

So this builds the record and nothing else. The panel stays where it was
written.

WHY NOT COPY IT. `save_artifact` would write a second complete Parquet under
SQT_RUNS_DIR, and the matrices this path exists for are partitioned
directories of them -- copying one is a materialization, which is the thing
the external-dataset contract was built to avoid. The engine loads the panel
whole either way, so the copy buys nothing except the integrity check, and
that survives without it: `hash_dataframe` runs on the frame AFTER it is
loaded, so an externally-referenced panel is verified exactly as strictly as
a built one.

WHAT THE CALLER MUST STILL DECLARE. The horizon. A panel arrives with a
`target` column and no statement of what that column MEANS, and the engine
needs the horizon for the target-overlap purge -- the rule that stops a
label spanning bars t..t+h from being trained on beside a fold boundary
inside that span. Inferring it from the data is not possible and defaulting
it would silently disable the purge, so it is required and it is the one
thing about an external panel this module refuses to guess.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from standard_quant_tools.data import external as _external
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.features.base import RESERVED_PANEL_COLUMNS

#: The canonical long-panel columns. `label_end_date` is optional: without
#: it the engine purges on the horizon alone, which is correct for a fixed
#: horizon and not for a triple-barrier label that can end early.
PANEL_DATE = "date"
PANEL_ENTITY = "entity"
PANEL_TARGET = "target"
PANEL_LABEL_END = "label_end_date"

#: Prefixes for the extra horizons. The PRIMARY target is also written to
#: plain `target`, so every existing reader -- `explain_dataset_row_loss`,
#: the feature report, anything that has ever looked for a target column --
#: keeps working unchanged on a multi-horizon panel.
#:
#: WHY NOT ONE COLUMN PER MODEL. A microstructure panel is labelled at 100ms
#: through 5min simultaneously and the features are identical across all of
#: them; building one dataset per horizon would recompute and re-store the
#: same feature matrix N times. One panel, N label columns, and the
#: experiment picks -- which is also what makes the horizons COMPARABLE,
#: since every model then sees the same rows and the same folds.
TARGET_PREFIX = "target__"
LABEL_END_PREFIX = "label_end_date__"


def target_column_for(name: str) -> str:
    return f"{TARGET_PREFIX}{name}"


def label_end_column_for(name: str) -> str:
    return f"{LABEL_END_PREFIX}{name}"


#: How many entities a spec may carry. Mirrors `DatasetSpec.universe`'s own
#: ceiling rather than inventing a second one -- a panel with more distinct
#: entities than a spec can hold cannot be described by the record this
#: module writes, and failing here says so before anything is persisted.
MAX_ENTITIES = 1000


def _resolve_columns(
    present: Sequence[str],
    *,
    date_column: str,
    entity_column: str,
    targets: Sequence[Dict[str, Any]],
    feature_columns: Optional[Sequence[str]],
) -> Tuple[Dict[str, str], List[str]]:
    """The rename map onto canonical names, and the feature columns."""
    columns = [str(c) for c in present]
    missing = [
        (name, role)
        for name, role in (
            (date_column, "date_column"),
            (entity_column, "entity_column"),
        )
        + tuple((str(t["column"]), f"target {t['name']!r}") for t in targets)
        if name not in columns
    ]
    if missing:
        raise ValidationError(
            "the panel is missing "
            + ", ".join(f"{role}={name!r}" for name, role in missing)
            + f". It has {len(columns)} column(s): {columns[:15]}"
            + (" ..." if len(columns) > 15 else "")
            + ". A model panel needs one column identifying the bar, one "
            "identifying the entity, and one holding the label."
        )
    for target in targets:
        ends = target.get("label_end_column")
        if ends is not None and str(ends) not in columns:
            raise ValidationError(
                f"target {target['name']!r} names label_end_column={ends!r}, "
                "which is not in the panel. Omit it when that label has a "
                "fixed horizon; supply it only for one that can end early, "
                "such as a triple barrier."
            )

    rename = {date_column: PANEL_DATE, entity_column: PANEL_ENTITY}
    # The PRIMARY target is the first one, and it also lands on plain
    # `target`/`label_end_date` so that a multi-horizon panel is still an
    # ordinary panel to everything that has only ever seen one.
    for index, target in enumerate(targets):
        rename[str(target["column"])] = target_column_for(target["name"])
        ends = target.get("label_end_column")
        if ends is not None:
            rename[str(ends)] = label_end_column_for(target["name"])

    if feature_columns is None:
        features = [c for c in columns if c not in rename]
    else:
        features = [str(c) for c in feature_columns]
        unknown = [c for c in features if c not in columns]
        if unknown:
            raise ValidationError(
                f"feature_columns names {unknown}, which the panel does not "
                f"have. Available: {[c for c in columns if c not in rename]}"
            )
        overlap = [c for c in features if c in rename]
        if overlap:
            raise ValidationError(
                f"feature_columns names {overlap}, which is/are already the "
                "date, entity, target or label-end column. A column cannot "
                "be both a feature and the thing being predicted."
            )

    if not features:
        raise ValidationError(
            "the panel has no feature columns -- every column is the date, "
            "the entity, the target or the label end. There is nothing to "
            "learn from."
        )

    reserved = sorted(set(features) & set(RESERVED_PANEL_COLUMNS))
    if reserved:
        raise ValidationError(
            f"feature column(s) {reserved} collide with the panel schema's "
            f"own reserved names {sorted(RESERVED_PANEL_COLUMNS)}. Rename "
            "them in the extract; left alone they would overwrite the "
            "column the engine reads rather than adding to it."
        )
    return rename, features


def load_external_panel(
    path: str,
    *,
    date_column: str = PANEL_DATE,
    entity_column: str = PANEL_ENTITY,
    targets: Optional[Sequence[Dict[str, Any]]] = None,
    target_column: str = PANEL_TARGET,
    label_end_column: Optional[str] = None,
    feature_columns: Optional[Sequence[str]] = None,
    fmt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Read a panel written elsewhere and put it in this library's shape.

    Returns the canonical frame plus everything the dataset record needs:
    the column mapping (so a later load reproduces it exactly), the feature
    ids, the entities and the date span.
    """
    if targets is None:
        targets = [
            {
                "name": "primary",
                "column": target_column,
                "label_end_column": label_end_column,
            }
        ]
    targets = [dict(t) for t in targets]
    names = [str(t["name"]) for t in targets]
    if len(set(names)) != len(names):
        raise ValidationError(
            f"target names must be unique; got {names}. They become panel "
            "column names, and two of one name would overwrite rather than "
            "add."
        )

    handle = _external.inspect(path, kind="model_panel", fmt=fmt)
    rename, features = _resolve_columns(
        handle.columns,
        date_column=date_column,
        entity_column=entity_column,
        targets=targets,
        feature_columns=feature_columns,
    )

    wanted = list(rename) + list(features)
    frames = [chunk for chunk in handle.batches(columns=wanted) if len(chunk)]
    if not frames:
        raise ValidationError(
            f"{handle.path} has a usable schema and no rows. An empty panel "
            "cannot be split into folds, so nothing downstream can run."
        )
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns=rename)

    panel[PANEL_DATE] = pd.to_datetime(panel[PANEL_DATE], errors="coerce")
    if panel[PANEL_DATE].isna().any():
        bad = int(panel[PANEL_DATE].isna().sum())
        raise ValidationError(
            f"{bad} of {len(panel):,} rows have an unparseable {date_column!r}. "
            "A row that cannot be placed in time cannot be assigned to a "
            "fold, and the walk-forward split is built on that axis."
        )
    for target in targets:
        ends = label_end_column_for(target["name"])
        if ends in panel.columns:
            panel[ends] = pd.to_datetime(panel[ends], errors="coerce")

    # The primary is duplicated onto the plain names, so a multi-horizon
    # panel still reads as an ordinary one to every consumer that has
    # only ever seen a single target.
    primary = targets[0]["name"]
    panel[PANEL_TARGET] = panel[target_column_for(primary)]
    primary_ends = label_end_column_for(primary)
    if primary_ends in panel.columns:
        panel[PANEL_LABEL_END] = panel[primary_ends]

    panel[PANEL_ENTITY] = panel[PANEL_ENTITY].astype(str)
    entities = sorted(panel[PANEL_ENTITY].unique())
    if len(entities) > MAX_ENTITIES:
        raise ValidationError(
            f"the panel holds {len(entities):,} distinct entities and a "
            f"dataset spec carries at most {MAX_ENTITIES}. Split the panel, "
            "or narrow the universe before registering it."
        )

    # Sorted the way the splitter reads it. `engine.py` derives its fold
    # boundaries from the sorted unique dates and `alignment.stack_long`
    # sorts by (date, entity); a panel arriving in another order would
    # produce the same folds but a different row order inside them, which
    # is a needless difference between a built dataset and this one.
    panel = panel.sort_values([PANEL_DATE, PANEL_ENTITY]).reset_index(drop=True)

    warnings: List[str] = []
    nulls = {c: int(panel[c].isna().sum()) for c in features}
    leaking = {c: n for c, n in nulls.items() if n}
    if leaking:
        worst = sorted(leaking.items(), key=lambda kv: -kv[1])[:5]
        warnings.append(
            "NOTE: "
            + ", ".join(f"{c} has {n:,} null(s)" for c, n in worst)
            + f" of {len(panel):,} rows. Alignment drops any row where a "
            "requested feature is missing, so this costs rows for every "
            "other feature too."
        )
    for target in targets:
        column = target_column_for(target["name"])
        nulls_here = int(panel[column].isna().sum())
        if nulls_here:
            warnings.append(
                f"NOTE: target {target['name']!r} is null on {nulls_here:,} "
                f"of {len(panel):,} rows, which is normal at the end of a "
                "sample where its forward window has not closed. Rows are "
                "dropped per EXPERIMENT, by the target it selects, so a long "
                "horizon costs its own rows and not the shorter ones'."
            )

    return {
        "panel": panel,
        "handle": handle,
        "rename": rename,
        "targets": targets,
        "feature_ids": features,
        "entities": entities,
        "start": str(panel[PANEL_DATE].iloc[0].date()),
        "end": str(panel[PANEL_DATE].iloc[-1].date()),
        "warnings": warnings,
    }


__all__ = [
    "LABEL_END_PREFIX",
    "MAX_ENTITIES",
    "PANEL_DATE",
    "PANEL_ENTITY",
    "PANEL_LABEL_END",
    "PANEL_TARGET",
    "TARGET_PREFIX",
    "label_end_column_for",
    "load_external_panel",
    "target_column_for",
]
