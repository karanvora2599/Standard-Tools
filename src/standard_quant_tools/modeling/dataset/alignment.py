"""Multi-entity panel construction: turns per-entity feature DataFrames
and target Series into one long-format (date, entity, <feature_ids>,
target) panel, and builds the dates x entities return panel universe-scope
features (features/factors.py) need."""

from typing import Dict

import pandas as pd


def build_returns_panel(close_by_entity: Dict[str, pd.Series]) -> pd.DataFrame:
    """dates x entities panel of Close-to-Close returns, aligned on the
    intersection of every entity's index — PCA needs a complete
    cross-section each date, not a ragged one, so a date missing for any
    single entity is dropped for all."""
    returns = {entity: close.pct_change() for entity, close in close_by_entity.items()}
    return pd.DataFrame(returns).dropna(how="any")


LABEL_END_COL = "label_end_date"


def stack_long(
    per_entity_features: Dict[str, pd.DataFrame],
    target_by_entity: Dict[str, pd.Series],
    label_end_by_entity: Dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """
    per_entity_features: {entity: DataFrame(index=date, columns=feature_ids)}.
    target_by_entity:    {entity: Series(index=date)}.
    label_end_by_entity: {entity: Series(index=date)} — the date each row's
        target finishes observing (see dataset/target.py::build_label_end_dates).
        Carried into the panel as `label_end_date` so walk-forward validation
        can purge training rows whose label crosses into the test window.

    Returns one long DataFrame, one row per (date, entity), columns
    [date, entity, <feature_ids>, target, label_end_date] — rows with any
    NaN feature or target dropped (unavoidable at the start of each
    feature's lookback window and the end of the target's forward horizon).
    """
    frames = []
    for entity, feat_df in per_entity_features.items():
        frame = feat_df.copy()
        frame["target"] = target_by_entity[entity]
        if label_end_by_entity is not None:
            frame[LABEL_END_COL] = label_end_by_entity[entity]
        frame["entity"] = entity
        frame["date"] = frame.index
        frames.append(frame.reset_index(drop=True))

    long_panel = pd.concat(frames, ignore_index=True)
    reserved = ("date", "entity", "target", LABEL_END_COL)
    feature_cols = [c for c in long_panel.columns if c not in reserved]
    long_panel = long_panel.dropna(subset=feature_cols + ["target"])
    ordered = ["date", "entity"] + feature_cols + ["target"]
    if label_end_by_entity is not None:
        ordered.append(LABEL_END_COL)
    return long_panel[ordered].sort_values(["date", "entity"]).reset_index(drop=True)


def stack_features_only(per_entity_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Same as stack_long, but for scoring rather than training: no target
    column, since score_model asks for a prediction as of a date where
    the forward-return target hasn't happened yet (it needs `horizon`
    bars of future data that don't exist for "today"). Using stack_long
    for scoring would silently drop exactly the most recent rows scoring
    needs, via its target-based dropna — this function exists specifically
    to avoid that bug, not as a convenience wrapper.
    """
    frames = []
    for entity, feat_df in per_entity_features.items():
        frame = feat_df.copy()
        frame["entity"] = entity
        frame["date"] = frame.index
        frames.append(frame.reset_index(drop=True))

    long_panel = pd.concat(frames, ignore_index=True)
    feature_cols = [c for c in long_panel.columns if c not in ("date", "entity")]
    long_panel = long_panel.dropna(subset=feature_cols)
    ordered = ["date", "entity"] + feature_cols
    return long_panel[ordered].sort_values(["date", "entity"]).reset_index(drop=True)
