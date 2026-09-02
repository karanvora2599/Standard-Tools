"""Multi-entity panel construction: turns per-entity feature DataFrames
and target Series into one long-format (date, entity, <feature_ids>,
target) panel, and builds the dates x entities return panel universe-scope
features (features/factors.py) need.

Both stackers return the aligned panel AND an attribution of what the
alignment dropped. Row loss here is normal — every feature consumes its
lookback window, and a forward-return target consumes `horizon` bars at
the end — but it was previously invisible, reported only as a final row
count with no way to tell "this is the warm-up I asked for" apart from
"one feature is silently costing me 60% of my data". The tuple return is
deliberate: an external caller breaks loudly on unpacking rather than
silently receiving un-dropped rows.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def build_returns_panel(close_by_entity: Dict[str, pd.Series]) -> pd.DataFrame:
    """dates x entities panel of Close-to-Close returns, aligned on the
    intersection of every entity's index — PCA needs a complete
    cross-section each date, not a ragged one, so a date missing for any
    single entity is dropped for all. dataset/coverage.py reports what that
    intersection cost when it is material."""
    # fill_method=None, EXPLICITLY. pandas' default is "pad": a gap in a
    # price series is forward-filled before the difference is taken, so a
    # name that did not trade is credited with a 0.00% return rather than
    # with no return at all. Measured on a two-name panel with one halted
    # day, the default produced B = 0.000000 on the halted date and the
    # explicit form produces NaN, which `dropna` then removes.
    #
    # A fabricated zero is not a harmless placeholder here. It biases the
    # covariance and the correlation toward understating a halted name's
    # volatility and its co-movement with everything else -- and this panel
    # is what PCA and the optimizer see. pandas has deprecated the default,
    # so leaving it implicit also meant every one of those numbers would
    # change silently on a pandas upgrade.
    returns = {
        entity: close.pct_change(fill_method=None)
        for entity, close in close_by_entity.items()
    }
    return pd.DataFrame(returns).dropna(how="any")


LABEL_END_COL = "label_end_date"


def attribute_drops(
    panel: pd.DataFrame,
    feature_cols: List[str],
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Which column cost which rows, computed on the panel BEFORE the drop.

    Two counts per column, because they answer different questions:

      n_missing       rows where this column is NaN. Every one of them is
                      dropped, but several columns are usually NaN in the
                      same row (every feature's warm-up overlaps), so these
                      sum to far more than the rows actually lost and
                      cannot be read as a cost per feature.
      n_sole_missing  rows where this column is the ONLY thing missing.
                      This is the actionable number: it is exactly what
                      dropping that one feature from the spec would give
                      back. A feature with a large lookback sitting behind
                      an even larger one has n_missing in the thousands and
                      n_sole_missing of zero — removing it would change
                      nothing, and only the second number says so.

    The target is attributed separately from the features. Its loss has a
    different cause (the forward horizon at the end of each entity's
    series) and a different remedy (a shorter horizon or more data), and
    unlike a feature it cannot simply be removed.
    """
    subset = list(feature_cols) + ([target_col] if target_col else [])
    if panel.empty or not subset:
        return {
            "rows_before_alignment": int(len(panel)),
            "rows_after_alignment": int(len(panel)),
            "rows_dropped": 0,
            "per_feature": {},
            "per_entity_rows_dropped": {},
        }

    missing = panel[subset].isna()
    n_missing_per_row = missing.sum(axis=1)
    dropped_mask = n_missing_per_row > 0

    per_feature: Dict[str, Dict[str, int]] = {}
    for col in subset:
        per_feature[col] = {
            "n_missing": int(missing[col].sum()),
            "n_sole_missing": int((missing[col] & (n_missing_per_row == 1)).sum()),
        }

    per_entity_dropped: Dict[str, int] = {}
    if "entity" in panel.columns:
        per_entity_dropped = {
            str(entity): int(count)
            for entity, count in panel.loc[dropped_mask, "entity"]
            .value_counts()
            .items()
        }

    return {
        "rows_before_alignment": int(len(panel)),
        "rows_after_alignment": int(len(panel) - dropped_mask.sum()),
        "rows_dropped": int(dropped_mask.sum()),
        "per_feature": per_feature,
        "per_entity_rows_dropped": per_entity_dropped,
    }


def _stack(
    per_entity_features: Dict[str, pd.DataFrame],
    target_by_entity: Optional[Dict[str, pd.Series]] = None,
    label_end_by_entity: Optional[Dict[str, pd.Series]] = None,
) -> pd.DataFrame:
    """The long frame, before any row is dropped."""
    frames = []
    for entity, feat_df in per_entity_features.items():
        frame = feat_df.copy()
        if target_by_entity is not None:
            frame["target"] = target_by_entity[entity]
        if label_end_by_entity is not None:
            frame[LABEL_END_COL] = label_end_by_entity[entity]
        frame["entity"] = entity
        frame["date"] = frame.index
        frames.append(frame.reset_index(drop=True))
    return pd.concat(frames, ignore_index=True)


def stack_long(
    per_entity_features: Dict[str, pd.DataFrame],
    target_by_entity: Dict[str, pd.Series],
    label_end_by_entity: Dict[str, pd.Series] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    per_entity_features: {entity: DataFrame(index=date, columns=feature_ids)}.
    target_by_entity:    {entity: Series(index=date)}.
    label_end_by_entity: {entity: Series(index=date)} — the date each row's
        target finishes observing (see dataset/target.py::build_label_end_dates).
        Carried into the panel as `label_end_date` so walk-forward validation
        can purge training rows whose label crosses into the test window.

    Returns (panel, attribution). The panel is one long DataFrame, one row
    per (date, entity), columns [date, entity, <feature_ids>, target,
    label_end_date] — rows with any NaN feature or target dropped
    (unavoidable at the start of each feature's lookback window and the end
    of the target's forward horizon). `attribution` says which column cost
    which rows; see attribute_drops.
    """
    long_panel = _stack(per_entity_features, target_by_entity, label_end_by_entity)
    reserved = ("date", "entity", "target", LABEL_END_COL)
    feature_cols = [c for c in long_panel.columns if c not in reserved]

    attribution = attribute_drops(long_panel, feature_cols, target_col="target")

    long_panel = long_panel.dropna(subset=feature_cols + ["target"])
    ordered = ["date", "entity"] + feature_cols + ["target"]
    if label_end_by_entity is not None:
        ordered.append(LABEL_END_COL)
    panel = long_panel[ordered].sort_values(["date", "entity"]).reset_index(drop=True)
    return panel, attribution


def stack_features_only(
    per_entity_features: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Same as stack_long, but for scoring rather than training: no target
    column, since score_model asks for a prediction as of a date where
    the forward-return target hasn't happened yet (it needs `horizon`
    bars of future data that don't exist for "today"). Using stack_long
    for scoring would silently drop exactly the most recent rows scoring
    needs, via its target-based dropna — this function exists specifically
    to avoid that bug, not as a convenience wrapper.
    """
    long_panel = _stack(per_entity_features)
    feature_cols = [c for c in long_panel.columns if c not in ("date", "entity")]

    attribution = attribute_drops(long_panel, feature_cols)

    long_panel = long_panel.dropna(subset=feature_cols)
    ordered = ["date", "entity"] + feature_cols
    panel = long_panel[ordered].sort_values(["date", "entity"]).reset_index(drop=True)
    return panel, attribution
