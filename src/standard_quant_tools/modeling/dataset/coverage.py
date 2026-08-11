"""
Dataset coverage and provenance diagnostics.

`BuildModelDatasetResult.warnings` existed as a field from the start and
was never populated by anything, so every one of these conditions was
silent. They share a shape: none of them is wrong enough to refuse to
build the dataset, and all of them change how the resulting OOS metrics
should be read. A model trained on a survivors-only universe with a
symbol contributing two years of a ten-year window is not broken — it is
answering a narrower question than its metrics appear to claim, and the
caller is the only one who can decide whether that matters.

Deliberately warnings and not errors: every provider this package ships
reports `point_in_time=False` and `survivorship_free=False`, so promoting
those to a hard failure would make the modeling runtime unusable against
its own default data source while teaching the caller nothing.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# An entity covering less than this fraction of the universe's combined
# date range is flagged. A threshold exists because isolated missing bars
# (a halt, a local holiday on a cross-listed name) are normal and
# uninteresting, while a materially short history is the thing that
# quietly changes what was trained: with complete-case alignment it either
# truncates the panel for everyone or contributes far fewer rows than its
# presence in `universe` suggests.
_COVERAGE_WARN_FRACTION = 0.98

# Fraction of the union's dates that must survive the universe-scope
# intersection before it is reported. Anything below this means the
# cross-sectional features were fit on a materially shorter history than
# the caller asked for.
_INTERSECTION_WARN_FRACTION = 0.95


# Share of pre-alignment rows that must be lost before the drop breakdown
# is reported. Warm-up loss is expected and universal -- warning about every
# dataset would train the reader to skip the warnings that matter.
_DROP_WARN_FRACTION = 0.30

# A single feature costing more than this share of pre-alignment rows on its
# own is named even when the overall drop rate is unremarkable, since it is
# individually removable.
_SOLE_CAUSE_WARN_FRACTION = 0.10


def _fmt_date(value: Any) -> str:
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except (ValueError, TypeError):  # pragma: no cover — defensive
        return str(value)


def provider_guarantee_warnings(metadata: Optional[Any]) -> List[str]:
    """
    Surface what the provider does NOT promise.

    `DataSetMetadata.point_in_time` / `.survivorship_free` have been
    reported honestly by every provider since they were added, and nothing
    in the modeling runtime read either one. The point-in-time gate in
    dataset/leakage.py checks each FEATURE's static temporal label, which
    is a statement about the formula and says nothing about whether the
    data underneath it gets revised — so a dataset could pass that gate
    while being built entirely from a source that makes no such guarantee.
    """
    if metadata is None:
        return []

    warnings: List[str] = []
    provider = getattr(metadata, "provider", "unknown")

    if not getattr(metadata, "point_in_time", False):
        warnings.append(
            f"provider {provider!r} does not guarantee point-in-time data "
            "(point_in_time=False): historical values may be silently revised after "
            "the fact, so a feature computed today can differ from what was actually "
            "observable on the date it is labelled with. The per-feature PIT check "
            "only verifies that each feature's FORMULA is causal — it cannot see "
            "revisions in the underlying series."
        )
    if not getattr(metadata, "survivorship_free", False):
        warnings.append(
            f"provider {provider!r} does not guarantee a survivorship-free universe "
            "(survivorship_free=False): delisted and defunct securities are not "
            "queryable, so any universe assembled from currently-listed symbols is a "
            "survivors-only sample. Backtested returns on such a universe are biased "
            "upward, and no amount of walk-forward validation corrects for it."
        )
    return warnings


def interval_warnings(interval: str) -> List[str]:
    """The built-in features are calibrated for daily bars — their default
    windows are stated in trading days and the volatility features
    annualize with a daily constant. A non-daily interval is supported and
    fetched correctly; what it does NOT do is reinterpret those defaults."""
    if interval == "1d":
        return []
    return [
        f"interval={interval!r} is not daily. Every feature lookback and "
        "target.horizon counts BARS of this interval, and the built-in features' "
        "default parameters are calibrated for daily bars (window=252 means one "
        "trading year at '1d', not at any other interval). The realized-volatility "
        "features additionally annualize with a daily constant, so their absolute "
        "scale is wrong here — harmless for a standardized model whose ranking is "
        "unaffected by a constant factor, misleading if you read the values "
        "directly. Set feature parameters explicitly for a non-daily interval."
    ]


def entity_coverage_warnings(
    ohlcv_by_entity: Dict[str, pd.DataFrame],
    start: str,
    end: str,
) -> List[str]:
    """
    Report entities whose history materially under-covers the universe, and
    a universe that under-covers the requested window.

    Time-varying universe membership is not modelled (see
    Documentation/15_modeling.md) — `universe` is a fixed list applied to
    the whole window. This function does not fix that; it makes the cases
    where it bites visible instead of silent.
    """
    if not ohlcv_by_entity:
        return []

    warnings: List[str] = []

    union_dates = pd.DatetimeIndex([])
    for frame in ohlcv_by_entity.values():
        union_dates = union_dates.union(pd.DatetimeIndex(frame.index))
    if len(union_dates) == 0:
        return []

    n_union = len(union_dates)
    short: List[str] = []
    for entity in sorted(ohlcv_by_entity):
        index = pd.DatetimeIndex(ohlcv_by_entity[entity].index)
        if len(index) == 0:  # pragma: no cover — empty frames rejected upstream
            continue
        fraction = len(index) / n_union
        if fraction < _COVERAGE_WARN_FRACTION:
            short.append(
                f"{entity} {len(index)}/{n_union} bars ({fraction:.0%}, "
                f"{_fmt_date(index[0])} to {_fmt_date(index[-1])})"
            )

    if short:
        warnings.append(
            "partial history: "
            + "; ".join(short)
            + f" — the universe spans {_fmt_date(union_dates[0])} to "
            f"{_fmt_date(union_dates[-1])}. A symbol that listed inside the window "
            "contributes rows only from its listing date, so it is weighted far less "
            "than its presence in `universe` suggests; one that stops early was "
            "delisted, halted, or renamed. Neither is corrected for."
        )

    # The requested window versus what actually came back. Distinct from the
    # per-entity check above: this fires when EVERY symbol starts late, which
    # per-entity coverage (measured against the union) cannot see.
    try:
        requested_start = pd.Timestamp(start)
        requested_end = pd.Timestamp(end)
    except (ValueError, TypeError):  # pragma: no cover — validated in DatasetSpec
        return warnings

    actual_start = union_dates[0]
    actual_end = union_dates[-1]
    # Ten calendar days absorbs weekends, holidays and a start date that
    # simply is not a trading day, without absorbing a genuinely
    # unavailable window.
    tolerance = pd.Timedelta(days=10)
    if actual_start - requested_start > tolerance:
        warnings.append(
            f"requested start {_fmt_date(requested_start)} but the earliest bar "
            f"available for any symbol is {_fmt_date(actual_start)} — the dataset "
            "covers a shorter window than requested, before any feature lookback is "
            "consumed."
        )
    if requested_end - actual_end > tolerance:
        warnings.append(
            f"requested end {_fmt_date(requested_end)} but the latest bar available "
            f"for any symbol is {_fmt_date(actual_end)}."
        )
    return warnings


def alignment_warnings(
    attribution: Dict[str, Any],
    entities_fetched: List[str],
    entities_surviving: List[str],
) -> List[str]:
    """
    What the feature/target alignment cost, and who to blame.

    Row loss during alignment is normal and was reported only as a final
    row count, which cannot distinguish the warm-up you asked for from one
    feature quietly consuming most of the panel. `n_sole_missing` is the
    number that makes this actionable: it is exactly what removing that one
    feature would give back, so a large-lookback feature sitting behind an
    even larger one correctly scores zero rather than looking like the
    culprit.
    """
    warnings: List[str] = []
    before = attribution.get("rows_before_alignment", 0)
    dropped = attribution.get("rows_dropped", 0)
    per_feature: Dict[str, Dict[str, int]] = attribution.get("per_feature", {})

    if before:
        sole = {
            name: counts.get("n_sole_missing", 0)
            for name, counts in per_feature.items()
            if counts.get("n_sole_missing", 0) > 0
        }
        heavy = {
            name: count
            for name, count in sole.items()
            if count / before >= _SOLE_CAUSE_WARN_FRACTION
        }

        if dropped / before >= _DROP_WARN_FRACTION or heavy:
            ranked = sorted(sole.items(), key=lambda kv: -kv[1])[:5]
            if ranked:
                breakdown = ", ".join(
                    f"{name} {count} ({count / before:.0%})" for name, count in ranked
                )
                attributable = (
                    f" Rows lost to a single column, i.e. recoverable by removing "
                    f"just that one: {breakdown}."
                )
            else:
                # Every dropped row was missing two or more columns, so no
                # single removal recovers anything -- worth saying outright,
                # since an empty breakdown otherwise reads as a bug.
                attributable = (
                    " No single column accounts for any dropped row on its own — "
                    "every loss is a warm-up window overlapping another, so removing "
                    "one feature recovers nothing. Shorten the longest lookback or "
                    "start the dataset earlier."
                )
            warnings.append(
                f"alignment dropped {dropped} of {before} row(s) "
                f"({dropped / before:.0%}) — each feature consumes its lookback "
                f"window and the target consumes its forward horizon." + attributable
            )

    missing_entities = sorted(set(entities_fetched) - set(entities_surviving))
    if missing_entities:
        warnings.append(
            f"fetched but contributed no rows to the panel: {missing_entities} — "
            "their history is shorter than the feature lookbacks plus the target "
            "horizon, so they were counted in `universe` but the model never saw "
            "them."
        )
    return warnings


def intersection_warnings(
    ohlcv_by_entity: Dict[str, pd.DataFrame],
    returns_panel: pd.DataFrame,
    has_universe_scope_features: bool,
) -> List[str]:
    """
    Complete-case alignment cost for universe-scope (PCA) features.

    `alignment.build_returns_panel` drops any date missing for ANY entity,
    because PCA needs a rectangular cross-section. That is the right
    call numerically and an invisible one operationally: adding a single
    recently-listed symbol to a ten-year universe silently truncates the
    cross-sectional features — and therefore the whole panel, since rows
    with a NaN feature are dropped — down to that symbol's history.

    Only reported when a universe-scope feature was actually requested;
    otherwise the returns panel is computed but never consumed, and the
    intersection costs nothing.
    """
    if not has_universe_scope_features or not ohlcv_by_entity:
        return []

    union_dates = pd.DatetimeIndex([])
    for frame in ohlcv_by_entity.values():
        union_dates = union_dates.union(pd.DatetimeIndex(frame.index))
    n_union = len(union_dates)
    n_kept = len(returns_panel.index)
    if n_union == 0 or n_kept / n_union >= _INTERSECTION_WARN_FRACTION:
        return []

    # Name the binding constraint rather than just the shortfall: with a
    # complete-case intersection, the entity whose history starts latest is
    # usually the entire explanation, and it is the one the caller would
    # drop or replace.
    latest_start_entity = max(
        ohlcv_by_entity,
        key=lambda e: (
            pd.DatetimeIndex(ohlcv_by_entity[e].index)[0]
            if len(ohlcv_by_entity[e].index)
            else pd.Timestamp.min
        ),
    )
    latest_start = pd.DatetimeIndex(ohlcv_by_entity[latest_start_entity].index)[0]

    return [
        f"universe-scope (PCA) features are fit on the {n_kept} date(s) present for "
        f"EVERY entity, out of {n_union} across the universe "
        f"({n_kept / n_union:.0%}) — a complete cross-section is required, so one "
        f"short history truncates the panel for all entities. The latest-starting "
        f"symbol is {latest_start_entity} ({_fmt_date(latest_start)}); dropping it, "
        "or starting the dataset later, recovers the rest."
    ]
