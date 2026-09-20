"""
Target construction -- the dispatcher over `modeling.targets`.

This module used to hold the arithmetic and a chain of `if spec.type ==`
branches; the arithmetic lives in `targets/builtin.py` now and the branches
are a registry lookup. The public names are unchanged so every caller and
test that imported them from here still does:

    build_target(close_or_ohlcv, spec, context=None) -> Series
    build_label_end_dates(close_or_ohlcv, spec, context=None) -> Series
    apply_cross_sectional_target(panel, spec, target_col="target") -> panel
    CROSS_SECTIONAL_TARGETS -- the labels defined against the date's peers

Two kinds live in the registry. Most targets are ENTITY-LOCAL: they need
only that entity's own history, so they are built inside the per-entity
loop alongside the features. A few are CROSS-SECTIONAL -- a rank within the
date, a return measured against the universe average -- and cannot be
built until every entity is stacked into one panel; those carry a
`cross_sectional_stage`, which the builder applies once.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from standard_quant_tools.error import ValidationError

from ..targets import (  # noqa: F401  (re-exports; importing registers the built-ins)
    CROSS_SECTIONAL_TARGETS,
    get_target,
    horizon_label_end,
)
from ..targets.builtin import _horizon_volatility, _triple_barrier  # noqa: F401


def _as_ohlcv(close_or_ohlcv: Any) -> pd.DataFrame:
    """A bare Close series is accepted for every caller that has only ever
    passed one; a registered builder always receives a frame."""
    if isinstance(close_or_ohlcv, pd.DataFrame):
        return close_or_ohlcv
    return pd.DataFrame({"Close": close_or_ohlcv})


def build_target(
    close_or_ohlcv: Any, spec: Any, context: Optional[Any] = None
) -> pd.Series:
    """
    Build the supervised target for one entity, from the registry.

    Raises:
        ValidationError: the label is external-only, or the frame lacks a
        column the label's builder reads.
    """
    definition = get_target(spec.type)
    if not definition.buildable:
        raise ValidationError(
            f"target type {spec.type!r} cannot be built from a price series: "
            f"{definition.description} Nothing in a Close column determines it -- "
            "it is a function of the book, of orders, or of fills. Compute "
            "it where that data lives and bring the panel in with "
            "register_external_panel, which records what a label IS rather "
            "than recomputing it. A bar-derived approximation would be a "
            "number with nothing behind it."
        )
    ohlcv = _as_ohlcv(close_or_ohlcv)
    missing = [c for c in definition.requires if c not in ohlcv.columns]
    if missing:
        raise ValidationError(
            f"target {spec.type!r} reads column(s) {missing}, which the frame "
            f"does not carry (it has {sorted(ohlcv.columns)}). A label that "
            "needs more than Close cannot be built from a Close series alone."
        )
    return definition.builder(ohlcv, spec, context)


def build_label_end_dates(
    close_or_ohlcv: Any, spec: Any, context: Optional[Any] = None
) -> pd.Series:
    """
    The date of the LAST bar each row's target actually observes.

    Row t's forward return reads close[t+horizon], so its label is only
    fully determined once bar t+horizon has printed; the engine purges any
    training row whose label end lands inside the test block. The default
    is `horizon` bars ahead on the entity's own calendar; a label that can
    close early supplies its own rule through `label_end_builder`.
    """
    definition = get_target(spec.type)
    builder = definition.label_end_builder or horizon_label_end
    return builder(_as_ohlcv(close_or_ohlcv), spec, context)


def apply_cross_sectional_target(
    panel: pd.DataFrame, spec: Any, target_col: str = "target"
) -> pd.DataFrame:
    """
    Turn a stacked placeholder column into a cross-sectional target.

    Called once by the builder after every entity is in one frame, because
    such a label is defined against the OTHER entities on the same date and
    simply does not exist per entity. A label without a stage passes
    through untouched.
    """
    definition = get_target(spec.type)
    if definition.cross_sectional_stage is None or panel.empty:
        return panel
    return definition.cross_sectional_stage(panel, spec, target_col)


__all__ = [
    "CROSS_SECTIONAL_TARGETS",
    "apply_cross_sectional_target",
    "build_label_end_dates",
    "build_target",
]
