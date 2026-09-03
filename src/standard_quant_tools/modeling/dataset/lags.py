"""
History as columns, which is the only way a sequence reaches this engine.

WHY THIS EXISTS RATHER THAN A SEQUENCE ESTIMATOR. `engine.py` hands an
estimator a 2-D X whose rows are (date, entity) observations and which
carries NO entity identity -- deliberately, because that contract is what
lets ridge, LightGBM and an SGD learner be interchangeable. A model that
wants yesterday's value of a feature therefore cannot reconstruct it from
what it is given; the window has to be in the columns before the engine
ever sees the data. Putting it here means EVERY estimator gets access to
history, not just one that ships with special plumbing.

WHY PER ENTITY, BEFORE STACKING. The expansion runs on each entity's own
feature frame, indexed by that entity's own bars, so a shift can only ever
reach that entity's earlier rows. Doing it after `stack_long` would shift
across the entity boundary and hand AAA a lagged value belonging to BBB --
a bug that produces a perfectly plausible panel and is invisible in every
aggregate statistic downstream.

WHY NEGATIVE LAGS ARE REFUSED AND NOT CLAMPED. A negative shift is next
week's value of a feature sitting on this week's row. It is the single most
effective way to make a model look brilliant, it survives every leakage
check that reasons about TARGETS rather than features, and a caller who
typed -1 meaning "one bar back" would get exactly that result with no
error. So the sign is rejected at the spec boundary with the reason
attached.

WHAT IT COSTS. Lag k needs k more bars of warm-up, so the deepest lag sets
where the panel can start; those rows are dropped by the existing alignment
and land in `drop_attribution` like any other missing feature. The width
cost is multiplicative -- ten features at ten lags is a hundred and ten
columns -- which is why both the depth and the count are bounded rather
than left to whatever a caller passes.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import pandas as pd

from standard_quant_tools.error import ValidationError

# Declared in `modeling/limits.py` rather than here, because `specs.py`
# needs the same numbers to put them in the JSON schema and cannot import
# this module without a cycle. Re-exported so this module's API is
# unchanged for everything that already reads them from here.
from ..limits import MAX_EXPANDED_COLUMNS, MAX_LAG, MAX_LAGS_PER_FEATURE

#: Separator between a feature's column name and its lag depth. Two
#: underscores because a single one appears inside feature ids already
#: (`technical.rsi`, `market.momentum`), and a name that can be split
#: unambiguously is what lets `analyze_model_errors` and the importance
#: summary report "this is rsi at lag 3" rather than an opaque string.
LAG_SUFFIX = "__lag"


def lag_column_name(output_name: str, lag: int) -> str:
    """The panel column carrying `output_name` as of `lag` bars earlier."""
    return f"{output_name}{LAG_SUFFIX}{lag}"


def parse_lag_column(column: str):
    """(feature, lag) for a lag column, or None. The inverse of the name."""
    if LAG_SUFFIX not in column:
        return None
    base, _, depth = column.rpartition(LAG_SUFFIX)
    if not base or not depth.isdigit():
        return None
    return base, int(depth)


def validate_lags(lags: Sequence[int], *, field: str = "lags") -> List[int]:
    """
    Check one feature's requested lags, and say why when refusing.

    Returns them sorted and de-duplicated, so two specs asking for [2, 1]
    and [1, 2] produce identical panels and therefore identical dataset
    hashes -- an ordering difference must not create a second dataset.
    """
    values = [int(v) for v in lags]
    for value in values:
        if value < 0:
            raise ValidationError(
                f"{field} contains {value}, a NEGATIVE lag. That is a shift "
                "FORWARD: it would put a future value of the feature on "
                "today's row, which is lookahead of the most damaging kind "
                "-- it survives every leakage check that reasons about the "
                "target, and it makes the model look excellent. If you "
                f"meant {abs(value)} bar(s) back, pass {abs(value)}."
            )
        if value == 0:
            raise ValidationError(
                f"{field} contains 0, which is the feature's own current "
                "value -- already in the panel as its ordinary column. Ask "
                "for 1 or more, or drop the 0."
            )
        if value > MAX_LAG:
            raise ValidationError(
                f"{field} contains {value}, deeper than the {MAX_LAG}-bar "
                "limit. Every entity pays that many bars of warm-up before "
                "the panel can start, and a value that old describes a "
                "different regime rather than the recent path."
            )
    unique = sorted(set(values))
    if len(unique) > MAX_LAGS_PER_FEATURE:
        raise ValidationError(
            f"{field} asks for {len(unique)} lags, more than the "
            f"{MAX_LAGS_PER_FEATURE} allowed for one feature. Each one adds a "
            "column for every feature it is applied to."
        )
    return unique


def expand_lags(frame: pd.DataFrame, lags_by_feature: Dict[str, List[int]]):
    """
    Add lagged copies to ONE ENTITY's feature frame.

    The frame is that entity's own features on its own bar index, so
    `shift(k)` reaches its earlier bars and nothing else. The caller
    guarantees that by calling this before the entities are stacked.
    """
    if not lags_by_feature:
        return frame
    additions: Dict[str, pd.Series] = {}
    for name, lags in lags_by_feature.items():
        if name not in frame.columns:
            continue
        column = frame[name]
        for lag in lags:
            additions[lag_column_name(name, lag)] = column.shift(lag)
    if not additions:
        return frame
    return pd.concat([frame, pd.DataFrame(additions, index=frame.index)], axis=1)


def expanded_feature_ids(specs: Iterable) -> List[str]:
    """
    Every column the panel will carry, feature then its lags, in order.

    Order is the panel's column order and therefore the order of X's
    columns and of the feature-importance vector, so it is generated in one
    place rather than reconstructed by each consumer.
    """
    names: List[str] = []
    for spec in specs:
        names.append(spec.output_name)
        for lag in getattr(spec, "lags", None) or []:
            names.append(lag_column_name(spec.output_name, lag))
    if len(names) > MAX_EXPANDED_COLUMNS:
        raise ValidationError(
            f"this spec expands to {len(names)} feature columns, past the "
            f"{MAX_EXPANDED_COLUMNS} limit. Lags multiply: every lag is a new "
            "column for each feature it is applied to. Ask for fewer lags, or "
            "apply them to the features whose history you actually believe in "
            "rather than to all of them."
        )
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValidationError(
            f"these panel columns are requested more than once: {duplicates}. "
            "A feature aliased to another feature's lag column name would "
            "silently overwrite it -- give one of them a different alias."
        )
    return names


def lags_by_output_name(specs: Iterable) -> Dict[str, List[int]]:
    """{panel column: its requested lags}, for the entity-scope expansion."""
    return {
        spec.output_name: list(spec.lags)
        for spec in specs
        if getattr(spec, "lags", None)
    }


def deepest_lag(specs: Iterable) -> int:
    """Extra warm-up bars the deepest lag costs, for the lookback report."""
    return max(
        (max(spec.lags) for spec in specs if getattr(spec, "lags", None)),
        default=0,
    )


__all__ = [
    "LAG_SUFFIX",
    "MAX_EXPANDED_COLUMNS",
    "MAX_LAG",
    "MAX_LAGS_PER_FEATURE",
    "deepest_lag",
    "expand_lags",
    "expanded_feature_ids",
    "lag_column_name",
    "lags_by_output_name",
    "parse_lag_column",
    "validate_lags",
]
