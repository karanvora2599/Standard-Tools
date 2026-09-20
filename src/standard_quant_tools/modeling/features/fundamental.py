"""
Fundamental features, computed from point-in-time filing records.

WHAT MAKES THESE DIFFERENT FROM EVERY OTHER FEATURE HERE. A technical
feature reads bars, and a bar is knowable at its own close. A filing is
not: it describes a quarter that ended weeks before anyone could read it,
and it may be restated months later. So these features never see OHLCV.
Their `fn` takes the provider's point-in-time RECORD SET -- one row per
version of a fact, stamped with `event_time` (the period it describes)
and `available_time` (when it could first be acted on) -- and returns a
value series in the same schema. The dataset builder then joins that onto
the panel by availability time, which is the join `dataset/point_in_time`
exists to make impossible to get wrong.

TWO KINDS OF TRANSFORM, AND WHY THE SECOND IS THE HARD ONE. A ratio within
one filing (net margin) is a row-wise map: each version of the filing
yields one version of the ratio, available when that filing was. A
comparison ACROSS filings (revenue growth, year over year) is not: its
value at time t is a function of the version of THIS quarter's filing
known at t and the version of LAST year's filing known at t, and either
can be restated. The derived series therefore has a version at every time
at which either input changed, and `_paired_versions` builds exactly that
set rather than pairing each current version with whatever the prior
fact's final value happened to be -- which would read a restatement nobody
had seen yet.

`max_staleness_days` is a parameter of every feature here, because a
quarterly figure older than about four months has been superseded and a
feed that stops updating would otherwise supply its last value forever.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import register_feature

#: The point-in-time record schema, written out here rather than imported
#: from `modeling.dataset.point_in_time`: the dataset package imports the
#: builder, which imports the specs, which import this package, and a
#: feature module importing the dataset package back would close a cycle
#: at import time. The names are the schema; the join module owns the
#: rules.
ENTITY = "entity"
EVENT_TIME = "event_time"
AVAILABLE_TIME = "available_time"

#: The column a point-in-time transform returns its value in; the builder
#: renames it to the feature's output name.
VALUE = "value"

#: Polygon's financials fields, as `<statement>.<key>`.
REVENUES = "income_statement.revenues"
NET_INCOME = "income_statement.net_income_loss"
DILUTED_EPS = "income_statement.diluted_earnings_per_share"

#: Period keys a cross-filing transform needs. A quarter is identified by
#: its fiscal year and period ("Q1".."Q4"), not by its calendar end date,
#: because fiscal calendars differ by company and a 52/53-week year moves
#: the end date by days.
FISCAL_YEAR = "fiscal_year"
FISCAL_PERIOD = "fiscal_period"

_SCHEMA = (ENTITY, EVENT_TIME, AVAILABLE_TIME)


def _require(records: pd.DataFrame, columns: List[str], feature_id: str) -> None:
    missing = [c for c in columns if c not in records.columns]
    if missing:
        raise ValidationError(
            f"feature {feature_id!r}: the point-in-time records carry no "
            f"column(s) {missing}. The provider was asked for them; a record "
            "set from another source needs the same columns to serve this "
            "feature."
        )


def _row_wise(records: pd.DataFrame, values: pd.Series) -> pd.DataFrame:
    """One derived version per record version."""
    out = records[list(_SCHEMA)].copy()
    out[VALUE] = values.to_numpy(dtype=float)
    return out


def diluted_eps(records: pd.DataFrame, context: FeatureContext, **params: Any):
    """Diluted earnings per share as filed, every version of it."""
    _require(records, [DILUTED_EPS], "fundamental.diluted_eps")
    return _row_wise(records, records[DILUTED_EPS].astype(float))


def net_margin(records: pd.DataFrame, context: FeatureContext, **params: Any):
    """Net income over revenues, within one filing. NaN where revenues are
    not positive: a margin on zero or negative revenue is not a margin."""
    _require(records, [NET_INCOME, REVENUES], "fundamental.net_margin")
    revenue = records[REVENUES].astype(float)
    income = records[NET_INCOME].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = income / revenue
    ratio = ratio.where(revenue > 0)
    return _row_wise(records, ratio)


def _paired_versions(
    records: pd.DataFrame, field: str, periods_back: int
) -> pd.DataFrame:
    """
    Every version of (this period's value, the value `periods_back` fiscal
    years earlier) as it was known at each point in time.

    For one fact C with versions at times c1 < c2 < ... and its prior fact
    P with versions at p1 < p2 < ..., the derived series changes at every
    time in the union of both, from the first time both exist. At each such
    time t the pair is (latest C version at or before t, latest P version
    at or before t). Rows: `entity`, `event_time` (C's), `available_time`
    (t), `current`, `prior`.
    """
    keys = [ENTITY, FISCAL_YEAR, FISCAL_PERIOD]
    frame = records[keys + [EVENT_TIME, AVAILABLE_TIME, field]].copy()
    frame[FISCAL_YEAR] = frame[FISCAL_YEAR].astype(int)
    frame[FISCAL_PERIOD] = frame[FISCAL_PERIOD].astype(str)
    frame = frame.sort_values(AVAILABLE_TIME, kind="stable")
    by_fact = {key: group for key, group in frame.groupby(keys, sort=False)}

    rows = []
    for (entity, year, period), current in by_fact.items():
        prior = by_fact.get((entity, year - periods_back, period))
        if prior is None:
            continue
        event_time = current[EVENT_TIME].iloc[0]
        change_points = sorted(
            set(current[AVAILABLE_TIME]).union(prior[AVAILABLE_TIME])
        )
        first = current[AVAILABLE_TIME].min()
        for t in change_points:
            if t < first:
                continue
            c = current[current[AVAILABLE_TIME] <= t]
            p = prior[prior[AVAILABLE_TIME] <= t]
            if p.empty:
                continue
            rows.append(
                {
                    ENTITY: entity,
                    EVENT_TIME: event_time,
                    AVAILABLE_TIME: t,
                    "current": float(c[field].iloc[-1]),
                    "prior": float(p[field].iloc[-1]),
                }
            )
    return pd.DataFrame(
        rows, columns=[ENTITY, EVENT_TIME, AVAILABLE_TIME, "current", "prior"]
    )


def revenue_growth_yoy(records: pd.DataFrame, context: FeatureContext, **params: Any):
    """
    Revenue growth against the same fiscal period one year earlier, with a
    version at every time either filing changed.

    NaN where the prior year's revenue is not positive.
    """
    feature_id = "fundamental.revenue_growth_yoy"
    _require(records, [REVENUES, FISCAL_YEAR, FISCAL_PERIOD], feature_id)
    pairs = _paired_versions(records, REVENUES, periods_back=1)
    out = pairs[list(_SCHEMA)].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = pairs["current"] / pairs["prior"] - 1.0
    out[VALUE] = growth.where(pairs["prior"] > 0).to_numpy(dtype=float)
    return out


#: Four months: a quarterly figure older than that has been superseded by
#: the next filing, and a row still reading it is reading a feed that
#: stopped.
_DEFAULT_STALENESS_DAYS = 120

register_feature(
    FeatureDefinition(
        id="fundamental.diluted_eps",
        description=(
            "Diluted earnings per share as filed, joined by filing date; a "
            "restatement is a new version from the date it was filed."
        ),
        fn=diluted_eps,
        default_params={"max_staleness_days": _DEFAULT_STALENESS_DAYS},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.POINT_IN_TIME,
        lookback=0,
        frame_kind="fundamentals",
        fields=[DILUTED_EPS],
    )
)

register_feature(
    FeatureDefinition(
        id="fundamental.net_margin",
        description=(
            "Net income over revenues within one filing, joined by filing "
            "date; NaN where revenues are not positive."
        ),
        fn=net_margin,
        default_params={"max_staleness_days": _DEFAULT_STALENESS_DAYS},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.POINT_IN_TIME,
        lookback=0,
        frame_kind="fundamentals",
        fields=[NET_INCOME, REVENUES],
    )
)

register_feature(
    FeatureDefinition(
        id="fundamental.revenue_growth_yoy",
        description=(
            "Revenue growth against the same fiscal period a year earlier, "
            "with a version at every time either filing changed, so a "
            "restated prior year is read from the day it was restated."
        ),
        fn=revenue_growth_yoy,
        default_params={"max_staleness_days": _DEFAULT_STALENESS_DAYS},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.POINT_IN_TIME,
        lookback=0,
        frame_kind="fundamentals",
        fields=[REVENUES],
    )
)

__all__ = [
    "DILUTED_EPS",
    "FISCAL_PERIOD",
    "FISCAL_YEAR",
    "NET_INCOME",
    "REVENUES",
    "VALUE",
    "diluted_eps",
    "net_margin",
    "revenue_growth_yoy",
]
