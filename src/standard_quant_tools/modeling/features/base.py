"""
The contract every FEATURE_REGISTRY entry satisfies.

Two scopes exist because PCA-derived factors (features/factors.py) need
the whole universe's return panel at once, not one symbol's OHLCV —
`dataset.builder` dispatches each feature differently depending on which
scope it declares:

  entity   : fn(ohlcv: pd.DataFrame, context: FeatureContext, **params) -> pd.Series
             called once per symbol, using that symbol's own OHLCV.
  universe : fn(returns_panel: pd.DataFrame, context: FeatureContext, **params) -> pd.DataFrame
             called once for the whole DatasetSpec.universe, on a
             dates x entities return panel; output is dates x entities.

Every feature function takes `context` even when it doesn't use it (only
risk.rolling_beta does, for the benchmark series) — one uniform call
signature in dataset.builder, not a special case per feature.
"""

from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TemporalSupport(str, Enum):
    # The FORMULA is causal: this feature at date t reads only data from t
    # and earlier. Price/volume-derived features qualify.
    #
    # This is a property of the formula ONLY. It does not assert that the
    # underlying dataset is true point-in-time data -- the data layer
    # tracks that separately (DataSetMetadata.point_in_time /
    # survivorship_free, both reported False by the default yfinance
    # provider), and the modeling PIT gate does not currently consult it.
    # Nor does the label alone constrain PARAMETERS: a negative lookback
    # turns a "pit_safe" formula into a forward-looking one, which is why
    # features/params.py validates resolved parameter values separately.
    PIT_SAFE = "pit_safe"
    # e.g. fundamentals as currently reported — no point-in-time-safe
    # historical provider wired up yet (see dataset/leakage.py). Nothing
    # in Phase 1 uses this value; it exists so a future fundamentals
    # feature is rejected by construction until a real PIT data source
    # backs it, instead of silently leaking.
    CURRENT_ONLY = "current_only"


class FeatureScope(str, Enum):
    ENTITY = "entity"
    UNIVERSE = "universe"
    # Computed from a point-in-time RECORD SET rather than from bars: the
    # provider's `get_point_in_time_records` supplies one row per version
    # of a fact stamped with when it became knowable, the feature's `fn`
    # transforms those records into a value series in the same schema, and
    # the builder joins it onto the stacked panel by availability time. A
    # feature of this scope never touches OHLCV and cannot be lagged in
    # bars, because its rows are filings, not sessions.
    POINT_IN_TIME = "point_in_time"


# Column names the long panel builds itself. A feature's output column --
# its alias, or its id when it has none -- must not collide with any of
# them. Defined here rather than inline at each check so the alias path and
# the feature-id path cannot drift apart: the alias path was validated and
# the id path was not, which let a custom feature registered as id="target"
# produce a column that shadowed the panel's supervised target.
RESERVED_PANEL_COLUMNS = frozenset(
    {"date", "entity", "target", "label_end_date", "event"}
)


# Bars per year, by interval, for annualizing a per-bar volatility.
#
# Only intervals whose constant is unambiguous are listed. Daily, weekly and
# monthly are calendar-derived and need no assumption about session length.
# INTRADAY IS DELIBERATELY ABSENT: bars-per-year at "1h" depends on how many
# trading hours the venue is open (6.5 for US equities, 8.5 for the LSE,
# ~24 for crypto), and only an exchange calendar can say which. Picking one
# silently would make an "annualized" volatility wrong by a fixed
# multiplicative factor for every other market -- a number that looks
# precise and is not. With a calendar named on the dataset, `modeling.calendar`
# reads the session length and the sessions per year off it.
_PERIODS_PER_YEAR = {
    "1d": 252,
    "5d": 52,
    "1wk": 52,
    "1mo": 12,
    "3mo": 4,
}


def periods_per_year_for_interval(
    interval: str, calendar: Optional[str] = None
) -> Optional[int]:
    """
    Bars per year for `interval`.

    A daily-or-coarser interval is a constant. An intraday interval is
    bars per session times sessions per year, both read off the named
    exchange calendar, and None without one -- the caller then refuses or
    warns rather than assuming a venue.
    """
    known = _PERIODS_PER_YEAR.get(str(interval).strip())
    if known is not None:
        return known
    if calendar is None:
        return None
    from ..calendar import interval_minutes, periods_per_year

    if interval_minutes(interval) is None:
        return None
    return periods_per_year(interval, calendar)


class FeatureContext(BaseModel):
    """Auxiliary cross-entity data a feature function may need beyond its
    own symbol's OHLCV. Optional fields only — most features ignore this."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    benchmark_close: Optional[pd.Series] = None
    # The dataset's bar interval, so a feature that annualizes can scale by
    # the right constant instead of assuming daily bars. None means "not
    # supplied", which callers treat as daily for backward compatibility.
    interval: Optional[str] = None
    # The dataset's exchange calendar (an `exchange_calendars` name), which
    # is what makes an INTRADAY interval annualizable: bars per session
    # and sessions per year are read off it. None means an intraday
    # feature that annualizes still refuses, as before.
    calendar: Optional[str] = None


class FeatureDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    description: str
    fn: Callable[..., Any]
    default_params: Dict[str, Any] = Field(default_factory=dict)
    temporal_support: TemporalSupport
    scope: FeatureScope = FeatureScope.ENTITY
    requires: List[str] = Field(default_factory=list)
    lookback: int = Field(
        ..., ge=0, description="Bars of history consumed before the first valid output."
    )
    #: POINT_IN_TIME scope only: which record set the provider is asked
    #: for (a `data.temporal.FRAME_KINDS` entry) and which of its fields
    #: the transform reads. `default_params` must carry
    #: `max_staleness_days`, the oldest record a panel row may still read.
    frame_kind: Optional[str] = None
    fields: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _point_in_time_fields_agree_with_scope(self) -> "FeatureDefinition":
        if self.scope == FeatureScope.POINT_IN_TIME:
            if not self.frame_kind or not self.fields:
                raise ValueError(
                    f"feature {self.id!r}: a POINT_IN_TIME feature names the "
                    "record set it reads (frame_kind) and the fields it needs."
                )
            if "max_staleness_days" not in self.default_params:
                raise ValueError(
                    f"feature {self.id!r}: a POINT_IN_TIME feature declares "
                    "default_params['max_staleness_days'], the oldest record a "
                    "panel row may still read -- without a bound a feed that "
                    "stops updating supplies its last value forever."
                )
            if self.requires or self.lookback:
                raise ValueError(
                    f"feature {self.id!r}: a POINT_IN_TIME feature reads "
                    "records, not bars; `requires` and `lookback` do not apply."
                )
        elif self.frame_kind or self.fields:
            raise ValueError(
                f"feature {self.id!r}: frame_kind and fields are read for "
                "scope=POINT_IN_TIME only."
            )
        return self
