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
from pydantic import BaseModel, ConfigDict, Field


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


# Column names the long panel builds itself. A feature's output column --
# its alias, or its id when it has none -- must not collide with any of
# them. Defined here rather than inline at each check so the alias path and
# the feature-id path cannot drift apart: the alias path was validated and
# the id path was not, which let a custom feature registered as id="target"
# produce a column that shadowed the panel's supervised target.
RESERVED_PANEL_COLUMNS = frozenset({"date", "entity", "target", "label_end_date"})


# Bars per year, by interval, for annualizing a per-bar volatility.
#
# Only intervals whose constant is unambiguous are listed. Daily, weekly and
# monthly are calendar-derived and need no assumption about session length.
# INTRADAY IS DELIBERATELY ABSENT: bars-per-year at "1h" depends on how many
# trading hours the venue is open (6.5 for US equities, 8 for many European
# venues, ~24 for crypto), and this package has no exchange calendar to
# resolve that from. Picking one silently would make an "annualized"
# volatility wrong by a fixed multiplicative factor for every other market —
# a number that looks precise and is not.
_PERIODS_PER_YEAR = {
    "1d": 252,
    "5d": 52,
    "1wk": 52,
    "1mo": 12,
    "3mo": 4,
}


def periods_per_year_for_interval(interval: str) -> Optional[int]:
    """Bars per year for `interval`, or None when it cannot be determined
    without an exchange calendar (every intraday interval)."""
    return _PERIODS_PER_YEAR.get(str(interval).strip())


class FeatureContext(BaseModel):
    """Auxiliary cross-entity data a feature function may need beyond its
    own symbol's OHLCV. Optional fields only — most features ignore this."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    benchmark_close: Optional[pd.Series] = None
    # The dataset's bar interval, so a feature that annualizes can scale by
    # the right constant instead of assuming daily bars. None means "not
    # supplied", which callers treat as daily for backward compatibility.
    interval: Optional[str] = None


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
