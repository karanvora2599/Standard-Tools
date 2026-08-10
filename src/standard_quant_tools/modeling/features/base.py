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
    # Price/volume-derived — safe to compute anywhere in a historical
    # training window since the source data itself isn't revised.
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


class FeatureContext(BaseModel):
    """Auxiliary cross-entity data a feature function may need beyond its
    own symbol's OHLCV. Optional fields only — most features ignore this."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    benchmark_close: Optional[pd.Series] = None


class FeatureDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    description: str
    fn: Callable[..., Any]
    default_params: Dict[str, Any] = Field(default_factory=dict)
    temporal_support: TemporalSupport
    scope: FeatureScope = FeatureScope.ENTITY
    requires: List[str] = Field(default_factory=list)
    lookback: int = Field(..., ge=0, description="Bars of history consumed before the first valid output.")
