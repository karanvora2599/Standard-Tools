"""Technical-indicator entity-scope features — thin wrappers over
indicators/momentum.py and indicators/trend.py, the exact primitives
analysis/rally.py::detect_rally already reuses rather than reimplementing."""

import pandas as pd

from standard_quant_tools.indicators.momentum import rsi as _rsi
from standard_quant_tools.indicators.trend import adx as _adx

from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import register_feature


def _technical_rsi(ohlcv: pd.DataFrame, context: FeatureContext, period: int = 14) -> pd.Series:
    return _rsi(ohlcv["Close"], period=period)


def _technical_adx(ohlcv: pd.DataFrame, context: FeatureContext, period: int = 14) -> pd.Series:
    return _adx(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], period=period)["ADX"]


register_feature(
    FeatureDefinition(
        id="technical.rsi",
        description="Relative Strength Index — momentum oscillator, 0-100.",
        fn=_technical_rsi,
        default_params={"period": 14},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["Close"],
        lookback=14,
    )
)
register_feature(
    FeatureDefinition(
        id="technical.adx",
        description="Average Directional Index — trend strength, unsigned.",
        fn=_technical_adx,
        default_params={"period": 14},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["High", "Low", "Close"],
        lookback=14,
    )
)
