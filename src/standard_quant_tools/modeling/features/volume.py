"""Volume-based entity-scope features — the one group of features that
needs the OHLCV panel's Volume column (every other feature file in this
package only needs Open/High/Low/Close)."""

import pandas as pd

from standard_quant_tools.indicators.volume import mfi as _mfi
from standard_quant_tools.indicators.volume import obv as _obv
from standard_quant_tools.indicators.volume import vwap as _vwap

from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import register_feature


def _volume_mfi(ohlcv: pd.DataFrame, context: FeatureContext, period: int = 14) -> pd.Series:
    return _mfi(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], ohlcv["Volume"], period=period)


def _volume_obv_roc(
    ohlcv: pd.DataFrame, context: FeatureContext, lookback: int = 20
) -> pd.Series:
    """Raw OBV is unbounded and cumulative (grows without limit over a
    long history), not stationary -- wrapped as a rate of change over
    `lookback` bars, the same trailing-window convention market.momentum
    already uses for Close."""
    obv_series = _obv(ohlcv["Close"], ohlcv["Volume"])
    return obv_series.pct_change(periods=lookback)


def _volume_vwap_deviation(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 20
) -> pd.Series:
    """Raw VWAP is a price level, not comparable across differently-priced
    stocks -- normalized as (Close - VWAP) / VWAP, the same "divide by a
    price level to make it stationary" convention risk.atr_pct uses."""
    vwap_series = _vwap(
        ohlcv["High"], ohlcv["Low"], ohlcv["Close"], ohlcv["Volume"], period=period
    )
    return (ohlcv["Close"] - vwap_series) / vwap_series


register_feature(
    FeatureDefinition(
        id="volume.mfi",
        description="Money Flow Index — volume-weighted RSI, 0-100.",
        fn=_volume_mfi,
        default_params={"period": 14},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["High", "Low", "Close", "Volume"],
        lookback=14,
    )
)
register_feature(
    FeatureDefinition(
        id="volume.obv_roc",
        description="Rate of change of On-Balance Volume over `lookback` bars.",
        fn=_volume_obv_roc,
        default_params={"lookback": 20},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["Close", "Volume"],
        lookback=20,
    )
)
register_feature(
    FeatureDefinition(
        id="volume.vwap_deviation",
        description="(Close - VWAP) / VWAP over a trailing `period`-bar window.",
        fn=_volume_vwap_deviation,
        default_params={"period": 20},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["High", "Low", "Close", "Volume"],
        lookback=20,
    )
)
