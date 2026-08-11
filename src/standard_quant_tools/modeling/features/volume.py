"""Volume-based entity-scope features — the one group of features that
needs the OHLCV panel's Volume column (every other feature file in this
package only needs Open/High/Low/Close)."""

import pandas as pd

from standard_quant_tools.indicators.volume import mfi as _mfi
from standard_quant_tools.indicators.volume import obv as _obv
from standard_quant_tools.indicators.volume import vwap as _vwap

from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import register_feature


def _volume_mfi(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 14
) -> pd.Series:
    return _mfi(
        ohlcv["High"], ohlcv["Low"], ohlcv["Close"], ohlcv["Volume"], period=period
    )


def _volume_obv_roc(
    ohlcv: pd.DataFrame, context: FeatureContext, lookback: int = 20
) -> pd.Series:
    """
    Raw OBV is unbounded and cumulative (grows without limit over a long
    history), not stationary -- wrapped as a normalized change over
    `lookback` bars.

    NOT `obv.pct_change(lookback)`, which is unusable on real data: OBV is
    a cumulative sum seeded at exactly 0.0 (the first bar has no prior
    close, so its direction is 0), so the first `lookback` valid rows all
    divide by OBV[0] == 0 and come out +/-inf. Dividing by a cumulative
    total is wrong in general anyway -- OBV crosses zero freely, so the
    denominator is arbitrarily near zero at any point in the series, not
    just at the start, and the ratio explodes there too.

    Normalized instead by the trailing volume actually transacted over the
    same window, which is strictly positive whenever any trading occurred
    and is the natural scale for a volume-flow delta: the result reads as
    "what fraction of the last `lookback` bars' volume was net directional
    flow", bounded in [-1, 1]. A window with zero total volume (a fully
    halted symbol) has no defined flow ratio and yields NaN, which
    alignment drops -- rather than an inf that would fail the dataset's
    finite-value guard and reject the entire panel.
    """
    obv_series = _obv(ohlcv["Close"], ohlcv["Volume"])
    obv_change = obv_series.diff(periods=lookback)
    volume_traded = ohlcv["Volume"].rolling(lookback).sum()
    return obv_change / volume_traded.where(volume_traded > 0)


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
