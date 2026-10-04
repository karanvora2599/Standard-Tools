"""Technical-indicator entity-scope features — thin wrappers over
indicators/momentum.py and indicators/trend.py, the exact primitives
analysis/rally.py::detect_rally already reuses rather than reimplementing."""

import pandas as pd

from standard_quant_tools.indicators.momentum import rsi as _rsi
from standard_quant_tools.indicators.momentum import (
    stochastic_oscillator as _stochastic,
)
from standard_quant_tools.indicators.trend import adx as _adx
from standard_quant_tools.indicators.trend import macd as _macd
from standard_quant_tools.indicators.trend import williams_r as _williams_r

from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import register_feature


def _technical_rsi(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 14
) -> pd.Series:
    return _rsi(ohlcv["Close"], period=period)


def _technical_adx(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 14
) -> pd.Series:
    return _adx(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], period=period)["ADX"]


def _technical_macd_histogram(
    ohlcv: pd.DataFrame,
    context: FeatureContext,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    return _macd(ohlcv["Close"], fast=fast, slow=slow, signal=signal)["Histogram"]


def _technical_macd_histogram_pct(
    ohlcv: pd.DataFrame,
    context: FeatureContext,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.Series:
    """The MACD histogram as a fraction of Close.

    The histogram is a difference of EMAs of price, so it is in price
    units: across a live 30-name panel its standard deviation ran from
    0.23 (KO) to 22.4 (GOOGL), a cross-sectional rank of it sorts names by
    price level as much as by momentum, and a 20:1 split changes its scale
    for good. Divided by Close it is comparable across names and returns
    to its scale once the smoothers have forgotten the pre-split bars.

    The denominator is guarded like risk.atr_pct's: a Close of zero or
    below gives NaN, which alignment drops, rather than an infinity that
    would refuse the whole panel."""
    close = ohlcv["Close"]
    histogram = _macd(close, fast=fast, slow=slow, signal=signal)["Histogram"]
    return histogram / close.where(close > 0)


def _technical_stochastic_k(
    ohlcv: pd.DataFrame, context: FeatureContext, k_period: int = 14, d_period: int = 3
) -> pd.Series:
    return _stochastic(
        ohlcv["High"],
        ohlcv["Low"],
        ohlcv["Close"],
        k_period=k_period,
        d_period=d_period,
    )["Stoch_K"]


def _technical_williams_r(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 14
) -> pd.Series:
    return _williams_r(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], period=period)


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
register_feature(
    FeatureDefinition(
        id="technical.macd_histogram",
        # The function above is never edited: its source is the
        # implementation hash a registered model is scored against. The
        # description is not part of that hash.
        description=(
            "MACD histogram (MACD line minus its signal line) — trend-momentum "
            "divergence. In price units, so its scale follows each name's price "
            "level and changes at a split; technical.macd_histogram_pct is the "
            "same quantity divided by Close."
        ),
        fn=_technical_macd_histogram,
        default_params={"fast": 12, "slow": 26, "signal": 9},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["Close"],
        lookback=26,
    )
)
register_feature(
    FeatureDefinition(
        id="technical.macd_histogram_pct",
        description=(
            "MACD histogram divided by Close — the histogram as a fraction of "
            "price, comparable across names. A split disturbs it only until the "
            "smoothers forget the bars before it (188 bars at the defaults), "
            "where the raw histogram's scale changes for good. NaN where Close "
            "is not positive."
        ),
        fn=_technical_macd_histogram_pct,
        default_params={"fast": 12, "slow": 26, "signal": 9},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["Close"],
        lookback=26,
    )
)
register_feature(
    FeatureDefinition(
        id="technical.stochastic_k",
        description="Stochastic oscillator %K — momentum vs. recent high-low range, 0-100.",
        fn=_technical_stochastic_k,
        default_params={"k_period": 14, "d_period": 3},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["High", "Low", "Close"],
        lookback=14,
    )
)
register_feature(
    FeatureDefinition(
        id="technical.williams_r",
        description="Williams %R momentum oscillator, -100 (oversold) to 0 (overbought).",
        fn=_technical_williams_r,
        default_params={"period": 14},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["High", "Low", "Close"],
        lookback=14,
    )
)
