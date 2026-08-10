"""Price/return-derived entity-scope features — the same trailing-return
and look-ahead-safe breakout conventions analysis/rally.py::detect_rally
and backtest/strategies.py already use (today's own bar excluded from its
own breakout comparison window via .shift(1))."""

import pandas as pd

from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import register_feature


def _market_momentum(ohlcv: pd.DataFrame, context: FeatureContext, lookback: int = 20) -> pd.Series:
    return ohlcv["Close"].pct_change(periods=lookback)


def _market_new_high_breakout(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 20
) -> pd.Series:
    breakout_high = ohlcv["High"].rolling(period).max().shift(1)
    return (ohlcv["Close"] > breakout_high).astype(float)


register_feature(
    FeatureDefinition(
        id="market.momentum",
        description="Trailing close-to-close return over `lookback` bars.",
        fn=_market_momentum,
        default_params={"lookback": 20},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["Close"],
        lookback=20,
    )
)
register_feature(
    FeatureDefinition(
        id="market.new_high_breakout",
        description="1.0 if Close breaks above the prior `period`-bar High "
        "(today's own bar excluded), else 0.0.",
        fn=_market_new_high_breakout,
        default_params={"period": 20},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["High", "Close"],
        lookback=20,
    )
)
