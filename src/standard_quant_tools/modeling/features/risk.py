"""Volatility/beta entity-scope features. risk.rolling_beta is the one
feature in this phase that actually uses FeatureContext.benchmark_close
— dataset.builder is responsible for populating it from DatasetSpec.benchmark
before calling any entity-scope feature."""

import pandas as pd

from standard_quant_tools.analysis.regression import rolling_beta as _rolling_beta
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.volatility import bollinger_bands as _bollinger_bands
from standard_quant_tools.indicators.volatility import wilder_atr as _wilder_atr
from standard_quant_tools.metrics.volatility_estimators import (
    garman_klass_volatility as _garman_klass_volatility,
)
from standard_quant_tools.metrics.volatility_estimators import (
    parkinson_volatility as _parkinson_volatility,
)
from standard_quant_tools.metrics.volatility_estimators import (
    yang_zhang_volatility as _yang_zhang_volatility,
)

from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import register_feature


def _risk_realized_volatility(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 20
) -> pd.Series:
    return _yang_zhang_volatility(
        ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"], period=period
    )


def _risk_rolling_beta(
    ohlcv: pd.DataFrame, context: FeatureContext, window: int = 60
) -> pd.Series:
    if context.benchmark_close is None:
        raise ValidationError(
            "risk.rolling_beta requires FeatureContext.benchmark_close — "
            "dataset.builder must fetch DatasetSpec.benchmark before computing it."
        )
    asset_returns = ohlcv["Close"].pct_change().dropna()
    benchmark_returns = context.benchmark_close.pct_change().dropna()
    return _rolling_beta(asset_returns, benchmark_returns, window=window)["Rolling_Beta"]


def _risk_atr_pct(ohlcv: pd.DataFrame, context: FeatureContext, period: int = 14) -> pd.Series:
    """Raw ATR is a price level, not stationary across differently-priced
    stocks -- divide by Close, the same normalization idea
    risk.realized_volatility already gets for free by being return-scale."""
    atr = _wilder_atr(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], period=period)
    return atr / ohlcv["Close"]


def _risk_bollinger_pct_b(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 20, num_std: float = 2.0
) -> pd.Series:
    """%B: where Close sits within the bands, 0=lower band, 1=upper band --
    the raw band levels themselves aren't cross-sectionally comparable,
    same reasoning as risk.atr_pct."""
    bands = _bollinger_bands(ohlcv["Close"], period=period, num_std=num_std)
    band_width = bands["BB_Upper"] - bands["BB_Lower"]
    return (ohlcv["Close"] - bands["BB_Lower"]) / band_width


def _risk_parkinson_volatility(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 20
) -> pd.Series:
    return _parkinson_volatility(ohlcv["High"], ohlcv["Low"], period=period)


def _risk_garman_klass_volatility(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 20
) -> pd.Series:
    return _garman_klass_volatility(
        ohlcv["Open"], ohlcv["High"], ohlcv["Low"], ohlcv["Close"], period=period
    )


def _risk_rolling_drawdown(
    ohlcv: pd.DataFrame, context: FeatureContext, window: int = 252
) -> pd.Series:
    """Not a direct wrap of metrics.risk_metrics.drawdown_series -- that
    function measures drawdown from the ALL-TIME peak since the start of
    the series (cummax()), which would give a stale, uninformative peak
    for a bar deep inside a multi-year training window. This uses a
    bounded trailing peak instead, the same .rolling(period).max()
    convention market.new_high_breakout already uses."""
    rolling_peak = ohlcv["Close"].rolling(window).max()
    return (ohlcv["Close"] - rolling_peak) / rolling_peak


register_feature(
    FeatureDefinition(
        id="risk.realized_volatility",
        description="Yang-Zhang realized volatility (annualized).",
        fn=_risk_realized_volatility,
        default_params={"period": 20},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["Open", "High", "Low", "Close"],
        lookback=20,
    )
)
register_feature(
    FeatureDefinition(
        id="risk.rolling_beta",
        description="Rolling OLS beta of the entity's returns against DatasetSpec.benchmark.",
        fn=_risk_rolling_beta,
        default_params={"window": 60},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["Close"],
        lookback=60,
    )
)
register_feature(
    FeatureDefinition(
        id="risk.atr_pct",
        description="Wilder's Average True Range as a fraction of Close (normalized, "
        "comparable across differently-priced stocks).",
        fn=_risk_atr_pct,
        default_params={"period": 14},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["High", "Low", "Close"],
        lookback=14,
    )
)
register_feature(
    FeatureDefinition(
        id="risk.bollinger_pct_b",
        description="Position of Close within its Bollinger Bands: 0=lower band, 1=upper band.",
        fn=_risk_bollinger_pct_b,
        default_params={"period": 20, "num_std": 2.0},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["Close"],
        lookback=20,
    )
)
register_feature(
    FeatureDefinition(
        id="risk.parkinson_volatility",
        description="Parkinson high-low range realized volatility (annualized).",
        fn=_risk_parkinson_volatility,
        default_params={"period": 20},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["High", "Low"],
        lookback=20,
    )
)
register_feature(
    FeatureDefinition(
        id="risk.garman_klass_volatility",
        description="Garman-Klass OHLC realized volatility (annualized).",
        fn=_risk_garman_klass_volatility,
        default_params={"period": 20},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["Open", "High", "Low", "Close"],
        lookback=20,
    )
)
register_feature(
    FeatureDefinition(
        id="risk.rolling_drawdown",
        description="Drawdown of Close from its trailing `window`-bar peak (0 at a new high, "
        "negative otherwise).",
        fn=_risk_rolling_drawdown,
        default_params={"window": 252},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["Close"],
        lookback=252,
    )
)
