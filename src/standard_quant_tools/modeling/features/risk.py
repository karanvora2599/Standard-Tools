"""Volatility/beta entity-scope features. risk.rolling_beta is the one
feature in this phase that actually uses FeatureContext.benchmark_close
— dataset.builder is responsible for populating it from DatasetSpec.benchmark
before calling any entity-scope feature."""

import pandas as pd

from standard_quant_tools.analysis.regression import rolling_beta as _rolling_beta
from standard_quant_tools.error import ValidationError
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
