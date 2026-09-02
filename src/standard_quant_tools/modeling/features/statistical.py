"""Regime-classification entity-scope feature — wraps analysis.hurst.rolling_hurst
(C++-accelerated), the per-bar variant of the same Hurst exponent
analysis/rally.py::detect_rally uses for its trending_regime signal."""

import pandas as pd

from standard_quant_tools.analysis.hurst import rolling_hurst as _rolling_hurst

from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import register_feature


def _statistical_hurst(
    ohlcv: pd.DataFrame, context: FeatureContext, window: int = 200, method: str = "dfa"
) -> pd.Series:
    returns = ohlcv["Close"].pct_change(fill_method=None).dropna()
    return _rolling_hurst(returns, window=window, method=method)


register_feature(
    FeatureDefinition(
        id="statistical.hurst",
        description="Rolling Hurst exponent — >0.55 trending, <0.45 mean-reverting.",
        fn=_statistical_hurst,
        default_params={"window": 200, "method": "dfa"},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.ENTITY,
        requires=["Close"],
        lookback=200,
    )
)
