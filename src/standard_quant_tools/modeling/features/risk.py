"""Volatility/beta entity-scope features. risk.rolling_beta is the one
feature in this phase that actually uses FeatureContext.benchmark_close
— dataset.builder is responsible for populating it from DatasetSpec.benchmark
before calling any entity-scope feature."""

import pandas as pd

from standard_quant_tools.analysis.regression import rolling_beta as _rolling_beta
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.volatility import (
    bollinger_bands as _bollinger_bands,
)
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

from .base import (
    FeatureContext,
    FeatureDefinition,
    FeatureScope,
    TemporalSupport,
    periods_per_year_for_interval,
)
from .registry import register_feature


def _annualization(context: FeatureContext, feature_id: str) -> int:
    """
    Bars per year for the dataset's interval.

    An "annualized" volatility scaled by sqrt(252) is only annualized when
    the bars ARE daily; at any other interval it is wrong by a fixed
    multiplicative factor while still looking like a volatility. Rather
    than emit that, features that annualize resolve the constant from the
    dataset interval and REJECT the intervals where no constant exists
    without an exchange calendar (every intraday one -- bars per year at
    "1h" is 6.5 hours/day for US equities, 8 for many European venues,
    ~24 for crypto).

    A missing interval means the caller predates this field and is treated
    as daily, which is what it was before.
    """
    if context is None or context.interval is None:
        return 252
    resolved = periods_per_year_for_interval(context.interval)
    if resolved is None:
        raise ValidationError(
            f"{feature_id}: cannot annualize at interval={context.interval!r}. "
            "Bars per year for an intraday interval depends on the venue's "
            "session length, which this package has no exchange calendar to "
            "resolve -- assuming one would make this 'annualized' volatility "
            "wrong by a fixed factor for every other market while still "
            "looking precise. Use a daily-or-coarser interval for this "
            "feature, or compute an explicitly per-bar volatility instead."
        )
    return resolved


def _risk_realized_volatility(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 20
) -> pd.Series:
    return _yang_zhang_volatility(
        ohlcv["Open"],
        ohlcv["High"],
        ohlcv["Low"],
        ohlcv["Close"],
        period=period,
        periods_per_year=_annualization(context, "risk.realized_volatility"),
    )


def _risk_rolling_beta(
    ohlcv: pd.DataFrame, context: FeatureContext, window: int = 60
) -> pd.Series:
    if context.benchmark_close is None:
        raise ValidationError(
            "risk.rolling_beta requires FeatureContext.benchmark_close — "
            "dataset.builder must fetch DatasetSpec.benchmark before computing it."
        )
    asset_returns = ohlcv["Close"].pct_change(fill_method=None).dropna()
    benchmark_returns = context.benchmark_close.pct_change(fill_method=None).dropna()
    return _rolling_beta(asset_returns, benchmark_returns, window=window)[
        "Rolling_Beta"
    ]


def _risk_atr_pct(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 14
) -> pd.Series:
    """Raw ATR is a price level, not stationary across differently-priced
    stocks -- divide by Close, the same normalization idea
    risk.realized_volatility already gets for free by being return-scale.

    The denominator is guarded because the division was unconditional: a
    single Close of exactly 0.0 (a bad print, a delisted stub, a provider
    filling a gap with zero) produced +/-inf, and inf does not merely
    corrupt that one row -- build_dataset's finite-value guard rejects the
    ENTIRE panel, so one bad bar in one symbol failed the whole build with
    an error pointing at the feature rather than the data. Exactly the
    failure mode volume.obv_roc was already fixed for. A non-positive price
    has no meaningful volatility ratio, so NaN is the right answer, and
    alignment drops it like any other missing value."""
    close = ohlcv["Close"]
    atr = _wilder_atr(ohlcv["High"], ohlcv["Low"], close, period=period)
    return atr_pct_from_atr(atr, close)


def atr_pct_from_atr(atr: pd.Series, close: pd.Series) -> pd.Series:
    """The normalization step of risk.atr_pct, split out so the panel fast
    path in dataset/panel_features.py applies the SAME guarded division to
    a batch-computed ATR. Two copies of this would be two chances for the
    fast path to drift away from the per-entity path silently."""
    return atr / close.where(close > 0)


def _risk_bollinger_pct_b(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 20, num_std: float = 2.0
) -> pd.Series:
    """%B: where Close sits within the bands, 0=lower band, 1=upper band --
    the raw band levels themselves aren't cross-sectionally comparable,
    same reasoning as risk.atr_pct.

    A window of perfectly flat prices (a halted symbol, a stale-quoted
    illiquid name, a synthetic constant series) collapses both bands onto
    the mean, making %B a 0/0 that came out NaN and was silently dropped by
    alignment -- so a halt removed rows rather than describing one. When the
    window IS flat, Close equals the moving average exactly, and 0.5 is the
    middle band, not a fallback: it is the value %B is defined to take
    there. Note this is the exactly-degenerate case only; a near-flat window
    followed by a jump is well behaved, because the jump enters the standard
    deviation that scales it (%B peaks around 1.56, not at infinity).

    Warm-up stays NaN. `band_width.notna()` distinguishes "the bands
    collapsed" from "there are not yet `period` bars to compute them" --
    conflating the two would fabricate a 0.5 for rows with no window at all,
    which is the mistake market.new_high_breakout was making with its own
    warm-up. Same guarded-denominator shape as the degenerate-window
    handling in stochastic_oscillator and spread_zscore."""
    bands = _bollinger_bands(ohlcv["Close"], period=period, num_std=num_std)
    return pct_b_from_bands(ohlcv["Close"], bands["BB_Upper"], bands["BB_Lower"])


def pct_b_from_bands(close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
    """The %B step of risk.bollinger_pct_b, including the collapsed-band
    rule, split out for the panel fast path. See _risk_bollinger_pct_b for
    why a flat window is 0.5 rather than NaN."""
    band_width = upper - lower
    pct_b = (close - lower) / band_width.where(band_width > 0)
    return pct_b.where(band_width.isna() | (band_width > 0), 0.5)


def _risk_parkinson_volatility(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 20
) -> pd.Series:
    return _parkinson_volatility(
        ohlcv["High"],
        ohlcv["Low"],
        period=period,
        periods_per_year=_annualization(context, "risk.parkinson_volatility"),
    )


def _risk_garman_klass_volatility(
    ohlcv: pd.DataFrame, context: FeatureContext, period: int = 20
) -> pd.Series:
    return _garman_klass_volatility(
        ohlcv["Open"],
        ohlcv["High"],
        ohlcv["Low"],
        ohlcv["Close"],
        period=period,
        periods_per_year=_annualization(context, "risk.garman_klass_volatility"),
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
        description="Position of Close within its Bollinger Bands: 0=lower band, "
        "1=upper band, 0.5 when a flat window collapses the bands onto the mean.",
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
