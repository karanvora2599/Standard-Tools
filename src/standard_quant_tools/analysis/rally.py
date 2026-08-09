"""
Rally detection: combines four already-proven signals from elsewhere in
this codebase (ADX trend strength, Donchian-style breakout, Hurst regime
classification, and trailing-return momentum) into one multi-confirmation
answer, rather than trusting any single indicator alone. ADX by itself
only says "trending," not which direction or how unusual; a raw
trailing-return threshold doesn't distinguish a genuine breakout from an
asset's normal drift; Hurst alone says "persistent," not "up." No new
indicator math is introduced here except the return z-score (see
`_return_zscore`'s docstring), which doesn't exist anywhere else in this
codebase.
"""

import logging
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd

from standard_quant_tools.analysis.hurst import hurst_exponent
from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators.trend import adx

logger = logging.getLogger(__name__)

# At least 3 of the 5 confirming signals below must agree for is_rally=True.
_RALLY_SCORE_THRESHOLD = 0.6
_RETURN_ZSCORE_THRESHOLD = 1.0
# Auto-tuning calibrates "strong trend" to what's actually strong FOR THIS
# SYMBOL (its own trailing ADX distribution) rather than one fixed number
# for every asset -- a chronically choppy stock and a chronically trending
# one each get a threshold calibrated to their own history. 60th percentile
# is the default: "stronger than 3 out of 5 of this symbol's own recent
# bars," a deliberately modest bar (not "strongest ADX this symbol has ever
# seen," which would almost never fire).
_AUTO_TUNE_DEFAULT_PERCENTILE = 60.0


def _return_zscore(close: pd.Series, lookback: int, zscore_window: int) -> pd.Series:
    """
    Z-score the trailing `lookback`-bar return against its own rolling
    `zscore_window`-bar history — volatility-normalized, so a low-vol
    utility stock and a high-vol growth stock are judged on the same
    relative-move scale rather than a fixed percentage threshold that
    would over-trigger on the latter and under-trigger on the former.
    Not implemented anywhere else in this codebase's metrics/ modules.
    """
    trailing_return = close.pct_change(periods=lookback)
    rolling_mean = trailing_return.rolling(zscore_window).mean()
    rolling_std = trailing_return.rolling(zscore_window).std()
    return (trailing_return - rolling_mean) / rolling_std


def detect_rally(
    df: pd.DataFrame,
    lookback: int = 20,
    zscore_window: int = 252,
    adx_period: int = 14,
    adx_threshold: float = 25.0,
    breakout_period: int = 20,
    hurst_method: Literal["dfa", "rs"] = "dfa",
    auto_tune_adx_threshold: bool = False,
    auto_tune_percentile: float = _AUTO_TUNE_DEFAULT_PERCENTILE,
) -> Dict[str, Any]:
    """
    Detect whether a symbol is currently rallying, via 5 independent
    confirming signals rather than any single indicator:

      1. unusual_positive_return — trailing `lookback`-bar return, z-scored
         against its own rolling `zscore_window`-bar history, > 1.0.
      2. strong_trend             — ADX(`adx_period`) > the effective ADX
                                     threshold (see `auto_tune_adx_threshold`).
      3. bullish_direction        — ADX's DI+ > DI-.
      4. trending_regime          — Hurst exponent regime == "trending"
                                     (H > 0.55), confirming genuine
                                     persistence rather than a random-walk
                                     blip that happens to look like a move
                                     today.
      5. new_high_breakout        — Close breaks above its own prior
                                     `breakout_period`-bar High (today's
                                     own bar excluded from the comparison
                                     window via .shift(1) — the same
                                     look-ahead-safe convention
                                     backtest/strategies.py's Donchian
                                     breakout uses).

    `rally_score` = fraction of the 5 signals that are true;
    `is_rally` = `rally_score >= 0.6` (at least 3 of 5).

    Parameters
    ----------
    df             : pd.DataFrame  OHLCV with Open/High/Low/Close columns.
    lookback       : Trailing-return window in bars (default 20, ~1 trading month).
    zscore_window  : Historical window the return z-score is measured
                     against (default 252, ~1 trading year).
    adx_period     : ADX lookback (default 14, the standard value).
    adx_threshold  : ADX level considered a "strong" trend (default 25).
                     Ignored when `auto_tune_adx_threshold=True`.
    breakout_period: Bars for the new-high breakout check (default 20).
    hurst_method   : "dfa" (default) or "rs", passed through to
                     hurst_exponent.
    auto_tune_adx_threshold : If True, ignore `adx_threshold` and instead
                     use the `auto_tune_percentile`-th percentile of this
                     symbol's OWN trailing ADX history over `df` as the
                     "strong trend" bar — a chronically choppy stock and a
                     chronically trending one each get a threshold
                     calibrated to their own history, rather than one fixed
                     number (25) applied to every asset regardless of its
                     normal ADX range. Default False: unchanged, exact
                     prior behavior (a fixed threshold every caller already
                     relied on).
    auto_tune_percentile : Percentile (0-100, exclusive) of the historical
                     ADX distribution used when `auto_tune_adx_threshold=True`.
                     Default 60 -- "stronger than most of this symbol's own
                     recent bars," a deliberately modest bar, not "the
                     strongest ADX this symbol has ever seen." Unused
                     otherwise.

    Returns
    -------
    dict with keys: is_rally, rally_score, trailing_return_pct,
    return_zscore, adx, di_plus, di_minus, trend_direction
    ("bullish"/"bearish"/"neutral"), hurst, regime, is_new_high, n_obs,
    adx_threshold_used (the actual threshold `strong_trend` was compared
    against -- equals `adx_threshold` unless auto-tuned), auto_tuned (bool,
    whether auto-tuning was actually applied).

    Raises
    ------
    ValidationError: required columns missing, `auto_tune_percentile` not
    strictly between 0 and 100, or fewer than `zscore_window + lookback`
    observations (not enough history for a meaningful z-score).
    """
    required = ("Open", "High", "Low", "Close")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(f"detect_rally: missing required columns {missing}")
    if auto_tune_adx_threshold and not (0.0 < auto_tune_percentile < 100.0):
        raise ValidationError(
            f"auto_tune_percentile must be strictly between 0 and 100, "
            f"got {auto_tune_percentile}"
        )

    df = df.dropna(subset=list(required))
    n = len(df)
    min_obs = zscore_window + lookback
    if n < min_obs:
        raise ValidationError(
            f"detect_rally needs at least {min_obs} observations "
            f"(zscore_window={zscore_window} + lookback={lookback}), got {n}"
        )

    close = df["Close"]

    trailing_return = close.pct_change(periods=lookback)
    zscore_series = _return_zscore(close, lookback, zscore_window)
    current_return = float(trailing_return.iloc[-1])
    current_zscore = (
        float(zscore_series.iloc[-1]) if not np.isnan(zscore_series.iloc[-1]) else 0.0
    )

    adx_df = adx(df["High"], df["Low"], close, period=adx_period)
    current_adx = float(adx_df["ADX"].iloc[-1])
    current_di_plus = float(adx_df["DI_Plus"].iloc[-1])
    current_di_minus = float(adx_df["DI_Minus"].iloc[-1])
    if current_di_plus > current_di_minus:
        trend_direction = "bullish"
    elif current_di_minus > current_di_plus:
        trend_direction = "bearish"
    else:
        trend_direction = "neutral"

    if auto_tune_adx_threshold:
        historical_adx = adx_df["ADX"].dropna()
        if historical_adx.empty:
            raise ValidationError(
                "detect_rally: auto_tune_adx_threshold=True but ADX has no "
                "non-NaN history to calibrate against (adx_period too "
                "large for the available data)."
            )
        effective_adx_threshold = float(
            np.percentile(historical_adx.to_numpy(), auto_tune_percentile)
        )
    else:
        effective_adx_threshold = adx_threshold

    returns = close.pct_change().dropna()
    hurst_result = hurst_exponent(returns, method=hurst_method)

    breakout_high = df["High"].rolling(breakout_period).max().shift(1)
    is_new_high = bool(close.iloc[-1] > breakout_high.iloc[-1])

    signals = {
        "unusual_positive_return": current_zscore > _RETURN_ZSCORE_THRESHOLD,
        "strong_trend": current_adx > effective_adx_threshold,
        "bullish_direction": trend_direction == "bullish",
        "trending_regime": hurst_result["regime"] == "trending",
        "new_high_breakout": is_new_high,
    }
    rally_score = sum(signals.values()) / len(signals)
    is_rally = rally_score >= _RALLY_SCORE_THRESHOLD

    logger.debug(
        "[rally] score=%.2f  is_rally=%s  adx_threshold_used=%.2f  auto_tuned=%s  signals=%s",
        rally_score,
        is_rally,
        effective_adx_threshold,
        auto_tune_adx_threshold,
        signals,
    )

    return {
        "is_rally": is_rally,
        "rally_score": rally_score,
        "trailing_return_pct": current_return,
        "return_zscore": current_zscore,
        "adx": current_adx,
        "di_plus": current_di_plus,
        "di_minus": current_di_minus,
        "trend_direction": trend_direction,
        "hurst": hurst_result["hurst"],
        "regime": hurst_result["regime"],
        "is_new_high": is_new_high,
        "n_obs": n,
        "adx_threshold_used": effective_adx_threshold,
        "auto_tuned": auto_tune_adx_threshold,
    }
