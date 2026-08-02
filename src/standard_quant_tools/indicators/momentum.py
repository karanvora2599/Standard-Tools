import logging
from typing import Any

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import require_finite_array, validate_series

logger = logging.getLogger(__name__)

_cpp_core: Any = None
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    # Dummy decorator if numba is missing
    def njit(func):
        return func


@njit
def _rsi_numba(prices: np.ndarray, period: int) -> np.ndarray:
    n = len(prices)
    rsi = np.full(n, np.nan)

    if n <= period:
        return rsi

    delta = prices[1:] - prices[:-1]

    # Initial Average (SMA)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder's Smoothing
    for i in range(period + 1, n):
        change = delta[i - 1]
        gain = change if change > 0 else 0.0
        loss = -change if change < 0 else 0.0

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@validate_series()
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    Uses C++ fast path when built, then Numba JIT, then pure Python fallback.
    All three paths use Wilder's smoothing (SMA seed, then alpha=1/period).
    """
    if series.empty:
        return pd.Series(dtype=float)
    if period <= 0:
        raise ValidationError(f"period must be > 0, got {period}")

    values: np.ndarray = np.asarray(series.values, dtype=np.float64)
    require_finite_array(values, "prices", "rsi")
    path = (
        "C++"
        if (HAS_CPP and _cpp_core is not None)
        else ("numba" if HAS_NUMBA else "python")
    )
    logger.debug("[rsi] period=%d  bars=%d  path=%s", period, len(values), path)

    if HAS_CPP and _cpp_core is not None:
        rsi_vals = _cpp_core.rsi(values, period)
    else:
        rsi_vals = _rsi_numba(values, period)

    result = pd.Series(rsi_vals, index=series.index)
    valid = result.dropna()
    if not valid.empty:
        logger.debug(
            "[rsi] last=%.2f  min=%.2f  max=%.2f",
            float(valid.iloc[-1]),
            float(valid.min()),
            float(valid.max()),
        )
    return result


@validate_series()
def stochastic_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.

    Uses C++ fused sliding min+max path when available (5-15× faster than two
    pandas rolling passes).  Falls back to pandas otherwise.
    """
    if k_period <= 0:
        raise ValidationError(f"k_period must be > 0, got {k_period}")
    if d_period <= 0:
        raise ValidationError(f"d_period must be > 0, got {d_period}")

    logger.debug(
        "[stochastic] k_period=%d  d_period=%d  bars=%d  path=%s",
        k_period,
        d_period,
        len(close),
        "C++" if (HAS_CPP and _cpp_core is not None) else "pandas",
    )

    # Checked once, unconditionally, BEFORE the C++ try/except below --
    # that except catches Exception broadly (to fall back to pandas on any
    # C++ failure), which would otherwise silently swallow a
    # ValidationError raised inside the try block and mask bad input
    # behind a confusing fallback instead of rejecting it.
    require_finite_array(
        high.to_numpy(dtype=np.float64), "high", "stochastic_oscillator"
    )
    require_finite_array(low.to_numpy(dtype=np.float64), "low", "stochastic_oscillator")
    require_finite_array(
        close.to_numpy(dtype=np.float64), "close", "stochastic_oscillator"
    )

    # ── C++ fast path ─────────────────────────────────────────────────────────
    if HAS_CPP and _cpp_core is not None:
        try:
            h_arr = high.to_numpy(dtype=np.float64)
            l_arr = low.to_numpy(dtype=np.float64)
            c_arr = close.to_numpy(dtype=np.float64)
            out = _cpp_core.stochastic_oscillator(
                h_arr, l_arr, c_arr, k_period, d_period
            )
            k = pd.Series(out[:, 0], index=close.index)
            d = pd.Series(out[:, 1], index=close.index)
            result = pd.DataFrame({"Stoch_K": k, "Stoch_D": d})
            valid_k = k.dropna()
            if not valid_k.empty:
                logger.debug(
                    "[stochastic] K last=%.2f  D last=%.2f",
                    float(valid_k.iloc[-1]),
                    float(d.dropna().iloc[-1]),
                )
            return result
        except Exception as exc:
            logger.warning("[stochastic] C++ failed (%s) — using pandas", exc)

    # ── Pandas fallback ───────────────────────────────────────────────────────
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()

    result = pd.DataFrame({"Stoch_K": k, "Stoch_D": d})
    valid_k = k.dropna()
    if not valid_k.empty:
        logger.debug(
            "[stochastic] K last=%.2f  D last=%.2f",
            float(valid_k.iloc[-1]),
            float(d.dropna().iloc[-1]),
        )
    return result
