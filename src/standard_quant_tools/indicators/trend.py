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

    def njit(func):
        return func


# ──────────────────────────────────────────────
# Existing indicators
# ──────────────────────────────────────────────


@validate_series()
def sma(series: pd.Series, period: int = 14) -> pd.Series:
    """Simple Moving Average."""
    if period <= 0:
        raise ValidationError(f"period must be > 0, got {period}")
    return series.rolling(window=period).mean()


@validate_series()
def ema(series: pd.Series, period: int = 14) -> pd.Series:
    """Exponential Moving Average."""
    if period <= 0:
        raise ValidationError(f"period must be > 0, got {period}")
    return series.ewm(span=period, adjust=False).mean()


@validate_series()
def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    MACD: Moving Average Convergence Divergence.
    Returns DataFrame with columns ['MACD', 'Signal', 'Histogram'].
    """
    for name, value in (("fast", fast), ("slow", slow), ("signal", signal)):
        if value <= 0:
            raise ValidationError(f"{name} must be > 0, got {value}")
    if fast >= slow:
        raise ValidationError(
            f"fast ({fast}) must be < slow ({slow}) — MACD is the fast EMA "
            "minus the slow one, so an inverted pair silently produces a "
            "sign-flipped indicator rather than an error."
        )
    logger.debug(
        "[macd] fast=%d  slow=%d  signal=%d  bars=%d", fast, slow, signal, len(series)
    )
    exp1 = ema(series, fast)
    exp2 = ema(series, slow)
    macd_line = exp1 - exp2
    signal_line = ema(macd_line, signal)
    result = pd.DataFrame(
        {
            "MACD": macd_line,
            "Signal": signal_line,
            "Histogram": macd_line - signal_line,
        }
    )
    valid = result.dropna()
    if not valid.empty:
        logger.debug(
            "[macd] last MACD=%.4f  Signal=%.4f  Hist=%.4f",
            float(valid["MACD"].iloc[-1]),
            float(valid["Signal"].iloc[-1]),
            float(valid["Histogram"].iloc[-1]),
        )
    return result


# ──────────────────────────────────────────────
# ADX — Average Directional Index (Numba JIT)
# ──────────────────────────────────────────────


@njit
def _adx_numba(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> np.ndarray:
    """
    Wilder's ADX using the same smoothing as RSI.
    Returns a (n, 3) array: [:, 0] = DI+, [:, 1] = DI-, [:, 2] = ADX.
    """
    n = len(close)
    result = np.full((n, 3), np.nan)

    # Wilder's seed needs `period` bars of DM/TR before the first DI can be
    # written at row `period`. With n <= period every write below
    # (result[period], dx_vals[period], result[2*period-1]) indexes past the
    # end of an n-row array -- and @njit compiles without bounds checking, so
    # that is an out-of-bounds heap write, not an IndexError. Return the
    # all-NaN warm-up result before any of them, matching what the C++ kernel
    # already does for this case.
    if n <= period:
        return result

    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    tr = np.zeros(n)

    # Step 1: raw DM and TR per bar
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]

        dm_plus[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        dm_minus[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Step 2: Wilder's initial sums (first `period` bars)
    atr_s = np.sum(tr[1 : period + 1])
    dmp_s = np.sum(dm_plus[1 : period + 1])
    dmm_s = np.sum(dm_minus[1 : period + 1])

    di_plus_0 = 100.0 * dmp_s / atr_s if atr_s != 0 else 0.0
    di_minus_0 = 100.0 * dmm_s / atr_s if atr_s != 0 else 0.0
    result[period, 0] = di_plus_0
    result[period, 1] = di_minus_0

    di_sum = di_plus_0 + di_minus_0
    dx_0 = 100.0 * abs(di_plus_0 - di_minus_0) / di_sum if di_sum != 0 else 0.0

    # Step 3: Wilder's smooth forward
    dx_vals = np.zeros(n)
    dx_vals[period] = dx_0

    for i in range(period + 1, n):
        atr_s = atr_s - (atr_s / period) + tr[i]
        dmp_s = dmp_s - (dmp_s / period) + dm_plus[i]
        dmm_s = dmm_s - (dmm_s / period) + dm_minus[i]

        di_p = 100.0 * dmp_s / atr_s if atr_s != 0 else 0.0
        di_m = 100.0 * dmm_s / atr_s if atr_s != 0 else 0.0
        result[i, 0] = di_p
        result[i, 1] = di_m

        di_sum = di_p + di_m
        dx_vals[i] = 100.0 * abs(di_p - di_m) / di_sum if di_sum != 0 else 0.0

    # Step 4: ADX = Wilder's smooth of DX (needs `period` DX values to initialise)
    adx_start = 2 * period - 1
    if adx_start < n:
        result[adx_start, 2] = np.mean(dx_vals[period : adx_start + 1])
        for i in range(adx_start + 1, n):
            result[i, 2] = (result[i - 1, 2] * (period - 1) + dx_vals[i]) / period

    return result


@validate_series()
def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Average Directional Index (ADX) with DI+ and DI-.
    ADX > 25 indicates a strong trend; direction determined by DI+/DI-.
    Uses C++ fast path when built, then Numba JIT, then pure Python fallback.
    Returns DataFrame with columns ['DI_Plus', 'DI_Minus', 'ADX'].
    """
    if period <= 0:
        raise ValidationError(f"period must be > 0, got {period}")
    # The kernels index high[i]/low[i] against a result array sized from
    # close, so a shorter high/low is an out-of-bounds read under @njit (no
    # bounds checking) rather than an IndexError. Reject up front.
    if not (len(high) == len(low) == len(close)):
        raise ValidationError(
            "adx: high/low/close must all be the same length, got "
            f"{len(high)}/{len(low)}/{len(close)}"
        )

    path = (
        "C++"
        if (HAS_CPP and _cpp_core is not None)
        else ("numba" if HAS_NUMBA else "python")
    )
    logger.debug("[adx] period=%d  bars=%d  path=%s", period, len(close), path)
    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)
    c = close.to_numpy(dtype=np.float64)
    require_finite_array(h, "high", "adx")
    require_finite_array(l, "low", "adx")
    require_finite_array(c, "close", "adx")

    if HAS_CPP and _cpp_core is not None:
        raw = _cpp_core.adx(h, l, c, period)
    else:
        raw = _adx_numba(h, l, c, period)

    result = pd.DataFrame(
        {"DI_Plus": raw[:, 0], "DI_Minus": raw[:, 1], "ADX": raw[:, 2]},
        index=close.index,
    )
    valid = result.dropna()
    if not valid.empty:
        logger.debug(
            "[adx] last DI+=%.2f  DI-=%.2f  ADX=%.2f  trend=%s",
            float(valid["DI_Plus"].iloc[-1]),
            float(valid["DI_Minus"].iloc[-1]),
            float(valid["ADX"].iloc[-1]),
            "strong" if float(valid["ADX"].iloc[-1]) > 25 else "weak",
        )
    return result


# ──────────────────────────────────────────────
# Parabolic SAR (Numba JIT)
# ──────────────────────────────────────────────


@njit
def _psar_numba(
    high: np.ndarray,
    low: np.ndarray,
    af_start: float,
    af_step: float,
    af_max: float,
) -> np.ndarray:
    """
    Parabolic SAR state machine.
    Returns a (n, 2) array: [:, 0] = SAR values, [:, 1] = trend (1=rising, -1=falling).
    """
    n = len(high)
    result = np.full((n, 2), np.nan)

    # The bootstrap below reads low[0]/high[0] unconditionally. @njit compiles
    # without bounds checking, so on an empty input that is an out-of-bounds
    # read rather than an IndexError -- return the (empty) result first.
    if n == 0:
        return result

    # Bootstrap: assume rising trend from bar 0
    sar = low[0]
    ep = high[0]
    af = af_start
    is_rising = True

    result[0, 0] = sar
    result[0, 1] = 1.0

    for i in range(1, n):
        prev_sar = sar

        if is_rising:
            sar = prev_sar + af * (ep - prev_sar)
            # SAR must be below the two prior lows
            sar = min(sar, low[i - 1])
            if i >= 2:
                sar = min(sar, low[i - 2])

            if high[i] > ep:
                ep = high[i]
                af = min(af + af_step, af_max)

            if low[i] < sar:
                # Bearish reversal
                is_rising = False
                sar = ep
                ep = low[i]
                af = af_start
        else:
            sar = prev_sar - af * (prev_sar - ep)
            # SAR must be above the two prior highs
            sar = max(sar, high[i - 1])
            if i >= 2:
                sar = max(sar, high[i - 2])

            if low[i] < ep:
                ep = low[i]
                af = min(af + af_step, af_max)

            if high[i] > sar:
                # Bullish reversal
                is_rising = True
                sar = ep
                ep = high[i]
                af = af_start

        result[i, 0] = sar
        result[i, 1] = 1.0 if is_rising else -1.0

    return result


@validate_series()
def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> pd.DataFrame:
    """
    Parabolic SAR — a dynamic trailing stop / trend-following indicator.
    Uses C++ fast path when built, then Numba JIT, then pure Python fallback.

    Returns DataFrame with:
        'SAR'   : Stop-and-reverse price level.
        'Trend' : 1 = rising (long), -1 = falling (short).
    """
    for name, value in (
        ("af_start", af_start),
        ("af_step", af_step),
        ("af_max", af_max),
    ):
        if not np.isfinite(value):
            raise ValidationError(f"{name} must be finite, got {value!r}")
    if af_start <= 0.0:
        raise ValidationError(f"af_start must be > 0, got {af_start!r}")
    if af_step < 0.0:
        raise ValidationError(f"af_step must be >= 0, got {af_step!r}")
    if af_max <= 0.0:
        raise ValidationError(f"af_max must be > 0, got {af_max!r}")
    if af_max < af_start:
        raise ValidationError(f"af_max ({af_max!r}) must be >= af_start ({af_start!r})")
    # Same out-of-bounds-read rationale as adx(): the state machine indexes
    # low[i] against a result array sized from high.
    if len(high) != len(low):
        raise ValidationError(
            f"parabolic_sar: high/low must be the same length, got "
            f"{len(high)}/{len(low)}"
        )

    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)
    # Consistent with adx()/rsi(): NaN/Inf must be rejected at the API
    # boundary rather than silently producing a garbage SAR path (the state
    # machine's comparisons are all false against NaN, so it would carry the
    # bootstrap value forward for the whole series and look like real output).
    require_finite_array(h, "high", "parabolic_sar")
    require_finite_array(l, "low", "parabolic_sar")

    if HAS_CPP and _cpp_core is not None:
        raw = _cpp_core.parabolic_sar(h, l, af_start, af_step, af_max)
    else:
        raw = _psar_numba(h, l, af_start, af_step, af_max)

    return pd.DataFrame(
        {"SAR": raw[:, 0], "Trend": raw[:, 1]},
        index=high.index,
    )


# ──────────────────────────────────────────────
# Williams %R
# ──────────────────────────────────────────────


@validate_series()
def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Williams %R — momentum oscillator, -100 to 0.
    Below -80 is oversold; above -20 is overbought.
    Vectorized rolling window; no Numba needed.

    A zero-range window (flat prices across the whole lookback) yields NaN —
    %R is a position within the range, which is undefined when there is no
    range — rather than an unguarded 0/0.
    """
    if period <= 0:
        raise ValidationError(f"period must be > 0, got {period}")
    if not (len(high) == len(low) == len(close)):
        raise ValidationError(
            "williams_r: high/low/close must all be the same length, got "
            f"{len(high)}/{len(low)}/{len(close)}"
        )
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    price_range = highest_high - lowest_low
    wr = -100.0 * (highest_high - close) / price_range.where(price_range > 0)
    return wr.rename("Williams_R")
