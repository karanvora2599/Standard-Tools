import logging
import math
from typing import Any

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.indicators._missing import refuse_infinities
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
    Wilder's ADX using the same smoothing as RSI -- the fallback for
    adx_into in indicators.cpp, operation for operation.
    Returns a (n, 3) array: [:, 0] = DI+, [:, 1] = DI-, [:, 2] = ADX.

    A bar whose high, low or close is not finite is a missing bar and is
    skipped: DM and TR are measured against the last PRESENT bar, every
    Wilder state is carried across the gap, and the row is NaN there. The
    warm-up thresholds count present bars. On a series with no gap that is
    the single pass the kernel always made. (The former four-array version
    carried a NaN into every later row, where the kernel's max() quietly
    dropped it -- the two backends disagreed about every bar after a gap.)
    """
    n = len(close)
    result = np.full((n, 3), np.nan)

    # Wilder's seed needs `period` moves before the first DI. With
    # n <= period there cannot be that many, and returning here keeps every
    # write below inside the array (@njit compiles without bounds checking).
    if period <= 0 or n <= period:
        return result

    adx_start = 2 * period - 1
    atr_s = 0.0
    dmp_s = 0.0
    dmm_s = 0.0
    dx_seed_sum = 0.0
    adx_val = 0.0
    j = 0  # position among the PRESENT bars
    have_prev = False
    prev_high = 0.0
    prev_low = 0.0
    prev_close = 0.0

    for i in range(n):
        if not (
            math.isfinite(high[i]) and math.isfinite(low[i]) and math.isfinite(close[i])
        ):
            continue
        if not have_prev:
            prev_high = high[i]
            prev_low = low[i]
            prev_close = close[i]
            have_prev = True
            continue
        j += 1

        up_move = high[i] - prev_high
        down_move = prev_low - low[i]
        dm_plus = up_move if (up_move > down_move and up_move > 0.0) else 0.0
        dm_minus = down_move if (down_move > up_move and down_move > 0.0) else 0.0
        tr = max(
            high[i] - low[i],
            abs(high[i] - prev_close),
            abs(low[i] - prev_close),
        )
        prev_high = high[i]
        prev_low = low[i]
        prev_close = close[i]

        if j <= period:
            # Wilder's seed sums over the first `period` moves.
            atr_s += tr
            dmp_s += dm_plus
            dmm_s += dm_minus
        else:
            # Wilder's smooth forward.
            atr_s = atr_s - (atr_s / period) + tr
            dmp_s = dmp_s - (dmp_s / period) + dm_plus
            dmm_s = dmm_s - (dmm_s / period) + dm_minus

        if j < period:
            continue

        di_p = 100.0 * dmp_s / atr_s if atr_s != 0.0 else 0.0
        di_m = 100.0 * dmm_s / atr_s if atr_s != 0.0 else 0.0
        result[i, 0] = di_p
        result[i, 1] = di_m

        di_sum = di_p + di_m
        dx = 100.0 * abs(di_p - di_m) / di_sum if di_sum != 0.0 else 0.0

        # ADX = Wilder's smooth of DX; `period` DX values initialise it.
        if j <= adx_start:
            dx_seed_sum += dx
            if j == adx_start:
                adx_val = dx_seed_sum / period
                result[i, 2] = adx_val
        else:
            adx_val = (adx_val * (period - 1) + dx) / period
            result[i, 2] = adx_val

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

    A bar with a NaN high, low or close is a missing bar: the recursions
    skip it, so the result is the ADX of the series with that bar dropped,
    and NaN at the bar itself. An infinite value is refused. See
    `indicators/_missing.py`.
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
    refuse_infinities(h, "high", "adx")
    refuse_infinities(l, "low", "adx")
    refuse_infinities(c, "close", "adx")

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
