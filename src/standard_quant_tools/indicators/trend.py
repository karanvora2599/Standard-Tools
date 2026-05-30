import pandas as pd
import numpy as np

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

def sma(series: pd.Series, period: int = 14) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int = 14) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    MACD: Moving Average Convergence Divergence.
    Returns DataFrame with columns ['MACD', 'Signal', 'Histogram'].
    """
    exp1 = ema(series, fast)
    exp2 = ema(series, slow)
    macd_line = exp1 - exp2
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': macd_line - signal_line,
    })


# ──────────────────────────────────────────────
# ADX — Average Directional Index (Numba JIT)
# ──────────────────────────────────────────────

@njit
def _adx_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Wilder's ADX using the same smoothing as RSI.
    Returns a (n, 3) array: [:, 0] = DI+, [:, 1] = DI-, [:, 2] = ADX.
    """
    n = len(close)
    result = np.full((n, 3), np.nan)

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


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """
    Average Directional Index (ADX) with DI+ and DI-.
    ADX > 25 indicates a strong trend; direction determined by DI+/DI-.
    Uses Numba JIT for high throughput.
    Returns DataFrame with columns ['DI_Plus', 'DI_Minus', 'ADX'].
    """
    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)
    c = close.to_numpy(dtype=np.float64)

    raw = _adx_numba(h, l, c, period)

    return pd.DataFrame(
        {'DI_Plus': raw[:, 0], 'DI_Minus': raw[:, 1], 'ADX': raw[:, 2]},
        index=close.index,
    )


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


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> pd.DataFrame:
    """
    Parabolic SAR — a dynamic trailing stop / trend-following indicator.
    Uses Numba JIT for correctness and speed.

    Returns DataFrame with:
        'SAR'   : Stop-and-reverse price level.
        'Trend' : 1 = rising (long), -1 = falling (short).
    """
    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)

    raw = _psar_numba(h, l, af_start, af_step, af_max)

    return pd.DataFrame(
        {'SAR': raw[:, 0], 'Trend': raw[:, 1]},
        index=high.index,
    )


# ──────────────────────────────────────────────
# Williams %R
# ──────────────────────────────────────────────

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
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    wr = -100.0 * (highest_high - close) / (highest_high - lowest_low)
    return wr.rename('Williams_R')
