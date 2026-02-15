import pandas as pd
import numpy as np
from standard_quant_tools.validation import validate_series

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
        change = delta[i-1]
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
    Uses Numba optimized Wilder's Smoothing if available.
    """
    if series.empty:
        return pd.Series(dtype=float)

    if HAS_NUMBA:
        values = series.values.astype(np.float64)
        rsi_vals = _rsi_numba(values, period)
        return pd.Series(rsi_vals, index=series.index)
    else:
        # Fallback to pandas-based (SMA for simplicity in fallback, or EWM)
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

@validate_series()
def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator.
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    
    return pd.DataFrame({'Stoch_K': k, 'Stoch_D': d})
