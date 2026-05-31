import numpy as np
import pandas as pd

def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()

    upper = sma + (std * num_std)
    lower = sma - (std * num_std)

    return pd.DataFrame({
        'BB_Upper': upper,
        'BB_Middle': sma,
        'BB_Lower': lower
    })

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    Uses np.maximum for a single-pass true range instead of pd.concat.
    """
    prev_close = close.shift(1).to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    tr = pd.Series(
        np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close))),
        index=close.index,
    )
    return tr.rolling(window=period).mean()
