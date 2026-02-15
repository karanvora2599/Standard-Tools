import pandas as pd
import numpy as np

def bollinger_bands(series: pd.Series, period: int = 20, num_std: int = 2) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    
    return pd.DataFrame({
        'BB_Upper': upper,
        'BB_Middle': sma, # Same as SMA
        'BB_Lower': lower
    })

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    """
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()
