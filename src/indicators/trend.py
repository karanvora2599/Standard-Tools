import pandas as pd
import numpy as np

def sma(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    """
    return series.rolling(window=period).mean()

def ema(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    """
    return series.ewm(span=period, adjust=False).mean()

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Returns:
        pd.DataFrame: Columns ['MACD', 'Signal', 'Histogram']
    """
    exp1 = ema(series, fast)
    exp2 = ema(series, slow)
    macd_line = exp1 - exp2
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })
