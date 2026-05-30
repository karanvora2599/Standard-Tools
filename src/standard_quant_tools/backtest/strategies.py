"""
Signal generation functions for built-in backtest strategies.

Each function accepts a full OHLCV DataFrame plus strategy-specific kwargs
and returns a pd.Series of signals (1 = long, 0 = flat, -1 = short).

All functions are defined at module level so they are picklable by
ProcessPoolExecutor (required for backtest_grid on Windows / spawn).
"""

import numpy as np
import pandas as pd

from standard_quant_tools.indicators.trend import sma, macd
from standard_quant_tools.indicators.momentum import rsi
from standard_quant_tools.indicators.volatility import bollinger_bands


def _sma_signals(
    df: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 30,
    **_,
) -> pd.Series:
    """Long when fast SMA > slow SMA, flat otherwise."""
    return pd.Series(
        np.where(sma(df["Close"], fast_period) > sma(df["Close"], slow_period), 1, 0),
        index=df.index,
    )


def _rsi_signals(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    **_,
) -> pd.Series:
    """Enter long when RSI < oversold; hold until RSI > overbought."""
    rsi_vals = rsi(df["Close"], period)
    rsi_arr = rsi_vals.to_numpy(dtype=float)
    values = np.zeros(len(df))
    in_pos = False
    for i in range(len(values)):
        if np.isnan(rsi_arr[i]):
            continue
        if not in_pos and rsi_arr[i] < oversold:
            in_pos = True
        elif in_pos and rsi_arr[i] > overbought:
            in_pos = False
        values[i] = 1.0 if in_pos else 0.0
    return pd.Series(values, index=df.index)


def _macd_signals(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    **_,
) -> pd.Series:
    """Long when MACD line > signal line, flat otherwise."""
    m = macd(df["Close"], fast, slow, signal)
    return pd.Series(
        np.where(m["MACD"] > m["Signal"], 1, 0),
        index=df.index,
    )


def _bollinger_signals(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    **_,
) -> pd.Series:
    """Enter long when price touches lower band; exit at middle band."""
    bb = bollinger_bands(df["Close"], period, num_std)
    close_arr = df["Close"].to_numpy(dtype=float)
    lower_arr = bb["BB_Lower"].to_numpy(dtype=float)
    middle_arr = bb["BB_Middle"].to_numpy(dtype=float)
    values = np.zeros(len(close_arr))
    in_pos = False
    for i in range(len(close_arr)):
        if np.isnan(lower_arr[i]):
            continue
        if not in_pos and close_arr[i] <= lower_arr[i]:
            in_pos = True
        elif in_pos and close_arr[i] >= middle_arr[i]:
            in_pos = False
        values[i] = 1.0 if in_pos else 0.0
    return pd.Series(values, index=df.index)


STRATEGY_REGISTRY = {
    "sma_crossover": _sma_signals,
    "rsi_mean_reversion": _rsi_signals,
    "macd_crossover": _macd_signals,
    "bollinger_reversion": _bollinger_signals,
}
