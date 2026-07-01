"""
Signal generation functions for built-in backtest strategies.

Each function accepts a full OHLCV DataFrame plus strategy-specific kwargs
and returns a pd.Series of signals (1 = long, 0 = flat, -1 = short).

All functions are defined at module level so they are picklable by
ProcessPoolExecutor (required for backtest_grid on Windows / spawn).
"""

import logging

import numpy as np
import pandas as pd

from standard_quant_tools.indicators.trend import sma, macd
from standard_quant_tools.indicators.momentum import rsi
from standard_quant_tools.indicators.volatility import bollinger_bands

logger = logging.getLogger(__name__)

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(func):  # type: ignore[misc]
        return func


@njit
def _rsi_state_machine(rsi_arr: np.ndarray, oversold: float, overbought: float) -> np.ndarray:
    n = len(rsi_arr)
    values = np.zeros(n)
    in_pos = False
    for i in range(n):
        if np.isnan(rsi_arr[i]):
            continue
        if not in_pos and rsi_arr[i] < oversold:
            in_pos = True
        elif in_pos and rsi_arr[i] > overbought:
            in_pos = False
        values[i] = 1.0 if in_pos else 0.0
    return values


@njit
def _bollinger_state_machine(
    close_arr: np.ndarray, lower_arr: np.ndarray, middle_arr: np.ndarray
) -> np.ndarray:
    n = len(close_arr)
    values = np.zeros(n)
    in_pos = False
    for i in range(n):
        if np.isnan(lower_arr[i]):
            continue
        if not in_pos and close_arr[i] <= lower_arr[i]:
            in_pos = True
        elif in_pos and close_arr[i] >= middle_arr[i]:
            in_pos = False
        values[i] = 1.0 if in_pos else 0.0
    return values


def _log_signals(name: str, signals: pd.Series) -> None:
    arr = signals.to_numpy()
    n = len(arr)
    n_long  = int((arr == 1).sum())
    n_short = int((arr == -1).sum())
    n_flat  = n - n_long - n_short
    logger.debug(
        "[signal] %s  bars=%d  long=%d(%.0f%%)  flat=%d(%.0f%%)  short=%d(%.0f%%)",
        name, n,
        n_long,  100 * n_long  / n if n else 0,
        n_flat,  100 * n_flat  / n if n else 0,
        n_short, 100 * n_short / n if n else 0,
    )


def _sma_signals(
    df: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 30,
    **_,
) -> pd.Series:
    """Long when fast SMA > slow SMA, flat otherwise."""
    logger.debug("[signal] sma_crossover  fast=%d  slow=%d  bars=%d", fast_period, slow_period, len(df))
    result = pd.Series(
        np.where(sma(df["Close"], fast_period) > sma(df["Close"], slow_period), 1, 0),
        index=df.index,
    )
    _log_signals("sma_crossover", result)
    return result


def _rsi_signals(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    **_,
) -> pd.Series:
    """Enter long when RSI < oversold; hold until RSI > overbought."""
    logger.debug("[signal] rsi_mean_reversion  period=%d  oversold=%.0f  overbought=%.0f  bars=%d",
                 period, oversold, overbought, len(df))
    rsi_arr = rsi(df["Close"], period).to_numpy(dtype=float)
    result = pd.Series(_rsi_state_machine(rsi_arr, oversold, overbought), index=df.index)
    _log_signals("rsi_mean_reversion", result)
    return result


def _macd_signals(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    **_,
) -> pd.Series:
    """Long when MACD line > signal line, flat otherwise."""
    logger.debug("[signal] macd_crossover  fast=%d  slow=%d  signal=%d  bars=%d", fast, slow, signal, len(df))
    m = macd(df["Close"], fast, slow, signal)
    result = pd.Series(
        np.where(m["MACD"] > m["Signal"], 1, 0),
        index=df.index,
    )
    _log_signals("macd_crossover", result)
    return result


def _bollinger_signals(
    df: pd.DataFrame,
    period: int = 20,
    num_std: float = 2.0,
    **_,
) -> pd.Series:
    """Enter long when price touches lower band; exit at middle band."""
    logger.debug("[signal] bollinger_reversion  period=%d  std=%.1f  bars=%d", period, num_std, len(df))
    bb = bollinger_bands(df["Close"], period, num_std)
    close_arr = df["Close"].to_numpy(dtype=float)
    lower_arr = bb["BB_Lower"].to_numpy(dtype=float)
    middle_arr = bb["BB_Middle"].to_numpy(dtype=float)
    result = pd.Series(_bollinger_state_machine(close_arr, lower_arr, middle_arr), index=df.index)
    _log_signals("bollinger_reversion", result)
    return result


STRATEGY_REGISTRY = {
    "sma_crossover": _sma_signals,
    "rsi_mean_reversion": _rsi_signals,
    "macd_crossover": _macd_signals,
    "bollinger_reversion": _bollinger_signals,
}
