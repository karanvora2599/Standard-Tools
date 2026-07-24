import logging
from typing import Any

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Uses C++ fused mean+std path when available (3-8× faster than two pandas
    rolling passes).  Falls back to pandas otherwise.
    """
    logger.debug(
        "[bollinger] period=%d  std=%.1f  bars=%d  path=%s",
        period,
        num_std,
        len(series),
        "C++" if (HAS_CPP and _cpp_core is not None) else "pandas",
    )

    # ── C++ fast path ─────────────────────────────────────────────────────────
    if HAS_CPP and _cpp_core is not None:
        try:
            arr = series.to_numpy(dtype=np.float64)
            out = _cpp_core.bollinger_bands(arr, period, num_std)
            upper = pd.Series(out[:, 0], index=series.index)
            middle = pd.Series(out[:, 1], index=series.index)
            lower = pd.Series(out[:, 2], index=series.index)
            result = pd.DataFrame(
                {"BB_Upper": upper, "BB_Middle": middle, "BB_Lower": lower}
            )
            valid_u = upper.dropna()
            if not valid_u.empty:
                logger.debug(
                    "[bollinger] last upper=%.4f  middle=%.4f  lower=%.4f  width=%.4f",
                    float(valid_u.iloc[-1]),
                    float(middle.dropna().iloc[-1]),
                    float(lower.dropna().iloc[-1]),
                    float(valid_u.iloc[-1]) - float(lower.dropna().iloc[-1]),
                )
            return result
        except Exception as exc:
            logger.warning("[bollinger] C++ failed (%s) — using pandas", exc)

    # ── Pandas fallback ───────────────────────────────────────────────────────
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)

    result = pd.DataFrame({"BB_Upper": upper, "BB_Middle": sma, "BB_Lower": lower})
    valid_u = upper.dropna()
    valid_l = lower.dropna()
    if not valid_u.empty:
        width = float(valid_u.iloc[-1]) - float(valid_l.iloc[-1])
        logger.debug(
            "[bollinger] last upper=%.4f  middle=%.4f  lower=%.4f  width=%.4f",
            float(valid_u.iloc[-1]),
            float(sma.dropna().iloc[-1]),
            float(valid_l.iloc[-1]),
            width,
        )
    return result


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    Uses np.maximum for a single-pass true range instead of pd.concat.
    """
    logger.debug("[atr] period=%d  bars=%d", period, len(close))
    prev_close = close.shift(1).to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    tr = pd.Series(
        np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close))),
        index=close.index,
    )
    result = tr.rolling(window=period).mean()
    valid = result.dropna()
    if not valid.empty:
        logger.debug("[atr] last=%.4f", float(valid.iloc[-1]))
    return result


def wilder_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range using Wilder's smoothing (not a simple rolling mean).

    TR[0] = high[0] - low[0]
    TR[i] = max(H[i]-L[i], |H[i]-C[i-1]|, |L[i]-C[i-1]|)
    Seed:    ATR[period-1] = mean(TR[0..period-1])
    Forward: ATR[i]        = (ATR[i-1]*(period-1) + TR[i]) / period

    Uses C++ fast path when built, otherwise falls back to a pure-Python loop.
    First period-1 values are NaN.
    """
    if period <= 0:
        raise ValidationError(f"period must be > 0, got {period}")

    h = high.to_numpy(dtype=np.float64)
    l = low.to_numpy(dtype=np.float64)
    c = close.to_numpy(dtype=np.float64)
    n = len(h)

    if HAS_CPP and _cpp_core is not None:
        raw = _cpp_core.wilder_atr(h, l, c, period)
        return pd.Series(raw, index=close.index, name="Wilder_ATR")

    # Pure-Python fallback (correct but slow; C++ path is preferred)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    result = np.full(n, np.nan)
    if n >= period:
        result[period - 1] = tr[:period].mean()
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + tr[i]) / period

    return pd.Series(result, index=close.index, name="Wilder_ATR")
