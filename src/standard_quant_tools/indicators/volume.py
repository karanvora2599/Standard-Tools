import logging
from typing import Optional

import numpy as np
import pandas as pd

from standard_quant_tools.validation import validate_series

logger = logging.getLogger(__name__)


@validate_series()
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On Balance Volume (OBV).
    Cumulative volume signed by the direction of price change.
    A rising OBV confirms an uptrend; divergence signals weakness.
    Pure numpy — no loops needed.
    """
    logger.debug("[obv] bars=%d", len(close))
    direction = pd.Series(
        np.sign(close.diff().fillna(0.0).to_numpy(dtype=np.float64)),
        index=close.index,
    )
    result = (direction * volume).cumsum().rename("OBV")
    logger.debug(
        "[obv] final=%.0f  trend=%s",
        float(result.iloc[-1]),
        "up" if float(result.iloc[-1]) > float(result.iloc[0]) else "down",
    )
    return result


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: Optional[int] = None,
) -> pd.Series:
    """
    Volume Weighted Average Price (VWAP).

    If `period` is None, computes a cumulative session VWAP from the first bar.
    If `period` is given, computes a rolling VWAP over that window.

    Typical price = (H + L + C) / 3 — the standard VWAP numerator.
    """
    mode = f"rolling({period})" if period is not None else "cumulative"
    logger.debug("[vwap] mode=%s  bars=%d", mode, len(close))
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume

    if period is None:
        result = (tp_vol.cumsum() / volume.cumsum()).rename("VWAP")
    else:
        result = (
            tp_vol.rolling(window=period, min_periods=period).sum()
            / volume.rolling(window=period, min_periods=period).sum()
        ).rename("VWAP")
    valid = result.dropna()
    if not valid.empty:
        logger.debug("[vwap] last=%.4f", float(valid.iloc[-1]))
    return result


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Money Flow Index (MFI) — a volume-weighted RSI.
    Oscillates 0–100. Values below 20 suggest oversold, above 80 overbought.
    Fully vectorized: uses rolling sums of positive/negative money flow.
    """
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume

    tp_diff = typical_price.diff()

    # Positive money flow: bars where typical price rose
    pos_flow = raw_money_flow.where(tp_diff > 0, 0.0).rolling(period).sum()
    # Negative money flow: bars where typical price fell (or was flat)
    neg_flow = raw_money_flow.where(tp_diff <= 0, 0.0).rolling(period).sum()

    # When neg_flow = 0 (all bars up), MFI = 100; when pos_flow = 0, MFI = 0.
    mfr = pos_flow / neg_flow.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + mfr))
    result = result.where(neg_flow != 0, 100.0)
    result = result.where(pos_flow != 0, 0.0)
    return result.rename("MFI")
