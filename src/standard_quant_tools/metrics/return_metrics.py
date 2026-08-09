import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cumulative_return(series: pd.Series) -> float:
    """
    Calculate Cumulative Return.
    """
    if series.empty:
        logger.warning("[cumulative_return] empty series — returning 0.0")
        return 0.0
    return (series.iloc[-1] / series.iloc[0]) - 1


def cagr(series: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).

    A terminal value at or below zero (a leveraged position wiped out — an
    equity curve run_strategy can produce, since it applies no bankruptcy
    floor) has no real compound growth rate: (1 + total_ret) is <= 0, and
    raising it to a fractional power yields NaN plus a RuntimeWarning. That
    NaN then propagates silently into calmar_ratio and every metric built on
    it. Reported as -1.0 (-100%/yr, total loss) instead — the conventional
    representation of a wiped-out position, and a value downstream ratios can
    actually use.
    """
    if series.empty:
        logger.warning("[cagr] empty series — returning 0.0")
        return 0.0

    total_ret = cumulative_return(series)
    num_years = len(series) / periods_per_year

    if num_years == 0:
        return 0.0

    if total_ret <= -1.0:
        logger.warning(
            "[cagr] terminal value is non-positive (cumulative return %.4f) — "
            "position wiped out; reporting -1.0 (total loss) rather than NaN",
            total_ret,
        )
        return -1.0

    return (1 + total_ret) ** (1 / num_years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Annualized Volatility.
    """
    return returns.std() * np.sqrt(periods_per_year)
