import logging

import numpy as np
import pandas as pd

from standard_quant_tools.numeric_contract import (
    require_finite_series,
    require_periods_per_year,
    require_positive_start_level,
)

logger = logging.getLogger(__name__)


def cumulative_return(series: pd.Series) -> float:
    """
    Calculate Cumulative Return over an equity/level series.

    The first value is the denominator, so it must be strictly positive.
    Unchecked, a starting level of 0.0 divided by zero (emitting a
    RuntimeWarning and returning inf), and a negative starting level returned
    a sign-flipped result that looked entirely ordinary.

    Only the START is constrained. A terminal value at or below zero is a
    wiped-out position — a real outcome run_strategy can produce, since it
    applies no bankruptcy floor — and cagr() has documented handling for it.
    """
    if series.empty:
        logger.warning("[cumulative_return] empty series — returning 0.0")
        return 0.0
    require_positive_start_level(series, "series", "cumulative_return")
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
    require_periods_per_year(periods_per_year, "cagr")

    total_ret = cumulative_return(series)
    # N observations span N-1 return INTERVALS, not N. Using len(series)
    # overstated the elapsed time and therefore understated the growth rate.
    # Negligible on a decade of daily bars, material on short windows: over 21
    # observations (one month) it is a 5% error in the exponent's denominator,
    # and the error grows as the window shortens — exactly where a CAGR is
    # already least reliable.
    num_years = (len(series) - 1) / periods_per_year

    if num_years <= 0:
        # A single observation spans no time at all, so it has no growth RATE.
        logger.warning(
            "[cagr] series spans no elapsed time (%d observation(s)) — "
            "returning 0.0",
            len(series),
        )
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

    periods_per_year is validated for the same reason it is everywhere else in
    this package: it is a bare multiplier, so a zero, negative or non-finite
    value produces a confidently wrong number rather than an error.
    """
    require_finite_series(returns, "returns", "annualized_volatility")
    require_periods_per_year(periods_per_year, "annualized_volatility")
    return returns.std() * np.sqrt(periods_per_year)
