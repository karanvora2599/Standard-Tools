import logging

import pandas as pd

from standard_quant_tools.analysis.microstructure_estimators import (
    corwin_schultz_pairs,
    require_tradeable_bars,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import validate_series

logger = logging.getLogger(__name__)


def _check_window(window: int) -> None:
    if window <= 0:
        raise ValidationError(f"window must be > 0, got {window!r}")


@validate_series()
def amihud_illiquidity(
    returns: pd.Series,
    dollar_volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio: rolling mean of |daily return| /
    dollar volume, scaled by 1e6 for readability (raw values are tiny).
    Higher = less liquid (a given dollar volume moves the price more).

    Days with non-positive dollar_volume are treated as missing (NaN) for
    that day's ratio rather than producing +inf, which would otherwise
    silently dominate the rolling mean.
    """
    _check_window(window)
    safe_volume = dollar_volume.where(dollar_volume > 0.0)
    ratio = (returns.abs() / safe_volume) * 1e6
    return ratio.rolling(window).mean()


@validate_series()
def corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    window: int = 1,
) -> pd.Series:
    """
    Corwin-Schultz (2012) high-low bid-ask spread estimator, using
    consecutive-day pairs of (High, Low) ranges — not the simpler
    single-bar (High-Low)/((High+Low)/2) range proxy already used
    elsewhere in this codebase (backtest.costs.pct_of_range_spread) for
    market-impact modeling; this is the actual named academic estimator.

    For each pair of consecutive bars (t-1, t):
        beta  = ln(H_t-1/L_t-1)^2 + ln(H_t/L_t)^2
        gamma = ln(max(H_t-1,H_t) / min(L_t-1,L_t))^2
        alpha = (sqrt(2*beta) - sqrt(beta)) / k - sqrt(gamma / k),  k = 3-2*sqrt(2)
        spread = 2*(exp(alpha)-1) / (1+exp(alpha))

    Clipped at 0 (the estimator can go slightly negative in low-volatility
    periods, a known property of the formula, not a bug). Result is
    indexed at the SECOND bar of each pair; the first bar is always NaN
    (no prior bar to pair with). When window > 1, the per-pair spread is
    additionally smoothed with a rolling mean of that length.

    THIS IS THE SERIES SHAPE of the same estimator that
    `analysis.microstructure_estimators.corwin_schultz_spread` returns as an
    aggregate dict, and both now run the one kernel. The two shapes are not
    interchangeable -- `check_spread_proxy` needs a per-bar series to roll a
    window over, and the dict form reports `negative_fraction`, which is the
    number that says whether the average means anything. Prefer the dict
    form when a single figure is what is wanted: this one floors negatives
    silently, so a series of zeros and a genuinely tight spread look alike.

    Returns:
        Fractional spread (e.g. 0.01 = 1%, not basis points) as a
        pd.Series aligned to high/low's index.
    """
    _check_window(window)
    require_tradeable_bars(high, low, "corwin_schultz_spread")

    spread = corwin_schultz_pairs(high.shift(1), low.shift(1), high, low)
    spread = spread.clip(lower=0.0)

    if window > 1:
        spread = spread.rolling(window).mean()

    return spread
