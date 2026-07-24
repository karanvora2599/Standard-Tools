import logging

import numpy as np
import pandas as pd

from standard_quant_tools.analysis.regression import calculate_beta
from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import validate_series

from .return_metrics import cagr

logger = logging.getLogger(__name__)


def _check_confidence(confidence: float) -> None:
    if not (0.0 < confidence < 1.0):
        raise ValidationError(
            f"confidence must be strictly between 0 and 1, got {confidence!r}"
        )

_scipy_stats = None
try:
    from scipy import stats as _scipy_stats  # type: ignore[assignment]

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Precomputed z-scores for common confidence levels when scipy is absent
_Z_TABLE = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326, 0.999: 3.090}


@validate_series()
def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    excess_returns = returns - risk_free_rate / periods_per_year
    std = returns.std()
    if std == 0:
        logger.warning("[sharpe_ratio] zero-volatility returns — returning 0.0")
        return 0.0
    return (excess_returns.mean() / std) * np.sqrt(periods_per_year)


@validate_series()
def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    excess_returns = returns - risk_free_rate / periods_per_year
    # Semi-deviation: RMS of clipped returns across ALL periods (zero contribution for
    # profitable bars). Dividing by N (not just n_negative) is the Sortino & Price (1994)
    # definition and gives a larger, more conservative denominator than std of negatives only.
    downside_sq = np.minimum(excess_returns.to_numpy(dtype=np.float64), 0.0) ** 2
    downside_dev = float(np.sqrt(downside_sq.mean())) * np.sqrt(periods_per_year)
    if downside_dev == 0 or np.isnan(downside_dev):
        return np.inf
    return (excess_returns.mean() * periods_per_year) / downside_dev


@validate_series()
def max_drawdown(series: pd.Series) -> float:
    cum_max = series.cummax()
    drawdown = (series - cum_max) / cum_max
    return drawdown.min()


@validate_series()
def calmar_ratio(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calmar Ratio: CAGR / |Max Drawdown|.
    Higher is better. A ratio > 1 means annual return exceeds worst drawdown.
    """
    annual_return = cagr(equity_curve, periods_per_year)
    mdd = max_drawdown(equity_curve)
    if mdd == 0.0:
        return np.inf
    return annual_return / abs(mdd)


@validate_series()
def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk (VaR) at the given confidence level.
    Returns the loss (positive number) not exceeded with probability `confidence`.
    Uses the empirical distribution — no normality assumption.

    Raises:
        ValidationError: confidence is not strictly between 0 and 1.
    """
    _check_confidence(confidence)
    arr = returns.dropna().to_numpy(dtype=np.float64)
    return float(-np.percentile(arr, (1 - confidence) * 100))


@validate_series()
def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric (Gaussian) VaR. Faster but assumes normally distributed returns.
    Uses scipy.stats if available; falls back to a precomputed z-table for a
    small set of common confidence levels.

    Raises:
        ValidationError: confidence is not strictly between 0 and 1, or scipy
            is not installed and confidence isn't one of the precomputed
            z-table levels (0.90, 0.95, 0.99, 0.999) — silently substituting
            the 95% z-score for an unsupported confidence would misreport
            the actual risk at the level the caller asked for.
    """
    _check_confidence(confidence)
    mu = float(returns.mean())
    sigma = float(returns.std())
    if HAS_SCIPY and _scipy_stats is not None:
        z = float(_scipy_stats.norm.ppf(1 - confidence))  # type: ignore[union-attr]
    elif confidence in _Z_TABLE:
        z = -_Z_TABLE[confidence]
    else:
        raise ValidationError(
            f"var_parametric(confidence={confidence}) requires scipy for an "
            f"arbitrary confidence level; scipy is not installed and {confidence} "
            f"is not one of the precomputed z-table levels {sorted(_Z_TABLE)}. "
            "Install scipy, or use one of the precomputed levels."
        )
    return float(-(mu + z * sigma))


@validate_series()
def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR / Expected Shortfall (CVaR).
    The expected loss given that the loss exceeds the VaR threshold.
    More conservative and coherent than VaR.

    Raises:
        ValidationError: confidence is not strictly between 0 and 1.
    """
    _check_confidence(confidence)
    arr = returns.dropna().to_numpy(dtype=np.float64)
    threshold = np.percentile(arr, (1 - confidence) * 100)
    tail = arr[arr <= threshold]
    return float(-tail.mean()) if len(tail) > 0 else float(-threshold)


@validate_series()
def information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = 252
) -> float:
    """
    Information Ratio: annualized active return divided by tracking error.
    Measures quality of active management. IR > 0.5 is considered strong.
    """
    common_idx = returns.index.intersection(benchmark_returns.index)
    active = returns.loc[common_idx] - benchmark_returns.loc[common_idx]
    tracking_error = active.std() * np.sqrt(periods_per_year)
    if tracking_error == 0:
        return 0.0
    return float((active.mean() * periods_per_year) / tracking_error)


@validate_series()
def treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Treynor Ratio: excess return per unit of systematic (beta) risk.
    Complements Sharpe (which uses total risk).
    """
    common_idx = returns.index.intersection(benchmark_returns.index)
    aligned_returns = returns.loc[common_idx]
    beta_stats = calculate_beta(aligned_returns, benchmark_returns.loc[common_idx])
    beta = beta_stats["beta"]
    if beta == 0:
        return 0.0
    # Use the SAME aligned window beta was estimated from -- the full,
    # unaligned `returns` series can cover a different date range (extra
    # dates the benchmark doesn't have), which would silently mix a mean
    # computed over one window with a beta computed over another.
    excess_return = (
        aligned_returns.mean() - risk_free_rate / periods_per_year
    ) * periods_per_year
    return float(excess_return / beta)


def drawdown_series(series: pd.Series) -> pd.Series:
    """Returns the full drawdown series (fraction from peak), useful for plotting."""
    cum_max = series.cummax()
    return (series - cum_max) / cum_max
