import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.analysis.regression import calculate_beta
from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import (
    require_finite_scalar,
    require_periods_per_year,
    require_positive_start_level,
)
from standard_quant_tools.validation import validate_series

from .return_metrics import cagr

logger = logging.getLogger(__name__)


def _check_confidence(confidence: float) -> None:
    if not (0.0 < confidence < 1.0):
        raise ValidationError(
            f"confidence must be strictly between 0 and 1, got {confidence!r}"
        )


_scipy_stats = None
_scipy_minimize = None
try:
    from scipy import stats as _scipy_stats  # type: ignore[assignment]
    from scipy.optimize import minimize as _scipy_minimize  # type: ignore[assignment]

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Precomputed z-scores for common confidence levels when scipy is absent
_Z_TABLE = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326, 0.999: 3.090}


@validate_series()
def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> float:
    require_periods_per_year(periods_per_year, "sharpe_ratio")
    require_finite_scalar(risk_free_rate, "risk_free_rate", "sharpe_ratio")
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
    require_periods_per_year(periods_per_year, "sortino_ratio")
    require_finite_scalar(risk_free_rate, "risk_free_rate", "sortino_ratio")
    downside_sq = np.minimum(excess_returns.to_numpy(dtype=np.float64), 0.0) ** 2
    downside_dev = float(np.sqrt(downside_sq.mean())) * np.sqrt(periods_per_year)
    # These two used to share a return value, and they mean opposite things:
    #
    #   downside_dev == 0   the strategy genuinely never lost -> +inf is the
    #                       correct, meaningful answer (infinite Sortino).
    #   downside_dev NaN    the deviation could not be computed at all.
    #
    # Returning +inf for both made "my inputs were unusable" read as "my
    # strategy has no downside whatsoever" — the single most flattering
    # possible misreading. The all-NaN input that produced it is now rejected
    # by validate_series upstream, but the branch is kept honest here too,
    # since a NaN can still arrive from a partially-NaN series.
    if np.isnan(downside_dev):
        return float("nan")
    if downside_dev == 0:
        return np.inf
    return (excess_returns.mean() * periods_per_year) / downside_dev


@validate_series()
def max_drawdown(series: pd.Series) -> float:
    # The START must be positive, not every level. (series - cum_max) /
    # cum_max divides by the running peak, which is seeded from the first
    # value and never decreases — so a positive open keeps the denominator
    # positive even for a curve that is later wiped out, which is a real
    # outcome this must keep supporting. A non-positive OPEN divided by zero
    # or flipped the sign: measured on a curve whose first value was replaced
    # with a negative, max_drawdown returned -1.0048519736842105, a drawdown
    # deeper than total loss.
    require_positive_start_level(series, "series", "max_drawdown")
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
    # Two distinct undefined cases, both previously collapsed to 0.0 — a
    # number that reads as a real, unremarkable Treynor ratio:
    #   * beta is NaN: it could not be estimated (too little overlap with
    #     the benchmark), so there is no systematic-risk denominator at all.
    #   * beta is exactly 0.0: a genuinely market-neutral asset. Treynor is
    #     excess return PER UNIT of systematic risk, and that unit is zero,
    #     so the ratio is undefined rather than zero.
    if not np.isfinite(beta) or beta == 0:
        return float("nan")
    # Use the SAME aligned window beta was estimated from -- the full,
    # unaligned `returns` series can cover a different date range (extra
    # dates the benchmark doesn't have), which would silently mix a mean
    # computed over one window with a beta computed over another.
    excess_return = (
        aligned_returns.mean() - risk_free_rate / periods_per_year
    ) * periods_per_year
    return float(excess_return / beta)


def drawdown_series(series: pd.Series) -> pd.Series:
    """Returns the full drawdown series (fraction from peak), useful for plotting.

    Requires a positive OPENING level for the same reason max_drawdown does:
    the ratio divides by a running peak seeded from it. A curve that is later
    wiped out remains supported."""
    require_positive_start_level(series, "series", "drawdown_series")
    cum_max = series.cummax()
    return (series - cum_max) / cum_max


# ── Extreme Value Theory (EVT) tail risk ─────────────────────────────────────
#
# var_historical/cvar above use the empirical distribution directly — fine
# in-sample, but the further out in the tail you go (99.5%, 99.9%) the fewer
# actual observations back it, and it can't say anything about losses beyond
# the worst one seen. The Peaks-Over-Threshold (POT) approach here instead
# fits a Generalized Pareto Distribution to just the exceedances beyond a
# high threshold (McNeil & Frey 2000) and extrapolates VaR/CVaR from that
# fitted tail model — the standard EVT approach to tail risk.


def _fit_gpd_pwm(exceedances: np.ndarray) -> Tuple[float, float]:
    """
    Fit a Generalized Pareto Distribution via probability-weighted moments
    (Hosking & Wallis 1987) — closed-form, no optimizer, pure vectorized
    numpy (one sort + cumulative arithmetic). This is the default fitting
    method specifically so evt_tail_risk has zero optional-dependency
    surface out of the box.

    Returns (xi, beta) — shape and scale.
    """
    n = len(exceedances)
    x_sorted = np.sort(exceedances)
    b0 = float(x_sorted.mean())
    # weight (n-1-j)/(n-1) for 0-indexed ascending order statistic j: the
    # smallest value gets weight 1, the largest gets weight 0 — this
    # estimates E[X*(1-F(X))], NOT E[X*F(X)] (weighting the other way is a
    # PWM sign bug that silently fits the wrong tail shape).
    weights = (n - 1 - np.arange(n, dtype=float)) / (n - 1)
    b1 = float(np.mean(weights * x_sorted))
    denom = b0 - 2.0 * b1
    # denom -> 0 when the exceedances carry no usable dispersion (e.g. every
    # exceedance identical), which would otherwise divide by ~zero and return
    # inf/nan shape and scale that then propagate silently into var_evt/
    # cvar_evt as if they were a real fitted tail.
    if abs(denom) < 1e-300 or not np.isfinite(denom):
        raise ValidationError(
            "GPD probability-weighted-moment fit is degenerate (b0 - 2*b1 ~ 0) "
            "— the exceedances above the threshold carry no usable dispersion "
            "to fit a tail shape to. Use a larger tail_fraction or a longer "
            "history."
        )
    xi = 2.0 - b0 / denom
    beta = 2.0 * b0 * b1 / denom
    return xi, beta


def _gpd_neg_loglik(params: np.ndarray, exceedances: np.ndarray) -> float:
    xi, beta = params
    if beta <= 0:
        return 1.0e10
    n = len(exceedances)
    if abs(xi) < 1.0e-6:
        # Exponential limiting case (xi -> 0) — avoids dividing by ~0.
        return n * np.log(beta) + float(np.sum(exceedances)) / beta
    z = xi * exceedances / beta
    if np.any(1.0 + z <= 0.0):
        return 1.0e10
    return n * np.log(beta) + (1.0 + 1.0 / xi) * float(np.sum(np.log(1.0 + z)))


def _fit_gpd_mle(exceedances: np.ndarray) -> Tuple[float, float]:
    """
    Refine a GPD fit via maximum likelihood, seeded from the PWM estimate.
    Requires scipy — call sites must guard with _require_scipy first.
    """
    xi0, beta0 = _fit_gpd_pwm(exceedances)
    beta0 = max(beta0, 1.0e-8)
    opt = _scipy_minimize(  # type: ignore[misc]
        _gpd_neg_loglik,
        x0=np.array([xi0, beta0]),
        args=(exceedances,),
        method="Nelder-Mead",
    )
    xi, beta = float(opt.x[0]), float(opt.x[1])
    return xi, beta


def _require_scipy_evt(context: str) -> None:
    if not HAS_SCIPY:
        raise ValidationError(
            f"{context} requires scipy, which is not installed. Install "
            "scipy, or use method='pwm' (the default), which has a "
            "closed-form solution needing only numpy."
        )


@validate_series()
def evt_tail_risk(
    returns: pd.Series,
    confidence: float = 0.99,
    tail_fraction: float = 0.05,
    method: str = "pwm",
) -> Dict[str, Any]:
    """
    Extreme Value Theory tail risk via the Peaks-Over-Threshold method:
    fits a Generalized Pareto Distribution to the worst `tail_fraction` of
    daily losses, then extrapolates VaR/CVaR at `confidence` from that
    fitted tail rather than the raw empirical quantile.

    Parameters
    ----------
    returns       : pd.Series  Return series (NOT price levels).
    confidence    : VaR/CVaR confidence level, strictly between 0 and 1, and
        additionally strictly greater than 1 - tail_fraction (the POT model
        can only extrapolate above the threshold it was fitted at — with the
        0.05 default that means confidence > 0.95).
    tail_fraction : Fraction of observations (by loss) treated as the tail
        for threshold selection — default 0.05 (top 5%), the standard POT
        choice. Must be strictly between 0 and 0.5.
    method        : "pwm" (default, closed-form, no dependencies) or "mle"
        (maximum likelihood, requires scipy, more statistically efficient
        but iterative).

    Returns
    -------
    dict with keys: confidence, tail_fraction, threshold, n_exceedances,
    n_obs, shape_xi, scale_beta, var_evt, cvar_evt, method,
    tail_classification ("heavy_tailed" if xi > 0.1, "light_tailed" if
    xi < -0.1, else "near_exponential").

    Raises
    ------
    ValidationError: confidence or tail_fraction out of range, confidence
        <= 1 - tail_fraction (outside what POT can extrapolate), method is
        neither "pwm" nor "mle", method="mle" requested without scipy
        installed, fewer than 20 exceedances result (the tail fit is
        unreliable below that — a documented threshold, not a silent bad
        fit), or the GPD moment fit is degenerate.
    """
    _check_confidence(confidence)
    if not (0.0 < tail_fraction < 0.5):
        raise ValidationError(
            f"tail_fraction must be strictly between 0 and 0.5, got {tail_fraction!r}"
        )
    if method not in ("pwm", "mle"):
        raise ValidationError(f"method must be 'pwm' or 'mle', got {method!r}")
    if method == "mle":
        _require_scipy_evt("EVT MLE fitting (method='mle')")
    # Peaks-Over-Threshold extrapolates BEYOND the threshold, so the requested
    # quantile has to sit in the tail the GPD was actually fitted to. When
    # confidence <= 1 - tail_fraction the exceedance probability
    # (n/n_u)*(1-confidence) is >= 1, which makes tail_prob**(-xi) - 1 negative
    # and returns a "VaR" BELOW the threshold -- a silently wrong number, not
    # a less precise one. Rejected rather than clamped: the caller asked for a
    # level this method cannot answer, and var_historical/cvar already cover
    # the in-sample region correctly.
    if confidence <= 1.0 - tail_fraction:
        raise ValidationError(
            f"evt_tail_risk(confidence={confidence}, tail_fraction={tail_fraction}): "
            f"Peaks-Over-Threshold can only extrapolate above its own threshold, "
            f"so confidence must exceed 1 - tail_fraction "
            f"({1.0 - tail_fraction:.4f}). Use a larger confidence, a larger "
            "tail_fraction, or var_historical/cvar for in-sample quantiles."
        )

    arr = returns.dropna().to_numpy(dtype=np.float64)
    n_obs = len(arr)
    losses = -arr
    threshold = float(np.percentile(losses, (1.0 - tail_fraction) * 100.0))
    exceedances = losses[losses > threshold] - threshold
    n_exceedances = len(exceedances)

    if n_exceedances < 20:
        raise ValidationError(
            f"evt_tail_risk needs at least 20 exceedances above the "
            f"tail_fraction={tail_fraction} threshold for a reliable GPD "
            f"fit, got {n_exceedances} (n_obs={n_obs}). Use a larger "
            "tail_fraction or a longer history."
        )

    if method == "mle":
        xi, beta = _fit_gpd_mle(exceedances)
    else:
        xi, beta = _fit_gpd_pwm(exceedances)

    n_over_nu = n_obs / n_exceedances
    tail_prob = n_over_nu * (1.0 - confidence)
    if abs(xi) < 1.0e-6:
        var_evt = threshold + beta * float(np.log(1.0 / tail_prob))
    else:
        var_evt = threshold + (beta / xi) * (tail_prob ** (-xi) - 1.0)

    if xi >= 1.0:
        cvar_evt = float("inf")
    else:
        cvar_evt = var_evt / (1.0 - xi) + (beta - xi * threshold) / (1.0 - xi)

    if xi > 0.1:
        tail_classification = "heavy_tailed"
    elif xi < -0.1:
        tail_classification = "light_tailed"
    else:
        tail_classification = "near_exponential"

    logger.debug(
        "[evt_tail_risk] method=%s  n_exceedances=%d  xi=%.4f  beta=%.6f  "
        "var_evt=%.6f  cvar_evt=%.6f",
        method,
        n_exceedances,
        xi,
        beta,
        var_evt,
        cvar_evt,
    )

    return {
        "confidence": confidence,
        "tail_fraction": tail_fraction,
        "threshold": threshold,
        "n_exceedances": n_exceedances,
        "n_obs": n_obs,
        "shape_xi": float(xi),
        "scale_beta": float(beta),
        "var_evt": float(var_evt),
        "cvar_evt": float(cvar_evt),
        "method": method,
        "tail_classification": tail_classification,
    }
