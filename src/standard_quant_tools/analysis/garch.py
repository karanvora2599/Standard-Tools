"""
GARCH(1,1) conditional volatility: unlike the realized-volatility estimators
in metrics/volatility_estimators.py (which only describe past variance from
OHLC bars), this module fits a model of how variance itself evolves —
today's variance depends on yesterday's shock and yesterday's variance — and
produces a genuine forward-looking forecast, not just a backward-looking
snapshot.

Scope, stated explicitly (same spirit as the data providers' docstrings):
this is GARCH(1,1) only (the standard, most commonly requested
specification) with normal innovations and a constant mean. EGARCH/GJR-GARCH
(asymmetric leverage effects) and Student-t innovations are real, useful
extensions but not built here — flagged as follow-up work, not silently
approximated.

The variance recursion is inherently sequential (sigma2[t] depends on
sigma2[t-1]) and can't be vectorized across time in plain numpy; it's
numba-@njit'd instead, the same tool this codebase already uses for
strategies.py's state-machine loops — no native build step required, unlike
the optional C++ extension. Fitting (MLE via scipy.optimize) calls that
njit recursion a few dozen times; even at millions of bars this is well
under a second (the recursion itself runs at C speed via numba, and each
optimizer iteration is a single O(n) pass).
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import require_finite_array

logger = logging.getLogger(__name__)

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(func):  # type: ignore[misc]
        return func


try:
    from scipy.optimize import minimize as _scipy_minimize

    HAS_SCIPY = True
except ImportError:
    _scipy_minimize = None
    HAS_SCIPY = False

_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass


_MIN_OBS = 100
_MIN_SIGMA2 = 1e-12


def _require_scipy(context: str) -> None:
    if not HAS_SCIPY:
        raise ValidationError(
            f"{context} requires scipy, which is not installed. Install "
            "scipy to fit a GARCH model — there is no meaningful scipy-free "
            "fallback for a maximum-likelihood fit (unlike, e.g., EVT's "
            "closed-form probability-weighted-moments default)."
        )


@njit
def _garch11_variance_recursion_numba(
    resid_sq: np.ndarray, omega: float, alpha: float, beta: float
) -> np.ndarray:
    n = len(resid_sq)
    sigma2 = np.empty(n)
    sigma2[0] = resid_sq.mean() if n > 0 else _MIN_SIGMA2
    if sigma2[0] < _MIN_SIGMA2:
        sigma2[0] = _MIN_SIGMA2
    for t in range(1, n):
        s2 = omega + alpha * resid_sq[t - 1] + beta * sigma2[t - 1]
        sigma2[t] = s2 if s2 >= _MIN_SIGMA2 else _MIN_SIGMA2
    return sigma2


def _garch11_variance_recursion(
    resid_sq: np.ndarray, omega: float, alpha: float, beta: float
) -> np.ndarray:
    """
    GARCH(1,1) conditional variance recursion -- dispatches to the compiled
    C++ kernel when `_sqt_core` is built, otherwise the numba-JIT'd
    reference above. Both are already fast once warm (numba compiles this
    to machine code on first call); the C++ path exists to eliminate
    numba's JIT cold-start latency (a few hundred ms on the first call in
    any fresh process) and immunity to future numpy ABI breakage -- the
    same permanent rationale this codebase already uses for RSI/ADX/PSAR
    (see Development/performance_insights.md).
    """
    if HAS_CPP and _cpp_core is not None:
        return _cpp_core.garch11_variance_recursion(resid_sq, omega, alpha, beta)
    return _garch11_variance_recursion_numba(resid_sq, omega, alpha, beta)


def _garch11_neg_loglik(
    params: np.ndarray, resid_sq: np.ndarray, penalize: bool = True
) -> float:
    """
    GARCH(1,1) negative log-likelihood -- dispatches to the compiled C++
    kernel when `_sqt_core` is built, otherwise falls back to the numba
    variance recursion plus a NumPy reduction. The C++ path computes the
    recursion and the NLL sum in one fused native call: unlike
    _garch11_variance_recursion (which still round-trips a full sigma2
    array so callers that actually need it, e.g. the final forecast step,
    still get one), scipy.optimize calls this function every single
    L-BFGS-B iteration purely for its scalar result, so fusing away that
    per-iteration array round-trip is the actual performance-relevant part
    of this port.
    """
    omega, alpha, beta = params
    if HAS_CPP and _cpp_core is not None:
        return float(
            _cpp_core.garch11_neg_loglik(resid_sq, omega, alpha, beta, penalize)
        )
    sigma2 = _garch11_variance_recursion_numba(resid_sq, omega, alpha, beta)
    nll = 0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + resid_sq / sigma2)
    if penalize:
        persistence = alpha + beta
        if persistence >= 1.0:
            nll += 1.0e6 * (persistence - 1.0) ** 2
    return float(nll)


def _garch11_neg_loglik_and_grad(
    params: np.ndarray, resid_sq: np.ndarray, penalize: bool = True
):
    """
    GARCH(1,1) NLL and its analytic gradient in one fused call, for
    scipy.optimize's `jac=True` convention (`fun` returns `(value, grad)`).
    C++-only: there is no numba/NumPy analytic-gradient fallback here --
    when `_sqt_core` isn't built, garch_volatility_forecast doesn't call
    this at all, and scipy falls back to its own finite-difference
    gradient with `_garch11_neg_loglik` instead. The analytic gradient was
    verified against a central-difference numerical check across a grid of
    random (resid_sq, omega, alpha, beta) inputs before being wired in
    here (see tests/cpp/test_garch.cpp).
    """
    omega, alpha, beta = params
    nll, grad = _cpp_core.garch11_neg_loglik_grad(
        resid_sq, omega, alpha, beta, penalize
    )
    return float(nll), np.asarray(grad, dtype=float)


def garch_volatility_forecast(
    returns: pd.Series,
    forecast_horizon: int = 10,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    """
    Fit GARCH(1,1) to a return series and forecast conditional volatility.

    Parameters
    ----------
    returns          : pd.Series  Simple or log returns (NOT price levels).
    forecast_horizon : Number of periods ahead to forecast (default 10).
    periods_per_year : Annualization factor (default 252, daily bars).

    Returns
    -------
    dict with keys: omega, alpha, beta, persistence, converged,
    log_likelihood, aic, bic, n_obs, current_annualized_vol,
    long_run_annualized_vol, forecast_annualized_vol (List[float], length
    forecast_horizon).

    Raises
    ------
    ValidationError: forecast_horizon <= 0, fewer than 100 observations
    (GARCH is known to be unstable/unreliable on small samples), or scipy
    is not installed.
    """
    if forecast_horizon <= 0:
        raise ValidationError(f"forecast_horizon must be > 0, got {forecast_horizon}")
    _require_scipy("GARCH(1,1) maximum-likelihood fitting")

    arr = returns.dropna().to_numpy(dtype=float)
    n = len(arr)
    if n < _MIN_OBS:
        raise ValidationError(
            f"garch_volatility_forecast needs at least {_MIN_OBS} "
            f"observations (GARCH fits are unstable on small samples), got {n}"
        )

    # dropna() above already strips NaN for this call path, but not
    # +/-Inf -- garch11_variance_recursion_into's floor-clamp
    # (mean < kMinSigma2) is false for both NaN and Inf, so either would
    # otherwise silently propagate through the entire native recursion
    # uncaught. Checked before the mean/resid computation below (not
    # after) so an Inf is caught at its source instead of producing a
    # NaN via inf-arithmetic in that subtraction first.
    require_finite_array(arr, "returns", "garch_volatility_forecast")
    resid = arr - arr.mean()
    resid_sq = resid**2

    alpha0, beta0 = 0.05, 0.90
    omega0 = resid_sq.mean() * (1.0 - alpha0 - beta0)
    x0 = np.array([omega0, alpha0, beta0])
    bounds = [(1e-12, None), (1e-8, 1.0 - 1e-8), (1e-8, 1.0 - 1e-8)]

    logger.debug("[garch] n_obs=%d  x0=%s", n, x0)
    if HAS_CPP and _cpp_core is not None:
        # jac=True: fun returns (value, grad) together, computed in one
        # fused C++ pass -- an optimizer using the gradient pays for one
        # recursion per iteration instead of scipy's default finite-
        # difference approach (2*3=6 extra NLL evaluations per iteration
        # to numerically estimate a 3-parameter gradient).
        opt = _scipy_minimize(  # type: ignore[misc]
            _garch11_neg_loglik_and_grad,
            x0,
            args=(resid_sq, True),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
        )
    else:
        opt = _scipy_minimize(  # type: ignore[misc]
            _garch11_neg_loglik,
            x0,
            args=(resid_sq, True),
            method="L-BFGS-B",
            bounds=bounds,
        )
    omega, alpha, beta = (float(v) for v in opt.x)
    persistence = alpha + beta
    converged = bool(opt.success) and persistence < 1.0

    # Report likelihood/AIC/BIC without the soft stationarity penalty — at a
    # converged optimum the penalty is zero anyway, but recomputing cleanly
    # avoids any ambiguity about what's actually being reported.
    log_likelihood = -_garch11_neg_loglik(
        np.array([omega, alpha, beta]), resid_sq, penalize=False
    )
    k_params = 3
    aic = 2 * k_params - 2 * log_likelihood
    bic = k_params * np.log(n) - 2 * log_likelihood

    sigma2 = _garch11_variance_recursion(resid_sq, omega, alpha, beta)
    # sigma2[-1] is the model's own conditional-variance estimate for the
    # LAST OBSERVED bar, computed from information only through resid_sq[-2]
    # -- it never incorporates the most recent actual squared return,
    # resid_sq[-1]. Take one more explicit recursion step to get the true
    # one-step-ahead forecast (the value GARCH would predict for the next,
    # not-yet-observed bar), which is what "current volatility" and the
    # forecast's own h=1 base are supposed to mean.
    current_var = float(omega + alpha * resid_sq[-1] + beta * sigma2[-1])

    persistence_safe = min(persistence, 0.9999)
    long_run_var = omega / (1.0 - persistence_safe)

    # current_var above is already the (deterministic, not decayed) T+1
    # value, so forecast step h=1,2,...,horizon needs exponent h-1 -- h=0 at
    # the first output -- or forecast_annualized_vol[0] would silently apply
    # one spurious extra decay step and stop matching current_annualized_vol.
    h = np.arange(0, forecast_horizon, dtype=float)
    forecast_var = long_run_var + (persistence_safe**h) * (current_var - long_run_var)
    forecast_var = np.clip(forecast_var, _MIN_SIGMA2, None)

    result = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "converged": converged,
        "log_likelihood": float(log_likelihood),
        "aic": float(aic),
        "bic": float(bic),
        "n_obs": n,
        "current_annualized_vol": float(np.sqrt(current_var * periods_per_year)),
        "long_run_annualized_vol": float(np.sqrt(long_run_var * periods_per_year)),
        "forecast_annualized_vol": [
            float(np.sqrt(v * periods_per_year)) for v in forecast_var
        ],
    }
    logger.debug(
        "[garch] omega=%.8f  alpha=%.4f  beta=%.4f  persistence=%.4f  " "converged=%s",
        omega,
        alpha,
        beta,
        persistence,
        converged,
    )
    return result
