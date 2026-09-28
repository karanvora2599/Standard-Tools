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
the optional C++ extension. Fitting (MLE via scipy.optimize, from three
starting points on returns rescaled to unit mean square) makes a few
hundred O(n) passes of that recursion; two million bars fit in about a
second with the C++ kernel.
"""

import logging
from typing import Any, Dict, List, Optional

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

# THE FIT RUNS ON RESCALED RETURNS. On daily returns omega is ~1e-6 while
# alpha and beta are ~0.1 and ~0.9, and the gradient at the old starting
# point was ~[4.6e7, -219, 1286]: the omega direction dominated by five
# orders of magnitude, L-BFGS-B's default ftol was met within a few
# iterations, and the fit stopped near where it started while `success`
# said True. Measured on simulated GARCH(1,1) with alpha = 0.25, n = 4000:
# a fitted alpha of 0.07, 111 nats of likelihood left on the table, and a
# different wrong answer from each of the two gradient sources below.
# Dividing the squared residuals by their mean makes every parameter O(1);
# the likelihood is scale-equivariant, so omega is multiplied back and
# alpha and beta need nothing. See the CHANGELOG entry of 2026-09-27.
_PARAM_NAMES = ("omega", "alpha", "beta")

#: (alpha, beta) starting pairs, each with omega = 1 - alpha - beta on the
#: rescaled problem, so every start sits at the sample variance. A single
#: start can stop on the (alpha -> 0, beta -> 1) ridge of a weak-ARCH
#: sample; three cut the mean shortfall against an independent optimizer on
#: iid samples from 0.15 nats to 0.02.
_STARTS = ((0.05, 0.90), (0.10, 0.80), (0.20, 0.60))

_BOUNDS = ((1e-12, None), (1e-8, 1.0 - 1e-8), (1e-8, 1.0 - 1e-8))

#: Tight enough that the stop is decided by the gradient, not by a relative
#: change in the objective that a flat ridge satisfies early.
_OPTIONS = {"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2000}

#: Largest projected-gradient component per observation that still counts
#: as a maximum. Fits that stopped short measured 0.07 to 0.53; fits at
#: the maximum measured 1e-8 to 1e-6, so the threshold is not a knife edge.
_GRADIENT_TOL = 1e-4

#: How close to a bound (on the rescaled problem) counts as on it.
#: L-BFGS-B projects onto the box, so a parameter pushed against a bound
#: lands on it exactly; this only absorbs the last rounding.
_BOUND_TOL = 1e-9


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
    (measured; see CHANGELOG for the figures).
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


def _finite_difference_gradient(params: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """
    Gradient of the penalized negative log-likelihood by differences, for
    the convergence verdict when there is no analytic gradient. Central
    where both neighbours are inside the bounds, one-sided at a bound so the
    check never evaluates a parameter the fit could not take.
    """
    grad = np.empty(3)
    for i, (lower, upper) in enumerate(_BOUNDS):
        step = 1e-6 * max(1.0, abs(float(params[i])))
        up, down = params.copy(), params.copy()
        up[i] += step
        down[i] -= step
        if lower is not None and down[i] < lower:
            grad[i] = (
                _garch11_neg_loglik(up, z2, True)
                - _garch11_neg_loglik(params, z2, True)
            ) / step
        elif upper is not None and up[i] > upper:
            grad[i] = (
                _garch11_neg_loglik(params, z2, True)
                - _garch11_neg_loglik(down, z2, True)
            ) / step
        else:
            grad[i] = (
                _garch11_neg_loglik(up, z2, True) - _garch11_neg_loglik(down, z2, True)
            ) / (2.0 * step)
    return grad


def _at_bound(params: np.ndarray) -> List[str]:
    """Names of the parameters sitting on a bound of the rescaled fit."""
    names = []
    for name, value, (lower, upper) in zip(_PARAM_NAMES, params, _BOUNDS):
        if (lower is not None and value <= lower + _BOUND_TOL) or (
            upper is not None and value >= upper - _BOUND_TOL
        ):
            names.append(name)
    return names


def _projected_gradient_norm(params: np.ndarray, grad: np.ndarray) -> float:
    """
    Largest absolute component of the projected gradient; the caller
    divides by the number of observations. A component pointing out of the
    box at a bound is zero, because the fit cannot move that way: alpha
    pinned at its floor with a positive gradient is at a constrained
    maximum, not short of one.
    """
    projected = np.array(grad, dtype=float)
    for i, (lower, upper) in enumerate(_BOUNDS):
        if lower is not None and params[i] <= lower + _BOUND_TOL and projected[i] > 0:
            projected[i] = 0.0
        if upper is not None and params[i] >= upper - _BOUND_TOL and projected[i] < 0:
            projected[i] = 0.0
    return float(np.max(np.abs(projected)))


def _fit_rescaled(z2: np.ndarray):
    """
    Maximum likelihood on squared residuals whose mean is 1, from every
    start in `_STARTS`; the best optimum and the gradient at it.

    Both gradient sources run the same starts, bounds and options, so they
    reach the same optimum: measured to 1.6e-7 in the parameters and 3e-11
    nats between them on simulated GARCH(1,1). Before the rescaling they
    stopped at different wrong points, about 10 nats apart on one real
    series.
    """
    analytic = HAS_CPP and _cpp_core is not None
    best = None
    for alpha0, beta0 in _STARTS:
        x0 = np.array([1.0 - alpha0 - beta0, alpha0, beta0])
        if analytic:
            # jac=True: fun returns (value, grad) together, computed in one
            # fused C++ pass -- one recursion per iteration instead of
            # scipy's finite-difference estimate of a 3-parameter gradient.
            opt = _scipy_minimize(  # type: ignore[misc]
                _garch11_neg_loglik_and_grad,
                x0,
                args=(z2, True),
                method="L-BFGS-B",
                jac=True,
                bounds=_BOUNDS,
                options=_OPTIONS,
            )
        else:
            opt = _scipy_minimize(  # type: ignore[misc]
                _garch11_neg_loglik,
                x0,
                args=(z2, True),
                method="L-BFGS-B",
                bounds=_BOUNDS,
                options=_OPTIONS,
            )
        if best is None or opt.fun < best.fun:
            best = opt
    params = np.asarray(best.x, dtype=float)
    if analytic:
        grad = _garch11_neg_loglik_and_grad(params, z2, True)[1]
    else:
        grad = _finite_difference_gradient(params, z2)
    return best, params, grad


#: Conventional level for the squared-residual test. A p-value below this is
#: the sample saying the fitted model left volatility clustering behind.
_MISSPECIFICATION_ALPHA = 0.05


def _residual_diagnostics(resid: np.ndarray, sigma2: np.ndarray) -> Dict[str, Any]:
    """
    Whether the fit removed the clustering it was fitted to remove.

    `converged` answers a question about the OPTIMIZER -- that L-BFGS-B
    reached a maximum of the likelihood (its projected gradient is
    negligible) inside the bounds with a stationary persistence. It says
    nothing about whether GARCH(1,1) with normal
    innovations and a constant mean is the right model for this series, and
    a fit can converge cleanly onto a specification the data rejects.

    The standardized residual z_t = e_t / sqrt(sigma2_t) is what answers
    that. If the model captured the variance dynamics, z is close to iid:
    z^2 has no autocorrelation left, because every predictable piece of it
    has been divided out. A Ljung-Box on z^2 that rejects means the variance
    path the model produced does not explain the variance path the sample
    had -- the single most direct evidence that the specification is wrong.

    The two tests answer different questions and are both reported. On the
    raw z, Ljung-Box tests the MEAN: this model assumes a constant mean, so
    a rejection there points at the mean equation (an AR term), not at the
    variance one. On z^2 it tests the VARIANCE, and that is the one
    `misspecified` is read off.

    Skew and excess kurtosis describe what is left: a normal innovation
    would give roughly 0 and 0, and a large positive excess kurtosis is the
    standard sign that Student-t innovations are wanted.
    """
    from standard_quant_tools.analysis.diagnostics import ljung_box

    standardized = resid / np.sqrt(sigma2)
    finite = standardized[np.isfinite(standardized)]

    def _p(squared: bool) -> Optional[float]:
        try:
            return float(ljung_box(pd.Series(finite), squared=squared)["p_value"])
        except ValidationError:
            # A degenerate residual series (no variance left to test) is not
            # evidence of misspecification; it is the absence of a test.
            return None

    squared_p = _p(squared=True)
    if finite.size >= 4:
        centred = finite - finite.mean()
        variance = float((centred**2).mean())
        if variance > 0:
            skew: Optional[float] = float((centred**3).mean() / variance**1.5)
            kurtosis: Optional[float] = float((centred**4).mean() / variance**2 - 3.0)
        else:
            skew = kurtosis = None
    else:
        skew = kurtosis = None

    return {
        "ljung_box_p": _p(squared=False),
        "ljung_box_squared_p": squared_p,
        "standardized_skew": skew,
        "standardized_kurtosis": kurtosis,
        "misspecified": bool(
            squared_p is not None and squared_p < _MISSPECIFICATION_ALPHA
        ),
    }


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
    gradient_norm, at_bound, log_likelihood, aic, bic, n_obs,
    current_annualized_vol, long_run_annualized_vol,
    forecast_annualized_vol (List[float], length forecast_horizon),
    conditional_variance (pd.Series on the returns' index -- the whole
    variance path, not just its last value), the residual diagnostics
    ljung_box_p, ljung_box_squared_p, standardized_skew,
    standardized_kurtosis (EXCESS) and misspecified, and warnings.

    `converged` is True only when L-BFGS-B reported success, persistence
    is below 1, AND `gradient_norm` -- the largest projected-gradient
    component of the negative log-likelihood per observation, on returns
    rescaled to unit mean square -- is below 1e-4. `at_bound` names the
    parameters sitting on a bound; 'alpha' there means no ARCH effect,
    with beta not identified, and a warning says so.

    `converged` and `misspecified` are independent: the first is about the
    optimizer, the second about the specification. See
    `_residual_diagnostics`.

    Raises
    ------
    ValidationError: forecast_horizon <= 0, fewer than 100 observations
    (GARCH is known to be unstable/unreliable on small samples), constant
    returns (no variance to model), or scipy is not installed.
    """
    if forecast_horizon <= 0:
        raise ValidationError(f"forecast_horizon must be > 0, got {forecast_horizon}")
    _require_scipy("GARCH(1,1) maximum-likelihood fitting")

    cleaned = returns.dropna()
    arr = cleaned.to_numpy(dtype=float)
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

    # The rescaling below divides by the mean squared residual, and a
    # constant series has none -- numerically a rounding residue, so the
    # test is the library's relative one rather than `== 0`. Imported here:
    # `metrics` imports `analysis` at package level, so a module-level
    # import would close a cycle.
    from standard_quant_tools.metrics.risk_metrics import has_no_dispersion

    if has_no_dispersion(arr):
        raise ValidationError(
            "garch_volatility_forecast: every return is the same value, so "
            "there is no variance for GARCH(1,1) to model. Pass returns "
            "(not a constant or forward-filled series) with some movement."
        )
    scale = float(resid_sq.mean())
    logger.debug("[garch] n_obs=%d  scale=%.3e", n, scale)
    opt, scaled_params, grad = _fit_rescaled(resid_sq / scale)
    omega = float(scaled_params[0]) * scale
    alpha, beta = float(scaled_params[1]), float(scaled_params[2])
    persistence = alpha + beta
    gradient_norm = _projected_gradient_norm(scaled_params, grad) / n
    at_bound = _at_bound(scaled_params)
    # Three conditions, because each has failed alone: L-BFGS-B said
    # success while stopped far from the maximum, a maximum can sit on the
    # non-stationary side where the soft penalty only discourages it, and
    # an iteration limit leaves the gradient large with success False.
    converged = (
        bool(opt.success) and persistence < 1.0 and gradient_norm < _GRADIENT_TOL
    )

    warnings: List[str] = []
    if "alpha" in at_bound:
        warnings.append(
            f"alpha sits at its lower bound ({alpha:.1e}): this sample shows "
            "no ARCH effect -- yesterday's squared shock does not move "
            "today's variance -- and with alpha at zero beta is not "
            "identified, because the variance path is then the constant "
            "omega / (1 - beta) for any beta. The fitted beta and persistence "
            "carry no information; read the series as constant-variance."
        )
    if not converged:
        reasons = []
        if not opt.success:
            reasons.append(f"the optimizer stopped without success ({opt.message})")
        if persistence >= 1.0:
            reasons.append(
                f"persistence is {persistence:.6f} >= 1, so the unconditional "
                "variance does not exist and long_run_annualized_vol is an "
                "artifact of the 0.9999 clamp"
            )
        if gradient_norm >= _GRADIENT_TOL:
            reasons.append(
                f"the projected gradient is {gradient_norm:.1e} per "
                f"observation, above {_GRADIENT_TOL:.0e}, so the parameters "
                "are not at a maximum of the likelihood"
            )
        warnings.append("NOT CONVERGED: " + "; ".join(reasons) + ".")

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

    # A fit with persistence >= 1 is non-stationary: the unconditional
    # variance omega/(1-persistence) does not exist. The clamp below keeps the
    # forecast recursion finite, but it means long_run_annualized_vol is then
    # an artifact of the 0.9999 clamp (~omega*10000), NOT an estimated
    # quantity — `converged` is always False in that case, so check it before
    # using long_run_annualized_vol for anything.
    persistence_safe = min(persistence, 0.9999)
    long_run_var = omega / (1.0 - persistence_safe)

    # current_var above is already the (deterministic, not decayed) T+1
    # value, so forecast step h=1,2,...,horizon needs exponent h-1 -- h=0 at
    # the first output -- or forecast_annualized_vol[0] would silently apply
    # one spurious extra decay step and stop matching current_annualized_vol.
    h = np.arange(0, forecast_horizon, dtype=float)
    forecast_var = long_run_var + (persistence_safe**h) * (current_var - long_run_var)
    forecast_var = np.clip(forecast_var, _MIN_SIGMA2, None)

    diagnostics = _residual_diagnostics(resid, sigma2)

    result = {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": persistence,
        "converged": converged,
        # Scale-free: measured on the rescaled problem, so it reads the same
        # for returns in decimals or in percent.
        "gradient_norm": float(gradient_norm),
        "at_bound": at_bound,
        "log_likelihood": float(log_likelihood),
        "aic": float(aic),
        "bic": float(bic),
        "n_obs": n,
        "current_annualized_vol": float(np.sqrt(current_var * periods_per_year)),
        "long_run_annualized_vol": float(np.sqrt(long_run_var * periods_per_year)),
        "forecast_annualized_vol": [
            float(np.sqrt(v * periods_per_year)) for v in forecast_var
        ],
        # The conditional-variance path, one value per observation, on the
        # returns' own index. It was computed on every call and only its
        # last element survived; the recursion is what the fit is FOR, and
        # the residual diagnostics below are read off it.
        "conditional_variance": pd.Series(
            sigma2, index=cleaned.index, name="conditional_variance"
        ),
        **diagnostics,
        "warnings": warnings,
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
