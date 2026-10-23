"""
Value at risk as a FORECAST: one-step-ahead VaR and expected shortfall over
a rolling window, each made from the returns before its date.

`var_historical`, `var_parametric` and `cvar` answer "what was the loss
quantile of this sample". A risk number is used as a statement about
TOMORROW, and a statement about tomorrow can be checked: line each day's
forecast up against the return that followed and count the days the loss
went through it. That needs a series of forecasts, each made with nothing
it could not have known, which is what this module produces and what
`metrics.var_backtest` tests.

THE ONE RULE: the forecast for date t is a function of the returns at
t - window, ..., t - 1 and of nothing after them. Every method below reads
its window as a slice that ends before t, the GARCH methods filter that same
slice, and a change to any return at or after t changes no forecast at or
before t (the tests perturb the future and compare).

VaR AND ES ARE LOSSES, POSITIVE, as `var_historical` and `cvar` give them:
an exception is a day whose return is below minus its VaR.

The methods:

- `historical` -- the window's empirical quantile and the mean of the
  returns at or below it: `var_historical` and `cvar` on the window, to the
  bit, with the tail count `cvar_tail_observations` reports.
- `normal` -- the window's mean and sample standard deviation:
  VaR = -mu + sigma z_c, ES = -mu + sigma phi(z_c) / (1 - c).
- `cornish_fisher` -- the normal quantile corrected for the window's
  skewness s and excess kurtosis k (Zangari 1996; Favre and Galeano 2002).
  The expansion is a quantile function only where it is monotone in z, and
  outside that region (Maillard 2012) it is refused rather than reported:
  its derivative 1 - k/8 + 5 s^2/36 + z s/3 + z^2 (k/8 - s^2/6) must stay
  non-negative for every z. ES is the exact tail mean of the distribution
  whose quantile function the expansion is.
- `ewma` -- RiskMetrics: zero mean, sigma^2_{t+1} = lambda sigma^2_t +
  (1 - lambda) r_t^2, seeded with the mean square of the first window, so
  the window is its warm-up.
- `filtered_historical` -- GARCH(1,1) standardized residuals of the window,
  their empirical quantile and tail mean rescaled by the one-step forecast
  volatility (Barone-Adesi, Giannopoulos and Vosper 1999).
- `evt_garch` -- the same filter, with a generalized Pareto tail fitted to
  the standardized losses above a threshold quantile (McNeil and Frey 2000),
  by `evt_tail_risk`.

The GARCH is the library's own (`analysis.garch`), refitted on the window
every `refit_every` forecasts and used to filter each window in between.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.analysis import garch as _garch
from standard_quant_tools.error import ValidationError

from .risk_metrics import (
    CVAR_MIN_TAIL_OBSERVATIONS,
    _check_confidence,
    _cvar_tail,
    evt_tail_risk,
)

__all__ = [
    "VAR_FORECAST_METHODS",
    "cornish_fisher_is_monotone",
    "cornish_fisher_tail_mean",
    "cornish_fisher_z",
    "forecast_var_series",
]

VAR_FORECAST_METHODS = (
    "historical",
    "normal",
    "cornish_fisher",
    "ewma",
    "filtered_historical",
    "evt_garch",
)

#: The shortest window any method takes. Below it a sample quantile at any
#: confidence worth forecasting is one or two points.
MIN_WINDOW = 20

#: The GARCH methods need what `garch_volatility_forecast` needs.
GARCH_MIN_WINDOW = 100

#: `evt_tail_risk` fits a tail only to this many exceedances or more.
EVT_MIN_EXCEEDANCES = 20

_GARCH_METHODS = ("filtered_historical", "evt_garch")


def _scipy_norm():
    from scipy import stats

    return stats.norm


# ── Cornish-Fisher ──────────────────────────────────────────────────────


def cornish_fisher_z(z: float, skew: float, excess_kurtosis: float) -> float:
    """The Cornish-Fisher quantile of a standardized variable at normal
    quantile z: z + (z^2-1)s/6 + (z^3-3z)k/24 - (2z^3-5z)s^2/36."""
    s, k = skew, excess_kurtosis
    return (
        z
        + (z * z - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * s * s / 36.0
    )


def cornish_fisher_is_monotone(skew: float, excess_kurtosis: float) -> bool:
    """
    Whether the expansion is increasing in z everywhere, i.e. a quantile
    function at all.

    Its derivative is the quadratic a z^2 + b z + c with a = k/8 - s^2/6,
    b = s/3 and c = 1 - k/8 + 5 s^2/36. It is non-negative for every z
    exactly when a >= 0 and b^2 <= 4ac (a = 0 needs b = 0 and c >= 0).
    Outside that region the expansion folds back on itself and the "VaR"
    it gives at some level is lower than at a lower level -- the domain of
    validity Maillard (2012) maps.
    """
    s, k = float(skew), float(excess_kurtosis)
    if not (math.isfinite(s) and math.isfinite(k)):
        return False
    a = k / 8.0 - s * s / 6.0
    b = s / 3.0
    c = 1.0 - k / 8.0 + 5.0 * s * s / 36.0
    if a < 0.0:
        return False
    if a == 0.0:
        return b == 0.0 and c >= 0.0
    return b * b <= 4.0 * a * c


def cornish_fisher_tail_mean(
    tail_probability: float, skew: float, excess_kurtosis: float
) -> float:
    """
    E[z_cf(Z) | Z < z_p] for a standard normal Z and p = tail_probability:
    the mean of the standardized Cornish-Fisher variable in its lower tail.

    With q(u) = z_cf(Phi^-1(u)) a quantile function (the expansion is
    monotone), (1/p) * integral_0^p q(u) du is this conditional mean, and
    each power of z integrates in closed form over the truncated normal:
    M0 = p, M1 = -phi(a), M2 = p - a phi(a), M3 = -(a^2 + 2) phi(a), with
    a = z_p.
    """
    norm = _scipy_norm()
    p = float(tail_probability)
    a = float(norm.ppf(p))
    phi = float(norm.pdf(a))
    m0, m1 = p, -phi
    m2 = p - a * phi
    m3 = -(a * a + 2.0) * phi
    s, k = float(skew), float(excess_kurtosis)
    total = (
        m1
        + (m2 - m0) * s / 6.0
        + (m3 - 3.0 * m1) * k / 24.0
        - (2.0 * m3 - 5.0 * m1) * s * s / 36.0
    )
    return total / p


def _moments(window: np.ndarray) -> Tuple[float, float, float, float]:
    """Mean, sample sd (ddof 1), skewness and EXCESS kurtosis, the last two
    from the window's own central moments (the 1/n estimator)."""
    mu = float(window.mean())
    sigma = float(window.std(ddof=1))
    centred = window - mu
    m2 = float(np.mean(centred**2))
    if m2 <= 0.0:
        return mu, sigma, float("nan"), float("nan")
    skew = float(np.mean(centred**3) / m2**1.5)
    kurt = float(np.mean(centred**4) / m2**2 - 3.0)
    return mu, sigma, skew, kurt


# ── input checks ────────────────────────────────────────────────────────


def _refuse_inputs(
    returns: Any,
    method: str,
    confidence: float,
    window: int,
    ewma_lambda: float,
    tail_fraction: float,
    refit_every: int,
    who: str,
) -> pd.Series:
    if method not in VAR_FORECAST_METHODS:
        raise ValidationError(
            f"{who}: method={method!r} is not one of {list(VAR_FORECAST_METHODS)}."
        )
    _check_confidence(confidence)
    if not isinstance(returns, pd.Series):
        raise ValidationError(
            f"{who}: returns must be a dated pandas Series, got "
            f"{type(returns).__name__}."
        )
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValidationError(
            f"{who}: returns must be indexed by date; a forecast is FOR a date, "
            "and an undated series has none to give it."
        )
    if returns.index.has_duplicates or not returns.index.is_monotonic_increasing:
        raise ValidationError(
            f"{who}: returns must be in date order with each date once; sort "
            "them and keep one return per date."
        )
    values = returns.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValidationError(
            f"{who}: returns hold a missing or non-finite value. A gap is a date "
            "the series does not have, not a NaN on it; drop those dates."
        )
    floor = GARCH_MIN_WINDOW if method in _GARCH_METHODS else MIN_WINDOW
    if isinstance(window, bool) or int(window) != window or window < floor:
        raise ValidationError(
            f"{who}: window={window!r} with method={method!r}; the window is at "
            f"least {floor} returns"
            + (
                " (a GARCH(1,1) fit is unstable on fewer)"
                if method in _GARCH_METHODS
                else ""
            )
            + ". Pass a longer window."
        )
    if len(values) <= window:
        raise ValidationError(
            f"{who}: {len(values)} returns and window={window}; the first "
            f"forecast is for the return after the first {window}, so at least "
            f"{window + 1} are needed. Pass a longer history or a shorter window."
        )
    if method == "ewma" and not (0.0 < ewma_lambda < 1.0):
        raise ValidationError(
            f"{who}: ewma_lambda={ewma_lambda!r} must be strictly between 0 and "
            "1; RiskMetrics' daily value is 0.94."
        )
    if method == "evt_garch":
        if not (0.0 < tail_fraction < 0.5):
            raise ValidationError(
                f"{who}: tail_fraction={tail_fraction!r} must be strictly between "
                "0 and 0.5; it is the share of standardized losses above the "
                "threshold the tail is fitted to."
            )
        if confidence <= 1.0 - tail_fraction:
            raise ValidationError(
                f"{who}: confidence={confidence} with tail_fraction="
                f"{tail_fraction}: the fitted tail starts at the "
                f"{1.0 - tail_fraction:.4f} quantile and can only say what lies "
                "beyond it. Raise confidence above that, or lower tail_fraction."
            )
    if method in _GARCH_METHODS and (
        isinstance(refit_every, bool)
        or int(refit_every) != refit_every
        or refit_every < 1
    ):
        raise ValidationError(
            f"{who}: refit_every={refit_every!r} must be a whole number of "
            "forecasts, at least 1 (1 refits the GARCH for every forecast)."
        )
    return pd.Series(values, index=returns.index)


# ── the methods, one window at a time ───────────────────────────────────


def _empirical(sample: np.ndarray, confidence: float) -> Tuple[float, float, int]:
    """(minus the quantile, minus the tail mean, tail size): `cvar`'s tail,
    so VaR is `var_historical`'s bits and ES is `cvar`'s."""
    threshold, tail = _cvar_tail(pd.Series(sample), confidence)
    es = float(-tail.mean()) if len(tail) > 0 else float(-threshold)
    return float(-threshold), es, int(len(tail))


class _GarchFilter:
    """The library's GARCH(1,1), fitted on a window and refitted every
    `refit_every` forecasts; each forecast filters its own window with the
    latest fit made from data before it."""

    def __init__(self, refit_every: int) -> None:
        self.refit_every = int(refit_every)
        self.fit: Optional[Dict[str, Any]] = None
        self.mu = 0.0
        self.n_fits = 0
        self.n_not_converged = 0
        self.n_alpha_at_bound = 0
        self.persistence: List[float] = []

    def step(self, window: np.ndarray, position: int) -> Tuple[np.ndarray, float]:
        """Standardized residuals of the window and the one-step forecast
        volatility for the date after it. `position` counts forecasts from 0;
        a refit happens at 0, refit_every, 2 * refit_every, ..."""
        if self.fit is None or position % self.refit_every == 0:
            fitted = _garch.garch_volatility_forecast(
                pd.Series(window), forecast_horizon=1
            )
            self.fit = fitted
            self.mu = float(window.mean())
            self.n_fits += 1
            self.n_not_converged += 0 if fitted["converged"] else 1
            self.n_alpha_at_bound += 1 if "alpha" in fitted["at_bound"] else 0
            self.persistence.append(float(fitted["persistence"]))
        fit = self.fit
        omega, alpha, beta = fit["omega"], fit["alpha"], fit["beta"]
        resid = window - self.mu
        resid_sq = resid**2
        sigma2 = _garch._garch11_variance_recursion(resid_sq, omega, alpha, beta)
        # One recursion step past the window: the variance of the date after
        # its last return, as garch_volatility_forecast takes it.
        next_var = float(omega + alpha * resid_sq[-1] + beta * sigma2[-1])
        return resid / np.sqrt(sigma2), math.sqrt(max(next_var, _garch._MIN_SIGMA2))


def forecast_var_series(
    returns: pd.Series,
    *,
    method: str = "historical",
    confidence: float = 0.99,
    window: int = 250,
    ewma_lambda: float = 0.94,
    tail_fraction: float = 0.10,
    refit_every: int = 20,
    who: str = "forecast_var_series",
) -> Dict[str, Any]:
    """
    One-step-ahead VaR and ES for every date after the first `window`, each
    from the `window` returns before it, and the forecast for the period
    after the last return.

    Parameters
    ----------
    returns       : dated Series of returns (not levels), in date order.
    method        : one of VAR_FORECAST_METHODS (see the module docstring).
    confidence    : the VaR level c, strictly between 0 and 1.
    window        : returns per estimate; at least 20, and 100 for the
        GARCH methods. For `ewma` it is the warm-up that seeds the variance.
    ewma_lambda   : `ewma`'s decay, strictly between 0 and 1.
    tail_fraction : `evt_garch`'s share of standardized losses above the
        threshold; confidence must exceed 1 - tail_fraction.
    refit_every   : the GARCH methods refit every this many forecasts.

    Returns
    -------
    dict with `forecasts` (a DataFrame on the forecast dates with columns
    `realized`, `var`, `es`, `confidence`), `next_var`, `next_es`, the
    counts and dates, the method's diagnostics and `warnings`.

    Raises
    ------
    ValidationError: an unknown method, a level out of range, an undated,
    unsorted or non-finite series, a window too short for the method or
    longer than the history, `cornish_fisher` on a window whose skewness
    and kurtosis fall outside the expansion's monotone region, or
    `evt_garch` on a window with fewer than 20 exceedances.
    """
    series = _refuse_inputs(
        returns,
        method,
        confidence,
        window,
        ewma_lambda,
        tail_fraction,
        refit_every,
        who,
    )
    window = int(window)
    values = series.to_numpy(dtype=np.float64)
    n = len(values)
    p = 1.0 - float(confidence)
    norm = _scipy_norm()
    z_c = float(norm.ppf(confidence))
    # The forecast for position t reads values[t - window : t]; t = n is the
    # forecast for the period after the last return, which has no date.
    positions = range(window, n + 1)
    var = np.empty(len(positions))
    es = np.empty(len(positions))
    warnings: List[str] = []
    diagnostics: Dict[str, Any] = {}
    tail_counts: List[int] = []

    if method == "historical":
        for i, t in enumerate(positions):
            var[i], es[i], count = _empirical(values[t - window : t], confidence)
            tail_counts.append(count)
    elif method == "normal":
        z_low = float(norm.ppf(p))
        density = float(norm.pdf(z_low))
        for i, t in enumerate(positions):
            sample = values[t - window : t]
            mu, sigma = float(sample.mean()), float(sample.std(ddof=1))
            var[i] = -(mu + z_low * sigma)
            es[i] = -mu + sigma * density / p
    elif method == "cornish_fisher":
        z_low = float(norm.ppf(p))
        outside: List[Tuple[int, float, float]] = []
        skews, kurts = [], []
        for i, t in enumerate(positions):
            mu, sigma, skew, kurt = _moments(values[t - window : t])
            skews.append(skew)
            kurts.append(kurt)
            if not cornish_fisher_is_monotone(skew, kurt):
                outside.append((t, skew, kurt))
                continue
            var[i] = -(mu + sigma * cornish_fisher_z(z_low, skew, kurt))
            es[i] = -(mu + sigma * cornish_fisher_tail_mean(p, skew, kurt))
        if outside:
            t0, s0, k0 = outside[0]
            when = (
                f"the forecast for {series.index[t0].date()}"
                if t0 < n
                else "the forecast after the last return"
            )
            if not (math.isfinite(s0) and math.isfinite(k0)):
                raise ValidationError(
                    f"{who}: method='cornish_fisher' has no skewness or kurtosis "
                    f"in the window of {when}: every return in it is the same, so "
                    "there is no shape to correct the normal quantile by. Pass "
                    "returns with some movement."
                )
            raise ValidationError(
                f"{who}: method='cornish_fisher' is refused on {len(outside)} of "
                f"{len(positions)} windows, first {when}: skewness {s0:.3f} and "
                f"excess kurtosis {k0:.3f} lie outside the region where the "
                "Cornish-Fisher expansion is monotone (Maillard 2012), so it is "
                "not a quantile function there and its VaR would be a number "
                "with no distribution behind it. A window with negative excess "
                "kurtosis is always outside (the cubic term turns the tails "
                "back), and method='normal' fits one; method='historical' and "
                "'filtered_historical' make no expansion at all."
            )
        diagnostics.update(
            skewness_min=float(np.min(skews)),
            skewness_max=float(np.max(skews)),
            excess_kurtosis_min=float(np.min(kurts)),
            excess_kurtosis_max=float(np.max(kurts)),
        )
    elif method == "ewma":
        density = float(norm.pdf(z_c))
        lam = float(ewma_lambda)
        variance = float(np.mean(values[:window] ** 2))
        diagnostics["ewma_seed_variance"] = variance
        for i, t in enumerate(positions):
            if i > 0:
                variance = lam * variance + (1.0 - lam) * values[t - 1] ** 2
            sigma = math.sqrt(variance)
            var[i] = sigma * z_c
            es[i] = sigma * density / p
        diagnostics["ewma_lambda"] = lam
    else:
        garch = _GarchFilter(refit_every)
        exceedances: List[int] = []
        shapes: List[float] = []
        n_unbounded = 0
        for i, t in enumerate(positions):
            standardized, sigma = garch.step(values[t - window : t], i)
            mu = garch.mu
            # var_z and es_z are the standardized LOSS quantile and tail mean:
            # r = mu + sigma z, so VaR = -mu + sigma var_z and likewise ES.
            if method == "filtered_historical":
                var_z, es_z, count = _empirical(standardized, confidence)
                tail_counts.append(count)
            else:
                losses = -standardized
                threshold = float(np.percentile(losses, (1.0 - tail_fraction) * 100))
                count = int(np.sum(losses > threshold))
                if count < EVT_MIN_EXCEEDANCES:
                    when = (
                        f"the forecast for {series.index[t].date()}"
                        if t < n
                        else "the forecast after the last return"
                    )
                    raise ValidationError(
                        f"{who}: method='evt_garch' has {count} standardized losses "
                        f"above the {1.0 - tail_fraction:.2f} quantile in the window "
                        f"of {when}, and a generalized Pareto tail is fitted to at "
                        f"least {EVT_MIN_EXCEEDANCES}. Use a longer window (about "
                        f"{math.ceil(EVT_MIN_EXCEEDANCES / tail_fraction) + 1} at "
                        f"tail_fraction={tail_fraction}) or a larger tail_fraction."
                    )
                fitted = evt_tail_risk(
                    pd.Series(standardized),
                    confidence=confidence,
                    tail_fraction=tail_fraction,
                )
                exceedances.append(int(fitted["n_exceedances"]))
                shapes.append(float(fitted["shape_xi"]))
                var_z = float(fitted["var_evt"])
                es_z = float(fitted["cvar_evt"])
                if not math.isfinite(es_z):
                    n_unbounded += 1
            var[i] = -mu + sigma * var_z
            es[i] = -mu + sigma * es_z if math.isfinite(es_z) else np.nan
        diagnostics.update(
            garch_fits=garch.n_fits,
            garch_not_converged=garch.n_not_converged,
            garch_persistence=garch.persistence[-1],
            refit_every=garch.refit_every,
        )
        if garch.n_not_converged:
            warnings.append(
                f"{garch.n_not_converged} of {garch.n_fits} GARCH(1,1) fits did not "
                "converge; the forecasts between them use those fits' parameters "
                "anyway. run_garch_volatility_forecast on one window says why."
            )
        if garch.n_alpha_at_bound:
            warnings.append(
                f"{garch.n_alpha_at_bound} of {garch.n_fits} GARCH fits put alpha at "
                "its lower bound: those windows show no ARCH effect, and their "
                "filter is a constant volatility."
            )
        if method == "evt_garch":
            diagnostics.update(
                min_exceedances=int(min(exceedances)),
                gpd_shape_min=float(min(shapes)),
                gpd_shape_max=float(max(shapes)),
                n_es_unbounded=n_unbounded,
                tail_fraction=float(tail_fraction),
            )
            if n_unbounded:
                warnings.append(
                    f"es is null on {n_unbounded} forecast(s): the fitted tail's "
                    "shape is 1 or more there, so the tail has no mean and the "
                    "expected shortfall does not exist."
                )

    if tail_counts:
        fewest = int(min(tail_counts))
        diagnostics["min_tail_observations"] = fewest
        if fewest < CVAR_MIN_TAIL_OBSERVATIONS:
            warnings.append(
                f"es averages as few as {fewest} of {window} "
                f"{'returns' if method == 'historical' else 'standardized residuals'}"
                f" at {confidence:.0%} confidence -- fewer than "
                f"{CVAR_MIN_TAIL_OBSERVATIONS}, so each is the mean of a handful of "
                "days and moves with any one of them. A longer window or a lower "
                "confidence gives it more."
            )

    dates = series.index[window:]
    frame = pd.DataFrame(
        {
            "realized": values[window:],
            "var": var[:-1],
            "es": es[:-1],
            "confidence": np.full(n - window, float(confidence)),
        },
        index=dates,
    )
    frame.index.name = "date"
    return {
        "method": method,
        "confidence": float(confidence),
        "window": window,
        "n_returns": n,
        "n_forecasts": int(n - window),
        "first_forecast_date": dates[0],
        "last_forecast_date": dates[-1],
        "forecasts": frame,
        "next_var": float(var[-1]),
        "next_es": float(es[-1]),
        "mean_var": float(np.mean(var[:-1])),
        "mean_es": float(np.mean(es[:-1])) if np.all(np.isfinite(es[:-1])) else None,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }
