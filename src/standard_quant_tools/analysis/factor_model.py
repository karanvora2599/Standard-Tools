"""
A factor model estimated from return series: the loadings B, the factor
covariance F and the specific variances D that `decompose_factor_risk`
takes, and the covariance they imply, Sigma = B F B' + D.

THE ESTIMATOR IS TIME-SERIES OLS, one regression per asset on the
caller's factor returns over the same dates:

    r_it = a_i + sum_k b_ik f_kt + e_it,

through `multi_factor.fit_least_squares`, the fit `multi_factor_regression`
runs -- the same lstsq, the same refusal of linearly dependent factors by
name, the same (X'X)^-1 for the standard errors -- so an asset's loadings
here are the bits that function gives on the same dates. The intercept is
on by default; without it the loadings also carry each asset's mean return.

WHAT EACH PIECE IS, and its standard error:

- B: the OLS slopes, each with sqrt(s_i^2 [(X'X)^-1]_jj), the standard
  error that assumes independent, homoskedastic residuals.
- F: the sample covariance of the factor returns on those dates (n - 1 in
  the denominator), with the normal-theory standard error of a sample
  covariance, sqrt((F_jk^2 + F_jj F_kk) / (n - 1)).
- D: each asset's residual variance with the degrees-of-freedom
  correction, s_i^2 = SSR_i / (n - p), p the number of coefficients
  (the factors, plus one for the intercept). Unbiased under the model,
  where SSR / n is not: with 5 factors on 60 months, SSR / n understates
  it by 10%. Its normal-theory standard error is s_i^2 sqrt(2 / (n - p)).

F and D are annualized by `periods_per_year`, as `estimate_covariance`
annualizes, and so is Sigma; the loadings are unitless. Nothing is
annualized twice: F and D are per-period moments times periods_per_year.

THE MODEL SETS THE RESIDUAL COVARIANCES TO ZERO. That is what makes D
diagonal and Sigma = B F B' + D, and it is an assumption, not a finding:
two share classes of one company, or two names in a factor the caller did
not supply, keep a residual correlation the model throws away. The largest
residual correlation between two assets in the sample is reported, so a
reader can see how far the data are from the assumption.

R-SQUARED IS IN-SAMPLE, the share of each asset's variance its own fitted
loadings explain on the dates they were fitted to. With the intercept off
it is still the centred R-squared, and it can be negative.

Sigma is built as (B F) B', averaged with its own transpose so it is
symmetric to the last bit, plus D on the diagonal, on one BLAS thread. It
is positive semidefinite: F is a sample covariance and every d_i >= 0.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.analysis.multi_factor import fit_least_squares, inverse_gram
from standard_quant_tools.error import ValidationError

__all__ = ["estimate_factor_model"]

#: How many offending names a refusal lists before counting the rest.
_SHOWN = 5

#: Fewer observations than this per coefficient, and the result says so.
_THIN_OBSERVATIONS_PER_COEFFICIENT = 10

#: A residual correlation this large or larger is named in the warnings.
_LARGE_RESIDUAL_CORRELATION = 0.5


def _listed(items: List[Any]) -> str:
    shown = ", ".join(repr(i) for i in items[:_SHOWN])
    more = len(items) - _SHOWN
    return f"[{shown}{f' and {more} more' if more > 0 else ''}]"


def _frame(value: Any, field: str, who: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise ValidationError(
            f"{who}: {field} is a {type(value).__name__}; give a frame of "
            "returns with one column per series and one row per date."
        )
    if value.shape[1] == 0:
        raise ValidationError(f"{who}: {field} has no columns.")
    names = [str(c) for c in value.columns]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValidationError(
            f"{who}: {field} names {_listed(repeated)} more than once; each "
            "column is one series."
        )
    out = value.copy()
    out.columns = names
    try:
        out = out.astype("float64")
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{who}: {field} holds values that are not numbers ({exc})."
        ) from exc
    bad = ~np.isfinite(out.to_numpy())
    if bad.any():
        rows, cols = np.nonzero(bad)
        cells = [f"({out.index[r]}, {names[c]})" for r, c in zip(rows[:3], cols[:3])]
        raise ValidationError(
            f"{who}: {field} is not finite at {int(bad.sum())} cell(s), e.g. "
            f"{', '.join(cells)}. Every series needs a finite return on every "
            "date used; align the series on the dates they share first."
        )
    return out


def estimate_factor_model(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    *,
    intercept: bool = True,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    """
    B, F and D of a time-series factor model, with their standard errors,
    and the covariance they imply.

    Args:
        asset_returns: one column per asset, one row per date.
        factor_returns: one column per factor, on exactly the same dates
            (align the two first; nothing is filled or dropped here).
        intercept: regress with an intercept (alpha). On by default.
        periods_per_year: what annualizes F, D and Sigma -- 252 for daily
            returns, 52 weekly, 12 monthly.

    Returns a dict: `assets`, `factors`, `n_observations`,
    `n_coefficients`, `degrees_of_freedom`, `intercept`,
    `periods_per_year`; per asset `loadings`, `loading_standard_errors`,
    `loading_t_stats` (asset -> factor -> value), `alpha`,
    `alpha_standard_error`, `alpha_t_stat` (per period; None without the
    intercept), `r_squared` (in-sample); `factor_covariance` and
    `factor_covariance_standard_errors` (factor -> factor -> value,
    annualized); `specific_variance` and
    `specific_variance_standard_errors` (asset -> value, annualized);
    `covariance` (Sigma = B F B' + D as rows parallel to `assets`,
    annualized); the largest residual correlation between two assets and
    the pair; and `warnings`.

    Refused by name: a frame that is not one, an empty or repeated column,
    a name used both as an asset and as a factor, frames on different
    dates, a non-finite value, too few dates for a residual variance (n
    must exceed the number of coefficients), and linearly dependent
    factors.
    """
    who = "estimate_factor_model"
    if isinstance(periods_per_year, bool) or not isinstance(
        periods_per_year, (int, np.integer)
    ):
        raise ValidationError(
            f"{who}: periods_per_year must be a whole number of periods a year "
            f"(252 daily, 12 monthly), got {periods_per_year!r}."
        )
    if int(periods_per_year) < 1:
        raise ValidationError(
            f"{who}: periods_per_year={periods_per_year} is not a number of "
            "periods in a year; it is at least 1 (252 daily, 12 monthly)."
        )
    ppy = int(periods_per_year)
    assets_frame = _frame(asset_returns, "asset_returns", who)
    factors_frame = _frame(factor_returns, "factor_returns", who)
    assets = list(assets_frame.columns)
    factors = list(factors_frame.columns)
    both = [a for a in assets if a in set(factors)]
    if both:
        raise ValidationError(
            f"{who}: {_listed(both)} named both as an asset and as a factor. "
            "An asset regressed on itself has a loading of exactly 1 and no "
            "specific risk, which is an identity rather than an estimate; "
            "rename one or leave it out of one side."
        )
    if not assets_frame.index.equals(factors_frame.index):
        raise ValidationError(
            f"{who}: asset_returns ({len(assets_frame)} rows) and "
            f"factor_returns ({len(factors_frame)} rows) are not on the same "
            "dates. A regression pairs each asset return with the factor "
            "returns of the same date; align them on the dates they share "
            "first (align_return_streams), so what was dropped is reported."
        )
    n = int(len(assets_frame))
    k = len(factors)
    p = k + (1 if intercept else 0)
    dof = n - p
    if dof < 1:
        raise ValidationError(
            f"{who}: {n} dates for {p} coefficient(s) "
            f"({k} factor(s){' and the intercept' if intercept else ''}) leave "
            f"{max(dof, 0)} degrees of freedom, and a residual variance needs "
            "at least one. Give more dates or fewer factors."
        )

    Y = assets_frame.to_numpy(dtype=float)
    Fr = factors_frame.to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), Fr]) if intercept else Fr.copy()
    columns = (["intercept"] if intercept else []) + factors
    m = len(assets)
    nan = float("nan")

    # One BLAS thread for every fit, Gram inverse, product and
    # decomposition, so each number is the same bits whatever thread count
    # the caller's BLAS has.
    with single_threaded_blas():
        coefficients = np.empty((m, p))
        residuals = np.empty((n, m))
        ss_res = np.empty(m)
        for i in range(m):
            beta, resid, ssr = fit_least_squares(
                X, np.ascontiguousarray(Y[:, i]), columns, who, intercept=intercept
            )
            coefficients[i] = beta
            residuals[:, i] = resid
            ss_res[i] = ssr
        XtX_inv = inverse_gram(X)
        factor_cov = np.atleast_2d(np.cov(Fr, rowvar=False, ddof=1))
        residual_corr = None
        if m > 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                residual_corr = np.atleast_2d(np.corrcoef(residuals, rowvar=False))

        s2 = ss_res / dof
        if XtX_inv is not None:
            diag = np.diag(XtX_inv)
            standard_errors = np.sqrt(np.maximum(diag[None, :] * s2[:, None], 0.0))
        else:
            standard_errors = np.full((m, p), nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            t_stats = np.where(
                standard_errors > 0, coefficients / standard_errors, np.nan
            )

        B = coefficients[:, 1:] if intercept else coefficients
        F = factor_cov * ppy
        d = s2 * ppy
        BF = B @ F
        S = BF @ B.T
        sigma = (S + S.T) / 2.0
        sigma[np.diag_indices(m)] += d

    # Normal-theory standard errors of the sample moments, in the annual
    # units the moments are reported in.
    vol = np.diag(factor_cov)
    with np.errstate(invalid="ignore"):
        factor_cov_se = (
            np.sqrt((factor_cov**2 + np.outer(vol, vol)) / max(n - 1, 1)) * ppy
        )
    d_se = d * math.sqrt(2.0 / dof)

    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_squared = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
    # A constant asset has no variance to explain, whatever the rounding of
    # its centred sum of squares says (calculate_beta's rule).
    for i in range(m):
        if bool(np.all(Y[:, i] == Y[0, i])):
            r_squared[i] = nan

    offset = 1 if intercept else 0
    loadings = {
        a: {f: float(B[i, j]) for j, f in enumerate(factors)}
        for i, a in enumerate(assets)
    }
    loading_se = {
        a: {f: float(standard_errors[i, j + offset]) for j, f in enumerate(factors)}
        for i, a in enumerate(assets)
    }
    loading_t = {
        a: {f: float(t_stats[i, j + offset]) for j, f in enumerate(factors)}
        for i, a in enumerate(assets)
    }
    alpha = alpha_se = alpha_t = None
    if intercept:
        alpha = {a: float(coefficients[i, 0]) for i, a in enumerate(assets)}
        alpha_se = {a: float(standard_errors[i, 0]) for i, a in enumerate(assets)}
        alpha_t = {a: float(t_stats[i, 0]) for i, a in enumerate(assets)}

    largest = nan
    pair: Optional[List[str]] = None
    if residual_corr is not None:
        upper = np.triu_indices(m, 1)
        values = np.abs(residual_corr[upper])
        finite = np.isfinite(values)
        if finite.any():
            where = int(np.flatnonzero(finite)[np.argmax(values[finite])])
            i, j = int(upper[0][where]), int(upper[1][where])
            largest = float(residual_corr[i, j])
            pair = [assets[i], assets[j]]

    warnings: List[str] = [
        "In-sample: the loadings, R-squared and residual variances are "
        "fitted and read on the same dates.",
        "The loadings' standard errors and t-statistics assume independent, "
        "homoskedastic residuals, and those of F and D normal returns. Daily "
        "residuals are neither, so read them as a scale for the estimate, "
        "not a test.",
    ]
    if n < _THIN_OBSERVATIONS_PER_COEFFICIENT * p:
        warnings.append(
            f"{n} dates for {p} coefficients per asset is fewer than "
            f"{_THIN_OBSERVATIONS_PER_COEFFICIENT} per coefficient: the "
            "loadings and residual variances are estimated loosely, and their "
            "standard errors say how loosely."
        )
    if pair is not None and abs(largest) >= _LARGE_RESIDUAL_CORRELATION:
        warnings.append(
            f"The residuals of {pair[0]!r} and {pair[1]!r} correlate at "
            f"{largest:.2f}. The model's D is diagonal -- it sets every "
            "residual covariance to zero -- so Sigma understates how those two "
            "move together; a factor they share is missing from the model."
        )
    if not intercept:
        warnings.append(
            "No intercept: each loading also carries its asset's mean return "
            "where the factors have a mean, the residuals need not average "
            "zero, and R-squared (still centred) can be negative."
        )

    return {
        "assets": assets,
        "factors": factors,
        "intercept": bool(intercept),
        "periods_per_year": ppy,
        "n_observations": n,
        "n_coefficients": p,
        "degrees_of_freedom": dof,
        "loadings": loadings,
        "loading_standard_errors": loading_se,
        "loading_t_stats": loading_t,
        "alpha": alpha,
        "alpha_standard_error": alpha_se,
        "alpha_t_stat": alpha_t,
        "r_squared": {a: float(r_squared[i]) for i, a in enumerate(assets)},
        "factor_covariance": {
            r: {c: float(F[x, y]) for y, c in enumerate(factors)}
            for x, r in enumerate(factors)
        },
        "factor_covariance_standard_errors": {
            r: {c: float(factor_cov_se[x, y]) for y, c in enumerate(factors)}
            for x, r in enumerate(factors)
        },
        "specific_variance": {a: float(d[i]) for i, a in enumerate(assets)},
        "specific_variance_standard_errors": {
            a: float(d_se[i]) for i, a in enumerate(assets)
        },
        "specific_variance_basis": (
            f"SSR / (n - p) = SSR / {dof}: n = {n} dates, p = {p} coefficients "
            f"({k} factor(s){' and the intercept' if intercept else ''}), "
            f"times {ppy} periods a year"
        ),
        "covariance": sigma.tolist(),
        "largest_residual_correlation": largest,
        "largest_residual_correlation_pair": pair,
        "warnings": warnings,
    }
