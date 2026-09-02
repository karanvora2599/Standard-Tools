import logging
import math
from typing import Any, Dict

import numpy as np
import pandas as pd

from standard_quant_tools._special import norm_cdf_array
from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import require_finite_array

logger = logging.getLogger(__name__)

_scipy_stats = None
try:
    from scipy import stats as _scipy_stats  # type: ignore[assignment]

    HAS_SCIPY = True
except ImportError:
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

_sqrt2 = math.sqrt(2.0)
_math_erf = math.erf


# See `_special`: this had 7 copies across the library, and the ones
# that were not identical disagreed at the edge of the domain.
_norm_cdf = norm_cdf_array


def multi_factor_regression(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
) -> Dict[str, Any]:
    """
    OLS regression of asset_returns on N factors.

    Parameters
    ----------
    asset_returns : pd.Series
        Daily (or periodic) returns of the asset being analysed.
    factor_returns : pd.DataFrame
        One column per factor (e.g. market, SMB, HML). Index must overlap
        with asset_returns — alignment is handled automatically.

    Returns
    -------
    dict with keys:
        alpha         : float  – per-period OLS intercept (raw coefficient; multiply by periods_per_year to annualize)
        loadings      : dict   – {factor_name: coefficient}
        t_stats       : dict   – {factor_name: t-statistic} (includes "alpha")
        p_values      : dict   – {factor_name: two-tailed p-value}
        r_squared     : float
        adj_r_squared : float
        n_obs         : int

    Note
    ----
    p_values use the t-distribution with `n - k` degrees of freedom when
    scipy is installed, and a normal approximation otherwise. The two agree
    closely for large samples but differ materially for small ones — check
    `n_obs` before comparing p-values computed in different environments.
    """
    common_idx = asset_returns.index.intersection(factor_returns.index)
    y = asset_returns.loc[common_idx].to_numpy(dtype=float)
    X_f = factor_returns.loc[common_idx].to_numpy(dtype=float)
    factor_names = list(factor_returns.columns)
    # Same finite-input contract rolling_factor_loadings below already
    # enforces — without it lstsq quietly returns all-NaN coefficients that
    # look like a completed regression.
    require_finite_array(y, "asset_returns", "multi_factor_regression")
    require_finite_array(X_f, "factor_returns", "multi_factor_regression")

    n = len(y)
    k = X_f.shape[1] + 1  # +1 for intercept
    cdf_path = "scipy" if HAS_SCIPY else "math.erf"
    logger.debug(
        "[multi_factor] factors=%s  n_obs=%d  cdf=%s", factor_names, n, cdf_path
    )

    _nan = float("nan")
    if n < k + 1:
        nan_factors = {name: _nan for name in factor_names}
        nan_all = {"alpha": _nan, **nan_factors}
        return {
            "alpha": _nan,
            "loadings": nan_factors,
            "t_stats": nan_all,
            "p_values": nan_all,
            "r_squared": _nan,
            "adj_r_squared": _nan,
            "n_obs": n,
        }

    X = np.column_stack([np.ones(n), X_f])

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ beta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    dof = n - k
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / dof if dof > 0 else _nan

    # Standard errors: SE_i = sqrt(s² * [(X'X)^{-1}]_ii)
    s2 = ss_res / dof if dof > 0 else _nan
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.maximum(np.diag(XtX_inv) * s2, 0.0))
    except np.linalg.LinAlgError:
        se = np.full(k, _nan)

    t_vals = np.where(se > 0, beta / se, np.nan)

    if HAS_SCIPY and _scipy_stats is not None:
        p_vals = 2.0 * _scipy_stats.t.sf(np.abs(t_vals), df=dof)
    else:
        p_vals = 2.0 * (1.0 - _norm_cdf(np.abs(t_vals)))

    all_names = ["alpha"] + factor_names
    t_stats = {name: float(t_vals[i]) for i, name in enumerate(all_names)}
    p_values = {name: float(p_vals[i]) for i, name in enumerate(all_names)}

    result = {
        "alpha": float(beta[0]),
        "loadings": {name: float(beta[i + 1]) for i, name in enumerate(factor_names)},
        "t_stats": t_stats,
        "p_values": p_values,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "n_obs": n,
    }
    sig = [f for f in factor_names if p_values.get(f, 1.0) < 0.05]
    logger.debug(
        "[multi_factor] alpha=%.6f  R²=%.4f  adj_R²=%.4f  significant=%s",
        result["alpha"],
        r_squared,
        adj_r_squared,
        sig,
    )
    return result


def rolling_factor_loadings(
    asset_returns: pd.Series,
    factor_returns: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Rolling OLS factor loadings over a sliding window.

    Returns a DataFrame indexed like asset_returns with columns
    ["alpha", factor1, factor2, ...]. The first (window-1) rows are NaN.

    A `window` smaller than k+2 (fewer observations than the intercept plus
    k loadings being estimated) makes every window underdetermined, and the
    whole result is NaN — there is no unique least-squares solution to
    report, and the minimum-norm one numpy would return is an artifact of
    the solver, not an estimated loading.

    Uses C++ incremental Cholesky path when available (50-200× faster than
    the Python numpy.linalg.lstsq loop).
    """
    if window <= 0:
        raise ValidationError(f"window must be > 0, got {window}")
    common_idx = asset_returns.index.intersection(factor_returns.index)
    y_arr = asset_returns.loc[common_idx].to_numpy(dtype=np.float64)
    X_arr = np.ascontiguousarray(
        factor_returns.loc[common_idx].to_numpy(dtype=np.float64)
    )
    factor_names = list(factor_returns.columns)
    col_names = ["alpha"] + factor_names

    require_finite_array(y_arr, "asset_returns", "rolling_factor_loadings")
    require_finite_array(X_arr, "factor_returns", "rolling_factor_loadings")

    n = len(y_arr)
    k = X_arr.shape[1] if X_arr.ndim == 2 else 1

    # Underdetermined window: fewer observations than the k+1 coefficients
    # (intercept + k loadings) the regression has to estimate. Checked here,
    # ahead of the path dispatch, so BOTH backends answer identically -- the
    # C++ kernel already bails to all-NaN for window < k+2
    # (rolling_regression.cpp), while numpy.linalg.lstsq below happily
    # returns its minimum-norm solution instead, so the same call produced
    # NaN or numbers depending only on whether the extension was built. The
    # minimum-norm solution is a numerical artifact of an underdetermined
    # system, not an estimated factor loading, so NaN is the honest answer
    # and the one both paths now give.
    if window < k + 2:
        return pd.DataFrame(
            np.full((n, len(col_names)), np.nan), index=common_idx, columns=col_names
        )

    path = "C++" if (HAS_CPP and _cpp_core is not None) else "python"
    logger.debug(
        "[rolling_factor_loadings] window=%d  factors=%d  bars=%d  path=%s",
        window,
        k,
        n,
        path,
    )

    # ── C++ fast path ─────────────────────────────────────────────────────────
    if HAS_CPP and _cpp_core is not None:
        try:
            out = _cpp_core.rolling_factor_loadings(y_arr, X_arr, window)
            return pd.DataFrame(out, index=common_idx, columns=col_names)
        except Exception as exc:
            logger.warning(
                "[rolling_factor_loadings] C++ failed (%s) — using Python", exc
            )

    # ── Python fallback ───────────────────────────────────────────────────────
    # Rank-deficiency policy, shared with the C++ kernel: a window whose design
    # matrix is not full rank (duplicated factors, a factor that is a linear
    # combination of the others, a constant factor) yields NaN, not numbers.
    #
    # lstsq's minimum-norm solution for such a window is one arbitrary member of
    # an infinite solution set chosen by a norm criterion nobody asked for --
    # it is not an estimated factor loading, and reporting it as one invites
    # exactly the misreading the underdetermined-window guard above already
    # rejects for the same reason. The backends used to differ here: the C++
    # path returned NaN while this one returned the minimum-norm vector, so
    # `rolling_factor_loadings(y, F, window)` on duplicated factors gave
    # [nan, nan, nan, nan] or [0.0059, 1.0077, 1.0077, 0.0091] from the same
    # inputs depending only on whether the extension had been compiled.
    #
    # np.linalg.lstsq already computes the rank (SVD-based) and returns it, so
    # this costs nothing extra.
    n_coef = 1 + k
    out = np.full((n, len(col_names)), np.nan)
    for i in range(window - 1, n):
        y_w = y_arr[i - window + 1 : i + 1]
        X_w = X_arr[i - window + 1 : i + 1]
        X_des = np.column_stack([np.ones(window), X_w])
        beta, _residuals, rank, _sv = np.linalg.lstsq(X_des, y_w, rcond=None)
        if rank < n_coef:
            continue  # leave the row NaN
        out[i] = beta

    return pd.DataFrame(out, index=common_idx, columns=col_names)
