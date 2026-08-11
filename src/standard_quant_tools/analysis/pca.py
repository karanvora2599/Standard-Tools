import logging
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_POWER_ITERATION_TOL = 1e-9
_POWER_ITERATION_MAX_ITER = 100


def _top_k_pc_power_iteration(
    arr: np.ndarray, n_comp: int, tol: float, max_iter: int
) -> tuple:
    """
    Top-`n_comp` principal components of the already mean-centered
    (and, if requested, standardized) n_obs x n_assets matrix `arr`, via
    power iteration + deflation applied directly to `arr` -- never forms
    the n_assets x n_assets covariance matrix explicitly. Forming it
    costs O(n_obs * n_assets^2), which for a wide matrix (n_assets >
    n_obs -- the common case: a few hundred trading days, a large
    universe) is MORE expensive than SVD itself and would defeat the
    entire reason to use power iteration. Each matrix-vector product
    `cov @ v` is instead computed as `arr.T @ (arr @ v) / (n_obs - 1)` --
    O(n_obs * n_assets) per iteration regardless of matrix shape.

    Deterministic (fixed start vector, no random_state needed), but NOT a
    uniform start. "Converges from almost any start" excludes starts
    orthogonal to the dominant eigenvector, and the uniform vector
    [1,...,1]/sqrt(n) is exactly orthogonal to one of the most common
    structures in real return data: a spread/long-short factor with
    loadings proportional to [1,-1,...]. There the very first matrix-
    vector product is the zero vector, the loop breaks on `norm == 0`, and
    the routine reports the ZERO-eigenvalue direction as PC1. On a
    two-asset spread this returned an explained-variance ratio of ~0.0001
    where SVD returned ~0.9999 -- silently the wrong component, not a less
    precise one.

    A fixed deterministic pseudo-random start has no such adversarial
    alignment (the probability of exact orthogonality is zero, and the
    vector is identical on every run and every machine because the seed is
    hardcoded). Convergence is then verified rather than assumed: the
    eigenpair residual ||Av - lambda*v|| is checked against the component's
    own scale, and `converged=False` is returned so the caller can fall
    back to SVD instead of trusting whatever state iteration happened to
    stop in.

    Deflation happens in data space (`working -= outer(working @ v, v)`)
    rather than on a covariance matrix, the standard Hotelling-deflation
    identity: this is exactly equivalent to deflating cov by
    `eigenvalue * outer(v, v)` when v is a true unit eigenvector, without
    ever forming cov.

    Returns (eigenvectors, eigenvalues, total_variance, converged):
    eigenvectors as an (n_comp x n_assets) array (row i unit-norm),
    eigenvalues as a length-n_comp array, the FULL matrix's total variance
    (needed for explained_variance_ratio, since power iteration never
    computes the full spectrum the way SVD does), and a bool that is False
    if ANY component failed its residual check.
    """
    n_obs, n_assets = arr.shape
    denom = n_obs - 1
    total_var = float(np.sum(arr * arr) / denom)  # trace(cov) without forming cov
    working = arr
    vecs = np.empty((n_comp, n_assets))
    vals = np.empty(n_comp)
    converged = True
    # Fixed seed: deterministic across runs/machines, but not aligned with
    # any structure the data might have (unlike the uniform vector).
    start = np.random.default_rng(0).standard_normal(n_assets)
    start /= np.linalg.norm(start)
    for k in range(n_comp):
        v = start.copy()
        for _ in range(max_iter):
            v_new = (working.T @ (working @ v)) / denom
            norm = np.linalg.norm(v_new)
            if norm == 0.0:
                # Genuinely no remaining variance in any direction (fully
                # deflated / zero matrix) -- distinct from the old uniform-
                # start bug, where this branch fired because the START was
                # orthogonal to a component that very much existed.
                break
            v_new /= norm
            if np.linalg.norm(v_new - v) < tol:
                v = v_new
                break
            v = v_new
        av = (working.T @ (working @ v)) / denom
        eigenvalue = float(v @ av)
        # Residual check: v is only an eigenvector if Av == lambda*v.
        # Scaled by the eigenvalue so the tolerance is relative, and
        # skipped when the eigenvalue is ~0 (a genuinely null direction has
        # nothing to converge to and no meaningful relative scale).
        residual = float(np.linalg.norm(av - eigenvalue * v))
        if abs(eigenvalue) > tol and residual > max(1e-6 * abs(eigenvalue), tol):
            converged = False
        vecs[k] = v
        vals[k] = eigenvalue
        if k < n_comp - 1:
            working = working - np.outer(working @ v, v)
    return vecs, vals, total_var, converged


def pca_returns(
    returns_df: pd.DataFrame,
    n_components: Optional[int] = None,
    standardize: bool = True,
    method: Literal["svd", "power_iteration"] = "svd",
) -> Dict[str, Any]:
    """
    Principal Component Analysis on a multi-asset return matrix.

    Uses full SVD (pure NumPy) by default — no sklearn or statsmodels
    required.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily (or periodic) returns, one column per asset. Rows with any
        NaN are dropped before fitting.
    n_components : int, optional
        Number of principal components to return. Defaults to all
        (min(n_assets, n_obs)).
    standardize : bool
        When True (default), each asset column is scaled to unit variance
        before PCA so that high-vol assets don't dominate. Set to False
        only when columns are already on a comparable scale.
    method : "svd" (default) or "power_iteration". "svd" computes every
        singular triplet via np.linalg.svd, exact prior behavior for
        every existing caller. "power_iteration" computes only the
        requested `n_components` via power iteration + deflation applied
        directly to the (mean-centered, optionally standardized) return
        matrix -- meaningfully cheaper when `n_components` is small
        relative to min(n_assets, n_obs) (e.g. a rolling PC1-only feature
        refit many times over a large universe), since SVD's cost doesn't
        depend on how many components you actually want but power
        iteration's does. Loadings/factor_returns/explained_variance_ratio
        are numerically equivalent between methods for any component
        whose eigenvalue is well-separated from its neighbors (true of
        PC1 for real, factor-structured market data -- the only component
        every current caller of "power_iteration" actually requests).
        Nearby or tied eigenvalues (e.g. PC2/PC3 of near-idiosyncratic-
        noise data) make the corresponding eigenvectors numerically
        unstable for either method -- eigenvalues still agree tightly,
        but the specific orthonormal basis chosen within a near-
        degenerate subspace can differ between methods. This is an
        inherent property of PCA near degenerate eigenvalues, not a bug
        in either path.

    Returns
    -------
    dict with keys:
        explained_variance_ratio  : pd.Series  – EVR per PC, indexed "PC1", "PC2", ...
        cumulative_variance_ratio : pd.Series  – cumulative EVR
        loadings                  : pd.DataFrame – (assets × components)
        factor_returns            : pd.DataFrame – (dates × components)
        n_components              : int
        n_obs                     : int
    """
    data = returns_df.dropna()
    n_obs, n_assets = data.shape
    logger.debug(
        "[pca] assets=%d  obs=%d  n_components=%s  standardize=%s  method=%s",
        n_assets,
        n_obs,
        n_components,
        standardize,
        method,
    )

    if n_obs < 2 or n_assets < 1:
        raise ValueError(
            f"Need at least 2 observations and 1 asset; got ({n_obs}, {n_assets})."
        )
    # `method` is only a type ANNOTATION -- nothing enforced it at runtime,
    # so any unrecognized string silently fell through to the SVD branch
    # and returned a result the caller never asked for.
    if method not in ("svd", "power_iteration"):
        raise ValueError(f"method must be 'svd' or 'power_iteration', got {method!r}.")
    if n_components is not None and n_components < 1:
        raise ValueError(f"n_components must be >= 1 when given, got {n_components}.")

    arr = data.to_numpy(dtype=float)
    arr = arr - arr.mean(axis=0)

    if standardize:
        stds = arr.std(axis=0, ddof=1)
        stds[stds == 0] = 1.0
        arr = arr / stds

    n_comp = min(
        n_components if n_components is not None else n_assets,
        n_assets,
        n_obs,
    )

    if method == "power_iteration":
        Vt, eigenvalues, total_var, converged = _top_k_pc_power_iteration(
            arr, n_comp, _POWER_ITERATION_TOL, _POWER_ITERATION_MAX_ITER
        )
        if not converged:
            # Fall back rather than return an unconverged eigenpair. Power
            # iteration is an optimization, not a different definition of
            # PCA -- if it can't verify its own answer (weakly separated
            # PC1/PC2, accumulated deflation error across many components),
            # the correct result is still SVD's.
            logger.warning(
                "[pca] power_iteration failed its residual check after %d iterations "
                "— falling back to SVD for a verified decomposition",
                _POWER_ITERATION_MAX_ITER,
            )
            method = "svd"

    # Not `else` on the branch above: `method` may have just been switched
    # to "svd" by the convergence fallback, and that switch must actually
    # take effect here.
    if method == "svd":
        _, s, Vt_full = np.linalg.svd(arr, full_matrices=False)
        full_eigenvalues = s**2 / (n_obs - 1)
        Vt = Vt_full[:n_comp]
        eigenvalues = full_eigenvalues[:n_comp]
        # Denominator for explained_variance_ratio must be the FULL
        # spectrum's total, not just the n_comp components kept below --
        # matches power_iteration's total_var, which is also the full
        # matrix's total variance despite only ever solving for n_comp
        # components.
        total_var = float(full_eigenvalues.sum())

    # Sign convention: flip each PC so its largest-magnitude loading is
    # positive -- applied identically regardless of method, so the two
    # methods' outputs agree exactly, not just "up to sign".
    for i in range(len(Vt)):
        if Vt[i, np.argmax(np.abs(Vt[i]))] < 0:
            Vt[i] = -Vt[i]

    evr = eigenvalues / total_var if total_var > 0 else eigenvalues

    comp_names = [f"PC{i + 1}" for i in range(n_comp)]

    loadings = pd.DataFrame(
        Vt[:n_comp].T,
        index=data.columns,
        columns=comp_names,
    )

    factor_rets = pd.DataFrame(
        arr @ Vt[:n_comp].T,
        index=data.index,
        columns=comp_names,
    )

    evr_series = pd.Series(
        evr[:n_comp], index=comp_names, name="explained_variance_ratio"
    )
    cumvar_series = evr_series.cumsum().rename("cumulative_variance_ratio")

    evr_strs = "  ".join(f"{k}={v:.3f}" for k, v in evr_series.items())
    logger.debug(
        "[pca] EVR: %s  (cumulative=%.3f)", evr_strs, float(cumvar_series.iloc[-1])
    )

    return {
        "explained_variance_ratio": evr_series,
        "cumulative_variance_ratio": cumvar_series,
        "loadings": loadings,
        "factor_returns": factor_rets,
        "n_components": n_comp,
        "n_obs": n_obs,
    }


def factor_contributions(
    returns_df: pd.DataFrame,
    n_components: int = 3,
    pca_result: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Marginal R² contribution of each principal component to each asset.

    For each asset, sequentially regresses returns on PC1, then PC1+PC2,
    and so on. The marginal R² at step k is the incremental variance
    explained by adding PC k.

    Because PCs are orthogonal, contributions are additive: the sum of
    all columns for a given asset equals the total R² from all PCs combined.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns, one column per asset.
    n_components : int
        Number of PCs to include (default 3).

    Returns
    -------
    pd.DataFrame  – shape (n_assets × n_components), indexed by ticker.
        Each cell is the marginal fraction of variance explained.
    """
    result = (
        pca_result
        if pca_result is not None
        else pca_returns(returns_df, n_components=n_components)
    )
    factor_rets = result["factor_returns"]
    comp_names = list(factor_rets.columns)

    data = returns_df.dropna()
    aligned = data.loc[data.index.intersection(factor_rets.index)]
    F = factor_rets.loc[aligned.index].to_numpy(dtype=float)

    rows: Dict[str, list] = {}
    for ticker in aligned.columns:
        y = aligned[ticker].to_numpy(dtype=float)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2_prev = 0.0
        contribs = []
        for k in range(n_components):
            X = np.column_stack([np.ones(len(y)), F[:, : k + 1]])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            ss_res = np.sum((y - X @ beta) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            contribs.append(r2 - r2_prev)
            r2_prev = r2
        rows[ticker] = contribs

    return pd.DataFrame(rows, index=comp_names).T
