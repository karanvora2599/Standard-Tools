import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def pca_returns(
    returns_df: pd.DataFrame,
    n_components: Optional[int] = None,
    standardize: bool = True,
) -> Dict[str, Any]:
    """
    Principal Component Analysis on a multi-asset return matrix.

    Uses full SVD (pure NumPy) — no sklearn or statsmodels required.

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
        "[pca] assets=%d  obs=%d  n_components=%s  standardize=%s",
        n_assets,
        n_obs,
        n_components,
        standardize,
    )

    if n_obs < 2 or n_assets < 1:
        raise ValueError(
            f"Need at least 2 observations and 1 asset; got ({n_obs}, {n_assets})."
        )

    arr = data.to_numpy(dtype=float)
    arr = arr - arr.mean(axis=0)

    if standardize:
        stds = arr.std(axis=0, ddof=1)
        stds[stds == 0] = 1.0
        arr = arr / stds

    U, s, Vt = np.linalg.svd(arr, full_matrices=False)

    # Sign convention: flip each PC so its largest-magnitude loading is positive
    for i in range(len(s)):
        if Vt[i, np.argmax(np.abs(Vt[i]))] < 0:
            Vt[i] = -Vt[i]
            U[:, i] = -U[:, i]

    eigenvalues = s**2 / (n_obs - 1)
    total_var = eigenvalues.sum()
    evr = eigenvalues / total_var if total_var > 0 else eigenvalues

    n_comp = min(
        n_components if n_components is not None else n_assets,
        n_assets,
        n_obs,
    )

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
