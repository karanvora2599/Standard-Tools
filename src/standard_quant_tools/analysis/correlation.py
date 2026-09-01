import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)


def diversification_ratio(
    returns_df: pd.DataFrame,
    weights: Optional[Union[List[float], np.ndarray]] = None,
) -> float:
    """
    Choueifaty-Coignard (2008) diversification ratio:

        DR = (sum_i w_i * sigma_i) / sigma_portfolio

    where sigma_i is asset i's own return volatility and sigma_portfolio is
    the actual portfolio volatility (sqrt(w' Cov w)). By Cauchy-Schwarz,
    DR >= 1 always; DR == 1 means zero diversification benefit (assets
    perfectly correlated), and higher values mean more of each asset's
    individual risk is being diversified away by the others.

    weights default to equal-weight when not supplied. Uses returns_df's
    own per-period std/covariance directly (the ratio is scale-invariant to
    periodization/annualization as long as both numerator and denominator
    use the same convention, which they do here).
    """
    n_assets = returns_df.shape[1]
    if n_assets < 2:
        raise ValidationError(
            f"diversification_ratio needs at least 2 assets, got {n_assets}"
        )

    w = (
        np.full(n_assets, 1.0 / n_assets)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if len(w) != n_assets:
        raise ValidationError(
            f"weights length ({len(w)}) must match number of assets ({n_assets})"
        )
    if not np.isclose(w.sum(), 1.0, atol=1e-4):
        raise ValidationError(f"weights must sum to 1.0, got {w.sum():.4f}")

    individual_vols = returns_df.std().to_numpy(dtype=np.float64)
    weighted_avg_vol = float(np.sum(w * individual_vols))

    cov = returns_df.cov().to_numpy(dtype=np.float64)
    portfolio_vol = float(np.sqrt(w @ cov @ w))

    if portfolio_vol <= 0.0:
        logger.warning(
            "[diversification_ratio] portfolio volatility is zero — returning NaN"
        )
        return float("nan")

    ratio = weighted_avg_vol / portfolio_vol
    logger.debug(
        "[diversification_ratio] assets=%d  weighted_avg_vol=%.6f  portfolio_vol=%.6f  ratio=%.4f",
        n_assets,
        weighted_avg_vol,
        portfolio_vol,
        ratio,
    )
    return ratio


def pairwise_correlation_summary(returns_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Full correlation matrix plus derived summary stats: average pairwise
    correlation, and the most/least correlated off-diagonal pair.

    Returns:
        Dict with keys: correlation_matrix (pd.DataFrame),
        avg_pairwise_correlation (float), highest_correlated_pair (dict
        with a/b/correlation), lowest_correlated_pair (dict with
        a/b/correlation).
    """
    # Local import: portfolio.portfolio imports metrics.risk_metrics, which
    # imports analysis.regression -- a module-level import here would make
    # `analysis` depend on `portfolio` at analysis/__init__.py's own import
    # time, creating a cycle (analysis -> portfolio -> metrics -> analysis)
    # that only surfaces in specific import orders (e.g. a fresh
    # ProcessPoolExecutor worker re-importing from scratch). Deferring the
    # import to call time sidesteps this entirely, matching the same
    # lazy-import pattern portfolio.py itself already uses for its own
    # cross-package DataFactory import.
    from standard_quant_tools.portfolio.portfolio import correlation_matrix

    tickers = list(returns_df.columns)
    n = len(tickers)
    if n < 2:
        raise ValidationError(
            f"pairwise_correlation_summary needs at least 2 assets, got {n}"
        )

    corr = correlation_matrix(returns_df)

    # Read the upper triangle out in ONE numpy step rather than with a
    # nested loop of `corr.iloc[i, j]`. The matrix is already computed by
    # the line above; the loop this replaces existed only to find an argmax
    # and an argmin over it, and spent 19,900 pandas scalar lookups doing
    # that on a 200-name universe -- 368 ms to answer a question the matrix
    # already contained. `np.triu_indices(n, k=1)` walks (i, j) in exactly
    # the order the nested loop did, so argmax and argmin break ties on the
    # same pair the old `max()`/`min()` returned.
    rows, cols = np.triu_indices(n, k=1)
    values = corr.to_numpy()[rows, cols]

    avg_corr = float(np.mean(values))
    hi, lo = int(np.argmax(values)), int(np.argmin(values))
    highest = (tickers[rows[hi]], tickers[cols[hi]], float(values[hi]))
    lowest = (tickers[rows[lo]], tickers[cols[lo]], float(values[lo]))

    logger.debug(
        "[pairwise_correlation_summary] assets=%d  avg_corr=%.4f  highest=%s/%s=%.4f  lowest=%s/%s=%.4f",
        n,
        avg_corr,
        highest[0],
        highest[1],
        highest[2],
        lowest[0],
        lowest[1],
        lowest[2],
    )

    return {
        "correlation_matrix": corr,
        "avg_pairwise_correlation": avg_corr,
        "highest_correlated_pair": {
            "a": highest[0],
            "b": highest[1],
            "correlation": highest[2],
        },
        "lowest_correlated_pair": {
            "a": lowest[0],
            "b": lowest[1],
            "correlation": lowest[2],
        },
    }
