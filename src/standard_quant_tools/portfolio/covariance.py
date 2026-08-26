"""
Covariance estimation, and why the sample one is usually the wrong choice.

The optimizer already warns about conditioning. Shrinkage is the ANSWER to
that warning rather than a caveat about it, and the reason is arithmetic
rather than taste: a sample covariance matrix estimated from T observations
of N assets has N(N+1)/2 parameters fitted from NT numbers. At N=50 and
T=252 that is 1,275 parameters from 12,600 observations, and the smallest
eigenvalues — the directions the optimizer will happily lever into, because
they look like free risk reduction — are the ones estimated worst.

Mean-variance optimization is an error-maximizer over exactly those
directions. It does not merely tolerate a noisy covariance matrix; it seeks
out the noisiest direction in it and puts the portfolio there.

THREE ESTIMATORS, AND WHEN EACH IS RIGHT:

- `sample` — unbiased, and unusable when N approaches T. Kept because it is
  the honest baseline and because with T >> N it is fine.
- `ledoit_wolf` — shrinks toward a scaled identity by an amount CHOSEN from
  the data rather than picked. The default for portfolio construction.
- `ewma` — weights recent observations more. A different question from
  shrinkage: it is about regime rather than about estimation error, and it
  makes the conditioning WORSE, because a half-life of 60 days on 252 days
  of data has an effective sample size closer to 87.

They are not alternatives to one another. `ewma` and `ledoit_wolf` answer
different questions and shrinking an EWMA estimate is a reasonable thing to
want; that is `ewma_shrunk`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

METHODS = ("sample", "ledoit_wolf", "ewma", "ewma_shrunk")

#: Condition number above which the optimizer's weights stop being
#: determined by the data. Not a hard failure -- a warning, because a
#: caller who is equal-weighting does not care and one who is levering the
#: minimum-variance direction cares enormously.
CONDITION_WARNING = 1e4


def estimate_covariance(
    returns: pd.DataFrame,
    *,
    method: str = "ledoit_wolf",
    halflife: Optional[float] = 60.0,
    periods_per_year: int = 252,
) -> Dict[str, Any]:
    """
    A covariance matrix, plus the diagnostics that say whether to trust it.

    Returns the matrix ANNUALIZED, because every other risk number in this
    library is annualized and a covariance in daily units silently produces
    a volatility 16 times too small.

    `shrinkage_intensity` is the number to read for `ledoit_wolf`: it is the
    weight put on the structured target, chosen analytically rather than
    tuned. Near 0 means the sample matrix was already well conditioned; near
    1 means almost nothing in the sample estimate survived, which is itself
    a finding about the data rather than a failure of the method.
    """
    if method not in METHODS:
        raise ValidationError(
            f"unknown covariance method {method!r}; expected one of {list(METHODS)}"
        )
    frame = returns.dropna(how="all", axis=1).dropna()
    if frame.shape[0] < 2:
        raise ValidationError(
            f"covariance needs at least 2 complete observations, got "
            f"{frame.shape[0]}. Every asset must have a return on every date "
            "used, and rows with any missing value were dropped."
        )
    if frame.shape[1] < 2:
        raise ValidationError("covariance needs at least 2 assets with usable history")

    n_obs, n_assets = frame.shape
    values = frame.to_numpy(dtype=float)
    shrinkage: Optional[float] = None

    if method == "sample":
        cov = np.cov(values, rowvar=False, ddof=1)
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        estimator = LedoitWolf().fit(values)
        cov = estimator.covariance_
        shrinkage = float(estimator.shrinkage_)
    else:
        cov = _ewma_covariance(values, halflife or 60.0)
        if method == "ewma_shrunk":
            cov, shrinkage = _shrink_to_identity(cov, n_obs, n_assets)

    annual = cov * periods_per_year
    eigenvalues = np.linalg.eigvalsh(annual)
    smallest = float(eigenvalues.min())
    condition = float(eigenvalues.max() / smallest) if smallest > 0 else float("inf")

    return {
        "method": method,
        "matrix": {
            row: {col: float(annual[i, j]) for j, col in enumerate(frame.columns)}
            for i, row in enumerate(frame.columns)
        },
        "assets": list(frame.columns),
        "n_observations": int(n_obs),
        "n_assets": int(n_assets),
        "observations_per_parameter": float(
            n_obs * n_assets / (n_assets * (n_assets + 1) / 2)
        ),
        "shrinkage_intensity": shrinkage,
        "condition_number": condition,
        "smallest_eigenvalue": smallest,
        "annualized": True,
        "warnings": _warnings(method, n_obs, n_assets, condition, smallest, shrinkage),
    }


def _ewma_covariance(values: np.ndarray, halflife: float) -> np.ndarray:
    """
    Exponentially weighted covariance about the WEIGHTED mean.

    Demeaning with the plain average would mix a full-sample centre into a
    recency-weighted spread, which shows up as extra variance whenever the
    mean has moved -- exactly the regimes EWMA is reached for.
    """
    if halflife <= 0:
        raise ValidationError("covariance: halflife must be positive")
    n = values.shape[0]
    decay = 0.5 ** (1.0 / halflife)
    weights = decay ** np.arange(n - 1, -1, -1)
    weights = weights / weights.sum()
    centered = values - (weights[:, None] * values).sum(axis=0)
    return (centered * weights[:, None]).T @ centered / (1.0 - (weights**2).sum())


def _shrink_to_identity(cov: np.ndarray, n_obs: int, n_assets: int):
    """
    Shrink toward a scaled identity with an intensity from the data's own
    shape.

    Not Ledoit-Wolf's analytic optimum -- that is derived for the sample
    covariance and does not carry over to a weighted one. This is the
    dimension-to-observations ratio, which is the quantity the optimum
    tracks, and it is labelled as the approximation it is.
    """
    average_variance = float(np.trace(cov) / n_assets)
    target = np.eye(n_assets) * average_variance
    intensity = float(min(1.0, n_assets / max(n_obs, 1)))
    return (1.0 - intensity) * cov + intensity * target, intensity


def _warnings(method, n_obs, n_assets, condition, smallest, shrinkage) -> List[str]:
    out: List[str] = []
    per_parameter = n_obs * n_assets / (n_assets * (n_assets + 1) / 2)
    if method == "sample" and n_assets > n_obs / 4:
        out.append(
            f"{n_assets} assets from {n_obs} observations is about "
            f"{per_parameter:.0f} numbers per estimated parameter. The sample "
            "covariance is unbiased and badly conditioned here, and "
            "mean-variance optimization is an error-maximizer over exactly "
            "its worst-estimated directions. Use ledoit_wolf."
        )
    if condition > CONDITION_WARNING:
        out.append(
            f"condition number {condition:.0f}: the smallest eigenvalue is "
            f"{condition:.0f} times below the largest, so the optimizer's "
            "weights in that direction are determined by estimation noise "
            "rather than by the data."
        )
    if smallest <= 0:
        out.append(
            "the covariance matrix is singular -- at least one asset is an "
            "exact linear combination of the others. An optimizer will lever "
            "that direction without limit. Drop a redundant asset or shrink."
        )
    if shrinkage is not None and shrinkage > 0.5:
        out.append(
            f"shrinkage intensity {shrinkage:.2f}: more than half of this "
            "estimate is the structured target rather than the sample. That "
            "is the method working, not failing -- but it means the data "
            "supported very little of the correlation structure."
        )
    if method == "ewma":
        out.append(
            "EWMA answers a different question from shrinkage -- regime, not "
            "estimation error -- and it makes conditioning WORSE by lowering "
            "the effective sample size. Use ewma_shrunk if you need both."
        )
    return out


__all__ = ["CONDITION_WARNING", "METHODS", "estimate_covariance"]
