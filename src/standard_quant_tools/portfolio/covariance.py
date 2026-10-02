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
  of data has an effective sample size (1 / sum of squared weights) of
  about 155, not 252. It is reported as `effective_observations`.

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
    kept_columns = returns.dropna(how="all", axis=1)
    frame = kept_columns.dropna()
    # Rows removed because SOME asset had no return there. One short
    # history removed 400 of 512 rows on a live panel with warnings: []
    # and moved risk-parity weights by 12.4% of NAV; the estimate is
    # then of the short history's window, not of the requested one.
    n_rows_dropped = int(len(kept_columns) - len(frame))
    shortest = (
        str(kept_columns.notna().sum().idxmin()) if kept_columns.shape[1] else None
    )
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
    effective = float(n_obs)

    if method == "sample":
        cov = np.cov(values, rowvar=False, ddof=1)
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        estimator = LedoitWolf().fit(values)
        cov = estimator.covariance_
        shrinkage = float(estimator.shrinkage_)
    else:
        # `halflife or 60.0` turned an explicit 0 into the default 60 while
        # -5 was refused: 0 is what an uninitialised config field arrives
        # as, and it was answered as though 60 had been asked for. Only an
        # absent halflife takes the default; 0 reaches the refusal below.
        cov, effective = _ewma_covariance(
            values, 60.0 if halflife is None else halflife
        )
        if method == "ewma_shrunk":
            cov, shrinkage = _shrink_to_identity(cov, n_obs, n_assets)

    annual = cov * periods_per_year
    eigenvalues = np.linalg.eigvalsh(annual)
    smallest = float(eigenvalues.min())
    condition = float(eigenvalues.max() / smallest) if smallest > 0 else float("inf")

    # The names once and the rows as Python floats once. Iterating the
    # column Index inside the comprehension and boxing `annual[i, j]` cell
    # by cell was nearly half of a sample-covariance call at 235 assets,
    # for the same dict: `tolist()` yields the same floats `float()` did.
    assets = list(frame.columns)
    return {
        "method": method,
        "matrix": {
            row: dict(zip(assets, values))
            for row, values in zip(assets, annual.tolist())
        },
        "assets": assets,
        "n_observations": int(n_obs),
        "n_assets": int(n_assets),
        "effective_observations": effective,
        # Numbers per parameter from the observations the estimate actually
        # rests on: under EWMA that is the effective count, not the rows.
        "observations_per_parameter": float(
            effective * n_assets / (n_assets * (n_assets + 1) / 2)
        ),
        "shrinkage_intensity": shrinkage,
        "condition_number": condition,
        "smallest_eigenvalue": smallest,
        "annualized": True,
        "n_rows_dropped": n_rows_dropped,
        "warnings": _warnings(
            method,
            n_obs,
            n_assets,
            condition,
            smallest,
            shrinkage,
            n_rows_dropped=n_rows_dropped,
            n_rows_total=int(len(kept_columns)),
            shortest=shortest,
            effective=effective,
        ),
    }


#: Fewer effective observations than this and there is no spread to
#: estimate: the weights sit on essentially one row.
MIN_EFFECTIVE_OBSERVATIONS = 2.0


def _ewma_covariance(values: np.ndarray, halflife: float) -> "tuple[np.ndarray, float]":
    """
    Exponentially weighted covariance about the WEIGHTED mean, and the
    effective number of observations it rests on.

    Demeaning with the plain average would mix a full-sample centre into a
    recency-weighted spread, which shows up as extra variance whenever the
    mean has moved -- exactly the regimes EWMA is reached for.

    THE EFFECTIVE COUNT IS CHECKED BEFORE THE DIVISION. The unbiasing
    denominator is 1 - sum(w^2), and a short enough half-life puts all the
    weight on the last row: below a half-life of about 0.019 the older
    weights underflow, sum(w^2) is exactly 1, and the division escaped as a
    raw numpy LinAlgError two calls later; between about 0.019 and 0.05 it
    returned a matrix with a condition number of 1e12 to 1e17 and no
    warning. 1 / sum(w^2) is the Kish effective sample size, and below two
    there is no second observation to measure a spread against.
    """
    halflife = float(halflife)
    if not np.isfinite(halflife) or halflife <= 0:
        raise ValidationError(
            f"covariance: halflife must be positive and finite, got {halflife!r}. "
            "Omit it for the default of 60 observations."
        )
    n = values.shape[0]
    decay = 0.5 ** (1.0 / halflife)
    weights = decay ** np.arange(n - 1, -1, -1)
    weights = weights / weights.sum()
    effective = float(1.0 / (weights**2).sum())
    if effective < MIN_EFFECTIVE_OBSERVATIONS:
        raise ValidationError(
            f"covariance: halflife={halflife:g} leaves {effective:.3f} "
            f"effective observations of the {n} rows (1 / sum of squared "
            "weights); an exponentially weighted covariance needs at least "
            f"{MIN_EFFECTIVE_OBSERVATIONS:g}. The weight has collapsed onto "
            "the last row, so there is no spread to estimate. Use a half-life "
            "of at least one observation -- the default is 60."
        )
    centered = values - (weights[:, None] * values).sum(axis=0)
    cov = (centered * weights[:, None]).T @ centered / (1.0 - (weights**2).sum())
    return cov, effective


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


def _warnings(
    method,
    n_obs,
    n_assets,
    condition,
    smallest,
    shrinkage,
    *,
    n_rows_dropped: int = 0,
    n_rows_total: int = 0,
    shortest=None,
    effective: Optional[float] = None,
) -> List[str]:
    out: List[str] = []
    if method.startswith("ewma") and effective is not None and effective <= n_assets:
        out.append(
            f"The half-life leaves {effective:.1f} effective observations for "
            f"{n_assets} assets. A covariance over more assets than effective "
            "observations is rank-deficient by construction, whatever its row "
            "count says: the smallest eigenvalues are weighting artefacts. "
            "Lengthen the half-life or use fewer assets."
        )
    if n_rows_dropped:
        out.append(
            f"{n_rows_dropped} of {n_rows_total} rows were dropped because at "
            f"least one asset had no return there (the shortest history is "
            f"{shortest!r}), so the estimate covers only the {n_obs} complete "
            "rows. Measured live, one short history removed 400 of 512 rows "
            "and moved risk-parity weights by 12.4% of NAV; drop the short "
            "asset or start the window where every asset has data."
        )
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
