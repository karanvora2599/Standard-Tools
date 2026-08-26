"""
Building a portfolio, and the ways the build is a lie.

`run_portfolio_optimization` maximizes a mean-variance objective. Everything
here exists because that objective, taken literally, produces portfolios no
one would hold -- and because the reasons why are specific and measurable
rather than a general warning about optimizers.

THE CORE PROBLEM IS THAT MEAN-VARIANCE IS AN ERROR MAXIMIZER. It puts weight
where expected return is high and covariance is low, which is exactly where
estimation error is most likely to have put them. With 50 assets and 2 years
of data you estimate 1,275 covariance parameters from 500 observations, and
the optimizer finds the corner of that estimate where the noise happens to
align. `risk_parity` and `hierarchical_risk_parity` exist because they do not
use expected returns at all, which removes the noisiest input entirely.

WHAT EACH TOOL IS FOR:

- **risk_parity** equalizes each asset's CONTRIBUTION to portfolio risk,
  not its weight. It needs only the covariance matrix, so it drops the input
  with the worst signal-to-noise ratio.
- **hierarchical_risk_parity** goes further: it never inverts the covariance
  matrix at all. Inversion is where an ill-conditioned estimate does its
  damage, and HRP replaces it with a tree of nested bisections.
- **factor_exposure_budget** answers the question that sinks more portfolios
  than any optimizer: "I hold 40 names and I thought I was diversified".
  Forty names with the same factor loading is one position.
- **liquidity_adjusted_var** is VaR that accounts for the fact that you
  cannot exit at the mark. A 1-day VaR on a position that takes 15 days to
  liquidate is not a 1-day risk.
- **concentration_analysis** turns "how concentrated is this" into numbers
  with known interpretations -- effective N, Herfindahl, the top-k share.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

#: Below this, a covariance estimate is not an estimate. With N assets you
#: are fitting N(N+1)/2 parameters, and this is the ratio of observations to
#: parameters below which the matrix is near-singular by construction.
MIN_OBS_PER_PARAMETER = 2.0


def _covariance_frame(covariance: Any, who: str) -> pd.DataFrame:
    frame = pd.DataFrame(covariance).astype(float)
    if frame.shape[0] != frame.shape[1]:
        raise ValidationError(f"{who}: covariance must be square, got {frame.shape}.")
    if frame.isna().any().any():
        raise ValidationError(f"{who}: the covariance matrix contains NaN.")
    array = frame.to_numpy()
    if not np.allclose(array, array.T, rtol=1e-8, atol=1e-12):
        raise ValidationError(
            f"{who}: the covariance matrix is not symmetric. That is a "
            "construction bug rather than a data problem -- a covariance "
            "matrix is symmetric by definition."
        )
    if (np.diag(array) <= 0).any():
        raise ValidationError(
            f"{who}: a diagonal entry is non-positive, so some asset has "
            "zero or negative variance. Usually a constant price series."
        )
    return frame


def _portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(math.sqrt(max(weights @ covariance @ weights, 0.0)))


def _risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """
    Each asset's share of total portfolio volatility.

    Marginal contribution is d(sigma_p)/d(w_i) = (Sigma w)_i / sigma_p, and
    the contribution is w_i times that. They sum to sigma_p exactly, which
    is what makes "contribution" the right word -- it is a genuine
    decomposition rather than an allocation of blame.
    """
    volatility = _portfolio_volatility(weights, covariance)
    if volatility <= 0:
        return np.zeros_like(weights)
    marginal = covariance @ weights / volatility
    return weights * marginal


# ── allocations that do not need expected returns ───────────────────────


def risk_parity(
    covariance: Any,
    *,
    max_iterations: int = 5000,
    tolerance: float = 1e-10,
    budget: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Weights at which every asset contributes the same amount of risk.

    NOT EQUAL WEIGHTS, and the distinction is the entire point. An equally
    weighted portfolio of a bond fund and a biotech stock is a biotech
    portfolio -- the equity contributes almost all the variance. Risk parity
    equalizes the CONTRIBUTIONS, so the volatile asset gets a smaller weight
    and the two matter equally to the outcome.

    IT USES NO EXPECTED RETURNS, which is the reason to prefer it over
    mean-variance in most real situations. Expected returns are the noisiest
    input in finance -- the standard error on a mean return estimated from
    two years of daily data is roughly the size of the estimate -- and
    mean-variance optimization is maximally sensitive to exactly that input.
    Dropping it removes the dominant source of error.

    SOLVED BY CYCLICAL COORDINATE DESCENT, which converges monotonically for
    this problem and needs no matrix inversion. The convergence is reported:
    a solution that did not converge is returned with `converged: false`
    rather than silently, because the iterate at that point is not a risk
    parity portfolio and using it as one is worse than not having it.

    `budget` allows a RISK budget other than equal -- pass [0.5, 0.3, 0.2] to
    give the first asset half the portfolio's risk. This is the useful
    generalization: most real mandates are stated as risk budgets, not as
    equal contributions.
    """
    frame = _covariance_frame(covariance, "risk_parity")
    matrix = frame.to_numpy()
    n = matrix.shape[0]
    if n < 2:
        raise ValidationError("risk_parity: needs at least two assets.")

    if budget is None:
        targets = np.full(n, 1.0 / n)
    else:
        targets = np.asarray([float(b) for b in budget], dtype=float)
        if targets.size != n:
            raise ValidationError(
                f"risk_parity: budget has {targets.size} entries for {n} assets."
            )
        if (targets <= 0).any():
            raise ValidationError(
                "risk_parity: every risk budget must be positive. A zero "
                "budget means 'do not hold this asset', which is an "
                "exclusion rather than a budget -- drop it from the "
                "covariance matrix instead."
            )
        targets = targets / targets.sum()

    # Start from inverse-volatility, which is the exact answer when the
    # correlation matrix is the identity and a good start otherwise.
    weights = 1.0 / np.sqrt(np.diag(matrix))
    weights = weights / weights.sum()

    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        previous = weights.copy()
        volatility = _portfolio_volatility(weights, matrix)
        if volatility <= 0:
            break
        marginal = matrix @ weights
        for i in range(n):
            # The coordinate update that solves w_i * (Sigma w)_i = b_i * sigma^2
            others = marginal[i] - matrix[i, i] * weights[i]
            discriminant = others**2 + 4.0 * matrix[i, i] * targets[i] * volatility**2
            weights[i] = (-others + math.sqrt(max(discriminant, 0.0))) / (
                2.0 * matrix[i, i]
            )
            marginal = matrix @ weights
        weights = weights / weights.sum()
        if np.max(np.abs(weights - previous)) < tolerance:
            converged = True
            break

    contributions = _risk_contributions(weights, matrix)
    total = contributions.sum()
    shares = contributions / total if total > 0 else contributions
    error = float(np.max(np.abs(shares - targets)))

    warnings: List[str] = []
    if not converged:
        warnings.append(
            f"DID NOT CONVERGE in {max_iterations} iterations (largest "
            f"risk-share error {error:.2e}). The weights returned are the "
            "last iterate, and they are not a risk parity portfolio. Using "
            "them as one is worse than not having them -- usually the "
            "covariance matrix is near-singular."
        )
    if error > 1e-4:
        warnings.append(
            f"The largest deviation from the target risk share is "
            f"{error:.2e}, which is above what a converged solution should "
            "show."
        )
    warnings.append(
        "Risk parity uses NO expected returns, which is the point: the "
        "standard error on a mean return from two years of daily data is "
        "about the size of the estimate itself, and mean-variance is "
        "maximally sensitive to exactly that input.",
    )
    warnings.append(
        "Equal RISK contribution is not equal weight and not equal return "
        "expectation. A risk parity portfolio is implicitly betting that "
        "Sharpe ratios are similar across assets; where they are not, it "
        "over-weights the low-Sharpe ones."
    )

    return {
        "n_assets": int(n),
        "assets": [str(c) for c in frame.columns],
        "weights": {str(c): float(w) for c, w in zip(frame.columns, weights)},
        "risk_contributions": {
            str(c): float(rc) for c, rc in zip(frame.columns, contributions)
        },
        "risk_shares": {str(c): float(s) for c, s in zip(frame.columns, shares)},
        "target_shares": {str(c): float(t) for c, t in zip(frame.columns, targets)},
        "portfolio_volatility": _portfolio_volatility(weights, matrix),
        "converged": converged,
        "iterations": iterations,
        "max_share_error": error,
        "warnings": warnings,
    }


def hierarchical_risk_parity(returns: pd.DataFrame) -> Dict[str, Any]:
    """
    Lopez de Prado's HRP: allocation without ever inverting the covariance
    matrix.

    WHY INVERSION IS THE PROBLEM. Mean-variance and minimum-variance both
    require Sigma inverse, and inversion is where an ill-conditioned estimate
    does its damage: the smallest eigenvalue becomes the largest one after
    inversion, so the direction the data says least about becomes the
    direction the portfolio bets most on. With 50 assets and 500
    observations that smallest eigenvalue is essentially noise.

    WHAT HRP DOES INSTEAD, in three steps. Cluster the assets by correlation
    distance into a tree. Order them so similar assets sit adjacent
    (quasi-diagonalization). Then walk the tree splitting capital between
    each pair of branches in inverse proportion to their variance. No
    inversion happens anywhere.

    THE TRADE-OFF IS REAL AND WORTH STATING. HRP has no optimality property
    -- it does not maximize anything. It is more robust out of sample than
    mean-variance in most published comparisons, and it is not the highest
    Sharpe portfolio under any model. It buys stability by giving up the
    claim to be optimal.

    The clustering here is single-linkage on correlation distance,
    sqrt(0.5 * (1 - rho)), implemented without scipy.
    """
    frame = pd.DataFrame(returns).astype(float).dropna()
    n_assets = frame.shape[1]
    if n_assets < 2:
        raise ValidationError("hierarchical_risk_parity: needs at least two assets.")
    if len(frame) < n_assets:
        raise ValidationError(
            f"hierarchical_risk_parity: {len(frame)} observations for "
            f"{n_assets} assets. The correlation matrix is rank-deficient "
            "and the clustering would be reading noise."
        )

    correlation = frame.corr().to_numpy()
    covariance = frame.cov().to_numpy()
    distance = np.sqrt(np.clip(0.5 * (1.0 - correlation), 0.0, None))
    order = _quasi_diagonal_order(distance)

    weights = np.ones(n_assets)
    clusters = [list(range(n_assets))]
    while clusters:
        # Bisect every cluster, then split capital between the halves in
        # inverse proportion to each half's variance.
        clusters = [
            part
            for cluster in clusters
            for part in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
            if len(part) > 0
        ]
        for i in range(0, len(clusters), 2):
            if i + 1 >= len(clusters):
                break
            left = [order[j] for j in clusters[i]]
            right = [order[j] for j in clusters[i + 1]]
            var_left = _cluster_variance(covariance, left)
            var_right = _cluster_variance(covariance, right)
            total = var_left + var_right
            if total <= 0:
                continue
            alpha = 1.0 - var_left / total
            for j in left:
                weights[j] *= alpha
            for j in right:
                weights[j] *= 1.0 - alpha
        clusters = [c for c in clusters if len(c) > 1]

    weights = weights / weights.sum()
    contributions = _risk_contributions(weights, covariance)

    warnings = [
        "HRP never inverts the covariance matrix, which is the point: "
        "inversion turns the smallest eigenvalue into the largest, so the "
        "direction the data says least about becomes the one the portfolio "
        "bets most on.",
        "HRP has NO optimality property -- it does not maximize anything. "
        "It is more robust out of sample than mean-variance in most "
        "published comparisons and it is not the highest-Sharpe portfolio "
        "under any model. That is the trade it makes.",
        "Clustering is single-linkage on correlation distance. Single "
        "linkage is prone to chaining -- a string of moderately correlated "
        "assets can merge into one cluster that has no common theme.",
    ]
    if len(frame) < n_assets * 5:
        warnings.append(
            f"{len(frame)} observations for {n_assets} assets is thin for a "
            "correlation matrix. The tree is being built from correlations "
            "with standard errors around "
            f"{1 / math.sqrt(max(len(frame) - 3, 1)):.2f}."
        )

    return {
        "n_assets": int(n_assets),
        "n_observations": int(len(frame)),
        "weights": {str(c): float(w) for c, w in zip(frame.columns, weights)},
        "cluster_order": [str(frame.columns[i]) for i in order],
        "risk_contributions": {
            str(c): float(rc) for c, rc in zip(frame.columns, contributions)
        },
        "portfolio_volatility": float(
            _portfolio_volatility(weights, covariance) * math.sqrt(TRADING_DAYS)
        ),
        "effective_n": float(1.0 / (weights**2).sum()),
        "warnings": warnings,
    }


def _cluster_variance(covariance: np.ndarray, index: Sequence[int]) -> float:
    """Inverse-variance weighted variance of a sub-portfolio."""
    sub = covariance[np.ix_(index, index)]
    inverse = 1.0 / np.diag(sub)
    weights = inverse / inverse.sum()
    return float(weights @ sub @ weights)


def _quasi_diagonal_order(distance: np.ndarray) -> List[int]:
    """
    Single-linkage clustering, returning the leaf order.

    Written out because scipy is not a dependency. Merges the closest pair
    of clusters repeatedly and concatenates their orderings, which puts
    similar assets adjacent -- the property HRP's bisection needs.
    """
    n = distance.shape[0]
    clusters = {i: [i] for i in range(n)}
    while len(clusters) > 1:
        keys = list(clusters)
        best = None
        for a_i in range(len(keys)):
            for b_i in range(a_i + 1, len(keys)):
                a, b = keys[a_i], keys[b_i]
                linkage = min(distance[i, j] for i in clusters[a] for j in clusters[b])
                if best is None or linkage < best[0]:
                    best = (linkage, a, b)
        if best is None:  # unreachable while len(clusters) > 1, but explicit
            break
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return list(clusters.values())[0]


# ── what you are actually exposed to ────────────────────────────────────


def factor_exposure_budget(
    weights: Dict[str, float],
    factor_loadings: pd.DataFrame,
    *,
    factor_covariance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    What the portfolio is actually betting on, once the names are collapsed
    into factors.

    THE FAILURE THIS EXISTS FOR: "I hold 40 names, so I am diversified."
    Forty names with the same factor loading is one position with extra
    transaction costs. The 2007 quant crisis and the 2020 growth unwind both
    hit portfolios that looked diversified by name count and were not
    diversified in any way that mattered.

    RISK IS DECOMPOSED, not just exposure. A large loading on a low-variance
    factor is not a large risk, and a small loading on a volatile one can be.
    When `factor_covariance` is supplied the result reports each factor's
    share of total portfolio VARIANCE, which is the number that answers "what
    am I actually taking risk on". Without it, only the exposures can be
    reported, and the result says so rather than implying the exposures are
    the risk.

    THE RESIDUAL MATTERS AS MUCH AS THE FACTORS. A portfolio whose variance
    is 90% explained by three factors is a factor bet. One where the factors
    explain 20% is a stock-picking portfolio, and its risk lives somewhere
    this decomposition cannot see. Both are reported.
    """
    loadings = pd.DataFrame(factor_loadings).astype(float)
    series = pd.Series(weights, dtype=float)
    common = [a for a in series.index if a in loadings.index]
    if not common:
        raise ValidationError(
            "factor_exposure_budget: no asset in `weights` appears in the "
            f"loadings index. Weights name {list(series.index)[:5]}, "
            f"loadings name {list(loadings.index)[:5]}."
        )
    missing = [a for a in series.index if a not in loadings.index]
    aligned_weights = series.loc[common].to_numpy()
    aligned_loadings = loadings.loc[common].to_numpy()

    exposures = aligned_weights @ aligned_loadings
    exposure_map = {str(f): float(e) for f, e in zip(loadings.columns, exposures)}

    variance_shares: Optional[Dict[str, float]] = None
    factor_variance = None
    if factor_covariance is not None:
        factor_frame = _covariance_frame(factor_covariance, "factor_exposure_budget")
        if factor_frame.shape[0] != loadings.shape[1]:
            raise ValidationError(
                f"factor_exposure_budget: factor covariance is "
                f"{factor_frame.shape[0]}x{factor_frame.shape[0]} but there "
                f"are {loadings.shape[1]} factors."
            )
        matrix = factor_frame.to_numpy()
        factor_variance = float(exposures @ matrix @ exposures)
        if factor_variance > 0:
            contributions = exposures * (matrix @ exposures)
            variance_shares = {
                str(f): float(c / factor_variance)
                for f, c in zip(loadings.columns, contributions)
            }

    ranked = sorted(exposure_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
    warnings: List[str] = []
    if missing:
        warnings.append(
            f"{len(missing)} position(s) have no factor loadings and were "
            f"excluded: {missing[:5]}. Their risk is invisible to this "
            "decomposition, so the picture is incomplete by exactly their "
            "weight."
        )
    if variance_shares:
        top_factor, top_share = max(variance_shares.items(), key=lambda kv: abs(kv[1]))
        if abs(top_share) > 0.6:
            warnings.append(
                f"{abs(top_share):.0%} of factor variance comes from "
                f"'{top_factor}' alone. However many names this portfolio "
                "holds, it is one bet."
            )
    else:
        warnings.append(
            "No factor covariance was supplied, so only EXPOSURES are "
            "reported -- not risk. A large loading on a quiet factor is not "
            "a large risk, and a small loading on a volatile one can be. "
            "Pass factor_covariance to get the variance decomposition."
        )
    warnings.append(
        "Name count is not diversification. Forty names with the same "
        "loading are one position with extra transaction costs."
    )

    return {
        "n_positions": len(common),
        "n_unmapped": len(missing),
        "unmapped": missing[:20],
        "n_factors": int(loadings.shape[1]),
        "exposures": exposure_map,
        "largest_exposures": [{"factor": f, "exposure": e} for f, e in ranked[:5]],
        "factor_variance": factor_variance,
        "factor_variance_shares": variance_shares,
        "gross_exposure": float(np.abs(aligned_weights).sum()),
        "net_exposure": float(aligned_weights.sum()),
        "warnings": warnings,
    }


def concentration_analysis(weights: Dict[str, float]) -> Dict[str, Any]:
    """
    How concentrated a portfolio is, in numbers with known interpretations.

    "CONCENTRATED" IS NOT A NUMBER and the whole point of this is to make it
    one. Three measures, because they disagree in informative ways:

    - **Effective N** (the inverse Herfindahl) is the count of equally
      weighted positions that would give the same concentration. A
      100-position portfolio with an effective N of 12 holds 100 names and
      has the concentration of 12. This is the number to quote.
    - **The Herfindahl index** is the sum of squared weights. Same
      information, different scale; it is what the effective N inverts.
    - **The top-k share** is what a risk committee actually asks about.

    LONG-SHORT PORTFOLIOS BREAK THE NAIVE VERSION. Weights that sum to zero
    make a share-of-total meaningless, and squaring signed weights loses the
    direction. Concentration here is computed on the GROSS weights, which is
    the economically meaningful denominator: a market-neutral book with 50
    longs and 50 shorts has an effective N in the tens, not an undefined one.
    """
    series = pd.Series(weights, dtype=float).dropna()
    if series.empty:
        raise ValidationError("concentration_analysis: no weights given.")
    gross = float(series.abs().sum())
    if gross <= 0:
        raise ValidationError(
            "concentration_analysis: every weight is zero, so there is no "
            "portfolio to describe."
        )

    shares = (series.abs() / gross).sort_values(ascending=False)
    herfindahl = float((shares**2).sum())
    effective_n = float(1.0 / herfindahl)
    is_long_short = bool((series < 0).any() and (series > 0).any())

    top = {}
    for k in (1, 3, 5, 10):
        if len(shares) >= k:
            top[f"top_{k}_share"] = float(shares.iloc[:k].sum())

    warnings: List[str] = []
    if effective_n < len(series) * 0.3:
        warnings.append(
            f"{len(series)} positions with an effective N of "
            f"{effective_n:.1f}. The portfolio has the concentration of "
            f"{effective_n:.0f} equally weighted names, not {len(series)}."
        )
    if top.get("top_1_share", 0) > 0.25:
        warnings.append(
            f"The largest single position is {top['top_1_share']:.0%} of "
            "gross exposure. Position-level risk dominates portfolio-level "
            "risk at that size, and diversification arguments do not apply "
            "to it."
        )
    if is_long_short:
        warnings.append(
            "This is a long-short book, so concentration is measured on "
            "GROSS weights. Net weights would make the denominator "
            "near-zero and the shares meaningless; gross is the "
            "economically relevant base."
        )
    warnings.append(
        "Concentration by WEIGHT is not concentration by risk. Ten names "
        "at 10% each is concentrated if they share a factor and diversified "
        "if they do not -- use factor_exposure_budget for that question."
    )

    return {
        "n_positions": int(len(series)),
        "gross_exposure": gross,
        "net_exposure": float(series.sum()),
        "is_long_short": is_long_short,
        "herfindahl": herfindahl,
        "effective_n": effective_n,
        "concentration_ratio": float(effective_n / len(series)),
        **top,
        "largest_positions": [
            {"asset": str(a), "weight": float(series[a]), "gross_share": float(s)}
            for a, s in shares.iloc[:10].items()
        ],
        "warnings": warnings,
    }


def liquidity_adjusted_var(
    positions: Dict[str, float],
    volatilities: Dict[str, float],
    daily_volumes: Dict[str, float],
    *,
    confidence: float = 0.95,
    participation_rate: float = 0.15,
    correlation: float = 0.0,
) -> Dict[str, Any]:
    """
    VaR that accounts for the fact that you cannot get out at the mark.

    THE STANDARD NUMBER ASSUMES INSTANT EXIT and that assumption is doing
    more work than anyone acknowledges. A 1-day 95% VaR says "95% of days you
    lose less than this", which is a statement about a position you could
    close today. A position that takes 15 days to liquidate at a sane
    participation rate is exposed for 15 days, and its risk is larger by
    roughly sqrt(15) -- a factor of four.

    THE ADJUSTMENT has two parts, and they are different things:

    1. **Holding-period extension.** Risk scales with the square root of the
       liquidation horizon. A position needing 15 days carries sqrt(15) times
       the 1-day risk. This is the larger effect and the one usually missed.
    2. **Liquidation cost.** Getting out moves the price against you. This is
       an expected cost rather than a risk, and it is reported separately
       because adding a cost to a quantile confuses two different things.

    THE CORRELATION ARGUMENT IS SET BY YOU AND MATTERS ENORMOUSLY. At zero
    the position risks add in quadrature, which understates a real portfolio.
    At 1.0 they add linearly, which is the crisis case -- and crisis is
    exactly when liquidation horizons matter, so the honest stress uses a
    correlation well above the historical average.
    """
    if not positions:
        raise ValidationError("liquidity_adjusted_var: no positions given.")
    if not 0 < confidence < 1:
        raise ValidationError(f"confidence must be in (0, 1), got {confidence!r}")
    if not 0 < participation_rate <= 1:
        raise ValidationError(
            f"participation_rate must be in (0, 1], got {participation_rate!r}"
        )
    if not -1 <= correlation <= 1:
        raise ValidationError(f"correlation must be in [-1, 1], got {correlation!r}")

    # Normal quantile via the erf inverse, bisected -- no scipy.
    z = _normal_quantile(confidence)

    rows: List[Dict[str, Any]] = []
    for asset, value in positions.items():
        value = float(value)
        volatility = float(volatilities.get(asset, float("nan")))
        volume = float(daily_volumes.get(asset, float("nan")))
        if not math.isfinite(volatility) or volatility <= 0:
            raise ValidationError(
                f"liquidity_adjusted_var: no usable volatility for {asset!r}."
            )
        if not math.isfinite(volume) or volume <= 0:
            raise ValidationError(
                f"liquidity_adjusted_var: no usable daily volume for "
                f"{asset!r}. Without it the liquidation horizon is unknown, "
                "which is the entire question."
            )
        daily_volatility = volatility / math.sqrt(TRADING_DAYS)
        days = abs(value) / (volume * participation_rate)
        naive = abs(value) * daily_volatility * z
        adjusted = naive * math.sqrt(max(days, 1.0))
        # Square-root impact, the standard functional form.
        impact_cost = abs(value) * 0.1 * volatility * math.sqrt(abs(value) / volume)
        rows.append(
            {
                "asset": str(asset),
                "position_value": value,
                "annual_volatility": volatility,
                "liquidation_days": float(days),
                "naive_1d_var": float(naive),
                "liquidity_adjusted_var": float(adjusted),
                "adjustment_multiple": float(adjusted / naive) if naive > 0 else None,
                "expected_liquidation_cost": float(impact_cost),
            }
        )
    rows.sort(key=lambda r: r["liquidity_adjusted_var"], reverse=True)

    naive_values = np.array([r["naive_1d_var"] for r in rows])
    adjusted_values = np.array([r["liquidity_adjusted_var"] for r in rows])

    def _aggregate(values: np.ndarray) -> float:
        if correlation <= 0:
            return float(math.sqrt((values**2).sum()))
        independent = (values**2).sum()
        cross = correlation * (values.sum() ** 2 - independent)
        return float(math.sqrt(max(independent + cross, 0.0)))

    total_naive = _aggregate(naive_values)
    total_adjusted = _aggregate(adjusted_values)
    total_cost = float(sum(r["expected_liquidation_cost"] for r in rows))
    worst = rows[0] if rows else None

    warnings: List[str] = []
    slow = [r["asset"] for r in rows if r["liquidation_days"] > 5]
    if slow:
        warnings.append(
            f"{len(slow)} position(s) take over 5 days to liquidate at a "
            f"{participation_rate:.0%} participation rate: {slow[:5]}. Their "
            "1-day VaR is not a 1-day risk."
        )
    if total_naive > 0 and total_adjusted / total_naive > 2:
        warnings.append(
            f"Liquidity adjustment multiplies portfolio VaR by "
            f"{total_adjusted / total_naive:.1f}x. The unadjusted number is "
            "describing a portfolio you could exit today, and this one "
            "cannot be."
        )
    if correlation < 0.5:
        warnings.append(
            f"Positions are aggregated at a correlation of {correlation:.2f}. "
            "Liquidation horizons matter most in a crisis, and correlations "
            "go to 1 in a crisis -- a stress version of this number should "
            "use a correlation well above the historical average."
        )
    warnings.append(
        "The liquidation COST is reported separately from the VaR on "
        "purpose. Cost is an expectation and VaR is a quantile; adding them "
        "produces a number that is neither."
    )

    return {
        "n_positions": len(rows),
        "confidence": float(confidence),
        "participation_rate": float(participation_rate),
        "assumed_correlation": float(correlation),
        "naive_var": total_naive,
        "liquidity_adjusted_var": total_adjusted,
        "adjustment_multiple": (
            float(total_adjusted / total_naive) if total_naive > 0 else None
        ),
        "expected_liquidation_cost": total_cost,
        "worst_position": worst,
        "by_position": rows,
        "warnings": warnings,
    }


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF by bisection on erf. No scipy."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def max_diversification(covariance: Any) -> Dict[str, Any]:
    """
    The portfolio that maximizes the DIVERSIFICATION RATIO: the weighted
    average of the assets' volatilities over the portfolio's own.

    WHAT THE RATIO MEANS. It is 1.0 when everything is perfectly correlated
    -- combining the assets bought nothing -- and grows as the correlations
    fall. Maximizing it is maximizing the volatility that CANCELS, which is
    a cleaner statement of "diversify" than either equal weight or minimum
    variance.

    HOW IT DIFFERS FROM MINIMUM VARIANCE, which is the usual confusion. The
    minimum-variance portfolio piles into the lowest-volatility assets,
    because low volatility is what it is minimizing; on a set containing one
    very quiet asset it concentrates there and is not diversified in any
    ordinary sense. Maximum diversification normalizes by each asset's own
    volatility first, so a quiet asset gets no advantage from being quiet --
    only from being UNCORRELATED.

    IT STILL INVERTS THE COVARIANCE MATRIX, and inherits everything that
    implies: on an ill-conditioned estimate the smallest eigenvalue becomes
    the largest and the portfolio bets on the direction the data says least
    about. The condition number is reported for exactly that reason.
    `hierarchical_risk_parity` is the version that avoids inversion.
    """
    frame = _covariance_frame(covariance, "max_diversification")
    matrix = frame.to_numpy()
    n = matrix.shape[0]
    if n < 2:
        raise ValidationError("max_diversification: needs at least two assets.")

    volatilities = np.sqrt(np.diag(matrix))
    # Maximizing w'v / sqrt(w'Sw) has the same solution as minimum variance
    # on the CORRELATION matrix, rescaled by volatility.
    correlation = matrix / np.outer(volatilities, volatilities)
    try:
        inverse = np.linalg.pinv(correlation)
    except np.linalg.LinAlgError as exc:
        raise ValidationError(
            f"max_diversification: the correlation matrix could not be "
            f"inverted ({exc})."
        ) from None
    raw = inverse @ np.ones(n)
    if raw.sum() == 0:
        raise ValidationError(
            "max_diversification: the solution is degenerate, which happens "
            "when the correlation matrix is singular."
        )
    weights = (raw / volatilities) / (raw / volatilities).sum()

    portfolio_volatility = _portfolio_volatility(weights, matrix)
    weighted_average = float(weights @ volatilities)
    ratio = (
        float(weighted_average / portfolio_volatility)
        if portfolio_volatility > 0
        else None
    )
    condition = float(np.linalg.cond(correlation))
    negative = {str(name): float(w) for name, w in zip(frame.columns, weights) if w < 0}

    warnings: List[str] = []
    if negative:
        warnings.append(
            f"{len(negative)} weight(s) came out NEGATIVE, so this solution "
            "requires shorting. Maximum diversification is unconstrained -- "
            "if the mandate is long-only, this is not the portfolio, and a "
            "constrained optimizer is needed rather than clipping these to "
            "zero."
        )
    if condition > 1000:
        warnings.append(
            f"The correlation matrix has a condition number of "
            f"{condition:,.0f}. This method INVERTS it, so the smallest "
            "eigenvalue becomes the largest and the portfolio bets hardest "
            "on the direction the data says least about. Use "
            "hierarchical_risk_parity, which never inverts, or shrink the "
            "estimate first."
        )
    warnings.append(
        "NOT the same as minimum variance. Minimum variance piles into the "
        "quietest assets because quiet is what it minimizes; this normalizes "
        "by each asset's own volatility first, so an asset is rewarded for "
        "being UNCORRELATED rather than for being quiet."
    )
    return {
        "n_assets": int(n),
        "weights": {str(c): float(w) for c, w in zip(frame.columns, weights)},
        "diversification_ratio": ratio,
        "portfolio_volatility": portfolio_volatility,
        "weighted_average_volatility": weighted_average,
        "condition_number": condition,
        "n_negative_weights": len(negative),
        "negative_weights": negative,
        "warnings": warnings,
    }


def marginal_risk_contribution(
    weights: Dict[str, float], covariance: Any
) -> Dict[str, Any]:
    """
    What one more unit of each position does to portfolio risk, for a
    portfolio you already hold.

    THE QUESTION THIS ANSWERS is the one that comes up when a portfolio
    already exists: not "what should I hold" but "where is my risk actually
    coming from, and what does adding to this position cost me". Those are
    different questions and an optimizer answers neither.

    THREE NUMBERS PER ASSET, and the distinction matters. MARGINAL risk is
    the derivative of portfolio volatility with respect to the weight -- the
    cost of the next unit. CONTRIBUTION is weight times marginal, and these
    sum exactly to portfolio volatility, which is what makes the
    decomposition real rather than an allocation of blame. And the
    contribution SHARE against the weight share is the diagnostic: an asset
    at 5% of the portfolio carrying 30% of the risk is the position to look
    at first.

    A NEGATIVE MARGINAL CONTRIBUTION IS THE INTERESTING CASE. It means
    adding to that position REDUCES portfolio risk, which happens when the
    asset is negatively correlated with the rest of the book. Those
    positions are hedges whether or not they were intended as such.
    """
    frame = _covariance_frame(covariance, "marginal_risk_contribution")
    series = pd.Series(weights, dtype=float)
    missing = [str(c) for c in frame.columns if str(c) not in series.index]
    if missing:
        raise ValidationError(
            f"marginal_risk_contribution: no weight given for {missing}. "
            "Every asset in the covariance matrix needs one -- omitting it "
            "would silently treat the position as zero."
        )
    ordered = np.array([float(series[str(c)]) for c in frame.columns])
    matrix = frame.to_numpy()

    volatility = _portfolio_volatility(ordered, matrix)
    if volatility <= 0:
        raise ValidationError(
            "marginal_risk_contribution: the portfolio has zero volatility, "
            "so there is no risk to attribute."
        )
    marginal = matrix @ ordered / volatility
    contribution = ordered * marginal
    gross = float(np.abs(ordered).sum())

    rows = [
        {
            "asset": str(name),
            "weight": float(w),
            "weight_share": float(abs(w) / gross) if gross > 0 else None,
            "marginal_risk": float(m),
            "risk_contribution": float(c),
            "risk_share": float(c / volatility),
            "concentration_flag": bool(
                gross > 0 and c / volatility > 2.0 * abs(w) / gross
            ),
        }
        for name, w, m, c in zip(frame.columns, ordered, marginal, contribution)
    ]
    rows.sort(key=lambda r: r["risk_contribution"], reverse=True)

    hedges = [r["asset"] for r in rows if r["marginal_risk"] < 0]
    outsized = [r for r in rows if r["concentration_flag"]]

    warnings: List[str] = []
    if outsized:
        worst = outsized[0]
        warnings.append(
            f"{len(outsized)} position(s) carry more than twice their weight "
            f"share of the risk. The largest is {worst['asset']}: "
            f"{worst['weight_share']:.1%} of gross exposure and "
            f"{worst['risk_share']:.1%} of the risk."
        )
    if hedges:
        warnings.append(
            f"{len(hedges)} position(s) have NEGATIVE marginal risk "
            f"({hedges[:5]}): adding to them REDUCES portfolio volatility, "
            "because they are negatively correlated with the rest of the "
            "book. They are hedges whether or not they were meant as such."
        )
    warnings.append(
        "Contributions sum exactly to portfolio volatility, which is what "
        "makes this a decomposition rather than an allocation of blame. "
        "Marginal risk is the cost of the NEXT unit and is the number to use "
        "when deciding whether to add."
    )
    return {
        "n_assets": len(rows),
        "portfolio_volatility": volatility,
        "gross_exposure": gross,
        "by_asset": rows,
        "n_hedges": len(hedges),
        "sum_of_contributions": float(contribution.sum()),
        "warnings": warnings,
    }


def portfolio_scenarios(
    weights: Dict[str, float],
    scenarios: Dict[str, Dict[str, float]],
    *,
    covariance: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    What a portfolio does under NAMED shocks, rather than under a
    distribution.

    WHY NAMED SCENARIOS AND NOT VaR. A 99% VaR is a statement about a
    distribution fitted to history, and its central weakness is that the
    event you care about is usually not in that history. A named scenario --
    "rates +200bp, equities -20%, credit spreads double" -- makes the
    assumption explicit and arguable, which a quantile does not. The two
    answer different questions and a risk process needs both.

    EACH SCENARIO IS A MAP OF ASSET TO RETURN. Assets in the portfolio but
    absent from a scenario are treated as UNCHANGED, and the count is
    reported: a scenario covering three of forty positions is a partial
    scenario and its loss is a lower bound, which is worth knowing before it
    is presented as the worst case.

    With `covariance`, each scenario's move is also reported in standard
    deviations of the portfolio -- which is the honest way to compare a
    scenario against the statistical measures rather than instead of them.
    """
    series = pd.Series(weights, dtype=float).dropna()
    if series.empty:
        raise ValidationError("portfolio_scenarios: no weights given.")
    if not scenarios:
        raise ValidationError(
            "portfolio_scenarios: no scenarios given. This tool exists to "
            "make an assumption explicit; with none, use run_stress_test."
        )

    portfolio_volatility = None
    if covariance is not None:
        frame = _covariance_frame(covariance, "portfolio_scenarios")
        common = [str(c) for c in frame.columns if str(c) in series.index]
        if len(common) == len(frame.columns):
            ordered = np.array([float(series[str(c)]) for c in frame.columns])
            portfolio_volatility = _portfolio_volatility(ordered, frame.to_numpy())

    rows: List[Dict[str, Any]] = []
    for name, shocks in scenarios.items():
        covered = [a for a in series.index if a in shocks]
        uncovered = [a for a in series.index if a not in shocks]
        impact = float(sum(series[a] * float(shocks[a]) for a in covered))
        rows.append(
            {
                "scenario": str(name),
                "portfolio_return": impact,
                "n_positions_shocked": len(covered),
                "n_positions_unchanged": len(uncovered),
                "coverage": float(len(covered) / len(series)),
                "sigma_move": (
                    float(impact / portfolio_volatility)
                    if portfolio_volatility
                    else None
                ),
                "largest_contributor": (
                    max(
                        ((a, float(series[a] * float(shocks[a]))) for a in covered),
                        key=lambda pair: abs(pair[1]),
                    )[0]
                    if covered
                    else None
                ),
            }
        )
    rows.sort(key=lambda r: r["portfolio_return"])

    partial = [r["scenario"] for r in rows if r["coverage"] < 0.9]
    warnings: List[str] = []
    if partial:
        warnings.append(
            f"Scenario(s) {partial[:5]} shock fewer than 90% of the "
            "positions. Unshocked positions are treated as UNCHANGED, so "
            "those losses are a LOWER BOUND rather than the scenario's full "
            "effect."
        )
    if rows:
        warnings.append(
            f"Worst scenario is {rows[0]['scenario']} at "
            f"{rows[0]['portfolio_return']:.2%}"
            + (
                f", a {abs(rows[0]['sigma_move']):.1f}-sigma move."
                if rows[0]["sigma_move"]
                else "."
            )
        )
    warnings.append(
        "A named scenario and a VaR answer different questions. VaR is a "
        "quantile of a distribution fitted to history, and its weakness is "
        "that the event you care about is usually not in that history; a "
        "named scenario makes the assumption explicit and arguable. A risk "
        "process needs both, not one instead of the other."
    )
    return {
        "n_positions": int(len(series)),
        "n_scenarios": len(rows),
        "portfolio_volatility": portfolio_volatility,
        "worst_scenario": rows[0] if rows else None,
        "best_scenario": rows[-1] if rows else None,
        "by_scenario": rows,
        "warnings": warnings,
    }


__all__ = [
    "marginal_risk_contribution",
    "max_diversification",
    "portfolio_scenarios",
    "MIN_OBS_PER_PARAMETER",
    "TRADING_DAYS",
    "concentration_analysis",
    "factor_exposure_budget",
    "hierarchical_risk_parity",
    "liquidity_adjusted_var",
    "risk_parity",
]
