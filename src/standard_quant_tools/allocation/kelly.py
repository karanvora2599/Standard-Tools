"""
The Kelly leverage of a return stream, or of several held together, and
how much of it the data can actually support.

THE MODEL. A stream with excess return mu and volatility sigma (annual,
over the risk-free rate), held at leverage f, grows at

    g(f) = f mu - f^2 sigma^2 / 2          (log, per year, over the rate)

which is greatest at f* = mu / sigma^2, where g = S^2 / 2 with S = mu /
sigma the Sharpe ratio; g turns negative past 2 f*. For several streams
the leverages are the vector f* = Sigma^-1 mu and the optimum is theta^2 / 2
with theta^2 = mu' Sigma^-1 mu, the squared Sharpe ratio of the tangency
portfolio. f* is a NOTIONAL -- the stream's exposure as a multiple of
equity -- which is what `portfolio.position_sizing`'s
`kelly_notional_fraction` reports for one trade; this module answers the
same question for a whole stream and does not size anything.

HOW SURE IT IS. The mean is estimated from T years with standard error
sigma / sqrt(T), and the variance far more precisely, so

    se(f-hat) = 1 / (sigma sqrt(T))    one stream
    se(f-hat_i) = sqrt((Sigma^-1)_ii / T)    several

(the variance's own error adds 2 f*^2 / n, n the observations -- 0.04 on a
standard error of 4.47 at five years of daily data). f-hat / f* then has a
relative error of 1 / (S sqrt(T)), and the chance the estimate is more
than twice the truth -- where its growth is no better than holding cash --
is Phi(-S sqrt(T)): 13% for a Sharpe of 0.5 over five years.

WHAT THE ERROR COSTS. Holding c f-hat when f-hat ~ N(f*, se^2) grows in
expectation at

    theta^2 (c - c^2 / 2) - c^2 N / (2 T)

so full Kelly from five years of a 0.5 Sharpe keeps 20% of the optimal
growth and half Kelly 55%. The fraction of f-hat that maximises it is the
plug-in shrinkage of Kan and Zhou (2007), k = theta^2 T / (theta^2 T + N),
Shat^2 T / (1 + Shat^2 T) for one stream. It is a plug-in: Shat^2 is biased
up by 1 / T, so k is optimistic about the data.

DRAWDOWNS. In the same continuous model, wealth held at c f* ever falls to
(1 - d) of where it starts with probability (1 - d)^(2/c - 1) (Browne
1999; Thorp 2006): a half at full Kelly, one in eight at half Kelly. It is
an infinite-horizon figure for any starting point, and it is 1 at c >= 2.
The bootstrap distribution of the maximum drawdown over a path as long as
the sample is reported beside it, at each fraction, from circular blocks of
the sample's own portfolio returns with f-hat held fixed.

EVERYTHING IS IN-SAMPLE. f-hat, its interval and every growth and drawdown
figure take the sample's mean and covariance as the truth; the empirical
Kelly is the leverage that would have grown this sample the most, which is
the most in-sample number of all. A negative mean excess return gives a
negative f*: Kelly shorts the stream, and it is reported as such, never as
zero.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools._resampling import circular_block_indices
from standard_quant_tools.error import ValidationError

__all__ = [
    "DEFAULT_DRAWDOWN_LEVELS",
    "DEFAULT_FRACTIONS",
    "MIN_PERIODS",
    "drawdown_probability",
    "empirical_kelly",
    "kelly_report",
]

_WHO = "kelly_report"

#: Fractions of Kelly reported by default: quarter, half and full.
DEFAULT_FRACTIONS = (0.25, 0.5, 1.0)

#: Losses (fractions of wealth) whose probability is reported by default.
DEFAULT_DRAWDOWN_LEVELS = (0.2, 0.5)

#: The fewest periods a report is computed from. A mean from fewer has an
#: interval wider than any leverage anyone would hold.
MIN_PERIODS = 20

#: The largest condition number of the streams' correlation matrix a
#: Sigma^-1 mu is taken from. Past it two streams are close enough to
#: copies that the solve levers into their difference.
MAX_CONDITION = 1e10

#: Newton's iterations for the empirical Kelly, and the norm past which the
#: in-sample growth is read as unbounded.
_NEWTON_STEPS = 200
_UNBOUNDED = 1e9


def drawdown_probability(fraction_of_kelly: float, loss: float) -> float:
    """
    P(wealth ever falls to 1 - loss of its start) at c = fraction_of_kelly
    of the true Kelly leverage, in the continuous model: (1 - loss)^(2/c -
    1), capped at 1 (c >= 2 grows at no more than zero and so hits every
    level eventually).
    """
    exponent = 2.0 / fraction_of_kelly - 1.0
    if exponent <= 0.0:
        return 1.0
    return float((1.0 - loss) ** exponent)


def _growth(excess: np.ndarray, rate: float, fraction: np.ndarray) -> float:
    """sum_t log(1 + rate + f'x_t) - log(1 + rate), or -inf off the domain."""
    wealth = 1.0 + rate + excess @ fraction
    if np.any(wealth <= 0.0):
        return -math.inf
    return float(np.sum(np.log(wealth)) - excess.shape[0] * math.log1p(rate))


def empirical_kelly(excess: np.ndarray, rate: float) -> Optional[np.ndarray]:
    """
    The leverages that maximise realized log growth on the sample,
    sum_t log(1 + rate + f'x_t), by Newton's method with a backtracking
    line search that never leaves the domain (every period's wealth
    positive). `excess` is periods x streams of per-period excess returns,
    `rate` the per-period risk-free rate.

    None when the growth has no maximum: a direction in which the sample
    never lost (every x_t'd >= 0), along which more leverage always grew
    the sample more. Concave, so the first stationary point is the answer.
    """
    n, k = excess.shape
    f = np.zeros(k)
    with single_threaded_blas():
        current = _growth(excess, rate, f)
        for _ in range(_NEWTON_STEPS):
            wealth = 1.0 + rate + excess @ f
            scaled = excess / wealth[:, None]
            gradient = scaled.sum(axis=0)
            hessian = scaled.T @ scaled
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                return None
            decrement = float(gradient @ step)
            if decrement <= 1e-18 * n:
                return f
            size = 1.0
            while size > 1e-12:
                trial = f + size * step
                value = _growth(excess, rate, trial)
                if value >= current + 1e-4 * size * decrement:
                    break
                size *= 0.5
            else:
                return f
            f, current = trial, value
            if float(np.max(np.abs(f))) > _UNBOUNDED:
                return None
    return None


def _max_drawdowns(paths: np.ndarray) -> np.ndarray:
    """
    The maximum drawdown of each row's compounded path, <= 0, as
    `metrics.risk_metrics.max_drawdown` gives it on `equity_from_returns`:
    wealth starts at 1 before the first return, so the first period counts.
    A period returning -1 or less is a total loss, -1.
    """
    ruined = np.any(paths <= -1.0, axis=1)
    safe = np.where(paths <= -1.0, 0.0, paths)
    log_wealth = np.cumsum(np.log1p(safe), axis=1)
    peak = np.maximum(np.maximum.accumulate(log_wealth, axis=1), 0.0)
    depth = np.expm1(np.min(log_wealth - peak, axis=1))
    depth = np.minimum(depth, 0.0)
    return np.where(ruined, -1.0, depth)


#: Bootstrap paths built at once: bounds the memory of a long sample's draws
#: without changing them, since the draws are taken in order either way.
_CHUNK = 250


def _bootstrap_depths(
    portfolio: np.ndarray,
    rate: float,
    fractions: Sequence[float],
    n_bootstrap: int,
    block: int,
    random_seed: int,
) -> Optional[Dict[float, np.ndarray]]:
    """
    Each fraction's maximum drawdown on `n_bootstrap` circular-block
    resamples of the portfolio's per-period excess returns, every fraction
    on the same draws. None when no draws were asked for.
    """
    if n_bootstrap <= 0:
        return None
    n = portfolio.shape[0]
    rng = np.random.default_rng(int(random_seed))
    out: Dict[float, List[np.ndarray]] = {c: [] for c in fractions}
    remaining = int(n_bootstrap)
    while remaining > 0:
        count = min(_CHUNK, remaining)
        index = np.stack([circular_block_indices(n, block, rng) for _ in range(count)])
        drawn = portfolio[index]
        for c in fractions:
            out[c].append(_max_drawdowns(rate + c * drawn))
        remaining -= count
    return {c: np.concatenate(parts) for c, parts in out.items()}


def _condition(covariance: np.ndarray) -> float:
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    values = np.linalg.eigvalsh(correlation)
    if values[0] <= 0.0:
        return math.inf
    return float(values[-1] / values[0])


def kelly_report(
    returns: pd.DataFrame,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    drawdown_levels: Sequence[float] = DEFAULT_DRAWDOWN_LEVELS,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    random_seed: int = 0,
) -> Dict[str, Any]:
    """
    Kelly leverage, its uncertainty, and growth and drawdowns at fractions
    of it, for the aligned per-period return streams in `returns` (one
    column a stream, every row a date they share).

    `risk_free_rate` is an annual decimal, charged per period as rate /
    periods_per_year; leverage is financed at it. The bootstrap resamples
    circular blocks of the Kelly portfolio's own excess returns (default
    block n^(1/3)) with `random_seed`, so the same input gives the same
    draws. See the module docstring for every formula.

    Raises:
        ValidationError: fewer than MIN_PERIODS rows or no more rows than
        streams; a stream that never moved; streams so nearly collinear
        that Sigma^-1 mu is not determined; a fraction outside (0, 10] or
        a loss outside (0, 1).
    """
    names = [str(c) for c in returns.columns]
    values = returns.to_numpy(dtype=float)
    n, k = values.shape
    if n < max(MIN_PERIODS, k + 2):
        raise ValidationError(
            f"{_WHO}: {n} periods for {k} stream(s); a Kelly report needs at "
            f"least {max(MIN_PERIODS, k + 2)}. A mean estimated from fewer has "
            "an interval wider than any leverage worth reporting."
        )
    if not np.all(np.isfinite(values)):
        raise ValidationError(f"{_WHO}: the returns are not all finite.")
    for c in fractions:
        if not (0.0 < float(c) <= 10.0):
            raise ValidationError(
                f"{_WHO}: fraction {c!r} is not in (0, 10]; a fraction is a "
                "multiple of the Kelly leverage, 0.5 for half Kelly."
            )
    for d in drawdown_levels:
        if not (0.0 < float(d) < 1.0):
            raise ValidationError(
                f"{_WHO}: drawdown level {d!r} is not in (0, 1); it is the "
                "fraction of wealth lost, 0.5 for half."
            )
    periods = int(periods_per_year)
    rate = float(risk_free_rate) / periods
    excess = values - rate
    years = n / periods

    with single_threaded_blas():
        mean = excess.mean(axis=0)
        covariance = np.atleast_2d(np.cov(excess, rowvar=False, ddof=1))
    still = [names[i] for i in range(k) if not covariance[i, i] > 0.0]
    if still:
        raise ValidationError(
            f"{_WHO}: {still} never moved over the sample (zero variance), "
            "and Kelly divides by the variance. Drop the stream, or pass "
            "the period it varied over."
        )
    mu = mean * periods
    sigma2 = covariance * periods
    warnings: List[str] = []
    with single_threaded_blas():
        if k == 1:
            condition = 1.0
            kelly = np.array([mu[0] / sigma2[0, 0]])
            inverse_diagonal = np.array([1.0 / sigma2[0, 0]])
        else:
            condition = _condition(sigma2)
            if not condition <= MAX_CONDITION:
                raise ValidationError(
                    f"{_WHO}: the streams' correlation matrix has condition "
                    f"number {condition:.3g}, above {MAX_CONDITION:.0e}: some "
                    "of them are close to copies of each other, and Sigma^-1 "
                    "mu would lever into the noise between them. Drop the "
                    "duplicates or combine them first."
                )
            kelly = np.linalg.solve(sigma2, mu)
            inverse_diagonal = np.diag(np.linalg.inv(sigma2))
        theta2 = float(mu @ kelly)
    theta2 = max(theta2, 0.0)
    sharpe = math.sqrt(theta2)
    se = np.sqrt(inverse_diagonal / years)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    shrink = theta2 * years / (theta2 * years + k)
    normal = NormalDist()

    by_stream: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(names):
        f_i = float(kelly[i])
        by_stream[name] = {
            "kelly_fraction": f_i,
            "standard_error": float(se[i]),
            "interval": [f_i - z * float(se[i]), f_i + z * float(se[i])],
            "p_estimate_over_twice_truth": (
                normal.cdf(-abs(f_i) / float(se[i])) if f_i != 0.0 else math.nan
            ),
            "shrunk_fraction": shrink * f_i,
            "mean_excess_return": float(mu[i]),
            "volatility": math.sqrt(float(sigma2[i, i])),
            "sharpe_ratio": float(mu[i]) / math.sqrt(float(sigma2[i, i])),
        }
        if f_i < 0.0:
            warnings.append(
                f"{_WHO}: {name!r} has a negative mean excess return "
                f"({float(mu[i]):.4g} a year), so its Kelly leverage is "
                f"negative ({f_i:.4g}): Kelly SHORTS it. Where it cannot be "
                "shorted, the long-only Kelly holding is none of it."
            )

    with single_threaded_blas():
        portfolio = excess @ kelly
    found = empirical_kelly(excess, rate)
    empirical: Optional[Dict[str, float]] = None
    empirical_growth = math.nan
    if found is not None:
        with single_threaded_blas():
            empirical_growth = _growth(excess, rate, found) / years
    if found is None:
        warnings.append(
            f"{_WHO}: empirical_kelly is null: along some leverage the sample "
            "never lost, so more of it always grew the sample more and its "
            "realized growth has no maximum."
        )
    else:
        empirical = {name: float(found[i]) for i, name in enumerate(names)}

    block = int(block_size) if block_size is not None else max(1, round(n ** (1 / 3)))
    block = min(block, n)
    depths = _bootstrap_depths(
        portfolio, rate, [float(c) for c in fractions], n_bootstrap, block, random_seed
    )

    rows: List[Dict[str, Any]] = []
    for c in fractions:
        c = float(c)
        growth = theta2 * (c - c * c / 2.0)
        with_error = growth - c * c * k / (2.0 * years)
        path = rate + c * portfolio
        realized = (
            float(np.sum(np.log1p(path)) - n * math.log1p(rate)) / years
            if np.all(path > -1.0)
            else math.nan
        )
        row: Dict[str, Any] = {
            "fraction_of_kelly": c,
            "leverage": {name: c * float(kelly[i]) for i, name in enumerate(names)},
            "gross_leverage": c * float(np.sum(np.abs(kelly))),
            "excess_growth_rate": growth,
            "excess_growth_rate_with_estimation_error": with_error,
            "share_of_optimal_growth": (
                with_error / (theta2 / 2.0) if theta2 > 0.0 else math.nan
            ),
            "realized_excess_growth_rate": realized,
            "drawdown_probability": {
                f"{float(d):g}": (
                    drawdown_probability(c, float(d)) if theta2 > 0.0 else math.nan
                )
                for d in drawdown_levels
            },
            "max_drawdown_in_sample": float(_max_drawdowns(path[None, :])[0]),
            "max_drawdown_median": math.nan,
            "max_drawdown_worst_5pct": math.nan,
            "share_of_paths_ruined": math.nan,
        }
        if depths is not None:
            drawn = depths[c]
            row["max_drawdown_median"] = float(np.quantile(drawn, 0.5))
            row["max_drawdown_worst_5pct"] = float(np.quantile(drawn, 0.05))
            row["share_of_paths_ruined"] = float(np.mean(drawn <= -1.0))
        rows.append(row)

    if theta2 > 0.0 and shrink < 0.5:
        warnings.append(
            f"{_WHO}: the plug-in shrinkage is {shrink:.3f}: at a Sharpe "
            f"ratio of {sharpe:.3g} over {years:.3g} years most of f-hat is "
            "noise, and the leverage that maximises expected growth is less "
            "than half of it."
        )

    return {
        "stream_names": names,
        "n_streams": k,
        "n_periods": n,
        "periods_per_year": periods,
        "years": years,
        "risk_free_rate": float(risk_free_rate),
        "confidence": float(confidence),
        "by_stream": by_stream,
        "sharpe_ratio": sharpe,
        "optimal_excess_growth_rate": theta2 / 2.0,
        "gross_kelly_leverage": float(np.sum(np.abs(kelly))),
        "net_kelly_leverage": float(np.sum(kelly)),
        "p_estimate_over_twice_truth": normal.cdf(-sharpe * math.sqrt(years)),
        "shrink_factor": shrink,
        "condition_number": condition,
        "empirical_kelly": empirical,
        "empirical_kelly_excess_growth_rate": empirical_growth,
        "by_fraction": rows,
        "n_bootstrap": int(n_bootstrap),
        "block_size": block,
        "random_seed": int(random_seed),
        "warnings": warnings,
    }
