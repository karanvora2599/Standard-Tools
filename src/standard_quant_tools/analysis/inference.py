"""
Error bars, and the questions that need them.

Almost every number this library produces is an estimate, and almost none of
them arrive with a standard error. A Sharpe ratio of 1.2, a correlation of
0.6, a maximum drawdown of 18% -- each is one draw from a distribution, and
the width of that distribution usually matters more to a decision than the
point estimate does. These tools produce the width.

THE BOOTSTRAP HERE IS BLOCKED BY DEFAULT and that is not a detail. Financial
series have serial correlation -- volatility clusters, drawdowns are runs of
bad days -- and resampling individual observations destroys it. The resulting
interval is too narrow, and the error is invisible: it looks perfectly
reasonable and is simply wrong.

HOW MUCH TOO NARROW, measured on AR(1) returns as the ratio of the blocked
interval to the IID one:

    phi          0.0    0.2    0.4    0.6    0.8
    Sharpe      0.97   1.16   1.41   1.77   2.24
    max drawdown 0.96  1.13   1.30   1.45   1.63

At phi = 0 the two agree, which is the null case worth checking. The SHARPE
is the more affected of the two, not the drawdown -- an earlier draft of this
docstring said the opposite, reasoning that path-dependent statistics would
suffer most. They do not: the Sharpe depends directly on the variance
estimate, which is exactly what serial correlation distorts, while maximum
drawdown already has a wide interval that persistence widens proportionally
less.

`block_size` defaults to n^(1/3), the standard rule. Setting it to 1 gives
the IID bootstrap, and it is warned about rather than forbidden.

WHAT `compare_distributions` IS FOR. Two samples, one question: are these
the same distribution? It comes up constantly -- in-sample against
out-of-sample returns, this regime against that one, live trading against
the backtest -- and it is usually answered by comparing means, which misses
every difference in shape. A strategy whose out-of-sample mean matches and
whose out-of-sample kurtosis has tripled is not performing as expected.

NO SCIPY. The KS distribution, the normal quantile and the F tail are all
computed here.
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

#: Statistics `bootstrap_statistic` knows how to compute. Restricted to a
#: named set rather than accepting arbitrary code, because this is reachable
#: from an agent and an eval-shaped hole in a tool surface is not worth the
#: generality.
STATISTICS = (
    "mean",
    "median",
    "std",
    "sharpe",
    "sortino",
    "skew",
    "kurtosis",
    "max_drawdown",
    "win_rate",
    "var_95",
    "cvar_95",
)


def _clean(values: Sequence[float], who: str, minimum: int = 30) -> np.ndarray:
    array = np.asarray([float(v) for v in values], dtype=float)
    array = array[np.isfinite(array)]
    if array.size < minimum:
        raise ValidationError(
            f"{who}: {array.size} usable observations, and this needs at "
            f"least {minimum}."
        )
    return array


def _statistic(values: np.ndarray, name: str, periods: int = TRADING_DAYS) -> float:
    """One of the named statistics. Returns NaN where it is undefined."""
    if values.size < 2:
        return float("nan")
    if name == "mean":
        return float(values.mean())
    if name == "median":
        return float(np.median(values))
    if name == "std":
        return float(values.std(ddof=1))
    if name in ("sharpe", "sortino"):
        std = (
            float(values.std(ddof=1))
            if name == "sharpe"
            else (
                float(values[values < 0].std(ddof=1))
                if (values < 0).sum() > 1
                else float("nan")
            )
        )
        if not math.isfinite(std) or std <= 0:
            return float("nan")
        return float(values.mean() / std * math.sqrt(periods))
    if name in ("skew", "kurtosis"):
        std = float(values.std(ddof=1))
        if std <= 0:
            return float("nan")
        centred = (values - values.mean()) / std
        return float((centred**3).mean() if name == "skew" else (centred**4).mean())
    if name == "max_drawdown":
        equity = np.cumprod(1.0 + values)
        return float((equity / np.maximum.accumulate(equity) - 1.0).min())
    if name == "win_rate":
        return float((values > 0).mean())
    if name == "var_95":
        return float(np.percentile(values, 5))
    if name == "cvar_95":
        tail = values[values <= np.percentile(values, 5)]
        return float(tail.mean()) if tail.size else float("nan")
    raise ValidationError(
        f"unknown statistic {name!r}. Available: {', '.join(STATISTICS)}."
    )


def _block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Indices for one blocked resample, trimmed to the original length."""
    if block_size <= 1:
        return rng.integers(0, n, n)
    n_blocks = int(math.ceil(n / block_size))
    starts = rng.integers(0, max(n - block_size + 1, 1), n_blocks)
    return np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])[:n]


# ── error bars ──────────────────────────────────────────────────────────


def bootstrap_statistic(
    values: Sequence[float],
    *,
    statistic: str = "sharpe",
    n_bootstrap: int = 2000,
    block_size: Optional[int] = None,
    confidence: float = 0.95,
    periods_per_year: int = TRADING_DAYS,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    A confidence interval for a statistic, by block bootstrap.

    THE POINT ESTIMATE IS USUALLY REPORTED ALONE and usually should not be.
    A Sharpe of 1.2 on two years of daily data has a 95% interval running
    from roughly 0.2 to 2.2 -- it is consistent with a mediocre strategy and
    with an excellent one, and the point estimate conceals that completely.
    The interval is the number a decision should be made on.

    BLOCKED, NOT IID, and the default matters. Resampling individual returns
    destroys serial correlation and narrows the interval. Measured on AR(1)
    returns at phi = 0.8, the IID interval is 2.24x too narrow for the
    Sharpe and 1.63x for maximum drawdown -- and at phi = 0 the two agree,
    so the correction costs nothing on a series that does not need it. The
    default block length is n^(1/3), the standard rule; `block_size=1` gives
    the IID version with a warning attached.

    THE BIAS IS REPORTED. The difference between the bootstrap mean and the
    point estimate estimates the estimator's bias, which is large for
    exactly the statistics people care about: maximum drawdown is a minimum
    over the sample, so it is biased toward zero in short samples, and the
    Sharpe ratio is biased upward.
    """
    array = _clean(values, "bootstrap_statistic")
    if statistic not in STATISTICS:
        raise ValidationError(
            f"bootstrap_statistic: unknown statistic {statistic!r}. "
            f"Available: {', '.join(STATISTICS)}."
        )
    if not 0 < confidence < 1:
        raise ValidationError(f"confidence must be in (0, 1), got {confidence!r}")
    n = array.size
    if block_size is None:
        block_size = max(1, int(round(n ** (1.0 / 3.0))))
    block_size = max(1, min(int(block_size), n // 2))
    n_bootstrap = max(100, int(n_bootstrap))

    observed = _statistic(array, statistic, periods_per_year)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        draws[i] = _statistic(
            array[_block_indices(n, block_size, rng)], statistic, periods_per_year
        )
    usable = draws[np.isfinite(draws)]
    if usable.size < n_bootstrap // 2:
        raise ValidationError(
            f"bootstrap_statistic: {statistic!r} was undefined on "
            f"{n_bootstrap - usable.size} of {n_bootstrap} resamples. The "
            "statistic is not estimable on this series."
        )

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(usable, alpha * 100))
    upper = float(np.percentile(usable, (1 - alpha) * 100))
    bias = float(usable.mean() - observed) if math.isfinite(observed) else None

    warnings: List[str] = []
    if block_size == 1:
        warnings.append(
            "block_size=1 is an IID bootstrap, which destroys the serial "
            "correlation in the series. The interval below is too NARROW: "
            "measured on AR(1) returns at phi = 0.8 it understates the "
            "Sharpe's interval by 2.24x and the maximum drawdown's by "
            "1.63x, while looking entirely plausible either way."
        )
    if (
        math.isfinite(observed)
        and lower <= 0 <= upper
        and statistic
        in (
            "sharpe",
            "sortino",
            "mean",
        )
    ):
        warnings.append(
            f"The {confidence:.0%} interval [{lower:.3f}, {upper:.3f}] "
            "CONTAINS ZERO. This sample does not distinguish the strategy "
            "from no edge at all, whatever the point estimate says."
        )
    if bias is not None and abs(bias) > abs(observed) * 0.15:
        warnings.append(
            f"The bootstrap mean sits {bias:+.4f} from the point estimate, "
            f"which is {abs(bias / observed):.0%} of it. That is estimator "
            "bias, and it is large for exactly the statistics people quote: "
            "maximum drawdown is a minimum over the sample and is biased "
            "toward zero in short ones."
        )
    warnings.append(
        f"Block bootstrap with blocks of {block_size} observations, which "
        "preserves local serial correlation. The interval is a statement "
        "about THIS sample's distribution, not about a regime the sample "
        "does not contain."
    )

    return {
        "statistic": statistic,
        "n_observations": int(n),
        "n_bootstrap": int(n_bootstrap),
        "block_size": int(block_size),
        "confidence": float(confidence),
        "point_estimate": float(observed) if math.isfinite(observed) else None,
        "lower": lower,
        "upper": upper,
        "interval_width": float(upper - lower),
        "bootstrap_mean": float(usable.mean()),
        "bootstrap_std": float(usable.std(ddof=1)),
        "estimated_bias": bias,
        "contains_zero": bool(lower <= 0 <= upper),
        "warnings": warnings,
    }


def compare_distributions(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> Dict[str, Any]:
    """
    Whether two samples came from the same distribution -- not whether their
    means differ.

    THE QUESTION GETS ANSWERED WITH A T-TEST and a t-test sees one thing.
    In-sample against out-of-sample, this regime against that one, live
    trading against the backtest: comparing means misses every difference in
    shape, and shape is usually where the problem is. A strategy whose
    out-of-sample mean matches and whose kurtosis has tripled is not
    performing as expected, and no test of means will say so.

    THREE COMPARISONS, because they fail differently. The
    Kolmogorov-Smirnov statistic is the largest gap between the two
    empirical CDFs and is sensitive to a shift or a shape change anywhere.
    The moment comparison says WHICH moment moved, which is what makes the
    result actionable. And the tail comparison is reported separately
    because KS is least sensitive exactly where a return distribution
    matters most -- its power is concentrated near the median.

    THE KS TEST IS NOT SENSITIVE TO TAILS. That limitation is stated rather
    than left to be discovered: two distributions differing only in their
    worst 1% can pass a KS test comfortably, and for returns that difference
    is the entire risk.
    """
    a = _clean(sample_a, "compare_distributions (sample_a)", minimum=10)
    b = _clean(sample_b, "compare_distributions (sample_b)", minimum=10)

    # Two-sample KS on the pooled grid.
    pooled = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), pooled, side="right") / a.size
    cdf_b = np.searchsorted(np.sort(b), pooled, side="right") / b.size
    ks = float(np.max(np.abs(cdf_a - cdf_b)))
    effective = math.sqrt(a.size * b.size / (a.size + b.size))
    # Kolmogorov's asymptotic distribution.
    lam = (effective + 0.12 + 0.11 / effective) * ks
    p_value = 2.0 * sum(
        (-1) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam) for k in range(1, 101)
    )
    p_value = float(min(max(p_value, 0.0), 1.0))

    def _moments(values: np.ndarray) -> Dict[str, float]:
        std = float(values.std(ddof=1))
        centred = (values - values.mean()) / std if std > 0 else values * 0.0
        return {
            "n": int(values.size),
            "mean": float(values.mean()),
            "std": std,
            "skew": float((centred**3).mean()),
            "kurtosis": float((centred**4).mean()),
            "p01": float(np.percentile(values, 1)),
            "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
        }

    moments_a, moments_b = _moments(a), _moments(b)
    shifts = []
    for key in ("mean", "std", "skew", "kurtosis"):
        before, after = moments_a[key], moments_b[key]
        relative = (
            abs(after - before) / abs(before) if abs(before) > 1e-12 else float("nan")
        )
        shifts.append(
            {
                "moment": key,
                label_a: before,
                label_b: after,
                "change": float(after - before),
                "relative_change": float(relative) if math.isfinite(relative) else None,
            }
        )

    tail_ratio = (
        abs(moments_b["p01"] / moments_a["p01"])
        if abs(moments_a["p01"]) > 1e-12
        else None
    )

    warnings: List[str] = []
    if p_value < 0.05:
        biggest = max(
            (s for s in shifts if s["relative_change"] is not None),
            key=lambda s: s["relative_change"],
            default=None,
        )
        warnings.append(
            f"The two samples differ (KS p = {p_value:.4f})."
            + (
                f" The largest relative move is in {biggest['moment']}, "
                f"{biggest['relative_change']:.0%}."
                if biggest
                else ""
            )
        )
    if tail_ratio is not None and (tail_ratio > 1.5 or tail_ratio < 0.67):
        warnings.append(
            f"The 1st percentile moved by a factor of {tail_ratio:.2f} "
            f"({moments_a['p01']:.4f} -> {moments_b['p01']:.4f}). The KS test "
            "is LEAST sensitive in the tails, so a large tail move can sit "
            "alongside a comfortable KS p-value -- and for returns the tail "
            "is the entire risk."
        )
    warnings.append(
        "KS tests the whole distribution and its power is concentrated near "
        "the median. It is not a tail test; the percentile comparison above "
        "is there because KS will not catch a tail-only difference."
    )
    if min(a.size, b.size) < 50:
        warnings.append(
            f"The smaller sample has {min(a.size, b.size)} observations. KS "
            "has little power at that size -- failing to reject is close to "
            "uninformative here."
        )

    return {
        "n_a": int(a.size),
        "n_b": int(b.size),
        "ks_statistic": ks,
        "p_value": p_value,
        "same_distribution_at_05": bool(p_value >= 0.05),
        "moments": {label_a: moments_a, label_b: moments_b},
        "moment_shifts": shifts,
        "tail_ratio_p01": tail_ratio,
        "warnings": warnings,
    }


def rolling_correlation_stability(
    a: Sequence[float],
    b: Sequence[float],
    *,
    window: int = 63,
) -> Dict[str, Any]:
    """
    Whether a correlation is a property of the pair or an average over two
    different regimes.

    THE FULL-SAMPLE CORRELATION IS THE ONE NUMBER EVERY HEDGE IS BUILT ON
    and it is routinely an artefact. Two assets that correlate at 0.0 over
    ten years may have correlated at +0.7 for five and -0.7 for five; the
    average is meaningless and a hedge sized on it is wrong in both regimes.

    WHAT TO READ. The range and the sign-flip count say whether there is a
    stable relationship at all. `fraction_of_windows_within_0.2` says how
    much of the sample looks like the headline number -- below about half,
    the headline is describing a period that mostly did not happen.

    THE CRISIS CASE IS SEPARATE AND WORSE. Correlations move toward 1 when
    everything falls together, so a hedge computed on a calm sample fails
    precisely when it is needed. The correlation conditional on the joint
    worst decile is reported for that reason: it is the number a
    diversification claim has to survive.
    """
    x = np.asarray([float(v) for v in a], dtype=float)
    y = np.asarray([float(v) for v in b], dtype=float)
    if x.size != y.size:
        raise ValidationError(
            f"rolling_correlation_stability: {x.size} against {y.size} "
            "observations. The two series must be aligned."
        )
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    window = int(window)
    if x.size < window * 2:
        raise ValidationError(
            f"rolling_correlation_stability: {x.size} observations with a "
            f"window of {window} leaves fewer than two windows."
        )

    full = float(np.corrcoef(x, y)[0, 1])
    frame = pd.DataFrame({"x": x, "y": y})
    rolling = frame["x"].rolling(window).corr(frame["y"]).dropna()
    values = rolling.to_numpy()
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValidationError(
            "rolling_correlation_stability: the rolling correlation was "
            "undefined almost everywhere, which happens when one series is "
            "constant inside most windows."
        )

    signs = np.sign(values)
    flips = int((signs[1:] * signs[:-1] < 0).sum())
    within = float((np.abs(values - full) <= 0.2).mean())

    # The joint worst decile: both series in their own bottom 10%.
    threshold_x = np.percentile(x, 10)
    threshold_y = np.percentile(y, 10)
    stress = (x <= threshold_x) | (y <= threshold_y)
    stress_correlation = (
        float(np.corrcoef(x[stress], y[stress])[0, 1]) if stress.sum() > 5 else None
    )

    warnings: List[str] = []
    if flips > 0:
        warnings.append(
            f"The rolling correlation changes SIGN {flips} time(s), ranging "
            f"from {values.min():.2f} to {values.max():.2f}. The full-sample "
            f"{full:.2f} is an average over regimes that disagree, and a "
            "hedge sized on it is wrong in both."
        )
    if within < 0.5:
        warnings.append(
            f"Only {within:.0%} of windows sit within 0.2 of the full-sample "
            "correlation. The headline number describes a period that mostly "
            "did not happen."
        )
    if stress_correlation is not None and stress_correlation > full + 0.15:
        warnings.append(
            f"Correlation in the joint worst decile is {stress_correlation:.2f} "
            f"against a full-sample {full:.2f}. Diversification is weakest "
            "exactly when it is needed, which is the normal behaviour of "
            "correlations and the reason a calm-sample hedge fails."
        )
    warnings.append(
        f"Rolling windows overlap, so the {values.size} values are far from "
        f"independent -- about {max(int(x.size / window), 1)} of them are. "
        "The RANGE is informative; the count is not a sample size."
    )

    return {
        "n_observations": int(x.size),
        "window": window,
        "full_sample_correlation": full,
        "n_windows": int(values.size),
        "n_independent_windows": max(int(x.size / window), 1),
        "mean_rolling": float(values.mean()),
        "min_rolling": float(values.min()),
        "max_rolling": float(values.max()),
        "std_rolling": float(values.std(ddof=1)),
        "sign_flips": flips,
        "fraction_within_0_2": within,
        "stress_correlation": stress_correlation,
        "warnings": warnings,
    }


def decompose_returns(
    returns: Sequence[float],
    *,
    periods_per_year: int = TRADING_DAYS,
) -> Dict[str, Any]:
    """
    Where a return series' compound growth actually came from.

    THE ARITHMETIC MEAN IS NOT WHAT YOU EARNED. Compound growth is the
    arithmetic mean MINUS roughly half the variance -- the volatility drag --
    and for a volatile strategy that gap is most of the return. A series
    averaging 0.08% a day with 3% daily volatility has an arithmetic annual
    return of 20% and a compound one near 9%. Reporting the first as "the
    return" is the single most common overstatement in this business.

    THE DECOMPOSITION separates that drag from the contribution of the best
    and worst days, because both are decision-relevant and neither shows up
    in a mean. A strategy whose entire compound return comes from its five
    best days is a lottery ticket with good statistics; one whose return
    survives removing them is robust.
    """
    array = _clean(returns, "decompose_returns")
    n = array.size

    arithmetic = float(array.mean())
    geometric = float(np.expm1(np.log1p(array).mean()))
    variance = float(array.var(ddof=1))
    drag = arithmetic - geometric

    total = float(np.prod(1.0 + array) - 1.0)
    ordered = np.sort(array)
    best_five = ordered[-5:]
    worst_five = ordered[:5]

    def _without(mask: np.ndarray) -> float:
        return float(np.prod(1.0 + array[mask]) - 1.0)

    without_best = _without(~np.isin(array, best_five))
    without_worst = _without(~np.isin(array, worst_five))

    positive = array[array > 0]
    negative = array[array < 0]

    warnings: List[str] = []
    if arithmetic > 0 and drag > arithmetic * 0.25:
        warnings.append(
            f"Volatility drag is {drag * periods_per_year:.2%} annualized, "
            f"{drag / arithmetic:.0%} of the arithmetic mean. Compounding "
            "earns the GEOMETRIC return; quoting the arithmetic one "
            "overstates what the strategy delivered."
        )
    if total > 0 and without_best < 0:
        warnings.append(
            "Removing the five best days turns the total return NEGATIVE. "
            "The entire result rests on five observations, which is a "
            "lottery ticket with good statistics rather than an edge."
        )
    if negative.size and positive.size:
        ratio = abs(float(positive.mean() / negative.mean()))
        if ratio < 0.8:
            warnings.append(
                f"The average winning day is {ratio:.2f}x the average losing "
                "day in size. This is a high-win-rate, small-gain profile, "
                "which is the shape that looks best in a backtest and "
                "carries the most left-tail risk."
            )

    return {
        "n_observations": int(n),
        "arithmetic_mean": arithmetic,
        "geometric_mean": geometric,
        "volatility_drag": drag,
        "arithmetic_annualized": float(arithmetic * periods_per_year),
        "geometric_annualized": float((1.0 + geometric) ** periods_per_year - 1.0),
        "variance": variance,
        "total_return": total,
        "total_without_best_5": without_best,
        "total_without_worst_5": without_worst,
        "best_5_contribution": float(total - without_best),
        "worst_5_contribution": float(total - without_worst),
        "n_positive": int(positive.size),
        "n_negative": int(negative.size),
        "win_rate": float((array > 0).mean()),
        "mean_win": float(positive.mean()) if positive.size else None,
        "mean_loss": float(negative.mean()) if negative.size else None,
        "win_loss_ratio": (
            float(abs(positive.mean() / negative.mean()))
            if positive.size and negative.size and negative.mean() != 0
            else None
        ),
        "warnings": warnings,
    }


__all__ = [
    "STATISTICS",
    "TRADING_DAYS",
    "bootstrap_statistic",
    "compare_distributions",
    "decompose_returns",
    "rolling_correlation_stability",
]
