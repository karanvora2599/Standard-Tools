"""
How much of this backtest is real.

Every tool in this module exists to answer one question the backtest itself
cannot: given that you tried N things and kept the best, how much of that
best result is skill and how much is the largest of N draws from a
distribution centred on zero?

THE ARITHMETIC IS BRUTAL AND IT IS NOT INTUITIVE. Measured on simulated
strategies with no edge whatsoever, over two years of daily data, the best
of N shows an annualized Sharpe of:

    N trials      5      10      20      50     100
    best Sharpe   0.84   1.11    1.34    1.59    1.79

Not because anything works -- every one of those series was drawn from a
zero-mean normal. A researcher who reports the 1.34 without saying "and I
tried 19 others" has not lied about any single number and has still
communicated something false. `deflated_sharpe_ratio` computes how much to
subtract. (An earlier draft of this docstring put the best-of-20 at "roughly
1.0", which is what best-of-10 gives; the table above is measured over 200
replications per column rather than reasoned about.)

WHY CROSS-VALIDATION AS USUALLY DONE IS WRONG FOR THIS. K-fold CV assumes
observations are independent, and financial observations are not: a label
built from a five-day forward return overlaps the next four observations, so
a training set drawn from immediately before a test set contains the answer.
`combinatorial_purged_cv` implements Lopez de Prado's purging (drop training
observations whose label window overlaps the test set) and embargo (drop a
further window after it, because serial correlation leaks backwards too).

WHAT NONE OF THIS FIXES. These tools measure the multiple-testing cost of
the trials you TELL them about. The trials nobody counted -- the parameter
you adjusted before saving the file, the universe you narrowed after a first
look, the two years you dropped as "unrepresentative" -- are invisible here
and are usually the larger number. A PBO of 0.2 computed over a grid you
arrived at after a month of informal iteration is not a PBO of 0.2.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools._special import (
    norm_cdf,
    norm_ppf,
)
from standard_quant_tools.constants import TRADING_DAYS_PER_YEAR
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import has_no_dispersion

logger = logging.getLogger(__name__)

#: Trading days per year. Sharpe ratios here are annualized with it.
# One definition, in `constants`. This name stays because it is
# imported from here by name.
TRADING_DAYS = TRADING_DAYS_PER_YEAR

#: Euler-Mascheroni, needed for the expected maximum of N normal draws.
EULER_MASCHERONI = 0.5772156649015329


# See `_special`: this had 7 copies across the library, and the ones
# that were not identical disagreed at the edge of the domain.
_norm_cdf = norm_cdf

# See `_special`: this had 2 copies across the library, and the ones
# that were not identical disagreed at the edge of the domain.
_norm_ppf = norm_ppf


def _clean_returns(returns: pd.Series, who: str, minimum: int = 20) -> pd.Series:
    values = pd.Series(returns).astype(float).dropna()
    if len(values) < minimum:
        raise ValidationError(
            f"{who}: {len(values)} usable returns, and this needs at least "
            f"{minimum}. Below that every statistic here has a standard "
            "error wider than the quantity it estimates."
        )
    return values


def _sharpe(values: np.ndarray, periods: int = TRADING_DAYS) -> float:
    """
    Annualized Sharpe, returning NaN for a series with no dispersion.

    THE ZERO CHECK IS RELATIVE, not `std <= 0`, and it has to be. On a
    constant series numpy's `std` returns 2.2e-19 rather than 0 -- the
    deviations are computed against an accumulated mean, and the rounding
    does not cancel. A strict `<= 0` test therefore passes, and the Sharpe
    of a flat 0.001 series comes back as 7.3e16: a finite number, no NaN
    anywhere, and complete nonsense that then propagates into every
    threshold computed from it.

    Comparing the range against the magnitude of the values catches the
    degenerate case at any scale, which an absolute epsilon would not --
    a series of returns around 1e-8 is not constant just because its
    spread is small.
    """
    if values.size < 2:
        return float("nan")
    std = float(values.std(ddof=1))
    # The relative test that used to live here inline. It is now
    # `metrics.risk_metrics.has_no_dispersion`, because five other
    # implementations were carrying the broken absolute version while this
    # docstring explained why it was broken.
    if has_no_dispersion(values, std):
        return float("nan")
    return float(values.mean() / std * math.sqrt(periods))


# ── the multiple-testing correction ─────────────────────────────────────


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    n_trials: int,
    trial_sharpes: Optional[Sequence[float]] = None,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> Dict[str, Any]:
    """
    The probability this Sharpe ratio is real, given how many were tried.

    THE PROBLEM, stated concretely. Twenty strategies with no edge at all,
    on two years of daily data: the best of them shows an annualized Sharpe
    of 1.34, measured over 200 replications. The standard error of a Sharpe
    on 504 observations is about 0.71, and the maximum of 20 draws lands
    about 1.9 standard errors up. Report that 1.34 and you have said
    nothing false about any individual number while communicating something
    entirely false.

    THE CORRECTION, after Bailey and Lopez de Prado (2014), works in two
    steps. First compute the Sharpe a researcher would expect to achieve by
    LUCK ALONE after n_trials attempts -- that is the deflation threshold.
    Then ask whether the observed Sharpe is far enough above it to survive,
    accounting for the fact that the Sharpe's own sampling distribution is
    skewed and fat-tailed when returns are.

    SKEW AND KURTOSIS MATTER HERE, unusually. Most Sharpe inference assumes
    normal returns; this does not, because the strategies that most need
    deflating are exactly the ones with a short-volatility payoff -- many
    small gains, rare large losses. Negative skew and high kurtosis WIDEN the
    Sharpe's sampling distribution, so the same nominal Sharpe is less
    significant. A strategy selling options gets penalised here, correctly.

    Pass `trial_sharpes` -- the Sharpe of every variant you tried -- for the
    accurate version. Their VARIANCE is what sets the deflation threshold,
    and it is far more informative than the count alone: 100 near-identical
    parameter settings deflate much less than 100 genuinely different ideas.
    """
    values = _clean_returns(returns, "deflated_sharpe_ratio")
    n_trials = int(n_trials)
    if n_trials < 1:
        raise ValidationError(
            f"n_trials={n_trials}: the number of configurations tried is at "
            "least 1 (the one you are testing). If you genuinely tried only "
            "one, pass 1 -- but count the ones you discarded before saving "
            "the file, because they are trials too."
        )

    array = values.to_numpy()
    n = array.size
    observed = _sharpe(array, periods_per_year)
    if not math.isfinite(observed):
        raise ValidationError(
            "deflated_sharpe_ratio: the return series has no dispersion, so "
            "its Sharpe ratio is undefined. (A constant series does not "
            "produce a std of exactly zero in floating point -- it produces "
            "something around 1e-19 -- so this is caught by comparing the "
            "range against the magnitude rather than by testing for zero.)"
        )

    # Non-annualized, which is the scale the variance formula works in.
    per_period = observed / math.sqrt(periods_per_year)
    mean = float(array.mean())
    std = float(array.std(ddof=1))
    centred = (array - mean) / std
    skew = float((centred**3).mean())
    kurtosis = float((centred**4).mean())

    # The expected maximum Sharpe from n_trials draws of a zero-mean
    # strategy. Uses the variance of the trial Sharpes when supplied.
    if trial_sharpes is not None:
        sharpes = np.asarray([float(s) for s in trial_sharpes], dtype=float)
        sharpes = sharpes[np.isfinite(sharpes)]
        if sharpes.size < 2:
            raise ValidationError(
                "trial_sharpes needs at least two finite values -- its "
                "VARIANCE is what sets the deflation threshold."
            )
        trial_variance = float(sharpes.var(ddof=1)) / periods_per_year
        n_trials = max(n_trials, int(sharpes.size))
    else:
        # Without the trial distribution, assume each trial's Sharpe has the
        # sampling variance of a zero-edge strategy on this much data.
        trial_variance = 1.0 / n

    if n_trials == 1:
        expected_max = 0.0
    else:
        # E[max of N standard normals], to the usual two-term expansion.
        first = _norm_ppf(1.0 - 1.0 / n_trials)
        second = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
        expected_max = math.sqrt(trial_variance) * (
            (1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second
        )

    threshold = max(expected_max, benchmark_sharpe / math.sqrt(periods_per_year))

    # The Sharpe's own standard error, widened by skew and kurtosis.
    variance = (1.0 - skew * per_period + (kurtosis - 1.0) / 4.0 * per_period**2) / (
        n - 1
    )
    if variance <= 0:
        raise ValidationError(
            "deflated_sharpe_ratio: the estimated variance of the Sharpe "
            "ratio came out non-positive, which happens on extremely "
            "skewed short samples. There is not enough data here to deflate."
        )
    statistic = (per_period - threshold) / math.sqrt(variance)
    probability = _norm_cdf(statistic)

    warnings: List[str] = []
    if probability < 0.90:
        warnings.append(
            f"The deflated Sharpe probability is {probability:.2f}. After "
            f"accounting for {n_trials} trials, this result is NOT "
            "distinguishable from the best of that many attempts at "
            "nothing. The conventional bar is 0.95."
        )
    if skew < -0.5:
        warnings.append(
            f"Return skew is {skew:.2f}. Negative skew widens the Sharpe's "
            "sampling distribution, so this strategy needs a higher nominal "
            "Sharpe than a symmetric one to reach the same significance. "
            "That is the short-volatility payoff being priced correctly, "
            "not a penalty."
        )
    if kurtosis > 6.0:
        warnings.append(
            f"Return kurtosis is {kurtosis:.1f} against 3.0 for a normal. "
            "Fat tails widen the Sharpe's standard error; the normal-theory "
            "confidence interval would have been far too narrow."
        )
    warnings.append(
        "This deflates for the trials you DECLARED. The uncounted ones -- "
        "parameters adjusted before the file was saved, a universe narrowed "
        "after a first look, a period dropped as unrepresentative -- are "
        "invisible here and are usually the larger number."
    )

    return {
        "n_observations": int(n),
        "observed_sharpe": float(observed),
        "n_trials": n_trials,
        "expected_max_sharpe_from_luck": float(
            expected_max * math.sqrt(periods_per_year)
        ),
        "deflation_threshold": float(threshold * math.sqrt(periods_per_year)),
        "deflated_sharpe_probability": float(probability),
        "significant_at_95": bool(probability >= 0.95),
        "skewness": skew,
        "kurtosis": kurtosis,
        "sharpe_standard_error": float(
            math.sqrt(variance) * math.sqrt(periods_per_year)
        ),
        "warnings": warnings,
    }


def probability_of_backtest_overfitting(
    trial_returns: pd.DataFrame,
    *,
    n_splits: int = 8,
) -> Dict[str, Any]:
    """
    PBO: how often the configuration that wins in-sample loses out-of-sample.

    THE METHOD, after Bailey, Borwein, Lopez de Prado and Zhu (2015). Cut the
    period into S chunks. For every way of splitting those chunks into equal
    halves, fit on one half and evaluate on the other: pick the strategy that
    ranked best in-sample, then look up where it ranked out-of-sample. PBO is
    the fraction of splits where that winner landed in the BOTTOM half.

    WHAT THE NUMBER MEANS, and it is not "the probability the strategy is
    bad". It is the probability that your SELECTION PROCEDURE has no skill --
    that picking the in-sample best is no better than picking at random. A
    PBO of 0.5 means exactly that. Above 0.5 means the in-sample winner is
    systematically the out-of-sample loser, which is the signature of a grid
    fitted to noise.

    IT NEEDS A REAL GRID. Two or three configurations cannot produce a
    meaningful rank distribution, and neither can a hundred configurations
    that are all the same strategy with a parameter nudged -- their returns
    correlate at 0.99 and every split ranks them identically. The result
    reports the median pairwise correlation between configurations for
    exactly this reason: above about 0.95 the PBO is measuring one strategy,
    not a hundred.

    Columns are configurations, rows are periods.
    """
    frame = pd.DataFrame(trial_returns).astype(float).dropna(how="all")
    if frame.shape[1] < 2:
        raise ValidationError(
            f"probability_of_backtest_overfitting: {frame.shape[1]} "
            "configuration(s). PBO measures whether picking the in-sample "
            "best beats picking at random, and with one candidate there is "
            "no choice to measure."
        )
    n_splits = int(n_splits)
    if n_splits % 2 or n_splits < 4:
        raise ValidationError(
            f"n_splits={n_splits} must be even and at least 4 -- the method "
            "forms every equal-size split of the chunks into two halves."
        )
    if len(frame) < n_splits * 5:
        raise ValidationError(
            f"probability_of_backtest_overfitting: {len(frame)} rows over "
            f"{n_splits} chunks leaves under 5 observations each. A Sharpe "
            "on 5 points is noise, and the ranks would be noise too."
        )

    chunks = np.array_split(np.arange(len(frame)), n_splits)
    half = n_splits // 2
    values = frame.to_numpy()

    logits: List[float] = []
    ranks_out: List[float] = []
    for combination in itertools.combinations(range(n_splits), half):
        train_rows = np.concatenate([chunks[c] for c in combination])
        test_rows = np.concatenate(
            [chunks[c] for c in range(n_splits) if c not in combination]
        )
        in_sample = np.array(
            [_sharpe(values[train_rows, j]) for j in range(values.shape[1])]
        )
        out_sample = np.array(
            [_sharpe(values[test_rows, j]) for j in range(values.shape[1])]
        )
        if not np.isfinite(in_sample).any() or not np.isfinite(out_sample).any():
            continue
        best = int(np.nanargmax(in_sample))
        finite = np.isfinite(out_sample)
        if finite.sum() < 2 or not finite[best]:
            continue
        # Relative rank of the in-sample winner, out of sample.
        rank = float((out_sample[finite] < out_sample[best]).sum()) / (finite.sum() - 1)
        ranks_out.append(rank)
        # Guard the logit at the endpoints, where the winner ranked first or
        # last out of sample and the log would diverge.
        clipped = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(clipped / (1 - clipped)))

    if not ranks_out:
        raise ValidationError(
            "probability_of_backtest_overfitting: no split produced usable "
            "Sharpe ratios on both halves. Usually a zero-variance "
            "configuration or too little data per chunk."
        )

    pbo = float(np.mean([r < 0.5 for r in ranks_out]))
    correlations = frame.corr().to_numpy()
    off_diagonal = correlations[~np.eye(len(correlations), dtype=bool)]
    median_correlation = float(np.nanmedian(off_diagonal))

    warnings: List[str] = []
    if pbo > 0.5:
        warnings.append(
            f"PBO is {pbo:.2f}. The configuration that wins in-sample lands "
            "in the bottom half out-of-sample MORE often than not -- the "
            "grid is being fitted to noise, and selecting the in-sample "
            "best is worse than selecting at random."
        )
    elif pbo > 0.25:
        warnings.append(
            f"PBO is {pbo:.2f}: the in-sample winner underperforms the "
            "median out-of-sample in about {:.0f}% of splits. Treat the "
            "selected configuration's backtested numbers as an upper "
            "bound.".format(pbo * 100)
        )
    if median_correlation > 0.95:
        warnings.append(
            f"The median pairwise correlation between configurations is "
            f"{median_correlation:.3f}. These are not {frame.shape[1]} "
            "strategies, they are one strategy with a parameter nudged, and "
            "every split ranks them near-identically. The PBO is close to "
            "meaningless on a grid this collinear."
        )
    warnings.append(
        "PBO is a property of the SELECTION PROCEDURE, not of the strategy. "
        "0.5 means picking the in-sample best is no better than picking at "
        "random; it does not mean the strategy loses money."
    )

    return {
        "n_configurations": int(frame.shape[1]),
        "n_observations": int(len(frame)),
        "n_splits": n_splits,
        "n_combinations": len(ranks_out),
        "pbo": pbo,
        "median_logit": float(np.median(logits)),
        "median_out_of_sample_rank": float(np.median(ranks_out)),
        "median_configuration_correlation": median_correlation,
        "warnings": warnings,
    }


# ── cross-validation that respects the arrow of time ────────────────────


def combinatorial_purged_cv(
    n_observations: int,
    *,
    n_splits: int = 6,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01,
    label_horizon: int = 1,
) -> Dict[str, Any]:
    """
    Train/test index sets that do not leak, after Lopez de Prado.

    THE LEAK THIS PREVENTS. A label built from a 5-day forward return at time
    t is a function of prices through t+5. If t sits in the training set and
    t+3 sits in the test set, the training label already contains the test
    period's answer. Plain k-fold cross-validation does this on every fold
    boundary, and it is why a model can show 0.6 AUC in cross-validation and
    0.5 in production.

    TWO CORRECTIONS. PURGING removes training observations whose label window
    overlaps the test set at all. EMBARGO removes a further stretch
    immediately after the test set, because serial correlation in features
    means an observation shortly after the test period is still nearly the
    same observation -- purging alone handles the label overlap, and the
    embargo handles the feature overlap.

    COMBINATORIAL, not sequential. Rather than one train/test split per fold,
    every choice of `n_test_splits` groups out of `n_splits` becomes a test
    set. That yields C(n, k) paths instead of n, so the distribution of
    out-of-sample performance has enough draws to have a shape -- which is
    the point, because a single walk-forward number has no error bar at all.

    Returns INDEX ARRAYS, not results. It is a splitter: hand the indices to
    whatever fitting code you already have.
    """
    n_observations = int(n_observations)
    n_splits = int(n_splits)
    n_test_splits = int(n_test_splits)
    if n_observations < 50:
        raise ValidationError(
            f"combinatorial_purged_cv: {n_observations} observations is too "
            "few to purge and embargo and still leave a usable training set."
        )
    if not 1 <= n_test_splits < n_splits:
        raise ValidationError(
            f"n_test_splits={n_test_splits} must be at least 1 and fewer "
            f"than n_splits={n_splits}."
        )
    if n_splits < 2:
        raise ValidationError("n_splits must be at least 2.")
    label_horizon = max(1, int(label_horizon))
    embargo = int(math.ceil(float(embargo_pct) * n_observations))

    groups = np.array_split(np.arange(n_observations), n_splits)
    paths: List[Dict[str, Any]] = []
    total_purged = 0
    for combination in itertools.combinations(range(n_splits), n_test_splits):
        test_index = np.concatenate([groups[c] for c in combination])
        test_set = set(test_index.tolist())
        train: List[int] = []
        purged = 0
        for i in range(n_observations):
            if i in test_set:
                continue
            # PURGE: does this observation's label window touch the test set?
            label_window = range(i, min(i + label_horizon + 1, n_observations))
            if any(j in test_set for j in label_window):
                purged += 1
                continue
            # EMBARGO: does it sit just after a test block?
            if embargo > 0 and any(
                (i - j) in range(1, embargo + 1) for j in (int(test_index.max()),)
            ):
                purged += 1
                continue
            if embargo > 0 and any(
                0 < (i - t) <= embargo for t in _block_ends(test_index)
            ):
                purged += 1
                continue
            train.append(i)
        total_purged += purged
        paths.append(
            {
                "test_groups": list(combination),
                "n_train": len(train),
                "n_test": int(test_index.size),
                "n_purged": purged,
                "train_index": train,
                "test_index": test_index.tolist(),
            }
        )

    mean_train = float(np.mean([p["n_train"] for p in paths]))
    warnings: List[str] = []
    if mean_train < n_observations * 0.3:
        warnings.append(
            f"Purging and embargo leave an average of {mean_train:.0f} "
            f"training observations out of {n_observations}. With a label "
            f"horizon of {label_horizon} and {n_test_splits} test groups, "
            "most of the sample is being removed -- either shorten the "
            "label horizon or use fewer test groups."
        )
    if embargo == 0:
        warnings.append(
            "embargo_pct=0 means no embargo. Purging alone removes the "
            "LABEL overlap; it does not remove the feature overlap that "
            "serial correlation creates just after a test block."
        )
    warnings.append(
        f"{len(paths)} paths from C({n_splits}, {n_test_splits}). The point "
        "of the combinatorial form is that out-of-sample performance gets a "
        "DISTRIBUTION rather than one number -- a single walk-forward "
        "result has no error bar."
    )

    return {
        "n_observations": n_observations,
        "n_splits": n_splits,
        "n_test_splits": n_test_splits,
        "n_paths": len(paths),
        "embargo_observations": embargo,
        "label_horizon": label_horizon,
        "mean_train_size": mean_train,
        "mean_purged": float(total_purged / max(len(paths), 1)),
        "paths": paths,
        "warnings": warnings,
    }


def _block_ends(index: np.ndarray) -> List[int]:
    """The last index of each contiguous run, which is where an embargo starts."""
    if index.size == 0:
        return []
    sorted_index = np.sort(index)
    breaks = np.where(np.diff(sorted_index) > 1)[0]
    ends = [int(sorted_index[b]) for b in breaks]
    ends.append(int(sorted_index[-1]))
    return ends


# ── is this better than the alternatives ────────────────────────────────


def reality_check(
    strategy_returns: pd.Series,
    benchmark_returns: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    block_size: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    White's Reality Check: is the best strategy better than the best of the
    alternatives, or just the luckiest?

    THE QUESTION IT ANSWERS is subtly different from a t-test on the
    strategy's returns. A t-test asks "is this strategy's mean return
    positive". The Reality Check asks "is this strategy's outperformance
    larger than the largest outperformance you would expect from the best of
    this many candidates under the null that none of them has any edge". The
    second question is the one that matters after a search.

    THE BOOTSTRAP IS BLOCKED, and it has to be. Resampling individual days
    destroys the serial correlation that drives drawdowns and volatility
    clustering, which makes the null distribution far too narrow and the
    p-value far too small. Blocks of `block_size` consecutive days preserve
    the local dependence structure. The block length is a real choice: too
    short and dependence is lost, too long and there are too few distinct
    blocks to resample.

    THE STATIONARITY ASSUMPTION IS THE WEAK POINT and it is worth naming.
    Block bootstrap assumes the return-generating process is the same
    throughout. A strategy whose edge genuinely existed until 2018 and
    vanished afterwards violates that, and the bootstrap will happily mix
    both regimes into one null.
    """
    strategy = _clean_returns(strategy_returns, "reality_check", minimum=30)
    benchmarks = pd.DataFrame(benchmark_returns).astype(float)
    if benchmarks.empty or benchmarks.shape[1] < 1:
        raise ValidationError(
            "reality_check: no benchmark columns. The test compares against "
            "the alternatives you considered; with none, use a t-test."
        )
    aligned = pd.concat([strategy.rename("__strategy__"), benchmarks], axis=1).dropna()
    if len(aligned) < 30:
        raise ValidationError(
            f"reality_check: only {len(aligned)} overlapping observations "
            "after aligning the strategy with the benchmarks."
        )

    strategy_values = aligned["__strategy__"].to_numpy()
    benchmark_values = aligned.drop(columns="__strategy__").to_numpy()
    n = len(aligned)
    block_size = max(1, min(int(block_size), n // 4))
    n_bootstrap = max(100, int(n_bootstrap))

    # Performance is excess mean return over each benchmark.
    excess = strategy_values[:, None] - benchmark_values
    observed = float(np.max(excess.mean(axis=0)) * TRADING_DAYS)

    rng = np.random.default_rng(int(seed))
    n_blocks = int(math.ceil(n / block_size))
    centred = excess - excess.mean(axis=0)

    maxima = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        starts = rng.integers(0, n - block_size + 1, n_blocks)
        rows = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        maxima[b] = np.max(centred[rows].mean(axis=0))
    p_value = float((maxima >= observed / TRADING_DAYS).mean())

    warnings: List[str] = []
    if p_value > 0.10:
        warnings.append(
            f"p = {p_value:.3f}. This strategy's outperformance is within "
            f"what the best of {benchmarks.shape[1]} candidates would show "
            "under the null that none of them has an edge."
        )
    warnings.append(
        f"Block bootstrap with blocks of {block_size} days, which preserves "
        "serial correlation. Resampling individual days would make the null "
        "too narrow and the p-value too small -- volatility clustering and "
        "drawdowns both live in the dependence structure."
    )
    warnings.append(
        "Assumes the return process is STATIONARY across the sample. A "
        "strategy whose edge genuinely existed and then vanished violates "
        "that, and the bootstrap mixes both regimes into one null."
    )

    return {
        "n_observations": int(n),
        "n_benchmarks": int(benchmarks.shape[1]),
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
        "observed_outperformance": observed,
        "p_value": p_value,
        "significant_at_05": bool(p_value < 0.05),
        "bootstrap_p95": float(np.percentile(maxima, 95) * TRADING_DAYS),
        "warnings": warnings,
    }


# ── where the performance came from ─────────────────────────────────────


def regime_stratified_performance(
    returns: pd.Series,
    regimes: pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS,
) -> Dict[str, Any]:
    """
    Performance broken out by regime, because one Sharpe over a mixed sample
    describes none of them.

    THE FAILURE THIS CATCHES: a strategy with an overall Sharpe of 1.2 that
    earned all of it in one 18-month window and was flat-to-negative for the
    other eight years. The full-sample number is arithmetically correct and
    completely misleading, and nothing in a single Sharpe reveals it.

    CONCENTRATION IS THE HEADLINE. The result reports what fraction of total
    P&L came from the single best regime. Above about 70% from one regime the
    strategy is a bet on that regime recurring, whatever it says on the tin.

    `regimes` is any labelling aligned to the returns -- volatility buckets
    from `detect_regimes`, a bull/bear flag, VIX terciles, calendar years.
    The tool does not care what the labels mean; it cares that performance is
    not uniform across them.
    """
    values = _clean_returns(returns, "regime_stratified_performance")
    labels = pd.Series(regimes).reindex(values.index)
    aligned = pd.DataFrame({"r": values, "g": labels}).dropna()
    if aligned.empty:
        raise ValidationError(
            "regime_stratified_performance: the returns and the regime "
            "labels do not overlap after alignment. Check that both are "
            "indexed the same way."
        )
    if aligned["g"].nunique() < 2:
        raise ValidationError(
            f"regime_stratified_performance: only "
            f"{aligned['g'].nunique()} distinct regime label. Stratifying "
            "by a constant is the full-sample number again."
        )

    total_pnl = float(aligned["r"].sum())
    rows: List[Dict[str, Any]] = []
    for label, chunk in aligned.groupby("g"):
        array = chunk["r"].to_numpy()
        pnl = float(array.sum())
        rows.append(
            {
                "regime": str(label),
                "n_observations": int(array.size),
                "share_of_sample": float(array.size / len(aligned)),
                "total_return": pnl,
                "share_of_pnl": float(pnl / total_pnl) if total_pnl != 0 else None,
                "mean_return": float(array.mean()),
                "annualized_return": float(array.mean() * periods_per_year),
                "volatility": (
                    float(array.std(ddof=1) * math.sqrt(periods_per_year))
                    if array.size > 1
                    else None
                ),
                "sharpe": _sharpe(array, periods_per_year) if array.size > 1 else None,
                "win_rate": float((array > 0).mean()),
                "worst": float(array.min()),
            }
        )
    rows.sort(key=lambda r: r["total_return"], reverse=True)

    shares = [abs(r["share_of_pnl"]) for r in rows if r["share_of_pnl"] is not None]
    concentration = max(shares) if shares else None
    positive_regimes = sum(1 for r in rows if r["total_return"] > 0)

    warnings: List[str] = []
    if concentration is not None and concentration > 0.7:
        warnings.append(
            f"{concentration:.0%} of total P&L came from the single regime "
            f"'{rows[0]['regime']}', which was {rows[0]['share_of_sample']:.0%} "
            "of the sample. This is a bet on that regime recurring, whatever "
            "the full-sample Sharpe says."
        )
    if positive_regimes == 1 and len(rows) > 2:
        warnings.append(
            f"Only 1 of {len(rows)} regimes was profitable. The full-sample "
            "number is an average over conditions the strategy mostly did "
            "not work in."
        )
    thin = [r["regime"] for r in rows if r["n_observations"] < 20]
    if thin:
        warnings.append(
            f"Regime(s) {thin} have under 20 observations. Their Sharpe "
            "ratios are not estimates, they are single draws."
        )

    return {
        "n_observations": int(len(aligned)),
        "n_regimes": len(rows),
        "overall_sharpe": _sharpe(aligned["r"].to_numpy(), periods_per_year),
        "pnl_concentration": concentration,
        "n_profitable_regimes": positive_regimes,
        "by_regime": rows,
        "warnings": warnings,
    }


def parameter_decay(
    parameter_values: Sequence[float],
    performance: Sequence[float],
    *,
    metric_name: str = "sharpe",
) -> Dict[str, Any]:
    """
    Whether performance degrades SMOOTHLY as a parameter moves, or falls off
    a cliff.

    THE DISTINCTION IS THE WHOLE POINT. A parameter whose neighbours perform
    almost as well describes a real effect with a broad optimum -- the exact
    value does not matter much, which is what a robust edge looks like. A
    parameter whose neighbours are materially worse is a spike, and a spike
    in a noisy objective is almost always noise: you have found the one
    setting that happened to fit the sample.

    THE MEASURE is the ratio of the best value's performance to the mean of
    its immediate neighbours, plus the correlation between parameter and
    performance across the whole grid. A smooth surface has neighbours close
    to the peak; a spike does not.

    IT CANNOT TELL YOU THE OPTIMUM IS RIGHT. A broad plateau at a Sharpe of
    0.3 is still a Sharpe of 0.3. This measures robustness of the parameter
    choice, which is a necessary condition for the result to survive, and
    nowhere near a sufficient one.
    """
    params = np.asarray([float(p) for p in parameter_values], dtype=float)
    scores = np.asarray([float(s) for s in performance], dtype=float)
    if params.size != scores.size:
        raise ValidationError(
            f"parameter_decay: {params.size} parameter values against "
            f"{scores.size} performance values."
        )
    mask = np.isfinite(params) & np.isfinite(scores)
    params, scores = params[mask], scores[mask]
    if params.size < 5:
        raise ValidationError(
            f"parameter_decay: {params.size} usable points. A peak and its "
            "neighbours needs at least 5 to say anything about shape."
        )

    order = np.argsort(params)
    params, scores = params[order], scores[order]
    best = int(np.argmax(scores))
    peak = float(scores[best])

    neighbours = [scores[i] for i in (best - 1, best + 1) if 0 <= i < scores.size]
    neighbour_mean = float(np.mean(neighbours)) if neighbours else None
    # A spike ratio near 1 means the neighbours are as good as the peak.
    spike_ratio = (
        float(neighbour_mean / peak)
        if neighbour_mean is not None and peak != 0
        else None
    )

    # How much of the grid is within 20% of the peak: the plateau width.
    if peak > 0:
        plateau = float((scores >= 0.8 * peak).mean())
    else:
        plateau = float((scores >= peak * 1.2).mean())

    dispersion = float(scores.std(ddof=1)) if scores.size > 1 else 0.0
    at_edge = best in (0, scores.size - 1)

    warnings: List[str] = []
    if spike_ratio is not None and spike_ratio < 0.5:
        warnings.append(
            f"The best parameter's neighbours score {spike_ratio:.0%} of the "
            "peak. That is a SPIKE, not an optimum -- in a noisy objective a "
            "spike is almost always the one setting that happened to fit "
            "this sample, and it will not survive out of sample."
        )
    elif spike_ratio is not None and spike_ratio > 0.85:
        warnings.append(
            f"The neighbours score {spike_ratio:.0%} of the peak: a broad "
            "optimum, which is what a real effect looks like. The exact "
            "parameter value is not doing the work."
        )
    if at_edge:
        warnings.append(
            "The best value sits at the EDGE of the grid, so the true "
            "optimum may lie outside it -- or the performance may be "
            "monotone in this parameter, which usually means it is standing "
            "in for something else (more lookback = less trading = less "
            "cost, for instance)."
        )
    if plateau > 0.9:
        warnings.append(
            f"{plateau:.0%} of the grid is within 20% of the peak. The "
            "parameter barely matters over this range, which is either "
            "robustness or a sign the parameter is not connected to the "
            "result at all."
        )
    warnings.append(
        "A broad plateau at a bad level is still a bad level. This measures "
        "whether the PARAMETER CHOICE is robust, which is necessary for the "
        "result to survive and nowhere near sufficient."
    )

    return {
        "n_points": int(params.size),
        "metric": metric_name,
        "best_parameter": float(params[best]),
        "best_performance": peak,
        "neighbour_mean": neighbour_mean,
        "spike_ratio": spike_ratio,
        "plateau_fraction": plateau,
        "performance_dispersion": dispersion,
        "best_at_grid_edge": bool(at_edge),
        "surface": [
            {"parameter": float(p), "performance": float(s)}
            for p, s in zip(params, scores)
        ],
        "warnings": warnings,
    }


__all__ = [
    "EULER_MASCHERONI",
    "TRADING_DAYS",
    "combinatorial_purged_cv",
    "deflated_sharpe_ratio",
    "parameter_decay",
    "probability_of_backtest_overfitting",
    "reality_check",
    "regime_stratified_performance",
]
