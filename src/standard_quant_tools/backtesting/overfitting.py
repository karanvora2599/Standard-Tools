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
observations whose label window overlaps the test set's, before it or after
it) and embargo (drop a further window after it, because serial correlation
leaks backwards too).

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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools._special import (
    norm_cdf,
    norm_ppf,
)
from standard_quant_tools.backtest.robustness import (
    expected_max_sharpe,
    sharpe_standard_error_factor,
)
from standard_quant_tools.constants import EULER_MASCHERONI as _EULER_MASCHERONI
from standard_quant_tools.constants import TRADING_DAYS_PER_YEAR
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import annualized_sharpe

logger = logging.getLogger(__name__)

#: Trading days per year. Sharpe ratios here are annualized with it.
# One definition, in `constants`. This name stays because it is
# imported from here by name.
TRADING_DAYS = TRADING_DAYS_PER_YEAR

#: Euler-Mascheroni, needed for the expected maximum of N normal draws.
# One definition, in `constants`. Re-exported because this name is
# in this module's __all__.
EULER_MASCHERONI = _EULER_MASCHERONI


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
    `metrics.risk_metrics.annualized_sharpe`, with this module's period
    count as the default.

    The arithmetic used to live here, and the docstring explaining WHY the
    dispersion test is relative rather than `std <= 0` lived here with it --
    while `analysis.diagnostics` and `backtesting.trade_analysis` carried
    their own copies of the same three statements without it. All three
    call one function now.

    This wrapper stays because the DEFAULT is load-bearing: the CSCV split
    below calls `_sharpe(column)` with no period argument, and the shared
    primitive deliberately defaults to un-annualized so the per-trade caller
    can use it too.
    """
    return annualized_sharpe(values, periods)


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

    # Bailey and Lopez de Prado's expected maximum, from
    # `backtest.robustness`. `deflated_sharpe_ratio` there computed the
    # same thing inline; they agreed, and nothing kept them agreeing.
    expected_max = expected_max_sharpe(math.sqrt(trial_variance), n_trials)

    threshold = max(expected_max, benchmark_sharpe / math.sqrt(periods_per_year))

    # The Sharpe's own standard error, widened by skew and kurtosis. Shared
    # with `robustness.deflated_sharpe_ratio`, which had its own copy: the
    # factor there multiplies 1/sqrt(n - 1), so this is the same quantity
    # written as a variance.
    factor = sharpe_standard_error_factor(per_period, skew, kurtosis)
    variance = factor**2 / (n - 1)
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
    the fraction of splits where that winner landed in the BOTTOM half, a
    split that puts it exactly on the median counting as half -- with an odd
    number of configurations the middle rank is neither half, and leaving it
    out made a no-skill grid of three read 0.33 rather than 0.5.

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
    # Each split's vote that the winner landed in the bottom half: 1 below
    # the median, 0 above it, and 1/2 exactly ON it. With an odd number of
    # configurations the middle rank is exactly 0.5, and counting it as
    # "not below" put the no-skill PBO at (N - 1) / 2N -- 0.33 for three
    # configurations, which reads as skill there is none of (see the
    # CHANGELOG entry of 2026-10-22). With an even number no rank is 0.5
    # and nothing changes.
    below_median: List[float] = []
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
        beaten = int((out_sample[finite] < out_sample[best]).sum())
        others = int(finite.sum()) - 1
        rank = float(beaten) / others
        ranks_out.append(rank)
        # Compared in integers: 2 x beaten against the others, so a rank of
        # exactly one half is found without a float equality.
        below_median.append(
            1.0 if 2 * beaten < others else 0.5 if 2 * beaten == others else 0.0
        )
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

    pbo = float(np.mean(below_median))
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
    overlaps the test set at all, on both sides: before a test block, the
    rows whose labels reach into it; after it, the rows inside the reach of
    its own labels, whose labels share bars with the test labels. EMBARGO
    removes a further stretch
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

    paths: List[Dict[str, Any]] = []
    total_purged = 0
    for combination, test_index, train, purged in _purged_paths(
        n_observations, n_splits, n_test_splits, embargo, label_horizon
    ):
        total_purged += purged
        paths.append(
            {
                "test_groups": list(combination),
                "n_train": int(train.size),
                "n_test": int(test_index.size),
                "n_purged": purged,
                "train_index": train.tolist(),
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


def _purged_paths(
    n_observations: int,
    n_splits: int,
    n_test_splits: int,
    embargo: int,
    label_horizon: int,
) -> Iterator[Tuple[Tuple[int, ...], np.ndarray, np.ndarray, int]]:
    """
    `(test groups, test index, train index, number purged)` for every path,
    each index an ascending integer array.

    An observation outside the test set is dropped from training when
    either rule reaches it, and counted once:

    - PURGE: a test observation lies in (i, i + label_horizon] -- its label
      window touches the test set -- or in [i - label_horizon, i) -- it
      sits inside a test observation's label window, so its own label is
      built from bars the test labels are still resolving over;
    - EMBARGO: the end of a test block lies in [i - embargo, i) -- it sits
      just after a block. Every block's end counts, including the last one.

    THE PURGE USED TO HAVE ONE SIDE. Only the rows before a block were
    purged, and the rows inside the block's label reach after it were kept
    unless the embargo happened to cover them. Measured on 1,000
    observations, six groups and two held out: with a 5-bar label and no
    embargo, 100 training rows across the 15 paths sat inside a test
    label's window; with a 20-bar label the default 1% embargo covered ten
    of the twenty bars and 200 rows leaked. Lopez de Prado purges a
    training observation whose label interval overlaps a test label's on
    either side, and `modeling.validation.label_overlap_mask` already did.
    The embargo stays measured from the block's last observation, as the
    modeling splitters measure it: it is the gap for feature lookback, a
    different leak from the label overlap the purge now covers.

    Both are answered for all observations at once, which is what keeps a
    path linear in the length of the series. Written as a loop over the
    observations -- scan each label window, then every block's end -- it
    cost the length times the horizon plus the length times the number of
    test observations, so 10,000 observations took seconds and 100,000
    took minutes. The two forms give the same indices; the test suite
    holds them to it on randomized inputs.
    """
    positions = np.arange(n_observations)
    # A label window never reaches past the last observation, and a gap
    # between two positions is never longer than the series: clipping both
    # here changes no answer and keeps the arithmetic inside int64.
    horizon = min(label_horizon, n_observations)
    window_end = np.minimum(positions + horizon, n_observations - 1)
    # The earliest observation whose label window can reach each position.
    window_start = np.maximum(positions - horizon, 0)
    reach = min(embargo, n_observations)
    groups = np.array_split(positions, n_splits)
    for combination in itertools.combinations(range(n_splits), n_test_splits):
        test_index = np.concatenate([groups[c] for c in combination])
        in_test = np.zeros(n_observations, dtype=bool)
        in_test[test_index] = True
        # How many test observations lie at or before each position, so the
        # count strictly after i and up to its window's end is a difference.
        seen = np.cumsum(in_test)
        dropped = seen[window_end] > seen
        # The other side: test observations strictly before each position,
        # so the count in [i - horizon, i) is a difference too.
        prior = seen - in_test
        dropped |= prior > prior[window_start]
        if reach > 0:
            block_ends = np.flatnonzero(in_test & ~np.append(in_test[1:], False))
            if block_ends.size:
                # The nearest block end strictly before each position.
                before = np.searchsorted(block_ends, positions, side="left") - 1
                nearest = block_ends[np.maximum(before, 0)]
                dropped |= (before >= 0) & (positions - nearest <= reach)
        dropped &= ~in_test
        train = np.flatnonzero(~in_test & ~dropped)
        yield combination, test_index, train, int(np.count_nonzero(dropped))


# ── is this better than the alternatives ────────────────────────────────

#: The hypotheses `reality_check` tests. Each is a different question, and
#: the caller names the one they mean.
REALITY_CHECK_QUESTIONS: Tuple[str, ...] = (
    "best_candidate_vs_benchmark",
    "beats_every_alternative",
)

#: How `best_candidate_vs_benchmark` builds its statistic.
REALITY_CHECK_METHODS: Tuple[str, ...] = ("white", "hansen_spa")


def _rc_frame(values: Any, name: str) -> pd.DataFrame:
    frame = pd.DataFrame(values).astype(float)
    if frame.empty or frame.shape[1] < 1:
        raise ValidationError(
            f"reality_check: {name} has no series. It is a frame (or a "
            "mapping) of return series, one column each."
        )
    return frame


def _rc_series(values: Any, name: str) -> pd.Series:
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValidationError(
                f"reality_check: {name} has {values.shape[1]} columns and is "
                "ONE series. Several series to compare against are "
                "alternative_returns under question='beats_every_alternative'."
            )
        values = values.iloc[:, 0]
    return pd.Series(values).astype(float)


def _rc_refuse_unread(question: str, **given: Any) -> None:
    unread = [name for name, value in given.items() if value is not None]
    if unread:
        raise ValidationError(
            f"reality_check: {', '.join(unread)} is not read under "
            f"question={question!r}. best_candidate_vs_benchmark reads "
            "candidate_returns and an optional benchmark_returns; "
            "beats_every_alternative reads strategy_returns and "
            "alternative_returns. Pass the inputs of the question you mean."
        )


def _rc_aligned(columns: Dict[str, Any]) -> pd.DataFrame:
    aligned = pd.concat(columns, axis=1).dropna()
    if len(aligned) < 30:
        raise ValidationError(
            f"reality_check: only {len(aligned)} observations remain where "
            "every series has a value, and the test needs at least 30."
        )
    return aligned


def _rc_bootstrap_means(
    differences: np.ndarray,
    n_bootstrap: int,
    block_size: int,
    seed: int,
    *,
    studentise: bool = False,
    circular: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Each bootstrap draw's mean of every column, minus the sample mean --
    the null distribution of each mean outperformance, centred at zero.
    One set of block resamples serves every column, so the columns keep
    their correlation inside a draw.

    With `studentise`, also each draw's own standard error of that mean,
    from the spread of the whole blocks it drew (`_rc_standard_errors` on
    the sample). A studentised statistic divided by ONE standard error
    estimated from the sample ignores that the estimate is itself noisy,
    and with a few dozen blocks the noise is large: dividing every draw by
    the sample's error rejected a no-edge best-of-ten 9% of the time at
    500 days, against 6% when each draw carries its own.

    THE RESAMPLE IS `circular_block_indices`'s (with `circular=False`,
    `block_indices`'s), summed rather than gathered. Each draw takes the
    same block starts from the same generator calls -- ceil(n / block_size)
    blocks, the last cut to n rows -- but adds up precomputed block sums
    instead of gathering n rows per column, which is what makes 20,000
    draws over a few hundred candidates affordable. The tests hold it to
    the row-by-row resample, times `_rc_bootstrap_scale`.

    THE NULL WAS TOO NARROW ON SHORT SERIES, for two reasons, and each fix
    answers one. Measured on true nulls with 20-day blocks and 500 draws,
    2,000 series per cell, at n = 60 / 120 / 250 / 500 against a nominal 5%:

        White, best of three vs a benchmark    11.4  10.2  6.7  6.1  (%)
          circular and scaled                   6.1   6.2  5.7  5.4
        a strategy vs one equal alternative    12.8   9.3  7.7  6.2
          circular and scaled                   7.8   6.6  6.2  6.0

    and under AR(0.3) and GARCH returns at 120 and 250 days, 7.5 / 6.6 and
    10.6 / 6.7 became 5.5 / 4.8 and 6.6 / 5.8 for White's, 9.8 / 5.3 and
    10.2 / 8.1 became 6.6 / 5.5 and 6.6 / 5.6 for the pair. What is left at
    60 days is the noise in a variance from four blocks. The price is a
    little power: a candidate 0.08% a day better, at 250 days, was found
    13.0% of the time against 15.3%, and a strategy that much better than
    its alternative 20.9% against 23.6%, part of which was the excess size.

    CIRCULAR BLOCKS. The null is the draws centred on the sample mean,
    which is exact only when a draw's expected mean IS the sample mean. A
    moving-block draw (starts in [0, n - block_size]) reaches the first and
    last block_size - 1 rows through fewer starts, so its expectation is a
    mean with the edges under-weighted; the circular draw wraps the series
    and reaches every row equally (see `_resampling`).

    THE SCALE. A block bootstrap's variance of a mean is the Bartlett sum of
    the autocovariances taken about the SAMPLE mean, and with no dependence
    at all its expectation is sigma^2 (1 - b/n) / n -- 17% short at n = 120
    and b = 20. The draws are multiplied by sqrt(n / (n - b)), which is
    Bessel's correction at b = 1, and the errors by the same factor, so a
    studentised draw would be unchanged by it.

    HANSEN'S SPA KEEPS THE MOVING DRAW, unscaled (`circular=False`), because
    it was not over-rejecting: 3.8 / 4.3 / 5.4 / 4.9% on the same nulls, and
    4.1 / 3.6 and 5.5 / 5.0 under AR(0.3) and GARCH. Its observed statistic
    is divided by `_rc_standard_errors`, which is biased low in the same
    way, and each draw by a standard error from a handful of blocks, which
    gives the null heavy tails; the two errors offset. The circular draw
    alone took it to 2.1% at 60 days and its power from 11.9% to 10.6%.
    """
    n, k = differences.shape
    centred = differences - differences.mean(axis=0)
    if circular:
        # The series bent into a circle: a block that runs past the last
        # row continues from the first, so every row starts a whole block.
        rows = np.vstack([centred, centred[:block_size]])
        n_starts = n
        scale = _rc_bootstrap_scale(n, block_size)
    else:
        rows = centred
        n_starts = n - block_size + 1
        scale = 1.0
    running = np.vstack([np.zeros((1, k)), np.cumsum(rows, axis=0)])
    # Sum of rows s .. s+block_size-1, and of the cut last block's rows.
    block_sums = running[block_size : block_size + n_starts] - running[:n_starts]
    n_blocks = -(-n // block_size)
    tail = n - (n_blocks - 1) * block_size
    tail_sums = running[tail : tail + n_starts] - running[:n_starts]
    whole = n // block_size

    rng = np.random.default_rng(int(seed))
    draws = np.empty((n_bootstrap, k))
    errors = np.empty((n_bootstrap, k)) if studentise else None
    chunk = max(1, 4_000_000 // (n_blocks * k))
    for low in range(0, n_bootstrap, chunk):
        high = min(low + chunk, n_bootstrap)
        starts = np.stack(
            [rng.integers(0, n_starts, n_blocks) for _ in range(high - low)]
        )
        totals = block_sums[starts[:, :-1]].sum(axis=1) + tail_sums[starts[:, -1]]
        draws[low:high] = scale * totals / n
        if errors is not None:
            means = block_sums[starts[:, :whole]] / block_size
            errors[low:high] = scale * np.sqrt(
                means.var(axis=1, ddof=1) * block_size / n
            )
    return draws, errors


def _rc_bootstrap_scale(n: int, block_size: int) -> float:
    """sqrt(n / (n - block_size)): what a circular block bootstrap's draws
    are widened by. See `_rc_bootstrap_means`."""
    return math.sqrt(n / (n - block_size)) if block_size < n else 1.0


def _rc_standard_errors(differences: np.ndarray, block_size: int) -> np.ndarray:
    """
    The standard error of each column's mean from the variance of its
    overlapping block means, var(block mean) x block_size / n -- the
    estimator the block bootstrap itself implies, so the sample and every
    draw are studentised the same way.

    BIASED LOW, AND LEFT SO ON PURPOSE. Block means taken about their own
    mean understate the variance of the mean by about (1 - b/n): on
    independent noise E[se^2] / (sigma^2 / n) measured 0.728, 0.822, 0.922
    and 0.962 at n = 60, 120, 250 and 500 with the reality check's blocks.
    Hansen's statistic divides by it, and the null's heavy tails offset it
    (see `_rc_bootstrap_means`); multiplied by sqrt(n / (n - b + 1))
    (Meketon and Schmeiser's overlapping batch means) it made the test
    conservative -- 2.6%, 3.3% and 4.6% false rejections at 60, 120 and 250
    days, power at 250 from 11.9% to 9.7%. The other use, Hansen's line
    for "far below the benchmark", is a threshold in these same units.
    """
    n = differences.shape[0]
    running = np.vstack([np.zeros((1, differences.shape[1])), differences.cumsum(0)])
    block_means = (running[block_size:] - running[:-block_size]) / block_size
    return np.sqrt(block_means.var(axis=0, ddof=1) * block_size / n)


def reality_check(
    *positional: Any,
    question: str = "best_candidate_vs_benchmark",
    candidate_returns: Optional[pd.DataFrame] = None,
    benchmark_returns: Optional[pd.Series] = None,
    strategy_returns: Optional[pd.Series] = None,
    alternative_returns: Optional[pd.DataFrame] = None,
    method: str = "white",
    n_bootstrap: int = 1000,
    block_size: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Is a result better than luck after a search -- under the hypothesis the
    caller names in `question`.

    TWO QUESTIONS, AND THEY ARE NOT THE SAME TEST.

    `best_candidate_vs_benchmark` (the default) is White's (2000) Reality
    Check. `candidate_returns` holds every candidate the search looked at,
    the one it kept among them; `benchmark_returns` is the one series they
    must beat (omitted, zero). The statistic is the LARGEST mean
    outperformance over the benchmark, max_k mean(c_k - b), and the block
    bootstrap gives its distribution under the null that no candidate beats
    the benchmark -- so the p-value pays for having picked the best of K.
    `method='hansen_spa'` is Hansen's (2005) test for superior predictive
    ability instead: each candidate's outperformance is divided by its
    bootstrap standard error, and a candidate far below the benchmark is
    not allowed to widen the null. White's form is conservative when the
    search included poor candidates; Hansen's is not.

    `beats_every_alternative` asks whether ONE strategy beats EVERY one of
    `alternative_returns`. It is an intersection-union test (Berger 1982):
    each pair gets a one-sided block-bootstrap test of mean(s - a_k) > 0,
    the statistic is the smallest margin, min_k mean(s - a_k), and the
    p-value is the LARGEST pairwise p-value -- the strategy beats them all
    only if it beats the hardest one, and needs no multiplicity correction
    for that.

    WHAT THIS USED TO TEST. Until the CHANGELOG entry of 2026-10-22 the
    statistic was max_k mean(s - a_k) over the alternatives: the strategy's
    margin over its WEAKEST alternative, bootstrapped as if it were the
    best. Its docstring said "better than the best of the alternatives". A
    strategy earning 5% a year against alternatives at -20% and +30% came
    back significant at p = 0.018 -- it beat the -20% one. Neither question
    above can say that.

    THE BOOTSTRAP IS BLOCKED, and it has to be. Resampling individual days
    destroys the serial correlation that drives drawdowns and volatility
    clustering, which makes the null distribution far too narrow and the
    p-value far too small. Blocks of `block_size` consecutive days preserve
    the local dependence structure. The block length is a real choice: too
    short and dependence is lost, too long and there are too few distinct
    blocks to resample.

    AND, FOR WHITE'S STATISTIC AND THE PAIRWISE ONE, CIRCULAR AND RESCALED,
    because on a short series the plain moving-block null was too narrow:
    10.2% and 9.3% false rejections at 120 days and 5% nominal. Hansen's
    studentised statistic was not, and keeps the moving draw.
    `bootstrap_scheme` and `bootstrap_scale` report which ran;
    `_rc_bootstrap_means` has the measurements.

    THE STATIONARITY ASSUMPTION IS THE WEAK POINT and it is worth naming.
    Block bootstrap assumes the return-generating process is the same
    throughout. A strategy whose edge genuinely existed until 2018 and
    vanished afterwards violates that, and the bootstrap will happily mix
    both regimes into one null.

    Every input is keyword-only: the two questions read different series,
    and a positional call written for the old statistic is refused rather
    than read as one of them.
    """
    if positional:
        raise ValidationError(
            "reality_check takes its series by keyword, because the question "
            "decides what they are: candidate_returns= (every candidate the "
            "search considered) and an optional benchmark_returns= for "
            "question='best_candidate_vs_benchmark', the default; or "
            "strategy_returns= and alternative_returns= for "
            "question='beats_every_alternative'. The positional "
            "(strategy, alternatives) form tested the strategy against its "
            "WEAKEST alternative and was withdrawn -- see the CHANGELOG entry "
            "of 2026-10-22."
        )
    if question not in REALITY_CHECK_QUESTIONS:
        raise ValidationError(
            f"reality_check: question={question!r} is not one of "
            f"{list(REALITY_CHECK_QUESTIONS)}."
        )
    if method not in REALITY_CHECK_METHODS:
        raise ValidationError(
            f"reality_check: method={method!r} is not one of "
            f"{list(REALITY_CHECK_METHODS)}."
        )
    n_bootstrap = max(100, int(n_bootstrap))

    if question == "beats_every_alternative":
        _rc_refuse_unread(
            question,
            candidate_returns=candidate_returns,
            benchmark_returns=benchmark_returns,
        )
        if method != "white":
            raise ValidationError(
                f"reality_check: method={method!r} applies to "
                "question='best_candidate_vs_benchmark'. Under "
                "beats_every_alternative each pair is tested on its own, "
                "with no maximum over candidates for Hansen's correction to "
                "act on; leave method at 'white'."
            )
        if strategy_returns is None or alternative_returns is None:
            raise ValidationError(
                "reality_check: question='beats_every_alternative' needs "
                "strategy_returns (the one strategy) and alternative_returns "
                "(every alternative it must beat)."
            )
        return _beats_every_alternative(
            _clean_returns(strategy_returns, "reality_check", minimum=30),
            _rc_frame(alternative_returns, "alternative_returns"),
            n_bootstrap,
            block_size,
            seed,
        )

    _rc_refuse_unread(
        question,
        strategy_returns=strategy_returns,
        alternative_returns=alternative_returns,
    )
    if candidate_returns is None:
        raise ValidationError(
            "reality_check: question='best_candidate_vs_benchmark' needs "
            "candidate_returns -- every candidate the search considered, the "
            "one it kept among them. The comparison is against "
            "benchmark_returns, or zero when that is omitted."
        )
    return _best_candidate_vs_benchmark(
        _rc_frame(candidate_returns, "candidate_returns"),
        (
            None
            if benchmark_returns is None
            else _rc_series(benchmark_returns, "benchmark_returns")
        ),
        method,
        n_bootstrap,
        block_size,
        seed,
    )


def _rc_common_warnings(block_size: int, n: int, circular: bool = True) -> List[str]:
    blocked = (
        f"Block bootstrap with blocks of {block_size} days, which preserves "
        "serial correlation. Resampling individual days would make the null "
        "too narrow and the p-value too small -- volatility clustering and "
        "drawdowns both live in the dependence structure."
    )
    if circular:
        blocked += (
            " The blocks wrap around the sample so every day is drawn equally "
            f"often, and the null is widened by "
            f"{_rc_bootstrap_scale(n, block_size):.3f} (sqrt(n / (n - block))) "
            "because a block bootstrap understates the variance of a mean by "
            "that much."
        )
    return [
        blocked,
        "Assumes the return process is STATIONARY across the sample. A "
        "strategy whose edge genuinely existed and then vanished violates "
        "that, and the bootstrap mixes both regimes into one null.",
    ]


def _best_candidate_vs_benchmark(
    candidates: pd.DataFrame,
    benchmark: Optional[pd.Series],
    method: str,
    n_bootstrap: int,
    block_size: int,
    seed: int,
) -> Dict[str, Any]:
    """White's Reality Check, or Hansen's SPA test, of the best candidate
    against one benchmark."""
    columns = {f"c{j}": candidates.iloc[:, j] for j in range(candidates.shape[1])}
    if benchmark is not None:
        columns["__benchmark__"] = benchmark
    aligned = _rc_aligned(columns)
    names = [str(name) for name in candidates.columns]
    values = aligned[[f"c{j}" for j in range(len(names))]].to_numpy()
    if benchmark is not None:
        values = values - aligned["__benchmark__"].to_numpy()[:, None]
    n, k = values.shape
    block_size = max(1, min(int(block_size), n // 4))

    means = values.mean(axis=0)
    # White's null is the draws themselves and takes the circular, scaled
    # draw; Hansen's divides each draw by its own error and keeps the moving
    # one, which was already calibrated (see `_rc_bootstrap_means`).
    circular = method == "white"
    draws, draw_errors = _rc_bootstrap_means(
        values,
        n_bootstrap,
        block_size,
        seed,
        studentise=method == "hansen_spa",
        circular=circular,
    )
    errors = _rc_standard_errors(values, block_size)
    distinct = errors > 0
    # Hansen's line between a candidate that may be on the null's boundary
    # and one so far below the benchmark it cannot be: sqrt(2 ln ln n)
    # standard errors.
    far = -errors * math.sqrt(2.0 * math.log(math.log(n)))
    warnings: List[str] = []

    if method == "white":
        best = int(np.argmax(means))
        statistic = float(means[best])
        null = draws.max(axis=1)
        p_value = float((null >= statistic).mean())
        statistic_units = TRADING_DAYS
        far_below = int(np.sum(distinct & (means < far)))
        if far_below:
            warnings.append(
                f"{far_below} of {k} candidates are far below the benchmark. "
                "In White's form they still widen the null and make the test "
                "conservative; method='hansen_spa' does not let them."
            )
    else:
        assert draw_errors is not None
        if not distinct.any():
            raise ValidationError(
                "reality_check: every candidate's outperformance has zero "
                "variance -- each one IS the benchmark -- so Hansen's "
                "studentised statistic is 0/0. Use method='white'."
            )
        # Hansen (2005): studentise, and recentre the null only for the
        # candidates that are not far below the benchmark (the consistent
        # p-value, SPA_c). A candidate with no variance is the benchmark
        # itself and contributes zero.
        t_values = np.divide(means, errors, out=np.zeros_like(means), where=distinct)
        best = int(np.argmax(np.where(distinct, t_values, -np.inf)))
        statistic = max(float(t_values[best]), 0.0)
        recentre = np.where(means >= far, means, 0.0)
        shifted = draws + means - recentre
        studentised = np.divide(
            shifted,
            draw_errors,
            out=np.zeros_like(shifted),
            where=draw_errors > 0,
        )
        null = np.maximum(studentised.max(axis=1), 0.0)
        p_value = float((null >= statistic).mean())
        statistic_units = 1.0

    if p_value > 0.10:
        warnings.append(
            f"p = {p_value:.3f}. The best of {k} candidate(s) outperforms the "
            "benchmark by no more than the best of that many would under the "
            "null that none of them beats it."
        )
    if benchmark is None:
        warnings.append(
            "No benchmark_returns: outperformance is measured against zero, "
            "so this asks whether the best candidate's mean return is above "
            "zero after the search."
        )
    if k == 1:
        warnings.append(
            "One candidate: this is a block-bootstrap test of its "
            "outperformance alone. The correction for a search needs every "
            "candidate the search looked at in candidate_returns."
        )
    warnings += _rc_common_warnings(block_size, n, circular=circular)

    return {
        "question": "best_candidate_vs_benchmark",
        "method": method,
        "n_observations": int(n),
        "n_candidates": int(k),
        "benchmark_given": benchmark is not None,
        "best_candidate": names[best],
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
        # What drew the null and how far it was widened, so a p-value can
        # be reproduced and told apart from one drawn the old way.
        "bootstrap_scheme": "circular_block" if circular else "moving_block",
        "bootstrap_scale": _rc_bootstrap_scale(n, block_size) if circular else 1.0,
        "observed_outperformance": float(means[best] * TRADING_DAYS),
        "observed_statistic": float(statistic * statistic_units),
        "p_value": p_value,
        "significant_at_05": bool(p_value < 0.05),
        "bootstrap_p95": float(np.percentile(null, 95) * statistic_units),
        "warnings": warnings,
    }


def _beats_every_alternative(
    strategy: pd.Series,
    alternatives: pd.DataFrame,
    n_bootstrap: int,
    block_size: int,
    seed: int,
) -> Dict[str, Any]:
    """The intersection-union test that one strategy beats every
    alternative."""
    columns: Dict[str, Any] = {"__strategy__": strategy}
    columns.update(
        {f"a{j}": alternatives.iloc[:, j] for j in range(alternatives.shape[1])}
    )
    aligned = _rc_aligned(columns)
    names = [str(name) for name in alternatives.columns]
    differences = (
        aligned["__strategy__"].to_numpy()[:, None]
        - aligned[[f"a{j}" for j in range(len(names))]].to_numpy()
    )
    n, k = differences.shape
    block_size = max(1, min(int(block_size), n // 4))

    margins = differences.mean(axis=0)
    draws, _ = _rc_bootstrap_means(differences, n_bootstrap, block_size, seed)
    # One-sided, per pair: how often a null centred at zero reaches the
    # observed margin.
    p_values = (draws >= margins).mean(axis=0)
    binding = int(np.argmax(p_values))
    p_value = float(p_values[binding])
    tightest = int(np.argmin(margins))

    warnings: List[str] = []
    if p_value > 0.10:
        warnings.append(
            f"p = {p_value:.3f}. The strategy does not beat {names[binding]!r} "
            f"(mean margin {margins[binding] * TRADING_DAYS:+.2%} a year) by "
            "more than luck would produce, so it is not shown to beat every "
            "alternative."
        )
    warnings += _rc_common_warnings(block_size, n)

    return {
        "question": "beats_every_alternative",
        "method": "white",
        "n_observations": int(n),
        "n_alternatives": int(k),
        "binding_alternative": names[binding],
        "per_alternative": {
            name: {
                "outperformance": float(margins[j] * TRADING_DAYS),
                "p_value": float(p_values[j]),
            }
            for j, name in enumerate(names)
        },
        "n_bootstrap": n_bootstrap,
        "block_size": block_size,
        # What drew the null and how far it was widened, so a p-value can
        # be reproduced and told apart from one drawn the old way.
        "bootstrap_scheme": "circular_block",
        "bootstrap_scale": _rc_bootstrap_scale(n, block_size),
        "observed_outperformance": float(margins[tightest] * TRADING_DAYS),
        "observed_statistic": float(margins[tightest] * TRADING_DAYS),
        "p_value": p_value,
        "significant_at_05": bool(p_value < 0.05),
        "bootstrap_p95": float(np.percentile(draws[:, binding], 95) * TRADING_DAYS),
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
