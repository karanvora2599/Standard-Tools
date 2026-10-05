"""
Is model B actually better than model A, or did it draw a kinder sample?

`compare_models` ranks registered models by a headline out-of-sample
metric. That answers which number is larger and nothing about whether the
difference is larger than the noise in one OOS sample -- and on a few
hundred dates of daily IC, a difference of 0.006 is routinely inside that
noise. Sorting on it selects the model that got the friendlier draw, which
is the quiet way to turn an out-of-sample sample into a tuning set.

THE COMPARISON IS PAIRED, ON THE INTERSECTION. Both models' predictions are
joined on (date, entity) and the same realized outcome, so each date
contributes one IC for A and one for B measured on identical rows. The
object of interest is the per-date DIFFERENCE series: its mean is the
improvement, and its dispersion -- which is far smaller than either
model's own, because the two share every good and bad day -- is what an
interval has to be built on. Comparing two unpaired intervals would find
nothing significant about two models that differ on every single day.

THE INTERVAL IS BLOCK-BOOTSTRAPPED. Daily ICs under an overlapping label
are serially correlated, and an IID resample destroys exactly that,
narrowing the interval by a factor the repository has already measured
(`analysis.inference.bootstrap_statistic`: 2.24x on AR(1) at phi 0.8).
Blocks of n^(1/3) consecutive dates are resampled with replacement, the
rule that function uses, and its block indexer is reused rather than
rewritten.

THE LOSS TEST IS DIEBOLD-MARIANO, where a loss exists. For a regression
the per-date mean squared error differential's mean is tested the way a
run's headline is: a long-run variance from the series' lowest cosine
frequencies, their number set by `headline_degrees_of_freedom` for the
dates and the label's horizon, and Student's t with that many degrees of
freedom; for a classifier the Brier differential; for a ranker there is no
loss with units, so there is no DM and the IC difference carries the
comparison alone.

MANY CANDIDATES AGAINST ONE REFERENCE need a multiple-testing correction,
and the p-values come back Holm-adjusted. What Holm does not do is control
for the candidates having been SELECTED on this same sample -- that is what
SPA-style tests exist for, and the report says so rather than pretending
the adjustment covers it.

THREE ADJUSTMENTS, TWO QUANTITIES. `holm_adjust` and `bonferroni_adjust`
control the family-wise error rate; `bh_adjust` controls the false
discovery rate, which is a weaker claim about each rejection and a
different one, so the method used travels with the numbers rather than
being inferred from them. See the CHANGELOG entry of 2026-09-21 for why
the second and third were added: the first was reachable only through a
comparison of two registered models, and a family of p-values from
anywhere else had no correction at all.

THE SAME BITS AT ANY THREAD COUNT. The long-run variances and the test
of a mean are dot products over the dates, and OpenBLAS splits a dot
product of more than 10,000 terms across its threads, so above 10,000
dates their last bits followed the caller's BLAS thread count -- and with
them a run's headline, `score_predictions`' headline, Diebold-Mariano and
`compare_signals`. `newey_west_variance`, `cosine_variance` and
`mean_vs_null_test` run their products under `single_threaded_blas()`.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.analysis.inference import _block_indices
from standard_quant_tools.error import ValidationError

from .metrics import cross_sectional_ic

#: The per-date correlations a comparison can be built on.
COMPARISON_METRICS = ("cs_rank_ic", "cs_ic")

#: The columns a prediction frame must carry to be compared.
REQUIRED_COLUMNS = ("date", "entity", "prediction", "target")


def _moving_block_means(
    values: np.ndarray, n_bootstrap: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap means of `values` under a moving-block resample."""
    n = values.size
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        draws[i] = float(values[_block_indices(n, block_size, rng)].mean())
    return draws


def compare_ic_series(
    ic_a: pd.Series,
    ic_b: pd.Series,
    *,
    n_bootstrap: int = 2000,
    block_size: Optional[int] = None,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    The statistics of the per-date difference `ic_b - ic_a`.

    The two series are aligned on their common dates -- which, when they
    came from `paired_comparison`, is every date, because both were
    computed on the same joined rows. Returns the mean difference, a
    block-bootstrap percentile interval on it, a two-sided bootstrap
    p-value (the share of resampled means on the far side of zero, doubled),
    and the fraction of dates on which B beat A.
    """
    if not 0 < confidence < 1:
        raise ValidationError(f"confidence must be in (0, 1), got {confidence!r}")
    joined = pd.concat(
        [ic_a.rename("a"), ic_b.rename("b")], axis=1, join="inner"
    ).dropna()
    n = int(len(joined))
    if n < 10:
        raise ValidationError(
            f"compare_ic_series: only {n} date(s) carry an IC for both models; "
            "a difference measured on fewer than ten dates has no interval "
            "worth reporting."
        )
    diff = (joined["b"] - joined["a"]).to_numpy(dtype=np.float64)
    if block_size is None:
        block_size = max(1, int(round(n ** (1.0 / 3.0))))
    block_size = max(1, min(int(block_size), n // 2))
    n_bootstrap = max(100, int(n_bootstrap))

    observed = float(diff.mean())
    wins = int(np.sum(diff > 0.0))
    losses = int(np.sum(diff < 0.0))
    rng = np.random.default_rng(int(seed))
    draws = _moving_block_means(diff, n_bootstrap, block_size, rng)
    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(draws, alpha * 100))
    upper = float(np.percentile(draws, (1 - alpha) * 100))
    # Percentile-bootstrap p-value: how often the resampled mean lands on
    # the far side of zero from the observed one, doubled for two sides
    # and floored at one resample so it is never reported as exactly zero.
    far_side = float(np.mean(draws <= 0.0) if observed > 0 else np.mean(draws >= 0.0))
    p_value = float(min(1.0, 2.0 * max(far_side, 1.0 / n_bootstrap)))

    return {
        "n_dates": n,
        "mean_a": float(joined["a"].mean()),
        "mean_b": float(joined["b"].mean()),
        "mean_difference": observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": float(confidence),
        "p_value": p_value,
        # Decided dates only: two identical models tie on every date, and
        # a share of ALL dates read that as 'b lost every day'.
        "hit_rate": (float(wins / (wins + losses)) if wins + losses else float("nan")),
        "n_b_better": wins,
        "n_a_better": losses,
        "n_ties": int(n - wins - losses),
        "block_size": int(block_size),
        "n_bootstrap": int(n_bootstrap),
        "verdict": _verdict(observed, lower, upper),
    }


def _verdict(mean: float, lower: float, upper: float) -> str:
    if lower <= 0.0 <= upper:
        return "indistinguishable"
    return "b_better" if mean > 0.0 else "a_better"


def newey_west_variance(values: np.ndarray, lag: int) -> float:
    """
    Long-run variance of the MEAN of `values`, Bartlett kernel to `lag`.

    gamma_0 + 2 * sum_{k=1..lag} (1 - k/(lag+1)) * gamma_k, divided by n.
    `lag=0` is the ordinary variance of the mean. It is what a test reads
    when its caller names a lag; with none named the tests here read
    `cosine_variance` instead (see `headline_degrees_of_freedom` for why).
    For a series of dates whose labels look h bars forward, these weights
    shrink the autocorrelations they keep: cut at h - 1 they recover 68%
    of a 5-bar overlap's long-run variance, at 2h 85%.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    if n < 2:
        return float("nan")
    centered = x - x.mean()
    with single_threaded_blas():
        variance = float(np.dot(centered, centered) / n)
        for k in range(1, min(int(lag), n - 1) + 1):
            weight = 1.0 - k / (lag + 1.0)
            autocov = float(np.dot(centered[k:], centered[:-k]) / n)
            variance += 2.0 * weight * autocov
    return variance / n


def cosine_variance(values: np.ndarray, n_frequencies: int) -> float:
    """
    Long-run variance of the MEAN of `values` from its `n_frequencies`
    lowest cosine frequencies, each weighted equally.

    The series is projected on the orthonormal cosines
    sqrt(2/n) * cos(pi * j * (t + 1/2) / n), j = 1..nu, each of which sums
    to zero over the dates, so the mean does not enter them. The average
    of the nu squared projections estimates the spectral density at zero
    frequency, which is n times the variance of the mean; it is a sum of
    squares, so it is never negative. Under a normal series whose spectrum
    is flat over those frequencies, mean / sqrt(this) is distributed as
    Student's t with nu degrees of freedom exactly: the reference
    distribution `mean_vs_null_test` and `diebold_mariano` read the
    p-value from. NaN with fewer than two values or no frequency.
    """
    x = np.asarray(values, dtype=np.float64)
    n = int(x.size)
    count = int(n_frequencies)
    if n < 2 or count < 1:
        return float("nan")
    count = min(count, n - 1)
    centered = x - x.mean()
    # cos(pi * j * (2t + 1) / (2n)) depends on j * (2t + 1) only modulo 4n,
    # so one table of 4n values serves every frequency, each angle read
    # from its reduced integer rather than from a large float product.
    period = 4 * n
    table = np.cos(np.arange(period, dtype=np.float64) * (math.pi / (2.0 * n)))
    odd = 2 * np.arange(n, dtype=np.int64) + 1
    total = 0.0
    with single_threaded_blas():
        for j in range(1, count + 1):
            projection = float(np.dot(table[(j * odd) % period], centered))
            total += projection * projection
    return (2.0 / n) * total / count / n


#: Fewer finite dates than this and a headline is not tested against its
#: null: a long-run variance read off a handful of dates is not an
#: estimate of anything. The same floor `compare_ic_series` refuses at.
MIN_HEADLINE_DATES = 10

#: The cosine frequencies a test of a mean over n dates averages, before
#: the label's horizon caps them: floor(0.4 * n ** (2/3)), the rule
#: Lazarus, Lewis, Stock and Watson (2018) give for this estimator.
COSINE_FREQUENCY_SCALE = 0.4

#: An h-bar label gets at most n / (3h) frequencies, so the highest one
#: read has a period of at least 6h dates, where the spectrum of an h-bar
#: overlap is still within 9% of its value at zero frequency.
HORIZONS_PER_FREQUENCY = 3


def andrews_lag(n_dates: int) -> int:
    """The rule-of-thumb Bartlett bandwidth for `n_dates` observations,
    floor(4 * (n / 100) ** (2/9)): 5 at 504 dates, 4 at 100."""
    if n_dates < 1:
        return 0
    return int(math.floor(4.0 * (float(n_dates) / 100.0) ** (2.0 / 9.0)))


def headline_lag(n_dates: int, horizon: Optional[int]) -> int:
    """
    The Newey-West lag a run's headline was tested at before the CHANGELOG
    entry of 2026-10-04 that replaced it: max(2h, the Andrews bandwidth),
    capped at `n_dates - 1`, with h the target horizon; without a horizon,
    the Andrews bandwidth alone.

    No test here reads it unless asked: a lag named to `mean_vs_null_test`,
    `diebold_mariano` or `compare_signals(hac_lag=...)` is used as named,
    and `mean_vs_null_test(values, lag=headline_lag(n, h))` returns the
    t and p a run reported before, to the bit. See
    `headline_degrees_of_freedom` for the rule that replaced it and why.
    """
    n = int(n_dates)
    if n < 2:
        return 0
    lag = andrews_lag(n)
    if horizon is not None and int(horizon) > 0:
        lag = max(lag, 2 * int(horizon))
    return int(min(lag, n - 1))


def headline_degrees_of_freedom(n_dates: int, horizon: Optional[int]) -> int:
    """
    How many cosine frequencies a test of a mean over `n_dates` dates
    averages, which is also the degrees of freedom of the Student t it is
    read against: min(floor(0.4 * n^(2/3)), floor(n / (3h))), at least 1
    and at most n - 1, with h the label's horizon (1 without one). 25 at
    504 dates for a 1- or 5-bar label, 16 for a 10-bar, 8 for a 20-bar.

    ONE RULE FOR EVERY MEAN OVER DATES. The run's headline
    (`modeling.engine`), `score_predictions`' headline, the Diebold-
    Mariano test of `paired_comparison` (`compare_models(method=
    'paired')`, `compare_signals(mode='paired')`) and `compare_signals(
    mode='ic_series')`'s long-run variance all read it. A Newey-West lag the
    caller names is used as named, at the normal critical value, as before.

    WHY NOT A NEWEY-WEST LAG. Daily values built on an h-bar label share
    up to h - 1 bars, so their autocorrelation falls roughly linearly to
    zero at lag h. Bartlett weights shrink those autocorrelations a second
    time, and a variance estimated from the same dates it is applied to is
    biased low and noisy besides; read against the normal, the t is too
    large. The previous rule, Bartlett at max(2h, the Andrews bandwidth),
    rejected a true zero mean, at a nominal 5%, 7.2% of the time for a
    5-bar label over 504 dates (10.9% for a 20-bar label) on simulated
    per-date rank ICs of a persistent feature against a 30-name cross-
    section, and 21.9% for a 20-bar label over 126 dates. The best
    Newey-West variant tried, lag 4h with a fixed-b critical value, still
    rejected 6.1% to 6.9% for a 20-bar label over 252 to 2,000 dates.

    This rule rejects 4.6% to 5.9% of the time on every one of the twenty
    cells of horizon 1, 5, 10, 20 by 126, 252, 504, 1,000, 2,000 dates
    (5,000 draws each), and 4.8% to 5.6% on simulated Diebold-Mariano loss
    differentials (20,000 each). The cost is power where the dates are few
    for the horizon: against a mean of 2.5 true standard errors it rejects
    67% of the time at 504 dates and a 5-bar label, where a test that knew
    the variance would reject 70%, and 34% at 126 dates and a 20-bar label,
    where only six non-overlapping labels exist (see the CHANGELOG entry of
    2026-10-04).
    """
    n = int(n_dates)
    if n < 2:
        return 0
    h = int(horizon) if horizon is not None and int(horizon) > 0 else 1
    nu = int(math.floor(COSINE_FREQUENCY_SCALE * float(n) ** (2.0 / 3.0)))
    nu = min(nu, n // (HORIZONS_PER_FREQUENCY * h))
    return int(max(1, min(nu, n - 1)))


def _student_t_p_value(statistic: float, degrees_of_freedom: int) -> float:
    """Two-sided p-value of `statistic` under Student's t."""
    from scipy.special import stdtr

    return float(2.0 * stdtr(float(degrees_of_freedom), -abs(statistic)))


def mean_vs_null_test(
    values: Any,
    *,
    null: float = 0.0,
    lag: Optional[int] = None,
    horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Whether the mean of a serially correlated series differs from `null`.

    Unless a `lag` is named, the t statistic divides the mean's distance
    from `null` by the square root of `cosine_variance` at
    `headline_degrees_of_freedom(n, horizon)` frequencies, and the p-value
    is two-sided under Student's t with that many degrees of freedom
    (`degrees_of_freedom`; `lag` is then None). A named `lag` is used as
    named: a Newey-West variance at that lag and a two-sided normal
    p-value, which at `headline_lag(n, horizon)` is the test a run's
    headline made before the CHANGELOG entry of 2026-10-04, to the bit.
    `t_stat_uncorrected` is the same distance over the ordinary standard
    error -- the series read as independent -- reported beside it so a
    reader can see how much the correction moved it, and
    `autocorrelation_lag1` is the plainest sign of why. Non-finite values
    are dropped first. With fewer than two values, or no variance, the
    statistics are NaN rather than a number that means nothing.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    nan = float("nan")
    degrees: Optional[int] = None
    if lag is None:
        degrees = headline_degrees_of_freedom(n, horizon) if n >= 2 else None
    out: Dict[str, Any] = {
        "n": n,
        "mean": float(x.mean()) if n else nan,
        "null": float(null),
        "lag": None if lag is None else int(lag),
        "degrees_of_freedom": degrees,
        "t_stat": nan,
        "t_stat_uncorrected": nan,
        "p_value": nan,
        "autocorrelation_lag1": nan,
    }
    if n < 2:
        return out
    # Its dot products on one BLAS thread, with the variances' own limits
    # nested inside this one: OpenBLAS splits a dot product of more than
    # 10,000 terms across threads, and the last bits then follow the count.
    with single_threaded_blas():
        centered = x - x.mean()
        denominator = float(np.dot(centered, centered))
        if denominator > 0.0:
            out["autocorrelation_lag1"] = float(
                np.dot(centered[1:], centered[:-1]) / denominator
            )
        if float(np.ptp(x)) == 0.0:
            # Every value the same. Their mean is rounded, so the deviations
            # from it are a few ulps rather than zero, and a variance of
            # those made a t of 1e31 out of a constant.
            return out
        distance = float(x.mean()) - float(null)
        if lag is None:
            variance = cosine_variance(x, int(degrees or 0))
        else:
            variance = newey_west_variance(x, int(lag))
        if math.isfinite(variance) and variance > 0.0:
            t_stat = distance / math.sqrt(variance)
            out["t_stat"] = float(t_stat)
            if lag is None:
                out["p_value"] = _student_t_p_value(t_stat, int(degrees or 0))
            else:
                out["p_value"] = float(math.erfc(abs(t_stat) / math.sqrt(2.0)))
        plain = newey_west_variance(x, 0)
        if math.isfinite(plain) and plain > 0.0:
            out["t_stat_uncorrected"] = float(distance / math.sqrt(plain))
    return out


def diebold_mariano(
    loss_a: pd.Series,
    loss_b: pd.Series,
    *,
    lag: Optional[int] = None,
    horizon: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Diebold-Mariano on the per-date loss differential `loss_a - loss_b`.

    A POSITIVE statistic means B's loss is smaller. Unless a lag is named,
    the variance is `cosine_variance` at `headline_degrees_of_freedom` for
    the dates and `horizon`, and the p-value is two-sided under Student's t
    with that many degrees of freedom (`degrees_of_freedom`; `lag` is then
    None) -- the test a run's headline makes. NaN when the differential
    has no variance, which two identical models produce and which is not
    evidence of anything.

    A NAMED LAG is used as named, as before: a Newey-West variance at that
    lag, the Harvey-Leybourne-Newbold correction at the forecast horizon
    (`horizon` when given, else `lag + 1`), and a normal p-value.
    `lag=h - 1` is the statistic this function returned first and
    `lag=headline_lag(n, h), horizon=h` the one it returned until the
    CHANGELOG entry of 2026-10-04, each to the bit. On 20,000 simulated
    differentials of a 5-bar overlap over 504 dates the second rejected a
    true zero 7.6% of the time at a nominal 5% (20-bar: 9.9%), and the
    first more (11.3% on 4,000); the default now rejects 5.2% (20-bar:
    5.6%).
    """
    from scipy.stats import norm

    joined = pd.concat(
        [loss_a.rename("a"), loss_b.rename("b")], axis=1, join="inner"
    ).dropna()
    n = int(len(joined))
    if n < 10:
        raise ValidationError(
            f"diebold_mariano: only {n} date(s) carry a loss for both models."
        )
    differential = (joined["a"] - joined["b"]).to_numpy(dtype=np.float64)
    mean = float(differential.mean())
    if lag is None:
        degrees = headline_degrees_of_freedom(n, horizon)
        variance = cosine_variance(differential, degrees)
        result: Dict[str, Any] = {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "mean_differential": mean,
            "lag": None,
            "degrees_of_freedom": int(degrees),
            "n_dates": n,
        }
        if math.isfinite(variance) and variance > 0.0:
            statistic = float(mean / math.sqrt(variance))
            result["statistic"] = statistic
            result["p_value"] = _student_t_p_value(statistic, degrees)
        return result

    if horizon is not None and int(horizon) > 0:
        h = int(horizon)
    else:
        h = int(lag) + 1
    variance = newey_west_variance(differential, int(lag))
    if not math.isfinite(variance) or variance <= 0.0:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "mean_differential": mean,
            "lag": int(lag),
            "degrees_of_freedom": None,
            "n_dates": n,
        }
    correction = math.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    statistic = float(mean / math.sqrt(variance) * correction)
    return {
        "statistic": statistic,
        "p_value": float(2.0 * norm.sf(abs(statistic))),
        "mean_differential": mean,
        "lag": int(lag),
        "degrees_of_freedom": None,
        "n_dates": n,
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjustment, monotone and capped at one."""
    p = np.asarray(list(p_values), dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * p[index])
        running = max(running, candidate)
        adjusted[index] = running
    return [float(v) for v in adjusted]


def bonferroni_adjust(p_values: Sequence[float]) -> List[float]:
    """
    Bonferroni: every p-value multiplied by the number of tests, capped at one.

    The most conservative of the three and the only one that needs no
    ordering, which is also what makes it the weakest: it spends the whole
    error budget on the possibility that every null is true at once, so on
    a family where several effects are real it rejects fewer of them than
    Holm does while controlling exactly the same quantity. Holm dominates
    it uniformly -- there is no configuration where Bonferroni rejects
    something Holm does not -- and it is here because it is the number a
    reader recognises and checks the other two against by hand.
    """
    p = np.asarray(list(p_values), dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    return [float(min(1.0, m * value)) for value in p]


def bh_adjust(p_values: Sequence[float]) -> List[float]:
    """
    Benjamini-Hochberg step-up adjustment, monotone and capped at one.

    A DIFFERENT QUANTITY FROM THE OTHER TWO. Holm and Bonferroni control
    the family-wise error rate, the probability of ONE false rejection
    anywhere in the family. This controls the false discovery rate, the
    expected SHARE of the rejections that are false. On twenty tests at
    0.05 the first promises that a false rejection is unlikely at all; the
    second allows one of twenty rejections to be false on average, and is
    therefore far more willing to reject. A rejection under it is a
    weaker claim, and nothing that reports it should imply otherwise.

    THE STEP IS UP, AND THE MONOTONE PASS IS WHAT MAKES THE ANSWER A
    P-VALUE. Ordered smallest to largest, the rank-`k` p-value is scaled by
    `m / k`: the critical constants ASCEND with the rank, where Holm's
    step-down multipliers descend. Those scaled values are not monotone on
    their own -- 3/2 * 0.03 exceeds 3/3 * 0.04 -- so the procedure sweeps
    from the largest rank down keeping a running minimum, exactly as
    `holm_adjust` sweeps up keeping a running maximum. That pass is not
    cosmetic: it is what makes `bh_adjust(p)[i] <= alpha` identical to the
    step-up procedure's own decision (reject every test up to the largest
    rank whose `p_(k) <= k / m * alpha`). Without it a test whose own
    scaled value missed alpha would still be rejected when a larger
    p-value cleared it, and a caller comparing the returned number to
    alpha would get a different answer from the procedure. These are the
    values `statsmodels.stats.multitest.multipletests(method='fdr_bh')`
    and R's `p.adjust(method="BH")` return.
    """
    p = np.asarray(list(p_values), dtype=np.float64)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m, dtype=np.float64)
    running = 1.0
    for rank in range(m, 0, -1):
        index = order[rank - 1]
        running = min(running, 1.0, m / rank * p[index])
        adjusted[index] = running
    return [float(value) for value in adjusted]


def _check_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValidationError(
            f"paired_comparison: {label} is missing column(s) {missing}; a "
            f"comparable frame carries {list(REQUIRED_COLUMNS)}."
        )
    out = frame[list(REQUIRED_COLUMNS)].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["entity"] = out["entity"].astype(str)
    return out


def _per_date_loss(joined: pd.DataFrame, side: str, task: str) -> Optional[pd.Series]:
    """Per-date mean loss for one model, or None for a task with no loss
    that has units."""
    if task == "regression":
        loss = (joined["target"] - joined[f"prediction_{side}"]) ** 2
    elif task == "classification":
        loss = (joined["target"] - joined[f"prediction_{side}"]) ** 2  # Brier
    else:
        return None
    return loss.groupby(joined["date"]).mean()


def paired_comparison(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    task: str,
    metric: str = "cs_rank_ic",
    horizon: Optional[int] = None,
    n_bootstrap: int = 2000,
    block_size: Optional[int] = None,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Compare two models' predictions on the rows both predicted.

    Each frame carries (date, entity, prediction, target). They are joined
    on (date, entity); the realized outcome must agree on the intersection,
    which refuses the comparison of two models fitted against different
    labels under the same name. The per-date `metric` is computed for each
    on the joined rows, the difference series is bootstrapped, and where
    the task has a loss with units the Diebold-Mariano test is reported
    beside it, at `headline_degrees_of_freedom` of the shared dates and
    `horizon`.
    """
    if metric not in COMPARISON_METRICS:
        raise ValidationError(
            f"paired_comparison: metric={metric!r}; expected one of "
            f"{list(COMPARISON_METRICS)}."
        )
    a = _check_frame(frame_a, "frame_a")
    b = _check_frame(frame_b, "frame_b")
    joined = a.merge(b, on=["date", "entity"], suffixes=("_a", "_b"), how="inner")
    if joined.empty:
        raise ValidationError(
            "paired_comparison: the two models share no (date, entity) row. "
            "Models validated over different windows cannot be compared "
            "paired; check their test spans."
        )
    disagreement = (joined["target_a"] - joined["target_b"]).abs()
    if not np.allclose(
        joined["target_a"], joined["target_b"], atol=1e-12, equal_nan=True
    ):
        raise ValidationError(
            "paired_comparison: the realized outcomes disagree on "
            f"{int((disagreement > 1e-12).sum())} shared row(s). The two models "
            "were fitted against different labels, and a comparison on rows "
            "whose 'truth' differs would be arithmetic on two questions."
        )
    joined = joined.rename(columns={"target_a": "target"}).drop(columns=["target_b"])
    joined = joined.dropna(subset=["prediction_a", "prediction_b", "target"])

    method = "spearman" if metric == "cs_rank_ic" else "pearson"
    dates = joined["date"].to_numpy()
    truth = joined["target"].to_numpy(dtype=np.float64)
    ic_a = cross_sectional_ic(
        truth, joined["prediction_a"].to_numpy(dtype=np.float64), dates, method
    )
    ic_b = cross_sectional_ic(
        truth, joined["prediction_b"].to_numpy(dtype=np.float64), dates, method
    )
    difference = compare_ic_series(
        ic_a,
        ic_b,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        confidence=confidence,
        seed=seed,
    )

    dm: Optional[Dict[str, Any]] = None
    loss_a = _per_date_loss(joined, "a", task)
    loss_b = _per_date_loss(joined, "b", task)
    if loss_a is not None and loss_b is not None:
        # The run headline's test, at `headline_degrees_of_freedom` for the
        # shared dates and the horizon. It was a Newey-West variance at
        # horizon - 1, then at `headline_lag`, both read against the normal,
        # and both too confident on an overlapping label: see
        # `headline_degrees_of_freedom`.
        dm = diebold_mariano(loss_a, loss_b, horizon=horizon)
        dm["loss"] = "squared_error" if task == "regression" else "brier"

    warnings: List[str] = []
    if difference["n_dates"] < 60:
        warnings.append(
            f"NOTE: the comparison rests on {difference['n_dates']} dates. An "
            "interval this short is wide by construction, and 'indistinguishable' "
            "on it means 'not enough dates to tell', not 'the same'."
        )
    if difference["verdict"] == "indistinguishable":
        warnings.append(
            f"The {confidence:.0%} interval on the IC difference "
            f"[{difference['ci_lower']:+.4f}, {difference['ci_upper']:+.4f}] "
            "contains zero: this sample does not separate the two models, "
            "whichever headline number is larger."
        )
    return {
        "metric": metric,
        "task": task,
        "n_rows": int(len(joined)),
        "n_entities": int(joined["entity"].nunique()),
        **difference,
        "diebold_mariano": dm,
        "warnings": warnings,
    }


__all__ = [
    "COMPARISON_METRICS",
    "COSINE_FREQUENCY_SCALE",
    "HORIZONS_PER_FREQUENCY",
    "MIN_HEADLINE_DATES",
    "REQUIRED_COLUMNS",
    "andrews_lag",
    "bh_adjust",
    "bonferroni_adjust",
    "compare_ic_series",
    "cosine_variance",
    "diebold_mariano",
    "headline_degrees_of_freedom",
    "headline_lag",
    "holm_adjust",
    "mean_vs_null_test",
    "newey_west_variance",
    "paired_comparison",
]
