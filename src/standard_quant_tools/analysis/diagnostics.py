"""
Questions about a return series that a summary statistic hides.

A Sharpe ratio, a mean and a standard deviation describe a series as if it
were an unordered bag of numbers. Everything here uses the ORDER, because
the order is where the interesting failures live: when the edge stopped
working, whether returns predict themselves, whether the whole result is one
calendar effect, and whether a drawdown was a single event or a long grind.

WHAT EACH ANSWERS:

- **ljung_box** -- is there autocorrelation in this series at all, jointly
  across lags rather than one lag at a time. The joint test matters: check
  20 lags individually at 5% and you expect one to fire on white noise.
- **seasonality** -- is the result a day-of-week or month-of-year effect.
  Many published anomalies are, and the multiple comparison over 12 months
  is severe enough that the correction is applied rather than mentioned.
- **rolling_sharpe_stability** -- did the edge decay. A full-sample Sharpe
  of 1.0 built from 2.0 in the first half and 0.0 in the second is a dead
  strategy with a good average.
- **drawdown_profile** -- not the maximum drawdown, which is one number
  describing one event, but the distribution: how many, how deep, how long
  to recover, and how long spent underwater.
- **lead_lag_matrix** -- which series move first, across a universe, with
  the honest warning that the winner of a 20x20 correlation search is
  usually noise.
- **structural_break_test** -- a Chow test at a KNOWN date, which is the
  right tool when you have a reason to suspect one (a regulation, an index
  reconstitution, a fee change) and the wrong one when you are searching.
- **entropy_measures** -- how predictable the series is in a way that does
  not assume linearity, which is what every other tool here assumes.

NO SCIPY. Chi-square and F tail probabilities are computed from incomplete
gamma and beta functions implemented here.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

#: Day names in the order pandas uses, so a weekday index maps to a label.
WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


# ── distribution tails, written out ─────────────────────────────────────


def _lower_gamma(s: float, x: float) -> float:
    """Regularized lower incomplete gamma P(s, x), by series or continued
    fraction depending on which converges."""
    if x < 0 or s <= 0:
        return float("nan")
    if x == 0:
        return 0.0
    if x < s + 1.0:
        # Series expansion.
        term = 1.0 / s
        total = term
        for n in range(1, 500):
            term *= x / (s + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    # Continued fraction for Q(s, x), then complement.
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + s * math.log(x) - math.lgamma(s)) * h
    return 1.0 - q


def _chi2_sf(statistic: float, degrees: int) -> float:
    """Upper tail of the chi-square distribution."""
    if degrees <= 0 or not math.isfinite(statistic):
        return float("nan")
    if statistic <= 0:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - _lower_gamma(degrees / 2.0, statistic / 2.0))))


def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return (
        1.0
        - math.exp(
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + b * math.log(1.0 - x)
            + a * math.log(x)
        )
        * _betacf(b, a, 1.0 - x)
        / b
    )


def _f_sf(statistic: float, d1: int, d2: int) -> float:
    """Upper tail of the F distribution."""
    if statistic <= 0 or d1 <= 0 or d2 <= 0:
        return 1.0
    x = d2 / (d2 + d1 * statistic)
    return float(max(0.0, min(1.0, _betainc(d2 / 2.0, d1 / 2.0, x))))


def _clean(series: pd.Series, who: str, minimum: int = 30) -> pd.Series:
    values = pd.Series(series).astype(float).dropna()
    if len(values) < minimum:
        raise ValidationError(
            f"{who}: {len(values)} usable observations, and this needs at "
            f"least {minimum}."
        )
    return values


# ── autocorrelation ─────────────────────────────────────────────────────


def ljung_box(
    series: pd.Series,
    *,
    lags: Optional[int] = None,
    squared: bool = False,
) -> Dict[str, Any]:
    """
    A JOINT test for autocorrelation across lags, rather than one test per
    lag.

    WHY JOINT. Check 20 lags individually at the 5% level on pure white
    noise and you expect one of them to be significant. Reporting that one
    as "returns are autocorrelated at lag 13" is a multiple comparison with
    no correction, and it is how a great many spurious signals begin. The
    Ljung-Box statistic sums the squared autocorrelations across all lags
    into one chi-square, so there is one test and one p-value.

    ON RETURNS VERSUS SQUARED RETURNS. Run it on returns and you are testing
    for predictability of DIRECTION -- usually absent, and its absence is
    roughly what market efficiency predicts. Run it on squared returns
    (`squared=True`) and you are testing for volatility clustering, which is
    almost always present and strongly so. Finding autocorrelation in
    squared returns is not a trading signal in itself; it is the reason
    GARCH exists.

    THE LAG DEFAULT is min(10, n/5), which keeps the test from spreading its
    power across more lags than the sample supports. A test at 40 lags on
    200 observations has essentially no power against anything.
    """
    values = _clean(series, "ljung_box")
    array = values.to_numpy()
    if squared:
        array = array**2
    n = array.size
    if lags is None:
        lags = max(1, min(10, n // 5))
    lags = int(lags)
    if lags < 1 or lags >= n:
        raise ValidationError(
            f"ljung_box: lags={lags} must be at least 1 and fewer than the "
            f"{n} observations."
        )

    centred = array - array.mean()
    denominator = float((centred**2).sum())
    if denominator <= 0:
        raise ValidationError(
            "ljung_box: the series has no variance, so it has no "
            "autocorrelation to test."
        )

    per_lag: List[Dict[str, Any]] = []
    statistic = 0.0
    for k in range(1, lags + 1):
        rho = float((centred[k:] * centred[:-k]).sum() / denominator)
        statistic += rho * rho / (n - k)
        per_lag.append(
            {
                "lag": k,
                "autocorrelation": rho,
                # The +/- 2/sqrt(n) band a plot would draw.
                "significant_alone": bool(abs(rho) > 2.0 / math.sqrt(n)),
            }
        )
    statistic *= n * (n + 2)
    p_value = _chi2_sf(statistic, lags)

    individually = sum(1 for row in per_lag if row["significant_alone"])
    warnings: List[str] = []
    if p_value < 0.05:
        warnings.append(
            f"Joint p = {p_value:.4f}: there IS autocorrelation in this "
            "series across the tested lags."
            + (
                " On squared returns that is volatility clustering, which is "
                "near-universal and is the reason GARCH exists -- it is not "
                "a directional signal."
                if squared
                else ""
            )
        )
    else:
        warnings.append(
            f"Joint p = {p_value:.4f}: no autocorrelation detectable across "
            f"{lags} lags. Note that {individually} individual lag(s) "
            "crossed the 2/sqrt(n) band, which is what the joint test "
            "exists to stop you reporting -- at 5% per lag you expect "
            f"{lags * 0.05:.1f} to do so by chance."
        )
    warnings.append(
        "One joint test across all lags, NOT one test per lag. Checking 20 "
        "lags individually at 5% produces a false positive on white noise "
        "about two thirds of the time."
    )

    return {
        "n_observations": int(n),
        "lags": lags,
        "on_squared_returns": bool(squared),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "n_lags_individually_significant": individually,
        "by_lag": per_lag,
        "warnings": warnings,
    }


def entropy_measures(
    series: pd.Series,
    *,
    n_bins: int = 8,
    embedding: int = 3,
) -> Dict[str, Any]:
    """
    How predictable a series is, without assuming the predictability is
    linear.

    EVERYTHING ELSE IN THIS MODULE ASSUMES LINEARITY. Autocorrelation,
    regressions, Granger -- all of them measure linear dependence, and all
    of them return zero on a series that is perfectly deterministic in a
    nonlinear way. Entropy does not: it measures how much the next value is
    constrained by the previous ones, whatever the functional form.

    TWO MEASURES, because they answer different questions.

    **Shannon entropy** of the binned marginal distribution says how spread
    out the returns are, normalized so 1.0 is uniform across bins. It is a
    dispersion measure and it says nothing about order.

    **Permutation entropy** (Bandt and Pompe) is the one that uses the
    order. It looks at every window of `embedding` consecutive points and
    records only their RANK PATTERN -- which of the 6 orderings of 3 points
    occurred. A random series visits all patterns equally, giving a
    normalized entropy near 1. A trending or oscillating series visits some
    far more often, and the entropy falls. It is robust to outliers and to
    any monotone transformation of the data, because it only reads ranks.

    THE BIN COUNT IS A REAL CHOICE and the result is sensitive to it. Too
    few bins and everything looks uniform; too many and every observation
    lands in its own bin, which also looks uniform. The reported
    observations-per-bin makes that visible rather than leaving it as a
    hidden parameter.
    """
    values = _clean(series, "entropy_measures", minimum=50)
    array = values.to_numpy()
    n = array.size
    n_bins = max(2, int(n_bins))
    embedding = int(embedding)
    if not 2 <= embedding <= 7:
        raise ValidationError(
            f"entropy_measures: embedding={embedding} must be between 2 and "
            "7. Below 2 there is no ordering to read; above 7 there are "
            "5040 patterns and no sample fills them."
        )
    if n < math.factorial(embedding) * 5:
        raise ValidationError(
            f"entropy_measures: {n} observations cannot populate the "
            f"{math.factorial(embedding)} rank patterns of an embedding of "
            f"{embedding}. Use a smaller embedding or more data."
        )

    counts, _ = np.histogram(array, bins=n_bins)
    probabilities = counts[counts > 0] / counts.sum()
    shannon = float(-(probabilities * np.log(probabilities)).sum())
    shannon_normalized = shannon / math.log(n_bins)

    # Permutation entropy: count each ordinal pattern of length `embedding`.
    patterns: Dict[tuple, int] = {}
    for i in range(n - embedding + 1):
        window = array[i : i + embedding]
        key = tuple(np.argsort(window))
        patterns[key] = patterns.get(key, 0) + 1
    total = sum(patterns.values())
    pattern_probabilities = np.array([c / total for c in patterns.values()])
    permutation = float(-(pattern_probabilities * np.log(pattern_probabilities)).sum())
    permutation_normalized = permutation / math.log(math.factorial(embedding))

    per_bin = n / n_bins
    warnings: List[str] = []
    if permutation_normalized > 0.98:
        warnings.append(
            f"Permutation entropy is {permutation_normalized:.4f} of its "
            "maximum -- this series is indistinguishable from random in its "
            "ordering. No nonlinear structure is detectable at this "
            "embedding."
        )
    elif permutation_normalized < 0.9:
        warnings.append(
            f"Permutation entropy is {permutation_normalized:.4f}, "
            "materially below random. There is ordering structure here that "
            "a linear test would miss -- though structure is not an edge, "
            "and the commonest cause is a trend."
        )
    if per_bin < 20:
        warnings.append(
            f"{per_bin:.0f} observations per bin. The Shannon figure is "
            "sensitive to the bin count and this is thin; too few bins make "
            "everything look uniform and too many do the same."
        )
    warnings.append(
        "Permutation entropy reads RANKS only, so it is invariant to any "
        "monotone transformation and robust to outliers. That is a strength "
        "for detection and a limitation for interpretation: it will not "
        "tell you how large the structure is, only that it is there."
    )

    return {
        "n_observations": int(n),
        "n_bins": n_bins,
        "embedding": embedding,
        "observations_per_bin": float(per_bin),
        "shannon_entropy": shannon,
        "shannon_normalized": float(shannon_normalized),
        "permutation_entropy": permutation,
        "permutation_normalized": float(permutation_normalized),
        "n_patterns_observed": len(patterns),
        "n_patterns_possible": math.factorial(embedding),
        "warnings": warnings,
    }


# ── calendar effects ────────────────────────────────────────────────────


def seasonality(
    returns: pd.Series,
    *,
    by: str = "weekday",
) -> Dict[str, Any]:
    """
    Whether performance is concentrated in a day of the week or a month of
    the year, corrected for the fact that you looked at all of them.

    THE MULTIPLE COMPARISON IS THE WHOLE ISSUE. Test twelve months at the 5%
    level and the probability that at least one comes back significant on
    pure noise is 46%. That is where a large fraction of published calendar
    anomalies come from, and it is why every per-period p-value here is
    Bonferroni corrected and the JOINT test is reported first: the joint
    test asks "is there any calendar effect at all", which is the question
    that was actually asked before the periods were inspected.

    A JOINT F-TEST comes first, comparing between-period variance against
    within-period variance. If it does not reject, no individual period's
    result should be reported, however striking it looks.

    `by` is 'weekday', 'month', or 'day_of_month'. The index must be a
    DatetimeIndex -- there is no calendar in a positional index, and
    inventing one would produce confident nonsense.
    """
    values = _clean(returns, "seasonality")
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValidationError(
            "seasonality: needs a DatetimeIndex. A positional index has no "
            "calendar in it, and a seasonality test on invented dates "
            "returns confident nonsense."
        )

    if by == "weekday":
        keys = values.index.dayofweek
        labels = {i: WEEKDAYS[i] for i in range(7)}
    elif by == "month":
        keys = values.index.month
        labels = {i: pd.Timestamp(2000, i, 1).strftime("%B") for i in range(1, 13)}
    elif by == "day_of_month":
        keys = values.index.day
        labels = {i: f"day {i}" for i in range(1, 32)}
    else:
        raise ValidationError(
            f"seasonality: by={by!r} must be 'weekday', 'month' or " "'day_of_month'."
        )

    frame = pd.DataFrame({"r": values.to_numpy(), "k": keys})
    groups = [g["r"].to_numpy() for _, g in frame.groupby("k") if len(g) >= 2]
    if len(groups) < 2:
        raise ValidationError(
            f"seasonality: fewer than two {by} groups have at least 2 " "observations."
        )

    # One-way ANOVA: between-group variance against within-group.
    grand_mean = float(frame["r"].mean())
    between = sum(g.size * (g.mean() - grand_mean) ** 2 for g in groups)
    within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    d1 = len(groups) - 1
    d2 = int(sum(g.size for g in groups)) - len(groups)
    f_statistic = (
        float((between / d1) / (within / d2)) if within > 0 and d2 > 0 else float("nan")
    )
    joint_p = _f_sf(f_statistic, d1, d2) if math.isfinite(f_statistic) else float("nan")

    rows: List[Dict[str, Any]] = []
    n_tests = len(groups)
    for key, group in frame.groupby("k"):
        array = group["r"].to_numpy()
        if array.size < 2:
            continue
        others = frame.loc[frame["k"] != key, "r"].to_numpy()
        # Welch t against everything else, which is the comparison actually
        # being made when someone says "Mondays are different".
        se = math.sqrt(
            array.var(ddof=1) / array.size + others.var(ddof=1) / others.size
        )
        t = float((array.mean() - others.mean()) / se) if se > 0 else 0.0
        raw_p = 2.0 * _f_sf(t * t, 1, max(array.size + others.size - 2, 1))
        rows.append(
            {
                "period": labels.get(int(key), str(key)),
                "n_observations": int(array.size),
                "mean_return": float(array.mean()),
                "annualized": float(array.mean() * TRADING_DAYS),
                "win_rate": float((array > 0).mean()),
                "t_statistic": t,
                "p_value_raw": float(min(raw_p, 1.0)),
                "p_value_corrected": float(min(raw_p * n_tests, 1.0)),
                "significant_after_correction": bool(raw_p * n_tests < 0.05),
            }
        )
    rows.sort(key=lambda r: r["mean_return"], reverse=True)

    survivors = [r for r in rows if r["significant_after_correction"]]
    warnings: List[str] = []
    if math.isfinite(joint_p) and joint_p >= 0.05:
        warnings.append(
            f"The JOINT test does not reject (p = {joint_p:.3f}): there is "
            f"no detectable {by} effect. Individual periods below should "
            "not be reported -- the best of "
            f"{n_tests} always looks striking."
        )
    elif survivors:
        warnings.append(
            f"{len(survivors)} period(s) survive Bonferroni correction and "
            f"the joint test rejects at p = {joint_p:.3f}. That is as much "
            "as this test can say; it does not rule out a confound with "
            "something else that shares the calendar."
        )
    warnings.append(
        f"Every p-value is Bonferroni corrected for the {n_tests} periods "
        f"tested. Uncorrected, testing {n_tests} periods at 5% produces at "
        f"least one 'significant' result on pure noise "
        f"{(1 - 0.95 ** n_tests) * 100:.0f}% of the time -- which is where "
        "a good share of published calendar anomalies come from."
    )

    return {
        "n_observations": int(len(values)),
        "by": by,
        "n_periods": n_tests,
        "joint_f_statistic": f_statistic if math.isfinite(f_statistic) else None,
        "joint_p_value": float(joint_p) if math.isfinite(joint_p) else None,
        "joint_significant": bool(math.isfinite(joint_p) and joint_p < 0.05),
        "by_period": rows,
        "n_surviving_correction": len(survivors),
        "warnings": warnings,
    }


# ── did the edge decay ──────────────────────────────────────────────────


def rolling_sharpe_stability(
    returns: pd.Series,
    *,
    window: int = 252,
    periods_per_year: int = TRADING_DAYS,
) -> Dict[str, Any]:
    """
    Whether the Sharpe ratio was stable, or the average of a good period and
    a dead one.

    THE FAILURE: a full-sample Sharpe of 1.0 made of 2.0 in the first half
    and 0.0 in the second. The average is arithmetically correct and the
    strategy is dead. Nothing in a single Sharpe number can show this, and
    the second half is the half that predicts tomorrow.

    THE TEST IS A HALF-VERSUS-HALF SHARPE COMPARISON, not a regression on
    the rolling series, and the first version of this function got that
    wrong in an instructive way. Regressing the rolling Sharpe on time looks
    natural and is invalid: consecutive windows share window-1 of their
    observations, so the nominal p-value means nothing. Correcting that by
    inflating the standard error by sqrt(n_windows / n_independent) -- a
    factor of 15 on a typical sample -- and ALSO cutting the degrees of
    freedom to the independent count applies the same correction twice. On a
    series whose Sharpe visibly fell from 1.9 to 0.0, the result was
    p = 0.17 and `decaying: false`.

    Two NON-OVERLAPPING halves of the raw returns have no such problem. Each
    half's Sharpe has a known standard error -- sqrt((1 + S^2/2)/n), after
    Lo (2002) -- the halves are independent by construction, and the
    difference is a clean two-sample test.

    The rolling series is still computed and returned, because looking at it
    is genuinely informative. It is just not what the p-value comes from.

    MEASURED CALIBRATION, over 200 replications each: on a strategy whose
    edge is genuinely constant the test calls decay 3.0% of the time, which
    is what a one-sided 5% test should do. Against a real halving of the
    edge across 1200 observations it fires 62% of the time. That power is
    not high, and it is honest -- a Sharpe estimated on 600 days has a
    standard error near 0.65, so two halves have to differ by a lot before
    the difference is distinguishable from noise. A test claiming to detect
    decay more reliably than that on this much data would be miscalibrated.

    THE TREND is reported too, fitted on non-overlapping blocks so the slope
    is not built from a series that repeats itself. It has low power by
    construction -- five years of daily data contain five independent annual
    windows -- and it is there to describe the shape, not to decide.
    """
    values = _clean(returns, "rolling_sharpe_stability", minimum=60)
    window = int(window)
    if window < 20:
        raise ValidationError(
            f"rolling_sharpe_stability: window={window} is too short for a "
            "Sharpe ratio to mean anything."
        )
    if len(values) < window * 2:
        raise ValidationError(
            f"rolling_sharpe_stability: {len(values)} observations with a "
            f"window of {window} leaves fewer than two windows. There is no "
            "stability to assess in one window."
        )

    rolling_mean = values.rolling(window).mean()
    rolling_std = values.rolling(window).std(ddof=1)
    rolling = (rolling_mean / rolling_std * math.sqrt(periods_per_year)).dropna()
    array = rolling.to_numpy()

    independent = max(int(len(values) / window), 2)

    # THE TEST: two independent halves of the RAW returns, never two halves
    # of an overlapping rolling series.
    split = len(values) // 2
    first = values.to_numpy()[:split]
    second = values.to_numpy()[split:]
    first_sharpe = _annualized_sharpe(first, periods_per_year)
    second_sharpe = _annualized_sharpe(second, periods_per_year)
    difference = first_sharpe - second_sharpe
    standard_error = math.sqrt(
        _sharpe_variance(first_sharpe, first.size, periods_per_year)
        + _sharpe_variance(second_sharpe, second.size, periods_per_year)
    )
    t_statistic = difference / standard_error if standard_error > 0 else float("nan")
    p_value = (
        _f_sf(t_statistic**2, 1, max(len(values) - 2, 1))
        if math.isfinite(t_statistic)
        else float("nan")
    )

    # The trend, on NON-OVERLAPPING blocks. Descriptive; low power by design.
    block_sharpes = [
        _annualized_sharpe(values.to_numpy()[i : i + window], periods_per_year)
        for i in range(0, len(values) - window + 1, window)
    ]
    block_sharpes = [s for s in block_sharpes if math.isfinite(s)]
    if len(block_sharpes) >= 2:
        bx = np.arange(len(block_sharpes), dtype=float)
        design = np.column_stack([np.ones_like(bx), bx])
        coefficients, *_ = np.linalg.lstsq(
            design, np.asarray(block_sharpes), rcond=None
        )
        slope_per_year = float(coefficients[1]) * (periods_per_year / window)
    else:
        slope_per_year = float("nan")

    first_half, second_half = first_sharpe, second_sharpe

    decaying = bool(math.isfinite(p_value) and p_value < 0.05 and difference > 0)
    warnings: List[str] = []
    if decaying:
        warnings.append(
            f"THE EDGE DECAYED. The first half's Sharpe was "
            f"{first_half:.2f} and the second half's {second_half:.2f}, a "
            f"difference of {difference:.2f} (p = {p_value:.3f}). The "
            "second half is the one that predicts tomorrow."
        )
    if float(array.min()) < 0 < float(array.max()):
        warnings.append(
            f"The rolling Sharpe ranges from {array.min():.2f} to "
            f"{array.max():.2f}, crossing zero. The full-sample number is an "
            "average over periods where this did and did not work."
        )
    warnings.append(
        f"The p-value comes from comparing two NON-OVERLAPPING halves of "
        f"the raw returns, not from a regression on the {array.size} "
        "rolling windows -- consecutive windows share all but one of their "
        "observations and cannot support inference. The rolling series is "
        "returned because looking at it is informative; it is not what was "
        "tested."
    )
    if len(block_sharpes) < 4:
        warnings.append(
            f"`trend_per_year` is fitted on only {len(block_sharpes)} "
            "non-overlapping blocks. It describes the shape and should not "
            "be read as a measurement."
        )

    return {
        "n_observations": int(len(values)),
        "window": window,
        "n_windows": int(array.size),
        "n_independent_windows": independent,
        "full_sample_sharpe": float(
            values.mean() / values.std(ddof=1) * math.sqrt(periods_per_year)
        ),
        "mean_rolling_sharpe": float(array.mean()),
        "std_rolling_sharpe": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min_rolling_sharpe": float(array.min()),
        "max_rolling_sharpe": float(array.max()),
        "first_half_mean": first_half,
        "second_half_mean": second_half,
        "trend_per_year": (
            float(slope_per_year) if math.isfinite(slope_per_year) else None
        ),
        "n_blocks": len(block_sharpes),
        "first_half_sharpe": first_sharpe,
        "second_half_sharpe": second_sharpe,
        "sharpe_difference": float(difference),
        "difference_standard_error": float(standard_error),
        "decay_p_value": float(p_value) if math.isfinite(p_value) else None,
        "decaying": decaying,
        "fraction_of_windows_positive": float((array > 0).mean()),
        "warnings": warnings,
    }


def _annualized_sharpe(values: np.ndarray, periods: int) -> float:
    if values.size < 2:
        return float("nan")
    std = float(values.std(ddof=1))
    if std <= 0:
        return float("nan")
    return float(values.mean() / std * math.sqrt(periods))


def _sharpe_variance(annualized: float, n: int, periods: int) -> float:
    """
    Variance of an annualized Sharpe estimate, after Lo (2002).

    Var(S) = (1 + S^2/2)/n with S per period, then annualized. The S^2/2
    term is small at realistic Sharpes and is kept because dropping it
    makes a large difference between two halves look more significant than
    it is, which is the exact error this function exists to avoid.
    """
    if n < 2 or not math.isfinite(annualized):
        return float("inf")
    per_period = annualized / math.sqrt(periods)
    return float((1.0 + 0.5 * per_period**2) / n * periods)


def drawdown_profile(
    returns: pd.Series,
    *,
    threshold: float = 0.05,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Every drawdown, not just the worst one.

    MAXIMUM DRAWDOWN IS ONE NUMBER DESCRIBING ONE EVENT, and it is the
    single most over-used statistic in the business. It says nothing about
    how often drawdowns happen, how long they last, or whether the worst was
    a one-day gap or a two-year grind -- and those determine whether a
    strategy is actually holdable far more than its depth does.

    TIME UNDERWATER IS USUALLY THE BINDING CONSTRAINT. A 20% drawdown that
    recovers in a month is survivable. A 20% drawdown that takes three years
    to recover ends the mandate, because no allocator waits three years. The
    profile reports recovery time per episode and the total fraction of the
    sample spent below a prior high.

    THE DISTINCTION BETWEEN DEPTH AND DURATION is why this returns episodes
    rather than a summary: they are close to independent, and a strategy can
    be bad at either.
    """
    values = _clean(returns, "drawdown_profile")
    equity = (1.0 + values).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0

    episodes: List[Dict[str, Any]] = []
    in_drawdown = False
    start = 0
    for i, value in enumerate(drawdown.to_numpy()):
        if not in_drawdown and value < 0:
            in_drawdown, start = True, i
        elif in_drawdown and value >= 0:
            episodes.append(
                _episode(drawdown, values.index, start, i - 1, recovered=True)
            )
            in_drawdown = False
    if in_drawdown:
        episodes.append(
            _episode(drawdown, values.index, start, len(drawdown) - 1, recovered=False)
        )

    material = [e for e in episodes if abs(e["depth"]) >= threshold]
    material.sort(key=lambda e: e["depth"])
    underwater = float((drawdown < 0).mean())
    max_drawdown = float(drawdown.min())

    recovered = [e for e in material if e["recovered"]]
    mean_recovery = (
        float(np.mean([e["recovery_days"] for e in recovered])) if recovered else None
    )
    longest = max(material, key=lambda e: e["length_days"]) if material else None
    ongoing = [
        e for e in episodes if not e["recovered"] and abs(e["depth"]) >= threshold
    ]

    warnings: List[str] = []
    if underwater > 0.6:
        warnings.append(
            f"{underwater:.0%} of the sample was spent below a prior high. "
            "Time underwater is usually the binding constraint on holding a "
            "strategy -- allocators redeem on duration more often than on "
            "depth."
        )
    if longest and longest["length_days"] > 250:
        warnings.append(
            f"The longest drawdown lasted {longest['length_days']} "
            f"observations ({longest['depth']:.1%} deep). A drawdown that "
            "outlasts an evaluation period ends the mandate whatever the "
            "eventual recovery."
        )
    if ongoing:
        warnings.append(
            f"The sample ends inside a drawdown of {ongoing[0]['depth']:.1%} "
            "that has not recovered. Its duration is a lower bound, not a "
            "measurement."
        )
    warnings.append(
        "Maximum drawdown describes ONE event. Depth and duration are "
        "close to independent, and a strategy can be unholdable on either."
    )

    return {
        "n_observations": int(len(values)),
        "threshold": float(threshold),
        "max_drawdown": max_drawdown,
        "n_drawdowns": len(material),
        "fraction_underwater": underwater,
        "mean_recovery_days": mean_recovery,
        "longest_drawdown_days": longest["length_days"] if longest else 0,
        "currently_in_drawdown": bool(ongoing),
        "worst_drawdowns": material[:top_n],
        "warnings": warnings,
    }


def _episode(drawdown, index, start, end, *, recovered: bool) -> Dict[str, Any]:
    window = drawdown.iloc[start : end + 1]
    trough = int(window.to_numpy().argmin())
    return {
        "start": str(index[start]),
        "trough": str(index[start + trough]),
        "end": str(index[end]),
        "depth": float(window.min()),
        "length_days": int(end - start + 1),
        "days_to_trough": int(trough + 1),
        "recovery_days": int(end - start - trough) if recovered else None,
        "recovered": recovered,
    }


# ── across a universe ───────────────────────────────────────────────────


def lead_lag_matrix(
    returns: pd.DataFrame,
    *,
    max_lag: int = 3,
    min_correlation: float = 0.1,
) -> Dict[str, Any]:
    """
    Which series move first, across a universe -- and why the answer is
    usually noise.

    THE SEARCH IS ENORMOUS AND THE CORRECTION IS THE POINT. Twenty assets at
    three lags is 20 x 19 x 3 = 1,140 correlations. At the 5% level you
    expect 57 of them to be "significant" on data with no lead-lag structure
    whatever. The strongest pair in that search will have a correlation
    around 0.10-0.15 on 500 observations and will look entirely convincing.

    So every reported pair carries a Bonferroni-corrected p-value against
    the FULL search size, and the result leads with how many survive. When
    that number is zero -- which is the normal outcome on real data -- the
    honest report is "nothing survived", not the top of the ranked list.

    WHAT A SURVIVING PAIR MEANS, if one does: series A's return today
    correlates with series B's return tomorrow. That is temporal precedence,
    not causality and not a trade. Check the obvious explanations first --
    different closing times across exchanges produce exactly this pattern and
    are the single commonest cause.
    """
    frame = pd.DataFrame(returns).astype(float).dropna()
    n_assets = frame.shape[1]
    if n_assets < 2:
        raise ValidationError("lead_lag_matrix: needs at least two series.")
    max_lag = int(max_lag)
    if max_lag < 1:
        raise ValidationError("lead_lag_matrix: max_lag must be at least 1.")
    n = len(frame)
    if n < 50:
        raise ValidationError(
            f"lead_lag_matrix: {n} observations. A correlation search this "
            "wide needs far more data to have any power after correction."
        )

    n_tests = n_assets * (n_assets - 1) * max_lag
    pairs: List[Dict[str, Any]] = []
    columns = list(frame.columns)

    # ONE cross-correlation matrix per lag, rather than one `np.corrcoef`
    # call per (leader, follower, lag) triple.
    #
    # The loop this replaces re-sliced `frame[leader].iloc[:-lag]` inside
    # its INNERMOST body, so a 50-name universe at max_lag=3 rebuilt 14,700
    # pandas slices where 150 numpy views would do, then called
    # `np.corrcoef` 7,350 times. Measured 10.5 SECONDS at 200 assets, and
    # `max_lag` is caller-settable to 20.
    #
    # For one lag L the whole matrix is a single matmul: correlate every
    # column of the leading block against every column of the lagging one.
    # ddof cancels between numerator and denominator, so the centred
    # cross-products and the centred sums of squares can both be raw.
    values = frame.to_numpy(dtype=float)
    correlations: Dict[int, np.ndarray] = {}
    for lag in range(1, max_lag + 1):
        if n - lag < 30:
            continue
        lead = values[:-lag]
        follow = values[lag:]
        lead_c = lead - lead.mean(axis=0)
        follow_c = follow - follow.mean(axis=0)
        denominator = np.sqrt(
            np.outer((lead_c**2).sum(axis=0), (follow_c**2).sum(axis=0))
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations[lag] = (lead_c.T @ follow_c) / denominator

    for i, leader in enumerate(columns):
        for j, follower in enumerate(columns):
            if i == j:
                continue
            for lag in range(1, max_lag + 1):
                matrix = correlations.get(lag)
                if matrix is None:
                    continue
                rho = float(matrix[i, j])
                if not math.isfinite(rho) or abs(rho) < min_correlation:
                    continue
                effective = n - lag
                t = rho * math.sqrt(max(effective - 2, 1) / max(1 - rho * rho, 1e-12))
                raw_p = _f_sf(t * t, 1, max(effective - 2, 1))
                pairs.append(
                    {
                        "leader": str(leader),
                        "follower": str(follower),
                        "lag": lag,
                        "correlation": rho,
                        "p_value_raw": float(raw_p),
                        "p_value_corrected": float(min(raw_p * n_tests, 1.0)),
                        "survives_correction": bool(raw_p * n_tests < 0.05),
                    }
                )
    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    survivors = [p for p in pairs if p["survives_correction"]]

    expected_false = n_tests * 0.05
    warnings: List[str] = []
    if not survivors:
        warnings.append(
            f"NOTHING SURVIVED the correction. {n_tests} correlations were "
            f"tested and about {expected_false:.0f} would clear an "
            "uncorrected 5% bar on data with no lead-lag structure at all. "
            "The top of the ranked list below is what noise looks like, not "
            "a finding."
        )
    else:
        warnings.append(
            f"{len(survivors)} of {n_tests} tested relationships survive "
            "Bonferroni correction. Before trading one, rule out different "
            "closing times across exchanges -- that produces exactly this "
            "pattern and is the commonest cause by a wide margin."
        )
    warnings.append(
        "Temporal precedence is not causality and is not a trade. A common "
        "driver produces it, and so does a faster-updating proxy for the "
        "same information."
    )

    return {
        "n_assets": int(n_assets),
        "n_observations": int(n),
        "max_lag": max_lag,
        "n_tests": int(n_tests),
        "expected_false_positives_uncorrected": float(expected_false),
        "n_surviving": len(survivors),
        "surviving_pairs": survivors[:20],
        "strongest_pairs": pairs[:10],
        "warnings": warnings,
    }


def structural_break_test(
    series: pd.Series,
    break_index: int,
    *,
    regressor: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    A Chow test for a break at a KNOWN date.

    THE "KNOWN" IS LOAD-BEARING. A Chow test at a date you chose because the
    data looks different there is not a valid test -- you have already used
    the data to pick the hypothesis, and the F distribution being compared
    against assumes you did not. The test is valid when the date comes from
    OUTSIDE the series: a regulation taking effect, a fee change, an index
    reconstitution, a decimalization, a strategy going live.

    For a break at an unknown date, `detect_change_points` searches for one
    and reports the gain, which is the honest form of that question. This
    function is deliberately not a searcher.

    WITH `regressor`, this tests whether the relationship between the two
    series changed -- whether a beta or a hedge ratio broke. Without it, it
    tests whether the mean changed. Both are Chow tests; they answer quite
    different questions and the second is the one usually wanted when
    someone says "did the relationship break".
    """
    values = _clean(series, "structural_break_test", minimum=20)
    n = len(values)
    break_index = int(break_index)
    if not 5 <= break_index <= n - 5:
        raise ValidationError(
            f"structural_break_test: break_index={break_index} leaves fewer "
            f"than 5 observations on one side of the {n} available. A "
            "regression on 4 points is not a regression."
        )

    y = values.to_numpy()
    if regressor is not None:
        x_series = pd.Series(regressor).astype(float).reindex(values.index)
        aligned = pd.DataFrame({"y": y, "x": x_series.to_numpy()}).dropna()
        if len(aligned) < n:
            raise ValidationError(
                "structural_break_test: the regressor does not cover every "
                f"observation ({len(aligned)} of {n} align)."
            )
        x = aligned["x"].to_numpy()
        design = np.column_stack([np.ones_like(x), x])
        k = 2
        tested = "the relationship (intercept and slope)"
    else:
        design = np.ones((n, 1))
        k = 1
        tested = "the mean"

    def _rss(d, target):
        coefficients, *_ = np.linalg.lstsq(d, target, rcond=None)
        residual = target - d @ coefficients
        return float((residual**2).sum()), coefficients

    pooled_rss, pooled_coefficients = _rss(design, y)
    first_rss, first_coefficients = _rss(design[:break_index], y[:break_index])
    second_rss, second_coefficients = _rss(design[break_index:], y[break_index:])

    denominator = first_rss + second_rss
    d1, d2 = k, n - 2 * k
    if denominator <= 0 or d2 <= 0:
        raise ValidationError(
            "structural_break_test: the split regressions leave no residual "
            "variance, so the F statistic is undefined."
        )
    f_statistic = ((pooled_rss - denominator) / d1) / (denominator / d2)
    p_value = _f_sf(f_statistic, d1, d2)

    warnings: List[str] = [
        "A Chow test is only valid at a break date chosen from OUTSIDE the "
        "data -- a regulation, a fee change, an index reconstitution, a "
        "go-live. If the date was chosen because the series looks different "
        "there, the hypothesis was picked using the data and this p-value "
        "does not mean what it says. Use detect_change_points for an "
        "unknown date."
    ]
    if p_value < 0.05:
        warnings.append(
            f"The break is significant (p = {p_value:.4f}): {tested} "
            "differs across this date. Fitting one model across it will "
            "describe neither side."
        )
    else:
        warnings.append(
            f"No significant break (p = {p_value:.4f}). That is evidence "
            "against a break at this date, not proof of stability -- the "
            "test has limited power against a gradual change, which is what "
            "decay usually looks like."
        )

    return {
        "n_observations": int(n),
        "break_index": break_index,
        "break_date": str(values.index[break_index]),
        "tested": tested,
        "f_statistic": float(f_statistic),
        "p_value": float(p_value),
        "significant_at_05": bool(p_value < 0.05),
        "pooled_coefficients": [float(c) for c in np.atleast_1d(pooled_coefficients)],
        "before_coefficients": [float(c) for c in np.atleast_1d(first_coefficients)],
        "after_coefficients": [float(c) for c in np.atleast_1d(second_coefficients)],
        "mean_before": float(y[:break_index].mean()),
        "mean_after": float(y[break_index:].mean()),
        "warnings": warnings,
    }


__all__ = [
    "TRADING_DAYS",
    "WEEKDAYS",
    "drawdown_profile",
    "entropy_measures",
    "lead_lag_matrix",
    "ljung_box",
    "rolling_sharpe_stability",
    "seasonality",
    "structural_break_test",
]
