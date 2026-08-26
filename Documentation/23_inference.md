# Statistical inference and diagnostics

Twenty tools in the `research` runtime that answer questions a summary
statistic hides. They fall into three groups: **error bars** on estimates
that are normally reported without any, **structure** in a series that a
mean and a standard deviation cannot see, and **corrections** for the fact
that you looked at more than one thing.

`08_analysis.md` covers the older statistical surface — factor regression,
cointegration, PCA, Hurst. This document covers what was added around it.

## The organising problem: almost nothing here is one test

Three of these tools exist because a naive version of the same question
produces a false positive rate several times its nominal one, and nothing
about the output reveals it. That pattern is worth stating once:

| Question | Naive approach | Its real false-positive rate |
|---|---|---|
| Does A lead B? | Test every lag, report the smallest p | ~15% at a nominal 5% |
| Is there a calendar effect? | Test twelve months at 5% | 46% chance of at least one hit |
| Which pair leads which? | 20 assets × 3 lags, take the best | ~57 of 1,140 clear an uncorrected bar |

In each case the individual test is correctly calibrated. What is wrong is
the claim built on top of it. All three are Bonferroni corrected here, the
correction is visible in the result, and a test pins the *flag* rather than
just the reported p-value — because a mutation that changed only the flag
survived the first version of those tests.

## Error bars

### `get_bootstrap_interval`

A confidence interval for a statistic, by **block** bootstrap.

The point estimate is usually reported alone and usually should not be. A
Sharpe of 1.2 on two years of daily data has a 95% interval running from
roughly 0.2 to 2.2 — consistent with a mediocre strategy and with an
excellent one. The interval is the number a decision should be made on.

**Blocked, not IID, and the default matters.** Resampling individual returns
destroys the serial correlation that drives drawdowns and volatility
clustering, and the resulting interval is too narrow in a way nothing about
it reveals. Measured on AR(1) returns, as the ratio of the blocked interval
to the IID one:

| φ | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---:|---:|---:|---:|---:|
| Sharpe | 0.97 | 1.16 | 1.41 | 1.77 | **2.24** |
| Max drawdown | 0.96 | 1.13 | 1.30 | 1.45 | **1.63** |

At φ = 0 the two agree, so the correction costs nothing on a series that
does not need it. The **Sharpe** is the more affected of the two — not the
drawdown, which is the intuition that reasoning from "path dependence"
produces. The Sharpe depends directly on the variance estimate, which is
exactly what serial correlation distorts.

The interval is validated by **coverage**: 150 samples from a process whose
true Sharpe is known by construction, and the nominal 95% interval covers it
93% of the time. That is the only property a confidence interval has, and it
is the only test that would catch an IID bootstrap masquerading as a blocked
one.

Eleven statistics are available (`sharpe`, `sortino`, `max_drawdown`,
`var_95`, `cvar_95`, `win_rate`, and the moments). The set is closed rather
than accepting arbitrary code, because this surface is reachable from an
agent and an eval-shaped hole is not worth the generality.

### `test_normality` and `estimate_tail_index`

Almost every risk number in this library assumes normality somewhere.
Parametric VaR multiplies a standard deviation by 1.645. A Sharpe's
confidence interval uses a normal approximation. Every "2-sigma move" quoted
anywhere is a normal statement. Returns are essentially never normal, and
these quantify by how much so the error can be sized.

`test_normality` runs Jarque-Bera, but **the p-value is not the interesting
output**. On 2,000 observations a trivially small departure is significant,
so the p-value is measuring sample length. The interesting output is the
**tail ratio**: observations beyond three standard deviations against the
0.27% a normal predicts. Three to five times that is common, and it is what
makes a parametric VaR understate the loss.

`estimate_tail_index` runs the Hill estimator. Alpha says which **moments
exist** — below 4 the kurtosis is infinite, so a sample kurtosis is an
artefact of the sample size; below 2 the *variance* is infinite and every
volatility, Sharpe and correlation computed on the series is meaningless.

**It is biased low**, measured: 2.39 for a true 3.0 and 3.48 for a true 5.0.
Hill assumes an exactly Pareto tail and a t-distribution is only
asymptotically so. Read alpha as a lower bound on tail thinness rather than
as a measurement — and note that the estimator flags its own worse case:
the t(5) estimate came with a 2.65 spread across thresholds, which trips the
instability warning, while the better t(3) estimate did not.

### `compare_distributions`

Two samples, one question: are these the same distribution? It comes up
constantly — in-sample against out-of-sample, this regime against that one,
live trading against the backtest — and it is usually answered by comparing
means, which misses every difference in shape.

**KS is not a tail test**, and the result says so because the demonstration
is built into the test suite. A normal against a t(3):

| | KS p-value | Kurtosis change | 1st percentile |
|---|---:|---:|---:|
| normal vs t(3) | 0.22 | **+3.8** | moved by 1.86× |

Reading only the p-value would conclude nothing changed. KS's power is
concentrated near the median, so the percentile comparison is reported
separately — for returns, the tail is the entire risk.

The result also names **which moment moved**, which is what makes it
actionable rather than merely a rejection.

## Structure

### `get_sharpe_stability`

Whether the edge decayed, or the full-sample Sharpe is the average of a good
period and a dead one. A Sharpe of 1.0 made of 2.0 in the first half and 0.0
in the second is arithmetically correct and describes a dead strategy — and
the second half is the half that predicts tomorrow.

**The p-value comes from two non-overlapping halves**, not from a regression
on the rolling series, and the reason is a bug this tool had. Regressing the
rolling Sharpe on time looks natural and is invalid: consecutive windows
share all but one observation. Correcting for that by inflating the standard
error *and* cutting the degrees of freedom applied the same correction twice
— and on a series whose Sharpe visibly fell from 1.9 to 0.0, the result was
`p = 0.17` and `decaying: false`.

Measured calibration of the current version: **4.0% false positives** on a
constant edge (one-sided 5% test, 6 of 150) with **62% power** against a real
halving (37 of 60).
That power is not high and it is honest — a Sharpe on 600 days has a
standard error near 0.65, and a test claiming better on this much data would
be miscalibrated.

### `get_drawdown_profile`

Maximum drawdown is one number describing one event, and it is the single
most over-used statistic in the business. It says nothing about how often
drawdowns happen, how long they last, or whether the worst was a one-day gap
or a two-year grind — and those determine whether a strategy is holdable far
more than depth does.

**Time underwater is usually the binding constraint.** A 20% drawdown
recovering in a month is survivable; one taking three years ends the
mandate, because no allocator waits three years. Depth and duration are
close to independent, so both are reported per episode.

### `get_correlation_stability`

The full-sample correlation is the one number every hedge is built on, and
it is routinely an artefact. Two assets correlating at 0.0 over ten years
may have correlated at +0.7 for five and −0.7 for five; the average is
meaningless and a hedge sized on it is wrong in both regimes.

The result reports the sign-flip count, the range, and — separately — the
correlation **conditional on the joint worst decile**. Correlations move
toward 1 when everything falls together, so a hedge computed on a calm
sample fails precisely when it is needed. That conditional number is what a
diversification claim has to survive.

### `test_autocorrelation`, `get_entropy_measures`

`test_autocorrelation` is a **joint** Ljung-Box across lags rather than one
test per lag. Check 20 lags individually at 5% on white noise and you expect
one to fire; reporting that as "returns are autocorrelated at lag 13" is an
uncorrected multiple comparison and is how a great many spurious signals
begin.

Set `squared=True` to test volatility clustering instead of direction. The
two are cleanly separated on a simulated GARCH series — returns p = 0.12,
squared returns p ≈ 0 — which is the textbook result and the reason the flag
exists. Finding autocorrelation in squared returns is not a trading signal;
it is why GARCH exists.

`get_entropy_measures` is the one tool here that does **not assume
linearity**. Every other statistical tool in this library measures linear
dependence and returns nothing on a series that is perfectly deterministic
in a nonlinear way. Permutation entropy reads only the rank order inside
each window, so it is invariant to any monotone transformation and robust to
outliers: a random series scores 0.9998 of maximum, an alternating one
0.387, and a monotone trend exactly 0.0 (it visits one ordinal pattern).

### `test_structural_break`

A Chow test at a **known** date, and the "known" is load-bearing. A test at
a date chosen because the data looks different there is not a valid test —
the hypothesis was picked using the data, and the F distribution being
compared against assumes it was not.

Valid when the date comes from outside: a regulation taking effect, a fee
change, an index reconstitution, a strategy going live. For an unknown date,
`detect_change_points` searches and reports the gain, which is the honest
form of that question. This function is deliberately not a searcher.

With a `regressor` it tests whether the **relationship** broke — a beta or a
hedge ratio — rather than whether the mean moved. On a planted break from
β = 0.5 to β = 2.0 it recovers 0.52 and 2.00.

### `decompose_returns`

**The arithmetic mean is not what you earned.** Compound growth is the
arithmetic mean minus roughly half the variance, and for a volatile strategy
that drag is most of the return: a series averaging 0.08% a day with 3%
daily volatility has an arithmetic annual return of about 20% and a compound
one near 9%. Reporting the first as "the return" is the most common
overstatement in the business.

The decomposition also separates the contribution of the best and worst five
days. A strategy whose entire compound return disappears when five days are
removed is a lottery ticket with good statistics; one whose return survives
that is robust.

## The tools

| Tool | Answers |
|---|---|
| `get_bootstrap_interval` | What is the error bar on this statistic |
| `compare_distributions` | Are these two samples the same distribution — in shape, not just mean |
| `test_normality` | How far from normal, and how much does the tail exceed what a normal predicts |
| `estimate_tail_index` | Which moments actually exist |
| `get_correlation_stability` | Is this correlation a property of the pair or an average over regimes |
| `decompose_returns` | Where did the compound growth come from |
| `get_sharpe_stability` | Did the edge decay |
| `get_drawdown_profile` | Every drawdown, not just the worst |
| `test_autocorrelation` | Is there autocorrelation at all, jointly across lags |
| `get_entropy_measures` | Is there structure a linear test would miss |
| `run_seasonality_analysis` | Is this a calendar effect, corrected for having looked at all of them |
| `get_lead_lag_matrix` | Which series move first — and does anything survive the search size |
| `test_structural_break` | Did something break at this known date |
| `detect_change_points` | When did the process change, at an unknown date |
| `run_stationarity_tests` | ADF, KPSS and variance ratio, with the four-way verdict |
| `detect_regimes` | Label each observation with a volatility regime |
| `test_granger_causality` | Does A precede B — Bonferroni corrected for the lags tested |
| `get_partial_correlation` | What is left once the common drivers are removed from both |
| `analyze_tail_dependence` | Do these move together in the tail, where it matters |

Full argument lists: [20_tool_index.md](20_tool_index.md#research--research).

## Related

- [08_analysis.md](08_analysis.md) — factor regression, cointegration, PCA, Hurst
- [24_overfitting.md](24_overfitting.md) — the same discipline applied to backtests
- [17_correctness.md](17_correctness.md) — how these are validated
