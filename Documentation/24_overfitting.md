# Overfitting and trade analysis

Eleven tools in the `backtest` runtime that answer a question the backtest
itself cannot: given that you tried N things and kept the best, how much of
that best result is skill and how much is the largest of N draws from a
distribution centred on zero?

`04_backtesting.md` covers running a strategy. This covers establishing
whether the result means anything.

## The arithmetic is brutal and it is not intuitive

Measured on simulated strategies with **no edge whatsoever** — every series
drawn from a zero-mean normal — over two years of daily data, the best of N
shows an annualized Sharpe of:

| N trials | 5 | 10 | 20 | 50 | 100 |
|---|---:|---:|---:|---:|---:|
| Best Sharpe | 0.84 | 1.11 | **1.34** | 1.59 | 1.79 |

Two hundred replications per column. Nothing works in any of those series.

A researcher who reports the 1.34 without saying "and I tried 19 others" has
not lied about any single number and has still communicated something false.
`get_deflated_sharpe_ratio` computes how much to subtract.

*(An earlier draft of this document put best-of-20 at "roughly 1.0", which
is what best-of-10 gives. The table above is measured rather than reasoned
about, which is the only reason it is right.)*

## What none of this fixes

These tools measure the multiple-testing cost of the trials you **tell them
about**. The trials nobody counted — the parameter you adjusted before
saving the file, the universe you narrowed after a first look, the two years
you dropped as unrepresentative — are invisible here and are usually the
larger number.

A PBO of 0.2 computed over a grid you arrived at after a month of informal
iteration is not a PBO of 0.2. Every result in this module says so.

## Multiple-testing corrections

### `get_deflated_sharpe_ratio`

After Bailey and Lopez de Prado (2014). Two steps: compute the Sharpe a
researcher would expect to achieve by **luck alone** after `n_trials`
attempts, then ask whether the observed Sharpe is far enough above that
threshold to survive.

**Skew and kurtosis matter here, unusually.** Most Sharpe inference assumes
normal returns; this does not, because the strategies that most need
deflating are exactly the ones with a short-volatility payoff — many small
gains, rare large losses. Negative skew and high kurtosis *widen* the
Sharpe's sampling distribution, so the same nominal Sharpe is less
significant. A strategy selling options gets penalised here, correctly.

Pass `trial_sharpes` — the Sharpe of every variant you tried — for the
accurate version. Their **variance** sets the deflation threshold and is far
more informative than the count alone: 100 near-identical parameter settings
deflate much less than 100 genuinely different ideas.

### `estimate_backtest_overfitting` (PBO)

After Bailey, Borwein, Lopez de Prado and Zhu (2015). Cut the period into S
chunks; for every way of splitting them into equal halves, pick the
configuration that ranked best in-sample and look up where it ranked
out-of-sample. PBO is the fraction of splits where the winner landed in the
bottom half.

**It is a property of your selection procedure, not of the strategy.** A PBO
of 0.5 means picking the in-sample best is no better than picking at random.
Above 0.5 means the in-sample winner is systematically the out-of-sample
loser, which is the signature of a grid fitted to noise. It does *not* mean
the strategy loses money.

Validated by construction:

| Grid | PBO |
|---|---:|
| 20 strategies, no edge anywhere | 0.457 |
| Same, with one genuine drift planted | 0.000 |
| 15 configurations correlated at 0.997 | 0.771 (and flagged) |

That last row is why the result reports the **median pairwise correlation**
between configurations. A hundred settings correlated at 0.99 are one
strategy with a parameter nudged; every split ranks them identically and the
PBO is measuring nothing.

### `run_reality_check`

White's Reality Check. Subtly different from a t-test on the strategy's
returns: a t-test asks "is this mean positive", while this asks "is this
outperformance larger than the largest you would expect from the best of
this many candidates under the null that none of them has an edge". The
second is the question that matters after a search.

**The bootstrap is blocked**, and it has to be. Resampling individual days
destroys the serial correlation that drives drawdowns and volatility
clustering, which makes the null far too narrow and the p-value far too
small.

Measured false-positive rate on no-edge strategies: **3.3%** against a
nominal 5%.

The stationarity assumption is the weak point and is named in every result:
block bootstrap assumes the return-generating process is the same
throughout, and a strategy whose edge genuinely existed until 2018 and
vanished afterwards violates that.

## Cross-validation that respects the arrow of time

### `build_purged_cv_splits`

**The leak this prevents.** A label built from a 5-day forward return at
time *t* is a function of prices through *t+5*. If *t* sits in the training
set and *t+3* sits in the test set, the training label already contains the
test period's answer. Plain k-fold cross-validation does this at every fold
boundary, and it is why a model shows 0.6 AUC in cross-validation and 0.5 in
production.

Two corrections, after Lopez de Prado:

- **Purging** removes training observations whose label window overlaps the
  test set at all.
- **Embargo** removes a further stretch immediately after the test set,
  because serial correlation in features means an observation shortly after
  the test period is still nearly the same observation. Purging handles the
  label overlap; the embargo handles the feature overlap.

Verified directly: at a 5-day label horizon, **zero** training observations
across all 15 paths have a label window touching their test set, and the
first training index after a test block sits at exactly `embargo + 1`.

**Combinatorial, not sequential.** Rather than one train/test split per
fold, every choice of `n_test_splits` groups out of `n_splits` becomes a
test set — C(n, k) paths instead of n. That gives out-of-sample performance
a *distribution* rather than one number with no error bar, which is the
whole point.

The result returns **half-open ranges** rather than every index. Purging
removes contiguous blocks, so a training set is a union of a few runs
however long the series is: 500 observations across 15 paths is 37 KB as
indices and 2.6 KB as ranges, and stays at 2.7 KB at 5,000 observations
where indices would have reached 360 KB.

## What the equity curve does not say

A backtest reports one path, in one order. Most of what is worth knowing is
about the paths that did not happen.

### `run_monte_carlo_trade_paths`

A strategy with a 20% backtested drawdown does not *have* a 20% drawdown. It
has a distribution of them, and the backtested one is a single draw that
routinely sits near the middle while the 95th percentile is half again as
deep. Sizing so the backtested drawdown is survivable is sizing on the
median outcome, and half of all realizations are worse.

**Reshuffling, not resampling with replacement.** Every path holds exactly
the same trades in a different order, so the total return is identical
across paths and only the *path* differs. That isolates sequence risk from
edge uncertainty; a bootstrap with replacement mixes the two, and the
drawdown distribution it produces is then partly about having drawn a
different strategy. The test suite pins this by checking that every path
ends at identical final equity — which cannot hold under replacement.

### `analyze_trade_clustering`

A 55% win rate is survivable if the losses are scattered and unholdable if
they arrive eleven in a row — and the win rate is identical in both cases.
The order is the data here, which is the one tool where shuffling the input
destroys the question.

A runs test gives the z-score: **negative is clustering**, positive is
alternation. Alternation is rarer and usually means the strategy is reacting
to its own last outcome, which is worth checking for a state-carrying bug.

**Read it alongside the Monte Carlo**, whose reshuffling destroys exactly
the clustering measured here — so that drawdown distribution is optimistic
by this much.

### `compare_against_random`

Comparing against zero or against buy-and-hold is the usual test and neither
is right after a search: a strategy can beat zero purely by holding a rising
asset with no timing skill at all.

The null keeps the trade **magnitudes** and the **win rate** and randomizes
the signs, so it tests whether the sequencing and sizing add anything — not
whether the win rate does. A strategy whose entire edge is a 55% win rate on
symmetric bets will look indistinguishable from this null, correctly: that
edge lives in the win rate, which the null was given.

**It is measurably conservative**: 1 of 150 skill-free strategies flagged
against a nominal 5%. That is the safe direction — it will not manufacture
significance — but it means a non-rejection is weaker evidence of no skill
than a nominal 5% test would suggest. It retains power against the case it
was built for: wins systematically larger than losses came back at p < 0.001.

### `get_exposure_attribution`

How much of the return came from being **right** versus from being
**invested**. A strategy's return is exposure times the market's move, and
splitting it shows whether the P&L came from timing — holding more before up
moves — or simply from average exposure to an asset that rose, which is beta
and available for a fee of zero.

The timing term is the covariance between exposure and the subsequent
return. It is usually far smaller than people expect, and often *negative*
in strategies that look profitable. The decomposition is exact —
`E[e·r] = E[e]E[r] + Cov(e,r)` is an identity, so constant exposure gives
exactly zero timing contribution.

Time in market is reported because it changes what the Sharpe means. A
strategy invested 20% of the time with a Sharpe of 1.5 and one invested 100%
of the time with the same Sharpe are different propositions: the first has
capital idle that has to earn something, and its Sharpe ignores that.

### `estimate_break_even_cost`

The per-trade cost at which the edge disappears — the number every backtest
should report and almost none do.

What decides whether a result survives contact with a real broker is not
whether it is profitable at 5 basis points but how far above 5 the
break-even sits. A strategy breaking even at 8bp when you modelled 5bp has
1.6× of headroom, and one bad fill, a widening spread or a venue change eats
it. One breaking even at 80bp is robust to all three.

**Under about 2×, the backtest is a statement about the cost assumption
rather than about the strategy.**

It models a flat per-trade charge and **not impact**. Impact grows with size
and is not flat, so a strategy with headroom here can still fail on capacity
— `get_capacity_report` and `estimate_trade_cost` answer that.

## Where the performance came from

### `get_regime_stratified_performance`

One Sharpe over a mixed sample describes none of the regimes in it. This
catches the strategy with an overall Sharpe of 1.2 that earned all of it in
one 18-month window and was flat-to-negative for the other eight years —
arithmetically correct and completely misleading.

**Concentration is the headline**: what fraction of total P&L came from the
single best regime. Above about 70% from one regime, the strategy is a bet
on that regime recurring, whatever it says on the tin.

### `analyze_parameter_decay`

Whether performance degrades smoothly as a parameter moves, or falls off a
cliff. A parameter whose neighbours perform almost as well describes a real
effect with a broad optimum — the exact value is not doing the work. A spike
in a noisy objective is almost always the one setting that happened to fit
the sample.

An optimum at the **grid edge** is flagged: the true one may lie outside it,
or performance is monotone in the parameter, which usually means it stands
in for something else (more lookback = less trading = less cost).

**A broad plateau at a bad level is still a bad level.** This measures
whether the parameter choice is robust, which is necessary for the result to
survive and nowhere near sufficient.

## The tools

| Tool | Answers |
|---|---|
| `get_deflated_sharpe_ratio` | Is this Sharpe real, given how many were tried |
| `estimate_backtest_overfitting` | Does my selection procedure have any skill |
| `build_purged_cv_splits` | Train/test indices that do not leak a forward-looking label |
| `run_reality_check` | Is this better than the best of the alternatives, or the luckiest |
| `run_monte_carlo_trade_paths` | What drawdown could this same edge have produced |
| `analyze_trade_clustering` | Do the losses arrive in runs I could not sit through |
| `compare_against_random` | Do the entry decisions beat a coin |
| `get_exposure_attribution` | Was it timing, or just being invested |
| `estimate_break_even_cost` | At what cost does this edge disappear |
| `get_regime_stratified_performance` | Which regime did the P&L actually come from |
| `analyze_parameter_decay` | Is this optimum a plateau or a spike |

Full argument lists: [20_tool_index.md](20_tool_index.md#backtest--backtest).

## Related

- [04_backtesting.md](04_backtesting.md) — running the strategy in the first place
- [23_inference.md](23_inference.md) — the same discipline on a return series
- [15_modeling.md](15_modeling.md) — leakage-purged validation for models
