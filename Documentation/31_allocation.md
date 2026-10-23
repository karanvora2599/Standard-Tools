# Capital Allocation

Eight tools for splitting capital ACROSS return streams -- strategies,
sleeves, models -- rather than across tickers: putting the streams on one
calendar, allocating among them with the uncertainty of the split, testing
the rule that chose it out of sample, how hard to lever a stream, an
overlay that changes its risk over time, how much money it can take before
its own trading eats it, where active return came from, and what a fund
with deposits and withdrawals actually returned.

It is not `portfolio`. That runtime builds a book out of ASSETS: it takes
tickers or an asset covariance and answers with weights per name. This one
takes the RETURN STREAMS those books produce -- a backtest's, a model
evaluation's, a sleeve the caller runs elsewhere -- and asks how much
capital each deserves. Streams come in as a `returns_panel` reference or
as dated values, and results go out as references, so a stream crosses
into this runtime from `backtest` or `modeling` by value.

---

## 1. The tools

| Tool | What it answers |
|---|---|
| `combine_return_streams` | Which dates do these streams share, and what did each lose to them? Publishes the aligned panel as one `returns_panel` reference |
| `allocate_across_sleeves` | How capital should be split across strategies, sleeves or models -- and how much of that split the sample chose: weights, capital per sleeve, risk contributions, effective N, binding caps, bootstrap weight bands and a stability number |
| `evaluate_allocation_rule` | Whether an allocation rule would have worked out of sample, after drift and rebalancing costs, against equal weight -- and how much of its in-sample look is look-ahead |
| `get_kelly_report` | How hard to lever a stream or a set of streams held together (f* = mu / sigma^2, or Sigma^-1 mu), how uncertain that is, and the growth and drawdown odds at fractions of it |
| `apply_risk_overlay` | What a volatility target, CPPI or TIPP does to a stream's volatility, drawdown and floor when each exposure is decided from the day before |
| `get_strategy_capacity_curve` | How much capital a strategy can take before spread and square-root impact on its own weight changes eat its return: the break-even AUM and the AUM at half the gross Sharpe |
| `attribute_returns_brinson` | Where a portfolio's return over its benchmark came from: Brinson-Fachler allocation, selection and interaction per segment and period, linked so they add up |
| `measure_book_returns` | What a portfolio with deposits and withdrawals returned: the true time-weighted return, Modified Dietz linked across sub-periods, and the money-weighted return (GIPS 2020), with large flows flagged and nothing under a year annualized |

---

## 2. Streams first

Every tool here takes return streams, never tickers. The allocating tools
take `returns` as {stream: {ISO date: return}} or a published
`returns_panel` as `returns_ref` (with `columns` to pick streams), and
fetch nothing. Streams that come from different places go through
`combine_return_streams` first: it takes `streams` -- a `returns_panel`
reference and column, dated values, or a symbol with both ends of its
window, the one door in this runtime that fetches -- keeps the dates every
stream has, reports per stream what it supplied and what it lost before,
after and inside the common window, and publishes the aligned panel at a
`run_id`. Undated lists, a short overlap (naming the stream that limits
it), repeated dates, non-finite values and levels passed as returns are
refused by name. An equity curve is levels, not returns: `convert_reference`
turns it into a `returns_panel` first.

---

## 3. Allocating, and how much of it the sample chose

`allocate_across_sleeves` splits capital by `equal`, `inverse_volatility`,
`risk_parity` (with `risk_budgets`), `hrp`, `min_variance`, `max_sharpe` or
`shrunk_sharpe`, on `estimate_covariance`'s estimators (`ledoit_wolf` by
default); the optimized rules are the exact solver's and match the critical
line. `shrunk_sharpe` shrinks the sleeves' Sharpe ratios toward their pooled
value by James-Stein on the Lo/Mertens standard errors, correlated as the
sleeves are, and takes the tangency on the shrunk means: with equal true
Sharpe ratios it pools them all about a third of the time at eight sleeves,
and at full pooling it is the long-only maximum-diversification portfolio.

Constraints act in order: `max_weight` caps the fully invested rule,
`target_volatility` scales the gross (refused above `max_leverage`, default
1; the rest is cash), and `capacity` in the currency of `capital` caps
last, its excess to cash or water-filled to the sleeves with room. Every
rule's risk is read off the same covariance, so the risk contributions sum
to the ex-ante volatility.

The block bootstrap refits the whole rule on circular resamples of the
dated rows and reports each weight's 5th and 95th percentile and the mean
L1 distance of a resampled allocation from the point one. Read that before
the weights: a max-Sharpe split of a few years of data moves most of its
capital between resamples, while risk parity's hardly moves. The Sharpe
ratio reported is `in_sample_sharpe` and is not a forecast.

---

## 4. Testing the rule, not the allocation

`evaluate_allocation_rule` walks a rule (any of the above, with its
constraints, as `rule`) forward: each fit sees only its trailing
`estimation_window` -- a copy, so a rule cannot read past it -- the weights
drift between refits every `rebalance_every` periods, each refit pays
`cost_bps` per unit of turnover, and uninvested capital earns `cash_rate`.
It scores the rule and a `benchmark_rule` (equal by default) on the same
out-of-sample periods: Sharpe over cash, return, volatility, drawdown,
turnover and cost; the difference of the two Sharpe ratios with a circular
block-bootstrap interval; the look-ahead gap, the same rule fitted once on
the full sample and held minus the walk forward; and the deflated Sharpe
ratio at `n_trials`, which is required -- count every rule and setting
tried.

On eight zero-edge sleeves a max-Sharpe rule fitted on the full sample
scores about 0.7, walked forward about 0.1, and the deflated Sharpe ratio
flags it as often as a 5% test should. Changing every return after a date
changes no weight chosen and no return reported on or before it. With
`run_id` the out-of-sample streams are published as a `returns_panel`
(columns `rule` and `benchmark`) and the chosen weights as a
`weight_panel`.

There is no overlay parameter: a volatility target or CPPI is
`apply_risk_overlay` on the published `rule` column, the same thing because
those overlays are causal on the stream. Nothing here picks the best rule;
choosing among several by this result is a selection the trial count must
include.

---

## 5. Leverage and overlays

### How hard to lever a stream

`get_kelly_report` takes one stream or several, puts them on the dates they
share, and reports the Kelly leverage over a stated `risk_free_rate` (at
which the leverage is financed): f* = mu / sigma^2 for one stream,
Sigma^-1 mu for several. f* is a multiple of equity, the stream counterpart
of the notional `get_position_size` reports for one trade. Beside it are
its standard error 1 / (sigma sqrt(T)) and an interval, and the chance the
estimate is more than twice the truth, Phi(-S sqrt(T)) -- 13% for a Sharpe
of 0.5 over five years, the point at which full Kelly grows no faster than
cash.

Each row of `by_fraction` (quarter, half and full Kelly by default) has the
growth if f* were the truth, the growth to expect when the estimate is held
instead (full Kelly keeps 20% of the optimum at a 0.5 Sharpe over five
years, half Kelly 55%), the realized in-sample growth, the probability of
ever losing a fraction d of wealth, (1 - d)^(2/c - 1) (Browne 1999; Thorp
2006), and the maximum drawdown in sample and over bootstrap paths.
`shrink_factor` is Kan and Zhou's (2007) plug-in theta^2 T / (theta^2 T +
N), the share of the estimate that maximises expected growth, and
`empirical_kelly` is the leverage that would have grown the sample the most
(null when that has no maximum, as in a sample that never lost). A negative
edge is a negative f* -- Kelly shorts the stream -- and says so; it is
never reported as zero. The tool reports a leverage and applies none.

### An overlay decided from the day before

`apply_risk_overlay` runs one of three overlays over a single stream.
`vol_target` holds target / sigma_t of it, sigma_t the EWMA volatility
(RiskMetrics, `ewma_lambda` 0.94) of the returns through t - 1, seeded on
the first `warmup` returns; `cppi` holds `multiplier` times the cushion over
a floor (a fraction of starting wealth, accruing at the cash rate); `tipp`
is CPPI whose floor also ratchets up with the peak. Exposure is capped at
`max_leverage` (1 by default); the part not in the stream earns `cash_rate`
and the part above 1x pays it, and `cost_bps` is charged per unit change in
exposure.

The result has realized against target volatility, return and maximum
drawdown before and after, the exposure's range, turnover and cost, and for
CPPI and TIPP the floor breaches -- a one-period loss worse than
-1/multiplier breaches the floor whatever the cushion (Black & Perold 1992)
-- and whether the overlay ended cash-locked, its exposure zero from a date
to the end because the cushion was gone. Every exposure is decided from
returns before it, so changing the stream after a date changes nothing
reported up to that date. With a `run_id` the overlaid stream is published
as a `returns_panel` and the exposure path as an `analytic_series`. Nothing
here trades or rebalances.

---

## 6. Capacity

`get_strategy_capacity_curve` takes a strategy's gross return stream and
its weight panel -- a `weights_ref` (`weight_panel`) or `weights` as
{instrument: {ISO date: weight}}, fractions of AUM -- and, per instrument,
its average daily volume in currency (`adv`), daily volatility and full
quoted spread in bps, as inline maps keyed by the panel's own instrument
labels. Nothing is fetched. At AUM A a weight change |dw| costs
|dw| (s/2 + k sigma sqrt(A |dw| / ADV)); k is 1 by default, the square-root
law's measured order (Toth et al. 2011; Bouchaud, Bonart, Donier & Gould
2018). A trade is charged on the first return date on or after its weight
date, the first row is bought from cash, and drift between weight dates is
not traded.

The curve has the net return and Sharpe ratio, the cost and its impact
part, and the largest trade's share of a day's volume at each AUM.
`breakeven_aum` is a closed form, ((mean gross - a) / b)^2 with a the mean
spread cost and b the mean impact per sqrt(AUM) -- for one instrument
trading tau every period, (ADV / tau) ((alpha / tau - s/2) / (k sigma))^2 --
and `half_sharpe_aum` is where the net Sharpe is half the gross. `binding`
ranks the instruments by their share of the impact, with each one's
participation at the break-even; past 20% of a day's volume the curve is an
extrapolation and the result says so.

---

## 7. Attribution and a fund's own return

### Where active return came from

`attribute_returns_brinson` takes portfolio and benchmark weights and
returns as {ISO period-end date: {segment: value}} -- segments are any
labels both sides are split by -- and splits each period's active return
R_p - R_b into allocation (w_p - w_b)(r_b - R_b), selection w_b (r_p - r_b)
and interaction (w_p - w_b)(r_p - r_b) per segment (Brinson & Fachler
1985). Each period's weights must sum to one; a cash segment with its return
covers the rest. A segment the portfolio holds none of needs no portfolio
return; every other missing return is refused, because zero is a return.

Single-period effects do not add up over time, because returns compound.
The periods are linked by `linking`: Carino (1999, the default) scales each
period's effects by its log-to-arithmetic ratio over the whole's, Menchero
(2000) by a near-constant coefficient with a least-squares correction;
either way the linked effects sum to the cumulative active return,
prod(1 + R_p) - prod(1 + R_b), to rounding.

### Time-weighted against money-weighted

`measure_book_returns` gives three returns for a portfolio with external
flows, because they answer different questions (GIPS 2020). The true
time-weighted return links the sub-period returns between valuations and is
the manager's result, independent of when money arrived; it needs a
valuation on every flow date, and without one it is null with
`missing_valuation_dates` naming the dates -- it is never approximated.
Modified Dietz is the approximation: each sub-period's gain over its opening
value plus its day-weighted flows, linked geometrically. The money-weighted
return is the IRR of the owner's cash, found by the same root search as
`analyze_cash_flows` and null unless it is unique. On a book that rises
20%, takes a deposit as large as itself at the top, then falls 10%, the TWR
is +8.000% and the MWR -1.2549% a year; `timing_gap` is the difference, the
owner's timing.

Inputs are `valuations` and `flows`, each {ISO date: amount} (a flow
positive for a contribution and negative for a withdrawal). Nothing under a
year is annualized, and a flow above `large_flow_share` of the value before
it (10% by default) is listed with whether it has its own valuation.

---

## 8. What this runtime will not do

- Pick the best rule, window or method. Choosing among several allocation
  rules by their out-of-sample result is itself a selection: state the
  trial count and let the deflated Sharpe ratio see it.
- Size a position or rebalance a book. A Kelly fraction and an overlay are
  reported, never applied.
- Fetch a stream other than through `combine_return_streams`'s symbol form.
- Net sleeves into one book engine: each stream is taken as its own net
  return.

## Related

- [05_portfolio.md](05_portfolio.md) -- weights across assets, and the
  library functions behind the streams.
- [24_overfitting.md](24_overfitting.md) -- the deflated Sharpe ratio, PBO
  and the reality check the walk forward reports beside.
- [32_finance.md](32_finance.md) -- what holding the resulting positions
  costs, and `analyze_cash_flows`, whose root search the money-weighted
  return uses.
- [09_advanced_agent_tools.md](09_advanced_agent_tools.md) -- position
  sizing for one trade and the capacity report for a set of target weights.
