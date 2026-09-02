# Microstructure

Sixteen tools for what the market will charge you to trade, in two halves
that answer the same question at two data fidelities — plus three that
publish the intermediate series the summary tools used to discard.

**Four MEASURE** from trades and quotes, and refuse to run without a tick
feed. **Eight ESTIMATE** the same quantities from OHLCV bars — which is the
normal case, because most environments have no tick data — and each one
says what it is a proxy *for* and how it fails.

## Why the refusal matters

The tick tools do not fall back to bars. A quoted spread is a quoted spread,
and nothing computed from a daily OHLCV row is one; approximating it would
produce a number that every downstream tool would then treat as a
measurement.

That refusal was right when there were four tools here and it left a gap:
"no tick data" is the normal situation and it does not make the questions go
away. The eight bar-based estimators fill it *without pretending* — they are
named for what they are (`estimate_roll_spread`, not `get_spread`), they
return a proxy with its failure modes attached, and `check_spread_proxy`
exists to measure the proxy's error on a specific name when both feeds are
available.

## The finding that shaped this module

**Roll's estimator returns a spread when there is none.**

Roll (1984) infers the effective spread from the negative serial covariance
of price changes: bid-ask bounce makes consecutive returns mean-revert, and
the size of the reversal is the spread. It needs no quotes at all.

Measured on a simulated random walk with a spread of **exactly zero** and 1%
daily volatility, it returned **0.098 on a $100 stock** — a confident-looking
10 basis points conjured entirely from sampling noise.

Two things produce it:

1. The lag-1 autocovariance has a standard error of `var(Δp)/√n`, which
   swamps `−(s/2)²` whenever the spread is small relative to volatility —
   which is to say, on every liquid name.
2. The estimator only takes a square root when the covariance lands
   *negative*. The positive half of the noise is silently discarded, so what
   survives is biased upward.

Nothing in Roll's algebra reveals either. So the result now reports
`smallest_detectable_spread` and a `significant` flag:

| Planted spread | Estimate | Noise floor | Significant |
|---:|---:|---:|:--|
| 0.00 | 0.098 | 0.289 | no |
| 0.02 | 0.107 | 0.289 | no |
| 0.10 | 0.166 | 0.291 | no |
| 0.50 | 0.563 | 0.327 | yes |
| 1.00 | 1.082 | 0.421 | yes |
| 2.00 | 2.125 | 0.674 | yes |

Above the floor it is accurate to within 13% — the 0.50 row is the worst
of them, and the error falls as the spread grows. Below the floor, any
number the formula returns is noise that happened to land on the negative
side.

**On a trending series it returns `null`, not zero.** The literature's usual
fix — substitute zero when the covariance is positive — produces a tidy
series with a systematic downward bias, and the zeros cluster in exactly the
trending periods where liquidity is most interesting. "We could not measure
it" and "it was zero" are different facts and only one of them is true.

## The bar-based estimators

### `estimate_corwin_schultz_spread`

A day's high-low range contains both volatility and the spread. Volatility
scales with the square root of time and the spread does not — so comparing a
one-day range against a two-day range identifies them separately, with no
quote data at all.

**It produces negative estimates routinely**, on 29–44% of days in the measurements below, as a
sampling artefact. Corwin and Schultz recommend flooring those at zero, and
that is done — *and reported*, because flooring turns a symmetric error into
a one-sided bias. Measured:

| Planted spread | Estimate | Negative fraction |
|---:|---:|---:|
| 20 bps | 56 bps | 44% |
| 50 bps | 72 bps | 40% |
| 100 bps | 103 bps | 29% |

`negative_fraction` is what separates the accurate case from the useless
one, which is why it is returned rather than swallowed. Above about a third,
read the average as noise.

### `get_amihud_illiquidity`

`|return| / dollar volume` — how far the price moves to absorb a dollar. The
most widely used liquidity proxy in the academic literature, largely because
it needs nothing but daily bars.

**The raw number is uninterpretable.** Its units are return-per-dollar, so
it scales inversely with dollar volume: a large-cap's reading is a thousand
times smaller than a microcap's and neither means anything alone. The result
therefore leads with the **percentile** of the current reading within the
name's own history, and returns the raw value second.

It is **not a spread**. It conflates the spread, the depth of the book and
the information content of trades, and cannot separate them — a genuinely
volatile stock scores as illiquid here even with a deep book.

### `estimate_kyle_lambda`

Market depth: the regression of price change on signed order flow. The one
measure here with a direct trading interpretation — multiply lambda by the
size you intend to trade and you have an estimate of the impact you will
cause.

**The signing is the weak link.** Kyle's model is about buyer- versus
seller-initiated volume, which needs trades matched against quotes. From
bars, the sign of the day's return stands in. The tick rule agrees with the
true classification about 85% of the time on a liquid name and materially
worse on an illiquid one — and illiquid names are where lambda matters.
Misclassification attenuates the slope toward zero, so this **understates**
impact, and understates it most exactly where impact is largest.

Check `r_squared` before sizing anything: a lambda from a regression
explaining 2% of the variance has a standard error larger than itself.

### `get_order_flow_imbalance`

Signed volume imbalance, with its own predictive test attached rather than
presented as a signal.

**Persistence is measured on non-overlapping windows**, and the reason is a
bug this tool had. A rolling sum at `window=5` shares four of its five
observations with the previous point, so its lag-1 autocorrelation is about
`1 − 1/w` whatever the data does. Measured on pure noise:

| Window | Overlapping (artefact) | Predicted `1 − 1/w` | Non-overlapping (truth) |
|---:|---:|---:|---:|
| 5 | +0.762 | +0.80 | +0.077 |
| 10 | +0.885 | +0.90 | +0.184 |
| 21 | +0.957 | +0.95 | +0.041 |

The overlapping figure describes the *window*, not the flow. It is still
returned as `overlapping_persistence` so the difference is visible rather
than assumed away.

### `estimate_vpin`

Flow one-sidedness measured in **volume time** rather than clock time —
information arrives with volume, so the series is cut into equal-volume
buckets rather than equal-time bars.

Two caveats are attached to every result, and both matter:

- **This is not the VPIN of the paper.** The original is a trade-level
  measure where each bucket holds hundreds of trades and bulk classification
  has something to work with. Built from daily bars with tick-rule signing,
  what comes back is a defensible series of flow one-sidedness and not that.
- **VPIN is contested.** Andersen and Bondarenko (2014) argue it is largely
  a transformation of volatility and that the flash-crash result depends on
  sample construction. What is not disputed is that it measures
  one-sidedness of flow; calling that "informed trading" is a model
  assumption, not a measurement.

### `get_intraday_volume_profile`

The U-shape every execution schedule is built on: volume concentrates at the
open and close, with a midday trough routinely a third of the opening
bucket. A schedule spread evenly across the **clock** over-participates at
lunch — paying impact into a thin book — and under-participates at the
close, missing the cheapest liquidity of the day.

Needs intraday bars with timestamps. **Daily bars are refused** rather than
aggregated into a meaningless single bucket, and the closing bucket's share
is flagged separately because closing-auction volume has risen for a decade
on index flows, so a profile fitted over several years understates today's.

### `get_implementation_shortfall`

Every other cost tool in this library is a model run *before* the fact —
`estimate_trade_cost` predicts, `get_capacity_report` bounds,
`plan_rebalance` schedules. This is the measurement, and it is what those
models should be checked against.

The Perold decomposition splits the gap between the decision price and what
was achieved into four parts, and the separation is the point:

| Component | What it is | Who owns it |
|---|---|---|
| **Delay** | Price moved before the order reached the market | Workflow. No algorithm recovers it, and it is frequently the largest term. |
| **Impact** | Price moved while the order worked | The execution algorithm |
| **Opportunity** | Shares never filled, priced at the close | An algorithm that beats VWAP by not completing has moved its cost here, not saved it |
| **Fees** | Commission | Known; separated so it neither flatters nor contaminates the measured parts |

**Positive is a cost.** Both sign conventions exist in the wild and it is
the first thing misread.

**The decision price is an input, not an inference**, because only the caller
knows it. Passing the arrival price for both — which is common, because it
is easy — sets the delay cost to zero *by construction*, and the result says
so when it detects it.

## What none of them do

Measure the spread you will actually pay. Every estimator here produces a
historical average under a model. The cost at the moment you send an order
depends on the book at that moment, and no daily bar contains it.

## The series the summary tools used to throw away

`get_microstructure_metrics` signs the tape, computes a spread per trade,
splits the effective spread into its realized and impact halves — and then
returns averages. The per-trade and per-quote series it built along the way
is what an event study, a CUSUM detector or a model's features would
actually consume, and it died inside the call.

Three tools publish it instead. They became worth adding only once a tape
could be fetched and handed around, which is what `fetch_tick_tape` in the
`data` runtime now does:

```
fetch_tick_tape  ──┐
                   ├──> classify_trade_direction ──> sqt://tick_tape/...
fetch_quote_panel ─┘                                        │
                                                            v
                                              event study / CUSUM /
                                              a model's features
```

**`classify_trade_direction` says which rule it used, and that matters
more than it sounds.** With a quote panel it is Lee-Ready, matching each
trade against the quote *preceding* it. Without one it falls back to the
tick rule, which agrees with the true classification about 85% of the time
on a liquid name and materially worse on an illiquid one. Every downstream
estimate inherits that error, and misclassification attenuates toward zero
— so the weaker rule makes an edge look smaller, not noisier.

**`get_effective_spread_series` without `realized_horizon_seconds` gives
you one number where there are two.** The realized half is what the
liquidity provider kept; the impact half is what the trade moved. They
imply opposite remedies — impact says trade smaller, realized says trade
somewhere else — and unsplit, neither is visible.

## The tools

| Tool | Needs ticks | Answers |
|---|:--:|---|
| `classify_trade_direction` | yes | Sign the tape, Lee-Ready or tick rule, published |
| `get_quoted_spread_series` | yes | Spread and imbalance per quote, not averaged |
| `get_effective_spread_series` | yes | What each trade paid, optionally split |
| `get_microstructure_metrics` | yes | Quoted and effective spread, realized/impact split, Lee-Ready signed flow |
| `get_trade_profile` | yes | Volume by trade size and time of day |
| `detect_liquidity_events` | yes | When a liquidity regime *changed*, by CUSUM |
| `check_spread_proxy` | yes | How wrong the OHLCV proxy is on this name |
| `estimate_roll_spread` | no | Effective spread from bid-ask bounce — with its noise floor |
| `estimate_corwin_schultz_spread` | no | Spread from the high-low range |
| `get_amihud_illiquidity` | no | Price move per dollar traded, as a percentile |
| `estimate_kyle_lambda` | no | Market depth, and the impact of a given size |
| `get_order_flow_imbalance` | no | Signed volume imbalance, and whether it predicts anything |
| `estimate_vpin` | no | Flow one-sidedness in volume time |
| `get_intraday_volume_profile` | no | The U-shape, for scheduling |
| `get_implementation_shortfall` | no | What an execution actually cost, decomposed |

Full argument lists:
[20_tool_index.md](20_tool_index.md#microstructure--microstructure).

## Related

- [05_portfolio.md](05_portfolio.md) — sizing and the cost models that predict
- [19_runtimes.md](19_runtimes.md) — why this left the portfolio runtime
