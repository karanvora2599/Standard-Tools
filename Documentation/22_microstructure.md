# Microstructure

Seventeen tools for what the market will charge you to trade, in two halves
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

**The guard runs in the windowed branch too.** With `window=` set it used
to apply only to the full-sample covariance, which that branch never
computed, so a rolling run on live bars reported 59 bp against its own
161 bp detection floor with `significant: null`. The windowed estimate is
now judged on the median window's covariance against a window-sized
standard error, and `significant` is always a verdict.

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

**Overnight gaps bias it DOWN**, and `n_gap_adjusted` counts them. The
derivation assumes the price is continuous between the two days; a bar
that sits entirely above or below the one before it has a two-bar range
inflated by the gap rather than by the spread. That range enters `gamma`,
and `gamma` is *subtracted*, so an unadjusted gap makes the spread look
smaller. The standard adjustment shifts the bar to touch its predecessor
before the range is measured.

The correction is mostly invisible in the headline, which is why it went
missing for so long. Measured on a name gapping 3% every twentieth bar,
31 of 399 pairs gapped and removing the gaps moved `raw_mean_bps` from
-39.80 to -21.85 while `spread_bps` stayed at 37.540011 — a gap large
enough to matter drives that pair's estimate deeply negative, and the
zero-floor then swallows the whole difference. Read `n_gap_adjusted`
against `n_estimates`: above about 5% the answer rests on the adjustment
rather than on the data, and a warning says so.

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

**The sign is the whole question, and from bars alone the estimate is
circular.** Kyle's model is about buyer- versus seller-initiated volume,
which needs trades matched against quotes. From bars the only sign
available is the bar's own return, so `x = sign(y) · V` is regressed on
`y`: lambda is positive by construction and `r_squared` measures nothing.
Measured on live AAPL bars, shuffling the returns and permuting the volume
— destroying every real relationship — left 80% of the estimate intact,
where a non-circular control gave a lambda of zero. The docstring used to
claim the opposite failure (that misclassification *understates* impact),
and the guard against a non-positive lambda protected against something
that could not happen.

The bars path is kept because it is what a daily dataset can offer, and it
now says what it is: the result carries `circular=True` and
`sign_source="return_sign"`, and the warning states that lambda is positive
by construction. Pass a tape instead — `kyle_lambda(trades=..., quotes=...,
freq="1min")` — and the flow is signed by Lee-Ready (99.7% accurate against
the venue's own aggressor flag; the tick rule without quotes), bucketed at
`freq`, and the **midpoint** change is regressed on it, because a last
trade price carries the bid-ask bounce whose sign is also in the signed
volume. That path returns `circular=False`, and on a market with no impact
at all it finds a lambda near zero, which is the test that separates a
measurement from an artefact.

Check `r_squared` before sizing anything: a lambda from a regression
explaining 2% of the variance has a standard error larger than itself — and
on the bars path, do not read it at all.

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

**The residue bucket is gone.** The bucket walk used to append whatever
volume was left after the last full bucket as one more bucket: a float
residue of a single bar, whose imbalance was exactly 1.0 by construction
and which landed *last* — so it dominated `current_vpin`, and 51 buckets
came back for 50 requested. The trailing partial bucket is now dropped and
its size reported as `residual_volume`; `n_buckets` is what was asked for.

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

**The session, not the observed range.** A feed that carries extended
hours — Databento's, whose bars sit on a UTC clock after normalisation —
was bucketed from 4am to 8pm, so the "open" and "close" buckets held the
pre- and post-market trickle: `open_share 0.00004`, `close_share 0.0`,
`u_shaped False`, and a warning that the caller's data was unusual.
Restricted to the regular session the same bars gave 0.234 / 0.153 and
`u_shaped True`. Pass `index_timezone` (`"UTC"` for Databento bars) or a
tz-aware index, and both profiles — the bars-based estimator and the
trades-based `get_trade_profile` — bucket `session` (default 09:30–16:00)
in `exchange_timezone` (default New York) and report the
`extended_hours_share` beside it. A naive index with no `index_timezone` is
taken as already in session time, which is what yfinance bars are.

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

**A repeated timestamp is two rows, not one label.** Real tapes repeat
timestamps — 32% of the prints in a live AAPL minute share one with the
print before — and three functions aligned trades to their signs by
*label*: `effective_spread` and `microstructure_summary` raised ("cannot
reindex on an axis with duplicate labels"), and the liquidity detector's
signed-volume channel fanned rows out by label and reported a net
imbalance of −229,340 on a tape whose whole volume was 64,780. Every
consumer now aligns by position, and the signed volume cannot exceed what
traded. The signing rule itself is unchanged; only the bookkeeping was
wrong.

**`detect_liquidity_events` reports the channel's memory, and isolates a
channel's failure.** The CUSUM threshold is calibrated for i.i.d. noise,
and real channels are not: a spread channel with lag-1 autocorrelation
+0.67 fired on 43% of quiet real windows at the default, where an i.i.d.
control gives 7%. Every channel result now carries `lag1_autocorrelation`,
the `threshold` it was judged against, and `false_alarm_rate_at_threshold`
— how often an AR(1) null with that memory crosses the threshold on a
window this long, simulated — so a detection is read against the rate it
was made at. `calibrate_threshold=True` (on the tool and the library
function) takes the threshold from that null's 95th percentile instead,
which puts the false-alarm rate back at 5% whatever the channel's memory.
And a channel that fails for any reason is reported as `unavailable` with
the exception's name; it used to catch only `ValidationError`, so one
channel's `ValueError` killed all six — and the failing channel was one
the module declared computable, so the obvious call was the one that died.

## The tools

| Tool | Needs ticks | Answers |
|---|:--:|---|
| `classify_trade_direction` | yes | Sign the tape, Lee-Ready or tick rule, published; signs positionally, so a tape whose timestamps repeat (a quarter of a live one) classifies instead of raising |
| `get_quoted_spread_series` | yes | Spread and imbalance per quote, not averaged |
| `get_effective_spread_series` | yes | What each trade paid, optionally split, with the realized and impact means beside the effective one |
| `get_order_book_metrics` | yes | Microprice, imbalance at the touch and cumulatively, and the depth slope -- what a top-of-book quote cannot say; inline snapshots may carry an ISO `timestamp`, which the per-second rates need, and `include_order_counts` returns the orders resting at each level when the feed carries them |
| `get_order_event_metrics` | Queue position, order lifetime, cancels per add and event intensity — from an ORDER feed, which a depth snapshot cannot produce; the lifetime summaries carry their tail (p25 to p99) beside a mean and median that can differ fifty-fold |
| `get_microstructure_metrics` | yes | Quoted and effective spread, realized/impact split, Lee-Ready signed flow |
| `get_trade_profile` | yes | Volume by trade size and time of day |
| `detect_liquidity_events` | yes | When a liquidity regime *changed*, by CUSUM |
| `check_spread_proxy` | yes | How wrong the OHLCV proxy is on this name |
| `estimate_roll_spread` | no | Effective spread from bid-ask bounce — with its noise floor |
| `estimate_corwin_schultz_spread` | no | Spread from the high-low range |
| `get_amihud_illiquidity` | no | Price move per dollar traded, as a percentile; with `run_id` and `name` the rolling series is published as an `analytic_series` |
| `estimate_kyle_lambda` | no | Market depth, and the impact of a given size |
| `get_order_flow_imbalance` | no | Signed volume imbalance, and whether it predicts anything |
| `estimate_vpin` | no | Flow one-sidedness in volume time |
| `get_intraday_volume_profile` | no | The U-shape, for scheduling; takes the venue's `exchange_timezone` and session, since a London tape under the New York session is refused rather than mis-measured |
| `get_implementation_shortfall` | no | What an execution actually cost, decomposed |

Full argument lists:
[20_tool_index.md](20_tool_index.md#microstructure--microstructure).

## What a depth book still cannot tell you

Everything above reads a BOOK: snapshots of aggregated size per price level.
That aggregation is the ceiling. A book showing 5,000 shares at the bid
cannot say

- whether that is one order or two hundred,
- which of them arrived first, or
- whether size that disappeared was **cancelled** or **filled** — and those
  mean opposite things about who wanted to trade.

`get_order_event_metrics` reads an ORDER feed instead: every add, cancel,
modify and fill, each with the venue's own `order_id`. `DatabentoProvider`
serves it through `get_order_events` (market-by-order), and it is a strictly
deeper feed than `get_order_book`.

| Measure | Why a book cannot produce it |
| --- | --- |
| **Queue ahead** | Resting size at an order's own price level when it arrives — the number that decides whether a passive order fills. Depth gives the level total and cannot say how much is in front of you |
| **Order lifetime** | Time from add to cancel or fill. No snapshot equivalent exists at all |
| **Cancel-to-add, cancel-to-trade** | A snapshot sees size vanish and cannot tell a cancel from a fill. A trade is counted ONCE: the `T` prints when the feed carries them, else the fills — an execution is a `T` and an `F` for the same event, and counting both counted every trade twice |
| **Event intensity by action** | A snapshot stream measures the SAMPLING rate when sampled and the update rate when not, and nothing in the frame says which |

### Censoring is counted, not folded in

An order already resting when the window opened has no add in it. Its true
lifetime is longer than anything the window can see, so it is counted
separately (`terminated_without_an_add`) and **excluded** from the lifetime
averages. Folding it in as the time since the window started would drag
every average downward — worst for exactly the long-resting orders a queue
study is about. Orders still open at the close are reported as
`still_resting` for the same reason.

A `CLEAR` wipes the book, so the queue accumulators reset on one rather than
carrying depth across a boundary where none existed. A `MODIFY` is counted
but does not adjust queue depth: whether it loses priority depends on the
venue's own rule, and guessing would be worse than saying so.

### A snapshot is the book, not an event

A window that opens with a snapshot — the vendor's flag bit 32, or a
`snapshot` column the caller supplies — carries the orders already resting
as `ADD` records that repeat state rather than report a change. Reading
them as events got three numbers wrong at once on a live CME reopen: the
resting orders were counted as arrivals, so `events_per_second` came out
16,000× too high on a snapshot-bearing window; a later cancel of one of
them was "terminated without an add", which was 54.5% of that count; and
dropping them instead understated the queue ahead of a real arrival by
33–79% against the `mbp-10` book for the same sequence numbers.

Snapshot records now seed the queue accumulators without counting as
arrivals (`n_snapshot_orders`), explain a later cancel or fill
(`terminated_from_snapshot`, with `resting_at_open` for the ones the
window saw resting) instead of leaving it censored, and are excluded from
the event counts, the rates and the clock (`n_snapshot_events`). The
reference path keeps `flags` and `snapshot` when the registered panel
carries them, so a Databento extract's snapshot bit reaches the metrics.

### Size

Market-by-order is one record per order event. An active name produces
millions in a session where mbp-10 produces thousands, so the reference path
is the normal one — pull it once, write it, and register it as an
`sqt://order_event_panel` rather than re-fetching a metered feed.


## Related

- [05_portfolio.md](05_portfolio.md) — sizing and the cost models that predict
- [19_runtimes.md](19_runtimes.md) — why this left the portfolio runtime
