# Delta One

Seventeen tools for the instruments that move one-for-one with an underlying —
cash, ETFs, baskets, futures, forwards and total return swaps. The runtime
answers one question that no other runtime could: **which instrument is the
cheapest way to own or hedge this exposure, and why do they differ.**

It is not `derivatives`. That runtime prices ONE convex contract and asks
what holding it does to you. This one prices the RELATIONSHIP between
several linear ones. The two meet at exactly one place, and it is a
feature: put-call parity produces a synthetic forward, and that forward is
one row in `compare_delta_one_expressions`.

---

## 1. What was already here, and what was missing

About two thirds of the mathematics existed before this runtime. The carry
equation `F = S·e^(r−q−b)T` has been in `analysis/derivatives.py` for a
long time, with financing, dividend and borrow deliberately kept apart.
Beta, rolling beta, factor regression, Ledoit-Wolf covariance and the
itemized trade-cost model were all present.

What was missing was the layer that connects instruments to each other:

```
        Research              Portfolio
       statistics              sizing
       regression              risk
            │                     │
            └────────┐   ┌────────┘
                     ▼   ▼
                  DELTA ONE
                     ▲   ▲
            ┌────────┘   └────────┐
       Derivatives          Microstructure
          carry                spreads
          forward              impact
```

Three things genuinely did not exist anywhere and had to be written:

| Missing | Now at |
|---|---|
| A year fraction. No `year_fraction`, no day-count convention, five inline `/365.0` sites | `delta_one/daycount.py` |
| Futures contract semantics — multiplier, tick value, expiry, settlement | `delta_one/contracts.py` |
| `tracking_error` as a function (it was a local variable inside `information_ratio`) | `delta_one/hedging.py` |

## 2. The tools

| Tool | Answers |
|---|---|
| `analyze_cash_futures_basis` | Is this future rich, and which carry component explains it |
| `solve_forward_carry` | What financing / dividend / borrow does this quote imply |
| `analyze_basis_history` | Is this basis wide *for this name* |
| `analyze_futures_curve` | What does the term structure look like, and what does a calendar spread price |
| `analyze_roll` | What does moving this position to the next contract cost |
| `size_futures_hedge` | How many contracts, and what does rounding leave behind |
| `analyze_hedge_effectiveness` | Did that hedge actually work |
| `analyze_index_basket` | Is this basket rich to its index, and which name explains it |
| `compare_delta_one_expressions` | Which of these six ways of holding it is cheapest |
| `optimize_replication_basket` | What is the smallest basket that tracks this |
| `analyze_etf_fair_value` | Is this ETF premium real after costs |
| `price_total_return_swap` | What is this swap worth, leg by leg |
| `analyze_total_return_future` | What financing spread does this TRF embed |
| `analyze_dividend_points` | How many index points of dividend before expiry |
| `analyze_index_rebalance` | What will this index change force people to trade |
| `detect_basis_dislocation` | Has this basis *structurally shifted*, or just moved |
| `monitor_spread_stream` | Watch any spread on a live feed, one stateful call at a time |

The first nine shipped alone, deliberately: the floor for a runtime is
eight, shipping exactly at it means one tool failing review makes the whole
runtime unshippable, and the nine needed nothing the library did not
already have. The six added second each needed something new -- a
constrained optimizer, a day-count convention, an index divisor -- and
holding them back got the runtime into use sooner.

## 3. Three things this surface gets wrong if you let it

### A wide basis is usually not an arbitrage

In descending order of frequency: a spot print that is not simultaneous
with the future, a wrong dividend assumption, a borrow that has moved,
expensive funding, and only then something tradeable.

This is why `analyze_cash_futures_basis` returns `implied_financing_rate`
rather than stopping at the mispricing. A future 40 bps rich against a
correct dividend is usually telling you what funding costs, and comparing
that number to SOFR settles it in one step.

### Points and basis points are not interchangeable

A basis of 41 points on a March contract and 62 on a December one is not a
comparison — they carry different amounts of time. The annualized rate is
comparable; the points figure is what the screen shows. Every tool here
returns both and labels which is which.

### The horizon decides which instrument wins

This is the reason `compare_delta_one_expressions` exists. Carry accrues
per year; execution is paid once. The same four expressions, priced on
identical inputs, at two horizons:

| | 1 month | 2 years |
|---|---:|---:|
| Cash basket | 371 bps | **279 bps** |
| SPY ETF | **301 bps** | 289 bps |
| TRS | 302 bps | 290 bps |
| ES futures | 302 bps | 295 bps |

The ranking fully reverses. At one month the cash basket's 4 bp one-way
execution amortizes to 96 bps a year and it is the worst choice; at two
years execution has vanished into the horizon and it is the best. Nothing
about the instruments changed — only how long they are held.

An answer from this tool that does not state its horizon is not an answer.

## 4. Nothing here fetches

Curves arrive as lists of contracts, baskets as lists of constituents,
financing as a number. This library has no futures data provider, no
index-constituent source and no dividend calendar, and a tool that
pretended otherwise would compute a curve that does not exist.

That is the same call the derivatives runtime made about option chains,
and it has the same side benefit: every tool here works on a hypothetical
curve, which is most of what they are used for.

The one thing that does reach a real provider is Bloomberg's identifier
pass-through — `"ESZ5 Index"` and `"SPX Index"` survive normalization
untouched. Note that the match is **case-sensitive**: `"SPX INDEX"`
silently becomes `"SPX INDEX US Equity"`, and the timezone metadata reports
`America/New_York` for a CME contract.

## 5. What is not here yet

Deliberately deferred, in the order they are worth building:

Nothing on the original roadmap is now deferred. What remains is not a
missing tool but a missing *source*: no provider shipped here serves L2 or
intraday futures. Everything that consumes them exists and is tested
against synthetic books, which was the sequencing
`DataProvider.get_order_book` chose on purpose — see §7.

## 6. The infrastructure underneath

Two tools landed outside this runtime, in the runtimes that own them:

| Tool | Runtime | Why there |
|---|---|---|
| `build_continuous_futures_series` | `data` | It produces references other runtimes read |
| `run_futures_backtest` | `backtest` | Delta One computes economics; backtest simulates them |

**A back-adjusted continuous future is not a price.** Adjustment changes
every historical level — a difference-adjusted series can go negative on a
contract that never traded below zero — so
`build_continuous_futures_series` publishes **two** references: an adjusted
`research_ref` for signals, and a `tradeable_ref` carrying which contract
was actually active on each date and what it actually traded at. Size
positions from the second. Collapsing the two is the error the tool exists
to make impossible.

**A futures account keeps different books.** The shared-cash engine rests on
`position value == shares × price == cash paid`, and a future breaks all
three: it costs margin rather than notional, its profit arrives as daily
variation margin credited to cash, and once credited the contract is worth
zero again. So `run_futures_backtest` reports equity as cash plus posted
margin, and its `max_leverage` is **economic exposure over equity** — not
the gross-market-value ratio the equity engine reports. A futures book is
at zero on that definition and many times its equity on this one, which is
why a limit written against one and measured against the other is how a
flat-looking book turns out not to be.

---

## Related

- [21_derivatives.md](21_derivatives.md) — options, greeks, and the
  put-call parity that produces a synthetic forward
- [05_portfolio.md](05_portfolio.md) — sizing and the itemized trade-cost
  model these tools compose
- [22_microstructure.md](22_microstructure.md) — whether the arbitrage
  survives execution
- [20_tool_index.md#delta_one--delta-one](20_tool_index.md#delta_one--delta-one)
  — generated argument-level reference
- [19_runtimes.md](19_runtimes.md) — why runtimes exist and how values
  cross between them

## 7. The live layer

`DataProvider.get_order_book` declared its column contract before any
provider implemented it, and said why: *"the analysis that consumes a book
(microprice, order-flow imbalance, depth slope) can be written and tested
against synthetic books now, so that when a source arrives the
correctness-critical part already exists rather than being invented under
deadline."* That analysis now exists.

### A depth book says what a quote cannot

`get_order_book_metrics` (in `microstructure`) reads that contract and
nothing else, so any feed shaped to it works — including one this library
has no provider for.

The midpoint ignores size, so a book with 5,000 bid and 100 offered reads
identically to its mirror, and the second is about to trade higher. The
**microprice** weights each side by the *opposite* side's resting size,
which reads backwards until you see why: the heavy side is the side that
absorbs, so price is pinned nearer the thin one. On a balanced book it
equals the mid exactly.

Touch and cumulative imbalance are both reported because they routinely
disagree and answer different questions — the touch predicts the next tick,
the cumulative predicts where *size* ends up. A book bid at the touch with
weight behind the offer is exactly the one that ticks up and fills badly.

### One monitor, three channels, five jobs

The roadmap asked for five monitors — live basis, ETF NAV, index arbitrage,
roll spread, and a generic cross-instrument spread. They are not five
computations. Four are `(a/b − 1)` in basis points and differ only in what
the legs are *called*; the fifth is a difference in points. Five tools for
three formulas would mint near-identical names for one rearrangement —
exactly what `solve_forward_carry` exists to avoid.

So the **channel** says how the legs combine, the **label** says what they
are:

| channel | value | serves |
|---|---|---|
| `relative_bps` | `(primary/reference − 1) × 10⁴` | live basis, ETF NAV, index arbitrage, any cross-instrument spread |
| `annualized_bps` | `ln(primary/reference)/T × 10⁴` | basis across more than one expiry — `relative_bps` *steps* at a roll and the detector will report the step |
| `absolute_points` | `primary − reference` | roll spread, and anything the market quotes in points |

One sign convention throughout: **positive means `primary` is dear to
`reference`.** Future over spot, ETF over NAV, basket over index, next
contract over front.

`absolute_points` is not a stylistic alternative to `relative_bps`. A
31-point calendar spread on a 6000 index is 51 bps — expressing a roll in
bps compresses the whole signal, and a difference needs no positive
denominator where a ratio does.

### A monitor cannot be a subscription

A tool call is asked a question and answers; there is nowhere for a
long-running loop to live between calls. So `monitor_spread_stream` returns
its state, and you pass it back:

```
state = None
while True:
    result = monitor_spread_stream(
        primary_prices=…, reference_prices=…, channel="relative_bps", state=state
    )
    state = result["state"]
    if result["alert"]:
        ...
```

Two properties fall out of that shape which a subscription does not have:
the state is inspectable at every step, and a monitor can be paused,
serialized, moved to another process and resumed without losing its
baseline.

**Accumulators are carried, not recomputed.** Feeding a hundred ticks in one
call and making a hundred single-tick calls reach byte-identical state, so
call frequency is a deployment decision rather than a modelling one — and
cost is constant per tick rather than a pass over a growing history.

**The baseline freezes after warm-up, deliberately.** A monitor that keeps
updating its own idea of normal adapts to the dislocation it is meant to
report and goes quiet exactly when it matters. The consequence is stated
rather than hidden: after a genuine regime change the monitor stays
triggered until reset, because by its own baseline the world is still
abnormal. Whether that is a spike to acknowledge or a new level to watch
from are opposite conclusions that look identical in the accumulators, so
`reset` asks which rather than guessing.
