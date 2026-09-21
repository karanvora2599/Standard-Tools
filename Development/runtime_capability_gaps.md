# The rest of the library — dead and unreachable capability

**Date:** 2026-09-21
**Companion to:** `modeling_capability_gaps.md` (which covered `modeling/` and the
feature lab). This one covers **everything else**: research, backtest, data,
delta one, derivatives, microstructure, portfolio, and the audit/meta layer.

Same question as before: which functions exist, work, and cannot be reached —
and which of them are worth turning into tools. Defects found along the way are
in section 7 because discarding them would be worse, but they are a by-product.

---

## 0. Scope and method

The agent-visible surface here is **211 tools across 10 runtimes**, which I
enumerated and checked for structural soundness before anything else:

| runtime | tools | schema cost |
|---|---:|---:|
| research | 42 | ~14.0k tokens |
| backtest | 35 | ~23.7k |
| modeling | 22 | ~24.8k |
| data | 18 | ~7.1k |
| delta one | 18 | ~10.7k |
| portfolio | 18 | ~8.9k |
| microstructure | 17 | ~6.7k |
| derivatives | 12 | ~5.0k |
| meta | 20 | ~4.9k |
| feature lab | 9 | ~8.8k |
| **total** | **211** | **~114.6k tokens** |

**The runtime layer itself is clean.** No tool appears in two runtimes, no tool
fails to resolve to an owner, and every runtime's dispatch-table size matches
its schema count exactly (10/10). That number on the right is worth keeping in
view: the full surface is ~115k tokens of schema, which is the real reason a
ranked list matters more than a long one. "Just add a tool" is never free.

Eight parallel sweeps ran against live data, one per area. Every load-bearing
claim below was then **re-verified personally**; where my numbers differ from a
sweep's, mine are the ones quoted and the difference is noted.

The corrected reachability scan over the domain packages:

```
domain modules scanned       : 114
public definitions           : 400
  touched by the tool layer  : 281  (70%)
  referenced, never by a tool:  73
  referenced by NOTHING      :  46
    ...and untested too      :  13
```

**70%, against 29% for `modeling/`.** That ratio is the headline, and it is why
this document is shorter than its companion.

---

## 1. Headline

**This half of the library is well exposed, and there is almost no dead code.
The failure mode here is not death — it is collapse.**

Three findings, in order of how much they change the picture:

1. **The static scan was wrong nearly every time it fired.** Out of the specific
   "dead" names it flagged across eight areas, the sweeps confirmed **almost
   none**. The data layer's six flagged names were **0-for-6**. The
   microstructure area's two were both false positives. The delta one area's
   "2 of 3 daycount symbols unreached" was wrong — all three are live. My own
   two cross-cutting suspicions were also false alarms (below). A dead-code scan
   is simply the wrong instrument for this question, and this is now the second
   document in a row to conclude that.

2. **The dominant real pattern is a series computed and collapsed to a scalar.**
   It repeats in every single area, independently found by eight sweeps that
   could not see each other's work:
   - every backtest engine builds four-to-five daily state curves and ships one;
   - `rolling_sharpe_stability` computes 623 rolling values and ships four
     summary numbers;
   - `kalman_hedge_ratio` builds an 875×4 path and ships six scalars;
   - `pca_returns` computes the PC score series and has no field for it;
   - `amihud_illiquidity` computes a 270-point rolling series and discards it;
   - `basis_history` computes three N-length series and returns one scalar each;
   - `book_metrics` collapses every per-snapshot array to its mean;
   - `bootstrap_statistic` draws 2000 samples and returns an interval.

   Verified personally: across **99 result models carrying 724 distinct declared
   field names**, there is **no output field anywhere** for `cash_curve`,
   `leverage_curve`, `margin_curve`, `position_curve` or `portfolio_returns`.
   The only net-exposure fields in the entire library are the scalars
   `avg_net_exposure` and `avg_gross_exposure`.

3. **Two live tools return an answer that is wrong and say it is fine.** These
   are not gaps, and they outrank every proposal in this document. Both are in
   section 3 as G1 and G2.

---

## 2. The genuinely dead list

Short, and shorter than it looks — most entries are correctly dead.

| symbol | works? | verdict |
|---|---|---|
| `data/base.py:167` + `databento_provider.py:905` `get_order_book` | yes — 3,208 mbp-10 snapshots pulled in 3.8 s | **expose** — G4. The only L2 implementation in the library. |
| `data/base.py:347` + `databento_provider.py:949` `get_order_events` | yes — 1,174 MBO events in 4.9 s | **expose** — G4 |
| `delta_one/contracts.py` `ContractSpec`, `SETTLEMENT_TYPES` | yes, all methods correct | **delete or use** — 177 lines, zero production callers. There is no shipped multiplier registry, so a tool here would answer a question this library cannot source. |
| `liquidity_events.py:351 available_channels` | yes | **delete** — the catalogue already reaches the agent twice (in a field description and in the unknown-channel error) |
| `data/bundle.py:115 DataBundle.frame` | yes | leave — in-process accessor; everything its PIT gate would say already crosses via `describe_data_bundle` + `validate_data_bundle` |
| `data/_cache.py:442 dead_generations` | yes — ran it, 0 of 37 files | **expose the count, not the deletion** — M-cache |

**That is the whole list.** Six entries, of which two are worth building on, two
are worth deleting, and two should stay as they are.

### My own two suspicions, both false alarms

Recorded because correcting a premise is worth as much as confirming one.

- **`run_portfolio_optimization` looked to be missing an `objective` axis.** It
  isn't. `method` is a *superset* of the module's `_OBJECTIVES`, adding
  `black_litterman` and `risk_parity`, and `target_return`/`target_volatility`
  are both present as fields. Independently confirmed by the portfolio sweep:
  there is no CVaR, CDaR, Kelly or robust objective in `optimize.py` **to**
  expose.
- **All eight registered strategies looked tool-less.** They aren't. Four have a
  dedicated `run_*_backtest` alias, and all eight are reachable by string key
  from seven different tools. `run_sma_backtest` and its three siblings are four
  aliases over one input model whose `strategy_type` Literal carries all ten
  values — a good pattern, not a gap.
- **Day-count conventions looked unreachable.** All four are selectable on
  `price_total_return_swap.day_count`. What *is* unreachable is the prose — see
  G7.

---

## 3. The arsenal — ranked

### G1 · `get_implied_volatility` reports a ceiling as an estimate — **fix first**

**HIGH, and it is a wrong answer, not a gap.** A deep-ITM call priced at its
no-arbitrage lower bound has no identifiable volatility: every σ at or below the
true one reproduces that price. The library knows this and returns
`at_bound: True`. The tool drops the field.

Verified personally. S=100, K=40, T=0.25, r=5%, priced at a true σ = **0.05**:

```
price                       : 60.4968879802
no-arb lower bound          : 60.4968879802   (equal to 1e-9)
LIBRARY returns: {'implied_volatility': 0.3491511836, 'converged': True,
                  'iterations': 29, 'method': 'bisection',
                  'price_error': 1.0e-07, 'at_bound': True}
TOOL    returns: {'implied_volatility': 0.349151,     'converged': True,
                  'iterations': 29, 'method': 'bisection'}
```

The agent is handed **7.0× the true volatility, flagged `converged: True`**,
with the two fields that would have said so discarded at the boundary.
`options.py`'s own docstring: *"a price AT the lower bound is reported with
`at_bound=True`: the number is a ceiling, not an estimate."*

**Fix:** add `at_bound` and `price_error` to the result model, plus a warning
when `at_bound`. **Cost: zero** — both values are already in the dict the
handler receives.

### G2 · GARCH reports `converged: True` on measurably misspecified fits

**HIGH, same class as G1.** `run_garch_volatility_forecast` fits GARCH(1,1),
computes the conditional-variance series, uses its last element, and discards
the rest. The standardized residuals are one line away and never computed — so a
misspecified fit is indistinguishable from a good one.

Verified personally on 875 bars, 2022-01-03 → 2025-07-01. The squared-residual
Ljung-Box is the check for whether the model removed the volatility clustering
it was fitted to remove:

| name | `converged` | LB(z) p | **LB(z²) p** |
|---|---|---|---|
| SPY | `True` | 0.0007 | **0.0000** |
| AAPL | `True` | 0.0300 | **0.0001** |
| XLE | `True` | 0.1618 | **0.0000** |
| TLT | `True` | 0.0032 | **0.0001** |

**All four fail decisively; all four report converged.** The library returns 12
keys — `omega`, `alpha`, `beta`, `persistence`, AIC, BIC — and **not one
residual diagnostic**. (The sweep that found this measured 3 of 6 names failing
on its own sample; mine is the stronger result and the one quoted.)

**Fix:** add `ljung_box_p`, `ljung_box_squared_p`, residual skew/kurtosis, and a
`conditional_vol_ref`. `ljung_box` is already imported into that runtime.
**Cost: +1.0 ms on a 4.3 ms fit.**

### G3 · The portfolio state curves — the same gap in five engines

**HIGH, and the highest-volume find in the sweep.** Every backtest engine
computes a family of daily state series and ships only the equity curve.

The sharpest instance, measured: a **dollar-neutral** long/short book on $100k.
Net exposure drifted from **−$12,781 to +$7,622** (−12.8% to +7.6%) and leverage
ranged 1.785 to 2.241. `make_dollar_neutral` is a tool **input**; there is no
tool **output** — not one field — that says whether neutrality held.

Futures is worse: a run that ended −34.6% with peak exposure $793,824 reported
`n_margin_calls: 0`, and the agent cannot distinguish "comfortably margined"
from "one tick away all quarter", because `margin_curve` and `position_curve`
are built and dropped.

**Fix:** `get_portfolio_state_curves` returning `cash`, `gross_exposure`,
`net_exposure` and `leverage` as `sqt://` refs (that plumbing already works),
plus `net_exposure_min/max/mean` as scalars so the common question needs no
dereference. Same for futures with `margin_curve`, `position_curve` and
`min_margin_cushion`. **Cost: zero marginal compute** — the engine builds all of
them today; a full simulation is 3.7 ms.

Ship `capped` with it: the engine's rebalance log records **which tickers** hit
the ADV cap (`['SPY','QQQ','IWM']` in a forced test) and the boundary emits only
`n_capped: 3`. One list field; the accumulator already exists.

### G4 · Depth and order-by-order data have no door

**HIGH.** `DatabentoProvider.get_order_book` (mbp-10) and `get_order_events`
(MBO) are implemented, tested, and have **no call path anywhere**.

I checked this carefully because a naive grep suggests otherwise — there are 26
matches for `get_order_book` in `src/`. **Every one outside the definitions is
prose**: docstrings in `data/tools.py` and `microstructure/book_tools.py`
describing the column contract, and refusal text. There is no `fetch_order_book`
tool; the only book-shaped tools are `get_order_book_metrics` and
`get_order_event_metrics`, which analyse a book you already have.

So **872 lines of microstructure analytics have no agent-reachable data
source** — the two tools that consume a book can only be fed by
`register_external_dataset` pointing at a file the agent has no way to create.

The `data` runtime's docstring argues the omission deliberately: *"a fetch tool
here would have to answer for every provider."* That argument is answerable —
`fetch_tick_tape` and `fetch_quote_panel` already answer for every provider
through the same mechanism, refusing by name on providers that do not serve.

**This one costs money and the schema must say so.** Measured via the free
`get_billable_size`: 5 minutes of AAPL mbp-10 on XNAS.ITCH = **41.9 MB**; MBO =
14.4 MB. Any such tool needs a default `limit` and the byte figure in its
description.

**What it unlocks, measured:** building the per-snapshot series from a live book
and running the shipped CUSUM detector on it lit up **7 of the 8 depth channels
that `liquidity_events` declares and refuses** — `microprice`, `book_imbalance`,
`l5_imbalance`, `ofi`, `bid_depth`, `ask_depth`, `depth_slope`, all clean in
0–2 ms each. The library already names the missing step: `CHANNELS['ofi']`'s own
`why_unavailable()` says it lacks *"the per-snapshot series to run on, which is
the get_order_book_series step."*

### G5 · `fetch_*` never says which vendor dataset answered

**HIGH, and it connects to a known live problem in this project.**

`frame.attrs["dataset"]` is set by the provider, survives publish/resolve across
processes, and is written into the audit record as `source="databento:EQUS.MINI"`.
It is then dropped **three times**: by `FetchResult`, by `describe_reference`,
and by `explain_decision`'s `DataSourceRef`.

Verified personally — `FetchResult` declares exactly
`['columns','end','entities','kind','ref','rows','start','warnings']`. No
`dataset`, no `provider`, no `adjusted`, no `source`.

Why it matters: the dataset preference list is date-dependent. `EQUS.SUMMARY`
starts 2024-07-01, so below that date **`EQUS.MINI` becomes first choice** — a
feed carrying ~3% of consolidated volume whose daily close is the last print of
the UTC day, frequently an after-hours trade. An agent fetching 2019 daily bars
gets that silently, with nothing in the result saying so.

**Fix:** three scalars on an existing result, read from a frame the handler
already holds. **Cost: zero, no network.**

Ship `preflight_vendor_request` alongside it (**MEDIUM-HIGH**): coverage windows
for six datasets took **1.5 s and $0.00** via free metadata endpoints, and
`get_billable_size` gives the real preflight quantity that `get_cost` (which
returns $0.00 on this subscription) does not.

### G6 · `get_efficient_frontier` — closed-form algebra with zero doors

**HIGH.** `portfolio/optimize.py` implements the Merton frontier constants and
exact frontier weights for any target return. Verified personally: the string
`frontier` appears **zero times** under `agent/` and **zero times** under `mcp/`.

Today an agent wanting a frontier calls `run_portfolio_optimization` with
`method="target_return"` once per point — 33 ms and one agent turn each, so 40
points is 40 turns — and gets an SLSQP approximation of a closed form that is
exact. The closed form does 40 points in **<1 ms**.

### G7 · Black-Litterman returns the weights and discards the reason

**HIGH.** `black_litterman()` computes four arrays; the handler uses the
posterior returns and covariance to derive two scalars, then returns weights
only. Dropped: `implied_equilibrium_returns` (π), `posterior_returns` (μ̄), and
the posterior covariance.

Measured: one view (AAPL − XOM = +5%, confidence 0.6) moved the posterior spread
by **+0.01707 against a stated 0.05 — the view was absorbed at 34% strength**.
That is the one number that says whether the view did anything, and no agent can
see it. **Cost: zero** — three dict comprehensions over arrays already in scope.

### G8 · `estimate_kyle_lambda` can only be called the circular way

**HIGH.** `kyle_lambda` has two paths: a bars path the library's own docstring
calls circular by construction, and a tick path signed by Lee-Ready. The input
model accepts only `close`/`volume`, so **every Kyle lambda an agent can obtain
has `circular=True`**.

Measured on a live AAPL tape at 1 s buckets: circular **6.61e-06** (r² 0.054)
against Lee-Ready **2.09e-06** (r² 0.020) — **3.16× too large, and it looks 2.7×
better fitted**. The docstring instructs the caller to "pass trades (and quotes)
for a Lee-Ready sign that does not know the answer"; no agent can follow that
instruction. **Cost: 20–30 ms**, and `fetch_tick_tape`/`fetch_quote_panel`
already produce the refs it would take.

### G9 · Give `calculate_series_metrics` a benchmark slot

**HIGH, small.** It is the only tool in the library that accepts an arbitrary
series. Four live `risk_metrics` functions are missing from its closed registry
and have **no other series door**: `information_ratio`, `treynor_ratio`,
`drawdown_series`, `evt_tail_risk`.

The consequence is that a **strategy** return series — a backtest output, a
synthetic path, anything the agent produced itself — can never be scored against
a benchmark or run through EVT, because every other door requires a listed
ticker. **Cost: ~1 ms per metric.**

### G10 · Extend the indicator panel past 5 of 14

**HIGH.** Verified personally: `_PANEL_SHAPES` holds exactly five names —
`rsi`, `atr`, `adx`, `bollinger_bands`, `stochastic_oscillator`. The
single-symbol doors reach all 14 but return **only the last value**.

Net: a MACD, SMA, EMA, Williams %R, OBV, VWAP, Parabolic SAR, Wilder ATR or MFI
**time series cannot be obtained from any tool in this library, for any number
of tickers**. Since `compute_indicator_panel` is what feeds
`build_model_dataset`, those nine indicator families also cannot become model
features. Measured: the panel does two indicators over 100 names in 24 ms; the
per-name equivalent is 168 ms of Python — or 100 separate tool calls each
returning a scalar.

Ship with it: `compute_indicator_panel` drops **all seven** indicator period
parameters that `get_technical_panel` exposes, so a persisted RSI feature is
always RSI(14).

---

## 4. MEDIUM

Grouped by the shape they share.

**Series that should be publishable as refs** (the plumbing exists and is
proven — `describe_reference`/`read_reference` already work):
`rolling_sharpe` (623 values, and see D3), `kalman_hedge_ratio`'s 875×4 path
including `Kalman_Gain`, `pca_returns`' PC score series plus the full eigenvalue
spectrum, `detect_regimes`' 874 per-bar labels, `amihud_illiquidity`'s rolling
series, `basis_history`'s three series, the per-snapshot book series (G4),
pairs' 753-point spread state machine, `run_signal_panel_backtest`'s
`portfolio_returns` — the last of which is a **chain break**, not just a missing
field: it is the only backtest tool whose output cannot be fed into any of the
library's return-consuming tools.

**Parameters the library takes and no tool input expresses:**
`get_data_quality_report` has no `source` (yfinance only, so a Databento or
external frame can never be quality-checked), no `calendar` (measured: the same
AAPL-2024 frame gives **0 gaps under XNYS and 9 under XCME**), and no volume
anomaly knobs (defaults → 0 anomalies, `(5, 0.5)` → 2).
`intraday_volume_profile` hard-wires New York 09:30–16:00 — measured on LSE data
the tool keeps 125 of 510 bars and reports `open_share` 0.101 against a true
0.255, a **2.5× error on the one number a VWAP schedule is built from**, while
warning that the *caller's* data is unusual; Tokyo it refuses outright.
`scan_basis_dislocations` hard-wires four detector parameters the underlying
detector exposes. `optimize_hierarchical_risk_parity` and
`get_volatility_estimators` pin `periods_per_year` at 252.
`get_liquidity_adjusted_var` pins `impact_coefficient`, which alone moves
`expected_liquidation_cost` across a **1,000× range**.
`run_stationarity_tests` cannot set `kpss_lags` (its `lags` is ADF-only) or
`vr_periods`, and never reports which lag Andrews chose.

**Diagnostics computed and dropped:** the SLSQP solver's iterations, status,
objective value and **Lagrange multipliers** (`_solve_constrained` returns
`result.x, result.success` and discards the rest); the covariance condition
number, which is computed unconditionally and emitted only inside a prose string
above 1e4; `select_features`-style redundancy in the scanner (`basis_scan` keeps
4 of 19 detector keys per pair, dropping `direction` and `shift`);
`effective_spread`'s realized/impact split (measured effective 1.238 bps =
realized 0.303 + impact 0.932); distribution quantiles where a mean stands alone
(cancelled-order lifetime: median **10.1 ms**, mean **475.7 ms** — a 47× ratio);
the mbp-10 **order counts** (20 columns the book code never reads, which answer
the exact question `order_events.py`'s docstring says a book cannot).

**Discoverability:** `describe_data_capabilities` is the tool an agent is told to
consult, and it cannot answer the question G4 depends on. Verified personally —
it declares thirteen fields (`available`, `cache_dir`, `financial_ratios`,
`guarantees`, `notes`, `ohlcv`, `ohlcv_async`, `provider`, `quotes`,
`supported_intervals`, `ticker_info`, `trades`, `unavailable_reason`) and omits
**all four that actually distinguish the providers**: `order_book`,
`order_events`, `point_in_time_records`, `temporal_contract`. Measured override
matrix — databento serves `order_book` and `order_events` and nothing else does;
polygon serves `point_in_time_records` and nothing else does. Worse, the
`source` field's own description reads *"'yfinance', 'polygon', or
'bloomberg'"*, **omitting databento entirely** — so the one provider that serves
depth is not named as a legal value of the tool that exists to report coverage.
All four provider checks cost **<8 ms combined and are free**.
No cache introspection exists anywhere in the 211 tools beyond a path string.
`inspect_model`-style provenance is missing here too: `DataSetMetadata.notes`
is orphaned between two tools, and the Databento notes are the ones that matter
(they name the EQUS.MINI sampling problem explicitly).

**A composition trap worth fixing:** `compare_ratio_frames` returns three
structurally-null fields (`classify_divergence` never emits the keys it reads),
drops the exact conversion `ratio`, and — passed the obvious sibling output from
`fetch_financial_ratios` — reported **8 of 8 fields disagreeing on two identical
inputs**, because the nesting differs and `no_overlap` counts as disagreement.

---

## 5. Not worth exposing

Stated so the question is closed.

- **Absent, not unreachable.** Verified by grep, 0 hits each: Kupiec,
  Christoffersen, Cornish-Fisher, Kelly, CVaR/CDaR objectives, Johansen,
  Zivot-Andrews, Bai-Perron, Engle ARCH-LM, BDS, Phillips-Ouliaris, VECM, EMO
  and BVC trade classifiers, Hasbrouck, MRR, Huang-Stoll, Glosten-Harris, SVI,
  SABR, Heston, and every exotic payoff. Group/sector, turnover, cardinality and
  tracking-error constraints likewise do not exist in `optimize.py`. **These are
  feature requests, not reachability fixes, and should be priced as such.**
- **`register_*` extension points** — they take Python callables and cannot be
  driven from a JSON tool call.
- **Raw ADF/KPSS/PP statistic functions** — the composite exists precisely
  because a single p-value invites "not significant, therefore random walk".
  Expose their outputs, not the functions.
- **A transition matrix for `detect_regimes`** — it is a Gaussian mixture, not
  an HMM. A post-hoc transition matrix would look like an HMM and not be one.
- **`bootstrap_statistic` with a caller-supplied expression** — the closed
  11-name registry is a deliberate security boundary.
- **Dedicated `run_donchian_backtest` etc.** — four more tools for strategies
  already reachable from seven. Pure surface inflation on a 35-tool runtime.
- **The full Monte Carlo path matrix, the PBO per-combination Sharpe matrices,
  the replication covariance matrix, raw per-path hedge P&L** — the reduced
  forms answer the decision and the raw forms are payload an agent reduces
  again.
- **`_cache.dead_generations` as a *mutating* tool** — expose the count, never
  the deletion. Eviction is an operator action with a CLI.
- **Single-expression helpers** (`microprice`, `sqrt_impact_bps`,
  `adv_participation`, `overnight_gap_shift`, `day_count`) — a tool per
  arithmetic step is surface without capability.
- **`stitch_oos_returns`** — deliberately superseded by a more correct method.
  Exposing it hands the agent the inferior option next to the better one.
- **`bloomberg_provider.py`** — 580 lines, inert here, and *correctly reported*
  as unavailable with install instructions. No action.

---

## 6. What was checked and found sound

- **The runtime layer**: no orphans, no duplicates, 10/10 dispatch/schema match.
- **Zero dead code** in backtest (57 symbols), econometrics (42), derivatives
  (16), portfolio (103 functions, proven by runtime instrumentation, not
  reading), and delta one (41 of 43).
- **Every registry is fully reachable**: all 6 delta-one registries mirror their
  tool Literals 1:1; all 10 cost models, all 5 sizing schemes, all 6 stress
  scenarios + `custom`, all 8 strategies, all 4 covariance estimators, all 9
  futures roll × adjustment combinations, all 13 screener filter keys, all 8
  preprocessing steps.
- **Lossless boundaries**: all 6 `overfitting.py` functions and all 5
  `trade_analysis.py` functions lose **zero** keys. The 8 `construction.py`
  functions map 1:1 onto 8 tools and drop nothing. `validate_external_dataset`
  drops nothing (diff = `[]`). The derivatives runtime's 10 result models are
  `extra="allow"` and measured **zero** drops.
- **Second-order greeks cross intact** (vanna, volga, charm, speed with a units
  map), the smile fitter returns R², residual σ and the Durrleman arbitrage
  checks, and put-call parity returns the violation in dollars and bps — all
  four were premises I expected to be gaps and none was.
- **The futures engine exposes roll handling richly**: per-roll date, contracts,
  cost, spread points, skipped variation margin, plus margin calls and
  collateral interest.
- **The `ES` ambiguity trap is guarded** — `get_ohlcv("ES")` raises and names
  `ES.c.0` / `ESZ6` / `ES.FUT` / `ES~equity` rather than silently returning the
  equity.
- **`specs`-style configuration is fully wired**: no declared-but-ignored tool
  input field anywhere, and no unwired input model.

---

## 7. Defects tripped over

Not the subject. Recorded because discarding them would be worse. G1, G2 and G5
are also defects and are ranked above as proposals.

**D1 · `classify_trade_direction` raises on live data — blocker.**
`ValueError: cannot reindex on an axis with duplicate labels`, 4/4 attempts on
live Databento AAPL, with and without a quote panel — along the exact chain the
module's own docstring draws (`fetch_tick_tape` → `classify_trade_direction`).
Cause: `sign_trades` drops unclassifiable rows and sorts an unsorted tape, so
the returned index differs from the tape's; live AAPL has **23.6% duplicate
timestamps**. The rest of the library solved this by aligning positionally and
says so in a comment; this one caller was not converted.

**D2 · `sort_by` returns the worst result for lower-is-better metrics.**
Verified personally: `ascending=False` is hardcoded at **five sites** in
`agent/runtimes/backtest/tools.py` (456, 614, 816, 964, 1591), while
`run_backtest_optimization.sort_by` offers both `max_drawdown` and
`annualized_volatility`. Measured: sorting by `annualized_volatility` returns the
grid's **highest** volatility (0.1173 against a 0.1089 minimum). Worse, four of
the sortable metrics are **absent from every returned row**, so
`top_results[0]["annualized_volatility"]` is `None` after sorting by it.

**D3 · `rolling_sharpe_stability` ships a warning asserting something false
about its own return value.** Verified personally: `diagnostics.py:547` states
*"The rolling series is still computed and returned, because looking at it is
genuinely informative"*, and that sentence is repeated in a **warning the agent
reads**. The library returns 21 keys and none is a series. The promise is broken
in the library, not merely dropped at the boundary.

**D4 · `run_pca_analysis` returns loadings and contributions from two different
decompositions.** `factor_contributions` accepts a `pca_result=` argument to
avoid recomputing; the handler never passes it, so it recomputes with its **own
defaults**. At `standardize=False` the loadings are from one decomposition and
the contributions byte-identical to the `standardize=True` run. Two halves of
one response describing different matrices. One argument fixes it.

**D5 · `monitor_spread_stream` silently ignores five known fields on every
resumed call.** The handler only constructs a monitor when state is `None`, so
`channel`, `label`, `warmup`, `threshold` and `slack` are accepted and discarded
thereafter. Demonstrated: a monitor opened on `relative_bps`, resumed with
`channel="absolute_points", threshold=25.0` — still `relative_bps` at 9.0, **zero
warnings**. A different *formula* was requested and a different one computed.

**D6 · The same tool's default threshold re-introduces a miscalibration the
library had corrected.** `STREAMING_THRESHOLD = 15.0` is set with a measured
table explaining why a stream cannot use the batch number; the tool input
defaults to **9.0** and its description quotes the *batch* calibration. Measured
false-alarm rate on pure noise at 5,000 ticks: **51.5% at 9.0 against 5.0% at
15.0**.

**D7 · `analyze_roll` hard-codes ACT/365F and names it nowhere** — not in the
result, not in its warnings, not in the description. Measured on a 91-day
quarterly roll: `roll_yield_bps` 160.376 under ACT/365F vs 162.603 under
ACT/360. `daycount.py`'s own docstring says it exists to kill "five inline
`/ 365.0` sites"; two remain.

**D8 · The day-count rationales never reach the agent.** Verified personally
against the full 458 KB schema payload: the sentence *"it accrues about 1.4%
more than ACT/365F over the same period"* — the one fact that would change a
financing-convention choice — returns `False`. Same shape as the modeling
sweep's parameter-bounds finding.

**D9 · `get_option_risk_scenarios` has no `dividend_yield` at all**, so every
cell prices a non-payer. Measured: a 1-year ATM call is overstated by **23.4%**
at a 4% yield, with no argument that can fix it. The same tool exceeds the
default MCP inline limit at its own defaults (4,877 B vs 4,096 B) and the
omitted field is `grid` — its entire reason to exist.

**D10 · Tool schemas forbid a sign the library supports.** Verified personally:
`option_greeks(dividend_yield=-0.01)` prices fine (11.099996); every tool input
pins `ge=0`. That makes FX options (negative foreign rates) and commodity
convenience yields unpriceable on the whole 12-tool derivatives surface.

**D11 · `run_screener` silently ignores an invalid `sort_by`** — returns input
order with no error, and `ScreenerResult` has no warnings field to carry one.

**D12 · `OrderBookInput.snapshots` is typed `Dict[str, float]`**, so the inline
path cannot carry a `timestamp` — the first column of the contract its own
description cites. Consequence: `ofi_per_second`, `updates_per_second` and
`mid_changes_per_second` are **always null** inline, and correct via a ref.

**D13 · `max_position_pct` is a hard rejection, not a cap** on the portfolio
simulation path — all four construction methods raise rather than clamp.

**D14 · `max_drawdown_pct` means a fraction in one reachable place and a
percentage in another** (stress test vs futures engine). Both modules already
flag it; it remains live across the boundary.

**D15 · `build_data_bundle` does not check a declared `frame_kind` against the
ref's actual kind** — a `returns_panel` labelled `fundamentals` was accepted, and
`validate_data_bundle` then gave a confident PIT verdict about the wrong thing.

**D16 · Two stale docstrings that cost the agent a correct call.**
`get_data_quality_report` claims it has no holiday calendar and will report every
holiday as a gap — measured **0 gaps across all of 2024**, because it uses
`exchange_calendars`. And `LiquidityEventsInput` says the depth channels need
"an order book nobody serves yet", which `DatabentoProvider.get_order_book`
serves today.

---

## 8. Suggested order

1. **G1 and G2** — the two tools that return a wrong answer labelled confident.
   Both are result-model changes costing ~0. Nothing else on this list outranks
   a 7× volatility reported as converged.
2. **D1** — a blocker on the library's own documented chain.
3. **D2, D3, D4, D5, D16** — a wrong sort, two false promises, a two-headed
   response, and a silently ignored configuration. All small, all misleading.
4. **G5** (which dataset answered) and **D10** (the `ge=0` bound) — zero-cost
   changes that stop silent wrongness.
5. **G3** (state curves) — one pattern, five engines, zero marginal compute.
6. **G6, G7, G9, G10** — closed-form frontier, BL posterior, series metrics,
   indicator panel. All small, all unlock something with no other door.
7. **G8** (Kyle tick path), then **G4** (depth fetch) — G4 last of the HIGHs
   because it is the only one that spends money, and its schema must carry the
   billable size.

The pattern from the companion document repeats exactly: **almost every item is
a wrapper, a passthrough, or a field addition.** What differs here is that this
half of the library is already well exposed, so the remaining gaps are narrower —
and correspondingly, three of the top four items are not gaps at all but answers
that are wrong while looking right.
