# Runtime Expansion Plan — five runtimes to nine

Status: **proposed**, not started. Written against `main` at the commit that
added the risk-free rate to the backtest engine.

The tool-selection problem is solved. `main` has five hard execution
runtimes (`research`, `backtest`, `portfolio`, `meta`, `modeling`) whose
dispatch tables refuse what they do not own, and typed `sqt://` references
already carry bulk values between them without bespoke bridges. The
question is no longer *how to stop one agent seeing 82 tools* — it is how
to grow past 82 without reintroducing that problem.

The target is **9 substantial runtimes averaging ~17 tools each**, not 20
micro-runtimes. Execution (broker writes) is deliberately out of scope for
this plan.

---

## 1. Four corrections to the starting assumptions

Three phases are cheaper than they look and one is dearer. This changes the
ordering, so it comes first.

| Assumption | Reality | What is actually there |
|---|---|---|
| Feature Lab needs building | **Exists** | `modeling/analysis/feature_report.py` already implements `build_feature_report`, `feature_distribution_stats`, `feature_predictive_stats`, `redundancy_report`, `lead_lag_ic_curve` and correlation clustering. The gap is that `analyze_features` returns all of it as one opaque `report` dict. |
| Point-in-time joins are new | **Exists** | `modeling/dataset/point_in_time.py` has `asof_join`, `validate_pit_frame`, `coverage_report`. `check_leakage` already surfaces part of it. What is missing is the *bundle* abstraction above them. |
| Ranking models are a later addition | **Exists** | `lightgbm_ranker` and `xgboost_ranker` are already registered under the `ranking` task in `ESTIMATOR_REGISTRY`. The real modeling gaps are calibration, ensembling, conformal intervals and multi-horizon. |
| Microstructure just needs more tools | **Dearer** | It needs a *data contract* first. `DataProvider` documents that quotes are top-of-book only and that no shipped provider exposes depth, queue position or resting size. Twelve L2 tools with no L2 feed is twelve tools that refuse. |

Net effect: **Feature Lab and Data are mostly promotion work** — typing and
splitting functions that already compute correctly — while
**Microstructure is a provider project wearing a runtime costume.**

---

## 2. Phase 0 — runtime-scoped exposure (blocking)

This is not a tuning problem to solve later. It blocks every runtime after
the fifth.

The MCP server selects tools by **category** and a client holds the whole
selection for the entire session. Measured on `main`:

```
82 tools
149.7 KB   input schema
179,088 B  over the wire        (ceiling: 180,000)
1,869 B    per tool, average
```

Projecting forward at today's per-tool average:

```
100 tools  ~183 KB input   ~213 KB over the wire     21% past the ceiling
151 tools  ~276 KB input   ~322 KB over the wire     83% past the ceiling
163 tools  ~298 KB input   ~348 KB over the wire     98% past the ceiling
```

151 is where section 9 lands, and 163 is that plus the optional `streaming`
runtime.

The projection is almost beside the point, because **the limit is not ahead
of us — it is here.** At 2,184 B per tool over the wire, the ceiling buys
82.4 tools. The library has 82. The remaining headroom is 912 bytes, which
is 0.42 of one tool: **the 83rd tool fails the budget test**, whatever it
is. There is no version of this plan, or of any smaller plan, that adds a
single tool without the exposure model changing first.

The ceiling has already been argued up once (150,000 → 180,000) and trimmed
against twice. A third raise would make it decorative, and
`--categories all` is exactly what the budget test pins.

### The fix

Runtimes are already the *execution* boundary. Make them the *exposure*
boundary too, because no single client ever needs more than one or two.

- `sqt-mcp --runtime research` → 23 tools, ~43 KB. A whole runtime costs
  less than a quarter of today's full surface.
- `--runtime research+meta` reuses the existing `combine()` semantics, so
  the flag and the library agree on what joining means.
- `--categories` survives unchanged as the narrower filter *within* a
  runtime.
- The budget ceiling becomes **per-runtime** — which is the number a client
  actually pays. Set it at **72 KB**, which clears the largest projected
  runtime (`research`, 30 tools, ~64 KB eager) with real headroom while
  still refusing a runtime that has quietly become the whole library again.
  Phase 0b lowers what that cap binds on — the same runtime is 21 KB tiered
  — so treat 72 KB as the guard on the eager path and re-derive it once
  tiering ships. Note the direction either way: a *per-runtime* cap
  constrains sprawl inside a runtime and no longer constrains the number of
  runtimes. Nothing should — the coarse-runtime rule in section 3 keeps that
  honest, enforced by a test rather than by a byte count.

The HTTP transport makes this more valuable: one host can serve several
runtime-scoped endpoints, and an agent's scope becomes a URL rather than a
flag it could have set differently.

**Ship Phase 0 before any new runtime.** It is small.

### Phase 0b — thin listings, schemas on demand

Scoping fixes *who sees what*. It does not fix the per-tool price, and 64 KB
for `research` is still most of a budget spent before the agent has done
anything. That price is worth measuring rather than assuming.

Where the 1,869 B/tool goes, measured on `main`:

```
 11,263 B   7%   tool descriptions
142,003 B  93%   input schemas
                  └─ 61,770 B   43% of schema is field DESCRIPTION text
                     29,505 B   19% of everything is byte-identical fields
                                repeated across tools
```

The duplication is stark. `end_date` is transmitted 43 times, `start_date`
40, `symbol` 22, `risk_free_rate` 20 — the same bytes, every time.

**`$ref` does not fix this.** MCP hands each tool a self-contained
`inputSchema`; the repetition is *between* tools, and there is no shared
document for a `$ref` to point into. Shortening the descriptions is the only
way to reach those bytes, and it wins maybe 15% on a number that has to fall
by 85%. Worth doing on its own merits — a shorter `risk_free_rate` blurb is
a better blurb — but it is not the lever.

The lever is that **`describe_tool` and `validate_tool_call` already exist**
in `meta`. The machinery for handing an agent a schema at the moment it
needs one is built and tested; the MCP server simply does not use it, and
instead ships all 82 schemas up front on the assumption the agent will need
every one. It will not. A session calls a handful.

Three exposure models, at the 151 tools of section 9:

| model | 64 KB | **21 KB** |
|---|---:|---:|
| eager — every schema up front (today) | 64 KB | 322 KB |
| **tiered** — 8 full, the tail thin | **21 KB** | **168 KB** |
| thin — name + one line, `describe_tool` on use | 6 KB | 28 KB |

A thin entry is ~165 B against ~1,869 B eager: **91% smaller**. The whole
151-tool surface, every runtime at once, fits in 28 KB — comfortably inside
a ceiling that today's 82 tools already strain.

**Take the tiered row, not the cheapest one.** The thin column is the better
number and the worse design. Runtimes exist to stop an agent inventing a
tool; a schema an agent has not read is a schema it will guess at, and
guessing arguments is the same failure one layer down — `extra="forbid"`
turns it into a clean rejection rather than a silent default, but a rejected
call is still a wasted turn. Tiering bounds that: the tools a runtime is
*for* stay fully described and zero-round-trip, and only the long tail costs
a lookup before first use. 168 KB is then the whole projected library inside
the existing ceiling, with runtime scoping cutting the actual session to 21.

Two things to settle with measurement rather than argument:

- **Which eight are the primary tier** should come from `explain_decision`
  call frequency, not from taste. The audit log already records it.
- **Whether the tail costs accuracy** needs an eval — thin-listed tools
  called correctly on first attempt versus fully-described ones. If the tail
  degrades, the tier grows; if it does not, it shrinks. Do not ship a fixed
  split and call the question answered.

This is the only work in the plan that makes the *existing* surface cheaper
rather than just accommodating a larger one. It should ship with Phase 0.

---

## 3. The splitting rule, written down

`tests/agent/test_runtimes.py` already enforces "nothing below eight". Three
more conditions belong alongside it.

1. **Build the domain inside its current home first.** A runtime is created
   by *moving* a mature cluster out, never by declaring an empty one.
2. **The new runtime lands at ≥ 8 tools.** Already enforced.
3. **The donor stays at ≥ 8 too.** *Nothing currently checks this.* Moving
   the three microstructure tools out of `portfolio` today leaves it at 7 —
   a floor violation produced by a legal-looking move. Extend the test to
   both sides of a split.
4. **A split is a breaking change.** Anyone scoped to the donor loses the
   tool.

### Deprecation path for a moved tool

When `get_option_pricing` moves from `research` to `derivatives`, a
research-scoped agent calling it should not get a bare refusal. The runtime
error already names the owning runtime; add a *moved-from* record so it can
say "this used to be in `research` and now lives in `derivatives`" for one
minor version. That turns a break into an instruction.

### Consequence for sequencing

Microstructure cannot leave `portfolio` until `portfolio` has grown back
past 8 on its own merits. That is why **portfolio depth (Phase 5) is
sequenced before the microstructure split**, not after it.

---

## 4. Dependency order

```
                    ┌─────────────────────────────┐
                    │ 0 · runtime-scoped exposure │  blocks everything below
                    └──────────────┬──────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
  ┌───────────────┐        ┌───────────────┐        ┌──────────────────┐
  │ 1 · feature   │◄───────┤ 2 · data +    ├───────►│ 4 · modeling     │
  │     _lab      │        │  available_at │        │     depth        │
  └───────────────┘        └───────┬───────┘        └──────────────────┘
                                   │
                                   ▼
  ┌───────────────┐        ┌───────────────┐
  │ 5 · portfolio ├───────►│ 3 · micro-    │
  │     depth     │        │  structure    │
  └───────────────┘        └───────┬───────┘
     keeps the donor ≥ 8           │
     so the split is legal         ├──────►  6 · derivatives
                                   └──────►  7 · streaming  (optional)

  Provider capability gates 3, 6 and 7 — no feed, no runtime.
```

---

## 5. The phases

### Phase 1 — Feature Lab

**New runtime · ~12 tools · donor: `modeling` (14 → 12, stays legal)**

The analysis exists and is correct. `analyze_features` hands back a single
untyped `report` dict, so an agent cannot ask a specific question — it gets
everything and parses prose out of a blob. **The work is typing and
splitting, not computing.**

| Tool | Source | Note |
|---|---|---|
| `build_feature_report` | exists | Typed result instead of a dict; stays the centerpiece |
| `analyze_feature` | exists | `feature_distribution_stats` + `feature_predictive_stats`, one feature |
| `get_feature_redundancy` | exists | `redundancy_report` + existing correlation clustering |
| `get_feature_ic_decay` | exists | `lead_lag_ic_curve`, surfaced as feature × horizon |
| `compare_feature_sets` | extend | Two `feature_panel` refs in, ranked delta out |
| `select_features` | extend | Redundancy clustering + IC, returns a `feature_panel` ref |
| `run_feature_ablation` | new | Refits without each feature |
| `run_feature_permutation_test` | new | The honest significance test for an IC |
| `get_feature_drift` | new | Distribution shift between two windows |
| `get_feature_regime_stability` | new | Composes with existing Hurst / regime work |

The per-feature report should carry coverage, missingness, distribution,
skew, kurtosis, autocorrelation, turnover, Pearson/rank/cross-sectional IC,
ICIR, hit rate, quantile monotonicity and spread, and stability by fold /
sector / regime — then the same feature across target horizons, which is
far more useful than a single tree-importance number.

Redundancy should be explicit: correlation, rank correlation, mutual
information, VIF, hierarchical clustering, condition number — so an agent
can see that RSI, 20-day momentum, MACD and stochastic are one momentum
cluster rather than four independent sources of alpha.

> **Failure mode to design against.** `run_feature_ablation` is a refit per
> feature. On a 40-feature panel with 8 folds that is 320 fits. It must go
> through the same estimated-fits reporting `validate_model_spec` already
> does, or an agent will call it casually and lose an afternoon.

### Phase 2 — Data and the `available_at` contract

**New runtime · ~13 tools · the highest-leverage phase here**

The most important idea in this plan is not the tool list — it is the
three-timestamp contract: `event_time`, `available_at`, `revision_time`.
That is what makes every non-price dataset safe to model on, and the
library already has the join that consumes it.

**Sequence it as a contract first, tools second.** Adding
`build_fundamental_panel` before the timestamp contract means building a
leak and retrofitting the fix.

- **2a — the contract.** Extend `DataSetMetadata` so every non-price frame
  declares which of the three timestamps it carries. A provider that cannot
  supply `available_at` says so, and `point_in_time_join` refuses rather
  than silently degrading to an ordinary timestamp join.
- **2b — the bundle.** `DataBundle` as a typed multi-frame container
  published as one reference: `bars`, `trades`, `quotes`, `orderbook`,
  `fundamentals`, `estimates`, `macro`, `events`, `options`,
  `reference_data`, `corporate_actions`.
- **2c — the builders.** `build_price_panel`, `build_returns_panel`
  (`alignment.build_returns_panel` exists), `build_fundamental_panel`,
  `build_macro_panel`, `build_event_panel`, `build_universe_snapshot`,
  `join_point_in_time`, `validate_dataset`, `compare_data_sources`,
  `list_data_snapshots`.

> **Why this pays for itself.** `compare_data_sources` is the underrated
> one. Two providers disagreeing about the same fundamental is currently
> invisible. The `debt_to_equity` unit divergence already documented in
> `FinancialRatios` — yfinance reports a percentage, Polygon a ratio, and
> Polygon derives it from total liabilities — is exactly the class of bug it
> would surface automatically rather than through a docstring someone has to
> read.

Unlocks safe handling of earnings, analyst estimates, fundamentals,
CPI/FOMC, alternative data, news, index membership and corporate actions.

### Phase 3 — Microstructure

**New runtime · ~12 tools · blocked by Phase 5 (donor floor) and by L2 access**

The proposed tool list is good and all twelve are worth building. The
correction is what comes first: **the library has no way to represent an
order book at all.**

1. **An L2 contract on `DataProvider`** — `get_order_book` returning a typed
   depth frame, defaulting to the same explicit refusal `get_trades` already
   gives. `describe_data_capabilities` gains `order_book` and becomes the
   gate.
2. **One provider that implements it.** Until this exists, every tool below
   is a tool that refuses.
3. **Then the twelve tools**, and only then the split from `portfolio`.

Target surface: `summarize_order_book`, `get_order_flow_imbalance`,
`get_microprice`, `get_book_imbalance`, `get_depth_profile`,
`detect_liquidity_events`, `analyze_queue_dynamics`,
`estimate_adverse_selection`, `estimate_impact_curve`,
`analyze_order_book_resilience`, `calibrate_fill_model`,
`run_microstructure_tca` — plus the three moved from `portfolio`.

#### Multi-channel change-point detection

This is the strongest single idea in the proposal and a genuinely new
capability rather than a repackaging. Allow CUSUM trigger channels over:

```
mid_price          microprice         spread
L1 imbalance       L5 imbalance       OFI
bid depth          ask depth          depth slope
cancel rate        trade intensity    signed volume
effective spread   short-term realized volatility
```

Which turns *"NVDA moved 1.4σ"* into a structured claim about which part of
the market changed:

```
NVDA
    price shock             low
    spread shock            high
    OFI shock               very high
    bid-depth collapse      high
    microprice divergence   high
```

Build it as **one tool over a declared channel set**, not one tool per
channel — the same rule that keeps `STRATEGY_REGISTRY` from becoming twelve
backtest tools. The channel list is data; the tool is
`detect_liquidity_events(channels=[...])`.

### Phase 4 — Modeling depth

**Extends `modeling` · registry entries, not tools**

Rankers are done. What is missing is everything downstream of a fitted
model:

- **Calibration** — isotonic and Platt. An uncalibrated classifier makes
  `proba_threshold` in `convert_reference` meaningless, and that is a live
  path today.
- **Conformal prediction** — the only interval method here that does not
  assume a distribution. Pairs naturally with the existing
  effective-sample-size reporting.
- **Ensembling** — stacking, blending, seed-bagging as `EstimatorSpec`
  compositions.
- **Robust regression** — Huber. Financial targets are heavy-tailed and
  every squared-loss fit here is being steered by its worst week.
- **Multi-horizon** — one spec, several target horizons, which the IC-decay
  curve from Phase 1 immediately motivates.
- Also: CatBoost, regime mixture-of-experts, online/partial-fit.

> **Hold the line.** All of this lands in `ESTIMATOR_REGISTRY` and
> `ModelSpec`. **Zero new tools.** `list_modeling_capabilities` already
> reports the registry, so a new estimator is discoverable the moment it is
> registered — that is the mechanism working as designed, and it is why
> modeling can gain a dozen capabilities without gaining a dozen tools.

### Phase 5 — Portfolio depth

**Extends `portfolio` · 10 → ~15 tools · must land before microstructure leaves**

Two jobs at once: real portfolio construction, and lifting the donor above
the floor so the microstructure split is legal.

- **Covariance** — Ledoit-Wolf, EWMA, factor. The optimizer already emits
  conditioning warnings; shrinkage is the answer to those warnings rather
  than a caveat about them.
- **Objectives** — HRP, HERC, mean-CVaR, minimum CVaR, maximum
  diversification. As *methods on the existing optimizer tool*, not five new
  tools.
- **Constraints** — factor-neutral, beta-neutral, sector, tracking-error,
  turnover-aware, transaction-cost-aware. Constraint spec, one tool.
- **New tools worth their own name** — `estimate_covariance`,
  `plan_rebalance`, `run_scenario_analysis`.

> **Build `plan_rebalance` first.** Every optimizer here currently implies
> an instantaneous jump between weight vectors. A transition path that takes
> current holdings, costs, liquidity and constraints and returns a
> *sequence* is what makes the optimizer's output actionable — and it
> consumes `weight_panel` references that already exist.

### Phase 6 — Derivatives

**New runtime · ~12 tools · donor: `research` (23 → 21, comfortably legal)**

Correctly sequenced: do not create a two-tool runtime. Build the surface
inside `research`, then move `get_option_pricing` and
`get_implied_volatility` across with it.

Target surface: `get_option_chain`, `price_option`,
`solve_implied_volatility`, `build_iv_surface`,
`analyze_iv_term_structure`, `analyze_skew`, `fit_svi_surface`,
`calculate_surface_greeks`, `calculate_portfolio_greeks`,
`calculate_gamma_exposure`, `run_option_scenario`,
`analyze_volatility_risk_premium`.

Gated on option-chain data, so it shares Phase 3's provider dependency. The
model families — Black-Scholes, Black-76, Bachelier, binomial, local vol,
SVI, SABR, Heston — go behind **one declarative `PricingSpec` with a
`model` field**, exactly as estimators do in modeling. Not seven tools.

New reference kinds: `option_chain`, `option_surface`, `greeks_panel`.

### Phase 7 — Streaming (optional, keep last)

**New runtime · ~12 tools · needs its own contract before its first tool**

A live stream **violates the reference contract**. `sqt://` promises that
resolving twice yields the same bytes, and `publish()` refuses to overwrite
specifically to protect that promise. A mutable stream cannot be a
reference without breaking every holder's assumption.

`stream_id` as a runtime-owned capability, with immutable
`sqt://stream_snapshot/…` for anything handed onward, is the right answer.
Two things to add:

- **A stream is not auditable the way a call is.** Every `dispatch()` writes
  one decision record; a stream runs for hours. The audit model needs a
  subscription record plus periodic snapshot records, or the log silently
  stops describing what the system did.
- **Streams outlive the process that made them.** `stream_id` needs an
  ownership and cleanup story before the first tool ships, or a fleet of
  agents leaks subscriptions.

Tools: `create_stream`, `configure_stream`, `attach_indicator`,
`attach_trigger`, `attach_model`, `get_stream_snapshot`,
`get_stream_health`, `list_streams`, `replay_stream`, `pause_stream`,
`resume_stream`, `close_stream`.

Defer indefinitely unless a live use case is actually waiting. It is the
only phase that requires changing an invariant the rest of the architecture
rests on.

---

## 6. Additions inside the existing five

### `research`

Change-point detection is the most valuable single addition, and it
composes with the existing regime work: Hurst answers *what kind of process
is this*, change points answer *when did the process itself change*.

**Nine new tools** (23 → 32, less the 2 that leave for `derivatives` → 30):

| tool | what it answers |
|---|---|
| `detect_change_points` | when did the process itself change |
| `detect_regimes` | Hidden-Markov regime labelling |
| `get_rolling_correlation` | correlation and covariance through time |
| `get_partial_correlation` | which link survives controlling for the rest |
| `analyze_tail_dependence` | copulas — does the correlation hold in the tail |
| `test_granger_causality` | which series leads which, formally |
| `run_stationarity_tests` | ADF, KPSS and the variance ratio in one report |
| `get_universe_breadth` | cross-sectional dispersion and participation |
| `run_event_study` | abnormal returns around a dated event |

**Not tools.** DCC-GARCH is a `model=` option on the existing GARCH tool,
and factor-neutral residuals belong in `get_factor_exposures`'s result
rather than beside it. Both would otherwise be a second way to ask a
question the surface already answers.

### `backtest`

Already mature. The missing capabilities are methodological, not another
strategy:

**Seven new tools** (21 → 28):

| tool | what it answers |
|---|---|
| `run_combinatorial_purged_cv` | many train/test splits without leakage |
| `get_overfitting_probability` | PBO — how likely the winner is noise |
| `run_multiple_testing_correction` | White's Reality Check and the SPA test |
| `get_regime_stratified_performance` | which regime the edge actually came from |
| `get_rolling_oos_performance` | is the edge decaying out of sample |
| `get_parameter_decay` | how fast a fitted parameter goes stale |
| `get_turnover_decomposition` | where the turnover is being spent |

**Not tools.** Bootstrap alpha intervals extend the existing
`block_bootstrap_ci`. Capacity limits, dynamic borrow, exchange calendars,
latency and market/limit/stop fills are all *cost and execution model*
configuration — they belong on the existing `CostModel` and execution spec,
where they compose with every backtest tool at once instead of forking the
surface into a with-latency and a without-latency version of each.

**Do not add one tool per strategy.** `STRATEGY_REGISTRY` already holds
eight strategies behind four named tools plus `run_backtest_compact` /
`run_strategy_matrix`, and `list_strategies` is the discoverability layer.
New strategies — carry, breakout, z-score, residual mean reversion,
volatility breakout, cross-sectional momentum — go in the registry. The
tool surface stays `strategy="donchian_breakout", parameters={...}`.

### `meta`

Already unusually strong. Worth adding: `estimate_tool_cost`,
`describe_runtime`, `trace_reference_lineage`, `compare_artifacts`.

`estimate_tool_cost` should return estimated fetches, rows, fits,
combinations, whether a native path is available, and memory/compute class.
There is already a proof of concept in `validate_model_spec`'s
`estimated_fits`.

---

## 7. Cross-cutting work

Four things that are cheap now and expensive at nine runtimes.

### Reference kinds are cheap; converters are not

Adding a kind is a table entry and a description. Adding a *converter* is
where cost compounds — pairs grow quadratically and each is a place two
domains' assumptions can diverge.

Make the existing policy an explicit rule: **a kind ships when a producer
exists; a converter ships when a consumer needs it and the semantics are
unambiguous.** Four converters for ten kinds is the right ratio, not an
omission.

Eventual vocabulary: `tick_tape`, `quote_panel`, `orderbook_panel`,
`microstructure_events`, `fundamental_panel`, `macro_panel`, `event_panel`,
`option_chain`, `option_surface`, `greeks_panel`, `feature_report`,
`experiment_table`, `scenario_cube`, `risk_report`, `tca_report`,
`stream_snapshot`. Ship each with its producer.

### The per-runtime tax, and a scaffold for it

Every runtime currently costs: a package with
`TOOL_DEFS`/`TOOL_DISPATCH`/`TOOL_CATEGORY`, a router description, a worker
with a system prompt, catalog wiring, facade re-exports, package exports,
partition tests, and a docs section. About nine touchpoints, three of them
hand-maintained lists that drift.

Tolerable at five runtimes. At nine it is a reliable source of exactly the
drift the existing tests were written to catch. **Write a `new_runtime`
scaffold before Phase 3**, not after the third time it is done by hand.

### Provider capability as a hard gate

`describe_data_capabilities` already probes by class override rather than by
calling. Extend it with `order_book`, `option_chain` and `streaming`, and
make it the documented first call for the three gated runtimes. Copy the
pattern that already works for ticks verbatim — *refuse by name, point at
the capability tool, never approximate* — rather than reinventing it per
domain.

---

## 8. What to cut

| Proposal | Call | Reason |
|---|---|---|
| A `risk` runtime | **No** | Risk and construction answer the same question; splitting leaves both below the floor and gains nothing. |
| `benchmark_tool` in `meta` | **Cut** | A developer concern, not an agent one. No agent decision changes on a microbenchmark, and it puts a timing harness inside the audited surface. |
| `list_execution_backends` | **Cut** | Already covered. `explain_decision` reports the execution path per call — the form that actually matters — and `describe_data_capabilities` covers the rest. |
| One tool per pricing model | **Cut** | Declarative spec, one tool. Same rule as estimators and strategies. |
| 20 reference kinds up front | **Trim** | Ship each kind with its producer. A kind with no producer documents an intention, not a capability. |
| `streaming` as a firm phase | **Defer** | Keep it last and optional. The only phase that requires changing an invariant the rest of the architecture rests on. |

---

## 9. Where this lands

Nine runtimes, `streaming` optional as a tenth, and **151 tools** — none of
which any single agent ever sees, because exposure moved to the runtime in
Phase 0.

| runtime | now | new | out | in | after | eager | tiered |
|---|---:|---:|---:|---:|---:|---:|---:|
| `research` | 23 | +9 | −2 | — | **30** | 64 KB | **21 KB** |
| `backtest` | 21 | +7 | — | — | **28** | 60 KB | 21 KB |
| `meta` | 14 | +4 | — | — | **18** | 38 KB | 19 KB |
| `modeling` | 14 | +2 | −2 | — | **14** | 30 KB | 18 KB |
| `data` | — | +13 | — | — | **13** | 28 KB | 18 KB |
| `portfolio` | 10 | +5 | −3 | — | **12** | 26 KB | 18 KB |
| `feature_lab` | — | +10 | — | +2 | **12** | 26 KB | 18 KB |
| `microstructure` | — | +9 | — | +3 | **12** | 26 KB | 18 KB |
| `derivatives` | — | +10 | — | +2 | **12** | 26 KB | 18 KB |
| | **82** | **+69** | −7 | +7 | **151** | 322 KB | 168 KB |

Read the tiered column down the page: **18–21 KB regardless of whether the
runtime holds 12 tools or 30.** That is not a coincidence — under tiering
the fixed primary tier dominates and the tail is nearly free, so session
cost decouples from runtime size. It is the property that makes a 30-tool
`research` and a 12-tool `derivatives` cost the same to hold, and it is why
the coarse-runtime rule in section 3 can stay a design judgement instead of
becoming a budget negotiation.

The number that matters is not the total. It is the **average runtime: 16.4
tools today, 16.8 after.** It holds flat, and that is the whole test of
whether this is a real expansion or a reorganization wearing one as a
costume. Nine runtimes covering four domains the library does not currently
touch, each landing at a size the existing five have already proved an
agent can hold — rather than the same 82 capabilities sliced thinner across
more boundaries, which would buy nothing and cost nine dispatch tables.

Two consequences worth stating plainly:

- **`research` at 30 and `backtest` at 28 are larger than any runtime
  today**, and that is fine. The runtime is the *execution* boundary; the
  category filter is the narrowing layer *inside* it, which is what
  `--categories screener` has always been for. A 30-tool runtime an agent
  can further narrow is a different thing from a 30-tool flat list it
  cannot.
- **151 tools is 322 KB eagerly, 83% past the ceiling — or 168 KB tiered,
  inside it.** The eager ceiling is 82.4 tools and the library has 82, so
  the eager path has already run out: the 83rd tool fails the budget test.
  The tiered number is what Phase 0b buys: the entire projected library fits the
  existing ceiling, and runtime scoping then cuts a real session to 21 KB.
  Phase 0 makes the surface safe to grow; Phase 0b makes it affordable.

**Start with three, in this order:**

1. **Feature Lab** — promotion work sitting on finished analysis.
2. **Data** — the timestamp contract has to precede every dataset built on it.
3. **Microstructure** — gated on a provider project *and* on Phase 5
   clearing the donor floor.

Phase 0 is not optional and is not large. It should ship before any of them.

---

## See also

- [`Documentation/19_runtimes.md`](../Documentation/19_runtimes.md) — the current runtime and handoff contract
- [`Documentation/18_mcp.md`](../Documentation/18_mcp.md) — the category budget this plan has to change
- [`Development/mcp_plan.md`](mcp_plan.md) — the measurements behind the original exposure design
- [`Development/modeling_native_plan.md`](modeling_native_plan.md) — the precedent for staging a large subsystem
