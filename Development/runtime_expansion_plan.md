# Runtime Expansion Plan — five runtimes to nine

Status: **proposed**, not started. Written against `main` at the commit that
added the risk-free rate to the backtest engine.

The tool-selection problem is solved. `main` has five hard execution
runtimes (`research`, `backtest`, `portfolio`, `meta`, `modeling`) whose
dispatch tables refuse what they do not own, and typed `sqt://` references
already carry bulk values between them without bespoke bridges. The
question is no longer *how to stop one agent seeing 82 tools* — it is how
to grow past 82 without reintroducing that problem.

**Except that we cannot currently grow past 82 at all.** The measurement
that came out of writing this plan is the thing to read first: at 2,184
bytes per tool over the wire, the MCP context ceiling buys 82.4 tools, and
the library has 82. The remaining headroom is 912 bytes — 0.42 of one tool.
The 83rd tool fails the budget test whatever it is. Every phase below is
blocked behind an exposure change (§2) that is not a scaling optimization
for a hypothetical future surface, but the precondition for adding one more
tool to this one.

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
  Phase 0b lowers what that cap binds on — `auto` holds every runtime
  under its own byte target
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

Three exposure models, at the 151 tools of section 9. **These are measured
now, not projected** — Phase 0b shipped, and the numbers below come from the
implementation rather than from an estimate:

| model | `research` (30) | all nine (151) |
|---|---:|---:|
| eager — every schema up front | 64 KB | 322 KB |
| tiered — a fixed 8 full, the rest thin | 28 KB | 195 KB |
| **`auto` — thin the most expensive until it fits** | **≤32 KB** | n/a |
| thin — everything thin | 16 KB | 78 KB |

A thin entry measures **531 bytes against 2,184 eager: 76% smaller.**

> **Correction.** This section originally projected 165 B per thin entry and
> concluded that a fixed 8-tool tier put the whole library inside the
> ceiling at 168 KB. Both were wrong. 165 B counted a name and one line of
> text and ignored the MCP envelope — annotations are 125 B a tool on their
> own, and the "call `describe_tool`" instruction has to appear in the
> description *and* the schema, because a client may show the model only
> one of them. The real figure is 531 B, and a fixed 8-tool tier lands at
> 195 KB, still 11% **over** the ceiling rather than inside it.

That correction is why the shipped mode is **`auto` rather than a fixed
tier**. A fixed 8 is a guess that happens to be wrong at this surface size
and would be wrong again at a different one. `auto` takes a byte target and
thins the most expensive tools until the runtime fits, which:

- **minimises round trips rather than bytes.** A runtime's cost is
  concentrated in a few large schemas — `modeling`'s top three are 65% of
  it — so thinning three tools buys what thinning fifteen cheap ones would
  not. Every tool left described is one an agent calls without a lookup.
- **costs nothing where nothing is needed.** `research`, `portfolio` and
  `meta` already fit the 32 KB target, so `auto` thins **zero** tools in
  them. `backtest` thins 8 and `modeling` 1.
- **stays correct as the surface grows.** The total across nine runtimes is
  not a number to optimise, because no client is served it. The per-runtime
  number is, and `auto` holds that by construction.

**Take `auto`, not `thin`.** The thin column is the better number and the
worse design. Runtimes exist to stop an agent inventing a tool; a schema an
agent has not read is one it will guess at, and guessing arguments is the
same failure one layer down — `extra="forbid"` turns it into a clean
rejection rather than a silent default, but a rejected call is still a
wasted turn.

One thing the implementation had to add that the plan did not anticipate: a
thin entry says *call `describe_tool` for the arguments*, and `describe_tool`
lives in `meta`. Under `--runtime backtest --tool-detail thin` that
instruction is unfollowable and every thinned tool becomes uncallable. The
server therefore **injects `describe_tool` whenever anything is thinned**,
and never thins it. That is the one place scope is widened automatically,
and it is justified by the alternative being a listing that lies.

Still to settle with measurement rather than argument:

- **Whether the tail costs accuracy** needs an eval — thinned tools called
  correctly on first attempt versus fully-described ones. If the tail
  degrades, lower the budget; if it does not, raise it. The mechanism is now
  a single number rather than a code change.
- **Call frequency would be a better ranking than cost.** The audit log
  records it and nothing has run enough to have one yet. `plan_detail` takes
  the ranking as an ordering step, so frequency can replace cost later
  without touching anything else.

This is the only work in the plan that makes the *existing* surface cheaper

rather than just accommodating a larger one. It should ship with Phase 0.

---

## 3. The splitting rule, written down

`tests/agent/test_runtimes.py` already enforces "nothing below eight". Three
more conditions belong alongside it.

1. **Build the domain inside its current home first.** A runtime is created
   by *moving* a mature cluster out, never by declaring an empty one.
2. **The new runtime lands at ≥ 8 tools.** Already enforced.
3. **The donor stays at ≥ 8 too.** ~~*Nothing currently checks this.*~~
   **Correction: it already does.**
   `test_no_runtime_is_too_small_to_be_worth_a_boundary` iterates
   `all_runtimes()` and asserts the floor on every one of them, and a donor
   is still a runtime after a split. Simulated: moving the three
   microstructure tools out of `portfolio` leaves it at 7 and fails that
   test today, with no change needed. The original note assumed the test
   checked only newly declared runtimes. No work here.
4. **A split is a breaking change.** Anyone scoped to the donor loses the
   tool.

### Deprecation path for a moved tool

When `get_option_pricing` moves from `research` to `derivatives`, a
research-scoped agent calling it should not get a bare refusal. The runtime
error already names the owning runtime; add a *moved-from* record so it can
say "this used to be in `research` and now lives in `derivatives`" for one
minor version. That turns a break into an instruction.

The sequencing consequence below is unaffected — the constraint is real,
it is simply already enforced rather than needing to be added.

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

Numbering starts at 1 here because **Phases 0 and 0b are in §2**, where the
constraint that motivates them is measured. They are not preliminaries to
this list — they gate all of it, and neither is large.

Phases 1, 2 and 6 are ready to start once §2 ships. Phase 3 is gated on a
provider project, Phase 5 on the donor floor, and Phase 7 on a contract that
does not exist yet.

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
temporal contract. That is what makes every non-price dataset safe to model
on, and the library already has the join that consumes it.

> **Correction, from building 2a.** This section specified three timestamps:
> `event_time`, `available_at`, `revision_time`. **Two are enough, and the
> third would have made things worse.** Measured against the join that
> already exists:
>
> ```
> Q3 EPS, reported at 1.20, later restated to 1.05
>   row 1: event_time=2024-09-30  available_time=2024-10-25  eps=1.20
>   row 2: event_time=2024-09-30  available_time=2025-02-10  eps=1.05
>
> asof_join at 2024-12-01 -> 1.20   (what was known then)
> asof_join at 2025-03-01 -> 1.05   (the restatement)
> ```
>
> A restatement is a **row**, not a column, and `available_time` on that row
> already IS its publication date. A separate `revision_time` would carry
> the same value — and would invite the one-row-per-fact encoding that
> cannot reproduce history at all, because the earlier value is gone.
>
> So the third thing worth declaring is not a timestamp. It is **how
> revisions are encoded** (`versioned` / `snapshot` / `none` / `unknown`),
> which is what separates a source you can backtest on from one that can
> only describe the present. The library's existing column is
> `available_time`, not `available_at`; the plan's name was a synonym and
> the code's name won.

**Sequence it as a contract first, tools second.** Adding
`build_fundamental_panel` before the timestamp contract means building a
leak and retrofitting the fix.

- **2a — the contract. SHIPPED.** `data/temporal.py` defines
  `TemporalContract`: what a source says about `event_time`,
  `available_time`, entity scope and revision encoding. `DataProvider`
  gained a `get_temporal_contract(frame_kind)` hook whose honest default is
  "bars are safe by construction, everything else is unsupported here", and
  `describe_temporal_contract` (in `meta`, for now) answers the question
  BEFORE anything is fetched.

  Two properties rather than one, because they come apart: `pit_safe` is
  whether the join can happen at all, and `reproduces_history` is stricter —
  a `snapshot` source joins without leaking the future and still shows a
  backtest final values nobody had at the time. That is not lookahead; it is
  a different history, and conflating them would have hidden it.

  `DataSetMetadata` was NOT extended. It is documented as describing one
  OHLCV pull, and adding a `frame_kind` to it to cover filings would have
  muddied a type that is currently precise. The contract is its own type and
  `price_contract()` is the bridge.
- **2b — the bundle. SHIPPED, narrower than specified.** `DataBundle`
  pairs every frame with the `TemporalContract` it was fetched under, and
  `validate_bundle` answers "is this safe to model on" from that pairing.
  `pit_safe` is the WEAKEST link rather than an average: a bundle is used as
  a unit, and "mostly safe" is not a number anybody can act on.

  It does **not** have eleven named slots. Eight of the eleven have no
  provider behind them, and a container with eight permanently empty slots
  is the "empty box with a correct label" `point_in_time.py` already warns
  about — worse, a slot named `fundamentals` reads as an invitation to fill
  it from a source with no availability timestamps, which is the exact leak
  the contract exists to stop. A bundle holds whatever frames it was given,
  keyed by frame kind. Adding `fundamentals` needs no change to the
  container; it needs a source.
- **2c — the builders. PARTIALLY SHIPPED, and the split is on purpose.**

  **Built, because they have a source behind them:**
  `compare_data_sources` (the underrated one — see below),
  `join_point_in_time` and `validate_pit_records`. The last two take records
  **inline**, which is a real use case rather than a placeholder: a caller
  who has an earnings calendar, a set of FOMC dates or index-membership
  changes can join them onto a panel today, capped at 5,000 rows so nobody
  moves a vendor history through a JSON argument.

  **Not built, because building them now would build a leak:**
  `build_fundamental_panel`, `build_macro_panel`, `build_event_panel`. No
  provider in this library supplies availability timestamps for any of them.
  A tool named `build_fundamental_panel` that could only join on `event_time`
  is not a partial implementation — it is a three-week lookahead with a
  reassuring name, and the plan's own instruction was "adding
  `build_fundamental_panel` before the timestamp contract means building a
  leak and retrofitting the fix". The contract now exists and refuses; the
  builder waits for a source.

  **Not built, because the thing they list does not exist:**
  `build_universe_snapshot`, `list_data_snapshots` need a snapshot store
  that has not been designed.

  **Renamed:** `validate_dataset` became `validate_pit_records`. A modeling
  dataset built from bars is point-in-time safe by construction, so a tool
  validating one would always say "fine"; what needs checking is the RECORD
  frame about to be joined onto it, where the event/available swap lives.

> **Why this pays for itself.** `compare_data_sources` was the underrated
> one and it shipped. Building it turned up that the hard part is not
> spotting a difference — a diff does that — but separating three cases
> that look identical and need opposite responses:
>
> - **scale**: a CONSTANT ratio across entities. A missed unit conversion;
>   the fix is arithmetic.
> - **definition**: systematic, ratio NOT constant. The two are computing
>   different quantities and no conversion exists.
> - **agree**: within rounding. Vendors differ at the margin about
>   everything and it is not a finding.
>
> The `debt_to_equity` case splits across two of them. yfinance reporting a
> percentage is `scale` — divide by 100. Polygon deriving it from total
> liabilities is `definition`, because payables and deferred revenue are not
> proportional to debt, so the ratio wanders. Telling a user "these disagree
> by 2x" when the verdict is `definition` invites them to divide by two,
> which is exactly wrong. That distinction is the tool's whole value and it
> was not visible from the docstring.

Unlocks safe handling of earnings, analyst estimates, fundamentals,
CPI/FOMC, alternative data, news, index membership and corporate actions.

### Phase 3 — Microstructure

**New runtime · ~12 tools · blocked by Phase 5 (donor floor) and by L2 access**

The proposed tool list is good and all twelve are worth building. The
correction is what comes first: **the library has no way to represent an
order book at all.**

1. **An L2 contract on `DataProvider`. SHIPPED.** `get_order_book` refuses
   explicitly and by name, warning against the substitution that would
   otherwise happen quietly — top-of-book quotes passed off as depth, which
   gives a one-level book with an imbalance of exactly zero and reads as a
   perfectly balanced market rather than as missing data. The column layout
   is declared so every consumer agrees what a book looks like before one
   exists.
2. **One provider that implements it. STILL BLOCKED**, and not by anything
   in this repository. Until a source exists, every L2 tool is a tool that
   refuses, so none were built.
3. **The twelve tools and the split: NOT STARTED.** The split is doubly
   blocked — `portfolio` is at 11 and the microstructure category is at 4,
   so moving them out leaves 7 and fails the floor. Phase 5 first, as this
   section already said.

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

> **SHIPPED, and it did not need L2 to be useful.** Six of the fourteen
> channels — `mid_return`, `spread`, `effective_spread`, `signed_volume`,
> `trade_intensity`, `realized_vol` — need only trades and quotes, which
> providers here already serve. The eight that need a book are DECLARED and
> refuse by name rather than being omitted, because a missing row in a
> report reads as a quiet channel.
>
> **Three bugs were found by running it on data with no shock in it**, and
> every one would have shipped as a confident alarm rather than an error:
>
> 1. **`mid_price` fired on 8 of 8 pure random walks.** A CUSUM over a
>    price LEVEL accumulates the walk itself, because a level is not
>    stationary. The channel is now `mid_return`; `mid_price` stays in the
>    table as an explicitly refused channel so the trap is visible rather
>    than merely absent.
> 2. **A frozen channel produced a peak statistic of 286,431** while moving
>    from 1.00 bps to 1.02 bps. The reference window was nearly constant, so
>    the denominator was near zero. Now labelled `degenerate_baseline`, with
>    the shift reported in the channel's own units — the detection is
>    labelled, not suppressed, because a calm period before a real shock is
>    the case this tool is for.
> 3. **The textbook threshold of 5.0 alarmed on pure noise 36% of the time
>    at n=120 and 82% at n=1000** — worse with more data, which is the tell.
>    CUSUM's design figure is an average RUN LENGTH, and "did anything
>    happen anywhere in this window" is a different question. Calibrated to
>    ~5%: 8.6–9.3 across 42 to 1050 observations, essentially FLAT, because
>    the maximum of the reflected random walk has a steep exponential tail.
>    So the fix is a better constant (9.0) rather than the log-scaling
>    formula I expected to need.

### Phase 4 — Modeling depth

**Extends `modeling` · registry entries, not tools**

Rankers are done. What is missing is everything downstream of a fitted
model.

**Shipped:**

- **Calibration.** `EstimatorSpec.calibration` takes `isotonic` or
  `sigmoid`, fitted on held-out folds inside each training window.

  The live path was real, and worse than "meaningless". Measured on a
  6,000-row synthetic panel, asking what a caller actually asks — if I set
  `proba_threshold=T`, what fraction of the rows I select actually win:

  ```
  T      raw forest          isotonic-calibrated
         realized    n       realized    n
  0.5      0.740    943        0.741   909
  0.7      0.875    353        0.829   532
  0.9        --       0        0.912   194
  ```

  A random forest's probabilities are compressed toward the middle, so it
  never emits one above ~0.9. A caller setting `proba_threshold=0.9` got an
  **empty signal panel with no error anywhere** — not a wrong answer, no
  answer. Calibrated, the same threshold selects 194 rows that win 0.912 of
  the time.

  Refused by name in the three ways it can be asked for wrongly: on a
  regressor (no probabilities to map), with more folds than the window has
  rows (sklearn's own error arrives three frames down talking about
  `n_splits`), and with a method the schema does not know.

- **Robust regression.** `huber` is registered with a bounded, documented
  `epsilon`. Financial targets are heavy-tailed and an 8-sigma day
  contributes 64 times what a 1-sigma day does to a squared loss.

**Still open — remaining work, not deferred work. None of it is blocked:**

- **Conformal prediction** — the only interval method here that does not
  assume a distribution. Pairs naturally with the existing
  effective-sample-size reporting. NOT a registry entry: it wraps the
  walk-forward loop, because its calibration set has to be held out the same
  way calibration's is.
- **Ensembling** — stacking, blending, seed-bagging as `EstimatorSpec`
  compositions. The open question is the composition shape: a spec that
  contains other specs is a different type from one that names an estimator.
- **Multi-horizon** — one spec, several target horizons, which
  `get_feature_ic_decay` from Phase 1 immediately motivates.
- Also: CatBoost, regime mixture-of-experts, online/partial-fit.

> **Hold the line.** All of this lands in `ESTIMATOR_REGISTRY` and
> `ModelSpec`. **Zero new tools.** `list_modeling_capabilities` already
> reports the registry, so a new estimator is discoverable the moment it is
> registered — that is the mechanism working as designed, and it is why
> modeling can gain a dozen capabilities without gaining a dozen tools.

### Phase 5 — Portfolio depth

**Extends `portfolio` · 10 → 13 tools · PARTIALLY SHIPPED · the donor floor
is now clear**

Two jobs at once: real portfolio construction, and lifting the donor above
the floor so the microstructure split is legal.

**The second job is done, and it removes one of two blockers.**
`portfolio` holds 13 tools — 9 `portfolio_risk` and 4 `microstructure` — so
moving the microstructure four out leaves 9, above the floor. A test in
`tests/portfolio/` pins that so a later move cannot quietly break it.

> **Correction.** This originally read "Phase 3's split is no longer
> blocked", which was half right and therefore wrong. Rule 2 is that the NEW
> runtime lands at ≥ 8, and microstructure has **four** tools:
> `check_spread_proxy`, `detect_liquidity_events`,
> `get_microstructure_metrics`, `get_trade_profile`. The donor floor is
> clear; the new-runtime floor is not, and the four missing tools are all L2
> ones. So the split remains blocked by the same absent order-book source it
> was blocked by from the start — Phase 5 removed one of the two conditions,
> which is worth having and is not the same as removing the blocker.

- **Covariance — SHIPPED** as `estimate_covariance`, with `sample`,
  `ledoit_wolf`, `ewma` and `ewma_shrunk`. Shrinkage is the answer to the
  conditioning warnings rather than a caveat about them, and the reason is
  arithmetic: a covariance over N assets has N(N+1)/2 parameters, so 40
  assets on 120 days is about six numbers per parameter, and the smallest
  eigenvalues — the directions an optimizer levers into because they look
  like free risk reduction — are estimated worst.

  Measured on exactly that panel: condition number 205 for `sample`, 143 for
  `ledoit_wolf`, **228 for `ewma`** and 35 for `ewma_shrunk`. EWMA making it
  *worse* is the point worth carrying: it answers a question about regime
  rather than about estimation error, and it lowers the effective sample
  size doing so. They are not alternatives, which is why `ewma_shrunk`
  exists.

  Returned ANNUALIZED, because every other risk number here is and a daily
  covariance silently produces a volatility 16 times too small.
**Still open — remaining work, nothing blocking it:**

- **Objectives** — HRP, HERC, mean-CVaR, minimum CVaR, maximum
  diversification. As *methods on the existing optimizer tool*, not five new
  tools.
- **Constraints** — factor-neutral, beta-neutral, sector, tracking-error,
  turnover-aware, transaction-cost-aware. Constraint spec, one tool.
- **`run_scenario_analysis`** — deliberately not built yet. `run_stress_test`
  already applies named historical windows, so the marginal tool is
  arbitrary user-specified shocks, and the two need separating carefully
  before a second one earns its name.
- **Wiring `estimate_covariance` into the optimizer** as a `covariance`
  option, so the optimizer's own conditioning warning has a fix reachable
  without a second call. A spec option, not a tool.

> **`plan_rebalance` was built first, and it was the right call.** Every
> optimizer here implied an instantaneous jump between weight vectors, and
> the transition has two costs pulling opposite ways — impact for trading
> fast, holding the old portfolio for trading slow. It returns the schedule
> and both costs rather than one number, with `urgency` exposed rather than
> chosen.
>
> **What it surfaces that nothing else does:** a target weight the market
> cannot supply. On a $100m book, a 60% target in a $2m-ADV name comes back
> as needing **295 days** at a 10% participation cap. The weight vector is
> valid, the backtest fills at the close, and until now nothing anywhere
> noticed.
>
> **A bug found building it, worth recording because it pointed the wrong
> way.** `total_cost_bps` summed each day's average impact rate, so on a
> transition where liquidity does not bind it reported 1.83 bps for trading
> everything at once against 4.08 for spreading it over five days — exactly
> backwards. Summing basis points across days adds rates, not costs. Both
> numbers were small and plausible, and a caller would have concluded "trade
> fast, it's cheaper". It now reports total impact dollars over total
> notional, and a test checks the ratio against the square-root law's own
> arithmetic: 5x the daily volume must cost sqrt(5) times the rate, and it
> does to two decimals.

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
model families go behind **one declarative spec with a `model` field**,
exactly as estimators do in modeling. Not seven tools.

> **The `model` field SHIPPED, and it needed no chain.** Pricing is
> evaluated from parameters, not calibrated to a surface, so it was
> buildable now. `get_option_pricing` gained `model`, `american` and
> `binomial_steps` — **zero new tools**, because adding `price_option`
> beside a tool that already prices options is exactly the redundancy the
> rule exists to prevent.
>
> Four models: `black_scholes`, `black_76` (options on futures),
> `bachelier` (NORMAL, so a negative underlying prices — WTI settled at
> -$37.63 on 20 April 2020 and every lognormal model returned a domain
> error), and `binomial` (the only one that prices AMERICAN exercise).
> `american=True` on a European model is refused by name rather than
> silently priced European, which would understate by exactly the
> early-exercise premium.
>
> **Local vol, SVI, SABR and Heston are deliberately absent.** Every one of
> them is CALIBRATED to a surface rather than evaluated from parameters, so
> they need the chain nobody serves. Shipping a calibration routine with
> nothing to calibrate against is the empty box `point_in_time.py` warns
> about. The `model` field is the extension point; they are entries it does
> not have yet.
>
> **A mutation survived and produced a test worth keeping.** Making
> `black_76` carry the forward a second time — the exact double-count the
> module warns about — passed all 65 tests. Put-call parity on a forward is
> `C - P = e^(-rT)(F - K)` *whatever the drift is*, so parity structurally
> cannot see it. The test that catches it is the identity that defines the
> relationship: Black-76 on `F = S·e^((r-q)T)` must equal Black-Scholes on
> `S`, exactly. The error also grows with expiry, so the horizons run to
> four years — a near-dated test would have missed it too.

Still gated on a chain: `get_option_chain`, `build_iv_surface`,
`analyze_iv_term_structure`, `analyze_skew`, `fit_svi_surface`,
`calculate_surface_greeks`, `calculate_gamma_exposure`,
`analyze_volatility_risk_premium`. Buildable without one but not yet built:
`calculate_portfolio_greeks`, `run_option_scenario`.

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

> **DEFERRED, and this is the decision rather than a postponement.** No
> live use case is waiting, and the cost is not the twelve tools — it is
> that `sqt://` currently promises resolving twice yields the same bytes,
> and `publish()` refuses to overwrite specifically to protect that. Every
> holder of a reference depends on it. Streaming would either break that
> promise or bolt a second, mutable identity system beside it, and both of
> those are decisions to make when something needs them rather than in
> advance.
>
> The two prerequisites the section already names — a subscription-plus-
> snapshot audit model, and an ownership and cleanup story for `stream_id`
> — are the real work, and neither is startable without knowing what the
> live use case actually is.

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

### Schema prose is a budget line — and a drift detector

43% of every input schema is field-description text, and 19% of the whole
surface is the same field descriptions retransmitted per tool. §2 explains
why `$ref` cannot reach those bytes; what remains is editing.

Measuring it turned up something more useful than a byte count.
**Nineteen shared field names have drifted into multiple distinct
schemas**, because each input model re-types its own copy:

```
tickers          15 uses,  14 variants   <- essentially all hand-written
fill_price       17 uses,   7 variants
start_date       45 uses,   6 variants
commission_pct   19 uses,   6 variants
symbol           32 uses,   5 variants
slippage_pct     18 uses,   5 variants
initial_capital  20 uses,   5 variants
```

Some of that variance is *correct*. Two of `risk_free_rate`'s four variants
are the Black-Scholes discount rate — a different quantity that is
deliberately required rather than defaulted, and
`test_risk_free_rate.py::test_the_option_pricing_rate_stays_required`
exists to keep the two apart. The rule is therefore not "make them
identical".

The rest is not correct, and one case is a live agent-facing problem.
`commission_pct` has a single default — `0.001` in all nineteen tools that
take it — but only **eight** descriptions say so; the other **eleven** read
*"Commission per trade (fraction)"* and leave the caller to guess. Two
`initial_capital` variants carry an **empty** description. An agent choosing
between those tools reads different documentation for identical behaviour,
which is precisely the class of thing that produces a confidently wrong
argument — and it is invisible to every test we have, because each model is
individually valid.

Two rules, cheap to enforce:

- **One canonical `Field(...)` per concept**, imported by every input model
  that uses it. A variant is then something a model does on purpose, in one
  place, with a reason — not the default outcome of re-typing.
- **A per-field description cap, tested**, the way the total is pinned now.
  The existing budget test caught two overruns during this work; a per-field
  cap catches the cause rather than the symptom.

Expect ~15% of the bytes. Not the lever — Phase 0b is — but it compounds
with it, since a thin listing still pays full price for whichever schemas
get fetched, and the correctness win lands whether or not 0b ships.

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
| Thin-everything exposure | **Rejected** | The cheapest number and the worst design. A schema an agent has not read is one it will guess at — the same failure runtimes exist to prevent, one layer down. `--tool-detail auto` thins only what the budget requires. |
| Raising the MCP ceiling a third time | **No** | Argued up once (150k → 180k) and trimmed against twice. A limit that moves whenever it binds is not a limit, and §2 removes the need. |
| `streaming` as a firm phase | **Defer** | Keep it last and optional. The only phase that requires changing an invariant the rest of the architecture rests on. |

---

## 9. Where this lands

Nine runtimes, `streaming` optional as a tenth, and **151 tools** — none of
which any single agent ever sees, because exposure moved to the runtime in
Phase 0.

| runtime | now | new | out | in | after | eager | `auto` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `research` | 23 | +9 | −2 | — | **30** | 64 KB | ≤32 KB |
| `backtest` | 21 | +7 | — | — | **28** | 60 KB | ≤32 KB |
| `meta` | 14 | +4 | — | — | **18** | 38 KB | ≤32 KB |
| `modeling` | 14 | +2 | −2 | — | **14** | 30 KB | ≤32 KB |
| `data` | — | +13 | — | — | **13** | 28 KB | ≤32 KB |
| `portfolio` | 10 | +5 | −3 | — | **12** | 26 KB | ≤32 KB |
| `feature_lab` | — | +10 | — | +2 | **12** | 26 KB | ≤32 KB |
| `microstructure` | — | +9 | — | +3 | **12** | 26 KB | ≤32 KB |
| `derivatives` | — | +10 | — | +2 | **12** | 26 KB | ≤32 KB |
| | **82** | **+69** | −7 | +7 | **151** | 322 KB | — |

The `auto` column has no total, and that is the point rather than an
omission. `auto` takes a per-runtime byte target and thins the most
expensive tools until that runtime fits, so **every row is under the target
by construction, whatever the row holds.** A 30-tool `research` and a
12-tool `derivatives` both come in under 32 KB; the first pays a few
lookups for it and the second pays none. Session cost decouples from
runtime size, which is why the coarse-runtime rule in section 3 stays a
design judgement instead of becoming a budget negotiation.

The `thin` column is the floor, if every schema were fetched on demand: 78
KB for the whole projected library, against a ceiling of 176. Nobody is
served that either.

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
- **151 tools is 322 KB eagerly, 83% past the ceiling.** The eager ceiling
  is 82.4 tools and the library has 82, so the eager path has already run
  out: the 83rd tool fails the budget test. Phase 0b removes that as the
  binding constraint — not by making the total fit, which no client pays
  anyway, but by making each *runtime* fit a target it holds by
  construction. Phase 0 makes the surface safe to grow; Phase 0b makes it
  affordable, and neither one is about the total.

**Start with three, in this order:**

1. **Feature Lab** — promotion work sitting on finished analysis.
2. **Data** — the timestamp contract has to precede every dataset built on it.
3. **Microstructure** — gated on a provider project *and* on Phase 5
   clearing the donor floor.

**Phases 0 and 0b ship first, and neither is optional.** Phase 0 is what
keeps a nine-runtime surface from re-becoming the flat list runtimes were
built to prevent. Phase 0b is what makes it fit at all — and it is the only
work in this document that makes the *existing* 82 tools cheaper rather than
just accommodating more. Together they are the smallest phase here and the
only one whose absence blocks everything else.

---

## See also

- [`Documentation/19_runtimes.md`](../Documentation/19_runtimes.md) — the current runtime and handoff contract
- [`Documentation/18_mcp.md`](../Documentation/18_mcp.md) — the category budget this plan has to change
- [`Development/mcp_plan.md`](mcp_plan.md) — the measurements behind the original exposure design
- [`Development/modeling_native_plan.md`](modeling_native_plan.md) — the precedent for staging a large subsystem
