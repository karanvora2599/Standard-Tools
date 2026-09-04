# Reachability: what is built, what can be called, and what answers wrongly

A sweep of the whole surface — 207 tools, 245 source files, ~88,800 lines —
asking three questions that ordinary testing does not: can each tool be
*reached*, does each argument *do* anything, and does each answer *mean* what
it says.

Five agents ran it. Every claim below that drives a code change was
re-measured by hand afterwards, because two agent findings were wrong in ways
that mattered (§0.1). Provenance is marked throughout: **[M]** measured
directly here, **[A]** agent-reported and read but not re-run.

---

## Status: executed, and five of its own claims were wrong

Every tier below is done. The surface is 208 tools now, not 207 — T1 added
one — and the facade is 179.

**Read this section before trusting a number further down.** The plan was
written from a sweep in which claims that did not drive a code change were
not re-measured. Acting on them turned five into corrections, and two share
a single cause worth naming.

### The mistake that produced two of them

`n_workers` (§3) and numba (§4.4) were both reported as creating nothing and
running never. Both measurements were taken with the C++ extension present,
which is the normal install — and in both cases the "dead" code is the
`else:` arm of `if HAS_CPP`. "Never runs" there means the fast path is
working, not that the slow path is dead.

Deleting either, as this plan recommended, would have removed real capability
from the exact configuration that needs it: a machine where the extension
failed to build. **A measurement of what executes is not a measurement of
what is reachable**, and the difference is a whole install configuration.

### The corrections

| § | the plan said | measured |
|---|---|---|
| 2.3 | `predictions→score_panel` gives 124 long / 0 short vs 52/72 | Not reproducible. Two of three sizers recentre, so the end weights were 100/100 — which is *why* it survived. The real defect was `task` accepted and ignored, and `vol_scaled` advertised and unable to run at all |
| 2.4 | two implementations disagree on degenerate input | They do, but one is the superseded predecessor the other was *built on*, with zero callers. Deleted, not reconciled |
| 3 | delete `n_workers` — creates zero processes | It creates a real pool when `HAS_CPP` is false. Forcing that, the measurement hung spawning eight Windows processes. Kept; descriptions corrected. Chasing it found the actual defect: the C++ gate's comment claims it is scoped to `fill_price="close"` and the condition never checked, because the kernel gained `grid_ref_arr` and the comment was left behind |
| 4.1 | `_POOLED_NATIVE_MIN_ROWS` is correctly placed, the contrast case | Same defect as the other gate, and worse: 2.39× at 200 rows against 1.43× at 12,600. Both gates won **hardest where they were forbidden**, because the conversion each guarded against is a smaller share of a small panel's cost, not larger |
| 4.4 | numba is a hard dependency whose JIT paths are 80% dead | Eight of ten are the C++ fallback. All ten call sites checked. Kept, with the reasoning recorded in `pyproject.toml` |
| 5 | `backtesting/` is a trap resolving as a namespace package | Live code — two runtime modules and eight test files import it. The real defect was narrower: the only subpackage here without an `__init__.py` |

§4.5 changed shape rather than being wrong: `HAS_SCIPY` is unfalsifiable as
described, but the defect underneath is that **six modules import scipy
directly and it was never declared**, reaching the environment only
transitively.

### Two more, from my own work rather than the plan's

**§1.5's demonstration was imprecise.** I first showed the change-point
penalty with a *volatility* change, and this detector splits on the mean —
those breaks came from sample-mean noise. Re-measured on a genuine level
shift, the finding is stronger: clearing the old default on 500 daily
returns needs a drift change of **0.28 per day**.

**§2.2's fix created a dead end.** Refusing a level series was right, and
left the caller told to run `.pct_change()` on a reference, which is not
possible. `convert_reference` gained `equity_curve → returns_panel`.

### What did not need correcting

§1.1 through §1.4, §2.1, §2.2, §2.5, §2.6, §3.1, §4.2, §4.3 and the rest of
§5 held up under measurement, several of them starker than written — the
`SPEARMAN` fallback is a 1.0000 → 0.6457 swing in a headline metric, and the
zero-copy kernels turned out to *refuse* the only input that would have
justified them.

---

## 0. The shape of the result

**The wiring is clean.** This is worth stating first because it bounds the
search. All 207 tools route through their own runtime's dispatcher; the
catalog and the runtimes agree in both directions; the 178-tool facade is
exactly the union of eight runtimes with no name collisions; 178 + 20
(`modeling`) + 9 (`feature_lab`) = 207 reconciles exactly; all 16 worker tool
lists resolve with zero bad references; and **all 207 input models set
`extra="forbid"`**, so a misspelled argument is refused rather than dropped.
The 11 CLI subcommands all have handlers, the 5 MCP prompts name real
categories, the 5 resource templates are coherent. **[M]**

So nothing here is a routing bug. The defects are in three other places:

1. **A door that was never cut** — a complete, tested subsystem with no entry
   point on the surface (§1.1), and one `Literal` missing one string (§1.2).
2. **Values, not names** — arguments accepted and ignored, defaults that
   forbid the only useful answer, a rate dropped between two call sites.
3. **Guards that outlived their reason** — fast paths gated off by thresholds
   measured before the fix that made them fast.

### 0.1 Two agent claims that did not survive checking

Recorded because they shape how much of the rest to trust.

- An agent reported `get_order_book_metrics` and `get_order_event_metrics` as
  equally dead, "no tool produces `order_book_panel`/`order_event_panel`."
  Half wrong: `register_external_dataset` **does** mint `order_book_panel`
  (`data/models.py:204`). The real defect is narrower and sharper — one
  missing `Literal` member — and is §1.2.
- The same sweep proposed a reference-inspection tool. `describe_reference`
  already exists (`meta/__init__.py:81`). Rejected, §7.

---

## 1. Tier 1 — cannot succeed

Calls that can never return a result, for any input.

### 1.1 The entire Databento ingestion layer has no door — `data/databento.py`

`src/standard_quant_tools/data/databento.py` is **643 lines with 39 dedicated
tests** (`tests/data/test_databento_normalization.py`) and **zero references
from anywhere under `agent/`**. It is not exported from `data/__init__.py`
either. **[M]**

All seven public functions are unreachable:

| function | produces | reachable |
|---|---|---|
| `normalize_book` | `order_book_panel` | no |
| `normalize_mbo` | `order_event_panel` | no |
| `normalize_quotes` | `quote_panel` | no |
| `normalize_trades` | `tick_tape` | no |
| `looks_like_databento` | vendor detection | no |
| `book_depth` | declared depth | no |
| `flag_warnings` | per-row vendor flags | no |

The four normalizers map exactly onto the four external kinds
`register_external_dataset` mints. This is not a missing feature — it is a
finished feature with no handle on it, and it is the single largest piece of
built capability the sweep found. It also lands precisely where it hurts
most: converting a vendor MBP-10 extract into this library's book contract is
the first thing anyone with a real L2 feed has to do, and right now it can
only be done by hand, outside the surface, with the judgement calls
(`price_scale`, `timestamp`) made silently rather than reported.

`normalize_book`'s own docstring is explicit that two of its judgements
"change the numbers and neither is inferable from the result" — which is
exactly why it returns notes alongside the frame, and exactly what a
hand-rolled conversion loses.

**Fix:** §6, tool **T1**.

### 1.2 `get_order_event_metrics` — one string missing from one `Literal`

`RegisterExternalDatasetInput.kind` (`agent/runtimes/data/models.py:204`):

```python
kind: Literal["order_book_panel", "event_panel", "tick_tape", "quote_panel"]
```

`event_tools.py:150` resolves with `expect="order_event_panel"`. That kind is
never offered, so no ref of it can ever be minted, and `convert_reference`
does not bridge to it. **[M]**

Everything else for it is already built:

- `KIND_COLUMNS["order_event_panel"]` — `(timestamp, order_id, action, side, size)` (`data/external.py:84`)
- `KIND_DESCRIPTIONS["order_event_panel"]` — written in full (`external.py:96`)
- `EXTERNAL_KINDS` includes it (`handoff.py:194`)
- the resolver, the consumer, and the batched reader all handle it

These are **not** a rename of each other — `event_panel` is
`(event_time, available_time)`, a point-in-time contract; `order_event_panel`
is an MBO feed. Both are real kinds. One is simply absent from the only tool
that can produce one. **[M]**

Blast radius is narrower than it looks and worse than it sounds: the inline
`events` path still works, so `get_order_event_metrics` is not wholly dead —
but the field description for `ref` says a real feed "cannot travel through a
tool argument," so what is stranded is the only path that scales. The toy
path works; the real one is unreachable.

Separately: **`event_panel` has no `expect=` consumer anywhere on the
surface** — an offered kind nothing reads. Flagged as an open question in
§8, not as a defect; it may be consumed by the point-in-time machinery
through another route.

**Fix:** add `"order_event_panel"` to the `Literal`. One string. Then a
surface test asserting `set(Literal) == EXTERNAL_KINDS`, so the next kind
added cannot drift the same way.

### 1.3 `construct_weights_from_scores` — broken on all four methods

Every method — `rank`, `zscore`, `top_bottom`, `vol_scaled` — fails with
`kind 'weight_panel' expects a non-empty {ticker: {date: value}} mapping`.
Committed with zero tests. **[M]**

### 1.4 `detect_liquidity_events` — 5 of 6 channels raise on a flat market

Five channels raise `ValidationError`; `cusum`'s degenerate-baseline early
return omits `severity` and `peak_statistic`, both required by
`ChannelResult`. A flat or thin market is exactly when a liquidity-event
detector is asked to run. **[M]**

### 1.5 `detect_change_points` — the default forbids the only useful answer

`penalty` defaults to `10.0` and is compared against a gain bounded above by
the series' total RSS about its mean. Measured on a series with a **10×
volatility regime change** (σ 0.005 → 0.05), an unmistakable break: **[M]**

| penalty | breaks found |
|---|---|
| ≤ 0.001 | 3 |
| ≥ 0.01 | **0** |
| **10.0 (default)** | **0** |

Total RSS of that returns series is **0.644** — the default penalty is ~15×
larger than the maximum gain *any* returns series can produce. The same
series as prices (RSS 343,709) finds 3 breaks at the identical penalty.

**`on="returns"`, the default channel, cannot report a break for any data.**

And it does not fail loudly. It reports *"no break cleared the penalty. That
is evidence the series is homogeneous at this threshold"* — a sentence that
reads as a finding about the market when it is a statement about units.

**Fix:** scale the penalty to the series (a multiple of its variance, or BIC
`k·σ²·log n`) rather than an absolute constant, and refuse a penalty that
exceeds total RSS rather than reporting homogeneity.

---

## 2. Tier 2 — silent wrong answers

The worst class: a plausible number, no warning.

### 2.1 `run_regime_adaptive_backtest` drops `risk_free_rate` — and contradicts itself

`backtest/tools.py:453` passes `risk_free_rate` into `backtest_grid`.
Sixteen lines later, `dummy_input = BacktestInput(...)` is rebuilt field by
field and **omits it**, so `_run_backtest` measures Sharpe against 0%. **[M]**

The tool therefore *selects* parameters using the caller's rate and *reports*
a Sharpe measured against zero. Two numbers in one result, computed under
different assumptions.

Agent measurement: rf = 0.20 gives 0.4344 where the control gives −1.0791. **[A]**

**Two sibling call sites already carry the fix and a comment naming this
exact bug** (`tools.py:1413` and `:1519`): *"Both of these dropped the
advertised `risk_free_rate`, so a caller asking for a 4.5% rate got a Sharpe
measured against 0% with no error and no warning."* The third site was
missed. **[M]**

**Fix:** pass it, and add the assertion the other two sites lack — a test
that varies rf and asserts the reported Sharpe moves.

### 2.2 `equity_curve` → `calculate_series_metrics` — Sharpe 232.475 vs −1.808

A reference conversion that feeds a level series where a return series is
expected, with no warning. **[A]** The magnitude is the tell: 232 is not a
plausible Sharpe, and nothing refuses it.

### 2.3 `convert_reference(predictions → score_panel)` drops classification recentring

124 long / 0 short where the correct conversion gives 52 / 72. **[A]** A
long-only book silently produced from a long-short signal.

### 2.4 `zscore_cross_sectional` and `standardize_cross_sectional` disagree

Two implementations of one operation, differing on degenerate input. **[A]**
One of the four duplicate-implementation pairs found; the only one where the
two disagree.

### 2.5 Six unguarded `str` fields silently pick a branch

`on='Returns'` — capitalisation alone — silently switches channel. Four
`feature_lab` `method` fields treat `SPEARMAN` as Pearson. **[A]**

The house norm is to refuse: `get_option_pricing.model` is a `Literal`, and
`tests/surface/synth.py` exists to catch bare `str` on constrained fields.
These six are the exceptions that slipped it. **Fix:** make them `Literal`,
which also makes the valid values visible in the schema an LLM reads.

### 2.6 `vol_scaled` in `convert_reference` raises a bare `TypeError` **[A]**

Not a wrong answer, but a raw exception where every neighbour raises
`ValidationError` with a remedy.

---

## 3. Tier 3 — accepted and ignored

Arguments in the JSON schema an LLM reads, that change nothing.

| tool | field | evidence |
|---|---|---|
| `profile_feature` | `include_ic_decay`, `max_shift` | byte-identical output across all four combinations; `include_ic_decay` is the only field of 1,143 on the whole surface with no read anywhere in `src`, and `FeatureProfile` has no field for the curve to land in **[A]** |
| `compare_artifacts` | `label_a`, `label_b` | inert **[A]** |
| walk-forward, `run_backtest_optimization` | `n_workers` | `pools_created=0` at every setting; `backtest_grid` takes the C++ batch path before reaching the pool, and `_sqt_core` ships with `pip install`. Two docstring claims at `engine.py:880-881` and `:887` are stale **[A]** |

`include_ic_decay` is the worst of these because its description *sells* the
feature — "Off by default because it costs (2·max_shift + 1) extra IC
passes" — a cost rationale for work that never happens.

**Fix:** implement or delete. For `n_workers`, delete the field and correct
the docstrings; the C++ path is the right one and the pool is vestigial.

### 3.1 `list_modeling_capabilities` overstates by 3×

Reports **18** target types from `_literal_options(TargetSpec, "type")`; only
**6** are buildable. `future_mid_return` raises `ValidationError: target type
'future_mid_return' cannot be built from a price series`. **[A]**

The capability reporter is what an agent consults *instead of* trial and
error, so an error here costs more than an error in a tool. `optional_dependencies`
likewise reports only `lightgbm`/`xgboost`/`native_extension` — nothing about
scipy, numba, polars, blpapi or cryptography, so an agent cannot learn that
the Bloomberg provider and the signing path are unavailable.

**Fix:** a `buildable` flag per target, and widen `optional_dependencies`.

---

## 4. Tier 4 — built capability that never activates

### 4.1 `_NATIVE_MIN_ROWS = 50_000` now suppresses a 3–5× win

`modeling/validation/weights.py:64`. Re-measured here, outputs matching
exactly at every size: **[M]**

| rows | native ms | python ms | speedup | gate |
|---|---|---|---|---|
| 5,040 | 0.22 | 0.79 | **3.54×** | BLOCKED |
| 12,600 | 0.70 | 1.83 | **2.60×** | BLOCKED |
| 25,200 | 1.05 | 4.11 | **3.91×** | BLOCKED |
| 49,392 | 2.00 | 10.25 | **5.14×** | BLOCKED |
| 50,400 | 2.38 | 15.38 | 6.46× | passes |
| 100,800 | 3.61 | 18.12 | 5.02× | passes |

The comment justifying the gate says the kernel "LOSES … measured at 0.4× on
a 12,600-row panel **before this guard existed**." `_as_int64_ns`
(`weights.py:67`) then removed that loss and shipped. The guard never moved.

The principle in that comment — *"A fast path that is slower is a bug"* — was
right, and the guard was right when written. It is now the inverse bug: a
fast path that is faster, forbidden.

It matters because per-fold rows are `train_window × entities`. A 200-name
universe on a 252-day window is 49,392 — **608 rows under the gate**. In a
realistic `run_model_experiment` the kernel fires on 1 call in 17, and that
one is the final whole-panel refit, not the fold loop it was written for. **[A]**

Contrast `_POOLED_NATIVE_MIN_ROWS = 5_000` (`validation/metrics.py:37`),
which fired on 16 of 32 calls in the same run. That gate is placed correctly.

**Fix:** re-derive the crossover and lower the threshold; add a benchmark
that asserts the kernel wins at the gate boundary, so the threshold cannot
outlive its measurement again.

Related: across the whole 6,893-test suite `label_uniqueness` is called
**4 times, all from the kernel's own unit test**. No end-to-end modeling test
crosses the gate, so the production path is untested as well as unreached. **[A]**

### 4.2 A stale `.pyd` silently costs ~18× under Python 3.11

`_sqt_core.cp311-win_amd64.pyd` (Aug 30) **loads** under the uv CPython
3.11.15 on this box and exports 40 symbols — missing `rank_by_date` and
`permutation_null_ic`. Both are `hasattr`-guarded, so they fall back
silently. **[M]**

The symbol-level guard is the right design and it is working exactly as
intended — the cost is that a stale artifact is indistinguishable from an
absent one. **Fix:** record the build's symbol count in
`list_modeling_capabilities`' `native_extension` entry, so a stale build is
visible rather than merely slow.

### 4.3 Six zero-copy kernels with no production caller — and leave them alone

`batch_run_strategy_zerocopy`, `rolling_beta_zerocopy`,
`rolling_factor_loadings_zerocopy`, `rolling_hurst_zerocopy`,
`simulate_forward_paths_zerocopy`, `technical_indicators_zerocopy` — declared
with 55 lines of helper machinery and a 250-line parity test, called from
exactly one test file and nothing in `src/`. **[A]**

**But wiring them would buy nothing:** measured head-to-head at 1.5k/100k/1M
rows, savings run −17.5% to +18.8%, mean ≈ 0. pybind11's `forcecast` does not
copy an already-conforming array, and every candidate caller already passes
`.to_numpy(dtype=np.float64)`. **[A]**

**Recommendation: delete them,** with the measurement recorded in the
CHANGELOG. This is the one place in the sweep where the fix is removal, and
saying so is the point — dead code that would not pay is worth less than the
schema and maintenance surface it occupies.

### 4.4 `numba` is a hard dependency, 8 of 10 JIT paths dead **[A]**

`pyproject.toml:41` requires `numba>=0.57.0`. Only
`_rsi_state_machine` and `_bollinger_state_machine` ever execute — the two
with no C++ sibling. The other eight always lose to the C++ path.

**Fix:** move `numba` to an optional extra and guard the two live call sites,
or port those two to C++ and drop the dependency. The second is cleaner and
the kernels are small.

### 4.5 `HAS_SCIPY` is unfalsifiable **[A]**

Five modules gate on it. scipy is not declared, but `statsmodels` and
`scikit-learn` both require it and both are hard dependencies. So three
"scipy is not installed" refusals and two `math.erf` normal-CDF fallbacks can
never run. **Fix:** declare scipy and delete the dead branches, or keep the
branches and add a CI job that actually removes scipy.

Also installed and unused: **`torch` 2.11.0, imported nowhere in `src/`.**
Consistent with the earlier decision to reject torch (the engine hands
estimators 2-D X with no entity identity) — but it should come out of the
environment spec. **Missing: `blpapi`, `cvxpy`.**

---

## 5. Tier 5 — dead and duplicate code

- **`backtest/` vs `backtesting/`** — the latter has no `__init__.py` and
  resolves as a namespace package. A directory that imports successfully and
  shadows nothing is a trap. **[A]**
- **5 genuinely dead functions of 490** examined. **[A]**
- **`cmd_replay`** (`cli.py:140`) is tested (`test_cli.py:150`) but the CLI
  re-implements the body inline at `cli.py:373-376` — so the test covers a
  function the CLI does not call. **[M]**
- **4 duplicate implementations**, one pair disagreeing (§2.4). **[A]**
- **`event_panel`** — offered kind, no `expect=` consumer (§8).

---

## 6. New tools that earn their schema bytes

**The rule, given the standing constraint not to reinvent the wheel:** a new
tool earns its place only if it makes existing, tested, currently-unreachable
capability callable. Wrapping something already reachable does not qualify.

Under that rule the sweep justifies **one** new tool. That is itself the
finding: at 207 tools the surface is close to saturated, and the gap is
reachability, not features.

### T1. `prepare_vendor_extract` — the door onto §1.1

One tool, dispatching across the four existing normalizers.

```
kind:        Literal["order_book_panel", "order_event_panel",
                     "tick_tape", "quote_panel"]   # → the four normalizers
path:        str                                    # the vendor extract
out_path:    str                                    # where to write the converted file
price_scale: Literal["auto", "fixed_1e9", "float"] = "auto"
timestamp:   Literal["auto", "ts_event", "ts_recv"] = "auto"
levels:      Optional[int] = None
keep_empty_levels: bool = False
dry_run:     bool = False      # detect and report only, write nothing
```

Returns the converted path, the **notes** (`normalize_book` already produces
them, and they record the two judgements that change the numbers), the
detected depth (`book_depth`), whether it looked like a Databento extract
(`looks_like_databento`), and any vendor flags (`flag_warnings`).

Why one tool and not four: it mirrors `register_external_dataset`'s shape, so
the pair reads as one pipeline —

```
prepare_vendor_extract → register_external_dataset → get_order_book_metrics
                                                   → get_order_event_metrics
```

— and `dry_run` folds what would otherwise be a second inspection tool into
this one. It unlocks 643 lines and 39 tests, and it is the workflow that
matters most given a real Databento L2 subscription.

**Prerequisite:** §1.2 must land first, or the `order_event_panel` arm of
this tool produces a file that cannot be registered.

### Conditional, pending §8

Nothing else is proposed. Two candidates are held behind the open questions
in §8 rather than committed to here.

---

## 7. Rejected proposals, and why

Recorded so they are not re-proposed.

| proposal | reason |
|---|---|
| `describe_reference` — inspect any `sqt://` ref | **Already exists** (`meta/__init__.py:81`) |
| `list_buildable_targets` | A `buildable` flag on `list_modeling_capabilities` (§3.1), not a tool |
| `normalize_order_book` as its own tool | Folded into T1; four one-per-normalizer tools quadruple the schema for one operation |
| `inspect_vendor_extract` | Folded into T1 as `dry_run` |
| Wire the six zero-copy kernels | Measured mean ≈ 0 (§4.3); delete instead |
| A polars-input tool for `hurst_exponent` | The only `pl.Series` signature in the package, and no tool input can carry one — every input is a JSON-shaped pydantic model |
| A torch estimator | Already rejected on measurement: the engine hands estimators 2-D X with no entity identity |
| A multi-output regression tool | Already rejected on measurement: +0.0014 R² |
| A second change-point tool with better defaults | Fix the defaults on the existing one (§1.5) |

---

## 8. Open questions

1. **`event_panel` has no `expect=` consumer.** Either something reads it
   through a route the sweep missed, or it should be dropped from the
   `Literal`. Resolve before touching that `Literal` for §1.2, since both
   changes edit the same line.
2. **Is the `cli.py:373` inline body a deliberate divergence from
   `cmd_replay`, or drift?** Determines whether the fix is to call the
   function or delete it.
3. **`numba`: port the two live kernels to C++, or make it an optional
   extra?** (§4.4)

---

## 9. Sequencing

Ordered by consequence per unit of work, not by tier.

**First — wrong answers reaching a caller.** §2.1 (rf, one line, fix already
written twice elsewhere), §2.2, §2.3, §2.4. Each needs a test that varies the
dropped input and asserts the output moves; the absence of exactly that test
is why all four survived.

**Second — the door.** §1.2 (one string, plus the
`set(Literal) == EXTERNAL_KINDS` guard), then §8.1, then T1. This is the
largest capability gain in the plan and the smallest amount of new code.

**Third — cannot-succeed tools.** §1.3, §1.4, §1.5. All three want the same
thing: a test that runs the tool on degenerate input — flat market, two-valued
column, no break — and asserts a *useful refusal* rather than an exception or
a false negative.

**Fourth — arguments that lie.** §2.5 (`Literal`s, plus extending
`tests/surface/synth.py` to catch the remaining bare `str` on constrained
fields), §3, §3.1.

**Fifth — activation and removal.** §4.1 (with a boundary benchmark), §4.2,
then the deletions: §4.3, §5.

**A cross-cutting note.** Nearly every defect here is invisible to the
existing test layers because they all assert that a call *succeeds*. The
missing layer asserts that an input *matters*: vary one argument, assert the
output changes. That single property would have caught §2.1, §3, §3.1 and
§2.5 — four of the five most consequential findings — and it is cheap to
apply across a surface where all 207 models already forbid extras.
