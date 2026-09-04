# Modeling native plan II — what is left, and what turned out not to be a kernel

A measured sweep of the `modeling` and `feature_lab` surfaces for C++
opportunities. Five parallel analyses plus an orchestrating measurement.
Every number below was produced on the project interpreter against
synthetic panels of stated shape; nothing here is an estimate unless it
says so.

The short version: **three correctness defects were found, only two new
kernels are justified, and the two largest costs in the whole subsystem are
pandas dispatch overhead that numpy removes entirely.**

---

## 0. The ceiling, which decides how much of this is worth doing

End-to-end, through the public tool surface, on a 300,000-row panel
(150 entities × 2,000 bars, 12 features):

| step | seconds | share |
|---|---:|---:|
| `register_external_panel` | 0.30 | 1.9% |
| `run_model_experiment` [ridge, 11 folds] | 1.96 | 12.2% |
| `run_model_experiment` [hist_gradient_boosting, 11 folds] | 12.61 | **78.5%** |
| `analyze_model_errors` | 1.20 | 7.5% |

**With a real estimator, 78.5% of wall-clock is sklearn fitting**, which no
kernel in this library can touch. `modeling_native_plan.md` §5 already said
so — *"sklearn's fit is 2-3% of a run"* was true of ridge and is not true of
a booster. Every proposal below is competing for the remaining fifth, and
that is the honest frame for judging whether any of it is worth the C++.

Note also `analyze_model_errors` at 1.20 s — **60% of what the entire ridge
walk-forward costs**, for a diagnostic. It is the newest code in the
subsystem and had never been profiled.

## 0.1 Two of the five existing kernels never run

Every native entry point was wrapped in a counter and a real walk-forward
run through the tool surface, at 200,000 rows:

| kernel | default spec | opt-in spec |
|---|---:|---:|
| `cross_sectional_correlation` | 55 | 55 |
| `fit_preprocess_stats` | 12 | 1 |
| `apply_preprocess_stats` | 23 | 1 |
| `standardize_by_date` | **0** | 22 |
| `label_uniqueness` | **0** | 12 |

`standardize_by_date` needs `preprocessing.normalization="cross_sectional"`.
`label_uniqueness` needs a weighting method AND a `label_end_date` column —
which an externally registered panel does not carry unless the registration
names `label_end_column`. So the kernel written because it was *"the worst
per-row cost in the module"* is unreachable on the default path and on the
external-panel path entirely.

This is not an argument for deleting them. It is an argument that the
measured "end-to-end 1.92x-2.55x" in the previous plan describes a
configuration most callers do not use.

---

## 1. Correctness defects — these come before any optimisation

### 1.1 `standardize_by_date` fabricates every entity to the mean when one is missing

**Both backends. Contract violated. Live.**

`panel_stats.hpp` states the contract:

> *NaN is skipped by the moments and preserved in the output ... a missing
> feature must stay missing rather than **be fabricated into an observation
> at the cross-section mean**.*

What actually happens is the precise inverse — the doc names the exact
failure mode it is meant to prevent:

```
input   [ 0.5,  1.4,  nan, -1.9,  1.6]
native  [ 0.0,  0.0,  0.0,  0.0,  0.0]
python  [ 0.0,  0.0,  0.0,  0.0,  0.0]
correct [ 0.07189, 0.71889, nan, -1.65344, 0.86266]
```

One NaN anywhere in a date's cross-section zeroes that entire (date,
column) cross-section. The Python fallback in
`modeling/features/transforms.py` does the same thing (`np.add.reduceat`
propagates the NaN into the date's mean, then `np.where(isfinite, ., 0.0)`
flattens it), so the two backends agree with each other and both disagree
with the documented behaviour.

**It was known, and that is the more useful finding.**
`panel_stats.cpp:564` documented it: *"NaN is NOT skipped here, and that is
a deliberate match rather than an oversight ... That is a wart, and it is
reproduced exactly because this kernel is a speed change and has no
business moving a number. In practice it never fires: alignment drops NaN
rows before the panel reaches the engine. If the rule is ever worth
changing, it should change on the Python side first, with both paths and
the tests moving together."* And `test_native_metrics.py` asserted it under
the name `test_nan_poisons_its_whole_date`.

So the kernel's own `.hpp` contract and its `.cpp` comment contradicted
each other, and the justification -- *"it never fires"* -- was retired by
`load_external_panel`, a path that did not exist when that note was
written. The fix followed the instruction it left: Python first, then the
kernel, then the tests.

**What the tests could not have caught.** Parity tests compare native
against Python by toggling `HAS_CPP`, so they are structurally incapable of
finding a misunderstanding both implementations share -- they would have
passed on any agreed-upon wrong answer. The replacement asserts against an
independent oracle written from the header, which is a different question
from backend agreement and needs its own test.

**Reachable, and not marginally.** An external panel carries NaN feature
values straight through `load_external_panel`. On a four-name universe
where one is in warm-up:

```
date        entity      f         z
2024-01-03  AAA     0.6404    0.0000
2024-01-03  BBB        NaN    0.0000
2024-01-03  CCC    -1.4870    0.0000
2024-01-03  DDD     7.0558    0.0000
```

Three present names with real dispersion, all reported as exactly average;
DDD is the largest value in the panel. 12 of 12 rows on warm-up dates
zeroed, 0 of 12 on clean dates. For a universe with staggered listings that
is most dates.

### 1.2 `network.mst_degree` drops the strongest edge in the graph

Written this session. `scipy.sparse.csgraph.minimum_spanning_tree` on a
dense array reads `0` as *absent edge*. A perfectly correlated pair has
Mantegna distance `sqrt(2(1-1)) = 0` — the edge the construction most wants
— so it is silently removed:

```
two entities, one an exact duplicate -> {'A': 0.0, 'B': 0.0}
```

A two-node spanning tree has exactly one edge, so both degrees must be 1.
In a larger universe the edge count stays correct and the tree reroutes
around the zero-distance pair, which corrupts exactly the topology this
feature exists to measure. Reachable through dual listings, a symbol
repeated in a universe, or stale repeated prices.

### 1.3 Lag columns bypass the infinity check

Written this session. `dataset/builder.py:542` checks
`[fs.output_name for fs in spec.features] + ["target"]`. Lag columns are
not in that list, and the check exists — by its own comment — so that inf
cannot *"feed straight into sklearn, which either raises a cryptic error or
silently produces garbage."*

Not unreachable, though it takes two steps to see: bar *t* has `X = inf`
but is dropped because feature `Y` is NaN at *t*; bar *t+1* survives;
`X__lag1` at *t+1* is `inf` while `long_panel["X"]` is clean.

---

## 2. The two kernels that are justified

### 2.1 `rank_by_date` — an extraction, not new math

Per-date average ranking, mirroring `standardize_by_date`'s signature:
`rank_by_date(values[n_rows, n_cols], date_codes, n_dates)`.

**Why a kernel and not numpy.** This is the one place where vectorisation
was measured and *lost*: pandas' `groupby.rank` is already the good
implementation, and a numpy rewrite of `ensemble._rank_within_date`
measured **395 ms against 407 ms at 3 models, and slower at 5** (702 vs
672 ms). There is no numpy win to take.

| 500 entities × 1000 dates | |
|---|---|
| `_rank_within_date`, 3 models | 407-455 ms |
| numpy rewrite | 395 ms — no gain |
| native, measured proxy | **~2.7 ms** |

Estimated **30-45x**. The proxy is sound: `cross_sectional_correlation`
with `spearman=True` costs 11.89 ms and ranks *two* arrays per date; the
pearson path costs 6.51 ms.

**It is an extraction.** `panel_stats.cpp:326 average_ranks()` already
implements `Series.rank(method="average")` tie semantics and
`panel_stats.cpp:272 bucket_by_date()` already counting-sorts rows by date.

**Three call sites**, which is what makes it worth the binding:
`ensemble._rank_within_date`, `analysis/feature_report._rank_turnover`, and
`dataset/target.apply_cross_sectional_target` — the last being the op
`modeling_native_plan.md` Phase 2 tabulated, measured at 1.4 us/row, then
never built and never declined.

### 2.2 `permutation_test_ic` — fuse the whole permutation loop

`analysis/feature_stability.py:358-362`. 200 permutations over 504k rows
costs **18.6 s**: group-index build 0.83 s, per-date Python shuffle 3.30 s,
`cross_sectional_ic` 6.27 s, and ~6 s of per-call pandas construction.

**Two vectorisations were tried and both measured slower**, which is why
this is a kernel and not a rewrite:

| attempt | result |
|---|---|
| one `np.lexsort` per permutation | **24.0 s vs 3.3 s (0.14x)** |
| rank-once + vectorised per-date pearson | **32.8 s vs 18.6 s (0.6x)** |

The rank-once insight is algorithmically right — shuffling values within a
date permutes their ranks, so ranking need not repeat — but it cannot be
expressed profitably in numpy, because the existing kernel counting-sorts
in O(n) and beats numpy's O(n log n). Fusing rank-once + within-date
Fisher-Yates + per-date accumulation across all permutations into one call
has a floor around 1 s: **~15-20x**. Reuses the counting-sort-by-date and
NaN-pair-drop machinery directly.

---

## 3. The big wins, none of which need C++

Ranked by seconds saved.

| # | site | now | after | gain |
|---|---|---:|---:|---|
| 1 | `feature_report._quantile_shape:272` | 78.3 s | 9.7 s | **7.8x** |
| 2 | `portfolio_eval.transform_predictions_to_weights:389` | 57.1 s | 16 ms | **3,545x** |
| 3 | `features/network.py` refit loop | 44-61 s | 8-15 s | **3-6x** |
| 4 | `feature_report._rank_turnover:170` | ~17 s of 40 s | — | regroup once |
| 5 | `dataset/alignment.stack_long` | 3.65 s | 1.94 s | **1.9-2.5x** |
| 6 | `audit.hash_dataframe` | 1.69 s | 0.1-0.2 s | 6-10x *(kernel)* |
| 7 | `bridge.py:319` `.dt.strftime` | 492 ms | 9.3 ms | **26x** |
| 8 | `diagnostics.error_attribution` | 721 ms | ~170 ms | **4.2-9.3x** |
| 9 | `bridge.py:169` `len(pd.bdate_range())` per pair | 121 ms | 0.04 ms | **1,107x** |
| 10 | `alignment.attribute_drops:92` | 0.20 s | 0.065 s | **3.0x** |
| 11 | `dataset/lags.expand_lags:125` | 1.33 s | 0.76 s | **1.8x** |
| 12 | `factors._pca_factor_return:94` | 202 ms | 0.13 ms | **1542x** (loop only) |

Three of these deserve their reasons recorded, because the shape of the
finding is the useful part:

**`_quantile_shape`** is `groupby.transform` with a Python callable — once
per date per feature. In `feature_predictive_stats`, the already-ported C++
kernel is **2% of runtime and the un-ported pandas glue is 81%**. That is
the general shape of this subsystem and the reason this plan proposes so
few kernels.

**`transform_predictions_to_weights`** costs 0.27 s on a dense panel and
**57 s on a panel whose only irregularity is staggered listing dates** —
a 37x penalty for entirely ordinary data, because the loop is over
*distinct availability patterns* and that count approaches `n_dates` as
soon as names enter and leave. The dominant single line is
`backtest/sizing.py:170`, `DataFrame.where()` with a **Series** condition,
which costs ~22 us *per column regardless of row count* — 46.6 ms on a
one-row 2000-column frame, against 0.019 ms for `np.where`.

**`hash_dataframe`** is the only assembly step whose cost is compute-bound
at a constant factor above the achievable floor: 0.47 s/M rows at 24
columns, 1.69 s at 84, 7.49 s at 204 — **3-6x more than reading the Parquet
it verifies**. `hash_pandas_object` is ~7x above the SHA-256 hardware floor
because it materialises a per-row uint64 Series. It is also the one step
paid on *every* load, from seven call sites. Two of those sites
(`tools.py:903` check_leakage, `tools.py:996` spec preflight) bind the
frame to `_panel`, read four keys out of `meta`, and discard it — 0.6-2.3 s
per call to read a dictionary. **That one is free to fix and needs no
kernel.**

---


## 3.1 The engine itself: the largest single win, and it is not a kernel

Measured on 200 entities x 1500 dates = 300,000 rows, 20 features, 16 folds,
by cost-by-deletion rather than by profiler attribution (a `settrace` line
profiler inflated anything making nested Python calls, so its numbers are
not reported).

**With a cheap estimator (ridge), 2.21 s total:**

| component | sec | % of run | % of NON-sklearn time |
|---|---:|---:|---:|
| `_preprocess` in the fold loop | 1.183 | 53.6% | **75%** |
| sklearn fit (16 folds + refit) | 0.641 | 29.1% | — |
| metrics | 0.189 | 8.6% | 12% |
| row selection + purge + bookkeeping | 0.193 | 8.8% | 12% |

Stable across scale (46.9 / 52.7 / 53.6 / 51.9% at 37.5k / 100k / 300k /
750k rows). **With a real estimator it is noise**: `hist_gradient_boosting`
spends 83.4% in the fit, `lightgbm_ranker` 68.8%. So this matters for the
LINEAR estimators and for hyperparameter search, which multiplies
fold-preparation by `n_candidates x inner_splits` (search: preprocessing
63%, sklearn 19%).

**And two thirds of `_preprocess` is not arithmetic.** Replaying the same
16 folds standalone: 0.839 s, of which **0.288 s (34%) is the two C++
kernels and 0.551 s (66%) is pandas-to-numpy materialisation.**
`engine.py:232` evaluates `train_frame[feature_ids]`, `:234` evaluates it
again, and `_native_matrix` runs `ascontiguousarray(to_numpy(float64))` on
each -- the same 100k x 20 block is built as a fresh float64 matrix twice
per fold, plus once for test. Building the matrix once before the loop and
gathering fold rows from it measured **1.97x on preprocessing and 2.63x on
the whole fold-prep pipeline, outputs bit-identical** (rtol 1e-9 on
train_X/test_X/train_y/test_y/weights across all 16 folds).

That is the shape of this whole subsystem: **the kernels are already only a
third of the cost of the path they sit inside.** Porting more of it would
attack the smaller half.

Other engine findings, all vectorisation or deletion:

| site | finding | gain |
|---|---|---|
| `validation/ranking.py:145` `ndcg_at_k` | the one genuine per-date Python loop, two `sort_values` per date, and the whole loop **reruns per k** (`ndcg_at=(5,10)` costs +113% for identical sorts) | **5.7-44.6x** |
| `engine.py:324` + `:327` | `adapter.metrics` computes pearson AND spearman `cross_sectional_ic`, then `adapter.fold_ic` **recomputes both with identical arguments** | 0.9% regression / 4.0% ranking, free |
| `validation/search.py:160` | `inner_train`/`inner_test` rebuilt per candidate although the slices do not depend on params | 75.0 -> 13.2 ms per outer fold |
| `validation/weights.py:110` | `pd.factorize(entities)` per fold on panel-invariant object strings: 3.87 ms/fold against 1.35 ms for the kernel it feeds | 75% of that function |
| `engine.py:420` + `:451` | fold rows taken through pandas including object and datetime columns, then a **second full copy** to drop purged rows | 11% |

**Measured negatives in the engine, recorded so they are not "fixed":**
the splitters and the purge/embargo contain no per-row iteration at all
(`np.arange`/`np.flatnonzero`, microseconds); `cross_sectional_ic`'s native
path wins at every size down to 150 rows; both row-count gates
(`_NATIVE_MIN_ROWS = 50_000`, `_POOLED_NATIVE_MIN_ROWS = 5_000`) are
correctly calibrated against measurement; the preprocessing kernels need no
gate at all (6x-200x from 300 rows up). An apparent 0.78x loss for pooled
pearson at 50k rows did not survive an interleaved 15-rep re-run (1.30x) --
**it was noise, not a defect.**

**No new C++ kernel is justified anywhere in the engine.**

## 4. Measured and rejected

Recorded so the next person does not re-derive them. *A fast path that is
slower is a defect* — these would have been.

| proposal | measured |
|---|---|
| lexsort shuffle in `permutation_test_ic` | **0.14x** |
| rank-once numpy in `permutation_test_ic` | **0.6x** |
| numpy rewrite of `_rank_within_date` | 0.97x at 3 models, **0.96x at 5** |
| numpy within-entity shifts in `lead_lag_ic_curve` | **0.53x** |
| `panel[features].isna().sum()` in `external_panel.py:257` | **0.33x** — materialises the bool frame |
| numpy block build in `expand_lags` | slower than grouped `shift` — `shift` is already a memcpy |
| `lexsort` + `take` for `sort_values(["date","entity"])` | 1.0x — memory-bandwidth bound, not a sort |
| `standardize_by_date` kernel for `portfolio_eval` | 12.9 ms vs numpy 16.2 ms — and wrong, see §1.1 |
| C++ for entity-scope features | un-ported ones are 0.23-5.9 ms/symbol, no slower than ported ones |

---

## 5. Infrastructure gap: the previous plan's figures are not reproducible

`modeling_native_plan.md` instructs: *"Reproduce every figure with
`python tests/bench/bench_modeling.py`."* That file exists and has four
sections, and **`HAS_CPP` appears nowhere in `tests/bench/`** — no script
can toggle the native path. No `pytest.mark.benchmark` test covers any of
the five modeling kernels either. So §6's multiples (5.5x-53.6x) cannot be
re-derived from committed code.

Two further gaps: the six speedup gates in `test_cpp_hurst.py` are marked
both `benchmark` and `slow`, and `build-cpp.yml` runs
`-m "not integration and not slow"`, so they are deselected despite a
comment saying they are included. And `optimization_plan.md` §8 asked for a
job failing on a >20% kernel regression; none of the four workflows has one.

**Any kernel added under this plan must ship with a `HAS_CPP`-toggling
benchmark**, or it inherits the same unverifiable status.

---

## 5.1 Status

Steps 1, 2 and the first three of step 3 are **done** (commits `ad5f245`,
this one). Achieved against the estimates above:

| item | estimated | achieved | output |
|---|---|---|---|
| §1.1 `standardize_by_date` NaN | — | fixed, both backends | contract test added |
| §1.2 `mst_degree` zero edge | — | fixed via dense Prim | regression tests added |
| §1.3 lag inf check | — | fixed | |
| §3.2 `transform_predictions_to_weights` | 3,545x* | **6.4x / 7.3x** | bit-identical |
| §3.1 `_quantile_shape` | 7.8x | **8.0-37.9x** | identical buckets |
| `sizing.zscore_normalized` | 3.2x | **224x** on 1x2000 | 3.2 ULPs |

\* the 3,545x estimate was for a full numpy rewrite that removes the group
loop entirely. What was done instead is the three targeted fixes inside it,
which keep the loop and every existing sizing method untouched. The
remaining gap is available if the loop is ever removed.

Since done as well:

| item | estimated | achieved | output |
|---|---|---|---|
| §3.1 engine `_preprocess` | 2.63x* | **10-11% end to end** | predictions hash identical |
| §3 item 6, meta-only load | "free" | **0.571 s -> 0.000 s** | see caveat |

\* 2.63x was fold PREPARATION measured in isolation. Fold preparation is
not the whole run, and 10-11% is what a caller experiences. Both numbers
are right about different things. The larger change the analysis proposed
-- building the feature matrix once before the loop and gathering fold rows
out of it -- is untaken and still available.

The meta-only load was applied to `validate_model_spec` and deliberately
NOT to `check_leakage`: that tool's own note says the coverage figures
"confirm the panel is the one that was built", which is only true because
the hash was checked. It was not a free fix on both sites.

`network.py` is done too: `avg_correlation` 5.605 -> 2.164 s and
`mst_degree` 6.105 -> 2.549 s at 2520 x 500, checksums identical. The
analysis reported the masked-correlation replacement as agreeing to
5.55e-16; on price-level data it returns **inf** and disagrees on which
pairs are NaN, because `E[x^2] - E[x]^2` cancels into a negative variance.
Centring per column fixes the inf and every NaN pattern and still leaves
1e-8 at that scale, which is the input's own precision rather than anyone's
bug -- so the fast path DECLINES when a column's mean exceeds a thousand
times its spread and the caller falls back to pandas.

**§2.1 `rank_by_date` is built.** 55.7 ms -> 3.4 ms on 100,000 rows by 3
columns, **16.5x**, against an estimate of 30-45x taken from a proxy. It
ships with the `HAS_CPP`-toggling benchmark §5 requires, which is the gate
the previous native work never had. Wired into all three call sites,
including `apply_cross_sectional_target` -- the operation the original
plan's phase 2 tabulated and left unbuilt, so that phase is now complete.

**Still open:** §2.2 `permutation_test_ic`, and the untaken engine change of
building the feature matrix once before the fold loop.

## 6. Order of work

1. **§1.1** `standardize_by_date` NaN semantics, both backends, plus a test
   that asserts the *contract* rather than backend agreement.
2. **§1.2** `mst_degree` zero-distance edge, and **§1.3** the lag inf check.
3. **§3** items 2, 1, 3 — the three that save tens of seconds, all numpy.
4. **§3** item 6's free half: stop loading the panel to read `meta`.
5. **§2.1** `rank_by_date`, with the benchmark §5 requires. Three call
   sites, one of them Phase 2's unfinished business.
6. **§2.2** `permutation_test_ic`, same discipline.
7. **§3** the remaining vectorisations, cheapest first.

Correctness before speed, and the two kernels last — because the
measurements say the pandas glue around them is worth more than they are.
