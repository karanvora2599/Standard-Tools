# Modeling module: analysis, and where the next gains are

Scope: `src/standard_quant_tools/modeling/`. Everything below was measured on
this machine against a synthetic in-memory universe (1,500 business-day bars per
ticker, `DataFactory` patched so no measurement includes network time). Numbers
are best-of-N with the GC disabled.

**Status: all ten items are implemented and merged.** The assessment below is
kept as written, and section 5 records what each prediction turned out to be
worth — including the three that were wrong. Every figure here is reproducible
with `python tests/bench/bench_modeling.py`.

## 1. What the module is

Six agent tools over a five-stage pipeline:

    build_model_dataset -> run_model_experiment -> inspect_model
                        -> score_model -> evaluate_model_portfolio
                        (+ list_features)

- **21 features** in a registry, namespaced `technical.*` (5), `risk.*` (7),
  `market.*` (3), `volume.*` (3), `factors.*` (2), `statistical.*` (1).
- **11 estimators**, all scikit-learn: 7 regression, 4 classification.
- **2 targets**: `forward_return`, `forward_direction`.
- **1 validation scheme**: `WalkForwardSplit`, rolling only.
- Specs are Pydantic, content-hashed (`dataset_spec_hash`), and the run is
  written to an audit trail. The leakage controls are real and correctly
  built: an `embargo` between train and test, plus a target-overlap purge on
  `LABEL_END_COL` that drops training rows whose forward-return label extends
  into the test window. That is the part most home-grown pipelines get wrong,
  and this one gets it right.

The architecture is sound. The gaps are in breadth, not in correctness.

## 2. Performance: what the runtime actually costs

### 2.1 The headline — 72% of a walk-forward run is the scorecard, not the model

`run_experiment` with `ridge`, 50 entities, 73,400 panel rows, 19 folds:

| component | time | share |
|---|---:|---:|
| **total** | **3.892 s** | 100% |
| `_predict_fold` | 2.898 s | 74% |
| - of which `cross_sectional_ic` | 2.798 s | **72%** |

Two independent checks confirm the fit is not the cost:

- **Cost is nearly flat in universe size.** Ridge takes 1,892 ms at 5 entities
  and 2,072 ms at 50 — the panel grew 10x, the runtime grew 1.1x. A cost driven
  by the linear algebra would not behave like that.
- **Fold sweep** (ridge, 20 entities): 2 folds = 1,282 ms, 40 folds = 2,499 ms.
  That is roughly **1,180 ms fixed + 33 ms per fold**. The per-fold marginal
  cost — which is where the estimator lives — is small.

`cross_sectional_ic` groups by date in Python and calls `Series.corr` once per
date. For a 19-fold run over 252-ish dates per fold that is thousands of tiny
pandas calls, each with full Series-construction and dispatch overhead.

### 2.2 Vectorizing the IC is exact, and 2-72x

I prototyped two replacements and checked both against the current
implementation on 30 randomized panels, deliberately including forced ties
(the part that is easy to get wrong for Spearman) and constant cross-sections
(where the correlation is undefined):

    worst |diff| spearman : 1.110e-16
    worst |diff| pearson  : 2.220e-14

and on a ragged panel where entities enter and leave, so the cross-section size
changes date to date, and single-row dates must be dropped:

    spearman  dates= 37  worst |diff| 1.110e-16
    pearson   dates= 37  worst |diff| 1.066e-14

Those are floating-point identity, not approximation. Speed (Spearman):

| dates | entities | rows | current | segment (`reduceat`) | rectangular | segment x | rect x |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 63 | 50 | 3,150 | 22.9 ms | 0.88 ms | 0.54 ms | 25.9x | **42.2x** |
| 252 | 50 | 12,600 | 91.5 ms | 3.72 ms | 1.28 ms | 24.6x | **71.6x** |
| 252 | 500 | 126,000 | 117.2 ms | 55.3 ms | 23.6 ms | 2.1x | **5.0x** |
| 1000 | 500 | 500,000 | 449.3 ms | 256.8 ms | 207.6 ms | 1.7x | **2.2x** |
| 252 | 2000 | 504,000 | 544.0 ms | 250.5 ms | 102.8 ms | 2.2x | **5.3x** |

Read the shape of that table honestly: **the speedup shrinks as the
cross-section grows.** At 50 entities the Python loop overhead dominates and
removing it is worth 70x; at 500 entities each date has enough real work that
the per-date overhead amortizes and the win drops to 2-5x. The good news is
that the large multiples land exactly on the shape the walk-forward runtime
actually uses today.

The rectangular path (reshape a balanced panel to `(n_dates, n_entities)` and
reduce along axis 1) beat the segment path at **every** size tested, so the
right design is: rectangular when the panel is balanced, segment `reduceat`
as the fallback when it is ragged. Both are exact, so the fallback is not a
quality compromise.

### 2.3 `build_dataset` re-validates the same OHLCV once per feature per entity

Every feature wrapper is decorated with `@validate_series`, and several call
`require_positive_price_series`, which runs a `pd.to_numeric` plus a
non-positive scan. With 6 features per entity, the same `ohlcv["Close"]` gets
re-scanned 6 times after the builder has already fetched and column-checked it.

Measured by neutralizing the checks and re-timing (measurement only, not a
proposed fix — the checks are load-bearing at the public API boundary):

| entities | baseline | checks disabled | cost of re-validation |
|---:|---:|---:|---:|
| 20 | 503.3 ms | 496.8 ms | 1% |
| 50 | 1,173.7 ms | 1,031.9 ms | 12% |
| 100 | 2,425.4 ms | 1,979.3 ms | **18%** |

A correction worth recording: reading this off a cProfile listing first, I
called it 41%. That was wrong — cProfile's *cumulative* time on a decorator
includes the wrapped function's own body. The direct A/B says 12-18%, and the
direct A/B is the number to trust.

### 2.4 Scaling, and the estimator cliff

`build_dataset`, 1,500 bars:

- ~6.5 ms per entity at 6 features
- ~21.5 ms per entity at 16 features

Extrapolated (**projection, not measured**): a 2,000-ticker, 16-feature build is
roughly **43 s**. That is tolerable. The estimator is not:

- `random_forest` at 50 entities: **174 seconds** for one walk-forward run.

At 2,000 entities that is not a slow run, it is an unusable one. Tree ensembles
are the first thing that stops working at universe scale, and no amount of IC
vectorization touches it.

### 2.5 Smaller items

- The panel's `entity` column is `object` dtype — 73,400 x 10 costs 9.2 MB.
  As `category` it is a few bytes per row plus a small dictionary, and the
  per-fold `panel["date"].isin(train_dates)` and every groupby get faster.
- The per-fold train mask is rebuilt with `.isin()` over the whole panel each
  fold. Dates are sorted; `searchsorted` on the fold boundaries is O(log n).
- `technical_indicators_panel` (the C++ panel kernel added in the previous
  session, 11.9x over the per-ticker wrappers) is **not** wired into
  `dataset/builder.py`, which still loops per entity calling per-ticker
  wrappers. This is the single largest piece of already-built, already-tested
  work sitting unused.

## 3. Feature gaps

Each of these was confirmed by reading the code, not inferred.

### 3.1 Normalization is pooled, which is wrong for a cross-sectional model

`fit_preprocessing`/`apply_preprocessing` winsorize and z-score **pooled across
the whole training panel**. Meanwhile `zscore_cross_sectional` exists in
`features/transforms.py:33` and has **zero callers**.

This matters more than it looks. Pooled z-scoring leaves the market factor in
the features: on a day when everything rallies, every entity's momentum feature
is high together, and the model learns "the market went up" instead of "this
name is strong relative to its peers." For a model whose scorecard is
*cross-sectional* IC, the normalization should be cross-sectional too. The
function to do it is already written and sitting unused.

### 3.2 `effective_sample_size` is computed and then ignored

`engine.py:369` computes it and writes it into `oos_metrics`. Nothing acts on
it. There is **no `sample_weight` anywhere in the module** (grep: 0 hits). With
a 5-day horizon, overlapping labels mean consecutive rows are ~80% redundant;
the code knows this well enough to report it, but every row still enters the
fit with weight 1.

### 3.3 No hyperparameter search

Zero hits for `GridSearchCV`, `RandomizedSearchCV`, `param_grid`, `optuna`.
`ridge(alpha=1.0)` is whatever the caller typed. Given a walk-forward splitter
already exists, an inner-loop search on the training window is a natural fit
and would be leak-free by construction.

### 3.4 Walk-forward is rolling-only

`walk_forward.py:38-43`: fixed `train_window`, `start += test_window`. No
expanding/anchored window, no purged K-fold, no combinatorial purged CV. For
short histories the rolling window throws away usable data.

### 3.5 Narrow model and target surface

- All 11 estimators are sklearn. No LightGBM/XGBoost — which matters given
  §2.4, since `LGBMRegressor` on 73k rows is seconds where `random_forest` is
  minutes.
- No quantile regression, so no uncertainty band on a prediction.
- Targets are `forward_return` and `forward_direction` only. No
  volatility-scaled return, no market-neutral (residual) return, no triple
  barrier, no rank target — even though the evaluation is rank-based.

## 4. What I would do, in order

Ordered by measured value per unit of risk.

| # | change | expected | confidence | risk |
|---|---|---|---|---|
| 1 | Vectorize `cross_sectional_ic` (rect + reduceat fallback) | 2-72x on 72% of the run | **measured, exact** | very low — bit-identical |
| 2 | Wire `technical_indicators_panel` into `build_dataset` | up to 11.9x on the indicator share | measured in isolation | low |
| 3 | Cross-sectional normalization option | correctness, not speed | high | **changes results** |
| 4 | `sample_weight` from label overlap | correctness | high | changes results |
| 5 | LightGBM/XGBoost estimators | minutes -> seconds | high | new dependency |
| 6 | Expanding window + purged K-fold | more usable folds | high | low, additive |
| 7 | Inner-loop hyperparameter search | model quality | high | cost x grid size |
| 8 | `entity` as `category`, `searchsorted` fold masks | few % + memory | medium | very low |
| 9 | Richer targets (vol-scaled, residual, triple-barrier) | breadth | high | additive |
| 10 | Hoist validation out of the per-feature inner loop | 12-18% of build | measured | must keep API-boundary checks |

Items 1, 2, 8 and 10 are pure speed and change no numbers. Items 3 and 4 change
model output — they are improvements I would argue for, but they are a modelling
decision, so they belong behind an explicit spec field with the current
behaviour as the default rather than as a silent change.

The single highest-value item is #1: it is the largest share of the runtime, the
replacement is provably exact including ties and ragged panels, and it carries
essentially no risk.

## 5. What the predictions were actually worth

All ten shipped. Three of the estimates above were wrong, and they are corrected
here rather than edited into the text above, because the pattern is the useful
part: every one of the three was wrong in the same direction, and for the same
reason — a measurement taken in isolation was quoted as though it were the
end-to-end effect.

| # | Predicted | Measured | Verdict |
|---|---|---|---|
| 1 | 2–72× on 72% of the run | **44.8×/47.8×/5.3×/1.8×**; run 3.892 s → 1.577 s | as predicted |
| 2 | up to 11.9× on the indicator share | **~2×** on the feature phase; whole build noisy | **over-stated** |
| 3 | correctness, not speed | correctness — and 898 ms → 469 ms, so also faster | better than predicted |
| 4 | correctness | correctness; no speed claim made or found | as predicted |
| 5 | minutes → seconds | **62.9 s → 3.55 s (17.7×)** | as predicted |
| 6 | more usable folds | purged K-fold tests every date once vs. walk-forward's tail | as predicted |
| 7 | model quality | works; picked a *different* alpha per fold, i.e. it was fitting noise | see below |
| 8 | few % + memory | folded into the fold-mask change; not separately measurable | not isolated |
| 9 | breadth | four new target types | as predicted |
| 10 | 12–18% of build | **~6% at 16 features, ~0% at 6** | **over-stated** |

### The three that were wrong

**Item 10 was wrong before this document was even finished** — and the
correction is already recorded in §2.3, where the profile said 41% and the
direct A/B said 12–18%. It was *still* wrong at 12–18%. That A/B disabled
**all** validation, so it measured the cost of every check rather than the cost
of the repeated ones. Instrumenting the actual memo hit rate showed most checks
are first checks on distinct columns: the six-feature benchmark set barely
shares columns between features, so there was little repetition to remove. The
mechanism helps in proportion to how much the requested features overlap — 56.7%
of checks avoided and 5.8% of the build saved at 16 features, nothing measurable
at 6. Two successive over-estimates of the same item, each corrected by
measuring one level closer to the thing itself.

**Item 2 quoted 11.9×, which was the indicator kernels in isolation.** Indicators
are only part of a build — fetching, stacking, alignment and hashing are
unchanged — so the end-to-end effect was never going to be 11.9×. Attributing
time directly to feature computation gives a stable **~2×**. The whole-build
number is worse than that and, more importantly, is not reliably measurable on
this machine: repeated interleaved A/B runs of the same change returned ratios
from **0.62× to 1.39×**, a spread wider than the effect. An earlier run of mine
reported the fast path as *slower*, and a later one as 1.35×; neither was
trustworthy. `bench_modeling.py` now reports the attributed feature-phase time
instead, and says why.

**Item 7 works, and the interesting result is not the speedup.** The search
selected a different `alpha` on most folds — `[10000, 0.01, 10000, 0.01, 100,
100, 100]`. That is the search fitting noise, and it is visible only because the
report keeps every candidate's score per fold instead of collapsing to one
"best" value. A tuned hyperparameter that changes every fold is a warning, not a
result.

### Two bugs the implementation surfaced

Neither was predictable from reading the code:

- The first LightGBM wrapper subclassed `LGBMRegressor` with a `**kwargs`
  `__init__` to pin `verbose=-1`. sklearn's `get_params()` reads parameter names
  off the `__init__` signature, so `random_state` silently vanished and the fit
  then failed inside LightGBM. The same defect was in the quantile booster.
- `triple_barrier` was first encoded `{0.0, 0.5, 1.0}`. sklearn reads a float
  target with those values as **continuous** and refuses to fit any classifier
  to it, so the obvious encoding does not work at all. Re-encoded as three
  integer classes with "up" deliberately at `1`, so `positive_class_proba` keeps
  returning P(up) — any ordering putting "neither" at class 1 would hand the
  downstream signal path P(nothing happened).

### What did not need changing

The leakage controls. The embargo plus the `LABEL_END_COL` target-overlap purge
were correct as found, and the only change made to them was generalizing the
purge from "label ends before the test starts" to "the label's span overlaps the
test block" — bit-identical under walk-forward, and required only because purged
K-fold puts training rows on both sides of the test window.
