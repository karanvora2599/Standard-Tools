# Modeling Runtime (`standard_quant_tools.modeling`)

A second, independent runtime alongside the 46-tool
`standard_quant_tools.agent` analysis/backtest surface — not tool #47.
This document explains why that split exists, what's built in this first
phase, and what's deliberately deferred.

---

## Why a separate runtime, not a 47th tool

`agent/tools.py`'s `TOOL_CATEGORY` router and `Multi_Agent_Implementation/`'s
7-worker split (see
[Documentation/13_agent_orchestration.md](13_agent_orchestration.md))
exist specifically because handing an LLM 46 similarly-shaped tools on
every call causes selection ambiguity. Fitting/validating/registering a
statistical model doesn't fit that surface's shape at all — it isn't a
point-in-time snapshot (`analyze_stock_risk`) or a single backtest run
(`run_sma_backtest`); it's a small, ordered pipeline (build data → fit →
validate → register → score) that needs its own vocabulary. Adding it as
tool #47 would make the ambiguity problem worse, not better.

So `standard_quant_tools.modeling` is a **second registry**:
`modeling.agent.get_modeling_tools()` / `modeling.agent.modeling_dispatch()`,
with exactly 5 tools, never merged into `agent.get_agent_tools()` /
`agent.TOOL_CATEGORY`. It reuses this codebase's existing indicator/analysis
math, the Parquet artifact store (`backtest.artifacts`), and the audit
pipeline (`audit.dispatch._run_and_record`) — the shared deterministic
core stays one thing; only the agent-facing vocabulary is separate.

```
                    STANDARD TOOLS
                          │
           ┌──────────────┴──────────────┐
           │                              │
     agent.get_agent_tools()      modeling.agent.get_modeling_tools()
     (46 tools, 7 categories)     (5 tools, one pipeline)
           │                              │
           └──────────────┬───────────────┘
                          │
              shared core: data / indicators /
              analysis / metrics / audit / artifacts
```

---

## The 5 tools

| Tool | Input → Output |
|---|---|
| `list_features` | optional category filter → the feature catalog (id, description, params, temporal_support, scope, lookback) |
| `build_model_dataset` | `DatasetSpec` → fetches OHLCV, computes features + target, persists a Parquet panel, returns a `dataset_id` |
| `run_model_experiment` | `dataset_id` + `ModelSpec` → walk-forward fit + validate + register, returns a `model_id` + out-of-sample metrics |
| `score_model` | `model_id` + `as_of` + `universe` → predictions, persisted as a Parquet artifact |
| `inspect_model` | `model_id` + `view` (`summary` \| `feature_importance` \| `validation` \| `lineage`) → that slice of the registered model's manifest |

`run_model_experiment` doing fit+validate+register in one call is
deliberate: there is no separate "just fit" tool, so it's structurally
impossible to register a model that was never walk-forward validated.
`inspect_model` is one tool with four views instead of four separate
inspection tools, for the same reason `get_rally_signal` returns five
signal fields in one call instead of five tools.

### End-to-end example

```python
from standard_quant_tools.modeling.agent import (
    build_model_dataset, run_model_experiment, score_model, inspect_model,
    BuildModelDatasetInput, RunModelExperimentInput, ScoreModelInput, InspectModelInput,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec, FeatureSpec, TargetSpec, ModelSpec, EstimatorSpec, ValidationSpec,
)

spec = DatasetSpec(
    universe=["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    start="2018-01-01", end="2024-01-01",
    features=[
        FeatureSpec(id="technical.rsi"),
        FeatureSpec(id="market.momentum", params={"lookback": 63}),
        FeatureSpec(id="risk.realized_volatility"),
        FeatureSpec(id="risk.rolling_beta"),
        FeatureSpec(id="factors.pca_loading"),
    ],
    target=TargetSpec(horizon=20),
    benchmark="SPY",
)
ds_result = build_model_dataset(BuildModelDatasetInput(spec=spec))

model_spec = ModelSpec(
    task="regression",
    estimator=EstimatorSpec(type="elastic_net", params={"alpha": 0.01, "l1_ratio": 0.5}),
    # embargo does NOT need to cover the target horizon — training rows whose
    # forward-return label would resolve inside the test window are purged
    # separately, per row. See "Walk-forward validation" below.
    validation=ValidationSpec(train_window=756, test_window=63, embargo=5, min_folds=2),
    random_seed=42,
)
exp_result = run_model_experiment(
    RunModelExperimentInput(dataset_id=ds_result.dataset_id, spec=model_spec)
)
print(exp_result.oos_metrics)
# r2, mae, ic, rank_ic  — plus the ones that actually matter for a
# cross-sectional model:
#   cs_ic_mean / cs_ic_icir / cs_ic_hit_rate       (per-date IC, summarized)
#   cs_rank_ic_mean / cs_rank_ic_icir / ...
#   baseline_mae                                   (predict-the-mean comparison)
#   n_oos_rows / effective_sample_size             (overlap-adjusted)
print(exp_result.validation_report["n_folds_completed"], "of",
      exp_result.validation_report["n_folds_expected"])

# score_model is for dates AFTER training only — see "Model registry".
score_result = score_model(ScoreModelInput(
    model_id=exp_result.model_id, as_of="2024-06-01",
    universe=["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
))
print(score_result.predictions_uri)

print(inspect_model(InspectModelInput(model_id=exp_result.model_id, view="feature_importance")).data)
```

**Classification** works through the same five tools — build the dataset
with a `forward_direction` target rather than binarizing the panel yourself:

```python
spec = DatasetSpec(
    universe=[...], start=..., end=..., features=[...],
    # 1.0 when the 20-bar forward return exceeds `threshold`, else 0.0.
    # threshold=0.0 is plain up/down; a positive value asks for a move of at
    # least that size and deliberately imbalances the classes — check
    # positive_rate / majority_class_accuracy in the result before reading
    # accuracy.
    target=TargetSpec(type="forward_direction", horizon=20, threshold=0.0),
)
model_spec = ModelSpec(
    task="classification",
    estimator=EstimatorSpec(type="logistic", params={"C": 1.0}),
    validation=ValidationSpec(train_window=756, test_window=63, embargo=5),
    random_seed=42,
)
```

Task and target must agree: `regression` requires `forward_return` and
`classification` requires `forward_direction`. Both mismatches are rejected
up front. (A 0/1 target handed to a regressor is the dangerous one — it
fits happily and reports meaningless R²/IC.)

**Requesting one feature at several horizons** needs an `alias`, which names
the panel column:

```python
features=[
    FeatureSpec(id="market.momentum", params={"lookback": 20},  alias="mom_20"),
    FeatureSpec(id="market.momentum", params={"lookback": 252}, alias="mom_252"),
]
```

Without an alias both would produce a `market.momentum` column and the spec
is rejected. `alias` defaults to the feature id, so single-use specs are
unaffected.

Routed through an LLM tool call, the same pipeline goes through
`modeling.agent.modeling_dispatch(tool_name, arguments)` — the exact
mirror of `agent.tools.dispatch()`, reusing `audit._run_and_record` as-is
so every modeling call is still audit-logged (the `ModelSpec`/`DatasetSpec`
hashes ride in the existing `DecisionRecord.input` payload — no separate
audit implementation).

---

## The feature catalog

`modeling.features.registry.FEATURE_REGISTRY` — 21 built-in features,
each a thin wrapper over a primitive that already exists elsewhere in
this codebase, not new indicator math:

| id | wraps | scope |
|---|---|---|
| `technical.rsi` | `indicators.momentum.rsi` | entity |
| `technical.adx` | `indicators.trend.adx` | entity |
| `technical.macd_histogram` | `indicators.trend.macd` — `Histogram` column | entity |
| `technical.stochastic_k` | `indicators.momentum.stochastic_oscillator` — `Stoch_K` column | entity |
| `technical.williams_r` | `indicators.trend.williams_r` | entity |
| `market.momentum` | trailing `pct_change` | entity |
| `market.new_high_breakout` | look-ahead-safe Donchian breakout (same `.shift(1)` convention as `backtest.strategies`/`analysis.rally`) | entity |
| `market.psar_trend` | `indicators.trend.parabolic_sar` — `Trend` column (±1; the raw `SAR` price level isn't cross-sectionally comparable) | entity |
| `risk.realized_volatility` | `metrics.volatility_estimators.yang_zhang_volatility` | entity |
| `risk.rolling_beta` | `analysis.regression.rolling_beta` against `DatasetSpec.benchmark` | entity |
| `risk.atr_pct` | `indicators.volatility.wilder_atr` ÷ Close (normalized — raw ATR is a price level) | entity |
| `risk.bollinger_pct_b` | `indicators.volatility.bollinger_bands` — %B, Close's position within the bands | entity |
| `risk.parkinson_volatility` | `metrics.volatility_estimators.parkinson_volatility` | entity |
| `risk.garman_klass_volatility` | `metrics.volatility_estimators.garman_klass_volatility` | entity |
| `risk.rolling_drawdown` | trailing `.rolling(window).max()` peak, **not** `metrics.risk_metrics.drawdown_series` (that function's whole-series `cummax()` gives a stale peak inside a multi-year training window) | entity |
| `volume.mfi` | `indicators.volume.mfi` | entity |
| `volume.obv_roc` | `indicators.volume.obv` change over `lookback`, normalized by the volume traded in that window (raw OBV is unbounded/cumulative). **Not** `obv.pct_change()` — OBV is seeded at exactly 0 and crosses zero freely, so a ratio against it blows up | entity |
| `volume.vwap_deviation` | `indicators.volume.vwap` — `(Close - VWAP) / VWAP` | entity |
| `statistical.hurst` | `analysis.hurst.rolling_hurst` | entity |
| `factors.pca_loading` | `analysis.pca.pca_returns` — entity's loading on PC1, refit every `refit_every` bars | universe |
| `factors.pca_factor_return` | same PCA fit, projected onto each date's realized return — a shared macro factor | universe |

The three `volume.*` features are the only ones that need the OHLCV
panel's `Volume` column — every other feature only needs Open/High/Low/Close.

**Two scopes** exist because PCA needs the whole universe's return panel
at once, not one symbol's OHLCV. `entity`-scope features get
`fn(ohlcv, context, **params) -> pd.Series`, called once per symbol.
`universe`-scope features get `fn(returns_panel, context, **params) -> pd.DataFrame`
(dates × entities), called once for the whole `DatasetSpec.universe`.
`dataset.builder.build_dataset` dispatches each feature to the right path
and merges both into one long `(date, entity, <feature_ids>, target)` panel.

PCA is refit only every `refit_every` bars (default 21, ~1 trading
month) over a trailing `window`-bar panel, not on every single date —
refitting daily would be wasted work for a value (a factor's own
composition) that barely moves day to day. `factors.pca_factor_return`
projects each day's realized return onto the currently-held loadings, so
it still updates every bar even though the loadings only change at each
refit.

Each refit calls `pca_returns(..., n_components=1, method="power_iteration")`
rather than the default full SVD — since only PC1 is ever needed here,
solving for just that one component (power iteration + deflation) is 12–45×
faster than computing every singular triplet.

Power iteration verifies its own answer: it checks the eigenpair residual
`‖Av − λv‖` and **falls back to SVD** when that fails, so it is never a
different definition of PCA, only a faster route to the same one. It also
starts from a fixed pseudo-random vector rather than the uniform
`[1,…,1]/√n`.

That start vector matters more than it sounds. "Power iteration converges
from almost any start" excludes starts orthogonal to the dominant
eigenvector — and the uniform vector is exactly orthogonal to a
spread/long-short factor with loadings proportional to `[1,−1,…]`, one of
the commonest structures in real return data. With the old uniform start the
first matrix-vector product was the zero vector and the routine reported the
zero-eigenvalue direction as PC1: on a two-asset spread, an explained-variance
ratio of 0.0001 where SVD gave 0.9999. Not a precision difference — the wrong
component. Fixed, and pinned by a test that asserts the two methods agree on
exactly that panel.

### Adding your own feature

`modeling.features.custom.register_feature` is the extension point — a
firm's proprietary alt-data, analyst scores, or internal signals register
through the exact same `FEATURE_REGISTRY` every built-in feature uses:

```python
from standard_quant_tools.modeling.features.custom import (
    FeatureDefinition, FeatureScope, TemporalSupport, register_feature,
)

register_feature(FeatureDefinition(
    id="firm.altdata.customer_growth",
    description="Proprietary customer-growth signal.",
    fn=my_feature_fn,               # (ohlcv, context, **params) -> pd.Series
    temporal_support=TemporalSupport.CURRENT_ONLY,  # see PIT-safety below
    scope=FeatureScope.ENTITY,
    lookback=0,
))
```

---

## Point-in-time safety

Every built-in feature is `TemporalSupport.PIT_SAFE` — price/volume-derived
only, so the formula is causal. `TemporalSupport.CURRENT_ONLY` exists
for features like today's fundamentals, where no point-in-time-safe
historical provider is wired up yet: using one in a multi-year training
dataset would silently leak future-only information into the past.

`dataset.leakage.check_point_in_time_safety` runs on every
`build_model_dataset` call and raises `PointInTimeViolation` (a
`ValidationError` subclass) if any requested feature is `CURRENT_ONLY`.
Nothing here registers a `CURRENT_ONLY` feature — the mechanism
is built now, not deferred, because retrofitting it after models already
exist that were (silently) trained on leaked data is much more expensive
than building the guardrail before the first fundamentals feature ships.

### PIT safety is a property of the resolved feature, not just its label

`TemporalSupport` describes the FORMULA. It says nothing about the
parameters the formula is called with, and that gap was exploitable:
`market.momentum` and `volume.obv_roc` pass `lookback` straight to
`Series.pct_change(periods=lookback)`, and pandas reads a **negative**
period as a **forward** window. `lookback=-20` made the feature at *t* read
`Close[t+20]` while its declared `PIT_SAFE` label — and therefore the
point-in-time gate — remained perfectly happy.

`features.params.resolve_params` now validates every feature call before
any data is fetched, deriving the rules from each feature's own
`default_params` so a newly registered (including custom) feature is
covered automatically:

- unknown parameter names raise a clean `ValidationError` naming the
  accepted ones, instead of a raw `TypeError` from inside the feature;
- values must match the type of the parameter's default;
- any bar-count parameter (`lookback`, `period`, `window`, `span`, …) must
  be a **positive whole number** — this is the rule that closes the
  negative-window leak;
- other numerics must be finite.

`FeatureDefinition.requires` is enforced too. It was previously
informational, so a provider frame missing `Volume` produced a raw
`KeyError` from inside whichever feature touched it first — naming the
column but not the feature, the symbol, or the fact that the provider was
at fault.

### What PIT safety here does *not* cover

Two separate properties are involved, and only the first is checked:

1. **The formula is causal** — `TemporalSupport`, enforced as above.
2. **The underlying dataset is true point-in-time data** — *not* checked.
   The data layer already records this: `DataSetMetadata.point_in_time` and
   `survivorship_free`, both of which the default yfinance provider reports
   as **false**. The modeling PIT gate never consults them.

So a model trained on the current ticker universe can still carry
survivorship bias, and a provider that silently revises history would not be
detected here. Treat `PIT_SAFE` as "this formula doesn't look forward", not
"this dataset is point-in-time correct".

---

## Estimators

`modeling.estimators.registry.ESTIMATOR_REGISTRY` — an explicit allowlist,
keyed by `(task, name)`, all from `scikit-learn>=1.3.0` (already a core
dependency — no new install):

- **regression**: `linear`, `ridge`, `lasso`, `elastic_net`, `hist_gradient_boosting`, `random_forest`, `gradient_boosting`
- **classification**: `logistic`, `hist_gradient_boosting`, `random_forest`, `gradient_boosting`

`engine.run_experiment` refuses any estimator type not in this registry —
no arbitrary `sklearn` import, no `exec()`. An LLM builds a declarative
`ModelSpec`; the engine decides exactly how it executes.

### Parameter values are bounded, not just named

An allowlist of parameter *names* is not a compute budget. `n_estimators`,
`max_iter` and `max_depth` were previously unbounded, so a single tool call
could request `n_estimators=10_000_000` and pin CPU and memory for as long
as sklearn kept fitting — an agent-triggerable denial of service.

`estimators.bounds` gives each parameter a typed `ParamBound` (kind, range,
choices), and `validate_params` checks values as well as names:

| Parameter | Bound |
|---|---|
| `n_estimators` | 1 – 2,000 |
| `max_iter` | 1 – 100,000 |
| `max_depth` | 1 – 64 (or `None` for unlimited) |
| `learning_rate` | 1e-6 – 10 |
| `alpha` / `C` | 0 (or 1e-9) – 1e9 |
| `l1_ratio` | 0 – 1 |

The ceilings are deliberately generous — they exist to stop a runaway
request taking the process down, not to express an opinion about good
hyperparameters, and a test asserts realistic values still pass so the
guard cannot quietly become an obstruction. Non-finite values, wrong types,
fractional counts, and `True` passed as a count (bool subclasses int) are
all rejected.

### Incompatible combinations are caught here, not inside sklearn

`penalty` used to be exposed for logistic regression without `solver`, so
`penalty="l1"` reached the default `lbfgs` solver — which doesn't support
it — and raised from deep inside `.fit()`. `solver` is now exposed
alongside it and the pair validated against sklearn's own compatibility
matrix, so `l1` and `elasticnet` remain usable rather than being silently
restricted:

- `penalty="l1"` requires `solver` in `{liblinear, saga}`
- `penalty="elasticnet"` requires `solver="saga"` **and** `l1_ratio`

`register_estimator` also requires `overwrite=True` to replace an existing
entry, matching `register_feature`. It previously overwrote silently, which
made the registry gating what an agent may *run* weaker than the one gating
features.

---

## Walk-forward validation & the leakage discipline

`modeling.validation.walk_forward.WalkForwardSplit(train_window, test_window, embargo)`
yields `(train_positions, test_positions)` pairs over the dataset's
unique dates, walking forward one `test_window` at a time, with an
`embargo` gap between each fold's train and test window.

There are **two distinct leakage channels**, and the split only closes one:

| Channel | What leaks | Prevented by |
|---|---|---|
| Feature overlap | a test row's features drawn from the training window | the split itself, plus `embargo` |
| **Label overlap** | a *training* row whose forward-return target is only resolved by prices inside the **test** window | the target-overlap purge |

The second is the one that bit. `target[t]` reads `Close[t+horizon]`, so a
training row's label isn't finished until bar `t+horizon` prints — but
`WalkForwardSplit` is never given the horizon, only an integer `embargo`.
With `horizon=20, embargo=0`, the last 20 training labels were built from
test-period prices.

`engine.run_experiment` now purges any training row whose label reaches the
first test date, using a per-row `label_end_date` recorded at build time
rather than an integer offset. The timestamp matters: `horizon` counts an
**entity's own bars**, so with missing trading days or entities on
different calendars (a mid-history IPO, a halted symbol), `t+horizon`
entity bars is a different calendar date than `t+horizon` global panel
dates — an integer embargo under-purges exactly there. The count of purged
rows is reported as `n_train_rows_purged_overlap` rather than applied
silently: a large value means the horizon is consuming a real fraction of
each training window, which changes how you read the metrics.

**`embargo` therefore does not need to cover the horizon.** It remains
useful for feature-side lookback bleed.

Preprocessing (`features.transforms.fit_preprocessing` — per-column
winsorize bounds + zscore mean/std) is fit **on each fold's training rows
only**, then applied unchanged to that fold's test rows — never refit on
test.

### What the metrics mean

For a cross-sectional model, the pooled `ic`/`rank_ic` across every
`(entity, date)` row conflate two unrelated things: whether the model ranks
names against each other on a given day, and whether it tracks the market's
overall level across days. A model with **no** cross-sectional skill can
post a pooled IC above 0.9 purely by following the market factor. The
per-date `cs_ic_*` / `cs_rank_ic_*` family is what to judge a
cross-sectional model on; the pooled values are kept for continuity.

- `cs_ic_mean` / `cs_rank_ic_mean` — IC computed within each date's
  cross-section, then averaged over dates.
- `cs_ic_icir` — mean ÷ std. Distinguishes a 0.03 IC that is positive on
  most days from a 0.03 IC driven by a handful of extreme ones; the mean
  alone cannot.
- `cs_ic_hit_rate` — fraction of dates with positive IC.
- `baseline_mae` — the predict-the-mean model, so R²/MAE has something to
  be judged against.
- `effective_sample_size` — `n_oos_rows` discounted for target overlap. A
  `horizon`-bar target generated every bar produces labels sharing
  `horizon−1` of their bars, so 2,000 daily rows of a 20-day target carry
  roughly 100 independent observations per entity, not 2,000. A first-order
  correction, not a full Newey-West adjustment.
- For classification: `positive_rate` and `majority_class_accuracy` — the
  number `accuracy` has to beat. A 95/5 split scores 0.95 by always
  guessing the majority class.

Fold metrics are averaged **weighted by each fold's prediction count**, not
equally: a fold covering 30 predictions shouldn't influence the headline
number as much as one covering 3,000.

### Fold accounting

`validation_report` (also surfaced by `inspect_model(view="validation")`)
records expected vs completed vs skipped folds, each skip's reason and
date, fold coverage, the target horizon, purged-row count, and **per-fold**
metrics with train/test windows. A single averaged number cannot show
performance decay over time, reveal which regime drove the result, or
expose that one fold carried everything — and a run where 8 of 10 folds
were skipped previously looked identical to a clean 2-fold run.

`ValidationSpec.min_folds` (default **2**) is enforced against *completed*
folds. One surviving fold is a single train/test split, not walk-forward
validation — it cannot show whether performance holds across time, which is
the entire point. Lower it to 1 only for a deliberately short exploratory
run.

After validation, the registered model is refit on the **full** panel —
folds are for validation, deployment uses every observation. The
preprocessing stats from that final refit are persisted alongside the model
so `score_model` applies the identical transform, rather than refitting on
whatever universe happens to be in the scoring call.

---

## Model registry

```
SQT_RUNS_DIR/<model_id>/
    manifest.json            # task, estimator, features, target, dataset lineage,
                              # oos_metrics, validation_report, content_hashes,
                              # feature_implementation_hashes, dataset_spec_hash,
                              # train_end_date, git_commit_sha, package_version,
                              # random_seed, oos_predictions_uri
    model.joblib
    model_spec.json
    preprocessing_stats.json
    dataset_spec.json        # the model's OWN copy of its training spec
    oos_predictions.parquet  # walk-forward OOS predictions -- see "Backtesting a
                              # trained model" below
```

Same directory-per-id convention `backtest.artifacts.save_artifact`
already uses for backtest runs (`ds_...`/`mdl_...` id prefixes never
collide), written with the same atomic temp-file-then-`os.replace`
pattern and identifier/path-escape validation.

### The package is content-addressed and verified

Every artifact's SHA-256 is recorded in the manifest, and **every loader
verifies before use**. Without this the directory was a collection of
individually-atomic files, not a verifiable package: anything with write
access could edit an RSI period in `dataset_spec.json`, shift a mean in
`preprocessing_stats.json`, or swap `model.joblib`, and every later
`score_model` call would use the altered version while still reporting the
original `model_id`.

`model.joblib` is verified **before** `joblib.load`, which is the
load-bearing ordering — joblib/pickle deserialization executes code from
the file, so a swapped binary is an arbitrary-code-execution vector, not
merely a wrong-answer one.

> **This is integrity, not authenticity.** `manifest.json` is the root of
> trust and cannot contain its own digest, so an attacker able to rewrite
> *both* an artifact and the manifest is out of scope. Signing the manifest
> — as `audit/signing.py` already does for decision records — is what would
> close that, and is the right step before this registry is trusted across
> a trust boundary.

### Models are self-contained

The training `DatasetSpec` is **copied into the model's own directory** and
read from there. `score_model` previously reached back into
`SQT_RUNS_DIR/<dataset_id>/dataset_spec.json` on every call, so archiving
the dataset made an otherwise-valid model unscoreable, and editing that one
file silently redefined the features of every model trained from it. Models
registered before this fall back to the dataset directory with an explicit
warning rather than failing.

`manifest.json` is written **last** and is the commit point: every loader
keys off it, so a crash mid-registration leaves a directory that is simply
not a registered model rather than a half-written one that looks loadable.
`dataset_meta.json` plays the same role for datasets.

`feature_implementation_hashes` records a hash of each feature function's
own source. `git_commit_sha` pins the repo but says nothing about a feature
registered at runtime from a notebook or an internal package; sources that
can't be introspected record an explicit `"unavailable"` marker rather than
being silently absent.

### `score_model` is for dates after training only

The registered estimator is refit on the entire training panel, so asking
it to "predict" a date inside that panel returns a **future-trained
prediction dressed as a historical one**. `train_end_date` is recorded in
the manifest and an `as_of` at or before it is rejected. For genuine
historical evaluation use the walk-forward OOS predictions (below), which
are actually out-of-sample.

Score artifacts are named with a digest of the scored universe, not just
the date — scoring `[AAPL, MSFT]` and then `[AAPL, NVDA]` for the same
`as_of` used to overwrite the first file, leaving an earlier audit record
pointing at contents produced by a later call.

---

## Backtesting a trained model

`run_model_experiment` answers "how did this model do out-of-sample."
It doesn't answer "does this work as a trading strategy" — that requires
an actual backtest, and this codebase already has one
(`run_signal_panel_backtest`, in the *other* 46-tool registry).
`modeling.bridge.oos_predictions_to_signal_panel` connects the two —
a plain Python function, **not a 6th agent tool** (the 5-tool surface
stays exactly 5; this is the "artifacts, not tool calls" boundary between
the two registries).

**Why the bridge reads `run_model_experiment`'s output, not
`score_model`'s**: `score_model` produces a single as-of snapshot for
live/production scoring — using it to "predict" historical dates would
mean using the final, fully-trained model to score data it was trained
on, which is leakage and would produce a falsely optimistic backtest.
`run_model_experiment`'s walk-forward out-of-sample fold predictions come
from models that never saw their own fold's dates, with training rows whose
labels would have resolved inside the test window purged (see "Walk-forward
validation" above — that purge is what makes the guarantee hold; fold
construction alone does not). They already span the whole dataset's date
range, so they're persisted (`oos_predictions.parquet`,
`RunModelExperimentResult.oos_predictions_uri`) specifically for this use.

**Why the bridge converts to `DIRECTION`, not `SCORE`**: investigated
`run_signal_panel_backtest` directly — it never normalizes a `SCORE`
value; the value is multiplied straight into
`strategy_return = lagged_signal * market_return` as a raw leverage
multiplier. A raw regression prediction like `0.02` passed through as
`SCORE` would become a ~2%-leveraged position — economically meaningless.
`oos_predictions_to_signal_panel` converts to `DIRECTION` instead (sign
of the prediction for regression, a thresholded positive-class
probability for classification), which is units-invariant regardless of
the model's prediction scale.

```python
from standard_quant_tools.agent.models import SignalPanelBacktestInput, SignalType
from standard_quant_tools.agent.tools import run_signal_panel_backtest
from standard_quant_tools.modeling.agent import run_model_experiment, RunModelExperimentInput
from standard_quant_tools.modeling.bridge import oos_predictions_to_signal_panel

exp = run_model_experiment(RunModelExperimentInput(dataset_id=ds.dataset_id, spec=model_spec))

# Prefer model_id=: it resolves the artifact AND the task from the manifest,
# so the two cannot disagree. Passing `task` by hand allows regression
# predictions to be thresholded as classification probabilities, which
# produces a nonsensical but perfectly valid-looking signal panel.
signal_panel = oos_predictions_to_signal_panel(model_id=exp.model_id)

result = run_signal_panel_backtest(SignalPanelBacktestInput(
    tickers=list(signal_panel.keys()),
    start_date="2018-01-01", end_date="2024-01-01",
    signal_panel=signal_panel,
    signal_type=SignalType.DIRECTION,
    # next_open, NOT the "close" default. Modeling features are computed
    # from bar t's own OHLC, so a signal dated t isn't knowable until t's
    # close has printed — filling at that same close is the look-ahead
    # run_strategy's own fill_price warning describes.
    fill_price="next_open",
))
print(result.portfolio_metrics["sharpe_ratio"])
```

The predictions artifact is validated on load — columns, emptiness, date
dtype, finiteness, and `(entity, date)` uniqueness. The last one mattered
most because it was silent: duplicate rows previously overwrote each other
in the output dict, producing a smaller but perfectly well-formed panel.

**Target horizon and holding period are different objects.** A 20-day
forward-return prediction converted to a daily direction signal is
re-evaluated every bar by `run_signal_panel_backtest`. That's a valid
strategy, but it is not "hold for 20 days", and nothing here enforces a
relationship between the two — choose the holding period deliberately
rather than inheriting it from the target.

---

## Error handling

Every failure mode below raises `standard_quant_tools.error.ValidationError`
(the codebase-wide exception for boundary/logic validation) *except*
Pydantic construction of a spec object itself (`DatasetSpec(...)`,
`ModelSpec(...)`, ...), which raises `pydantic.ValidationError` — the
normal, expected split between "this input shape is invalid" (Pydantic)
and "this input is well-formed but semantically wrong" (this codebase's
own `ValidationError`).

- **Ambiguous input, rejected at construction**: duplicate panel column
  names (the same feature twice without distinct `alias`es, or two
  colliding aliases), duplicate universe symbols, `start >= end`, a
  malformed date string anywhere a date is expected, an `alias` that
  collides with a reserved panel column.
- **Invalid feature parameters**: an unknown parameter name, a wrong type,
  or a non-positive bar count (`lookback`/`period`/`window`/…) is rejected
  before any data is fetched. The last is a leakage guard, not a nicety —
  pandas reads a negative period as a *forward* window.
- **Missing required columns**: a provider frame lacking a column a
  requested feature declares in `requires` names the feature, the symbol,
  and what the provider actually returned.
- **Out-of-range estimator parameters**: values outside their documented
  bounds, or an incompatible `penalty`/`solver` pair, are rejected at the
  modeling boundary rather than deep inside `sklearn`.
- **Data fetch failures**: a provider error or empty OHLCV for any
  universe symbol *or* the benchmark names the failing symbol in the
  error, rather than propagating an unattributed raw exception.
- **Non-finite feature values**: `build_model_dataset` runs
  `require_finite_array` over every feature/target column before
  returning — a degenerate computation producing `inf` is caught here,
  not left to fail confusingly inside `sklearn`.
- **Task/target mismatch**: `regression` requires a `forward_return`
  target and `classification` a `forward_direction` one; both directions
  are rejected before fitting. The regression-on-binary case is the
  dangerous one — it would otherwise fit happily and report meaningless
  R²/IC.
- **Degenerate walk-forward folds**: a fold whose training window is
  empty, whose training rows were entirely purged for target overlap, or
  (classification only) that lands entirely on one class, is skipped
  rather than failing the whole experiment. Skips are recorded with their
  reason in `validation_report`. `run_model_experiment` raises if *every*
  fold was skipped, or if fewer than `min_folds` completed.
- **Historical scoring**: `score_model` rejects an `as_of` at or before
  the model's `train_end_date` — the registered estimator saw those dates
  in training, so the "prediction" would be future-informed.
- **Tampered artifacts**: any model artifact whose content hash no longer
  matches the manifest is rejected on load, `model.joblib` before it is
  deserialized.
- **Partial scoring**: `score_model`'s `ScoreModelResult.missing_entities`
  lists any requested universe symbol that had no scoreable row as of
  `as_of` (e.g. insufficient history within `lookback_days`) — never
  silently absent from a result that otherwise looks complete.

## Audit and replay

Every modeling tool call routed through
`modeling.agent.modeling_dispatch` writes a `DecisionRecord`, using the
same `audit._run_and_record` the 46-tool surface uses — no parallel audit
implementation.

`audit.verify_replay` covers **both** surfaces: it resolves a record's tool
against the agent registry and then the modeling registry. (Each is looked
up lazily, since both tool packages import the audit package, and the
modeling runtime is deliberately independent of the 46-tool registry rather
than importable from it.)

Replay comparison for modeling is **semantic**, not literal. Modeling mints
a fresh `ds_`/`mdl_` identifier on every run and embeds it in artifact
paths, so a byte-identical re-run can never match the literal output hash —
every modeling replay would report a mismatch, which is worse than no
replay support because a false mismatch reads as evidence of drift. Records
therefore carry `output_hash_normalized` alongside `output_hash`: the same
output with those identifiers rewritten to a placeholder. Replay falls back
to it only when the literal comparison fails *and* the output actually
contains such an identifier, so deterministic tools keep their exact check.
Normalization is narrow — a changed metric, feature list or fold count
still surfaces as a mismatch.

Records written before that field existed report `output_match=None`
("not comparable") rather than `False`: their literal mismatch cannot be
distinguished from a genuine one, and reporting drift would be a false
accusation.

---

## Explicitly deferred

Not built here, and not accidentally half-built either:

- **Semantic feature search** — `list_features` is a plain catalog
  lookup. A 21-entry catalog doesn't need ranking; revisited only if the
  catalog grows large enough that it does.
- **Fundamentals / true point-in-time revision tracking** — the
  `CURRENT_ONLY` mechanism exists, but no fundamentals feature is
  registered, since none of the current data providers expose
  point-in-time-safe historical fundamentals. This is the honest blocker
  on the "analyze fundamentals → turn them into model features → train"
  workflow: it needs a PIT fundamentals provider first, not a feature
  wrapper over today's reported ratios.
- **Provider selection and interval** — `DatasetSpec` has no provider or
  interval field; modeling calls `DataFactory.get_provider()` and is
  effectively a daily-OHLCV system, even though the data layer supports
  both. Provider identity is likewise not recorded in model lineage.
- **Async / batched universe fetching** — the dataset builder loops
  symbols one at a time, while the repo already has async full-OHLCV
  portfolio fetching. This dominates runtime for large universes.
- **Time-varying universe membership** — universes are static ticker
  lists, with no as-of membership, so historical models over a
  present-day universe carry survivorship bias (which the default
  provider itself reports it cannot rule out).
- **Dataset coverage diagnostics** — rows before/after alignment, rows per
  entity, date coverage, missing-data rate, and which features caused
  drops. `BuildModelDatasetResult.warnings` exists as the natural home for
  these but is currently unused. Relatedly, `entities` reports the symbols
  FETCHED, not necessarily those that survived into the training panel.
- **Universe-scope features under a changed scoring universe** —
  `factors.pca_loading` / `pca_factor_return` are computed from the whole
  current universe, so a model trained on `[AAPL, MSFT, NVDA]` and scored
  on `[AAPL, XOM, JPM]` receives a different variable under the same
  feature id.
- **Scoring staleness** — `score_model` uses each entity's latest
  surviving feature row and reports the requested `as_of`, without
  exposing the effective observation date per entity or enforcing a
  maximum staleness.
- **Model lifecycle states** — registration means "persisted and
  validated enough to load", not "approved for production". There is no
  trained/validated/approved/production distinction.
- **Prediction transforms beyond sign** — the bridge is sign-only; rank,
  quantile, normalized-score, neutralized and volatility-scaled
  transformations are not built.
- **Hyperparameter tuning, custom estimator import, multi-model
  comparison tooling.**
- **Declarative preprocessing** — winsorize 1/99 + pooled z-score is
  hardwired for every estimator, though trees, linear and cross-sectional
  models want different treatment.
- **Extracting a codebase-wide generic `standard_quant_tools.artifacts`
  package** — `modeling.artifacts` reuses `backtest.artifacts` directly
  today; a shared package is only worth building once there's a second
  real pattern to generalize from, not speculatively.

See [Documentation/08_analysis.md](08_analysis.md) for the underlying
analysis primitives (`hurst_exponent`, `pca_returns`, `rolling_beta`,
...) every built-in feature wraps.
