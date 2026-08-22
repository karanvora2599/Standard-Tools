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
with exactly 6 tools, never merged into `agent.get_agent_tools()` /
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
     (46 tools, 7 categories)     (6 tools, one pipeline)
           │                              │
           └──────────────┬───────────────┘
                          │
              shared core: data / indicators /
              analysis / metrics / audit / artifacts
```

---

## The 6 tools

| Tool | Input → Output |
|---|---|
| `list_features` | optional category filter → the feature catalog (id, description, params, temporal_support, scope, lookback) |
| `build_model_dataset` | `DatasetSpec` → fetches OHLCV, computes features + target, persists a Parquet panel, returns a `dataset_id` |
| `run_model_experiment` | `dataset_id` + `ModelSpec` → walk-forward fit + validate + register, returns a `model_id` + out-of-sample metrics |
| `score_model` | `model_id` + `as_of` + `universe` → predictions, persisted as a Parquet artifact |
| `inspect_model` | `model_id` + `view` (`summary` \| `feature_importance` \| `validation` \| `lineage`) → that slice of the registered model's manifest |
| `evaluate_model_portfolio` | `model_id` + `PredictionTransformSpec` + `PortfolioSimSpec` → OOS predictions turned into target weights and simulated as one shared-cash account, returning Sharpe/drawdown/turnover/exposure plus a persisted weights artifact |

`run_model_experiment` doing fit+validate+register in one call is
deliberate: there is no separate "just fit" tool, so it's structurally
impossible to register a model that was never walk-forward validated.
`inspect_model` is one tool with four views instead of four separate
inspection tools, for the same reason `get_rally_signal` returns five
signal fields in one call instead of six tools.

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

**Classification** works through the same tools — build the dataset
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

Task and target must agree — see **Targets** below for the full set.
Historically this was one type each: `regression` requires `forward_return` and
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
| `market.new_high_breakout` | look-ahead-safe Donchian breakout (same `.shift(1)` convention as `backtest.strategies`/`analysis.rally`). Warm-up bars are NaN, not 0.0 — `NaN > x` is False, so `.astype(float)` alone asserted "no breakout" for every bar before the window existed | entity |
| `market.psar_trend` | `indicators.trend.parabolic_sar` — `Trend` column (±1; the raw `SAR` price level isn't cross-sectionally comparable) | entity |
| `risk.realized_volatility` | `metrics.volatility_estimators.yang_zhang_volatility` | entity |
| `risk.rolling_beta` | `analysis.regression.rolling_beta` against `DatasetSpec.benchmark` | entity |
| `risk.atr_pct` | `indicators.volatility.wilder_atr` ÷ Close (normalized — raw ATR is a price level). A non-positive Close yields NaN rather than ±inf | entity |
| `risk.bollinger_pct_b` | `indicators.volatility.bollinger_bands` — %B, Close's position within the bands. A flat window collapses both bands onto the mean, where %B is 0.5 (the middle band) rather than 0/0 | entity |
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

That verification had a hole worth naming, because it is the same failure
the fixed start vector was introduced to prevent. The residual check was
*skipped entirely* whenever the eigenvalue came out ≈ 0 — a zero matvec was
read as a legitimate zero eigenvalue, which sets `‖Av − λv‖ = 0` and passes
trivially. A zero eigenvalue is only legitimate when there is no remaining
variance to find; otherwise the iteration landed on a null direction while
real structure was still there, and the fallback that exists for exactly
that case never fired. The check now discriminates on **remaining
variance**, so a null direction found in a matrix that still has variance is
a convergence failure. Confirmed to add no SVD fallbacks across the existing
suites (0 before, 0 after) — it closes a hole rather than trading speed for
safety.

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

**The output contract is enforced, and names your feature when it isn't
met.** An `ENTITY`-scope `fn` must return a `pd.Series` indexed by a
**subset** of the bars it was given; a `UNIVERSE`-scope `fn` must return a
`pd.DataFrame` with one column per entity, indexed like the shared returns
panel. These were documented but unchecked — the return value went straight
into a `DataFrame` constructor, so a wrong type or a foreign index surfaced
as a generic pandas error several frames away with no mention of which
feature caused it.

Subset, not equality, is deliberate: a feature legitimately produces fewer
rows than it consumes (`risk.rolling_beta` works from returns, which lose
the first bar to `pct_change`). Panel assembly is index-aligned, so the
absent bars become NaN and the existing alignment step handles them. What is
rejected is an index carrying labels the entity does not have — those either
invent rows or indicate the feature was computed against the wrong entity
entirely.

`id` may not be one of the reserved panel columns (`date`, `entity`,
`target`, `label_end_date`). `FeatureSpec.alias` was already checked; the id
is equally the output column name when no alias is set, so a feature
registered as `id="target"` produced a column shadowing the panel's
supervised target. Both now read the same `RESERVED_PANEL_COLUMNS` — two
independent copies is how the id path drifted from the alias path in the
first place.

---

## Where the data comes from

`DatasetSpec` names the provider and the bar interval. Both were
previously implicit — the builder called `DataFactory.get_provider()` with
no arguments and fetched whatever that returned at whatever interval it
defaulted to, so the runtime was a yfinance-daily system by accident and
no model recorded which source it had been trained on.

```python
DatasetSpec(
    universe=["AAPL", "MSFT", "NVDA"],
    start="2015-01-01",
    end="2024-12-31",
    provider="polygon",     # "yfinance" (default) | "polygon" | "bloomberg"
    interval="1d",          # default
    features=[FeatureSpec(id="technical.rsi")],
    target=TargetSpec(horizon=20),
)
```

Both fields live on the spec, so both are covered by `spec_hash`, bundled
into the model, and reused verbatim when scoring — a model trained on one
provider will not silently score against another.

**Credentials are deliberately not spec fields.** The spec is written to
disk, hashed into model lineage and embedded in decision records, so an
`api_key` field would leak the key into all three. Providers read their
own credentials from the environment (`SQT_POLYGON_API_KEY`).

**Interval is validated by the provider, not here.** The supported set
genuinely differs — `BloombergProvider` rejects intraday outright — so
duplicating a union of them in `DatasetSpec` would only drift. Note that
`target.horizon` and every feature lookback count BARS of the chosen
interval, and that the built-in defaults are calibrated for daily bars:
`window=252` is one trading year at `1d` and about six weeks at `1h`.
Non-daily intervals are fetched correctly and warned about, not rescaled.

#### Interval-aware annualization

Annualization is the exception: it *is* rescaled. `risk.realized_volatility`
(Yang-Zhang), `risk.parkinson_volatility` and `risk.garman_klass_volatility`
all used to multiply by `sqrt(252)` regardless of interval, so weekly bars
were reported at roughly 2.2× their true annualized volatility. Harmless for
a standardized model whose ranking is unaffected by a constant factor,
misleading the moment anyone reads the value.

`FeatureContext` now carries the interval and each feature scales by its
own constant: 252 for `1d`, 52 for `5d`/`1wk`, 12 for `1mo`, 4 for `3mo`.

**Intraday intervals raise rather than guess.** There is no correct constant
without knowing the session length, which is venue-specific — 6.5 hours on
US equities, 23 on CME futures, 24 on crypto — and not derivable from the
interval string. Inventing one would produce a number that looks
authoritative and is wrong by whatever the venue mismatch happens to be. A
missing interval still means daily, so callers that never passed one are
unaffected.

The universe is fetched **concurrently**, through each provider's
`get_ohlcv_async`. Bounded by `SQT_MODELING_FETCH_CONCURRENCY` (default
8) — high enough to matter for a large universe, low enough not to look
like abuse to a public endpoint. Two details worth knowing:

- Every symbol is awaited to completion and **all** failures are reported
  in one message, sorted. `asyncio.gather`'s default propagates only the
  first exception and abandons the rest, which would mean fixing a
  universe one bad ticker per run, in nondeterministic order.
- Called from inside a running event loop (a notebook, an async agent
  runtime, a web handler), it falls back to a sequential fetch instead of
  raising — `asyncio.run` refuses to nest. The same fallback covers a
  duck-typed provider that implements only `get_ohlcv`. Both paths report
  failures identically, so the error you see does not depend on which ran.

### Coverage and provenance warnings

`build_model_dataset` returns a `warnings` list — a field that had existed
from the start and was never populated. These are conditions that change
how the resulting OOS metrics should be read but are not grounds to refuse
to build the dataset:

| Warning | What it means |
|---|---|
| `point_in_time=False` | The provider does not guarantee historical values are never revised, so a feature computed today may differ from what was observable on its label date. The per-feature PIT gate checks the FORMULA; it cannot see revisions in the underlying series. |
| `survivorship_free=False` | Delisted securities are not queryable, so any universe of currently-listed symbols is a survivors-only sample. Backtested returns are biased upward and walk-forward validation does not correct for it. |
| partial history | A symbol covers materially less than the universe's date range — it listed inside the window, or stopped early. It is weighted far less than its presence in `universe` suggests. |
| requested start/end unavailable | The window that came back is shorter than the one asked for, before any feature lookback is consumed. |
| PCA intersection | Universe-scope features need a complete cross-section, so one short history truncates the panel *for every entity*. The warning names the latest-starting symbol, which is usually the whole explanation. |
| non-daily interval | Feature default *parameters* are stated in trading days and are not rescaled (above). Annualization is no longer part of this caveat — see below. |
| provider guarantees undetermined | `get_metadata` was unavailable or failed, so the point-in-time and survivorship caveats could not be checked at all. Deliberately **not** the same as an empty warning list: absence of evidence is reported as absence of evidence. |

Every provider this package ships reports `point_in_time=False` and
`survivorship_free=False` honestly, which is why these are warnings rather
than errors: promoting them to a hard failure would make the runtime
unusable against its own default data source while teaching the caller
nothing. The list is empty for a provider making both guarantees over
aligned daily histories, so a non-empty one means something.

### What alignment dropped, and why

A row survives into the panel only when every feature and the target are
present for it. That loss is normal — each feature consumes its lookback
window, the forward-return target consumes its horizon — and it used to be
reported as a final row count and nothing else, which cannot separate the
warm-up you asked for from one feature quietly eating the panel.

`build_model_dataset` returns `drop_attribution`: rows before and after,
per-entity drop counts, and **two** counts per column.

| Count | Meaning |
|---|---|
| `n_missing` | rows where that column was NaN |
| `n_sole_missing` | rows where it was the **only** thing missing |

The second is the actionable one. Warm-up windows overlap, so `n_missing`
sums to far more than the rows actually lost, and a short-lookback feature
sitting entirely inside a longer one looks just as guilty. Only
`n_sole_missing` says what removing that one feature would give back:

```
rows 860 -> 288 (dropped 572)
  risk.rolling_drawdown   n_missing=562   n_sole_missing=515
  target                  n_missing= 15   n_sole_missing= 10
  technical.rsi           n_missing= 42   n_sole_missing=  0
```

`technical.rsi` is missing in 42 rows and is worth removing in none of
them — its 14-bar warm-up is entirely inside `rolling_drawdown`'s 252-bar
one. The target is attributed separately because its cause (the forward
horizon) and its remedy (a shorter horizon, or more data) are different,
and unlike a feature it cannot simply be removed.

A warning is raised when alignment costs more than 30% of the rows, or
when any single feature is solely responsible for more than 10% — not on
every dataset, since a warning that always fires trains the reader to skip
the ones that matter. When no column is ever the sole cause, the warning
says so explicitly rather than showing an empty breakdown.

The same attribution appears in the error raised when *nothing* survives,
which previously left the caller to guess which feature was too long for
the window they asked for.

**`entities` reports what reached the panel**, not what was fetched. The
two differ whenever a symbol's history is shorter than the feature
lookbacks plus the target horizon; reporting the fetched list made a
dataset look like it covered a universe the model never saw a single row
of. The fetched list is still available as `entities_fetched`, and any
symbol that dropped out entirely is named in `warnings`.

Warnings are persisted with the dataset and carried onto any model trained
from it as `ModelManifest.dataset_warnings`, surfaced by
`inspect_model(view="lineage")` — the caveats belong next to the metrics
they qualify, and the build-time tool response is transient. An empty list
on an older model is indistinguishable from "no warnings" by design:
absence of a recorded warning is not evidence the condition did not hold.

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

1. **The formula is causal** — `TemporalSupport`, *enforced* as above: a
   `CURRENT_ONLY` feature is rejected outright.
2. **The underlying dataset is true point-in-time data** — *reported, not
   enforced.* The data layer records this as
   `DataSetMetadata.point_in_time` and `survivorship_free`, both of which
   every provider this package ships reports as **false**. Modeling now
   reads them and emits a warning (see [Coverage and provenance
   warnings](#coverage-and-provenance-warnings)), which is a genuinely
   weaker guarantee than the gate applied to the formula — a warning you
   can ignore, and building proceeds either way.

The asymmetry is deliberate rather than an oversight: a `CURRENT_ONLY`
feature has a PIT-safe alternative (compute it from prices instead), so
refusing it is actionable. A provider that revises history does not — every
shipped provider reports the same thing, so failing on it would leave no
usable path at all. That is a limitation of the available data, not a
policy choice worth encoding as an error.

So a model trained on the current ticker universe still carries
survivorship bias, and a provider that silently revises history is still
not *detected* — only disclosed. Treat `PIT_SAFE` as "this formula doesn't
look forward", not "this dataset is point-in-time correct".

---

## Estimators

`modeling.estimators.registry.ESTIMATOR_REGISTRY` — an explicit allowlist,
keyed by `(task, name)`:

- **regression** (scikit-learn): `linear`, `ridge`, `lasso`, `elastic_net`,
  `hist_gradient_boosting`, `random_forest`, `gradient_boosting`,
  `quantile`, `quantile_gradient_boosting`
- **classification** (scikit-learn): `logistic`, `hist_gradient_boosting`,
  `random_forest`, `gradient_boosting`
- **both, when installed**: `lightgbm`, `xgboost`

`engine.run_experiment` refuses any estimator type not in this registry —
no arbitrary `sklearn` import, no `exec()`. An LLM builds a declarative
`ModelSpec`; the engine decides exactly how it executes.

### The two optional boosters, and why they are worth installing

Measured on this pipeline — 50 entities, 73,400 panel rows, one
walk-forward run:

| Estimator | Time |
|---|---:|
| `random_forest` (n_estimators=100) | 62.9 s |
| `hist_gradient_boosting` | 10.9 s |
| `lightgbm` | **3.55 s** |
| `xgboost` | **3.50 s** |

`random_forest` is the estimator that stops being usable first as the
universe grows, and no amount of tuning the surrounding Python changes that
— the cost is inside sklearn's tree building. LightGBM and XGBoost are
roughly **18×** faster on the same panel.

Neither is a declared dependency. Registration is guarded, so a missing
library leaves the registry reporting what *is* installed rather than
breaking an import of `standard_quant_tools.modeling`. Install with
`pip install lightgbm xgboost` if you want them; `hist_gradient_boosting`
closes most of the gap with no extra install.

### Quantile regression

`quantile` (a linear program, exact and cheap) and
`quantile_gradient_boosting` (captures interactions, one model per
quantile) predict a chosen quantile of the target rather than its mean.
Fitting the 10th, 50th and 90th gives an uncertainty band — the spread
between them is the model's own statement about its confidence. The median
is also far less sensitive to a fat tail than the mean, which matters for
return data specifically.

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

The same reasoning now covers the rest of the request surface, which it did
not originally reach:

| Bound | Value |
|---|---|
| Feature window params | ≤ 100,000 bars |
| `DatasetSpec.universe` | ≤ 1,000 symbols |
| `ModelSpec.random_seed` | 0 – 2³²−1 (NumPy's RNG range) |

Integer-valued **feature** params are enforced from the default's *type*
rather than a vocabulary of parameter names. `refit_every` is not a
window-name, so `refit_every=1.5` passed the generic finite-number check and
reached `range(window, n+1, refit_every)`, raising a bare `TypeError` from
inside Python that named neither the feature nor the parameter. An
integer-valued float (`5.0`) is still accepted and coerced.

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

`modeling.validation.walk_forward.WalkForwardSplit(train_window, test_window, embargo, scheme)`
yields `(train_positions, test_positions)` pairs over the dataset's
unique dates, walking forward one `test_window` at a time, with an
`embargo` gap between each fold's train and test window.

### Three schemes

`ValidationSpec.method` and `.scheme` choose between them:

| Setting | Training window | Use it to answer |
|---|---|---|
| `walk_forward` + `rolling` (default) | fixed length, slides forward | what would this have earned |
| `walk_forward` + `expanding` | anchored at the start, grows | same, on a short history |
| `purged_kfold` | everything outside the test block | is there a signal here at all |

**`expanding`** keeps the same fold *boundaries* as rolling — the test
windows are identical — and only anchors the training start at the
beginning of the sample, so the two remain directly comparable. It stops a
short history being discarded, at the cost that later folds train on more
data than earlier ones, so a trend across folds mixes "the model improved"
with "the model got more data".

**`purged_kfold`** splits the date axis into K contiguous blocks and tests
each exactly once, with overlapping labels purged and an embargo band on
*both* sides. It uses a short history far better than walk-forward, which
can never test its earliest `train_window` dates at all, and its metric is
not dominated by whatever happened at the end of the sample.

Its cost is stated rather than buried: **folds after the first train partly
on data that postdates their test block.** That is not leakage in the label
sense — the purge and the two-sided embargo remove the rows whose
information touches the test window — but it is not a simulation of live
trading either, because a live model cannot be fitted on next year's data.
Use `purged_kfold` to decide whether a signal exists; use `walk_forward` to
estimate what it would have earned.

The target-overlap purge below is generalized to match: a training row is
purged when its label's span *overlaps* the test block, rather than merely
ending after the test starts. Under walk-forward the two rules are
identical, since training always precedes testing; the general form exists
because purged K-fold puts training rows on both sides.

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
  be judged against. The constant comes from the **training** fold, not the
  test fold: at prediction time nobody knows the future window's average
  realized return, so a test-derived constant holds the model against a
  standard no real forecaster could meet. `baseline_is_oracle` reports
  which is in force, so a caller never has to infer it.
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

**Except the IC dispersion statistics, which are pooled rather than
averaged.** A weighted mean is right for `cs_ic_mean` and wrong for
everything built on it:

```
mean(fold standard deviations)  !=  std(all OOS daily ICs)
mean(fold ICIRs)                !=  mean(all ICs) / std(all ICs)
```

A fold's std measures dispersion *within* that fold's dates only, so
averaging folds' stds discards the between-fold variation entirely — which
is exactly the variation ICIR exists to measure. Two folds each internally
rock-steady (std < 0.02) but centred at +0.20 and −0.20 average to a
"dependable" ICIR, while the pooled series has ~zero mean and std > 0.15.
Walk-forward folds are disjoint in time, so the per-date IC series are
concatenated and the dispersion computed once over the whole OOS window.
The per-fold versions stay in `validation_report`, where they answer the
different question of how each individual fold did.

### Fold accounting

`validation_report` (also surfaced by `inspect_model(view="validation")`)
records expected vs completed vs skipped folds, each skip's reason and
date, fold coverage, the target horizon, purged-row count, and **per-fold**
metrics with train/test windows. A single averaged number cannot show
performance decay over time, reveal which regime drove the result, or
expose that one fold carried everything — and a run where 8 of 10 folds
were skipped previously looked identical to a clean 2-fold run.

Each fold records both `train_end` — the last date **actually fit**, after
label-overlap purging — and `scheduled_train_end`, the window end the
splitter planned. Their difference is exactly how much the purge removed.
Only the scheduled value used to be reported, so a fold whose last two
weeks were entirely purged still claimed to have trained through them.

`ValidationSpec.min_folds` (default **2**) is enforced against *completed*
folds. One surviving fold is a single train/test split, not walk-forward
validation — it cannot show whether performance holds across time, which is
the entire point. Lower it to 1 only for a deliberately short exploratory
run.

### Feature importance, with direction

`inspect_model(view="feature_importance")` reports five numbers per
feature, computed across folds:

| Key | Meaning |
|---|---|
| `mean` | average \|value\| — the ranking quantity, unchanged, so older manifests stay comparable |
| `std` | spread of \|value\| |
| `signed_mean` | average coefficient **with its sign** |
| `signed_std` | spread of the signed coefficient — the real stability metric |
| `sign_consistency` | fraction of folds agreeing with the majority sign, in [0.5, 1.0] |

The last three exist because taking the absolute value *first* broke the
stability report this section is for. A feature whose coefficient
alternates `+0.5, −0.5, +0.5, −0.5` across folds — the maximally unstable
case, and a textbook sign of fitting noise — is `|0.5|` in every fold, so
its `std` is exactly **0.0**: reported as perfectly stable by the very
number whose job was catching it. `signed_std` reports 0.5 and
`sign_consistency` 0.5 for that case.

Direction also matters on its own: a stably negative coefficient is a
working contrarian signal, and magnitude alone made it indistinguishable
from a positive one of the same size.

All three signed keys are **NaN for tree estimators**, whose
`feature_importances_` are non-negative by construction and carry no
direction — deliberately NaN rather than a plausible default, so "no sign
information exists" cannot be misread as "the sign was stable". Exact-zero
coefficients (routine under L1) are excluded from `sign_consistency` rather
than counted as agreeing with either side.

After validation, the registered model is refit on the **full** panel —
folds are for validation, deployment uses every observation. The
preprocessing stats from that final refit are persisted alongside the model
so `score_model` applies the identical transform, rather than refitting on
whatever universe happens to be in the scoring call.


---

## Targets

`TargetSpec.type` selects what the model is trained to predict. The choice
matters more than the estimator does, and the default is the simplest rather
than the best.

| Type | Task | What it is |
|---|---|---|
| `forward_return` (default) | regression | `(close[t+h] - close[t]) / close[t]` |
| `forward_return_vol_scaled` | regression | that return over the entity's own trailing volatility, scaled to the horizon |
| `forward_return_rank` | regression | the return's rank within its date's cross-section, mapped to `[-0.5, 0.5]` |
| `forward_return_market_neutral` | regression | the return minus that date's equal-weighted universe return |
| `forward_direction` | classification | `1.0` when the forward return exceeds `threshold`, else `0.0` |
| `triple_barrier` | classification | `1.0` upper barrier first, `0.0` lower first, `2.0` neither |

**Why not just use the raw return.** An unscaled return target lets the
highest-volatility names dominate a squared-error loss — the model spends
its capacity on whichever handful of entities moved most, which is rarely
what you wanted it to learn. `forward_return_vol_scaled` divides that out.

**Why a rank target is worth considering.** The model is *scored* on
cross-sectional rank IC. Training it to predict a magnitude and then judging
it on an ordering optimizes one thing and reports another;
`forward_return_rank` aligns the two. It is also immune to the fat tail that
lets a few extreme returns dominate the fit.

**Market-neutral takes the market out of the label,** rather than leaving it
in and hoping the model learns to ignore it. What remains is the relative
performance a cross-sectional model is supposed to forecast.

`forward_return_rank` and `forward_return_market_neutral` are *cross-sectional*:
they are defined against the other entities on the same date, so they cannot be
built per entity and are applied once after the panel is stacked. Rows on a
date carrying a single entity are dropped with a `NOTE` — a one-name
cross-section has no rank, and a market-relative return of exactly zero by
construction is not a measurement.

### Triple barrier

`triple_barrier` asks which of two barriers the price touches first within
the horizon. The third outcome — *neither* — is deliberately its own class
rather than being folded into "down", because "the price went nowhere" is a
real and common answer and a plain up/down label teaches the model something
false about it.

Left at `barrier=0.0` the barriers are placed at trailing volatility scaled
to the horizon, which is the volatility-adaptive form. A fixed 5% barrier is
a coin flip in a quiet name and unreachable in a volatile one, so the same
label would mean different things for different entities.

The class ids are `0` down, `1` up, `2` neither — nominal, not an ordered
scale. Two constraints forced that specific numbering. It has to be
integer-valued, because sklearn reads a float target whose values are
`0.0/0.5/1.0` as *continuous* and refuses to fit any classifier to it. And
"up" has to be class `1`, so `positive_class_proba` keeps returning P(up) —
the probability the downstream signal path consumes as a score. Any ordering
putting "neither" at class 1 would hand it P(nothing happened).

Only closes are examined, not intrabar highs and lows, which makes this a
conservative barrier test: a level touched and reversed inside one bar is
not counted.

---

## Preprocessing: pooled vs cross-sectional

`ModelSpec.preprocessing.normalization` chooses how feature columns are
standardized before the estimator sees them.

**`pooled`** (default) fits one mean and standard deviation over the whole
training panel, and applies them unchanged to the test rows.

**`cross_sectional`** standardizes within each date, so what reaches the
model is each entity's position relative to its peers that day.

The difference is not cosmetic. Pooled z-scoring leaves the market factor
inside every feature: on a day the whole market rallies, every entity's
momentum reads high together, and a model fed those features can score well
by learning *"today was an up day"* rather than *"this name is strong
relative to its peers"*. For a model judged on cross-sectional IC, that is
the wrong thing to have learned.

It is not the default only because switching it changes what every existing
model predicts.

Two properties worth knowing:

- **No fold-boundary question.** Unlike the pooled statistics, these are not
  fitted on train and carried to test — each date uses only its own
  cross-section, which is contemporaneous information a live model would
  also have. Nothing crosses the split.
- **Clipping, not quantile winsorizing.** The pooled path clips to the
  1st/99th percentile. That is meaningless inside a single date: the 1st
  percentile of a 20-name cross-section *is* its minimum, so clipping to it
  does nothing at all. `clip_sigma` (default 3.0) bounds outliers at the
  sample size that actually exists.

It is also cheaper — measured at 469 ms against 898 ms for pooled on a
50-entity walk-forward, because it skips the quantile fitting entirely.

---

## Sample weighting

`ModelSpec.weighting.method` decides how much each training row counts.
Default `none`: every row at weight 1.

| Method | Corrects for |
|---|---|
| `label_uniqueness` | overlapping forward returns making consecutive rows redundant |
| `time_decay` | a relationship that drifts, so older evidence is less relevant |
| `uniqueness_and_time_decay` | both |

`effective_sample_size` has always been reported next to the OOS metrics: a
`horizon`-bar forward return generated every bar produces labels that share
`horizon - 1` of their bars, so 2,000 daily rows of a 20-day return carry
roughly 100 independent observations per entity. That number was computed
and then acted on by nothing. These are the weights that act on it.

`label_uniqueness` weights each row by the mean of `1/concurrency` over the
bars its own label spans (López de Prado, *Advances in Financial Machine
Learning*, ch. 4). It is computed **per entity**, because two entities'
labels are different series and do not make each other redundant. Weights
are normalized to mean 1, so turning weighting on does not also rescale the
effective regularization strength.

`time_decay`'s `half_life_days` is in calendar days rather than bars, so the
intent survives a change of data frequency.

An estimator that does not accept `sample_weight` raises rather than
silently ignoring it. A weighting the caller believes is active but which
never reached the fit is worse than an error — the model looks like it
corrected for label overlap and did not.

---

## Hyperparameter search

`ModelSpec.search` is optional and off by default. When set, each fold runs
a grid or random search **on its own training window** before the real fit.

```python
ModelSpec(
    task="regression",
    estimator=EstimatorSpec(type="ridge", params={}),
    validation=ValidationSpec(train_window=250, test_window=125, embargo=5),
    search=SearchSpec(
        param_grid={"alpha": [0.01, 1.0, 100.0, 10000.0]},
        inner_splits=3,
        scoring="cs_rank_ic",
    ),
)
```

**Why not `GridSearchCV`.** sklearn's search helpers cross-validate by
splitting *rows*. A modeling panel is stacked `(entity, date)` rows, so an
ordinary K-fold puts the same date on both sides of an inner split — every
entity on that date is a near-duplicate of the others, and the search then
selects whichever hyperparameter best memorizes them. The selection is
leaked even though the outer walk-forward split is clean. Splitting on
*dates*, forward in time, is the only version of this that means anything
here.

`scoring` defaults to `cs_rank_ic` because that is what the outer report
leads with — selecting on `r2` and then quoting rank IC optimizes one thing
and reports another.

**Read the report, not just the winner.** `validation_report.hyperparameter_search`
carries one entry per fold, and each keeps *every* candidate's score:

```python
report = result["validation_report"]["hyperparameter_search"]
[r["best_params"]["alpha"] for r in report if r["searched"]]
# [10000.0, 0.01, 10000.0, 0.01, 100.0, 100.0, 100.0]
```

That output is from a real run, and it is the useful signal: the search
picked a *different* alpha on most folds, which means it was fitting noise.
A single averaged "best alpha" would have hidden that completely.

**What it costs.** Roughly `(grid size × inner_splits)` extra fits per outer
fold. A 12-point grid with 3 inner splits over 20 outer folds is 720 fits
where there was 20. That is why it is opt-in. If the training window is too
short to be split `inner_splits` times, the search declines for that fold
and says so in `reason`, rather than selecting on two dates.

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

`dataset_spec.json` is verified against its recorded `spec_hash` too, not
just `panel.parquet`. The spec is the more dangerous of the pair to tamper
with: the panel is only *read* during training, while the spec is copied
into the model and becomes the definition `score_model` rebuilds features
from for the rest of that model's life. Editing `RSI(14)` → `RSI(100)`
trained on the original panel — whose hash still matched — and registered a
model that would score every future prediction with different feature
definitions.

**Persisted JSON is strict.** `save_json` used Python's default
`allow_nan=True`, which writes bare `NaN` / `Infinity` tokens that are not
valid JSON per RFC 8259. The runtime legitimately produces NaN — AUC on a
single-class fold, ICIR with no dispersion, unsupported feature importance —
so manifests were being written that this package could read back and
stricter parsers could not. Non-finite values now go through the same
`sanitize_for_json` the agent boundary uses, so a manifest on disk and the
tool response describing it agree that a non-finite value is `null`, with
`allow_nan=False` set so any future path that sneaks one through fails at
the *write* rather than producing a subtly unparseable file.

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
prediction dressed as a historical one**.

The gate is `training_information_cutoff`, not `train_end_date`. A row
dated `t` with a horizon-`h` forward-return target reads `Close[t+h]` to
build its label, so the estimator has indirectly seen prices through
`max(label_end_date)` — later than the last feature date by roughly the
horizon. On a 120-bar panel with `h=20` that gap was 28 calendar days, and
every `as_of` inside it was accepted. `train_end_date` is still recorded,
demoted to lineage.

> Models registered before this field existed fall back to the old,
> too-permissive guard, and the error says so. They are detectable
> (`training_information_cutoff is None`) but not retroactively safe.
> Retraining is the only way to get an exact cutoff.

For genuine historical evaluation use the walk-forward OOS predictions
(below), which are actually out-of-sample.

**One cross-section means one date.** `score_model` used to take each
entity's own most recent surviving row, so a halted symbol or one with a
shorter history contributed an older bar inside what the response called a
single `as_of` cross-section. For a cross-sectional model that is not a
smaller cross-section — it is a ranking that no longer compares
contemporaneous information. `missing_entities` never caught it, because
the entity *was* present.

- `effective_score_date` — the most recent date actually available,
  returned alongside `as_of`. The two are deliberately separate: `as_of` is
  what was **requested**, and the newest bar at or before it is legitimately
  earlier on a holiday, a weekend, or when the provider's window ends
  sooner.
- `stale_entities` — `{symbol: its actual last observation date}` for
  entities excluded because their latest bar predates
  `effective_score_date`. Kept separate from `missing_entities`: "no data at
  all" and "data, but older" have different causes and different fixes.
- `staleness_days` — always reported, whether or not a limit was requested.
  `max_staleness_days` rejects the call when the gap is too large; it is
  opt-in, since how much staleness is still decision-useful is a property of
  the strategy, not something `score_model` can pick for you.

**A different scoring universe is allowed — unless a universe-scope feature
says otherwise.** `factors.pca_loading` and `factors.pca_factor_return` are
computed from the entire universe's return matrix, so a model trained on
`[AAA, BBB, CCC]` and scored on `[AAA, BBB]` receives a different PCA basis
under the same column name. The estimator gets a silently different
variable, not a smaller sample. When any trained feature is universe-scope,
the scoring universe must match training exactly (compared as sets, so
ordering is irrelevant; a subset is rejected too — it is a different basis,
not a narrower one). A model built purely from entity-scope features keeps
the freedom to score a new universe, which is the whole point of the
permission.

**Feature implementations are checked before scoring.** The manifest's
`feature_provenance` records, per output column, the feature id, its
resolved params, and a hash of the feature function's own source. Editing a
feature function and then scoring an existing model would otherwise feed the
registered estimator a differently-defined input under the same column
name — coefficients learned against the old definition, a prediction that
looks valid, and not the model that was validated. Mismatches are named per
column, with the model's `git_commit_sha` offered as the way to score
against the code it was trained on.

> Scoped honestly: the hash covers the **feature function's own source**,
> not its transitive dependencies. Rewriting `indicators.momentum.rsi`
> leaves `_technical_rsi`'s source identical, so that case is not caught
> here — `git_commit_sha` / `package_version` are the coarser signal for it.
> What this does catch is what those two cannot: a feature registered at
> runtime from outside the repo.

Score artifacts are **content-addressed**: the filename carries a digest of
the predictions, returned as `predictions_hash`. An identical re-score
resolves to the same path (idempotent, no proliferation), while any change
in the predictions writes a new path and leaves the recorded one intact.
The name used to cover only (date, universe) and was written with
`overwrite=True`, so re-scoring after a provider revised its data replaced
the file in place — and an audit record written by the earlier call still
pointed at that URI, which now returned different bytes. A silently wrong
provenance trail, and the harder kind to notice, because the link still
resolves.

---

## Backtesting a trained model

`run_model_experiment` answers "how did this model do out-of-sample."
It doesn't answer "does this work as a trading strategy" — that requires
an actual backtest, and this codebase already has one
(`run_signal_panel_backtest`, in the *other* 46-tool registry).
`modeling.bridge.oos_predictions_to_signal_panel` connects the two —
a plain Python function, deliberately **not** a tool, because it only
reshapes an artifact the caller already holds and hands it to a tool in
the other registry. That is argument-shaping, not a decision, and it is
the "artifacts, not tool calls" boundary between the two registries.

For the *portfolio* question — one shared cash balance, weights rather
than direction signals — see [Evaluating a model as a
portfolio](#evaluating-a-model-as-a-portfolio) below, which **is** a tool
(`evaluate_model_portfolio`). The distinction is not the count but the
kind: that one runs a simulation and produces new persisted artifacts.

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

**In `model_id` mode the artifact's content hash is verified before it is
loaded.** All of the structural validation above is shape-based, so a
shape-preserving edit passes every one of it: flipping the sign of the
prediction column keeps the same columns, dtypes, pairs and finiteness, and
produces a clean, entirely plausible backtest of numbers the registered
model never emitted. Verified *before* loading, for the same reason
`load_model` does — checking afterwards tells you about a problem you have
already acted on.

> Direct-URI mode is explicitly **unverified**. With no manifest there is no
> root of trust to check a digest against, so that path gets structural
> validation only. A documented boundary, not an oversight, and one more
> reason `model_id` is the preferred entry point.

**A gap in the prediction calendar is not "flat".** `run_strategy`
intersects price data down to the supplied signal index and then takes
`pct_change()` over what remains, so an absent span disappears from the
price axis entirely and the bars either side become adjacent. Measured on a
90-day series with February missing, the Jan→Mar boundary bar carried
**26×** a normal daily return — a month of price movement collapsed into
one bar. Total return can still look right (it did in that repro, because
the position never changed) while per-bar volatility, Sharpe and drawdown
are all distorted, which is why this was easy to miss.

Two gaps, deliberately handled differently:

- **A skipped walk-forward fold is rejected.** Its dates are absent from
  every entity, so there is nothing to densify against — only the caller,
  who has the price data, knows what the missing calendar was. In
  `model_id` mode detection is authoritative: the engine already records
  `validation_report.skipped_folds` with a reason per fold, so the bridge
  reads that rather than inferring a hole from date spacing. Direct-URI mode
  falls back to flagging any gap longer than 10 business days — comfortably
  above any holiday cluster, well below a skipped test window.
- **An entity-level gap is filled.** The date exists in the artifact, just
  not for that entity (a symbol that IPO'd mid-window, or whose features
  were NaN on that bar). `run_signal_panel_backtest` runs `run_strategy` per
  ticker against that ticker's *own* signal series, so leaving the hole
  compressed one symbol's price axis while its peers kept the full calendar.
  Every entity is now densified onto the panel's shared calendar with
  `0.0`, which is the honest fill: the model expressed no view, and
  `DIRECTION`'s `0.0` means exactly "flat".

`task` and `deadband` are validated at runtime. `task`'s `Literal`
annotation is a static hint and this is public Python, so `task="banana"`
used to fall through into classification handling and threshold raw
forward-return predictions against a probability cutoff. A non-finite
`deadband` is rejected for a subtler reason: NaN compares False against
everything, silently *disabling* the deadband, while `inf` compares True and
silently flattens every prediction to zero. Both looked like success.

**Target horizon and holding period are different objects.** A 20-day
forward-return prediction converted to a daily direction signal is
re-evaluated every bar by `run_signal_panel_backtest`. That's a valid
strategy, but it is not "hold for 20 days", and nothing here enforces a
relationship between the two — choose the holding period deliberately
rather than inheriting it from the target.

---

## Evaluating a model as a portfolio

The bridge answers "what if I traded each name independently on the sign
of this model's forecast". `evaluate_model_portfolio` answers the
question an agent actually has after training: **would this model have
made money in an account?**

The two differ in more than presentation:

| | `bridge` → `run_signal_panel_backtest` | `evaluate_model_portfolio` → `run_portfolio_simulation` |
|---|---|---|
| Signal | `-1 / 0 / +1` per ticker | target weight per ticker per rebalance date |
| Capital | each ticker gets its **own** `initial_capital`; per-ticker return streams blended afterwards | **one shared cash balance**, positions sized against current account equity |
| Ranking | discarded | drives position size |
| Between rebalances | signal re-evaluated every bar | share counts held, weights drift (as a real account does) |
| Costs | per-ticker | commission, spread, borrow, margin interest, ADV limits on one book |

Reducing a cross-sectional model's ordering to three values, and then
giving every name its own capital, throws away both the rank and the
fact that the names compete for the same dollars. That is why a strong
`cs_rank_ic` can sit alongside a negative portfolio Sharpe — and why
`run_model_experiment`'s statistical metrics are not a substitute for
this one.

```python
from standard_quant_tools.modeling.agent import (
    EvaluateModelPortfolioInput, evaluate_model_portfolio,
)
from standard_quant_tools.modeling.specs import (
    PredictionTransformSpec, PortfolioSimSpec,
)

result = evaluate_model_portfolio(EvaluateModelPortfolioInput(
    model_id=exp.model_id,
    transform=PredictionTransformSpec(
        method="cross_sectional_rank",   # or zscore / top_bottom_quantile / sign
        gross_exposure=1.0,
        net_exposure=0.0,                # dollar neutral
        max_position_weight=0.05,
        rebalance_frequency="weekly",
    ),
    portfolio=PortfolioSimSpec(
        fill_price="next_open",          # default, and the only lookahead-free choice
        commission_pct=0.001,
        slippage_pct=0.0005,
        borrow_fee_bps=50.0,             # a long/short book shorts for free at 0.0
    ),
))
print(result.metrics["sharpe_ratio"], result.metrics["annualized_turnover"])
print(result.target_weights_uri)
```

### Same leakage discipline as the bridge

This reads `run_model_experiment`'s walk-forward OOS predictions and
**never** `score_model` — no parameter can select that path.
`score_model`'s estimator is the final full-panel refit, so scoring
historical dates with it would be in-sample and the equity curve would be
fiction. The predictions artifact's recorded content hash is verified
*before* loading, for the same reason the bridge verifies it: structural
validation passes on an edited file that kept its shape, so a rewritten
prediction column would otherwise produce a clean and entirely fictional
track record.

The OOS calendar-continuity check applies here too, though the failure it
prevents is different in kind. A hole does *not* compress the price axis
(the simulator runs on its own master trading calendar, not on the signal
index) — but the portfolio would hold a stale position across the gap
while equity keeps marking to market, which is not the model's
out-of-sample performance either.

### The score → weight step reuses `backtest.sizing`

The ranking math is not reimplemented. `rank_weighted`,
`zscore_normalized`, `equal_weight_top_bottom` and `vol_scaled` in
[`backtest/sizing.py`](../src/standard_quant_tools/backtest/sizing.py)
already build gross-normalized weight panels and are called as-is. What
`portfolio_eval` adds is what sizing.py has no concept of:

- **An exact gross *and* net target.** A single rescale can control one
  or the other, never both. The signed vector is split into books sized
  to `(gross + net) / 2` and `(gross − net) / 2`, which gives
  `sum(|w|) = gross` and `sum(w) = net` by construction for any
  `|net| ≤ gross`.
- **A per-position cap that redistributes.** Excess above
  `max_position_weight` is pushed onto the uncapped names in the same
  book, repeatedly — redistribution can lift another name over the cap,
  which must then be capped in turn. A cap that merely truncated would
  quietly deliver less gross exposure than requested.
- **Honest infeasibility.** A book with too few names to hold its target
  gross under the cap (2 names × 0.1 cap cannot reach 0.5) reports the
  shortfall in `transform_diagnostics` and `warnings`. It does not breach
  the cap, and it does not rescale the *other* book — that would break
  the net target instead.
- **A rebalance schedule.** `daily` / `weekly` / `monthly`, taking the
  **first** prediction date in each period. First, not last: "last date
  in the month" is only knowable once the month has ended, so a schedule
  built that way cannot be reproduced live.
- **Sparse cross-sections.** Dates sharing the same set of available
  entities are grouped and weighted together. A missing `(entity, date)`
  stays `NaN` through the score panel and receives zero *weight* — never
  a `0.0` *score*, which is the middle of a centered cross-section and
  would rank a name the model said nothing about above every name it was
  bearish on. A date with fewer than two entities is left flat: one name
  is not a cross-section, and allocating to it would be acting on a
  comparison that was never made.

Classification predictions are recentred to `proba − 0.5` when the panel
is built. Ranking is unaffected (a constant shift is monotone), but it
makes `method="sign"` mean the same thing for both tasks — a raw
probability lives in `[0, 1]`, so its sign is `+1` for every name on
every date, a "long everything" book dressed up as a signal.

### What the result carries

`metrics` is the economic answer: cumulative return, CAGR, annualized
volatility, Sharpe, Sortino, max drawdown, Calmar, mean and annualized
turnover, mean gross/net exposure, position count.
`estimated_cost_drag_pct` is explicitly **derived, not measured** — the
simulator deducts costs from cash without reporting a total, so this
reconstructs the commission + spread component from realized turnover and
therefore *excludes* borrow, margin interest and any impact model. It is
a floor on cost drag, not the whole of it.

`transform_diagnostics` reports what the weighting actually produced
(names per date, book sizes, realized gross/net, dates below target
gross, dates with no position). `provenance` records the prediction,
weight and equity-curve hashes alongside the dataset/estimator lineage
and both specs, so a reported Sharpe traces back to the exact bytes that
produced it. The target-weight artifact is content-addressed, so changing
the transform writes a *new* artifact rather than replacing one an audit
record still points at.

`warnings` carries anything that changes how the number should be read:
a `close` fill convention (look-ahead, and the metrics will look better
than they are), dropped rebalance dates, books that could not reach
target gross, an interval with no defined annualization factor, the
dataset coverage warnings carried from the model manifest, and any the
simulator itself raised (insolvency, negative cash).

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
- **Time-varying universe membership** — universes are static ticker
  lists, with no as-of membership, so historical models over a
  present-day universe carry survivorship bias (which the default
  provider itself reports it cannot rule out). Still deferred, since it
  needs index-constituent history no shipped provider exposes; what is
  built is the *diagnosis* — see [Coverage and provenance
  warnings](#coverage-and-provenance-warnings).
- **Non-daily feature *parameter* calibration** — the built-in features'
  default parameters are stated in trading days (`window=252` is one
  trading year at `1d`, about six weeks at `1h`) and nothing rescales them
  for a non-daily interval; you get a warning and are expected to set them
  yourself. *Annualization* is no longer part of this gap — see
  [Interval-aware annualization](#interval-aware-annualization).
- **Model lifecycle states** — registration means "persisted and
  validated enough to load", not "approved for production". There is no
  trained/validated/approved/production distinction.
- **Prediction transforms beyond sign** — *closed for the portfolio
  path.* `PredictionTransformSpec` (see [Evaluating a model as a
  portfolio](#evaluating-a-model-as-a-portfolio)) provides sign, rank,
  z-score, quantile and volatility-scaled transforms with gross/net
  exposure targets and a position cap. The **bridge** remains sign-only
  by design — `DIRECTION` is the only units-invariant signal for an
  engine that treats a `SCORE` as a raw leverage multiplier. Still not
  built: beta- and sector-neutralization, which need per-ticker
  beta/sector metadata this repo does not carry (the same blocker
  `backtest.sizing` documents for its own deferred list).
- **Hyperparameter tuning, custom estimator import, multi-model
  comparison tooling.** Note that selecting among candidates on
  `evaluate_model_portfolio`'s reported Sharpe would turn those OOS folds
  into tuning data — a nested inner-fold selection is what that needs,
  and it is not built.
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
