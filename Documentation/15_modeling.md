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
    validation=ValidationSpec(train_window=756, test_window=63, embargo=20),
    random_seed=42,
)
exp_result = run_model_experiment(
    RunModelExperimentInput(dataset_id=ds_result.dataset_id, spec=model_spec)
)
print(exp_result.oos_metrics)  # {'r2': ..., 'mae': ..., 'ic': ..., 'rank_ic': ...}

score_result = score_model(ScoreModelInput(
    model_id=exp_result.model_id, as_of="2024-06-01",
    universe=["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
))
print(score_result.predictions_uri)

print(inspect_model(InspectModelInput(model_id=exp_result.model_id, view="feature_importance")).data)
```

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
| `volume.obv_roc` | `indicators.volume.obv`, rate of change over `lookback` (raw OBV is unbounded/cumulative) | entity |
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
faster than computing every singular triplet, with no accuracy cost for a
real, factor-structured universe (see `analysis.pca.pca_returns`'s `method`
docstring for the full explanation, including the one case — near-degenerate
eigenvalues beyond the single dominant component used here — where the two
methods can pick different eigenvectors within an unstable subspace).

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
only, safe to use anywhere in a historical training window since the
source data itself isn't revised. `TemporalSupport.CURRENT_ONLY` exists
for features like today's fundamentals, where no point-in-time-safe
historical provider is wired up yet: using one in a multi-year training
dataset would silently leak future-only information into the past.

`dataset.leakage.check_point_in_time_safety` runs on every
`build_model_dataset` call and raises `PointInTimeViolation` (a
`ValidationError` subclass) if any requested feature is `CURRENT_ONLY`.
Nothing in this phase registers a `CURRENT_ONLY` feature — the mechanism
is built now, not deferred, because retrofitting it after models already
exist that were (silently) trained on leaked data is much more expensive
than building the guardrail before the first fundamentals feature ships.

---

## Estimators

`modeling.estimators.registry.ESTIMATOR_REGISTRY` — an explicit allowlist,
keyed by `(task, name)`, all from `scikit-learn>=1.3.0` (already a core
dependency — no new install):

- **regression**: `linear`, `ridge`, `lasso`, `elastic_net`, `hist_gradient_boosting`, `random_forest`, `gradient_boosting`
- **classification**: `logistic`, `hist_gradient_boosting`, `random_forest`, `gradient_boosting`

`engine.run_experiment` refuses any estimator type not in this registry,
and `validate_params` refuses any constructor param outside that
estimator's explicit allowed-params set — no arbitrary `sklearn` import,
no `exec()`. An LLM builds a declarative `ModelSpec`; the engine decides
exactly how it executes.

---

## Walk-forward validation & the leakage discipline

`modeling.validation.walk_forward.WalkForwardSplit(train_window, test_window, embargo)`
yields `(train_positions, test_positions)` pairs over the dataset's
unique dates, walking forward one `test_window` at a time, with an
`embargo` gap between each fold's train and test window.

`engine.run_experiment` fits preprocessing (`features.transforms.fit_preprocessing`
— per-column winsorize bounds + zscore mean/std) **on each fold's training
rows only**, then applies those same fitted stats to that fold's test
rows via `apply_preprocessing` — never refit on test. This is the
highest-value part of the original design discussion: it's structurally
impossible to leak test-fold statistics into training through this path.

After walk-forward validation reports out-of-sample metrics
(`r2`/`mae`/`ic`/`rank_ic` for regression, `accuracy`/`auc` for
classification), the registered/deployed model is refit on the **full**
panel — walk-forward folds are for validation only, deployment uses every
available observation, the same convention real factor-model practice
uses. The preprocessing stats from that final refit are persisted
alongside the model so `score_model` applies the identical transform to
new data, rather than refitting stats on whatever universe happens to be
in the scoring call.

---

## Model registry

```
SQT_RUNS_DIR/<model_id>/
    manifest.json            # task, estimator, features, target, dataset lineage,
                              # oos_metrics, feature_importance_summary, git_commit_sha,
                              # package_version, random_seed, oos_predictions_uri
    model.joblib
    model_spec.json
    preprocessing_stats.json
    oos_predictions.parquet  # walk-forward OOS predictions -- see "Backtesting a
                              # trained model" below
```

Same directory-per-id convention `backtest.artifacts.save_artifact`
already uses for backtest runs (`ds_...`/`mdl_...` id prefixes never
collide), written with the same atomic temp-file-then-`os.replace`
pattern and identifier/path-escape validation — `modeling.artifacts`
reuses `backtest.artifacts.save_artifact`/`load_artifact` directly for
Parquet panels/predictions, and adds small JSON/joblib helpers following
the identical pattern for manifest/model files.

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
`run_model_experiment`'s walk-forward out-of-sample fold predictions are
leakage-safe by construction (each fold's predictions come from a model
that never saw that fold's dates) and already span the whole dataset's
date range, so they're persisted (`oos_predictions.parquet`,
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

signal_panel = oos_predictions_to_signal_panel(exp.oos_predictions_uri, task="regression")

result = run_signal_panel_backtest(SignalPanelBacktestInput(
    tickers=list(signal_panel.keys()),
    start_date="2018-01-01", end_date="2024-01-01",
    signal_panel=signal_panel,
    signal_type=SignalType.DIRECTION,
))
print(result.portfolio_metrics["sharpe_ratio"])
```

---

## Error handling

Every failure mode below raises `standard_quant_tools.error.ValidationError`
(the codebase-wide exception for boundary/logic validation) *except*
Pydantic construction of a spec object itself (`DatasetSpec(...)`,
`ModelSpec(...)`, ...), which raises `pydantic.ValidationError` — the
normal, expected split between "this input shape is invalid" (Pydantic)
and "this input is well-formed but semantically wrong" (this codebase's
own `ValidationError`).

- **Ambiguous input, rejected at construction**: duplicate feature ids or
  duplicate universe symbols in a `DatasetSpec`, `start >= end`, a
  malformed date string anywhere a date is expected.
- **Data fetch failures**: a provider error or empty OHLCV for any
  universe symbol *or* the benchmark names the failing symbol in the
  error, rather than propagating an unattributed raw exception.
- **Non-finite feature values**: `build_model_dataset` runs
  `require_finite_array` over every feature/target column before
  returning — a degenerate computation producing `inf` is caught here,
  not left to fail confusingly inside `sklearn`.
- **Classification target mismatch**: `run_model_experiment` validates a
  `task="classification"` model's target is binary `{0, 1}` before
  fitting anything — `TargetSpec` only builds a continuous
  `forward_return` today, so this is the expected failure mode until a
  classification-ready target type exists.
- **Degenerate walk-forward folds**: a fold whose training window is
  empty, or (classification only) lands entirely on one class, is
  skipped rather than failing the whole experiment — as long as at least
  one fold survives. `run_model_experiment` only raises if *every* fold
  was skipped.
- **Partial scoring**: `score_model`'s `ScoreModelResult.missing_entities`
  lists any requested universe symbol that had no scoreable row as of
  `as_of` (e.g. insufficient history within `lookback_days`) — never
  silently absent from a result that otherwise looks complete.

## Explicitly deferred

Not built in this phase, and not accidentally half-built either:

- **Semantic feature search** — `list_features` is a plain catalog
  lookup. A ~21-entry catalog doesn't need ranking; this is revisited
  only if the catalog grows large enough that it does.
- **Fundamentals / true point-in-time revision tracking** — the
  `CURRENT_ONLY` mechanism exists, but no fundamentals feature is
  registered yet, since none of the current data providers expose
  point-in-time-safe historical fundamentals.
- **Hyperparameter tuning, custom estimator import, multi-model
  comparison tooling.**
- **Extracting a codebase-wide generic `standard_quant_tools.artifacts`
  package** — `modeling.artifacts` reuses `backtest.artifacts` directly
  today; a shared package is only worth building once there's a second
  real pattern to generalize from, not speculatively.

See [Documentation/08_analysis.md](08_analysis.md) for the underlying
analysis primitives (`hurst_exponent`, `pca_returns`, `rolling_beta`,
...) every built-in feature wraps.
