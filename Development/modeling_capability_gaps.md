# Modeling and feature lab — dead and unreachable capability

**Date:** 2026-09-21
**Question asked:** which functions in `modeling/` and the feature lab exist but
cannot be reached — and which of them are worth turning into tools.

This is **not** a defect hunt. Defects found along the way are recorded in
section 7 because it would be wrong to drop them, but they are a by-product.
The subject is capability that is built, works, and has no door.

---

## 0. Method, and why the first answer was wrong

The agent-visible surface is **31 tools**: 22 modeling tools
(`modeling/agent/dispatch.py:MODELING_TOOL_DISPATCH`) plus 9 feature-lab tools,
both wired through `agent/runtimes/__init__.py:458,483` and
`mcp/catalog.py:260,272`.

A static AST reachability scan over `modeling/` reported:

```
public definitions in modeling/ : 381
  referenced by the tool layer  : 109  (29%)
  referenced, but never by tools: 184
  referenced by NOTHING in src  :  88
    ...and not in tests either  :  25
```

**That 29% is not the finding, and treating it as one would have been the
error.** The first version of the scan counted string constants as references
and reported every tool in the library as dead, because tools are reached by
string key from a dispatch table that no code-reference walk can see. The
corrected scan excludes `modeling/agent/**` from the target set for the same
reason.

Even corrected, the scan's individual hits were unreliable. Of the names it
flagged in `modeling/analysis/`, **all three were false positives** — confirmed
not by reading but by object identity:

```
feature_tools._feature_stability   is feature_stability.feature_stability   -> True
feature_tools._permutation_test_ic is feature_stability.permutation_test_ic -> True
```

So the scan was used only to aim. Every claim below was then **executed** —
seven parallel investigations against the Carbon engine venv, and the
load-bearing ones re-verified personally afterwards. Claims that could not be
executed are marked as such rather than asserted.

Four dependencies are absent in this venv and it is stated wherever it bites:
`lightgbm`, `xgboost`, `scikit-survival`, `optuna`. `cryptography` **is**
present; `skops` and `fsspec` are not.

---

## 1. Headline

**There is very little dead code, and a great deal of unreachable capability.**
Those are different problems with different fixes, and conflating them is how
you end up deleting something valuable.

The shape of the finding, stated plainly:

1. **Almost nothing is broken.** Across seven sweeps, essentially every function
   called worked on real data the first time. This is a library whose parts are
   finished.
2. **The dead list is short and mostly correct.** Of ~30 genuinely dead symbols,
   about a third are constants, a third are protocol conformance or plumbing
   that should stay, and a third are real capability.
3. **The big gaps are not dead functions at all.** They are *live* functions
   whose output is computed, used once inside an engine loop, and thrown away
   before it reaches the tool boundary — or whose parameters no tool input can
   express. A function with a live caller is invisible to a "dead code" scan and
   is exactly as unreachable to an agent as one with none.
4. **One dead function is worse than dead — it is the safe half of a fork whose
   unsafe half is the one an agent gets.** `bridge.py:209` has a `model_id`
   branch that reads the task from the manifest and verifies the predictions
   against their registered digest, and an `oos_predictions_uri` branch its own
   docstring marks *"Explicitly unverified — prefer `model_id`"*. The verified
   branch is reachable from no tool. The unverified one is the only route an
   agent has. Section 7, D-10 and D-11, is what that costs.

The three richest examples, all verified personally:

- **`plan.py` is a complete, panel-aware, hash-carrying experiment planner
  reduced to one integer.** `validate_model_spec` calls
  `plan_experiment(spec, pd.RangeIndex(n_dates)).n_fits` — on a *fake* date axis,
  with no panel. `FoldPlan` has 18 fields and a `to_dict()`; `ExperimentPlan`
  has 15 and a `to_dict()`. None of it is serialized anywhere, and
  `RunModelExperimentInput` has exactly three fields, so there is no `dry_run`.
- **`calendar.py` computes real numbers for 102 exchanges off an installed
  dependency, and an agent's only view of it is one boolean.**
- **Parameter bounds exist, are enforced on every call, and cannot be read.**
  `N_ESTIMATORS` is capped at 2000 with a written rationale; `num_leaves` at
  4096; the logistic solver×penalty matrix is written down in full. Searching
  the serialized capability report for `2000`, `4096`, `liblinear`,
  `invscaling`, `modified_huber` and `calibration` returns **False for every
  one**. `allowed_params` ships bare names.

---

## 2. The genuinely dead list

"Dead" = no caller anywhere in `src/`. Verified by scoped grep, then executed
to confirm the function still works.

| symbol | works? | keep / expose / delete |
|---|---|---|
| `features/params.py resolved_lookback` | ✅ | **expose** — G1 |
| `features/custom.py` (whole module) | — | **keep** — an extension point by design, correctly empty |
| `analysis/feature_ablation.py DEFAULT_MAX_FITS` | n/a | **delete** — a constant that collides by name with the *live* `limits.DEFAULT_MAX_FITS=500`, while `FeatureAblationInput.max_fits` hardcodes 200 independently. A drift trap. |
| `validation/distributional.py pinball_loss` | ✅ 0.01020 | **expose** — G3 |
| `validation/conformal.py MIN_CALIBRATION_ROWS` | n/a | constant; leave |
| `validation/comparison.py compare_ic_series` | ✅ | **expose** — G4 |
| `validation/comparison.py newey_west_variance` | ✅ | **expose** — G4 |
| `validation/survival.py censoring_distribution` | ✅ | expose — M2 |
| `validation/survival.py brier_time_grid` | ✅ 64-pt grid | expose — M2 |
| `validation/survival.py brier_scores` | ✅ | expose — M2 |
| `validation/survival.py integrated_brier_score` | ✅ 0.1195 | expose — M2 |
| `validation/search.py search_candidates` | ✅ | **expose** — fold into G2 |
| `validation/search.py rank_turnover` | ✅ 0.329 / 0.000 | **expose** — M5, 3 lines |
| `dataset/lags.py parse_lag_column` | ✅ | **give it a caller** — M1 |
| `dataset/lags.py deepest_lag` | ✅ | **expose** — G1 |
| `samples.py SampleIndex.take` | ✅ | leave — engine-shaped, no question behind it |
| `targets/registry.py list_targets` | ✅ | **expose** — G6 |
| `calendar.py calendar_names` | ✅ 102 names | **expose** — G5 |
| `calendar.py sessions_per_year` | ✅ XNYS 251.55 | **expose** — G5 |
| `calendar.py session_minutes` | ✅ XNYS 390 | **expose** — G5 |
| `calendar.py bars_per_session` | ✅ XNYS@1h = 7 | **expose** — G5 |
| `estimators/boosting.py OPTIONAL_ESTIMATORS` | ✅ 8 pairs | **expose** — G7 |
| `estimators/survival.py aft_survival_function` | ✅ all 3 distributions | dep-gated (xgboost); reachable when present |
| `estimators/survival.py CoxPHRegressor.set_params` | ✅ | **leave** — 3 lines of sklearn protocol, costs nothing |
| `estimators/survival.py HAS_XGBOOST_SURVIVAL` | ✅ | **delete** — exactly equal to `HAS_XGBOOST`, which is reported |
| `registry/artifacts.py local_store` | ✅ | **delete** — returns the *local* store; every store-taking API wants a *remote* one. 0 src callers, 0 test refs. |
| `registry/package.py list_remote_models` | ✅ | **expose** — G9 |
| `registry/package.py pull_model_package` | ✅ full round trip | **expose** — G9, after Defect A |
| `registry/signing.py sign_manifest` | ✅ | **expose** — G8 |
| `registry/signing.py signing_available` | ✅ `True` | leave — `optional_dependencies.cryptography` already answers it |
| `bridge.py oos_predictions_to_signal_panel` — **the `model_id` branch** | ✅ | **expose** — G10. The *other* branch of the same function is reachable and is the one its docstring says not to use. |

Note how few of these are worth deleting: **three**. The rest are either
capability with no door, or plumbing that is correctly plumbing.

### Corrections to the scan, recorded so they are not re-derived

`feature_stability`, `permutation_test_ic`, `summarize_feature_set`,
`group_sizes`, `load_dataset_spec` (2 callers) and `load_distribution`
(1 caller) were all flagged and are all **live**. `leakage.py` contains no
stranded empirical screen — the screen lives in `analysis/feature_report.py` and
`check_leakage` already calls it.

---

## 3. The arsenal — ranked

Nine proposals. Every one is backed by functions that were executed and worked;
none needs new mathematics. Ranked by what they unlock against what they cost.

### G1 · `estimate_feature_warmup` — the catalog's lookback is wrong the moment you override a parameter

**HIGH.** `FeatureDefinition.lookback` is a static number recorded at
registration against *default* parameters. `resolved_lookback(definition,
resolved)` exists to compute the true one and **has zero callers**. Its own
docstring names the two callers that need it; neither calls it.

Verified personally:

| feature | catalog says | requested | actually consumes |
|---|---|---|---|
| `statistical.hurst` | 200 | `{window: 500}` | **500** |
| `market.momentum` | 20 | `{lookback: 900}` | **900** |
| `risk.realized_volatility` | 20 | `{window: 252}` | **252** |
| `technical.rsi` | 14 | `{period: 14}` | 14 |

`market.momentum` understates by **45x**. And `scoring.py` mentions
`lookback_days` four times and `resolved_lookback` **zero** times — so
`score_model(lookback_days=400)` is a number a human supplies by hand, with
nothing in the library deriving it, against features that may need 900 bars.

Pair it with `deepest_lag` (also dead): `max(resolved_lookback) + deepest_lag`
is exactly "how many bars this spec burns before row one" — the question
`explain_dataset_row_loss` answers only *after* a build has already been paid
for.

```
estimate_feature_warmup(features: List[FeatureSpec], interval: str = "1d")
  -> {bars_required, per_feature: {id: {declared, resolved, deepest_lag}},
      binding_feature, calendar_days_estimate, warnings}
```

Cost: S. Wraps two dead functions and one live registry read.

### G2 · `plan_model_experiment` — see the split before paying for it

**HIGH.** The largest stranded capability found. `plan.py` computes, per fold,
before any fit: real train/test date spans, `n_train_dates` *after* the purge,
`n_train_rows`, `n_test_rows`, `n_purged`, the `n_inner_folds` the window
supports, `n_candidates`, `n_fits`, and two content hashes.

Verified: `validate_model_spec` reduces all of that to
`plan_experiment(spec, pd.RangeIndex(int(n_dates))).n_fits` — **one integer, on
a RangeIndex, with no panel**. Without a panel `n_purged` is `None` on every
fold. With one, the same spec reports **180 rows purged per fold, 1,080 total**,
and fold 0's training window loses nine dates.

Three things it would surface that nothing can today:

- `inner_fold_count` returns **0** for a window too short, and the plan then
  prices that fold at 1 fit instead of `1 + grid×inner`. Visible in
  `plan.to_dict()` and nowhere else.
- `fits_per_estimator` with 3 quantiles + 5 conformal blocks = **9**. A 9×
  cost multiplier invisible behind a single integer.
- `CombinatorialPurgedSplit.n_paths` = 15, each with two disjoint test blocks.

Fold in `search_candidates` (dead) behind `include_candidates: bool` — an agent
that can see the actual grid can spot a log-spacing mistake before spending 720
fits on it.

Cost: S. `plan_experiment(spec, dates, panel=panel)` + `.to_dict()` + a result
model. The same planner `run_experiment` executes, so the numbers are the
numbers that will run.

### G3 · `score_prediction_intervals` — the library produces intervals and cannot check them

**HIGH.** `distributional.py` is complete and correct, with exactly one caller:
`engine.py:1000`, inside a fold loop, averaged into `oos_metrics` and never
re-runnable.

Meanwhile the interval columns *are* handed to the agent — `engine.py:1067` puts
`q05`/`q50`/`q95`/`lower`/`upper` into the OOS frame and `tools.py:727`
publishes that frame — and `score_model` re-emits them on new data.

Verified personally: `ScorePredictionsInput` has exactly eight fields —
`predictions_ref, task, target_column, prediction_column, ic_method,
ndcg_cutoffs, horizon, event_column`. **No lower, no upper, no quantile, no
interval input of any kind.**

So the library produces intervals, publishes them, re-emits them at scoring
time, and has no way to ask whether they cover. There is a point-calibration
diagnostic (`analyze_model_errors`) and no distributional one.

The `by="date"` axis is the point: `conformal.py`'s own docstring says
exchangeability fails across regimes and "the reported OOS coverage is the
check". One pooled number cannot show a band that covered 97% in calm and 62%
in a selloff.

Working output, measured: `{'pinball_q05': 0.00284, 'pinball_q50': 0.00971,
'quantile_crossing_rate': 0.0, 'quantile_coverage_90': 0.784,
'interval_coverage': 0.897, 'interval_nominal_coverage': 0.9}`.

Cost: S, plus a few lines for the `by` axis.

### G4 · `compare_signals` — no multiple-testing correction is reachable

**HIGH.** `holm_adjust` *is* called by the tool layer — but only at
`tools.py:1911,1960`, inside `_paired_against_reference`, which requires both
sides to be **registered models** sharing a task and target, refuses cpcv models
by name, and enforces a one-against-many star topology.

Verified: `compare_ic_series` and `newey_west_variance` have callers **only
inside `comparison.py` itself**. So a researcher with 12 candidate signals, or
12 p-values from anywhere at all, cannot apply a family-wise correction. There
is no Holm, no Bonferroni, no BH anywhere an agent can reach. The three
multiple-testing tools that do exist live in the backtest runtime and all take
*return series*, not p-values.

The HAC estimator is the sharper irony: the portfolio and research runtimes
**print a warning** that their OLS t-stats "may not survive HAC (Newey-West)
errors" — and `newey_west_variance` sits one import away, unreachable. Measured,
the correction is a factor of **2.8** on a real series (`2.561e-04` at lag 0 vs
`9.139e-05` at lag 19). Not cosmetic.

One tool, three modes: `paired` (two arbitrary frames, no registry),
`ic_series` (two per-date IC series inline), `adjust` (p-values from anywhere).
`mode="adjust"` alone is a one-line wrapper that closes an entire missing
category.

Carry forward the honesty requirement `comparison.py` already states: Holm
controls the family-wise error of *these* tests and does not control for the
candidates having been selected on the same sample — that is what
`run_reality_check` is for, and the two should cross-reference.

Cost: S–M.

### G5 · `describe_exchange_calendar` — 102 venues, one boolean

**HIGH.** `exchange_calendars` is installed and `calendar.py` computes real
numbers off it. An agent's entire view is `optional_dependencies.exchange_calendars`.

To set `DatasetSpec.calendar` — required for any intraday feature that
annualizes — the agent must guess a code or provoke a `ValidationError` to read
the names out of the error text. Verified working: `calendar_names()` → 102
names; `sessions_per_year("XNYS")` → 251.55, `XLON` → 252.70, 24/7 → 365.25;
`session_minutes`: XNYS 390, XLON 510, XASX 360; `bars_per_session("XNYS","1h")`
→ 7.

Cost: S. A pure wrapper over five dead functions, all `lru_cache`d.

### G6 · surface the four target-registry fields no tool reports

**HIGH, and the cheapest on the list.** Verified personally: `targets.detail`
reports exactly `{buildable, continuous, description, tasks}`. The registry
carries and the report drops **`censored`, `cross_sectional`, `requires`,
`param_schema`, `default_params`**.

Two of those change what a correct spec says:

- **`censored`** — `time_to_fill` is the only survival target and is censored by
  construction. `register_external_panel` **refuses** it without an
  `event_column`. No tool says so before the refusal.
- **`cross_sectional`** — `forward_return_rank` and
  `forward_return_market_neutral` are defined against the date's *other*
  entities, so on a one-name universe they are degenerate.
  `CROSS_SECTIONAL_TARGETS` is read by exactly one line in `builder.py` and
  reported nowhere.

Cheapest correct fix: switch `capabilities.py` to call `list_targets()` — which
is currently dead and returns precisely these definitions — and add the four
keys. Cost: XS.

### G7 · `describe_estimator` — the bounds are enforced and unreadable

**HIGH.** Behind each bare name in `allowed_params` sits a populated
`ParamBound` with `kind`, `minimum`, `maximum`, `choices`, `allow_none` and a
hand-written `note`, plus per-estimator compatibility rules. Verified: **none of
it crosses the tool boundary.**

What an agent cannot learn without a failed call:

- `n_estimators` capped at 2000 (note: *"Ensembles above ~2000 trees are a
  compute budget concern"*), `num_leaves` at 4096, `max_depth` at 64.
- `logistic.penalty="l1"` needs `solver ∈ {liblinear, saga}`; `elasticnet` needs
  `saga` **and** an explicit `l1_ratio`. Written down in `bounds.py:128`,
  unreadable.
- `sgd.loss` accepts a different set per task, and the note explaining why
  (hinge has no `predict_proba`, and this library asks every classifier for one)
  is the most decision-changing sentence in the module.
- `mlp` takes `n_hidden_units` (1–512) and `n_hidden_layers` (1–3), **not**
  sklearn's `hidden_layer_sizes` tuple. An agent that knows sklearn guesses
  wrong on its first call, every time.
- **`calibration` is absent from the capability report entirely**, despite that
  report being the thing an agent is told to read before choosing a model.

Size it as a separate filterable tool, not a fold-in: the full payload for all
27 entries is ~30 KB / ~7.5k tokens. One estimator is ~300 tokens.

Cost: S. Also fixes G7b for free.

### G7b · `OPTIONAL_ESTIMATORS` is agent-invisible

`boosting.OPTIONAL_ESTIMATORS` is a static, machine-independent declaration of
8 optional `(task, name)` pairs *with full parameter schemas* — including
`lightgbm_ranker` and `xgboost_ranker`, the estimators that make the `ranking`
task real. Verified: its only consumers in the whole repo are
`Development/generate_modeling_reference.py` (a doc script) and one doc test.
**No tool reads it.** A user without lightgbm sees
`optional_dependencies.lightgbm: false` and has no way to learn what it costs
them. Subsumed by G7 via `include_unavailable=True`.

### G8 · `attest_model_package` — a working security control with no interface

**HIGH.** Signing works end to end: `sign_manifest`,
`verify_manifest_signature`, and `verify_model_package(require_signature=True,
public_key=…)` all run, and key pinning behaves as documented.

Verified personally, and this is the part that matters:

- `verify_model_package` defaults to `require_signature=False, public_key=None`.
- `inspect_model` calls it at `tools.py:823` as
  `verify_model_package(input_data.model_id).to_dict()` — **bare defaults**.
- **No modeling tool input anywhere accepts a `signature`, `public_key`,
  `signing` or `store_url` field.** Scanned every Pydantic input model in
  `agent/models.py`: the result is `NONE`.

The consequence, reproduced live by the investigation: edit `manifest.json` to
change the headline metric to `r2 = 0.91` and *leave* `manifest.sig` in place,
and `inspect_model` correctly reports `ok: False`. **Delete `manifest.sig`
instead, and it reads clean** — `ok: True`, `signature: null`,
`signature_error: null`, r2 0.91. Deleting a signature is strictly easier than
forging one, so the only defence against manifest tampering is a flag no tool
can pass.

**And nothing gates on it.** `lifecycle.py` contains no verification call at
all — verified by grep. A model with a byte appended to `model.joblib` was
driven `candidate → validated → staging → production` through the tool; every
call succeeded, and `inspect_model` was reporting `mismatched: ['model.joblib']`
the entire time. `PromoteModelInput` has five fields and no verification gate.

Note the asymmetry: the audit subsystem **did** get this control —
`meta/tools.py:373` takes a `public_key_path`, and its docstring says *"only the
Ed25519 checkpoint signature catches a wholesale rewrite."* The model registry
has the identical control and no tool.

Ship the promotion gate with it — `promote_model(require_verified_package: bool)`
— or the tool is advisory and production stays ungated. `Promotion.evidence`
already exists as the place to record the `manifest_sha256` the decision rested
on.

Cost: S for both.

### G9 · `list_remote_models` + `pull_model_package` — the mirror is write-only

**HIGH, sequenced second.** The subsystem works end to end: mirror copied all 11
files re-hashing each through the target, `list_remote_models` found the model,
`pull_model_package(require_signature=True, public_key=…)` registered it into a
second root with `ok=True, key_pinned=True`, the promotion log travelled, a
re-pull was refused without `overwrite=True`, and a flipped byte made the pull
refuse by filename.

Zero of the 31 tools take a store. Push happens only if an operator sets
`SQT_MODEL_MIRROR_URL` before the process starts. **A model can be pushed to a
mirror and never listed, pulled, or verified back through a tool.**

**Blocked on Defect A** (section 7). A pulled model that cannot be monitored,
and whose error message tells the operator to retrain, is half a model. Fix the
URI resolution first; ship G8 alone in the meantime.

### G10 · `backtest_model_signal` — the safe branch of the bridge is the dead one

**HIGH.** A researcher *can* get from a trained model to a backtest through
tools alone: `run_model_experiment` → `oos_predictions_ref` →
`convert_reference(to_kind="signal_panel", task=…)` →
`run_signal_panel_backtest`. That route was run end to end and returned
Sharpe 1.027.

But it goes through the `oos_predictions_uri` branch — the one the bridge's own
docstring calls *"Explicitly unverified"*. The `model_id` branch, which exists
precisely to make the following impossible, is reachable from no tool. Three
consequences, all reproduced live:

- **A wrong `task` is accepted silently and backtests to nonsense.**
  `convert_reference(ref=<a regression model's OOS ref>, task="classification")`
  succeeded, produced an all-zero panel for all 8 entities, and the backtest
  returned `annualized_return 0.0, sharpe nan, sortino inf, calmar inf` with no
  error anywhere. The `model_id` branch reads `task` from the manifest and
  rejects a mismatch.
- **The tamper check is bypassed.** `run_model_experiment` publishes a *copy* of
  the predictions, separate from the artifact the manifest hashed. Flipping the
  sign of every prediction in that copy was accepted without complaint. The
  `model_id` branch calls `verify_file` against
  `manifest.content_hashes["oos_predictions"]` first.
- **cpcv degrades to a useless error.** `bridge(model_id=…)` and
  `evaluate_model_portfolio` both refuse it by name with a remedy;
  `convert_reference` fails with *"19200 duplicate (entity, date) row(s)"*
  naming a temp file — correct, and actionable by nobody.

Cost: S, ~15 lines wrapping a function that already does all of this. It also
drops the `meta` runtime from the path: a model→backtest workflow currently
costs **77 tools of context** (modeling + meta + backtest) against 22 for
modeling alone.

### G11 · `attach_model_outcomes` — `score_predictions` cannot score this library's own output

**HIGH, and the most clearly broken thing in the sweep.** Neither
`run_model_experiment`'s ref (`['date','entity','prediction','lower','upper']`)
nor `build_model_ensemble`'s ref (`['date','entity','prediction']`) carries a
`target` column, so both fail `score_predictions` with *"the predictions frame
has no 'target' column"*.

Verified personally: `build_model_ensemble`'s own tool description
(`tools.py:2221`) says it publishes a reference *"that score_predictions and the
backtest bridge read like any other."* The backtest half is true. **The scoring
half is false.** That is a documentation claim, not an inference.

**The library already knows how to fix it.** `tools.py:1885 _oos_with_actuals`
joins a model's OOS predictions to the realized target via
`_panel_with_selected_target`, refusing rather than guessing when a
multi-horizon dataset is ambiguous. It is private and used only by
`compare_models(method="paired")` and `analyze_model_errors`. Prototyped:
publishing its output as a ref and scoring it gave ridge ICIR 0.107, rf 0.119,
**ensemble 0.128** — the diversification benefit, three lines away and
unreachable.

Without this the library can build an ensemble and backtest it but cannot
produce a single statistical number for it — which is to say it cannot answer
whether the ensemble was worth building, the exact question `ensemble.py` exists
to ask. Returning `horizon` also feeds `ScorePredictionsInput.horizon`, which an
agent currently has to remember from dataset-build time.

Cost: S.

### G12 · `evaluate_predictions_portfolio` — the simulator only accepts a model id

**HIGH.** `evaluate_model_portfolio` takes a `model_id` and nothing else;
passing an ensemble ref fails at identifier validation. Everything below the
`model_id` lookup in `portfolio_eval.py:656-922` already works on a frame.

Generalizing it to any `predictions_ref` — an ensemble, an externally computed
alpha, a converted panel — is what makes `build_model_ensemble` a tool that
produces a tradeable result rather than a correlation report. Same
`PredictionTransformSpec`, same `PortfolioSimSpec`, same result shape with
provenance naming the ref.

Cost: S–M. A generalization, not new math.

### Also HIGH, but two-line fixes rather than tools

- **`score_predictions` always uses an oracle baseline.**
  `baseline_regression_metrics(y_true, train_y)` has two arms and its own
  docstring says one is wrong: using the test fold's own mean makes it an
  *oracle*, so `model MAE vs baseline MAE` is not a valid comparison.
  `engine.py` passes `train_y`. Verified: `tools.py:1730` calls
  `baseline_regression_metrics(y_true)` with **one argument**, and
  `regression_metrics(y_true, y_pred, dates=dates)` never passes `train_y`
  either. So every external prediction ever scored gets
  `baseline_is_oracle = 1.0`, and the oracle's R² is 0.0 *by construction* —
  which is exactly why it is not a baseline. `ScorePredictionsInput` has no
  input that could fix it. **Add `train_mean`.**
- **`estimate_feature_warmup`'s sibling:** `score_model`'s `lookback_days` should
  be derivable, not hand-supplied (G1).

---

## 4. MEDIUM

- **`screen_feature_significance`** — panel-wide permutation floor.
  `run_feature_permutation_test`'s own docstring calls `null_p95_abs` "the
  honest floor for `select_features(min_abs_rank_ic=...)`", but answers for one
  feature while `select_features` takes the floor as a number the agent invents.
  Closing the loop inverted the result on a real 12-name panel: a naive
  `min_abs_rank_ic=0.02` keeps 4 of 10 features; the permutation floor (max
  `null_p95_abs` = **0.0694**) keeps **0**. Every feature showed per-date IC
  autocorrelation +0.65 to +0.74, and `technical.rsi` is p=0.0099
  ("significant") under a `within_date` null and p=0.308 under `circular_shift`.
  Measured cost 3.15s for 10 features × 200 permutations. *Promotable to HIGH —
  it is ranked MEDIUM only because it is a loop around an existing tool.*
- **`screen_feature_stability`** — `feature_stability` and `feature_drift` are
  both single-feature-only, so `analyze_features` gives a whole-panel view that
  is **silent about time**. On a real panel `statistical.hurst` was the only
  feature with PSI 0.289 (`significant`) against 0.006–0.061 for everything
  else — a feature that is no longer the same measurement, invisible unless you
  happened to call `get_feature_drift` on that one name. Cost 0.83s.
- **`select_features` discards diagnostics it already paid for.** It calls
  `redundancy_report` and `feature_predictive_stats` internally and returns
  `n_clusters` and `rank_ic_mean` from them. The `vif`, `condition_number`,
  correlation matrices and the actual **cluster membership** are computed and
  dropped. An agent wanting "which features were dropped as duplicates of what"
  reads a prose string and must re-run `get_feature_redundancy`, paying for the
  same correlation matrix twice. **Free capability.**
- **`predict_survival_curve`.** A survival model trains end-to-end today
  (`concordance 0.705`, `integrated_brier 0.104`) but no curve can come out:
  `run_model_experiment` returns the scalar integral and discards the
  `(n_rows × n_times)` matrix; `inspect_model` has no survival view;
  `score_model` returns a scalar risk. Today a survival model is a regression
  model with extra steps — you learn *who fills first*, never *how likely this
  order is to still be resting in 30 seconds*, which is the number a desk sizes
  on. The machinery is intact: a model loaded back off disk had 4,130 baseline
  knots and produced monotone curves with recoverable median survival times.
  *Ranked MEDIUM only because the survival path itself is narrow; within that
  path it is the whole point.*
- **`inspect_model(view="provenance")` — promotable to HIGH.** Two independent
  sweeps arrived at this same proposal from opposite ends (registry lifecycle
  and deployment/consumption), which is the strongest signal in the document.
  **Twelve manifest fields reach no view**: `training_information_cutoff`,
  `train_end_date`, `distribution`, `content_hashes`, `dataset_spec_hash`
  (+version), `feature_provenance`, `feature_implementation_hashes`, `formats`,
  `monitoring`, `version`, `model_id`. Three are load-bearing:
  - **`training_information_cutoff`** is the field `score_model` gates `as_of`
    on, so the earliest legal scoring date is discoverable only by eating a
    `ValidationError`. (Measured: 2024-06-28 against a `train_end_date` of
    2024-06-21 — a week apart, and neither readable.)
  - **`feature_provenance`** is *enforced* at scoring time to refuse a run when
    a feature's implementation hash has moved, and cannot be inspected first.
  - **`distribution`** says whether the model carries a conformal band, which is
    the precondition for `transform.method="uncertainty_scaled"` — so today the
    only way to learn that transform is available is to run
    `evaluate_model_portfolio` and read the refusal.

  Add an environment comparison too (~15 lines: flatten the stored fingerprint,
  flatten `environment_fingerprint()`, diff). Prototyped output on a moved
  environment: `{"blas.blas": {trained: "scipy-openblas", current: "mkl"},
  "packages.numpy": {trained: "2.4.6", current: "2.5.0"}, ...}`. That function
  is the *only* route to the current numerics — the capability report contains
  no `numpy`, no `blas`, no `OMP_NUM_THREADS`, no version of anything.
- **Give `score_model` a `predictions_ref`.** One `handoff.publish` call at
  `tools.py:740`, mirroring what `run_model_experiment` already does at :727.
  Today `score_model` returns a filesystem path, not an `sqt://` reference, and
  `handoff.resolve` rejects a path — so a live scoring run is a dead end. It can
  be monitored for drift but never scored against outcomes or turned into a
  tradeable signal, because there is no publish-a-file-as-a-reference tool
  anywhere in the fleet. With G11 it becomes scoreable; with G10's sibling
  conversion, tradeable.
- **Fold the two `predictions → score_panel` implementations together.**
  `portfolio_eval.py:106` recenters classification by a fixed `0.5` and runs
  full structural validation; `meta/convert.py:93` recenters by a
  caller-settable `proba_threshold` and validates nothing. For
  `proba_threshold != 0.5` they produce different panels under the same name.
  The `portfolio_eval` one is the better of the two.
- **`apply_pit_transform`** — the fundamental transforms are restatement-correct
  but reachable only as a model feature via `provider="polygon"`, never as an
  answer, though the library already accepts inline PIT records in two tools.
- **`resolve_universe`** — `fetch_plan` refuses two keys resolving to one
  provider symbol (`BHP@XNYS` + `BHP@XASX`) only mid-build, after a universe
  fetch has been budgeted; and `common_venue` **silently mutates**
  `DatasetSpec.calendar` with no tool reporting which calendar was adopted.
- **`preview_sample_weights`** — `WeightingSpec.method` is selectable and the
  resulting distribution is never reported. Measured: uniqueness weights span
  0.983–3.261 (a 3.3× spread); time-decay at a 180-day half-life spans
  0.284–2.425. An agent choosing a half-life is choosing blind.
- **`preview_preprocessing`** — all 8 steps fit+apply cleanly outside the engine.
  Two traps are discoverable only at fit time: `pca_whiten` refuses NaN *and*
  refuses `n_components > n_columns` (it raised on a 4-column frame at its own
  default of 8), and `missing_indicator` doubles the column count.
- **`rank_turnover` in `score_predictions`** — three lines, no new input.
  Turnover is the bridge between a signal's IC and its net-of-cost P&L and no
  tool reports it.
- **`monitor_model` should return the `profile` it already loads** —
  `tools.py:992` binds it and never uses it again.
- **`parse_lag_column` needs a caller.** Its docstring claims it is "what lets
  `analyze_model_errors` and the importance summary report 'this is rsi at lag
  3'". That never happened — nothing outside `lags.py` matches `__lag`. A spec
  with `lags=[1,2,3]` on ten features produces 40 importance rows of opaque
  strings.
- **`explain_dataset_row_loss` drops `per_entity_rows_dropped`**, which
  `attribute_drops` computes and `build_model_dataset` already returns. "Which
  name lost the rows" is a different question from "which feature".
- **`list_features` drops `frame_kind` and `fields`**, so an agent cannot learn
  that `fundamental.*` needs `provider="polygon"`.
- **External survival curve scoring** — the four dead IPCW/Brier functions.
  Narrower demand; awkward matrix input.

---

## 5. Deliberately not worth exposing

Stated so the question is closed rather than re-asked:

- **`register_estimator`, `register_target`, `register_preprocessor`** — the
  estimator allowlist **is** the security boundary (no arbitrary import, no
  `exec`); an agent-callable registration hole would defeat `bounds.py`'s compute
  budget entirely. The other two take Python callables and cannot be driven from
  a JSON tool call. Library APIs by design, not stranded capability.
- **`partial_fit` / incremental fitting** — reachable from nowhere, and
  correctly so: a warm-started estimator carries state from every previous fold,
  including rows inside the current fold's purge and embargo window, which
  breaks the one guarantee every downstream number rests on. *But the capability
  report advertises `supports_partial_fit: True` for four estimators anyway —
  the only field in the report that describes sklearn rather than this runtime.
  Delete the flag.*
- **`limits.py`** — every constant is already in the JSON schemas by explicit
  design: *"a bound enforced only in a `field_validator` is invisible in
  `model_json_schema()`, which is the document an LLM actually reads."* A
  `get_modeling_limits` tool would be a worse copy of something the agent
  already reads.
- **`cache.py`** — hits/misses/projections already ride out in
  `validation_report["cache"]` and `preprocessing_reused`. In-process, no disk,
  nothing to size or evict.
- **`lifecycle.py`** — promotion, demotion/rollback, full history and stage
  filtering are all already on the surface. Its only gap is the verification
  gate in G8. (Rollback specifically **is** exposed: `PromoteModelInput.to_stage`
  is the full 5-stage Literal and backward moves are permitted.)
- **Plumbing with a live caller and no standalone question** — `stack_long`,
  `stack_features_only`, `build_returns_panel`, `compute_panel_features` (a pure
  perf fast path), `fetch_universe_ohlcv` (duplicates the data runtime),
  `run_dir`, `hash_file`, `verify_file`, `save_model`, `new_model_id`, every
  `load_*` in `model_registry`, all of `serialization.py` and `mirror.py`,
  `ablation_contributions`, `summarize_ablation`, `estimator_capabilities`,
  `group_sizes`, `relevance_grades`, `fold_ic_series`, `horizon_label_end`,
  `quantile_estimators`, and all six `diagnostics.py` publics.
- **`adapters.input_kind`** — one possible value (`"tabular"`) across all four
  adapters, shipped on every entry of every call.

---

## 6. What was checked and found sound

Recorded because a sweep that only reports gaps is not an honest sweep:

- **No dead feature.** 30 registered, all 30 listed by `list_features`, 27/27
  non-PIT built successfully. No unregistered "free" features.
- **12 of 14 analysis→tool paths are lossless.** The "computes twelve, returns
  three" hypothesis was worth testing and came back mostly negative;
  `select_features` is the single real offender.
- **Preprocessing:** 8/8 registered, 8/8 namable from a spec, 8/8 listed with
  id/description/params/defaults. Registry and capability list match exactly.
- **Targets:** `TargetType` validates against the *live* registry rather than a
  frozen Literal, so a runtime registration is immediately spec-namable. All 6
  buildable targets built correctly; all 12 external ones refused by name with
  the right remedy.
- **Estimators registered vs advertised agree exactly** — set difference `[]` in
  both directions. No invisible estimator (at the estimator level; the `tasks`
  list is a separate problem, D-1 below).
- **Classification is in better shape than the scan suggested** — probability,
  calibration (`isotonic`/`sigmoid`, AUC 0.6845 → 0.6980 on the same panel) and
  thresholding are all reachable, and the calibration field descriptions ship in
  the tool's JSON schema.
- **PIT works end to end**, and `asof_join` is genuinely revision-aware
  (1.0 → 1.1 → 1.3 across dates).
- **The purge is exact and the OOS stream is genuinely out of sample** (carried
  forward from the prior pass, unchanged).
- **`specs.py` has no stranded configuration. Not one field.** Every field of
  every spec class was traced to a consumer outside `specs.py` and confirmed
  both settable by an agent and read by a handler — `min_folds` →
  `engine.py:1100`, `quantiles` → `:473`, `intervals.*` → `:528`, `weighting.*`
  → `:407-414`, `ranking.*` → `adapters.py:340/352`, every `search.*` field,
  `benchmark` → `builder.py:356`, and the rest. The structural reason is that
  tool inputs embed whole spec objects (`RunModelExperimentInput.spec:
  ModelSpec`) rather than re-declaring fields, which makes the "exists in
  Python, absent from the schema" failure mode impossible here. The only
  degenerate field is `ConformalSpec.method: Literal["split"]` — one legal
  value, harmless.
- **No tool input field is declared and ignored, and no result field is left
  unpopulated.** Every field of all 31 input models was diffed against its
  handler's source. The three that did not appear literally
  (`compare_models`' `comparison_metric`, `n_bootstrap`, `block_size`) are
  consumed one frame down in `_paired_against_reference`. No unwired `*Input`
  model exists anywhere, and no handler-returned dict key is silently dropped.
- **Ensemble member correlation is exposed and well done** — pairwise, with the
  basis labelled and a >0.95 hot-pair warning. It is the *diversification
  benefit* that is unreachable (G11), not the correlation.
- **Three more scan false positives, all live**: `predictions_to_score_panel`,
  `select_rebalance_dates` and `scale_by_uncertainty` are all called by
  `evaluate_model_portfolio`. `scale_by_uncertainty` in particular runs —
  `transform={"method":"uncertainty_scaled"}` returned a different, plausible
  equity curve (Sharpe 0.109 vs 0.299 for the default rank transform). It is not
  dead; it is *undiscoverable*, which the provenance view above fixes.

---

## 7. Defects found along the way

Not the object of this sweep. Recorded because discarding them would be worse.

**D-1 · `capabilities.tasks` advertises `ranking` with zero estimators.**
Verified personally: `tasks: ['classification', 'ranking', 'regression',
'survival']` against a registry holding `{classification: 6, regression: 12,
survival: 1}`. `available_tasks()` reads the static `_ADAPTERS` dict, not the
registry. This is the exact failure `capabilities.py:101-111` says the module
exists to prevent — they fixed it for `targets` (splitting `buildable` from
`external_only`, with the comment *"an overstatement here is not one failed
call, it is a plan built on a tool that cannot do what it was told"*) and left
`tasks` a bare list. **MEDIUM** — `validate_model_spec` catches it cleanly
(`"Available for ranking: []"`), so the cost is one wasted round trip. One-line
shape fix.

**D-2 · Calibration silently voids feature importances while the capability
report still promises them.** `EstimatorSpec.calibration` wraps the estimator in
`CalibratedClassifierCV`, which has neither `coef_` nor `feature_importances_`,
so `fold_feature_importance` falls through to NaN. Proved on identical data:

```
calibration=none      -> {f0: 0.389, f1: 0.318, f2: 0.293}
calibration=isotonic  -> {f0: NaN,   f1: NaN,   f2: NaN}
```

…while `capabilities()` reports `exposes_feature_importance: True` in both
cases, and nothing warns. `adapters.py:88` states the contract this breaks:
*"the two must agree, or the capability report promises a diagnostic the run
then does not produce."*

**D-3 · The signature downgrade** — see G8. Deleting `manifest.sig` makes a
tampered manifest read clean through `inspect_model`, and no tool can pass
`require_signature`.

**D-4 · Promotion does not consult integrity** — see G8. `lifecycle.py` has no
verification call; a model with a corrupted joblib was promoted to production
through the tool while `inspect_model` reported it mismatched.

**D-5 · A pulled package's monitoring is broken, and the error blames the wrong
thing.** `manifest.monitoring.*_uri` fields are stored as **absolute local
paths** of the registering machine. The pull copies the parquet files correctly;
the manifest still points at the source path. Same machine, different
`SQT_RUNS_DIR` → `resolved path … escapes SQT_RUNS_DIR`. Different machine →
`monitor_model` says *"model was registered before monitoring references were
kept … Retrain to register a model with references"* while
`feature_reference.parquet` sits in its own directory. `evaluate_model_portfolio`
and the OOS bridge read the same field and fail the same way. **Blocks G9.**

**D-6 · `mirror_model_package(prefix=…)` produces a write-only artifact.** All
11 files land; `list_remote_models` then returns `[]` because it hard-filters
`key.startswith("mdl_")`, and the pull refuses because the manifest under the
prefix names a different `model_id`. Either don't expose `prefix`, or key
list/pull off the manifest.

**D-7 · `get_feature_drift` is single-split only**, and no rolling or per-block
PSI exists anywhere. A drift *curve* would be genuinely useful but is new code,
not unlocked code — flagged so it is not lost.

**D-8 · `summarize_feature_set` summarises the whole panel with no holdout**, so
`CompareFeatureSetsResult.left/right.mean_abs_rank_ic` is in-sample by
construction and carries no warning — while its sibling `select_features` went
to considerable trouble over exactly that.

**D-9 · `BuildModelDatasetInput`'s JSON schema enum lists all 18 target ids**,
including the 12 external-only ones that `build_target` refuses. The capability
report distinguishes them; the tool schema does not.

**D-10 · A wrong `task` on the agent-reachable bridge path is accepted silently
and backtests to nonsense** — all-zero panel, `sharpe nan, sortino inf, calmar
inf`, no error anywhere. See G10.

**D-11 · The predictions tamper check is bypassed on the agent path.**
`run_model_experiment` publishes a *copy* of the predictions, separate from the
artifact the manifest hashed; `convert_reference` accepted a copy with every
prediction's sign flipped. See G10.

**D-12 · `build_model_ensemble`'s tool description is false.** It states the
published reference is one *"that score_predictions and the backtest bridge read
like any other"*. Verified: the ref carries `['date','entity','prediction']`,
`score_predictions` requires a `target` column, and the call fails. The backtest
half of the sentence is true; the scoring half is not. See G11.

**D-13 · `ConvertReferenceInput.task`'s description advertises the capability
that is missing.** `agent/models.py:5113` says *"Required unless the reference
carries a model_id."* Verified: `convert.py` raises unconditionally when `task is
None`, and the string `model_id` appears **zero times** in that module.

**D-14 · `score_model`'s forward predictions are a dead end.** It returns a
filesystem path rather than an `sqt://` reference, and `handoff.resolve` rejects
a path. `monitor_model` accepts a path; `score_predictions` and
`convert_reference` require a ref; no publish-a-file-as-a-reference tool exists
in the fleet. A live scoring run can be monitored for drift and can never be
scored against outcomes or traded.

**D-15 · The conformal band a scored model emits is invisible and, in one
measured case, implausible.** Scoring a ridge model wrote `lower`/`upper` with a
mean width of **0.169** against a prediction range of 0.0034–0.0058 — a band
roughly thirty times the entire cross-section's spread.
`ScoreModelResult.summary_stats` reports mean/std/min/max of the point
prediction only and says nothing about the interval, and `manifest.distribution`
is surfaced by no view. *(Whether the width itself is a defect or correct
behaviour on a low-signal panel was not established — it is recorded as
something no tool would let anyone notice.)*

**D-16 · Two implementations of `predictions → score_panel` that disagree** on
the classification offset and on whether the frame is validated at all. See
section 4.

**Three deletions:** `feature_ablation.DEFAULT_MAX_FITS` (name-collides with the
live `limits.DEFAULT_MAX_FITS`), `HAS_XGBOOST_SURVIVAL` (equal to the reported
`HAS_XGBOOST`), `artifacts.local_store` (returns the wrong kind of store; zero
references anywhere). Plus one flag: `supports_partial_fit`.

---

## 8. Suggested order

Cheapest-first, respecting the one sequencing constraint:

1. **Fix the two false descriptions first** — D-12 and D-13. They are text, they
   take minutes, and each currently advertises a capability that does not
   exist, which is worse for an agent than silence.
2. **G6** (target fields) and the **`train_mean`** fix — both XS, both make an
   existing answer correct rather than adding surface. Then **D-1** (`tasks`
   shape) and **D-2** (calibration warning), one line each.
3. **G10** (`backtest_model_signal`) — S, and it closes D-10 and D-11 with it.
   A silently-wrong backtest outranks every missing diagnostic on this list.
4. **G11** (`attach_model_outcomes`) — S. Makes D-12's sentence true and makes
   the ensemble measurable at all.
5. **G1** (warm-up), **G5** (calendar), **G7** (estimator bounds), and the
   **provenance view** — four S-cost wrappers over dead functions and unread
   fields; together they remove most of the discover-by-failing in the library.
6. **G2** (plan) — the largest single unlock, still S.
7. **G8** (attest + promotion gate) — S, and it is a security control.
8. **G3** (intervals), **G4** (compare signals), **G12** (portfolio on a ref) —
   S–M.
9. **D-5**, then **G9** (mirror) — sequenced last because the defect blocks it.

Two patterns worth carrying forward. First: **almost every item is a wrapper, a
passthrough, or a field addition.** Nothing here proposes new mathematics,
because the mathematics is already written and tested — it just has no door.

Second, and more useful as a habit than any single item: **the reliable way to
find this class of gap is to ask what a function computes and then diff it
against what crosses the tool boundary.** A dead-code scan finds constants. The
real losses — `plan.py`'s 18 fields behind one integer, the conformal metrics
averaged away inside a fold loop, the parameter bounds enforced on every call
and readable by nobody, the verified branch of a two-branch bridge — all have
live callers and all look perfectly healthy to a reachability analysis.
