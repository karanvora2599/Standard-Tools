# Opening the doors: a plan for `modeling_capability_gaps.md`

**Date:** 2026-09-21. **Source:** `Development/modeling_capability_gaps.md`
(HEAD `7f20478`). **Status: plan; nothing implemented yet.**

The gaps document found very little dead code and a great deal of
unreachable capability: functions with live callers whose output is
thrown away before the tool boundary, or whose parameters no tool input
can express. This plan turns its twelve HIGH proposals (G1-G12), the two
"two-line" fixes, the MEDIUM list, the sixteen defects (D-1..D-16) and the
three deletions into six phases, each one commit, each green offline
before it lands. Every claim in the document was re-grounded against HEAD
by five parallel read-only passes before this was written; where the
document was wrong about the code, section 8 says so and the phase
follows the code.

---

## 0. Ground rules

- **A phase is one commit** on `main`, `KV - <the insight>`, sole author,
  the CHANGELOG entry written with it. Offline suites green before the
  commit: `tests/modeling`, then `tests --ignore=tests/modeling
  --ignore=tests/data/test_databento_live.py
  --ignore=tests/data/test_databento_pipeline_live.py`, plus `tests/docs`
  (which regenerates the indexes and fails on drift).
- **Multi-agent inside a phase, single-threaded at the seams.** Work is
  split by FILE SET so that no two agents edit the same module. The two
  files every new tool touches -- `modeling/agent/tools.py` and
  `modeling/agent/models.py` -- are owned by exactly one agent per phase;
  library-side work lands in parallel in other modules and the tool
  wiring for it is done afterwards by the owner. Surface bookkeeping
  (section 7) is done once, at the end of the phase, by the integrator.
- **Tests against planted answers, with a null case for every detector.**
  Each phase gets its own test files named for it
  (`tests/modeling/test_capability_gaps_phase<N>_<part>.py`), and the
  planted cases listed under each item are the minimum.
- **Conventions that are not optional:** Input models
  `ConfigDict(extra="forbid")` (and `protected_namespaces=()` when a field
  starts with `model_`); typed pydantic results with a `warnings: List[str]`
  field; non-finite floats through a `Stat` type (`modeling/agent/models.py`
  has none today -- declare one there from
  `agent.runtimes._json_safe.finite_or_none`, `feature_models.py:63` has its
  own); refusals are `ValidationError`s that name the remedy; a tool earns
  its place by being a decision, not plumbing.
- **Nothing here proposes new mathematics.** Two small exceptions are
  named where they occur (Bonferroni and Benjamini-Hochberg beside
  `holm_adjust`; a per-block PSI beside the single-split one). Everything
  else is a wrapper, a passthrough or a field.
- **Interpreter:** `C:/Users/karan/AppData/Local/Programs/Python/Python312/python.exe`;
  isort + black on every touched file.

---

## 1. Phase 1 -- The surface tells the truth (no new tools)

Section 8 of the gaps document puts the text fixes and the one-liners
first because each currently advertises something that does not exist,
which is worse for an agent than silence. This phase is every item that
adds no tool: two false descriptions, five capability-report corrections,
one baseline fix, four dropped fields, three deletions, and the feature
lab's two lossy results. Three agents, disjoint file sets.

### 1A -- the modeling tool layer (owner of `modeling/agent/tools.py` and `modeling/agent/models.py`)

| # | item | where (HEAD) | change |
|---|---|---|---|
| 1A.1 | **D-12** `build_model_ensemble` description says `score_predictions` reads its ref | `modeling/agent/tools.py:2220-2234`; the ref carries `date,entity,prediction` only (`ensemble.py:301-303`) and `score_predictions` refuses without `target` (`tools.py:1697-1703`) | Rewrite the sentence: the reference carries no realized outcome, so `score_predictions` cannot read it until the outcomes are attached (phase 2 names the tool). Add the same caveat to `BuildEnsembleResult.ref`'s description and a `warnings` entry from `build_model_ensemble` (`tools.py:269-289` already builds one). |
| 1A.2 | **D-13** `ConvertReferenceInput.task` says "Required unless the reference carries a model_id" | `src/standard_quant_tools/agent/models.py:5113-5114`; `meta/convert.py:47-53` refuses `task=None` for `signal_panel` unconditionally, tolerates it for `score_panel` (`:127-133`); `model_id` appears 0x in `convert.py` | Replace with: required for `to_kind='signal_panel'`; optional for `'score_panel'`, where omitting it passes the predictions through unchanged. |
| 1A.3 | **`train_mean`** -- every external prediction is scored against an oracle baseline | `tools.py:1729-1730` calls `baseline_regression_metrics(y_true)` with one argument and `regression_metrics(..., dates=dates)` without `train_y`; both take it (`validation/metrics.py:385-387`, `:428-433`, `is_oracle=1.0` when absent at `:415-417`) | `ScorePredictionsInput.train_mean: Optional[float]` after `horizon` (`models.py:1353-1362`), description saying why the scored set's own mean is not a baseline. Pass `train_y = np.full(1, train_mean)` to BOTH calls. Keep the oracle note at `tools.py:1787-1794`. |
| 1A.4 | **`rank_turnover`** into `score_predictions` | `validation/search.py:115-117` (live inside `_score` at `:180`, absent from `__all__` `:551-558`); no tool reports prediction turnover (`feature_report.py:172` is the FEATURE quantity) | In the `entities > 1` arm (`tools.py:1770-1773`): `prediction_turnover: Stat` on `ScorePredictionsResult`, guarded on `"entity" in frame.columns`; add `rank_turnover` to `search.__all__`. Description: the bridge between an IC and a net-of-cost P&L. |
| 1A.5 | **`list_features`** drops `frame_kind` and `fields` | `FeatureCatalogEntry` `models.py:48-56`; populated `tools.py:120-136`; registry carries both at `features/base.py:150-151`; exactly three `fundamental.*` features declare them | Add `frame_kind: Optional[str] = None`, `fields: List[str] = []` and pass them through. |
| 1A.6 | **`explain_dataset_row_loss`** drops `per_entity_rows_dropped` | `attribute_drops` returns it (`dataset/alignment.py:113`); the tool reads five keys (`modeling/agent/dataset_tools.py:111-165`) | Add `per_entity_rows_dropped: Dict[str, int]` to `ExplainRowLossResult` (`dataset_tools.py:57-72`) and a warning when one entity holds more than half of `rows_lost`. |
| 1A.7 | **`monitor_model`** binds `profile` and never returns it | `tools.py:992` (sole occurrence); shape `monitoring.py:87-108` (`bins`, per-feature `n, missing_rate, quantile_edges, mean, std`) | `MonitorModelResult.training_profile: Dict[str, Any]` (`models.py:680-708`): the reference the drift numbers were read against. |
| 1A.8 | **`parse_lag_column`** has no caller | `dataset/lags.py:63-70`, `LAG_SUFFIX="__lag"` `:55`, docstring claim at `:51-54`; importance keys reach the agent raw at `tools.py:781-782`; `analyze_model_errors` names raw columns at `:2036-2044` | In the `feature_importance` view add a sibling `feature_labels: {column: {"feature": base, "lag": depth}}` for every column that parses; use the same label in the `analyze_model_errors` refusal list. `InspectModelResult.data` is `Dict[str, Any]`, no model change. |
| 1A.9 | **convention** `ScorePredictionsResult` has `notes` and no `warnings` | `models.py:1409` | Add `warnings: List[str] = []` beside `notes` (keep `notes`; existing tests read it) and route the new turnover/baseline caveats through `warnings`. |

Tests (`tests/modeling/test_capability_gaps_phase1_tools.py`): the
ensemble description no longer names `score_predictions`, and publishing
an ensemble ref then scoring it still refuses with `no 'target' column`
(the sentence made honest, not the behaviour); `task=None` refused for
`signal_panel`, accepted for `score_panel`; the same frame scored with
and without `train_mean` gives `baseline_is_oracle` 1.0 / 0.0 and the
honest run's `baseline_r2 < 0` when the training mean is off; constant
ordering gives `prediction_turnover == 0.0`, reversed ordering each date
gives 0.5, one entity gives `None`; every `fundamental.*` entry has a
`frame_kind` and every other entry has none; `per_entity_rows_dropped`
sums to `rows_lost` and names the short-history entity; `training_profile`
keys equal the manifest's feature ids with `PROFILE_BINS + 1` edges;
`FeatureSpec(id="technical.rsi", lags=[1,2,3])` produces
`feature_labels` with the right `(feature, lag)` for every lagged column.

### 1B -- capabilities, adapters, engine, and the deletions

| # | item | where (HEAD) | change |
|---|---|---|---|
| 1B.1 | **G6** target detail drops five fields; `list_targets` dead | `capabilities.py:130-137` emits `{buildable, tasks, continuous, description}`; `targets/registry.py:108-109` `list_targets()` has zero callers; `TargetDefinition` carries `censored` (`targets/base.py:77`), `requires` (`:83`), `param_schema` (`:89`), `default_params` (`:90`), `cross_sectional` (`:102-104`) | Build `detail` from `list_targets()`; add `censored`, `cross_sectional`, `requires`, `param_schema` (allowed names), `default_params`. Extend the `note`: censored labels need `event_column` on `register_external_panel`; cross-sectional labels are degenerate on a one-name universe. `generate_modeling_reference.py:190-209` reads `TARGET_KINDS`, not the report, so `29` does not regenerate differently. |
| 1B.2 | **D-1** `capabilities.tasks` is a static list | `capabilities.py:94` -> `available_tasks()` = `sorted(_ADAPTERS)` (`adapters.py:410-428`). The document's "ranking with zero estimators" is environment-specific (this machine has lightgbm and xgboost, so ranking has two); the structural defect holds everywhere | `"tasks": {"fitted": <tasks with a registered estimator>, "no_estimator_installed": <the rest>, "all": available_tasks(), "note": ...}` -- the shape `targets` already uses (`:112-137`). Update `tests/modeling/test_adapters.py:201` (`set(caps["tasks"]) == set(available_tasks())` reads the keys of a dict now -> compare `["all"]`). |
| 1B.3 | **D-2** calibration voids importances while the report promises them | Wrap at `engine.py:106-130` (`_calibrated`), applied `:963` and `:1375`; importance read `:1066` -> `validation/diagnostics.py:33-65` falls to NaN; flag from the UNWRAPPED class at `adapters.py:257-263`; contract comment `adapters.py:88-91`; `calibration` is absent from the capability report | (i) In the engine, when `estimator.calibration != "none"` and the base class exposes coefficients or importances, append a run warning naming `CalibratedClassifierCV` and saying `feature_importance_summary` is NaN by construction. (ii) Add a top-level `calibration` section to `capabilities()`: the three methods (`specs.py:655`), `calibration_folds`, and the sentence that the importance flags describe the uncalibrated estimator. Keep `exposes_feature_importance` as is (the docs generator reads it at `generate_modeling_reference.py:221-226`). |
| 1B.4 | **D-9** the dataset schema's target enum lists the twelve external-only ids | `specs.py:71-82` `_target_choices` sets `schema["enum"] = sorted(TARGET_KINDS)`; pinned by `tests/modeling/test_target_registry.py:48-49, 57-59` and `test_target_extension.py:144-145` | Keep the enum complete (an external label IS legal on `ExternalTarget.target_type`) and make the same hook write the split: `schema["x-buildable"]` and a `description` sentence naming the buildable ids and `register_external_panel` for the rest. The separate-Literal alternative breaks all three pins and is rejected. |
| 1B.5 | **`supports_partial_fit`** is the one capability flag that describes sklearn, not this runtime | set at `adapters.py:244`, read by nothing in `src/`, `tests/` or `Documentation/`; `estimators/online.py:20` says why `partial_fit` is deliberately unreachable | Delete the line. |
| 1B.6 | **deletion** `analysis/feature_ablation.DEFAULT_MAX_FITS = 200` | `feature_ablation.py:39-44`, `__all__` `:160`; collides by name with the live `limits.DEFAULT_MAX_FITS = 500` (`limits.py:39`); `FeatureAblationInput.max_fits` hardcodes 200 at `feature_models.py:795-803` | Delete the constant and its export. Do NOT re-point the input at `limits.DEFAULT_MAX_FITS` -- that would silently raise the ablation cap from 200 to 500. |
| 1B.7 | **deletion** `estimators/survival.HAS_XGBOOST_SURVIVAL` -- the document's premise is wrong twice | `survival.py:483` is `HAS_XGBOOST_SURVIVAL = _register_xgboost()`, and `_register_xgboost()` (`:472-481`) REGISTERS `xgboost_cox`/`xgboost_aft` as a side effect; three test references exist (`tests/modeling/test_survival.py:55`, `test_survival_brier.py:20,181`) | Keep the call on its own line, drop the name from `__all__` (`:486`), rewrite the three test sites to `boosting.HAS_XGBOOST` (the flag the report publishes, `capabilities.py:181`). |
| 1B.8 | **deletion** `modeling/artifacts.local_store` | `artifacts.py:85-87`, export `:58`; zero callers in `src/` and `tests/`; undocumented (grep of `Documentation/` and `README` is empty); `Development/tool_surface_analysis.md:309-311` argued it stays as "the documented entry point", which it is not | Delete the function, its export, and the now-unused `LocalArtifactStore` import (`:46`). Correct the sentence in `tool_surface_analysis.md`. |

Tests (`tests/modeling/test_capability_gaps_phase1_capabilities.py`):
`detail["time_to_fill"]["censored"] is True`,
`detail["forward_return_rank"]["cross_sectional"] is True`,
`detail["forward_return"]["cross_sectional"] is False`, every `requires`
non-empty, `set(detail) == {d.id for d in list_targets()}`; with the
ranking pairs filtered out of `ESTIMATOR_REGISTRY`, `ranking` moves from
`fitted` to `no_estimator_installed`; two identical classification specs,
`calibration="none"` vs `"isotonic"`: the isotonic run's importances are
all NaN AND its warnings name `CalibratedClassifierCV`, the plain run has
neither; the target enum still equals `sorted(TARGET_KINDS)` and the
buildable subset is discoverable from the schema; no estimator entry
carries `supports_partial_fit`; exactly one `DEFAULT_MAX_FITS` exists;
`("survival","xgboost_cox") in ESTIMATOR_REGISTRY` with xgboost present
and `survival` has no `HAS_XGBOOST_SURVIVAL`; `artifacts` has no
`local_store`.

### 1C -- the feature lab's two lossy results (owner of `analysis/feature_selection.py`, `feature_models.py`, `feature_tools.py`)

| # | item | where (HEAD) | change |
|---|---|---|---|
| 1C.1 | **`select_features`** discards what it paid for | `feature_selection.py:144-147` calls `feature_predictive_stats` and `redundancy_report`; the dict at `:252-264` keeps `n_clusters`, `selection_ic`, `holdout_ic`; `redundancy_report` (`feature_report.py:341+`) computed `correlation`, `spearman_correlation`, `vif`, `condition_number`, `clusters`; the drop reason is prose (`:164-171`) | Return `clusters` (member lists), `vif`, `condition_number`, `correlation` from the library; on `SelectFeaturesResult` (`feature_models.py:456-495`) add `clusters: List[FeatureCluster]` (the model `get_feature_redundancy` already returns, so the shapes agree), `vif: Dict[str, Stat]`, `condition_number: Stat`, and `correlation` gated behind `SelectFeaturesInput.include_correlation: bool = False` (O(n^2)); `DroppedFeature.duplicate_of: Optional[str]` so "dropped as a duplicate of what" is machine-readable. Keep `n_clusters`. `tests/modeling/test_feature_tools.py:521-534` (selected == redundancy representatives) must keep holding. |
| 1C.2 | **D-8** `summarize_feature_set` is whole-panel, in-sample, and silent | `feature_selection.py:268-296`; `compare_feature_sets` `:299+` calls it twice; `CompareFeatureSetsResult` (`feature_models.py:551-563`) is the only feature-lab result WITHOUT a `warnings` field; `select_features` holds dates out via `_selection_cutoff` (`:54-88`) and documents why (`:105-116`) | (i) `warnings` on `CompareFeatureSetsResult`, always carrying the in-sample sentence when `holdout_fraction == 0` (quote the D4 figures already at `feature_selection.py:107-113`: five noise columns chosen in-sample scored +0.045 OOS against +0.002 chosen blind). (ii) `CompareFeatureSetsInput.holdout_fraction: float = 0.0` (default 0 so today's numbers do not move) and `selection_end`, reusing `_selection_cutoff`; when non-zero, `FeatureSetSummary` gains `holdout_mean_abs_rank_ic`, `holdout_max_abs_rank_ic`, `selection_window`, `holdout_window`. |

Tests (`tests/modeling/test_capability_gaps_phase1_feature_lab.py`):
with `cluster_threshold=0.0, holdout_fraction=0.0`,
`select_features(...).clusters == get_feature_redundancy(...).clusters`
and `vif`/`condition_number` agree, so the second call is unnecessary;
`duplicate_of` names the keeper; sixty noise columns plus one real
feature: at `holdout_fraction=0` a noise set's `mean_abs_rank_ic` sits
near the real one and the in-sample warning fires, at 0.3 the noise set's
`holdout_mean_abs_rank_ic` collapses while the real set's holds; identical
sets give a zero delta and the warning still fires (it is unconditional
at fraction 0); a one-date panel at 0.3 gets `_selection_cutoff`'s
refusal by name; every feature-lab result model has a `warnings` field.

No count pin moves in phase 1. `tests/docs` still runs: 1B.3's
`calibration` section and 1B.2's `tasks` shape are not read by the
reference generator, so `29_modeling_reference.md` is unchanged.

---

## 2. Phase 2 -- The verified branch is the one the agent gets (G10, G11, G12, D-10..D-16)

`bridge.py:209-402` has two branches. The `model_id` branch (`:271-301`)
reads the task from the manifest, refuses cpcv by name, and verifies the
predictions file against `manifest.content_hashes["oos_predictions"]`.
The `oos_predictions_uri` branch (`:302-306`) checks only that `task` is a
task. Exactly one caller exists in `src/` -- `meta/convert.py:73-79` --
and it passes `oos_predictions_uri=`. So the agent's only route is the
unverified one, and the document's D-10 (a wrong task backtests to an
all-zero panel) and D-11 (a sign-flipped copy is accepted) both reproduce
at HEAD. Three new modeling tools close it, and three field-level fixes
make the scoring path continuous.

### 2A -- the two bridge tools and the scoring publish (owner of `tools.py`/`models.py`)

**2A.1 `backtest_model_signal` (G10).** Wraps the `model_id` branch and
PUBLISHES a `signal_panel` reference; it does not run the backtest. Why:
`signal_panel` is a mapping kind (`handoff.py:75-81`), so
`handoff.publish(panel, "signal_panel", run_id, name, producer=...)` takes
the bridge's return value unchanged and `run_signal_panel_backtest`
already consumes `signal_panel_ref` (`backtest/tools.py:1080-1089`);
running it inline would make the modeling runtime own `tickers`,
`start_date`, `fill_price` and costs, which are backtest decisions. It
also removes the `meta` runtime from the path (77 tools of context to
22 + backtest).

- `BacktestModelSignalInput(protected_namespaces=(), extra="forbid")`:
  `model_id`, `run_id`, `name`, `deadband: float = 0 (ge=0)`,
  `proba_threshold: float = 0.5 (gt=0, lt=1)`, `long_only: bool = True`.
  **No `task` field** -- that is the point; `extra="forbid"` makes a
  mismatch unrepresentable.
- Result: `signal_panel_ref`, `model_id`, `task`, `entities`, `n_dates`,
  `first_date`, `last_date`, `n_long/n_flat/n_short`,
  `oos_predictions_hash` (the verified `content_hashes["oos_predictions"]`),
  `warnings` (seed with the `fill_price="next_open"` advisory at
  `bridge.py:15-21` and `manifest.dataset_warnings`).
- Refusals inherited by name: cpcv (`bridge.py:280` -> `:93-101`, with the
  walk-forward remedy), venue-qualified entities (`:343-353`), tamper
  (`:297-301` -> `artifacts.py:191-197` "has changed since it was
  registered"), calendar hole (`:194-206`).
- `tests/modeling/test_bridge.py:372-388` pins the URI branch as
  deliberately unverified at the LIBRARY level; leave it.

**2A.2 `attach_model_outcomes` (G11).** `_oos_with_actuals(model_id)`
(`tools.py:1885-1906`) already joins a model's OOS predictions to the
realized target via `_panel_with_selected_target` (`:1849-1882`), refusing
a multi-horizon ambiguity (`:1858-1866`) rather than guessing; it is
private, used by `compare_models(method="paired")` and
`analyze_model_errors`. Neither `run_model_experiment`'s ref nor
`build_model_ensemble`'s carries `target`, so `score_predictions` refuses
both (D-12).

- Input (`protected_namespaces=(), extra="forbid"`): exactly one of
  `model_id` / `predictions_ref` (model validator, wording as
  `bridge.py:265-269`); `dataset_id` (required with `predictions_ref`);
  `target: Optional[str]` (label name, disambiguates a multi-horizon
  panel); `run_id`, `name`.
- Lift the dataset half of `_panel_with_selected_target` into
  `_outcomes_frame(dataset_id, target_id_or_label, purpose)` so both
  paths share the ambiguity refusal; parse `horizon` as `tools.py:1936-1939`
  does and return it so it can feed `ScorePredictionsInput.horizon`.
- Publishes `sqt://predictions/<run_id>/<name>` with columns exactly
  `date, entity, prediction, target`. Result: `ref`, `task`, `target_id`,
  `horizon`, `columns`, `n_rows`, `n_predictions_unmatched`,
  `first_date`, `last_date`, `warnings`. Refuse cpcv by name via
  `bridge._refuse_cpcv`.
- Finish D-12: the `build_model_ensemble` description (phase 1A.1) now
  names this tool as the way to make the ref scoreable.

**2A.3 D-14 `score_model` returns a path.** `tools.py:740-749` never
publishes; `handoff.resolve` refuses a raw path under `expect=`
(`handoff.py:549-556`). After `_score_model`, publish the frame as
`sqt://predictions/<model_id>/scored_<date>_<hash8>` with
`producer="modeling.score_model"` and `overwrite=True` (re-scoring is
content-addressed and idempotent, `tests/modeling/test_scoring.py:402-455`;
without `overwrite` the second call hits the collision at
`handoff.py:377`). `ScoreModelResult.predictions_ref: Optional[str]`
after `predictions_uri` (`models.py:553`).

Tests (`tests/modeling/test_capability_gaps_phase2_bridge.py`):
`backtest_model_signal` publishes a resolvable `signal_panel` whose values
are in {-1, 0, 1} and `run_signal_panel_backtest(signal_panel_ref=...)`
returns a finite Sharpe; a sign-flipped `oos_predictions` artifact makes
it raise "has changed since it was registered" (the case
`convert_reference` accepts today); cpcv refused by name; `task=` is
rejected by the schema; venue-qualified entities refused pointing at
`evaluate_model_portfolio`. `run_model_experiment` ->
`attach_model_outcomes` -> `score_predictions` returns a finite
`cross_sectional_ic["ic_mean"]` (the loop that fails today); the same for
an ensemble ref plus `dataset_id`, and the ensemble's ICIR is comparable
to its members'; a two-horizon dataset with no `target` is refused, not
guessed; returned `horizon` equals `TargetSpec.horizon`; published
columns are exactly the four. `score_model` -> `handoff.resolve(ref,
expect="predictions")` equals `load_artifact(predictions_uri)`; scoring
twice does not raise; `score_model` -> `attach_model_outcomes` ->
`score_predictions` runs end to end.

### 2B -- the simulator on a reference, one score-panel, interval stats (library files; tool wiring after 2A lands)

**2B.1 G12 `evaluate_predictions_portfolio`.** `portfolio_eval.py:615-922`
resolves the manifest at `:656-687` and everything from `:688` on is
frame-only. What the model path takes from the manifest, and its
substitute for a ref: `task` (new required input); `validation_method`
cpcv (duplicates are already refused by `_validate_predictions_frame`,
keep an explicit refusal when a `path` column is present);
`oos_predictions_uri` + hash (the handoff store is the root of trust;
record the ref and its sidecar producer in provenance instead of
`verify_file`); `skipped_folds` (None); the dataset spec's `interval`,
`provider`, `calendar`, `start`, `end` (an optional `dataset_id` to
inherit them, or explicit fields). `distribution` is NOT read --
`scale_by_uncertainty` (`:362-392`) reads the frame's `lower`/`upper`
columns (`:375-381`), so a ref carrying them works unchanged; monitoring
URIs are not read.

- Split at `:688` into `_simulate_predictions_portfolio(predictions_df,
  task, *, interval, provider_name, calendar, start, end, transform,
  portfolio, run_id, source, skipped_folds=None, extra_warnings=())`
  holding `:688-921` verbatim; `evaluate_model_portfolio` keeps
  `:656-687` and calls it. `tests/modeling/test_portfolio_eval.py:455-577`
  must stay green through the refactor.
- Tool input (`extra="forbid"`): `predictions_ref`, `task`, `dataset_id`
  (optional, inherits the five fields), explicit overrides for each,
  `transform`, `portfolio`, `run_id`. Result: `EvaluateModelPortfolioResult`'s
  fields with `model_id` replaced by `source_ref`, `provenance` carrying
  `source_ref` and `producer` from `handoff.describe`.

**2B.2 D-16 two `predictions -> score_panel` implementations.**
`portfolio_eval.predictions_to_score_panel` (`:106-140`) validates
structurally through `_validate_predictions_frame` and recentres
classification by a fixed 0.5 (`:138-139`); `meta/convert.py:93-140`
checks three column names, recentres by a caller-settable
`proba_threshold`, and a duplicate `(entity, date)` silently collapses at
`:137`. Keep the `portfolio_eval` one (it is what the simulator trusts):
give it `proba_threshold: float = 0.5` and `task: str | None` (None ->
offset 0, the passthrough `convert` documents at `:127-133`); make
`convert._predictions_to_score_panel` resolve, coerce dates (`:64-68`),
call it, `.to_dict()` into `{entity: {date: score}}`, and return the
EXISTING note strings unchanged (`tests/agent/test_handoff.py:309-355`
asserts them). Net change: `convert_reference` now refuses duplicates,
non-finite predictions and an empty frame, which is the fix.

**2B.3 D-15 the conformal band a scored model emits is invisible.**
`scoring.py:564-573` builds `summary_stats` from the point prediction;
`lower`/`upper` are written at `:487-498`. Add `interval_stats` (populated
only when both columns exist): `interval_mean_width`, `interval_median_width`,
`interval_min_width`, `interval_max_width`,
`interval_width_over_prediction_spread` (mean width / (max - min) of the
prediction -- the document's 0.169 / 0.0024 case), `n_intervals`; a
`warnings` entry when the ratio exceeds 1 ("the band is wider than the
entire cross-section's spread"). No coverage field: coverage needs
realized outcomes, which do not exist at `as_of`; that is phase 4's
`score_prediction_intervals`. `ScoreModelResult.interval_stats: Dict[str,
float] = {}` after `summary_stats` (`models.py:569`), non-finite values
omitted. The fixture at `tests/modeling/test_distributional.py:335-351`
(a conformal model with a known radius) is the planted case.

Tests (`tests/modeling/test_capability_gaps_phase2_portfolio.py`): an
ensemble ref plus `dataset_id` gives a finite Sharpe and a persisted
weights artifact; the same predictions through both entry points give
byte-identical `target_weights_hash`; a ref with a `path` column is
refused by name; `uncertainty_scaled` on a ref without `lower`/`upper`
reproduces the existing refusal; provenance names the ref and producer.
`convert_reference(to_kind="score_panel", proba_threshold=0.6)` and
`predictions_to_score_panel(frame, "classification", proba_threshold=0.6)`
agree cell for cell; a duplicate `(entity, date)` ref is refused by
`convert_reference` with the bridge's message; a non-finite prediction is
refused; the default reproduces every existing expected panel.
`interval_mean_width == 2 * radius` exactly for the conformal fixture; a
point-only model returns `{}` and `summary_stats` unchanged; the
implausible-width warning fires when and only when the ratio exceeds 1.

**Surface after phase 2:** modeling 22 -> 25, catalog 211 -> 214; section 7.

---

## 3. Phase 3 -- Discovery: the numbers an agent can only learn by failing (G1, G2, G5, G7, provenance, previews)

Every item here wraps something that is computed on every call and
readable by nobody: a lookback the catalog understates by 45x, an
experiment plan reduced to one integer on a fake date axis, 85-102
exchange calendars behind one boolean, parameter bounds enforced on
every call and absent from the report, twelve manifest fields no view
returns, and two spec choices (sample weights, preprocessing) whose
consequences appear only after a fit. Six new modeling tools and one new
view. To keep the seam small, each agent owns a NEW module pair under
`modeling/agent/` (the precedent is `dataset_tools.py`, imported into
`tools.py`); the integrator adds the imports and the two registry
entries per tool.

### 3A -- `discovery_tools.py` / `discovery_models.py`: calendar, estimator, warm-up

**3A.1 `describe_exchange_calendar` (G5).** `modeling/calendar.py` is
all `lru_cache`d and public (`calendar_names:81`, `sessions_per_year:104`,
`session_minutes:122`, `bars_per_session(interval, calendar):136` --
INTERVAL FIRST, the document has the arguments reversed --
`periods_per_year:148`, `interval_minutes:69`, `calendar_available:46`).
The agent's only view is `optional_dependencies.exchange_calendars`, and
`DatasetSpec.calendar` (`specs.py:499-523`) can be set only by guessing
or by reading eight names out of a refusal (`calendar.py:96-100`).
Measured here: XNYS 251.6 sessions, 390 minutes, 7 one-hour bars; `24/7`
365.25 and 1440. `calendar_names()` returned 85 on this machine, not the
document's 102 -- it is version-dependent, so nothing pins the count.

- Input (`extra="forbid"`): `calendar: Optional[str]`, `interval:
  Optional[str]`, `name_contains: Optional[str]`. Result: `available`,
  `n_calendars`, `calendar_names` (filtered), `calendar`,
  `sessions_per_year`, `session_minutes`, `interval_minutes`,
  `bars_per_session`, `periods_per_year`, `warnings`.
- Unknown code -> `validate_calendar_name`'s refusal, IDENTICAL to the one
  `DatasetSpec` gives (that is the point). Library absent and a calendar
  asked for -> `require_calendar_library`'s refusal; library absent with
  no calendar -> `available=False`, empty list, a warning, never a raise
  (this is the discovery path). A non-intraday `interval` ->
  `bars_per_session=None` with the `calendar.py:140-144` sentence as a
  warning rather than the refusal.

**3A.2 `describe_estimator` (G7, G7b).** Behind each bare name in
`allowed_params` (`estimators/registry.py:118-124`, all `capabilities.py:58`
publishes) sits a `ParamBound(kind, minimum, maximum, choices,
allow_none, note)` (`bounds.py:32-41`) and compatibility rules:
`N_ESTIMATORS` 1..2000 with its note (`:108-110`), `MAX_DEPTH` 1..64
(`:112-118`), `NUM_LEAVES` 2..4096 (`boosting.py:50-52`), the logistic
solver x penalty matrix (`bounds.py:128-157`: `l1` needs liblinear or
saga, `elasticnet` needs saga AND an explicit `l1_ratio`), the per-task
`sgd.loss` sets (`online.py:67-96`, with the "hinge has no
`predict_proba`" note at `:92-95`), `mlp`'s `n_hidden_units` /
`n_hidden_layers` (`neural.py:159-169`; `hidden_layer_sizes` is built
internally, `:24` says why). Searching the serialized capability report
for `2000`, `4096`, `liblinear`, `invscaling`, `modified_huber`,
`calibration` returns False for every one. `OPTIONAL_ESTIMATORS`
(`boosting.py:233-244`, eight pairs with schemas) is read by the doc
generator and one doc test, never by a tool. Measured payload: all 27
registered entries 24,410 bytes (~6.1k tokens); `classification.logistic`
778 bytes; `regression.mlp` 1,743 -- so a separate, filterable tool, not a
fold-in to `list_modeling_capabilities`.

- One library addition: a public `param_schema(task, name) ->
  EstimatorParamSchema` in `estimators/registry.py` beside `allowed_params`
  (today only the private `_PARAM_SCHEMAS`, `:16`).
- Input: `task: Optional[Task]`, `name: Optional[str]`,
  `include_unavailable: bool = False`. Result: `estimators: List[{task,
  name, available, requires_library, class_path, quantile_param, params:
  {name: {kind, minimum, maximum, choices, allow_none, note}},
  compatibility_notes, calibration}]` (calibration on classification
  entries, from `EstimatorSpec.calibration` `specs.py:655-685` -- the
  spec's note about a raw forest selecting 0 rows at `proba_threshold=0.9`
  against 194 calibrated is the decision-changing sentence),
  `n_estimators_described`, `warnings` (the byte cost when unfiltered).
  `include_unavailable=True` merges the eight optional pairs with
  `available=False, requires_library=...`, which is what makes
  `lightgbm_ranker` nameable on a machine without lightgbm. An unknown
  pair -> `get_estimator_class`'s refusal (`registry.py:88-91`) UNLESS it
  is an optional pair, which is described as unavailable.

**3A.3 `estimate_feature_warmup` (G1).** `resolved_lookback(definition,
resolved)` (`features/params.py:164-186`, `max(definition.lookback,
*window_values)` over `_WINDOW_PARAM_NAMES` `:38-40` and the
`_period|_window|_lookback` suffix rule `:51-54`) has zero callers in
`src/`; `deepest_lag(specs)` (`dataset/lags.py:177-182`) likewise.
`FeatureSpec` lives in `specs.py:94-168` (`id, params, alias, lags`;
lags are per feature, not per dataset; `validate_lags` `lags.py:73-112`
against `MAX_LAG=60`, `MAX_LAGS_PER_FEATURE=20`, `MAX_EXPANDED_COLUMNS=400`).
`scoring.py` mentions `lookback_days` four times (`:110` default 400,
`:342`, `:358` "try a larger lookback_days") and `resolved_lookback`
never. Executed: `statistical.hurst` declared 200, `{window: 500}`
consumes 500; `market.momentum` declared 20, `{lookback: 900}` consumes
900. Modeling runtime, not feature lab: every feature-lab input takes a
`dataset_id` and feature NAMES (post-build); this is the pre-build
question, beside `validate_model_spec`.

- Input: `features: List[FeatureSpec]` (min 1), `interval: str = "1d"`,
  `calendar: Optional[str]`. Result: `bars_required =
  max(resolved_lookback) + deepest_lag`, `per_feature: {output_name:
  {declared, resolved, lags, deepest_lag}}`, `binding_feature`,
  `deepest_lag`, `calendar_days_estimate` (daily: `bars / 252 * 365.25`,
  or `sessions_per_year(calendar)` when given; intraday: through
  `bars_per_session`, `None` without a calendar plus a warning),
  `warnings` (POINT_IN_TIME features contribute 0 and are named; duplicate
  `output_name`). Refusals are the registry's and the lag validator's,
  unchanged. Description says the number is what `score_model(lookback_days=)`
  needs and what `explain_dataset_row_loss` explains only after a build.

### 3B -- `preview_tools.py` / `preview_models.py`: plan, weights, preprocessing

**3B.1 `plan_model_experiment` (G2).** `plan_experiment(model_spec, dates,
*, panel=None, dataset_hash=None, feature_ids=None) -> ExperimentPlan`
(`plan.py:243-249`); `FoldPlan` has 18 fields (`:131-158`, `to_dict()`
emits 15), `ExperimentPlan` 15 (`:181-202`, `within_budget:204`,
`refuse_over_budget:208`, `to_dict:223`). `validate_model_spec` calls
`plan_experiment(spec, pd.RangeIndex(int(n_dates))).n_fits` (`tools.py:1516-1518`)
-- one integer, on a fake axis, no panel, so `n_purged` is None on every
fold; `run_experiment` calls it with the panel and hashes
(`engine.py:682-688`) and refuses over budget at `:689`.
`inner_fold_count` (`search.py:243-251`) returns 0 for a window too short
(executed: `(3,3,0) -> 0`, `(4,3,0) -> 3`) and the plan then prices that
fold at 1 fit; `fits_per_estimator` (`plan.py:56-73`) is 9 for 3
quantiles + 5 conformal blocks. `search_candidates` (`search.py:70-98`)
is reached only by `search_best_params`.

- Input: `dataset_id`, `spec: ModelSpec` (the schema the runtime already
  pays for twice, `models.py:410`, `:1229`), `target: Optional[str]`,
  `include_folds: bool = True`, `include_candidates: bool = False`.
- Body: `_load_dataset_panel` -> sorted unique dates ->
  `plan_experiment(spec, dates, panel=panel, dataset_hash=meta["data_hash"],
  feature_ids=...)` -> `to_dict()`. **Over budget is reported, not
  refused** (`within_budget=False` plus the `refuse_over_budget` text as a
  warning): refusing would defeat a dry run, and `run_model_experiment`
  still refuses. `include_candidates` on a `tpe` spec -> `candidates=None`
  and a warning naming `n_search_candidates` (`search.py:82-85, 101-112`),
  not a raise.
- Result: the 15 `ExperimentPlan` keys, `within_budget`, `folds` (the 15
  `FoldPlan` keys each), `candidates`, `warnings` (a fold with
  `n_inner_folds == 0` is named: the window is too short for the inner
  search and that fold is priced at one fit).

**3B.2 `preview_sample_weights`.** `WeightingSpec` (`specs.py:960-989`:
`none | label_uniqueness | time_decay | uniqueness_and_time_decay`,
`half_life_days` in DAYS) is applied inside the engine and its
distribution is reported nowhere. `build_sample_weights(method, dates,
label_end_dates, entities, half_life)` (`validation/weights.py:199-245`)
returns `None` for `none`, refuses `label_uniqueness` on a panel without
`label_end_date` with a rebuild instruction (`:214-221`), composes both
(`:233-234`) and normalizes to mean 1.0. The document measured uniqueness
weights spanning 0.983-3.261 and a 180-day decay spanning 0.284-2.425:
an agent choosing a half-life chooses blind.

- Input: `dataset_id`, `weighting: WeightingSpec`, `target: Optional[str]`.
  Result: `method`, `half_life_days`, `n_rows`, `min`, `p05`, `p25`,
  `median`, `p75`, `p95`, `max`, `mean` (1.0 by construction), `std`,
  `ratio_max_min`, `effective_sample_size_kish` (`sum(w)^2 / sum(w^2)`)
  BESIDE `effective_sample_size` (`metrics.py:365`, the overlap-based
  one `check_leakage` reports) with a sentence that they measure
  different things, `weight_share_newest_decile`, `n_zero_weight`,
  `warnings` (`ratio_max_min > 10`; a half-life shorter than a tenth of
  the panel's span). `method="none"` returns a flat summary with a
  warning, not a refusal.

**3B.3 `preview_preprocessing`.** Eight registered steps
(`preprocessing/steps.py`; `list_preprocessors()`), `fit_and_apply_pipeline(steps,
train, test, train_ctx, test_ctx)` (`pipeline.py:194-215`), `FoldContext.from_frame`
(`base.py:64-69`, never the target). Two traps are discoverable only at
fit time: `pca_whiten` refuses NaN (`steps.py:346-351`) and
`n_components > n_columns` (`:341-345`) at its own default of 8
(`:512-521`), so it raises on any panel narrower than eight columns; and
`missing_indicator` appends one `<col>__missing` per column (`:271-274`),
doubling the width by design (`:252-257`).

- Input: `dataset_id`, `preprocessing: PreprocessingSpec`, `sample_rows:
  int = 5000`, `split_fraction: float = 0.7` -- split BY DATE, never by
  row. Result: `steps: List[{type, params, stateless, column_wise,
  n_columns_in, n_columns_out, columns_added, columns_removed}]`,
  `n_columns_in`, `n_columns_out`, `output_columns` (truncated),
  `per_column_before/after: {mean, std, min, max, n_missing}` for a bounded
  number of columns, `n_nan_after`, `explained_variance_ratio` when a
  `pca_whiten` step is present (state key at `steps.py:373`), `warnings`
  (width doubled by `missing_indicator`; a non-column-wise step means
  `run_feature_ablation` must refit, `base.py:30-34`; `pc1..pcK` output
  means importances no longer name features). Refusals are the steps'
  own. The default pooled pair takes the fused native path
  (`pipeline.py:152-155`); the preview must report the same state shape
  the engine produces, and a test pins it against
  `tests/modeling/test_native_preprocessing.py`'s fixture.

### 3C -- the provenance view and two in-place fixes (owner of `tools.py`/`models.py`; `registry/`)

**3C.1 `inspect_model(view="provenance")`.** The four views
(`models.py:724-726`; body `tools.py:752-828`) read none of
`training_information_cutoff` (`manifests.py:147`; what `scoring.py:171`
gates `as_of` on), `train_end_date` (`:130`), `distribution` (`:198`;
written at `model_registry.py:276`; the precondition for
`uncertainty_scaled`), `content_hashes` (`:95`), `dataset_spec_hash`
(`:84`, `+_version :88`), `feature_provenance` (`:126`; enforced at
`scoring.py:252`), `feature_implementation_hashes` (`:115`), `formats`
(`:101` AND `:107` -- declared twice with identical comments; the second
wins; fix it), `monitoring` (`:206`), `version` (`:54`). The environment
fingerprint is stored at registration (`model_registry.py:282`) and
`environment_fingerprint()` (`registry/environment.py:79-108`) gives the
current one; the report contains no numpy, no BLAS, no thread count.

- Add `"provenance"` to the Literal and a branch after `tools.py:796`
  returning: `model_id`, `version`, `created_at_utc`,
  `training_information_cutoff`, `train_end_date` (the earliest legal
  `as_of`, readable at last), `dataset_id`, `dataset_spec_hash`,
  `dataset_spec_hash_version`, `content_hashes`, `formats`,
  `feature_provenance`, `feature_implementation_hashes`, `distribution:
  {has_conformal, quantile_levels, alpha, raw}`, `monitoring:
  {has_feature_reference, has_prediction_reference, rows}`, `environment:
  {trained, current, differences: {"packages.numpy": {trained, current}},
  matches}` via a new `environment_differences(trained, current)` in
  `registry/environment.py` (flatten both, diff), and `warnings` (a
  pre-field manifest with no cutoff gets the weaker `train_end_date`
  guarantee, `manifests.py:143-146`; empty provenance; empty environment;
  one line per differing key). `verify_model_package` stays OUT of this
  view; it is already the expensive part of `lineage`.
- The synthesized fuzzer enumerates `get_args` on the Literal
  (`tests/surface/synth.py`), so the fifth view is fuzzed for free.

**3C.2 `resolve_universe`, done in place instead of as a tool.**
`fetch_plan(universe)` (`assets.py:123-144`, duplicate-symbol refusal at
`:133-141`) is called only mid-build (`builder.py:338`,
`pit_features.py:207`, `portfolio_eval.py:728`), after a universe fetch
was budgeted; `common_venue` (`:114-120`) feeds
`DatasetSpec._calendar_from_the_universe_s_venue` (`specs.py:603-626`),
which ASSIGNS `self.calendar` at `:623` with no report, and the calendar
is part of the dataset's identity. The payload of a tool would be
`{entity: symbol}`, one refusal and one adopted code, so: (i)
`validate_model_spec` (`tools.py:1460-1523`, loads metadata only) calls
`fetch_plan(spec.universe)` so the duplicate-symbol refusal arrives
before any fetch; (ii) `build_model_dataset` and `validate_model_spec`
add `calendar adopted from the universe's venue: <code>` to `warnings`
when `:623` fired (record the adoption on the spec, e.g. a private
attribute or a `notes` entry the validator sets).

**3C.3 `apply_pit_transform` -- deferred** (section 8): three
`fundamental.*` transforms (`features/fundamental.py:183-233`) reachable
via `provider="polygon"`; the hard part is the input, and
`validate_pit_records` / `join_point_in_time` already accept inline PIT
records for the checking and joining halves. Revisit as a `transform`
option on `join_point_in_time` when the namespace grows.

Tests (`tests/modeling/test_capability_gaps_phase3_discovery.py`,
`..._previews.py`, `..._provenance.py`): listing mode returns sorted
names including `XNYS`, `XLON`, `24/7` without pinning the count;
`interval="1d"` gives `bars_per_session None` plus the warning; an
unknown code's message equals `DatasetSpec(calendar=...)`'s; a patched
`calendar_available() -> False` returns `available=False` without
raising. `describe_estimator("classification", "logistic")` reports five
solvers and a note naming liblinear/saga; `sgd` reports different loss
sets per task; `mlp` reports `n_hidden_units` and not
`hidden_layer_sizes`; `n_estimators.maximum == 2000` with its note,
`num_leaves.maximum == 4096`; `include_unavailable=True` names all eight
optional pairs with `available` matching `find_spec`; every name in
`allowed_params(task, name)` appears in `params` (the anti-drift pin);
classification entries carry `calibration`. Momentum at 900 binds and
`bars_required == 900 + deepest_lag`; an aliased spec is keyed by alias;
hurst at 500 has declared != resolved; a PIT feature reports 0 and never
binds; `bars_required` fed to `score_model(lookback_days=)` scores where
400 refuses with the `scoring.py:358` message. The plan's `n_fits`
equals what `run_model_experiment` executes on the same dataset; a
too-short fold reports `n_inner_folds == 0` and `n_fits == fits_per_fit`;
three quantiles plus five conformal folds give `fits_per_fit == 9`; with
a panel `n_purged` is not None while the RangeIndex path leaves it None;
`tpe` with `include_candidates` warns instead of raising; `node_hash`
moves with the estimator and not with an unrelated field. Every
weighting method has mean 1.0; a decay whose half-life equals the span
gives `max/min ~ 2`; Kish ESS is at most `n_rows` and falls as the
half-life shrinks; the composite differs from the product of the halves.
The two preprocessing traps refuse with their own messages through the
tool; `missing_indicator` doubles the width; the fused default path
returns a state `step_types` still reads; reordered columns refuse on
apply. `training_information_cutoff > train_end_date` for a horizon-h
target and equals the value `score_model` refuses below; a conformal
model reports `has_conformal=True` and `uncertainty_scaled` then
succeeds while a point model reports False and that call refuses; a
patched fingerprint with a bumped numpy produces exactly one difference;
a pre-field manifest yields `None` with the named warning. A universe
with two keys resolving to one provider symbol is refused by
`validate_model_spec` before any fetch; a venue-inferred calendar is
named in `build_model_dataset`'s warnings.

**Surface after phase 3:** modeling +6; section 7.

---

## 4. Phase 4 -- The statistics that were computed and averaged away (G3, G4, the two screens, the survival curve)

### 4A -- modeling: intervals and signal comparison (owner of `tools.py`/`models.py`)

**4A.1 `score_prediction_intervals` (G3).** `validation/distributional.py`
(`quantile_column:27`, `pinball_loss:36`, `distributional_metrics:45`) has
one production caller, `engine.py:999-1010`, inside the fold loop, averaged
away at `:1110`. The OOS frame carries the columns (`engine.py:988,
996-997, 1067-1074`), `run_model_experiment` publishes it (`tools.py:727-736`),
`score_model` re-emits them (`scoring.py:481-489`), and
`ScorePredictionsInput` has eight fields and no interval input
(`models.py:1319-1370`).

- Input (`extra="forbid"`): `predictions_ref`, `target_column="target"`,
  `quantile_columns: Optional[Dict[str, float]]` (column -> level; default
  auto-detect via the `quantile_column` round trip), `lower_column="lower"`,
  `upper_column="upper"`, `nominal_coverage: float = 0.9 (gt=0, lt=1)`,
  `by: Literal["all","date","entity"] = "all"`, `min_group_rows: int = 20`,
  `max_groups: int = 500 (le=5000)` -- refuse rather than emit a
  4,000-row frame.
- Result: `n_rows`, `n_groups`, `quantile_levels`, `pinball: Dict[str,
  Stat]`, `quantile_crossing_rate`, `quantile_coverage`, `quantile_width`,
  `interval_coverage`, `interval_width`, `interval_nominal_coverage`,
  `groups: List[{key, n_rows, interval_coverage, quantile_coverage_*,
  crossing_rate}]`, `worst_group`, `best_group`, `warnings`. The keys are
  the ones `distributional_metrics` returns and
  `tests/modeling/test_distributional.py:186-211` pins.
- Refusals: no quantile column and no lower/upper; `lower` without
  `upper`; target column absent; every row NaN; `by="entity"` without an
  `entity` column; more groups than `max_groups`.
- Warnings: when `by="all"`, the exchangeability sentence from
  `conformal.py:23-27` (a pooled coverage cannot separate 97% in calm
  from 62% in a selloff); crossing rate above 0 (the pair is not a
  distribution on those rows); groups under `min_group_rows` flagged as
  too short to read; |coverage - nominal| beyond a binomial 2-sigma band.

**4A.2 `compare_signals` (G4).** `holm_adjust` reaches the tool layer only
inside `_paired_against_reference` (`tools.py:1908-1999`), which requires
two registered models, refuses cpcv (`:1917-1923`), and enforces a star
topology; `compare_ic_series` (`comparison.py:73`) and
`newey_west_variance` (`:151`) have no caller outside their module. The
research and portfolio runtimes print "may not survive HAC (Newey-West)
errors" (`research/tools.py:564`, `portfolio/tools.py:401`) with the
estimator one import away. The backtest runtime's three multiple-testing
tools all take return series, none a p-value.

- Modeling runtime (its `paired` inputs are `predictions` refs, which only
  modeling tools produce). Input: `mode: Literal["paired","ic_series","adjust"]`;
  paired -> `predictions_ref_a/_b`, `task`, `metric: Literal["cs_rank_ic","cs_ic"]`,
  `horizon`, `n_bootstrap (100..20000)`, `block_size`, `confidence`, `seed`;
  ic_series -> `ic_a`, `ic_b` as `Dict[date, float]` (>= 10 shared dates,
  the refusal at `comparison.py:98-103`) plus the bootstrap knobs, and
  `hac_variance_lag0`, `hac_variance`, `hac_ratio` from `newey_west_variance`
  (the measured 2.8x); adjust -> `p_values: Dict[label, float]` (1..1000,
  each in [0, 1]), `method: Literal["holm","bonferroni","bh"]`, `alpha`.
  Fields for another mode present -> refuse naming them.
- **The one piece of new arithmetic:** `bonferroni_adjust` and `bh_adjust`
  beside `holm_adjust` in `comparison.py`, same contract (monotone,
  capped at 1, `[] -> []`), one shared test; a `bh` warning that it
  controls FDR, not FWER, so a rejection is a different claim.
- Result: `mode`, the per-mode block (`compare_ic_series` keys;
  `paired_comparison` keys), `adjusted: List[{label, p_value, p_adjusted,
  reject_at_alpha}]`, `n_tests`, `method`, `warnings` -- always carrying
  the `tools.py:1990-1999` sentence: Holm controls the family-wise error
  of THESE tests and not for the candidates having been selected on the
  same sample, which is `run_reality_check`'s job.

### 4B -- feature lab: the two panel screens (owner of `feature_tools.py`/`feature_models.py`; `limits.py`)

**4B.1 `screen_feature_significance`.** `run_feature_permutation_test`
(`feature_tools.py:485`) takes one feature; its docstring calls
`null_p95_abs` "the honest floor for `select_features(min_abs_rank_ic=...)`"
(`:518-521`) while `select_features` takes the floor as a number the agent
invents (`feature_models.py:410-419`). The document measured the loop on a
real panel: a naive 0.02 floor keeps 4 of 10 features, the permutation
floor (max `null_p95_abs` = 0.0694) keeps 0.

- Input: `dataset_id`, `features: Optional[List[str]]` (default all, via
  `_resolve_features` `:91`), `n_permutations: int = 200 (20..5000)`,
  `method`, `null: Literal["circular_shift","within_date"] = "circular_shift"`,
  `random_seed`, `max_draws: int = 20_000`. Budget: refuse when
  `len(features) * n_permutations > max_draws`, naming the product and the
  two ways out, in the shape of `run_feature_ablation`'s refusal
  (`:608-616`). Measured 1.6 ms/draw, so the default is ~30 s and the
  ceiling `limits.MAX_PERMUTATION_DRAWS = 200_000` (new, declared in
  `limits.py` for the reason at `limits.py:10-18`) is ~5 min.
- Result per feature: `feature, rank_ic, p_value, null_p95_abs,
  ic_autocorrelation_lag1, significant_at_05, n_usable_permutations`; top
  level `honest_floor = max(null_p95_abs)`, `floor_feature`, `n_features`,
  `n_significant`, `n_kept_at_floor`, `warnings`: the floor sentence with
  the count each floor keeps; the `within_date` caveat from
  `feature_stability.py:379-384` when any lag-1 autocorrelation exceeds
  0.3; the family-wise sentence (`0.05 * n` significant by chance)
  cross-referencing `compare_signals(mode="adjust")`.

**4B.2 `screen_feature_stability`, with the drift curve folded in (D-7).**
`feature_stability` (`feature_stability.py:214`) and `feature_drift`
(`:123`) are single-feature; `analyze_features` is silent about time.
Thresholds `PSI_MODERATE = 0.10`, `PSI_SIGNIFICANT = 0.25` (`:50-51`),
verdicts `significant|moderate|stable` (`:182-186`). `monitoring.py:45-68`
carries a parallel vocabulary (`severe`); the tool uses the feature-lab
one so it agrees with `get_feature_drift`, and says so in a comment.

- Input: `dataset_id`, `features`, `n_blocks: int = 4 (2..20)`, `method`,
  `split_date`, `reference: Literal["first","previous"] = "first"`,
  `max_features: int = 200`.
- Result per feature: the `feature_drift` fields (`psi`, `psi_verdict`,
  `ks_statistic`, `ic_before/after`, `ic_flipped`) and the
  `feature_stability` fields (`ic_overall`, `ic_block_mean/std`,
  `sign_consistency`, `worst_block`), plus `psi_by_block: List[{block,
  start, end, psi, psi_verdict}]` -- the D-7 curve, ~25 lines: the block
  split at `:245` already exists, and `population_stability_index(reference,
  block)` with the reference fixed (`first`) or rolling (`previous`) is
  one call per block. Top level: `n_significant`, `n_moderate`,
  `n_stable`, `psi_thresholds`, `most_drifted`, `warnings` (a feature at
  or above 0.25 is no longer the same measurement, so its full-sample IC
  describes neither side; the "conventions, not tests" caveat from
  `:46-49`; high `sign_consistency` with decaying block ICs).

### 4C -- `predict_survival_curve` (library in `scoring.py`; tool wiring after 4A)

A survival model trains end to end (`concordance`, `integrated_brier`)
and no curve comes out: `adapters.py:386-402` builds the
`survival_function(times)` closure and hands it to `survival_metrics`,
which creates the `(n_rows x n_times)` matrix at `validation/survival.py:250`
and keeps three scalars; `score_model` returns `estimator.predict`
(`adapters.py:383-384`), the risk. The machinery is intact:
`CoxPHRegressor.predict_survival_function(X, times)` (`estimators/survival.py:272`),
the xgboost variants (`:362`, `:459`), and `load_model`
(`model_registry.py:516`) returns the fitted baseline knots.

- Factor `score_model`'s feature-matrix build (`scoring.py:106+`, after
  every provenance, universe and staleness gate) so a
  `survival_curves(model_id, as_of, ..., times)` sibling reuses it.
- Input: `ScoreModelInput`'s fields (`models.py:474-515`) plus `times:
  Optional[List[float]]`, `n_times: int = 32 (2..256)` (default grid from
  the baseline knots' quantiles), `include_matrix: bool = False`,
  `max_matrix_cells: int = 50_000`.
- Result: `model_id`, `as_of`, `times`, `per_entity: List[{entity, risk,
  survival_at_times: List[Stat], median_survival: Stat}]`,
  `survival_mean_curve`, `n_baseline_knots`, `warnings`.
  `median_survival` is the first `t` with `S(t) <= 0.5` and `None` when
  the curve never crosses inside the grid -- never the grid's last point.
- Refusals: `manifest.task != "survival"`; an estimator without
  `predict_survival_function`, naming the ones that have it;
  `include_matrix` over the cell cap; every gate `score_model` already
  enforces. Warnings: the grid ends before the last knot (truncated, a
  `None` median is not "never"); proportional hazards means the ordering
  is the model's claim and the level is the baseline's.
- **External survival-curve scoring (IPCW/Brier on caller-supplied
  matrices) is deferred**, recorded in section 8: the four functions are
  live inside `survival_metrics`, and no caller but this tool can produce
  the matrices they take.

Tests (`tests/modeling/test_capability_gaps_phase4_intervals.py`,
`..._signals.py`, `..._screens.py`, `..._survival.py`): 18 of 20 rows
covered gives `interval_coverage == 0.9` and a hand-computed pinball; a
100%-covered band ten times the target's IQR still scores 1.0 and the
width exposes it; `by="date"` with identical per-date coverage gives zero
spread and no regime warning; `q05 > q95` on 1 of 4 rows gives
`quantile_crossing_rate == 0.25` and the crossing warning; no interval
columns refused naming what was looked for. Through the tool,
`[0.01, 0.04, 0.03]` -> holm `[0.03, 0.06, 0.06]`, bonferroni
`[0.03, 0.12, 0.09]`, bh `[0.03, 0.04, 0.045]`; twelve U(0,1) p-values
reject nothing after Holm in >= 95% of seeds; two identical refs give
`mean_difference == 0`, `indistinguishable`, `hit_rate is None`; an AR(1)
phi=0.8 IC difference has a strictly wider block-bootstrap CI than
`block_size=1`; white noise has blocked ~ iid. One planted feature among
nine noise columns is the only `significant_at_05` and `honest_floor`
exceeds every noise |IC|; ten noise features give `n_significant <= 1`
and the floor sentence says a naive 0.02 keeps zero; a constant column
is never significant and its IC is `None`; 50 x 5000 is refused naming
250,000 against the cap. A feature shifted +3 sigma in its second half
is `significant` and `most_drifted`, the rest `stable`; a stationary panel
has zero significant; a stable distribution whose IC dies is `stable`
AND `ic_flipped` (the two failures stay separable); a monotone drift
rises block over block under `first` and stays flat under `previous`. A
Cox fit where the high-risk row's `median_survival` is strictly below the
low-risk row's and every curve is monotone; an estimator without curves
is refused with no fallback to the risk; a grid before the first knot
gives `S == 1.0` everywhere and a `None` median with the truncation
warning; a non-survival model is refused.

**Surface after phase 4:** modeling +3, feature_lab +2; section 7.

---

## 5. Phase 5 -- The registry's security control gets an interface (G8, G9, D-3..D-6)

D-3 holds by construction: `verify_model_package` (`package.py:83-88`)
runs the signature branch only `if signed or require_signature`
(`:109-110`); `manifest.json` cannot hash itself and is subtracted at
`:108`, so an edited manifest with its `.sig` deleted verifies `ok=True`
under the defaults `inspect_model` uses (`tools.py:823`, bare). D-4
holds: `registry/lifecycle.py` has no verification call (grep is empty),
`promote()` refuses five things and integrity is not one (`:119-148`).
The audit subsystem got the control (`meta/tools.py:373-394`,
`VerifyAuditIntegrityInput.public_key_path` `agent/models.py:4596-4612`);
the model registry did not. Order inside the phase: D-6, then D-5, then
G8, then G9 -- the defect blocks the mirror tools.

### 5A -- `package.py` / `model_registry.py` / `engine.py` (the two defects)

**5A.1 D-6 `mirror_model_package(prefix=...)` is write-only.** `prefix`
(`package.py:124, 145-146`) has zero callers in `src/` and `tests/`
(`model_registry.py:308` and `mirror.py:46` use the model id);
`list_remote_models` filters `key.startswith("mdl_")` (`:173`) and
`pull_model_package` refuses a manifest naming another id (`:216-220`).
Delete the parameter. One deleted argument beats a remote-prefix concept
threaded through two functions and a tool schema.

**5A.2 D-5 monitoring and OOS URIs are absolute paths of the registering
machine.** Writers: `model_registry.py:201-212` (`feature_reference_uri`,
`prediction_reference_uri`) and `:251` (`oos_predictions_uri`, from
`engine.py:1462-1464`, all `str(path)` out of `save_artifact`). Readers:
`load_monitoring_reference` (`model_registry.py:434-449`, the
`Path(uri).exists()` gate at `:435/:443` decides which D-5 symptom fires),
`portfolio_eval.py:682-694`, `bridge.py:281-300`, `ensemble.py:66`,
`tools.py:803`. `pull_model_package` copies the bytes into the new root
correctly (`:243-256`) and the manifest verbatim, so the URI still names
the source root: on another machine `.exists()` is False and the tool
says "registered before monitoring references were kept ... Retrain"
(`tools.py:996-1001`); in a second `SQT_RUNS_DIR` on the same machine
`verify_file` passes against the OLD root and `load_artifact` raises
`escapes SQT_RUNS_DIR` (`_runspath.py:86-88`).

- Store bare filenames relative to the model directory
  (`feature_reference.parquet`, `prediction_reference.parquet`,
  `oos_predictions.parquet`) -- exactly what `_filename_for`
  (`package.py:77-80`) derives from `content_hashes`, so mirror and pull
  already land them there.
- One resolver in `model_registry.py`, `resolve_model_artifact(model_id,
  uri) -> Path`: `Path(uri).name` under `_artifacts.run_dir(model_id)`
  when that file exists (covers the new relative form AND a legacy
  absolute path naming the same file); else the literal legacy absolute
  path when it lies within this runs root; else a refusal naming the
  model directory. Route through `run_dir` so the legacy fallback cannot
  re-open the escape. All five read sites use it. No manifest rewrite,
  no version bump; `content_hashes` keys were never paths.
- Split the D-5 message: keep "registered before monitoring references
  were kept ... Retrain" only when `manifest.monitoring` has no `*_uri`
  key; when the key exists and the file is absent, say the reference is
  not in the model's directory and, if pulled, to re-pull.
  `tests/modeling/test_lifecycle_monitoring.py:292-306` matches "Retrain"
  on the no-key branch; keep that wording there.

### 5B -- attestation and the promotion gate (owner of `tools.py`/`models.py`)

**5B.1 `attest_model_package` (G8).** Input (`extra="forbid"`): `model_id`,
`require_signature: bool = True`, `public_key_path: Optional[str]`
(description cloned from `agent/models.py:4598-4602`). Handler:
`verify_model_package(model_id, require_signature=..., public_key=path)`
(a str path is accepted, `signing.py:103-107`). Result: `ok`,
`verified`, `mismatched`, `missing`, `unhashed`, `signature`,
`signature_error`, `key_pinned` (from `signature["key_pinned"]` -- there
is no top-level field, the document is wrong there), `manifest_sha256`,
`warnings` (a valid signature under an unknown key proves the manifest
and signature were written together, not that anyone you trust wrote
them -- `signing.py:242-245`). **The tool's default is
`require_signature=True`; the library default stays `False`**: the tool
is new, so nothing depends on it, and an attestation that returns ok on
an unsigned package reproduces D-3 at a new address; the library default
cannot move because `mirror_model_package` (`package.py:134`) and
`inspect_model` call it bare.

**5B.2 The promotion gate.** `PromoteModelInput` (`models.py:598-623`) gains
`require_verified_package: bool = True`, `require_signature: bool =
False`, `public_key_path: Optional[str] = None`. Default True is safe:
`extra="forbid"` means no caller passes it today, and every promotion in
the suite is of a clean package; a model registered before content
hashing has empty `content_hashes` -> empty `verified` -> `ok=True`.
Default False on the signature: a shop with no signing key would be
locked out of production, and `signing_available()` may be False
entirely. The gate lives in the HANDLER `promote_model` (`tools.py:938`),
not in `lifecycle.promote`: `package.py:34` imports `lifecycle`, so the
other direction is a cycle (`mirror.py:12-14` records the same
reasoning), and the library call sites in tests stay green. Refusal
text names what mismatched and says a stage is a statement that somebody
read the evidence, so the evidence must be the one that was registered.
Prepend `manifest_sha256=<sha>` and `package_verified=<n> files` to
`evidence` (`signing._manifest_bytes` `:75-80` gives the bytes; the full
64-hex sha as `signing.py:259` computes it). `PromoteModelResult` gains
`manifest_sha256`, `package_ok`.

**5B.3 `inspect_model`** gains `public_key_path` (`InspectModelInput`
`models.py:714-727`) threaded into `tools.py:823`. Not `require_signature`
-- it would do nothing in three of four views, and
`tests/modeling/test_artifact_store.py:143-148` pins the bare shape.

### 5C -- the mirror tools (after 5A and 5B; owner of `tools.py`/`models.py`)

**5C.1 `list_remote_models`, `pull_model_package` (G9).** Zero of the 31
tools take a store; the push happens only if `SQT_MODEL_MIRROR_URL` was
set before the process started (`mirror.py:27-47`, called at
`model_registry.py:305-308` and `lifecycle.py:163`). Store construction
is `store_from_url` (`artifact_store.py:270-277`): a bare path or
`file://` is local, anything else `FsspecArtifactStore` (`:211-225`),
and an unknown scheme is a bare `ValueError` from fsspec ("Protocol not
known"), which the tool must re-raise as a `ValidationError` naming the
schemes that work.

- `ListRemoteModelsInput`: `store_url: Optional[str]` (default the env
  var; refuse by name when neither), `limit: int = 200 (le=1000)`.
  Result: `store_url`, `models`, `n_total`, `warnings`.
- `PullModelPackageInput`: `model_id`, `store_url`, `require_signature:
  bool = False`, `public_key_path`, `overwrite: bool = False`. Result:
  `model_id`, `store_url`, `ok`, `verified/mismatched/missing/unhashed`,
  `signature`, `key_pinned`, `stage` (from `current_stage`; the promotion
  log travels), `registry_dir`, `warnings` (always one when `signature is
  None`, so the agent sees what it got).
- Refusals already produced downstream, by name: unsigned when required
  (`package.py:222-226`), already registered without `overwrite`
  (`:204-208`), wrong pinned key (`signing.py:240-245`), changed remote
  file (`package.py:249-254`), missing hashed file (`:239-242`).
  Containment holds through `LocalArtifactStore._path` ->
  `require_within` (`artifact_store.py:147-153`); a nested remote key is
  refused mid-pull by `validate_key` and the manifest is written last, so
  nothing becomes a registered model -- plant that.

Tests (`tests/modeling/test_capability_gaps_phase5_registry.py`, the
`memory://` + `uuid4()` pattern of `test_remote_registry.py`): `manifest.sig`
deleted and `manifest.json` edited -> `attest_model_package` says
`ok=False` matching "is not signed" while the bare library call still
says `ok=True` (D-3 as the gap the default closes); a wrong key ->
"not the pinned", `key_pinned False`; a corrupted `model.joblib` ->
`promote_model` refuses "does not verify", `promotions(model_id) == []`,
stage still `candidate`; the same with `require_verified_package=False`
succeeds and the evidence carries `manifest_sha256=`; a clean promotion's
`manifest_sha256` equals the sha of the manifest bytes. After mirror,
root switch and pull: `load_monitoring_reference` returns both frames
equal to the originals, `score_model` then `monitor_model` returns drift
rows in the second root (the document's "a pulled model that cannot be
monitored", closed), a legacy absolute URI in the current root still
resolves, `evaluate_model_portfolio` runs on the pulled model, and
`mirror_model_package(prefix=...)` is gone. Tool round trip: register
with the env var set -> `list_remote_models` finds it -> second root ->
`pull_model_package` gives `ok=True` and `stage == "validated"`;
`store_url="zzz://bucket/x"` is a `ValidationError`; unsigned with
`require_signature=True` refuses and the model directory does not exist;
second pull refuses without `overwrite`; a wrong `public_key_path`
registers nothing; a nested remote key refuses and leaves no
`manifest.json`.

**Surface after phase 5:** modeling +3; section 7.

---

## 6. Phase 6 -- Documentation, and the two documents go

- `Documentation/15_modeling.md`: the new tools, one subsection each in
  the pipeline order; the counts ("exactly 22 tools" at `:25`, the diagram
  at `:37`, "Those nine tools" at `:84`); a paragraph on the verified
  bridge (`backtest_model_signal` replaces the convert route for a
  registered model, and why); the promotion gate; the capability report's
  new `tasks`, `calibration` and target-detail shapes.
- `Documentation/29_modeling_reference.md` and `20_tool_index.md`
  regenerated (`tests/docs` fails otherwise); README runtime rows (`:20`,
  `:26`), the "211" total (`:12`, `:256`), "22-tool" (`:96`, `:251`);
  `13_agent_orchestration.md:41-43`, `18_mcp.md:601` (already stale at
  "20-tool"), `19_runtimes.md:128`; `25_testing.md` if the count of test
  layers' examples changes; `Development/tool_surface_analysis.md` where
  its section 2 proposals are now built (the joins it lists).
- `CHANGELOG.md`: one entry per phase, written at the phase's commit.
- Then delete `Development/modeling_capability_gaps.md` and this plan, as
  the Databento findings and plan were deleted once implemented: the
  CHANGELOG is the record, and section 8's deliberate non-changes move
  into the documentation where a reader would look for them.

---

## 7. Surface bookkeeping (once per phase that adds a tool)

Every new tool trips all of these; the integrator does them together at
the end of the phase, after the agents' file sets are merged.

| pin | what moves |
|---|---|
| `tests/modeling/test_agent_tools.py:128-200` | the EXACT modeling name set; `:203-210` advertised == dispatchable |
| `tests/modeling/test_feature_tools.py:639-658` | the EXACT feature-lab name set; `:670-678` no overlap; `:681-690` parametrized |
| `tests/modeling/test_calibration.py:256`, `test_lifecycle_monitoring.py:310` | `len(MODELING_TOOL_DISPATCH) == 22` |
| `tests/test_wrong_numbers.py:1911-1923` | 10 runtimes, 211 total, 211 catalog, 180 facade (unchanged: modeling sits outside it) |
| `tests/agent/test_multi_agent_tool_coverage.py:66-78` | every tool claimed by exactly one worker in `Multi_Agent_Implementation/worker_agents.py` (`_MODEL_BUILDER_TOOLS` at `:125-139` and the feature-lab list) |
| `tests/docs/test_documentation.py` | `20_tool_index.md` regenerated byte for byte; `29_modeling_reference.md` when the reference inputs change; README counts; per-runtime "serves N tools" strings; spelled-out counts |
| `tests/mcp/test_mcp_surface.py:195-239` | the whole surface at `detail_budget=98_304` must keep every tool advertised (thinning, never dropping); self-adjusting, but the modeling runtime is already the largest at 82 KB and a dozen new schemas will push it, so check `context_bytes()` after phase 4 |
| README rows | `modeling | 22 | 82 KB`, `feature_lab | 9 | 29 KB`: count and measured KB |

New tools register in exactly two places each: `_MODELING_TOOL_DEFS`
(`tools.py:2187`) + `MODELING_TOOL_DISPATCH` (`:2656`), or
`FEATURE_TOOL_DEFS` (`feature_tools.py:671`) + `FEATURE_TOOL_DISPATCH`
(`:774`), with the input model in the matching models module; the runtime
(`agent/runtimes/__init__.py:440-483`) and the MCP catalog
(`mcp/catalog.py:259-280`) pick them up by name.

Expected totals when every phase has landed: modeling 22 -> 37 (+3 phase
2, +6 phase 3, +3 phase 4, +3 phase 5), feature_lab 9 -> 11 (+2 phase
4), catalog 211 -> 228. The facade's 180 does not move. The heaviest-
runtime share (`tests/surface/test_invariants.py:370-381`, under 45% of
the total) has room: modeling is ~85 KB of ~358 KB today, and the new
inputs are small (the one expensive schema, `ModelSpec`, is already
inlined twice).

New tools may live in their own module pair under `modeling/agent/`
(the precedent is `dataset_tools.py`), which is how a phase's agents
avoid the `tools.py` seam; a tool that needs `tools.py`'s private helpers
(`_oos_with_actuals`, `_panel_with_selected_target`, `_load_dataset_panel`)
lives in `tools.py` and belongs to that phase's owner of the file.

---

## 8. Where the document was wrong, and what is deliberately not done

**Corrections to the document, from the grounding passes:**

- `HAS_XGBOOST_SURVIVAL` is not an alias of `HAS_XGBOOST`: its right-hand
  side registers two estimators, and three tests import it. The name is
  deleted; the registration stays (1B.7).
- `rank_turnover` is not dead: `search.py:180` calls it. What is true is
  that no tool reports the quantity (1A.4).
- `capabilities.tasks` advertising `ranking` with zero estimators is an
  artefact of the document's environment (no lightgbm, no xgboost); here
  ranking has two. The static-list defect is real everywhere and is fixed
  structurally (1B.2).
- `PackageVerification` has no top-level `key_pinned`; it lives in
  `signature` (5B.1). `tests/modeling/test_registry.py` pins nothing about
  packages; `test_artifact_store.py` does.
- `monitoring.py` reads no URI; `model_registry.load_monitoring_reference`
  does (5A.2). And `distribution` is not what `uncertainty_scaled` reads
  -- the frame's columns are (2B.1).
- The four IPCW/Brier functions are not dead: `survival_metrics` calls
  them. They are dead as ENTRY POINTS, because `score_predictions(task=
  "survival")` never passes a survival function (4C).
- `local_store`: the earlier tool-surface analysis kept it as "the
  documented entry point"; it is documented nowhere and called by nothing
  (1B.8).

**Deliberately not done, and why:**

- `register_estimator` / `register_target` / `register_preprocessor`,
  `partial_fit`, a `get_modeling_limits` tool, a cache tool: the
  document's section 5 closes these and the reasons hold (the allowlist
  is the security boundary; warm starts break the purge; the limits are
  already in the schemas).
- A standalone external survival-curve scorer taking three inline
  matrices: no caller but `predict_survival_curve` can produce them; the
  curve tool is the door, and IPCW scoring on caller matrices waits for a
  caller (4C).
- A standalone drift-curve tool: folded into `screen_feature_stability`
  as `psi_by_block` (4B.2), which avoids a tool that is one field of
  another.
- Changing `verify_model_package`'s library default to require a
  signature: it would make every unsigned in-house model unmirrorable and
  uninspectable; the tool default is the strict one instead (5B.1).
- Running the backtest inside `backtest_model_signal`: the fill price and
  the costs are backtest decisions; the tool publishes the panel and the
  backtest runtime prices it (2A.1).
- `adapters.input_kind` (one value on every entry): left alone; a field
  with one value is noise, and removing it moves the generated reference
  for no reader's benefit.
