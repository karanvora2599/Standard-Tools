# Survey: `modeling/` (excluding `modeling/agent/`) — implemented vs. exposed

Scope: every module under `src/standard_quant_tools/modeling/` except `modeling/agent/`, ~22,100 lines across 80 files. Tool surface checked against: `modeling/agent/tools.py` (22 tools, `MODELING_TOOL_DISPATCH`), `modeling/agent/feature_tools.py` (9 tools, `FEATURE_TOOL_DISPATCH`), `modeling/agent/dataset_tools.py` (`explain_dataset_row_loss`), the other runtimes' `tools.py` (only `meta.convert_reference` and `data.validate_external_dataset` reach into this slice), and `mcp/resources.py` (`sqt://model/{id}`, `sqt://dataset/{id}`, `sqt://catalog/features|capabilities`).

Method: enumerated `^def |^class ` in every module, read each module in full, then grepped the tool files (and every `src/` file outside `modeling/`) for each name. Exposure codes used in every table:

| code | meaning |
|---|---|
| **D** | direct — a tool function calls it by name (tool named) |
| **S** | indirect through a spec field — reachable by setting the named field on `DatasetSpec` / `ModelSpec` / `PredictionTransformSpec` / etc. |
| **I** | internal — called by something a tool calls; carries no decision surface of its own (plumbing, correctly unexposed) |
| **N** | NOT exposed — a real capability (or a real mode of one) that no tool, spec field or resource reaches |
| **X** | dead — no production caller anywhere in `src/` (tests only, or nothing) |

Counts (rows in the inventory below, computed with grep on this file): **235 inventory rows; D=76, S=73, I=69, N=12, X=5.** "Exposed" = D+S+I = 218; "unexposed" = N+X = 17. A further 13 rows are exposed in their primary mode but carry a mode no tool reaches (marked **N** inside the cell, e.g. `verify_model_package(require_signature=...)`, `load_oos_predictions(keep_path=True)`, the bridge's `model_id` mode); counting those, 17 + 13 = 30 distinct unexposed capabilities.

---

## 1. Tool surface recap (what the 31 tools actually reach)

**modeling (22):** `list_features`, `build_model_dataset`, `register_external_panel`, `run_model_experiment`, `score_model`, `inspect_model` (views: summary / feature_importance / validation / lineage), `list_models`, `list_datasets`, `promote_model`, `monitor_model`, `compare_models` (headline / paired), `check_leakage`, `validate_model_spec`, `score_predictions`, `analyze_model_errors`, `explain_dataset_row_loss`, `validate_pit_records`, `join_point_in_time`, `build_model_ensemble`, `evaluate_model_portfolio`, `analyze_features`, `list_modeling_capabilities`.

**feature_lab (9):** `profile_feature`, `get_feature_redundancy`, `get_feature_ic_decay`, `select_features`, `compare_feature_sets`, `get_feature_drift`, `get_feature_regime_stability`, `run_feature_permutation_test`, `run_feature_ablation`.

**Indirect routes that count as exposure:** `EstimatorSpec.type` (19 always-on + 8 optional estimators), `PreprocessingSpec.steps[].type` (8 steps) / `normalization`, `TargetSpec.type` (6 buildable + 12 external labels), `FeatureSpec.id` (26 features) / `.lags` / `.alias`, `ValidationSpec.method` (3 splitters), `WeightingSpec.method`, `SearchSpec.method` (grid/random/tpe), `ModelSpec.quantiles` / `.intervals` / `.budget`, `DatasetSpec.missing` / `.calendar` / `.provider`, `PredictionTransformSpec`, `PortfolioSimSpec`; `meta.convert_reference(to_kind="signal_panel")` -> `bridge.oos_predictions_to_signal_panel` (URI mode only); MCP resources for manifests, dataset panels, the feature catalog and capabilities.

---

## 2. Inventory

### 2.1 `modeling/*.py` (top-level)

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| adapters.py | `FitArrays` | exactly what `estimator.fit` receives, with the `SampleIndex` of those rows | X, y, sample_weight, group, index | **I** — engine |
| adapters.py | `accepts_missing` | does a default instance accept NaN in X (sklearn tags, both APIs) | cls -> bool | **D** — `list_modeling_capabilities` (`accepts_missing` per estimator); engine/scoring refusals |
| adapters.py | `ModelAdapter` | base: prepare / score / metrics / fold_ic / capabilities | spec, SampleIndex, X, y -> FitArrays, scores, metric dicts | **S** — `ModelSpec.task` |
| adapters.py | `RegressionAdapter` | regression metrics + pooled/CS IC | as above | **S** — `task="regression"` |
| adapters.py | `ClassificationAdapter` | positive-class probability as score; accuracy/AUC/base rate | as above | **S** — `task="classification"` |
| adapters.py | `RankingAdapter` | sort by (date, entity), relevance grades, query groups; NDCG + CS IC | as above | **S** — `task="ranking"` + `RankingSpec` |
| adapters.py | `SurvivalAdapter` | (duration, event) label; risk score; concordance | as above | **S** — `task="survival"` |
| adapters.py | `get_adapter`, `available_tasks` | adapter lookup / task list | task -> adapter; () -> list | **D** — `list_modeling_capabilities.tasks` |
| adapters.py | `_exposes_coefficients`, `_probes_true` | capability probes (class membership, default-instance hasattr) | cls -> bool | **I** — via `capabilities()` |
| artifacts.py | `run_dir` | `SQT_RUNS_DIR/<id>` resolved inside the runs root | id -> Path | **I** — every tool |
| artifacts.py | `local_store` | the runs dir as a `LocalArtifactStore` | () -> store | **X** — no caller in src or tests |
| artifacts.py | `save_json`, `load_json`, `save_joblib`, `load_joblib` | atomic, NaN-safe JSON / joblib IO | dir, name, payload -> path | **I** |
| artifacts.py | `hash_file`, `verify_file` | 16-hex SHA-256; refuse on digest mismatch (None = predates hashing) | Path, expected -> None/raise | **I** — every loader; surfaced via `inspect_model(view="lineage").package` |
| bridge.py | `oos_predictions_to_signal_panel` | OOS predictions -> `{ticker: {date: -1/0/+1}}`; `model_id` mode verifies digest, refuses cpcv and skipped folds; URI mode is structural-only | model_id \| uri, task, proba_threshold, deadband, long_only -> dict | **D** — `meta.convert_reference(to_kind="signal_panel")` in **URI mode only**; the safer `model_id` mode is **N** (no tool passes a model id, so hash verification / cpcv refusal / skipped-fold refusal never run on the tool path) |
| bridge.py | `_refuse_cpcv`, `_validate_predictions_frame`, `_assert_continuous_calendar` | guards reused by portfolio_eval | manifest/frame/dates -> raise | **I** — `evaluate_model_portfolio` |
| cache.py | `FoldCache` | preprocessed (train, test) matrices keyed by plan hash, exact column projection for column-wise pipelines | key, feature_ids -> frames; stats | **I** — `run_model_experiment` (`validation_report.cache`), `run_feature_ablation` (`preprocessing_reused`) |
| cache.py | `column_wise_pipeline` | every step column-wise? (read off registry) | steps -> bool | **I** |
| calendar.py | `calendar_available` | is `exchange_calendars` importable | () -> bool | **D** — `list_modeling_capabilities.optional_dependencies.exchange_calendars` |
| calendar.py | `require_calendar_library` | refuse by name | where -> raise | **I** |
| calendar.py | `interval_minutes` | minutes per intraday bar ('5m','1h') or None | str -> int\|None | **I** |
| calendar.py | `calendar_names` | every exchange_calendars code | () -> list[str] | **N** — only the first 8 appear inside a refusal message; nothing lets an agent list them |
| calendar.py | `validate_calendar_name` | refuse an unknown code, listing a sample | name -> name | **S** — `DatasetSpec.calendar` validator |
| calendar.py | `sessions_per_year` | sessions counted over complete calendar years | calendar -> float | **I** — via `periods_per_year` |
| calendar.py | `session_minutes` | median full-session length over the last 250 sessions | calendar -> float | **I** |
| calendar.py | `bars_per_session` | ceil(session / interval) | interval, calendar -> int | **I** |
| calendar.py | `periods_per_year` | bars per year for an intraday interval on a named venue | interval, calendar -> int | **I** — `features.base.periods_per_year_for_interval` -> `risk.*` annualization, `evaluate_model_portfolio`; **N** as a question ("what does '1h' on XLON annualize to") |
| capabilities.py | `estimator_capabilities` | one record per (task, estimator) actually registered | () -> list[dict] | **D** — `list_modeling_capabilities` |
| capabilities.py | `modeling_capabilities` | tasks, estimators, features, targets (from registry), validation, preprocessing steps (from registry), weighting (hand-written), search, optional deps, native detail | () -> dict | **D** — `list_modeling_capabilities`; MCP `sqt://catalog/capabilities` |
| capabilities.py | `_native_detail`, `_native_available`, `_importable`, `_literal_options` | extension staleness, find_spec probes, Literal choices | -> dict/bool/list | **I** — inside the above; `_native_detail` also feeds `environment_fingerprint` |
| diagnostics.py | `residual_summary` | n, bias, MAE, RMSE, std, skew, kurtosis, tails | actual, predicted -> dict | **D** — `analyze_model_errors` |
| diagnostics.py | `heteroskedasticity` | corr(\|error\|, \|prediction\|) | arrays -> float\|None | **D** — `analyze_model_errors` |
| diagnostics.py | `residual_autocorrelation` | lag-1 pooled within-entity, centred per entity | joined frame -> float\|None | **D** — `analyze_model_errors` |
| diagnostics.py | `calibration` | regression slope/intercept/dispersion; classification Brier/ECE/reliability (refuses non-0/1) | arrays, task -> dict | **D** — `analyze_model_errors` |
| diagnostics.py | `error_attribution` | RMSE etc. by entity / period / prediction decile / feature decile, thin flags | joined, feature, period -> dict | **D** — `analyze_model_errors` |
| diagnostics.py | `worst_buckets` | headline sentences for buckets >= 1.5x the best | report -> list[str] | **D** — `analyze_model_errors` |
| diagnostics.py | `_bucket_report`, `_sort_key`, `_finite` | helpers | | **I** |
| engine.py | `run_experiment` | plan -> fold loop (purge, preprocess, search, fit, quantiles, conformal, metrics) -> pooled IC -> refit -> register | dataset dict, ModelSpec, dataset_id, register, fold_cache -> result dict | **D** — `run_model_experiment`; `run_feature_ablation` (`register=False`) |
| engine.py | `_calibrated` | wrap classifier in `CalibratedClassifierCV` on inner folds | estimator, spec, n -> estimator | **S** — `EstimatorSpec.calibration`, `.calibration_folds` |
| engine.py | `_fit_quantile_models` | one fit per quantile level with the registry's fixed objective | -> {q: model} | **S** — `ModelSpec.quantiles` |
| engine.py | `_conformal_radius` | split-conformal radius from held-out date blocks | -> (radius, n) | **S** — `ModelSpec.intervals` |
| engine.py | `_fold_sample_weights` | weights for one fold | spec, SampleIndex -> array\|None | **S** — `WeightingSpec.method`, `.half_life_days` |
| engine.py | `_preprocess`, `_refuse_missing_after_preprocessing` | fold-boundary pipeline; NaN refusal by name | -> (train_X, test_X) | **S** — `PreprocessingSpec`, `MissingDataSpec.policy="keep"` |
| engine.py | `_check_task_target_compatibility`, `_validate_classification_target`, `_validate_survival_target` | pre-fit refusals | panel/target_id -> raise | **I** — (`validate_model_spec` re-derives the first from metadata) |
| engine.py | `_fit`, `_predict_fold`, `_labels`, `_instantiate`, `_target_horizon` | fit with weights/groups; per-fold metrics; label reader; ctor with seed | | **I** |
| ensemble.py | `load_oos_predictions` | one model's OOS frame, digest-verified; `keep_path` keeps cpcv paths | model_id, keep_path -> DataFrame | **D** — `build_model_ensemble`, `compare_models(paired)`, `analyze_model_errors`; `keep_path=True` mode **N** |
| ensemble.py | `combine_predictions` | mean / median / rank_mean / weighted over the intersection, with per-model coverage, pairwise correlation and its basis | model_ids, method, weights -> dict | **D** — `build_model_ensemble` |
| ensemble.py | `_check_tasks`, `_rank_within_date`, `_pairwise_correlation` | task-compatibility refusal; centred within-date ranks; corr matrix -> pairs | | **I** |
| limits.py | `MAX_LAG`, `MAX_LAGS_PER_FEATURE`, `MAX_EXPANDED_COLUMNS`, `DEFAULT_MAX_FITS`, `MAX_FITS_CEILING` | schema-visible bounds | constants | **S** — JSON schema of `FeatureSpec.lags`, `ComputeBudgetSpec.max_fits` |
| monitoring.py | `reference_sample` | seeded row sample kept at registration | frame, cols, rows, seed -> frame | **I** — `run_experiment` -> `save_model` |
| monitoring.py | `feature_profile` | decile edges, missing rate, moments per feature | frame, ids -> dict | **I** — same; also persisted as `feature_profile.json` (not shown by any view) |
| monitoring.py | `drift_report` | PSI, KS, missing rates, status per feature | ref, cur, ids -> list[dict] | **D** — `monitor_model` |
| monitoring.py | `prediction_drift` | same on the prediction column + moments | ref, cur -> dict | **D** — `monitor_model` |
| monitoring.py | `realized_ic` | per-date rank IC of scored predictions vs outcomes, z vs validation | preds, outcomes, val mean/std -> dict | **D** — `monitor_model(outcomes_ref=...)` |
| monitoring.py | `THRESHOLDS`, `_status` | PSI/KS conventions, status rule | | **D** — reported by `monitor_model` |
| plan.py | `fits_per_estimator` | fits one "fit" costs (calibration folds, quantiles, conformal blocks) | spec -> int | **I** |
| plan.py | `fit_count` | spec-only fit arithmetic over n folds | spec, n_folds -> int | **D** — `validate_model_spec` (no dataset) |
| plan.py | `FoldPlan` | one fold: positions, date span, purged rows, inner folds, fits, node/preprocessing hash, `to_dict()` | | **N** — `to_dict()` (the per-fold schedule, purge counts and content hashes) is never returned by any tool |
| plan.py | `ExperimentPlan` | the schedule + `within_budget`, `refuse_over_budget`, `to_dict()` | | **D (partial)** — `validate_model_spec` reads only `.n_fits`; `run_model_experiment` executes it; `to_dict()` **N** |
| plan.py | `plan_experiment` | pure function of spec + date axis (+ panel: purge rows) | spec, dates, panel, hash, feature_ids -> ExperimentPlan | **D (partial)** — as above |
| portfolio_eval.py | `predictions_to_score_panel` | long OOS -> wide date x entity, classification recentred | frame, task -> panel | **D** — `evaluate_model_portfolio` (note: `meta.convert_reference` has its own copy of this logic) |
| portfolio_eval.py | `select_rebalance_dates` | first date per week/month | dates, freq -> DatetimeIndex | **S** — `PredictionTransformSpec.rebalance_frequency` |
| portfolio_eval.py | `apply_exposure_targets`, `_cap_book` | hit gross AND net exactly via long/short book split; iterative cap redistribution; shortfall reported | row, gross, net, cap -> (weights, diag) | **S** — `PredictionTransformSpec.gross_exposure/net_exposure/max_position_weight` |
| portfolio_eval.py | `scale_by_uncertainty` | prediction / conformal width | frame -> frame | **S** — `method="uncertainty_scaled"` |
| portfolio_eval.py | `transform_predictions_to_weights`, `_raw_weights_for_group`, `_quantile_counts` | score panel -> target weights, grouped by availability pattern, per-date exposure enforcement + diagnostics | panel, spec, returns -> (weights, diag) | **S** — `PredictionTransformSpec.method`; **N** as a standalone conversion (only reachable through a *registered* model; `meta.convert_reference(score_panel->weight_panel)` uses `backtest.sizing` and has no cap / net target / rebalance schedule / uncertainty scaling) |
| portfolio_eval.py | `_summarize_simulation` | Sharpe, CAGR, vol, Sortino, MDD, Calmar, turnover, cost drag floor | sim result -> dict | **I** |
| portfolio_eval.py | `evaluate_model_portfolio` | manifest -> verified OOS -> score panel -> rebalance dates -> weights -> `run_portfolio_simulation` -> metrics + provenance | model_id, transform, portfolio -> dict | **D** — `evaluate_model_portfolio` |
| samples.py | `SampleIndex` | dates / entities / label_end of the rows an estimator sees, in row order; `take`, `context` | | **I** |
| scoring.py | `score_model` | as_of guard (information cutoff), implementation-drift refusal, universe-scope pin, rebuild features, one cross-section, staleness, apply persisted pipeline, distribution columns, content-addressed artifacts | model_id, as_of, universe, lookback_days, max_staleness_days -> dict | **D** — `score_model` |
| scoring.py | `_deployed_preprocessing` | legacy-manifest transform resolution / refusal | manifest, id -> dict | **I** |
| specs.py | `FeatureSpec`, `TargetSpec`, `MissingDataSpec`, `DatasetSpec` | the dataset contract (aliases, lags, horizons, missing policy, provider, calendar) | pydantic | **D** — `build_model_dataset` input schema |
| specs.py | `EstimatorSpec`, `ValidationSpec`, `StepSpec`, `PreprocessingSpec`, `WeightingSpec`, `ParamRange`, `SearchSpec`, `RankingSpec`, `ComputeBudgetSpec`, `ConformalSpec`, `ModelSpec` | the experiment contract | pydantic | **D** — `run_model_experiment`, `validate_model_spec`, `run_feature_ablation` |
| specs.py | `PredictionTransformSpec`, `PortfolioSimSpec` | predictions -> weights; simulator subset | pydantic | **D** — `evaluate_model_portfolio` |
| specs.py | `_parse_date`, `TargetType`, `_known_target_type`, `_target_choices` | date parsing shared with `ScoreModelInput`; registry-backed target enum in the schema | | **I** |
| tasks.py | `TASKS`, `Task`, `SCORE_TASKS` | the four tasks; the three whose score is an ordering | constants | **S** — `ModelSpec.task` |

### 2.2 `modeling/analysis/`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| feature_ablation.py | `estimate_ablation_fits` | (n_features + 1) x n_folds | ints -> int | **D** — `run_feature_ablation` (refusal before fitting) |
| feature_ablation.py | `ablation_contributions` | baseline vs leave-one-out -> ranked contributions (sign-aware) | baseline, {f: score}, metric -> rows | **D** — `run_feature_ablation` |
| feature_ablation.py | `summarize_ablation` | best/worst/negative contributions + warnings | rows, metric -> dict | **D** — `run_feature_ablation` |
| feature_ablation.py | `_lower_is_better`, `_metric_value`, `DEFAULT_MAX_FITS` (=200) | metric orientation; value lookup; a second `DEFAULT_MAX_FITS` distinct from `limits.DEFAULT_MAX_FITS` (=500) | | **D** — imported by the tool (`_lower_is_better`, `_metric_value`); the constant is **X** (tool uses `FeatureAblationInput.max_fits`) |
| feature_report.py | `feature_distribution_stats` | coverage, moments, outlier rate, within-entity autocorr, rank turnover | panel, ids -> {f: stats} | **D** — `analyze_features`, `profile_feature` |
| feature_report.py | `feature_predictive_stats` | CS IC/rank IC + ICIR + quantile spread + monotonicity (against `target` only) | panel, ids, n_quantiles -> {f: stats} | **D** — `analyze_features`, `profile_feature`, `get_feature_redundancy`, `select_features`, `compare_feature_sets` |
| feature_report.py | `redundancy_report` | Pearson/Spearman matrices, VIF, condition number, union-find clusters | panel, ids, threshold -> dict | **D** — `analyze_features`, `get_feature_redundancy`, `select_features` |
| feature_report.py | `lead_lag_ic_curve` | IC vs feature shift; tent-shape leak screen with persistence abstention | panel, feature, max_shift, method -> dict | **D** — `analyze_features`, `get_feature_ic_decay`, `profile_feature(include_ic_decay)` |
| feature_report.py | `build_feature_report` | everything above + warnings, JSON-safe | panel, ids, ... -> dict | **D** — `analyze_features` |
| feature_report.py | `_rank_turnover`, `_quantile_shape`, `_correlation_clusters`, `_feature_persistence`, `_report_warnings`, `_frame_to_nested`, `_require_columns`, `_safe` | helpers | | **I** |
| feature_selection.py | `select_features` | one keeper per cluster, IC floor, cap; reason per drop | panel, ids, threshold, min_ic, max -> dict | **D** — `select_features` |
| feature_selection.py | `summarize_feature_set` | n, independent signals, mean/max \|IC\|, condition number | panel, ids -> dict | **I** — inside `compare_feature_sets` |
| feature_selection.py | `compare_feature_sets` | two sets on one panel with deltas and per-feature IC | panel, left, right -> dict | **D** — `compare_feature_sets` |
| feature_stability.py | `population_stability_index` | PSI with reference-quantile edges, floored | ref, cur, bins -> float | **D** — `monitor_model` (via `drift_report`), `get_feature_drift` |
| feature_stability.py | `ks_statistic` | two-sample KS (numpy) | ref, cur -> float | **D** — same |
| feature_stability.py | `feature_drift` | PSI/KS + IC before/after a split date, `ic_flipped` | panel, feature, split_date, method -> dict | **D** — `get_feature_drift` |
| feature_stability.py | `feature_stability` | IC per contiguous block, sign consistency, worst block | panel, feature, n_blocks, method -> dict | **D** — `get_feature_regime_stability` |
| feature_stability.py | `permutation_test_ic`, `_null_distribution` | within-date shuffle null, two-sided p, `null_p95_abs` (kernel-backed) | panel, feature, n_perm, method, seed -> dict | **D** — `run_feature_permutation_test` |

### 2.3 `modeling/dataset/`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| alignment.py | `build_returns_panel` | dates x entities close-to-close returns on the intersection | {e: close} -> frame | **I** — `build_dataset` (also `portfolio/portfolio.py`) |
| alignment.py | `attribute_drops` | `n_missing` / `n_sole_missing` per column, per-entity drops | panel, cols, target -> dict | **D** — `explain_dataset_row_loss` |
| alignment.py | `stack_long`, `stack_features_only`, `_stack`, `_record_kept_rows` | long panel with target(s)/label ends; scoring variant without target; keep-policy accounting | -> (panel, attribution) | **I** — `build_dataset` |
| builder.py | `build_dataset` | fetch -> features (entity / universe / panel fast path / PIT join) -> target(s) -> alignment -> warnings -> hashes -> `temporal_bundle` verdict | DatasetSpec, include_target -> dict | **D** — `build_model_dataset`, `score_model` |
| builder.py | `dataset_spec_hash`, `SPEC_HASH_VERSION` | canonical spec hash, v1 (all fields) / v2 (non-defaults) | spec, version -> hex | **D** — `build_model_dataset`, `register_external_panel`, `run_model_experiment` (verify) |
| builder.py | `_check_required_columns`, `_fetch_ohlcv`, `_provider_contract`, `_provider_metadata`, `_check_entity_output`, `_check_universe_output` | contract enforcement and provenance probes | | **I** |
| builder.py | result key `temporal_bundle` (the `DataBundle` verdict) | what the frames can and cannot support as one verdict | | **N** — computed by `build_dataset`, then dropped by `build_model_dataset` (not persisted in `dataset_meta`, not returned) |
| coverage.py | `provider_guarantee_warnings`, `interval_warnings`, `entity_coverage_warnings`, `alignment_warnings`, `missing_policy_warnings`, `intersection_warnings` | the six warning families | -> list[str] | **I** — `build_model_dataset.warnings`, carried onto the manifest as `dataset_warnings` |
| external_panel.py | `load_external_panel` | read a panel by reference, canonicalize columns, multi-target / event columns, MAX_ENTITIES, null warnings | path, columns, targets, fmt -> dict | **D** — `register_external_panel`; `_load_external_panel_for` |
| external_panel.py | `target_column_for`, `event_column_for`, `label_end_column_for`, `_resolve_columns` | naming and rename map | | **D** — `_select_target` in tools.py |
| fetch.py | `fetch_universe_ohlcv` | bounded-concurrency universe fetch, all failures reported, running-loop fallback | provider, symbols, start, end, interval -> {s: frame} | **I** — `build_dataset`, `evaluate_model_portfolio` |
| fetch.py | `_fetch_all_async`, `_fetch_all_sequential`, `_fetch_one`, `_validate_frame`, `_describe_failure`, `_max_concurrency`, `_in_running_loop` | helpers (`SQT_MODELING_FETCH_CONCURRENCY`) | | **I** |
| lags.py | `validate_lags` | refuse negative / zero / >60 / >20 lags, sorted+deduped | list -> list | **S** — `FeatureSpec.lags` validator |
| lags.py | `expand_lags`, `expanded_feature_ids`, `lags_by_output_name`, `lag_column_name` | per-entity shift before stacking; column order; 400-column ceiling | | **S** — `FeatureSpec.lags` |
| lags.py | `parse_lag_column` | `<name>__lag<k>` -> (name, k) | str -> tuple\|None | **X** — no production caller (module comment promises `analyze_model_errors` and the importance summary use it; they do not) |
| lags.py | `deepest_lag` | extra warm-up the deepest lag costs | specs -> int | **X** — no production caller |
| leakage.py | `PointInTimeViolation`, `check_point_in_time_safety` | refuse CURRENT_ONLY features | defs -> raise | **D** — `check_leakage`; `build_dataset` |
| missing.py | `forward_fill_bounded` | per-entity bounded ffill on an allowlist, counts filled | frame, features, max -> (frame, counts) | **S** — `MissingDataSpec.policy="forward_fill_bounded"` |
| panel_features.py | `compute_panel_features`, `_batch`, `_extract`, `_indices_identical` | native panel fast path for rsi/adx/stoch_k/atr_pct/bollinger_pct_b when indices are identical | specs, defs, params, ohlcv -> {name: {sym: Series}} | **I** — `build_dataset` |
| pit_features.py | `point_in_time_requests` | the POINT_IN_TIME features requested; refuses lags on them | -> list | **S** — `FeatureSpec.id="fundamental.*"` |
| pit_features.py | `gate_point_in_time` | provider temporal contract per frame kind BEFORE any fetch; `require_pit` | provider, requests, getter -> {kind: contract} | **S** — same (only PolygonProvider serves `fundamentals`) |
| pit_features.py | `join_point_in_time_features`, `_fetch_records`, `_transform` | fetch records once per kind, transform per feature, `asof_join` with staleness, attribution + `observed_revisions` warning | -> (panel, attribution, warnings, frames) | **S** — same |
| point_in_time.py | `validate_pit_frame` | schema + dtype + `available_time >= event_time` | frame -> frame | **D** — `validate_pit_records`, `join_point_in_time`; `data.validate_external_dataset` |
| point_in_time.py | `asof_join` | backward merge_asof on `available_time`, by entity or global, prefix, `max_staleness` | panel, records, fields, ... -> panel | **D** — `join_point_in_time` |
| point_in_time.py | `observed_revisions` | n_facts / n_restated / max_versions in a record set | records, by_entity -> dict | **N** for caller-supplied records — reached only inside `build_dataset`'s fundamental-feature path; `validate_pit_records` re-implements a weaker version inline (True/False) |
| point_in_time.py | `coverage_report` | never-available / not-yet-available warnings per field | joined, fields -> list[str] | **D** — `join_point_in_time` |
| target.py | `build_target`, `build_label_end_dates`, `apply_cross_sectional_target`, `_as_ohlcv` | registry dispatch; refuses external labels and missing `requires` | ohlcv, spec, ctx -> Series / panel | **S** — `TargetSpec.type` |

### 2.4 `modeling/estimators/`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| registry.py | `register_estimator` | allowlist extension point (no silent overwrite; optional `QuantileSupport`) | task, name, cls, schema -> None | **N** — Python-only extension point; deliberately not a tool (would violate the no-exec contract) |
| registry.py | `get_estimator_class` | lookup with "allowed:" refusal | task, name -> cls | **I** |
| registry.py | `validate_params`, `validate_param_value`, `allowed_params` | names AND values against bounds; per-axis check for grids | -> None / list | **D** — `validate_model_spec` |
| registry.py | `quantile_support`, `quantile_estimators`, `QuantileSupport` | which estimators can fit a quantile and how | -> support / list | **D** — `validate_model_spec`, `list_modeling_capabilities.quantile_param` |
| bounds.py | `ParamBound`, `EstimatorParamSchema`, `_logistic_compatibility`, shared bounds | typed, bounded params; solver/penalty matrix | | **I** — reached by `validate_model_spec` and every registration |
| regression.py | `linear`, `ridge`, `lasso`, `elastic_net`, `huber` registrations | sklearn regressors with bounded params | | **S** — `EstimatorSpec.type` |
| classification.py | `logistic` registration | with solver/penalty compatibility | | **S** — `EstimatorSpec.type` |
| trees.py | `hist_gradient_boosting` (x2), `random_forest` (x2), `gradient_boosting` (x2) | | | **S** — `EstimatorSpec.type` |
| boosting.py | `_register_lightgbm`, `_register_xgboost`, `_register_rankers`, `_register_quantile` | guarded registrations: `lightgbm`, `xgboost`, `lightgbm_ranker`, `xgboost_ranker`, `quantile`, `quantile_gradient_boosting` | | **S** — `EstimatorSpec.type` (optional ones only when installed) |
| boosting.py | `QuantileGradientBoostingRegressor` | GBR with quantile loss pinned | | **S** — `type="quantile_gradient_boosting"` |
| boosting.py | `OPTIONAL_ESTIMATORS`, `HAS_LIGHTGBM`, `HAS_XGBOOST` | static table for the doc generator; install flags | | **D** — flags via `list_modeling_capabilities.optional_dependencies`; table **I** (docs generator) |
| neural.py | `PanelMLPRegressor`, `PanelMLPClassifier`, `_sizes` | MLP with bounded width/depth scalars | | **S** — `type="mlp"` |
| online.py | `SGDRegressor` registration, `ProbabilisticSGDClassifier`, `_elasticnet_needs_a_ratio` | SGD learners; classifier defaults to a probabilistic loss | | **S** — `type="sgd"` |
| survival.py | `CoxPHRegressor` | numpy Cox PH (Breslow ties, L2, Newton) | fit(X, (n,2)) ; predict -> log hazard | **S** — `type="cox_ph"` |
| survival.py | `XGBCoxSurvival`, `XGBAFTSurvival`, `_XGBSurvivalBase`, `_split_labels`, `_register_xgboost` | XGBoost cox / AFT objectives wrapped with named params | | **S** — `type="xgboost_cox"` / `"xgboost_aft"` |

### 2.5 `modeling/features/`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| base.py | `TemporalSupport`, `FeatureScope`, `FeatureContext`, `FeatureDefinition`, `RESERVED_PANEL_COLUMNS` | the feature contract (entity / universe / point_in_time scopes; PIT fields) | | **D** — fields surfaced by `list_features` (temporal_support, scope, requires, lookback) |
| base.py | `periods_per_year_for_interval` | 252/52/12/4 for calendar intervals; calendar-derived for intraday | interval, calendar -> int\|None | **I** — `risk.*`, `evaluate_model_portfolio` |
| registry.py | `register_feature` | extension point (reserved-name refusal, no silent overwrite) | definition -> None | **N** — Python-only, deliberately |
| registry.py | `get_feature` | lookup with refusal | id -> def | **D** — `check_leakage`, `score_model` |
| registry.py | `list_features` | sorted catalog, category filter | category -> list | **D** — `list_features`; MCP `sqt://catalog/features` |
| params.py | `resolve_params` | merge onto defaults; type checks; window params >= 1 and <= 100k (closes the negative-lookback leak) | def, requested -> dict | **I** — `build_dataset`, `feature_provenance_from_spec` |
| params.py | `resolved_lookback` | lookback given resolved params | def, resolved -> int | **X** — no production caller (docstring says scoring / warm-up budgeting need it; `score_model` uses a flat `lookback_days=400` instead) |
| custom.py | re-exports | documented entry point for custom features | | **N** — Python-only |
| factors.py | `_pca_loading` (`factors.pca_loading`) | PC1 loading, rolling refit, power iteration | returns panel -> panel | **S** — `FeatureSpec.id` |
| factors.py | `_pca_factor_return` (`factors.pca_factor_return`) | realized return projected on held PC1 | | **S** — `FeatureSpec.id` |
| fundamental.py | `diluted_eps` (`fundamental.diluted_eps`) | EPS as filed, every version | records -> record-schema frame | **S** — `FeatureSpec.id` (needs a PIT provider contract; Polygon only) |
| fundamental.py | `net_margin` (`fundamental.net_margin`) | net income / revenue within a filing | | **S** — same |
| fundamental.py | `revenue_growth_yoy`, `_paired_versions` | YoY growth with a version at every change point of either filing | | **S** — same |
| market.py | `_market_momentum` (`market.momentum`) | trailing return over `lookback` | | **S** — `FeatureSpec.id` |
| market.py | `_market_new_high_breakout` (`market.new_high_breakout`) | close > prior `period` high, NaN warm-up | | **S** |
| market.py | `_market_psar_trend` (`market.psar_trend`) | parabolic SAR +/-1 | | **S** |
| network.py | `_avg_correlation` (`network.avg_correlation`) | mean correlation to the universe, rolling refit | | **S** |
| network.py | `_mst_degree` (`network.mst_degree`), `_prim_degrees` | MST degree on Mantegna distance (dense Prim) | | **S** |
| network.py | `_pairwise_correlation`, `_correlation`, `_rolling_network`, `_validate` | masked-corr matrix products; refit loop | | **I** |
| risk.py | `_risk_realized_volatility` (`risk.realized_volatility`) | Yang-Zhang, annualized by interval/calendar | | **S** |
| risk.py | `_risk_rolling_beta` (`risk.rolling_beta`) | OLS beta vs `DatasetSpec.benchmark` | | **S** |
| risk.py | `_risk_atr_pct` (`risk.atr_pct`), `atr_pct_from_atr` | ATR / Close, guarded | | **S**; helper **I** (panel fast path) |
| risk.py | `_risk_bollinger_pct_b` (`risk.bollinger_pct_b`), `pct_b_from_bands` | %B with collapsed-band rule | | **S**; helper **I** |
| risk.py | `_risk_parkinson_volatility`, `_risk_garman_klass_volatility` | annualized range estimators | | **S** |
| risk.py | `_risk_rolling_drawdown` (`risk.rolling_drawdown`) | drawdown from trailing peak | | **S** |
| risk.py | `_annualization` | bars/year or refuse without a calendar | ctx -> int | **I** |
| statistical.py | `_statistical_hurst` (`statistical.hurst`) | rolling Hurst (dfa) | | **S** |
| technical.py | `technical.rsi`, `.adx`, `.macd_histogram`, `.stochastic_k`, `.williams_r` (5 fns) | indicator wrappers | | **S** |
| transforms.py | `fit_preprocessing`, `apply_preprocessing`, `fit_and_apply_preprocessing`, `_fit_and_apply_with_stats` | winsorize(1/99)+zscore, native fused path | | **S** — default `PreprocessingSpec` (`normalization="pooled"`) |
| transforms.py | `standardize_cross_sectional` | per-date standardize + sigma clip, NaN-preserving | frame, dates, clip -> frame | **S** — `normalization="cross_sectional"` / step `cross_sectional_standardize` |
| transforms.py | `rank_within_date`, `cross_sectional_counts` | kernel-backed per-date average ranks and counts | | **I** — ensemble `rank_mean`, `forward_return_rank` target, rank turnover |
| transforms.py | `_native_matrix` | C-contiguous float64 or None | | **I** |
| volume.py | `volume.mfi`, `volume.obv_roc`, `volume.vwap_deviation` (3 fns) | volume features, guarded denominators | | **S** |

### 2.6 `modeling/preprocessing/`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| base.py | `FoldContext`, `Preprocessor`, `PreprocessorDefinition` | step contract: fit(X, ctx) -> state; transform(X, state, ctx); `stateless`, `column_wise` flags | | **I** |
| registry.py | `register_preprocessor` | extension point (id must match class) | definition -> None | **N** — Python-only, deliberately |
| registry.py | `get_preprocessor` | lookup with refusal | id -> def | **I** |
| registry.py | `validate_step_params` | names AND values against the step's bounds | id, params -> None | **S** — `StepSpec` validator |
| registry.py | `list_preprocessors` | sorted catalog | -> list | **D** — `list_modeling_capabilities.preprocessing.steps` |
| pipeline.py | `build_step`, `fit_pipeline`, `apply_pipeline`, `fit_and_apply_pipeline`, `step_types`, `STATE_VERSION` | fit on train / apply anywhere; state as JSON; fused native path for the default pair | | **I** — engine (folds, search, refit) and `score_model` |
| pipeline.py | `legacy_stats`, `_fused_state`, `_fused_stats`, `_is_default_pooled`, `_normalize`, `_resolved_params` | `preprocessing_stats.json` projection; helpers | | **I** |
| steps.py | `Winsorize`, `ZScore` | training-fold quantile clip; mean/std | | **S** — `PreprocessingSpec.steps[].type` (or default) |
| steps.py | `CrossSectionalStandardize` | stateless per-date standardize | | **S** |
| steps.py | `RobustScale` | median / MAD (x1.4826) | | **S** — `type="robust_scale"` |
| steps.py | `QuantileTransform` | rank-gauss / uniform via a training quantile grid | | **S** — `type="quantile_transform"` |
| steps.py | `MissingIndicator` | `<col>__missing` for every column | | **S** — `type="missing_indicator"` |
| steps.py | `Impute` | median / mean / constant from training rows | | **S** — `type="impute"` |
| steps.py | `PCAWhiten` | leading components, sign-fixed, whitened; not column-wise; refuses NaN | | **S** — `type="pca_whiten"` |

### 2.7 `modeling/registry/`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| environment.py | `environment_fingerprint`, `_version`, `_blas` | python / packages / BLAS / native extension / thread caps, no host identity | () -> dict | **I** — `save_model` -> `manifest.environment` -> `inspect_model(view="lineage")` |
| feature_provenance.py | `feature_implementation_hash`, `feature_implementation_hashes` | SHA-256 of each feature function's own source (or `"unavailable"`) | id(s) -> hash(es) | **I** — `save_model` |
| feature_provenance.py | `feature_provenance_from_spec` | output column -> {feature_id, resolved params, implementation_hash} | spec features -> dict | **I** — `save_model`; `score_model` drift refusal. **N** as a question: no view shows `manifest.feature_provenance`, and nothing lets an agent ask "has any feature implementation changed since this model was trained" before `score_model` refuses |
| lifecycle.py | `Promotion`, `promotions`, `current_stage`, `promote`, `STAGES` | append-only `promotions.jsonl`; one stage forward at a time, any earlier live stage backward, archived terminal | model_id, to_stage, reason, actor, evidence -> Promotion | **D** — `promote_model`, `inspect_model(summary)`, `list_models` |
| manifests.py | `ModelManifest`, `_nulls_to_nan` | the manifest schema (hashes, provenance, cutoff, warnings, preprocessing, distribution, monitoring, environment) | | **I** — `inspect_model` shows a subset of fields per view |
| model_registry.py | `new_model_id`, `save_model` | write every artifact, hash each, manifest last, auto-sign when configured | ... -> ModelManifest | **I** — `run_experiment` |
| model_registry.py | `load_manifest` | parse manifest; `require_signature=True` verifies `manifest.sig` before parsing | model_id, require_signature -> manifest | **D** — most tools; `require_signature=True` mode **N** (no tool passes it) |
| model_registry.py | `load_model`, `load_model_spec`, `load_preprocessing_stats`, `load_preprocessing_state`, `load_distribution`, `load_dataset_spec` | verified loaders (digest checked before joblib.load) | model_id -> object | **I** — `score_model`, `evaluate_model_portfolio` |
| model_registry.py | `load_monitoring_reference` | profile + feature/prediction reference frames, verified | model_id -> (profile, frame, frame) | **D** — `monitor_model` |
| model_registry.py | `_expected_hash` | digest for one artifact; manifest failure propagates | | **I** |
| package.py | `PackageVerification`, `verify_model_package` | hash every covered file, list unhashed files, verify signature (optionally required / pinned key) | model_id, require_signature, public_key -> report | **D** — `inspect_model(view="lineage").package` **with defaults only**; `require_signature` / `public_key` modes **N** |
| package.py | `mirror_model_package` | copy a verified package to any `ArtifactStore`, re-hash through the target, manifest last | model_id, store, prefix -> {file: uri} | **N** — no tool, no CLI |
| signing.py | `signing_available` | `cryptography` importable | () -> bool | **N** — only approximated by `list_modeling_capabilities.optional_dependencies.cryptography` |
| signing.py | `signing_configured` | `SQT_MODEL_SIGNING_KEY_PATH` set | () -> bool | **I** — `save_model` auto-sign |
| signing.py | `sign_manifest` | Ed25519 over `manifest.json` bytes; key file, env var, or signer callback + public key | model_id, key_path, signer, public_key -> record | **N** — only runs automatically at registration when the env var is set; a model registered unsigned cannot be signed later through any tool or CLI (`sqt keygen` writes an *audit* keypair) |
| signing.py | `verify_manifest_signature` | verify against embedded key, or refuse unless it equals the pinned key (`SQT_MODEL_VERIFY_KEY_PATH`) | model_id, public_key -> record | **I (partial)** — via `verify_model_package` when a `.sig` exists; pinned-key verification **N** |
| signing.py | `_public_key_bytes`, `_manifest_bytes`, `_require` | key parsing, bytes, dependency refusal | | **I** |

### 2.8 `modeling/targets/`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| base.py | `TargetKind`, `TargetDefinition` | the label contract: tasks, buildable, continuous, censored, requires, builder, label_end_builder, cross_sectional_stage, param schema | | **I** |
| registry.py | `register_target` | extension point with refusals (censored <-> survival, external has no builder, description length) | definition -> None | **N** — Python-only, deliberately |
| registry.py | `get_target`, `validate_target_params` | lookup; `TargetSpec.params` against the label's bounds | | **S** — `TargetSpec.type` / `.params` validators |
| registry.py | `targets_for_task` | labels a task may fit | task -> tuple | **D** — `validate_model_spec` target check; engine |
| registry.py | `list_targets` | sorted definitions | -> list | **I** — docs generator only (capabilities reads `TARGET_KINDS`) |
| registry.py | `TARGET_KINDS`, `EXTERNAL_TARGETS`, `CROSS_SECTIONAL_TARGETS` (live views) | what / external-only / cross-sectional | | **D** — `list_modeling_capabilities.targets` |
| builtin.py | `forward_return`, `forward_direction`, `forward_return_vol_scaled`, `forward_return_rank`, `forward_return_market_neutral`, `triple_barrier` builders; `_stage_rank`, `_stage_market_neutral`; `horizon_label_end` | the six buildable labels and their label-end rule | ohlcv, spec, ctx -> Series | **S** — `TargetSpec.type`, `.horizon(s)`, `.threshold`, `.vol_window`, `.barrier` |
| builtin.py | 12 external labels (`future_mid_return` ... `time_to_fill`, `adverse_selection`) | recorded-only labels, `time_to_fill` censored | registrations | **S** — `register_external_panel(targets[].target_type)` |
| builtin.py | `_forward_return`, `_horizon_volatility`, `_triple_barrier` | arithmetic (re-exported by `dataset/target.py` for tests) | | **I** |

### 2.9 `modeling/validation/`

| module | function/class | purpose | inputs -> outputs | exposed by |
|---|---|---|---|---|
| comparison.py | `paired_comparison` | join two prediction frames on (date, entity), same truth required; per-date IC difference, block-bootstrap CI, DM where a loss exists | frame_a, frame_b, task, metric, horizon, ... -> dict | **D** — `compare_models(method="paired")` (registered models only) |
| comparison.py | `holm_adjust` | step-down family-wise adjustment | p-values -> list | **D** — `compare_models(paired)` |
| comparison.py | `compare_ic_series` | difference statistics of two per-date IC series | ic_a, ic_b -> dict | **I** — inside `paired_comparison`; **N** for two arbitrary series (an ensemble vs a base model, a model vs a single feature, two external prediction refs) |
| comparison.py | `diebold_mariano`, `newey_west_variance` | DM with HLN correction on a NW long-run variance | losses, lag -> dict | **I** — inside `paired_comparison` |
| comparison.py | `_check_frame`, `_per_date_loss`, `_moving_block_means`, `_verdict` | helpers | | **I** |
| conformal.py | `held_out_residuals`, `conformal_radius`, `MIN_CALIBRATION_ROWS` | split-conformal on embargoed, purged date blocks; finite-sample quantile | fit_predict, dates, label_end, n_folds, embargo -> residuals; -> radius | **S** — `ModelSpec.intervals` |
| diagnostics.py | `FoldImportance`, `fold_feature_importance`, `summarize_importance`, `_sign_consistency` | coef_/feature_importances_ per fold; magnitude + signed mean/std + sign consistency | estimator, cols -> FoldImportance; list -> summary | **I** — `run_experiment` -> `manifest.feature_importance_summary` -> `inspect_model(view="feature_importance")` |
| distributional.py | `quantile_column`, `pinball_loss`, `distributional_metrics` | q-column naming; pinball per level; crossing rate; coverage/width per symmetric pair and per conformal interval | y, {q: preds}, lower, upper, alpha -> dict | **S** — `ModelSpec.quantiles` / `.intervals` (in `oos_metrics`); **N** for external predictions (`score_predictions` ignores `q05`/`lower`/`upper` columns) |
| metrics.py | `cross_sectional_ic` | per-date IC, balanced/ragged vectorized paths, native kernel | y, pred, dates, method -> Series | **D** — `score_predictions`; everywhere |
| metrics.py | `summarize_cross_sectional_ic` | mean/std/ICIR/hit rate/n_dates | series, prefix -> dict | **D** — `score_predictions` |
| metrics.py | `effective_sample_size` | n / horizon per entity | n, horizon, entities -> float | **D** — `score_predictions` (but with `horizon=1`, see 5.3); engine |
| metrics.py | `baseline_regression_metrics` | predict-the-mean baseline, training mean when given (`baseline_is_oracle`) | y, train_y -> dict | **D** — `score_predictions` (always oracle: no `train_y` available) |
| metrics.py | `regression_metrics`, `classification_metrics` | R2/MAE/pooled IC + CS blocks; accuracy/AUC/positive_rate/majority_class_accuracy | | **D** — `score_predictions`; adapters |
| metrics.py | `positive_class_proba`, `aggregate_cross_sectional_ic`, `average_fold_metrics`, `check_ic_method`, `IC_METHODS` | class-index-safe proba; pooled-series dispersion; weighted fold mean; refuse unknown IC method | | **I** |
| ranking.py | `ranking_metrics` | CS IC/rank IC + NDCG@k on graded target | y, scores, dates, n_grades, ks -> dict | **D** — `score_predictions(task="ranking")`; `RankingAdapter` |
| ranking.py | `relevance_grades`, `group_sizes`, `ndcg_at_k`, `fold_ic_series` | within-date integer grades; consecutive query counts with ordering check; NDCG; per-fold IC | | **I** (`RankingSpec.n_grades`, `.ndcg_at` via **S**) |
| search.py | `search_best_params` | inner walk-forward with the outer embargo and label purge; grid / random / tpe; per-candidate report | ... -> (params, report) | **S** — `ModelSpec.search` (report in `validation_report.hyperparameter_search`) |
| search.py | `n_search_candidates`, `optuna_available` | candidate count without enumerating; optuna probe | | **D** — `validate_model_spec`, `list_modeling_capabilities` |
| search.py | `search_candidates`, `inner_fold_count`, `require_optuna`, `_tpe_trials`, `_score`, `_inner_splitter`, `_labels_for` | deterministic enumeration; inner-fold count for the plan; TPE with median pruning; higher-is-better scoring | | **I** |
| splits.py | `holdout_split` | time-ordered train/test positions | dates, frac -> (idx, idx) | **X** — no caller anywhere; docstring stale ("ValidationSpec.method is currently a Literal['walk_forward'] only") |
| survival.py | `survival_labels`, `concordance_index`, `survival_metrics`, `EVENT_COL` | (duration, event) label; Harrell's C pooled and per date; event rate | | **S** — `task="survival"`; **N** for external predictions (`score_predictions` accepts `task="survival"` in its schema but routes it to `ranking_metrics`, see 5.3) |
| walk_forward.py | `WalkForwardSplit` | rolling / expanding windows with embargo | dates -> (train, test) positions | **S** — `ValidationSpec.method="walk_forward"`, `.scheme` |
| walk_forward.py | `PurgedKFoldSplit` | K contiguous blocks, embargo both sides | | **S** — `method="purged_kfold"` |
| walk_forward.py | `CombinatorialPurgedSplit` | C(n, k) paths, `n_paths` | | **S** — `method="cpcv"`, `.n_test_splits` (<= 60 paths) |
| walk_forward.py | `build_splitter` | spec -> splitter | | **D** — `validate_model_spec` (exact fold count), `run_feature_ablation` |
| walk_forward.py | `contiguous_runs`, `label_overlap_mask` | test blocks of a combinatorial set; the one purge rule (outer, inner, conformal) | | **I** |
| weights.py | `build_sample_weights` | none / label_uniqueness / time_decay / both, mean-1 normalized | method, dates, label_end, entities, half_life -> weights | **S** — `WeightingSpec` |
| weights.py | `label_uniqueness_weights`, `time_decay_weights`, `_as_int64_ns` | Lopez de Prado average uniqueness per entity (kernel-backed); exponential decay in calendar days | | **I**; **N** as a diagnostic (nothing reports the uniqueness distribution or what a half-life implies before fitting) |

---

## 3. Unexposed capabilities, grouped by theme

**A. The experiment plan as a document (plan.py).** `plan_experiment(...).to_dict()` carries, per fold: date spans, scheduled vs. surviving training dates, purged-row counts, rows per side, inner-fold count, fits, and two content hashes. Tools surface only `n_fits` (`validate_model_spec`) and, after the fact, `validation_report.folds` (`inspect_model`). Nothing answers "show me the schedule this spec implies on this dataset before I pay for it".

**B. Package trust: sign / require / pin / mirror (registry/signing.py, package.py, model_registry.py).** `sign_manifest`, `verify_manifest_signature(public_key=...)`, `verify_model_package(require_signature=True, public_key=...)`, `load_manifest(require_signature=True)` and `mirror_model_package` are all unreachable. `inspect_model(view="lineage")` verifies with defaults, so an unsigned package reads `ok=True`, and `promote_model` does not gate on verification. The module docstrings describe these as the trust-boundary steps, and no tool stands at that boundary.

**C. Consuming predictions that are not a registered model's.** `build_model_ensemble` publishes a `predictions` ref, `score_predictions` accepts one, but: (i) the ensemble frame carries no `target`, and `score_predictions` needs one in the frame (nothing joins outcomes from the dataset the way `_oos_with_actuals` does for registered models); (ii) `compare_models(paired)` only takes `model_ids`, so `paired_comparison`/`compare_ic_series`/`diebold_mariano` cannot answer "does the ensemble beat its best base model"; (iii) `evaluate_model_portfolio` only takes a `model_id`, so an ensemble or an external prediction set cannot be run through the shared-cash simulator; (iv) `transform_predictions_to_weights` (gross+net exact, per-position cap, rebalance schedule, uncertainty scaling) is unreachable except through a registered model, while `meta.convert_reference(score_panel->weight_panel)` uses `backtest.sizing` with none of that.

**D. Distribution- and survival-aware scoring of external predictions.** `distributional_metrics` (pinball, crossing, coverage/width) and `survival_metrics` (concordance) exist and run inside the engine, but `score_predictions` ignores `q*`/`lower`/`upper` columns and mis-routes `task="survival"` to ranking metrics.

**E. Dataset-spec pre-flight (the `validate_model_spec` counterpart for `DatasetSpec`).** `resolve_params`, `resolved_lookback` (dead), `deepest_lag` (dead), `expanded_feature_ids`, `check_point_in_time_safety`, `point_in_time_requests`, `gate_point_in_time`, `_provider_metadata` -> `provider_guarantee_warnings`, `interval_warnings`, `periods_per_year_for_interval`, `validate_calendar_name` can all run without fetching a bar. Today the only way to learn "this spec needs 500 bars of warm-up, expands to 140 columns, and its `fundamental.*` features will be refused by yfinance" is to run the build.

**F. Calendar as a question (calendar.py).** `calendar_names`, `bars_per_session`, `sessions_per_year`, `session_minutes`, `periods_per_year` — an agent choosing `DatasetSpec.calendar` for an intraday panel has no way to list valid codes or see what annualization a (interval, calendar) pair implies.

**G. Provenance as a question (registry/feature_provenance.py, manifests.py).** `manifest.feature_provenance`, `feature_implementation_hashes`, `content_hashes`, `distribution`, `monitoring`, `model_input_columns` (partly) and `feature_profile.json` are persisted and never shown. The implementation-drift check runs only as a refusal inside `score_model`.

**H. Point-in-time record evidence (dataset/point_in_time.py).** `observed_revisions` (n_facts / n_restated / max_versions) is unreachable for caller records; `validate_pit_records` reports only a boolean from an inline re-implementation. The `DataBundle` verdict (`temporal_bundle`) is computed by `build_dataset` and discarded by the tool.

**I. Weighting and overlap diagnostics (validation/weights.py).** `label_uniqueness_weights` / `time_decay_weights` run only inside a fit; nothing lets an agent see the uniqueness distribution, the effective row count per entity, or what `half_life_days` does to the training window before choosing `WeightingSpec.method`.

**J. Multi-horizon panels vs. the feature lab.** `TargetSpec.horizons` and `register_external_panel(targets=[...])` produce `target__h5`, `target__h30`, ... but every `feature_lab` tool and `analyze_features` read `target` only; the feature report docstring still says multi-horizon targets do not exist yet.

**K. The bridge's safe mode.** The only tool path to `oos_predictions_to_signal_panel` is `meta.convert_reference` in URI mode, which skips digest verification, the cpcv refusal and the skipped-fold refusal that the `model_id` mode was written to enforce.

**L. Extension points (deliberately unexposed).** `register_feature`, `register_target`, `register_estimator`, `register_preprocessor` are Python-only. Exposing them would let an LLM hand the engine an arbitrary callable, which is the exec() path the specs exist to prevent. Correctly not tools; listed for completeness.

---

## 4. Proposed tools

Ordered by expected value. "Runtime" is where it belongs given the existing split (modeling = the model lifecycle; feature_lab = one dataset, no model; meta = handoff references).

### 4.1 `plan_model_experiment` — or `validate_model_spec(include_plan=True)` (modeling) — **S**

- **Inputs:** `dataset_id`, `spec: ModelSpec`, `target: Optional[str]`, `max_folds_listed: int = 20`.
- **Outputs:** `ExperimentPlan.to_dict()` (method, n_dates, n_candidates, fits per fold and total, `within_budget`, `max_fits`, total purged rows, and per fold: train/test spans, scheduled vs surviving train dates, `n_train_rows` / `n_test_rows` / `n_purged`, `n_inner_folds`, `node_hash`, `preprocessing_hash`) plus warnings.
- **Backing:** `plan.plan_experiment(spec, dates, panel=panel, dataset_hash=..., feature_ids=...)` with the panel loaded through `_load_dataset_panel`; `search.inner_fold_count`; `walk_forward.build_splitter`.
- **Why a decision:** the fold schedule *is* the validation design. A purge that removes 40% of a fold's training rows, an inner search that silently does not run on short folds (`n_inner_folds=0`), or an expanding scheme whose first fold is one third the size of its last, all change what the OOS number means, and the agent chooses windows, embargo and scheme. Today those facts arrive only after the fits.
- **Caveats to state:** the plan is a function of dates and label ends, not of fitting; a fold can still be skipped at run time (single-class window, no observed event); hashes describe the fitted estimator's inputs, not a persisted cache.
- **Effort:** S (all functions exist; `validate_model_spec` already loads the metadata and could load the panel on request).

### 4.2 `validate_dataset_spec` (modeling) — **M**

- **Inputs:** `spec: DatasetSpec`; `provider_probe: bool = True`.
- **Outputs:** per feature: registry id, resolved params (`resolve_params`), `resolved_lookback`, scope, temporal support, `requires`; deepest warm-up in bars (`deepest_lag` + max resolved lookback + horizon); expanded column count vs `MAX_EXPANDED_COLUMNS`; PIT verdict (`check_point_in_time_safety`; `point_in_time_requests` and `gate_point_in_time` against the named provider's `get_temporal_contract`, which is a method call, not a fetch); provider guarantees (`_provider_metadata` -> `provider_guarantee_warnings`); `interval_warnings`; calendar resolution (`validate_calendar_name`, `periods_per_year_for_interval`) and whether any annualizing feature will refuse; target/task hints (`TARGET_KINDS`).
- **Backing:** listed above; all under `features/params.py`, `dataset/lags.py`, `dataset/leakage.py`, `dataset/pit_features.py`, `dataset/coverage.py`, `calendar.py`, `targets/registry.py`.
- **Why a decision:** the cost of a bad `DatasetSpec` is a universe fetch plus a refusal several minutes in; and the panel's start date, its column width, and whether fundamentals can be joined are choices the agent makes before building. This is the same argument `validate_model_spec` already won.
- **Caveats to state:** fetches nothing, so it cannot report coverage, survivorship, or alignment loss (those need the build); the PIT gate answers "does the provider declare a contract", not "does the data exist for these symbols"; the warm-up estimate is bars, and bars are not calendar days.
- **Effort:** M (one new tool; revives two dead helpers).

### 4.3 `compare_predictions` (modeling) — **M**

- **Inputs:** `reference_ref` and `candidate_refs` (each an `sqt://predictions` ref *or* a `model_id`), `task`, `outcomes`: either `dataset_id` (join `target` from the panel via the existing `_panel_with_selected_target` logic) or `target_column` present in the frames, `metric` (cs_rank_ic / cs_ic), `horizon`, `n_bootstrap`, `block_size`.
- **Outputs:** the same `PairedComparison` rows `compare_models(paired)` returns (mean difference, block-bootstrap CI, p, Holm p, hit rate, DM where a loss exists, verdict), plus coverage of the joined rows.
- **Backing:** `validation/comparison.py` (`paired_comparison`, `holm_adjust`, `diebold_mariano`), `ensemble.load_oos_predictions`, `handoff.resolve`.
- **Why a decision:** it closes the loop `build_model_ensemble` opens: the only question worth asking after combining models is whether the combination beats the best member on the same rows, and today that comparison is impossible because the ensemble has no `model_id`. It also lets a feature (its column as `prediction`) be compared paired against a model, which is the honest test of "did the model add anything over its best input".
- **Caveats to state:** every frame must carry the *same* realized outcome on the intersection (refused otherwise); intervals rest on the shared dates, "indistinguishable" under 60 dates means "not enough dates"; Holm controls these tests, not the selection of candidates on this sample; a classification comparison uses Brier and needs probabilities.
- **Effort:** M.

### 4.4 `build_target_weights` (meta or modeling) + `evaluate_model_portfolio(predictions_ref=...)` — **M**

- **Inputs:** `predictions_ref` (or `model_id`), `task`, `transform: PredictionTransformSpec`, optional `returns_ref` for `volatility_scale`; publishes a `weight_panel` ref and returns `transform_predictions_to_weights` diagnostics (names per date, realized gross/net, shortfall dates, empty dates, max weight).
- **Backing:** `portfolio_eval.predictions_to_score_panel`, `select_rebalance_dates`, `scale_by_uncertainty`, `transform_predictions_to_weights`, `apply_exposure_targets`.
- **Why a decision:** sizing is the decision that turns an ordering into P&L, and the library's only implementation of "gross and net exactly, cap redistributed, first-of-period rebalance, size by conformal width" is locked behind a registered `model_id`. An ensemble, a `score_predictions`-scored external model, or a hand-built score panel cannot reach it. Extending `evaluate_model_portfolio` to accept `predictions_ref` (with `interval`, `provider`, and an explicit universe/date range, since there is no manifest to read them from) gives those the shared-cash simulation too.
- **Caveats to state:** without a manifest there is no digest to verify, no cpcv refusal and no skipped-fold list, so continuity is checked from date gaps only; classification predictions are recentred at `proba - 0.5`; `uncertainty_scaled` needs `lower`/`upper`; `volatility_scale` makes the transform depend on price history; a `close` fill is look-ahead.
- **Effort:** M.

### 4.5 `attest_model_package` (modeling) — **S/M**

- **Inputs:** `model_id`, `action: Literal["verify", "sign", "mirror"]`, `require_signature: bool`, `public_key: Optional[str]` (hex or path), `mirror_uri: Optional[str]` (a directory or an fsspec URI; credentials from the environment, never from the input).
- **Outputs:** `PackageVerification.to_dict()` (verified / mismatched / missing / unhashed, signature record with `key_pinned`, `signature_error`); for `sign`, the signature record; for `mirror`, `{filename: uri}` on the target.
- **Backing:** `package.verify_model_package`, `signing.sign_manifest`, `signing.verify_manifest_signature`, `package.mirror_model_package`, `artifact_store.LocalArtifactStore` / `FsspecArtifactStore`, `signing.signing_available`.
- **Why a decision:** promoting to `staging`/`production` and copying a package somewhere else are the two moments the docstrings name as trust boundaries, and both are currently taken on an `ok=True` that never required a signature. `promote_model` should gain `require_verified_package: bool` (default False to keep behaviour) that calls this before writing the log line, and `evidence` could carry the manifest SHA the verification saw.
- **Caveats to state:** integrity is not authenticity: a valid signature under an unpinned key proves the manifest and signature were written together, not by anyone trusted; key custody is outside the library (`signer` callback for HSM/KMS is Python-only and stays so); mirroring refuses a package that does not verify locally; `unhashed` files (promotion log, scoring outputs) are not covered by the hashes.
- **Effort:** S for verify/sign, M for mirror (store selection + URI validation).

### 4.6 `score_predictions` extensions (modeling) — **S**

Not a new tool; three parameters and one fix on the existing one, because each is a question the tool already claims to answer:

- `horizon: Optional[int]` (or `dataset_id`/`model_id` to read it): today `effective_sample_size` is called with `horizon=1`, so the reported "effective sample size adjusted for overlapping forward returns" equals `n_observations`. The tool description promises an adjustment the code cannot make.
- `outcomes_dataset_id: Optional[str]`: join `target` (and `event`, `label_end_date`) from a dataset panel so an ensemble's `predictions` ref can be scored without the caller materializing a frame with a target column. Backing: the existing `_panel_with_selected_target` helper in `tools.py`.
- `task="survival"`: route to `survival_metrics` with an `event_column`; today the schema admits `"survival"` (it is `Task`) and the code's `else` branch scores it with `ranking_metrics`, i.e. NDCG over a duration.
- Auto-detect `q*` / `lower` / `upper` columns and add `distributional_metrics` (pinball, crossing rate, coverage, width), so a model trained with `quantiles`/`intervals` can be judged on its distribution out of the engine, and an external quantile forecast can be judged at all.
- Caveats: the baseline is always the oracle (scored-set mean) because no training mean exists here — already stated; classification binarizes the outcome at 0 and thresholds at 0.5, which is wrong for a `triple_barrier` (3-class) outcome and should be refused rather than computed.

### 4.7 `inspect_model(view="provenance")` (modeling) — **S**

- Adds a fifth view returning `feature_provenance` (column -> feature_id, resolved params, implementation hash), `feature_implementation_hashes`, `content_hashes`, `distribution`, `monitoring`, `model_input_columns`, `environment`, plus a computed `implementation_drift` block: `feature_provenance_from_spec(load_dataset_spec(model_id)["features"])` diffed against the recorded hashes, exactly the check `score_model` performs as a refusal.
- **Why a decision:** "retrain, or score with the old code" is decided before a scoring call, not learned from its error; and a reader deciding whether to trust a model months later needs the provenance map the manifest already holds.
- **Caveats:** the hash covers a feature function's own source, not shared primitives (`git_commit_sha`/`package_version` are the coarser signal); `"unavailable"` means identity was not captured, not that it changed.
- **Effort:** S.

### 4.8 `resolve_calendar` — or `list_modeling_capabilities(calendar=..., interval=...)` (modeling) — **S**

- **Inputs:** `interval`, `calendar: Optional[str]`, `list_names: bool`.
- **Outputs:** `calendar_names()` (or a filtered sample), and for a pair: `interval_minutes`, `session_minutes`, `bars_per_session`, `sessions_per_year`, `periods_per_year`, and which registered features would refuse to annualize without it.
- **Backing:** `calendar.py`, `features.base.periods_per_year_for_interval`.
- **Why a decision:** naming a venue on `DatasetSpec.calendar` changes every annualized number (Sharpe, volatility features, CAGR) by a fixed factor, and the agent cannot currently discover valid codes or the factor.
- **Caveats:** needs the optional `exchange_calendars` package; session length is a median over recent sessions (early closes are not modelled per bar); a daily-or-coarser interval needs no calendar.
- **Effort:** S.

### 4.9 `validate_pit_records` gains `observed_revisions` (modeling) — **S**

- Replace the inline `versions = frame.groupby(keys)[AVAILABLE_TIME].nunique()` with `observed_revisions(frame, by_entity=...)` and return `n_facts`, `n_restated`, `max_versions` (the numbers the Polygon contract's own docstring says it is waiting on) beside `median_publication_lag_days`. Also persist and return `build_dataset`'s `temporal_bundle` verdict from `build_model_dataset` (into `dataset_meta.json`, and via `check_leakage(dataset_id=...)`), which today is computed and thrown away.
- **Why a decision:** whether restatements arrive as rows decides whether an as-of join reproduces history or the final numbers; a single boolean hides how much of the set is affected.
- **Effort:** S.

### 4.10 `diagnose_label_overlap` (feature_lab) — **S/M**

- **Inputs:** `dataset_id`, `target: Optional[str]`, `half_life_days: Optional[float]`.
- **Outputs:** the horizon, `effective_sample_size` (and per-entity), the distribution of `label_uniqueness_weights` (quantiles, share of rows below 0.5), the share of training weight the last N calendar days would carry under `time_decay_weights` for the given half-life, and the fraction of rows the label-overlap purge would remove per fold for a given `ValidationSpec` (via `label_overlap_mask` / the plan).
- **Backing:** `validation/weights.py`, `validation/metrics.effective_sample_size`, `plan.plan_experiment` (purge counts).
- **Why a decision:** `WeightingSpec.method` and `half_life_days` are modelling choices the spec asks the agent to make blind; the numbers that inform them are one function call each and currently only exist inside a fit.
- **Caveats:** needs `label_end_date` (external panels without it get no uniqueness); weights are normalized to mean 1 so a ratio, not a count; a half-life in days on an hourly panel is still days by design.
- **Effort:** S/M.

### 4.11 `preview_preprocessing` (feature_lab) — **M**

- **Inputs:** `dataset_id`, `preprocessing: PreprocessingSpec`, `fit_on: Literal["first_fold", "full_panel"]` with an optional `ValidationSpec` for the fold.
- **Outputs:** the fitted state (`fit_pipeline` JSON: winsor bounds, means/stds, MAD scales, PCA `explained_variance_ratio`, quantile knot counts), output column set (`model_input_columns` equivalent), `column_wise` / `stateless` per step (so the agent knows ablation can project), and post-transform distribution stats via `feature_distribution_stats`.
- **Backing:** `preprocessing/pipeline.py`, `preprocessing/registry.py`, `cache.column_wise_pipeline`, `analysis.feature_report.feature_distribution_stats`.
- **Why a decision:** choosing between `zscore`, `robust_scale`, `quantile_transform` and `pca_whiten(n_components)` is currently a guess confirmed only by a full experiment; the PCA variance ratio in particular decides `n_components`.
- **Caveats:** a state fitted on the full panel describes no fold and must not be persisted or applied to test rows; this is a diagnostic, never the deployed transform.
- **Effort:** M.

### 4.12 `feature_lab` tools gain `target: Optional[str]` — **S**

- Every feature_lab tool and `analyze_features` should accept the declared label name of a multi-horizon panel and select it through the existing `_select_target`, so "at what horizon is this feature predictive" becomes a loop over `target` rather than a rebuild per horizon. Backing exists (`_select_target` in tools.py). Caveat: rows are dropped per selected label, so counts differ between horizons and the numbers are not on identical rows unless the intersection is taken.

### 4.13 `meta.convert_reference` gains `model_id` — **S**

- Let the `predictions -> signal_panel` conversion take a `model_id` so the bridge runs in its verified mode (digest check, cpcv refusal, skipped-fold refusal, task read from the manifest). The repo's decision that the bridge is not a *modeling* tool stands; this only stops the one tool that reaches it from bypassing its guards.

### Where a parameter or view beats a new tool (summary)

| existing tool | addition | replaces proposal |
|---|---|---|
| `validate_model_spec` | `include_plan: bool` -> `plan.to_dict()` | 4.1 |
| `inspect_model` | `view="provenance"` (+ `implementation_drift`) | 4.7 |
| `inspect_model(view="lineage")` | `require_signature`, `public_key` passed through to `verify_model_package` | half of 4.5 |
| `promote_model` | `require_verified_package: bool` | half of 4.5 |
| `score_predictions` | `horizon`, `outcomes_dataset_id`, survival routing, distribution columns | 4.6 |
| `evaluate_model_portfolio` | `predictions_ref` (+ `interval`, `provider`, `universe`, `start`, `end`) | half of 4.4 |
| `compare_models` | `predictions_refs` beside `model_ids` (needs an outcomes source) | most of 4.3 |
| `list_modeling_capabilities` | `calendar`, `interval` -> resolved annualization; `calendar_names` | 4.8 |
| `validate_pit_records` | `observed_revisions` numbers | 4.9 |
| `build_model_dataset` / `check_leakage` | persist and return `temporal_bundle` | 4.9 |
| all `feature_lab` tools, `analyze_features` | `target: Optional[str]` | 4.12 |
| `meta.convert_reference` | `model_id` | 4.13 |

---

## 5. Dead code, duplicated implementations, docstring drift

### 5.1 Dead or effectively dead

| where | item | finding |
|---|---|---|
| `artifacts.py` | `local_store()` | no caller in `src/` or `tests/`; `package.py` constructs `LocalArtifactStore()` directly |
| `validation/splits.py` | `holdout_split` | no caller anywhere; re-exported from `validation/__init__`; docstring says `ValidationSpec.method` "is currently a Literal['walk_forward'] only" (there are three methods). Either delete or wire as `method="holdout"` |
| `dataset/lags.py` | `parse_lag_column`, `deepest_lag` | test-only; the `LAG_SUFFIX` comment says the name "lets `analyze_model_errors` and the importance summary report 'this is rsi at lag 3'" — neither does |
| `features/params.py` | `resolved_lookback` | test-only; docstring: "Callers that need to size a history window (scoring, warm-up budgeting) need the resolved value" — `score_model` uses a flat `lookback_days=400` and never calls it (so a custom feature with `lookback=500` is silently under-fetched, as the `score_model` docstring half-admits) |
| `analysis/feature_ablation.py` | `DEFAULT_MAX_FITS = 200` | unused by the tool (which reads `FeatureAblationInput.max_fits`); same name as `limits.DEFAULT_MAX_FITS = 500` with a different meaning |
| `ensemble.load_oos_predictions(keep_path=True)` | cpcv path column | no caller uses `keep_path=True`; the per-path frame is never consumed |
| `targets/registry.list_targets` | | used only by the documentation generator; `capabilities` reads `TARGET_KINDS` instead |
| `registry/model_registry.load_manifest(require_signature=True)` | | never passed by any tool or by `verify_model_package` (which calls `verify_manifest_signature` directly) |

### 5.2 Duplicated implementations

| items | note |
|---|---|
| `validate_pit_records` (tools.py) inline version counting vs `point_in_time.observed_revisions` | same groupby; the tool version discards `n_facts`/`n_restated`/`max_versions` |
| `monitoring.realized_ic` per-date rank IC via `group["prediction"].rank().corr(...)` vs `validation.metrics.cross_sectional_ic` | a second, un-kernelled IC implementation with a different minimum (3 per date vs 2) and no NaN-pair rule; should call `cross_sectional_ic` + `summarize_cross_sectional_ic` |
| `monitoring.PSI_MODERATE/PSI_SEVERE/KS_FLAG` vs `feature_stability.PSI_MODERATE/PSI_SIGNIFICANT` | identical values under two names in two modules; `get_feature_drift` and `monitor_model` could disagree if one moves |
| `meta/convert.py::_predictions_to_score_panel` vs `portfolio_eval.predictions_to_score_panel` | two long-to-wide pivots with different classification recentring rules (`proba_threshold` vs a fixed 0.5) |
| `meta/convert.py::_score_panel_to_weight_panel` (via `backtest.sizing`) vs `portfolio_eval.transform_predictions_to_weights` | two score->weight paths; only the second enforces gross/net/cap and rebalance dates |
| `bridge._validate_predictions_frame` vs `ensemble.load_oos_predictions` column/dtype checks vs `comparison._check_frame` | three validators of the same (date, entity, prediction) shape |
| `estimators/boosting._register_xgboost` and `estimators/survival._register_xgboost` | same name, two modules, each probing `import xgboost` |
| `ensemble._pairwise_correlation` and `network._pairwise_correlation` | same name, unrelated semantics (corr dict of columns vs masked corr matrix of a panel) |
| `capabilities.modeling_capabilities()["weighting"]` | hand-written four-item list, while the module's own docstring says hand-maintained lists are the failure it exists to avoid; `WeightingSpec.method` is a Literal that `_literal_options` already reads for other fields |
| `validation/walk_forward.PurgedKFoldSplit` vs the backtest runtime's `build_purged_cv_splits` tool | likely two purged-CV cutters in the repo (the backtest one lives outside this slice); worth confirming they share `label_overlap_mask` semantics |
| `diagnostics._finite`, `feature_stability._finite`, `feature_report._safe` | trivial, but three spellings of "finite float or NaN" |

### 5.3 Docstrings / descriptions promising more than the code does

| where | claim | reality |
|---|---|---|
| `score_predictions` tool description and result field | "an effective sample size adjusted for overlapping forward returns" | `effective_sample_size(len(frame), horizon=1, entities)` == `n_observations`; no horizon input exists; the "Horizon inferred as 1 bar; pass a target horizon through the dataset spec" note is emitted only when `label_end_date` is present and there is no such input |
| `ScorePredictionsInput.task: Task` | admits `"survival"` | falls into the `else` branch and is scored with `ranking_metrics` (NDCG over a duration); `survival_metrics` is never called |
| `score_predictions` classification branch | scores any classification outcome | binarizes `y_true > 0` and thresholds at 0.5; a `triple_barrier` (0/1/2) outcome is silently mangled rather than refused (the engine and `diagnostics.calibration` both refuse non-0/1) |
| `dataset/lags.py` module comment on `LAG_SUFFIX` | analyze_model_errors / importance summary decode lag columns | `parse_lag_column` has no caller |
| `features/params.resolved_lookback` | scoring uses it | it does not |
| `validation/splits.py` | "ValidationSpec.method is currently a Literal['walk_forward'] only" | three methods; function unused |
| `engine.py` module docstring | "Preprocessing (features/transforms.py's winsorize + zscore)" | preprocessing is a registry pipeline of 8 steps |
| `adapters.py` / `features/base.py` docstrings | "only risk.rolling_beta ... uses context" | every annualizing `risk.*` feature reads `context.interval`/`.calendar`; PIT features read it too |
| `bridge.py` docstring | "modeling, 1 of 6 tools" | 22 tools |
| `analysis/feature_report.py` module docstring | "ONE HORIZON, FOR NOW ... needs multi-horizon targets in the dataset first" | `TargetSpec.horizons` and multi-target external panels exist; the report still reads only `target` and the tools cannot select another |
| `inspect_model(view="lineage")` description ("says whether the package is still the one that was registered") | | true for integrity; it cannot say whether it is the one anyone *signed* unless a `.sig` happens to exist, and it never requires one |
| `capabilities.modeling_capabilities` docstring ("Everything here is READ OFF the live registries") | | `weighting` is a literal list |
| `PromoteModelInput`/`promote_model` description ("candidate -> validated -> staging -> production one stage at a time, or archived") | | `promote()` also allows a *demotion* to any earlier live stage; the tool text does not say so |
| `registry/model_registry.py` layout docstring | lists `manifest.json`, `model.joblib`, `model_spec.json`, `preprocessing_stats.json`, `manifest.sig`, `promotions.jsonl` | the directory also holds `preprocessing_state.json`, `distribution.json`, `quantile_models.joblib`, `feature_profile.json`, `feature_reference.parquet`, `prediction_reference.parquet`, `dataset_spec.json`, `oos_predictions.parquet` |

### 5.4 Behavioural notes worth a caveat in any new tool

- `cross_sectional_ic` emits `0.0` for a date whose usable rows fell below 2 after the NaN drop (a documented, deliberately preserved rule); every IC mean is dragged toward zero by such dates. Any tool that reports IC on a panel with holes (external panels, `missing.policy="keep"`) should say so.
- `select_features` and `compare_feature_sets` score on the same panel they select from (no holdout); the docstring is honest about avoiding greedy search but the IC floor is still in-sample.
- `combine_predictions` labels a regression+ranking mix as `task="ranking"`; the published ref then feeds `score_predictions` with a task the caller must supply consistently.
- `build_model_dataset` drops `temporal_bundle`; `register_external_panel` records `drop_attribution={}` and no bundle, so a `check_leakage(dataset_id=...)` on either can only echo what was recorded, never a contract verdict.
