# The modeling runtime: from research engine to model-governance system

A plan for generalizing the modeling runtime's preprocessing, target,
validation, representation and lifecycle primitives, in the order that the
code's own constraints impose rather than the order the ideas arrived in.

**Status: phases 0-4 are implemented and merged; phases 5-10 are
proposals.** Phase 4 landed as three commits on 2026-09-20: `56727b8`
(`plan_experiment`, `ModelSpec.budget`, the fit count in
`validate_model_spec`), `85128dd` (the fold cache, with the measurement
in the guide: a 20-feature ablation on 40k rows from 4.4 s to 3.3 s under
the fused default pipeline and from 43.8 s to 8.4 s with a quantile
transform in it, identical numbers) and the commit that carries this
paragraph (`SearchSpec.method="tpe"` over `param_ranges`, seeded, with
median pruning). `max_parallelism` was not built: nothing in the engine
runs in parallel and no registered estimator exposes `n_jobs`, so the
knob would control nothing. Phase 2 landed as `9e574ee` (the target registry) and phase
3 as `77dfffd` (paired comparison, `validation/comparison.py`) plus the
commit that carries this paragraph (`CombinatorialPurgedSplit`,
`method="cpcv"`, per-block purge, the path distribution in the validation
report, and the by-name refusals in the bridge, the portfolio evaluator
and the ensemble), all on 2026-09-20. Phase 1 landed as three commits on
2026-09-20 -- `ee5ad55`
(the preprocessing registry, pipeline and persisted state, replacing the
phase 0 stop-gap), `dcee66b` (the five remaining built-in steps, and
importance labelled by the pipeline's output) and the commit that carries
this paragraph (`DatasetSpec.missing`, in two layers) -- and added 78
modeling tests, 1,247 to 1,325. Two of the phase's open measurements were
answered from the source rather than by experiment: the native
preprocessing kernel already skips NaN when fitting and passes it through
when applying, so the `keep` policy uses the fused path unchanged; and
scikit-learn's own tags say which estimators accept NaN, so
`accepts_missing` is read rather than listed. Sector and beta
neutralization stay out, as section 4 says, for want of per-entity
metadata.
Every claim in sections 1-3 was checked against `main` at `f9c7008` on
2026-09-20 by reading the code, and the two defects marked *reproduced*
were reproduced by script. The baseline was 1,182 modeling tests passing
in 66 s on a clean tree. Phase 0 landed as six commits on 2026-09-20 --
`56e55c8` (F2, F4, F5, F6, F8), `230a7fe` (F3), `92c1ed6` (F1 stop-gap),
`4c49e2a` (hash v2), `3496b31` (environment fingerprint) and the commit
that carries this status line (generated reference, F7) -- and added 69
tests: the modeling suite went from 1,182 to 1,247 and `tests/docs` gained
four. One finding, F8 below, was not in the original read and was found
while fixing the others. Where a later phase depends on something not yet
measured, the measurement is named as the first task of that phase rather
than assumed.

The external review this responds to is largely right about what is built
and what is missing. Where this plan departs from it, section 5 says so and
why. The one-line difference: the review sequences by architectural value;
this plan sequences by **what is already wrong**, then by what each change
unblocks, because two of the review's "next steps" turn out to be the fix for
defects that exist today.

---

## 1. What is built, verified against the source

The review's description of the pipeline is accurate. The table records
where each property actually lives, so a later reader can check it rather
than trust it.

| Property | Where it is enforced |
|---|---|
| Per-fold target-overlap purge on each row's own `label_end_date`, correct for entities on different calendars and for purged k-fold's two-sided case | `engine.py:449-482` |
| Preprocessing fitted on training rows only, applied unchanged to test | `engine.py:206-253`, `features/transforms.py` |
| Inner hyperparameter search on dates, forward in time, never touching the outer test window | `validation/search.py` |
| Sample weights by label uniqueness / time decay, applied to folds AND the deployed refit | `validation/weights.py`, `engine.py:758` |
| Classifier calibration fitted on held-out inner folds, never on rows the estimator trained on | `engine.py:69-116` |
| Task-specific prepare/score/metrics behind three adapters | `adapters.py` |
| Deployed model refit on the full panel; `score_model` refuses any `as_of` at or before `max(label_end_date)` | `engine.py:825`, `scoring.py:71-119` |
| Registry: content hashes for every artifact, manifest written last as the commit point, bundled training `DatasetSpec`, feature implementation hashes checked at scoring | `registry/model_registry.py`, `scoring.py:149-192` |
| Two feature scopes, lags expanded per entity before stacking, negative lags refused at the schema | `features/base.py`, `dataset/lags.py` |
| One target registry (`TARGET_KINDS`), 6 buildable + 12 external-only labels, task compatibility derived from it | `specs.py:66-256`, `engine.py:165-203` |
| Several horizons from one build, selected per experiment, rows dropped by the chosen label only | `dataset/builder.py:463-475`, `agent/tools.py:376-456` |
| Estimator allowlist with bounded parameter values, 24 `(task, name)` pairs on this machine | `estimators/*.py` |
| Point-in-time join with `available_time` as the key, restatements as rows, staleness bound | `dataset/point_in_time.py` |
| Provider temporal contract declared, not inferred; every shipped provider says non-bar frames are unsupported | `data/temporal.py`, `data/base.py:206-241` |
| Intraday annualization refused rather than guessed | `features/base.py:62-84`, `features/risk.py:34-64` |
| Ensemble reads only OOS predictions, refuses to average across tasks | `ensemble.py` |
| Feature ablation refuses past a fit budget it computes first | `analysis/feature_ablation.py` |

Two small corrections to the review's numbers: the feature catalog is
**23** entries on this machine (the two `network.*` features postdate the
documentation's "21"), and `capabilities.py` reports the estimator list
from the registry but **hand-writes** the preprocessing, weighting and
search option lists (`capabilities.py:124-131`), contrary to its own
docstring.

The review is also right that the OOS prediction stream and the deployed
estimator are kept apart, and that the adapter contract assumes a 2-D
matrix. `adapters.py:22-29` says so explicitly and declines to abstract
past it until a real case arrives. That stance is kept here (section 4,
phase 8).

---

## 2. Defects found during the read

These are not in the review. They were found by reading the code the review
describes, and the first two are the reason the sequencing in section 4
differs from the review's. Fix these before building on them.

### F1. A model validated cross-sectionally is deployed pooled — *reproduced*

`_preprocess` (`engine.py:240-253`) branches on
`preprocessing.normalization`: the folds are standardized within each date
when `cross_sectional` is asked for. The full-panel refit at
`engine.py:732-734` does not branch: it calls `fit_preprocessing` and
`apply_preprocessing` unconditionally, which is the pooled winsorize/zscore
path. Those pooled statistics are what `save_model` persists and what
`score_model` applies (`scoring.py:304`).

So a model whose `validation_report.normalization` says `cross_sectional`
was validated on one transform and deployed on another. Measured on a
six-entity synthetic panel, ridge, three features: the deployed estimator's
predictions on the pooled transform and on the cross-sectional transform of
the same rows agree at Spearman **0.84**, and the mean absolute difference
between the two feature matrices is 0.40 standard deviations. The manifest
records nothing that would let a reader notice, and no test covers
cross-sectional scoring. This is exactly the failure the engine's own
comment at `engine.py:743-753` describes for weighting, one field over.

**Fix:** the deployed pipeline must be the fitted, serialized object the
folds used. That is phase 1's preprocessing registry, and it is why that
phase is first rather than merely valuable. The stop-gap for phase 0 is to
make the refit branch the same way `_preprocess` does and to persist
`normalization` in the manifest so `score_model` can apply the right one.

### F2. `score_model` cannot score a ranking model — *reproduced*

`scoring.py:305-308` branches `task == "regression"` → `predict`, else
`positive_class_proba`. `LGBMRanker` and `XGBRanker` have no
`predict_proba` (checked on the installed versions), so a registered ranker
fails inside scoring. The adapter that knows how to score a ranker exists
(`RankingAdapter.score`) and is not consulted. **Fix:** `score_model` calls
`get_adapter(manifest.task).score(estimator, X)`.

### F3. The inner search neither purges nor embargoes

`search.py:97-114` builds the inner walk-forward with `embargo=0`, and the
inner train/test selection at `search.py:163-169` never looks at
`label_end_date`. Training rows whose label resolves inside the inner test
window are scored on, so the hyperparameter selection is optimistically
biased in exactly the way the outer loop was fixed for. The outer OOS
metric stays clean; what is affected is *which* parameters get chosen.
**Fix:** thread the outer `embargo` and the panel's label-end column into
the inner split and apply the same overlap rule the engine applies.

### F4. `validate_model_spec` estimates the wrong fit count and runs a vacuous check

`agent/tools.py:1005` reads `validation.n_splits`, which every
`ValidationSpec` carries (default 5) whether or not the method is
`purged_kfold`. For the default `walk_forward` the fold count depends on
the dataset's date span, so the estimate is 5 regardless of the spec. And
at `agent/tools.py:1031-1036` the dataset branch builds both `available`
and `wanted` from the same `meta["feature_ids"]`, so `missing` is empty by
construction; a `ModelSpec` carries no feature list to check, and the
branch should either check `search.param_grid` names or be removed.
**Fix:** with a `dataset_id`, build the real splitter over the recorded
date axis and count; without one, say the count is unknown for walk-forward.
Phase 4 turns this into `ComputeBudget`.

### F5. `list_datasets` and `check_leakage` read keys the metadata never writes

`agent/tools.py:813-817` and `:935` read `rows`, `start_date` and
`end_date` from `dataset_meta.json`. Neither `build_model_dataset`
(`:160-187`) nor `register_external_panel` (`:323-358`) writes any of the
three, so every dataset lists with `rows=None` and no span, and the list is
sorted by a field that is always `None`. **Fix:** write them at build and
registration time; the doc says the list has them.

### F6. `compare_models` ranks regression by the metric the docs say not to use

`_HEADLINE_METRIC["regression"] = ("ic", ...)` at `agent/tools.py:737` is
the *pooled* Pearson IC, which `15_modeling.md` ("What the metrics mean")
explains conflates cross-sectional skill with market timing. The engine
"leads with" `cs_rank_ic_mean`. **Fix** in phase 3 alongside the paired
comparison, since the headline changes anyway.

### F7. Counts in docstrings and prose have drifted

Verified by grep at HEAD:

| Location | Says | Is |
|---|---|---|
| `modeling/agent/tools.py:2`, `:1902`; `agent/models.py:2`; `modeling/__init__.py:7` | 6-tool / 6-entry | 20 |
| `modeling/__init__.py:3`; `agent/dispatch.py:4`; `bridge.py:6` | 46-tool / 46-entry analysis surface | 180 |
| `dataset/target.py:41` | five-tool pipeline | 20 |
| `Documentation/15_modeling.md:98` | "17 + 9 is 26, and the whole library is 200" | 20 + 9 = 29; 209 |
| `15_modeling.md:110`, `:115` | seventeen / sixteen tools | 20 |
| `15_modeling.md:294` | 21 built-in features | 23 |
| `15_modeling.md`, "Explicitly deferred" | listed comparison tooling, prediction transforms, hyperparameter search and annualization as gaps that had already closed | pruned in the commit that added this plan; the section now names only what is still deferred |
| `Multi_Agent_Implementation/worker_agents.py:11`, `:83` | 16-tool runtime; "eight tools in one pipeline" | 20 |

The tool index is already generated and tested. Nothing generates the
feature, estimator or target tables, and the docs test's count guards match
specific phrasings that none of the above use. **Fix** in phase 0: generate
the modeling reference from the registries and stop writing counts into
docstrings.

### F8. `build_model_ensemble` raised `NameError` on every call — *found during phase 0*

`agent/tools.py`'s ensemble tool called a bare `publish` that nothing in
the module defined. Nothing ever reached it: the one test naming the tool
checked that it was registered, and the surface fuzzer's synthesized model
ids fail at `load_manifest` first. A tool that is advertised, dispatchable
and cannot run is the gap the advertised-equals-dispatchable invariant
cannot see; the fix routes through `handoff.publish` and an end-to-end
test combines two registered models into a reference that resolves.

---

## 3. Constraints every phase must respect

These come from the code and the tests, not from preference. A phase that
breaks one of them is not done.

1. **`dataset_spec_hash` covers every field of `DatasetSpec`.** Adding a
   field with a default changes the hash of every persisted dataset and
   makes `run_model_experiment` refuse it (`15_modeling.md`, "One upgrade
   note", records this happening for `horizons`). Phases 1, 5 and 8 each
   want a new dataset-level field. Phase 0 therefore moves the hash to a
   canonical form that excludes default-valued fields and records
   `spec_hash_version` in the metadata. Datasets built before that change
   need one rebuild, which is the documented remedy already; every
   additive field after it is free. `ModelSpec` fields are not hashed into
   dataset identity and can be added at will.
2. **The deployed pipeline is the validated pipeline.** Any state fitted
   per fold (preprocessing, calibration, conformal quantiles, weights) is
   fitted the same way for the refit and serialized so scoring applies it.
   F1 is what violating this looks like.
3. **Purge on `label_end_date`, never on an integer offset.** Every new
   splitter, inner or outer, reuses the engine's overlap rule.
4. **Allowlists, not imports.** New preprocessors, targets, estimators and
   search backends register with bounded parameter schemas
   (`estimators/bounds.py`'s `ParamBound` is reusable as is). No class
   paths in JSON.
5. **The OOS prediction frame is `(date, entity, prediction)` and five
   consumers read it** (`bridge`, `ensemble`, `portfolio_eval`,
   `score_predictions`, `analyze_model_errors`). Columns may be added;
   `prediction` keeps its meaning as the continuous score.
6. **The tool surface has budgets.** The modeling runtime must fit
   73,728 bytes of schema at default detail
   (`tests/surface/test_invariants.py:300`); every tool needs a typed
   `Result` and a strict input model; the worker split in
   `Multi_Agent_Implementation/worker_agents.py` and the per-runtime counts
   in `README.md` and `19_runtimes.md` are pinned by tests. Prefer a new
   `ModelSpec` field over a new tool: the invariant is "every tool is a
   decision".
7. **The native fast paths stay reachable.** `fit_preprocess_stats`,
   `apply_preprocess_stats`, `standardize_by_date`, `rank_by_date`,
   `label_uniqueness` and `cross_sectional_correlation` are the kernels
   that took preprocessing from ~50% of a run to a fraction of it
   (`Development/modeling_native_plan.md` in history). A preprocessing
   registry must dispatch the default steps to them.
8. **Tests plant the answer.** A new statistic gets a synthetic panel whose
   truth is known and a null case where it must decline to find anything;
   corrections get mutation-checked (`Documentation/25_testing.md`).

---

## 4. The phases

Each phase lists what it is for, the design at the level of files and spec
fields, what must stay true, how it is tested, and what "done" means. Size
is S (a day), M (a few days), L (a week or more) of focused work.

### Phase 0 — Correctness and hygiene (S–M)

Fixes F1–F7 and lays the two foundations later phases need.

- **F1 stop-gap.** `run_experiment`'s refit branches like `_preprocess`;
  `ModelManifest` gains `preprocessing: Dict` (the spec's normalization and
  clip) and `score_model` applies `standardize_cross_sectional` when that
  is what was validated. A test registers a cross-sectional model and
  asserts the deployed estimator, fed the fold transform, reproduces the
  final fold's predictions on that fold's rows. Superseded by phase 1's
  pipeline state, but the test survives.
- **F2.** `scoring.py` scores through `get_adapter(task).score`. Test:
  a registered `lightgbm_ranker` scores when the library is installed,
  skipped otherwise.
- **F3.** `search_best_params` takes `embargo` and the label-end array;
  the inner selection applies the engine's overlap mask. Test: a planted
  panel where a leaked inner fold prefers a wrong parameter and a purged
  one does not.
- **F4, F5, F6** as described. `list_datasets` also gains `provider` and
  `interval`, which the metadata already has.
- **Hash v2.** `dataset_spec_hash` uses `model_dump_json(exclude_defaults=True)`;
  `dataset_meta.json` records `spec_hash_version: 2`;
  `run_model_experiment` verifies with the recorded version and names the
  rebuild remedy for version 1.
- **Environment fingerprint.** `ModelManifest.environment`: Python,
  numpy, pandas, scikit-learn, scipy, lightgbm, xgboost versions; the
  native extension's export count and path (already computed by
  `capabilities._native_detail`); BLAS name from `numpy.show_config`;
  `platform.machine()`; `OMP_NUM_THREADS`/`MKL_NUM_THREADS`. Cheap, and
  every later phase's reproducibility claim rests on it.
- **Generated modeling reference.** `Development/generate_modeling_reference.py`
  writes `Documentation/29_modeling_reference.md` from `FEATURE_REGISTRY`,
  `ESTIMATOR_REGISTRY` + adapter capabilities, `TARGET_KINDS`, and the
  Literal options of `ValidationSpec`, `PreprocessingSpec`, `WeightingSpec`
  and `SearchSpec`. `tests/docs/test_documentation.py` regenerates and
  compares it exactly as it does the tool index. `15_modeling.md`'s
  hand-written feature and estimator tables are replaced by a link. The
  drifting docstrings drop their numbers (say "the modeling surface", not
  "the 6-tool surface"); a docs test asserts no `\d+-tool` or `\d+-entry`
  phrase remains under `src/standard_quant_tools/modeling`.

Done when: the seven findings each have a test that fails on HEAD and
passes after; the reference document is generated and pinned; the full
suite is green including `tests/docs` and `tests/surface`.

### Phase 1 — Preprocessing registry and missing-data policy (M–L)

The highest-return change in the review, and the real fix for F1.

**Design.** New package `modeling/preprocessing/`:

- `base.py` — `Preprocessor` protocol: `fit(X, ctx) -> state`,
  `transform(X, state, ctx) -> X`, `state` JSON-serializable, plus flags
  `stateless` (cross-sectional steps fit nothing) and `column_wise` (the
  ablation cache in phase 4 needs to know). `ctx` carries dates, entities
  and the fold role, never the target.
- `registry.py` — `PREPROCESSOR_REGISTRY: Dict[str, PreprocessorDefinition]`
  with a `ParamBound` schema per step and `register_preprocessor(...,
  overwrite=False)`, mirroring the feature and estimator registries.
- `steps.py` — the built-ins: `winsorize(lo, hi)`, `zscore`,
  `robust_scale` (median/MAD), `quantile_transform` (rank-gauss),
  `cross_sectional_standardize(clip_sigma)`, `missing_indicator`,
  `impute(strategy=median|mean|constant)`, `pca_whiten(n_components ≤ 64)`.
  Sector and beta neutralization are **not** in this list: they need
  per-entity metadata the repository does not carry, the same blocker
  `backtest.sizing` documents. Named so nobody looks for them.
- `pipeline.py` — `fit_pipeline(steps, X, ctx) -> PipelineState` and
  `apply_pipeline(state, X, ctx)`. When the step list begins
  `winsorize, zscore` on a float64 matrix, it dispatches to the fused
  native kernel; the Python steps remain the reference and the oracle.

**Spec.** `PreprocessingSpec` gains `steps: List[StepSpec]` where
`StepSpec(type, params)` has `extra="forbid"`. `normalization` and
`clip_sigma` stay for compatibility and are translated at validation:
`pooled` → `[winsorize(0.01, 0.99), zscore]`, `cross_sectional` →
`[cross_sectional_standardize(clip_sigma)]`. Supplying both `steps` and
`normalization` is refused. The default spec produces the byte-identical
transform of today, which a test pins against the current
`fit_preprocessing`/`apply_preprocessing` pair.

**Engine and registry.** `_preprocess` becomes fit-on-train /
apply-to-both through the pipeline; the search closure and the full refit
use the same function; `save_model` persists `preprocessing_state.json`
(the fitted `PipelineState`, content-hashed) and keeps writing
`preprocessing_stats.json` for one release so older readers load;
`score_model` applies the state. `capabilities()` reads the registry,
closing F7's hand-written list.

**Missing data.** Two layers, because the two remedies live in different
places:

- *Dataset layer* (`DatasetSpec.missing: MissingDataSpec`): `policy` is
  `drop` (today's behaviour and the default), `forward_fill_bounded` with
  `max_staleness_bars` and an explicit `features` allowlist (a bar's
  volume must not be carried; a slowly-updating level may be), or `keep`.
  Forward fill runs per entity before stacking, beside `expand_lags`, for
  the same reason lags do. Under `keep`, alignment drops on the target
  only, `drop_attribution` still reports what `drop` would have removed,
  and the finiteness check at `builder.py:548-552` becomes an
  infinity-only check for features. **First task of this sub-phase is a
  measurement:** whether the fused native `fit_preprocess_stats` skips NaN
  or propagates it. If it propagates, `keep` routes to the Python path
  until the kernel is extended.
- *Fold layer*: `impute` and `missing_indicator` are ordinary pipeline
  steps, fitted on the training rows only. Universe-scope features keep
  their complete-case intersection; a PCA needs a rectangle.

Since `MissingDataSpec` is a new `DatasetSpec` field, this phase depends on
phase 0's hash v2.

**Tests.** Every step: fit on train, apply to test, state round-trips
through JSON and reproduces the transform exactly; a step that peeks at
test rows is caught by a planted test-only outlier; `keep` + `impute`
recovers rows that `drop` loses and the recovered rows carry the training
median, never a value from the test fold; the deployed model's transform
equals the last fold's (the F1 test, now general).

Done when: a spec with no preprocessing fields produces the same model as
today (same predictions, same hash of `preprocessing_stats`); the
cross-sectional model deploys the transform it was validated on;
`list_modeling_capabilities` lists the steps from the registry.

### Phase 2 — Target registry (M)

Make a label as registrable as a feature or an estimator.

**Design.** New package `modeling/targets/`:

- `base.py` — `TargetDefinition(id, description, tasks, buildable,
  continuous, requires, builder, label_end_builder, cross_sectional_stage,
  default_params, param_bounds)`. `builder` takes the entity's full OHLCV
  and the resolved `TargetSpec` (today's builders take `Close` only; a
  realized-volatility or range-based label needs High/Low).
  `cross_sectional_stage` replaces the `CROSS_SECTIONAL_TARGETS` frozenset.
- `registry.py` — `TARGET_REGISTRY`, `register_target(definition,
  overwrite=False)`, `get_target`. `dataset/target.py` becomes the module
  that registers the six built-ins and re-exports `build_target`,
  `build_label_end_dates` and `apply_cross_sectional_target` as thin
  dispatchers, so nothing that imports them today changes.
- `specs.py` — `TargetSpec.type: str` validated against the registry
  (the `TargetType` Literal goes), with `json_schema_extra` writing the
  registered ids into the field's schema so an LLM still sees the choice
  list. `TargetSpec.params: Dict[str, object]` for custom labels, bounded
  like feature params through `features/params.resolve_params`. `TARGET_KINDS`
  stays as a read-only view for the three modules that import it.
  `test_the_literal_matches_the_registry` becomes "the schema's listed
  ids equal the registry".

**Tests.** A custom target registered in a test (`20d residual return`
against the benchmark) builds, purges on its own label ends, and refuses a
task it did not name; an external-only registration with a builder is
refused at `register_target`; every built-in reproduces today's panel
byte for byte.

Done when: the six built-ins and twelve external labels are registry
entries; a firm can `register_target` without touching `specs.py`,
`target.py` or `engine.py`; capabilities and the generated reference read
the registry.

### Phase 3 — CPCV and paired model comparison (M)

Strengthen selection before growing the model zoo.

**CPCV.** `validation/walk_forward.py` gains `CombinatorialPurgedSplit(n_groups,
n_test_groups, embargo)` over the date axis, yielding one `(train, test)`
per combination; the engine's two-sided overlap purge already handles
training rows on both sides of a test block. `ValidationSpec.method` gains
`"cpcv"` with `n_test_splits`. The combinatorics exist already in
`backtesting/overfitting.py::combinatorial_purged_cv` for the strategy
layer; that one is index-based with an integer horizon, so the modeling
splitter is written on dates and the docstring names its sibling. The
validation report gains per-path metrics and the distribution across paths
(mean, p05, p95), which is the number CPCV exists to produce.

A CPCV run tests each date in several paths, so its OOS frame is not a
single trading path. Rule, stated in the spec description and enforced by
name: `evaluate_model_portfolio` and the bridge refuse a `cpcv` model with
the doc's own sentence ("use walk-forward for what it would have earned").
The OOS artifact carries a `path` column (additive, constraint 5).

**Paired comparison.** New `validation/comparison.py`:
`paired_comparison(model_a, model_b, metric)` on the **intersection** of the
two OOS frames: per-date IC series for each, the difference series, a
moving-block bootstrap interval on its mean (reusing the block machinery in
`analysis/inference.bootstrap_statistic`), and for regression a
Diebold-Mariano test on the per-date squared-error differential with a
Newey-West variance at the label horizon. With more than two models,
Holm-adjusted p-values against a named reference, and a warning that a
family of candidates selected on the same OOS sample is what SPA-style
tests exist for. `compare_models` gains `method: "headline" | "paired"`,
`reference_model_id`, `n_bootstrap`, `block_size`; the result gains
`pairs`. F6 is fixed here: regression and ranking headline on
`cs_rank_ic_mean`.

**Tests.** Planted: two models whose OOS series differ by a known constant
plus AR(1) noise recover the constant inside the interval; two models with
identical skill and different noise produce an interval that covers zero
(the null); the block interval is wider than the IID one on autocorrelated
differences and equal on white noise.

Done when: `compare_models(method="paired")` reports an interval, not an
ordering; a CPCV run reports a distribution; both are in the generated
reference.

### Phase 4 — Experiment plan, fold cache, compute budget, search backend (M)

**Plan before run.** `engine.plan_experiment(dataset, model_spec) ->
ExperimentPlan`: a pure function producing the folds with date ranges,
purge counts, the candidate list, the fit count and a content hash per node
(dataset hash × fold × preprocessing steps × estimator params × seed).
`run_experiment` executes a plan. `validate_model_spec` reports the plan's
fit count when given a `dataset_id` (fixing F4 properly) and refuses past
`ComputeBudget`.

**`ComputeBudget`** on `ModelSpec`: `max_fits` (default 500, the same
shape as `feature_ablation.DEFAULT_MAX_FITS`), `max_parallelism`. Refuse
before the first fit, never truncate.

**Fold cache.** In-process, keyed by the plan's node hashes: preprocessed
train/test matrices per fold. The inner search re-preprocesses the same
inner folds once per candidate today; feature ablation refits the whole
walk-forward per feature and, for column-wise steps, can drop a column from
the cached matrix exactly (`column_wise` from phase 1 is what makes that
safe; `pca_whiten` is not column-wise and falls back). A persistent,
cross-process cache is deliberately not built: the hashes make it possible,
and orchestration is the layer above this library.

**Search backend.** `SearchSpec.method` gains `"tpe"` behind an optional
`optuna` import guarded like lightgbm (installed here, not declared);
`max_trials`, `early_pruning`. Same inner splitter (purged and embargoed
after F3), same `fit_predict` closure, sampler seeded from `random_seed`,
trials recorded in `search_reports`. Capabilities report `optuna`.

Done when: a search-heavy spec is refused by budget before fetching
anything; ablation on a 20-feature panel measurably reuses fold
preprocessing (record the before/after time in the doc); `tpe` selects
on the same purged inner folds `grid` does.

### Phase 5 — Point-in-time source and calendar metadata (L)

The information-set expansion. The join, the contract and the bundle
already exist; what is missing is a provider that supplies vintages and a
feature scope that consumes them.

**Provider.** `DataProvider.get_point_in_time_records(symbols, frame_kind,
fields, start, end)` returning the `point_in_time.py` schema
(`event_time`, `available_time`, `entity`, fields). The base raises
`NotImplementedError` by name, as `get_trades` does. First implementation:
Polygon's financials endpoint, which returns `filing_date` and
`period_of_report_date` per filing and lists amended filings as separate
rows. **First task is a measurement:** pull one symbol's history and
confirm restatements arrive as rows (`versioned`) rather than overwritten
(`snapshot`), then declare the `TemporalContract` accordingly; the claim is
not made until it is checked.

**Feature scope.** `FeatureScope.POINT_IN_TIME`: a `FeatureDefinition`
with `frame_kind`, `fields`, `max_staleness_days` and a `transform`
(e.g. surprise = actual − estimate, revision momentum = change in
consensus over 30 days). The builder joins these after stacking with
`asof_join`, bounded by staleness, and refuses at build time when the
provider's contract is not `pit_safe` — the moment `CURRENT_ONLY` stops
being a label nothing uses. The `DataBundle` records the fundamentals
frame and its contract, so the verdict travels with the dataset. Feature
candidates once the source exists: valuation, earnings surprise, estimate
revisions, profitability, balance-sheet momentum; macro surprise through
`by_entity=False`.

**Calendar.** `DatasetSpec.calendar: Optional[str]` (an `exchange_calendars`
name; optional dependency), and `periods_per_year_for_interval(interval,
calendar)` derives bars per session × sessions per year for intraday
intervals. The refusal in `risk.py:_annualization` stays for a missing
calendar. `AssetKey` (venue, asset class, contract) is not built here:
`universe` stays a list of symbols, and the collision cases the review
lists are recorded as the reason to revisit.

Both fields depend on phase 0's hash v2.

Done when: a dataset joins a PIT fundamental feature with a recorded
contract and a coverage warning; a July-15 row carries nothing for a
July-29 filing and an August-20 row carries the revision; an intraday
dataset with a calendar annualizes and one without still refuses.

### Phase 6 — Distributional predictions (M)

**Spec.** `ModelSpec.quantiles: Optional[List[float]]` (regression only).
The estimator registry gains a `quantile_param` capability: `quantile` →
`quantile`, `quantile_gradient_boosting` → `alpha`, `lightgbm` → objective
`quantile` + `alpha` (added to its bounded schema), `xgboost` →
`reg:quantileerror` + `quantile_alpha`. An estimator without one refuses
the field. One fit per quantile per fold; the OOS frame gains `q05`,
`q50`, … columns; `prediction` stays the point score (the median when 0.5
is requested).

**Conformal.** `ModelSpec.intervals: ConformalSpec(alpha, method="split")`
fits residual quantiles on the inner held-out folds inside each training
window, exactly as calibration does, and serializes them with the model so
scoring emits intervals.

**Metrics.** Pinball loss per quantile, central-interval coverage and mean
width, quantile-crossing rate; reported beside the point metrics.
`PredictionTransformSpec.method` gains `uncertainty_scaled` (score over
interval width), which is where the portfolio layer starts consuming the
distribution.

Done when: a quantile model's OOS frame carries the requested columns and
every downstream consumer still reads `prediction` unchanged; coverage on
a planted Gaussian panel matches the nominal level within bootstrap error.

### Phase 7 — Survival task (M–L)

`time_to_fill` is declared censored in its own description and is fitted
today as a regression, which the description says biases every estimate.

**Task.** `TASKS` gains `"survival"`. `SurvivalAdapter.prepare` builds the
structured label from `(duration, event)`; `score` returns a risk score
(higher means the event sooner), so the bridge and the portfolio path read
it like any other score; `metrics` returns Harrell's concordance (own
implementation, no dependency) and the integrated Brier score when
`scikit-survival` is present. **Estimators:** `xgboost_aft` and
`xgboost_cox` first, because xgboost is already installed and both are
objectives on a class already in the registry; `cox_ph`,
`random_survival_forest` and `gradient_boosting_survival` behind an
optional `scikit-survival` import.

**Labels.** External panels declare `event_column` per target;
`time_to_fill`'s tasks become `("survival",)` — a deliberate break, since a
regression on a censored label is the thing being removed.
`fill_probability` stays a classification label; a survival curve can
derive it at any horizon, which is a later convenience.

Done when: a planted panel with known hazard recovers the ordering
(concordance near the truth's) and a regression on the same censored label
is refused by the task check.

### Phase 8 — Representation contract, sequence kind later (S now, L later)

**Now (S).** Formalize what the engine already does implicitly: a
`SampleIndex(dates, entities, label_end)` passed to adapters beside the
arrays, so the purge, the weights and the metrics are defined on sample
metadata rather than on the shape of `X`. `FitArrays.X` is typed by the
adapter's declared `input_kind`. No spec field is added: a
`RepresentationSpec(kind="tabular")` with one allowed value would churn
every persisted `ModelSpec` for no behaviour, and the spike in
`Development/spike_lags_and_multioutput.py` measured a shared
representation at +0.0014 R² on the most favourable panel it could be
given.

**Later (L), gated on a concrete model.** `ModelSpec.representation`
(on the model spec, not the dataset spec, so dataset hashes are untouched)
with `kind="sequence"`, `lookback`, `stride`; a `SequenceAdapter` that
builds `(n, T, F)` per entity **within the fold** from the tabular panel;
a torch-backed estimator behind an optional extra with bounded width,
depth and epochs. The gate is a measured case on an external microstructure
panel where an MLP over lag columns loses to a sequence model by more than
bootstrap noise. Until then the lag columns are the sequence.

### Phase 9 — Lifecycle stages and monitoring (M)

**Stages.** `registry/lifecycle.py`: `candidate → validated → staging →
production → archived`. Promotions are an append-only `promotions.jsonl`
in the model directory (from, to, reason, actor, timestamp, evidence
references); the manifest stays immutable and content-hashed; `stage` is
derived by reading the log. One new tool, `promote_model`, because a
promotion is a decision; `list_models` gains a `stage` filter.

**Monitoring.** Registration persists `feature_profile.json` (per-feature
quantile edges and missing rate on the training panel) and the OOS
prediction distribution. `score_model` persists the scored feature matrix
beside the predictions. `monitor_model(model_id, predictions_uri)` then
reports feature PSI and KS (the code exists in
`analysis/feature_stability.py`), prediction-distribution drift, and —
when outcomes for the scored dates exist — realized IC against the
validation IC. Status thresholds are reported with the numbers, never
alone.

Two tools take the runtime to 22; check the schema budget, the worker
lists and the pinned counts (constraint 6).

### Phase 10 — Artifact store and signed manifests (M)

`ArtifactStore` protocol (`put`, `get`, `exists`, `list`, `hash`) behind
`_runspath` and `artifacts.py`; `LocalArtifactStore` is the default and
the only one tested in-tree; an object-store implementation behind an
optional `fsspec` extra. The manifest-last commit point and the content
hashes survive unchanged.

Signing: `manifest.sig` over the manifest bytes with the Ed25519 helpers in
`audit/signing.py`; `SQT_MODEL_SIGNING_KEY_PATH`; `load_manifest(...,
require_signature=True)` and a `verify_model_package` entry point. The
registry's own docstring at `model_registry.py:278-283` names this as the
next step before the registry crosses a trust boundary. Serialization stays
joblib; `skops` export for sklearn-only estimators as an optional bundle
format; no ONNX.

---

## 5. Where this departs from the review

- **Preprocessing is first for a different reason.** The review ranks it
  by return; here it is first because the current two-scheme
  implementation deploys the wrong transform (F1). The registry is the fix,
  not only an enrichment.
- **No `RepresentationSpec` now.** Adding a one-valued field is hash churn
  with no behaviour (constraint 1), and the repository's own measurement
  argues against the model that would use it. The `SampleIndex` contract
  gets the engine ready without the field.
- **Missing data is not one spec.** Bounded forward fill is a per-entity,
  pre-stacking operation and belongs beside lags; imputation is a fitted,
  per-fold step and belongs in the pipeline. One `MissingDataSpec` that
  did both would put a fold-fitted operation in the dataset's identity.
- **The inner economic objective (review §22) is deferred.** Scoring inner
  folds on IC − λ·turnover − λ·cost needs a per-fold cost model and a
  rebalance schedule inside the search loop; the outer path already
  answers "which model survives costs" through `evaluate_model_portfolio`
  without contaminating the OOS sample. Revisit after phase 4's plan and
  cache make inner-fold work cheap.
- **Feature additions wait for the source, except the cheap ones.**
  Realized semivariance, bipower variation, Amihud and volume surprise are
  price/volume features that register today with no architectural change;
  they do not need a phase. Fundamental, macro and event features need
  phase 5 and are not worth adding as `CURRENT_ONLY` placeholders.
- **CPCV models are not portfolio-evaluable**, by rule rather than by
  accident, because the doc already draws that line for purged k-fold.
- **`partial_fit` stays out** (review §16 agrees). An `OnlineModelRuntime`
  with a version chain is a separate execution mode and is not scheduled
  here; nothing in phases 0–10 depends on it.

---

## 6. The first milestone

Phase 0 and phase 1, as one sequence of commits, each green on the full
suite:

1. F2, F4, F5 — three small, independent fixes with tests.
2. F3 — inner search purge and embargo, with the planted-leak test.
3. F1 stop-gap, manifest `preprocessing` field, cross-sectional scoring
   test.
4. Hash v2 and `spec_hash_version`.
5. Environment fingerprint in the manifest.
6. Generated modeling reference, docstring counts removed, docs test
   extended. F7 closes here.
7. Preprocessing registry with the two built-in step lists, byte-identical
   default; engine and scoring on `PipelineState`; F1's stop-gap replaced.
8. Remaining built-in steps; `capabilities()` reads the registry.
9. `MissingDataSpec` after the native-kernel NaN measurement.

After that milestone the runtime deploys what it validates, an agent can
compose a preprocessing pipeline from a bounded catalog, and every count in
the documentation is generated. Phases 2 and 3 follow; phases 4–10 are
sequenced but each is worth re-planning against what the previous one
measured, which is the lesson `f576d75` recorded.
