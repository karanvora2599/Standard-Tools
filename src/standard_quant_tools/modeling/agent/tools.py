"""
The 6-tool modeling agent surface — kept structurally separate from
standard_quant_tools.agent.get_agent_tools()/TOOL_CATEGORY (the existing
46-tool analysis/backtest registry) per Documentation/15_modeling.md's
architecture rationale: fitting/validating/registering a statistical
model doesn't fit that surface's shape (a point-in-time snapshot or a
single backtest run), and adding it there would make the tool-selection
ambiguity problem TOOL_CATEGORY already exists to mitigate worse, not
better.

    list_features        — the feature catalog, not semantic search
                            (21 entries doesn't need ranking).
    build_model_dataset  — DatasetSpec -> persisted panel + dataset_id.
    run_model_experiment — dataset_id + ModelSpec -> fit + walk-forward
                            validate + register, one call. Structurally
                            impossible to fit without validation, since
                            there is no separate "just fit" tool.
    score_model           — model_id + as_of + universe -> predictions
                            artifact.
    inspect_model          — one tool, four views, instead of four
                            separate inspection tools.
    evaluate_model_portfolio — model_id + transform/portfolio specs ->
                            OOS predictions run through the shared-cash
                            portfolio simulator as target weights.

Why evaluate_model_portfolio is a TOOL while `bridge`'s signal-panel
conversion deliberately is not: the bridge only RESHAPES an artifact the
caller already has, and hands it to a tool in the other registry — there
is no decision in it, so exposing it would have been a sixth name for an
argument-shaping step. This one runs a simulation, produces new persisted
artifacts, and answers the question an agent actually asks after training
("is this model worth trading"), which is the same shape of operation
score_model already occupies. The 5-tool count was never the invariant;
"every tool is a decision the agent makes, not plumbing" was.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from standard_quant_tools.audit.hashing import hash_dataframe
from standard_quant_tools.error import ValidationError

from .. import artifacts as _artifacts
from ..analysis import build_feature_report
from ..capabilities import modeling_capabilities
from ..dataset.builder import build_dataset as _build_dataset
from ..dataset.builder import dataset_spec_hash
from ..engine import run_experiment as _run_experiment
from ..features.registry import list_features as _list_features
from ..portfolio_eval import evaluate_model_portfolio as _evaluate_model_portfolio
from ..registry.model_registry import load_manifest
from ..scoring import score_model as _score_model
from ..specs import DatasetSpec
from .models import (
    AnalyzeFeaturesInput,
    AnalyzeFeaturesResult,
    BuildModelDatasetInput,
    BuildModelDatasetResult,
    CheckLeakageInput,
    CheckLeakageResult,
    CompareModelsInput,
    CompareModelsResult,
    DatasetSummary,
    EvaluateModelPortfolioInput,
    EvaluateModelPortfolioResult,
    FeatureCatalogEntry,
    InspectModelInput,
    InspectModelResult,
    LeakageFinding,
    ListDatasetsInput,
    ListDatasetsResult,
    ListFeaturesInput,
    ListFeaturesResult,
    ListModelingCapabilitiesInput,
    ListModelingCapabilitiesResult,
    ListModelsInput,
    ListModelsResult,
    ModelComparison,
    ModelSummary,
    RunModelExperimentInput,
    RunModelExperimentResult,
    ScoreModelInput,
    ScoreModelResult,
    ScorePredictionsInput,
    ScorePredictionsResult,
    SpecProblem,
    ValidateModelSpecInput,
    ValidateModelSpecResult,
)

logger = logging.getLogger(__name__)


def list_features(input_data: ListFeaturesInput) -> ListFeaturesResult:
    """Return the feature catalog, optionally filtered to one category."""
    defs = _list_features(category=input_data.category)
    return ListFeaturesResult(
        features=[
            FeatureCatalogEntry(
                id=d.id,
                description=d.description,
                default_params=d.default_params,
                temporal_support=d.temporal_support.value,
                scope=d.scope.value,
                requires=d.requires,
                lookback=d.lookback,
            )
            for d in defs
        ]
    )


def build_model_dataset(input_data: BuildModelDatasetInput) -> BuildModelDatasetResult:
    """Fetch OHLCV for DatasetSpec.universe, compute the requested
    features/target, and persist the resulting panel — never returned
    inline, the same "don't dump the full curve into the agent context"
    discipline BacktestResultV2's equity_curve_uri uses."""
    logger.debug(
        "[build_model_dataset] universe=%s  %s -> %s  features=%d  provider=%s  interval=%s",
        input_data.spec.universe,
        input_data.spec.start,
        input_data.spec.end,
        len(input_data.spec.features),
        input_data.spec.provider,
        input_data.spec.interval,
    )
    built = _build_dataset(input_data.spec)

    dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
    panel_uri = _artifacts.save_artifact(
        built["panel"], run_id=dataset_id, name="panel"
    )
    directory = Path(panel_uri).parent
    # dataset_spec: the exact DatasetSpec used, so score_model can later
    # rebuild identical features (see scoring.py).
    _artifacts.save_json(directory, "dataset_spec", input_data.spec.model_dump())
    # dataset_meta: build_dataset's own lineage fields, so
    # run_model_experiment can reload them without recomputing anything
    # or re-deriving them from dataset_spec.
    # Written LAST: every reader keys off dataset_meta.json, so a crash
    # partway through leaves a directory that is simply not a dataset
    # rather than a half-written one that looks loadable. Same write-order
    # transaction boundary save_model uses for manifest.json.
    _artifacts.save_json(
        directory,
        "dataset_meta",
        {
            "feature_ids": built["feature_ids"],
            "target_id": built["target_id"],
            "data_hash": built["data_hash"],
            # spec_hash was computed by build_dataset and then discarded.
            # Persisted so a model can be tied to the exact feature/target
            # DEFINITION, not just to the resulting data.
            "spec_hash": built["spec_hash"],
            "entities": built["entities"],
            # Persisted so run_model_experiment can carry them into the
            # manifest without re-reading dataset_spec.json, and so the
            # conditions the caller was warned about at build time remain
            # attached to the dataset rather than living only in the
            # tool response they may not have kept.
            "provider": input_data.spec.provider,
            "interval": input_data.spec.interval,
            "warnings": built["warnings"],
            # Per-column row loss from feature/target alignment. Kept with
            # the dataset rather than only in the tool response, so "why is
            # this panel so small" is answerable later without a rebuild.
            "drop_attribution": built["drop_attribution"],
            "entities_fetched": built["entities_fetched"],
        },
    )

    return BuildModelDatasetResult(
        dataset_id=dataset_id,
        rows=len(built["panel"]),
        entities=built["entities"],
        feature_ids=built["feature_ids"],
        target_id=built["target_id"],
        warnings=built["warnings"],
        drop_attribution=built["drop_attribution"],
    )


def _load_dataset_panel(dataset_id: str):
    """
    Load a persisted dataset panel, verifying it is the one that was built.

    Extracted so every consumer of a dataset_id gets the integrity check,
    not just the training path. The check matters as much for analysis as
    for fitting: a feature report computed from an edited panel.parquet
    would describe data that no recorded dataset hash covers, which is a
    quieter failure than a wrong model but the same kind.

    Returns (panel, meta, directory). The directory comes back because
    callers need it for the sibling artifacts written next to the panel --
    run_model_experiment reads dataset_spec.json from it to verify the spec
    hash as well as the data hash.
    """
    directory = _artifacts.run_dir(dataset_id)
    meta_path = directory / "dataset_meta.json"
    if not meta_path.exists():
        raise ValidationError(
            f"no dataset with dataset_id={dataset_id!r} — "
            "dataset_meta.json is written last, so its absence also means a "
            "previous build_model_dataset call did not complete."
        )
    panel = _artifacts.load_artifact(str(directory / "panel.parquet"))
    meta = _artifacts.load_json(str(meta_path))

    # The panel is reloaded from disk and its hash was recorded at build
    # time, but nothing previously re-derived it -- so an edited
    # panel.parquet trained a model whose manifest recorded the ORIGINAL
    # panel's hash, making the lineage actively misleading rather than
    # merely incomplete.
    stored_hash = meta.get("data_hash")
    if stored_hash is not None:
        actual_hash = hash_dataframe(panel)
        if actual_hash != stored_hash:
            raise ValidationError(
                f"dataset {dataset_id!r}: panel.parquet no longer matches the "
                f"hash recorded when it was built (expected {stored_hash}, found "
                f"{actual_hash}). Using it would record a lineage hash that does "
                "not describe the data actually used — rebuild the dataset instead."
            )
    return panel, meta, directory


def run_model_experiment(
    input_data: RunModelExperimentInput,
) -> RunModelExperimentResult:
    """Load the persisted dataset panel + its lineage metadata,
    fit+walk-forward-validate+register a model — one call, no separate
    "just fit" path."""
    logger.debug(
        "[run_model_experiment] dataset_id=%s  task=%s  estimator=%s",
        input_data.dataset_id,
        input_data.spec.task,
        input_data.spec.estimator.type,
    )
    panel, meta, directory = _load_dataset_panel(input_data.dataset_id)

    # dataset_spec.json gets the same treatment panel.parquet just got, and
    # for a sharper reason. The panel is only READ during training, but the
    # spec is COPIED INTO THE MODEL and becomes the definition score_model
    # rebuilds features from for the rest of that model's life. So an edited
    # spec (say RSI period 14 -> 100) trains on the original RSI(14) panel --
    # its hash still matches -- and then registers a model that will score
    # every future prediction with RSI(100). Verifying the panel but not the
    # spec left exactly the mismatch the self-contained-model work existed to
    # prevent.
    stored_spec_hash = meta.get("spec_hash")
    spec_dict = _artifacts.load_json(str(directory / "dataset_spec.json"))
    if stored_spec_hash is not None:
        actual_spec_hash = dataset_spec_hash(DatasetSpec(**spec_dict))
        if actual_spec_hash != stored_spec_hash:
            raise ValidationError(
                f"dataset {input_data.dataset_id!r}: dataset_spec.json no longer matches "
                f"the hash recorded when it was built (expected {stored_spec_hash}, found "
                f"{actual_spec_hash}). The panel was built from the original spec, so "
                "training would register a model whose bundled feature definitions differ "
                "from the data it learned on — rebuild the dataset instead."
            )

    dataset = {
        "panel": panel,
        "feature_ids": meta["feature_ids"],
        "target_id": meta["target_id"],
        "data_hash": meta["data_hash"],
        "spec_hash": stored_spec_hash,
        # Bundled into the model so it becomes self-contained -- see
        # registry.model_registry.save_model.
        "dataset_spec": spec_dict,
        # .get, not [...]: datasets built before coverage diagnostics
        # existed have no such key, and a missing warning list is not the
        # same claim as an empty one -- see ModelManifest.dataset_warnings.
        "warnings": meta.get("warnings", []),
    }
    result = _run_experiment(dataset, input_data.spec, dataset_id=input_data.dataset_id)
    # Republished with a content kind so the rest of the interconnect can
    # type-check it. Same rows, addressed as `predictions` rather than as
    # a path that says nothing about what it holds.
    from standard_quant_tools.agent.runtimes import handoff
    from standard_quant_tools.backtest.artifacts import load_artifact

    predictions_ref = handoff.publish(
        load_artifact(result["oos_predictions_uri"]),
        "predictions",
        result["model_id"],
        "oos_predictions_ref",
        producer="modeling.run_model_experiment",
        overwrite=True,
    )

    result["oos_predictions_ref"] = predictions_ref
    return RunModelExperimentResult(**result)


def score_model(input_data: ScoreModelInput) -> ScoreModelResult:
    result = _score_model(
        model_id=input_data.model_id,
        as_of=input_data.as_of,
        universe=input_data.universe,
        lookback_days=input_data.lookback_days,
        max_staleness_days=input_data.max_staleness_days,
    )
    return ScoreModelResult(**result)


def inspect_model(input_data: InspectModelInput) -> InspectModelResult:
    """One tool, four views — avoids five separate inspection tools for
    what's ultimately reading different slices of the same manifest."""
    manifest = load_manifest(input_data.model_id)

    if input_data.view == "summary":
        data: Dict[str, Any] = {
            "task": manifest.task,
            "estimator_type": manifest.estimator_type,
            "oos_metrics": manifest.oos_metrics,
            "n_folds": manifest.n_folds,
            "feature_ids": manifest.feature_ids,
            "target_id": manifest.target_id,
            "created_at_utc": manifest.created_at_utc,
        }
    elif input_data.view == "feature_importance":
        data = {"feature_importance_summary": manifest.feature_importance_summary}
    elif input_data.view == "validation":
        data = {
            "validation_method": manifest.validation_method,
            "n_folds": manifest.n_folds,
            "oos_metrics": manifest.oos_metrics,
            # Per-fold detail plus fold accounting. Averages alone cannot
            # show performance decay across time, which window carried the
            # result, or how many folds were skipped and why.
            "validation_report": manifest.validation_report,
        }
    else:  # lineage
        data = {
            "dataset_id": manifest.dataset_id,
            "dataset_hash": manifest.dataset_hash,
            "oos_predictions_uri": manifest.oos_predictions_uri,
            "random_seed": manifest.random_seed,
            "git_commit_sha": manifest.git_commit_sha,
            "package_version": manifest.package_version,
            "created_at_utc": manifest.created_at_utc,
            # What the data behind these metrics does not guarantee. This
            # is the view someone opens months later to decide whether to
            # trust a model, and it previously reported hashes and a commit
            # sha while staying silent about a survivors-only universe.
            "dataset_warnings": manifest.dataset_warnings,
        }

    return InspectModelResult(
        model_id=input_data.model_id, view=input_data.view, data=data
    )


def evaluate_model_portfolio(
    input_data: EvaluateModelPortfolioInput,
) -> EvaluateModelPortfolioResult:
    """
    Run a registered model's walk-forward out-of-sample predictions through
    the shared-cash portfolio simulator and report what they were worth
    after costs.

    This is the economic counterpart to run_model_experiment's statistical
    metrics. A model with a strong IC can still lose money once its ranking
    is turned into position sizes, held for a realistic period, and charged
    commission, spread and (if configured) borrow — and nothing in
    oos_metrics can show that.

    Uses OOS predictions only, never score_model: score_model's estimator is
    the final full-panel refit, so scoring historical dates with it would be
    in-sample and the equity curve would be fiction.
    """
    result = _evaluate_model_portfolio(
        model_id=input_data.model_id,
        transform=input_data.transform,
        portfolio=input_data.portfolio,
    )
    return EvaluateModelPortfolioResult(**result)


#: Metric each task is ranked by when the caller names none. Ranking a
#: regression R2 against a classification AUC would produce an ordering
#: that looks meaningful and is not, so tasks are ranked separately.
_HEADLINE_METRIC = {
    "regression": ("ic", "spearman_ic", "r2"),
    "classification": ("auc", "roc_auc", "accuracy"),
    "ranking": ("ndcg_at_10", "ndcg", "ic"),
}


def _headline(task: str, metrics: dict, preferred=None):
    """(metric name, value) for one model, or (None, None)."""
    candidates = (preferred,) if preferred else _HEADLINE_METRIC.get(task, ())
    for name in candidates:
        if name and name in metrics:
            return name, float(metrics[name])
    return None, None


def list_models(input_data: ListModelsInput) -> ListModelsResult:
    """
    Every registered model, newest first.

    `inspect_model` and `score_model` both require a model_id the caller
    already holds, and nothing enumerated them — so a session that lost the
    id, or any new session, could not find a model it had trained. The
    registry has been on disk the whole time; this reads it.

    Manifests that fail to load are skipped rather than failing the call: a
    half-written run should not make the other twenty models unfindable.
    """
    directory = _artifacts._runs_dir()
    summaries = []
    for path in sorted(directory.glob("mdl_*/manifest.json")):
        try:
            manifest = load_manifest(path.parent.name)
        except Exception:  # a partial or corrupt run, not a reason to fail
            continue
        if input_data.task and manifest.task != input_data.task:
            continue
        metric, value = _headline(manifest.task, manifest.oos_metrics)
        summaries.append(
            ModelSummary(
                model_id=manifest.model_id,
                task=manifest.task,
                estimator=manifest.estimator_type,
                created_at=manifest.created_at_utc,
                n_features=len(manifest.feature_ids),
                n_folds=manifest.n_folds,
                headline_metric=metric,
                headline_value=value,
                dataset_id=manifest.dataset_id,
            )
        )
    summaries.sort(key=lambda s: s.created_at or "", reverse=True)
    return ListModelsResult(
        models=summaries[: input_data.limit],
        n_total=len(summaries),
        registry_dir=str(directory),
    )


def list_datasets(input_data: ListDatasetsInput) -> ListDatasetsResult:
    """
    Every built dataset panel, newest first.

    Same gap as `list_models`: `run_model_experiment` needs a dataset_id
    and nothing could produce the list of them.
    """
    directory = _artifacts._runs_dir()
    summaries = []
    for path in sorted(directory.glob("ds_*/dataset_meta.json")):
        try:
            meta = _artifacts.load_json(str(path))
        except Exception:
            continue
        summaries.append(
            DatasetSummary(
                dataset_id=path.parent.name,
                rows=meta.get("rows"),
                entities=len(meta.get("entities", []) or []) or None,
                features=len(meta.get("feature_ids", []) or []) or None,
                start_date=meta.get("start_date"),
                end_date=meta.get("end_date"),
            )
        )
    summaries.sort(key=lambda s: s.end_date or "", reverse=True)
    return ListDatasetsResult(
        datasets=summaries[: input_data.limit],
        n_total=len(summaries),
        runs_dir=str(directory),
    )


def compare_models(input_data: CompareModelsInput) -> CompareModelsResult:
    """
    Rank several registered models side by side.

    Models are ranked WITHIN their task, never across it. A regression IC
    and a classification AUC are both "about 0.6" on the same scale and
    mean entirely different things, so a combined ordering would look
    authoritative while being arithmetic on incomparable quantities.
    """
    comparisons = []
    tasks = {}
    for model_id in input_data.model_ids:
        manifest = load_manifest(model_id)
        metric, value = _headline(
            manifest.task, manifest.oos_metrics, input_data.metric
        )
        comparisons.append(
            ModelComparison(
                model_id=model_id,
                task=manifest.task,
                metric=metric,
                value=value,
                n_features=len(manifest.feature_ids),
                dataset_id=manifest.dataset_id,
            )
        )
        tasks.setdefault(manifest.task, []).append(comparisons[-1])

    notes = []
    best = {}
    for task, group in tasks.items():
        scored = [c for c in group if c.value is not None]
        scored.sort(key=lambda c: c.value, reverse=True)
        for position, comparison in enumerate(scored, start=1):
            comparison.rank = position
        if scored:
            best[task] = scored[0].model_id
        missing = [c.model_id for c in group if c.value is None]
        if missing:
            notes.append(
                f"{task}: {missing} carry no usable metric and are unranked. "
                "Pass `metric` explicitly if the manifest records one under "
                "a different name."
            )
    if len(tasks) > 1:
        notes.append(
            f"These models span {sorted(tasks)}. They are ranked separately "
            "because their metrics are not on a common scale — comparing "
            "across tasks here would be arithmetic on incomparable numbers."
        )

    return CompareModelsResult(comparisons=comparisons, best_by_task=best, notes=notes)


def check_leakage(input_data: CheckLeakageInput) -> CheckLeakageResult:
    """
    Ask whether a set of features is safe to fit on before fitting on it.

    `check_point_in_time_safety` runs implicitly inside dataset building;
    this makes it a question a caller can ask directly, which is the
    agent-shaped version — the answer changes which features you pick, and
    finding out during a build means the build already happened.

    With a `dataset_id`, also reports that panel's point-in-time coverage:
    how much of it is genuinely as-of rather than back-filled.
    """
    from standard_quant_tools.modeling.dataset.leakage import (
        PointInTimeViolation,
        check_point_in_time_safety,
    )
    from standard_quant_tools.modeling.features.registry import get_feature

    ids = input_data.feature_ids
    if ids is None:
        from standard_quant_tools.modeling.features.registry import FEATURE_REGISTRY

        ids = sorted(FEATURE_REGISTRY)

    definitions, findings, notes = [], [], []
    for feature_id in ids:
        try:
            definitions.append(get_feature(feature_id))
        except Exception as exc:
            findings.append(
                LeakageFinding(
                    feature_id=feature_id,
                    temporal_support="unknown",
                    problem=f"not in the feature registry: {exc}",
                )
            )

    try:
        check_point_in_time_safety(definitions)
    except PointInTimeViolation as exc:
        findings.append(
            LeakageFinding(
                feature_id="(set)",
                temporal_support="violation",
                problem=str(exc),
            )
        )

    coverage = {}
    if input_data.dataset_id is not None:
        _panel, meta, _directory = _load_dataset_panel(input_data.dataset_id)
        coverage = {
            key: meta[key]
            for key in ("rows", "start_date", "end_date", "warnings")
            if key in meta
        }
        notes.append(
            "Dataset coverage is what the build RECORDED. It confirms the "
            "panel is the one that was built; it does not re-derive whether "
            "each value was available on its own date."
        )

    return CheckLeakageResult(
        n_features_checked=len(ids),
        safe=not findings,
        findings=findings,
        dataset_coverage=coverage,
        notes=notes,
    )


def validate_model_spec(input_data: ValidateModelSpecInput) -> ValidateModelSpecResult:
    """
    Check a ModelSpec before spending an experiment on it.

    `run_model_experiment` is the most expensive call in this library: it
    fetches a universe, builds a panel, and fits once per walk-forward fold
    — times the search grid if one is set. A misspelled estimator parameter
    surfaced only after all of that. The estimator registry has always known
    the answer in microseconds; it just could not be asked.

    `estimated_fits` is the other half of the point. A spec that looks
    modest can imply thousands of fits once an inner search grid multiplies
    through every fold, and the difference between seconds and an afternoon
    is not visible anywhere in the spec itself.
    """
    from standard_quant_tools.modeling.estimators.registry import (
        ESTIMATOR_REGISTRY,
        allowed_params,
        validate_params,
    )

    spec = input_data.spec
    task = spec.task
    estimator = spec.estimator.type
    problems: List[SpecProblem] = []
    notes: List[str] = []
    allowed: List[str] = []

    if (task, estimator) not in ESTIMATOR_REGISTRY:
        available = sorted(name for t, name in ESTIMATOR_REGISTRY if t == task)
        problems.append(
            SpecProblem(
                where="estimator",
                problem=f"{estimator!r} is not registered for task {task!r}.",
                suggestion=f"Available for {task}: {available}",
            )
        )
    else:
        allowed = sorted(allowed_params(task, estimator))
        try:
            validate_params(task, estimator, spec.estimator.params)
        except ValidationError as exc:
            problems.append(
                SpecProblem(
                    where="estimator.params",
                    problem=str(exc),
                    suggestion=f"Accepted parameters: {allowed}",
                )
            )

    # How much work this implies.
    estimated_fits: Optional[int] = None
    folds = getattr(spec.validation, "n_splits", None)
    if folds:
        estimated_fits = int(folds)
        search = spec.search
        if search is not None:
            grid = getattr(search, "param_grid", None) or {}
            combinations = 1
            for values in grid.values():
                combinations *= max(1, len(values))
            inner = getattr(search, "inner_splits", 1) or 1
            estimated_fits = int(folds * (1 + combinations * inner))
            notes.append(
                f"A search grid of {combinations} combination(s) over "
                f"{inner} inner split(s) multiplies through {folds} fold(s). "
                "That is the difference between a quick experiment and a "
                "long one, and nothing in the spec shows it."
            )

    if input_data.dataset_id is not None:
        try:
            _panel, meta, _directory = _load_dataset_panel(input_data.dataset_id)
        except ValidationError as exc:
            problems.append(SpecProblem(where="dataset_id", problem=str(exc)))
        else:
            available = set(meta.get("feature_ids", []) or [])
            wanted = {
                f.output_name() if hasattr(f, "output_name") else str(f)
                for f in (meta.get("feature_ids", []) or [])
            }
            missing = sorted(w for w in wanted if w not in available)
            if missing:
                problems.append(
                    SpecProblem(
                        where="features",
                        problem=f"not present in dataset {input_data.dataset_id!r}: {missing}",
                    )
                )
            notes.append(
                f"Checked against dataset {input_data.dataset_id!r} "
                f"({meta.get('rows')} rows)."
            )

    if not problems:
        notes.append("Valid. Nothing was fetched, built or fitted.")

    return ValidateModelSpecResult(
        valid=not problems,
        task=task,
        estimator=estimator,
        problems=problems,
        allowed_estimator_params=allowed,
        estimated_fits=estimated_fits,
        notes=notes,
    )


def score_predictions(input_data: ScorePredictionsInput) -> ScorePredictionsResult:
    """
    Score a prediction frame against its realized outcome, from a reference.

    Works on anything published as `predictions`, including predictions this
    library never produced — which is the point. A model built elsewhere can
    be measured with the same yardstick as one built here, and the yardstick
    includes the two things a headline metric leaves out.

    THE BASELINE. The same metrics for predicting the training mean. A model
    that does not beat it has learned nothing, and an R2 that looks strong
    beside a baseline that also looks strong usually means the target was
    easy rather than the model clever.

    THE EFFECTIVE SAMPLE SIZE. A 20-day forward return sampled daily has far
    fewer independent observations than rows, so any t-statistic computed
    from the raw count is overstated — often by a factor of four or five.
    """
    import numpy as np

    from standard_quant_tools.agent.runtimes import handoff
    from standard_quant_tools.modeling.validation.metrics import (
        baseline_regression_metrics,
        classification_metrics,
        cross_sectional_ic,
        effective_sample_size,
        regression_metrics,
        summarize_cross_sectional_ic,
    )
    from standard_quant_tools.modeling.validation.ranking import ranking_metrics

    frame = handoff.resolve(input_data.predictions_ref, expect="predictions")
    for column in (input_data.target_column, input_data.prediction_column, "date"):
        if column not in frame.columns:
            raise ValidationError(
                f"the predictions frame has no {column!r} column; it holds "
                f"{list(frame.columns)}. Scoring needs the prediction, the "
                "realized outcome, and the date each pair belongs to."
            )

    frame = frame.dropna(
        subset=[input_data.target_column, input_data.prediction_column]
    )
    if frame.empty:
        raise ValidationError(
            "every row is missing either the prediction or the outcome, so "
            "there is nothing to score."
        )

    y_true = frame[input_data.target_column].to_numpy(dtype=float)
    y_pred = frame[input_data.prediction_column].to_numpy(dtype=float)
    dates = frame["date"].to_numpy()
    entities = frame["entity"].nunique() if "entity" in frame.columns else 1

    notes: List[str] = []
    if input_data.task == "regression":
        metrics = regression_metrics(y_true, y_pred, dates=dates)
        baseline = baseline_regression_metrics(y_true)
    elif input_data.task == "classification":
        metrics = classification_metrics(
            (y_true > 0).astype(int), (y_pred > 0.5).astype(int), y_pred
        )
        baseline = {}
        notes.append(
            "The outcome was binarized at zero and the prediction treated as "
            "a positive-class probability. If either is not what the column "
            "holds, these numbers are meaningless rather than merely wrong."
        )
    else:
        metrics = ranking_metrics(
            y_true, y_pred, dates, ks=tuple(input_data.ndcg_cutoffs)
        )
        baseline = {}

    ic_summary: Dict[str, float] = {}
    if entities > 1:
        ic = cross_sectional_ic(y_true, y_pred, dates, method=input_data.ic_method)
        ic_summary = summarize_cross_sectional_ic(ic, prefix="ic")
    else:
        notes.append(
            "One entity only, so there is no cross-section to correlate "
            "within — the IC block is empty by construction, not by failure."
        )

    beats = None
    if baseline and "r2" in metrics and "baseline_r2" in baseline:
        beats = bool(metrics["r2"] > baseline["baseline_r2"])
        if not beats:
            notes.append(
                "This does NOT beat predicting the mean. Whatever the "
                "headline metric says, the model has not learned anything "
                "the baseline did not already know."
            )
        if baseline.get("baseline_is_oracle"):
            notes.append(
                "The baseline used the SCORED set's own mean, not a "
                "training mean — an oracle it could not have known in "
                "advance. That makes it harder than a real baseline: "
                "beating it is strong evidence, and failing to beat it is "
                "weaker evidence than it looks."
            )

    ess = None
    horizon = 1
    if "label_end_date" in frame.columns:
        notes.append(
            "Horizon inferred as 1 bar; pass a target horizon through the "
            "dataset spec for a sharper effective sample size."
        )
    ess = float(effective_sample_size(len(frame), horizon, int(entities)))

    return ScorePredictionsResult(
        task=input_data.task,
        n_observations=int(len(frame)),
        n_dates=int(len(np.unique(dates))),
        n_entities=int(entities),
        metrics={k: float(v) for k, v in metrics.items()},
        cross_sectional_ic={k: float(v) for k, v in ic_summary.items()},
        baseline={k: float(v) for k, v in baseline.items()},
        beats_baseline=beats,
        effective_sample_size=ess,
        notes=notes,
    )


# ── Registration (mirrors agent.tools.get_agent_tools()/_TOOL_DISPATCH,
# but a separate registry — never merged into that one) ────────────────

_MODELING_TOOL_DEFS: List[tuple] = [
    (
        "validate_model_spec",
        "Check a ModelSpec before spending an experiment on it: that the "
        "estimator exists for the task, that its parameters are accepted, "
        "and how many fits the spec implies once a search grid multiplies "
        "through every fold. Fetches nothing and fits nothing.",
        ValidateModelSpecInput,
    ),
    (
        "score_predictions",
        "Score a predictions reference against its realized outcome — "
        "accuracy metrics, cross-sectional IC and ICIR, a predict-the-mean "
        "baseline, and an effective sample size adjusted for overlapping "
        "forward returns. Works on predictions this library never produced.",
        ScorePredictionsInput,
    ),
    ("list_features", "Feature catalog for the modeling runtime.", ListFeaturesInput),
    (
        "build_model_dataset",
        "Fetch OHLCV, compute requested features/target, persist the panel.",
        BuildModelDatasetInput,
    ),
    (
        "run_model_experiment",
        "Fit + walk-forward validate + register a model from a persisted dataset.",
        RunModelExperimentInput,
    ),
    (
        "score_model",
        "Score a registered model for a universe as of a date.",
        ScoreModelInput,
    ),
    (
        "inspect_model",
        "Inspect a registered model's summary/importance/validation/lineage.",
        InspectModelInput,
    ),
    (
        "list_modeling_capabilities",
        "What this modeling runtime can do: tasks, estimators and the "
        "capabilities of each (sample weights, probabilities, query groups, "
        "coefficients, feature importance), features, target types, "
        "validation schemes, preprocessing and weighting options, and which "
        "optional libraries are installed. Call this before choosing a model "
        "rather than assuming an estimator is available.",
        ListModelingCapabilitiesInput,
    ),
    (
        "analyze_features",
        "Score a built dataset's FEATURES before fitting anything: coverage, "
        "turnover, cross-sectional IC and ICIR, decile spread and monotonicity, "
        "which features are near-duplicates of one another, and a lead-lag "
        "causality screen for features whose information arrives too early. "
        "Use this to choose features; use inspect_model(view='feature_importance') "
        "to see what one fitted model then leaned on.",
        AnalyzeFeaturesInput,
    ),
    (
        "list_models",
        "Every registered model, newest first, with its task, estimator, "
        "headline out-of-sample metric and source dataset. Call this when "
        "you need a model_id you do not already hold.",
        ListModelsInput,
    ),
    (
        "list_datasets",
        "Every built dataset panel, newest first, with row/entity/feature "
        "counts and date span.",
        ListDatasetsInput,
    ),
    (
        "compare_models",
        "Rank registered models side by side on their out-of-sample "
        "metrics. Models are ranked within their own task, never across "
        "tasks, because those metrics are not on a common scale.",
        CompareModelsInput,
    ),
    (
        "check_leakage",
        "Ask whether a set of features is temporally safe to fit on — "
        "before building a dataset with them. Optionally reports a built "
        "dataset's recorded point-in-time coverage too.",
        CheckLeakageInput,
    ),
    (
        "evaluate_model_portfolio",
        "Evaluate a model's out-of-sample predictions as a shared-cash "
        "portfolio: transform predictions into target weights and simulate "
        "them with costs, returning Sharpe, drawdown, turnover and exposure.",
        EvaluateModelPortfolioInput,
    ),
]


def list_modeling_capabilities(
    input_data: ListModelingCapabilitiesInput,
) -> ListModelingCapabilitiesResult:
    """
    What this runtime can currently do: tasks, estimators and what each one
    supports, features, targets, validation schemes, preprocessing,
    weighting, and which optional libraries are installed.

    One operation rather than one tool per model. Everything is read off the
    live registries and the model adapters, so a newly registered estimator
    describes itself correctly without anyone updating a table.

    The part an agent most needs and cannot infer is `optional_dependencies`:
    lightgbm and xgboost are not declared dependencies of this package, so
    the ranking task and the fast boosters exist on one machine and not
    another. An estimator list that is silently shorter is much harder to act
    on than an explicit absence.
    """
    capabilities = modeling_capabilities()
    if not input_data.include_estimators:
        capabilities = {
            key: value for key, value in capabilities.items() if key != "estimators"
        }
    return ListModelingCapabilitiesResult(capabilities=capabilities)


def analyze_features(input_data: AnalyzeFeaturesInput) -> AnalyzeFeaturesResult:
    """
    Score the FEATURES of a built dataset, before any model is fitted.

    `inspect_model(view="feature_importance")` answers "which columns did
    this estimator lean on" — a statement about one fit, in units that
    differ per estimator, and only available after the features have already
    been chosen. This answers the earlier and more useful question: is this
    a good feature at all.

    Four things come back per feature — how well populated and how fast it
    turns over, its cross-sectional IC and the decile shape behind it, which
    other features are restatements of it, and whether its information is
    actually available when it claims to be.
    """
    logger.debug("[analyze_features] dataset_id=%s", input_data.dataset_id)
    panel, meta, _ = _load_dataset_panel(input_data.dataset_id)
    features = input_data.features or list(meta.get("feature_ids", []))
    if not features:
        raise ValidationError(
            f"dataset {input_data.dataset_id!r} records no feature_ids, and none "
            "were supplied. Pass `features` explicitly."
        )

    report = build_feature_report(
        panel,
        features,
        n_quantiles=input_data.n_quantiles,
        cluster_threshold=input_data.cluster_threshold,
        leakage_max_shift=input_data.leakage_max_shift,
        include_leakage=input_data.include_leakage,
    )
    # Lifted out of the nested dict so an agent reading only the top level
    # still sees them. They are the part of this result most likely to
    # change what it does next.
    warnings = list(report.pop("warnings", []))
    return AnalyzeFeaturesResult(
        dataset_id=input_data.dataset_id, report=report, warnings=warnings
    )


MODELING_TOOL_DISPATCH = {
    "validate_model_spec": (validate_model_spec, ValidateModelSpecInput),
    "score_predictions": (score_predictions, ScorePredictionsInput),
    "list_features": (list_features, ListFeaturesInput),
    "build_model_dataset": (build_model_dataset, BuildModelDatasetInput),
    "run_model_experiment": (run_model_experiment, RunModelExperimentInput),
    "score_model": (score_model, ScoreModelInput),
    "inspect_model": (inspect_model, InspectModelInput),
    "evaluate_model_portfolio": (
        evaluate_model_portfolio,
        EvaluateModelPortfolioInput,
    ),
    "list_models": (list_models, ListModelsInput),
    "list_datasets": (list_datasets, ListDatasetsInput),
    "compare_models": (compare_models, CompareModelsInput),
    "check_leakage": (check_leakage, CheckLeakageInput),
    "analyze_features": (analyze_features, AnalyzeFeaturesInput),
    "list_modeling_capabilities": (
        list_modeling_capabilities,
        ListModelingCapabilitiesInput,
    ),
}


def get_modeling_tools() -> List[Dict[str, Any]]:
    """Tool definitions for the modeling runtime, in the exact same
    OpenAI-style {"type": "function", "function": {...}} envelope
    agent.tools.get_agent_tools() returns — a separate 6-entry list,
    never merged into that 46-entry one, but shaped identically so the
    same LLM client code can consume either registry."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_model.model_json_schema(),
            },
        }
        for name, description, input_model in _MODELING_TOOL_DEFS
    ]
