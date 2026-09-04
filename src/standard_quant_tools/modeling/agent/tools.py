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
from ..specs import DatasetSpec, FeatureSpec, TargetSpec
from .dataset_tools import (  # noqa: F401
    ExplainRowLossInput,
    explain_dataset_row_loss,
)
from .models import (
    AnalyzeFeaturesInput,
    AnalyzeFeaturesResult,
    AnalyzeModelErrorsInput,
    AnalyzeModelErrorsResult,
    BuildEnsembleInput,
    BuildEnsembleResult,
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
    JoinPointInTimeInput,
    JoinPointInTimeResult,
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
    PitRecordsInput,
    PitValidationResult,
    RegisterExternalPanelInput,
    RegisterExternalPanelResult,
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
            "targets": built.get("targets", []),
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


def build_model_ensemble(input_data: BuildEnsembleInput) -> BuildEnsembleResult:
    """Combine registered models' out-of-sample predictions into one series."""
    from ..ensemble import combine_predictions

    combined = combine_predictions(
        input_data.model_ids,
        method=input_data.method,
        weights=input_data.weights,
    )
    ref = publish(
        combined["predictions"],
        kind="predictions",
        run_id=input_data.run_id,
        name=input_data.name,
        producer="build_model_ensemble",
    )
    warnings = list(combined["warnings"])
    hottest = max(combined["correlations"].values(), default=None)
    if hottest is not None and hottest > 0.95:
        pair = max(combined["correlations"], key=combined["correlations"].get)
        warnings.append(
            f"WARNING: {pair} are correlated at {hottest:.3f}. An ensemble of "
            "models that agree is approximately either of them, and the "
            "diversification a combination is supposed to buy is not there -- "
            "which the ensemble's own score will not show you."
        )
    return BuildEnsembleResult(
        ref=ref,
        model_ids=combined["model_ids"],
        method=combined["method"],
        task=combined["task"],
        n_rows=combined["n_rows"],
        rows_per_model=combined["rows_per_model"],
        rows_covered_by_all=combined["rows_covered_by_all"],
        correlations=combined["correlations"],
        correlation_basis=combined["correlation_basis"],
        warnings=warnings,
    )


def register_external_panel(
    input_data: RegisterExternalPanelInput,
) -> RegisterExternalPanelResult:
    """Register a feature matrix computed elsewhere as a modeling dataset."""
    from ..dataset.external_panel import load_external_panel

    if input_data.targets:
        declared = [
            {
                "name": target.name,
                "column": target.column,
                "horizon": target.horizon,
                "target_type": target.target_type,
                "label_end_column": target.label_end_column,
            }
            for target in input_data.targets
        ]
    else:
        declared = [
            {
                "name": "primary",
                "column": input_data.target_column,
                "horizon": input_data.horizon,
                "target_type": input_data.target_type,
                "label_end_column": input_data.label_end_column,
            }
        ]

    loaded = load_external_panel(
        input_data.path,
        date_column=input_data.date_column,
        entity_column=input_data.entity_column,
        targets=declared,
        feature_columns=input_data.feature_columns,
        fmt=input_data.file_format,
    )
    primary = declared[0]
    panel = loaded["panel"]
    handle = loaded["handle"]

    # A real DatasetSpec, synthesized from what the panel actually holds.
    # Not decoration: run_model_experiment verifies its hash, bundles it
    # into the registered model, and reads `universe`/`interval` into the
    # lineage. Writing a placeholder would put a false claim in all three.
    spec = DatasetSpec(
        universe=loaded["entities"],
        start=loaded["start"],
        end=loaded["end"],
        features=[FeatureSpec(id=column) for column in loaded["feature_ids"]],
        target=TargetSpec(type=primary["target_type"], horizon=primary["horizon"]),
        provider="external",
        interval=input_data.interval,
    )

    dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
    directory = _artifacts.run_dir(dataset_id)
    directory.mkdir(parents=True, exist_ok=True)
    _artifacts.save_json(directory, "dataset_spec", spec.model_dump())

    warnings = list(loaded["warnings"])
    warnings.append(
        "REGISTERED BY REFERENCE. No copy of this panel was written -- the "
        f"dataset reads {handle.path} on every load. Its CONTENT hash is "
        "recorded and verified each time, so an edit fails loudly; but if "
        "the file is moved or deleted the dataset stops loading."
    )
    warnings.append(
        "score_model cannot run on a model trained from this dataset. "
        "Scoring rebuilds features from the bundled spec, and these "
        "features were computed outside this library, so there is nothing "
        "to rebuild them from. Score by registering a new panel for the "
        "scoring window and calling score_predictions."
    )

    if len(declared) > 1:
        warnings.append(
            f"{len(declared)} labels registered: "
            + ", ".join(f"{d['name']}(h={d['horizon']})" for d in declared)
            + f". run_model_experiment trains on {primary['name']!r} unless "
            "given `target`. Every model then sees the same rows and the "
            "same folds, which is what makes the horizons comparable."
        )

    _artifacts.save_json(
        directory,
        "dataset_meta",
        {
            "feature_ids": loaded["feature_ids"],
            "target_id": f"{primary['target_type']}:{primary['horizon']}",
            "targets": declared,
            "data_hash": hash_dataframe(panel),
            "spec_hash": dataset_spec_hash(spec),
            "entities": loaded["entities"],
            "provider": "external",
            "interval": input_data.interval,
            "warnings": warnings,
            # Nothing was aligned here -- the panel arrived aligned -- so
            # there is no per-column row loss to attribute. An empty map is
            # the honest answer; explain_dataset_row_loss reports it as
            # such rather than implying no rows were ever lost upstream.
            "drop_attribution": {},
            "entities_fetched": loaded["entities"],
            # How to read the panel again. Stored as the ORIGINAL column
            # names rather than the rename map, so reloading takes the same
            # code path as registering did and cannot drift from it.
            "storage": "external",
            "panel_path": str(handle.path),
            "panel_format": handle.fmt,
            "panel_fingerprint": handle.fingerprint,
            "panel_columns": {
                "date_column": input_data.date_column,
                "entity_column": input_data.entity_column,
                "target_column": input_data.target_column,
                "label_end_column": input_data.label_end_column,
                "feature_columns": loaded["feature_ids"],
            },
            "source": input_data.source,
        },
    )

    return RegisterExternalPanelResult(
        dataset_id=dataset_id,
        rows=int(len(panel)),
        entities=loaded["entities"],
        feature_ids=loaded["feature_ids"],
        target_id=f"{primary['target_type']}:{primary['horizon']}",
        targets=[str(d["name"]) for d in declared],
        start=loaded["start"],
        end=loaded["end"],
        interval=input_data.interval,
        source_path=str(handle.path),
        fingerprint=handle.fingerprint,
        warnings=warnings,
    )


def _select_target(panel, meta, requested, dataset_id: str):
    """
    Point the panel's `target` at the label this experiment asked for.

    A panel registered with several horizons carries them all as
    `target__<name>`, with the primary duplicated onto plain `target` so
    every consumer that has only ever seen one keeps working. Selecting is
    therefore a rename plus a drop, and the ENGINE is unchanged -- it reads
    `target` and `label_end_date` exactly as it always has.

    Rows are dropped by the CHOSEN label. A 30-second horizon has more
    unclosed rows at the end of a sample than a 1-second one, and dropping
    on the union would make every short-horizon model pay for the longest
    one's warm-down.
    """
    from ..dataset.external_panel import label_end_column_for, target_column_for

    declared = meta.get("targets") or []
    names = [str(d["name"]) for d in declared]
    notes = []

    if requested is None:
        if len(names) > 1:
            notes.append(
                f"NOTE: this dataset carries {len(names)} labels {names}; "
                f"trained on the primary, {names[0]!r}. Pass `target` to fit "
                "another."
            )
        return panel, meta["target_id"], notes

    if not declared:
        raise ValidationError(
            f"target={requested!r} was asked for, but dataset "
            f"{dataset_id!r} carries a single unnamed label. Only a dataset "
            "registered with `targets` can be selected from."
        )
    if requested not in names:
        raise ValidationError(
            f"dataset {dataset_id!r} has no label named {requested!r}. It "
            f"carries {names}."
        )

    chosen = declared[names.index(requested)]
    column = target_column_for(requested)
    if column not in panel.columns:
        raise ValidationError(
            f"dataset {dataset_id!r} records a label {requested!r} that its "
            f"panel does not contain ({column!r} is absent). The file behind "
            "it has changed shape since registration."
        )

    out = panel.copy()
    out["target"] = out[column]
    ends = label_end_column_for(requested)
    if ends in out.columns:
        out["label_end_date"] = out[ends]
    elif "label_end_date" in out.columns:
        # The primary's label-end would otherwise be applied to a different
        # horizon's rows, which is a purge computed against the wrong window.
        out = out.drop(columns=["label_end_date"])

    before = len(out)
    out = out[out["target"].notna()]
    dropped = before - len(out)
    if dropped:
        notes.append(
            f"NOTE: {dropped:,} of {before:,} rows have no {requested!r} "
            "label and were dropped for THIS experiment only. A longer "
            "horizon has more unclosed rows at the end of a sample; dropping "
            "on the union would make every shorter horizon pay for it."
        )
    if out.empty:
        raise ValidationError(
            f"every row's {requested!r} label is null, so there is nothing "
            "to fit. Check the horizon against the panel's own span."
        )
    return (
        out.reset_index(drop=True),
        f"{chosen.get('target_type', 'forward_return')}:{chosen['horizon']}",
        notes,
    )


def _load_external_panel_for(meta, dataset_id: str):
    """Re-read a referenced panel through the path that registered it."""
    from ..dataset.external_panel import load_external_panel

    columns = meta.get("panel_columns") or {}
    path = meta.get("panel_path")
    if not path:
        raise ValidationError(
            f"dataset {dataset_id!r} is recorded as external but names no "
            "panel_path. The registration is unusable; register the panel "
            "again."
        )
    try:
        return load_external_panel(
            str(path),
            date_column=columns.get("date_column", "date"),
            entity_column=columns.get("entity_column", "entity"),
            targets=meta.get("targets"),
            target_column=columns.get("target_column", "target"),
            label_end_column=columns.get("label_end_column"),
            feature_columns=columns.get("feature_columns"),
            fmt=meta.get("panel_format"),
        )["panel"]
    except ValidationError as exc:
        raise ValidationError(
            f"dataset {dataset_id!r} points at {path}, which cannot be read "
            f"now -- {exc} Nothing was copied when it was registered, so an "
            "externally referenced dataset is only as good as the file it "
            "names."
        ) from exc


def _load_dataset_meta(dataset_id: str):
    """
    A dataset's recorded metadata, WITHOUT reading or hashing its panel.

    `_load_dataset_panel` verifies the panel against the hash recorded at
    build time, which costs 0.5 s per million rows at 24 columns and 1.7 s
    at 84 -- three to six times what reading the Parquet costs. That is the
    right price for anything that USES the data. It is the wrong price for
    a caller that wants four keys out of a JSON file, which is what
    `validate_model_spec` was paying: it read and hashed the whole panel,
    bound it to `_panel`, and discarded it.

    Deliberately NOT used by `check_leakage`, whose own note tells the
    caller the coverage figures "confirm the panel is the one that was
    built". That sentence is only true because the hash was checked, so
    that tool keeps paying for it.
    """
    directory = _artifacts.run_dir(dataset_id)
    meta_path = directory / "dataset_meta.json"
    if not meta_path.exists():
        raise ValidationError(
            f"no dataset with dataset_id={dataset_id!r} — "
            "dataset_meta.json is written last, so its absence also means a "
            "previous build_model_dataset call did not complete."
        )
    return _artifacts.load_json(str(meta_path)), directory


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
    meta = _artifacts.load_json(str(meta_path))
    # An externally registered panel has no panel.parquet -- the whole point
    # is that nothing was copied. The hash check below is UNCHANGED and runs
    # on both, because it hashes the loaded frame rather than the file: the
    # engine reads the panel whole either way, so integrity costs one pass
    # over data already in memory.
    if meta.get("storage") == "external":
        panel = _load_external_panel_for(meta, dataset_id)
    else:
        panel = _artifacts.load_artifact(str(directory / "panel.parquet"))

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
    panel, selected_target_id, selection_notes = _select_target(
        panel, meta, input_data.target, input_data.dataset_id
    )

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
                "from the data it learned on — rebuild the dataset instead. "
                "An UPGRADE can also cause this without anything being edited: "
                "the hash covers every field of the spec, so a release that "
                "adds one (TargetSpec.horizons did) changes it for every "
                "dataset persisted before that release. Rebuilding is the "
                "same remedy either way."
            )

    dataset = {
        "panel": panel,
        "feature_ids": meta["feature_ids"],
        "target_id": selected_target_id,
        "data_hash": meta["data_hash"],
        "spec_hash": stored_spec_hash,
        # Bundled into the model so it becomes self-contained -- see
        # registry.model_registry.save_model.
        "dataset_spec": spec_dict,
        # .get, not [...]: datasets built before coverage diagnostics
        # existed have no such key, and a missing warning list is not the
        # same claim as an empty one -- see ModelManifest.dataset_warnings.
        "warnings": list(meta.get("warnings", [])) + selection_notes,
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
            # Metadata only: this branch reads `feature_ids` and nothing
            # else, so it does not need the panel loaded or verified.
            meta, _directory = _load_dataset_meta(input_data.dataset_id)
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


def _label_name_for_target_id(meta, target_id: str):
    """
    Which declared label a registered model was actually fit on.

    A manifest records `target_id` -- "forward_return:5" -- and a panel
    registered with several horizons carries its labels by NAME. The two are
    joined by reconstructing the id from each declaration, which is exact
    when the declarations differ and ambiguous when two of them describe the
    same type and horizon. Ambiguous returns None rather than picking the
    first: the wrong label produces residuals against the wrong outcome, and
    every number downstream would be confidently wrong.
    """
    declared = meta.get("targets") or []
    matches = [
        str(d["name"])
        for d in declared
        if f"{d.get('target_type', 'forward_return')}:{d['horizon']}" == target_id
    ]
    return matches[0] if len(matches) == 1 else None


def _trim(rows, top_n: int):
    """Worst and best buckets by RMSE, with the middle counted not listed."""
    if len(rows) <= 2 * top_n:
        return sorted(rows, key=lambda r: -r["rmse"]), 0
    ordered = sorted(rows, key=lambda r: -r["rmse"])
    return ordered[:top_n] + ordered[-top_n:], len(ordered) - 2 * top_n


def analyze_model_errors(
    input_data: AnalyzeModelErrorsInput,
) -> AnalyzeModelErrorsResult:
    """Residuals, calibration and error attribution for a registered model.

    The persisted out-of-sample frame carries date, entity and prediction
    and NO outcome, so the actuals are joined back from the dataset panel
    the model was fit on -- which is also what makes a breakdown by feature
    decile possible, since the panel is where the features live."""
    import numpy as np
    import pandas as pd

    from .. import diagnostics as _diag
    from ..ensemble import load_oos_predictions

    manifest = load_manifest(input_data.model_id)
    predictions = load_oos_predictions(input_data.model_id)
    panel, meta, _directory = _load_dataset_panel(manifest.dataset_id)

    warnings: List[str] = []
    label = _label_name_for_target_id(meta, manifest.target_id)
    if label is not None:
        panel, _target_id, notes = _select_target(
            panel, meta, label, manifest.dataset_id
        )
        warnings.extend(notes)
    elif (meta.get("targets") or []) and "target" not in panel.columns:
        raise ValidationError(
            f"model {input_data.model_id!r} was fit on target_id="
            f"{manifest.target_id!r}, and dataset {manifest.dataset_id!r} "
            "declares no single label matching it. Residuals cannot be "
            "computed against a label that cannot be identified -- the wrong "
            "one would produce numbers that look fine and describe a "
            "different outcome."
        )

    if "target" not in panel.columns:
        raise ValidationError(
            f"dataset {manifest.dataset_id!r} has no 'target' column, so "
            "there are no outcomes to compare this model's predictions "
            "against."
        )

    actuals = panel[["date", "entity", "target"]].copy()
    actuals["date"] = pd.to_datetime(actuals["date"])
    actuals["entity"] = actuals["entity"].astype(str)
    features = [
        c for c in panel.columns if c not in {"date", "entity", "label_end_date"}
    ]
    if input_data.feature is not None:
        if input_data.feature not in panel.columns:
            raise ValidationError(
                f"feature={input_data.feature!r} is not a column of dataset "
                f"{manifest.dataset_id!r}. Its panel carries {features[:15]}"
                f"{' ...' if len(features) > 15 else ''}."
            )
        actuals[input_data.feature] = panel[input_data.feature].to_numpy()

    joined = predictions.merge(actuals, on=["date", "entity"], how="inner")
    if joined.empty:
        raise ValidationError(
            f"none of this model's {len(predictions):,} out-of-sample rows "
            f"match a row in dataset {manifest.dataset_id!r}. The predictions "
            "and the panel do not describe the same (date, entity) universe."
        )
    if len(joined) < len(predictions):
        warnings.append(
            f"NOTE: {len(predictions) - len(joined):,} of {len(predictions):,} "
            "predicted rows found no outcome in the panel and are excluded. "
            "Rows at the end of the sample have an unclosed label."
        )

    joined["_predicted"] = pd.to_numeric(joined["prediction"], errors="coerce")
    joined["_actual"] = pd.to_numeric(joined["target"], errors="coerce")
    joined["_residual"] = joined["_actual"] - joined["_predicted"]

    actual = joined["_actual"].to_numpy(dtype="float64")
    predicted = joined["_predicted"].to_numpy(dtype="float64")
    residuals = _diag.residual_summary(actual, predicted)
    calibration = _diag.calibration(actual, predicted, manifest.task)
    attribution = _diag.error_attribution(
        joined, feature=input_data.feature, period=input_data.period
    )

    if attribution.get("feature_note"):
        warnings.append(attribution["feature_note"])
    if attribution.get("rows_without_feature"):
        missing = attribution["rows_without_feature"]
        warnings.append(
            f"NOTE: {missing:,} row(s) have no value for "
            f"{input_data.feature!r} and are absent from the feature "
            "breakdown -- a rolling feature has a warm-up window. They are "
            "still counted in every other section, so the feature deciles "
            "cover fewer rows than n_rows."
        )

    findings = _diag.worst_buckets(attribution)

    # Bias, qualified by how uncertain the mean itself is. The t-statistic
    # below OVERSTATES significance whenever the label overlaps -- the
    # autocorrelation reported alongside is the measure of by how much --
    # so the threshold is deliberately blunt rather than a p-value.
    n, std = residuals["n"], residuals["std_error"]
    if n > 30 and std and np.isfinite(std) and std > 0:
        t_stat = residuals["mean_error"] / (std / np.sqrt(n))
        if abs(t_stat) > 3:
            direction = "LOW" if residuals["mean_error"] > 0 else "HIGH"
            findings.append(
                f"the model reads systematically {direction}: mean error "
                f"{residuals['mean_error']:.6g} over {n:,} rows (t={t_stat:.1f}). "
                "This is bias, and no amount of rank skill corrects it -- a "
                "constant offset would."
            )

    slope = calibration.get("slope")
    if slope is not None and np.isfinite(slope):
        if slope < 0:
            findings.append(
                f"calibration slope {slope:.3f} is NEGATIVE: out of sample "
                "the predictions point the wrong way. Trading them inverted "
                "is not the fix -- a sign that flips out of sample usually "
                "means the fit found something that was not there."
            )
        elif not 0.5 <= slope <= 2.0:
            findings.append(
                f"calibration slope {slope:.3f}, against 1.0 for a calibrated "
                "model. "
                + (
                    "The predictions are spread wider than the outcomes, so "
                    "anything sized directly from them over-trades."
                    if slope < 1
                    else "The predictions are compressed relative to the "
                    "outcomes, so sizing from them under-trades."
                )
            )

    if calibration.get("note"):
        warnings.append("NOTE: " + calibration["note"])

    ece = calibration.get("expected_calibration_error")
    if ece is not None and ece > 0.1:
        findings.append(
            f"expected calibration error {ece:.3f}: the stated probabilities "
            "are not the observed frequencies, so any threshold applied to "
            "them fires somewhere other than where you set it."
        )

    autocorrelation = _diag.residual_autocorrelation(joined)
    if autocorrelation is not None and autocorrelation > 0.3:
        warnings.append(
            f"NOTE: residual autocorrelation is {autocorrelation:.3f}. For an "
            "overlapping label this is EXPECTED and not a defect -- an "
            "h-bar forward return sampled every bar shares h-1 bars with its "
            "neighbour. It does mean the effective sample is smaller than "
            f"{n:,} rows, so read every t-statistic here as optimistic."
        )

    trimmed = {}
    omitted = {}
    for key in (
        "by_entity",
        "by_period",
        "by_prediction_decile",
        "by_feature_decile",
    ):
        rows, dropped = _trim(attribution.get(key, []), input_data.top_n)
        trimmed[key] = rows
        if dropped:
            omitted[key] = dropped

    thin = sum(1 for rows in trimmed.values() for r in rows if r.get("thin"))
    if thin:
        warnings.append(
            f"NOTE: {thin} listed bucket(s) hold fewer than "
            f"{_diag.MIN_BUCKET_ROWS} rows and are marked `thin`. They are "
            "excluded from the findings above, because the worst bucket of a "
            "breakdown is almost always the emptiest one."
        )

    return AnalyzeModelErrorsResult(
        model_id=input_data.model_id,
        task=manifest.task,
        target_id=manifest.target_id,
        n_rows=int(len(joined)),
        residuals=residuals,
        calibration=calibration,
        heteroskedasticity=_diag.heteroskedasticity(actual, predicted),
        residual_autocorrelation=autocorrelation,
        buckets_omitted=omitted,
        findings=findings,
        warnings=warnings,
        **trimmed,
    )


_MODELING_TOOL_DEFS: List[tuple] = [
    (
        "analyze_model_errors",
        "WHERE a registered model is wrong, not merely how wrong on average. "
        "An R2 cannot separate a broadly mediocre model from one that is "
        "excellent except in the conditions you trade -- those have the same "
        "headline number and opposite consequences. Breaks the out-of-sample "
        "errors down by entity, by period, by the model's OWN prediction "
        "decile, and optionally by the decile of any feature in its panel, "
        "which is the breakdown that turns 'the model is mediocre' into 'the "
        "model fails when the spread is wide'. Also reports CALIBRATION, a "
        "separate question from accuracy: a model can rank perfectly and "
        "still be systematically too confident, which is invisible in an R2 "
        "or an IC and changes every position size computed from it. "
        "Residual autocorrelation is reported and is EXPECTED to be positive "
        "for an overlapping label rather than being a defect.",
        AnalyzeModelErrorsInput,
    ),
    (
        "explain_dataset_row_loss",
        'Which column cost which training rows, and which are free to drop. Reports n_missing beside n_sole_missing, and the second is the actionable one: a 252-day feature sitting behind a 500-day one has n_missing in the hundreds of thousands and n_sole_missing of zero, so removing it gives back nothing. Reading only the first number produces a decision that feels informed and changes nothing, which is why "you lost 44% of the data" is not an answer.',
        ExplainRowLossInput,
    ),
    (
        "validate_pit_records",
        "Check point-in-time records BEFORE joining them onto anything. The error worth catching is the two timestamps the wrong way round: event_time is when a fact is ABOUT, available_time is when it could first be ACTED ON, and swapped they make every model look prescient. Also reports median_publication_lag_days -- exactly how much hindsight a naive join on event_time would have handed you. Fetches nothing.",
        PitRecordsInput,
    ),
    (
        "join_point_in_time",
        "Attach point-in-time records to a built dataset, each panel row getting the most recent record AVAILABLE by then. Strictly backward and inclusive: a filing released before a bar's close is usable on it, one released after is not, and a row with nothing available yet gets NaN rather than zero or the eventual value. A restatement is a second row with the same event_time and a later available_time, and the join returns whichever version was current at each date.",
        JoinPointInTimeInput,
    ),
    (
        "build_model_ensemble",
        "Combine several registered models into one prediction series, and "
        "publish it as an `sqt://predictions` reference that score_predictions "
        "and the backtest bridge read like any other. What gets combined is "
        "each model's OUT-OF-SAMPLE predictions -- rows predicted by a fold "
        "that did not train on them -- so the combination cannot inherit the "
        "optimism that makes naive stacking look excellent until it meets a "
        "new day. The default is rank_mean rather than mean, because two "
        "models on different scales average into a number dominated by "
        "whichever has the wider spread, which is its units and not its "
        "skill. Reports the pairwise correlation between the base models: "
        "two agreeing at 0.98 combine into approximately either of them, and "
        "the ensemble's own score cannot show you that.",
        BuildEnsembleInput,
    ),
    (
        "register_external_panel",
        "Register a feature matrix computed OUTSIDE this library -- by a C++ "
        "pipeline over an L2 feed, a warehouse query, another system -- as a "
        "modeling dataset, without copying it. Use this when the features "
        "already exist and build_model_dataset has nothing to fetch or "
        "compute. `horizon` is required and is the one thing not inferable "
        "from the file: the engine purges training rows whose label window "
        "overlaps the test fold, and a missing horizon disables that purge "
        "silently rather than failing. The panel's content hash is recorded "
        "and verified on every load, so an edited file fails loudly; a moved "
        "one stops loading. score_model cannot run on a model trained this "
        "way, because rebuilding features needs definitions this library "
        "does not have.",
        RegisterExternalPanelInput,
    ),
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
    (
        "list_features",
        "Which features this library can build, what each one measures, and "
        "what it costs to compute. Call it BEFORE build_model_dataset rather "
        "than guessing names -- a feature name that does not exist is a "
        "failed call and an error round trip, which costs more than reading "
        "the catalogue does. The feature_lab runtime then answers what each "
        "one is worth once the dataset exists.",
        ListFeaturesInput,
    ),
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
        "Run a registered model forward and get its predictions for a "
        "universe as of a date. The step that turns a fitted model into "
        "something a backtest can consume, and the one where point-in-time "
        "discipline matters most: the `as_of` date is what stops the model "
        "seeing features that did not exist yet. Raw probabilities from a "
        "tree ensemble are NOT calibrated, so a 0.9 threshold may select no "
        "rows at all -- check the distribution before thresholding.",
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


def _json_safe(value):
    """
    Recursively replace non-finite floats with None.

    `analyze_features` returns the report dict verbatim, and that dict
    carries NaN wherever a cross-sectional statistic was undefined -- a
    single entity per date, a constant feature. `json.dumps` writes those as
    a bare `NaN` token, which is not valid JSON per RFC 8259 and is rejected
    by strict parsers, including JSON-RPC clients. Measured on a legal
    panel: twelve of them in one report.

    `null` is both valid and more truthful. A consumer that reads 0.0 for a
    monotonicity that was never calculable concludes "no relationship" when
    the answer is "no measurement".

    Done here, at the tool boundary, rather than in `feature_report.py`:
    NaN is the right in-memory representation for a numpy pipeline, and
    Python callers of `build_feature_report` should keep getting it. It is
    only wrong once it has to cross a wire.
    """
    import math

    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
        dataset_id=input_data.dataset_id,
        report=_json_safe(report),
        warnings=warnings,
    )


def _pit_frame(records, entity_scoped: bool):
    """Records -> a validated PIT frame, with the caller's error surfaced."""
    import pandas as pd

    from standard_quant_tools.modeling.dataset.point_in_time import (
        validate_pit_frame,
    )

    frame = pd.DataFrame(records)
    return validate_pit_frame(frame, name="records", require_entity=entity_scoped)


def validate_pit_records(input_data: PitRecordsInput) -> PitValidationResult:
    """
    Check point-in-time records BEFORE joining them onto anything.

    The one error worth catching here is the two timestamps the wrong way
    round. `event_time` is when a fact is about; `available_time` is when it
    could first be acted on. Swapped, every value arrives weeks before it
    existed, and the model built on it looks prescient rather than wrong —
    which is why this is checked rather than assumed.

    `median_publication_lag_days` is the number to read even when everything
    passes: it is exactly how much hindsight a naive join on `event_time`
    would have handed you. Three weeks for quarterly filings, and it does
    not announce itself anywhere else.
    """
    import numpy as np
    import pandas as pd

    from standard_quant_tools.error import ValidationError as _VE
    from standard_quant_tools.modeling.dataset.point_in_time import (
        AVAILABLE_TIME,
        ENTITY,
        EVENT_TIME,
    )

    logger.debug("[validate_pit_records] n=%d", len(input_data.records))
    try:
        frame = _pit_frame(input_data.records, input_data.entity_scoped)
    except _VE as exc:
        return PitValidationResult(
            valid=False, n_records=len(input_data.records), problem=str(exc)
        )

    reserved = {EVENT_TIME, AVAILABLE_TIME, ENTITY}
    fields = sorted(c for c in frame.columns if c not in reserved)
    lag = (frame[AVAILABLE_TIME] - frame[EVENT_TIME]).dt.total_seconds() / 86400.0

    keys = [EVENT_TIME] + ([ENTITY] if input_data.entity_scoped else [])
    versions = frame.groupby(keys)[AVAILABLE_TIME].nunique()
    versioned = bool((versions > 1).any())

    warnings = []
    if not versioned:
        warnings.append(
            "every fact appears once, so these records cannot show whether a "
            "restated value would be kept as a new row or overwritten. That "
            "is not a problem with the data -- it is a limit on what can be "
            "concluded from it."
        )
    if not fields:
        warnings.append(
            "the records carry no value columns, only timestamps. A join "
            "would add nothing."
        )
    zero_lag = int((lag <= 0).sum())
    if zero_lag:
        warnings.append(
            f"{zero_lag} record(s) were available at the instant they "
            "describe. That is right for a market bar and wrong for anything "
            "reported -- check these are not event_time copied across."
        )

    return PitValidationResult(
        valid=True,
        n_records=int(len(frame)),
        n_entities=(int(frame[ENTITY].nunique()) if input_data.entity_scoped else None),
        fields=fields,
        event_time_range=[
            str(frame[EVENT_TIME].min().date()),
            str(frame[EVENT_TIME].max().date()),
        ],
        available_time_range=[
            str(frame[AVAILABLE_TIME].min().date()),
            str(frame[AVAILABLE_TIME].max().date()),
        ],
        revisions="versioned" if versioned else "unknown",
        reproduces_history=versioned,
        median_publication_lag_days=(float(np.median(lag)) if len(lag) else None),
        warnings=warnings,
    )


def join_point_in_time(input_data: JoinPointInTimeInput) -> JoinPointInTimeResult:
    """
    Attach point-in-time records to a built dataset, each panel row getting
    the most recent record that was AVAILABLE by then.

    The join is strictly backward and inclusive: a filing released before a
    bar's close is usable on that bar, and one released after it is not. A
    row with nothing available yet gets NaN, which is the honest answer for
    "nobody knew this yet" — not zero, and not the eventual value.

    A restatement is a SECOND ROW with the same `event_time` and a later
    `available_time`. The join then returns whichever version was current at
    each date, which is what reproducing a past decision means: seeing the
    numbers as they were, mistakes included.

    Call `validate_pit_records` first if the records came from anywhere you
    did not construct yourself.
    """
    import pandas as pd

    from standard_quant_tools.modeling.dataset.point_in_time import (
        AVAILABLE_TIME,
        ENTITY,
        EVENT_TIME,
        asof_join,
    )

    logger.debug(
        "[join_point_in_time] dataset_id=%s n_records=%d",
        input_data.dataset_id,
        len(input_data.records),
    )
    panel, _meta, _dir = _load_dataset_panel(input_data.dataset_id)
    records = _pit_frame(input_data.records, input_data.entity_scoped)

    reserved = {EVENT_TIME, AVAILABLE_TIME, ENTITY}
    fields = input_data.fields or sorted(
        c for c in records.columns if c not in reserved
    )
    staleness = (
        pd.Timedelta(days=input_data.max_staleness_days)
        if input_data.max_staleness_days
        else None
    )

    joined = asof_join(
        panel,
        records,
        fields=fields,
        by_entity=input_data.entity_scoped,
        prefix=input_data.prefix,
        max_staleness=staleness,
    )
    added = [f"{input_data.prefix}{f}" for f in fields]
    coverage = {column: float(joined[column].notna().mean()) for column in added}

    warnings = []
    for column, fraction in sorted(coverage.items()):
        if fraction == 0.0:
            warnings.append(
                f"{column}: no panel row received a value. Every record "
                "became available after the panel ends, or the entities do "
                "not match."
            )
        elif fraction < 0.5:
            warnings.append(
                f"{column}: only {fraction:.0%} of rows received a value. "
                "Usually the panel starts before the first release, which is "
                "expected -- but check the entity names match."
            )

    uri = _artifacts.save_artifact(
        joined, run_id=input_data.dataset_id, name="pit_joined"
    )
    return JoinPointInTimeResult(
        dataset_id=input_data.dataset_id,
        joined_uri=uri,
        n_rows=int(len(joined)),
        fields_added=added,
        coverage=coverage,
        warnings=warnings,
    )


MODELING_TOOL_DISPATCH = {
    "analyze_model_errors": (analyze_model_errors, AnalyzeModelErrorsInput),
    "explain_dataset_row_loss": (
        explain_dataset_row_loss,
        ExplainRowLossInput,
    ),
    "validate_pit_records": (validate_pit_records, PitRecordsInput),
    "join_point_in_time": (join_point_in_time, JoinPointInTimeInput),
    "build_model_ensemble": (build_model_ensemble, BuildEnsembleInput),
    "register_external_panel": (
        register_external_panel,
        RegisterExternalPanelInput,
    ),
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
