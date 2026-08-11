"""
The 5-tool modeling agent surface — kept structurally separate from
standard_quant_tools.agent.get_agent_tools()/TOOL_CATEGORY (the existing
46-tool analysis/backtest registry) per Documentation/15_modeling.md's
architecture rationale: fitting/validating/registering a statistical
model doesn't fit that surface's shape (a point-in-time snapshot or a
single backtest run), and adding it there would make the tool-selection
ambiguity problem TOOL_CATEGORY already exists to mitigate worse, not
better.

    list_features        — the feature catalog, not semantic search
                            (~9 entries doesn't need ranking).
    build_model_dataset  — DatasetSpec -> persisted panel + dataset_id.
    run_model_experiment — dataset_id + ModelSpec -> fit + walk-forward
                            validate + register, one call. Structurally
                            impossible to fit without validation, since
                            there is no separate "just fit" tool.
    score_model           — model_id + as_of + universe -> predictions
                            artifact.
    inspect_model          — one tool, four views, instead of four
                            separate inspection tools.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from .. import artifacts as _artifacts
from ..dataset.builder import build_dataset as _build_dataset
from ..engine import run_experiment as _run_experiment
from ..features.registry import list_features as _list_features
from ..registry.model_registry import load_manifest
from ..scoring import score_model as _score_model
from .models import (
    BuildModelDatasetInput,
    BuildModelDatasetResult,
    FeatureCatalogEntry,
    InspectModelInput,
    InspectModelResult,
    ListFeaturesInput,
    ListFeaturesResult,
    RunModelExperimentInput,
    RunModelExperimentResult,
    ScoreModelInput,
    ScoreModelResult,
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
        "[build_model_dataset] universe=%s  %s -> %s  features=%d",
        input_data.spec.universe,
        input_data.spec.start,
        input_data.spec.end,
        len(input_data.spec.features),
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
    _artifacts.save_json(
        directory,
        "dataset_meta",
        {
            "feature_ids": built["feature_ids"],
            "target_id": built["target_id"],
            "data_hash": built["data_hash"],
            "entities": built["entities"],
        },
    )

    return BuildModelDatasetResult(
        dataset_id=dataset_id,
        rows=len(built["panel"]),
        entities=built["entities"],
        feature_ids=built["feature_ids"],
        target_id=built["target_id"],
    )


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
    directory = _artifacts.run_dir(input_data.dataset_id)
    panel = _artifacts.load_artifact(str(directory / "panel.parquet"))
    meta = _artifacts.load_json(str(directory / "dataset_meta.json"))

    dataset = {
        "panel": panel,
        "feature_ids": meta["feature_ids"],
        "target_id": meta["target_id"],
        "data_hash": meta["data_hash"],
    }
    result = _run_experiment(dataset, input_data.spec, dataset_id=input_data.dataset_id)
    return RunModelExperimentResult(**result)


def score_model(input_data: ScoreModelInput) -> ScoreModelResult:
    result = _score_model(
        model_id=input_data.model_id,
        as_of=input_data.as_of,
        universe=input_data.universe,
        lookback_days=input_data.lookback_days,
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
        }

    return InspectModelResult(
        model_id=input_data.model_id, view=input_data.view, data=data
    )


# ── Registration (mirrors agent.tools.get_agent_tools()/_TOOL_DISPATCH,
# but a separate registry — never merged into that one) ────────────────

_MODELING_TOOL_DEFS: List[tuple] = [
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
]

MODELING_TOOL_DISPATCH = {
    "list_features": (list_features, ListFeaturesInput),
    "build_model_dataset": (build_model_dataset, BuildModelDatasetInput),
    "run_model_experiment": (run_model_experiment, RunModelExperimentInput),
    "score_model": (score_model, ScoreModelInput),
    "inspect_model": (inspect_model, InspectModelInput),
}


def get_modeling_tools() -> List[Dict[str, Any]]:
    """Tool definitions for the modeling runtime, in the exact same
    OpenAI-style {"type": "function", "function": {...}} envelope
    agent.tools.get_agent_tools() returns — a separate 5-entry list,
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
