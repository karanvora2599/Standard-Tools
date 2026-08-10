"""
File-based model registry: save_model/load_model/load_manifest, using
modeling.artifacts' atomic-write helpers. Layout:

    SQT_RUNS_DIR/<model_id>/
        manifest.json
        model.joblib
        model_spec.json
        preprocessing_stats.json
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from standard_quant_tools.audit.provenance import _git_sha, _package_version
from standard_quant_tools.error import ValidationError

from .. import artifacts as _artifacts
from ..specs import ModelSpec
from .manifests import ModelManifest


def new_model_id() -> str:
    return f"mdl_{uuid.uuid4().hex[:12]}"


def save_model(
    estimator: Any,
    model_spec: ModelSpec,
    feature_ids: List[str],
    target_id: str,
    dataset_id: str,
    dataset_hash: str,
    oos_metrics: Dict[str, float],
    feature_importance_summary: Dict[str, Dict[str, float]],
    n_folds: int,
    preprocessing_stats: Dict[str, Dict[str, float]],
    model_id: Optional[str] = None,
) -> ModelManifest:
    """
    preprocessing_stats: the fit_preprocessing() output computed on the
    FULL training panel that produced `estimator` (engine.py's final
    refit, not any one walk-forward fold) — persisted so scoring.py can
    apply the identical winsorize/zscore transform to new data instead of
    refitting stats on whatever happens to be in the scoring universe.
    """
    model_id = model_id or new_model_id()
    directory = _artifacts.run_dir(model_id)
    manifest = ModelManifest(
        model_id=model_id,
        version=1,
        task=model_spec.task,
        estimator_type=model_spec.estimator.type,
        estimator_params=model_spec.estimator.params,
        feature_ids=feature_ids,
        target_id=target_id,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        validation_method=model_spec.validation.method,
        oos_metrics=oos_metrics,
        feature_importance_summary=feature_importance_summary,
        n_folds=n_folds,
        random_seed=model_spec.random_seed,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit_sha=_git_sha(),
        package_version=_package_version(),
    )
    _artifacts.save_joblib(directory, "model", estimator)
    _artifacts.save_json(directory, "manifest", manifest.model_dump())
    _artifacts.save_json(directory, "model_spec", model_spec.model_dump())
    _artifacts.save_json(directory, "preprocessing_stats", preprocessing_stats)
    return manifest


def load_preprocessing_stats(model_id: str) -> Dict[str, Dict[str, float]]:
    directory = _artifacts.run_dir(model_id)
    path = directory / "preprocessing_stats.json"
    if not path.exists():
        raise ValidationError(f"no registered model with model_id={model_id!r}")
    return _artifacts.load_json(str(path))


def load_manifest(model_id: str) -> ModelManifest:
    directory = _artifacts.run_dir(model_id)
    path = directory / "manifest.json"
    if not path.exists():
        raise ValidationError(f"no registered model with model_id={model_id!r}")
    return ModelManifest(**_artifacts.load_json(str(path)))


def load_model(model_id: str) -> Any:
    directory = _artifacts.run_dir(model_id)
    path = directory / "model.joblib"
    if not path.exists():
        raise ValidationError(f"no registered model with model_id={model_id!r}")
    return _artifacts.load_joblib(str(path))


def load_model_spec(model_id: str) -> ModelSpec:
    directory = _artifacts.run_dir(model_id)
    path = directory / "model_spec.json"
    if not path.exists():
        raise ValidationError(f"no registered model with model_id={model_id!r}")
    return ModelSpec(**_artifacts.load_json(str(path)))
