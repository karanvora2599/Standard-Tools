"""
File-based model registry: save_model/load_model/load_manifest, using
modeling.artifacts' atomic-write helpers. Layout:

    SQT_RUNS_DIR/<model_id>/
        manifest.json
        model.joblib
        model_spec.json
        preprocessing_stats.json
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from standard_quant_tools.audit.provenance import _git_sha, _package_version
from standard_quant_tools.error import ValidationError

from .. import artifacts as _artifacts
from ..specs import ModelSpec
from .feature_provenance import feature_implementation_hashes
from .manifests import ModelManifest

logger = logging.getLogger(__name__)


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
    oos_predictions_uri: str,
    model_id: Optional[str] = None,
    train_end_date: Optional[str] = None,
    training_information_cutoff: Optional[str] = None,
    dataset_spec: Optional[Dict[str, Any]] = None,
    dataset_spec_hash: Optional[str] = None,
    validation_report: Optional[Dict[str, Any]] = None,
    dataset_warnings: Optional[List[str]] = None,
) -> ModelManifest:
    """
    preprocessing_stats: the fit_preprocessing() output computed on the
    FULL training panel that produced `estimator` (engine.py's final
    refit, not any one walk-forward fold) — persisted so scoring.py can
    apply the identical winsorize/zscore transform to new data instead of
    refitting stats on whatever happens to be in the scoring universe.

    oos_predictions_uri: where engine.py already persisted the
    walk-forward out-of-sample fold predictions (date, entity, prediction)
    — recorded here (not re-saved) so inspect_model can surface it and
    modeling.bridge.oos_predictions_to_signal_panel can find it from just
    a model_id, matching the "the model_id is the entry point to every
    one of its artifacts" convention every other file in this directory
    already follows.
    """
    model_id = model_id or new_model_id()
    directory = _artifacts.run_dir(model_id)

    # ── Self-containment ──────────────────────────────────────────────
    # The training DatasetSpec is COPIED into the model's own directory
    # rather than referenced by dataset_id. Previously score_model reached
    # back into SQT_RUNS_DIR/<dataset_id>/dataset_spec.json every time,
    # which made an otherwise-valid model unscoreable once the dataset
    # directory was archived or deleted, and meant editing that file
    # silently redefined the features of every model trained from it.
    directory.mkdir(parents=True, exist_ok=True)
    dataset_spec_path = None
    if dataset_spec is not None:
        dataset_spec_path = _artifacts.save_json(
            directory, "dataset_spec", dataset_spec
        )

    # Written before the manifest so their digests can go INTO it.
    model_path = _artifacts.save_joblib(directory, "model", estimator)
    model_spec_path = _artifacts.save_json(
        directory, "model_spec", model_spec.model_dump()
    )
    preprocessing_path = _artifacts.save_json(
        directory, "preprocessing_stats", preprocessing_stats
    )

    content_hashes: Dict[str, str] = {
        "model.joblib": _artifacts.hash_file(Path(model_path)),
        "model_spec.json": _artifacts.hash_file(Path(model_spec_path)),
        "preprocessing_stats.json": _artifacts.hash_file(Path(preprocessing_path)),
    }
    if dataset_spec_path is not None:
        content_hashes["dataset_spec.json"] = _artifacts.hash_file(
            Path(dataset_spec_path)
        )
    if oos_predictions_uri:
        oos_path = Path(oos_predictions_uri)
        if oos_path.exists():
            content_hashes["oos_predictions"] = _artifacts.hash_file(oos_path)

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
        validation_report=validation_report or {},
        oos_predictions_uri=oos_predictions_uri,
        random_seed=model_spec.random_seed,
        dataset_spec_hash=dataset_spec_hash,
        content_hashes=content_hashes,
        feature_implementation_hashes=feature_implementation_hashes(feature_ids),
        train_end_date=train_end_date,
        training_information_cutoff=training_information_cutoff,
        dataset_warnings=list(dataset_warnings or []),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit_sha=_git_sha(),
        package_version=_package_version(),
    )
    # manifest.json is written LAST and is the commit point for the whole
    # package. Each individual file was already written atomically, but a
    # crash partway through still left a half-registered directory that
    # looked loadable. Every loader keys off manifest.json's existence, so
    # a directory without it is simply not a registered model -- the write
    # order is the transaction boundary.
    _artifacts.save_json(directory, "manifest", manifest.model_dump())
    return manifest


def _expected_hash(model_id: str, filename: str) -> Optional[str]:
    """The registered digest for one artifact, or None when this model
    predates content hashing (or the manifest is unreadable — in which
    case the caller's own missing-file check reports the real problem)."""
    try:
        return load_manifest(model_id).content_hashes.get(filename)
    except ValidationError:
        return None


def load_preprocessing_stats(model_id: str) -> Dict[str, Dict[str, float]]:
    directory = _artifacts.run_dir(model_id)
    path = directory / "preprocessing_stats.json"
    if not path.exists():
        raise ValidationError(f"no registered model with model_id={model_id!r}")
    # Editing a mean or scale here silently shifts every future prediction
    # from this model while the model_id stays the same.
    _artifacts.verify_file(
        path,
        _expected_hash(model_id, "preprocessing_stats.json"),
        "preprocessing_stats.json",
    )
    return _artifacts.load_json(str(path))


def load_dataset_spec(model_id: str) -> Dict[str, Any]:
    """
    The DatasetSpec this model was TRAINED with, read from the model's own
    verified copy.

    scoring.py previously re-read SQT_RUNS_DIR/<dataset_id>/dataset_spec.json
    on every call, so changing an RSI period from 14 to 100 in that file
    silently fed the registered estimator a differently-defined feature —
    with no integrity check and no change to the model_id.
    """
    directory = _artifacts.run_dir(model_id)
    path = directory / "dataset_spec.json"
    if path.exists():
        _artifacts.verify_file(
            path, _expected_hash(model_id, "dataset_spec.json"), "dataset_spec.json"
        )
        return _artifacts.load_json(str(path))

    # Fallback for models registered before models carried their own spec.
    # Deliberately a warning rather than a hard error: refusing to score
    # every previously-registered model would make an upgrade look like
    # data loss. The fallback keeps the OLD weakness (an unverified file in
    # a directory that may be deleted), so it says so.
    legacy_path = (
        _artifacts.run_dir(load_manifest(model_id).dataset_id) / "dataset_spec.json"
    )
    if legacy_path.exists():
        logger.warning(
            "[score_model] model %s predates self-contained models — falling back to "
            "the dataset directory's dataset_spec.json (%s), which is NOT content-"
            "verified and disappears if that dataset is archived. Re-run the experiment "
            "to register a model that bundles its own training spec.",
            model_id,
            legacy_path,
        )
        return _artifacts.load_json(str(legacy_path))

    raise ValidationError(
        f"model {model_id!r} has no bundled dataset_spec.json, and its source dataset "
        f"directory no longer has one either. The model cannot be scored without the "
        "feature definitions it was trained on — re-run the experiment to register a "
        "self-contained model."
    )


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
    # Verified BEFORE joblib.load, which is the important ordering:
    # joblib/pickle deserialization executes code from the file, so a
    # swapped binary is an arbitrary-code-execution vector, not merely a
    # wrong-predictions one. Checking the digest first means a tampered
    # blob is rejected without ever being deserialized.
    #
    # This is integrity, not authenticity: it detects an artifact that no
    # longer matches its manifest, but an attacker who can rewrite BOTH
    # model.joblib and manifest.json is still out of scope. Signing the
    # manifest (as audit/signing.py does for decision records) is what
    # would close that, and is the right next step before this registry is
    # trusted across a trust boundary.
    _artifacts.verify_file(
        path, _expected_hash(model_id, "model.joblib"), "model.joblib"
    )
    return _artifacts.load_joblib(str(path))


def load_model_spec(model_id: str) -> ModelSpec:
    directory = _artifacts.run_dir(model_id)
    path = directory / "model_spec.json"
    if not path.exists():
        raise ValidationError(f"no registered model with model_id={model_id!r}")
    _artifacts.verify_file(
        path, _expected_hash(model_id, "model_spec.json"), "model_spec.json"
    )
    return ModelSpec(**_artifacts.load_json(str(path)))
