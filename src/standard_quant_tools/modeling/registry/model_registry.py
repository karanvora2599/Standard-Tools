"""
File-based model registry: save_model/load_model/load_manifest, using
modeling.artifacts' atomic-write helpers. Layout:

    SQT_RUNS_DIR/<model_id>/
        manifest.json
        model.joblib
        model_spec.json
        preprocessing_stats.json
        manifest.sig        (when a signing key is configured; see signing.py)
        promotions.jsonl    (once promoted; see lifecycle.py)
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
from . import signing as _signing
from .environment import environment_fingerprint
from .feature_provenance import (
    feature_implementation_hashes,
    feature_provenance_from_spec,
)
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
    dataset_spec_hash_version: Optional[int] = None,
    validation_report: Optional[Dict[str, Any]] = None,
    dataset_warnings: Optional[List[str]] = None,
    preprocessing: Optional[Dict[str, Any]] = None,
    environment: Optional[Dict[str, Any]] = None,
    preprocessing_state: Optional[Dict[str, Any]] = None,
    model_input_columns: Optional[List[str]] = None,
    distribution: Optional[Dict[str, Any]] = None,
    quantile_models: Optional[Dict[str, Any]] = None,
    feature_profile: Optional[Dict[str, Any]] = None,
    feature_reference: Optional[Any] = None,
    prediction_reference: Optional[Any] = None,
) -> ModelManifest:
    """
    preprocessing_stats: the fit_preprocessing() output computed on the
    FULL training panel that produced `estimator` (engine.py's final
    refit, not any one walk-forward fold) — persisted so scoring.py can
    apply the identical winsorize/zscore transform to new data instead of
    refitting stats on whatever happens to be in the scoring universe.
    Empty for a cross-sectional model, which fits nothing per column.

    preprocessing: the PreprocessingSpec the estimator was refit under, so
    scoring can tell whether to apply those statistics or to standardize
    within the scoring date's own cross-section. See ModelManifest.

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
    # Resolved from the training DatasetSpec's feature entries, which carry
    # the registry id, the requested params AND the alias -- everything the
    # panel's column names alone had thrown away.
    _provenance = feature_provenance_from_spec(
        (dataset_spec or {}).get("features") if dataset_spec else None
    )

    model_path = _artifacts.save_joblib(directory, "model", estimator)
    model_spec_path = _artifacts.save_json(
        directory, "model_spec", model_spec.model_dump()
    )
    preprocessing_path = _artifacts.save_json(
        directory, "preprocessing_stats", preprocessing_stats
    )
    # The fitted pipeline state -- every step's type, parameters and fitted
    # values, in order. This is what scoring applies; the statistics file
    # above is its legacy projection. Written whenever the engine supplies
    # one, which is every registration through run_experiment.
    state_path = None
    if preprocessing_state is not None:
        state_path = _artifacts.save_json(
            directory, "preprocessing_state", preprocessing_state
        )

    # The deployed distribution: the quantile levels, their columns and
    # the conformal radius as JSON, and the fitted quantile estimators as
    # a joblib beside the point estimator. Both are hashed into the
    # manifest like every other artifact, because an edited radius shifts
    # every interval while the model_id stays the same.
    distribution_path = None
    quantile_models_path = None
    if distribution is not None:
        distribution_path = _artifacts.save_json(
            directory, "distribution", distribution
        )
    if quantile_models:
        quantile_models_path = _artifacts.save_joblib(
            directory, "quantile_models", quantile_models
        )
    content_hashes: Dict[str, str] = {
        "model.joblib": _artifacts.hash_file(Path(model_path)),
        "model_spec.json": _artifacts.hash_file(Path(model_spec_path)),
        "preprocessing_stats.json": _artifacts.hash_file(Path(preprocessing_path)),
    }
    if state_path is not None:
        content_hashes["preprocessing_state.json"] = _artifacts.hash_file(
            Path(state_path)
        )
    if distribution_path is not None:
        content_hashes["distribution.json"] = _artifacts.hash_file(
            Path(distribution_path)
        )
    # The monitoring reference: a profile a reader can inspect, and the
    # seeded samples PSI and KS are computed against. Hashed like every
    # other artifact, because a reference that could be edited is a
    # drift report that could be made to say anything.
    monitoring: Dict[str, Any] = {}
    if feature_profile is not None:
        profile_path = _artifacts.save_json(
            directory, "feature_profile", feature_profile
        )
        content_hashes["feature_profile.json"] = _artifacts.hash_file(
            Path(profile_path)
        )
        monitoring["profile_bins"] = feature_profile.get("bins")
    if feature_reference is not None:
        uri = _artifacts.save_artifact(
            feature_reference, run_id=model_id, name="feature_reference"
        )
        content_hashes["feature_reference"] = _artifacts.hash_file(Path(uri))
        monitoring["feature_reference_uri"] = uri
        monitoring["feature_reference_rows"] = int(len(feature_reference))
    if prediction_reference is not None:
        uri = _artifacts.save_artifact(
            prediction_reference, run_id=model_id, name="prediction_reference"
        )
        content_hashes["prediction_reference"] = _artifacts.hash_file(Path(uri))
        monitoring["prediction_reference_uri"] = uri
        monitoring["prediction_reference_rows"] = int(len(prediction_reference))
    if quantile_models_path is not None:
        content_hashes["quantile_models.joblib"] = _artifacts.hash_file(
            Path(quantile_models_path)
        )
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
        model_input_columns=list(model_input_columns or []),
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
        dataset_spec_hash_version=dataset_spec_hash_version,
        content_hashes=content_hashes,
        # Derived from the DatasetSpec's own feature entries, so an aliased
        # column resolves through its real registry id instead of having
        # its alias looked up as one (which recorded "unavailable", or --
        # when the alias happened to name another feature -- that other
        # feature's hash). Falls back to the id-keyed form only when no
        # spec was supplied, which is the pre-alias behavior.
        feature_provenance=_provenance,
        feature_implementation_hashes=(
            {
                column: record["implementation_hash"]
                for column, record in _provenance.items()
            }
            if _provenance
            else feature_implementation_hashes(feature_ids)
        ),
        train_end_date=train_end_date,
        training_information_cutoff=training_information_cutoff,
        dataset_warnings=list(dataset_warnings or []),
        preprocessing=dict(preprocessing or {}),
        distribution=dict(distribution or {}),
        monitoring=monitoring,
        # Read from the process at registration, never declared by the
        # caller -- a caller-supplied value is only for tests that need a
        # known one.
        environment=(
            dict(environment) if environment is not None else environment_fingerprint()
        ),
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
    # An attestation on the package that now exists, written after the
    # commit point rather than as part of it. Only when a key is
    # configured: an unsigned registration is the default and not a
    # failure; `load_manifest(require_signature=True)` is where a caller
    # says unsigned is not enough.
    if _signing.signing_configured():
        _signing.sign_manifest(model_id)
    return manifest


def _expected_hash(model_id: str, filename: str) -> Optional[str]:
    """
    The registered digest for one artifact, or None when this model predates
    content hashing.

    Only that second case may return None. This used to also swallow a
    ValidationError from load_manifest() and return None, which quietly
    turned "I cannot read the manifest" into "this artifact has no expected
    hash" -- and verify_file() treats expected=None as "skip verification".

    manifest.json is the commit point of the package: it is written last,
    and every other artifact's digest lives in it. So deleting it downgraded
    every integrity check at once. Measured on a registered model whose
    model.joblib had been swapped:

        manifest present  -> refused (hash mismatch)
        manifest deleted  -> DESERIALIZED the tampered file

    Removing the manifest is strictly easier than forging a hash inside it,
    so the bypass was cheaper than the attack it was meant to stop -- and
    joblib.load executes code from the file, making this an arbitrary-code-
    execution path rather than merely a wrong-answer one.

    load_manifest's error now propagates. A registered model always has a
    manifest; if it is absent or unreadable, the package is not intact and
    no loader should proceed on the assumption that it is.
    """
    return load_manifest(model_id).content_hashes.get(filename)


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


def load_preprocessing_state(model_id: str) -> Optional[Dict[str, Any]]:
    """
    The fitted preprocessing pipeline the deployed estimator expects, or
    None for a model registered before the state file existed.

    None is a real answer rather than an error: the scoring path falls back
    to the legacy statistics file for such a model, and refuses by name
    when that file cannot describe the transform that was validated.
    """
    directory = _artifacts.run_dir(model_id)
    path = directory / "preprocessing_state.json"
    if not path.exists():
        if not (directory / "manifest.json").exists():
            raise ValidationError(f"no registered model with model_id={model_id!r}")
        return None
    # Same immutability contract as the statistics: an edited state shifts
    # every prediction while the model_id stays the same.
    _artifacts.verify_file(
        path,
        _expected_hash(model_id, "preprocessing_state.json"),
        "preprocessing_state.json",
    )
    return _artifacts.load_json(str(path))


def load_distribution(model_id: str) -> "tuple[Dict[str, Any], Dict[str, Any]]":
    """
    The deployed distribution: (state, quantile models by column), both
    empty for a model registered without quantiles or intervals -- which
    every model before they existed was, and which a point-only model
    still is. Verified against the manifest before either is read, the
    joblib before it is deserialized.
    """
    directory = _artifacts.run_dir(model_id)
    if not (directory / "manifest.json").exists():
        raise ValidationError(f"no registered model with model_id={model_id!r}")
    state: Dict[str, Any] = {}
    models: Dict[str, Any] = {}
    state_path = directory / "distribution.json"
    if state_path.exists():
        _artifacts.verify_file(
            state_path,
            _expected_hash(model_id, "distribution.json"),
            "distribution.json",
        )
        state = _artifacts.load_json(str(state_path))
    models_path = directory / "quantile_models.joblib"
    if models_path.exists():
        _artifacts.verify_file(
            models_path,
            _expected_hash(model_id, "quantile_models.joblib"),
            "quantile_models.joblib",
        )
        models = _artifacts.load_joblib(str(models_path))
    return state, models


def load_monitoring_reference(
    model_id: str,
) -> "tuple[Dict[str, Any], Optional[Any], Optional[Any]]":
    """
    (feature profile, feature reference frame, prediction reference frame)
    for `monitor_model`, each verified against the manifest first; the
    frames are None for a model registered before they were kept.
    """
    manifest = load_manifest(model_id)
    directory = _artifacts.run_dir(model_id)
    profile: Dict[str, Any] = {}
    profile_path = directory / "feature_profile.json"
    if profile_path.exists():
        _artifacts.verify_file(
            profile_path,
            manifest.content_hashes.get("feature_profile.json"),
            "feature_profile.json",
        )
        profile = _artifacts.load_json(str(profile_path))
    features = predictions = None
    feature_uri = manifest.monitoring.get("feature_reference_uri")
    if feature_uri and Path(str(feature_uri)).exists():
        _artifacts.verify_file(
            Path(str(feature_uri)),
            manifest.content_hashes.get("feature_reference"),
            "feature_reference",
        )
        features = _artifacts.load_artifact(str(feature_uri))
    prediction_uri = manifest.monitoring.get("prediction_reference_uri")
    if prediction_uri and Path(str(prediction_uri)).exists():
        _artifacts.verify_file(
            Path(str(prediction_uri)),
            manifest.content_hashes.get("prediction_reference"),
            "prediction_reference",
        )
        predictions = _artifacts.load_artifact(str(prediction_uri))
    return profile, features, predictions


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


def load_manifest(model_id: str, *, require_signature: bool = False) -> ModelManifest:
    """
    The manifest, parsed. With `require_signature`, `manifest.sig` is
    verified over the manifest's bytes BEFORE they are parsed -- against
    `SQT_MODEL_VERIFY_KEY_PATH` when it is set -- so a manifest that fails
    is never turned into an object anything can act on. Unsigned is the
    default and is not a failure; requiring a signature is how a caller
    at a trust boundary says it is not enough.
    """
    directory = _artifacts.run_dir(model_id)
    path = directory / "manifest.json"
    if not path.exists():
        raise ValidationError(f"no registered model with model_id={model_id!r}")
    if require_signature:
        _signing.verify_manifest_signature(model_id)
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
    # model.joblib and manifest.json is still out of scope HERE. Closing
    # it is `registry/signing.py`: `manifest.sig` over the manifest bytes,
    # checked by `load_manifest(require_signature=True)` and by
    # `verify_model_package` against a pinned public key.
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
