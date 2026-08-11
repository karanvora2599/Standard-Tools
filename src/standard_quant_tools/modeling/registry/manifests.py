"""ModelManifest — the schema written to manifest.json for every
registered model. Mirrors the provenance fields audit.models.DecisionRecord
already captures (git_commit_sha, package_version) so a model's lineage
is legible by reading manifest.json alone, without cross-referencing the
audit log."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelManifest(BaseModel):
    # model_id (and other model_*-prefixed fields below) would otherwise
    # collide with pydantic's own "model_" protected-namespace warning.
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    version: int
    task: str
    estimator_type: str
    estimator_params: Dict[str, Any]
    feature_ids: List[str]
    target_id: str
    dataset_id: str
    dataset_hash: str
    validation_method: str
    oos_metrics: Dict[str, float]
    feature_importance_summary: Dict[str, Dict[str, float]]
    n_folds: int
    # Per-fold metrics/windows plus fold accounting (expected vs completed
    # vs skipped, with reasons). Averaged metrics alone hide performance
    # decay, single-fold dominance, and how much of the walk-forward
    # schedule actually ran. Optional so older manifests still load.
    validation_report: Dict[str, Any] = Field(default_factory=dict)
    oos_predictions_uri: str
    random_seed: int
    # SHA-256 of the DatasetSpec that produced the training data.
    # build_dataset already computed this and threw it away; recorded here
    # so a model can be tied to the exact feature/target definition it was
    # trained under, independently of the mutable dataset_spec.json file.
    dataset_spec_hash: Optional[str] = None
    # {filename: content hash} for every artifact in this model's own
    # directory, plus its OOS predictions. Verified on load so an edited
    # spec, tampered preprocessing stats or swapped estimator binary is
    # rejected instead of silently changing predictions. Empty for models
    # registered before content hashing existed -- verify_file skips a
    # None expectation rather than failing every older model.
    content_hashes: Dict[str, str] = Field(default_factory=dict)
    # {feature_id: hash of that feature's implementation source}. A git
    # SHA covers the repo, but says nothing about a custom feature
    # registered at runtime from outside it.
    feature_implementation_hashes: Dict[str, str] = Field(default_factory=dict)
    # Last date in the panel the deployed (full-refit) estimator saw.
    # score_model needs this to reject a historical as_of: the registered
    # estimator is refit on the ENTIRE training panel, so asking it to
    # "predict" a date inside that panel produces a future-trained
    # prediction dressed up as a historical one. Optional so manifests
    # written before this field existed still load.
    train_end_date: Optional[str] = None
    created_at_utc: str
    git_commit_sha: Optional[str] = None
    package_version: Optional[str] = None
