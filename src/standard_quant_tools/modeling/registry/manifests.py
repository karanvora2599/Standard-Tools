"""ModelManifest — the schema written to manifest.json for every
registered model. Mirrors the provenance fields audit.models.DecisionRecord
already captures (git_commit_sha, package_version) so a model's lineage
is legible by reading manifest.json alone, without cross-referencing the
audit log."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


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
    oos_predictions_uri: str
    random_seed: int
    created_at_utc: str
    git_commit_sha: Optional[str] = None
    package_version: Optional[str] = None
