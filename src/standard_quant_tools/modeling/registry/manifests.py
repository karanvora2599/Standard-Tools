"""ModelManifest — the schema written to manifest.json for every
registered model. Mirrors the provenance fields audit.models.DecisionRecord
already captures (git_commit_sha, package_version) so a model's lineage
is legible by reading manifest.json alone, without cross-referencing the
audit log."""

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _nulls_to_nan(mapping: Any) -> Any:
    """
    JSON has no NaN, so a NaN metric is written as `null` and comes back as
    None. Map it to NaN on the way in.

    This is not a nicety. Without it, EVERY tree-based model was unloadable:
    summarize_importance correctly reports signed_mean / signed_std /
    sign_consistency as NaN for an estimator with no coefficient sign, those
    became `null` in manifest.json, and ModelManifest then rejected the file
    it had itself written -- so load_model and, through it, score_model failed
    for random_forest, gradient boosting, LightGBM and XGBoost alike. Linear
    models were unaffected, which is why it survived: they have a sign, so
    none of their fields were ever NaN.

    oos_metrics has the same exposure and gets the same treatment: a
    single-class fold's AUC and a single-date fold's IC standard deviation are
    both legitimately NaN.

    NaN rather than dropping the key, because the two say different things --
    a NaN entry means "this quantity does not exist for this model", and an
    absent one means "this version did not record it".
    """
    if not isinstance(mapping, dict):
        return mapping
    out: Dict[str, Any] = {}
    for key, value in mapping.items():
        if value is None:
            out[key] = math.nan
        elif isinstance(value, dict):
            out[key] = _nulls_to_nan(value)
        else:
            out[key] = value
    return out


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

    _coerce_metric_nulls = field_validator(
        "oos_metrics", "feature_importance_summary", mode="before"
    )(_nulls_to_nan)
    # {feature_id: hash of that feature's implementation source}. A git
    # SHA covers the repo, but says nothing about a custom feature
    # registered at runtime from outside it.
    feature_implementation_hashes: Dict[str, str] = Field(default_factory=dict)
    # Full per-column provenance: panel column -> {feature_id, params,
    # implementation_hash}. Keyed by the OUTPUT column (the alias when one
    # is set), with the registry id carried as a value rather than being
    # inferred from the key.
    #
    # feature_implementation_hashes alone could not express this. Its keys
    # were the panel's column names, which it then looked up in the feature
    # registry -- so an aliased column recorded "unavailable", and a column
    # aliased to another feature's name recorded THAT feature's hash. Making
    # the alias a key and never a lookup removes both failure modes.
    feature_provenance: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # Last FEATURE date in the training panel. Lineage only -- this is NOT
    # the cutoff score_model gates on; see training_information_cutoff.
    # Optional so manifests written before this field existed still load.
    train_end_date: Optional[str] = None
    # Last date whose PRICES the training data actually consumed, i.e.
    # max(panel["label_end_date"]).
    #
    # This is the field score_model must gate on. A row dated t with a
    # horizon-h forward-return target reads Close[t+h] to build its label,
    # so a full-refit estimator has indirectly seen prices h bars past the
    # last feature date. Gating on train_end_date (the feature date) left a
    # horizon-wide window -- ~28 calendar days at h=20 -- in which
    # score_model would accept an as_of the model had already consumed the
    # future of, returning a future-trained prediction that looks
    # point-in-time.
    #
    # Optional for the same back-compat reason as train_end_date: manifests
    # written before this existed fall back to the older, weaker guarantee
    # rather than failing to load or silently claiming a cutoff they never
    # recorded.
    training_information_cutoff: Optional[str] = None
    # Coverage/provenance warnings raised when the training dataset was
    # built: survivors-only universe, a provider that revises history, a
    # symbol contributing a fraction of the window, a non-daily interval.
    # Carried onto the model because the dataset's tool response is
    # transient and these are exactly the caveats that must travel WITH the
    # metrics -- inspect_model(view="lineage") is where someone reads the
    # OOS numbers months later, and none of these conditions were visible
    # there before. Empty for models registered before this existed, which
    # is indistinguishable from "no warnings" by design: absence of a
    # recorded warning is not evidence the condition did not hold.
    dataset_warnings: List[str] = Field(default_factory=list)
    created_at_utc: str
    git_commit_sha: Optional[str] = None
    package_version: Optional[str] = None
