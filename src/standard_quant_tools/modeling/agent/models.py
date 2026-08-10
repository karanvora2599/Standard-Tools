"""
Pydantic Input/Result models for the 5-tool modeling agent surface.
DatasetSpec/ModelSpec (modeling.specs) are embedded directly as nested
fields rather than flattened — an LLM constructs one declarative spec
object per call, the same ModelSpec-not-exec() contract described in
Documentation/15_modeling.md, matching how agent/models.py's own Input
models nest structured params for the existing 46-tool surface.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..specs import DatasetSpec, ModelSpec, _parse_date

# Every Input/Result model below with a model_id field sets this to
# silence pydantic's "model_" protected-namespace warning (the same fix
# ModelManifest uses in registry/manifests.py).
_NO_PROTECTED_NAMESPACES = ConfigDict(protected_namespaces=())

# ── list_features ──────────────────────────────────────────────────────


class ListFeaturesInput(BaseModel):
    category: Optional[str] = Field(
        None, description="Filter to one category, e.g. 'technical' or 'factors'. Omit for the full catalog."
    )


class FeatureCatalogEntry(BaseModel):
    id: str
    description: str
    default_params: Dict[str, Any]
    temporal_support: str
    scope: str
    requires: List[str]
    lookback: int


class ListFeaturesResult(BaseModel):
    features: List[FeatureCatalogEntry]


# ── build_model_dataset ────────────────────────────────────────────────


class BuildModelDatasetInput(BaseModel):
    spec: DatasetSpec


class BuildModelDatasetResult(BaseModel):
    dataset_id: str
    rows: int
    entities: List[str]
    feature_ids: List[str]
    target_id: str
    warnings: List[str] = Field(default_factory=list)


# ── run_model_experiment ───────────────────────────────────────────────


class RunModelExperimentInput(BaseModel):
    dataset_id: str = Field(..., description="An id returned by build_model_dataset.")
    spec: ModelSpec


class RunModelExperimentResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    oos_metrics: Dict[str, float]
    feature_importance_summary: Dict[str, Dict[str, float]]
    n_folds: int


# ── score_model ─────────────────────────────────────────────────────────


class ScoreModelInput(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str = Field(..., description="An id returned by run_model_experiment.")
    as_of: str = Field(..., description="Date YYYY-MM-DD to score as of.")
    universe: List[str] = Field(..., min_length=1)
    lookback_days: int = Field(
        400,
        gt=0,
        description="Calendar days of history fetched before as_of — widen for models "
        "using features with unusually large lookback windows.",
    )

    @field_validator("as_of")
    @classmethod
    def _valid_date(cls, v: str) -> str:
        _parse_date(v, "as_of")
        return v

    @field_validator("universe")
    @classmethod
    def _no_duplicate_symbols(cls, v: List[str]) -> List[str]:
        dupes = sorted({s for s in v if v.count(s) > 1})
        if dupes:
            raise ValueError(f"universe contains duplicate symbols: {dupes}")
        return v


class ScoreModelResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    as_of: str
    predictions_uri: str
    n_entities: int
    summary_stats: Dict[str, float]
    missing_entities: List[str] = Field(
        default_factory=list,
        description="Requested universe symbols that had no scoreable row as of "
        "as_of (e.g. insufficient history within lookback_days) — silently absent "
        "from predictions_uri, listed here instead of being dropped without a trace.",
    )


# ── inspect_model ───────────────────────────────────────────────────────


class InspectModelInput(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    view: Literal["summary", "feature_importance", "validation", "lineage"] = Field(
        "summary", description="Which slice of the registered model to return."
    )


class InspectModelResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    view: str
    data: Dict[str, Any]
