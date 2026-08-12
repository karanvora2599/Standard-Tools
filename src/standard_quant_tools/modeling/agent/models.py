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
        None,
        description="Filter to one category, e.g. 'technical' or 'factors'. Omit for the full catalog.",
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
    entities: List[str] = Field(
        ...,
        description=(
            "Entities present in the built panel. This reports what the "
            "model will actually be trained on, not the symbols fetched — "
            "the two differ whenever a symbol's history is shorter than the "
            "feature lookbacks plus the target horizon, and reporting the "
            "fetched list overstated coverage. A symbol that dropped out "
            "entirely is named in `warnings`."
        ),
    )
    feature_ids: List[str]
    target_id: str
    drop_attribution: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "What feature/target alignment cost, per column: `n_missing` "
            "(rows where that column was NaN) and `n_sole_missing` (rows "
            "where it was the ONLY thing missing — what removing just that "
            "feature would give back), plus rows before/after and per-entity "
            "drop counts. Row loss here is normal, but a final row count "
            "alone cannot separate the warm-up you asked for from one "
            "feature quietly consuming the panel."
        ),
    )
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Coverage and provenance conditions that change how this "
            "dataset's models should be read, but are not grounds to refuse "
            "to build it: a provider that makes no point-in-time or "
            "survivorship guarantee, a symbol covering only part of the "
            "requested window, a complete-case intersection that truncated "
            "the cross-sectional features, or a non-daily interval against "
            "daily-calibrated feature defaults. See "
            "dataset/coverage.py. Carried onto any model trained from this "
            "dataset as ModelManifest.dataset_warnings."
        ),
    )


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
    validation_report: Dict[str, Any] = Field(
        default_factory=dict,
        description="Per-fold metrics and windows, plus fold accounting "
        "(expected/completed/skipped with reasons, rows purged for target "
        "overlap, and the target horizon). Averaged oos_metrics alone cannot "
        "show performance decay across folds, reveal that one fold carried "
        "the result, or expose how much of the walk-forward schedule "
        "actually ran.",
    )
    n_train_rows_purged_overlap: int = Field(
        0,
        description="Training rows dropped because their forward-return "
        "label would have resolved inside the test window. A large count "
        "means the target horizon consumes a real fraction of each training "
        "window — relevant when reading the OOS metrics.",
    )
    oos_predictions_uri: str = Field(
        ...,
        description="Walk-forward out-of-sample predictions (date, entity, prediction). "
        "Each fold's predictions come from a model that never saw that fold's dates in "
        "training, and training rows whose forward-return label would have resolved "
        "inside the test window are purged — so these are genuinely out-of-sample, "
        "unlike score_model's single as-of snapshot (which uses the full-panel refit). "
        "Feed to modeling.bridge.oos_predictions_to_signal_panel to backtest this model "
        "as a strategy via the existing run_signal_panel_backtest tool.",
    )


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
    max_staleness_days: Optional[int] = Field(
        None,
        gt=0,
        description="Reject the call if the newest available observation "
        "(effective_score_date) is more than this many calendar days before "
        "as_of. Enforcing a single cross-section date makes every returned "
        "prediction internally consistent, but says nothing about how OLD that "
        "shared date is — a universe whose data stopped six months ago still "
        "produces a perfectly uniform, entirely stale cross-section. Set this "
        "to state how far behind as_of a prediction is still decision-useful. "
        "None (default) does not check; staleness_days is reported either way, "
        "so the gap is never invisible.",
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
    effective_score_date: str = Field(
        "",
        description="The single observation date every returned prediction was "
        "actually computed from. Distinct from as_of, which is only the date "
        "REQUESTED: the most recent bar available at or before as_of can be "
        "earlier (a market holiday, a provider whose `end` excluded as_of, a "
        "symbol that stopped trading). Reported so a caller never has to assume "
        "as_of and the data behind the prediction are the same date.",
    )
    staleness_days: int = Field(
        0,
        description="Calendar days between effective_score_date and as_of. "
        "Always reported, whether or not max_staleness_days was set: a "
        "uniform cross-section can still be an entirely stale one, and that "
        "should never be something the caller has to go and derive.",
    )
    predictions_uri: str
    predictions_hash: str = Field(
        "",
        description="Content digest of the written predictions artifact. The "
        "artifact path is content-addressed, so a URI recorded by one call "
        "always resolves to the bytes that call produced — re-scoring after a "
        "data revision writes a NEW path rather than replacing an older one an "
        "audit record still points at.",
    )
    n_entities: int
    summary_stats: Dict[str, float]
    missing_entities: List[str] = Field(
        default_factory=list,
        description="Requested universe symbols that had no scoreable row as of "
        "as_of (e.g. insufficient history within lookback_days) — silently absent "
        "from predictions_uri, listed here instead of being dropped without a trace.",
    )
    stale_entities: Dict[str, str] = Field(
        default_factory=dict,
        description="Symbol -> its most recent available observation date, for "
        "symbols whose latest bar predates effective_score_date. These are "
        "EXCLUDED from predictions_uri rather than scored on an older bar: "
        "scoring each entity on whatever date it last traded silently mixes "
        "observation dates into one 'cross-section', which for a "
        "cross-sectional model means the ranking no longer compares "
        "contemporaneous information.",
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
