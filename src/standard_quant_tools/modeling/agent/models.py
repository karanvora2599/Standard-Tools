"""
Pydantic Input/Result models for the 6-tool modeling agent surface.
DatasetSpec/ModelSpec (modeling.specs) are embedded directly as nested
fields rather than flattened — an LLM constructs one declarative spec
object per call, the same ModelSpec-not-exec() contract described in
Documentation/15_modeling.md, matching how agent/models.py's own Input
models nest structured params for the existing 46-tool surface.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..specs import (
    DatasetSpec,
    ModelSpec,
    PortfolioSimSpec,
    PredictionTransformSpec,
    _parse_date,
)

# Every Input/Result model below with a model_id field sets this to
# silence pydantic's "model_" protected-namespace warning (the same fix
# ModelManifest uses in registry/manifests.py).
_NO_PROTECTED_NAMESPACES = ConfigDict(protected_namespaces=())

# ── list_features ──────────────────────────────────────────────────────


class ListFeaturesInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(extra="forbid")

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
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(extra="forbid")

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
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(extra="forbid")

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
    oos_predictions_ref: Optional[str] = Field(
        None,
        description=(
            "Typed handoff reference for the same predictions "
            "(sqt://predictions/...). This is the one to pass onward: "
            "convert_reference turns it into a signal_panel or a "
            "score_panel, and any runtime can resolve it without the "
            "predictions ever passing through the conversation."
        ),
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
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

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
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: str
    view: Literal["summary", "feature_importance", "validation", "lineage"] = Field(
        "summary", description="Which slice of the registered model to return."
    )


class InspectModelResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    view: str
    data: Dict[str, Any]


# ── analyze_features ────────────────────────────────────────────────────


class ListModelingCapabilitiesInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    include_estimators: bool = Field(
        True,
        description="Include the per-estimator capability table. Turn it off "
        "for a compact answer when only the task/target/validation options "
        "are needed.",
    )


class ListModelingCapabilitiesResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    capabilities: Dict[str, Any]


class AnalyzeFeaturesInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    features: Optional[List[str]] = Field(
        None,
        description="Feature columns to analyze. Defaults to every feature in "
        "the dataset.",
    )
    n_quantiles: int = Field(
        10,
        ge=2,
        le=100,
        description="Buckets used for the quantile spread and monotonicity. "
        "Deciles by default; fewer hides the shape of the relationship, more "
        "puts too few entities per bucket to mean anything on a small "
        "universe.",
    )
    cluster_threshold: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Absolute correlation at or above which two features are "
        "grouped as near-duplicates.",
    )
    include_leakage: bool = Field(
        True,
        description="Run the lead-lag causality screen. It costs "
        "(2 * leakage_max_shift + 1) IC passes per feature, which is the "
        "expensive part of the report — but a screen nobody runs catches "
        "nothing, so it is on by default.",
    )
    leakage_max_shift: int = Field(
        5,
        ge=1,
        le=60,
        description="How many bars either side to shift each feature for the "
        "causality screen.",
    )


class AnalyzeFeaturesResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    dataset_id: str
    report: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)


# ── evaluate_model_portfolio ────────────────────────────────────────────


class EvaluateModelPortfolioInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: str = Field(
        ..., description="A model_id returned by run_model_experiment."
    )
    transform: PredictionTransformSpec = Field(
        default_factory=PredictionTransformSpec,
        description="How the model's out-of-sample predictions become target "
        "weights. Defaults to a dollar-neutral, weekly-rebalanced, "
        "rank-weighted portfolio capped at 5% per name.",
    )
    portfolio: PortfolioSimSpec = Field(
        default_factory=PortfolioSimSpec,
        description="Simulation parameters (capital, costs, fill convention, "
        "leverage limits). Defaults to next-open fills, 10bps commission and "
        "5bps slippage, unlevered.",
    )


class EvaluateModelPortfolioResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    metrics: Dict[str, float] = Field(
        ...,
        description=(
            "Economic performance of the simulated account: cumulative "
            "return, CAGR, annualized volatility, Sharpe, Sortino, max "
            "drawdown, Calmar, turnover, mean gross/net exposure, position "
            "count, and estimated_cost_drag_pct. These are what the model is "
            "worth AFTER costs and position sizing — a different question "
            "from run_model_experiment's oos_metrics (R2, IC), which measure "
            "predictive accuracy and can be strong while these are negative."
        ),
    )
    transform_diagnostics: Dict[str, Any] = Field(
        ...,
        description="What the prediction -> weight step actually produced: "
        "names per date, book sizes, realized gross/net exposure, dates that "
        "could not reach the target gross under the position cap, and dates "
        "with no position at all.",
    )
    coverage: Dict[str, Any] = Field(
        ...,
        description="Entities, prediction dates, rebalance dates actually "
        "traded, and simulated bars — how much of a track record this number "
        "rests on.",
    )
    target_weights_uri: str = Field(
        ...,
        description="Persisted (date x entity) target-weight panel that drove "
        "the simulation. Content-addressed, so re-running with different "
        "transform settings writes a new artifact rather than replacing one an "
        "audit record still points at.",
    )
    equity_curve_uri: str
    provenance: Dict[str, Any] = Field(
        ...,
        description="Prediction, weight and equity-curve hashes plus the "
        "dataset/estimator lineage and both specs — everything needed to "
        "reproduce the reported metrics.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Conditions that change how these metrics should be read: "
        "a look-ahead fill convention, rebalance dates dropped, books that "
        "could not reach target gross, an ambiguous annualization factor, plus "
        "the dataset coverage warnings carried from the model manifest and any "
        "raised by the simulator itself (insolvency, negative cash).",
    )


# ── list_models / list_datasets / compare_models ────────────────────────
#
# inspect_model and score_model both require a model_id the caller already
# holds. Nothing enumerated them, so a session that lost the id -- or a new
# session entirely -- could not find a model it had trained.


class ListModelsInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(extra="forbid")

    task: Optional[Literal["regression", "classification", "ranking"]] = Field(
        None, description="Only models trained for this task."
    )
    limit: int = Field(50, gt=0, le=500, description="Most recent first.")


class ModelSummary(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    task: str
    estimator: Optional[str] = None
    created_at: Optional[str] = None
    n_features: Optional[int] = None
    n_folds: Optional[int] = None
    headline_metric: Optional[str] = Field(
        None, description="Which metric `headline_value` reports."
    )
    headline_value: Optional[float] = None
    dataset_id: Optional[str] = None


class ListModelsResult(BaseModel):
    models: List[ModelSummary]
    n_total: int = Field(
        ..., description="Registered models before `limit` was applied."
    )
    registry_dir: str


class ListDatasetsInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(extra="forbid")

    limit: int = Field(50, gt=0, le=500)


class DatasetSummary(BaseModel):
    dataset_id: str
    rows: Optional[int] = None
    entities: Optional[int] = None
    features: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ListDatasetsResult(BaseModel):
    datasets: List[DatasetSummary]
    n_total: int
    runs_dir: str


class CompareModelsInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(extra="forbid")

    model_ids: List[str] = Field(
        ...,
        min_length=2,
        max_length=20,
        description="Models to rank side by side.",
    )
    metric: Optional[str] = Field(
        None,
        description=(
            "Metric to rank by. None picks each task's usual headline. "
            "Models trained for different tasks are reported but NOT ranked "
            "against each other — the metrics are not comparable."
        ),
    )

    @field_validator("model_ids")
    @classmethod
    def _distinct(cls, ids: List[str]) -> List[str]:
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(
                f"model_ids contains duplicates {duplicates}; a model "
                "compared against itself contributes nothing to a ranking."
            )
        return ids


class ModelComparison(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    task: str
    metric: Optional[str] = None
    value: Optional[float] = None
    rank: Optional[int] = Field(
        None, description="Within its own task. None when the metric is missing."
    )
    n_features: Optional[int] = None
    dataset_id: Optional[str] = None


class CompareModelsResult(BaseModel):
    comparisons: List[ModelComparison]
    best_by_task: Dict[str, str] = Field(
        default_factory=dict, description="task -> winning model_id."
    )
    notes: List[str] = Field(default_factory=list)


# ── check_leakage ───────────────────────────────────────────────────────


class CheckLeakageInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(extra="forbid")

    feature_ids: Optional[List[str]] = Field(
        None,
        description=(
            "Feature ids to check for temporal safety. Omit to check every "
            "feature in the registry."
        ),
    )
    dataset_id: Optional[str] = Field(
        None,
        description=(
            "Also report that dataset's point-in-time coverage — how much "
            "of the panel is genuinely as-of rather than back-filled."
        ),
    )


class LeakageFinding(BaseModel):
    feature_id: str
    temporal_support: str
    problem: str


class CheckLeakageResult(BaseModel):
    n_features_checked: int
    safe: bool
    findings: List[LeakageFinding] = Field(default_factory=list)
    dataset_coverage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Point-in-time coverage, when a dataset_id was supplied.",
    )
    notes: List[str] = Field(default_factory=list)


# ── validate_model_spec ─────────────────────────────────────────────────
#
# run_model_experiment is the most expensive call in the library: it fetches
# a universe, builds a panel, and fits once per walk-forward fold. A bad
# estimator parameter surfaced only after all of that. The registry has
# always known the answer in microseconds.


class ValidateModelSpecInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    spec: ModelSpec = Field(
        ..., description="The ModelSpec you intend to pass to run_model_experiment."
    )
    dataset_id: Optional[str] = Field(
        None,
        description=(
            "Also check the spec against a built dataset: that its features "
            "exist in the panel and that the target is present. Omit to "
            "check the spec alone."
        ),
    )


class SpecProblem(BaseModel):
    where: str = Field(
        ...,
        description="Which part of the spec — 'estimator', 'features', 'target', ...",
    )
    problem: str
    suggestion: Optional[str] = None


class ValidateModelSpecResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    valid: bool
    task: str
    estimator: str
    problems: List[SpecProblem] = Field(default_factory=list)
    allowed_estimator_params: List[str] = Field(
        default_factory=list,
        description="Every parameter this estimator accepts, from the registry.",
    )
    estimated_fits: Optional[int] = Field(
        None,
        description=(
            "Fits this spec implies: folds, times the search grid if one is "
            "set. The number that decides whether the experiment takes "
            "seconds or an afternoon."
        ),
    )
    notes: List[str] = Field(default_factory=list)


# ── score_predictions ───────────────────────────────────────────────────


class ScorePredictionsInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(extra="forbid")

    predictions_ref: str = Field(
        ...,
        description=(
            "A 'predictions' handoff reference — from run_model_experiment, "
            "or published by anything at all. Predictions computed entirely "
            "outside this library score the same way."
        ),
    )
    task: Literal["regression", "classification", "ranking"] = Field(
        ...,
        description=(
            "How to read the prediction column. Scoring a raw forward-return "
            "prediction as a probability produces numbers that look like "
            "metrics and are not."
        ),
    )
    target_column: str = Field(
        "target",
        description="Column holding the realized outcome each prediction is scored against.",
    )
    prediction_column: str = Field(
        "prediction", description="Column holding the prediction."
    )
    ic_method: Literal["spearman", "pearson"] = Field(
        "spearman",
        description=(
            "Rank correlation (default) or linear. Spearman is the usual "
            "choice for a cross-sectional alpha, where the ORDER is the "
            "claim and the magnitude is not."
        ),
    )
    ndcg_cutoffs: List[int] = Field(
        [5, 10], description="task='ranking' only: the k values for NDCG@k."
    )


class ScorePredictionsResult(BaseModel):
    task: str
    n_observations: int
    n_dates: int
    n_entities: int
    metrics: Dict[str, float] = Field(
        ..., description="Task-appropriate accuracy metrics."
    )
    cross_sectional_ic: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Mean IC, its standard deviation, ICIR and hit rate across "
            "dates. For a cross-sectional model this matters more than any "
            "pooled metric, which can look strong purely from time-series "
            "level differences."
        ),
    )
    baseline: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "The same metrics for predicting the training mean. A model that "
            "does not beat this has not learned anything, and a good-looking "
            "R2 next to a good-looking baseline usually means the target was "
            "easy rather than the model clever."
        ),
    )
    beats_baseline: Optional[bool] = None
    effective_sample_size: Optional[float] = Field(
        None,
        description=(
            "Observations adjusted for overlapping forward-return windows. "
            "A 20-day target sampled daily has far fewer independent "
            "observations than rows, and every t-statistic computed from the "
            "raw count is overstated."
        ),
    )
    notes: List[str] = Field(default_factory=list)
