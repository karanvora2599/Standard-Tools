"""
Pydantic Input/Result models for the 6-tool modeling agent surface.
DatasetSpec/ModelSpec (modeling.specs) are embedded directly as nested
fields rather than flattened — an LLM constructs one declarative spec
object per call, the same ModelSpec-not-exec() contract described in
Documentation/15_modeling.md, matching how agent/models.py's own Input
models nest structured params for the existing 46-tool surface.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..specs import (
    DatasetSpec,
    ModelSpec,
    PortfolioSimSpec,
    PredictionTransformSpec,
    TargetType,
    Task,
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


class ExternalTarget(BaseModel):
    """One label column in an externally computed panel."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description=(
            "What to call this horizon -- 'h1s', 'h30s'. Becomes a panel "
            "column, and is what run_model_experiment selects by."
        ),
    )
    column: str = Field(..., description="The column in the file holding it.")
    horizon: int = Field(
        ...,
        gt=0,
        description=(
            "Bars ahead THIS label was measured over. Per target, because "
            "that is the whole point of declaring several."
        ),
    )
    target_type: TargetType = Field(
        "forward_return",
        description=(
            "What this column already holds. Recorded, never recomputed -- "
            "which is what lets a microstructure label say what it is: a "
            "markout, a fill probability, a time to fill. Mislabelling one "
            "as a forward return puts a false claim in the manifest and "
            "leaves the task/target check unable to tell a probability from "
            "a return."
        ),
    )
    label_end_column: Optional[str] = Field(
        None,
        description=(
            "Column holding when THIS label's window closes, for a label "
            "that can end early."
        ),
    )


class BuildEnsembleInput(BaseModel):
    """Several registered models, combined into one prediction series."""

    model_config = ConfigDict(extra="forbid")

    model_ids: List[str] = Field(
        ...,
        min_length=2,
        max_length=20,
        description=(
            "Registered models to combine. Their OUT-OF-SAMPLE predictions "
            "are what gets combined -- each row predicted by a fold that did "
            "not train on it -- so the combination is honest whatever it "
            "does with them."
        ),
    )
    method: Literal["rank_mean", "mean", "median", "weighted"] = Field(
        "rank_mean",
        description=(
            "'rank_mean' (default) converts each model to a within-date rank "
            "first, so every model contributes its ORDERING and none "
            "contributes its variance -- two models on different scales "
            "would otherwise average into a number dominated by whichever "
            "has the wider spread, which is its units and not its skill. "
            "'mean' and 'median' combine the raw levels; 'weighted' takes "
            "the weights from you rather than learning them."
        ),
    )
    weights: Optional[List[float]] = Field(
        None,
        description=(
            "One per model, in the same order. Only read by "
            "method='weighted', and REFUSED with any other method rather "
            "than ignored."
        ),
    )
    run_id: str = Field(..., description="Groups this workflow's artifacts.")
    name: str = Field(..., description="Names the combined series within the run.")


class BuildEnsembleResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    ref: str = Field(
        ...,
        description=(
            "An `sqt://predictions/...` reference to the combined series. "
            "Score it with score_predictions, or backtest it through "
            "convert_reference -- it is an ordinary prediction frame."
        ),
    )
    model_ids: List[str] = Field(default_factory=list)
    method: str = ""
    task: str = ""
    n_rows: int = 0
    rows_per_model: Dict[str, int] = Field(default_factory=dict)
    rows_covered_by_all: int = Field(
        0,
        description="Rows every model predicted. Only these are combined, so "
        "a model validated over a shorter window shortens the ensemble.",
    )
    correlations: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Pairwise correlation between the base models' predictions. The "
            "number that says whether the ensemble was worth building: two "
            "models correlated at 0.98 average into approximately either of "
            "them, which the ensemble's own score cannot show you. Read it "
            "with `correlation_basis`, which says whether it was taken on "
            "ranks or on levels."
        ),
    )
    correlation_basis: str = Field(
        "",
        description=(
            "'rank' or 'level' -- which series `correlations` was computed "
            "on. `rank_mean` correlates the within-date RANKS, every other "
            "method the raw predictions, so the same field means a "
            "Spearman-like number in one case and a Pearson one in the "
            "other. Stated rather than left to be inferred from `method`."
        ),
    )
    warnings: List[str] = Field(default_factory=list)


class RegisterExternalPanelInput(BaseModel):
    """A finished feature matrix, and the one thing it cannot carry."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        ...,
        description=(
            "A Parquet or CSV file, or a directory read as one partitioned "
            "dataset, holding a long panel: one row per (bar, entity) with "
            "the feature columns and the label already computed. Nothing is "
            "copied -- the dataset record points at this path."
        ),
    )
    horizon: Optional[int] = Field(
        None,
        gt=0,
        description=(
            "Bars ahead the target was measured over, for a panel with ONE "
            "label. Not inferable and not defaulted: the engine purges "
            "training rows whose label window overlaps the test fold, and an "
            "absent horizon silently disables that purge rather than "
            "failing. Supply this or `targets`, never both."
        ),
    )
    targets: Optional[List[ExternalTarget]] = Field(
        None,
        min_length=1,
        description=(
            "Several labels in ONE panel, each with its own horizon. This is "
            "the microstructure case: a book is labelled at 1s, 5s and 30s "
            "simultaneously off identical features, and building one dataset "
            "per horizon would recompute and re-store the same matrix three "
            "times. Registered together they also stay COMPARABLE -- every "
            "model then sees the same rows and the same folds. "
            "run_model_experiment picks one by name."
        ),
    )
    target_type: TargetType = Field(
        "forward_return",
        description="What the target column already holds. Recorded, not recomputed.",
    )
    interval: str = Field(
        "1d",
        min_length=1,
        description=(
            "Bar interval of the panel's own rows -- '1d', '1s', '100ms'. "
            "Recorded on the spec, because `horizon` counts BARS of it and "
            "a horizon of 20 means a month at '1d' and two seconds at "
            "'100ms'."
        ),
    )
    date_column: str = Field("date", description="Column identifying the bar.")
    entity_column: str = Field("entity", description="Column identifying the symbol.")
    target_column: str = Field("target", description="Column holding the label.")
    label_end_column: Optional[str] = Field(
        None,
        description=(
            "Column holding when each label's window CLOSES. Supply it for a "
            "label that can end early -- a triple barrier -- so the purge "
            "uses the real end rather than the nominal horizon. Omit it for "
            "a fixed horizon."
        ),
    )
    feature_columns: Optional[List[str]] = Field(
        None,
        description=(
            "Which columns are features. Omitted, every column that is not "
            "the date, entity, target or label end is taken as one."
        ),
    )
    source: str = Field(
        "external",
        description="Where the panel came from, recorded on the dataset.",
    )
    file_format: Optional[Literal["parquet", "csv"]] = Field(
        None, description="Override the format inferred from the suffix."
    )

    @model_validator(mode="after")
    def _one_way_of_declaring_labels(self) -> "RegisterExternalPanelInput":
        if (self.horizon is None) == (self.targets is None):
            raise ValueError(
                "declare the panel's labels with EITHER `horizon` (one "
                "label, named by target_column) OR `targets` (several, each "
                "with its own horizon) -- not both and not neither. Both "
                "would make the precedence rule part of the contract, and "
                "neither leaves the purge with no horizon to purge on."
            )
        return self


class RegisterExternalPanelResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="Pass to run_model_experiment.")
    rows: int = 0
    entities: List[str] = Field(default_factory=list)
    feature_ids: List[str] = Field(default_factory=list)
    target_id: str = Field("", description="The PRIMARY target's id.")
    targets: List[str] = Field(
        default_factory=list,
        description=(
            "Every label this panel carries, by name. The first is the "
            "primary; run_model_experiment trains on it unless told another."
        ),
    )
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = ""
    source_path: str = Field("", description="Where the panel stayed.")
    fingerprint: str = Field(
        "",
        description=(
            "Name, size and mtime of the files behind it -- not a content "
            "hash. The content hash is recorded separately and IS verified "
            "on every load, because the engine reads the whole panel anyway."
        ),
    )
    warnings: List[str] = Field(default_factory=list)


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
    target: Optional[str] = Field(
        None,
        description=(
            "Which label to train on, for a dataset registered with several. "
            "Omitted, the primary is used. Rows whose CHOSEN label is null "
            "are dropped for this experiment only, so a long horizon costs "
            "its own rows and not the shorter ones'."
        ),
    )


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

    task: Optional[Task] = Field(None, description="Only models trained for this task.")
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
    task: Task = Field(
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


# ── point-in-time records ───────────────────────────────────────────────

#: A record frame arrives inline because no provider in this library serves
#: one yet. That is a real use case rather than a placeholder: a caller who
#: has FOMC dates, an earnings calendar or a set of index-membership changes
#: can join them onto a panel today. The cap is what stops somebody pasting
#: a whole vendor history through a JSON argument, which would work and be
#: a terrible way to move it.
MAX_INLINE_PIT_RECORDS = 5000


class PitRecordsInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    records: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=MAX_INLINE_PIT_RECORDS,
        description=(
            "Point-in-time records. Each needs `event_time` (when the fact "
            "is ABOUT -- the quarter end, the reference month), "
            "`available_time` (when it could first be ACTED ON -- the "
            "publication or release), plus `entity` unless the series is "
            "global, plus the value column(s). A value that was later "
            "restated is a SECOND ROW with the same event_time and a later "
            "available_time -- never an edit to the first."
        ),
    )
    entity_scoped: bool = Field(
        True,
        description="False for a global series -- CPI, Fed Funds, VIX -- "
        "which has no `entity` and joins to every entity on each date.",
    )


class PitValidationResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    valid: bool
    n_records: int
    n_entities: Optional[int] = None
    fields: List[str] = Field(
        default_factory=list, description="Value columns the records carry."
    )
    event_time_range: Optional[List[str]] = None
    available_time_range: Optional[List[str]] = None
    revisions: str = Field(
        "unknown",
        description="'versioned' when some fact carries more than one "
        "version, so a past decision is reproducible; 'unknown' when every "
        "fact appears once, which proves nothing either way.",
    )
    reproduces_history: bool = False
    median_publication_lag_days: Optional[float] = Field(
        None,
        description="Median (available_time - event_time). This is the "
        "hindsight a naive join on event_time would have given you, in days.",
    )
    problem: Optional[str] = Field(
        None, description="Why the records were rejected, if they were."
    )
    warnings: List[str] = Field(default_factory=list)


class JoinPointInTimeInput(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    records: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=MAX_INLINE_PIT_RECORDS,
        description="Point-in-time records -- see validate_pit_records, "
        "which checks the same input without joining anything.",
    )
    fields: Optional[List[str]] = Field(
        None,
        description="Value columns to attach. Defaults to every column that "
        "is not event_time, available_time or entity.",
    )
    entity_scoped: bool = Field(
        True, description="False for a global series joined to every entity."
    )
    prefix: str = Field(
        "", description="Namespace for the added columns, to avoid collisions."
    )
    max_staleness_days: Optional[int] = Field(
        None,
        ge=1,
        description="Refuse to carry a record older than this. Without it, a "
        "series that stops updating supplies its last value forever and the "
        "model learns from a number that stopped being a measurement.",
    )


class JoinPointInTimeResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    dataset_id: str
    joined_uri: str = Field(
        ..., description="sqt:// reference to the panel with the fields added."
    )
    n_rows: int
    fields_added: List[str]
    coverage: Dict[str, float] = Field(
        default_factory=dict,
        description="Fraction of panel rows that received a value, per field. "
        "A low number is not a failure -- it is how much of the panel "
        "predates the first release.",
    )
    warnings: List[str] = Field(default_factory=list)


class AnalyzeModelErrorsInput(BaseModel):
    """Where a registered model is wrong, not merely how wrong on average."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: str = Field(
        ...,
        description=(
            "A model registered by run_model_experiment. Its OUT-OF-SAMPLE "
            "predictions are the ones analysed -- each row predicted by a "
            "fold that did not train on it -- so these errors are the errors "
            "the model would have made."
        ),
    )
    feature: Optional[str] = Field(
        None,
        description=(
            "A column of the model's dataset panel to break errors down by, "
            "in deciles. This is the conditional question: does the model "
            "fail when the spread is wide, when volatility is high, when the "
            "book is thin. Omit it for the unconditional breakdowns only."
        ),
    )
    period: Literal["M", "Q", "Y"] = Field(
        "M",
        description=(
            "Calendar granularity for the by-period breakdown: month, "
            "quarter or year. Use a coarser one on a short sample, where "
            "monthly buckets are too thin to say anything."
        ),
    )
    top_n: int = Field(
        5,
        ge=1,
        le=50,
        description=(
            "How many buckets to return from each end of a breakdown, worst "
            "and best by RMSE. A 500-name universe produces 500 entity rows "
            "and the extremes are the whole finding; the rest are counted, "
            "not listed."
        ),
    )


class AnalyzeModelErrorsResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    task: str = ""
    target_id: str = ""
    n_rows: int = Field(
        0,
        description="Out-of-sample rows that matched an outcome in the "
        "dataset panel. The predictions frame carries no target column, so "
        "the actuals are joined back from the panel the model was fit on.",
    )
    residuals: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Shape of actual-minus-predicted: mean error (a non-zero mean is "
            "BIAS, which no amount of rank skill corrects), MAE, RMSE, the "
            "5th/95th percentiles, skew and excess kurtosis. A fat residual "
            "tail means the model is usually close and occasionally very "
            "wrong, which sizing from its average error will not survive."
        ),
    )
    calibration: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Whether the prediction's SCALE is right, which is a separate "
            "question from whether its ordering is. Regression/ranking: the "
            "slope and intercept of actual regressed on predicted -- slope 1, "
            "intercept 0 is calibrated, and below 1 means the predictions are "
            "spread wider than the outcomes. Classification: Brier score, "
            "expected calibration error and the reliability bins."
        ),
    )
    heteroskedasticity: Optional[float] = Field(
        None,
        description=(
            "Correlation between the absolute error and the prediction's "
            "magnitude. Positive means the model is least reliable exactly "
            "where it is most confident -- the direction that costs money, "
            "since the large predictions are the ones sized on."
        ),
    )
    residual_autocorrelation: Optional[float] = Field(
        None,
        description=(
            "Lag-1 residual autocorrelation, averaged over entities rather "
            "than computed on the stacked panel. EXPECTED to be positive for "
            "an overlapping target: a 20-bar forward return sampled every bar "
            "shares 19 bars with its neighbour, so consecutive residuals are "
            "correlated by construction. Read it as how few INDEPENDENT "
            "observations there were, not as misspecification."
        ),
    )
    by_entity: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Worst and best entities by RMSE, each with its row "
        "count -- a bias measured on nine rows is not a bias, so `thin` "
        "marks buckets under the floor.",
    )
    by_period: List[Dict[str, Any]] = Field(default_factory=list)
    by_prediction_decile: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Where in its OWN range the model is wrong. Accurate in "
        "the middle and wrong at the extremes is backwards for trading, "
        "because the extremes are the positions you take.",
    )
    by_feature_decile: List[Dict[str, Any]] = Field(default_factory=list)
    buckets_omitted: Dict[str, int] = Field(
        default_factory=dict,
        description="Buckets computed but not listed, per breakdown, because "
        "only the extremes were returned.",
    )
    findings: List[str] = Field(
        default_factory=list,
        description=(
            "The sentences the breakdown exists to produce -- which bucket "
            "is materially worse than the rest, and whether the model is "
            "biased or mis-scaled. Empty means the errors are spread evenly, "
            "which is itself the answer."
        ),
    )
    warnings: List[str] = Field(default_factory=list)
