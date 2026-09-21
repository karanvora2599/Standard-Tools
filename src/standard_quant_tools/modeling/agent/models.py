"""
Pydantic Input/Result models for the modeling agent surface.
DatasetSpec/ModelSpec (modeling.specs) are embedded directly as nested
fields rather than flattened — an LLM constructs one declarative spec
object per call, the same ModelSpec-not-exec() contract described in
Documentation/15_modeling.md, matching how agent/models.py's own Input
models nest structured params for the existing analysis surface.
"""

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from standard_quant_tools.agent.runtimes._json_safe import finite_or_none

from ..registry.lifecycle import LifecycleStage
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

#: A float statistic that may legitimately be absent, and that must never
#: reach the wire as `NaN`. `NaN` is not valid JSON, and several MCP
#: clients reject the whole response at the transport layer rather than
#: the one field that could not be computed -- so a statistic that did not
#: exist would fail every other number beside it. `feature_models.py:63`
#: carries the same alias for the feature-lab results.
Stat = Annotated[Optional[float], BeforeValidator(finite_or_none)]

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
    frame_kind: Optional[str] = Field(
        None,
        description=(
            "POINT_IN_TIME features only: which record set the provider is "
            "asked for -- 'fundamentals' and the like. None for every "
            "feature computed from bars alone. This is the field that says "
            "a feature needs a provider that serves that record set, which "
            "no other part of the catalog reveals: a spec naming one "
            "against a bars-only provider fails at build time, not here."
        ),
    )
    fields: List[str] = Field(
        default_factory=list,
        description=(
            "POINT_IN_TIME features only: the record fields the transform "
            "reads out of `frame_kind`. Empty for everything else. Read it "
            "with `default_params['max_staleness_days']`, the oldest record "
            "a panel row is still allowed to use."
        ),
    )


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
    event_column: Optional[str] = Field(
        None,
        description=(
            "For a censored label such as time_to_fill: the 0/1 column saying "
            "whether the event was observed (1) or the window ended first "
            "(0). Required by a censored target_type and fitted by "
            "task='survival'; without it an unfilled order would be read as "
            "filling at the horizon."
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
            "It carries date, entity and prediction and NO realized "
            "outcome, which is the difference between the two things you "
            "might do with it: backtest it through convert_reference -- "
            "which needs no outcome -- and it is an ordinary prediction "
            "frame; score it, and score_predictions refuses it for having "
            "no 'target' column. attach_model_outcomes("
            "predictions_ref=<this ref>, "
            "dataset_id=<the dataset the base models were fit on>) joins the "
            "realized label and publishes the reference that scores."
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
            "training rows whose label END falls inside a test window, and "
            "when no label_end_column is given the registration derives each "
            "row's label end from this horizon (that many rows ahead on the "
            "entity's own calendar). Without a horizon there is no label end "
            "and no purge. Supply this or `targets`, never both."
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
    event_column: Optional[str] = Field(
        None,
        description=(
            "For a panel with ONE censored label: the 0/1 column saying whether "
            "its event was observed. See ExternalTarget.event_column."
        ),
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
    data_sources: Dict[str, str] = Field(
        default_factory=dict,
        description="Entity -> the feed its bars came from, '<provider>:<dataset>' "
        "for a provider that chooses a dataset per window (Databento) and the "
        "provider name otherwise. Persisted in dataset_meta.json and carried onto "
        "every model trained from this dataset, so two builds of one spec from "
        "two feeds can be told apart.",
    )
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
    n_train_rows_purged_overlap: Optional[int] = Field(
        0,
        description="Training rows dropped because their label would have "
        "resolved inside the test window, purged on each row's recorded "
        "label_end_date. None when the panel carries no such column and the "
        "purge could not run at all -- that used to read 0, the same value a "
        "clean run gives, on a panel with 280 overlapping rows. A large count "
        "means the target horizon consumes a real fraction of each training "
        "window — relevant when reading the OOS metrics; "
        "validation_report.purge says which case this is.",
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
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Caveats about THIS RUN, as opposed to the dataset's -- those "
            "travel on the manifest as `dataset_warnings`. What lands here "
            "is a thing the run did that changes how a number above should "
            "be read: a calibrated estimator, for instance, whose "
            "`feature_importance_summary` is NaN by construction because "
            "the wrapper does not expose the base class's coefficients. "
            "The engine has always produced these; this result had no "
            "field for them, so they were dropped between the engine and "
            "the agent."
        ),
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
    universe_policy: Literal["strict", "allow"] = Field(
        "strict",
        description="What to do when the model standardizes within the scoring "
        "cross-section (a cross_sectional preprocessing step) and `universe` is "
        "not the training universe. 'strict' (default) refuses: every row's score "
        "depends on which other entities are in the call, so a subset is a "
        "different transform, not a smaller sample. 'allow' scores anyway and "
        "returns a warning saying the transform was refit on this cross-section "
        "and how its width compares with the training one. A model with "
        "universe-scope features is refused either way.",
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
    predictions_ref: Optional[str] = Field(
        None,
        description=(
            "Typed handoff reference for the same predictions "
            "(sqt://predictions/...). This is the one to pass onward: "
            "`handoff.resolve` refuses a raw artifact path under `expect=`, "
            "so `predictions_uri` alone could be monitored for drift and "
            "never scored against outcomes or traded. Feed it to "
            "attach_model_outcomes (which joins the realized label and makes "
            "it scoreable) or to convert_reference. Content-addressed and "
            "idempotent: re-scoring the same model, date and universe "
            "republishes the same reference."
        ),
    )
    predictions_hash: str = Field(
        "",
        description="Content digest of the written predictions artifact. The "
        "artifact path is content-addressed, so a URI recorded by one call "
        "always resolves to the bytes that call produced — re-scoring after a "
        "data revision writes a NEW path rather than replacing an older one an "
        "audit record still points at.",
    )
    n_entities: int
    features_uri: Optional[str] = Field(
        None,
        description="The raw feature rows these predictions were made from, "
        "written beside them so monitor_model can measure how far the scored "
        "universe has drifted from the training panel.",
    )
    summary_stats: Dict[str, float]
    interval_stats: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "The conformal band's WIDTH, when the model emits one: mean, "
            "median, min and max width, how many rows carry an interval, "
            "and the width as a multiple of the prediction spread. "
            "`summary_stats` describes the point prediction and says "
            "nothing about the interval beside it, so a band thirty times "
            "the width of the whole cross-section came back looking "
            "exactly like a tight one. Empty for a point-only model: an "
            "absent interval is not a zero-width one. No coverage figure "
            "-- coverage needs realized outcomes, and at `as_of` they do "
            "not exist yet."
        ),
    )
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
    warnings: List[str] = Field(
        default_factory=list,
        description="Conditions the caller waived or that qualify these scores: "
        "today, that a cross-sectional transform was refit on the scoring "
        "cross-section under universe_policy='allow', with the width it had "
        "against the training width. Empty when nothing qualifies them.",
    )


# ── promote_model / monitor_model ────────────────────────────────────────


class PromoteModelInput(BaseModel):
    """A lifecycle decision, recorded with its reason and evidence."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: str
    to_stage: LifecycleStage = Field(
        ...,
        description=(
            "Where to move the model: candidate -> validated -> staging -> "
            "production, one stage at a time, or 'archived' from anywhere "
            "(terminal). A demotion to an earlier live stage is allowed and "
            "recorded like any other decision."
        ),
    )
    reason: str = Field(
        ...,
        min_length=8,
        description="Why, in a sentence somebody can read months later.",
    )
    actor: str = Field("agent", min_length=1, description="Who decided.")
    evidence: List[str] = Field(
        default_factory=list,
        description="References the decision rests on: an evaluate_model_portfolio "
        "weights_uri, a monitor_model predictions_uri, a compare_models run.",
    )


class PromoteModelResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    from_stage: str
    to_stage: str
    timestamp_utc: str
    actor: str
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Every promotion recorded for this model, oldest first, "
        "from the append-only promotions.jsonl beside the manifest.",
    )


class MonitorModelInput(BaseModel):
    """Has the scored universe drifted from the training panel, and is the
    model still right where outcomes exist."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: str
    predictions_uri: str = Field(
        ..., description="A score_model predictions_uri for this model."
    )
    features_uri: Optional[str] = Field(
        None,
        description="The score_model features_uri that goes with it. Found "
        "beside the predictions by name when omitted.",
    )
    outcomes_ref: Optional[str] = Field(
        None,
        description=(
            "Optional: an artifact or sqt:// reference holding `entity` and "
            "`realized` (and `date` when the predictions span several dates) "
            "for the scored rows. With it the realized cross-sectional rank IC "
            "is reported beside the validation's."
        ),
    )


class FeatureDriftRow(BaseModel):
    feature: str
    psi: float = Field(
        ...,
        description="Population stability index against the "
        "training reference sample; NaN when undefined.",
    )
    ks: float = Field(..., description="Two-sample Kolmogorov-Smirnov statistic.")
    missing_rate_reference: float
    missing_rate_current: float
    status: Literal["stable", "moderate", "severe", "unknown"]


class MonitorModelResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    stage: str
    predictions_uri: str
    features_uri: Optional[str] = None
    n_scored: int
    feature_drift: List[FeatureDriftRow] = Field(default_factory=list)
    n_features_moderate: int = 0
    n_features_severe: int = 0
    prediction_drift: Dict[str, Any] = Field(
        default_factory=dict,
        description="PSI, KS and moments of the scored predictions against the "
        "model's out-of-sample prediction sample.",
    )
    realized_ic: Optional[Dict[str, Any]] = Field(
        None,
        description="With outcomes_ref: the realized cross-sectional rank IC, the "
        "validation's mean and dispersion, and how many validation standard "
        "deviations the realized value sits from the mean.",
    )
    thresholds: Dict[str, Any] = Field(
        default_factory=dict,
        description="The lines the statuses were read against. Conventions, "
        "reported with every number rather than instead of it.",
    )
    training_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The reference the drift numbers above were read AGAINST: per "
            "training feature, the quantile edges (`bins` + 1 of them), the "
            "missing rate, the mean and the standard deviation of the panel "
            "the model was fit on. A PSI of 0.3 says the distribution "
            "moved; only this says what it moved FROM, which is what "
            "decides whether the new regime is one the model ever saw. "
            "Recorded at registration, so it cannot be contaminated by the "
            "window being monitored."
        ),
    )
    overall_status: Literal["stable", "moderate", "severe", "unknown"]
    warnings: List[str] = Field(default_factory=list)


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
    view: Literal[
        "summary",
        "feature_importance",
        "validation",
        "lineage",
        "provenance",
    ] = Field(
        "summary",
        description=(
            "Which slice of the registered model to return. 'provenance' is "
            "the manifest's recorded identity: the training information "
            "cutoff score_model gates `as_of` on, the dataset spec hash and "
            "per-artifact content hashes, the per-column feature provenance "
            "scoring re-checks, whether a conformal band was deployed, and "
            "how the environment that fitted the model differs from this "
            "one. It verifies nothing and re-reads no artifact, so it is "
            "cheap where 'lineage' is not."
        ),
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


# ── backtest_model_signal ───────────────────────────────────────────────
#
# The VERIFIED route from a registered model to a backtest. The other one
# -- publish the predictions, convert_reference(task=...) -- reads a COPY
# of the artifact with no manifest behind it, so it cannot check the task
# and cannot check the digest. This input exists to make both of those
# unrepresentable rather than merely discouraged.


class BacktestModelSignalInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    #
    # There is deliberately NO `task` field. The manifest is the source of
    # the task, and with extra="forbid" a caller cannot even spell a
    # mismatch -- which is the one thing the other route to a backtest
    # accepts silently, thresholding raw forward returns against a
    # probability cutoff into an all-zero panel that backtests to
    # `sharpe nan` with no error anywhere.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: str = Field(
        ...,
        description=(
            "A model_id from run_model_experiment. Its manifest resolves the "
            "predictions artifact, the task and the recorded content digest "
            "together, so none of the three can disagree with the others."
        ),
    )
    run_id: str = Field(
        ...,
        description=(
            "Run id to publish the signal panel under. Letters, digits, '_' "
            "and '-' only."
        ),
    )
    name: str = Field(..., description="Artifact name for the published signal panel.")
    deadband: float = Field(
        0.0,
        ge=0.0,
        description=(
            "Score tasks only (regression and ranking): a prediction whose "
            "magnitude is at or below this becomes flat (0.0) instead of a "
            "full-size position on what is probably noise. 0 (default) takes "
            "every prediction's sign."
        ),
    )
    proba_threshold: float = Field(
        0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "Classification only: long above this probability. With "
            "long_only=False it must also be >= 0.5, because a symmetric "
            "decision boundary below the midpoint would make the long and "
            "short conditions overlap."
        ),
    )
    long_only: bool = Field(
        True,
        description=(
            "Classification only: treat the negative class as FLAT rather "
            "than short. 'Not predicted up' is not the same claim as "
            "'predicted down', which is why this is the default."
        ),
    )


class BacktestModelSignalResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    signal_panel_ref: str = Field(
        ...,
        description=(
            "An `sqt://signal_panel/...` reference holding {ticker: {date: "
            "value}} with every value exactly -1.0, 0.0 or 1.0. Pass it to "
            "run_signal_panel_backtest as `signal_panel_ref` with "
            "signal_type='direction' and fill_price='next_open'. Nothing was "
            "backtested here: the fill convention, the costs, the tickers "
            "and the date range are backtest decisions, and this runtime "
            "does not own them."
        ),
    )
    model_id: str
    task: str = Field(
        ...,
        description=(
            "Read from the manifest, never from the caller. A score task "
            "(regression or ranking) becomes the sign of the prediction; "
            "classification thresholds the positive-class probability."
        ),
    )
    entities: List[str] = Field(
        default_factory=list,
        description=(
            "The panel's outer keys, which is what "
            "SignalPanelBacktestInput.tickers must match."
        ),
    )
    n_dates: int = Field(
        0,
        description=(
            "Dates on the panel's shared calendar. Every entity carries all "
            "of them -- an entity with no prediction on a date is explicitly "
            "flat, because a hole would vanish from the price axis rather "
            "than reading as no position."
        ),
    )
    first_date: str = ""
    last_date: str = ""
    n_long: int = Field(0, description="Panel cells equal to 1.0.")
    n_flat: int = Field(
        0,
        description=(
            "Panel cells equal to 0.0, including the densified ones. A "
            "figure close to the whole panel is the symptom of a signal that "
            "sits inside its deadband or under its probability threshold "
            "almost everywhere -- which backtests to a flat curve rather "
            "than an error."
        ),
    )
    n_short: int = Field(0, description="Panel cells equal to -1.0.")
    oos_predictions_hash: Optional[str] = Field(
        None,
        description=(
            "The digest recorded in the manifest at registration, VERIFIED "
            "against the file before it was read. None only for a model "
            "registered before content hashing existed, where there was no "
            "root of trust to check against."
        ),
    )
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "The fill-price advisory this panel must be backtested under, "
            "plus the dataset coverage warnings carried from the model's "
            "manifest."
        ),
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
    stage: Optional[LifecycleStage] = Field(
        None,
        description="Only models at this lifecycle stage. A registered model "
        "is a 'candidate' until promote_model moves it.",
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
    stage: Optional[str] = Field(
        None, description="Where the model is in its lifecycle; see promote_model."
    )


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
    # Distinct dates in the panel. This is what decides how many
    # walk-forward folds a spec yields, which is why validate_model_spec
    # reads it and why it is recorded rather than re-derived from a panel
    # that would have to be loaded and hashed to answer.
    n_dates: Optional[int] = None
    provider: Optional[str] = None
    interval: Optional[str] = None
    target_id: Optional[str] = None


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
    method: Literal["headline", "paired"] = Field(
        "headline",
        description=(
            "'headline' (default): rank by each model's recorded OOS metric, "
            "which says which number is larger and nothing about whether the "
            "gap is larger than the noise in one OOS sample. 'paired': test "
            "every other model against `reference_model_id` on the rows both "
            "predicted -- a block-bootstrap interval on the per-date IC "
            "difference, a Diebold-Mariano test on the loss where the task has "
            "one, and Holm-adjusted p-values across the candidates. Paired "
            "comparison needs models trained on the SAME label; the realized "
            "outcomes must agree on every shared row."
        ),
    )
    reference_model_id: Optional[str] = Field(
        None,
        description=(
            "method='paired' only: the model every other one is compared "
            "against. Must be one of `model_ids`. Omitted, the first is used."
        ),
    )
    comparison_metric: Literal["cs_rank_ic", "cs_ic"] = Field(
        "cs_rank_ic",
        description="method='paired' only: the per-date correlation the "
        "difference is measured on.",
    )
    n_bootstrap: int = Field(
        2000,
        ge=100,
        le=20_000,
        description="method='paired' only: block-bootstrap resamples of the "
        "per-date difference.",
    )
    block_size: Optional[int] = Field(
        None,
        ge=1,
        description="method='paired' only: dates per bootstrap block. None "
        "uses n_dates^(1/3), the same rule get_bootstrap_interval uses; 1 is "
        "an IID resample, which understates the interval on an overlapping "
        "label.",
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

    @model_validator(mode="after")
    def _reference_is_a_candidate(self) -> "CompareModelsInput":
        if (
            self.reference_model_id is not None
            and self.reference_model_id not in self.model_ids
        ):
            raise ValueError(
                f"reference_model_id={self.reference_model_id!r} is not in "
                "model_ids; the reference is compared against the others and "
                "must be one of them."
            )
        return self


class PairedComparison(BaseModel):
    """One candidate against the reference, on the rows both predicted."""

    model_config = _NO_PROTECTED_NAMESPACES

    model_id: str
    reference_model_id: str
    metric: str
    n_dates: int
    n_rows: int
    mean_reference: float = Field(..., description="Mean per-date IC of the reference.")
    mean_candidate: float = Field(..., description="Mean per-date IC of the candidate.")
    mean_difference: float = Field(..., description="candidate minus reference.")
    ci_lower: float
    ci_upper: float
    confidence: float
    p_value: float = Field(..., description="Two-sided bootstrap p-value, unadjusted.")
    p_value_holm: float = Field(
        ..., description="The same, Holm-adjusted across every candidate in this call."
    )
    hit_rate: Optional[float] = Field(
        None,
        description="Share of DECIDED dates -- ties excluded -- on which the "
        "candidate's IC exceeded the reference's. None when every date tied, "
        "which is what two identical models produce and what a rate of 0.0 "
        "read as: the candidate losing every day.",
    )
    n_ties: int = Field(
        0,
        description="Dates on which the two per-date ICs were exactly equal.",
    )
    block_size: int
    verdict: Literal["candidate_better", "reference_better", "indistinguishable"]
    diebold_mariano: Optional[Dict[str, Any]] = Field(
        None,
        description="The loss test where the task has a loss with units; "
        "None for a ranker. A positive statistic means the candidate's loss "
        "is smaller.",
    )
    warnings: List[str] = Field(default_factory=list)


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
    method: Literal["headline", "paired"] = "headline"
    comparisons: List[ModelComparison]
    best_by_task: Dict[str, str] = Field(
        default_factory=dict, description="task -> winning model_id."
    )
    reference_model_id: Optional[str] = None
    pairs: List[PairedComparison] = Field(
        default_factory=list,
        description="method='paired' only: one entry per candidate against "
        "the reference.",
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
    safe: bool = Field(
        ...,
        description="No finding. Read `scope` before trusting it: without a "
        "dataset_id this rests on each feature's DECLARED temporal support, "
        "and a feature that reads its own target passes that check.",
    )
    scope: Literal["declared_temporal_support_only", "declared_and_empirical"] = Field(
        "declared_temporal_support_only",
        description="What `safe` rests on: the registry's declarations alone, "
        "or those plus the empirical lead-lag screen run on the built panel "
        "(a dataset_id was supplied and the panel carries a target).",
    )
    findings: List[LeakageFinding] = Field(default_factory=list)
    screen: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per screened feature column: IC at shift 0, peak ratio, "
        "persistence, whether it was flagged and why. Empty without a "
        "dataset_id.",
    )
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
            "Also check the spec against a built dataset: that the dataset's "
            "label is one the task can consume, and how many folds the "
            "validation spec yields over the dataset's own date axis -- which "
            "is what turns `estimated_fits` from a guess into a count. Omit "
            "to check the spec alone."
        ),
    )
    target: Optional[str] = Field(
        None,
        description=(
            "For a dataset registered with several labels: the label name the "
            "experiment would select, exactly as run_model_experiment's "
            "`target`. Omitted, the primary label is checked."
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
            "Estimator fits this spec implies: every fold's fit, every search "
            "candidate on every inner fold, each calibration fold, the "
            "full-panel refit and the search on the full panel that chooses "
            "its parameters -- the same plan run_model_experiment executes. "
            "The number that decides whether the experiment takes seconds or "
            "an afternoon. None when it cannot be known: a walk-forward fold "
            "count depends on the dataset's date axis, so without a "
            "`dataset_id` it is not estimated rather than guessed."
        ),
    )
    max_fits: Optional[int] = Field(
        None,
        description="The spec's budget.max_fits, the ceiling the experiment "
        "is checked against before its first fit.",
    )
    within_budget: Optional[bool] = Field(
        None,
        description="Whether estimated_fits fits under max_fits. False is "
        "reported as a problem at where='budget'; None when the fit count "
        "is unknown.",
    )
    estimated_folds: Optional[int] = Field(
        None,
        description=(
            "Folds the validation spec yields. Exact when a `dataset_id` "
            "supplies the date axis; for purged k-fold without one, the "
            "requested `n_splits`; for walk-forward without one, None."
        ),
    )
    notes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Conditions that do not make the spec invalid but change what "
            "gets built -- notably a calendar this dataset adopted from its "
            "universe's venue rather than being given one, which is part of "
            "the dataset's identity and was visible nowhere."
        ),
    )


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
    horizon: int = Field(
        1,
        ge=1,
        description=(
            "The label's forward horizon in bars, for the effective sample "
            "size under overlapping labels. 1 applies no adjustment; pass the "
            "horizon the target was built with (TargetSpec.horizon) and the "
            "count of independent observations is deflated accordingly."
        ),
    )
    train_mean: Optional[float] = Field(
        None,
        description=(
            "The mean of the TRAINING outcomes -- the constant a forecaster "
            "who had seen only the training data would have predicted. It "
            "is what the 'predict the mean' baseline should be built from, "
            "and the scored set's own mean is not a substitute: nobody "
            "knows the future window's average realized return in advance, "
            "so a baseline built from it is an ORACLE that a model is being "
            "held to a standard no real forecaster could meet. Omitted, the "
            "baseline falls back to that oracle and says so through "
            "`baseline_is_oracle=1.0`, whose R2 is then 0.0 by construction "
            "-- which is exactly why it is not a baseline."
        ),
    )
    event_column: str = Field(
        "event",
        description=(
            "task='survival' only: the 0/1 column saying whether each "
            "duration's event was observed. The prediction is read as a RISK, "
            "higher meaning sooner, and scored on concordance."
        ),
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
    prediction_turnover: Stat = Field(
        None,
        description=(
            "How much the signal's ORDERING moves from one date to the "
            "next: the mean absolute change in each entity's percentile "
            "rank between consecutive dates, in [0, 1]. The bridge between "
            "an IC and a net-of-cost P&L -- a 0.05 IC at 0.05 turnover and "
            "the same IC at 0.60 turnover are different strategies, and "
            "only the second one's costs can eat the whole edge. Zero for "
            "an ordering that never changes; a signal reshuffled at random "
            "every date sits near a third. None for a single entity, where "
            "there is no cross-section to reorder."
        ),
    )
    notes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Caveats that change how the numbers above should be read -- "
            "an oracle baseline, a turnover that would have to be paid on "
            "every date. `notes` carries the explanatory commentary; "
            "these are the ones that say a headline figure is not what it "
            "looks like."
        ),
    )


# ── attach_model_outcomes ───────────────────────────────────────────────
#
# What makes this library's own output scoreable. Neither
# run_model_experiment's reference (date, entity, prediction, lower, upper)
# nor build_model_ensemble's (date, entity, prediction) carries the
# realized label, so score_predictions refused both for having no 'target'
# column -- the library could build an ensemble and backtest it and could
# not produce one statistical number for it.


class AttachModelOutcomesInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it
    # had configured something -- the same failure strategy_params.py
    # exists to stop one layer down, at the boundary where a model is
    # the one choosing the names.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    model_id: Optional[str] = Field(
        None,
        description=(
            "PREFERRED. A registered model, whose manifest resolves the "
            "predictions, the dataset and the label it was actually fit on "
            "together -- so none of them can disagree. The predictions "
            "artifact is verified against its recorded digest on the way in."
        ),
    )
    predictions_ref: Optional[str] = Field(
        None,
        description=(
            "An `sqt://predictions/...` reference instead of a model: an "
            "ensemble from build_model_ensemble, a scored universe from "
            "score_model, or predictions this library never produced. "
            "Requires dataset_id, because nothing on a reference says which "
            "dataset's realized label its rows should be joined to."
        ),
    )
    dataset_id: Optional[str] = Field(
        None,
        description=(
            "The dataset carrying the realized outcomes. REQUIRED with "
            "predictions_ref and REFUSED with model_id, where it is read "
            "from the manifest rather than taken on trust."
        ),
    )
    target: Optional[str] = Field(
        None,
        description=(
            "Which declared label to join, by NAME ('h5', 'h30'), for a "
            "dataset registered with several horizons. Omitted on such a "
            "dataset the call is REFUSED rather than silently scored "
            "against the primary: the wrong label produces numbers that "
            "look fine and describe a different outcome. Unnecessary for a "
            "single-label dataset, and unnecessary with model_id, whose "
            "manifest names the label it was fit on."
        ),
    )
    run_id: str = Field(
        ...,
        description=(
            "Run id to publish the joined frame under. Letters, digits, '_' "
            "and '-' only."
        ),
    )
    name: str = Field(..., description="Artifact name for the joined frame.")

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "AttachModelOutcomesInput":
        if (self.model_id is None) == (self.predictions_ref is None):
            raise ValueError(
                "pass exactly one of model_id (preferred -- resolves the "
                "predictions, the dataset and the label together from the "
                "manifest) or predictions_ref."
            )
        if self.predictions_ref is not None and not self.dataset_id:
            raise ValueError(
                "dataset_id is required with predictions_ref: a predictions "
                "reference carries no realized outcome and nothing on it "
                "says which dataset's label these rows should be joined to. "
                "Pass the dataset the predictions were made against."
            )
        if self.model_id is not None and self.dataset_id is not None:
            raise ValueError(
                "dataset_id is read from the model's manifest, so passing it "
                "here could only contradict the model's own lineage. Pass it "
                "with predictions_ref, or pass model_id alone."
            )
        return self


class AttachModelOutcomesResult(BaseModel):
    model_config = _NO_PROTECTED_NAMESPACES

    ref: str = Field(
        ...,
        description=(
            "An `sqt://predictions/...` reference carrying exactly date, "
            "entity, prediction and target. This is what score_predictions "
            "reads: pass it as `predictions_ref` with the `task` and the "
            "`horizon` reported below."
        ),
    )
    task: Optional[str] = Field(
        None,
        description=(
            "What score_predictions should read the prediction column as. "
            "Read from the manifest on the model path. On the reference "
            "path there is no manifest, so this is the task the dataset's "
            "label admits when it admits exactly one, and None when it "
            "admits several -- a forward return is scoreable as regression "
            "or as ranking, and which one is being claimed is the caller's "
            "decision rather than the dataset's."
        ),
    )
    target_id: Optional[str] = Field(
        None,
        description=(
            "The label that was actually joined, in the manifest's own "
            "'<type>:<horizon>' spelling -- so what was scored is on the "
            "record beside the numbers."
        ),
    )
    horizon: Optional[int] = Field(
        None,
        description=(
            "Bars the joined label looks forward, parsed from target_id. "
            "Pass it straight to score_predictions' `horizon`: an "
            "overlapping forward return has far fewer independent "
            "observations than rows, and a t-statistic read off the raw "
            "count is overstated by roughly that factor. Currently an agent "
            "has to remember it from dataset-build time."
        ),
    )
    columns: List[str] = Field(
        default_factory=list,
        description="The published frame's columns, in order.",
    )
    n_rows: int = Field(0, description="Rows that matched an outcome.")
    n_predictions_unmatched: int = Field(
        0,
        description=(
            "Prediction rows with no outcome row on their (date, entity) -- "
            "dropped rather than carried as a null target. Nonzero is normal "
            "at the end of a sample, where the forward label has not "
            "resolved yet; a figure close to the whole frame usually means "
            "the predictions and the dataset are not about the same panel."
        ),
    )
    first_date: str = ""
    last_date: str = ""
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Label-selection notes, how many predictions found no outcome, "
            "and the dataset coverage warnings carried from the manifest."
        ),
    )


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
