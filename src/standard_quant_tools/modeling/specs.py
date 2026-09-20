"""
Declarative specs the modeling runtime executes — the ModelSpec-not-exec()
contract: an LLM (or any caller) builds one of these Pydantic objects and
hands it to `dataset.builder.build_dataset` / `engine.run_experiment`,
never arbitrary Python. Every field here is validated once, at the
boundary, the same discipline `agent/models.py` uses for the analysis
tool surface.
"""

import math
from typing import Annotated, ClassVar, Dict, List, Literal, Optional

import pandas as pd
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .features.base import RESERVED_PANEL_COLUMNS
from .limits import DEFAULT_MAX_FITS, MAX_FITS_CEILING, MAX_LAG, MAX_LAGS_PER_FEATURE


def _parse_date(value: str, field_name: str) -> pd.Timestamp:
    """Shared by DatasetSpec's start/end cross-check and
    modeling.agent.models.ScoreModelInput.as_of — raises the same
    ValueError shape pydantic validators elsewhere in this codebase use
    (e.g. PortfolioInput._check_weights), not a raw pandas parse error."""
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field_name}={value!r} is not a valid date: {exc}") from None


# ── The labels and the tasks live in registries, not here ──────────────
#
# `TARGET_KINDS`, `EXTERNAL_TARGETS`, `targets_for_task` and `TargetKind`
# used to be defined in this module, and the `TargetType` Literal beside
# them was pinned equal to the dict by test. They are re-exported from
# `modeling.targets` so every consumer keeps its import, and they are LIVE:
# a label registered through `register_target` is seen by the validator
# below, by the capability report and by the generated reference without
# anyone editing this file. `TASKS`/`Task`/`SCORE_TASKS` come from the
# `tasks` leaf for the reason its docstring gives.
from .targets import (  # noqa: F401  (re-exports; importing registers the built-ins)
    EXTERNAL_TARGETS,
    TARGET_KINDS,
    TargetKind,
    get_target,
    targets_for_task,
    validate_target_params,
)
from .tasks import SCORE_TASKS, TASKS, Task  # noqa: F401  (re-exports)


def _known_target_type(value: str) -> str:
    """A target id is a registry lookup, not a Literal member."""
    get_target(value)
    return value


def _target_choices(schema: Dict[str, object]) -> None:
    """
    Write the registered ids into the JSON schema as an enum.

    A Literal put the choices in the schema for free and could not be
    extended; a plain string can be extended and puts nothing in the
    schema. This is the third option: the schema is generated when a tool
    definition is built, which is after every registration, so an LLM
    reading it sees the same list the validator enforces.
    """
    schema["enum"] = sorted(TARGET_KINDS)


#: A registered target id. Validated against the registry and advertised
#: in the schema as an enum of whatever is registered when the schema is
#: built. Keeps the name every input model used for the Literal.
TargetType = Annotated[
    str,
    AfterValidator(_known_target_type),
    Field(json_schema_extra=_target_choices),
]


class FeatureSpec(BaseModel):
    """One requested feature: a `features.registry.FEATURE_REGISTRY` id
    plus caller-supplied overrides for that feature's `default_params`."""

    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Feature id, e.g. 'technical.rsi'.")
    params: Dict[str, object] = Field(
        default_factory=dict,
        description="Overrides merged onto the feature's default_params.",
    )
    alias: Optional[str] = Field(
        None,
        description=(
            "Column name for this feature in the output panel. Defaults to "
            "`id`. Supply one to request the SAME feature at more than one "
            "parameter setting — e.g. market.momentum at lookback 20 and 252 "
            "as 'mom_20' and 'mom_252', a completely standard multi-horizon "
            "model spec that was previously impossible because the panel "
            "keyed one column per feature id."
        ),
    )

    lags: List[Annotated[int, Field(ge=1, le=MAX_LAG)]] = Field(
        default_factory=list,
        max_length=MAX_LAGS_PER_FEATURE,
        description=(
            "Bars of HISTORY of this feature to add as extra columns, e.g. "
            "[1, 2, 3] adds its value 1, 2 and 3 bars ago as "
            "`<name>__lag1/2/3`. This is how a sequence reaches the "
            "estimator: the engine hands every estimator a 2-D matrix with "
            "no entity identity, so a model that wants yesterday's value "
            "cannot reconstruct it and the window has to be in the columns. "
            "Shifted within each entity, so a lag never reaches another "
            "entity's rows. A NEGATIVE lag is refused rather than clamped: "
            "it is a shift forward, which puts a future value on today's row "
            "and survives every leakage check that reasons about the target. "
            "Costs warm-up -- the deepest lag decides where the panel can "
            "start -- and columns, which multiply."
        ),
    )

    @field_validator("lags", mode="before")
    @classmethod
    def _lags_are_backward_and_bounded(cls, v):
        # Returned sorted and de-duplicated so [2, 1] and [1, 2] build the
        # SAME panel and hash to the same dataset id -- an ordering
        # difference must not silently create a second dataset.
        from .dataset.lags import validate_lags

        return validate_lags(v)

    @field_validator("alias")
    @classmethod
    def _alias_is_a_usable_column_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not v.strip():
            raise ValueError("alias must be a non-empty string")
        # These are reserved by the long panel's own schema; an alias
        # colliding with one would overwrite the column rather than add to it.
        if v in RESERVED_PANEL_COLUMNS:
            raise ValueError(
                f"alias={v!r} is reserved by the panel schema "
                "(date/entity/target/label_end_date)"
            )
        return v

    @property
    def output_name(self) -> str:
        """The panel column this feature produces. `id` when no alias is
        given, so every existing spec keeps its current column name."""
        return self.alias or self.id


class TargetSpec(BaseModel):
    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    type: TargetType = Field(
        "forward_return",
        description=(
            "'forward_return' (default) — continuous forward return, for "
            "task='regression'. 'forward_direction' — 1.0 when that forward "
            "return exceeds `threshold`, else 0.0, for task='classification'. "
            "'forward_return_vol_scaled' — that return divided by the entity's "
            "own trailing volatility, so a 2% move in a quiet name and a 2% "
            "move in a volatile one are not treated as equal evidence; an "
            "unscaled return target otherwise lets the highest-volatility "
            "names dominate the loss. 'forward_return_rank' — the return's "
            "rank within its date's cross-section mapped to [-0.5, 0.5], which "
            "matches how the model is SCORED (cross-sectional rank IC) and is "
            "immune to a fat-tailed return distribution. "
            "'forward_return_market_neutral' — the return minus that date's "
            "equal-weighted universe return, removing the market factor from "
            "the LABEL rather than hoping the model learns to ignore it. "
            "'triple_barrier' — 1.0 if an upper barrier is touched first, 0.0 "
            "if a lower one is, 2.0 if neither is touched within the horizon; "
            "for task='classification'. Those are three nominal class ids, not "
            "an ordered scale: 'up' is 1 so the predicted probability the "
            "downstream signal path reads is P(up)."
        ),
    )
    horizon: Optional[int] = Field(
        None, gt=0, description="Bars ahead the target return is measured over."
    )
    horizons: Optional[List[int]] = Field(
        None,
        min_length=1,
        max_length=12,
        description=(
            "Several horizons from ONE build, for when the same features "
            "answer the same question at more than one distance -- a "
            "microstructure panel labelled at 1, 5 and 30 bars at once. The "
            "features are computed once and the panel carries every label as "
            "`target__h<n>`, which is also what makes the resulting models "
            "COMPARABLE: each sees the same rows and the same folds. "
            "run_model_experiment picks one with `target`. Supply this or "
            "`horizon`, never both."
        ),
    )

    @model_validator(mode="after")
    def _one_way_of_saying_how_far(self) -> "TargetSpec":
        """
        Normalize so BOTH are always populated after validation.

        `horizon` is read in six places -- the forward return, the
        volatility scaling, the barrier walk, the label-end dates, the
        target id and the engine's purge -- and leaving it None for a
        multi-horizon spec would mean touching all six. Setting it to the
        first horizon instead means every one of them keeps working
        unchanged and reads the PRIMARY, which is exactly what they should
        read when no target has been selected.
        """
        if self.horizon is None and self.horizons is None:
            raise ValueError(
                "a target needs `horizon` (one distance) or `horizons` "
                "(several from one build); got neither, which leaves the "
                "walk-forward purge with no label window to purge on."
            )
        if self.horizon is not None and self.horizons is not None:
            # BOTH set is the NORMALIZED state, not a contradiction: this
            # validator populates the other one, and a spec is round-tripped
            # through `model_dump()` constantly -- `dataset_spec_hash`
            # rebuilds one to re-derive the hash, and every persisted
            # dataset_spec.json is read back the same way. Rejecting it
            # outright made a spec unable to survive its own serialization.
            #
            # What is still rejected is the pair DISAGREEING, which is a
            # caller saying two different things about the same label.
            first = sorted({int(h) for h in self.horizons})[0]
            if int(self.horizon) != first:
                raise ValueError(
                    f"horizon={self.horizon} and horizons={self.horizons} "
                    "disagree: the primary horizon is the smallest of "
                    f"`horizons` ({first}). Supply one or the other, or make "
                    "them agree."
                )
        if self.horizons is None:
            object.__setattr__(self, "horizons", [int(self.horizon)])
        else:
            ordered = sorted({int(h) for h in self.horizons})
            if len(ordered) != len(self.horizons):
                raise ValueError(
                    f"horizons={self.horizons} repeats a value. Each becomes "
                    "a panel column, and two of one would overwrite rather "
                    "than add."
                )
            object.__setattr__(self, "horizons", ordered)
            object.__setattr__(self, "horizon", ordered[0])
        return self

    @property
    def horizon_names(self) -> List[str]:
        """The panel name of each horizon; the first is the primary."""
        return [f"h{h}" for h in (self.horizons or [])]

    threshold: float = Field(
        0.0,
        description=(
            "forward_direction only: the forward return a bar must EXCEED to "
            "be labelled 1.0. Default 0.0 = plain up/down. A positive value "
            "(e.g. 0.02) asks for a move of at least that size, which also "
            "makes the classes deliberately imbalanced — check the resulting "
            "class balance before reading accuracy."
        ),
    )

    vol_window: int = Field(
        20,
        gt=1,
        description="forward_return_vol_scaled and triple_barrier: bars of "
        "trailing return history used for the volatility scale.",
    )
    barrier: float = Field(
        0.0,
        ge=0.0,
        description="triple_barrier only: the symmetric barrier as a fraction "
        "of the entry price (0.05 = +/-5%). Left at 0.0 the barriers are set "
        "from `vol_window` trailing volatility scaled to the horizon, which is "
        "the volatility-adaptive form — a fixed 5% barrier is a coin flip in a "
        "quiet name and unreachable in a volatile one.",
    )

    @model_validator(mode="after")
    def _threshold_only_for_direction(self) -> "TargetSpec":
        # Read off the registry rather than restated. The set used to be
        # inlined here and listed four of the six types that existed then,
        # so a new continuous label would have silently been allowed a
        # threshold that means nothing for it.
        kind = TARGET_KINDS.get(self.type)
        if kind is not None and kind.continuous and self.threshold != 0.0:
            raise ValueError(
                "threshold applies to a binarized target ('forward_direction' "
                f"or 'triple_barrier'); {self.type!r} is a continuous value."
            )
        if not math.isfinite(self.threshold):
            raise ValueError(f"threshold must be finite, got {self.threshold}")
        # A custom label's parameters, against the bounds it registered.
        # The built-ins register none, so any `params` on one is refused
        # by name rather than silently ignored.
        validate_target_params(self.type, dict(self.params))
        return self

    params: Dict[str, object] = Field(
        default_factory=dict,
        description=(
            "Parameters of a label registered with its own bounds -- a "
            "firm's residual return, an earnings drift -- merged onto that "
            "label's defaults. The built-in labels take none: their "
            "parameters are the fields above (horizon, threshold, "
            "vol_window, barrier)."
        ),
    )

    @property
    def resolved_params(self) -> Dict[str, object]:
        """The label's defaults with this spec's overrides on top -- what a
        registered builder reads."""
        return {**get_target(self.type).default_params, **dict(self.params)}


class MissingDataSpec(BaseModel):
    """
    What happens to a row whose feature is missing, decided at the DATASET
    level because two of the three answers are dataset operations.

    Complete-case alignment -- `drop`, the default and the only behaviour
    there was -- is conservative and honest, and on a 50-200 feature panel
    it is also expensive: one feature's warm-up or one halted bar costs the
    whole row for every other feature. The two alternatives recover rows
    without inventing observations:

      forward_fill_bounded  carry a feature's last value forward, within
                            the entity, for at most `max_staleness_bars`,
                            and only for the features named. A carried
                            value is a STALE one, which is defensible for a
                            slowly-updating level and not for a bar's
                            volume; the allowlist is what makes the caller
                            say which. Warm-up NaN has no prior value and
                            is never fabricated. Rows still missing after
                            the fill are dropped as under `drop`.
      keep                  keep the row, keep the NaN, drop on the target
                            only. The hole then reaches the engine, where a
                            fold-fitted `impute` step or a `missing_indicator`
                            handles it with training-fold statistics -- the
                            fold layer's half of the policy -- or an
                            estimator that accepts missing values reads it
                            directly. An estimator that does not is refused
                            by name before any fold is fitted.
    """

    model_config = ConfigDict(extra="forbid")

    policy: Literal["drop", "forward_fill_bounded", "keep"] = Field(
        "drop",
        description=(
            "'drop' (default): complete-case alignment, a row survives only "
            "when every feature and the target are present. "
            "'forward_fill_bounded': carry the named features' last value "
            "forward within each entity for at most max_staleness_bars, then "
            "drop what is still missing. 'keep': keep rows with missing "
            "features and drop on the target only; pair with an `impute` or "
            "`missing_indicator` preprocessing step, or an estimator that "
            "accepts missing values."
        ),
    )
    max_staleness_bars: int = Field(
        0,
        ge=0,
        le=MAX_LAG,
        description=(
            "forward_fill_bounded only: how many bars a value may be carried. "
            "Counted in the entity's own bars. Bounded like a lag, and for "
            "the same reason: a value that old describes a different regime."
        ),
    )
    features: List[str] = Field(
        default_factory=list,
        description=(
            "forward_fill_bounded only: the feature output names (alias, or "
            "id when there is none) that may be carried. Explicit rather "
            "than 'all', because carrying a stale value is a claim about the "
            "feature's semantics that only the caller can make."
        ),
    )

    @model_validator(mode="after")
    def _fields_match_the_policy(self) -> "MissingDataSpec":
        if self.policy == "forward_fill_bounded":
            if self.max_staleness_bars < 1:
                raise ValueError(
                    "missing.policy='forward_fill_bounded' needs "
                    "max_staleness_bars >= 1: a bound of zero carries nothing."
                )
            if not self.features:
                raise ValueError(
                    "missing.policy='forward_fill_bounded' needs `features`, the "
                    "output names allowed to be carried. Carrying every feature "
                    "would assert that a stale value is a fair stand-in for each "
                    "of them, which is not true of a volume."
                )
        else:
            # Checked against the defaults, not against which fields were
            # set, so a spec survives its own model_dump() round trip.
            if self.max_staleness_bars != 0 or self.features:
                raise ValueError(
                    f"missing.policy={self.policy!r} does not read "
                    "max_staleness_bars or features; they belong to "
                    "'forward_fill_bounded'. Drop them, or change the policy."
                )
        return self


class DatasetSpec(BaseModel):
    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    # max_length alongside min_length: universe fetching creates a task per
    # symbol, and while a semaphore bounds how many run at once it does not
    # bound how many are created. One valid-looking tool call could
    # therefore request an unbounded workload -- the same
    # agent-triggerable resource-exhaustion path the estimator registry's
    # parameter ceilings close. 1000 is far above any realistic modeling
    # universe and is a budget, not a modeling opinion.
    universe: List[str] = Field(
        ..., min_length=1, max_length=1000, description="Ticker symbols."
    )
    start: str = Field(..., description="Start date YYYY-MM-DD.")
    end: str = Field(..., description="End date YYYY-MM-DD.")
    features: List[FeatureSpec] = Field(..., min_length=1)
    target: TargetSpec
    benchmark: str = Field(
        "SPY",
        min_length=1,
        description="Benchmark symbol — only consumed by features that need one "
        "(e.g. risk.rolling_beta).",
    )
    provider: Literal["yfinance", "polygon", "bloomberg", "databento", "external"] = (
        Field(
            "yfinance",
            description=(
                "Data provider for this dataset. Previously hardcoded to the "
                "DataFactory default, so a model could not be built on anything "
                "else and its lineage never recorded which source it came from. "
                "Credentials are deliberately NOT part of this spec — it is "
                "persisted to disk, hashed into the model's lineage and written "
                "into decision records, so an api_key field here would leak the "
                "key into all three. Providers read their own credentials from "
                "the environment (e.g. SQT_POLYGON_API_KEY). "
                "'external' is not a provider at all: it marks a panel whose "
                "features were computed OUTSIDE this library and registered by "
                "register_external_panel, so nothing here can rebuild them — "
                "which is why score_model refuses such a model by name rather "
                "than recomputing features it does not have the definitions for."
            ),
        )
    )
    interval: str = Field(
        "1d",
        min_length=1,
        description=(
            "Bar interval passed to the provider, e.g. '1d' (default), '1h', "
            "'1wk'. The VALUE is validated by the selected provider, which "
            "owns the authoritative list — they differ (BloombergProvider "
            "rejects intraday outright). Note that `target.horizon` and every "
            "feature's lookback count BARS of this interval, and that the "
            "built-in features' default parameters and annualization "
            "constants are calibrated for daily bars: window=252 means one "
            "year at '1d' and about six weeks at '1h'. build_model_dataset "
            "warns when this is not '1d' rather than silently reinterpreting "
            "those defaults."
        ),
    )
    calendar: Optional[str] = Field(
        None,
        description=(
            "The venue's exchange calendar, as an `exchange_calendars` name "
            "such as 'XNYS', 'XLON' or '24/7'. What makes an INTRADAY "
            "interval annualizable: bars per session and sessions per year "
            "are read off it, so a volatility at '1h' is scaled by the bars "
            "this venue actually has rather than by a constant chosen for "
            "another. Optional; a daily-or-coarser interval needs none, and "
            "an intraday feature that annualizes refuses without one. Needs "
            "the optional exchange_calendars package."
        ),
    )

    @field_validator("calendar")
    @classmethod
    def _calendar_is_a_known_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        # Lazy: the calendar module imports nothing from specs, but the
        # library it wraps is optional and is only needed once a spec
        # names a calendar.
        from .calendar import validate_calendar_name

        return validate_calendar_name(v, "DatasetSpec.calendar")

    missing: MissingDataSpec = Field(
        default_factory=MissingDataSpec,
        description=(
            "What happens to a row whose feature is missing. The default, "
            "'drop', is complete-case alignment and the only behaviour there "
            "was; the alternatives recover rows without inventing "
            "observations. See MissingDataSpec. A dataset built with the "
            "default hashes identically to one built before this field "
            "existed."
        ),
    )

    @field_validator("universe")
    @classmethod
    def _no_duplicate_symbols(cls, v: List[str]) -> List[str]:
        dupes = sorted({s for s in v if v.count(s) > 1})
        if dupes:
            raise ValueError(f"universe contains duplicate symbols: {dupes}")
        return v

    @field_validator("features")
    @classmethod
    def _no_duplicate_output_names(cls, v: List["FeatureSpec"]) -> List["FeatureSpec"]:
        """
        Uniqueness is enforced on the OUTPUT COLUMN, not the feature id.

        Keying on the id meant momentum(20) + momentum(252) — an ordinary
        multi-horizon spec — was rejected outright. What actually cannot
        collide is the panel column name, so that is what is checked; an
        `alias` distinguishes repeated uses of one feature.
        """
        names = [f.output_name for f in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if not dupes:
            return v

        # Distinguish the two causes: a genuine alias collision is a
        # different mistake from repeating a feature without aliasing it.
        repeated_ids = sorted(
            {f.id for f in v if [s.id for s in v].count(f.id) > 1 and f.alias is None}
        )
        if repeated_ids:
            raise ValueError(
                f"features would produce duplicate panel column(s): {dupes}. "
                f"Feature id(s) {repeated_ids} are requested more than once without an "
                "alias — give each use a distinct `alias` (e.g. "
                "FeatureSpec(id='market.momentum', params={'lookback': 20}, "
                "alias='mom_20') alongside alias='mom_252')."
            )
        raise ValueError(
            f"features would produce duplicate panel column(s): {dupes} — two aliases "
            "(or an alias and another feature's id) resolve to the same column name."
        )

    @model_validator(mode="after")
    def _start_before_end(self) -> "DatasetSpec":
        start_ts = _parse_date(self.start, "start")
        end_ts = _parse_date(self.end, "end")
        if start_ts >= end_ts:
            raise ValueError(
                f"start ({self.start!r}) must be before end ({self.end!r})"
            )
        return self

    @model_validator(mode="after")
    def _fillable_features_exist(self) -> "DatasetSpec":
        """A feature named for forward filling must be one this spec
        produces, by its OUTPUT name -- the alias when there is one."""
        names = {f.output_name for f in self.features}
        unknown = sorted(set(self.missing.features) - names)
        if unknown:
            raise ValueError(
                f"missing.features names {unknown}, which this spec does not "
                f"produce. Its feature output names are {sorted(names)}. Name "
                "the alias where one is set."
            )
        return self


class EstimatorSpec(BaseModel):
    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description="Estimator name — must be in estimators.registry.ESTIMATOR_REGISTRY.",
    )
    params: Dict[str, object] = Field(default_factory=dict)
    calibration: Literal["none", "isotonic", "sigmoid"] = Field(
        "none",
        description=(
            "CLASSIFICATION ONLY. Map the estimator's raw scores onto "
            "probabilities that mean what they say, fitted on held-out folds "
            "inside each training window. "
            "This matters because `proba_threshold` is a live path: a random "
            "forest's probabilities are compressed toward the middle, an "
            "artefact of averaging trees, so it never emits one above about "
            "0.9. Measured on a noisy synthetic signal, at "
            "proba_threshold=0.9 the raw forest selected ZERO rows and the "
            "caller got an empty signal panel with no error anywhere, while "
            "the isotonic-calibrated one selected 194 at a realized hit rate "
            "of 0.912. "
            "'isotonic' is non-parametric and the usual choice with a few "
            "thousand rows; 'sigmoid' (Platt) fits two parameters and is "
            "safer on a short history, where isotonic will happily "
            "interpolate noise. 'none' (default) leaves scores untouched, "
            "which is right when you rank on them and never threshold."
        ),
    )
    calibration_folds: int = Field(
        3,
        ge=2,
        le=10,
        description=(
            "Inner folds used to fit the calibration map. Fitting it on the "
            "same rows the estimator trained on would calibrate against "
            "memorized labels and report a confidence nobody has."
        ),
    )


class ValidationSpec(BaseModel):
    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    method: Literal["walk_forward", "purged_kfold", "cpcv"] = Field(
        "walk_forward",
        description=(
            "'walk_forward' (default) — train on the past, test on the "
            "immediate future, repeatedly. The only scheme here that simulates "
            "live trading, and the one to quote a return from. 'purged_kfold' "
            "— K contiguous test blocks covering every date exactly once, with "
            "overlapping labels purged and an embargo on both sides. Uses a "
            "short history far better and is not dominated by the end of the "
            "sample, but later folds train partly on data that postdates their "
            "test block, so it answers 'is there a signal here', not 'what "
            "would this have earned'. 'cpcv' — combinatorial purged CV: every "
            "choice of n_test_splits blocks out of n_splits is a test set, so "
            "the OOS metric has a DISTRIBUTION across C(n_splits, n_test_splits) "
            "paths rather than one draw; that is the number to select a model "
            "on. Like purged_kfold it is not a trading simulation, and a cpcv "
            "model is refused by evaluate_model_portfolio and the bridge."
        ),
    )
    n_test_splits: int = Field(
        2,
        ge=1,
        description=(
            "cpcv only: how many of the n_splits blocks form each test set. "
            "The path count is C(n_splits, n_test_splits), bounded at 60 -- "
            "six choose two is 15 paths; eight choose two is 28."
        ),
    )
    scheme: Literal["rolling", "expanding"] = Field(
        "rolling",
        description=(
            "walk_forward only. 'rolling' (default) keeps the training window a "
            "fixed length so every fold is fit on comparable data. 'expanding' "
            "anchors it at the start of the sample and lets it grow, which "
            "stops a short history being discarded but makes a trend across "
            "folds mix skill with sample size."
        ),
    )
    n_splits: int = Field(
        5,
        ge=2,
        description="purged_kfold and cpcv: how many contiguous blocks the date "
        "axis is cut into.",
    )
    train_window: Optional[int] = Field(
        None, gt=0, description="Bars per training fold (walk_forward)."
    )
    test_window: Optional[int] = Field(
        None, gt=0, description="Bars per test fold (walk_forward)."
    )
    embargo: int = Field(
        0,
        ge=0,
        description="Bars excluded between train and test folds to prevent "
        "lookback leakage across the boundary. Note this does NOT need to "
        "cover the target horizon: training rows whose forward-return label "
        "would resolve inside the test window are purged separately, using "
        "each row's own label end date.",
    )
    min_folds: int = Field(
        2,
        ge=1,
        description="Minimum walk-forward folds that must actually COMPLETE "
        "before a model is registered. One surviving fold is a single "
        "train/test split, not walk-forward validation — it cannot show "
        "whether performance holds across time, which is the entire reason "
        "for validating this way. Lower to 1 only for a deliberately short "
        "exploratory run.",
    )

    @model_validator(mode="after")
    def _windows_required_for_walk_forward(self) -> "ValidationSpec":
        if self.method == "walk_forward":
            missing = [
                name
                for name in ("train_window", "test_window")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"method='walk_forward' requires {' and '.join(missing)} "
                    "(method='purged_kfold' does not, since its fold sizes come "
                    "from n_splits)."
                )
        return self

    #: The most paths a cpcv spec may imply. Each path is a full fit, so
    #: this is a compute budget of the same kind the estimator bounds are:
    #: 60 keeps eight choose two (28) and ten choose two (45) reachable and
    #: refuses fourteen choose seven (3,432) before it starts.
    MAX_CPCV_PATHS: ClassVar[int] = 60

    @model_validator(mode="after")
    def _cpcv_paths_are_bounded(self) -> "ValidationSpec":
        from math import comb

        if self.method == "cpcv":
            if self.n_test_splits >= self.n_splits:
                raise ValueError(
                    f"method='cpcv' needs n_test_splits ({self.n_test_splits}) "
                    f"fewer than n_splits ({self.n_splits}); a test set of every "
                    "block leaves nothing to train on."
                )
            paths = comb(self.n_splits, self.n_test_splits)
            if paths > self.MAX_CPCV_PATHS:
                raise ValueError(
                    f"method='cpcv' with n_splits={self.n_splits} and "
                    f"n_test_splits={self.n_test_splits} implies {paths} paths, "
                    f"each a full fit, past the ceiling of {self.MAX_CPCV_PATHS}. "
                    "Six choose two is 15 paths and is the usual choice."
                )
        # Checked against the default rather than against which fields were
        # set, so a spec survives its own model_dump() round trip.
        elif self.n_test_splits != 2:
            raise ValueError(
                f"n_test_splits={self.n_test_splits} is read by method='cpcv' "
                f"only; method={self.method!r} does not use it."
            )
        return self


class StepSpec(BaseModel):
    """One preprocessing step: a `PREPROCESSOR_REGISTRY` id plus overrides
    for that step's bounded parameters -- the same shape as a FeatureSpec
    against the feature registry."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description=(
            "Step id, e.g. 'winsorize', 'zscore', 'cross_sectional_standardize'. "
            "list_modeling_capabilities lists every registered step with its "
            "parameters."
        ),
    )
    params: Dict[str, object] = Field(
        default_factory=dict,
        description="Overrides merged onto the step's default parameters; "
        "names and values are checked against the step's bounds.",
    )

    @model_validator(mode="after")
    def _known_step_with_valid_params(self) -> "StepSpec":
        # Lazy import: the preprocessing package imports the transforms,
        # which do not import specs, but keeping the registry out of this
        # module's import graph is what keeps `specs` a leaf.
        from .preprocessing.registry import validate_step_params

        validate_step_params(self.type, dict(self.params))
        return self


class PreprocessingSpec(BaseModel):
    """
    How feature columns are transformed before the estimator sees them.

    Either a `steps` pipeline composed from the preprocessor registry, or --
    the original form -- a `normalization` scheme that resolves to one.
    `resolved_steps` is what the engine runs in both cases.
    """

    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    steps: List[StepSpec] = Field(
        default_factory=list,
        max_length=16,
        description=(
            "An explicit pipeline, applied in order: each step is fitted on "
            "the fold's TRAINING rows and its state applied unchanged to the "
            "test rows, then persisted with the model so scoring applies the "
            "same state. Empty (default) means `normalization` decides: "
            "'pooled' is [winsorize(0.01, 0.99), zscore] and 'cross_sectional' "
            "is [cross_sectional_standardize(clip_sigma)]. When `steps` is "
            "given, `normalization` and `clip_sigma` must be left at their "
            "defaults -- a pipeline and a scheme that disagree would be two "
            "claims about one transform."
        ),
    )

    normalization: Literal["pooled", "cross_sectional"] = Field(
        "pooled",
        description=(
            "'pooled' (default, and the original behaviour) — one mean and "
            "standard deviation fitted over the whole training panel. "
            "'cross_sectional' — standardize within each date, so what reaches "
            "the model is each entity's position relative to its peers that "
            "day. Pooled normalization leaves the market factor inside every "
            "feature, which lets a model score well by learning 'today was an "
            "up day' rather than 'this name is strong relative to its peers' — "
            "for a model judged on cross-sectional IC that is the wrong thing "
            "to have learned. It is not the default only because switching it "
            "changes what every existing model predicts."
        ),
    )
    clip_sigma: float = Field(
        3.0,
        ge=0.0,
        description="cross_sectional only: clip standardized features at this "
        "many standard deviations (0 disables). Replaces the pooled path's "
        "1st/99th percentile winsorizing, which is meaningless within a single "
        "date — the 1st percentile of a 20-name cross-section is its minimum, "
        "so clipping to it would do nothing at all.",
    )

    @model_validator(mode="after")
    def _steps_and_scheme_do_not_disagree(self) -> "PreprocessingSpec":
        """
        With `steps` given, the scheme fields must be at their defaults.

        Checked against the DEFAULT VALUES rather than against which fields
        were set, because a spec round-trips through `model_dump()` on
        every persist and reload, and a dump writes every field -- so
        "was normalization passed" is not a question a reloaded spec can
        answer, while "does it say something other than the default" is.
        """
        if self.steps and self.normalization != "pooled":
            raise ValueError(
                f"preprocessing.steps was given together with normalization="
                f"{self.normalization!r}. The steps ARE the pipeline; a scheme "
                "beside them would be a second claim about the same transform. "
                "Drop `normalization`, or express it as a step "
                "(cross_sectional_standardize)."
            )
        if self.steps and self.clip_sigma != 3.0:
            raise ValueError(
                f"preprocessing.steps was given together with clip_sigma="
                f"{self.clip_sigma}. clip_sigma is read only by "
                "normalization='cross_sectional'; with steps, pass it as the "
                "cross_sectional_standardize step's own parameter."
            )
        return self

    @property
    def resolved_steps(self) -> List[StepSpec]:
        """The pipeline the engine runs: `steps` when given, else the
        scheme's translation. Read here and nowhere else, so the two forms
        cannot be interpreted differently by two consumers."""
        if self.steps:
            return list(self.steps)
        if self.normalization == "cross_sectional":
            return [
                StepSpec(
                    type="cross_sectional_standardize",
                    params={"clip_sigma": self.clip_sigma},
                )
            ]
        return [
            StepSpec(type="winsorize", params={"lower": 0.01, "upper": 0.99}),
            StepSpec(type="zscore"),
        ]

    def resolved_dump(self) -> Dict[str, object]:
        """`model_dump()` with `steps` replaced by the RESOLVED pipeline --
        what the manifest records, so a reader sees what ran rather than
        the scheme that implied it."""
        dumped = self.model_dump()
        dumped["steps"] = [s.model_dump() for s in self.resolved_steps]
        return dumped


class WeightingSpec(BaseModel):
    """How much each training row counts."""

    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "none", "label_uniqueness", "time_decay", "uniqueness_and_time_decay"
    ] = Field(
        "none",
        description=(
            "'none' (default) — every row at weight 1. 'label_uniqueness' — "
            "weight by the average uniqueness of each row's label, correcting "
            "for overlapping forward returns making consecutive rows largely "
            "redundant; this is the quantity effective_sample_size already "
            "reports and that nothing acted on. 'time_decay' — exponential "
            "decay in calendar time, for a relationship that drifts. "
            "'uniqueness_and_time_decay' — both."
        ),
    )
    half_life_days: float = Field(
        252.0,
        gt=0.0,
        description="time_decay only: calendar days after which a row's weight "
        "halves. Days rather than bars, so the intent survives a change of data "
        "frequency.",
    )


class ParamRange(BaseModel):
    """
    A continuous axis for `SearchSpec.method='tpe'`: sampled between `low`
    and `high` by the sampler rather than enumerated by hand. A grid over a
    regularization strength is either coarse or enormous; a log-spaced
    range is what the question actually is.
    """

    model_config = ConfigDict(extra="forbid")

    low: float
    high: float
    log: bool = Field(
        False,
        description="Sample uniformly in log space -- for a strength or a "
        "learning rate whose sensible values span decades. Needs low > 0.",
    )
    integer: bool = Field(
        False,
        description="Round to an integer: a tree count, a depth, a window. "
        "Both endpoints must be integers.",
    )

    @model_validator(mode="after")
    def _well_formed(self) -> "ParamRange":
        if not self.low < self.high:
            raise ValueError(
                f"a range needs low < high; got low={self.low}, high={self.high}"
            )
        if self.log and self.low <= 0:
            raise ValueError(
                f"a log-spaced range needs low > 0; got low={self.low}. Sample "
                "linearly, or start the range above zero."
            )
        if self.integer and (self.low != int(self.low) or self.high != int(self.high)):
            raise ValueError(
                f"an integer range needs integer endpoints; got low={self.low}, "
                f"high={self.high}"
            )
        return self


class SearchSpec(BaseModel):
    """
    Hyperparameter search on the TRAINING window of each fold.

    The search runs its own inner walk-forward inside the training data and
    never sees the fold's test window, so the outer out-of-sample metric
    stays out-of-sample. That is why this exists rather than a sklearn
    GridSearchCV wrapped around the panel: an ordinary K-fold over stacked
    (entity, date) rows puts the same date on both sides of a split and
    would select hyperparameters on leaked information.
    """

    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    method: Literal["grid", "random", "tpe"] = Field(
        "grid",
        description="'grid' — every combination. 'random' — `n_iter` samples "
        "from the grid, the better use of a fixed budget once the grid has "
        "more than a couple of axes. 'tpe' — optuna's Tree-structured Parzen "
        "Estimator chooses each next candidate from what the previous ones "
        "scored, over `param_ranges` (continuous) and `param_grid` "
        "(categorical), for `max_trials` trials; needs optuna installed. All "
        "three score candidates on the same purged, embargoed inner folds.",
    )
    param_grid: Dict[str, List[object]] = Field(
        default_factory=dict,
        description="Estimator parameter name -> candidate values. Every name "
        "must be allowed for the chosen estimator, checked at the same boundary "
        "as EstimatorSpec.params. grid and random: every axis; tpe: the "
        "categorical axes, beside `param_ranges`.",
    )
    param_ranges: Dict[str, ParamRange] = Field(
        default_factory=dict,
        description="tpe only: estimator parameter name -> a continuous range "
        "the sampler draws from. A name may not appear in both this and "
        "`param_grid`.",
    )
    n_iter: int = Field(
        20, gt=0, description="random only: how many combinations to sample."
    )
    max_trials: int = Field(
        30,
        ge=1,
        le=1000,
        description="tpe only: trials the sampler runs per fold. Each is "
        "fitted on every inner fold, so the cost is max_trials x inner_splits "
        "fits per outer fold, which the plan counts against the budget.",
    )
    early_pruning: bool = Field(
        False,
        description="tpe only: stop a trial after an inner fold when its "
        "running mean is below the median of the completed trials at the "
        "same fold (optuna's MedianPruner, after three complete trials). "
        "Fewer fits; a pruned trial keeps the score it had and is reported "
        "as pruned.",
    )
    inner_splits: int = Field(
        3, ge=2, description="Inner walk-forward folds used to score a candidate."
    )
    scoring: Literal[
        "cs_rank_ic", "cs_ic", "r2", "neg_mae", "accuracy", "auc", "concordance"
    ] = Field(
        "cs_rank_ic",
        description="What the search maximizes. Defaults to cross-sectional "
        "rank IC because that is what the outer report leads with — selecting "
        "on r2 and then quoting rank IC optimizes one thing and reports "
        "another. 'concordance' is the survival task's score and the only one "
        "it accepts: an IC of a risk against a censored duration measures "
        "nothing.",
    )

    @model_validator(mode="after")
    def _axes_agree_with_method(self) -> "SearchSpec":
        for name, values in self.param_grid.items():
            if not values:
                raise ValueError(f"param_grid[{name!r}] has no candidate values")
        both = sorted(set(self.param_grid) & set(self.param_ranges))
        if both:
            raise ValueError(
                f"{both} appear in both param_grid and param_ranges; an axis is "
                "either enumerated or sampled, not both."
            )
        if self.method == "tpe":
            if not self.param_grid and not self.param_ranges:
                raise ValueError(
                    "method='tpe' needs at least one axis: a continuous range in "
                    "param_ranges or candidate values in param_grid."
                )
            return self
        # The fields the other methods do not read must be at their
        # defaults, so a reloaded spec cannot carry a setting nothing used.
        if not self.param_grid:
            raise ValueError("param_grid must name at least one parameter")
        if self.param_ranges:
            raise ValueError(
                f"param_ranges is read by method='tpe' only; method="
                f"{self.method!r} enumerates param_grid. Drop the ranges, or "
                "set method='tpe'."
            )
        if self.max_trials != 30:
            raise ValueError(
                f"max_trials={self.max_trials} is read by method='tpe' only; "
                f"method={self.method!r} does not use it."
            )
        if self.early_pruning:
            raise ValueError(
                f"early_pruning is read by method='tpe' only; method="
                f"{self.method!r} does not use it."
            )
        return self


class RankingSpec(BaseModel):
    """
    How a continuous target becomes something a ranker can learn from.

    Only consulted for task='ranking'. It exists because the conversion is
    not optional: LightGBM and XGBoost both REJECT a continuous label for a
    ranking objective outright, and neither accepts a merely non-negative
    one — the requirement is integer relevance grades.
    """

    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    n_grades: int = Field(
        8,
        ge=2,
        le=31,
        description=(
            "Relevance levels the target is cut into WITHIN each date, 0 "
            "(worst) to n_grades-1 (best). Fewer grades tell the objective "
            "less about the ordering it is meant to learn; more make each "
            "level thinner than the noise in the target, so the model spends "
            "capacity on distinctions that are not really there. Measured on "
            "a 40-entity cross-section, 8 grades beat 16 — five names per "
            "grade carried more signal than two and a half. "
            "The ceiling of 31 is LightGBM's, not a preference: its default "
            "label_gain table holds 31 entries (2^i - 1 for i in 0..30), and "
            "a 32nd grade fails at fit time with 'Label 31 is not less than "
            "the number of label mappings'. Bounded here so that surfaces as "
            "a spec error rather than a crash several folds in."
        ),
    )
    ndcg_at: List[int] = Field(
        default_factory=lambda: [5, 10],
        description=(
            "Cut-offs for the reported NDCG. Rank IC weighs the whole "
            "cross-section equally; NDCG's logarithmic discount weighs the "
            "top of the ranking far more heavily, which is closer to how a "
            "concentrated book actually uses a score. Both are reported "
            "because a model can improve one and not the other."
        ),
    )

    @model_validator(mode="after")
    def _cutoffs_are_positive(self) -> "RankingSpec":
        if not self.ndcg_at:
            raise ValueError("ndcg_at must name at least one cut-off")
        bad = [k for k in self.ndcg_at if k < 1]
        if bad:
            raise ValueError(f"ndcg_at cut-offs must be >= 1, got {bad}")
        return self


class ComputeBudgetSpec(BaseModel):
    """
    The most an experiment may cost, checked before it costs anything.

    A spec that looks modest can imply thousands of estimator fits once a
    search grid multiplies through every fold's inner splits, and nothing
    in the spec shows it. The plan counts the fits before the first one and
    REFUSES a spec over the ceiling, by name, with the count and what would
    bring it under. Never truncates: a search that ran half its grid is not
    the search the spec described.
    """

    model_config = ConfigDict(extra="forbid")

    max_fits: int = Field(
        DEFAULT_MAX_FITS,
        ge=1,
        le=MAX_FITS_CEILING,
        description=(
            "Ceiling on estimator fits for one run_model_experiment call: "
            "every fold's fit, every search candidate on every inner fold, "
            "each calibration fold, and the full-panel refit. Over it the "
            "run is refused before anything is fitted and the message says "
            "the count. Raise it on purpose to accept a long run; "
            "validate_model_spec reports the count without running."
        ),
    )


class ConformalSpec(BaseModel):
    """
    A prediction interval from split conformal calibration.

    Inside each training window the estimator is refit without each of
    `calibration_folds` contiguous date blocks -- embargoed and purged like
    every other split here -- and the absolute residuals on the held-out
    blocks are collected. The (1 - alpha) quantile of those is the radius:
    prediction +/- radius covers a new outcome with probability at least
    1 - alpha for exchangeable rows. A return panel is not exchangeable
    across regimes, so the OOS coverage is reported as the check.
    """

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(
        0.1,
        gt=0.0,
        lt=1.0,
        description="Miscoverage: 0.1 asks for a 90% interval.",
    )
    method: Literal["split"] = Field(
        "split",
        description="'split' -- residual quantiles on held-out date blocks. The "
        "only method; named so a second one is a choice rather than a change.",
    )
    calibration_folds: int = Field(
        3,
        ge=2,
        le=10,
        description="Contiguous date blocks the training window is cut into "
        "for the residuals. Each is one more fit per fold.",
    )


class ModelSpec(BaseModel):
    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    task: Task
    estimator: EstimatorSpec
    validation: ValidationSpec
    quantiles: List[float] = Field(
        default_factory=list,
        max_length=9,
        description=(
            "task='regression' only: quantile levels in (0, 1) to fit beside "
            "the point prediction, one fit per level per fold, on an "
            "estimator that can fit a quantile (list_modeling_capabilities "
            "reports `quantile_param`). The OOS frame gains a column per "
            "level (`q05`, `q50`, `q95`), `prediction` stays the base fit's "
            "point score, and the metrics gain the pinball loss per level, "
            "the crossing rate, and coverage and width for each symmetric "
            "pair such as 0.05 and 0.95."
        ),
    )
    intervals: Optional[ConformalSpec] = Field(
        None,
        description="task='regression' only: a split-conformal prediction "
        "interval, `lower`/`upper` on the OOS frame and at scoring, with "
        "coverage and width reported. See ConformalSpec.",
    )
    budget: ComputeBudgetSpec = Field(
        default_factory=ComputeBudgetSpec,
        description="How much compute the experiment may spend; refused, not "
        "truncated, when the plan exceeds it.",
    )
    preprocessing: PreprocessingSpec = Field(default_factory=PreprocessingSpec)
    ranking: RankingSpec = Field(
        default_factory=RankingSpec,
        description="task='ranking' only: how the target is graded and which "
        "NDCG cut-offs are reported.",
    )
    weighting: WeightingSpec = Field(default_factory=WeightingSpec)
    search: Optional[SearchSpec] = Field(
        None,
        description="Optional hyperparameter search on each fold's training "
        "window. Costs roughly (grid size x inner_splits) extra fits per fold.",
    )
    random_seed: int = Field(
        42,
        ge=0,
        le=2**32 - 1,
        description="Seed passed to the estimator's constructor. Bounded to "
        "numpy/sklearn's accepted RandomState range [0, 2**32-1]: an arbitrary "
        "Python int outside it (negative, or wider than 32 bits) is rejected "
        "deep inside sklearn rather than at this boundary, where the message "
        "can say which field was wrong.",
    )

    @field_validator("quantiles")
    @classmethod
    def _quantiles_are_levels(cls, v: List[float]) -> List[float]:
        levels = sorted({float(q) for q in v})
        for q in levels:
            if not 0.0 < q < 1.0:
                raise ValueError(
                    f"quantiles must lie strictly inside (0, 1); got {q}. The "
                    "0th and 100th percentiles are the sample's own extremes, "
                    "not something a model fits."
                )
        return levels

    @model_validator(mode="after")
    def _search_scoring_fits_the_task(self) -> "ModelSpec":
        if self.search is None:
            return self
        if self.task == "survival" and self.search.scoring != "concordance":
            raise ValueError(
                f"task='survival' selects on search.scoring='concordance'; "
                f"{self.search.scoring!r} correlates a risk score with a "
                "censored duration, which measures nothing about the ordering "
                "the task is judged on."
            )
        if self.task != "survival" and self.search.scoring == "concordance":
            raise ValueError(
                "search.scoring='concordance' is the survival task's score; "
                f"task={self.task!r} has no event indicator to compute it from."
            )
        return self

    @model_validator(mode="after")
    def _distribution_is_a_regression_question(self) -> "ModelSpec":
        if self.task == "regression":
            return self
        if self.quantiles:
            raise ValueError(
                f"quantiles are fitted for task='regression' only; task="
                f"{self.task!r} predicts a probability or an ordering, which "
                "has no quantiles of a continuous outcome to fit."
            )
        if self.intervals is not None:
            raise ValueError(
                f"intervals are calibrated for task='regression' only; task="
                f"{self.task!r} has no residual to place an interval around."
            )
        return self


class PredictionTransformSpec(BaseModel):
    """
    How a model's out-of-sample predictions become portfolio target
    weights.

    This is the piece `bridge.oos_predictions_to_signal_panel` cannot
    express. That function collapses every prediction to -1/0/+1 because
    its consumer (`run_signal_panel_backtest`) treats a SCORE value as a
    raw leverage multiplier, so passing a 0.02 forward-return prediction
    through would size a 2%-leveraged position. Sign is the correct answer
    for THAT engine, but it throws away both the ranking and the magnitude
    — which is most of what a cross-sectional model actually predicts.

    `run_portfolio_simulation` takes target WEIGHTS (fractions of account
    equity) rather than direction signals, so the rank survives all the
    way to position size. The score -> weight step itself is not
    reimplemented here: `backtest.sizing` already builds exactly these
    panels and is reused as-is (see portfolio_eval._SIZERS). What this
    spec adds is the declarative selection an agent can construct, plus
    the three pieces sizing.py does not have — a per-position cap, an
    explicit net-exposure target, and a rebalance schedule.
    """

    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "sign",
        "cross_sectional_rank",
        "cross_sectional_zscore",
        "top_bottom_quantile",
        "uncertainty_scaled",
    ] = Field(
        "cross_sectional_rank",
        description=(
            "'cross_sectional_rank' (default) — weight proportional to the "
            "prediction's rank within each date's cross-section, centered on "
            "the mean rank. Robust to the prediction scale being wrong, which "
            "for a return-forecasting model it usually is. "
            "'cross_sectional_zscore' — proportional to the standardized "
            "prediction, so magnitude carries through and an outlier gets a "
            "bigger position. 'top_bottom_quantile' — equal weight in the top "
            "`long_quantile` and bottom `short_quantile` of each cross-section, "
            "flat in between; the classic quantile-portfolio construction. "
            "'sign' — equal weight on the sign of the (centered) prediction. "
            "Reproduces the bridge's information content, but sized as a "
            "portfolio rather than as per-ticker direction signals. "
            "'uncertainty_scaled' — the prediction divided by the width of its "
            "conformal interval, then standardized within the cross-section: "
            "a confident forecast gets a bigger position than an equal but "
            "uncertain one. Needs a model trained with ModelSpec.intervals; "
            "refused by name otherwise."
        ),
    )
    long_quantile: float = Field(
        0.2,
        gt=0.0,
        le=1.0,
        description="top_bottom_quantile only: fraction of each cross-section "
        "held long. 0.2 = top quintile.",
    )
    short_quantile: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description="top_bottom_quantile only: fraction of each cross-section "
        "held short. Set 0.0 for a long-only quantile portfolio (which also "
        "requires net_exposure == gross_exposure).",
    )
    gross_exposure: float = Field(
        1.0,
        gt=0.0,
        le=10.0,
        description="Target sum(|weight|) per rebalance date. 1.0 = fully "
        "invested, unlevered. Must be <= the portfolio spec's "
        "max_gross_leverage, which is what the simulator actually enforces.",
    )
    net_exposure: float = Field(
        0.0,
        description=(
            "Target sum(weight) per rebalance date. 0.0 (default) = dollar "
            "neutral; set equal to gross_exposure for long-only. |net| must "
            "be <= gross. The two targets are hit exactly by sizing the long "
            "book to (gross + net)/2 and the short book to (gross - net)/2, "
            "which is why they compose rather than fighting each other — a "
            "single rescale cannot control both."
        ),
    )
    max_position_weight: float = Field(
        0.05,
        gt=0.0,
        le=1.0,
        description=(
            "Cap on any single |weight|. Excess above the cap is redistributed "
            "to the uncapped names in the same book (repeatedly, since "
            "redistribution can push another name over), so the cap does not "
            "quietly reduce gross exposure. A book with too few names to "
            "absorb its target gross at this cap is reported as a shortfall "
            "rather than silently levered past the cap."
        ),
    )
    volatility_scale: bool = Field(
        False,
        description=(
            "Divide each raw prediction by that entity's trailing realized "
            "volatility before weighting (backtest.sizing.vol_scaled), so an "
            "equally-ranked high-vol name takes a smaller position. Default "
            "False keeps the transform a pure function of the predictions; "
            "True makes it depend on price history too, and therefore on "
            "`volatility_lookback`. Ignored by method='top_bottom_quantile' "
            "and 'sign', whose weights are membership-based — scaling a score "
            "cannot change an equal weight."
        ),
    )
    volatility_lookback: int = Field(
        20,
        gt=1,
        le=500,
        description="Bars of trailing returns used for volatility_scale. "
        "Counted in BARS of the dataset's own interval, not calendar days.",
    )
    rebalance_frequency: Literal["daily", "weekly", "monthly"] = Field(
        "weekly",
        description=(
            "Which of the OOS prediction dates actually become rebalance "
            "dates. 'weekly'/'monthly' take the FIRST prediction date in each "
            "calendar week/month — first, not last, so the choice never "
            "depends on a date later than the one being traded. Between "
            "rebalances the simulator holds share counts constant and lets "
            "weights drift, which is the whole reason to rebalance less than "
            "daily: it is the turnover, not the prediction, that costs money."
        ),
    )

    @model_validator(mode="after")
    def _coherent_exposures(self) -> "PredictionTransformSpec":
        if not math.isfinite(self.net_exposure):
            raise ValueError(f"net_exposure must be finite, got {self.net_exposure}")
        if abs(self.net_exposure) > self.gross_exposure + 1e-12:
            raise ValueError(
                f"|net_exposure| ({abs(self.net_exposure)}) cannot exceed "
                f"gross_exposure ({self.gross_exposure}) — the long book would "
                "have to be larger than the whole portfolio."
            )
        if self.method == "top_bottom_quantile":
            if self.long_quantile + self.short_quantile > 1.0 + 1e-12:
                raise ValueError(
                    f"long_quantile + short_quantile ({self.long_quantile} + "
                    f"{self.short_quantile}) exceeds 1.0 — the two books would "
                    "have to overlap, putting the same name long and short."
                )
            long_only = abs(self.net_exposure - self.gross_exposure) <= 1e-12
            if self.short_quantile == 0.0 and not long_only:
                raise ValueError(
                    "short_quantile=0.0 selects no short names, so the short "
                    "book cannot be filled to (gross - net)/2 = "
                    f"{(self.gross_exposure - self.net_exposure) / 2}. Set "
                    "net_exposure == gross_exposure for a long-only portfolio."
                )
        return self


class PortfolioSimSpec(BaseModel):
    """
    The subset of `run_portfolio_simulation`'s parameters an agent may set
    when evaluating a model, with defaults chosen for evaluation rather
    than for backward compatibility.

    Deliberately narrower than the simulator's own signature. The omitted
    parameters (per-share commissions, the impact model, hl2 fills) either
    need calibration this layer cannot supply, or exist for exploratory use
    that would make a model-selection number misleading.
    `run_portfolio_simulation` is still importable directly for those.
    """

    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    initial_capital: float = Field(100_000.0, gt=0.0)
    commission_pct: float = Field(
        0.001, ge=0.0, le=0.1, description="Commission per trade notional."
    )
    slippage_pct: float = Field(
        0.0005, ge=0.0, le=0.1, description="Spread cost per trade notional."
    )
    fill_price: Literal["close", "next_open"] = Field(
        "next_open",
        description=(
            "'next_open' (default) — a weight dated t executes at t+1's open. "
            "This is the only lookahead-free choice: modeling features close "
            "on bar t's own OHLC, so a prediction dated t is not knowable "
            "until t's close has printed, and filling it AT that close is the "
            "look-ahead run_strategy's own fill_price warning describes. "
            "'close' is accepted for like-for-like comparison against an "
            "existing close-filled backtest, and is reported in `warnings` "
            "because a model evaluated that way will look better than it is."
        ),
    )
    max_gross_leverage: float = Field(
        1.0,
        gt=0.0,
        le=10.0,
        description="Hard limit the simulator enforces on each date's target "
        "gross. Must be >= the transform's gross_exposure, or every rebalance "
        "is rejected.",
    )
    max_position_pct: float = Field(1.0, gt=0.0, le=1.0)
    borrow_fee_bps: float = Field(
        0.0,
        ge=0.0,
        description="Annualized bps accrued daily on short notional. A "
        "long/short model evaluated at 0.0 is being credited with free "
        "shorting.",
    )
    margin_interest_rate: float = Field(0.0, ge=0.0)
    max_adv_participation: Optional[float] = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Reject any rebalance trade exceeding this fraction of the "
        "ticker's rolling average dollar volume. None = unconstrained, which "
        "for a large-universe model means capacity is untested.",
    )
    risk_free_rate: float = Field(
        0.0,
        description="Annualized rate used for the reported Sharpe/Sortino "
        "only — it does not enter the simulation itself.",
    )
