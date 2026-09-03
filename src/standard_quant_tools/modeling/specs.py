"""
Declarative specs the modeling runtime executes — the ModelSpec-not-exec()
contract: an LLM (or any caller) builds one of these Pydantic objects and
hands it to `dataset.builder.build_dataset` / `engine.run_experiment`,
never arbitrary Python. Every field here is validated once, at the
boundary, the same discipline `agent/models.py` uses for the analysis
tool surface.
"""

import math
from dataclasses import dataclass
from typing import Annotated, Dict, List, Literal, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .features.base import RESERVED_PANEL_COLUMNS
from .limits import MAX_LAG, MAX_LAGS_PER_FEATURE


def _parse_date(value: str, field_name: str) -> pd.Timestamp:
    """Shared by DatasetSpec's start/end cross-check and
    modeling.agent.models.ScoreModelInput.as_of — raises the same
    ValueError shape pydantic validators elsewhere in this codebase use
    (e.g. PortfolioInput._check_weights), not a raw pandas parse error."""
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{field_name}={value!r} is not a valid date: {exc}") from None


@dataclass(frozen=True)
class TargetKind:
    """One supervised label: what consumes it, and who can build it."""

    #: The tasks that can be fitted against it. A continuous label suits a
    #: regressor and a ranker; a discrete one suits a classifier. This is
    #: read by the engine's compatibility check rather than restated there.
    tasks: Tuple[str, ...]
    #: Whether `build_target` can produce it from a Close series. FALSE for
    #: every microstructure and execution label: none is a function of
    #: closing prices, and pretending otherwise would silently hand back a
    #: forward return under another name.
    buildable: bool
    #: Continuous labels reject a `threshold`, which only means something
    #: for a binarized one.
    continuous: bool
    description: str


#: Every label this library understands, and the ONE place that says so.
#:
#: WHY A REGISTRY AND NOT FIVE LITERALS. The task set was written five times
#: in two widths, and the narrow copies were where `ranking` had been
#: forgotten -- a model that could be fitted and never traded. The target
#: set was on the same path: four copies, two of them added in the same
#: week this was written. A type declared here is a type every consumer
#: sees, and the Literal below is pinned equal to it by test.
#:
#: EXTERNAL-ONLY IS NOT A GAP. A markout, a fill probability or a time to
#: fill is a function of the order book and of orders, not of closing
#: prices. `build_target` refuses them by name and says to compute them
#: where the book is and register the panel -- which is a real answer,
#: whereas a bar-derived approximation of a fill probability would be a
#: number with nothing behind it.
TARGET_KINDS: Dict[str, TargetKind] = {
    "forward_return": TargetKind(
        tasks=("regression", "ranking"),
        buildable=True,
        continuous=True,
        description="The return from t to t+horizon.",
    ),
    "forward_direction": TargetKind(
        tasks=("classification",),
        buildable=True,
        continuous=False,
        description="That forward return binarized against `threshold`.",
    ),
    "forward_return_vol_scaled": TargetKind(
        tasks=("regression", "ranking"),
        buildable=True,
        continuous=True,
        description="Forward return over the entity's own trailing volatility.",
    ),
    "forward_return_rank": TargetKind(
        tasks=("regression", "ranking"),
        buildable=True,
        continuous=True,
        description="Its rank within the date's cross-section, in [-0.5, 0.5].",
    ),
    "forward_return_market_neutral": TargetKind(
        tasks=("regression", "ranking"),
        buildable=True,
        continuous=True,
        description="Forward return minus that date's equal-weighted mean.",
    ),
    "triple_barrier": TargetKind(
        tasks=("classification",),
        buildable=True,
        continuous=False,
        description="Which barrier is touched first: up, down, or neither.",
    ),
    # ── microstructure labels, computed where the book is ─────────────
    "future_mid_return": TargetKind(
        tasks=("regression", "ranking"),
        buildable=False,
        continuous=True,
        description=(
            "Return of the MIDPOINT over the horizon. Not the same as a "
            "trade-price return: the mid moves without a trade and is where "
            "a passive order is measured from."
        ),
    ),
    "future_microprice_return": TargetKind(
        tasks=("regression", "ranking"),
        buildable=False,
        continuous=True,
        description=(
            "Return of the size-weighted touch price. Leads the mid when the "
            "book is lopsided, which is exactly when the mid is least "
            "informative."
        ),
    ),
    "future_markout": TargetKind(
        tasks=("regression", "ranking"),
        buildable=False,
        continuous=True,
        description=(
            "Mid move measured FROM a fill, signed by the side taken. The "
            "standard read on whether a trade was well-placed."
        ),
    ),
    "next_mid_direction": TargetKind(
        tasks=("classification",),
        buildable=False,
        continuous=False,
        description="Whether the midpoint's next move is up or down.",
    ),
    "future_spread": TargetKind(
        tasks=("regression", "ranking"),
        buildable=False,
        continuous=True,
        description=(
            "The quoted spread at t+horizon. A liquidity forecast rather "
            "than a price one -- what it will COST to cross, not where the "
            "price goes."
        ),
    ),
    "future_depth": TargetKind(
        tasks=("regression", "ranking"),
        buildable=False,
        continuous=True,
        description=(
            "Resting size at t+horizon. What will be THERE to trade against, "
            "which a spread forecast does not answer -- a tight quote for a "
            "hundred shares and a tight quote for fifty thousand cost the "
            "same to cross and are not the same liquidity."
        ),
    ),
    "future_ofi": TargetKind(
        tasks=("regression", "ranking"),
        buildable=False,
        continuous=True,
        description=(
            "Signed order-flow imbalance over the horizon, from book "
            "updates. Predicting FLOW rather than price: the quantity that "
            "moves the price, one step earlier."
        ),
    ),
    "future_volume": TargetKind(
        tasks=("regression", "ranking"),
        buildable=False,
        continuous=True,
        description=(
            "Traded volume over the horizon. Bar volume can approximate this "
            "at daily frequency, but not at the horizons this exists for, "
            "where the question is how much prints in the next thirty "
            "seconds."
        ),
    ),
    "future_trade_intensity": TargetKind(
        tasks=("regression", "ranking"),
        buildable=False,
        continuous=True,
        description=(
            "Trades per unit time over the horizon. Distinct from volume: "
            "one block and two hundred odd lots are the same volume and "
            "completely different information."
        ),
    ),
    "fill_probability": TargetKind(
        tasks=("classification",),
        buildable=False,
        continuous=False,
        description=(
            "Whether a passive order resting at a stated level fills within "
            "the horizon. Needs queue position and cancellations, so no "
            "bar-derived series can produce it."
        ),
    ),
    "time_to_fill": TargetKind(
        tasks=("regression",),
        buildable=False,
        continuous=True,
        description=(
            "How long that order waits before filling. CENSORED by "
            "construction -- an order that never fills has no time, and "
            "recording it as the horizon rather than as unfilled biases "
            "every estimate toward patience."
        ),
    ),
    "adverse_selection": TargetKind(
        tasks=("regression", "ranking"),
        buildable=False,
        continuous=True,
        description=(
            "How much the mid moves against a fill after it happens. The "
            "cost of being the one who was willing to trade."
        ),
    ),
}

#: The same set as a Literal, so a bad value is refused at the schema
#: boundary. Written out because a Literal cannot be built from a dict at
#: type-check time; `test_the_target_literal_matches_the_registry` fails the
#: moment the two disagree.
TargetType = Literal[
    "forward_return",
    "forward_direction",
    "forward_return_vol_scaled",
    "forward_return_rank",
    "forward_return_market_neutral",
    "triple_barrier",
    "future_mid_return",
    "future_microprice_return",
    "future_markout",
    "next_mid_direction",
    "future_spread",
    "future_depth",
    "future_ofi",
    "future_volume",
    "future_trade_intensity",
    "fill_probability",
    "time_to_fill",
    "adverse_selection",
]

#: Labels no Close series can produce.
EXTERNAL_TARGETS = tuple(
    name for name, kind in TARGET_KINDS.items() if not kind.buildable
)


def targets_for_task(task: str) -> Tuple[str, ...]:
    """Every label a given task can be fitted against."""
    return tuple(name for name, kind in TARGET_KINDS.items() if task in kind.tasks)


#: The supervised tasks this library fits, declared ONCE.
#:
#: It was written five times, in two widths: three copies said
#: regression/classification/ranking and two said
#: regression/classification. The narrow pair was not a different opinion,
#: it was a place ranking had been forgotten -- which is exactly the drift
#: a repeated literal produces and the reason this name exists.
TASKS = ("regression", "classification", "ranking")
Task = Literal["regression", "classification", "ranking"]

#: Tasks whose prediction is a CONTINUOUS SCORE rather than a probability.
#: A ranker emits a relative score exactly as a regressor emits a
#: magnitude, so everything downstream that asks "which side is this" reads
#: the sign of both the same way. Classification is the odd one out, being
#: bounded in [0, 1] with a decision boundary in the middle.
SCORE_TASKS = ("regression", "ranking")


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

    method: Literal["walk_forward", "purged_kfold"] = Field(
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
            "would this have earned'."
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
        5, ge=2, description="purged_kfold only: how many contiguous test blocks."
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


class PreprocessingSpec(BaseModel):
    """How feature columns are normalized before the estimator sees them."""

    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

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

    method: Literal["grid", "random"] = Field(
        "grid",
        description="'grid' — every combination. 'random' — `n_iter` samples "
        "from the grid, the better use of a fixed budget once the grid has "
        "more than a couple of axes.",
    )
    param_grid: Dict[str, List[object]] = Field(
        ...,
        description="Estimator parameter name -> candidate values. Every name "
        "must be allowed for the chosen estimator, checked at the same boundary "
        "as EstimatorSpec.params.",
    )
    n_iter: int = Field(
        20, gt=0, description="random only: how many combinations to sample."
    )
    inner_splits: int = Field(
        3, ge=2, description="Inner walk-forward folds used to score a candidate."
    )
    scoring: Literal["cs_rank_ic", "cs_ic", "r2", "neg_mae", "accuracy", "auc"] = Field(
        "cs_rank_ic",
        description="What the search maximizes. Defaults to cross-sectional "
        "rank IC because that is what the outer report leads with — selecting "
        "on r2 and then quoting rank IC optimizes one thing and reports "
        "another.",
    )

    @model_validator(mode="after")
    def _grid_not_empty(self) -> "SearchSpec":
        if not self.param_grid:
            raise ValueError("param_grid must name at least one parameter")
        for name, values in self.param_grid.items():
            if not values:
                raise ValueError(f"param_grid[{name!r}] has no candidate values")
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


class ModelSpec(BaseModel):
    # extra="forbid" like every top-level input model. Without it a
    # nested typo was silently dropped: `validate_model_spec` -- the
    # tool whose job is catching exactly this -- certified a spec
    # `valid: True` while the embargo the caller asked for was 0.
    model_config = ConfigDict(extra="forbid")

    task: Task
    estimator: EstimatorSpec
    validation: ValidationSpec
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
            "portfolio rather than as per-ticker direction signals."
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
