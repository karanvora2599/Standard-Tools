"""
The inputs and results for the three questions a spec author has to
answer BEFORE anything is built or fitted.

WHY THESE ARE A MODULE PAIR. `discovery_tools.py` needs none of
`tools.py`'s private helpers -- it reads registries, bounds and a
calendar, and touches no panel and no manifest -- so its models live
beside it rather than in `models.py`, the way `dataset_tools.py` and
`portfolio_models.py` do. The seam is the point: nothing here can be
broken by an edit to the tool file that every other modeling tool
shares.

WHAT THE THREE RESULTS HAVE IN COMMON. Each one carries a number that
the library already computes on every call and that no view returned,
so the only way to learn it was to make a call that failed:

    describe_exchange_calendar   the venue's sessions, session length and
                                 bars per session -- previously reachable
                                 only by provoking the refusal that lists
                                 the first eight codes of dozens
    describe_estimator           the value bounds and the cross-parameter
                                 rules that `validate_params` enforces on
                                 every fit, plus the calibration choice
                                 the capability report omits entirely
    estimate_feature_warmup      bars of history a feature spec consumes
                                 at its REQUESTED parameters, which is a
                                 different number from the catalog's the
                                 moment a window is overridden

See the CHANGELOG entry of 2026-09-21 for why each was added.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..specs import FeatureSpec
from .models import Stat

# ── describe_exchange_calendar ─────────────────────────────────────────


class DescribeExchangeCalendarInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it had
    # configured something.
    model_config = ConfigDict(extra="forbid")

    calendar: Optional[str] = Field(
        None,
        description=(
            "An `exchange_calendars` venue code to resolve, e.g. 'XNYS', "
            "'XLON' or '24/7' -- the same value DatasetSpec.calendar takes. "
            "Omit it to list the codes instead of resolving one. An "
            "unrecognised code is refused with the identical message "
            "DatasetSpec gives, so a code that passes here passes there."
        ),
    )
    interval: Optional[str] = Field(
        None,
        description=(
            "A bar interval to place inside a session, e.g. '1h', '5m', "
            "'90m'. With a calendar this yields bars_per_session and "
            "periods_per_year -- the factor that annualizes an intraday "
            "volatility or Sharpe. A daily-or-coarser interval is not "
            "refused here: it reports bars_per_session=None with the reason "
            "in `warnings`, because daily bars annualize by calendar "
            "arithmetic and need no venue at all."
        ),
    )
    name_contains: Optional[str] = Field(
        None,
        description=(
            "Case-insensitive substring filter on the returned codes. There "
            "are dozens of venues; 'XL' narrows them to the three that "
            "start that way. `n_calendars` still reports the unfiltered "
            "total, so a filter cannot make the catalog look smaller than "
            "it is."
        ),
    )


class DescribeExchangeCalendarResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = Field(
        ...,
        description=(
            "Whether the optional `exchange_calendars` package is importable "
            "here. False is a fact about this machine, not a failure: "
            "listing returns an empty catalog with a warning, while asking "
            "to RESOLVE a named calendar without the library is refused, "
            "because there is no honest number to return for it."
        ),
    )
    n_calendars: int = Field(
        0,
        description=(
            "Venue codes this installation knows, BEFORE `name_contains` is "
            "applied. Version-dependent -- the count moves with the "
            "installed package -- so it is reported rather than assumed."
        ),
    )
    calendar_names: List[str] = Field(
        default_factory=list,
        description="The codes, sorted, after `name_contains` is applied.",
    )
    calendar: Optional[str] = Field(
        None,
        description="The resolved code, when one was asked for.",
    )
    sessions_per_year: Stat = Field(
        None,
        description=(
            "Trading sessions per year, COUNTED over the calendar's complete "
            "years rather than estimated, so holidays are in the number. "
            "This is the factor a daily Sharpe is annualized by; 252 is a "
            "convention, and a venue's own count is not 252."
        ),
    )
    session_minutes: Stat = Field(
        None,
        description=(
            "Length of a full session in minutes, as the median over recent "
            "sessions -- a median so an early close before a holiday does "
            "not shorten every day."
        ),
    )
    interval_minutes: Optional[int] = Field(
        None,
        description=(
            "Minutes per bar for the requested interval, or None when the "
            "interval is daily-or-coarser or is not one this library "
            "recognises. Computed without the calendar library."
        ),
    )
    bars_per_session: Optional[int] = Field(
        None,
        description=(
            "Whole bars in one session, the partial last bar COUNTED: a "
            "6.5-hour session at '1h' is seven bars, not six, because the "
            "provider emits the stub. None for a daily-or-coarser interval, "
            "with the reason in `warnings`."
        ),
    )
    periods_per_year: Optional[int] = Field(
        None,
        description=(
            "bars_per_session x sessions_per_year: what annualizes a "
            "statistic computed on intraday bars. Wrong by a fixed factor "
            "if a constant chosen for another venue is used instead, which "
            "is why nothing in this library guesses it."
        ),
    )
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Conditions that change how this result reads: the library "
            "absent, an interval that cannot be placed in a session (quoted "
            "verbatim from the refusal the bars-per-session computation "
            "would have raised), or an intraday interval given with no "
            "calendar to place it in."
        ),
    )


# ── describe_estimator ─────────────────────────────────────────────────


class ParamBoundDescription(BaseModel):
    """One parameter's accepted type, range and choices -- the bound the
    estimator allowlist enforces on every fit."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., description="'int', 'float', 'bool' or 'str'.")
    minimum: Stat = Field(None, description="Smallest accepted value, if bounded.")
    maximum: Stat = Field(
        None,
        description=(
            "Largest accepted value, if bounded. A ceiling here is a "
            "RESOURCE BUDGET rather than a modelling opinion: unbounded, one "
            "tool call could exhaust CPU and memory."
        ),
    )
    choices: Optional[List[Optional[str]]] = Field(
        None,
        description=(
            "The exact accepted values for a categorical parameter. A null "
            "in this list means the parameter also accepts None."
        ),
    )
    allow_none: bool = Field(False, description="Whether None is an accepted value.")
    note: str = Field(
        "",
        description=(
            "The hand-written reason behind the bound, where one exists. "
            "These carry the decision-changing facts -- which loss has no "
            "predict_proba, which schedule ignores the step size, why the "
            "tree ceiling is where it is."
        ),
    )


class CalibrationDescription(BaseModel):
    """CLASSIFICATION ONLY: the probability-calibration choice on
    EstimatorSpec, which the capability report does not publish."""

    model_config = ConfigDict(extra="forbid")

    choices: List[str] = Field(
        default_factory=list,
        description="Accepted values of EstimatorSpec.calibration.",
    )
    default: str = Field("none", description="What a spec gets without asking.")
    folds_default: int = Field(
        3, description="EstimatorSpec.calibration_folds default."
    )
    folds_minimum: Optional[int] = None
    folds_maximum: Optional[int] = None
    note: str = Field(
        "",
        description=(
            "Why this field decides an outcome rather than polishing one, "
            "quoted from the spec itself."
        ),
    )
    folds_note: str = Field(
        "", description="Why the calibration map is fitted on held-out folds."
    )


class EstimatorDescription(BaseModel):
    """One (task, name) entry of the estimator allowlist, with everything
    a spec author has to get right before the first fit."""

    model_config = ConfigDict(extra="forbid")

    task: str
    name: str = Field(
        ..., description="The value EstimatorSpec.type takes for this entry."
    )
    available: bool = Field(
        ...,
        description=(
            "Whether this estimator is registered on THIS machine. False "
            "means the entry is real and its library is not installed -- the "
            "name, its bounds and the package that would provide it are "
            "still described, which is the only way to learn what an "
            "uninstalled library would buy."
        ),
    )
    requires_library: Optional[str] = Field(
        None,
        description=(
            "The optional package this entry needs, or None when it comes "
            "from scikit-learn and is always present."
        ),
    )
    class_path: Optional[str] = Field(
        None,
        description=(
            "Fully-qualified class the allowlist maps this name to. None "
            "when the library is absent, because nothing was imported to "
            "have a path."
        ),
    )
    quantile_param: Optional[str] = Field(
        None,
        description=(
            "The constructor argument that names the quantile, for the "
            "estimators that can fit one -- which is what makes a prediction "
            "interval possible. The engine sets it itself, one fit per "
            "requested quantile, so it is NOT a params key a spec supplies. "
            "None means this estimator predicts a mean or a probability "
            "only."
        ),
    )
    params: Dict[str, ParamBoundDescription] = Field(
        default_factory=dict,
        description=(
            "Every parameter name EstimatorSpec.params accepts for this "
            "entry, each with its bound. The keys are exactly the allowlist "
            "the fit validates against -- anything outside them is refused "
            "before sklearn is reached."
        ),
    )
    compatibility_notes: List[str] = Field(
        default_factory=list,
        description=(
            "Rules BETWEEN parameters, which no per-parameter bound can "
            "express: a penalty that only some solvers implement, a mixing "
            "ratio that must be stated when the penalty mixes. Each one is "
            "checked on the whole params dict at validation time, so a "
            "combination that violates one is refused at the modeling "
            "boundary rather than from inside a fit."
        ),
    )
    calibration: Optional[CalibrationDescription] = Field(
        None,
        description=(
            "Present on classification entries only. Absent for regression, "
            "ranking and survival, where the field is not read."
        ),
    )


class DescribeEstimatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Optional[str] = Field(
        None,
        description=(
            "Restrict to one supervised task ('regression', "
            "'classification', 'ranking', 'survival'). The same estimator "
            "name can be registered under more than one task with a "
            "different schema -- sgd's accepted losses differ by task -- so "
            "a name alone may describe several entries."
        ),
    )
    name: Optional[str] = Field(
        None,
        description=(
            "Restrict to one estimator name, the value EstimatorSpec.type "
            "takes. With `task`, an unknown pair is refused with the same "
            "message a spec naming it would get, unless it is an optional "
            "estimator, which is described as unavailable instead."
        ),
    )
    include_unavailable: bool = Field(
        False,
        description=(
            "Also describe the estimators whose optional library is not "
            "installed here. These are declared statically, so their names "
            "and bounds are the same on every machine -- which is what makes "
            "a ranking estimator nameable, and the cost of not installing "
            "its library legible, on a machine that does not have it."
        ),
    )


class DescribeEstimatorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estimators: List[EstimatorDescription] = Field(default_factory=list)
    n_estimators_described: int = 0
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Conditions that change how this result reads: the size of an "
            "unfiltered description, and entries whose library is absent "
            "here."
        ),
    )


# ── estimate_feature_warmup ────────────────────────────────────────────


class FeatureWarmup(BaseModel):
    """What one requested feature costs in bars of history."""

    model_config = ConfigDict(extra="forbid")

    declared: int = Field(
        ...,
        description=(
            "The catalog's lookback: a static number recorded at "
            "registration against the feature's DEFAULT parameters."
        ),
    )
    resolved: int = Field(
        ...,
        description=(
            "Bars this feature consumes at the parameters actually "
            "requested. Equal to `declared` until a window parameter is "
            "overridden, and the two diverge without warning when one is -- "
            "a momentum feature declared at 20 and asked for 900 consumes "
            "900."
        ),
    )
    lags: List[int] = Field(
        default_factory=list,
        description="The lag columns requested for this feature.",
    )
    deepest_lag: int = Field(
        0,
        description=(
            "This feature's own deepest lag. Warm-up is charged ONCE at the "
            "deepest lag across the whole spec, not per feature, because the "
            "panel starts where the last column becomes computable."
        ),
    )
    point_in_time: bool = Field(
        False,
        description=(
            "Whether this feature reads a point-in-time RECORD SET rather "
            "than bars. One of these costs no bar warm-up at all; its "
            "freshness is bounded by max_staleness_days instead, which is a "
            "different unit and not comparable with the numbers above."
        ),
    )


class EstimateFeatureWarmupInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: List[FeatureSpec] = Field(
        ...,
        min_length=1,
        description=(
            "The feature specs to price, exactly as DatasetSpec.features "
            "would carry them -- id, params, alias and lags. Nothing is "
            "fetched and nothing is built."
        ),
    )
    interval: str = Field(
        "1d",
        description=(
            "The bar interval these features would be computed on. Every "
            "lookback counts BARS of this interval, so window=252 is a year "
            "at '1d' and about six weeks at '1h' -- which is what makes the "
            "calendar-day estimate below interval-dependent."
        ),
    )
    calendar: Optional[str] = Field(
        None,
        description=(
            "An `exchange_calendars` venue code, the same value "
            "DatasetSpec.calendar takes. Turns bars into calendar days with "
            "the venue's own session count instead of the 252 convention, "
            "and is REQUIRED for that conversion at an intraday interval, "
            "where bars per session is a property of the venue."
        ),
    )


class EstimateFeatureWarmupResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bars_required: int = Field(
        ...,
        description=(
            "Bars of history this spec burns before its first usable row: "
            "the deepest RESOLVED lookback plus the deepest lag. This is the "
            "number that decides where a panel can start, and the one to "
            "size a scoring history window with."
        ),
    )
    per_feature: Dict[str, FeatureWarmup] = Field(
        default_factory=dict,
        description=(
            "Keyed by the feature's OUTPUT NAME -- its alias where it has "
            "one, its id otherwise -- because that is the panel's column "
            "name and the only key that distinguishes the same feature "
            "requested twice at different windows."
        ),
    )
    binding_feature: Optional[str] = Field(
        None,
        description=(
            "The output name whose resolved lookback set `bars_required`. "
            "The one to shorten if the warm-up is unaffordable; shortening "
            "any other changes nothing."
        ),
    )
    deepest_lag: int = Field(
        0,
        description=(
            "Deepest lag across the whole spec, charged once on top of the "
            "binding lookback."
        ),
    )
    calendar_days_estimate: Stat = Field(
        None,
        description=(
            "`bars_required` expressed in CALENDAR days -- the unit a "
            "scoring history window is given in, which is not the unit the "
            "lookbacks are counted in. None for an intraday interval with "
            "no calendar, because bars per session is a property of the "
            "venue and guessing one would understate the window by whatever "
            "factor the guess was wrong by."
        ),
    )
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Conditions that change how this reads: features that consume no "
            "bars at all, output names requested more than once, and a "
            "conversion to calendar days that could not be made."
        ),
    )


__all__ = [
    "CalibrationDescription",
    "DescribeEstimatorInput",
    "DescribeEstimatorResult",
    "DescribeExchangeCalendarInput",
    "DescribeExchangeCalendarResult",
    "EstimateFeatureWarmupInput",
    "EstimateFeatureWarmupResult",
    "EstimatorDescription",
    "FeatureWarmup",
    "ParamBoundDescription",
]
