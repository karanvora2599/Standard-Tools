"""
The three numbers a spec author could previously only learn by failing.

Each tool here wraps something this library already computes on every
call and that no view returned. The shared shape of the problem: a
constraint is enforced at the boundary, its reason is written down in the
module that enforces it, and the only way an agent could read either was
to send a call that got refused and parse the refusal.

    VENUE ARITHMETIC. `modeling/calendar.py` reads sessions per year,
    session length and bars per session off an installed calendar
    library. The published view of all of it was one boolean saying the
    library was importable, and `DatasetSpec.calendar` could be filled
    only by guessing a code or by reading the eight names that fit in a
    refusal message out of the dozens installed.

    PARAMETER BOUNDS. Behind every bare name in `allowed_params` sits a
    bound with a type, a range, a choice list and a hand-written reason,
    plus cross-parameter rules that no per-parameter bound can express.
    `validate_params` enforces all of it on every fit and nothing
    published any of it, so an agent that knew scikit-learn guessed
    `hidden_layer_sizes` and was refused, every time.

    WARM-UP. `FeatureDefinition.lookback` is a static number recorded at
    registration against DEFAULT parameters. Override a window and it is
    simply wrong -- a momentum feature declared at 20 bars and requested
    at 900 consumes 900 -- and the function that computes the true one
    had no callers at all.

WHY THESE ARE NOT FOLDED INTO THE CAPABILITY REPORT. Describing every
registered estimator runs to tens of kilobytes of JSON -- most of a
capability report's whole budget, spent on a question the caller has
usually already narrowed to one estimator. A separate, filterable tool
costs a call and returns what was asked for, and measures its own
payload rather than quoting a figure that was true on another machine.
See the CHANGELOG entry of 2026-09-21.

WHAT NONE OF THESE DO. No fetching, no building, no fitting, nothing
written to disk. Every refusal they raise is a refusal some other entry
point already raises, deliberately word for word: a calendar code
refused here is refused by `DatasetSpec`, an estimator name refused here
is refused by the allowlist, a feature id refused here is refused by the
registry. A name that passes one of these tools passes the call it was
being prepared for.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, get_args

from standard_quant_tools.error import ValidationError

# Imported as a MODULE, not as names. Every function in it is lru_cached
# and several of them branch on whether the optional library is importable
# at all, so a test that has to simulate a machine without it patches one
# attribute and both this module and the library's own refusal see it --
# which is exactly the environment the "library absent" path exists for.
from .. import calendar as _calendar
from ..dataset.lags import deepest_lag
from ..estimators.boosting import OPTIONAL_ESTIMATORS
from ..estimators.registry import (
    ESTIMATOR_REGISTRY,
    get_estimator_class,
    param_schema,
    quantile_support,
)
from ..features.base import FeatureScope, periods_per_year_for_interval
from ..features.params import resolve_params, resolved_lookback
from ..features.registry import get_feature
from ..specs import EstimatorSpec
from ..tasks import TASKS
from .discovery_models import (
    CalibrationDescription,
    DescribeEstimatorInput,
    DescribeEstimatorResult,
    DescribeExchangeCalendarInput,
    DescribeExchangeCalendarResult,
    EstimateFeatureWarmupInput,
    EstimateFeatureWarmupResult,
    EstimatorDescription,
    FeatureWarmup,
    ParamBoundDescription,
)

logger = logging.getLogger(__name__)

#: Days per year used to turn a bar count into a calendar window. The
#: Julian year, so a leap year does not make the estimate short.
_DAYS_PER_YEAR = 365.25

#: Sessions per year assumed for a daily bar when no venue is named. The
#: convention every annualization in this library falls back to, stated
#: here so the estimate says which number it used rather than implying a
#: venue it was not given.
_CONVENTIONAL_SESSIONS = 252.0


# ── describe_exchange_calendar ─────────────────────────────────────────

DESCRIBE_EXCHANGE_CALENDAR_DESCRIPTION = (
    "List the exchange calendars this installation knows, and resolve one "
    "into the numbers that annualize an intraday model: sessions per year "
    "(counted over the calendar's complete years, so holidays are in it -- "
    "NYSE is 251.6, not 252), session length in minutes (390 for NYSE, 510 "
    "for London, 1440 for a crypto venue), bars per session at a given "
    "interval (a 6.5-hour session at '1h' is SEVEN bars, the stub counted, "
    "because the provider emits it) and bars per year. This is the value "
    "DatasetSpec.calendar takes, and without this tool the only ways to "
    "find a valid code were to guess one or to read the handful of names "
    "that fit inside a refusal. An unrecognised code is refused with the "
    "identical message DatasetSpec gives, so a code that passes here "
    "passes there. A daily-or-coarser interval is NOT refused: it reports "
    "bars_per_session=None and says in warnings why a venue is not needed "
    "for it -- daily, weekly and monthly bars annualize by calendar "
    "arithmetic, and only an intraday interval depends on how long the "
    "venue is open. Where the optional calendar package is not installed, "
    "listing returns an empty catalog with a warning rather than failing, "
    "and resolving a named calendar is refused by name. Fetches nothing."
)


def describe_exchange_calendar(
    input_data: DescribeExchangeCalendarInput,
) -> DescribeExchangeCalendarResult:
    """The venue catalog, and one venue's session arithmetic."""
    warnings: List[str] = []

    # Minutes per bar is a regex over the interval string and needs no
    # library at all, so it is answered even on a machine with no
    # calendars installed -- an agent asking "is '90m' intraday" should
    # not be blocked by a missing optional package.
    minutes = (
        _calendar.interval_minutes(input_data.interval)
        if input_data.interval is not None
        else None
    )

    if not _calendar.calendar_available():
        if input_data.calendar is not None:
            # The refusal by name, identical to the one a spec naming this
            # calendar would get. Asking to RESOLVE a venue is the one
            # request that has no honest answer here.
            _calendar.require_calendar_library("describe_exchange_calendar")
        warnings.append(
            "The optional `exchange_calendars` package is not installed in "
            "this environment, so no venue codes can be listed and no "
            "session arithmetic can be computed (pip install "
            "exchange_calendars). A daily-or-coarser interval needs none of "
            "it; an intraday interval cannot be annualized without it, and "
            "DatasetSpec.calendar will refuse any name on this machine."
        )
        return DescribeExchangeCalendarResult(
            available=False,
            n_calendars=0,
            calendar_names=[],
            interval_minutes=minutes,
            warnings=warnings,
        )

    names = _calendar.calendar_names()
    if input_data.name_contains:
        needle = input_data.name_contains.strip().lower()
        filtered = sorted(n for n in names if needle in n.lower())
        if not filtered:
            warnings.append(
                f"No venue code contains {input_data.name_contains!r}. "
                f"n_calendars reports the {len(names)} codes this "
                "installation knows; call again without name_contains to "
                "see them."
            )
    else:
        filtered = sorted(names)

    resolved: Optional[str] = None
    sessions: Optional[float] = None
    length: Optional[float] = None
    bars: Optional[int] = None
    per_year: Optional[int] = None

    if input_data.calendar is not None:
        # The default `where` on purpose: this refusal must be the one
        # DatasetSpec gives, character for character, or a code that
        # passes this tool could still fail the spec it was being
        # prepared for.
        resolved = _calendar.validate_calendar_name(input_data.calendar)
        sessions = _calendar.sessions_per_year(resolved)
        length = _calendar.session_minutes(resolved)

        if input_data.interval is not None:
            try:
                bars = _calendar.bars_per_session(input_data.interval, resolved)
            except ValidationError as exc:
                # Quoted verbatim rather than paraphrased. Here it is a
                # warning and not a refusal: the sessions and the session
                # length asked for are still true, and the interval is a
                # separate question that has an answer ("you do not need a
                # venue for this one").
                warnings.append(str(exc))
            else:
                per_year = _calendar.periods_per_year(input_data.interval, resolved)
    elif minutes is not None:
        warnings.append(
            f"interval={input_data.interval!r} is an intraday interval "
            f"({minutes} minutes per bar), but bars_per_session and "
            "periods_per_year are properties of a VENUE -- 6.5 hours on "
            "NYSE, 8.5 in London, 24 on a crypto exchange -- so neither can "
            "be computed without a calendar. Pass calendar=<a code from "
            "calendar_names> to get them."
        )

    logger.debug(
        "[describe_exchange_calendar] calendar=%s interval=%s listed=%d/%d",
        resolved,
        input_data.interval,
        len(filtered),
        len(names),
    )
    return DescribeExchangeCalendarResult(
        available=True,
        n_calendars=len(names),
        calendar_names=filtered,
        calendar=resolved,
        sessions_per_year=sessions,
        session_minutes=length,
        interval_minutes=minutes,
        bars_per_session=bars,
        periods_per_year=per_year,
        warnings=warnings,
    )


# ── describe_estimator ─────────────────────────────────────────────────

DESCRIBE_ESTIMATOR_DESCRIPTION = (
    "What an estimator's params actually accept: every parameter name, its "
    "type, its range, its exact choices and the reason behind each bound, "
    "plus the rules BETWEEN parameters that no single bound can express. "
    "All of it is enforced on every fit and none of it was readable, so "
    "the bounds could only be discovered by tripping them -- n_estimators "
    "caps at 2000 and num_leaves at 4096; logistic's penalty='l1' needs "
    "solver='liblinear' or 'saga' and 'elasticnet' needs 'saga' AND an "
    "explicit l1_ratio; sgd accepts a DIFFERENT set of losses per task, "
    "because a classifier here is asked for probabilities unconditionally "
    "and hinge has none; mlp takes n_hidden_units and n_hidden_layers, not "
    "scikit-learn's hidden_layer_sizes tuple, so knowing scikit-learn is "
    "what makes the first call wrong. Also reports `calibration`, which the "
    "capability report omits entirely and which decides outcomes: a raw "
    "random forest's probabilities are compressed by averaging and rarely "
    "exceed 0.9, so at proba_threshold=0.9 one selected ZERO rows where the "
    "isotonic-calibrated model selected 194. Filter with task and/or name; "
    "unfiltered it describes every registered entry and says how large that "
    "is. include_unavailable=True adds the estimators whose optional "
    "library is not installed here -- declared statically, so their names "
    "and bounds are the same on every machine, which is how you learn what "
    "a missing library costs and what a ranking model would be called. An "
    "unknown name is refused with the same message a spec naming it gets. "
    "Fits nothing."
)


def _calibration_description() -> CalibrationDescription:
    """
    The calibration choice, read off `EstimatorSpec` rather than retyped.

    The choices, the default, the fold bounds and the note all come from
    the spec field itself, for the reason the capability report reads its
    Literals instead of listing them: a choice added to the spec and
    copied by hand here is a choice this description gets wrong.
    """
    field = EstimatorSpec.model_fields["calibration"]
    folds = EstimatorSpec.model_fields["calibration_folds"]

    minimum: Optional[int] = None
    maximum: Optional[int] = None
    for constraint in folds.metadata:
        for attribute, setter in (("ge", "minimum"), ("le", "maximum")):
            value = getattr(constraint, attribute, None)
            if value is not None:
                if setter == "minimum":
                    minimum = int(value)
                else:
                    maximum = int(value)

    return CalibrationDescription(
        choices=[a for a in get_args(field.annotation) if isinstance(a, str)],
        default=str(field.default),
        folds_default=int(folds.default),
        folds_minimum=minimum,
        folds_maximum=maximum,
        note=" ".join(str(field.description or "").split()),
        folds_note=" ".join(str(folds.description or "").split()),
    )


def _logistic_solver_matrix() -> str:
    """
    The solver x penalty matrix, rendered from the dict that enforces it.

    The check itself is a plain function with no docstring, so there is no
    text to quote; reading the table it validates against is the only
    rendering that cannot drift from what a fit would accept.
    """
    from ..estimators.bounds import _LOGISTIC_SOLVER_PENALTIES

    rows = "; ".join(
        f"{solver} accepts "
        + ", ".join("None" if p is None else repr(p) for p in penalties)
        for solver, penalties in _LOGISTIC_SOLVER_PENALTIES.items()
    )
    return (
        "penalty and solver are a MATRIX, not two independent choices -- "
        f"{rows}. So penalty='l1' needs solver='liblinear' or 'saga', and "
        "penalty='elasticnet' needs solver='saga' and an explicit l1_ratio "
        "beside it. Rendered from the solver/penalty table the estimator "
        "allowlist validates against (estimators/bounds.py), which is "
        "scikit-learn's own matrix, so it cannot disagree with what a fit "
        "would accept."
    )


#: Cross-parameter checks whose rule is a table rather than a sentence.
#: A check that carries a docstring is quoted; one that does not is
#: rendered here from the constant it enforces, because a note invented
#: for it would be a second source of truth.
_COMPATIBILITY_RENDERERS = {"_logistic_compatibility": _logistic_solver_matrix}


def _compatibility_notes(name: str, schema: Any) -> List[str]:
    """Every cross-parameter rule on this estimator, in words."""
    notes: List[str] = []
    for check in schema.compatibility:
        check_name = getattr(check, "__name__", "")
        renderer = _COMPATIBILITY_RENDERERS.get(check_name)
        if renderer is not None:
            notes.append(renderer())
            continue
        doc = inspect.getdoc(check)
        if doc:
            notes.append(" ".join(doc.split()))
            continue
        notes.append(
            f"estimator {name!r} enforces a rule across its parameters "
            f"({check_name or 'unnamed check'}) that carries no description "
            "of its own. The combination is checked when the spec is "
            "validated, so the refusal states the rule; nothing here can."
        )
    return notes


def _bound_description(bound: Any) -> ParamBoundDescription:
    return ParamBoundDescription(
        kind=str(bound.kind),
        minimum=bound.minimum,
        maximum=bound.maximum,
        choices=(
            None
            if bound.choices is None
            else [None if c is None else str(c) for c in bound.choices]
        ),
        allow_none=bool(bound.allow_none),
        note=str(bound.note or ""),
    )


def _describe_one(task: str, name: str) -> EstimatorDescription:
    """One entry, registered here or declared and unavailable."""
    key = (task, name)
    optional = OPTIONAL_ESTIMATORS.get(key)
    available = key in ESTIMATOR_REGISTRY

    if available:
        schema = param_schema(task, name)
        cls = ESTIMATOR_REGISTRY[key]
        class_path: Optional[str] = f"{cls.__module__}.{cls.__qualname__}"
        support = quantile_support(task, name)
        quantile_param = support.param if support is not None else None
    else:
        # The static declaration, which exists whether or not the library
        # does. This is the whole reason an uninstalled estimator can be
        # named and budgeted for at all.
        schema = optional[1]
        class_path = None
        quantile_param = None

    return EstimatorDescription(
        task=task,
        name=name,
        available=available,
        requires_library=optional[0] if optional is not None else None,
        class_path=class_path,
        quantile_param=quantile_param,
        params={
            param: _bound_description(bound) for param, bound in schema.bounds.items()
        },
        compatibility_notes=_compatibility_notes(name, schema),
        calibration=(_calibration_description() if task == "classification" else None),
    )


def _selected_keys(
    task: Optional[str], name: Optional[str], include_unavailable: bool
) -> List[Tuple[str, str]]:
    """Which entries the filters ask for, refusing an unknown name the way
    the allowlist does."""
    keys = set(ESTIMATOR_REGISTRY)
    if include_unavailable:
        keys |= set(OPTIONAL_ESTIMATORS)
    selected = sorted(
        key
        for key in keys
        if (task is None or key[0] == task) and (name is None or key[1] == name)
    )
    if selected or name is None:
        return selected

    # A name that matched nothing. An OPTIONAL pair is described as
    # unavailable rather than refused even without include_unavailable:
    # the caller named it, so "that estimator exists and its library does
    # not" is the answer, and the allowlist's "unknown name" would be a
    # lie about a machine-independent declaration.
    optional_matches = sorted(
        key
        for key in OPTIONAL_ESTIMATORS
        if (task is None or key[0] == task) and key[1] == name
    )
    if optional_matches:
        return optional_matches

    if task is not None:
        # Identical to what a spec naming this estimator is refused with.
        get_estimator_class(task, name)
    raise ValidationError(
        f"describe_estimator: no estimator named {name!r} is registered "
        f"under any task — registered names: "
        f"{sorted({n for _t, n in ESTIMATOR_REGISTRY})}. Pass "
        "include_unavailable=True to also describe the optional estimators "
        "whose library is not installed here, or pass task= to get the "
        "allowlist for one task."
    )


def describe_estimator(input_data: DescribeEstimatorInput) -> DescribeEstimatorResult:
    """The bounds, the cross-parameter rules and the calibration choice."""
    task = input_data.task
    if task is not None and task not in TASKS:
        raise ValidationError(
            f"describe_estimator: task={task!r} is not a supervised task "
            f"this library fits — allowed: {sorted(TASKS)}."
        )

    keys = _selected_keys(task, input_data.name, input_data.include_unavailable)
    entries = [_describe_one(entry_task, name) for entry_task, name in keys]

    warnings: List[str] = []
    if not entries:
        warnings.append(
            f"No estimator is registered for task={task!r} on this machine. "
            "Pass include_unavailable=True to see the optional estimators "
            "that would serve it and the library each one needs."
        )
    if task is None and input_data.name is None:
        # Measured on what is actually about to be returned, not quoted
        # from a reading taken once: the payload moves with the installed
        # libraries, and a number that was true on another machine would
        # be the wrong thing to budget against.
        size = len(json.dumps([entry.model_dump() for entry in entries], default=str))
        warnings.append(
            f"Unfiltered: this describes {len(entries)} estimator(s) and is "
            f"about {size / 1024:.0f} KB of JSON. Pass task= and/or name= "
            "to describe the one you are choosing between instead -- a "
            "single entry is a fraction of that."
        )
    unavailable = [f"{e.task}.{e.name}" for e in entries if not e.available]
    if unavailable:
        libraries = sorted(
            {
                e.requires_library
                for e in entries
                if not e.available and e.requires_library
            }
        )
        warnings.append(
            f"{len(unavailable)} of these are NOT installed here and cannot "
            f"be fitted as things stand: {unavailable}. They need "
            f"{libraries}. Their names and bounds are declared statically "
            "and are the same on every machine, which is what this listing "
            "is for -- a spec naming one of them is refused until the "
            "library is installed."
        )

    logger.debug(
        "[describe_estimator] task=%s name=%s entries=%d unavailable=%d",
        task,
        input_data.name,
        len(entries),
        len(unavailable),
    )
    return DescribeEstimatorResult(
        estimators=entries,
        n_estimators_described=len(entries),
        warnings=warnings,
    )


# ── estimate_feature_warmup ────────────────────────────────────────────

ESTIMATE_FEATURE_WARMUP_DESCRIPTION = (
    "How many bars of history a feature spec burns before its first usable "
    "row, at the parameters you are actually requesting. The catalog's "
    "lookback is a static number recorded against a feature's DEFAULT "
    "parameters, so it is simply wrong the moment a window is overridden: "
    "market.momentum is catalogued at 20 bars and consumes 900 when asked "
    "for lookback=900, and statistical.hurst is catalogued at 200 and "
    "consumes 500 at window=500. This reports declared beside resolved for "
    "every feature, names the one that BINDS (the only one worth "
    "shortening), adds the deepest lag on top -- lags are warm-up too, "
    "charged once at the deepest -- and converts the total into calendar "
    "days, which is the unit score_model's lookback_days is given in. That "
    "argument has no default that can be right for every spec: too small "
    "and scoring refuses with an empty panel, and nothing in the library "
    "derived it. Point-in-time features contribute nothing and are named, "
    "because their freshness is a staleness bound on records rather than a "
    "count of bars. This is the pre-build form of the question "
    "explain_dataset_row_loss answers afterwards, once a build has already "
    "been paid for. Unknown feature ids and invalid lags are refused here "
    "exactly as the dataset builder would refuse them. Fetches nothing."
)


def _calendar_days(
    bars: int, interval: str, calendar: Optional[str], warnings: List[str]
) -> Optional[float]:
    """`bars` as a span of calendar days, or None with a reason."""
    minutes = _calendar.interval_minutes(interval)

    if minutes is not None:
        if calendar is None:
            warnings.append(
                f"interval={interval!r} is intraday, and bars per session is "
                "a property of the venue -- 6.5 hours on NYSE, 24 on a "
                "crypto exchange. Without calendar= there is no honest way "
                "to turn a bar count into days, and a guess would be wrong "
                "by whatever factor the venue differs by, so "
                "calendar_days_estimate is left empty. Pass a calendar code "
                "(describe_exchange_calendar lists them)."
            )
            return None
        per_session = _calendar.bars_per_session(interval, calendar)
        sessions = bars / float(per_session)
        return sessions / _calendar.sessions_per_year(calendar) * _DAYS_PER_YEAR

    # Daily or coarser: calendar arithmetic, no venue required.
    per_year = periods_per_year_for_interval(interval, None)
    if per_year is None:
        warnings.append(
            f"interval={interval!r} is neither an intraday interval this "
            "library can place in a session nor one of the daily-or-coarser "
            "intervals it annualizes by arithmetic, so bars_required could "
            "not be converted to calendar days. bars_required itself is "
            "unaffected -- it counts bars of whatever interval you named."
        )
        return None

    sessions_a_year = float(per_year)
    # A DAILY bar is a session, so a named venue's own session count
    # replaces the 252 convention. A weekly or monthly bar is not a
    # session, and no venue changes how many weeks are in a year.
    if calendar is not None and sessions_a_year == _CONVENTIONAL_SESSIONS:
        sessions_a_year = _calendar.sessions_per_year(calendar)
    return bars / sessions_a_year * _DAYS_PER_YEAR


def estimate_feature_warmup(
    input_data: EstimateFeatureWarmupInput,
) -> EstimateFeatureWarmupResult:
    """Bars of history this feature spec consumes before its first row."""
    warnings: List[str] = []
    per_feature: Dict[str, FeatureWarmup] = {}
    point_in_time: List[str] = []

    for spec in input_data.features:
        # The registry's refusal and the parameter validator's, unchanged:
        # an id or a window this tool accepts is one build_model_dataset
        # accepts, and there is no second opinion about either here.
        definition = get_feature(spec.id)
        resolved_params = resolve_params(definition, spec.params)
        is_pit = definition.scope == FeatureScope.POINT_IN_TIME

        # A point-in-time feature reads filings, not bars, and its
        # definition is forbidden from declaring a bar lookback at all.
        # Charged as zero explicitly rather than by relying on that, so a
        # record-set parameter that happens to be named like a window can
        # never leak into a bar count.
        resolved = 0 if is_pit else int(resolved_lookback(definition, resolved_params))
        entry = FeatureWarmup(
            declared=int(definition.lookback),
            resolved=resolved,
            lags=list(spec.lags),
            deepest_lag=max(spec.lags) if spec.lags else 0,
            point_in_time=is_pit,
        )
        if is_pit:
            point_in_time.append(spec.output_name)

        existing = per_feature.get(spec.output_name)
        if existing is not None:
            warnings.append(
                f"output name {spec.output_name!r} is requested more than "
                "once. The panel keys one column per output name, so these "
                "specs collide -- build_model_dataset refuses the "
                "collision, and the deeper of the two is reported here. "
                "Give one of them an alias."
            )
            if existing.resolved >= entry.resolved:
                continue
        per_feature[spec.output_name] = entry

    if point_in_time:
        warnings.append(
            f"{len(point_in_time)} point-in-time feature(s) contribute NO "
            f"bar warm-up and cannot bind: {sorted(point_in_time)}. They "
            "read a record set rather than bars, and how fresh a record may "
            "be is bounded by their max_staleness_days parameter -- a "
            "different unit, not comparable with bars_required and not "
            "included in it."
        )

    from_bars = {
        name: entry for name, entry in per_feature.items() if not entry.point_in_time
    }
    max_lookback = max((entry.resolved for entry in from_bars.values()), default=0)
    binding: Optional[str] = None
    if max_lookback > 0:
        binding = max(from_bars, key=lambda name: from_bars[name].resolved)

    # The deepest lag across the WHOLE spec, charged once: the panel
    # starts where its last column becomes computable, not once per
    # feature that asked for history.
    deepest = int(deepest_lag(input_data.features))
    bars_required = int(max_lookback + deepest)

    calendar = input_data.calendar
    if calendar is not None:
        calendar = _calendar.validate_calendar_name(
            calendar, "estimate_feature_warmup.calendar"
        )
    days = _calendar_days(bars_required, input_data.interval, calendar, warnings)

    logger.debug(
        "[estimate_feature_warmup] bars=%d binding=%s deepest_lag=%d days=%s",
        bars_required,
        binding,
        deepest,
        days,
    )
    return EstimateFeatureWarmupResult(
        bars_required=bars_required,
        per_feature=per_feature,
        binding_feature=binding,
        deepest_lag=deepest,
        calendar_days_estimate=days,
        warnings=warnings,
    )


__all__ = [
    "DESCRIBE_ESTIMATOR_DESCRIPTION",
    "DESCRIBE_EXCHANGE_CALENDAR_DESCRIPTION",
    "ESTIMATE_FEATURE_WARMUP_DESCRIPTION",
    "DescribeEstimatorInput",
    "DescribeEstimatorResult",
    "DescribeExchangeCalendarInput",
    "DescribeExchangeCalendarResult",
    "EstimateFeatureWarmupInput",
    "EstimateFeatureWarmupResult",
    "describe_estimator",
    "describe_exchange_calendar",
    "estimate_feature_warmup",
]
