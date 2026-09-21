"""
Two statistics this library computed on every call and nobody could read.

WHETHER THE BAND COVERED. An experiment that asks for quantiles or a
conformal interval scores them on every fold -- pinball loss per level,
the crossing rate, coverage and width of every symmetric pair -- and the
fold loop averages the result into the headline metrics, where it is one
number per name and cannot be recomputed. Meanwhile the columns
themselves ARE handed over: the out-of-sample frame carries `q05`, `q50`,
`q95`, `lower` and `upper` when the spec asked for them, it is published
as a reference, and scoring re-emits them on new data. So the library
produced intervals, published them, re-emitted them, and had no way to
ask whether they covered.

The `by` axis is the reason this is a tool and not a field. Exchangeability
is the assumption behind a conformal band, and the conformal module's own
docstring states the limit plainly: "a return panel is not exchangeable
across regimes: coverage is honest on average over the window and can
fail in a regime the calibration window did not contain. The reported OOS
coverage is the check." One pooled number is exactly what cannot separate
a band that covered 97% in a calm stretch from the same band covering 62%
in a selloff. Per date, the two are two rows.

WHETHER B BEAT A, AND HOW MANY OTHERS WERE ASKED. The Holm adjustment
reached the tool layer at one point only: a comparison of two REGISTERED
models sharing a task and a label, refusing one validation method by name
and arranged as a one-against-many star. A researcher with twelve
candidate signals, or twelve p-values from anywhere at all, had no
family-wise correction available -- and the multiple-testing tools that
do exist elsewhere take return series, not p-values. The
autocorrelation-consistent variance of a mean had no caller outside its
own module while two other runtimes printed a warning that their
t-statistics might not survive it.

WHAT NEITHER OF THESE DOES. No fitting, no fetching, nothing written to
disk, and no registry lookup: both read what they are handed. A
prediction frame from another library scores here exactly as one from
this one, and a p-value computed in a notebook adjusts here exactly as
one computed by a model comparison -- which is the whole point of the
third mode.

See the CHANGELOG entry of 2026-09-21.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.agent.runtimes._json_safe import finite_or_none
from standard_quant_tools.error import ValidationError

from ..validation.comparison import (
    bh_adjust,
    bonferroni_adjust,
    compare_ic_series,
    holm_adjust,
    newey_west_variance,
    paired_comparison,
)
from ..validation.distributional import distributional_metrics, quantile_column
from .statistics_models import (
    AdjustedPValue,
    CompareSignalsInput,
    CompareSignalsResult,
    IntervalGroup,
    ScorePredictionIntervalsInput,
    ScorePredictionIntervalsResult,
)

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Every non-finite float in a nested structure turned into a null.

    `NaN` is not valid JSON and several clients reject the whole response
    at the transport layer rather than the one field that could not be
    computed. The comparison blocks below are dictionaries the validation
    layer built for a Python caller, so they are sanitized on the way out
    rather than field by field.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    return finite_or_none(value)


# ── score_prediction_intervals ─────────────────────────────────────────

SCORE_PREDICTION_INTERVALS_DESCRIPTION = (
    "Score the quantile and interval columns of a published predictions "
    "reference against the realized outcome: pinball loss per level, the "
    "quantile crossing rate, and coverage AND width for every symmetric "
    "quantile pair and for the lower/upper band. These are the numbers an "
    "experiment computes on every fold and averages away, on columns it "
    "hands over and that nothing could previously check. Read coverage and "
    "width together, always: a band covering 90% of outcomes by spanning "
    "ten times the target's spread has told nobody anything, and coverage "
    "alone cannot tell the two apart. Set by='date' (or 'entity') for the "
    "breakdown that matters most -- exchangeability is the assumption "
    "behind a conformal interval and a return panel is not exchangeable "
    "across regimes, so a band that covered 97% in a calm stretch and 62% "
    "in a selloff pools to a healthy-looking 90%, and only the per-date "
    "view separates them. Quantile columns are auto-detected by the "
    "convention the library writes them under (q05, q50, q95); a frame "
    "with neither quantile nor lower/upper columns is refused naming what "
    "was looked for, rather than returning empty statistics. Nothing is "
    "fitted and nothing is fetched: this reads the frame it is given, so "
    "predictions from another library score on the same yardstick."
)

#: Levels closer together than this are the same level, matching the
#: tolerance `distributional_metrics` pairs quantiles with.
_LEVEL_TOLERANCE = 1e-9


def _detect_quantile_columns(frame: pd.DataFrame) -> Dict[str, float]:
    """
    Which columns are quantiles, by the round trip that wrote them.

    A column is a quantile column when parsing its name back to a level
    and re-deriving the name yields the same string -- so `q05` is 0.05
    and `q02.5` is 0.025, while `q5`, `quantile` and `q120` are not
    quantile columns at all. Recognising them by the writer's own
    function rather than a second regex is what keeps a rename on one
    side from silently producing an empty result on the other.
    """
    found: Dict[str, float] = {}
    for column in frame.columns:
        name = str(column)
        if not name.startswith("q"):
            continue
        try:
            percent = float(name[1:])
        except ValueError:
            continue
        level = percent / 100.0
        if not 0.0 < level < 1.0:
            continue
        if quantile_column(level) == name:
            found[name] = level
    return found


def _resolve_quantile_columns(
    frame: pd.DataFrame, requested: Optional[Dict[str, float]]
) -> Dict[str, float]:
    """The explicit mapping, checked against the frame, or the detected one."""
    if requested is None:
        return _detect_quantile_columns(frame)
    resolved: Dict[str, float] = {}
    for column, level in requested.items():
        if column not in frame.columns:
            raise ValidationError(
                f"score_prediction_intervals: quantile_columns names "
                f"{column!r}, which the frame does not carry; it holds "
                f"{list(frame.columns)}. Drop it from quantile_columns, or "
                "omit quantile_columns entirely to score whichever q-columns "
                "are present."
            )
        value = float(level)
        if not 0.0 < value < 1.0:
            raise ValidationError(
                f"score_prediction_intervals: quantile_columns maps "
                f"{column!r} to {value!r}, which is not a quantile level. A "
                "level is strictly between 0 and 1 -- 0.05 for the 5th "
                "percentile, not 5."
            )
        clash = next(
            (
                other
                for other, seen in resolved.items()
                if abs(seen - value) < _LEVEL_TOLERANCE
            ),
            None,
        )
        if clash is not None:
            raise ValidationError(
                f"score_prediction_intervals: quantile_columns maps both "
                f"{clash!r} and {column!r} to level {value}. Two columns at "
                "one level would each be scored as the other's pair; give "
                "each level exactly one column."
            )
        resolved[column] = value
    return resolved


def _nominal_for_pair_key(key: str) -> Optional[float]:
    """The claimed coverage behind a `quantile_coverage_90` key."""
    tail = key.rsplit("_", 1)[-1]
    try:
        return float(tail) / 100.0
    except ValueError:
        return None


def _two_sigma_band(nominal: float, n_rows: int) -> Optional[float]:
    """
    Two standard errors of a coverage proportion measured on `n_rows`.

    Coverage is a share of Bernoulli draws, so its own sampling error is
    sqrt(p(1-p)/n): on 200 rows a truthful 90% band measures between 86%
    and 94% routinely. A gap inside this is the sample, not the model,
    and saying otherwise would make every honest interval look broken.
    """
    if n_rows < 1 or not 0.0 < nominal < 1.0:
        return None
    return 2.0 * math.sqrt(nominal * (1.0 - nominal) / float(n_rows))


def _group_metrics(
    subset: pd.DataFrame,
    target_column: str,
    levels: Dict[str, float],
    interval_columns: Optional[Tuple[str, str]],
    alpha: Optional[float],
) -> Dict[str, float]:
    """`distributional_metrics` on one slice of the frame."""
    y_true = subset[target_column].to_numpy(dtype=float)
    quantile_predictions = {
        level: subset[column].to_numpy(dtype=float) for column, level in levels.items()
    }
    lower = upper = None
    if interval_columns is not None:
        lower = subset[interval_columns[0]].to_numpy(dtype=float)
        upper = subset[interval_columns[1]].to_numpy(dtype=float)
    return distributional_metrics(
        y_true, quantile_predictions, lower=lower, upper=upper, alpha=alpha
    )


def score_prediction_intervals(
    input_data: ScorePredictionIntervalsInput,
) -> ScorePredictionIntervalsResult:
    """
    Did the band cover, and what did the coverage cost in width?

    COVERAGE ALONE IS NOT A VERDICT. An interval wide enough to contain
    every outcome covers 100% and is worth nothing, so the width of every
    band scored is reported beside its coverage in the target's own units.
    Pinball loss is the score that cannot be gamed this way: it is
    minimized by a quantile that is exceeded exactly as often as it
    claims.

    THE POOLED NUMBER IS THE ONE THAT MISLEADS. Coverage averaged over a
    whole window is honest on average and can be wrong everywhere: the
    calm stretch over-covers, the selloff under-covers, and the mean of
    the two looks like the promise being kept. `by='date'` is the axis
    that separates them, and with `by='all'` the caveat is in `warnings`
    rather than left to be remembered.
    """
    from standard_quant_tools.agent.runtimes import handoff

    frame = handoff.resolve(input_data.predictions_ref, expect="predictions")
    target_column = input_data.target_column
    if target_column not in frame.columns:
        raise ValidationError(
            f"score_prediction_intervals: the frame has no "
            f"{target_column!r} column to judge the band against; it holds "
            f"{list(frame.columns)}. Coverage is the share of REALIZED "
            "outcomes that fell inside the band, so there is nothing to "
            "compute without one -- pass target_column=<the realized "
            "outcome>, or attach outcomes to these predictions first."
        )

    levels = _resolve_quantile_columns(frame, input_data.quantile_columns)
    lower_column = input_data.lower_column
    upper_column = input_data.upper_column
    has_lower = lower_column in frame.columns
    has_upper = upper_column in frame.columns
    if has_lower != has_upper:
        present, absent = (
            (lower_column, upper_column) if has_lower else (upper_column, lower_column)
        )
        raise ValidationError(
            f"score_prediction_intervals: the frame carries {present!r} but "
            f"not {absent!r}. Half a band has no coverage and no width -- "
            "there is no edge on the other side for an outcome to fall "
            f"inside of. Name the column that holds it "
            f"({'upper_column' if has_lower else 'lower_column'}=...), or "
            "drop the one-sided column and score the quantile columns "
            "instead."
        )
    interval_columns = (lower_column, upper_column) if has_lower and has_upper else None

    if not levels and interval_columns is None:
        raise ValidationError(
            "score_prediction_intervals: the frame carries nothing "
            f"distributional to score. It holds {list(frame.columns)}; this "
            f"looked for {lower_column!r} and {upper_column!r} (a prediction "
            "interval) and for quantile columns named the way this library "
            "writes them -- 'q05', 'q50', 'q95', or 'q02.5' for a level that "
            "is not a whole percent. A point prediction has no coverage to "
            "check: score it with score_predictions, or re-run the "
            "experiment with quantiles or intervals requested in the spec."
        )

    scored_columns = [target_column, *sorted(levels)]
    if interval_columns is not None:
        scored_columns.extend(interval_columns)
    data = frame.dropna(subset=scored_columns)
    if data.empty:
        raise ValidationError(
            f"score_prediction_intervals: all {len(frame)} row(s) are "
            "missing either the realized outcome or a band edge, so no row "
            f"can be judged. The columns scored were {scored_columns}. A "
            "frame of predictions whose outcomes have not been realized "
            "yet cannot be checked for coverage -- wait for the labels, or "
            "score the window that has them."
        )
    n_rows = int(len(data))

    group_column = {"date": "date", "entity": "entity"}.get(input_data.by)
    keys: Optional[pd.Series] = None
    if group_column is not None:
        if group_column not in data.columns:
            raise ValidationError(
                f"score_prediction_intervals: by={input_data.by!r} needs a "
                f"{group_column!r} column and the frame holds "
                f"{list(data.columns)}. Use by='all' for a pooled coverage, "
                f"or publish a frame that carries {group_column!r}."
            )
        if group_column == "date":
            keys = pd.to_datetime(data[group_column]).dt.strftime("%Y-%m-%d")
        else:
            keys = data[group_column].astype(str)
        n_keys = int(keys.nunique())
        if n_keys > input_data.max_groups:
            raise ValidationError(
                f"score_prediction_intervals: by={input_data.by!r} would "
                f"return {n_keys} groups and max_groups is "
                f"{input_data.max_groups}. A row per group for a panel this "
                "wide is more JSON than any reader gets through, and the "
                "spread it is meant to show is already in worst_group and "
                "best_group. Narrow the frame to the window in question, "
                "use by='entity' (or 'all'), or raise max_groups "
                "deliberately."
            )

    alpha = 1.0 - input_data.nominal_coverage if interval_columns is not None else None
    overall = _group_metrics(data, target_column, levels, interval_columns, alpha)

    pinball = {
        key: value for key, value in overall.items() if key.startswith("pinball_")
    }
    quantile_coverage = {
        key: value
        for key, value in overall.items()
        if key.startswith("quantile_coverage_")
    }
    quantile_width = {
        key: value
        for key, value in overall.items()
        if key.startswith("quantile_width_")
    }

    groups: List[IntervalGroup] = []
    if keys is not None:
        # Iterating the groups themselves rather than looking their index
        # labels back up: a published frame's index is whatever it was
        # published with, and a duplicated label would make a `.loc`
        # lookup pull rows from a neighbouring group into this one's
        # coverage.
        for key, subset in data.groupby(keys, sort=True):
            metrics = _group_metrics(
                subset, target_column, levels, interval_columns, alpha
            )
            groups.append(
                IntervalGroup(
                    key=str(key),
                    n_rows=int(len(subset)),
                    interval_coverage=metrics.get("interval_coverage"),
                    quantile_coverage={
                        name: value
                        for name, value in metrics.items()
                        if name.startswith("quantile_coverage_")
                    },
                    crossing_rate=metrics.get("quantile_crossing_rate"),
                )
            )
        groups.sort(key=lambda group: group.key)

    warnings = _interval_warnings(
        input_data=input_data,
        n_rows=n_rows,
        overall=overall,
        quantile_coverage=quantile_coverage,
        quantile_width=quantile_width,
        groups=groups,
    )
    worst, best = _worst_and_best(
        groups, input_data.min_group_rows, float(input_data.nominal_coverage)
    )

    logger.debug(
        "[score_prediction_intervals] ref=%s rows=%d levels=%s groups=%d by=%s",
        input_data.predictions_ref,
        n_rows,
        sorted(levels),
        len(groups),
        input_data.by,
    )
    return ScorePredictionIntervalsResult(
        n_rows=n_rows,
        n_groups=len(groups),
        quantile_levels=dict(sorted(levels.items())),
        pinball=pinball,
        quantile_crossing_rate=overall.get("quantile_crossing_rate"),
        quantile_coverage=quantile_coverage,
        quantile_width=quantile_width,
        interval_coverage=overall.get("interval_coverage"),
        interval_width=overall.get("interval_width"),
        interval_nominal_coverage=overall.get("interval_nominal_coverage"),
        groups=groups,
        worst_group=worst,
        best_group=best,
        warnings=warnings,
    )


def _group_coverage_and_nominal(
    group: IntervalGroup, interval_nominal: float
) -> Tuple[Optional[float], Optional[float]]:
    """
    A group's headline coverage and the level it claimed.

    The conformal band when there is one -- its claim comes from the
    input, because two columns of numbers do not carry the alpha they
    were built at -- otherwise the WIDEST symmetric quantile pair, whose
    claim is in its own name.
    """
    if group.interval_coverage is not None:
        return group.interval_coverage, interval_nominal
    pairs = {
        key: value
        for key, value in group.quantile_coverage.items()
        if value is not None
    }
    if not pairs:
        return None, None
    widest = max(pairs, key=lambda key: _nominal_for_pair_key(key) or 0.0)
    return pairs[widest], _nominal_for_pair_key(widest)


def _group_deviation(group: IntervalGroup, interval_nominal: float) -> Optional[float]:
    """|coverage - claimed| for one group, or None when it has neither."""
    coverage, nominal = _group_coverage_and_nominal(group, interval_nominal)
    if coverage is None or nominal is None:
        return None
    return abs(coverage - nominal)


def _worst_and_best(
    groups: List[IntervalGroup], min_group_rows: int, interval_nominal: float
) -> Tuple[Optional[str], Optional[str]]:
    """
    The groups furthest from and closest to what the band claimed.

    Only groups that clear `min_group_rows` are eligible. Coverage on
    eight rows takes one of nine values, and nominating the lowest of
    them as the regime where the band failed would be reporting sampling
    noise with a date attached.
    """
    readable = [
        (group, _group_deviation(group, interval_nominal))
        for group in groups
        if group.n_rows >= min_group_rows
    ]
    readable = [pair for pair in readable if pair[1] is not None]
    if not readable:
        return None, None
    worst = max(readable, key=lambda pair: pair[1])[0]
    best = min(readable, key=lambda pair: pair[1])[0]
    return worst.key, best.key


def _interval_warnings(
    *,
    input_data: ScorePredictionIntervalsInput,
    n_rows: int,
    overall: Dict[str, float],
    quantile_coverage: Dict[str, float],
    quantile_width: Dict[str, float],
    groups: List[IntervalGroup],
) -> List[str]:
    """Everything that changes how these coverage numbers should be read."""
    warnings: List[str] = []
    interval_nominal = float(input_data.nominal_coverage)

    if input_data.by == "all":
        warnings.append(
            "Coverage here is POOLED over every row. Exchangeability is the "
            "assumption behind an interval like this, and the conformal "
            "module states its limit: 'a return panel is not exchangeable "
            "across regimes: coverage is honest on average over the window "
            "and can fail in a regime the calibration window did not "
            "contain. The reported OOS coverage is the check.' One pooled "
            "number cannot separate a band that covered 97% in a calm "
            "stretch from the same band covering 62% in a selloff -- their "
            "average is the 90% printed here. Re-run with by='date' to see "
            "which of the two this was."
        )

    crossing = overall.get("quantile_crossing_rate")
    if crossing is not None and crossing > 0.0:
        warnings.append(
            f"The quantiles are out of order on {crossing:.1%} of rows: a "
            "lower level was predicted ABOVE a higher one, which separately "
            "fitted quantile models can do and a single fit cannot. On "
            "those rows the pair is not a distribution at all and the width "
            "between them is NEGATIVE, which drags the mean width reported "
            "here below the typical honest width. Fit the quantiles jointly, "
            "or sort each row's quantiles before reading a width off them."
        )

    interval_coverage = overall.get("interval_coverage")
    if interval_coverage is not None:
        nominal = float(input_data.nominal_coverage)
        band = _two_sigma_band(nominal, n_rows)
        if band is not None and abs(interval_coverage - nominal) > band:
            direction = "below" if interval_coverage < nominal else "above"
            bought = (
                "the band is too narrow and its promise is not being kept"
                if interval_coverage < nominal
                else "the extra coverage was bought with width -- read "
                "interval_width before calling this good"
            )
            warnings.append(
                f"Realized interval coverage {interval_coverage:.3f} is "
                f"{direction} the claimed {nominal:.3f} by more than the "
                f"sampling error of {n_rows} rows (two standard errors is "
                f"{band:.3f}), so {bought}."
            )

    for key, coverage in quantile_coverage.items():
        nominal = _nominal_for_pair_key(key)
        if coverage is None or nominal is None:
            continue
        band = _two_sigma_band(nominal, n_rows)
        if band is None or abs(coverage - nominal) <= band:
            continue
        width_key = key.replace("quantile_coverage_", "quantile_width_")
        width = quantile_width.get(width_key)
        width_note = f" Its mean width is {width:.6g}." if width is not None else ""
        warnings.append(
            f"{key} is {coverage:.3f} against a claimed {nominal:.3f}, "
            f"outside the two-standard-error band of {band:.3f} on "
            f"{n_rows} rows. A quantile pair that misses its own level is "
            "mis-calibrated, not merely unlucky, at this sample size."
            f"{width_note}"
        )

    if groups:
        short = [group for group in groups if group.n_rows < input_data.min_group_rows]
        if short:
            named = ", ".join(group.key for group in short[:5])
            more = "" if len(short) <= 5 else f" (and {len(short) - 5} more)"
            warnings.append(
                f"{len(short)} of {len(groups)} group(s) hold fewer than "
                f"{input_data.min_group_rows} rows: {named}{more}. Their "
                "coverage is reported and is not readable -- on a handful "
                "of rows a proportion can only take a handful of values, so "
                "neither a 1.0 nor a 0.5 there is evidence about the band. "
                "They are excluded from worst_group, best_group and the "
                "spread check."
            )
        measured = [
            (group, *_group_coverage_and_nominal(group, interval_nominal))
            for group in groups
            if group.n_rows >= input_data.min_group_rows
        ]
        coverages = [
            (group, coverage)
            for group, coverage, nominal in measured
            if coverage is not None and nominal is not None
        ]
        claimed = next(
            (nominal for _, _, nominal in measured if nominal is not None),
            interval_nominal,
        )
        values = [value for _, value in coverages if value is not None]
        if len(values) >= 2:
            spread = max(values) - min(values)
            typical_rows = max(
                int(np.median([group.n_rows for group, _ in coverages])), 1
            )
            band = _two_sigma_band(claimed, typical_rows)
            if band is not None and spread > band:
                low = min(coverages, key=lambda pair: pair[1])
                high = max(coverages, key=lambda pair: pair[1])
                warnings.append(
                    f"Coverage VARIES ACROSS {input_data.by}s by {spread:.3f}, "
                    f"from {low[1]:.3f} on {low[0].key} to {high[1]:.3f} on "
                    f"{high[0].key}, which is wider than the {band:.3f} the "
                    "sampling error of a typical group explains. The pooled "
                    "number is an average over regimes the band did not "
                    "treat alike, which is exactly the failure "
                    "exchangeability does not rule out."
                )
    return warnings


# ── compare_signals ────────────────────────────────────────────────────

COMPARE_SIGNALS_DESCRIPTION = (
    "Decide whether one signal actually beat another, and correct a family "
    "of p-values for having asked more than once. Three modes. "
    "mode='paired' compares two published prediction frames on the rows "
    "both predicted -- the per-date difference series, a block-bootstrap "
    "interval on its mean, and a Diebold-Mariano loss test where the task "
    "has a loss with units -- with no registry and no shared manifest "
    "required, so an externally computed alpha compares against a model "
    "here. mode='ic_series' takes two per-date information-coefficient "
    "series inline from anywhere and adds the Newey-West variance of the "
    "difference beside the ordinary one: the ratio says how much a "
    "t-statistic computed without the correction was overstated, and on a "
    "real series it has been measured at 2.8. mode='adjust' applies Holm, "
    "Bonferroni or Benjamini-Hochberg to p-values from ANY source -- the "
    "correction that a researcher with twelve candidate signals otherwise "
    "has no way to reach. Read the verdict, not the headline: a difference "
    "of 0.006 in daily IC is routinely inside the noise of one "
    "out-of-sample sample, and sorting on it selects the model that got "
    "the friendlier draw. None of these corrections controls for the "
    "candidates having been SELECTED on this same sample; that is what "
    "run_reality_check exists for, and the result says so."
)

#: Which mode reads which field. A field from another mode is refused by
#: name rather than ignored: silently dropping `p_values` on a paired call
#: would run the comparison and never mention that the correction the
#: caller asked for did not happen.
_FIELD_OWNER: Dict[str, str] = {
    "predictions_ref_a": "'paired'",
    "predictions_ref_b": "'paired'",
    "task": "'paired'",
    "metric": "'paired'",
    "horizon": "'paired'",
    "ic_a": "'ic_series'",
    "ic_b": "'ic_series'",
    "hac_lag": "'ic_series'",
    "n_bootstrap": "'paired' or 'ic_series'",
    "block_size": "'paired' or 'ic_series'",
    "confidence": "'paired' or 'ic_series'",
    "seed": "'paired' or 'ic_series'",
    "p_values": "'adjust'",
    "method": "'adjust'",
    "alpha": "'adjust'",
}

_MODE_FIELDS: Dict[str, Tuple[str, ...]] = {
    "paired": (
        "predictions_ref_a",
        "predictions_ref_b",
        "task",
        "metric",
        "horizon",
        "n_bootstrap",
        "block_size",
        "confidence",
        "seed",
    ),
    "ic_series": (
        "ic_a",
        "ic_b",
        "hac_lag",
        "n_bootstrap",
        "block_size",
        "confidence",
        "seed",
    ),
    "adjust": ("p_values", "method", "alpha"),
}

_REQUIRED_FIELDS: Dict[str, Tuple[str, ...]] = {
    "paired": ("predictions_ref_a", "predictions_ref_b", "task"),
    "ic_series": ("ic_a", "ic_b"),
    "adjust": ("p_values",),
}

#: The honesty requirement the paired model comparison already carries,
#: restated for every family this tool corrects. An adjustment is about
#: the tests that were RUN; it knows nothing about the candidates that
#: were tried and discarded before them.
_SELECTION_TAIL = (
    "It does not control for the candidates having been SELECTED on the "
    "same sample that produced these p-values: pick the best of twenty "
    "signals on one window and correct only the survivor, and the "
    "correction is measuring the wrong family. That is what "
    "run_reality_check exists for, and no adjustment here substitutes for "
    "it."
)


def _refuse_foreign_fields(input_data: CompareSignalsInput) -> None:
    """A field belonging to another mode is named, not ignored."""
    supplied = set(input_data.model_fields_set) - {"mode"}
    stray = sorted(supplied - set(_MODE_FIELDS[input_data.mode]))
    if not stray:
        return
    detail = "; ".join(
        f"{field!r} is read by mode {_FIELD_OWNER.get(field, 'another mode')}"
        for field in stray
    )
    raise ValidationError(
        f"compare_signals(mode={input_data.mode!r}) does not read "
        f"{stray}: {detail}. Dropping them silently would run a different "
        "call from the one you wrote -- a paired comparison that quietly "
        "ignored `p_values`, say, would return a p-value nobody corrected. "
        "Remove them, or set mode to the one that reads them."
    )


def _require_fields(input_data: CompareSignalsInput) -> None:
    """The fields the chosen mode cannot run without."""
    missing = [
        field
        for field in _REQUIRED_FIELDS[input_data.mode]
        if getattr(input_data, field) is None
    ]
    if not missing:
        return
    raise ValidationError(
        f"compare_signals(mode={input_data.mode!r}) needs {missing}, which "
        "were not given. There is no default for them: a comparison needs "
        "both sides named, and a correction needs the family it is "
        "correcting over."
    )


def _ic_series(mapping: Dict[str, float], label: str) -> pd.Series:
    """A date -> IC mapping as a chronologically ordered series."""
    try:
        index = pd.to_datetime(list(mapping.keys()))
    except (ValueError, TypeError) as exc:
        raise ValidationError(
            f"compare_signals: {label} has a key that is not a date "
            f"({exc}). The keys are the dates each information coefficient "
            "was measured on, e.g. '2023-01-03', because the two series are "
            "aligned on them and resampled in blocks of consecutive dates."
        ) from exc
    if index.has_duplicates:
        duplicated = sorted({str(d.date()) for d in index[index.duplicated()]})
        raise ValidationError(
            f"compare_signals: {label} carries more than one value for "
            f"{duplicated[:5]}. One date holds one cross-sectional IC; two "
            "would each be aligned against the other series' single value "
            "and counted twice in the bootstrap."
        )
    return pd.Series(
        [float(value) for value in mapping.values()], index=index
    ).sort_index()


def _hac_block(
    difference: np.ndarray, requested_lag: Optional[int]
) -> Dict[str, Optional[float]]:
    """
    The variance of the mean difference, without and with the correction.

    The lag defaults to the usual data-driven rule floor(4 (n/100)^(2/9)).
    `hac_ratio` above 1 means the difference series is positively
    autocorrelated, so the ordinary standard error is too small and any
    t-statistic built on it is overstated by its square root.
    """
    n = int(difference.size)
    if requested_lag is None:
        lag = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    else:
        lag = int(requested_lag)
    lag = max(0, min(lag, max(n - 1, 0)))
    variance_lag0 = newey_west_variance(difference, 0)
    variance = newey_west_variance(difference, lag)
    ratio: Optional[float] = None
    if math.isfinite(variance_lag0) and variance_lag0 > 0.0:
        ratio = float(variance / variance_lag0)
    return {
        "hac_variance_lag0": float(variance_lag0),
        "hac_variance": float(variance),
        "hac_lag": float(lag),
        "hac_ratio": ratio,
    }


def compare_signals(input_data: CompareSignalsInput) -> CompareSignalsResult:
    """
    Is B better than A, and how many other candidates were asked?

    THE COMPARISON IS PAIRED. Two signals are joined on the rows both
    predicted and the per-date DIFFERENCE is the object: its dispersion is
    far smaller than either signal's own, because the two share every good
    and bad day, and comparing two unpaired intervals would find nothing
    significant about signals that differ on every single date.

    THE INTERVAL IS BLOCK-BOOTSTRAPPED. Daily ICs under an overlapping
    label are serially correlated and an IID resample destroys exactly
    that, narrowing the interval by a factor this repository has measured
    at 2.24x on an AR(1) at phi=0.8.

    THE CORRECTION IS THE THIRD MODE, and it takes p-values from anywhere,
    because a family of tests is a family however the tests were
    computed. What no correction does is account for the candidates having
    been chosen on the same sample -- the result says so every time.
    """
    _refuse_foreign_fields(input_data)
    _require_fields(input_data)

    if input_data.mode == "paired":
        return _compare_paired(input_data)
    if input_data.mode == "ic_series":
        return _compare_ic_series_mode(input_data)
    return _adjust_p_values(input_data)


def _compare_paired(input_data: CompareSignalsInput) -> CompareSignalsResult:
    """Two published prediction frames, on the rows both predicted."""
    from standard_quant_tools.agent.runtimes import handoff

    frame_a = handoff.resolve(input_data.predictions_ref_a, expect="predictions")
    frame_b = handoff.resolve(input_data.predictions_ref_b, expect="predictions")
    comparison = paired_comparison(
        frame_a,
        frame_b,
        task=str(input_data.task),
        metric=input_data.metric,
        horizon=input_data.horizon,
        n_bootstrap=input_data.n_bootstrap,
        block_size=input_data.block_size,
        confidence=input_data.confidence,
        seed=input_data.seed,
    )
    warnings = [str(note) for note in comparison.pop("warnings", [])]
    warnings.append(
        "This is ONE test, so nothing has been adjusted. Comparing several "
        "candidates against the same baseline means correcting the whole "
        "family -- compare_signals(mode='adjust') does that, and what it "
        "then controls is the family-wise error of THOSE tests. " + _SELECTION_TAIL
    )
    logger.debug(
        "[compare_signals] mode=paired a=%s b=%s verdict=%s",
        input_data.predictions_ref_a,
        input_data.predictions_ref_b,
        comparison.get("verdict"),
    )
    return CompareSignalsResult(
        mode="paired",
        comparison=_json_safe(comparison),
        hac=None,
        adjusted=[],
        n_tests=1,
        method=None,
        n_rejected=0,
        warnings=warnings,
    )


def _compare_ic_series_mode(input_data: CompareSignalsInput) -> CompareSignalsResult:
    """Two per-date IC series inline, plus the HAC variance of the difference."""
    series_a = _ic_series(input_data.ic_a or {}, "ic_a")
    series_b = _ic_series(input_data.ic_b or {}, "ic_b")
    # Raises when fewer than ten dates carry an IC for both, which is the
    # library's own refusal and not a second opinion about it.
    comparison = compare_ic_series(
        series_a,
        series_b,
        n_bootstrap=input_data.n_bootstrap,
        block_size=input_data.block_size,
        confidence=input_data.confidence,
        seed=input_data.seed,
    )
    joined = pd.concat(
        [series_a.rename("a"), series_b.rename("b")], axis=1, join="inner"
    ).dropna()
    difference = (joined["b"] - joined["a"]).to_numpy(dtype=np.float64)
    hac = _hac_block(difference, input_data.hac_lag)

    warnings: List[str] = []
    n_dates = int(comparison["n_dates"])
    if n_dates < 60:
        warnings.append(
            f"NOTE: the comparison rests on {n_dates} dates. An interval "
            "this short is wide by construction, and 'indistinguishable' on "
            "it means 'not enough dates to tell', not 'the same'."
        )
    ratio = hac["hac_ratio"]
    if ratio is not None and ratio > 1.25:
        warnings.append(
            f"The difference series is autocorrelated: its long-run "
            f"variance at lag {int(hac['hac_lag'] or 0)} is {ratio:.2f}x the "
            "ordinary one, so a t-statistic computed without the correction "
            f"is overstated by about {math.sqrt(ratio):.2f}x. The bootstrap "
            "interval above resamples in blocks and already accounts for "
            "this; an ordinary OLS standard error on the same series does "
            "not."
        )
    logger.debug(
        "[compare_signals] mode=ic_series dates=%d hac_ratio=%s",
        n_dates,
        ratio,
    )
    return CompareSignalsResult(
        mode="ic_series",
        comparison=_json_safe(comparison),
        hac=hac,
        adjusted=[],
        n_tests=1,
        method=None,
        n_rejected=0,
        warnings=warnings,
    )


def _adjust_p_values(input_data: CompareSignalsInput) -> CompareSignalsResult:
    """A family of p-values from anywhere, corrected for having been asked."""
    p_values = input_data.p_values or {}
    labels = list(p_values.keys())
    raw = [float(p_values[label]) for label in labels]
    alpha = float(input_data.alpha)
    method = input_data.method

    adjust = {
        "holm": holm_adjust,
        "bonferroni": bonferroni_adjust,
        "bh": bh_adjust,
    }[method]
    adjusted_values = adjust(raw)
    # One rejection rule for all three, because all three return a
    # monotone adjusted p-value: comparing it to alpha IS each
    # procedure's own decision, step-down or step-up.
    rows = [
        AdjustedPValue(
            label=label,
            p_value=value,
            p_adjusted=adjusted,
            reject_at_alpha=bool(adjusted <= alpha),
        )
        for label, value, adjusted in zip(labels, raw, adjusted_values)
    ]
    rows.sort(key=lambda row: (row.p_adjusted, row.p_value, row.label))
    n_rejected = sum(1 for row in rows if row.reject_at_alpha)

    quantity = "false discovery rate" if method == "bh" else "family-wise error rate"
    warnings = [
        f"{method!r} controls the {quantity} across THESE {len(rows)} "
        f"test(s) at alpha={alpha:g}. " + _SELECTION_TAIL
    ]
    if method == "bh":
        warnings.append(
            "Benjamini-Hochberg controls the FALSE DISCOVERY RATE -- the "
            "expected share of the rejections that are false -- and not the "
            "family-wise error rate, so a rejection here is a DIFFERENT "
            "CLAIM from one under 'holm'. It says this test is unlikely to "
            "be more than its share of the false ones, not that a false "
            "rejection anywhere in the family is unlikely. It rejects more, "
            "and each rejection is worth less; use 'holm' when one false "
            "positive would be acted on."
        )
    if len(rows) == 1:
        warnings.append(
            "A family of one is not a multiple-testing problem: every "
            "method leaves a single p-value unchanged. The correction only "
            "means something when every candidate that was tried is in the "
            "call, and leaving the discarded ones out is the difference "
            "between an honest correction and a flattering one."
        )
    logger.debug(
        "[compare_signals] mode=adjust method=%s tests=%d rejected=%d",
        method,
        len(rows),
        n_rejected,
    )
    return CompareSignalsResult(
        mode="adjust",
        comparison=None,
        hac=None,
        adjusted=rows,
        n_tests=len(rows),
        method=method,
        n_rejected=n_rejected,
        warnings=warnings,
    )


__all__ = [
    "COMPARE_SIGNALS_DESCRIPTION",
    "SCORE_PREDICTION_INTERVALS_DESCRIPTION",
    "AdjustedPValue",
    "CompareSignalsInput",
    "CompareSignalsResult",
    "IntervalGroup",
    "ScorePredictionIntervalsInput",
    "ScorePredictionIntervalsResult",
    "compare_signals",
    "score_prediction_intervals",
]
