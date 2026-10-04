"""
Choosing a feature set, and comparing two of them.

Selection here is deliberately BORING: drop what is redundant, drop what
does not pass a permutation test against the target, keep the rest, and
say why for every drop. There is no search, no wrapper method, no greedy
forward pass. That is a deliberate limit rather than an unfinished one.

A greedy selector scored on the same panel it selects from is a machine for
manufacturing overfit, and it is a particularly bad one to hand an agent:
the output looks like a decision backed by evidence, the evidence is the
training data, and nothing in the result says so. The two criteria used
here -- "this is the same feature twice" and "this has no measurable
relationship with the target" -- are the two an agent can defend to a human
afterwards.

`compare_feature_sets` exists for the same reason. An agent that wants to
know whether adding six features helped should get an answer with the cost
attached (more collinearity, more turnover) rather than a single score that
went up.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from ..dataset.alignment import LABEL_END_COL
from ..limits import MAX_PERMUTATION_DRAWS
from ..preprocessing.base import refuse_mixed_time_zones
from ..validation.comparison import bh_adjust
from .feature_report import (
    _abs_rank_ic,
    _boundary_date,
    _named_once,
    _panel_dates,
    cluster_records,
    collinearity_warnings,
    condition_warning,
    feature_predictive_stats,
    redundancy_report,
)
from .feature_stability import estimate_draw_seconds, permutation_test_ic

logger = logging.getLogger(__name__)

#: The nulls `select_features` can gate on, and "none" for no gate.
SELECTION_NULLS = ("entity_shuffle", "circular_shift", "none")

#: The multiple-testing corrections the gate can apply. No family-wise
#: correction is offered: see `_check_gate_arguments`.
SELECTION_CORRECTIONS = ("none", "bh")

#: How each null reads in a sentence.
_NULL_PROSE = {
    "entity_shuffle": "the entity-shuffle null",
    "circular_shift": "the circular-shift null",
}

#: What "none of them beat it" means, per null.
_NULL_MEANING = {
    "entity_shuffle": "the same series assigned to random entities",
    "circular_shift": "its own series rolled in time by a random offset per entity",
}

#: The draw budget a selection may spend before it is refused, the same
#: default the significance screen uses.
DEFAULT_MAX_DRAWS = 20_000


def _signed_rank_ic(
    stats: Dict[str, Dict[str, float]], feature: str
) -> Optional[float]:
    value = (stats.get(feature) or {}).get("rank_ic_mean")
    return float(value) if value is not None and np.isfinite(value) else None


def _date_label(value: Any) -> str:
    return str(pd.Timestamp(value).date())


def _selection_cutoff(
    panel: pd.DataFrame,
    selection_end: Any,
    holdout_fraction: float,
    *,
    caller: str = "select_features",
) -> "tuple[pd.DatetimeIndex, Optional[pd.Timestamp]]":
    """
    The last date the selection may read. `selection_end` names it;
    otherwise the first `1 - holdout_fraction` of the panel's dates select
    and the rest are held out; a zero fraction selects on everything and
    holds out nothing, which the result then says in so many words.

    `caller` prefixes the refusals. Two functions share this cutoff now,
    and a refusal that named the wrong one would send the reader to fix an
    argument they did not pass.

    `selection_end` is parsed by the rule the tool doors use. It used to go
    straight to `pd.Timestamp`, where an unreadable string raised pandas'
    own parse error and an empty one became NaT: NaT compares false against
    every date, so it passed the range check below and left a holdout with
    no dates in it, which failed later as an IndexError.
    """
    dates = pd.DatetimeIndex(sorted(_panel_dates(panel, caller).unique()))
    if len(dates) < 2:
        raise ValidationError(
            f"{caller}: the panel has fewer than two dates, so nothing "
            "can be held out and no cross-sectional IC can be trusted."
        )
    if selection_end is not None:
        cutoff = _boundary_date(selection_end, "selection_end", caller, dates)
        if cutoff < dates[0] or cutoff >= dates[-1]:
            raise ValidationError(
                f"{caller}: selection_end={_date_label(cutoff)!r} must "
                f"fall inside the panel's dates ({_date_label(dates[0])}.."
                f"{_date_label(dates[-1])}) and leave at least one date after "
                "it to hold out."
            )
        return dates, cutoff
    if holdout_fraction <= 0:
        return dates, None
    if holdout_fraction >= 1:
        raise ValidationError(
            f"{caller}: holdout_fraction={holdout_fraction} would hold "
            "out every date; it must be below 1."
        )
    n_select = int(np.floor(len(dates) * (1.0 - holdout_fraction)))
    n_select = min(max(n_select, 1), len(dates) - 1)
    return dates, dates[n_select - 1]


def _window(dates: pd.DatetimeIndex) -> Dict[str, Any]:
    """First date, last date and count of a non-empty window.

    Every caller passes a window `_selection_cutoff` guarantees non-empty;
    the check keeps an empty one from surfacing as an IndexError that
    names no argument, should that guarantee ever slip.
    """
    if len(dates) == 0:
        raise ValidationError(
            "feature selection: a window with no dates in it. selection_end "
            "or holdout_fraction must leave at least one date on each side."
        )
    return {
        "start": _date_label(dates[0]),
        "end": _date_label(dates[-1]),
        "n_dates": int(len(dates)),
    }


def _split_at_holdout(
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
    cutoff: Optional[pd.Timestamp],
    embargo_dates: int,
    caller: str,
) -> Dict[str, Any]:
    """
    The rows a selection or a summary reads, the rows it holds out, and the
    embargo between them.

    THE EMBARGO IS BY ROW. With `embargo_dates` = h above zero, a row of
    the selection window is dropped when it is dated on one of the
    window's last h dates -- its label, h bars long, ends on or after the
    holdout's first date -- or when its recorded `label_end_date` falls on
    or after that date. The second catches what the first cannot: on a
    ragged panel, an entity with dates missing reaches its h-th bar later
    than the panel's own calendar does, so its label can end in the holdout
    from earlier in the window. A row with no recorded end is taken to end
    h of the panel's dates after its own. Zero is no embargo, the window as
    it was. The holdout is the same either way.

    Returns `selection` and `holdout` (the rows), `selection_window`,
    `holdout_window` (None without a holdout), `embargoed` (the window's
    last h dates), `embargo_rows` (every row dropped) and `reach_rows` (the
    earlier rows dropped for their recorded end).
    """
    empty = dates[0:0]
    if cutoff is None:
        return {
            "selection": panel,
            "holdout": panel.iloc[0:0],
            "selection_window": _window(dates),
            "holdout_window": None,
            "embargoed": empty,
            "embargo_rows": 0,
            "reach_rows": 0,
        }
    date_values = pd.to_datetime(panel["date"])
    held = dates[dates > cutoff]
    holdout = panel[date_values > cutoff]
    if not embargo_dates:
        return {
            "selection": panel[date_values <= cutoff],
            "holdout": holdout,
            # `end` is the cutoff as asked for, not the last date at or
            # before it: a caller who named `selection_end` should read
            # their own date back rather than the nearest trading day to it.
            "selection_window": {
                "start": _date_label(dates[0]),
                "end": _date_label(cutoff),
                "n_dates": int((dates <= cutoff).sum()),
            },
            "holdout_window": _window(held),
            "embargoed": empty,
            "embargo_rows": 0,
            "reach_rows": 0,
        }

    h = int(embargo_dates)
    before = dates[dates <= cutoff]
    if h >= len(before):
        raise ValidationError(
            f"{caller}: embargo_dates={embargo_dates} would "
            f"leave no date to select on: the selection window holds "
            f"{len(before)} date(s) through {_date_label(cutoff)}. "
            "Lower embargo_dates, or hold out fewer dates."
        )
    embargoed = before[-h:]
    last_read = before[-h - 1]
    in_window = date_values <= cutoff
    dropped = in_window & (date_values > last_read)
    reach_rows = 0
    if LABEL_END_COL in panel.columns:
        refuse_mixed_time_zones(panel["date"], panel[LABEL_END_COL], caller)
        ends = pd.to_datetime(panel[LABEL_END_COL])
        reaching = in_window & ~dropped & (ends >= held[0])
        reach_rows = int(reaching.sum())
        dropped = dropped | reaching
    read = in_window & ~dropped
    read_dates = dates[dates <= last_read]
    if reach_rows:
        present = pd.DatetimeIndex(date_values[read].unique())
        read_dates = read_dates[read_dates.isin(present)]
        if len(read_dates) == 0:
            raise ValidationError(
                f"{caller}: every row of the selection window through "
                f"{_date_label(last_read)} has a recorded label_end_date "
                f"on or after the holdout's first date, {_date_label(held[0])}, "
                "so the embargo leaves nothing to select on. Hold out fewer "
                "dates, or check the label_end_date column."
            )
    return {
        "selection": panel[read],
        "holdout": holdout,
        "selection_window": {
            "start": _date_label(read_dates[0]),
            "end": _date_label(read_dates[-1]),
            "n_dates": int(len(read_dates)),
        },
        "holdout_window": _window(held),
        "embargoed": embargoed,
        "embargo_rows": int(dropped.sum()),
        "reach_rows": reach_rows,
    }


def _embargo_sentence(split: Dict[str, Any], what: str) -> str:
    """What the embargo dropped, as the clause that follows "Selected on
    dates through <end>; " or "Summarised on dates through <end>; "."""
    window = _window(split["embargoed"])
    count = window["n_dates"]
    if split["reach_rows"]:
        return (
            f"the {count} date(s) after it, through {window['end']}, were "
            f"embargoed, and so were {split['reach_rows']} earlier row(s) "
            "whose recorded label_end_date falls in the holdout, so no "
            f"label {what} read ends inside the holdout."
        )
    return (
        f"the {count} date(s) after it, through {window['end']}, were "
        f"embargoed, so no label of up to {count} bar(s) that {what} read "
        "ends inside the holdout."
    )


def _check_gate_arguments(
    significance: str,
    alpha: float,
    n_permutations: int,
    random_seed: int,
    correction: str = "none",
) -> None:
    """
    The gate's arguments, refused by name for a direct caller the way the
    tool's schema refuses them.

    WHY BENJAMINI-HOCHBERG AND NOT HOLM. A permutation p-value from N draws
    is at least 1/(N + 1): 1/201 at the default 200. Holm's first step
    compares the smallest p-value with alpha/m, so at alpha 0.05 and 200
    draws it can pass nothing at all once more than ten features are
    tested (0.05/11 < 1/201), however strong they are. Benjamini-Hochberg's
    k-th step compares the k-th smallest with k x alpha/m, so several
    strong features pass together, and what it controls -- the expected
    share of the kept features that noise kept -- is the question a
    selection asks.
    """
    if significance not in SELECTION_NULLS:
        raise ValidationError(
            f"select_features: significance={significance!r}; expected "
            "'entity_shuffle' (default), 'circular_shift' or 'none'."
        )
    if correction not in SELECTION_CORRECTIONS:
        raise ValidationError(
            f"select_features: correction={correction!r}; expected 'none' "
            "(default) or 'bh'."
        )
    if significance == "none":
        if correction != "none":
            raise ValidationError(
                f"select_features: correction={correction!r} adjusts the "
                "significance test's p-values, and significance='none' runs "
                "no test. Pass a significance null, or leave correction at "
                "'none'."
            )
        return
    if not (0.0 < float(alpha) < 1.0):
        raise ValidationError(
            f"select_features: alpha={alpha!r} must lie strictly between 0 and 1."
        )
    if not (20 <= int(n_permutations) <= 5000):
        raise ValidationError(
            f"select_features: n_permutations={n_permutations!r} must be "
            "between 20 and 5000. 200 resolves a p-value to about 0.005."
        )
    if int(random_seed) < 0:
        raise ValidationError(
            f"select_features: random_seed={random_seed!r} must be non-negative."
        )


def _significance_gate(
    selection_panel: pd.DataFrame,
    candidates: Sequence[str],
    predictive: Dict[str, Dict[str, float]],
    *,
    null: str,
    alpha: float,
    n_permutations: int,
    random_seed: int,
    correction: str = "none",
) -> tuple[
    List[str],
    Dict[str, Optional[float]],
    Dict[str, Optional[float]],
    List[Dict[str, Any]],
]:
    """
    Each candidate's two-sided permutation p-value on the selection window;
    what passes, every p-value, the Benjamini-Hochberg adjusted p-values
    (empty without the correction), and a drop record for what does not.

    Without a correction a feature passes at p < alpha. With
    `correction='bh'` it passes at a Benjamini-Hochberg adjusted p-value
    below alpha, over every candidate tested -- one that could not be
    tested enters the family at p = 1, since it was asked and cannot pass.
    The comparison is strict in both, so the corrected selection is always
    a subset of the uncorrected one.

    Reads `selection_panel` only. The holdout is not in it, and the caller
    reads the holdout after this returns, for the features this kept.
    """
    p_values: Dict[str, Optional[float]] = {}
    dropped: List[Dict[str, Any]] = []
    prose = _NULL_PROSE[null]
    tested: Dict[str, float] = {}
    for feature in candidates:
        try:
            result = permutation_test_ic(
                selection_panel,
                feature,
                n_permutations=n_permutations,
                method="spearman",
                random_seed=random_seed,
                null=null,
            )
        except ValidationError as exc:
            # One untestable column does not sink the selection: it cannot
            # pass a test it cannot take, and the reason is the record.
            p_values[feature] = None
            dropped.append(
                {
                    "feature": feature,
                    "reason": "insignificant",
                    "duplicate_of": None,
                    "p_value": None,
                    "detail": f"not testable on the selection window: {exc}",
                }
            )
            continue
        p_values[feature] = tested[feature] = float(result["p_value"])

    adjusted: Dict[str, Optional[float]] = {}
    if correction == "bh":
        family = [p_values[f] if p_values[f] is not None else 1.0 for f in candidates]
        for feature, value in zip(candidates, bh_adjust(family)):
            adjusted[feature] = value if p_values[feature] is not None else None

    passed: List[str] = []
    for feature, p_value in tested.items():
        decided_on = adjusted[feature] if correction == "bh" else p_value
        if decided_on is not None and decided_on < alpha:
            passed.append(feature)
            continue
        ic = (predictive.get(feature) or {}).get("rank_ic_mean")
        shown = f"{ic:+.4f}" if ic is not None and np.isfinite(ic) else "None"
        corrected = (
            f" (Benjamini-Hochberg adjusted {adjusted[feature]:.3f} over "
            f"{len(candidates)} tested)"
            if correction == "bh"
            else ""
        )
        dropped.append(
            {
                "feature": feature,
                "reason": "insignificant",
                "duplicate_of": None,
                "p_value": p_value,
                "detail": (
                    f"rank IC {shown} on the selection window, p={p_value:.3f}"
                    f"{corrected} against {prose} at alpha {alpha:g}"
                ),
            }
        )
    return passed, p_values, adjusted, dropped


def select_features(
    panel: pd.DataFrame,
    feature_ids: Sequence[str],
    *,
    cluster_threshold: float = 0.9,
    min_abs_rank_ic: float = 0.0,
    max_features: int = 0,
    selection_end: Any = None,
    holdout_fraction: float = 0.3,
    significance: str = "entity_shuffle",
    alpha: float = 0.05,
    n_permutations: int = 200,
    random_seed: int = 0,
    max_draws: int = DEFAULT_MAX_DRAWS,
    embargo_dates: int = 0,
    correction: str = "none",
) -> Dict[str, Any]:
    """
    Keep one feature per redundancy cluster, drop what does not pass a
    permutation test on the selection window, and record a reason for every
    exclusion.

    THE SELECTION DOES NOT SEE THE WHOLE PANEL. Redundancy and the IC floor
    were measured over every date, including the ones a later walk-forward
    run holds out, so the run started biased: sixty columns of pure noise
    added to a live panel, the top five by full-panel IC, and the same
    walk-forward on those five scored +0.045 against +0.002 for five chosen
    blind -- about 70% of the real model's headline, manufactured from
    noise by following the documented workflow (findings D4). Selection
    now reads dates up to `selection_end`, or the first
    `1 - holdout_fraction` of the panel's dates, and reports each selected
    feature's IC on the dates after that beside its selection IC. The
    holdout IC is the number to believe; the selection IC is in-sample by
    construction.

    Order matters and is not arbitrary: redundancy, then the IC floor, then
    the significance test, then the cap. Redundancy is resolved FIRST. The
    other way round, a cluster whose members are all individually below the
    floor would be dropped entirely -- but a cluster is one signal, and the
    right question is whether that one signal clears the floor, asked once
    via its representative. The test is asked of the representatives that
    cleared the floor, for the same reason and because it is the expensive
    step.

    THE SIGNIFICANCE TEST. The floor defaults to 0.0, so before the test a
    call with no arguments kept every feature that was not a duplicate --
    on the live panel of 2026-10-03, all eight, while one screen called one
    of them significant. `significance` (default 'entity_shuffle') tests
    each representative's mean rank IC on the selection window against a
    permutation null and drops what does not reach p < `alpha`, as reason
    'insignificant' with its p-value. The entity-shuffle null hands each
    entity's whole feature series to another entity (see
    `permutation_test_ic`): it keeps each series' serial correlation and
    breaks the feature-entity link, including a static tilt, which a
    circular shift keeps. On that panel it kept beta_60 (p 0.005, holdout
    IC +0.015) and rvol_20 (p 0.005, +0.027) and dropped the six others
    (p 0.28-0.89). 'circular_shift' is the screen's null; 'none' applies no
    test and returns what this function returned before it had one, to the
    bit, with a warning saying no test was applied. By default the p-values
    are not corrected for the number of features tested, and the warning
    says how many would clear from noise alone; `correction='bh'` passes a
    feature on its Benjamini-Hochberg adjusted p-value instead, which
    controls the expected share of the kept features that noise kept (see
    `_check_gate_arguments` for why no family-wise correction is offered).

    The test reads the selection window ONLY, and is fixed before the
    holdout is read: `holdout_ic` is computed afterwards, for the features
    the test kept.

    THE EMBARGO. A label dated on one of the selection window's last h
    dates looks h bars forward, into the holdout, so the selection IC and
    the test read outcomes the holdout is later scored on. `embargo_dates`
    drops that many dates from the end of the selection window; the
    holdout is unchanged. Every number the selection computes then reads
    only dates whose labels end before the holdout starts, for a label of
    up to that many bars. The function does not know the label, so the
    default is 0, the window as it was, to the bit; `select_features` the
    tool passes the target horizon. The embargo applies only when there is
    a holdout.

    `max_features` truncates by absolute rank IC after every filter. It is
    a cap for a caller who has a hard budget, not a ranking to trust: the
    difference between the 20th and 21st feature by IC on one panel is
    usually noise.

    The redundancy diagnostics come back with the selection rather than
    being recomputed: `clusters` in the shape `get_feature_redundancy`
    publishes, `vif`, `condition_number`, `correlation`,
    `collinear_features` and a `duplicate_of` on every redundant drop. All
    of them were already computed to make the decision, and returning them
    is what makes the decision auditable without paying for the same
    correlation matrix twice.
    """
    feature_ids = list(feature_ids)
    if not feature_ids:
        raise ValidationError("select_features: no features to choose from")
    missing = [f for f in feature_ids if f not in panel.columns]
    if missing:
        raise ValidationError(f"panel has no features: {sorted(missing)}")
    _named_once(panel, feature_ids, "select_features")
    _check_gate_arguments(significance, alpha, n_permutations, random_seed, correction)
    if int(embargo_dates) < 0:
        raise ValidationError(
            f"select_features: embargo_dates={embargo_dates!r} must be zero or more."
        )

    dates, cutoff = _selection_cutoff(panel, selection_end, holdout_fraction)
    split = _split_at_holdout(
        panel, dates, cutoff, int(embargo_dates), "select_features"
    )
    selection_panel, holdout_panel = split["selection"], split["holdout"]
    embargoed = split["embargoed"]

    predictive = feature_predictive_stats(selection_panel, feature_ids)
    redundancy = redundancy_report(
        selection_panel, feature_ids, cluster_threshold=cluster_threshold
    )

    clusters = cluster_records(
        redundancy["clusters"], redundancy["correlation"], predictive
    )

    dropped: List[Dict[str, Any]] = []
    survivors: List[str] = []
    for cluster in clusters:
        keeper = cluster["representative"]
        survivors.append(keeper)
        for member in cluster["members"]:
            if member != keeper:
                dropped.append(
                    {
                        "feature": member,
                        "reason": "redundant",
                        # The prose stays, because it carries the threshold
                        # the drop was made at; `duplicate_of` sits beside
                        # it so that "a duplicate of what" needs no parser.
                        "duplicate_of": keeper,
                        "detail": (
                            f"same signal as {keeper!r} at "
                            f"|rho| >= {cluster_threshold:.2f}"
                        ),
                    }
                )

    kept: List[str] = []
    for feature in survivors:
        strength = _abs_rank_ic(predictive, feature)
        if strength < min_abs_rank_ic:
            dropped.append(
                {
                    "feature": feature,
                    "reason": "weak",
                    "duplicate_of": None,
                    "detail": (
                        f"|rank IC| {strength:.4f} below the "
                        f"{min_abs_rank_ic:.4f} floor"
                    ),
                }
            )
        else:
            kept.append(feature)

    # The test, on the representatives that cleared the floor. Its cost is
    # counted before the first draw, as the significance screen counts its
    # own: a long run is chosen rather than discovered.
    p_values: Dict[str, Optional[float]] = {}
    adjusted: Dict[str, Optional[float]] = {}
    gate: Optional[Dict[str, Any]] = None
    if significance != "none":
        n_draws = len(kept) * int(n_permutations)
        if n_draws > max_draws:
            seconds = estimate_draw_seconds(significance, len(selection_panel))
            remedy = (
                f"pass max_draws={n_draws} to accept the cost"
                if n_draws <= MAX_PERMUTATION_DRAWS
                else (
                    f"max_draws stops at {MAX_PERMUTATION_DRAWS:,} draws, so "
                    "this test cannot be bought -- it has to be narrowed"
                )
            )
            raise ValidationError(
                f"select_features: the significance test needs {n_draws:,} "
                f"permutation draws ({len(kept)} features x {n_permutations} "
                f"permutations), over the max_draws={max_draws:,} ceiling. At "
                f"about {seconds * 1e3:.1f} ms a draw under "
                f"{_NULL_PROSE[significance]} on {len(selection_panel):,} rows "
                f"that is roughly {max(1, round(n_draws * seconds / 60))} "
                "minute(s). Narrow `features`, lower `n_permutations`, pass "
                f"significance='none' to skip the test, or {remedy}."
            )
        passed, p_values, adjusted, insignificant = _significance_gate(
            selection_panel,
            kept,
            predictive,
            null=significance,
            alpha=alpha,
            n_permutations=n_permutations,
            random_seed=random_seed,
            correction=correction,
        )
        gate = {
            "null": significance,
            "alpha": float(alpha),
            "n_permutations": int(n_permutations),
            "random_seed": int(random_seed),
            "n_tested": len(kept),
            "n_passed": len(passed),
            "correction": correction,
            # What the uncorrected rule would have kept, so the cost of
            # the correction is a number rather than a second call.
            "n_passed_uncorrected": sum(
                1 for p in p_values.values() if p is not None and p < alpha
            ),
        }
        dropped.extend(insignificant)
        kept = passed

    kept.sort(key=lambda f: (-_abs_rank_ic(predictive, f), f))
    if max_features and len(kept) > max_features:
        for feature in kept[max_features:]:
            dropped.append(
                {
                    "feature": feature,
                    "reason": "capped",
                    "duplicate_of": None,
                    "detail": (
                        f"ranked {kept.index(feature) + 1} by |rank IC|, past "
                        f"the max_features={max_features} cap"
                    ),
                }
            )
        kept = kept[:max_features]

    warnings: List[str] = []
    holdout_ic: Dict[str, Optional[float]] = {}
    selection_window = split["selection_window"]
    holdout_window = split["holdout_window"]
    if cutoff is None:
        warnings.append(
            "WARNING: the selection read the WHOLE panel, holdout included. "
            "Every selection IC below is in-sample by construction, and a "
            "walk-forward run on these features starts biased: measured, the "
            "top five of sixty pure-noise columns chosen this way scored "
            "+0.045 out of sample against +0.002 for five chosen blind. Pass "
            "holdout_fraction or selection_end to select on an earlier window "
            "and read the holdout IC instead."
        )
    else:
        # Under an embargo `end` is the last date the selection read;
        # without one, the cutoff as asked for (see `_split_at_holdout`).
        if kept:
            holdout_stats = feature_predictive_stats(holdout_panel, kept)
            holdout_ic = {f: _signed_rank_ic(holdout_stats, f) for f in kept}
        if len(embargoed):
            warnings.append(
                f"Selected on dates through {selection_window['end']}; "
                + _embargo_sentence(split, "the selection")
                + " `holdout_ic` is each selected feature's rank IC on the "
                f"{holdout_window['n_dates']} date(s) after the embargo, which "
                "the selection never read. That is the number to believe: "
                "`selection_ic` chose the features and is optimistic by "
                "construction."
            )
        else:
            warnings.append(
                f"Selected on dates through {selection_window['end']}; "
                f"`holdout_ic` is each selected feature's rank IC on the "
                f"{holdout_window['n_dates']} date(s) after it, which the "
                "selection never read. That is the number to believe: "
                "`selection_ic` chose the features and is optimistic by "
                "construction."
            )
        if holdout_window["n_dates"] < 20:
            warnings.append(
                f"NOTE: the holdout is {holdout_window['n_dates']} date(s), too "
                "few for a rank IC to mean much; widen holdout_fraction or the "
                "panel."
            )

    warnings.extend(
        _gate_warnings(
            gate,
            selection_panel,
            n_selection_dates=int(selection_window["n_dates"]),
            min_abs_rank_ic=min_abs_rank_ic,
        )
    )
    warnings.extend(
        collinearity_warnings(
            redundancy["collinear"],
            cluster_threshold=cluster_threshold,
            pair_list="The redundancy drops are made for",
        )
    )
    condition = condition_warning(redundancy["condition_number"])
    if condition:
        warnings.append(condition)

    return {
        "selected": kept,
        "dropped": sorted(dropped, key=lambda d: d["feature"]),
        "n_considered": len(feature_ids),
        "n_selected": len(kept),
        "n_clusters": len(clusters),
        "clusters": clusters,
        "cluster_threshold": cluster_threshold,
        "min_abs_rank_ic": min_abs_rank_ic,
        "selection_window": selection_window,
        "holdout_window": holdout_window,
        "embargo_dates": int(len(embargoed)),
        "embargo_window": _window(embargoed) if len(embargoed) else None,
        "embargo_rows": int(split["embargo_rows"]),
        "selection_ic": {f: _signed_rank_ic(predictive, f) for f in feature_ids},
        "selection_p_value": p_values,
        "selection_p_value_adjusted": adjusted,
        "significance": gate,
        "holdout_ic": holdout_ic,
        # Paid for by the `redundancy_report` call above and previously
        # thrown away, which forced an agent that wanted "dropped as a
        # duplicate of what", or the collinearity of what survived, to run
        # get_feature_redundancy and buy the same correlation matrix twice.
        "vif": redundancy["vif"],
        "condition_number": redundancy["condition_number"],
        "correlation": redundancy["correlation"],
        "collinear_features": redundancy["collinear"],
        "warnings": warnings,
    }


def _gate_warnings(
    gate: Optional[Dict[str, Any]],
    selection_panel: pd.DataFrame,
    *,
    n_selection_dates: int,
    min_abs_rank_ic: float,
) -> List[str]:
    """What the significance test did, in sentences: how many passed and
    how many noise alone would pass, the empty result said in so many
    words, a panel too small for the null to reach `alpha`, and the
    absence of any test when there was none."""
    if gate is None:
        return [
            "No significance test was applied: every feature that was not "
            f"redundant and cleared min_abs_rank_ic={min_abs_rank_ic:g} was kept."
        ]
    alpha, null, n_tested = gate["alpha"], gate["null"], gate["n_tested"]
    prose = _NULL_PROSE[null]
    warnings: List[str] = []
    if null == "entity_shuffle":
        n_entities = int(selection_panel["entity"].nunique())
        assignments = math.factorial(n_entities) if n_entities < 20 else math.inf
        if assignments * alpha < 2.0:
            warnings.append(
                f"With {n_entities} entities there are only {assignments} ways "
                "to assign the feature series to them, so the entity-shuffle "
                "null cannot produce a p-value much below "
                f"1/{assignments} = {1.0 / assignments:.3f}, and at alpha "
                f"{alpha:g} it cannot reliably pass anything. Pass "
                "significance='circular_shift' or 'none' on a panel this small."
            )
    if n_tested == 0:
        return warnings
    if gate.get("correction") == "bh":
        warnings.append(
            _bh_sentence(gate, prose, n_selection_dates, _NULL_MEANING[null])
        )
        return warnings
    if gate["n_passed"] == 0:
        warnings.append(
            f"No feature cleared p < {alpha:g} against {prose} on the "
            f"{n_selection_dates} selection dates, so `selected` is empty. None "
            "of these features ranks these entities better than "
            f"{_NULL_MEANING[null]}."
        )
    else:
        warnings.append(
            f"{gate['n_passed']} of {n_tested} features cleared p < {alpha:g} "
            f"against {prose} on the {n_selection_dates} selection dates; at "
            f"that alpha about {alpha * n_tested:.1f} of {n_tested} clear from "
            "noise alone, and these p-values are not corrected for the "
            f"{n_tested} tests."
        )
    return warnings


def _bh_sentence(
    gate: Dict[str, Any], prose: str, n_selection_dates: int, meaning: str
) -> str:
    """What the Benjamini-Hochberg gate kept, beside what the uncorrected
    rule would have kept."""
    alpha, n_tested = gate["alpha"], gate["n_tested"]
    n_passed, uncorrected = gate["n_passed"], gate["n_passed_uncorrected"]
    beside = (
        f" Uncorrected, {uncorrected} cleared p < {alpha:g}."
        if uncorrected != n_passed
        else ""
    )
    if n_passed == 0:
        tail = (
            f" None of these features ranks these entities better than {meaning}."
            if uncorrected == 0
            else ""
        )
        return (
            f"No feature cleared a Benjamini-Hochberg adjusted p < {alpha:g} "
            f"over the {n_tested} tested, against {prose} on the "
            f"{n_selection_dates} selection dates, so `selected` is empty."
            f"{beside}{tail}"
        )
    return (
        f"{n_passed} of {n_tested} features cleared a Benjamini-Hochberg "
        f"adjusted p < {alpha:g} against {prose} on the {n_selection_dates} "
        f"selection dates.{beside} The correction holds the expected share "
        f"of the kept features that noise kept at or below {alpha:g} when the "
        "tests are independent or positively dependent; it does not say "
        "which of them those are."
    )


def summarize_feature_set(
    panel: pd.DataFrame,
    feature_ids: Sequence[str],
    *,
    cluster_threshold: float = 0.9,
    selection_end: Any = None,
    holdout_fraction: float = 0.0,
    caller: str = "summarize_feature_set",
    embargo_dates: int = 0,
) -> Dict[str, Any]:
    """
    One feature set, as the handful of numbers worth comparing.

    `n_independent_signals` is the one to read rather than `n_features`. A
    set of twelve features in three clusters carries three ideas, and
    reporting twelve overstates the diversification by four times.

    THE SUMMARY IS IN-SAMPLE UNLESS DATES ARE HELD OUT. `holdout_fraction`
    (or `selection_end`) summarises the set on the earlier dates through
    the same `_selection_cutoff` `select_features` uses, and re-measures
    |rank IC| on the later ones as `holdout_mean_abs_rank_ic` /
    `holdout_max_abs_rank_ic`. The default is 0.0 -- every date, no holdout
    -- so the numbers a caller already has do not move; a zero fraction is
    then a statement the caller's warnings have to make, not a silence.

    `embargo_dates` is the embargo `select_features` applies, by the same
    rule (see `_split_at_holdout`): with a holdout, the summary does not
    read the rows whose labels end inside it. 0, the default, reads the
    window as it was, to the bit.
    """
    feature_ids = list(feature_ids)
    _named_once(panel, feature_ids, caller)
    if int(embargo_dates) < 0:
        raise ValidationError(
            f"{caller}: embargo_dates={embargo_dates!r} must be zero or more."
        )
    dates, cutoff = _selection_cutoff(
        panel, selection_end, holdout_fraction, caller=caller
    )
    split = _split_at_holdout(panel, dates, cutoff, int(embargo_dates), caller)
    selection_panel, holdout_panel = split["selection"], split["holdout"]
    selection_window = split["selection_window"]
    holdout_window = split["holdout_window"]
    embargoed = split["embargoed"]

    predictive = feature_predictive_stats(selection_panel, feature_ids)
    redundancy = redundancy_report(
        selection_panel, feature_ids, cluster_threshold=cluster_threshold
    )
    strengths = np.array(
        [_abs_rank_ic(predictive, f) for f in feature_ids], dtype=float
    )

    holdout_mean: Optional[float] = None
    holdout_max: Optional[float] = None
    if cutoff is not None and feature_ids:
        holdout_stats = feature_predictive_stats(holdout_panel, feature_ids)
        held = np.array(
            [_abs_rank_ic(holdout_stats, f) for f in feature_ids], dtype=float
        )
        holdout_mean = float(np.mean(held))
        holdout_max = float(np.max(held))

    return {
        "features": sorted(feature_ids),
        "n_features": len(feature_ids),
        "n_independent_signals": len(redundancy["clusters"]),
        "mean_abs_rank_ic": float(np.mean(strengths)) if strengths.size else 0.0,
        "max_abs_rank_ic": float(np.max(strengths)) if strengths.size else 0.0,
        "condition_number": float(redundancy["condition_number"]),
        "selection_window": selection_window,
        "holdout_window": holdout_window,
        "embargo_dates": int(len(embargoed)),
        "embargo_window": _window(embargoed) if len(embargoed) else None,
        "embargo_rows": int(split["embargo_rows"]),
        "holdout_mean_abs_rank_ic": holdout_mean,
        "holdout_max_abs_rank_ic": holdout_max,
    }


def compare_feature_sets(
    panel: pd.DataFrame,
    left: Sequence[str],
    right: Sequence[str],
    *,
    cluster_threshold: float = 0.9,
    selection_end: Any = None,
    holdout_fraction: float = 0.0,
    embargo_dates: int = 0,
) -> Dict[str, Any]:
    """
    Two feature sets on the same panel, with the cost of the difference
    attached.

    Deliberately NOT a single score. A larger set almost always has a higher
    max IC and almost always has more collinearity, and an agent handed one
    number cannot see the trade it just made. What comes back is per-set
    diagnostics, the features unique to each side, and a per-feature IC
    table for everything in either.

    Both sets are measured on the same window, so the comparison is like for
    like. Comparing sets scored on different date ranges would be comparing
    the ranges.

    AT `holdout_fraction == 0` -- the default, kept so that existing numbers
    do not move -- BOTH SETS ARE SUMMARISED ON EVERY DATE, and every IC here
    is therefore in-sample. The result says so unconditionally rather than
    only when it looks suspicious, because "in-sample" is a property of how
    the numbers were made and not of how they came out. That is the same
    trouble `select_features` goes to, for the same measured reason: five
    noise columns chosen on a full panel scored +0.045 out of sample against
    +0.002 for five chosen blind (findings D4).

    Pass `holdout_fraction` (or `selection_end`) and each set is summarised
    on the earlier dates and re-measured on the later ones, which is the
    comparison worth acting on. `embargo_dates` then leaves out the rows
    whose labels end inside the holdout, by the rule `select_features`
    applies (see `_split_at_holdout`); 0, the default, reads the window as
    it was.
    """
    left, right = list(left), list(right)
    if not left or not right:
        raise ValidationError("compare_feature_sets: both sets must be non-empty")
    unknown = sorted({f for f in left + right if f not in panel.columns})
    if unknown:
        raise ValidationError(f"panel has no features: {unknown}")
    # Each side on its own: a name shared by both sides is the comparison,
    # a name twice on one side is a mistake.
    _named_once(panel, left, "compare_feature_sets", field="left")
    _named_once(panel, right, "compare_feature_sets", field="right")

    everything = sorted(set(left) | set(right))
    if int(embargo_dates) < 0:
        raise ValidationError(
            f"compare_feature_sets: embargo_dates={embargo_dates!r} must be "
            "zero or more."
        )
    dates, cutoff = _selection_cutoff(
        panel, selection_end, holdout_fraction, caller="compare_feature_sets"
    )
    # The per-feature table reads the same rows the summaries do. A table
    # measured on the whole panel beside a summary that held dates out would
    # be two answers to one question, and the wider one is the optimistic one.
    split = _split_at_holdout(
        panel, dates, cutoff, int(embargo_dates), "compare_feature_sets"
    )
    predictive = feature_predictive_stats(split["selection"], everything)

    left_summary = summarize_feature_set(
        panel,
        left,
        cluster_threshold=cluster_threshold,
        selection_end=selection_end,
        holdout_fraction=holdout_fraction,
        caller="compare_feature_sets",
        embargo_dates=embargo_dates,
    )
    right_summary = summarize_feature_set(
        panel,
        right,
        cluster_threshold=cluster_threshold,
        selection_end=selection_end,
        holdout_fraction=holdout_fraction,
        caller="compare_feature_sets",
        embargo_dates=embargo_dates,
    )

    warnings: List[str] = []
    if cutoff is None:
        warnings.append(
            "WARNING: both sets were summarised on EVERY date, so every IC "
            "here is in-sample by construction. This is a comparison BETWEEN "
            "the two sets, not an estimate of either one's out-of-sample "
            "strength: measured, the top five of sixty pure-noise columns "
            "chosen on a full panel scored +0.045 out of sample against "
            "+0.002 for five chosen blind (findings D4). Pass "
            "holdout_fraction or selection_end and compare "
            "holdout_mean_abs_rank_ic instead."
        )
    else:
        n_held = int(left_summary["holdout_window"]["n_dates"])
        if len(split["embargoed"]):
            warnings.append(
                "Summarised on dates through "
                f"{left_summary['selection_window']['end']}; "
                + _embargo_sentence(split, "either summary")
                + " `holdout_mean_abs_rank_ic` is each set's mean |rank IC| "
                f"on the {n_held} date(s) after the embargo, which neither "
                "summary read. Compare the sets on that: `mean_abs_rank_ic` "
                "is in-sample."
            )
        else:
            warnings.append(
                "Summarised on dates through "
                f"{left_summary['selection_window']['end']};"
                f" `holdout_mean_abs_rank_ic` is each set's mean |rank IC| on "
                f"the {n_held} date(s) after it, which neither summary read. "
                "Compare the sets on that: `mean_abs_rank_ic` is in-sample."
            )
        if n_held < 20:
            warnings.append(
                f"NOTE: the holdout is {n_held} date(s), too few for a rank IC "
                "to mean much; widen holdout_fraction or the panel."
            )

    per_feature = [
        {
            "feature": feature,
            "in_left": feature in set(left),
            "in_right": feature in set(right),
            "abs_rank_ic": _abs_rank_ic(predictive, feature),
        }
        for feature in everything
    ]
    per_feature.sort(key=lambda row: (-row["abs_rank_ic"], row["feature"]))

    return {
        "left": left_summary,
        "right": right_summary,
        "only_in_left": sorted(set(left) - set(right)),
        "only_in_right": sorted(set(right) - set(left)),
        "shared": sorted(set(left) & set(right)),
        "features": per_feature,
        "delta": {
            "n_features": right_summary["n_features"] - left_summary["n_features"],
            "n_independent_signals": (
                right_summary["n_independent_signals"]
                - left_summary["n_independent_signals"]
            ),
            "mean_abs_rank_ic": (
                right_summary["mean_abs_rank_ic"] - left_summary["mean_abs_rank_ic"]
            ),
            "condition_number": (
                right_summary["condition_number"] - left_summary["condition_number"]
            ),
        },
        "warnings": warnings,
    }


__all__ = ["compare_feature_sets", "select_features", "summarize_feature_set"]
