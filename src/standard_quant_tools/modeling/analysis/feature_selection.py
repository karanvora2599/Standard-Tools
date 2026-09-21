"""
Choosing a feature set, and comparing two of them.

Selection here is deliberately BORING: drop what is redundant, drop what
does not predict, keep the rest, and say why for every drop. There is no
search, no wrapper method, no greedy forward pass. That is a deliberate
limit rather than an unfinished one.

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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .feature_report import feature_predictive_stats, redundancy_report

logger = logging.getLogger(__name__)


def _abs_rank_ic(stats: Dict[str, Dict[str, float]], feature: str) -> float:
    value = (stats.get(feature) or {}).get("rank_ic_mean")
    return abs(float(value)) if value is not None and np.isfinite(value) else 0.0


def _signed_rank_ic(
    stats: Dict[str, Dict[str, float]], feature: str
) -> Optional[float]:
    value = (stats.get(feature) or {}).get("rank_ic_mean")
    return float(value) if value is not None and np.isfinite(value) else None


def _date_label(value: Any) -> str:
    return str(pd.Timestamp(value).date())


def _selection_cutoff(
    panel: pd.DataFrame, selection_end: Any, holdout_fraction: float
) -> "tuple[pd.DatetimeIndex, Optional[pd.Timestamp]]":
    """
    The last date the selection may read. `selection_end` names it;
    otherwise the first `1 - holdout_fraction` of the panel's dates select
    and the rest are held out; a zero fraction selects on everything and
    holds out nothing, which the result then says in so many words.
    """
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"]).unique()))
    if len(dates) < 2:
        raise ValidationError(
            "select_features: the panel has fewer than two dates, so nothing "
            "can be held out and no cross-sectional IC can be trusted."
        )
    if selection_end is not None:
        cutoff = pd.Timestamp(selection_end)
        if cutoff < dates[0] or cutoff >= dates[-1]:
            raise ValidationError(
                f"select_features: selection_end={_date_label(cutoff)!r} must "
                f"fall inside the panel's dates ({_date_label(dates[0])}.."
                f"{_date_label(dates[-1])}) and leave at least one date after "
                "it to hold out."
            )
        return dates, cutoff
    if holdout_fraction <= 0:
        return dates, None
    if holdout_fraction >= 1:
        raise ValidationError(
            f"select_features: holdout_fraction={holdout_fraction} would hold "
            "out every date; it must be below 1."
        )
    n_select = int(np.floor(len(dates) * (1.0 - holdout_fraction)))
    n_select = min(max(n_select, 1), len(dates) - 1)
    return dates, dates[n_select - 1]


def select_features(
    panel: pd.DataFrame,
    feature_ids: Sequence[str],
    *,
    cluster_threshold: float = 0.9,
    min_abs_rank_ic: float = 0.0,
    max_features: int = 0,
    selection_end: Any = None,
    holdout_fraction: float = 0.3,
) -> Dict[str, Any]:
    """
    Keep one feature per redundancy cluster, drop what does not predict, and
    record a reason for every exclusion.

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

    Order matters and is not arbitrary. Redundancy is resolved FIRST, then
    the IC floor is applied. The other way round, a cluster whose members
    are all individually below the floor would be dropped entirely -- but a
    cluster is one signal, and the right question is whether that one signal
    clears the floor, asked once via its representative.

    `max_features` truncates by absolute rank IC after both filters. It is
    a cap for a caller who has a hard budget, not a ranking to trust: the
    difference between the 20th and 21st feature by IC on one panel is
    usually noise.
    """
    feature_ids = list(feature_ids)
    if not feature_ids:
        raise ValidationError("select_features: no features to choose from")
    missing = [f for f in feature_ids if f not in panel.columns]
    if missing:
        raise ValidationError(f"panel has no features: {sorted(missing)}")

    dates, cutoff = _selection_cutoff(panel, selection_end, holdout_fraction)
    if cutoff is None:
        selection_panel, holdout_panel = panel, panel.iloc[0:0]
    else:
        date_values = pd.to_datetime(panel["date"])
        selection_panel = panel[date_values <= cutoff]
        holdout_panel = panel[date_values > cutoff]

    predictive = feature_predictive_stats(selection_panel, feature_ids)
    redundancy = redundancy_report(
        selection_panel, feature_ids, cluster_threshold=cluster_threshold
    )

    dropped: List[Dict[str, str]] = []
    survivors: List[str] = []
    for members in redundancy["clusters"]:
        members = sorted(members)
        # Strongest |rank IC|, ties broken by the FIRST name alphabetically.
        # Written as a sort rather than a max because `max` on a
        # (value, name) key breaks ties toward the LAST name, and the drop
        # list has to agree with get_feature_redundancy's representative or
        # the two tools contradict each other on the same panel.
        keeper = sorted(members, key=lambda f: (-_abs_rank_ic(predictive, f), f))[0]
        survivors.append(keeper)
        for member in members:
            if member != keeper:
                dropped.append(
                    {
                        "feature": member,
                        "reason": "redundant",
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
                    "detail": (
                        f"|rank IC| {strength:.4f} below the "
                        f"{min_abs_rank_ic:.4f} floor"
                    ),
                }
            )
        else:
            kept.append(feature)

    kept.sort(key=lambda f: (-_abs_rank_ic(predictive, f), f))
    if max_features and len(kept) > max_features:
        for feature in kept[max_features:]:
            dropped.append(
                {
                    "feature": feature,
                    "reason": "capped",
                    "detail": (
                        f"ranked {kept.index(feature) + 1} by |rank IC|, past "
                        f"the max_features={max_features} cap"
                    ),
                }
            )
        kept = kept[:max_features]

    warnings: List[str] = []
    holdout_ic: Dict[str, Optional[float]] = {}
    holdout_window: Optional[Dict[str, Any]] = None
    if cutoff is None:
        selection_window = {
            "start": _date_label(dates[0]),
            "end": _date_label(dates[-1]),
            "n_dates": int(len(dates)),
        }
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
        held = dates[dates > cutoff]
        selection_window = {
            "start": _date_label(dates[0]),
            "end": _date_label(cutoff),
            "n_dates": int((dates <= cutoff).sum()),
        }
        holdout_window = {
            "start": _date_label(held[0]),
            "end": _date_label(held[-1]),
            "n_dates": int(len(held)),
        }
        if kept:
            holdout_stats = feature_predictive_stats(holdout_panel, kept)
            holdout_ic = {f: _signed_rank_ic(holdout_stats, f) for f in kept}
        warnings.append(
            f"Selected on dates through {selection_window['end']}; "
            f"`holdout_ic` is each selected feature's rank IC on the "
            f"{holdout_window['n_dates']} date(s) after it, which the selection "
            "never read. That is the number to believe: `selection_ic` chose "
            "the features and is optimistic by construction."
        )
        if holdout_window["n_dates"] < 20:
            warnings.append(
                f"NOTE: the holdout is {holdout_window['n_dates']} date(s), too "
                "few for a rank IC to mean much; widen holdout_fraction or the "
                "panel."
            )

    return {
        "selected": kept,
        "dropped": sorted(dropped, key=lambda d: d["feature"]),
        "n_considered": len(feature_ids),
        "n_selected": len(kept),
        "n_clusters": len(redundancy["clusters"]),
        "cluster_threshold": cluster_threshold,
        "min_abs_rank_ic": min_abs_rank_ic,
        "selection_window": selection_window,
        "holdout_window": holdout_window,
        "selection_ic": {f: _signed_rank_ic(predictive, f) for f in feature_ids},
        "holdout_ic": holdout_ic,
        "warnings": warnings,
    }


def summarize_feature_set(
    panel: pd.DataFrame,
    feature_ids: Sequence[str],
    *,
    cluster_threshold: float = 0.9,
) -> Dict[str, Any]:
    """
    One feature set, as the handful of numbers worth comparing.

    `n_independent_signals` is the one to read rather than `n_features`. A
    set of twelve features in three clusters carries three ideas, and
    reporting twelve overstates the diversification by four times.
    """
    feature_ids = list(feature_ids)
    predictive = feature_predictive_stats(panel, feature_ids)
    redundancy = redundancy_report(
        panel, feature_ids, cluster_threshold=cluster_threshold
    )
    strengths = np.array(
        [_abs_rank_ic(predictive, f) for f in feature_ids], dtype=float
    )
    return {
        "features": sorted(feature_ids),
        "n_features": len(feature_ids),
        "n_independent_signals": len(redundancy["clusters"]),
        "mean_abs_rank_ic": float(np.mean(strengths)) if strengths.size else 0.0,
        "max_abs_rank_ic": float(np.max(strengths)) if strengths.size else 0.0,
        "condition_number": float(redundancy["condition_number"]),
    }


def compare_feature_sets(
    panel: pd.DataFrame,
    left: Sequence[str],
    right: Sequence[str],
    *,
    cluster_threshold: float = 0.9,
) -> Dict[str, Any]:
    """
    Two feature sets on the same panel, with the cost of the difference
    attached.

    Deliberately NOT a single score. A larger set almost always has a higher
    max IC and almost always has more collinearity, and an agent handed one
    number cannot see the trade it just made. What comes back is per-set
    diagnostics, the features unique to each side, and a per-feature IC
    table for everything in either.

    Both sets are measured on the same panel, so the comparison is like for
    like. Comparing sets scored on different date ranges would be comparing
    the ranges.
    """
    left, right = list(left), list(right)
    if not left or not right:
        raise ValidationError("compare_feature_sets: both sets must be non-empty")
    unknown = sorted({f for f in left + right if f not in panel.columns})
    if unknown:
        raise ValidationError(f"panel has no features: {unknown}")

    everything = sorted(set(left) | set(right))
    predictive = feature_predictive_stats(panel, everything)

    left_summary = summarize_feature_set(
        panel, left, cluster_threshold=cluster_threshold
    )
    right_summary = summarize_feature_set(
        panel, right, cluster_threshold=cluster_threshold
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
    }


__all__ = ["compare_feature_sets", "select_features", "summarize_feature_set"]
