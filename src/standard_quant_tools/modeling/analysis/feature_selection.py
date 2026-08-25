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
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .feature_report import feature_predictive_stats, redundancy_report

logger = logging.getLogger(__name__)


def _abs_rank_ic(stats: Dict[str, Dict[str, float]], feature: str) -> float:
    value = (stats.get(feature) or {}).get("rank_ic_mean")
    return abs(float(value)) if value is not None and np.isfinite(value) else 0.0


def select_features(
    panel: pd.DataFrame,
    feature_ids: Sequence[str],
    *,
    cluster_threshold: float = 0.9,
    min_abs_rank_ic: float = 0.0,
    max_features: int = 0,
) -> Dict[str, Any]:
    """
    Keep one feature per redundancy cluster, drop what does not predict, and
    record a reason for every exclusion.

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

    predictive = feature_predictive_stats(panel, feature_ids)
    redundancy = redundancy_report(
        panel, feature_ids, cluster_threshold=cluster_threshold
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

    return {
        "selected": kept,
        "dropped": sorted(dropped, key=lambda d: d["feature"]),
        "n_considered": len(feature_ids),
        "n_selected": len(kept),
        "n_clusters": len(redundancy["clusters"]),
        "cluster_threshold": cluster_threshold,
        "min_abs_rank_ic": min_abs_rank_ic,
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
