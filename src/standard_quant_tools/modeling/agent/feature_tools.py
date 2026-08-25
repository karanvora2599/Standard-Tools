"""
Feature analysis, one question per tool.

`analyze_features` answers every question at once and returns a nested,
untyped `report`. That is the right tool when an agent wants an overview and
the wrong one for everything else: to find out whether `momentum_20d` is
worth keeping, an agent had to profile all forty features, then guess at key
names no schema promised.

These tools ask one question each, and their answers are typed. Nothing here
computes anything new -- every number comes from
`modeling/analysis/feature_report.py`, which already produced it.

WHAT IS ACTUALLY NEW is the interpretation the types make room for:

- `get_feature_redundancy` names a *representative* per cluster rather than
  handing back a list of correlated names. "These four are the same signal"
  leaves an agent with a decision; "keep this one, drop those three" does
  not.
- `get_feature_ic_decay` returns the curve as ordered points with a named
  peak, rather than a dict keyed by stringified shifts. An agent reading
  `{"-2": ..., "-1": ...}` has to sort string keys numerically to see the
  shape, and sorting them as strings puts "-1" after "-2" and "0" before
  "1" -- correct by luck, on this range only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.feature_models import (
    AnalyzeFeatureInput,
    CompareFeatureSetsInput,
    CompareFeatureSetsResult,
    FeatureCluster,
    FeatureDistribution,
    FeatureDriftInput,
    FeatureDriftResult,
    FeatureICDecayInput,
    FeatureICDecayResult,
    FeaturePredictive,
    FeatureProfile,
    FeatureRedundancyInput,
    FeatureRedundancyResult,
    FeatureStabilityInput,
    FeatureStabilityResult,
    ICDecayPoint,
    PermutationTestInput,
    PermutationTestResult,
    SelectFeaturesInput,
    SelectFeaturesResult,
)
from standard_quant_tools.modeling.analysis.feature_report import (
    feature_distribution_stats,
    feature_predictive_stats,
    lead_lag_ic_curve,
    redundancy_report,
)
from standard_quant_tools.modeling.analysis.feature_selection import (
    compare_feature_sets as _compare_feature_sets,
)
from standard_quant_tools.modeling.analysis.feature_selection import (
    select_features as _select_features,
)
from standard_quant_tools.modeling.analysis.feature_stability import (
    feature_drift as _feature_drift,
)
from standard_quant_tools.modeling.analysis.feature_stability import (
    feature_stability as _feature_stability,
)
from standard_quant_tools.modeling.analysis.feature_stability import (
    permutation_test_ic as _permutation_test_ic,
)

logger = logging.getLogger(__name__)


def _resolve_features(meta: Dict[str, Any], requested, dataset_id: str) -> List[str]:
    features = list(requested or meta.get("feature_ids", []) or [])
    if not features:
        raise ValidationError(
            f"dataset {dataset_id!r} records no feature_ids, and none were "
            "supplied. Pass `features` explicitly."
        )
    return features


def _require_feature(panel, feature: str, dataset_id: str) -> None:
    if feature in panel.columns:
        return
    from difflib import get_close_matches

    known = [c for c in panel.columns if c not in ("date", "entity", "target")]
    near = get_close_matches(feature, known, n=3)
    hint = f" Did you mean: {near}?" if near else f" Available: {sorted(known)[:8]}"
    raise ValidationError(f"dataset {dataset_id!r} has no feature {feature!r}.{hint}")


def _pick_representative(
    members: Sequence[str], predictive: Dict[str, Dict[str, float]]
) -> str:
    """
    The member of a cluster worth keeping.

    Strongest absolute rank IC, because that is the one that would survive a
    selection made on merit. Ties -- and a genuine tie is common when the
    cluster is an exact restatement -- fall back to the alphabetically first
    name, so the answer is stable across runs rather than dependent on dict
    ordering. A representative that moved between identical calls would make
    every downstream drop-list unreproducible.
    """

    def key(name: str):
        stats = predictive.get(name) or {}
        ic = stats.get("rank_ic_mean")
        return (-abs(ic) if isinstance(ic, (int, float)) else 0.0, name)

    return sorted(members, key=key)[0]


def analyze_feature(input_data: AnalyzeFeatureInput) -> FeatureProfile:
    """
    Profile ONE feature: how well populated it is, how fast it turns over,
    what it predicts, and the quantile shape behind that prediction.

    The single-feature counterpart to `analyze_features`. Reach for this
    when a specific feature is in question -- it costs one feature's work
    rather than the whole panel's, and every number comes back named and
    typed rather than nested inside a report dict.
    """
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

    logger.debug(
        "[analyze_feature] dataset_id=%s feature=%s",
        input_data.dataset_id,
        input_data.feature,
    )
    panel, _meta, _dir = _load_dataset_panel(input_data.dataset_id)
    _require_feature(panel, input_data.feature, input_data.dataset_id)

    feature = input_data.feature
    distribution = feature_distribution_stats(panel, [feature])[feature]
    predictive = feature_predictive_stats(
        panel, [feature], n_quantiles=input_data.n_quantiles
    )[feature]

    return FeatureProfile(
        feature=feature,
        distribution=FeatureDistribution(**distribution),
        predictive=FeaturePredictive(**predictive),
    )


def get_feature_redundancy(
    input_data: FeatureRedundancyInput,
) -> FeatureRedundancyResult:
    """
    Which features are restatements of one another, and which one to keep.

    RSI, 20-day momentum, MACD and stochastic are one momentum cluster, not
    four independent sources of alpha. A panel that treats them as four
    will report a model leaning on "many" features while it leans on one
    idea, and will size positions as though it had diversified.

    Returns the clusters with a representative each, the drop list already
    worked out, and the collinearity diagnostics (VIF, condition number)
    that say whether linear coefficients on this panel mean anything.
    """
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

    logger.debug("[get_feature_redundancy] dataset_id=%s", input_data.dataset_id)
    panel, meta, _dir = _load_dataset_panel(input_data.dataset_id)
    features = _resolve_features(meta, input_data.features, input_data.dataset_id)
    for feature in features:
        _require_feature(panel, feature, input_data.dataset_id)

    report = redundancy_report(
        panel, features, cluster_threshold=input_data.cluster_threshold
    )
    predictive = feature_predictive_stats(panel, features)
    correlation = report["correlation"]

    clusters: List[FeatureCluster] = []
    redundant: List[str] = []
    for members in report["clusters"]:
        members = list(members)
        representative = _pick_representative(members, predictive)
        pairs = [
            abs(correlation.get(a, {}).get(b, 0.0))
            for a in members
            for b in members
            if a != b
        ]
        clusters.append(
            FeatureCluster(
                members=sorted(members),
                representative=representative,
                max_abs_correlation=max(pairs) if pairs else 1.0,
                size=len(members),
            )
        )
        redundant.extend(m for m in members if m != representative)

    warnings: List[str] = []
    for cluster in clusters:
        if cluster.size > 1:
            others = sorted(set(cluster.members) - {cluster.representative})
            warnings.append(
                f"{cluster.size} features are one signal at "
                f"|rho| >= {input_data.cluster_threshold:.2f}: keep "
                f"{cluster.representative!r}, drop {others}."
            )
    condition_number = float(report["condition_number"])
    if condition_number > 30.0:
        warnings.append(
            f"condition number {condition_number:.0f} -- the panel is "
            "collinear enough that a linear model's coefficients are not "
            "individually interpretable."
        )

    return FeatureRedundancyResult(
        dataset_id=input_data.dataset_id,
        n_features=len(features),
        clusters=sorted(clusters, key=lambda c: (-c.size, c.representative)),
        redundant_features=sorted(redundant),
        condition_number=condition_number,
        vif=report["vif"],
        correlation=correlation,
        spearman_correlation=report["spearman_correlation"],
        warnings=warnings,
    )


def get_feature_ic_decay(input_data: FeatureICDecayInput) -> FeatureICDecayResult:
    """
    How a feature's IC behaves when the feature is displaced in time.

    Two questions at once. The first is leakage: a feature whose IC spikes
    sharply at shift 0 and collapses on both sides already contains the
    answer, because a real signal degrades gracefully rather than
    knife-edging. The second is tradeability: `persistence` says how much IC
    survives one bar of staleness, and a feature that loses all of it cannot
    be traded on a one-bar delay no matter how good shift 0 looks.

    Positive shifts advance the feature -- letting it see further into the
    target window. That is not something to do in production; it is the
    control that makes the shape readable.
    """
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

    logger.debug(
        "[get_feature_ic_decay] dataset_id=%s feature=%s",
        input_data.dataset_id,
        input_data.feature,
    )
    panel, _meta, _dir = _load_dataset_panel(input_data.dataset_id)
    _require_feature(panel, input_data.feature, input_data.dataset_id)

    result = lead_lag_ic_curve(
        panel,
        input_data.feature,
        max_shift=input_data.max_shift,
        method=input_data.method,
    )
    # Sorted NUMERICALLY. The underlying dict is keyed by stringified
    # shifts, and sorting those as text puts "-10" before "-2".
    points = [
        ICDecayPoint(shift=int(shift), ic=float(ic))
        for shift, ic in sorted(result["curve"].items(), key=lambda kv: int(kv[0]))
    ]
    peak = max(points, key=lambda p: abs(p.ic)) if points else None

    return FeatureICDecayResult(
        dataset_id=input_data.dataset_id,
        feature=input_data.feature,
        curve=points,
        ic_at_zero=float(result["ic_at_zero"]),
        peak_shift=peak.shift if peak else 0,
        peak_ratio=float(result["peak_ratio"]),
        persistence=float(result["persistence"]),
        flagged=bool(result["flagged"]),
        reason=str(result["reason"]),
    )


def select_features(input_data: SelectFeaturesInput) -> SelectFeaturesResult:
    """
    Choose a feature set: drop the duplicates, drop what does not predict,
    and give a reason for every exclusion.

    Deliberately boring. There is no greedy search here, because a selector
    scored on the same panel it selects from manufactures overfit that looks
    like evidence -- and an agent handed that output has no way to tell. The
    two criteria used instead, "this is the same feature twice" and "this has
    no measurable relationship with the target", are the two a human can be
    shown afterwards.

    Redundancy is resolved BEFORE the IC floor. A cluster is one signal, so
    the question is whether that signal clears the floor, asked once through
    its representative -- not whether each restatement clears it separately.
    """
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

    logger.debug("[select_features] dataset_id=%s", input_data.dataset_id)
    panel, meta, _dir = _load_dataset_panel(input_data.dataset_id)
    features = _resolve_features(meta, input_data.features, input_data.dataset_id)
    for feature in features:
        _require_feature(panel, feature, input_data.dataset_id)

    result = _select_features(
        panel,
        features,
        cluster_threshold=input_data.cluster_threshold,
        min_abs_rank_ic=input_data.min_abs_rank_ic,
        max_features=input_data.max_features,
    )
    return SelectFeaturesResult(
        dataset_id=input_data.dataset_id,
        selected=result["selected"],
        dropped=result["dropped"],
        n_considered=result["n_considered"],
        n_selected=result["n_selected"],
        n_clusters=result["n_clusters"],
    )


def compare_feature_sets(
    input_data: CompareFeatureSetsInput,
) -> CompareFeatureSetsResult:
    """
    Two feature sets on the same panel, with the cost of the difference
    attached.

    Not a single score, on purpose. A larger set almost always has a higher
    maximum IC and almost always carries more collinearity, so one number
    hides half of the trade. What comes back is per-set diagnostics, what is
    unique to each side, and the per-feature IC for everything in either --
    including `n_independent_signals`, which is the honest count of ideas
    where `n_features` is the count of columns.
    """
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

    logger.debug("[compare_feature_sets] dataset_id=%s", input_data.dataset_id)
    panel, _meta, _dir = _load_dataset_panel(input_data.dataset_id)
    for feature in set(input_data.left) | set(input_data.right):
        _require_feature(panel, feature, input_data.dataset_id)

    result = _compare_feature_sets(
        panel,
        input_data.left,
        input_data.right,
        cluster_threshold=input_data.cluster_threshold,
    )
    return CompareFeatureSetsResult(dataset_id=input_data.dataset_id, **result)


def get_feature_drift(input_data: FeatureDriftInput) -> FeatureDriftResult:
    """
    Whether a feature is still the same measurement, and still predicts, on
    either side of a date.

    Two failures that look alike in a full-sample report and need different
    fixes. A feature can drift in DISTRIBUTION while keeping its IC, which is
    a preprocessing problem -- rescale it. Or it can hold its distribution
    while losing its IC, which means the edge is gone and no amount of
    normalizing brings it back. Reporting only one invites fixing the wrong
    one.

    The full-sample IC averages across the break, and an average across a
    break describes neither side of it.
    """
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

    logger.debug(
        "[get_feature_drift] dataset_id=%s feature=%s",
        input_data.dataset_id,
        input_data.feature,
    )
    panel, _meta, _dir = _load_dataset_panel(input_data.dataset_id)
    _require_feature(panel, input_data.feature, input_data.dataset_id)

    result = _feature_drift(
        panel,
        input_data.feature,
        split_date=input_data.split_date,
        method=input_data.method,
    )
    return FeatureDriftResult(dataset_id=input_data.dataset_id, **result)


def get_feature_regime_stability(
    input_data: FeatureStabilityInput,
) -> FeatureStabilityResult:
    """
    The feature's IC inside each of several CONTIGUOUS time blocks.

    Contiguous, never shuffled. A feature's usual problem is that it worked
    in one regime and not others, and randomly interleaved folds average
    exactly that away -- reproducing the failure this tool exists to expose.

    Read `sign_consistency` first, then the block ICs. Consistency alone
    misses decay: a feature going 0.44, 0.44, 0.01, 0.02 keeps a sign
    consistency of 1.0 while its edge disappears.
    """
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

    logger.debug(
        "[get_feature_regime_stability] dataset_id=%s feature=%s",
        input_data.dataset_id,
        input_data.feature,
    )
    panel, _meta, _dir = _load_dataset_panel(input_data.dataset_id)
    _require_feature(panel, input_data.feature, input_data.dataset_id)

    result = _feature_stability(
        panel,
        input_data.feature,
        n_blocks=input_data.n_blocks,
        method=input_data.method,
    )
    return FeatureStabilityResult(dataset_id=input_data.dataset_id, **result)


def run_feature_permutation_test(
    input_data: PermutationTestInput,
) -> PermutationTestResult:
    """
    How often noise on THIS panel produces an IC as large as the observed
    one.

    An IC of 0.03 over 60 dates and 20 entities is a number noise produces
    routinely, and no amount of staring at it reveals that. The feature is
    shuffled within each date, which states the null exactly: the feature
    carries no cross-sectional information within a date.

    The p-value is TWO-SIDED. A feature with an IC of -0.20 is a strong
    feature with a sign, not a weak one, and a one-sided test would report
    it as unremarkable.

    `null_p95_abs` is the number to keep: it is the IC this panel yields from
    noise alone 5% of the time, and it is the honest floor for
    `select_features(min_abs_rank_ic=...)` on this data.
    """
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel

    logger.debug(
        "[run_feature_permutation_test] dataset_id=%s feature=%s n=%d",
        input_data.dataset_id,
        input_data.feature,
        input_data.n_permutations,
    )
    panel, _meta, _dir = _load_dataset_panel(input_data.dataset_id)
    _require_feature(panel, input_data.feature, input_data.dataset_id)

    result = _permutation_test_ic(
        panel,
        input_data.feature,
        n_permutations=input_data.n_permutations,
        method=input_data.method,
        random_seed=input_data.random_seed,
    )
    return PermutationTestResult(dataset_id=input_data.dataset_id, **result)


FEATURE_TOOL_DISPATCH = {
    "analyze_feature": (analyze_feature, AnalyzeFeatureInput),
    "get_feature_redundancy": (get_feature_redundancy, FeatureRedundancyInput),
    "get_feature_ic_decay": (get_feature_ic_decay, FeatureICDecayInput),
    "select_features": (select_features, SelectFeaturesInput),
    "compare_feature_sets": (compare_feature_sets, CompareFeatureSetsInput),
    "get_feature_drift": (get_feature_drift, FeatureDriftInput),
    "get_feature_regime_stability": (
        get_feature_regime_stability,
        FeatureStabilityInput,
    ),
    "run_feature_permutation_test": (
        run_feature_permutation_test,
        PermutationTestInput,
    ),
}

__all__ = [
    "FEATURE_TOOL_DISPATCH",
    "analyze_feature",
    "compare_feature_sets",
    "get_feature_drift",
    "get_feature_ic_decay",
    "get_feature_redundancy",
    "get_feature_regime_stability",
    "run_feature_permutation_test",
    "select_features",
]
