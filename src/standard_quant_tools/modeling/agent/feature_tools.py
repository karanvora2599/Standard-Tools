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

import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.agent.feature_models import (
    AnalyzeFeatureInput,
    CompareFeatureSetsInput,
    CompareFeatureSetsResult,
    FeatureAblationInput,
    FeatureAblationResult,
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
from standard_quant_tools.modeling.analysis.feature_ablation import (
    _lower_is_better,
    _metric_value,
    ablation_contributions,
    estimate_ablation_fits,
    summarize_ablation,
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


def _ic_decay_result(
    panel,
    *,
    dataset_id: str,
    feature: str,
    max_shift: int,
    method: str,
) -> FeatureICDecayResult:
    """The lead-lag curve as a result model.

    Extracted so `profile_feature` reaches the same computation rather than
    growing a second one -- `include_ic_decay`'s own description says
    "get_feature_ic_decay is the same computation on its own", and it has to
    stay true.
    """
    result = lead_lag_ic_curve(panel, feature, max_shift=max_shift, method=method)
    # Sorted NUMERICALLY. The underlying dict is keyed by stringified
    # shifts, and sorting those as text puts "-10" before "-2".
    points = [
        ICDecayPoint(shift=int(shift), ic=float(ic))
        for shift, ic in sorted(result["curve"].items(), key=lambda kv: int(kv[0]))
    ]
    peak = max(points, key=lambda p: abs(p.ic)) if points else None
    return FeatureICDecayResult(
        dataset_id=dataset_id,
        feature=feature,
        curve=points,
        ic_at_zero=float(result["ic_at_zero"]),
        peak_shift=peak.shift if peak else 0,
        peak_ratio=float(result["peak_ratio"]),
        persistence=float(result["persistence"]),
        flagged=bool(result["flagged"]),
        reason=str(result["reason"]),
    )


def profile_feature(input_data: AnalyzeFeatureInput) -> FeatureProfile:
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
        "[profile_feature] dataset_id=%s feature=%s",
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

    # The two arguments that were accepted and dropped. Off by default
    # because the cost their description advertises is real: the curve is
    # (2 * max_shift + 1) extra IC passes over the panel.
    decay = (
        _ic_decay_result(
            panel,
            dataset_id=input_data.dataset_id,
            feature=feature,
            max_shift=input_data.max_shift,
            method="spearman",
        )
        if input_data.include_ic_decay
        else None
    )

    return FeatureProfile(
        feature=feature,
        distribution=FeatureDistribution(**distribution),
        predictive=FeaturePredictive(**predictive),
        ic_decay=decay,
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

    return _ic_decay_result(
        panel,
        dataset_id=input_data.dataset_id,
        feature=input_data.feature,
        max_shift=input_data.max_shift,
        method=input_data.method,
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


def _first_numeric_metric(metrics):
    """
    The metric to compare when the caller did not name one.

    Deterministic rather than arbitrary: sorted, so two runs on the same
    dataset compare the same thing. The chosen name is echoed in the result,
    because an ablation ranked on a metric the caller did not choose and
    cannot see is a table of numbers with no stated meaning.
    """
    import math

    for name in sorted(metrics):
        value = metrics[name]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isfinite(float(value)):
                return name
    raise ValidationError(
        "the baseline experiment reported no finite numeric OOS metric, so "
        "there is nothing to rank an ablation by"
    )


def run_feature_ablation(input_data: FeatureAblationInput) -> FeatureAblationResult:
    """
    Refit the model without each feature in turn, and report what each one
    was worth.

    This is the only feature tool that asks a MODEL-RELATIVE question. Every
    other one scores a feature on its own -- its IC, its stability, its
    distribution -- which is cheap and usually right, but cannot tell you
    what happens to a fitted model when the feature goes away. The two
    differ in the direction that costs money: a strong feature that
    duplicates another contributes nothing marginal, and a mediocre feature
    that is the sole source of some information can be the one holding the
    model up.

    It is also the most expensive tool here by a wide margin. One baseline
    plus one refit per feature, each across every walk-forward fold: a
    40-feature panel with 8 folds is 328 fits. The count is computed BEFORE
    anything is fit and the run is refused if it exceeds `max_fits`, so an
    afternoon is spent on purpose rather than by accident.

    None of these refits are registered. They are candidate models nobody
    asked for, and 41 of them in the registry to answer one question is not
    a trade worth making -- but the fits themselves are the fits
    run_model_experiment would do, so the metrics are the real ones.
    """
    from standard_quant_tools.modeling.agent.tools import _load_dataset_panel
    from standard_quant_tools.modeling.engine import build_splitter, run_experiment
    from standard_quant_tools.modeling.specs import ModelSpec

    logger.debug("[run_feature_ablation] dataset_id=%s", input_data.dataset_id)
    spec = (
        input_data.spec
        if isinstance(input_data.spec, ModelSpec)
        else ModelSpec(**input_data.spec)
    )
    panel, meta, _dir = _load_dataset_panel(input_data.dataset_id)
    features = _resolve_features(meta, input_data.features, input_data.dataset_id)
    for feature in features:
        _require_feature(panel, feature, input_data.dataset_id)
    if len(features) < 2:
        raise ValidationError(
            "ablation needs at least two features: removing the only feature "
            "leaves nothing to fit, which is not a comparison."
        )

    # Folds come from the real splitter against the real dates, so the
    # estimate and the run cannot disagree.
    dates = pd.Index(sorted(panel["date"].unique()))
    n_folds = int(build_splitter(spec.validation).n_splits(dates))
    if n_folds < 1:
        raise ValidationError(
            f"dataset {input_data.dataset_id!r} has {len(dates)} dates, not "
            "enough for one fold under this validation spec. Ablation cannot "
            "compare models that cannot be validated."
        )

    n_fits = estimate_ablation_fits(len(features), n_folds)
    if n_fits > input_data.max_fits:
        raise ValidationError(
            f"this ablation needs {n_fits:,} fits "
            f"({len(features)} features + 1 baseline, x {n_folds} folds), over "
            f"the max_fits={input_data.max_fits:,} ceiling. That is minutes to "
            "hours of compute. Either narrow `features` to the candidates you "
            f"actually doubt, or pass max_fits={n_fits} to accept the cost."
        )

    def _fit(subset):
        dataset = {
            "panel": panel,
            "feature_ids": list(subset),
            "target_id": meta["target_id"],
            "data_hash": meta["data_hash"],
            "spec_hash": meta.get("spec_hash"),
            "warnings": meta.get("warnings", []),
        }
        return run_experiment(
            dataset, spec, dataset_id=input_data.dataset_id, register=False
        )["oos_metrics"]

    baseline_metrics = _fit(features)
    metric = input_data.metric or _first_numeric_metric(baseline_metrics)
    baseline = _metric_value(baseline_metrics, metric)

    without = {}
    for feature in features:
        remaining = [f for f in features if f != feature]
        without[feature] = _metric_value(_fit(remaining), metric)

    rows = ablation_contributions(baseline, without, metric)
    summary = summarize_ablation(rows, metric)

    return FeatureAblationResult(
        dataset_id=input_data.dataset_id,
        metric=metric,
        lower_is_better=_lower_is_better(metric),
        baseline_metric=baseline,
        n_folds=n_folds,
        n_fits=n_fits,
        contributions=rows,
        **summary,
    )


FEATURE_TOOL_DEFS: List[tuple] = [
    (
        "profile_feature",
        "Profile ONE feature of a built dataset: coverage, turnover, "
        "autocorrelation, cross-sectional IC and ICIR, quantile spread and "
        "monotonicity. The single-feature counterpart to analyze_features — "
        "reach for it when a specific feature is in question, since it costs "
        "one feature's work instead of the whole panel's and returns every "
        "number named rather than nested in a report dict.",
        AnalyzeFeatureInput,
    ),
    (
        "get_feature_redundancy",
        "Which features are restatements of one another, and which one to "
        "keep. RSI, 20-day momentum, MACD and stochastic are one momentum "
        "cluster, not four independent sources of alpha. Returns each "
        "cluster with a representative chosen by strongest rank IC, the drop "
        "list already worked out, and the collinearity diagnostics (VIF, "
        "condition number) that say whether linear coefficients on this "
        "panel mean anything.",
        FeatureRedundancyInput,
    ),
    (
        "get_feature_ic_decay",
        "How one feature's IC behaves when the feature is displaced in time. "
        "Answers two questions: whether it leaks (an IC that spikes at shift "
        "0 and collapses on both sides already contains the answer) and "
        "whether it is tradeable (how much IC survives one bar of "
        "staleness). Returns the curve as ordered points with the peak "
        "named.",
        FeatureICDecayInput,
    ),
    (
        "select_features",
        "Choose a feature set from a built dataset: keep one feature per "
        "redundancy cluster, drop what falls below an IC floor, and return a "
        "reason for every exclusion. Deliberately has no greedy search -- a "
        "selector scored on the panel it selects from manufactures overfit "
        "that looks like evidence. Redundancy is resolved before the IC "
        "floor, because a cluster is one signal and the question is whether "
        "THAT signal clears the floor.",
        SelectFeaturesInput,
    ),
    (
        "compare_feature_sets",
        "Two feature sets measured on the same panel, with the cost of the "
        "difference attached: per-set IC, independent-signal count and "
        "condition number, what is unique to each side, and the per-feature "
        "IC table. Not a single score, because a larger set almost always "
        "has a higher maximum IC and almost always more collinearity, and "
        "one number hides half of that trade.",
        CompareFeatureSetsInput,
    ),
    (
        "get_feature_drift",
        "Whether a feature is still the same measurement, and still "
        "predicts, either side of a date. Returns PSI and a two-sample KS "
        "for the distribution, plus the IC computed separately on each half. "
        "The two fail differently and need different fixes: distribution "
        "drift with a stable IC is a preprocessing problem, while a stable "
        "distribution with a collapsed IC means the edge is gone.",
        FeatureDriftInput,
    ),
    (
        "get_feature_regime_stability",
        "The feature's IC inside each of several CONTIGUOUS time blocks, "
        "never shuffled -- a feature's usual problem is that it worked in "
        "one regime, and interleaved folds average exactly that away. "
        "Returns per-block IC plus sign consistency against the full-sample "
        "IC. Read both: consistent sign with collapsing magnitude is decay, "
        "and sign consistency stays at 1.0 through it.",
        FeatureStabilityInput,
    ),
    (
        "run_feature_permutation_test",
        "How often noise on THIS panel produces an IC as large as the "
        "observed one, in either direction. Shuffles the feature within each "
        "date, which states the null exactly -- the feature carries no "
        "cross-sectional information within a date -- and returns a "
        "TWO-SIDED empirical p-value, so a strongly negative IC is "
        "significant rather than ignored. null_p95_abs is the IC this panel "
        "yields from noise alone 5% of the time, which is the defensible "
        "floor for select_features(min_abs_rank_ic=...). Cost is linear in "
        "n_permutations.",
        PermutationTestInput,
    ),
    (
        "run_feature_ablation",
        "Refit the model without each feature in turn and report what each "
        "one was worth. The only feature tool that asks a MODEL-RELATIVE "
        "question: a strong feature that duplicates another contributes "
        "nothing marginal, and a mediocre feature that is the sole source of "
        "some information can be the one holding the model up. Neither shows "
        "in a per-feature report or in tree importance. EXPENSIVE -- one "
        "baseline plus one refit per feature across every fold, so 40 "
        "features at 8 folds is 328 fits. The count is computed before "
        "anything is fit and the run is REFUSED past max_fits, so narrow "
        "`features` to the candidates you actually doubt.",
        FeatureAblationInput,
    ),
]


FEATURE_TOOL_DISPATCH = {
    "profile_feature": (profile_feature, AnalyzeFeatureInput),
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
    "run_feature_ablation": (run_feature_ablation, FeatureAblationInput),
}

__all__ = [
    "FEATURE_TOOL_DEFS",
    "FEATURE_TOOL_DISPATCH",
    "feature_dispatch",
    "get_feature_tools",
    "profile_feature",
    "compare_feature_sets",
    "get_feature_drift",
    "get_feature_ic_decay",
    "get_feature_redundancy",
    "get_feature_regime_stability",
    "run_feature_ablation",
    "run_feature_permutation_test",
    "select_features",
]


def get_feature_tools() -> List[Dict[str, Any]]:
    """Tool definitions for the feature_lab runtime, in the same
    OpenAI-style envelope as get_agent_tools() and get_modeling_tools()."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": input_model.model_json_schema(),
            },
        }
        for name, description, input_model in FEATURE_TOOL_DEFS
    ]


def feature_dispatch(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute one feature_lab tool by name.

    Refuses anything it does not own, by name, for the same reason every
    other runtime dispatcher does: a scoped agent that hallucinates a tool
    must get an error rather than a successful result from somewhere else.
    """
    entry = FEATURE_TOOL_DISPATCH.get(name)
    if entry is None:
        raise ValidationError(
            f"unknown feature_lab tool {name!r}. This runtime provides: "
            f"{sorted(FEATURE_TOOL_DISPATCH)}"
        )
    fn, model = entry
    return fn(model(**arguments)).model_dump()
