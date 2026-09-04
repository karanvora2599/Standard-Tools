"""
Typed results for feature analysis.

WHY THESE EXIST. `modeling/analysis/feature_report.py` already computes
everything here, correctly, and has for a while. What it hands back is
nested untyped dicts, and `analyze_features` passed that straight through as
`report: Dict[str, Any]`. An agent could therefore not ask a question -- it
asked for everything and then parsed a blob, guessing at key names that no
schema ever promised.

That is the same failure `extra="forbid"` fixes on the way IN, left
unfixed on the way OUT. A tool that accepts a typo'd argument and a tool
that returns an undocumented shape are the same problem from opposite ends:
in both cases the contract is in someone's head rather than in the schema.

So the work here is typing and splitting, not computing. Every number below
comes from a function that already produced it.

THESE LIVE IN THEIR OWN MODULE ON PURPOSE. The expansion plan moves feature
analysis into a `feature_lab` runtime once the cluster is big enough to
carry one. Keeping the models and tools in dedicated files makes that a file
move rather than an extraction, which is the difference between a split that
is reviewable and one that is a diff of the whole package.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing_extensions import Annotated

from ..specs import ModelSpec


def _finite_or_none(value: Any) -> Optional[float]:
    """
    A statistic, or None when it could not be computed.

    Every number in this module comes from a cross-sectional calculation
    that is undefined on some legal inputs -- a single entity per date has
    no cross-section, a constant feature has no rank correlation. The
    library represents those as NaN, which is fine in numpy and fatal at the
    protocol boundary: `json.dumps(float("nan"))` emits a bare `NaN` token
    that is not valid JSON and that a strict JSON-RPC client rejects.

    So non-finite becomes `null`. That also happens to be the more truthful
    encoding: `0.0` for an IC that was never calculable reads as "no signal"
    when the answer is "no measurement".
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


#: A float statistic that may legitimately be absent. See `_finite_or_none`.
Stat = Annotated[Optional[float], BeforeValidator(_finite_or_none)]

#: Pydantic reserves the `model_` prefix. These results carry `model_id` and
#: friends, which is the domain's word, not Pydantic's.
_NO_PROTECTED = ConfigDict(protected_namespaces=())

#: Inputs reject what they do not declare, for the reason described above.
_FORBID_EXTRA = ConfigDict(protected_namespaces=(), extra="forbid")


# ── the per-feature numbers ─────────────────────────────────────────────


class FeatureDistribution(BaseModel):
    """How well populated and how well behaved one feature is."""

    model_config = _NO_PROTECTED

    coverage: Stat = Field(
        ..., description="Fraction of rows where the feature is present."
    )
    n_missing: int = Field(..., description="Rows where the feature is null.")
    mean: Stat
    std: Stat
    skew: Stat
    kurtosis: Stat
    outlier_rate: Stat = Field(
        ...,
        description="Fraction of observations beyond the outlier threshold. A "
        "high rate is not automatically wrong -- it is what a jump-driven "
        "feature looks like -- but it decides whether standardizing is safe.",
    )
    autocorrelation: Stat = Field(
        ...,
        description="Lag-1 autocorrelation of the feature per entity. Near 1 "
        "means a slow-moving feature; near 0 means it is re-drawn each bar.",
    )
    turnover: Stat = Field(
        ...,
        description="Mean absolute change in cross-sectional RANK per bar. "
        "This is the number that decides whether a signal survives costs: a "
        "feature with real IC and near-1.0 turnover pays the spread every bar "
        "to keep it.",
    )


class FeaturePredictive(BaseModel):
    """Cross-sectional IC and the quantile shape behind it."""

    model_config = _NO_PROTECTED

    ic_mean: Stat = Field(..., description="Mean Pearson IC across dates.")
    ic_std: Stat
    ic_icir: Stat = Field(
        ...,
        description="ic_mean / ic_std. The IC's own Sharpe -- a mean IC of "
        "0.05 that is 0.05 every month is a different asset from one that is "
        "0.30 in three months and -0.20 in the rest.",
    )
    ic_hit_rate: Stat = Field(
        ...,
        description="Fraction of dates where the IC had the same sign as " "its mean.",
    )
    ic_n_dates: int
    rank_ic_mean: Stat = Field(
        ...,
        description="Mean Spearman IC. Usually the one to trust: it does not "
        "let a handful of extreme values carry the correlation.",
    )
    rank_ic_std: Stat
    rank_ic_icir: Stat
    rank_ic_hit_rate: Stat
    rank_ic_n_dates: int
    n_quantiles: int
    quantile_spread: Stat = Field(
        ...,
        description="Mean target in the top bucket minus the bottom one. The "
        "tradeable version of the IC.",
    )
    monotonicity: Stat = Field(
        ...,
        description="Rank correlation between bucket index and bucket mean "
        "target, in [-1, 1]. A high IC with low monotonicity means the edge "
        "lives in the tails rather than across the whole distribution, which "
        "is a different strategy and often a fragile one.",
    )


class ICDecayPoint(BaseModel):
    """One point on the lead-lag curve."""

    model_config = _NO_PROTECTED

    shift: int = Field(
        ...,
        description="Bars the FEATURE was displaced by. Negative means the "
        "feature was moved back in time (made staler); positive means it was "
        "advanced, which is only physically meaningful as a test.",
    )
    ic: Stat


class FeatureICDecayResult(BaseModel):
    model_config = _NO_PROTECTED

    dataset_id: str
    feature: str
    curve: List[ICDecayPoint] = Field(
        ...,
        description="IC against the same target at each shift, ordered from "
        "most negative shift to most positive.",
    )
    ic_at_zero: Stat = Field(
        ..., description="The IC as the feature is actually aligned."
    )
    peak_shift: int = Field(
        ...,
        description="The shift with the strongest absolute IC. Anything other "
        "than a smooth decay away from a sensible peak is worth reading the "
        "reason for.",
    )
    peak_ratio: Stat = Field(
        ...,
        description="|IC at the peak| divided by the mean |IC| elsewhere. A "
        "large value means the alignment is knife-edged, which is what "
        "look-ahead looks like.",
    )
    persistence: Stat = Field(
        ...,
        description="How much IC survives one bar of staleness. A feature "
        "whose IC vanishes when delayed by a bar cannot be traded on a bar "
        "delay.",
    )
    flagged: bool = Field(
        ...,
        description="True when the curve has the shape of look-ahead rather "
        "than of a real signal.",
    )
    reason: str = Field(
        ...,
        description="Why it was or was not flagged, in words. This is the "
        "part an agent should surface to a human before trusting the feature.",
    )


class FeatureProfile(BaseModel):
    """One feature, both halves."""

    model_config = _NO_PROTECTED

    feature: str
    distribution: FeatureDistribution
    predictive: FeaturePredictive
    # `include_ic_decay` and `max_shift` were accepted and dropped: this
    # model had nowhere for a curve to land, so the two arguments were in
    # the schema an agent reads, with a cost rationale attached, and every
    # combination of them returned byte-identical output.
    #
    # It is `FeatureICDecayResult` rather than a trimmed copy of it, so what
    # arrives here is exactly what get_feature_ic_decay returns -- the same
    # object from the same function, reached in one call instead of two.
    ic_decay: Optional[FeatureICDecayResult] = Field(
        None,
        description="The lead-lag IC curve, present only when "
        "include_ic_decay was set. Identical to get_feature_ic_decay's "
        "result for this feature.",
    )


# ── redundancy ──────────────────────────────────────────────────────────


class FeatureCluster(BaseModel):
    """
    A set of features that are restatements of one another.

    `representative` is the members-only reason this type exists rather than
    a bare list. Telling an agent that four features are correlated leaves it
    with a decision; telling it which one to keep does not. The pick is the
    member with the strongest absolute rank IC, so it is the one that would
    survive on merit rather than on alphabetical order.
    """

    model_config = _NO_PROTECTED

    members: List[str]
    representative: str = Field(
        ...,
        description="The member with the strongest |rank IC|: keep this one "
        "and drop the rest, unless something outside the data argues "
        "otherwise.",
    )
    max_abs_correlation: Stat = Field(
        ...,
        description="Strongest absolute pairwise correlation inside the "
        "cluster. 1.0 means an exact restatement.",
    )
    size: int


class FeatureRedundancyResult(BaseModel):
    model_config = _NO_PROTECTED

    dataset_id: str
    n_features: int
    clusters: List[FeatureCluster] = Field(
        ...,
        description="Every cluster, including singletons -- a feature that is "
        "nobody's duplicate is a result, not an omission.",
    )
    redundant_features: List[str] = Field(
        ...,
        description="Every non-representative member of every multi-feature "
        "cluster: the drop list, already worked out.",
    )
    condition_number: Stat = Field(
        ...,
        description="Condition number of the feature correlation matrix. "
        "Above ~30 the panel is collinear enough that linear coefficients "
        "stop meaning what they appear to mean.",
    )
    vif: Dict[str, Stat] = Field(
        ...,
        description="Variance inflation factor per feature. Above 10 is the "
        "usual line; above 100 the feature is nearly a linear combination of "
        "the others.",
    )
    correlation: Dict[str, Dict[str, Stat]] = Field(
        ..., description="Pearson correlation matrix."
    )
    spearman_correlation: Dict[str, Dict[str, Stat]] = Field(
        ..., description="Rank correlation matrix."
    )
    warnings: List[str] = Field(default_factory=list)


# ── IC decay / lead-lag ─────────────────────────────────────────────────


# ── inputs ──────────────────────────────────────────────────────────────


class AnalyzeFeatureInput(BaseModel):
    """One feature, in depth. `analyze_features` is the whole-panel version."""

    model_config = _FORBID_EXTRA

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    feature: str = Field(
        ...,
        description="The feature column to profile. Call list_features or "
        "analyze_features for the names in this dataset.",
    )
    n_quantiles: int = Field(
        10,
        ge=2,
        le=100,
        description="Buckets for the quantile spread and monotonicity.",
    )
    include_ic_decay: bool = Field(
        False,
        description="Also run the lead-lag screen for this one feature. Off "
        "by default because it costs (2 * max_shift + 1) extra IC passes; "
        "get_feature_ic_decay is the same computation on its own.",
    )
    max_shift: int = Field(
        5,
        ge=1,
        le=60,
        description="Bars either side for the lead-lag screen, when it runs.",
    )


class FeatureRedundancyInput(BaseModel):
    model_config = _FORBID_EXTRA

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    features: Optional[List[str]] = Field(
        None,
        description="Features to consider. Defaults to every feature in the "
        "dataset.",
    )
    cluster_threshold: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Absolute correlation at or above which two features are "
        "grouped as near-duplicates.",
    )


class FeatureICDecayInput(BaseModel):
    model_config = _FORBID_EXTRA

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    feature: str = Field(..., description="The feature column to screen.")
    max_shift: int = Field(
        5, ge=1, le=60, description="Bars either side to displace the feature."
    )
    method: Literal["spearman", "pearson"] = Field(
        "spearman",
        description="Correlation used for the IC: 'spearman' (default, rank) "
        "or 'pearson'.",
    )


# ── selection ───────────────────────────────────────────────────────────


class DroppedFeature(BaseModel):
    """A feature that did not make the cut, and why."""

    model_config = _NO_PROTECTED

    feature: str
    reason: str = Field(
        ...,
        description="'redundant' (the same signal as a kept feature), "
        "'weak' (below the IC floor), or 'capped' (past max_features).",
    )
    detail: str = Field(..., description="The specific numbers behind the reason.")


class SelectFeaturesInput(BaseModel):
    model_config = _FORBID_EXTRA

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    features: Optional[List[str]] = Field(
        None,
        description="Features to choose from. Defaults to every feature in "
        "the dataset.",
    )
    cluster_threshold: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Absolute correlation at or above which two features are "
        "one signal, and only the strongest is kept.",
    )
    min_abs_rank_ic: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Drop a surviving feature whose |rank IC| is below this. "
        "0.0 (default) keeps everything that is not redundant. A floor around "
        "0.01-0.02 is where a cross-sectional signal stops being measurable "
        "on a few hundred dates -- but set it from what THIS panel supports, "
        "which run_feature_permutation_test answers directly.",
    )
    max_features: int = Field(
        0,
        ge=0,
        description="Hard cap after both filters, by |rank IC|. 0 (default) "
        "means no cap. A cap for a caller with a budget, not a ranking to "
        "trust -- the gap between the 20th and 21st feature is usually noise.",
    )


class SelectFeaturesResult(BaseModel):
    model_config = _NO_PROTECTED

    dataset_id: str
    selected: List[str] = Field(
        ..., description="The kept features, strongest |rank IC| first."
    )
    dropped: List[DroppedFeature] = Field(
        ...,
        description="Every exclusion with its reason. Read this before "
        "accepting the selection: a feature dropped as 'weak' may simply be "
        "unmeasurable on this panel rather than useless.",
    )
    n_considered: int
    n_selected: int
    n_clusters: int = Field(
        ...,
        description="Independent signals found among the candidates. This, "
        "not n_considered, is how many ideas the panel actually held.",
    )


class FeatureSetSummary(BaseModel):
    model_config = _NO_PROTECTED

    features: List[str]
    n_features: int
    n_independent_signals: int = Field(
        ...,
        description="Redundancy clusters. Twelve features in three clusters "
        "carry three ideas; reporting twelve overstates the diversification.",
    )
    mean_abs_rank_ic: Stat
    max_abs_rank_ic: Stat
    condition_number: Stat


class FeatureSetDelta(BaseModel):
    model_config = _NO_PROTECTED

    n_features: int
    n_independent_signals: int
    mean_abs_rank_ic: Stat
    condition_number: Stat = Field(
        ...,
        description="Right minus left. A rise here is the COST of the extra "
        "features -- more collinearity -- and is the half of the trade a "
        "single score would hide.",
    )


class FeatureSetMembership(BaseModel):
    model_config = _NO_PROTECTED

    feature: str
    in_left: bool
    in_right: bool
    abs_rank_ic: Stat


class CompareFeatureSetsInput(BaseModel):
    model_config = _FORBID_EXTRA

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    left: List[str] = Field(..., min_length=1, description="The baseline feature set.")
    right: List[str] = Field(
        ..., min_length=1, description="The candidate feature set."
    )
    cluster_threshold: float = Field(
        0.9, ge=0.0, le=1.0, description="Redundancy threshold for both sets."
    )


class CompareFeatureSetsResult(BaseModel):
    model_config = _NO_PROTECTED

    dataset_id: str
    left: FeatureSetSummary
    right: FeatureSetSummary
    delta: FeatureSetDelta
    only_in_left: List[str]
    only_in_right: List[str]
    shared: List[str]
    features: List[FeatureSetMembership] = Field(
        ..., description="Every feature in either set, strongest IC first."
    )


# ── drift ───────────────────────────────────────────────────────────────


class FeatureDriftInput(BaseModel):
    model_config = _FORBID_EXTRA

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    feature: str = Field(..., description="The feature column to check.")
    split_date: Optional[str] = Field(
        None,
        description="YYYY-MM-DD boundary between the two windows. Defaults to "
        "the median date, which splits by TIME rather than row count.",
    )
    method: Literal["spearman", "pearson"] = Field(
        "spearman",
        description="Correlation for the IC halves: 'spearman' or " "'pearson'.",
    )


class FeatureDriftResult(BaseModel):
    model_config = _NO_PROTECTED

    dataset_id: str
    feature: str
    split_date: str
    n_before: int
    n_after: int
    psi: Stat = Field(
        ...,
        description="Population Stability Index. Below 0.10 is stable, "
        "0.10-0.25 moderate, above 0.25 significant. These are conventions, "
        "not a test -- there is no null distribution behind them.",
    )
    psi_bins: int
    psi_verdict: str = Field(..., description="'stable', 'moderate' or 'significant'.")
    ks_statistic: Stat = Field(
        ..., description="Largest gap between the two empirical CDFs."
    )
    mean_before: Stat
    mean_after: Stat
    std_before: Stat
    std_after: Stat
    ic_before: Stat
    ic_after: Stat
    ic_flipped: bool = Field(
        ...,
        description="True when the IC changed SIGN across the split and was "
        "non-trivial on both sides. Distribution drift and IC decay are "
        "different failures: the first is a preprocessing problem, the second "
        "means the edge is gone.",
    )


# ── stability ───────────────────────────────────────────────────────────


class StabilityBlock(BaseModel):
    model_config = _NO_PROTECTED

    block: int
    start: str
    end: str
    n_dates: int
    ic_mean: Stat


class FeatureStabilityInput(BaseModel):
    model_config = _FORBID_EXTRA

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    feature: str = Field(..., description="The feature column to check.")
    n_blocks: int = Field(
        4,
        ge=2,
        le=50,
        description="Contiguous time blocks to split the panel into. Never "
        "shuffled: a feature's usual problem is that it worked in one regime, "
        "and interleaved folds average exactly that away.",
    )
    method: Literal["spearman", "pearson"] = Field(
        "spearman", description="'spearman' or 'pearson'."
    )


class FeatureStabilityResult(BaseModel):
    model_config = _NO_PROTECTED

    dataset_id: str
    feature: str
    n_blocks: int
    blocks: List[StabilityBlock]
    ic_overall: Stat
    ic_block_mean: Stat
    ic_block_std: Stat
    ic_block_min: Stat
    ic_block_max: Stat
    sign_consistency: Stat = Field(
        ...,
        description="Fraction of blocks whose IC has the same sign as the "
        "full-sample IC. Read this first: a mean IC of 0.04 at 0.5 sign "
        "consistency is a coin flip with a good average. But read the block "
        "ICs too -- consistent sign with collapsing magnitude is decay, and "
        "this number stays at 1.0 through it.",
    )
    worst_block: Optional[int] = None


# ── permutation ─────────────────────────────────────────────────────────


class PermutationTestInput(BaseModel):
    model_config = _FORBID_EXTRA

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    feature: str = Field(..., description="The feature column to test.")
    n_permutations: int = Field(
        200,
        ge=20,
        le=5000,
        description="Shuffles used to build the null. 200 resolves a p-value "
        "to about 0.005, which is enough to separate 'real' from 'noise' and "
        "not enough to defend a 0.001 claim. Cost is linear in this.",
    )
    method: Literal["spearman", "pearson"] = Field(
        "spearman", description="'spearman' or 'pearson'."
    )
    random_seed: int = Field(
        0, ge=0, description="Seed, so the p-value is reproducible."
    )


class PermutationTestResult(BaseModel):
    model_config = _NO_PROTECTED

    dataset_id: str
    feature: str
    observed_ic: Stat
    n_permutations: int
    n_usable_permutations: int
    null_mean: Stat
    null_std: Stat
    null_p95_abs: Stat = Field(
        ...,
        description="95th percentile of |IC| under the null: the IC this "
        "panel produces from noise alone 5% of the time. An observed IC below "
        "it is not evidence of anything.",
    )
    p_value: Stat = Field(
        ...,
        description="Two-sided empirical p-value, with the +1 correction in "
        "numerator and denominator so an exact 0 is never claimed -- 200 "
        "shuffles cannot tell 'p < 0.005' from 'p = 0'.",
    )
    significant_at_05: bool
    random_seed: int


# ── ablation ────────────────────────────────────────────────────────────


class FeatureContribution(BaseModel):
    """What one feature is worth to the fitted model."""

    model_config = _NO_PROTECTED

    feature: str
    rank: int
    metric_with: Stat
    metric_without: Stat
    contribution: Stat = Field(
        ...,
        description="How much the model LOSES without this feature, already "
        "sign-corrected for metrics where lower is better. Positive means the "
        "feature earns its place. Negative means the model was better without "
        "it, which is reported rather than clipped -- on one sample it is "
        "also what noise looks like.",
    )
    relative_contribution: Stat = Field(
        ..., description="contribution / |baseline metric|."
    )


class FeatureAblationInput(BaseModel):
    model_config = _FORBID_EXTRA

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    spec: ModelSpec = Field(
        ...,
        description="The ModelSpec to refit. Use the same spec the real "
        "experiment uses -- an ablation of a different model answers a "
        "different question.",
    )
    features: Optional[List[str]] = Field(
        None,
        description="Features to ablate one at a time. Defaults to every "
        "feature in the dataset. Narrow this FIRST if the fit budget is the "
        "problem: ablating six candidates is six refits, not forty.",
    )
    metric: Optional[str] = Field(
        None,
        description="Which OOS metric to compare. Defaults to the first "
        "finite numeric metric the baseline reports, and the chosen name is "
        "echoed back.",
    )
    max_fits: int = Field(
        200,
        ge=1,
        le=100_000,
        description="Refuse to start if the run needs more fits than this. "
        "One baseline plus one per feature, times the folds -- a 40-feature "
        "panel at 8 folds is 328 fits, which is minutes to hours. Raise it "
        "deliberately once you have seen the estimate.",
    )


class FeatureAblationResult(BaseModel):
    model_config = _NO_PROTECTED

    dataset_id: str
    metric: str = Field(..., description="The OOS metric that was compared.")
    lower_is_better: bool
    baseline_metric: Stat = Field(
        ..., description="The metric with every feature present."
    )
    n_folds: int
    n_fits: int = Field(..., description="Fits actually run: (features + 1) x folds.")
    contributions: List[FeatureContribution] = Field(
        ..., description="Ranked, most valuable first."
    )
    n_features: int
    best_feature: Optional[str] = None
    worst_feature: Optional[str] = None
    mean_contribution: Stat = None
    n_negative_contributions: int = 0
    warnings: List[str] = Field(default_factory=list)


__all__ = [
    "FeatureContribution",
    "FeatureAblationResult",
    "FeatureAblationInput",
    "StabilityBlock",
    "SelectFeaturesResult",
    "SelectFeaturesInput",
    "PermutationTestResult",
    "PermutationTestInput",
    "FeatureStabilityResult",
    "FeatureStabilityInput",
    "FeatureSetSummary",
    "FeatureSetMembership",
    "FeatureSetDelta",
    "FeatureDriftResult",
    "FeatureDriftInput",
    "DroppedFeature",
    "CompareFeatureSetsResult",
    "CompareFeatureSetsInput",
    "AnalyzeFeatureInput",
    "Stat",
    "FeatureCluster",
    "FeatureDistribution",
    "FeatureICDecayInput",
    "FeatureICDecayResult",
    "FeaturePredictive",
    "FeatureProfile",
    "FeatureRedundancyInput",
    "FeatureRedundancyResult",
    "ICDecayPoint",
]
