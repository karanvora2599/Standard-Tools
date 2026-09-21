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

import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator
from typing_extensions import Annotated

from ..analysis.feature_stability import PSI_MODERATE, PSI_SIGNIFICANT
from ..limits import MAX_PERMUTATION_DRAWS
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
    warnings: List[str] = Field(default_factory=list)


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
    warnings: List[str] = Field(default_factory=list)


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
    duplicate_of: Optional[str] = Field(
        None,
        description="For a 'redundant' drop, the kept feature this one "
        "restates -- the same name its cluster reports as representative. "
        "It is here so that 'dropped as a duplicate of what' is a field "
        "rather than a sentence to parse. None for 'weak' and 'capped', "
        "which are not about another feature.",
    )


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
    selection_end: Optional[str] = Field(
        None,
        description="Last date (YYYY-MM-DD) the selection may read; the dates "
        "after it are held out and each selected feature's IC on them is "
        "reported as `holdout_ic`. Overrides holdout_fraction.",
    )
    holdout_fraction: float = Field(
        0.3,
        ge=0.0,
        lt=1.0,
        description="Share of the panel's dates, from the end, that the "
        "selection never reads. 0.3 (default) selects on the first 70% and "
        "reports the selected features' IC on the last 30%. 0 selects on the "
        "whole panel -- measured, that manufactures about 70% of a real "
        "model's headline from pure noise -- and the result warns so.",
    )
    include_correlation: bool = Field(
        False,
        description="Also return the pairwise correlation matrix over the "
        "candidates. Off by default because it is O(n^2) in the payload -- "
        "forty candidates is sixteen hundred numbers -- and `clusters`, "
        "`vif` and `condition_number`, which always come back, answer what "
        "it is usually opened for.",
    )

    @field_validator("selection_end")
    @classmethod
    def _valid_selection_end(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            pd.Timestamp(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"selection_end must be a date: {exc}") from exc
        return v


class SelectFeaturesResult(BaseModel):
    model_config = _NO_PROTECTED

    dataset_id: str
    selected: List[str] = Field(
        ..., description="The kept features, strongest |rank IC| first."
    )
    selection_window: Dict[str, Any] = Field(
        default_factory=dict,
        description="start/end/n_dates of the dates the selection read.",
    )
    holdout_window: Optional[Dict[str, Any]] = Field(
        None,
        description="start/end/n_dates of the dates held out, or None when the "
        "selection read the whole panel.",
    )
    selection_ic: Dict[str, Stat] = Field(
        default_factory=dict,
        description="Each candidate's rank IC on the selection window. "
        "In-sample for the features it chose.",
    )
    holdout_ic: Dict[str, Stat] = Field(
        default_factory=dict,
        description="Each SELECTED feature's rank IC on the held-out dates, "
        "which the selection never read. The number to believe.",
    )
    warnings: List[str] = Field(default_factory=list)
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
    clusters: List[FeatureCluster] = Field(
        default_factory=list,
        description="Every redundancy cluster the selection resolved, "
        "singletons included. Identical to what get_feature_redundancy "
        "returns for this panel and threshold -- same members, same "
        "representative -- so reading which features were one signal does "
        "not need a second call and a second correlation matrix.",
    )
    vif: Dict[str, Stat] = Field(
        default_factory=dict,
        description="Variance inflation factor per candidate, over the "
        "selection window. Above 10 is the usual line; above 100 the feature "
        "is nearly a linear combination of the others.",
    )
    condition_number: Stat = Field(
        None,
        description="Condition number of the candidates' correlation matrix "
        "over the selection window. Above ~30 the panel is collinear enough "
        "that linear coefficients stop meaning what they appear to mean.",
    )
    correlation: Dict[str, Dict[str, Stat]] = Field(
        default_factory=dict,
        description="Pearson correlation over the candidates, present only "
        "when include_correlation was set; empty otherwise, because it is "
        "O(n^2) and the clusters already carry the decision it supports.",
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
    mean_abs_rank_ic: Stat = Field(
        ...,
        description="Mean |rank IC| over the set on the dates the summary "
        "read. In-sample whenever nothing was held out.",
    )
    max_abs_rank_ic: Stat
    condition_number: Stat
    selection_window: Dict[str, Any] = Field(
        default_factory=dict,
        description="start/end/n_dates of the dates this summary read.",
    )
    holdout_window: Optional[Dict[str, Any]] = Field(
        None,
        description="start/end/n_dates of the dates held out, or None when "
        "the summary read the whole panel.",
    )
    holdout_mean_abs_rank_ic: Stat = Field(
        None,
        description="Mean |rank IC| over the set on the held-out dates, "
        "which this summary never read. The number to compare two sets on. "
        "None when holdout_fraction was 0 and nothing was held out.",
    )
    holdout_max_abs_rank_ic: Stat = Field(
        None,
        description="Strongest |rank IC| in the set on the held-out dates. "
        "None when nothing was held out.",
    )


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
    abs_rank_ic: Stat = Field(
        ...,
        description="|rank IC| on the same dates both summaries read, so the "
        "table and the summaries cannot disagree about the window.",
    )


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
    selection_end: Optional[str] = Field(
        None,
        description="Last date (YYYY-MM-DD) either summary may read; the "
        "dates after it are held out and each set's |rank IC| on them comes "
        "back as holdout_mean_abs_rank_ic. Overrides holdout_fraction.",
    )
    holdout_fraction: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
        description="Share of the panel's dates, from the end, that neither "
        "summary reads. 0.0 (default) summarises both sets on every date, "
        "which is in-sample by construction and warned about in so many "
        "words. 0.3 summarises on the first 70% and reports each set's IC on "
        "the last 30%, which is the comparison worth acting on.",
    )

    @field_validator("selection_end")
    @classmethod
    def _valid_selection_end(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            pd.Timestamp(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"selection_end must be a date: {exc}") from exc
        return v


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
    warnings: List[str] = Field(
        default_factory=list,
        description="Always carries the in-sample caveat when nothing was "
        "held out, which is the default. A comparison whose numbers were "
        "made on every date is a comparison between the sets, not an "
        "estimate of either one's out-of-sample strength.",
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
    warnings: List[str] = Field(default_factory=list)


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
    warnings: List[str] = Field(default_factory=list)


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
    null: Literal["circular_shift", "within_date"] = Field(
        "circular_shift",
        description="How the null is drawn. 'circular_shift' (default) rolls "
        "each entity's feature series by a random offset, destroying its link "
        "to the target while keeping the feature's own serial correlation, so "
        "an autocorrelated feature against an overlapping label is tested "
        "against the null it actually lives under. 'within_date' shuffles "
        "within each date, which also destroys the serial correlation and "
        "rejected a true null 27-35% of the time on live features.",
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
    null: str = Field("within_date", description="The null the p-value is against.")
    ic_autocorrelation_lag1: Stat = Field(
        None,
        description="Lag-1 autocorrelation of the observed per-date IC series. "
        "Near zero the two nulls agree; at +0.6, where every live feature "
        "sat, only circular_shift is calibrated.",
    )
    warnings: List[str] = Field(default_factory=list)


# ── panel-wide screens ──────────────────────────────────────────────────


class FeatureSignificance(BaseModel):
    """One feature's observed IC against what this panel's noise produces."""

    model_config = _NO_PROTECTED

    feature: str
    rank_ic: Stat = Field(
        ...,
        description="Observed mean cross-sectional IC. None when the panel "
        "cannot produce one for this feature -- a constant column has no "
        "rank correlation, and 0.0 would read as 'no signal' where the "
        "answer is 'no measurement'.",
    )
    p_value: Stat = Field(
        ...,
        description="Two-sided empirical p-value against the null, with the "
        "+1 correction in numerator and denominator so an exact 0 is never "
        "claimed.",
    )
    null_p95_abs: Stat = Field(
        ...,
        description="95th percentile of |IC| under the null for THIS "
        "feature: the IC noise alone produces 5% of the time.",
    )
    ic_autocorrelation_lag1: Stat = Field(
        ...,
        description="Lag-1 autocorrelation of the observed per-date IC "
        "series. Near zero the two nulls agree; at +0.6 only "
        "'circular_shift' is calibrated.",
    )
    significant_at_05: bool
    n_usable_permutations: int = Field(
        ...,
        description="Draws that produced a finite IC. 0 means the feature "
        "was not testable at all.",
    )


class ScreenFeatureSignificanceInput(BaseModel):
    model_config = _FORBID_EXTRA

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    features: Optional[List[str]] = Field(
        None,
        description="Features to test. Defaults to every feature in the "
        "dataset, which is the point of the screen -- the floor is a "
        "property of the whole candidate set.",
    )
    n_permutations: int = Field(
        200,
        ge=20,
        le=5000,
        description="Shuffles per feature. 200 resolves a p-value to about "
        "0.005, which separates 'real' from 'noise' and does not defend a "
        "0.001 claim. Cost is linear in this and in the feature count.",
    )
    method: Literal["spearman", "pearson"] = Field(
        "spearman",
        description="Correlation used for the IC: 'spearman' (default, "
        "rank) or 'pearson'.",
    )
    null: Literal["circular_shift", "within_date"] = Field(
        "circular_shift",
        description="How the null is drawn. 'circular_shift' (default) "
        "rolls each entity's feature series by a random offset, destroying "
        "its link to the target while keeping its own serial correlation. "
        "'within_date' shuffles inside each date, which destroys that "
        "serial correlation too and rejected a true null 27-35% of the time "
        "on autocorrelated features.",
    )
    random_seed: int = Field(0, ge=0, description="Seed, so the floor is reproducible.")
    max_draws: int = Field(
        20_000,
        ge=1,
        le=MAX_PERMUTATION_DRAWS,
        description="Refuse to start if features x n_permutations exceeds "
        "this. At about 1.6 ms a draw the default is roughly half a minute "
        "and the ceiling several minutes; the product is computed before "
        "the first shuffle and the screen is REFUSED rather than truncated.",
    )


class ScreenFeatureSignificanceResult(BaseModel):
    model_config = _NO_PROTECTED

    dataset_id: str
    features: List[FeatureSignificance] = Field(
        ..., description="Ordered by |IC|, strongest first."
    )
    honest_floor: Stat = Field(
        ...,
        description="The largest null_p95_abs across the screened features: "
        "the min_abs_rank_ic that select_features can be given on THIS "
        "panel without keeping features whose IC this panel's noise "
        "reproduces. None when nothing was testable.",
    )
    floor_feature: Optional[str] = Field(
        None, description="The feature whose null set the floor."
    )
    n_features: int
    n_significant: int = Field(
        ..., description="Features with p < 0.05 against the chosen null."
    )
    n_kept_at_floor: int = Field(
        ...,
        description="Features whose |IC| clears honest_floor. Compare with "
        "how many a floor picked by eye would keep -- the warnings say.",
    )
    n_draws: int = Field(
        ...,
        description="features x n_permutations: the product that was "
        "checked against max_draws before the first shuffle.",
    )
    null: str
    random_seed: int
    warnings: List[str] = Field(default_factory=list)


class BlockPSI(BaseModel):
    """One block of the drift curve."""

    model_config = _NO_PROTECTED

    block: int
    start: str
    end: str
    n_dates: int
    psi: Stat = Field(
        None,
        description="PSI of this block against the reference block. None "
        "for block 0, which IS the reference and has no predecessor -- "
        "reporting 0.0 there would put a measurement where there is none.",
    )
    psi_verdict: Optional[str] = Field(
        None, description="'stable', 'moderate' or 'significant'; None where psi is."
    )


class FeatureStabilityScreen(BaseModel):
    """One feature's two failures, and the shape of the first over time."""

    model_config = _NO_PROTECTED

    feature: str
    split_date: str = Field(
        ...,
        description="The boundary the before/after numbers were measured "
        "across. Defaults to this feature's median date.",
    )
    psi: Stat = Field(
        ...,
        description="Population Stability Index across the split. Below "
        "0.10 stable, 0.10-0.25 moderate, above 0.25 significant.",
    )
    psi_verdict: str
    ks_statistic: Stat = Field(
        ..., description="Largest gap between the two empirical CDFs."
    )
    ic_before: Stat
    ic_after: Stat
    ic_flipped: bool = Field(
        ...,
        description="True when the IC changed SIGN across the split and was "
        "non-trivial on both sides. Independent of psi: a feature can hold "
        "its distribution and lose its edge, or drift and keep it.",
    )
    ic_overall: Stat = Field(
        ..., description="Full-sample IC. Across a break it describes neither side."
    )
    ic_block_mean: Stat
    ic_block_std: Stat
    sign_consistency: Stat = Field(
        ...,
        description="Fraction of blocks whose IC has the same sign as the "
        "full-sample IC. Read with the block ICs: consistent sign with "
        "collapsing magnitude is decay, and this stays at 1.0 through it.",
    )
    worst_block: Optional[int] = None
    psi_by_block: List[BlockPSI] = Field(
        ...,
        description="The drift curve. Under reference='first' a monotone "
        "slide accumulates and rises; under 'previous' the same slide reads "
        "flat and only a jump stands out.",
    )


class ScreenFeatureStabilityInput(BaseModel):
    model_config = _FORBID_EXTRA

    dataset_id: str = Field(
        ..., description="A dataset_id returned by build_model_dataset."
    )
    features: Optional[List[str]] = Field(
        None,
        description="Features to screen. Defaults to every feature in the " "dataset.",
    )
    n_blocks: int = Field(
        4,
        ge=2,
        le=20,
        description="Contiguous time blocks for the per-block IC and the "
        "drift curve. Never shuffled.",
    )
    method: Literal["spearman", "pearson"] = Field(
        "spearman", description="'spearman' or 'pearson'."
    )
    split_date: Optional[str] = Field(
        None,
        description="YYYY-MM-DD boundary for the before/after halves. "
        "Defaults to each feature's median date, which splits by TIME "
        "rather than row count. A date outside the panel is refused.",
    )
    reference: Literal["first", "previous"] = Field(
        "first",
        description="What each block of the drift curve is measured "
        "against: the FIRST block (drift accumulates, so a monotone slide "
        "rises) or the PREVIOUS one (a slide reads flat and a jump stands "
        "out). Run both to tell a break from a drift.",
    )
    max_features: int = Field(
        200,
        ge=1,
        le=1000,
        description="Refuse rather than return a row per feature past this.",
    )


class ScreenFeatureStabilityResult(BaseModel):
    model_config = _NO_PROTECTED

    dataset_id: str
    features: List[FeatureStabilityScreen] = Field(
        ..., description="Ordered by PSI, most drifted first."
    )
    n_features: int
    n_significant: int = Field(
        ..., description=f"Features with PSI >= {PSI_SIGNIFICANT}."
    )
    n_moderate: int = Field(
        ..., description=f"Features with PSI in [{PSI_MODERATE}, {PSI_SIGNIFICANT})."
    )
    n_stable: int
    psi_thresholds: Dict[str, float] = Field(
        ...,
        description="The two lines the verdicts are drawn at. Conventions "
        "rather than tests -- there is no null distribution behind them.",
    )
    most_drifted: Optional[str] = Field(
        None, description="The feature with the largest PSI, or None."
    )
    n_blocks: int
    reference: str
    warnings: List[str] = Field(default_factory=list)


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
    preprocessing_fitted: int = Field(
        0,
        description=(
            "Fold pipelines fitted across the whole ablation. For a "
            "column-wise pipeline this is the baseline's folds only: each "
            "leave-one-out run reads its matrices off the baseline's, minus "
            "the column, which is exact."
        ),
    )
    preprocessing_reused: int = Field(
        0,
        description="Fold pipelines read from the cache instead of refitted.",
    )
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
    "BlockPSI",
    "FeatureContribution",
    "FeatureAblationResult",
    "FeatureAblationInput",
    "FeatureSignificance",
    "FeatureStabilityScreen",
    "ScreenFeatureSignificanceInput",
    "ScreenFeatureSignificanceResult",
    "ScreenFeatureStabilityInput",
    "ScreenFeatureStabilityResult",
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
