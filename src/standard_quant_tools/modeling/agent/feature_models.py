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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing_extensions import Annotated


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


class FeatureProfile(BaseModel):
    """One feature, both halves."""

    model_config = _NO_PROTECTED

    feature: str
    distribution: FeatureDistribution
    predictive: FeaturePredictive


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
    method: str = Field(
        "spearman",
        description="Correlation used for the IC: 'spearman' (default, rank) "
        "or 'pearson'.",
    )


__all__ = [
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
