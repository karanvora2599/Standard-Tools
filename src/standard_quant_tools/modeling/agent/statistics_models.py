"""
The inputs and results for the two statistics the library computed and
threw away before anything could read them.

WHY THESE ARE A MODULE PAIR. `statistics_tools.py` needs none of
`tools.py`'s private helpers -- it resolves a published reference, reads
columns off the frame it gets back, and calls two validation modules --
so its models live beside it rather than in `models.py`, the way
`dataset_tools.py`, `portfolio_models.py` and `discovery_models.py` do.
The seam is the point: nothing here can be broken by an edit to the tool
file every other modeling tool shares.

WHAT THE TWO HAVE IN COMMON. Each wraps a function with exactly one
caller deep inside something else, whose output was averaged or narrowed
away before the tool boundary:

    INTERVALS. `validation/distributional.py` scores quantiles and
                intervals on every fold of an experiment, and the fold
                loop averages the result into one headline number per
                metric. The interval columns themselves ARE handed to the
                agent -- the out-of-sample frame carries `q05`/`q50`/`q95`
                and `lower`/`upper` when the spec asked for them, and
                scoring re-emits them on new data -- and nothing could ask
                whether they covered. The `by` axis is the whole reason:
                one pooled coverage cannot separate a band that covered
                97% in a calm stretch from the same band covering 62% in
                a selloff.

    COMPARISON.  The Holm adjustment was reachable only through a
                comparison of two REGISTERED models sharing a task and a
                label, in a one-against-many star; a family of p-values
                from anywhere else had no correction at all. The
                heteroskedasticity- and autocorrelation-consistent
                variance had no caller outside its own module, while two
                other runtimes printed a warning that their t-statistics
                might not survive it.

See the CHANGELOG entry of 2026-09-21.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..tasks import Task
from .models import Stat

# ── score_prediction_intervals ─────────────────────────────────────────


class ScorePredictionIntervalsInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it had
    # configured something.
    model_config = ConfigDict(extra="forbid")

    predictions_ref: str = Field(
        ...,
        description=(
            "A published `predictions` reference carrying the distribution "
            "columns: the out-of-sample frame an experiment publishes, a "
            "scored frame, or any (date, entity, target) frame published "
            "with kind='predictions' that also holds `lower`/`upper` or "
            "`qNN` columns. A raw artifact path is refused -- it carries no "
            "kind, so nothing can check that it is predictions at all."
        ),
    )
    target_column: str = Field(
        "target",
        description=(
            "The realized outcome the band is judged against. Coverage is "
            "the share of rows whose outcome fell inside the band, so this "
            "must be the SAME label the band was built for: judging a "
            "5-day forward return against a band fitted for a 1-day one "
            "reports a number that describes neither."
        ),
    )
    quantile_columns: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "Column -> quantile level, e.g. {'q05': 0.05, 'q95': 0.95}. "
            "Left unset, the columns are auto-detected by the naming "
            "convention this library writes them under: `q05` is the 5th "
            "percentile, `q95` the 95th, `q02.5` a level that is not a "
            "whole percent. Set it to read differently named columns, or "
            "to score a SUBSET of the ones present. Pinball loss is "
            "reported per level; coverage and width are reported for every "
            "SYMMETRIC pair (0.05 with 0.95), because a lone quantile has "
            "no interval to cover anything."
        ),
    )
    lower_column: str = Field(
        "lower",
        description=(
            "Lower edge of a conformal (or any other) prediction interval. "
            "Absent from the frame it is simply not scored; present "
            "WITHOUT its upper counterpart the call is refused, because "
            "half a band has no coverage."
        ),
    )
    upper_column: str = Field(
        "upper",
        description="Upper edge of the prediction interval. See lower_column.",
    )
    nominal_coverage: float = Field(
        0.9,
        gt=0.0,
        lt=1.0,
        description=(
            "What the `lower`/`upper` band CLAIMS to cover -- 0.9 for a "
            "band built at alpha=0.1, which is the library's default. This "
            "is not read off the frame because the frame does not carry it: "
            "the columns are two numbers per row and the claim behind them "
            "lives in the spec that built them. Realized coverage is "
            "compared against it, and a gap wider than the sampling error "
            "of this many rows is reported in `warnings`."
        ),
    )
    by: Literal["all", "date", "entity"] = Field(
        "all",
        description=(
            "Pool every row ('all'), or report coverage per date or per "
            "entity as well. THE POOLED NUMBER IS THE ONE THAT MISLEADS: "
            "exchangeability is the assumption behind a conformal interval "
            "and a return panel is not exchangeable across regimes, so a "
            "band covering 97% in a calm stretch and 62% in a selloff "
            "pools to a healthy-looking 90%. 'date' is the axis that "
            "separates them; 'entity' separates a band that works on the "
            "liquid names and not on the rest."
        ),
    )
    min_group_rows: int = Field(
        20,
        ge=2,
        description=(
            "Groups with fewer rows than this are still reported and are "
            "FLAGGED as too short to read, and are left out of worst_group, "
            "best_group and the spread warning. Coverage on eight rows "
            "takes one of nine values; calling the lowest of them a regime "
            "failure would be reading sampling noise."
        ),
    )
    max_groups: int = Field(
        500,
        ge=1,
        le=5000,
        description=(
            "Refuse rather than emit a row per group for a panel with "
            "thousands of them. A per-date breakdown of four years of daily "
            "data is a thousand rows of JSON that no reader gets through; "
            "narrow the frame, or ask for 'entity' instead, or raise this "
            "deliberately."
        ),
    )


class IntervalGroup(BaseModel):
    """Coverage for one date or one entity, on that slice's rows alone."""

    key: str = Field(
        ...,
        description="The date (YYYY-MM-DD) or entity this slice covers.",
    )
    n_rows: int = Field(
        ...,
        description=(
            "Rows in the slice after dropping those missing the outcome or "
            "a band edge. Below min_group_rows the coverage below is noise "
            "and `warnings` says which groups those were."
        ),
    )
    interval_coverage: Stat = Field(
        None,
        description=(
            "Share of this slice's outcomes that fell inside "
            "[lower, upper]. Null when the frame carries no interval "
            "columns."
        ),
    )
    quantile_coverage: Dict[str, Stat] = Field(
        default_factory=dict,
        description=(
            "Coverage of each symmetric quantile pair on this slice, keyed "
            "`quantile_coverage_90` as the library's own metrics are."
        ),
    )
    crossing_rate: Stat = Field(
        None,
        description=(
            "Share of this slice's rows where a lower quantile was "
            "predicted ABOVE a higher one. Those rows are not a "
            "distribution and their width is negative."
        ),
    )


class ScorePredictionIntervalsResult(BaseModel):
    n_rows: int = Field(
        ...,
        description=(
            "Rows scored, after dropping those missing the outcome or any "
            "column being scored. Coverage is a proportion on this many "
            "draws, and its own standard error is sqrt(p(1-p)/n) -- on a "
            "few hundred rows a 90% band routinely measures 87% or 93% "
            "while being exactly right."
        ),
    )
    n_groups: int = Field(
        ...,
        description=(
            "How many groups `groups` describes. Zero when by='all': the "
            "top-level figures ARE the whole sample, and repeating them as "
            "a single group would say nothing."
        ),
    )
    quantile_levels: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "The columns read as quantiles and the level each was read at. "
            "Empty when the frame carried none and only the interval was "
            "scored."
        ),
    )
    pinball: Dict[str, Stat] = Field(
        default_factory=dict,
        description=(
            "Mean pinball (quantile) loss per level, keyed `pinball_q05` as "
            "the fold metrics are, so a number here is the same number an "
            "experiment recorded under the same name. This is the PROPER "
            "score for a quantile: a 95th percentile exceeded 5% of the "
            "time minimizes it and one exceeded 20% of the time does not, "
            "however tight it looks. At 0.5 it is half the mean absolute "
            "error. Lower is better; the scale is the target's, so it is "
            "comparable across models on one label and across nothing else."
        ),
    )
    quantile_crossing_rate: Stat = Field(
        None,
        description=(
            "Share of rows where the quantiles were not in order -- the 5th "
            "predicted above the 95th. Separately fitted quantile models "
            "can do this and a single fit cannot. Above zero, the pair is "
            "not a distribution on those rows and anything reading a width "
            "off it reads a negative number."
        ),
    )
    quantile_coverage: Dict[str, Stat] = Field(
        default_factory=dict,
        description=(
            "Share of outcomes inside each symmetric quantile pair, keyed "
            "`quantile_coverage_90`. Read beside the width below: coverage "
            "alone is maximized by a band wide enough to contain "
            "everything."
        ),
    )
    quantile_width: Dict[str, Stat] = Field(
        default_factory=dict,
        description=(
            "Mean width of each symmetric quantile pair, keyed "
            "`quantile_width_90`, in the target's units. THE OTHER HALF OF "
            "COVERAGE: an interval that covers 90% of outcomes by spanning "
            "ten times the target's spread has told nobody anything, and "
            "only this field shows it."
        ),
    )
    interval_coverage: Stat = Field(
        None,
        description=(
            "Share of outcomes inside [lower, upper]. Null when the frame "
            "carries no interval columns. This is the honest check on a "
            "conformal band -- the theory promises coverage under "
            "exchangeability, and this number is what the band did."
        ),
    )
    interval_width: Stat = Field(
        None,
        description=(
            "Mean width of [lower, upper] in the target's units. A "
            "conformal radius is constant within a fold, so this is close "
            "to twice that radius."
        ),
    )
    interval_nominal_coverage: Stat = Field(
        None,
        description=(
            "What the band claimed, echoed from `nominal_coverage` so the "
            "realized figure beside it can be read without the input. "
            "Null when there was no interval to score."
        ),
    )
    groups: List[IntervalGroup] = Field(
        default_factory=list,
        description=(
            "Per-date or per-entity coverage, ordered by key. Empty for "
            "by='all'. This is where a regime failure shows: the spread "
            "across these, not the pooled number, is what says whether the "
            "band's promise held everywhere."
        ),
    )
    worst_group: Optional[str] = Field(
        None,
        description=(
            "The group whose coverage is FURTHEST from nominal in either "
            "direction -- under-covering is a broken promise and "
            "over-covering was bought with width. Chosen among groups that "
            "clear min_group_rows; null when none do, or when by='all'."
        ),
    )
    best_group: Optional[str] = Field(
        None,
        description=(
            "The group whose coverage is closest to nominal, among those "
            "clearing min_group_rows. With worst_group it brackets the "
            "spread the pooled number hides."
        ),
    )
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Conditions that change how these numbers should be read: the "
            "exchangeability caveat behind a pooled coverage, quantiles "
            "that crossed, groups too short to read, coverage outside the "
            "sampling band around nominal, and coverage that varies across "
            "groups by more than that band."
        ),
    )


# ── compare_signals ────────────────────────────────────────────────────

#: A p-value is a probability. Out of [0, 1] it is not a p-value that is
#: slightly wrong, it is a number that was not a p-value, and adjusting it
#: would return a confident-looking answer about nothing.
_Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class AdjustedPValue(BaseModel):
    """One test's p-value before and after the multiple-testing adjustment."""

    label: str = Field(
        ...,
        description="The caller's name for this test, echoed unchanged.",
    )
    p_value: float = Field(
        ...,
        description="The p-value as supplied, before any adjustment.",
    )
    p_adjusted: float = Field(
        ...,
        description=(
            "The adjusted p-value under `method`, monotone in the original "
            "p-value and capped at one. All three methods return the "
            "quantity whose comparison to alpha IS the procedure's own "
            "decision, so this can be read against any level without "
            "another call -- the same numbers statsmodels' multipletests "
            "and R's p.adjust return. Under 'bh' it is still a different "
            "quantity from the other two: see `method`."
        ),
    )
    reject_at_alpha: bool = Field(
        ...,
        description=(
            "Whether the procedure rejects this null at `alpha`, which is "
            "exactly `p_adjusted <= alpha` for every method. Reported "
            "beside it so the decision does not have to be re-derived, not "
            "because the two could differ."
        ),
    )


class CompareSignalsInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it had
    # configured something.
    model_config = ConfigDict(extra="forbid")

    mode: Literal["paired", "ic_series", "adjust"] = Field(
        ...,
        description=(
            "Which question is being asked. 'paired' compares two published "
            "prediction frames on the rows both predicted -- no registry, no "
            "shared task required beyond the one you name. 'ic_series' "
            "compares two per-date information-coefficient series computed "
            "anywhere, and adds the autocorrelation-consistent variance of "
            "the difference. 'adjust' applies a multiple-testing correction "
            "to p-values from ANYWHERE -- this library, another one, a "
            "notebook. A field belonging to a different mode is refused by "
            "name rather than ignored."
        ),
    )

    # -- mode="paired" ------------------------------------------------
    predictions_ref_a: Optional[str] = Field(
        None,
        description=(
            "mode='paired': the BASELINE frame, published as `predictions` "
            "with date, entity, prediction and target columns. The reported "
            "difference is b minus a, so this is the one being improved on."
        ),
    )
    predictions_ref_b: Optional[str] = Field(
        None,
        description=(
            "mode='paired': the CANDIDATE frame. Joined to a on "
            "(date, entity); the realized outcomes must agree on the "
            "intersection, which refuses a comparison of two signals fitted "
            "against different labels under the same name."
        ),
    )
    task: Optional[Task] = Field(
        None,
        description=(
            "mode='paired': how to read the prediction column, which "
            "decides whether a loss with units exists. 'regression' gets a "
            "squared-error Diebold-Mariano test beside the correlation "
            "difference and 'classification' a Brier one; 'ranking' and "
            "'survival' have no loss in the target's units, so the "
            "correlation difference carries the comparison alone."
        ),
    )
    metric: Literal["cs_rank_ic", "cs_ic"] = Field(
        "cs_rank_ic",
        description=(
            "mode='paired': the per-date cross-sectional correlation the "
            "difference series is built from. Rank IC (Spearman) is the "
            "default because it is not moved by the handful of extreme "
            "outcomes that dominate a Pearson IC on returns."
        ),
    )
    horizon: int = Field(
        1,
        ge=1,
        description=(
            "mode='paired': bars the label looks forward. Sets the lag of "
            "the Diebold-Mariano long-run variance to horizon - 1, because "
            "two rows fewer than that many bars apart still share a bar. "
            "Left at 1 on an overlapping label, the loss test's standard "
            "error is too small and its p-value too confident."
        ),
    )

    # -- mode="ic_series" ---------------------------------------------
    ic_a: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "mode='ic_series': date -> information coefficient for the "
            "BASELINE, e.g. {'2023-01-03': 0.021, ...}. Any source at all; "
            "nothing is checked against a model. At least ten dates must "
            "carry an IC for both series."
        ),
    )
    ic_b: Optional[Dict[str, float]] = Field(
        None,
        description=(
            "mode='ic_series': date -> information coefficient for the "
            "CANDIDATE. Aligned to ic_a on the shared dates; the object of "
            "interest is the per-date difference, whose dispersion is far "
            "smaller than either series' own because the two share every "
            "good and bad day."
        ),
    )
    hac_lag: Optional[int] = Field(
        None,
        ge=0,
        description=(
            "mode='ic_series': Bartlett-kernel lag for the "
            "autocorrelation-consistent variance of the mean difference. "
            "Unset, the usual data-driven rule floor(4 * (n/100)^(2/9)) is "
            "used. Zero is the ordinary variance of the mean and makes "
            "hac_ratio exactly 1."
        ),
    )

    # -- shared by paired and ic_series -------------------------------
    n_bootstrap: int = Field(
        2000,
        ge=100,
        le=20000,
        description=(
            "Resamples behind the interval on the mean difference. The "
            "smallest reportable two-sided p-value is 2 / n_bootstrap, so "
            "at the floor of 100 nothing below 0.02 can be distinguished "
            "from it."
        ),
    )
    block_size: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Consecutive dates resampled together. Unset, n^(1/3), the rule "
            "this library's other bootstraps use. 1 is an IID resample and "
            "DESTROYS the serial correlation an overlapping label induces, "
            "narrowing the interval by a factor already measured at 2.24x "
            "on an AR(1) at phi=0.8 -- set it to 1 only to see that effect, "
            "never to report a result."
        ),
    )
    confidence: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description="Two-sided coverage of the interval on the mean difference.",
    )
    seed: int = Field(
        0,
        description="Seed for the resampler, so the interval is reproducible.",
    )

    # -- mode="adjust" ------------------------------------------------
    p_values: Optional[Dict[str, _Probability]] = Field(
        None,
        min_length=1,
        max_length=1000,
        description=(
            "mode='adjust': label -> p-value, from anywhere. The labels are "
            "echoed back beside the adjusted values so a family of a dozen "
            "candidate signals stays readable. Each must be in [0, 1]; an "
            "empty family is refused, because a correction across zero "
            "tests is not a null answer, it is a mistake upstream."
        ),
    )
    method: Literal["holm", "bonferroni", "bh"] = Field(
        "holm",
        description=(
            "mode='adjust': which correction. 'holm' and 'bonferroni' "
            "control the FAMILY-WISE error rate -- the chance of one false "
            "rejection anywhere -- and Holm dominates Bonferroni uniformly, "
            "which is why it is the default. 'bh' controls the FALSE "
            "DISCOVERY RATE, the expected share of the rejections that are "
            "false: it rejects far more and each rejection is a weaker "
            "claim. None of them controls for the candidates having been "
            "chosen on the same sample that produced these p-values."
        ),
    )
    alpha: float = Field(
        0.05,
        gt=0.0,
        lt=1.0,
        description=(
            "mode='adjust': the level each adjusted p-value is compared "
            "against, which sets `reject_at_alpha` and `n_rejected`. The "
            "unadjusted p-values are returned beside them, so a different "
            "level does not need another call."
        ),
    )


class CompareSignalsResult(BaseModel):
    mode: str = Field(..., description="The mode that produced this result.")
    comparison: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "mode='paired' and 'ic_series': the statistics of the per-date "
            "difference b minus a. `mean_difference` is the improvement, "
            "`ci_lower`/`ci_upper` the block-bootstrap interval on it, "
            "`p_value` the two-sided bootstrap p-value, `hit_rate` the "
            "share of DECIDED dates b won (null when every date tied, which "
            "is what two identical signals produce and what a share of all "
            "dates would read as 'b lost every day'), and `verdict` one of "
            "b_better / a_better / indistinguishable. mode='paired' adds "
            "the row and entity counts and, where the task has a loss with "
            "units, `diebold_mariano`. 'indistinguishable' means this "
            "sample does not separate the two, whichever headline number "
            "is larger."
        ),
    )
    hac: Optional[Dict[str, Stat]] = Field(
        None,
        description=(
            "mode='ic_series': the variance of the MEAN difference without "
            "and with the autocorrelation correction -- `hac_variance_lag0`, "
            "`hac_variance`, the `hac_lag` used, and `hac_ratio` between "
            "them. Above 1 the difference series is positively "
            "autocorrelated and any t-statistic computed from the ordinary "
            "variance is overstated by sqrt(hac_ratio); on a real series "
            "that factor has been measured at 2.8. Near 1 the correction "
            "changes nothing and the ordinary standard error stands."
        ),
    )
    adjusted: List[AdjustedPValue] = Field(
        default_factory=list,
        description=(
            "mode='adjust': every test, ordered by adjusted p-value, with "
            "its original value and the rejection decision. Empty in the "
            "other modes, which run one comparison and have nothing to "
            "adjust across."
        ),
    )
    n_tests: int = Field(
        0,
        description=(
            "How many tests this call ran: one for a paired or IC-series "
            "comparison, the size of the family for mode='adjust'. In the "
            "adjust case it is the number the correction divides the error "
            "budget by, so leaving a candidate out of the call is the "
            "difference between an honest correction and a flattering one."
        ),
    )
    method: Optional[str] = Field(
        None,
        description=(
            "The correction applied, null outside mode='adjust'. Reported "
            "because 'p_adjusted' means a different thing under 'bh' than "
            "under the other two."
        ),
    )
    n_rejected: int = Field(
        0,
        description=(
            "How many nulls were rejected at alpha after the adjustment. "
            "Zero on a family of pure noise is the expected answer, not a "
            "failure of the call."
        ),
    )
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "What these numbers do not cover: that no multiple-testing "
            "correction accounts for the candidates having been SELECTED on "
            "this same sample, that a false-discovery-rate rejection is a "
            "weaker claim than a family-wise one, and that an interval on a "
            "short series is wide by construction so 'indistinguishable' on "
            "it means 'not enough dates to tell'."
        ),
    )


__all__ = [
    "AdjustedPValue",
    "CompareSignalsInput",
    "CompareSignalsResult",
    "IntervalGroup",
    "ScorePredictionIntervalsInput",
    "ScorePredictionIntervalsResult",
]
