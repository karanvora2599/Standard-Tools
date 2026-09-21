"""
Inputs and results for the three dry runs: the schedule, the weights, the
transform.

Each of these describes something the engine already computes on every
call and then keeps to itself. The plan is a pure function of the spec and
the date axis; the weights are a pure function of the panel's dates,
entities and label ends; the fitted pipeline is a pure function of the
training rows. None of the three needs an estimator, so all three can be
answered before the first fit -- which is the only time the answer can
still change a decision.

Why the models live beside `preview_tools.py` rather than in `models.py`:
the module pair is how a tool that needs none of the tool module's own
result types stays out of that file (`dataset_tools.py` and
`portfolio_models.py` are the precedents). The spec classes are imported
from `modeling.specs` and nested whole, so a caller composes the SAME
`ModelSpec`, `WeightingSpec` and `PreprocessingSpec` it would hand
`run_model_experiment` -- a preview of a spec that had to be retyped in a
different shape would be a preview of a different spec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..specs import ModelSpec, PreprocessingSpec, WeightingSpec
from .models import Stat

# ── plan_model_experiment ───────────────────────────────────────────────


class PlanModelExperimentInput(BaseModel):
    # An argument this tool does not take is REJECTED, not ignored.
    # Pydantic's default would drop it silently, so a typo or a
    # hallucinated name ran on defaults while the caller believed it had
    # configured something.
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(
        ...,
        description="A dataset built by build_model_dataset. The panel is "
        "loaded and verified against the hash recorded when it was built, "
        "because the purge counts and the fold hashes below describe THOSE "
        "rows.",
    )
    spec: ModelSpec = Field(
        ...,
        description="The same ModelSpec run_model_experiment would be given. "
        "Nothing is fitted: the validation method, the search, the "
        "quantiles, the conformal blocks and the budget are read, and the "
        "estimator is read only for its identity in the fold hashes.",
    )
    target: Optional[str] = Field(
        None,
        description=(
            "Which label to plan against, for a dataset registered with "
            "several. Omitted, the primary is used. The choice matters here "
            "and not only at fit time: rows whose CHOSEN label is null are "
            "dropped, which shortens the date axis the folds are cut from, "
            "and a longer horizon purges more training rows per fold."
        ),
    )
    include_folds: bool = Field(
        True,
        description="Return the per-fold schedule. Set False for the totals "
        "alone on a plan with many folds.",
    )
    include_candidates: bool = Field(
        False,
        description=(
            "Return the parameter combinations the search will actually "
            "score, in the order it will score them. Off by default because "
            "a large grid is a large payload; on, it is the only way to see "
            "a log-spacing mistake before paying for it. A 'tpe' search has "
            "no list -- it chooses each candidate from what the previous "
            "ones scored -- and reports its trial budget in a warning "
            "instead."
        ),
    )


class FoldPlanDescription(BaseModel):
    """One outer fold: which dates, which rows, what it will cost."""

    model_config = ConfigDict(extra="forbid")

    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_dates: int = Field(
        ...,
        description="Dates in the training window that SURVIVE the purge, "
        "which is what the inner search is sized against -- not the dates "
        "the splitter scheduled.",
    )
    n_test_dates: int
    n_inner_folds: int = Field(
        ...,
        description=(
            "Inner folds this training window supports for the search. Zero "
            "means the window is too short: the search does not run on that "
            "fold, the fold is priced at one fit, and the parameters it uses "
            "are the spec's base values rather than anything that was "
            "scored."
        ),
    )
    n_candidates: int
    n_fits: int
    node_hash: str = Field(
        ...,
        description="Content hash of everything that determines this fold's "
        "fitted estimator: the dataset, the fold's rows, the resolved "
        "pipeline, the estimator with its parameters, and the seed.",
    )
    preprocessing_hash: str = Field(
        ...,
        description="Content hash of everything that determines this fold's "
        "preprocessed matrices -- the node hash without the estimator, its "
        "parameters and the seed. Two specs differing only in estimator "
        "share it.",
    )
    n_train_rows: Optional[int] = Field(
        None, description="Training rows left after the purge."
    )
    n_test_rows: Optional[int] = None
    n_purged: Optional[int] = Field(
        None,
        description=(
            "Training rows this fold loses because their label window "
            "reaches into its test block. None only when the plan was built "
            "without a panel, which cannot happen here -- validate_model_"
            "spec plans on a date COUNT and reports None on every fold, and "
            "the difference between that None and this number is the "
            "difference between an estimate and the schedule."
        ),
    )


class PlanModelExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str = Field(..., description="The validation method planned.")
    n_dates: int = Field(
        ...,
        description="Dates on the axis the folds were cut from: the panel's "
        "sorted unique dates after the target selection, which is the axis "
        "the engine walks.",
    )
    n_folds: int
    n_candidates: int = Field(
        ...,
        description="Parameter combinations the search scores per fold. Zero "
        "without a search.",
    )
    fits_per_fit: int = Field(
        ...,
        description=(
            "Estimator fits one 'fit' of this spec costs. One, unless the "
            "classifier is calibrated (one fit per calibration fold), the "
            "regression asks for quantiles (one more per level), or it asks "
            "for a conformal interval (one more per calibration block). "
            "Three quantiles beside five conformal blocks is nine, and that "
            "multiplier is invisible in the spec."
        ),
    )
    n_fits_folds: int
    n_fits_refit: int = Field(
        ...,
        description="The full-panel refit that produces the deployed "
        "estimator, plus the full-panel search that chooses its parameters.",
    )
    n_inner_final: int
    n_fits_final_search: int
    n_fits: int = Field(
        ...,
        description="Total estimator fits this spec implies. The number "
        "run_model_experiment checks against the budget and then runs.",
    )
    max_fits: int = Field(..., description="The spec's own ceiling, budget.max_fits.")
    within_budget: bool = Field(
        ...,
        description=(
            "False is REPORTED here and refused by run_model_experiment. "
            "Refusing a dry run would defeat it: the count and what brings "
            "it under the ceiling are exactly what the caller came for, and "
            "nothing is fitted either way."
        ),
    )
    dataset_hash: Optional[str] = None
    has_panel: bool = Field(
        ...,
        description="Always true here, and stated because the same planner "
        "answers a panel-free question elsewhere with every row count and "
        "purge count absent.",
    )
    n_purged: Optional[int] = Field(
        None, description="Training rows the purge removes across all folds."
    )
    final_search_hash: Optional[str] = Field(
        None,
        description="Content hash of the full-panel search that chooses the "
        "deployed parameters. None without a search.",
    )
    folds: List[FoldPlanDescription] = Field(default_factory=list)
    candidates: Optional[List[Dict[str, Any]]] = Field(
        None,
        description=(
            "The parameter combinations the search will score, when "
            "include_candidates was set and the method enumerates them. "
            "None otherwise -- which is not 'no candidates': n_candidates is "
            "the count in every case."
        ),
    )
    warnings: List[str] = Field(default_factory=list)


# ── preview_sample_weights ──────────────────────────────────────────────


class PreviewSampleWeightsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="A dataset built by build_model_dataset.")
    weighting: WeightingSpec = Field(
        ...,
        description="The same WeightingSpec a ModelSpec carries: the method "
        "and, for the decay, the half-life in calendar DAYS.",
    )
    target: Optional[str] = Field(
        None,
        description=(
            "Which label to weight against, for a dataset registered with "
            "several. The label decides which rows exist and how far each "
            "row's label window reaches, so uniqueness weights differ "
            "between horizons on the same panel."
        ),
    )


class PreviewSampleWeightsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    half_life_days: Optional[float] = Field(
        None,
        description="Read only by the decaying methods; None for the others, "
        "rather than the spec's unused default.",
    )
    n_rows: int = Field(
        ...,
        description=(
            "Rows summarized: the WHOLE panel under the chosen label. The "
            "engine builds these weights per training fold, so a fold's own "
            "spread is narrower than this one -- the shape of the "
            "distribution and the ratio below are what carry over, not the "
            "row count."
        ),
    )
    min: Stat = None
    p05: Stat = None
    p25: Stat = None
    median: Stat = None
    p75: Stat = None
    p95: Stat = None
    max: Stat = None
    mean: Stat = Field(
        None,
        description="1.0 by construction: the weights are normalized so that "
        "switching weighting on does not also rescale the effective "
        "regularization strength.",
    )
    std: Stat = None
    ratio_max_min: Stat = Field(
        None,
        description="How many times more the heaviest row counts than the "
        "lightest. The single number that says whether this weighting is a "
        "correction or a re-selection of the sample.",
    )
    effective_sample_size_kish: Stat = Field(
        None,
        description=(
            "sum(w)^2 / sum(w^2): the rows this weighting leaves, counted by "
            "the DISPERSION of the weights. Equals n_rows for flat weights "
            "and falls as they spread."
        ),
    )
    effective_sample_size: Stat = Field(
        None,
        description=(
            "The overlap-based count reported beside every out-of-sample "
            "metric: rows discounted for the fact that a horizon-bar forward "
            "return generated every bar produces labels sharing horizon-1 of "
            "their bars. A DIFFERENT quantity from the Kish number above -- "
            "that one measures weight dispersion and this one measures label "
            "redundancy -- and the two are not comparable. None when the "
            "label's horizon cannot be read from the dataset."
        ),
    )
    weight_share_newest_decile: Stat = Field(
        None,
        description="Share of total weight carried by the newest 10% of "
        "dates. Under a short half-life this approaches 1, at which point "
        "the fit is on the recent window and the rest of the panel is "
        "paying for storage.",
    )
    n_zero_weight: int = Field(
        0,
        description="Rows the weighting removes from the fit entirely.",
    )
    warnings: List[str] = Field(default_factory=list)


# ── preview_preprocessing ───────────────────────────────────────────────


class PreviewPreprocessingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(..., description="A dataset built by build_model_dataset.")
    preprocessing: PreprocessingSpec = Field(
        ...,
        description="The same PreprocessingSpec a ModelSpec carries: an "
        "explicit `steps` pipeline, or the `normalization` scheme that "
        "resolves to one. The resolved pipeline is what runs here.",
    )
    sample_rows: int = Field(
        5000,
        gt=0,
        le=1_000_000,
        description="Rows to fit and apply on, taken from the END of the "
        "panel and trimmed to whole dates. The transform's shape does not "
        "need the whole panel, and a preview that costs what a fold costs "
        "is not a preview.",
    )
    split_fraction: float = Field(
        0.7,
        gt=0.0,
        lt=1.0,
        description=(
            "Where the sample is cut into the rows the pipeline is FITTED on "
            "and the rows the fitted state is APPLIED to. Split by DATE, "
            "never by row: a random row split would put the same date on "
            "both sides, and every statistic fitted here would already have "
            "seen the rows it is applied to."
        ),
    )


class ColumnStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: Stat = None
    std: Stat = None
    min: Stat = None
    max: Stat = None
    n_missing: int = 0


class StepPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="The step's defaults with the spec's overrides merged "
        "on: what actually ran, not what was typed.",
    )
    stateless: bool = Field(
        ...,
        description="The step fits nothing, so nothing crosses the fold "
        "boundary and nothing is persisted with the model.",
    )
    column_wise: bool = Field(
        ...,
        description="Each output column depends on that input column alone. "
        "False means a feature cannot be dropped from a fitted matrix "
        "without refitting the pipeline.",
    )
    n_columns_in: int
    n_columns_out: int
    columns_added: List[str] = Field(default_factory=list)
    columns_removed: List[str] = Field(default_factory=list)


class PreviewPreprocessingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: List[StepPreview] = Field(
        default_factory=list,
        description="The resolved pipeline, in order, each step with the "
        "width it received and the width it produced.",
    )
    n_columns_in: int
    n_columns_out: int
    output_columns: List[str] = Field(
        default_factory=list,
        description="The columns the estimator would see, and the names any "
        "importance report would be labelled with. Truncated on a wide "
        "pipeline; the truncation is stated in warnings.",
    )
    n_rows_sampled: int = 0
    n_train_rows: int = 0
    n_test_rows: int = 0
    train_end: Optional[str] = Field(
        None, description="Last date the pipeline was fitted on."
    )
    test_start: Optional[str] = Field(
        None,
        description="First date the fitted state was applied to. Never "
        "earlier than train_end -- that is what splitting by date means.",
    )
    per_column_before: Dict[str, ColumnStats] = Field(
        default_factory=dict,
        description="Per-column mean, dispersion, range and missing count on "
        "the sampled rows BEFORE the pipeline. Bounded in width; the "
        "truncation is stated in warnings.",
    )
    per_column_after: Dict[str, ColumnStats] = Field(
        default_factory=dict,
        description="The same statistics on the pipeline's output, so a step "
        "that did nothing -- or that flattened a column to a constant -- is "
        "visible without a fit.",
    )
    n_nan_after: int = Field(
        0,
        description="Missing values left in the output. Anything above zero "
        "is refused by the engine before it fits, so it is a failure "
        "discovered here rather than at the first fold.",
    )
    explained_variance_ratio: Optional[List[float]] = Field(
        None,
        description="Present when the pipeline contains a pca_whiten step: "
        "the share of the training rows' variance each retained component "
        "carries. None otherwise.",
    )
    warnings: List[str] = Field(default_factory=list)


__all__ = [
    "ColumnStats",
    "FoldPlanDescription",
    "PlanModelExperimentInput",
    "PlanModelExperimentResult",
    "PreviewPreprocessingInput",
    "PreviewPreprocessingResult",
    "PreviewSampleWeightsInput",
    "PreviewSampleWeightsResult",
    "StepPreview",
]
