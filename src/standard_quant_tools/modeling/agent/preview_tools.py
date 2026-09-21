"""
Three dry runs: the schedule, the weights, the transform.

WHAT THESE HAVE IN COMMON. Each answers a question the library already
answers internally, on every experiment, and then throws away before
anything reaches the caller. The cost of a spec is decided by a planner
before the first fit and reported as one integer on a fake date axis. The
sample weights are built per fold, multiplied into the fit, and never
summarized. The preprocessing pipeline is fitted per fold and its two
sharpest failures -- a PCA wider than the panel and an indicator step that
doubles the width -- are discoverable only by hitting them. In all three
cases the answer is cheap, the question is decision-changing, and the only
way to ask it was to spend the experiment.

WHAT A DRY RUN MUST NOT DO. Refuse. `plan_model_experiment` reports an
over-budget plan with the count and the remedy rather than raising:
refusing a preview of a cost is refusing to answer the question that was
asked, and `run_model_experiment` still refuses the same spec.
`preview_sample_weights` summarizes `method='none'` as the flat weights it
is rather than declining to describe them. The refusals that DO come
through are the library's own and are unchanged -- uniqueness weighting on
a panel with no label end dates, a PCA on a panel narrower than its
component count, a PCA on a panel with holes -- because a preview whose
refusals differ from the engine's is a preview of a different pipeline.

See the CHANGELOG entry of 2026-09-21 for what each of these was before.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from ..plan import plan_experiment
from ..preprocessing import FoldContext, build_step, fit_and_apply_pipeline
from ..preprocessing.registry import get_preprocessor
from ..samples import SampleIndex
from ..validation.metrics import effective_sample_size
from ..validation.search import search_candidates
from ..validation.weights import build_sample_weights
from .preview_models import (
    ColumnStats,
    FoldPlanDescription,
    PlanModelExperimentInput,
    PlanModelExperimentResult,
    PreviewPreprocessingInput,
    PreviewPreprocessingResult,
    PreviewSampleWeightsInput,
    PreviewSampleWeightsResult,
    StepPreview,
)

logger = logging.getLogger(__name__)

#: How many columns the per-column blocks describe, and how many output
#: names are listed. A missing_indicator on a 200-feature panel produces
#: 400 columns, and a result that carried them all would be mostly
#: column names.
_MAX_COLUMNS_DESCRIBED = 24
_MAX_OUTPUT_COLUMNS = 64
_MAX_NAMES_PER_STEP = 24


# ── plan_model_experiment ───────────────────────────────────────────────

PLAN_MODEL_EXPERIMENT_DESCRIPTION = (
    "The schedule run_model_experiment would execute, on the real panel, "
    "before it executes any of it: every fold with its train and test "
    "dates, the training ROWS the label-overlap purge removes from each, "
    "the inner folds each training window can actually support, the "
    "candidates the search will score, and the estimator fits all of that "
    "implies against the spec's own budget.max_fits. validate_model_spec "
    "answers the cost question with one integer computed on a date COUNT "
    "and no panel, so it cannot see the purge, cannot see a training "
    "window too short for its inner search, and reports n_purged as null "
    "on every fold; this runs the same planner the engine runs, with the "
    "panel, so the numbers are the numbers that will run. Three things it "
    "shows that nothing else does: fits_per_fit, the multiplier that turns "
    "three quantiles and five conformal blocks into nine fits per fold; a "
    "fold with n_inner_folds=0, which silently costs one fit and deploys "
    "the spec's base parameters rather than searched ones; and, with "
    "include_candidates, the actual parameter combinations -- where a "
    "log-spacing mistake is visible before 720 fits are spent on it. An "
    "over-budget plan is REPORTED here (within_budget=false plus the "
    "refusal text as a warning), not refused: refusing a dry run of a cost "
    "would defeat it, and run_model_experiment still refuses. Fits "
    "nothing."
)


def plan_model_experiment(
    input_data: PlanModelExperimentInput,
) -> PlanModelExperimentResult:
    """What the experiment will cost, counted before it is spent."""
    from .tools import _load_dataset_panel, _select_target

    # The verifying loader, not a bare artifact read: every row count,
    # purge count and content hash below describes THIS panel, and an
    # edited panel.parquet would be planned over as though it were the
    # dataset that was built.
    panel, meta, _directory = _load_dataset_panel(input_data.dataset_id)
    panel, _target_id, notes = _select_target(
        panel, meta, input_data.target, input_data.dataset_id
    )
    warnings: List[str] = list(notes)

    # The same three arguments the engine passes: its own date axis, the
    # dataset's recorded content hash, and the recorded feature ids. The
    # fold hashes are a function of all three, so anything else here would
    # produce hashes that no run will ever reproduce.
    dates = pd.Index(sorted(panel["date"].unique()))
    plan = plan_experiment(
        input_data.spec,
        dates,
        panel=panel,
        dataset_hash=meta.get("data_hash"),
        feature_ids=list(meta.get("feature_ids") or []),
    )

    if not plan.within_budget:
        try:
            plan.refuse_over_budget("plan_model_experiment")
        except ValidationError as exc:
            warnings.append(
                f"{exc} This is a dry run, so the plan is reported rather "
                "than refused; run_model_experiment refuses it."
            )

    search = input_data.spec.search
    if search is not None:
        starved = [fold.index for fold in plan.folds if fold.n_inner_folds == 0]
        if starved:
            shown = starved[:8]
            warnings.append(
                f"{len(starved)} of {len(plan.folds)} fold(s) have "
                f"n_inner_folds=0 (fold(s) {shown}"
                f"{', ...' if len(starved) > len(shown) else ''}): the "
                "training window that survives the purge is too short for "
                f"search.inner_splits={search.inner_splits}, so the search "
                "does not run on them. Each is priced at fits_per_fit=1 fit "
                "and uses the spec's BASE parameters -- a cheaper plan that "
                "validated nothing the search was asked to choose. Lengthen "
                "validation.train_window, or lower inner_splits."
            )

    candidates: Optional[List[Dict[str, Any]]] = None
    if input_data.include_candidates:
        if search is None:
            warnings.append(
                "include_candidates was set on a spec with no `search`: "
                "there are no candidates to enumerate, and every fold fits "
                "the estimator's declared parameters once."
            )
        elif search.method == "tpe":
            warnings.append(
                "search.method='tpe' samples its candidates one at a time "
                "from what the previous ones scored, so there is no list to "
                "return; `candidates` is null. The count is "
                f"n_candidates={plan.n_candidates}, which is "
                "n_search_candidates of this spec -- its max_trials -- and "
                "it is what the fit count above was computed from."
            )
        else:
            candidates = [
                dict(candidate)
                for candidate in search_candidates(search, input_data.spec.random_seed)
            ]

    payload = plan.to_dict()
    fold_dicts = payload.pop("folds")
    logger.debug(
        "[plan_model_experiment] dataset_id=%s  folds=%d  fits=%d/%d",
        input_data.dataset_id,
        len(fold_dicts),
        plan.n_fits,
        plan.max_fits,
    )
    return PlanModelExperimentResult(
        **payload,
        has_panel=plan.has_panel,
        final_search_hash=plan.final_search_hash,
        folds=(
            [FoldPlanDescription(**fold) for fold in fold_dicts]
            if input_data.include_folds
            else []
        ),
        candidates=candidates,
        warnings=warnings,
    )


# ── preview_sample_weights ──────────────────────────────────────────────

PREVIEW_SAMPLE_WEIGHTS_DESCRIPTION = (
    "What a WeightingSpec would actually do to the training rows, before "
    "a model is fitted under it. The weighting method is selectable on "
    "every ModelSpec, is applied inside the engine, and its distribution "
    "is reported nowhere -- so choosing a half_life_days is choosing "
    "blind. Returns the percentiles of the weights, the ratio of the "
    "heaviest row to the lightest, the share of total weight sitting on "
    "the newest tenth of the dates, and two effective sample sizes that "
    "measure different things: the Kish size sum(w)^2/sum(w^2), which is "
    "these weights' own dispersion, beside the overlap-based count that "
    "every out-of-sample metric is already reported against. A weighting "
    "whose max/min is 30 is not a correction, it is a re-selection of the "
    "sample under another name, and that is visible here and in no result "
    "afterwards. method='none' is summarized as the flat weights it is "
    "rather than refused. Uniqueness weighting on a panel that carries no "
    "label end dates gets the library's own refusal, unchanged, which is "
    "the point of asking here first. Fits nothing."
)


def _percentiles(weights: np.ndarray) -> Dict[str, float]:
    keys = ("min", "p05", "p25", "median", "p75", "p95", "max")
    values = np.percentile(weights, [0.0, 5.0, 25.0, 50.0, 75.0, 95.0, 100.0])
    return {key: float(value) for key, value in zip(keys, values)}


def preview_sample_weights(
    input_data: PreviewSampleWeightsInput,
) -> PreviewSampleWeightsResult:
    """The distribution of the training-row weights a spec implies."""
    from ..engine import _target_horizon
    from .tools import _load_dataset_panel, _select_target

    panel, meta, _directory = _load_dataset_panel(input_data.dataset_id)
    panel, target_id, notes = _select_target(
        panel, meta, input_data.target, input_data.dataset_id
    )
    warnings: List[str] = list(notes)
    if panel.empty:
        raise ValidationError(
            f"dataset {input_data.dataset_id!r} has no rows under the chosen "
            "label, so there are no training weights to describe."
        )

    method = input_data.weighting.method
    decays = method in ("time_decay", "uniqueness_and_time_decay")
    half_life = float(input_data.weighting.half_life_days)

    # The same three arrays the engine hands the builder, off the same
    # index type -- dates, label ends and entities in row order -- so the
    # refusal for a panel with no label end dates is raised here exactly
    # where it would be raised there.
    index = SampleIndex.from_frame(panel)
    weights = build_sample_weights(
        method, index.dates, index.label_end, index.entities, half_life
    )
    if weights is None:
        # 'none' is not an absent answer: it is every row at weight 1, and
        # saying so is what lets the caller compare it against the spread
        # of a weighting they are considering.
        weights = np.ones(len(panel), dtype=np.float64)
        warnings.append(
            "weighting.method='none': every row enters the fit at weight 1, "
            "so the summary below is flat by construction. Run this again "
            "with 'label_uniqueness' or 'time_decay' to see what turning "
            "one on would do to the same panel."
        )

    n_rows = int(weights.size)
    total = float(weights.sum())
    stats = _percentiles(weights)
    ratio = stats["max"] / stats["min"] if stats["min"] > 0 else None
    kish = (total * total / float((weights**2).sum())) if total > 0 else None

    # The newest tenth of the DATE axis, not of the rows: an unbalanced
    # panel has more rows on recent dates, and a share computed over rows
    # would report the panel's shape as the weighting's effect.
    unique_dates = np.unique(index.dates)
    newest_count = max(1, int(round(0.1 * unique_dates.size)))
    newest = np.isin(index.dates, unique_dates[-newest_count:])
    share = float(weights[newest].sum() / total) if total > 0 else None

    horizon = _target_horizon(target_id)
    n_entities = int(pd.Series(index.entities).nunique())
    overlap_ess = (
        float(effective_sample_size(n_rows, horizon, n_entities))
        if horizon is not None
        else None
    )
    warnings.append(
        "effective_sample_size_kish and effective_sample_size are not two "
        "estimates of one quantity. The first counts the rows these WEIGHTS "
        "leave, from their dispersion alone; the second counts the "
        "independent observations the LABELS leave, from the overlap a "
        f"{horizon if horizon is not None else 'h'}-bar forward return "
        "generated every bar creates. A weighting that fixes the second is "
        "still measured by the first, and neither bounds the other."
    )

    if ratio is not None and ratio > 10:
        warnings.append(
            f"the heaviest row counts {ratio:,.1f} times the lightest. At "
            "this spread the fit is dominated by a minority of the rows, "
            "which is a re-selection of the sample rather than a "
            "correction to it -- compare effective_sample_size_kish "
            f"({kish:,.0f} of {n_rows:,} rows) against what you believe you "
            "are training on."
        )

    span_days = 0.0
    if unique_dates.size:
        span_days = float(
            (pd.Timestamp(unique_dates[-1]) - pd.Timestamp(unique_dates[0])).days
        )
    if decays and span_days > 0 and half_life < span_days / 10.0:
        warnings.append(
            f"half_life_days={half_life:g} is shorter than a tenth of this "
            f"panel's {span_days:,.0f}-day span, so the oldest rows enter "
            f"the fit at roughly 2^-{span_days / half_life:.0f} of the "
            "newest ones. The fit is effectively on the recent window; "
            "shorten the dataset instead if that is the intent, so the "
            "validation folds are cut from the window being trained on."
        )

    logger.debug(
        "[preview_sample_weights] dataset_id=%s  method=%s  rows=%d",
        input_data.dataset_id,
        method,
        n_rows,
    )
    return PreviewSampleWeightsResult(
        method=method,
        half_life_days=half_life if decays else None,
        n_rows=n_rows,
        mean=float(weights.mean()) if n_rows else None,
        std=float(pd.Series(weights).std()) if n_rows > 1 else None,
        ratio_max_min=ratio,
        effective_sample_size_kish=kish,
        effective_sample_size=overlap_ess,
        weight_share_newest_decile=share,
        n_zero_weight=int((weights == 0.0).sum()),
        warnings=warnings,
        **stats,
    )


# ── preview_preprocessing ───────────────────────────────────────────────

PREVIEW_PREPROCESSING_DESCRIPTION = (
    "Fit a preprocessing pipeline on a sample of a built dataset and "
    "report what it does to the columns, before an experiment runs it on "
    "every fold. Two of the eight registered steps carry traps that are "
    "discoverable only at fit time: pca_whiten refuses a panel with any "
    "missing value AND refuses n_components greater than the column count, "
    "at its own default of 8 -- so it raises on any dataset narrower than "
    "eight features -- and missing_indicator appends one <column>__missing "
    "per column, doubling the width by design. Both refusals and both "
    "widths appear here, on the real panel, for the cost of one fit of the "
    "pipeline and no fit of an estimator. Reports each step's width in and "
    "out with the columns it added and removed, the per-column statistics "
    "before and after, the missing values left (which the engine refuses "
    "before it fits), and a pca_whiten step's explained variance ratio. "
    "The sample is split BY DATE, never by row: the statistics are fitted "
    "on the earlier rows and applied to the later ones, the way a fold "
    "does it. Warns when the output columns are no longer feature names, "
    "because an importance report is then labelled pc1..pcK, and when a "
    "step is not column-wise, because run_feature_ablation must refit "
    "rather than project. Fits no estimator."
)


def _column_stats(frame: pd.DataFrame, limit: int) -> Dict[str, ColumnStats]:
    """Per-column statistics for the first `limit` columns."""
    out: Dict[str, ColumnStats] = {}
    for column in list(frame.columns)[:limit]:
        series = frame[column]
        out[str(column)] = ColumnStats(
            mean=float(series.mean()),
            std=float(series.std()),
            min=float(series.min()),
            max=float(series.max()),
            n_missing=int(series.isna().sum()),
        )
    return out


def _date_sample(
    panel: pd.DataFrame, dataset_id: str, sample_rows: int
) -> Tuple[pd.DataFrame, bool, int]:
    """
    The most recent `sample_rows` rows, trimmed to WHOLE dates.

    A partial date would put some of a cross-section in the sample and the
    rest outside it, which is exactly what a cross-sectional step reads --
    so the preview would describe a transform of a cross-section that does
    not exist.
    """
    ordered = panel.sort_values(["date", "entity"], kind="stable").reset_index(
        drop=True
    )
    n_total = int(len(ordered))
    if n_total <= sample_rows:
        return ordered, False, n_total

    trimmed = ordered.iloc[n_total - sample_rows :]
    first = trimmed["date"].iloc[0]
    if int((ordered["date"] == first).sum()) != int((trimmed["date"] == first).sum()):
        trimmed = trimmed[trimmed["date"] != first]
    if trimmed.empty:
        widest = int(ordered["date"].value_counts().max())
        raise ValidationError(
            f"sample_rows={sample_rows} is smaller than one date's "
            f"cross-section in dataset {dataset_id!r} ({widest} rows), and a "
            "partial date cannot be previewed -- a cross-sectional step "
            "would be fitted on part of a cross-section. Pass "
            f"sample_rows>={widest * 2}."
        )
    return trimmed.reset_index(drop=True), True, n_total


def _split_by_date(
    frame: pd.DataFrame, dataset_id: str, split_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Earlier dates to fit on, later dates to apply to."""
    unique_dates = np.array(sorted(frame["date"].unique()))
    if unique_dates.size < 2:
        raise ValidationError(
            f"dataset {dataset_id!r} has {unique_dates.size} date(s) in the "
            "previewed sample, and a pipeline fitted on training rows and "
            "applied to later rows needs at least two. Raise sample_rows, or "
            "preview a dataset with a longer date axis."
        )
    n_train_dates = int(round(unique_dates.size * split_fraction))
    n_train_dates = min(max(n_train_dates, 1), unique_dates.size - 1)
    boundary = unique_dates[n_train_dates - 1]
    train = frame[frame["date"] <= boundary]
    test = frame[frame["date"] > boundary]
    return train, test, unique_dates


def _step_previews(
    state: Dict[str, Any], train_features: pd.DataFrame, train_ctx: FoldContext
) -> List[StepPreview]:
    """
    Each step's width in and out, by replaying the FITTED state one step
    at a time.

    Replayed from the state rather than fitted again: the default pooled
    pair takes the fused native path inside the pipeline, which fits the
    two steps in one pass, and a second independent fit would describe a
    transform the engine did not run.
    """
    previews: List[StepPreview] = []
    current = train_features
    for entry in state.get("steps") or []:
        step_type = str(entry["type"])
        params = dict(entry.get("params") or {})
        definition = get_preprocessor(step_type)
        before = list(current.columns)
        current = build_step(step_type, params).transform(
            current, entry.get("state") or {}, train_ctx
        )
        after = list(current.columns)
        previews.append(
            StepPreview(
                type=step_type,
                params=params,
                stateless=definition.stateless,
                column_wise=definition.column_wise,
                n_columns_in=len(before),
                n_columns_out=len(after),
                columns_added=[str(c) for c in after if c not in set(before)][
                    :_MAX_NAMES_PER_STEP
                ],
                columns_removed=[str(c) for c in before if c not in set(after)][
                    :_MAX_NAMES_PER_STEP
                ],
            )
        )
    return previews


def _preprocessing_warnings(
    previews: Sequence[StepPreview],
    output_columns: List[str],
    n_columns_out: int,
    n_nan_after: int,
    n_columns_in: int,
) -> List[str]:
    warnings: List[str] = []
    for preview in previews:
        if (
            preview.type == "missing_indicator"
            and preview.n_columns_out == 2 * preview.n_columns_in
        ):
            warnings.append(
                f"missing_indicator doubled the width from "
                f"{preview.n_columns_in} to {preview.n_columns_out} columns: "
                "one <column>__missing indicator per input column, for every "
                "column and not only the ones that are actually missing, "
                "because a column set that depended on the fold could not be "
                "applied at scoring. On a panel that alignment already made "
                "complete every indicator is zero and the width is the whole "
                "of the cost."
            )
    not_column_wise = sorted({p.type for p in previews if not p.column_wise})
    if not_column_wise:
        warnings.append(
            f"step(s) {not_column_wise} are not column-wise: every output "
            "column depends on every input column. run_feature_ablation "
            "cannot project a feature subset out of a fitted matrix under "
            "this pipeline and has to refit it per subset, which costs one "
            "pipeline fit per feature dropped."
        )
    # `pc1`, not anything beginning with "pc": a feature legitimately
    # named `pcf_ratio` is not a principal component.
    if any(
        str(column).startswith("pc") and str(column)[2:].isdigit()
        for column in output_columns
    ):
        warnings.append(
            "the estimator's columns are no longer feature names "
            f"({output_columns[:4]}): a component is a combination of every "
            "input feature, so the importance summary, analyze_model_errors' "
            "feature deciles and run_feature_ablation all name components "
            "rather than the features a decision would be made about."
        )
    if n_nan_after:
        warnings.append(
            f"{n_nan_after:,} missing value(s) remain after the pipeline. "
            "run_model_experiment refuses these before it fits, for any "
            "estimator that does not accept them: add an `impute` step "
            "(with `missing_indicator` before it to keep the information), "
            "fit an estimator that accepts missing values, or build the "
            "dataset with missing.policy='drop'."
        )
    if n_columns_out > len(output_columns):
        warnings.append(
            f"output_columns lists the first {len(output_columns)} of "
            f"{n_columns_out} columns."
        )
    if max(n_columns_in, n_columns_out) > _MAX_COLUMNS_DESCRIBED:
        warnings.append(
            f"per_column_before and per_column_after describe the first "
            f"{_MAX_COLUMNS_DESCRIBED} column(s) of each frame."
        )
    return warnings


def preview_preprocessing(
    input_data: PreviewPreprocessingInput,
) -> PreviewPreprocessingResult:
    """Fit the pipeline on a date-split sample and report what it did."""
    from .tools import _load_dataset_panel

    panel, meta, _directory = _load_dataset_panel(input_data.dataset_id)
    feature_columns = [
        column for column in (meta.get("feature_ids") or []) if column in panel.columns
    ]
    if not feature_columns:
        raise ValidationError(
            f"dataset {input_data.dataset_id!r} records no feature columns "
            "that are present in its panel, so there is no pipeline to fit. "
            "Rebuild it with build_model_dataset."
        )

    sample, truncated, n_total = _date_sample(
        panel, input_data.dataset_id, int(input_data.sample_rows)
    )
    train, test, _unique_dates = _split_by_date(
        sample, input_data.dataset_id, float(input_data.split_fraction)
    )

    # The engine's own call, with the engine's own contexts: the dates and
    # entities of the rows, and never the target. A step that could read
    # the label would be a step that could leak it.
    train_ctx = FoldContext.from_frame(train)
    state, train_out, test_out = fit_and_apply_pipeline(
        input_data.preprocessing.resolved_steps,
        train[feature_columns],
        test[feature_columns],
        train_ctx,
        FoldContext.from_frame(test),
    )
    previews = _step_previews(state, train[feature_columns], train_ctx)

    before_frame = sample[feature_columns]
    after_frame = pd.concat([train_out, test_out], axis=0)
    output_columns = [str(c) for c in train_out.columns]
    n_columns_out = len(output_columns)
    n_nan_after = int(after_frame.isna().to_numpy().sum())

    explained: Optional[List[float]] = None
    for entry in state.get("steps") or []:
        if entry["type"] == "pca_whiten":
            ratios = (entry.get("state") or {}).get("explained_variance_ratio")
            if ratios is not None:
                explained = [float(v) for v in ratios]

    warnings = _preprocessing_warnings(
        previews,
        output_columns[:_MAX_OUTPUT_COLUMNS],
        n_columns_out,
        n_nan_after,
        len(feature_columns),
    )
    if truncated:
        warnings.insert(
            0,
            f"previewed on the most recent {len(sample):,} of {n_total:,} "
            "rows, trimmed to whole dates. The column widths and the step "
            "sequence are exact; the per-column statistics describe the "
            "sample, and a fold earlier in the panel can differ. Raise "
            "sample_rows to widen it.",
        )
    if explained is not None and explained:
        retained = float(math.fsum(explained))
        warnings.append(
            f"pca_whiten keeps {len(explained)} component(s) carrying "
            f"{retained:.1%} of the training rows' variance. The rest is "
            "discarded before the estimator sees it."
        )

    logger.debug(
        "[preview_preprocessing] dataset_id=%s  steps=%s  %d -> %d columns",
        input_data.dataset_id,
        [p.type for p in previews],
        len(feature_columns),
        n_columns_out,
    )
    return PreviewPreprocessingResult(
        steps=previews,
        n_columns_in=len(feature_columns),
        n_columns_out=n_columns_out,
        output_columns=output_columns[:_MAX_OUTPUT_COLUMNS],
        n_rows_sampled=int(len(sample)),
        n_train_rows=int(len(train)),
        n_test_rows=int(len(test)),
        train_end=str(pd.Timestamp(train["date"].max()).date()),
        test_start=str(pd.Timestamp(test["date"].min()).date()),
        per_column_before=_column_stats(before_frame, _MAX_COLUMNS_DESCRIBED),
        per_column_after=_column_stats(after_frame, _MAX_COLUMNS_DESCRIBED),
        n_nan_after=n_nan_after,
        explained_variance_ratio=explained,
        warnings=warnings,
    )


__all__ = [
    "PLAN_MODEL_EXPERIMENT_DESCRIPTION",
    "PREVIEW_PREPROCESSING_DESCRIPTION",
    "PREVIEW_SAMPLE_WEIGHTS_DESCRIPTION",
    "PlanModelExperimentInput",
    "PlanModelExperimentResult",
    "PreviewPreprocessingInput",
    "PreviewPreprocessingResult",
    "PreviewSampleWeightsInput",
    "PreviewSampleWeightsResult",
    "plan_model_experiment",
    "preview_preprocessing",
    "preview_sample_weights",
]
