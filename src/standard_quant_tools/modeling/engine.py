"""
run_experiment: the ModelSpec executor. One call does
split -> fit-preprocessing-on-train-only -> fit -> walk-forward evaluate
-> refit on all data -> register — structurally impossible to fit
without validation through this function, since there is no separate
"just fit" entry point in the modeling agent surface.

Only estimators in estimators.registry.ESTIMATOR_REGISTRY can be used —
no arbitrary sklearn import, no exec(). Preprocessing (features/transforms.py's
winsorize + zscore) is fit on each fold's training rows only and applied
unchanged to that fold's test rows, per validation/walk_forward.py's
leakage discipline.
"""

import inspect
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from standard_quant_tools._blas import openmp_thread_limit, single_threaded_blas
from standard_quant_tools.error import ValidationError

from . import artifacts as _artifacts
from .adapters import _exposes_coefficients, accepts_missing, get_adapter
from .cache import FoldCache, column_wise_pipeline
from .dataset.alignment import LABEL_END_COL
from .estimators.registry import (
    estimator_cost,
    get_estimator_class,
    quantile_estimators,
    quantile_support,
    validate_params,
)
from .monitoring import feature_profile, reference_sample
from .plan import plan_experiment
from .preprocessing import (
    FoldContext,
    fit_and_apply_pipeline,
    fit_pipeline,
    legacy_stats,
    step_types,
)
from .registry.model_registry import new_model_id, save_model
from .samples import SampleIndex, datetime_values
from .specs import TASKS, ModelSpec, targets_for_task
from .validation.comparison import (
    COSINE_FREQUENCY_SCALE,
    HORIZONS_PER_FREQUENCY,
    MIN_HEADLINE_DATES,
    mean_vs_null_test,
)
from .validation.conformal import conformal_radius, held_out_residuals
from .validation.diagnostics import fold_feature_importance, summarize_importance
from .validation.distributional import distributional_metrics, quantile_column
from .validation.metrics import (
    aggregate_cross_sectional_ic,
    average_fold_metrics,
    classification_metrics,
    cross_sectional_ic,
    effective_sample_size_report,
    label_cross_sectional_corr,
    positive_class_proba,
    regression_metrics,
    summarize_cross_sectional_ic,
)
from .validation.ranking import (
    fold_ic_series,
    group_sizes,
    ranking_metrics,
    relevance_grades,
)
from .validation.search import (
    grid_duplicates_warning,
    inner_fold_count,
    n_search_candidates,
    require_optuna,
    search_best_params,
    search_pool_workers,
)
from .validation.survival import EVENT_COL, survival_labels
from .validation.walk_forward import build_splitter, contiguous_runs
from .validation.weights import build_sample_weights


def _target_horizon(target_id: "str | None") -> "int | None":
    """`target_id` is built as "<type>:<horizon>" (dataset/builder.py), so
    the horizon is recoverable without threading DatasetSpec through the
    engine. Returns None for a malformed or absent id rather than raising —
    a missing horizon only costs the overlap adjustment."""
    if not target_id or ":" not in target_id:
        return None
    try:
        return int(target_id.rsplit(":", 1)[1])
    except ValueError:
        return None


def refuse_duplicate_entity_dates(panel: pd.DataFrame, where: str) -> None:
    """
    Refuse a panel in which an (entity, date) pair appears more than once,
    naming the count and a sample.

    A repeated row is not a second observation. Every fold would train on
    it twice and test on it twice, so `n_oos_rows` and the effective sample
    size double while the metrics barely move -- a panel written out twice
    reported 960 out-of-sample rows for 480. The predictions frame the
    bridge reads has always been refused on the same condition.
    """
    duplicated = panel.duplicated(subset=["entity", "date"])
    if duplicated.any():
        sample = panel.loc[duplicated, ["entity", "date"]].head(3)
        raise ValidationError(
            f"{where}: {int(duplicated.sum())} duplicate (entity, date) row(s), "
            f"e.g. {sample.to_dict('records')}. Each pair must be unique -- a "
            "repeated row is trained on and tested on twice, which multiplies "
            "the out-of-sample row count and the effective sample size without "
            "adding evidence. Drop the repeats (keep one row per entity and "
            "date) and register or build the panel again."
        )


#: How `validation_report["purge"]` names the label end the purge ran on.
PURGE_ON_LABEL_END = "label_end"
PURGE_ON_DERIVED_LABEL_END = "label_end_derived_from_horizon"


def panel_with_label_end(
    panel: pd.DataFrame, target_id: "str | None", where: str
) -> "tuple[pd.DataFrame, str, List[str]]":
    """
    The panel with a `label_end_date` column the purge can read, how that
    column came to be, and the warning that says so.

    A panel built by `build_dataset` or registered through
    `register_external_panel` carries the column, and is returned as it
    came. One without it -- a dataset dict assembled by hand, or a dataset
    persisted before the column existed -- used to skip the purge with
    nothing but `purge="not_applicable"` to show for it, although its
    `target_id` named the horizon the end can be derived from. The end is
    derived the way the registration derives it: the date `horizon` rows
    ahead on the entity's own dates, NaT where the panel ends first. Exact
    for a fixed-horizon label, and a superset for one that can end early,
    so it purges at least the rows the true end would.

    With no horizon either there is no end to purge on, and the panel is
    refused rather than validated on training rows whose labels may reach
    into the test window.
    """
    if LABEL_END_COL in panel.columns:
        return panel, PURGE_ON_LABEL_END, []
    horizon = _target_horizon(target_id)
    if horizon is None or horizon < 0:
        raise ValidationError(
            f"{where}: the panel has no {LABEL_END_COL!r} column and its "
            f"target_id {target_id!r} names no horizon, so the purge has no "
            "label end to run on: training rows whose labels reach into the "
            "test window would be kept, and the embargo alone does not "
            f"separate them. Add a {LABEL_END_COL!r} column (the date each "
            "row's label is finished), or give target_id as "
            "'<type>:<horizon>' so the end can be derived from the horizon."
        )
    from .dataset.external_panel import _label_end_from_horizon

    derived = panel.assign(
        **{LABEL_END_COL: _label_end_from_horizon(panel, int(horizon)).to_numpy()}
    )
    warning = (
        f"the panel carries no {LABEL_END_COL!r} column, so each row's label "
        f"end was derived from target_id {target_id!r}: the date {horizon} "
        "row(s) ahead on the entity's own dates, NaT where the panel ends "
        "first. The label-overlap purge and label-uniqueness weighting read "
        "those ends (a run reports validation_report.purge="
        f"{PURGE_ON_DERIVED_LABEL_END!r}). Exact for a fixed-horizon label; "
        "for a label that can end early it purges a superset of the "
        "overlapping rows. Record the column to use the true ends."
    )
    return derived, PURGE_ON_DERIVED_LABEL_END, [warning]


def _oos_effective_sample_size(
    panel: pd.DataFrame,
    tested_rows: np.ndarray,
    n_oos_rows: int,
    n_oos_dates: int,
    horizon: "int | None",
) -> Dict[str, Any]:
    """
    The effective sample size behind the out-of-sample metrics, with the
    inputs and bounds it was computed from.

    The labels' cross-sectional correlation is measured on the rows that
    were tested, because those are the rows the metrics are averaged
    over; see `validation.metrics.effective_sample_size` for why it, and
    not only the overlap along time, decides how many of those rows are
    independent.
    """
    tested = panel.loc[tested_rows, ["date", "entity", "target"]]
    labels = pd.to_numeric(tested["target"], errors="coerce").to_numpy(dtype=float)
    rho = label_cross_sectional_corr(
        datetime_values(tested["date"]), tested["entity"].to_numpy(), labels
    )
    return effective_sample_size_report(n_oos_rows, horizon, n_oos_dates, rho)


def _calibrated(estimator, model_spec, n_rows: int):
    """
    Wrap a classifier so its probabilities mean what they say.

    Returns the estimator unchanged unless calibration was asked for, which
    keeps this a no-op for every existing spec.

    THREE THINGS THIS REFUSES TO DO SILENTLY.

    Calibrating a regressor is meaningless -- there are no probabilities to
    map -- so a spec that asks for it is a mistake worth naming rather than
    a request to ignore.

    Calibrating with more folds than rows will support cannot work, and
    sklearn's own error for it arrives from three frames down talking about
    `n_splits`. The estimator needs enough rows per fold to fit at all.

    And the calibration map is fitted on HELD-OUT folds inside the training
    window, never on the rows the estimator itself trained on. Fitting it
    there would calibrate against memorized labels and report a confidence
    nobody has, which is the same failure as scoring a model on its training
    set and one layer more obscure.
    """
    method = getattr(model_spec.estimator, "calibration", "none")
    if method == "none":
        return estimator

    if model_spec.task != "classification":
        raise ValidationError(
            f"estimator.calibration={method!r} was requested for a "
            f"{model_spec.task!r} task. Calibration maps scores onto "
            "probabilities and only classification has any. Remove it, or "
            "set task='classification'."
        )

    folds = int(getattr(model_spec.estimator, "calibration_folds", 3))
    if n_rows < folds * 2:
        raise ValidationError(
            f"estimator.calibration needs at least {folds * 2} training rows "
            f"for {folds} folds and this window has {n_rows}. Lower "
            "calibration_folds, widen train_window, or drop calibration -- "
            "an uncalibrated score is honest, and a calibration map fitted "
            "on a handful of rows is not."
        )

    from sklearn.calibration import CalibratedClassifierCV

    return CalibratedClassifierCV(estimator, method=method, cv=folds)


def _calibration_importance_warning(
    model_spec: ModelSpec, estimator_cls: Any
) -> List[str]:
    """
    Said once, at the top of a run that asks for calibration from an
    estimator that would otherwise have reported importances.

    `_calibrated` wraps the estimator in `CalibratedClassifierCV`, which
    exposes neither `coef_` nor `feature_importances_`, so
    `fold_feature_importance` falls through to its NaN branch for every
    fold and `feature_importance_summary` comes back NaN for every feature.
    Measured on identical data: {f0: 0.389, f1: 0.318, f2: 0.293} without
    calibration, {f0: NaN, f1: NaN, f2: NaN} with isotonic.

    Nothing is broken by that and nothing said so. The capability report
    still advertises `exposes_feature_importance: True` -- correctly, it
    describes the UNWRAPPED class, which is what `adapters.py:88-91` says
    the flag must agree with -- so a reader of the two together concludes
    the model failed to fit rather than that the spec traded one output
    for another. One sentence, at the point where the trade was made.
    """
    method = getattr(model_spec.estimator, "calibration", "none")
    if method == "none":
        return []
    if not (
        _exposes_coefficients(estimator_cls)
        or hasattr(estimator_cls, "feature_importances_")
    ):
        # The estimator had no importances to lose -- HistGradientBoosting
        # reports NaN calibrated or not -- so there is nothing to warn
        # about and a warning here would be noise on every such run.
        return []
    return [
        f"estimator.calibration={method!r} wraps "
        f"{estimator_cls.__name__} in CalibratedClassifierCV, which exposes "
        "neither `coef_` nor `feature_importances_`: "
        "`feature_importance_summary` is therefore NaN for every feature on "
        "this run, by construction rather than because the fit failed. The "
        "capability report's `exposes_coefficients` / "
        "`exposes_feature_importance` flags describe the uncalibrated "
        "estimator. Run the same spec with calibration='none' to read the "
        "importances, and keep the calibrated model for anything that "
        "thresholds a probability."
    ]


def _importance_source(per_fold: List[Any]) -> str:
    """
    Where `feature_importance_summary` came from: 'coefficients' when a
    fold's estimator exposed `coef_`, 'feature_importances' when one
    exposed non-negative importances, 'none' when no fold exposed either
    and every number in the summary is NaN.
    """
    if any(fold.signed for fold in per_fold):
        return "coefficients"
    for fold in per_fold:
        if any(math.isfinite(v) for v in fold.values.values()):
            return "feature_importances"
    return "none"


def _importance_note(
    model_spec: ModelSpec,
    estimator_cls: Any,
    summary: Dict[str, Dict[str, float]],
    headline: str,
) -> List[str]:
    """
    One sentence for an uncalibrated run whose estimator has no `coef_`,
    saying which parts of `feature_importance_summary` are null by
    construction and where to measure what a feature is worth instead.

    The predicates are the capability report's own (`_exposes_coefficients`
    and `feature_importances_` on the class), so the sentence and the flags
    it points at cannot disagree; and it is said only when the summary
    really is null where the sentence says, so an estimator that sets its
    importances during the fit is not told it has none. A calibrated run
    has its own warning, `_calibration_importance_warning`.

    A null block without a word read as a failed fit: hist_gradient_boosting
    returned 40 of 40 importance fields null and random_forest 24 of 40.
    """
    if getattr(model_spec.estimator, "calibration", "none") != "none":
        return []
    if not summary or _exposes_coefficients(estimator_cls):
        return []
    entries = list(summary.values())
    name = estimator_cls.__name__
    estimator = model_spec.estimator.type
    pointer = (
        f"run_feature_ablation with metric='{headline}' measures what each "
        "feature is worth to this model by refitting without it."
    )
    if not hasattr(estimator_cls, "feature_importances_"):
        if all(not math.isfinite(e.get("mean", math.nan)) for e in entries):
            return [
                "feature_importance_summary is null for every feature by "
                f"construction, not because the fit failed: {name} exposes "
                "neither coef_ nor feature_importances_, as "
                f"list_modeling_capabilities reports for {estimator} "
                "(exposes_coefficients false, exposes_feature_importance "
                f"false). {pointer}"
            ]
        return []
    signed = ("signed_mean", "signed_std", "sign_consistency")
    if all(
        not math.isfinite(e.get(key, math.nan)) for e in entries for key in signed
    ) and any(math.isfinite(e.get("mean", math.nan)) for e in entries):
        return [
            "feature_importance_summary carries mean and std for every "
            "feature and null signed_mean, signed_std and sign_consistency, "
            f"by construction: {name}'s feature_importances_ are non-negative "
            "and carry no direction, as list_modeling_capabilities reports "
            f"for {estimator} (exposes_coefficients false, "
            "exposes_feature_importance true). mean says how much the fitted "
            "trees relied on a feature, not which way it moves the "
            f"prediction; {pointer}"
        ]
    return []


def _r2_note(
    oos_metrics: Dict[str, float],
    fold_metrics: List[Dict[str, float]],
    fold_weights: List[float],
    headline: str,
) -> List[str]:
    """
    Why r2 sits below its baseline, said whenever it does.

    r2 can be no larger than the squared correlation between prediction and
    label: the best rescaling of the predictions reaches it and any other
    falls short. So the ceiling is each fold's squared pooled `ic`,
    averaged with the weights the folds' r2 are averaged with (their test
    rows). On a cross-sectionally ranked label the baseline is zero by
    construction and that ceiling was 0.002 to 0.005 on a 30-name daily
    equity panel: every model there reported a negative r2 beside a
    positive rank IC, and nothing said which of the two to read.
    """
    r2 = oos_metrics.get("r2")
    baseline = oos_metrics.get("baseline_r2")
    if r2 is None or baseline is None:
        return []
    if not (math.isfinite(r2) and math.isfinite(baseline)) or r2 >= baseline:
        return []
    below = sum(
        1
        for m in fold_metrics
        if math.isfinite(m.get("r2", math.nan))
        and math.isfinite(m.get("baseline_r2", math.nan))
        and m["r2"] < m["baseline_r2"]
    )
    squares = np.array([m.get("ic", math.nan) ** 2 for m in fold_metrics])
    weights = np.asarray(fold_weights, dtype=float)
    usable = np.isfinite(squares) & (weights > 0)
    ceiling = (
        float(np.average(squares[usable], weights=weights[usable]))
        if usable.any()
        else math.nan
    )
    which = "a negative r2" if r2 < 0 else "an r2 below the baseline's"
    bound = (
        f", {ceiling:.4f} averaged over these folds" if math.isfinite(ceiling) else ""
    )
    return [
        f"r2 is {r2:.4f} against baseline_r2 {baseline:.4f}, the "
        "training-fold mean predicted for every row, and below it in "
        f"{below} of {len(fold_metrics)} folds. r2 can be no larger than the "
        "square of each fold's correlation between prediction and label"
        f"{bound}, so {which} says the predictions are more dispersed, or "
        "further from the label's level, than that correlation supports: as "
        "values they did worse than the constant, and as an ordering they "
        f"are measured by {headline}, not by r2."
    ]


def _frequency_reason(n_dates: int, horizon: "int | None") -> str:
    """Why the headline test read the frequencies it did, as a clause."""
    h = int(horizon) if horizon and int(horizon) > 0 else 1
    by_dates = math.floor(COSINE_FREQUENCY_SCALE * float(n_dates) ** (2.0 / 3.0))
    if h > 1 and n_dates // (HORIZONS_PER_FREQUENCY * h) < by_dates:
        return (
            f"at most one per {HORIZONS_PER_FREQUENCY * h} dates, "
            f"{HORIZONS_PER_FREQUENCY} times the {h}-day label horizon"
        )
    return f"{COSINE_FREQUENCY_SCALE:g} x {n_dates:,}^(2/3)"


def _headline_report(
    adapter: Any,
    task: str,
    oos_metrics: Dict[str, float],
    series: "pd.Series | None",
    horizon: "int | None",
    *,
    scope: str = "out-of-sample",
) -> "Tuple[Dict[str, Any], List[str]]":
    """
    The headline metric against what a model with no skill scores, and the
    warning when it does not beat it.

    `scope` names the dates in the warning: a run's are out-of-sample;
    `score_predictions` passes "scored", since a frame from anywhere may
    not be.

    For a headline that is the mean of a per-date series (the
    cross-sectional rank IC of a regression or a ranker) the test is a t
    on the pooled series -- under cpcv, each date's mean across the paths
    that tested it -- over a long-run variance from its lowest cosine
    frequencies, read against Student's t at
    `headline_degrees_of_freedom` for the dates and the horizon, two-sided
    at 5%. For one that is not (a classifier's AUC, a survival model's
    concordance) it is a point comparison with 0.5. Measured on a 30-name
    daily equity panel, none of sixteen recorded runs beat zero at 5%, and
    every run had reported the headline with nothing to say so.
    """
    metric = adapter.headline
    null = adapter.headline_null
    value = oos_metrics.get(metric)
    block: Dict[str, Any] = {
        "metric": metric,
        "null": null,
        "value": None if value is None else float(value),
        "n_dates": None,
        "t_stat": None,
        "t_stat_uncorrected": None,
        "p_value": None,
        "hac_lag": None,
        "hac_degrees_of_freedom": None,
        "ic_autocorrelation_lag1": None,
        "beats_null": None,
    }
    if value is None or null is None or not math.isfinite(value):
        return block, []
    rank_by = (
        f"{metric} is the metric compare_models and list_models rank {task} "
        "models by."
    )

    if adapter.headline_series is None:
        beats = bool(value > null)
        block["beats_null"] = beats
        if beats:
            return block, []
        where = "out of sample" if scope == "out-of-sample" else f"on the {scope} rows"
        what = (
            "the predicted probabilities did not separate the classes"
            if task == "classification"
            else "the risk scores did not order the durations"
        )
        return block, [
            f"{metric} is {value:.4f}, at or below the {null} a random "
            f"ordering scores: {where}, {what}. {rank_by}"
        ]

    values = (
        series.to_numpy(dtype=np.float64)
        if series is not None
        else np.empty(0, dtype=np.float64)
    )
    n = int(np.isfinite(values).sum())
    block["n_dates"] = n
    if n < MIN_HEADLINE_DATES:
        return block, [
            f"{metric} is {value:.4f} over {n} {scope} date(s), fewer than "
            f"the {MIN_HEADLINE_DATES} a long-run variance needs, so whether "
            "it differs from zero was not tested."
        ]
    test = mean_vs_null_test(values, null=float(null), horizon=horizon)
    degrees = test["degrees_of_freedom"]
    block.update(
        {
            "t_stat": test["t_stat"],
            "t_stat_uncorrected": test["t_stat_uncorrected"],
            "p_value": test["p_value"],
            "hac_degrees_of_freedom": degrees,
            "ic_autocorrelation_lag1": test["autocorrelation_lag1"],
        }
    )
    t_stat, p_value = test["t_stat"], test["p_value"]
    if not (math.isfinite(t_stat) and math.isfinite(p_value)):
        return block, [
            f"{metric} is {value:.4f} over {n:,} {scope} dates whose daily "
            "values do not vary, so whether it differs from zero was not "
            "tested."
        ]
    beats = bool(p_value < 0.05 and t_stat > 0)
    block["beats_null"] = beats
    if beats:
        return block, []
    variance = (
        f"t = {t_stat:.2f}, two-sided p = {p_value:.3f} on Student's t with "
        f"{degrees} degrees of freedom, from a long-run variance over the "
        f"series' {degrees} lowest cosine frequencies "
        f"({_frequency_reason(n, horizon)})"
    )
    if p_value < 0.05:
        return block, [
            f"{metric} is {value:.4f} over {n:,} {scope} dates, below "
            f"zero by more than noise explains: {variance}. The predictions "
            "order the names in reverse: ranking by the negated prediction "
            f"would have scored {-value:+.4f} on these dates. {rank_by}"
        ]
    autocorrelation = test["autocorrelation_lag1"]
    plain = test["t_stat_uncorrected"]
    independent = (
        f" The daily values have lag-1 autocorrelation {autocorrelation:.2f}; "
        f"read as independent, the same series gives t = {plain:.2f}."
        if math.isfinite(autocorrelation) and math.isfinite(plain)
        else ""
    )
    return block, [
        f"{metric} is {value:.4f} over {n:,} {scope} dates and is not "
        f"distinguishable from zero: {variance}.{independent} {rank_by}"
    ]


#: Panel rows from which a `gradient_boosting` run is pointed at
#: `hist_gradient_boosting`. See `_gradient_boosting_advice` for why here.
_GRADIENT_BOOSTING_ADVICE_ROWS = 10_000


def _gradient_boosting_advice(model_spec: ModelSpec, n_rows: int) -> List[str]:
    """
    Said once, at the top of a `gradient_boosting` run on a panel large
    enough that the booster is where the run's time goes.

    scikit-learn's GradientBoosting sorts every feature at every node of
    every tree, on one core; HistGradientBoosting bins each feature once,
    splits on the bins, uses every core, and from 10,000 rows stops early
    by default. Measured on a 71,070-row, 12-feature synthetic panel on 16
    logical cores (see the CHANGELOG entry of 2026-10-01): the walk-forward
    experiment took 124 s under gradient_boosting (n_estimators=150,
    max_depth=3) and 2.5 s under hist_gradient_boosting at its defaults,
    about 50x; a single like-for-like fit, 150 trees of depth 3 with early
    stopping off, was 128x. On live data of the same shape it measured
    16x; the gap depends on the cores and on how soon early stopping ends
    the fit, which is why the sentence says where its number came from.

    WHY 10,000 ROWS. One gradient_boosting(150, depth 3) fit, against
    hist_gradient_boosting at its defaults, by rows: 1,000 rows 0.34 s vs
    0.21 s (1.6x); 3,000 rows 1.1 s vs 0.23 s (4.8x); 10,000 rows 4.1 s vs
    0.26 s (15x); 20,000 rows 8.5 s vs 0.07 s, early stopping now on
    (118x). Below 10,000 rows both fit in about a second and the gap is a
    few times, so the sentence would be noise; from 10,000 rows a fit costs
    seconds, an experiment makes one per fold plus the refit, and the gap
    is an order of magnitude and widening.

    Guidance, never a substitution: the two are different models -- binned
    splits, a different default depth and stopping rule -- so which one to
    fit is the caller's decision, and this run fits the one it was asked to.
    """
    if model_spec.estimator.type != "gradient_boosting":
        return []
    if n_rows < _GRADIENT_BOOSTING_ADVICE_ROWS:
        return []
    return [
        f"gradient_boosting on a {n_rows:,}-row panel: scikit-learn's "
        "exact-split booster sorts every feature at every node on one core, "
        "and at this size it is where the run's time goes. "
        "hist_gradient_boosting is its histogram-binned equivalent: on a "
        "71,070-row synthetic panel the same walk-forward experiment ran "
        "about 50x faster with it at its defaults than with gradient_boosting "
        "(n_estimators=150, max_depth=3), on 16 logical cores. It is a "
        "different model, so it was not substituted: this run fitted "
        "gradient_boosting as specified."
    ]


#: Panel rows from which a one-thread `random_forest` run is told what a
#: budget would buy it. The `gradient_boosting` threshold, for the same
#: reason: below it a forest's fits take about a second each and the
#: sentence would be noise.
_RANDOM_FOREST_ADVICE_ROWS = 10_000


def _random_forest_advice(model_spec: ModelSpec, n_rows: int, budget: int) -> List[str]:
    """
    Said once, at the top of a `random_forest` run that will build every
    tree on one core on a panel large enough for that to be where the
    run's time goes.

    A forest's trees are independent, so its `n_jobs` builds them side by
    side, and the folds of a walk-forward experiment fit side by side
    too; at a budget of one neither happens. The budget defaults to
    'auto', so this is a spec that asked for 1, or a machine that gives
    the process one CPU. Measured on a 30-name daily equity panel (31,680
    rows, 8 features, 16 logical cores shared with other work; 200 trees
    of depth 6, 8 folds and the refit): 71 to 124 s at budget 1, 18.1 to
    18.9 s at 'auto' with every content hash of the budget-1 run, and
    hist_gradient_boosting at its defaults 3.9 to 7.9 s at 'auto'. The
    same 10,000-row threshold as `_gradient_boosting_advice`.

    Guidance, never a substitution, as for gradient boosting.
    """
    if model_spec.estimator.type != "random_forest":
        return []
    if budget > 1 or "n_jobs" in model_spec.estimator.params:
        return []
    if n_rows < _RANDOM_FOREST_ADVICE_ROWS:
        return []
    setting = (
        "budget.max_parallelism='auto', which is one thread on this machine,"
        if model_spec.budget.max_parallelism == "auto"
        else "budget.max_parallelism=1"
    )
    return [
        f"random_forest on a {n_rows:,}-row panel at {setting} builds every "
        "tree on one core, and at this size that is where the run's time "
        "goes. On a 31,680-row, 8-feature panel on 16 logical cores, 200 "
        "trees of depth 6 took 71 to 124 s for 8 walk-forward folds and the "
        "refit, 97% of it building trees; at budget.max_parallelism='auto' "
        "the same experiment took 18 to 19 s with the same predictions, and "
        "hist_gradient_boosting at its defaults took 4 to 8 s. It is a "
        "different model, so it was not substituted: this run fitted "
        "random_forest as specified."
    ]


def _instantiate(
    cls: Any,
    params: Dict[str, Any],
    random_seed: int,
    n_jobs: Optional[int] = None,
    *,
    exact_n_jobs: bool = False,
) -> Any:
    """
    Build an estimator from its params, plus what the run supplies: the
    seed, and -- for a constructor that accepts `n_jobs` and params that
    do not set it -- the budget's parallelism. An estimator whose
    signature has neither is built from its params alone.

    `n_jobs` of 1 is left to the constructor's default, which for
    scikit-learn is one thread, unless `exact_n_jobs`: an OpenMP booster
    reads its default as every core. LightGBM ignores the OpenMP runtime's
    thread count when its n_jobs is unset -- measured on 400,000 x 20 rows,
    9.4 CPU-seconds in 2.4 s under a one-thread limit -- so its share is
    handed to it explicitly, one included.
    """
    sig = inspect.signature(cls.__init__)
    kwargs = dict(params)
    if "random_state" in sig.parameters:
        kwargs["random_state"] = random_seed
    if (
        n_jobs
        and (int(n_jobs) > 1 or exact_n_jobs)
        and "n_jobs" in sig.parameters
        and "n_jobs" not in kwargs
    ):
        kwargs["n_jobs"] = int(n_jobs)
    return cls(**kwargs)


#: Estimators whose walk-forward folds may be fitted side by side. For
#: these nothing a run records depends on what runs beside a fit:
#: scikit-learn's gradient boosting is single-threaded and seeded per
#: estimator, and a random forest builds its trees from seeds drawn up
#: front. Their tree builder releases the GIL, which is what lets threads
#: overlap at all (measured: threads 2.6x over one fold at a time for
#: gradient boosting where processes managed 1.4x, the difference being
#: process start-up and copying each fold's matrices).
#:
#: Every other estimator keeps its folds one at a time. Histogram
#: boosting, LightGBM and XGBoost spread one fit over OpenMP threads,
#: which `openmp_thread_limit` holds to the fit's share of the budget, and
#: folds side by side would not overlap anyway: the GIL is held between
#: their parallel regions. Histogram boosting's predictions do not depend
#: on its thread count in scikit-learn 1.9 (measured bit-identical at one
#: and at sixteen threads), but its pickle does -- the bin mapper records
#: the count it was fitted with -- so `model.joblib` differs between two
#: thread counts while every number a run reports agrees. XGBoost's
#: predictions measured identical at 1, 4 and 16 threads too; LightGBM's
#: agreed on a 25,000-row fit and differed in the last bits (4e-19) on a
#: 160,000-row one, so under 'auto' -- one thread below 2,000,000 training
#: cells -- its numbers on such a panel no longer depend on the machine,
#: and above that they follow the budget as they did before. The linear
#: models fit a fold in a fraction of a second through BLAS, whose
#: reductions are not promised to be independent of the threads running
#: beside them.
_FOLD_PARALLEL_ESTIMATORS = frozenset(
    {
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    }
)

#: Why folds ran one at a time, or side by side up to the budget, as
#: `validation_report["fits"]["fold_parallel_limit"]` names it.
FOLD_LIMIT_BUDGET = "budget"
FOLD_LIMIT_ESTIMATOR = "estimator fits one fold at a time"
FOLD_LIMIT_N_JOBS = "n_jobs set in params"
FOLD_LIMIT_SEARCH = "search"
FOLD_LIMIT_ONE_FOLD = "one fold"


def _fold_schedule(
    model_spec: ModelSpec,
    estimator_cls: Any,
    n_folds: int,
    budget: Optional[int] = None,
) -> "Tuple[int, Optional[str]]":
    """
    How many walk-forward folds run side by side, and what limited it:
    up to the budget for an estimator in `_FOLD_PARALLEL_ESTIMATORS`, else
    1, which is the sequential loop exactly as it always ran. `budget` is
    the resolved thread count; None resolves the spec's.

    The limit is None when every fold ran side by side; otherwise, in the
    order checked: `FOLD_LIMIT_ONE_FOLD`, `FOLD_LIMIT_ESTIMATOR`,
    `FOLD_LIMIT_N_JOBS` (the caller chose that parallelism, and it would
    multiply with this one), `FOLD_LIMIT_SEARCH` (the search spends the
    budget scoring its candidates side by side inside each fold), and
    `FOLD_LIMIT_BUDGET` (fewer threads than folds, a budget of one
    included).

    Folds beside each other share the budget: each fold's estimators get
    n_jobs = budget // workers, so the threads in use never exceed it.
    Before, the budget reached only an estimator's own n_jobs, which a
    gradient booster does not have: its folds ran one at a time on one
    core whatever the budget said. A random forest fits on its share of
    threads -- each tree from a seed drawn before any thread starts, so the
    fit does not depend on them -- and predicts on one, which adds its
    trees in order (see `_predict_on_one_thread`); so its numbers are its
    budget-1 numbers at any budget.
    """
    if budget is None:
        budget = model_spec.budget.resolved_max_parallelism()
    budget = int(budget)
    if n_folds < 2:
        return 1, FOLD_LIMIT_ONE_FOLD
    if estimator_cls not in _FOLD_PARALLEL_ESTIMATORS:
        return 1, FOLD_LIMIT_ESTIMATOR
    if "n_jobs" in model_spec.estimator.params:
        return 1, FOLD_LIMIT_N_JOBS
    if model_spec.search is not None:
        return 1, FOLD_LIMIT_SEARCH
    workers = max(1, min(budget, n_folds))
    return workers, (FOLD_LIMIT_BUDGET if workers < n_folds else None)


def _fold_workers(
    model_spec: ModelSpec,
    estimator_cls: Any,
    n_folds: int,
    budget: Optional[int] = None,
) -> int:
    """How many walk-forward folds run side by side; see `_fold_schedule`."""
    return _fold_schedule(model_spec, estimator_cls, n_folds, budget)[0]


#: Training cells (rows x columns) below which an OpenMP estimator fits on
#: one thread under budget 'auto'. Measured on hist_gradient_boosting under
#: the PASSIVE wait policy this package sets: a 15,000-row, 8-feature fold
#: fitted in 0.28 to 0.36 s on one thread and 1.5 to 1.7 s on sixteen, with
#: the same predictions, while at 1,000,000 x 8 rows one thread was 1.5x
#: slower than sixteen. The crossing is near 2,000,000 cells. An explicit
#: budget is taken as asked.
_OPENMP_ONE_THREAD_CELLS = 2_000_000


def _fit_threads(
    threads_kind: Optional[str],
    budget_asked: Any,
    share: int,
    n_cells: int,
) -> "Tuple[int, Optional[int]]":
    """
    (the n_jobs one fit's constructor is handed, the OpenMP thread count it
    runs under) for a fit given `share` threads of the budget.

    An OpenMP estimator gets the same count for both: one thread under
    'auto' below `_OPENMP_ONE_THREAD_CELLS` training cells, else its share.
    Every other estimator is handed its share as n_jobs, which reaches only
    a constructor that takes it, and runs under no OpenMP limit.
    """
    share = max(1, int(share))
    if threads_kind != "openmp":
        return share, None
    if budget_asked == "auto" and n_cells < _OPENMP_ONE_THREAD_CELLS:
        return 1, 1
    return share, share


def _reported_fold_schedule(
    budget_asked: Any, workers: int, limit: Optional[str]
) -> Dict[str, Any]:
    """
    `fold_workers` and `fold_parallel_limit` as the tool output reports
    them: the folds that ran side by side and what limited it.

    Under budget 'auto' the count depends on the machine wherever the
    budget decides it, and a recorded run is replayed by comparing its
    output on whatever machine checks it -- so there it reads 'auto',
    limited by 'budget', and the count is min(the resolved budget, folds),
    with the resolved budget in the manifest's
    `environment.threads.auto_parallelism`. A limit that is not the
    budget's -- the estimator, a search, n_jobs in params, a single fold --
    is the same on every machine and is reported with its count of 1.
    """
    if budget_asked == "auto" and limit in (FOLD_LIMIT_BUDGET, None):
        return {"fold_workers": "auto", "fold_parallel_limit": FOLD_LIMIT_BUDGET}
    return {"fold_workers": int(workers), "fold_parallel_limit": limit}


def _cache_report(
    after: Dict[str, int], before: Dict[str, int], shared: bool, projectable: bool
) -> Dict[str, Any]:
    """The cache block of `validation_report`: hits, misses and whether the
    cache was shared, plus projections and projectability for a shared one
    -- the only kind that can project."""
    report: Dict[str, Any] = {
        key: after[key] - before[key] for key in ("hits", "misses")
    }
    report["shared"] = shared
    if shared:
        report["projections"] = after["projections"] - before["projections"]
        report["projectable"] = projectable
    return report


def _forests_in(estimator: Any) -> List[Any]:
    """The scikit-learn forests inside a fitted estimator: itself, and the
    per-fold clones a `CalibratedClassifierCV` fitted around it."""
    found: List[Any] = []
    if isinstance(estimator, (RandomForestClassifier, RandomForestRegressor)):
        found.append(estimator)
    for calibrated in getattr(estimator, "calibrated_classifiers_", None) or ():
        found.extend(_forests_in(getattr(calibrated, "estimator", None)))
    return found


def _predict_on_one_thread(estimator: Any) -> None:
    """
    Put every forest inside a fitted estimator back on its constructor's
    n_jobs (None, one thread), so that it predicts, reports importances and
    pickles the same at any budget.

    A random forest fits its trees side by side from seeds drawn up front,
    so its fit does not depend on n_jobs; its predictions do. Above one
    job, scikit-learn adds the trees' outputs in whatever order its threads
    finish, and the last bits followed: at a budget of 16 over 8 folds,
    5,798 of 15,120 out-of-sample predictions differed from the budget-1
    run (by up to 1.7e-13 relative), and two budget-16 runs differed from
    each other. The deployed forest kept n_jobs at the budget, so its
    pickle -- and `score_model` -- depended on the budget too. Predicting on
    one thread costs little: prediction is a small share of a forest's fit.

    The allowlist admits no `n_jobs`, so any n_jobs above one on a forest
    here is the budget's.
    """
    for forest in _forests_in(estimator):
        if forest.n_jobs not in (None, 1):
            forest.set_params(n_jobs=None)


def _run_folds_side_by_side(
    folds: List[Any],
    prepare: Callable[[Any], "Dict[str, Any] | None"],
    fit: Callable[[Dict[str, Any], int], Dict[str, Any]],
    record: Callable[[Dict[str, Any], Dict[str, Any]], None],
    workers: int,
    n_jobs: int,
) -> None:
    """
    The fold loop on `workers` threads, with the sequential loop's outcome.

    Every fold is prepared first, in order, on this thread -- so the skips,
    the purge count, the fold cache and every refusal before a fit happen
    exactly as they do one fold at a time -- stopping at the first that
    raises. The prepared folds are then fitted on the pool and recorded in
    fold order, whatever order they finish in. The error raised is the one
    the loop would have met first: a fit's, when a fold before the failed
    preparation failed to fit, and otherwise the preparation's.

    Each fit runs on one BLAS thread. At the BLAS default every fold's
    linear algebra started one thread per logical CPU, `workers` times
    over; one each was measured 2.4x to 2.8x faster for a 235-asset solve
    and eigendecomposition on four to eight workers, and gives the same
    bits whatever the machine's core count.
    """

    def fit_on_one_blas_thread(ready: Dict[str, Any], jobs: int) -> Dict[str, Any]:
        with single_threaded_blas():
            return fit(ready, jobs)

    prepared: List[Dict[str, Any]] = []
    failure: "Exception | None" = None
    for fold in folds:
        try:
            ready = prepare(fold)
        except Exception as exc:  # noqa: BLE001 - re-raised below, in fold order
            failure = exc
            break
        if ready is not None:
            prepared.append(ready)
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sqt-fold")
    try:
        futures = [
            pool.submit(fit_on_one_blas_thread, ready, n_jobs) for ready in prepared
        ]
        for ready, future in zip(prepared, futures):
            record(ready, future.result())
    finally:
        # A fit that failed leaves the folds after it unrecorded, as the
        # loop would; the ones not yet started are not started.
        pool.shutdown(wait=True, cancel_futures=True)
    if failure is not None:
        raise failure


def _validate_classification_target(panel: pd.DataFrame) -> None:
    """
    Every allowlisted classification estimator (logistic, hist_gradient_boosting,
    random_forest) needs a categorical target, but TargetSpec only builds a
    continuous forward_return -- without this check, task='classification'
    against an unmodified target reaches sklearn and fails deep inside
    .fit() with a confusing "Unknown label type: continuous" error instead
    of a clear, actionable one raised before any fold is even attempted.
    """
    # Python set equality treats 0.0/1.0 and 0/1 as equal members, so this
    # single comparison covers both int- and float-dtype target columns.
    #
    # {0, 1, 2} is admitted alongside {0, 1} because that is what
    # TargetSpec(type='triple_barrier') produces: lower barrier first, upper
    # barrier first, or neither touched within the horizon. AUC is undefined
    # for three classes and comes back NaN there, which
    # classification_metrics already handles; accuracy and the class balance
    # stay meaningful.
    unique_values = set(pd.unique(panel["target"].dropna()))
    if not unique_values or not unique_values <= {0, 1, 2}:
        sample = sorted(unique_values)[:10]
        raise ValidationError(
            "run_model_experiment: task='classification' requires a discrete "
            "target — {0, 1} from TargetSpec(type='forward_direction') or "
            "{0, 1, 2} from TargetSpec(type='triple_barrier') — but the "
            f"dataset's target column has values {sample}"
            + ("..." if len(unique_values) > 10 else "")
            + "."
        )
    if len(unique_values) < 2:
        raise ValidationError(
            "run_model_experiment: task='classification' requires a discrete "
            f"target with at least two classes, but every row is "
            f"{sorted(unique_values)[0]!r}. A threshold that no bar exceeds "
            "produces exactly this."
        )


def _labels(model_spec: ModelSpec, frame: pd.DataFrame) -> np.ndarray:
    """
    The label an estimator of this task fits, read off a panel slice.

    `target` for every task but survival, whose label is two columns --
    the duration and whether the event was observed -- and whose
    estimators refuse the duration alone. One reader, used by the fold
    loop, the inner search closure and the full-panel refit, so the three
    cannot disagree about what a survival row is.
    """
    if model_spec.task == "survival":
        return survival_labels(frame)
    return frame["target"].to_numpy()


def _validate_survival_target(panel: pd.DataFrame) -> None:
    """
    A survival panel carries a positive duration and a 0/1 event
    indicator, with at least one event observed; anything else cannot be
    ordered, and is refused here rather than inside an estimator.
    """
    if EVENT_COL not in panel.columns:
        raise ValidationError(
            "run_model_experiment: task='survival' needs an `event` column "
            "beside `target` -- 1 where the event was observed, 0 where the "
            "window ended first. A panel registered with "
            "register_external_panel declares it as `event_column` on the "
            "target; a duration without it would be fitted as though every "
            "row's event had been seen."
        )
    labels = survival_labels(panel)
    duration, event = labels[:, 0], labels[:, 1]
    if not np.isfinite(duration).all() or (duration <= 0).any():
        raise ValidationError(
            "run_model_experiment: task='survival' needs a finite, positive "
            "duration on every row; a zero or negative duration has no place "
            "in the ordering."
        )
    if event.sum() == 0:
        raise ValidationError(
            "run_model_experiment: every row of the survival label is censored, "
            "so there is no observed event and nothing to order."
        )


def _check_task_target_compatibility(task: str, target_id: "str | None") -> None:
    """
    A classification model needs a binary target and a regression model a
    continuous one. Both were previously accepted against either target,
    so the mismatch only surfaced as a confusing sklearn error (or, for
    regression on a 0/1 target, not at all — it would happily fit and
    report meaningless R2/IC).
    """
    if not target_id or ":" not in target_id:
        return
    target_type = target_id.split(":", 1)[0]
    # DERIVED from the target registry rather than restated here. This map
    # was three hand-written sets, so every label added anywhere had to be
    # remembered in this file too -- and a ranker consumes the same
    # continuous labels a regressor does, which meant two of the three sets
    # were duplicates of each other kept in sync by hand.
    allowed = set(targets_for_task(task)) if task in TASKS else None
    if allowed is None:
        # NOT a pass. An unrecognized task used to return here, so the one
        # check standing between a task and an incompatible target skipped
        # itself for exactly the task nobody had thought about yet -- and a
        # regressor on a 0/1 target fits happily and reports meaningless
        # R2. The Literal makes this unreachable today; it is here so that
        # widening the taxonomy fails LOUDLY at the map that was not
        # updated rather than quietly at the model that was fitted.
        raise ValidationError(
            f"run_model_experiment: task={task!r} has no entry in the "
            "task/target compatibility map, so nothing can say whether "
            f"{target_type!r} is a target it can consume. Add the task to "
            "_check_task_target_compatibility in modeling/engine.py."
        )
    if target_type in allowed:
        return
    raise ValidationError(
        f"run_model_experiment: task={task!r} expects one of "
        f"{sorted(allowed)}, but this dataset was built with {target_type!r}. "
        "Rebuild the dataset with a compatible TargetSpec(type=...), or change "
        "the model's task."
    )


def _preprocess(
    model_spec: ModelSpec,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_ids: List[str],
    train_features: "pd.DataFrame | None" = None,
    test_features: "pd.DataFrame | None" = None,
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Transform the feature columns for one fold.

    The pipeline is fitted on the training rows and its state applied
    unchanged to the test rows, which is the fold-boundary discipline that
    keeps the test window genuinely out of sample. A stateless step --
    cross-sectional standardization, which uses only each date's own
    cross-section, contemporaneous information a live model also has --
    fits nothing, so for it the two sides are transformed independently
    and no statistic crosses the split. Which steps run is
    `PreprocessingSpec.resolved_steps`, read here and in the refit and
    nowhere else.
    """
    # Selected ONCE each. `train_frame[feature_ids]` was evaluated twice on
    # the pooled path -- once to fit and once to apply -- and the take is
    # about 4.9 ms on a 100,000 x 20 block, paid on every fold of every run
    # and every candidate of every hyperparameter search.
    #
    # The outer fold loop hands these in already gathered out of one
    # panel-wide float64 matrix, which skips the column take and the
    # C-order copy entirely. The hyperparameter search does not, because
    # its inner folds are slices of a frame rather than of the panel, so it
    # selects here as before.
    if train_features is None:
        train_features = train_frame[feature_ids]
    if test_features is None:
        test_features = test_frame[feature_ids]

    # ONE call, whatever the spec asked for. This branched on
    # `normalization` -- pooled statistics fitted on train and applied to
    # test, or per-date standardization of each side -- and so did the
    # full-panel refit, except the refit did not, which is how a model
    # validated cross-sectionally was deployed pooled. The pipeline is the
    # one place that knows how a step is fitted and applied; the fold loop,
    # the search closure and the refit all hand it the same resolved steps.
    # The default pair still goes through the fused native kernel inside.
    _state, train_X, test_X = fit_and_apply_pipeline(
        model_spec.preprocessing.resolved_steps,
        train_features,
        test_features,
        FoldContext.from_frame(train_frame),
        FoldContext.from_frame(test_frame),
    )
    return train_X, test_X


def _refuse_missing_after_preprocessing(
    train_X: pd.DataFrame, test_X: pd.DataFrame, model_spec: ModelSpec, where: str
) -> None:
    """
    Refuse, by name, a NaN that the pipeline left for an estimator that
    cannot take one.

    Without this the fit failed several frames down with sklearn's own
    "Input X contains NaN", naming neither the policy that let the hole
    through nor the step that would close it.
    """
    holes = int(np.isnan(train_X.to_numpy(dtype=np.float64)).sum()) + int(
        np.isnan(test_X.to_numpy(dtype=np.float64)).sum()
    )
    if not holes:
        return
    raise ValidationError(
        f"{where}: {holes} missing value(s) reach estimator "
        f"{model_spec.estimator.type!r} after the preprocessing pipeline "
        f"{step_types(model_spec.preprocessing.resolved_steps)}, and it does "
        "not accept missing values. Either add an `impute` step (with or "
        "without `missing_indicator` before it) to preprocessing.steps, fit "
        "an estimator that accepts them -- list_modeling_capabilities reports "
        "`accepts_missing` per estimator -- or build the dataset with "
        "missing.policy='drop'."
    )


def _training_missing_rates(
    train_matrix: np.ndarray, feature_ids: List[str]
) -> Dict[str, float]:
    """Each feature's share of NaN in one fold's training rows."""
    if len(train_matrix) == 0:
        rates = np.ones(len(feature_ids))
    else:
        rates = np.isnan(train_matrix).mean(axis=0)
    return {f: round(float(r), 4) for f, r in zip(feature_ids, rates)}


def _refuse_absent_features(
    rates: Dict[str, float], fold_number: int, test_dates, model_spec: ModelSpec
) -> None:
    """
    Refuse, by name, a feature that is missing in EVERY training row of a
    fold. The impute step would fit it as a constant and the fold would
    train on a feature that does not exist there; an estimator that accepts
    missing values (hist_gradient_boosting) died inside numpy on the same
    column instead of saying so.
    """
    absent = sorted(f for f, r in rates.items() if r >= 1.0)
    if not absent:
        return
    first = str(pd.Timestamp(test_dates[0]).date())
    last = str(pd.Timestamp(test_dates[-1]).date())
    raise ValidationError(
        f"run_model_experiment: feature(s) {absent} are missing in every "
        f"training row of fold {fold_number} (test window {first}..{last}). "
        "A column with no training values cannot be imputed -- the fold "
        "would train on a constant under the feature's name and its "
        "importance would be averaged in as if the feature existed there. "
        "Drop the feature, start the dataset where it is available, build "
        "with missing.policy='drop', or use a validation scheme whose "
        "training windows cover it (this run: "
        f"{model_spec.validation.method}, "
        f"train_window={getattr(model_spec.validation, 'train_window', None)})."
    )


def _fold_sample_weights(
    model_spec: ModelSpec, index: SampleIndex
) -> "np.ndarray | None":
    """Training-row weights for one fold, or None for an unweighted fit --
    defined on the rows' sample index, which is what a weight is about."""
    if model_spec.weighting.method == "none":
        return None
    return build_sample_weights(
        model_spec.weighting.method,
        index.dates,
        index.label_end,
        index.entities,
        model_spec.weighting.half_life_days,
    )


def _fit(
    estimator: Any,
    X: "np.ndarray",
    y: Any,
    weights: "np.ndarray | None",
    group: "np.ndarray | None" = None,
):
    """
    Fit, passing sample weights and query groups only when they apply.

    An estimator that does not accept `sample_weight` fails loudly rather
    than silently ignoring the request — a weighting the caller believes is
    active but which never reached the fit is worse than an error, because
    the model looks like it corrected for label overlap and did not.

    `group` is the ranking path. Both LightGBM and XGBoost take it as
    consecutive per-query counts and ASSUME the rows are already ordered by
    group without checking, which is why the caller sorts by date first and
    group_sizes() verifies the ordering rather than trusting it.

    Every fit in a run comes through here -- each fold's, each search
    candidate's, each quantile and conformal fit's, and the refit's -- so
    this is where a forest fitted on several threads is put back on one
    before anything predicts with it (see `_predict_on_one_thread`).
    """
    kwargs: Dict[str, Any] = {}
    if weights is not None:
        kwargs["sample_weight"] = weights
    if group is not None:
        kwargs["group"] = group
    if not kwargs:
        estimator.fit(X, y)
        _predict_on_one_thread(estimator)
        return
    try:
        estimator.fit(X, y, **kwargs)
    except TypeError as exc:
        unsupported = " or ".join(sorted(kwargs))
        raise ValidationError(
            f"run_model_experiment: estimator {type(estimator).__name__} does not "
            f"accept {unsupported}. For sample_weight, use weighting.method='none' "
            "or an estimator that supports weighted fitting; for group, use a "
            "task='ranking' estimator."
        ) from exc
    _predict_on_one_thread(estimator)


def _fit_quantile_models(
    estimator_cls: Any,
    support: Any,
    params: Dict[str, Any],
    model_spec: ModelSpec,
    arrays: Any,
    n_jobs: Optional[int] = None,
    exact_n_jobs: bool = False,
) -> Dict[float, Any]:
    """
    One estimator per requested quantile, fitted on the same rows and
    weights as the point estimator, with the registry's quantile parameter
    set and its fixed objective switched on. The point estimator is left
    exactly as it was: `prediction` is the base fit, and the quantiles
    stand beside it. `n_jobs` defaults to the budget's parallelism; a fold
    fitted beside others passes its own share, and `exact_n_jobs` is
    `_instantiate`'s.
    """
    if n_jobs is None:
        n_jobs = model_spec.budget.resolved_max_parallelism()
    models: Dict[float, Any] = {}
    for q in model_spec.quantiles:
        quantile_params = {**params, **support.fixed, support.param: float(q)}
        model = _instantiate(
            estimator_cls,
            quantile_params,
            model_spec.random_seed,
            n_jobs=n_jobs,
            exact_n_jobs=exact_n_jobs,
        )
        _fit(model, arrays.X, arrays.y, arrays.sample_weight)
        models[float(q)] = model
    return models


def _conformal_radius(
    estimator_cls: Any,
    params: Dict[str, Any],
    model_spec: ModelSpec,
    arrays: Any,
    n_jobs: Optional[int] = None,
    exact_n_jobs: bool = False,
) -> "tuple[float, int]":
    """
    The split-conformal radius for one training window: absolute
    residuals on held-out date blocks, the estimator refit without each
    under the embargo and the label purge, and their (1 - alpha) quantile.
    Returns (radius, number of residuals it was read from). The blocks are
    cut on the sample index the arrays carry, so they are the rows of `X`
    whatever order the adapter put them in. `n_jobs` as for
    `_fit_quantile_models`.
    """
    intervals = model_spec.intervals
    assert intervals is not None
    if arrays.index is None:
        raise ValidationError(
            "conformal calibration needs the sample index beside the arrays; "
            "the adapter did not carry it."
        )
    weights = arrays.sample_weight
    if n_jobs is None:
        n_jobs = model_spec.budget.resolved_max_parallelism()

    def fit_predict(train_mask, test_mask):
        model = _instantiate(
            estimator_cls,
            params,
            model_spec.random_seed,
            n_jobs=n_jobs,
            exact_n_jobs=exact_n_jobs,
        )
        _fit(
            model,
            arrays.X[train_mask],
            arrays.y[train_mask],
            weights[train_mask] if weights is not None else None,
        )
        return arrays.y[test_mask], np.asarray(model.predict(arrays.X[test_mask]))

    residuals = held_out_residuals(
        fit_predict,
        arrays.index.dates,
        arrays.index.label_end,
        n_folds=int(intervals.calibration_folds),
        embargo=int(model_spec.validation.embargo),
    )
    return conformal_radius(residuals, float(intervals.alpha)), int(residuals.size)


def _predict_fold(
    adapter: Any,
    model_spec: ModelSpec,
    estimator: Any,
    test_X: pd.DataFrame,
    test_y: Any,
    test_dates: "np.ndarray | None" = None,
    train_y: "np.ndarray | None" = None,
) -> "tuple[Dict[str, float], np.ndarray, Dict[str, pd.Series]]":
    """
    Returns (metrics, prediction_values, ic_series).

    `prediction_values` is always a continuous score suitable for downstream
    signal construction (see modeling.bridge) -- the raw prediction for a
    regression task, the positive-class probability for a classification one,
    the ordering score for a ranker. Which of those it is, and which metrics
    mean anything against it, is the adapter's decision rather than a chain
    of task comparisons here.

    `ic_series` carries this fold's PER-DATE cross-sectional IC series so the
    caller can pool every fold's dates and compute the OOS dispersion
    statistics once. Averaging each fold's own std/ICIR is a different
    quantity -- see aggregate_cross_sectional_ic.
    """
    score = adapter.score(estimator, test_X)
    metrics = adapter.metrics(
        model_spec, estimator, test_X, test_y, score, test_dates, train_y
    )
    return metrics, score, adapter.fold_ic(test_y, score, test_dates)


def run_experiment(
    dataset: Dict[str, Any],
    model_spec: ModelSpec,
    dataset_id: str,
    *,
    register: bool = True,
    fold_cache: Optional[FoldCache] = None,
) -> Dict[str, Any]:
    """
    Args:
        dataset: the dict returned by dataset.builder.build_dataset
            (include_target=True — run_model_experiment always trains,
            never scores).
        model_spec: task/estimator/validation spec.
        dataset_id: id under which the dataset panel was persisted (see
            modeling/agent/tools.py::build_model_dataset) — recorded in
            the registered model's manifest for lineage.
        fold_cache: a cache of preprocessed fold matrices to read from and
            add to, keyed by the plan's preprocessing hashes. Pass one
            across several runs over the SAME dataset -- the feature
            ablation does, once per feature -- and a column-wise pipeline
            is fitted once per fold for all of them. Without one the run
            keeps a private cache, which is what lets the inner search
            preprocess each inner fold once rather than once per
            candidate. Needs the dataset's `data_hash` to key on.
        register: persist the refit estimator and its OOS predictions, and
            return a model_id. True for anything a caller might want to
            score later. False for a comparison that fits many candidate
            specs and keeps none of them — feature ablation refits once per
            feature, and registering 41 models to answer one question about
            a 40-feature panel would fill the registry with models nobody
            asked for. The FITS are identical either way; only the writing
            down is skipped, so an unregistered run's metrics are the same
            numbers a registered one would have reported.

    Raises:
        ValidationError: unknown estimator, disallowed estimator param,
        or the dataset has too few dates for even one walk-forward fold.
    """
    adapter = get_adapter(model_spec.task)
    estimator_cls = get_estimator_class(model_spec.task, model_spec.estimator.type)
    validate_params(
        model_spec.task, model_spec.estimator.type, model_spec.estimator.params
    )

    # The budget as the spec asked for it -- a whole number, or 'auto' --
    # and the thread count that means in this process, read once so every
    # fit of the run shares one answer. Only the first reaches the tool
    # output: the second depends on the machine, and a recorded run is
    # replayed on whatever machine checks it. The manifest's
    # `environment.threads.auto_parallelism` keeps the second.
    budget_asked = model_spec.budget.max_parallelism
    budget = model_spec.budget.resolved_max_parallelism()
    # What sets the cores one fit uses, as the registry declares it: a
    # forest's own n_jobs, an OpenMP runtime, or one core.
    cost = estimator_cost(model_spec.task, model_spec.estimator.type)
    threads_kind = cost.threads if cost is not None else None

    # Caveats about THIS run, returned beside its metrics. Seeded with the
    # one the run can state before it starts.
    run_warnings: List[str] = []
    run_warnings.extend(_calibration_importance_warning(model_spec, estimator_cls))
    repeated_grid_values = grid_duplicates_warning(model_spec.search)
    if repeated_grid_values:
        run_warnings.append(repeated_grid_values)

    # Whether the estimator can fit a quantile at all, before any data is
    # touched: the registry declares the parameter, and an estimator
    # without one cannot be asked, whatever the spec says.
    quantile = None
    if model_spec.quantiles:
        quantile = quantile_support(model_spec.task, model_spec.estimator.type)
        if quantile is None:
            raise ValidationError(
                f"run_model_experiment: estimator {model_spec.estimator.type!r} "
                "has no quantile parameter, so it cannot fit the requested "
                f"quantiles {list(model_spec.quantiles)}. Estimators that can: "
                f"{quantile_estimators(model_spec.task)}; "
                "list_modeling_capabilities reports `quantile_param` per "
                "estimator. Drop `quantiles`, or use `intervals` for a "
                "conformal band around any regressor's point prediction."
            )

    panel = dataset["panel"]
    refuse_duplicate_entity_dates(panel, "run_model_experiment")
    _check_task_target_compatibility(model_spec.task, dataset.get("target_id"))
    if model_spec.task == "classification":
        _validate_classification_target(panel)
    if model_spec.task == "survival":
        _validate_survival_target(panel)
    # The purge needs each row's label end. A panel without the column gets
    # one derived from the target's horizon, said so in `warnings`, or is
    # refused when there is no horizon to derive it from.
    panel, purge_basis, purge_warnings = panel_with_label_end(
        panel, dataset.get("target_id"), "run_model_experiment"
    )
    run_warnings.extend(purge_warnings)
    run_warnings.extend(_gradient_boosting_advice(model_spec, len(panel)))
    run_warnings.extend(_random_forest_advice(model_spec, len(panel), budget))
    feature_ids = dataset["feature_ids"]
    dates = pd.Index(sorted(panel["date"].unique()))

    splitter = build_splitter(model_spec.validation)
    if splitter.n_splits(dates) < 1:
        if model_spec.validation.method == "purged_kfold":
            raise ValidationError(
                f"run_model_experiment: dataset has {len(dates)} dates, not enough "
                f"for {model_spec.validation.n_splits} purged k-fold blocks with "
                f"embargo={model_spec.validation.embargo}."
            )
        raise ValidationError(
            f"run_model_experiment: dataset has {len(dates)} dates, not enough for one "
            f"walk-forward fold with train_window={model_spec.validation.train_window}, "
            f"test_window={model_spec.validation.test_window}, embargo={model_spec.validation.embargo}."
        )

    has_label_end = LABEL_END_COL in panel.columns
    is_cpcv = model_spec.validation.method == "cpcv"
    fold_metrics = []
    fold_importance = []
    # The columns the ESTIMATOR sees: the pipeline's output, which is the
    # dataset's feature columns for every step that maps columns onto
    # themselves and something else for a step that adds or replaces them
    # (a missingness indicator, a PCA). Importance is labelled by these,
    # not by `feature_ids`, and every fold and the refit must agree on
    # them or the per-fold vectors could not be summarized.
    model_columns: "List[str] | None" = None
    fold_records: List[Dict[str, Any]] = []
    fold_weights: List[float] = []
    # The dates some completed fold tested on: the rows the out-of-sample
    # metrics are averaged over, and so the rows whose labels the effective
    # sample size is measured on.
    tested_dates = np.zeros(len(dates), dtype=bool)
    # metric prefix -> list of each fold's per-date IC series.
    pooled_ic: Dict[str, List[pd.Series]] = {}
    oos_prediction_frames = []
    n_purged_total = 0
    # Skipped folds were previously invisible: the result reported only how
    # many folds SURVIVED, so a run where 8 of 10 folds were dropped looked
    # identical to a clean 2-fold run.
    skipped: List[Dict[str, str]] = []
    # One entry per fold that ran a hyperparameter search, so a reader can
    # see whether the chosen parameters were stable across folds or whether
    # each fold picked something different -- the latter means the search
    # was fitting noise, and the report is the only place that shows it.
    search_reports: List[Dict[str, Any]] = []
    # The schedule, decided before anything is fitted: the folds, the rows
    # the label-overlap purge removes from each, the inner folds each
    # training window supports, and the fit count all of that implies.
    # Refused here by name when it costs more than the spec's budget
    # allows, and executed as planned below, so the count the plan reports
    # is the count that runs.
    plan = plan_experiment(
        model_spec,
        dates,
        panel=panel,
        dataset_hash=dataset.get("data_hash"),
        feature_ids=feature_ids,
    )
    plan.refuse_over_budget("run_model_experiment")
    if model_spec.search is not None and model_spec.search.method == "tpe":
        # Before the first fold, not inside it: a missing library is a
        # fact about the environment, and the first fold's preprocessing
        # is work spent learning it.
        require_optuna("run_model_experiment")
    n_expected_folds = len(plan.folds)
    if fold_cache is not None and not dataset.get("data_hash"):
        raise ValidationError(
            "run_model_experiment: a shared fold_cache keys its entries on the "
            "dataset's data_hash, and this dataset carries none, so two datasets "
            "could not be told apart in it. Build the dataset with "
            "build_dataset, or run without the cache."
        )
    cache = fold_cache if fold_cache is not None else FoldCache()
    cache_before = cache.stats()
    # Whether a feature subset's matrices may be read off a wider run's:
    # exact for a pipeline whose every step is column-wise, and only then.
    projectable = column_wise_pipeline(model_spec.preprocessing.resolved_steps)
    # Row -> position in `dates`, computed once instead of hashing the whole
    # date column against a fresh set on every fold. `dates` is sorted and
    # every row's date is in it by construction, so searchsorted is exact.
    # A fold then selects rows by gathering a small per-date boolean, which
    # also keeps working for splitters whose folds are not contiguous
    # (purged K-fold) -- an interval slice would not. Both sides as
    # datetime64 instants: a timezone-aware column's `to_numpy()` builds a
    # Timestamp per row and searchsorted then compares them one at a time.
    date_code = np.searchsorted(datetime_values(dates), datetime_values(panel["date"]))
    # The whole panel's features as ONE C-contiguous float64 matrix, built
    # here rather than per fold. A fold then gathers its rows with a numpy
    # take (5.5 ms on 100,000 rows) instead of a pandas column selection
    # plus the C-order copy the kernels need (7.2 + 4.2 ms).
    feature_matrix = np.ascontiguousarray(panel[feature_ids].to_numpy(dtype=np.float64))
    # A panel built under missing.policy='keep', or an external panel with
    # holes, carries NaN into the folds. Whether that is a problem depends
    # on what is left after preprocessing and on the estimator, and is
    # decided per fold below; this is the one pass that says whether the
    # question needs asking at all.
    panel_has_missing = bool(np.isnan(feature_matrix).any())
    estimator_accepts_missing = accepts_missing(estimator_cls)
    # Whether a search scores its candidates on a pool: the rule
    # `search_best_params` applies (a grid or random search, a budget above
    # one, more than one candidate; a tpe search runs one trial at a time).
    search_on_pool = (
        model_spec.search is not None
        and budget > 1
        and model_spec.search.method != "tpe"
        and n_search_candidates(model_spec.search) > 1
    )

    def _search_on(frame: pd.DataFrame, *, prefix: str):
        """
        Choose estimator parameters on `frame` alone: an inner walk-forward
        under the spec's embargo, purged on each row's own label end, every
        candidate scored the way the real fit will run it -- the same
        preprocessing, the same weighting, the adapter's own score, so the
        search cannot select for a pipeline that is never used. `prefix`
        keys the inner folds' matrices in the cache: they are a function of
        the frame and the search's shape, not of the candidate, so each
        inner fold is preprocessed once rather than once per candidate.

        The folds call this on their training rows; the refit calls it on
        the full panel, which is how the deployed estimator comes to carry
        parameters a search actually chose (findings D14).

        Candidates scored side by side share the budget the way folds do:
        each gets `budget // workers` threads, `workers` being the pool
        `search_best_params` runs (`search_pool_workers`, one rule for
        both). Each candidate was handed the whole budget inside a pool of
        budget-many workers, so a random-forest grid at a budget of 4 ran
        sixteen tree builders and each histogram-boosting candidate started
        an OpenMP team on every core. Candidates scored one at a time get
        the whole budget, as before.
        """
        n_inner = inner_fold_count(
            int(frame["date"].nunique()),
            model_spec.search.inner_splits,
            model_spec.validation.embargo,
        )
        pool_workers = search_pool_workers(
            model_spec.search.method,
            n_search_candidates(model_spec.search),
            n_inner,
            budget,
        )
        candidate_share = max(1, budget // pool_workers)

        def _fit_predict(params, inner_train, inner_test, fold_index):
            inner_key = f"{prefix}{fold_index}"
            matrices = cache.lookup(inner_key, feature_ids)
            if matrices is None:
                matrices = _preprocess(model_spec, inner_train, inner_test, feature_ids)
                cache.store(inner_key, feature_ids, *matrices, projectable=projectable)
            inner_train_X, inner_test_X = matrices
            fit_jobs, openmp_threads = _fit_threads(
                threads_kind,
                budget_asked,
                candidate_share,
                int(inner_train_X.shape[0]) * int(inner_train_X.shape[1]),
            )
            candidate = _instantiate(
                estimator_cls,
                params,
                model_spec.random_seed,
                n_jobs=fit_jobs,
                exact_n_jobs=openmp_threads is not None,
            )
            inner_index = SampleIndex.from_frame(inner_train)
            inner_arrays = adapter.prepare(
                model_spec,
                inner_index,
                inner_train_X,
                _labels(model_spec, inner_train),
                _fold_sample_weights(model_spec, inner_index),
            )
            with openmp_thread_limit(openmp_threads):
                _fit(
                    candidate,
                    inner_arrays.X,
                    inner_arrays.y,
                    inner_arrays.sample_weight,
                    group=inner_arrays.group,
                )
                # The adapter's score, so a search on a ranker selects using
                # the ordering score the real fit will produce rather than
                # whatever `predict` happens to return.
                predictions = adapter.score(candidate, inner_test_X)
            probabilities = predictions if model_spec.task == "classification" else None
            return predictions, probabilities

        # When the search scores its candidates side by side, every
        # candidate's fit runs on one BLAS thread, as the walk-forward
        # folds' do (see `_run_folds_side_by_side`) -- the first candidate,
        # which runs alone to fill the cache, included, so one search's
        # scores all come from the same BLAS setting.
        fit_predict = _fit_predict
        if search_on_pool:

            def fit_predict(params, inner_train, inner_test, fold_index):
                with single_threaded_blas():
                    return _fit_predict(params, inner_train, inner_test, fold_index)

        return search_best_params(
            task=model_spec.task,
            search_spec=model_spec.search,
            base_params=model_spec.estimator.params,
            train_frame=frame,
            feature_ids=feature_ids,
            random_seed=model_spec.random_seed,
            fit_predict=fit_predict,
            # The inner folds are cut under the SAME discipline as the outer
            # ones: the spec's embargo, and a purge on each row's own label
            # end. They were cut with neither, so the candidate that won was
            # the one that scored best on training rows whose labels had
            # already seen the inner test window.
            embargo=model_spec.validation.embargo,
            label_end=(frame[LABEL_END_COL].to_numpy() if has_label_end else None),
            purge_basis=purge_basis,
            max_parallelism=budget,
            # The report says what the spec asked for: 'auto', not this
            # machine's count.
            reported_parallelism=budget_asked,
        )

    # How many folds may be fitted side by side, what limited it, and the
    # n_jobs each fold's estimators then get -- see `_fold_schedule`. One
    # worker is the loop as it always ran, with the budget's n_jobs.
    fold_workers, fold_limit = _fold_schedule(
        model_spec, estimator_cls, len(plan.folds), budget
    )
    fold_n_jobs = budget // fold_workers
    # Folds prepared and not skipped: the number the next completed fold
    # will carry. One at a time it is `len(fold_records)`; it is counted
    # separately so that it is the same number when the fits run later.
    n_ready = 0

    # A fold runs in three steps. `_prepare_fold` takes it up to its fit on
    # the calling thread, in fold order, so the skips, the purge count, the
    # fold cache and the refusals happen exactly as they did in one loop.
    # `_fit_fold` is the fit and everything read off it; it touches no
    # state shared between folds, so it is the part that can run beside
    # other folds. `_record_fold` appends the outcome, again in fold order,
    # so nothing recorded depends on which fit finished first.
    def _prepare_fold(fold: Any) -> "Dict[str, Any] | None":
        """One fold up to its fit, or None when the fold is skipped (and
        recorded in `skipped`)."""
        nonlocal n_purged_total, model_columns, n_ready
        train_dates = dates[fold.train_positions]
        test_dates = dates[fold.test_positions]
        in_train = np.zeros(len(dates), dtype=bool)
        in_train[fold.train_positions] = True
        in_test = np.zeros(len(dates), dtype=bool)
        in_test[fold.test_positions] = True
        train_mask = in_train[date_code]
        test_mask = in_test[date_code]

        # ── Target-overlap purge ──────────────────────────────────────────
        # A forward-return label on training row t is only finished once
        # bar t+horizon prints. With embargo < horizon that bar lies inside
        # the test window, so the row's LABEL is built from test-period
        # prices even though its FEATURES are entirely in the past. The
        # embargo alone never enforced this (WalkForwardSplit is not given
        # the horizon at all), so horizon=20/embargo=0 trained on 20 labels
        # that had already seen the test period.
        #
        # The rows are the PLAN's: a training row is purged when the bars
        # its label spans overlap a test block, decided per contiguous
        # block by walk_forward.label_overlap_mask -- the rule the inner
        # hyperparameter search shares -- on each row's own recorded
        # label_end_date, which also handles entities on different
        # calendars. Applied to the row mask in place so the fold is taken
        # ONCE; selecting and then dropping made a second full copy of the
        # training block, 11.5 ms on 100,000 rows.
        if fold.purged_rows is not None and fold.purged_rows.size:
            train_mask[fold.purged_rows] = False
            n_purged_total += int(fold.purged_rows.size)

        train_df = panel[train_mask]
        test_df = panel[test_mask]

        if train_df.empty or test_df.empty:
            skipped.append(
                {
                    "test_start": str(pd.Timestamp(test_dates[0]).date()),
                    "reason": (
                        "no training rows survived the target-overlap purge"
                        if train_df.empty and n_purged_total > 0
                        else "empty train or test slice"
                    ),
                }
            )
            return None

        # ── What each feature IS in this fold, before anything imputes ────
        # A column that is missing in EVERY training row of a fold has no
        # median; the impute step fell back to its constant and the fold
        # trained on a feature that does not exist there, while the run
        # reported full fold coverage and averaged that fold's importance
        # in with the rest (findings D18: five of eight folds, 8.87% of the
        # importance on a feature present in three). Refused by name, and
        # the per-fold rate recorded beside the averaged importance.
        fold_missing: "Dict[str, float] | None" = None
        if panel_has_missing:
            fold_missing = _training_missing_rates(
                feature_matrix[train_mask], feature_ids
            )
            _refuse_absent_features(fold_missing, n_ready, test_dates, model_spec)

        train_y = _labels(model_spec, train_df)
        test_y = _labels(model_spec, test_df)
        # A fold whose training window happens to land entirely on one
        # side of a binary target can't fit a classifier at all (sklearn
        # raises deep inside .fit()) -- skip it, same as an empty
        # train/test slice above, rather than letting the whole
        # experiment crash over one unlucky window.
        if model_spec.task == "classification" and len(np.unique(train_y)) < 2:
            skipped.append(
                {
                    "test_start": str(pd.Timestamp(test_dates[0]).date()),
                    "reason": "training window contained only one class",
                }
            )
            return None
        # The survival analogue: a window in which every row was censored
        # has no observed event and therefore no ordering to learn.
        if model_spec.task == "survival" and train_y[:, 1].sum() == 0:
            skipped.append(
                {
                    "test_start": str(pd.Timestamp(test_dates[0]).date()),
                    "reason": "training window contained no observed event",
                }
            )
            return None

        # From the cache when a run over the same dataset and fold has
        # fitted this pipeline already -- exactly, for a column-wise
        # pipeline, even when that run had more features than this one.
        cached = cache.lookup(fold.preprocessing_hash, feature_ids)
        if cached is None:
            train_X, test_X = _preprocess(
                model_spec,
                train_df,
                test_df,
                feature_ids,
                pd.DataFrame(
                    feature_matrix[train_mask],
                    index=train_df.index,
                    columns=feature_ids,
                ),
                pd.DataFrame(
                    feature_matrix[test_mask],
                    index=test_df.index,
                    columns=feature_ids,
                ),
            )
            cache.store(
                fold.preprocessing_hash,
                feature_ids,
                train_X,
                test_X,
                projectable=projectable,
            )
        else:
            train_X, test_X = cached
        if panel_has_missing and not estimator_accepts_missing:
            _refuse_missing_after_preprocessing(
                train_X, test_X, model_spec, "run_model_experiment"
            )
        # What each training row IS -- its date, entity and label end --
        # beside what it contains. The weights, the adapter and the
        # conformal calibration are defined on this, not on the frame.
        train_index = SampleIndex.from_frame(train_df)
        sample_weight = _fold_sample_weights(model_spec, train_index)
        fold_columns = list(train_X.columns)
        if model_columns is None:
            model_columns = fold_columns
        elif fold_columns != model_columns:
            raise ValidationError(
                "run_model_experiment: the preprocessing pipeline produced "
                f"{len(fold_columns)} columns on this fold and "
                f"{len(model_columns)} on an earlier one. A step whose output "
                "columns depend on the rows it was fitted on cannot be "
                "summarized across folds; every registered step emits a "
                "column set that depends only on its input columns."
            )
        n_ready += 1
        return {
            "fold": fold,
            "train_dates": train_dates,
            "test_dates": test_dates,
            "train_df": train_df,
            "test_df": test_df,
            "fold_missing": fold_missing,
            "train_y": train_y,
            "test_y": test_y,
            "train_X": train_X,
            "test_X": test_X,
            "train_index": train_index,
            "sample_weight": sample_weight,
            "fold_columns": fold_columns,
        }

    def _fit_fold(prepared: Dict[str, Any], n_jobs: int) -> Dict[str, Any]:
        """The fit of one prepared fold and everything read off it."""
        fold = prepared["fold"]
        train_df, test_df = prepared["train_df"], prepared["test_df"]
        train_X, test_X = prepared["train_X"], prepared["test_X"]
        train_y, test_y = prepared["train_y"], prepared["test_y"]

        fold_params = model_spec.estimator.params
        search_report = None
        if model_spec.search is not None:
            # The inner folds are a function of this outer fold and the
            # search's shape, not of the candidate, so their matrices are
            # keyed under the outer fold's hash and fitted once per inner
            # fold rather than once per candidate per inner fold.
            fold_params, search_report = _search_on(
                train_df,
                prefix=f"{fold.preprocessing_hash}/inner/{model_spec.search.inner_splits}/",
            )

        # The fold's n_jobs, or for an OpenMP estimator the threads its
        # runtime is held to -- one under 'auto' on a matrix this small.
        # Taken after the search, whose candidates set their own.
        fit_jobs, openmp_threads = _fit_threads(
            threads_kind,
            budget_asked,
            n_jobs,
            int(train_X.shape[0]) * int(train_X.shape[1]),
        )
        exact = openmp_threads is not None
        estimator = _instantiate(
            estimator_cls,
            fold_params,
            model_spec.random_seed,
            n_jobs=fit_jobs,
            exact_n_jobs=exact,
        )
        arrays = adapter.prepare(
            model_spec,
            prepared["train_index"],
            train_X,
            train_y,
            prepared["sample_weight"],
        )
        # Calibration is fitted INSIDE the training window, on folds held out
        # from it, so the map never sees a label the estimator memorized --
        # and never sees a test row at all.
        estimator = _calibrated(estimator, model_spec, len(arrays.y))
        with openmp_thread_limit(openmp_threads):
            _fit(
                estimator, arrays.X, arrays.y, arrays.sample_weight, group=arrays.group
            )

            metrics, prediction_values, fold_ic = _predict_fold(
                adapter,
                model_spec,
                estimator,
                test_X,
                test_y,
                datetime_values(test_df["date"]),
                train_y=train_y,
            )
            # ── The distribution beside the point ────────────────────────
            # One more fit per requested quantile, on the same rows, and a
            # conformal radius read off held-out date blocks inside this
            # training window; both become OOS columns beside `prediction`,
            # which is left exactly as the base fit produced it, and
            # metrics beside the point metrics.
            distribution_columns: Dict[str, np.ndarray] = {}
            quantile_values: Dict[float, np.ndarray] = {}
            if quantile is not None:
                for q, model in _fit_quantile_models(
                    estimator_cls,
                    quantile,
                    fold_params,
                    model_spec,
                    arrays,
                    n_jobs=fit_jobs,
                    exact_n_jobs=exact,
                ).items():
                    quantile_values[q] = np.asarray(model.predict(test_X.to_numpy()))
                    distribution_columns[quantile_column(q)] = quantile_values[q]
            lower = upper = None
            if model_spec.intervals is not None:
                radius, _n_calibration = _conformal_radius(
                    estimator_cls,
                    fold_params,
                    model_spec,
                    arrays,
                    n_jobs=fit_jobs,
                    exact_n_jobs=exact,
                )
                lower = np.asarray(prediction_values, dtype=float) - radius
                upper = np.asarray(prediction_values, dtype=float) + radius
                distribution_columns["lower"] = lower
                distribution_columns["upper"] = upper
        if distribution_columns:
            metrics.update(
                distributional_metrics(
                    test_y,
                    quantile_values,
                    lower=lower,
                    upper=upper,
                    alpha=(
                        float(model_spec.intervals.alpha)
                        if model_spec.intervals is not None
                        else None
                    ),
                )
            )
        return {
            "search_report": search_report,
            "metrics": metrics,
            "prediction_values": prediction_values,
            "fold_ic": fold_ic,
            "distribution_columns": distribution_columns,
            "importance": fold_feature_importance(estimator, prepared["fold_columns"]),
        }

    def _record_fold(prepared: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        """Append one fitted fold's outcome to the run's records."""
        fold = prepared["fold"]
        train_dates, test_dates = prepared["train_dates"], prepared["test_dates"]
        train_df, test_df = prepared["train_df"], prepared["test_df"]
        metrics = outcome["metrics"]
        if outcome["search_report"] is not None:
            search_reports.append(outcome["search_report"])
        # Every fold's per-date IC dates are kept so the OOS dispersion
        # statistics can be computed once over the pooled series -- see
        # aggregate_cross_sectional_ic for why averaging per-fold std/ICIR
        # is a different quantity.
        for ic_key, ic_values in outcome["fold_ic"].items():
            pooled_ic.setdefault(ic_key, []).append(ic_values)
        # Per-fold detail is retained, not only its contribution to the
        # average: one averaged number cannot show performance decay over
        # time, reveal which regime drove the result, or expose that a
        # single fold carried everything.
        fold_records.append(
            {
                "fold": len(fold_records),
                "train_start": str(pd.Timestamp(train_dates[0]).date()),
                # The date range actually FIT, after label-overlap purging.
                # This reported the scheduled window end, so a fold whose
                # last two weeks were entirely purged still claimed to have
                # trained through them -- lineage describing the split that
                # was planned rather than the one that ran.
                "train_end": str(pd.Timestamp(train_df["date"].max()).date()),
                # Kept alongside it: the difference between the two is
                # exactly how much the purge removed, which is worth being
                # able to see rather than having to infer.
                "scheduled_train_end": str(pd.Timestamp(train_dates[-1]).date()),
                "test_start": str(pd.Timestamp(test_dates[0]).date()),
                "test_end": str(pd.Timestamp(test_dates[-1]).date()),
                # The test BLOCKS, one per contiguous run of dates. Under cpcv
                # a fold's test set is several blocks with training dates
                # between them, and the start..end span above read as one
                # window: 1,912 rows against n_test_rows 1,304.
                "test_blocks": [
                    {
                        "start": str(pd.Timestamp(dates[first]).date()),
                        "end": str(pd.Timestamp(dates[last]).date()),
                    }
                    for first, last in contiguous_runs(fold.test_positions)
                ],
                "n_train_rows": int(len(train_df)),
                "n_test_rows": int(len(test_df)),
                "metrics": metrics,
                # Each feature's missing rate in the rows this fold trained
                # on; None when the panel carries no holes at all.
                "missing_rate_train": prepared["fold_missing"],
                # What determined this fold's estimator: dataset, rows,
                # pipeline, estimator, parameters, seed. Two runs that
                # agree here fitted the same thing.
                "node_hash": fold.node_hash,
            }
        )
        # Weight by out-of-sample prediction count -- see
        # average_fold_metrics for why equal weighting distorts the
        # headline number when coverage varies across folds.
        fold_weights.append(float(len(test_df)))
        tested_dates[fold.test_positions] = True
        fold_metrics.append(metrics)
        fold_importance.append(outcome["importance"])
        oos_frame = pd.DataFrame(
            {
                "date": test_df["date"].to_numpy(),
                "entity": test_df["entity"].to_numpy(),
                "prediction": outcome["prediction_values"],
                **outcome["distribution_columns"],
            }
        )
        if is_cpcv:
            # Under combinatorial CV a (date, entity) is predicted once per
            # path it was tested in, so the row is not unique without the
            # path that produced it. Additive: every consumer that reads the
            # three canonical columns still can, and the ones that need one
            # prediction per row refuse a cpcv model by name.
            oos_frame["path"] = len(fold_records) - 1
        oos_prediction_frames.append(oos_frame)

    if fold_workers == 1:
        for fold in plan.folds:
            prepared = _prepare_fold(fold)
            if prepared is not None:
                _record_fold(prepared, _fit_fold(prepared, fold_n_jobs))
    else:
        _run_folds_side_by_side(
            plan.folds,
            _prepare_fold,
            _fit_fold,
            _record_fold,
            fold_workers,
            fold_n_jobs,
        )

    if not fold_metrics:
        raise ValidationError(
            "run_model_experiment: every walk-forward fold was skipped -- either an empty "
            "train/test slice (the requested universe/date range likely doesn't cover every "
            "entity on every date), or, for classification, a fold whose training window "
            "landed entirely on one class of the binary target, or every training row was "
            "purged because its forward-return label overlapped the test window (raise "
            "train_window, or lower the target horizon)."
        )

    # A model used to be registered after a single surviving fold, which is
    # one train/test split rather than walk-forward validation -- it cannot
    # show whether performance holds across time. Enforced after the loop
    # (not from n_splits) because it is COMPLETED folds that matter: folds
    # skipped for a single-class window or an empty slice provide no
    # evidence.
    if len(fold_metrics) < model_spec.validation.min_folds:
        raise ValidationError(
            f"run_model_experiment: only {len(fold_metrics)} of {n_expected_folds} "
            f"walk-forward fold(s) completed, below min_folds="
            f"{model_spec.validation.min_folds}. "
            + (f"Skipped: {skipped}. " if skipped else "")
            + "Widen the date range, shorten train_window/test_window, or lower "
            "min_folds if a single split is genuinely what you want."
        )

    oos_metrics = average_fold_metrics(fold_metrics, fold_weights)
    # Recompute the cross-sectional IC dispersion statistics over the POOLED
    # OOS daily IC series, overwriting the fold-averaged versions.
    #
    # A weighted mean across folds is right for cs_ic_mean but wrong for
    # everything built on top of it: mean(fold stds) is not std(all OOS
    # daily ICs), and mean(fold ICIRs) is not mean(all ICs)/std(all ICs).
    # Averaging folds' stds throws away the BETWEEN-fold variation, which
    # is precisely the variation ICIR exists to measure -- so a model whose
    # IC was stable inside each fold but swung between them scored as
    # dependable. The per-fold numbers remain in validation_report, where
    # they answer the different question of how each fold did.
    # The pooled per-date series behind the headline, kept for its test
    # against the null: the series the headline's mean was taken over.
    headline_series: "pd.Series | None" = None
    for prefix, series_list in pooled_ic.items():
        if is_cpcv:
            # Paths are NOT disjoint in time: a date is tested in several,
            # so a plain concat would count it once per path and the
            # dispersion would mix across-path spread into the across-date
            # spread ICIR exists to measure. Each date's IC is averaged
            # across the paths that tested it first, then summarized once.
            merged = pd.concat(series_list)
            per_date = merged.groupby(level=0).mean().sort_index()
            oos_metrics.update(summarize_cross_sectional_ic(per_date, prefix))
            pooled = per_date
        else:
            oos_metrics.update(aggregate_cross_sectional_ic(series_list, prefix))
            # The concatenation `aggregate_cross_sectional_ic` summarizes.
            usable = [s for s in series_list if s is not None and not s.empty]
            pooled = pd.concat(usable).sort_index() if usable else None
        if prefix == adapter.headline_series:
            headline_series = pooled
    importance_summary = summarize_importance(fold_importance, model_columns or [])

    # Sample size discounted for target overlap. A `horizon`-bar forward
    # return generated every bar produces labels sharing horizon-1 of their
    # bars, so the raw OOS row count overstates the independent evidence
    # behind every metric above -- often by an order of magnitude.
    n_oos_rows = int(sum(fold_weights))
    if is_cpcv:
        # Once per row, not once per path: a row tested in five paths is
        # one observation of the world, however many fits looked at it.
        n_oos_rows = int(
            pd.concat(oos_prediction_frames)[["date", "entity"]]
            .drop_duplicates()
            .shape[0]
        )
    horizon = _target_horizon(dataset.get("target_id"))
    oos_metrics["n_oos_rows"] = float(n_oos_rows)
    # And discounted again across entities: rows on one date are one
    # cluster, and a panel of names that move together holds fewer
    # independent observations than rows / horizon -- as few as dates /
    # horizon. The entity count was passed here and cancelled out of the
    # arithmetic, so the count assumed independent entities without
    # saying so; the report below says what was measured and assumed.
    ess_report = _oos_effective_sample_size(
        panel, tested_dates[date_code], n_oos_rows, int(tested_dates.sum()), horizon
    )
    oos_metrics["effective_sample_size"] = ess_report["value"]
    if horizon is None:
        run_warnings.append(
            f"target_id {dataset.get('target_id')!r} names no horizon, so "
            "effective_sample_size is not discounted for label overlap along "
            "time -- only for the labels' correlation across entities. For an "
            "h-bar label it overstates the independent observations by up to "
            "a factor of h."
        )

    # ── The headline against what no skill scores ─────────────────────
    # The metric compare_models and list_models rank this task by, tested
    # against its null; a warning when it does not beat it.
    headline_block, headline_warnings = _headline_report(
        adapter, model_spec.task, oos_metrics, headline_series, horizon
    )
    run_warnings.extend(headline_warnings)
    # Explanations of numbers that read as failures and are not: an r2
    # below its baseline beside a positive rank IC, and an importance block
    # that is null by construction. Apart from `warnings`, which say the
    # run did something that changes how a number should be read.
    run_notes: List[str] = []
    run_notes.extend(
        _r2_note(oos_metrics, fold_metrics, fold_weights, adapter.headline)
    )
    run_notes.extend(
        _importance_note(
            model_spec, estimator_cls, importance_summary, adapter.headline
        )
    )

    paths_report = None
    if is_cpcv:
        # The distribution across paths IS the result combinatorial CV
        # exists to produce: a fifth percentile and a median of the OOS
        # metric rather than one draw of it.
        distribution: Dict[str, Dict[str, float]] = {}
        for key in (
            "cs_rank_ic_mean",
            "cs_ic_mean",
            "r2",
            "mae",
            "auc",
            "accuracy",
            "ndcg_at_5",
            "ndcg_at_10",
            "concordance",
            "integrated_brier",
        ):
            values = np.array([m.get(key, np.nan) for m in fold_metrics], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                distribution[key] = {
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "std": (
                        float(values.std(ddof=1)) if values.size > 1 else float("nan")
                    ),
                    "min": float(values.min()),
                    "p05": float(np.percentile(values, 5)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "max": float(values.max()),
                }
        paths_report = {
            "n_paths": len(fold_metrics),
            "n_groups": int(model_spec.validation.n_splits),
            "n_test_groups": int(model_spec.validation.n_test_splits),
            "ic_pooling": "mean per date across paths",
            "metric_distribution": distribution,
        }

    # ── The deployed parameters ───────────────────────────────────────
    # The refit instantiated from `model_spec.estimator.params` -- the BASE
    # values -- while each fold had reassigned `fold_params` from its own
    # inner search, which never reached the refit, the quantile models or
    # the conformal radius. So a ridge searched over {0.001, 100, 10000}
    # was deployed at alpha=1.0, a value not in the grid, and a forest
    # searched over max_depth {6, 8} was deployed at the base depth of 1;
    # the deployed forest's predictions correlated with the correctly
    # refitted ones at Spearman 0.30 (findings D14). The rule this file
    # states for weighting and for preprocessing applies to parameters
    # too: the deployed pipeline is the validated pipeline.
    #
    # One final search on the FULL panel, under the same purge and embargo
    # the folds searched under, chooses the deployed values; the plan
    # counted its fits and refused them over budget like any other. When
    # the panel cannot support the inner folds (it always can when any
    # fold could) the last searched fold's choice is deployed, and when
    # nothing searched, the spec's. One variable feeds all three call
    # sites below, and the manifest says which of the three it was.
    deployed_params: Dict[str, Any] = dict(model_spec.estimator.params)
    deployed_params_source = "spec"
    final_search_report: "Dict[str, Any] | None" = None
    if model_spec.search is not None:
        deployed_params, final_search_report = _search_on(
            panel, prefix=f"{plan.final_search_hash}/final/"
        )
        if final_search_report.get("searched"):
            deployed_params_source = "full_panel_search"
        else:
            last_searched = next(
                (r for r in reversed(search_reports) if r.get("searched")), None
            )
            if last_searched is not None:
                deployed_params = {
                    **model_spec.estimator.params,
                    **last_searched["best_params"],
                }
                deployed_params_source = "last_fold"
    # Per feature, the missing rate in each completed fold's training
    # rows -- the number that says whether an averaged importance was
    # averaged over folds that actually had the feature (findings D18).
    missing_rate_by_fold = (
        {
            feature: [record["missing_rate_train"][feature] for record in fold_records]
            for feature in feature_ids
        }
        if panel_has_missing
        else None
    )
    # How wide the cross-sections the model was fitted on were. A
    # cross-sectional transform standardizes within whatever rows a
    # scoring call contains, so this is what a scoring width is judged
    # against (findings D15).
    per_date_width = panel.groupby("date")["entity"].nunique()
    training_cross_section = {
        "min": int(per_date_width.min()),
        "median": float(per_date_width.median()),
        "max": int(per_date_width.max()),
        "n_dates": int(len(per_date_width)),
    }

    validation_report = {
        "method": model_spec.validation.method,
        # cpcv only: the metric distribution across paths, which is the
        # number to select a model on; None for the other methods.
        "paths": paths_report,
        "scheme": (
            model_spec.validation.scheme
            if model_spec.validation.method == "walk_forward"
            else None
        ),
        "normalization": model_spec.preprocessing.normalization,
        # The resolved pipeline, by step id. `normalization` alone cannot
        # describe an explicit `steps` spec, for which it reads 'pooled'.
        "preprocessing_steps": step_types(model_spec.preprocessing.resolved_steps),
        "weighting": model_spec.weighting.method,
        # Per fold, so a reader can see whether the search settled on the
        # same parameters every time or picked something different each
        # fold. The second is the useful signal: it means the search was
        # fitting noise, and an averaged "best alpha" would have hidden it.
        "hyperparameter_search": search_reports or None,
        # The parameters the DEPLOYED estimator was refit with, where they
        # came from, and the full-panel search that chose them -- beside
        # the per-fold selections above, so a reader can see whether the
        # deployed choice agrees with what the folds validated.
        "deployed_params": deployed_params,
        "deployed_params_source": deployed_params_source,
        "final_search": final_search_report,
        "missing_rate_by_fold": missing_rate_by_fold,
        "n_folds_expected": int(n_expected_folds),
        "n_folds_completed": len(fold_metrics),
        "n_folds_skipped": len(skipped),
        "fold_coverage": (
            round(len(fold_metrics) / n_expected_folds, 4) if n_expected_folds else 0.0
        ),
        "skipped_folds": skipped,
        # The purge runs on each row's label end: the recorded one
        # ('label_end'), or one derived from the target's horizon when the
        # panel carries none ('label_end_derived_from_horizon', with a
        # warning). It used to report 'not_applicable' and purge nothing on
        # such a panel, although the horizon was known; a panel with
        # neither is now refused before a fold is cut.
        "purge": purge_basis if has_label_end else "not_applicable",
        "n_train_rows_purged_overlap": n_purged_total if has_label_end else None,
        "target_horizon": horizon,
        # The count behind every metric above, the two bounds it lies
        # between and what placed it there: the overlap along time and the
        # labels' correlation across entities.
        "effective_sample_size": ess_report,
        # What the plan said this would cost, against the ceiling it was
        # checked against. A fold skipped at run time cost less than
        # planned; nothing costs more.
        "fits": {
            "planned": plan.n_fits,
            "folds": plan.n_fits_folds,
            "refit": plan.n_fits_refit,
            "final_search": plan.n_fits_final_search,
            "candidates_per_fold": plan.n_candidates,
            "max_fits": plan.max_fits,
            # As the spec asked for it: a whole number, or 'auto'.
            "max_parallelism": budget_asked,
            **_reported_fold_schedule(budget_asked, fold_workers, fold_limit),
        },
        # Pipeline fits this run did and did not have to do: `misses` were
        # fitted here, `hits` were read off an earlier candidate of this
        # run's own inner search. A run given a shared cache also reports
        # `projections`, the folds read off a wider run's matrices -- a
        # column-wise pipeline's, exact by construction -- and
        # `projectable`, whether this pipeline allows that. A private cache
        # never projects: its runs have one feature set, so the two were
        # reported as `projections: 0` beside `projectable: true` on every
        # run_model_experiment call, which read as a reuse that failed.
        "cache": _cache_report(
            cache.stats(), cache_before, fold_cache is not None, projectable
        ),
        # The headline metric against what a model with no skill scores,
        # with the test that compared them; see `_headline_report`.
        "headline": headline_block,
        # Where feature_importance_summary came from: 'coefficients',
        # 'feature_importances' or 'none'.
        "importance_source": _importance_source(fold_importance),
        "folds": fold_records,
    }

    # Walk-forward folds are for out-of-sample validation only; the
    # registered/deployed model is refit on the full panel so it uses
    # every available observation, the same "validate on folds, deploy
    # on everything" convention real factor-model practice uses.
    # One take of the whole panel, not two. The fused helper is not used
    # here because these statistics are persisted into the manifest, and it
    # returns only the transformed frames.
    #
    # UNDER THE SAME TRANSFORM THE FOLDS USED. This did not branch: it
    # fitted the pooled winsorize/zscore statistics whatever
    # `preprocessing.normalization` said, persisted them, and score_model
    # applied them -- so a model validated under `cross_sectional` was
    # deployed on a transform it was never validated on, with nothing in
    # the manifest to show it. Measured on a six-entity panel, ridge, three
    # features: the deployed estimator's predictions under the two
    # transforms agreed at Spearman 0.84. This is the weighting mistake
    # recorded below, one field over, and the same rule applies: the
    # deployed pipeline is the validated pipeline. A cross-sectional model
    # fits nothing per column, so its persisted statistics are empty and
    # the manifest's `preprocessing` field says which transform to apply.
    full_features = panel[feature_ids]
    # The same pipeline the folds ran, fitted once on the whole panel. Its
    # state is what gets persisted and what scoring applies; the legacy
    # per-column statistics file is projected from it for one release so an
    # older reader of `preprocessing_stats.json` keeps loading.
    full_state, full_X = fit_pipeline(
        model_spec.preprocessing.resolved_steps,
        full_features,
        FoldContext.from_frame(panel),
    )
    full_stats = legacy_stats(full_state)
    if model_columns is not None and list(full_X.columns) != model_columns:
        raise ValidationError(
            "run_model_experiment: the preprocessing pipeline produced "
            f"{list(full_X.columns)[:6]} on the full-panel refit and "
            f"{model_columns[:6]} on the folds. The deployed estimator would "
            "be fitted on different columns than the ones that were validated."
        )
    full_y = _labels(model_spec, panel)
    # The refit runs alone, after the folds, on the whole budget: a forest
    # builds its trees on it and is put back on one thread before it
    # predicts or is written down, so the deployed model carries no budget.
    refit_jobs, refit_openmp_threads = _fit_threads(
        threads_kind,
        budget_asked,
        budget,
        int(full_X.shape[0]) * int(full_X.shape[1]),
    )
    refit_exact = refit_openmp_threads is not None
    final_estimator = _instantiate(
        estimator_cls,
        deployed_params,
        model_spec.random_seed,
        n_jobs=refit_jobs,
        exact_n_jobs=refit_exact,
    )
    # The deployed estimator is calibrated the same way the folds were. A
    # model validated with calibrated probabilities and deployed without
    # would report one threshold's behaviour and exhibit another's.
    final_estimator = _calibrated(final_estimator, model_spec, len(full_y))
    # Refit through the SAME adapter the folds used. The deployed model is
    # what actually scores, so fitting it differently from the one that was
    # validated is the quietest way to make a validation number describe
    # something else -- for a ranker that would mean no grading and no
    # grouping at all.
    # WEIGHTED THE SAME WAY THE FOLDS WERE. This passed None, so a model
    # validated under weighting.method='time_decay' was DEPLOYED
    # UNWEIGHTED while the manifest still recorded the weighted config --
    # which is precisely what the comment above says not to do. On a
    # regime-switching panel the two fits disagreed in sign on 4 of 10
    # as-of predictions, Spearman 0.75.
    #
    # `_fold_sample_weights` needs only `date`, `entity` and optionally the
    # label-end column, all of which the full panel carries, so this is the
    # same function the folds call rather than a second weighting path.
    full_index = SampleIndex.from_frame(panel)
    full_weights = _fold_sample_weights(model_spec, full_index)
    full_arrays = adapter.prepare(model_spec, full_index, full_X, full_y, full_weights)
    with openmp_thread_limit(refit_openmp_threads):
        _fit(
            final_estimator,
            full_arrays.X,
            full_arrays.y,
            full_arrays.sample_weight,
            group=full_arrays.group,
        )
        # The deployed distribution, fitted the way the folds' were: one
        # quantile model per level on the full panel, and a conformal
        # radius read off held-out blocks of it. Persisted with the model
        # so scoring emits the same columns the validation reported on.
        distribution_state: "Dict[str, Any] | None" = None
        quantile_models: "Dict[str, Any] | None" = None
        if quantile is not None or model_spec.intervals is not None:
            distribution_state = {
                "quantiles": [float(q) for q in model_spec.quantiles],
                "columns": {quantile_column(q): float(q) for q in model_spec.quantiles},
                "conformal": None,
            }
            if quantile is not None:
                fitted = _fit_quantile_models(
                    estimator_cls,
                    quantile,
                    deployed_params,
                    model_spec,
                    full_arrays,
                    n_jobs=refit_jobs,
                    exact_n_jobs=refit_exact,
                )
                quantile_models = {quantile_column(q): m for q, m in fitted.items()}
            if model_spec.intervals is not None:
                radius, n_calibration = _conformal_radius(
                    estimator_cls,
                    deployed_params,
                    model_spec,
                    full_arrays,
                    n_jobs=refit_jobs,
                    exact_n_jobs=refit_exact,
                )
                distribution_state["conformal"] = {
                    "method": model_spec.intervals.method,
                    "alpha": float(model_spec.intervals.alpha),
                    "calibration_folds": int(model_spec.intervals.calibration_folds),
                    "radius": float(radius),
                    "n_calibration": int(n_calibration),
                }

    # model_id generated here (not left to save_model's own default)
    # so the OOS predictions artifact lands in the same
    # SQT_RUNS_DIR/<model_id>/ directory as the model's other files --
    # these predictions are the leakage-safe source
    # modeling.bridge.oos_predictions_to_signal_panel needs to turn a
    # model into an actual strategy backtest (never score_model, whose
    # single as-of snapshot comes from this same final_estimator and
    # would leak if used to "predict" dates it was trained on).
    model_id = new_model_id()
    if not register:
        # Everything above has already happened: the folds are fit, the OOS
        # metrics are computed, the importance is summarized. What is
        # skipped is writing any of it down.
        return {
            "model_id": None,
            "oos_metrics": oos_metrics,
            "feature_importance_summary": importance_summary,
            "model_input_columns": model_columns,
            "n_folds": len(fold_metrics),
            "validation_report": validation_report,
            "oos_predictions_uri": None,
            "n_train_rows_purged_overlap": (n_purged_total if has_label_end else None),
            "warnings": list(run_warnings),
            "notes": list(run_notes),
        }

    oos_predictions_df = pd.concat(oos_prediction_frames, ignore_index=True)
    oos_predictions_uri = _artifacts.save_artifact(
        oos_predictions_df, run_id=model_id, name="oos_predictions"
    )
    # What `monitor_model` will compare a scored universe against: a
    # seeded sample of the RAW feature rows the model was trained on and a
    # sample of its out-of-sample predictions, plus a profile per feature.
    # The reference window's own values, so a later window cannot move the
    # edges it is measured by.
    monitoring_profile = feature_profile(panel, feature_ids)
    feature_reference_frame = reference_sample(
        panel, ["date", "entity", *feature_ids], seed=model_spec.random_seed
    )
    prediction_reference_frame = reference_sample(
        oos_predictions_df,
        ["date", "entity", "prediction"],
        seed=model_spec.random_seed,
    )

    manifest = save_model(
        estimator=final_estimator,
        model_spec=model_spec,
        feature_ids=feature_ids,
        target_id=dataset["target_id"],
        dataset_id=dataset_id,
        dataset_hash=dataset["data_hash"],
        dataset_hash_version=dataset.get("data_hash_version"),
        oos_metrics=oos_metrics,
        feature_importance_summary=importance_summary,
        n_folds=len(fold_metrics),
        validation_report=validation_report,
        preprocessing_stats=full_stats,
        # The fitted pipeline state the deployed estimator expects, and the
        # RESOLVED step list in the manifest so a reader sees what ran
        # rather than the scheme that implied it.
        preprocessing_state=full_state,
        preprocessing=model_spec.preprocessing.resolved_dump(),
        model_input_columns=model_columns,
        oos_predictions_uri=oos_predictions_uri,
        model_id=model_id,
        distribution=distribution_state,
        quantile_models=quantile_models,
        feature_profile=monitoring_profile,
        feature_reference=feature_reference_frame,
        prediction_reference=prediction_reference_frame,
        # The last FEATURE date in the training panel. Kept for lineage, but
        # deliberately NOT the cutoff score_model gates on -- see below.
        train_end_date=pd.Timestamp(panel["date"].max()).strftime("%Y-%m-%d"),
        # The last date whose PRICES the training data actually consumed.
        #
        # A row dated t with a horizon-h forward-return target reads
        # Close[t+h] to build its label, so the deployed estimator (refit on
        # the whole panel) has indirectly seen prices through
        # max(label_end_date), not max(date). Those differ by the horizon --
        # ~28 calendar days for h=20 -- and gating on max(date) left exactly
        # that window open: score_model would accept an as_of the model had
        # already consumed the future of, returning a future-trained
        # prediction that looks point-in-time. Falls back to max(date) only
        # for a panel with no label_end_date column (datasets built before it
        # existed), which is the old, weaker guarantee rather than none.
        # The later of the two: a label end derived from the horizon is NaT
        # on every row of an entity shorter than the horizon, and a NaT
        # maximum has no date to write.
        training_information_cutoff=max(
            pd.Timestamp(value)
            for value in (
                panel[LABEL_END_COL].max() if has_label_end else pd.NaT,
                panel["date"].max(),
            )
            if pd.notna(value)
        ).strftime("%Y-%m-%d"),
        # Copied into the model directory so the model is self-contained:
        # scoring must not depend on the dataset directory still existing,
        # or on its spec file not having been edited since training.
        dataset_spec=dataset.get("dataset_spec"),
        dataset_spec_hash=dataset.get("spec_hash"),
        dataset_spec_hash_version=dataset.get("spec_hash_version"),
        # Carried from the dataset build onto the model: survivorship,
        # revised history, partial coverage and interval caveats belong
        # next to the OOS metrics they qualify, not only in the
        # build_model_dataset response the caller may never look at again.
        dataset_warnings=dataset.get("warnings"),
        # The parameters the deployed estimator actually carries, and
        # where they came from -- the manifest's `estimator_params` are
        # these, not the spec's base values (findings D14).
        deployed_params=deployed_params,
        deployed_params_source=deployed_params_source,
        # Which feed each entity's bars came from (findings D16): two
        # models built from the same spec on two feeds differed by 22%
        # on the headline metric with identical recorded identity.
        data_sources=dataset.get("data_sources"),
        training_cross_section=training_cross_section,
    )

    return {
        "model_id": manifest.model_id,
        "oos_metrics": oos_metrics,
        "feature_importance_summary": importance_summary,
        "model_input_columns": model_columns,
        "n_folds": len(fold_metrics),
        "validation_report": validation_report,
        "oos_predictions_uri": oos_predictions_uri,
        # Surfaced rather than silently applied: a large purge count means
        # the target horizon is consuming a real fraction of each training
        # window, which is information the caller needs when reading the
        # OOS metrics.
        "n_train_rows_purged_overlap": (n_purged_total if has_label_end else None),
        # Caveats about the run itself, as opposed to the dataset's (which
        # travel on the manifest as `dataset_warnings`): what calibration
        # costs the importances, a derived label end, a costly estimator
        # on a budget of one, a headline that does not beat its null.
        "warnings": list(run_warnings),
        # What a number that reads as a failure means: an r2 below its
        # baseline, an importance block null by construction.
        "notes": list(run_notes),
    }
