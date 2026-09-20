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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from . import artifacts as _artifacts
from .adapters import accepts_missing, get_adapter
from .cache import FoldCache, column_wise_pipeline
from .dataset.alignment import LABEL_END_COL
from .estimators.registry import (
    get_estimator_class,
    quantile_estimators,
    quantile_support,
    validate_params,
)
from .plan import plan_experiment
from .preprocessing import (
    FoldContext,
    fit_and_apply_pipeline,
    fit_pipeline,
    legacy_stats,
    step_types,
)
from .registry.model_registry import new_model_id, save_model
from .specs import TASKS, ModelSpec, targets_for_task
from .validation.conformal import conformal_radius, held_out_residuals
from .validation.diagnostics import fold_feature_importance, summarize_importance
from .validation.distributional import distributional_metrics, quantile_column
from .validation.metrics import (
    aggregate_cross_sectional_ic,
    average_fold_metrics,
    classification_metrics,
    cross_sectional_ic,
    effective_sample_size,
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
from .validation.search import require_optuna, search_best_params
from .validation.walk_forward import build_splitter
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


def _instantiate(cls: Any, params: Dict[str, Any], random_seed: int) -> Any:
    sig = inspect.signature(cls.__init__)
    kwargs = dict(params)
    if "random_state" in sig.parameters:
        kwargs["random_state"] = random_seed
    return cls(**kwargs)


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


def _fold_sample_weights(
    model_spec: ModelSpec, train_frame: pd.DataFrame
) -> "np.ndarray | None":
    """Training-row weights for one fold, or None for an unweighted fit."""
    if model_spec.weighting.method == "none":
        return None
    label_end = (
        train_frame[LABEL_END_COL].to_numpy()
        if LABEL_END_COL in train_frame.columns
        else None
    )
    return build_sample_weights(
        model_spec.weighting.method,
        train_frame["date"].to_numpy(),
        label_end,
        train_frame["entity"].to_numpy(),
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
    """
    kwargs: Dict[str, Any] = {}
    if weights is not None:
        kwargs["sample_weight"] = weights
    if group is not None:
        kwargs["group"] = group
    if not kwargs:
        estimator.fit(X, y)
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


def _fit_quantile_models(
    estimator_cls: Any,
    support: Any,
    params: Dict[str, Any],
    model_spec: ModelSpec,
    arrays: Any,
) -> Dict[float, Any]:
    """
    One estimator per requested quantile, fitted on the same rows and
    weights as the point estimator, with the registry's quantile parameter
    set and its fixed objective switched on. The point estimator is left
    exactly as it was: `prediction` is the base fit, and the quantiles
    stand beside it.
    """
    models: Dict[float, Any] = {}
    for q in model_spec.quantiles:
        quantile_params = {**params, **support.fixed, support.param: float(q)}
        model = _instantiate(estimator_cls, quantile_params, model_spec.random_seed)
        _fit(model, arrays.X, arrays.y, arrays.sample_weight)
        models[float(q)] = model
    return models


def _conformal_radius(
    estimator_cls: Any,
    params: Dict[str, Any],
    model_spec: ModelSpec,
    arrays: Any,
    frame: pd.DataFrame,
) -> "tuple[float, int]":
    """
    The split-conformal radius for one training window: absolute
    residuals on held-out date blocks, the estimator refit without each
    under the embargo and the label purge, and their (1 - alpha) quantile.
    Returns (radius, number of residuals it was read from).
    """
    intervals = model_spec.intervals
    assert intervals is not None
    weights = arrays.sample_weight

    def fit_predict(train_mask, test_mask):
        model = _instantiate(estimator_cls, params, model_spec.random_seed)
        _fit(
            model,
            arrays.X[train_mask],
            arrays.y[train_mask],
            weights[train_mask] if weights is not None else None,
        )
        return arrays.y[test_mask], np.asarray(model.predict(arrays.X[test_mask]))

    residuals = held_out_residuals(
        fit_predict,
        frame["date"].to_numpy(),
        frame[LABEL_END_COL].to_numpy() if LABEL_END_COL in frame.columns else None,
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
    _check_task_target_compatibility(model_spec.task, dataset.get("target_id"))
    if model_spec.task == "classification":
        _validate_classification_target(panel)
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
    # (purged K-fold) -- an interval slice would not.
    date_code = np.searchsorted(dates.to_numpy(), panel["date"].to_numpy())
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
    for fold in plan.folds:
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
            continue

        train_y = train_df["target"].to_numpy()
        test_y = test_df["target"].to_numpy()
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
            continue

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
        sample_weight = _fold_sample_weights(model_spec, train_df)
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

        fold_params = model_spec.estimator.params
        if model_spec.search is not None:
            # The inner folds are a function of this outer fold and the
            # search's shape, not of the candidate, so their matrices are
            # keyed under the outer fold's hash and fitted once per inner
            # fold rather than once per candidate per inner fold.
            inner_prefix = (
                f"{fold.preprocessing_hash}/inner/{model_spec.search.inner_splits}/"
            )

            def _fit_predict(params, inner_train, inner_test, fold_index):
                """Score one candidate the way the real fit will run it —
                same preprocessing, same weighting — so the search cannot
                select for a pipeline that is never used."""
                inner_key = f"{inner_prefix}{fold_index}"
                matrices = cache.lookup(inner_key, feature_ids)
                if matrices is None:
                    matrices = _preprocess(
                        model_spec, inner_train, inner_test, feature_ids
                    )
                    cache.store(
                        inner_key, feature_ids, *matrices, projectable=projectable
                    )
                inner_train_X, inner_test_X = matrices
                candidate = _instantiate(estimator_cls, params, model_spec.random_seed)
                inner_arrays = adapter.prepare(
                    model_spec,
                    inner_train,
                    inner_train_X,
                    inner_train["target"].to_numpy(),
                    _fold_sample_weights(model_spec, inner_train),
                )
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
                probabilities = (
                    predictions if model_spec.task == "classification" else None
                )
                return predictions, probabilities

            fold_params, search_report = search_best_params(
                task=model_spec.task,
                search_spec=model_spec.search,
                base_params=model_spec.estimator.params,
                train_frame=train_df,
                feature_ids=feature_ids,
                random_seed=model_spec.random_seed,
                fit_predict=_fit_predict,
                # The inner folds are cut under the SAME discipline as the
                # outer ones: the spec's embargo, and a purge on each row's
                # own label end. They were cut with neither, so the
                # candidate that won was the one that scored best on
                # training rows whose labels had already seen the inner
                # test window.
                embargo=model_spec.validation.embargo,
                label_end=(
                    train_df[LABEL_END_COL].to_numpy() if has_label_end else None
                ),
            )
            search_reports.append(search_report)

        estimator = _instantiate(estimator_cls, fold_params, model_spec.random_seed)
        arrays = adapter.prepare(model_spec, train_df, train_X, train_y, sample_weight)
        # Calibration is fitted INSIDE the training window, on folds held out
        # from it, so the map never sees a label the estimator memorized --
        # and never sees a test row at all.
        estimator = _calibrated(estimator, model_spec, len(arrays.y))
        _fit(estimator, arrays.X, arrays.y, arrays.sample_weight, group=arrays.group)

        metrics, prediction_values, fold_ic = _predict_fold(
            adapter,
            model_spec,
            estimator,
            test_X,
            test_y,
            test_df["date"].to_numpy(),
            train_y=train_y,
        )
        # ── The distribution beside the point ────────────────────────────
        # One more fit per requested quantile, on the same rows, and a
        # conformal radius read off held-out date blocks inside this
        # training window; both become OOS columns beside `prediction`,
        # which is left exactly as the base fit produced it, and metrics
        # beside the point metrics.
        distribution_columns: Dict[str, np.ndarray] = {}
        quantile_values: Dict[float, np.ndarray] = {}
        if quantile is not None:
            for q, model in _fit_quantile_models(
                estimator_cls, quantile, fold_params, model_spec, arrays
            ).items():
                quantile_values[q] = np.asarray(model.predict(test_X.to_numpy()))
                distribution_columns[quantile_column(q)] = quantile_values[q]
        lower = upper = None
        if model_spec.intervals is not None:
            radius, _n_calibration = _conformal_radius(
                estimator_cls, fold_params, model_spec, arrays, train_df
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
        # Every fold's per-date IC dates are kept so the OOS dispersion
        # statistics can be computed once over the pooled series -- see
        # aggregate_cross_sectional_ic for why averaging per-fold std/ICIR
        # is a different quantity.
        for ic_key, ic_values in fold_ic.items():
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
                "n_train_rows": int(len(train_df)),
                "n_test_rows": int(len(test_df)),
                "metrics": metrics,
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
        fold_metrics.append(metrics)
        fold_importance.append(fold_feature_importance(estimator, fold_columns))
        oos_frame = pd.DataFrame(
            {
                "date": test_df["date"].to_numpy(),
                "entity": test_df["entity"].to_numpy(),
                "prediction": prediction_values,
                **distribution_columns,
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
        else:
            oos_metrics.update(aggregate_cross_sectional_ic(series_list, prefix))
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
    n_entities = int(panel["entity"].nunique())
    oos_metrics["n_oos_rows"] = float(n_oos_rows)
    oos_metrics["effective_sample_size"] = (
        effective_sample_size(n_oos_rows, horizon, n_entities)
        if horizon is not None
        else float(n_oos_rows)
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
        "n_folds_expected": int(n_expected_folds),
        "n_folds_completed": len(fold_metrics),
        "n_folds_skipped": len(skipped),
        "fold_coverage": (
            round(len(fold_metrics) / n_expected_folds, 4) if n_expected_folds else 0.0
        ),
        "skipped_folds": skipped,
        "n_train_rows_purged_overlap": n_purged_total,
        "target_horizon": horizon,
        # What the plan said this would cost, against the ceiling it was
        # checked against. A fold skipped at run time cost less than
        # planned; nothing costs more.
        "fits": {
            "planned": plan.n_fits,
            "folds": plan.n_fits_folds,
            "refit": plan.n_fits_refit,
            "candidates_per_fold": plan.n_candidates,
            "max_fits": plan.max_fits,
        },
        # Pipeline fits this run did and did not have to do: `misses` were
        # fitted here, `hits` and `projections` were read off an earlier
        # fit -- of this run's inner search, or of a run that shared the
        # cache. A projection is a column-wise pipeline's matrix read for
        # a feature subset, exact by construction.
        "cache": {
            key: cache.stats()[key] - cache_before[key]
            for key in ("hits", "misses", "projections")
        }
        | {"shared": fold_cache is not None, "projectable": projectable},
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
    full_y = panel["target"].to_numpy()
    final_estimator = _instantiate(
        estimator_cls, model_spec.estimator.params, model_spec.random_seed
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
    full_weights = _fold_sample_weights(model_spec, panel)
    full_arrays = adapter.prepare(model_spec, panel, full_X, full_y, full_weights)
    _fit(
        final_estimator,
        full_arrays.X,
        full_arrays.y,
        full_arrays.sample_weight,
        group=full_arrays.group,
    )
    # The deployed distribution, fitted the way the folds' were: one
    # quantile model per level on the full panel, and a conformal radius
    # read off held-out blocks of it. Persisted with the model so scoring
    # emits the same columns the validation reported on.
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
                model_spec.estimator.params,
                model_spec,
                full_arrays,
            )
            quantile_models = {quantile_column(q): m for q, m in fitted.items()}
        if model_spec.intervals is not None:
            radius, n_calibration = _conformal_radius(
                estimator_cls,
                model_spec.estimator.params,
                model_spec,
                full_arrays,
                panel,
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
            "n_train_rows_purged_overlap": n_purged_total,
        }

    oos_predictions_df = pd.concat(oos_prediction_frames, ignore_index=True)
    oos_predictions_uri = _artifacts.save_artifact(
        oos_predictions_df, run_id=model_id, name="oos_predictions"
    )

    manifest = save_model(
        estimator=final_estimator,
        model_spec=model_spec,
        feature_ids=feature_ids,
        target_id=dataset["target_id"],
        dataset_id=dataset_id,
        dataset_hash=dataset["data_hash"],
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
        training_information_cutoff=pd.Timestamp(
            panel[LABEL_END_COL].max() if has_label_end else panel["date"].max()
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
        "n_train_rows_purged_overlap": n_purged_total,
    }
