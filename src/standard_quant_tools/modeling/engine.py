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
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from . import artifacts as _artifacts
from .adapters import get_adapter
from .dataset.alignment import LABEL_END_COL
from .estimators.registry import get_estimator_class, validate_params
from .features.transforms import (
    apply_preprocessing,
    fit_and_apply_preprocessing,
    fit_preprocessing,
    standardize_cross_sectional,
)
from .registry.model_registry import new_model_id, save_model
from .specs import TASKS, ModelSpec, targets_for_task
from .validation.diagnostics import fold_feature_importance, summarize_importance
from .validation.metrics import (
    aggregate_cross_sectional_ic,
    average_fold_metrics,
    classification_metrics,
    cross_sectional_ic,
    effective_sample_size,
    positive_class_proba,
    regression_metrics,
)
from .validation.ranking import (
    fold_ic_series,
    group_sizes,
    ranking_metrics,
    relevance_grades,
)
from .validation.search import search_best_params
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
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Normalize the feature columns for one fold.

    The two modes differ in more than their arithmetic. Pooled statistics
    are FITTED on the training rows and applied unchanged to the test rows,
    which is the fold-boundary discipline that keeps the test window
    genuinely out of sample. Cross-sectional standardization has nothing to
    fit: each date is normalized against its own cross-section, which is
    contemporaneous information a live model would also have, so train and
    test are transformed independently and no statistic crosses the split.
    """
    # Selected ONCE each. `train_frame[feature_ids]` was evaluated twice on
    # the pooled path -- once to fit and once to apply -- and the take is
    # about 4.9 ms on a 100,000 x 20 block, paid on every fold of every run
    # and every candidate of every hyperparameter search.
    train_features = train_frame[feature_ids]
    test_features = test_frame[feature_ids]

    if model_spec.preprocessing.normalization == "cross_sectional":
        clip = model_spec.preprocessing.clip_sigma
        return (
            standardize_cross_sectional(
                train_features, train_frame["date"].to_numpy(), clip
            ),
            standardize_cross_sectional(
                test_features, test_frame["date"].to_numpy(), clip
            ),
        )
    # Fused so the training block becomes a C-contiguous matrix once rather
    # than once per kernel call; identical arithmetic, and it falls back to
    # the fit/apply pair whenever the fast path does not apply.
    return fit_and_apply_preprocessing(train_features, test_features)


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
    fold_metrics = []
    fold_importance = []
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
    n_expected_folds = splitter.n_splits(dates)
    # Row -> position in `dates`, computed once instead of hashing the whole
    # date column against a fresh set on every fold. `dates` is sorted and
    # every row's date is in it by construction, so searchsorted is exact.
    # A fold then selects rows by gathering a small per-date boolean, which
    # also keeps working for splitters whose folds are not contiguous
    # (purged K-fold) -- an interval slice would not.
    date_code = np.searchsorted(dates.to_numpy(), panel["date"].to_numpy())
    for train_pos, test_pos in splitter.split(dates):
        train_dates = dates[train_pos]
        test_dates = dates[test_pos]
        in_train = np.zeros(len(dates), dtype=bool)
        in_train[train_pos] = True
        in_test = np.zeros(len(dates), dtype=bool)
        in_test[test_pos] = True
        train_df = panel[in_train[date_code]]
        test_df = panel[in_test[date_code]]

        # ── Target-overlap purge ──────────────────────────────────────────
        # A forward-return label on training row t is only finished once
        # bar t+horizon prints. With embargo < horizon that bar lies inside
        # the test window, so the row's LABEL is built from test-period
        # prices even though its FEATURES are entirely in the past. The
        # embargo alone never enforced this (WalkForwardSplit is not given
        # the horizon at all), so horizon=20/embargo=0 trained on 20 labels
        # that had already seen the test period.
        #
        # Purging on the row's own recorded label_end_date rather than on
        # an integer offset also handles entities on different calendars,
        # where t+horizon entity bars != t+horizon global panel dates.
        if has_label_end and not train_df.empty and len(test_dates) > 0:
            first_test_date = test_dates[0]
            last_test_date = test_dates[-1]
            # A training row is purged when the bars its label spans
            # OVERLAP the test block: the label ends on or after the block
            # starts, and the row itself begins on or before the block
            # ends. Under walk-forward the second condition is always true
            # (training precedes testing), so this reduces exactly to the
            # previous rule; it is written in full because purged k-fold
            # puts training rows on BOTH sides of the test block, and there
            # the rows after it must not be purged for the wrong reason.
            overlaps = (train_df[LABEL_END_COL] >= first_test_date) & (
                train_df["date"] <= last_test_date
            )
            keep = ~overlaps
            n_purged_total += int(overlaps.sum())
            train_df = train_df[keep]

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

        train_X, test_X = _preprocess(model_spec, train_df, test_df, feature_ids)
        sample_weight = _fold_sample_weights(model_spec, train_df)

        fold_params = model_spec.estimator.params
        if model_spec.search is not None:

            def _fit_predict(params, inner_train, inner_test):
                """Score one candidate the way the real fit will run it —
                same preprocessing, same weighting — so the search cannot
                select for a pipeline that is never used."""
                inner_train_X, inner_test_X = _preprocess(
                    model_spec, inner_train, inner_test, feature_ids
                )
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
            }
        )
        # Weight by out-of-sample prediction count -- see
        # average_fold_metrics for why equal weighting distorts the
        # headline number when coverage varies across folds.
        fold_weights.append(float(len(test_df)))
        fold_metrics.append(metrics)
        fold_importance.append(fold_feature_importance(estimator, feature_ids))
        oos_prediction_frames.append(
            pd.DataFrame(
                {
                    "date": test_df["date"].to_numpy(),
                    "entity": test_df["entity"].to_numpy(),
                    "prediction": prediction_values,
                }
            )
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
    for prefix, series_list in pooled_ic.items():
        oos_metrics.update(aggregate_cross_sectional_ic(series_list, prefix))
    importance_summary = summarize_importance(fold_importance, feature_ids)

    # Sample size discounted for target overlap. A `horizon`-bar forward
    # return generated every bar produces labels sharing horizon-1 of their
    # bars, so the raw OOS row count overstates the independent evidence
    # behind every metric above -- often by an order of magnitude.
    n_oos_rows = int(sum(fold_weights))
    horizon = _target_horizon(dataset.get("target_id"))
    n_entities = int(panel["entity"].nunique())
    oos_metrics["n_oos_rows"] = float(n_oos_rows)
    oos_metrics["effective_sample_size"] = (
        effective_sample_size(n_oos_rows, horizon, n_entities)
        if horizon is not None
        else float(n_oos_rows)
    )

    validation_report = {
        "method": model_spec.validation.method,
        "scheme": (
            model_spec.validation.scheme
            if model_spec.validation.method == "walk_forward"
            else None
        ),
        "normalization": model_spec.preprocessing.normalization,
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
        "folds": fold_records,
    }

    # Walk-forward folds are for out-of-sample validation only; the
    # registered/deployed model is refit on the full panel so it uses
    # every available observation, the same "validate on folds, deploy
    # on everything" convention real factor-model practice uses.
    # One take of the whole panel, not two. The fused helper is not used
    # here because these statistics are persisted into the manifest, and it
    # returns only the transformed frames.
    full_features = panel[feature_ids]
    full_stats = fit_preprocessing(full_features)
    full_X = apply_preprocessing(full_features, full_stats)
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
        oos_predictions_uri=oos_predictions_uri,
        model_id=model_id,
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
        "n_folds": len(fold_metrics),
        "validation_report": validation_report,
        "oos_predictions_uri": oos_predictions_uri,
        # Surfaced rather than silently applied: a large purge count means
        # the target horizon is consuming a real fraction of each training
        # window, which is information the caller needs when reading the
        # OOS metrics.
        "n_train_rows_purged_overlap": n_purged_total,
    }
