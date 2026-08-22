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
from .dataset.alignment import LABEL_END_COL
from .estimators.registry import get_estimator_class, validate_params
from .features.transforms import apply_preprocessing, fit_preprocessing
from .registry.model_registry import new_model_id, save_model
from .specs import ModelSpec
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
from .validation.walk_forward import WalkForwardSplit


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
    unique_values = set(pd.unique(panel["target"].dropna()))
    if unique_values != {0, 1}:
        sample = sorted(unique_values)[:10]
        raise ValidationError(
            "run_model_experiment: task='classification' requires a binary {0, 1} "
            f"target, but the dataset's target column has values {sample}"
            + ("..." if len(unique_values) > 10 else "")
            + ". Build the dataset with TargetSpec(type='forward_direction', "
            "horizon=..., threshold=...) — that is the target type that produces a "
            "binary label through the normal pipeline."
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
    expected = {
        "regression": "forward_return",
        "classification": "forward_direction",
    }.get(task)
    if expected is None or target_type == expected:
        return
    raise ValidationError(
        f"run_model_experiment: task={task!r} expects a {expected!r} target, but this "
        f"dataset was built with {target_type!r}. Rebuild the dataset with "
        f"TargetSpec(type={expected!r}, ...), or change the model's task."
    )


def _predict_fold(
    task: str,
    estimator: Any,
    test_X: pd.DataFrame,
    test_y: Any,
    test_dates: "np.ndarray | None" = None,
    train_y: "np.ndarray | None" = None,
) -> "tuple[Dict[str, float], np.ndarray, Dict[str, pd.Series]]":
    """
    Returns (metrics, prediction_values, ic_series). `prediction_values` is
    always a continuous score suitable for downstream signal construction
    (see modeling.bridge) -- the raw regression prediction for a regression
    task, or the positive-class probability (not the discrete 0/1
    predicted label) for a classification task.

    `ic_series` carries this fold's PER-DATE cross-sectional IC series so
    the caller can pool every fold's dates and compute the OOS dispersion
    statistics once. Averaging each fold's own std/ICIR is a different
    quantity -- see aggregate_cross_sectional_ic.
    """
    preds = estimator.predict(test_X.to_numpy())
    if task == "regression":
        metrics = regression_metrics(test_y, preds, dates=test_dates, train_y=train_y)
        ic_series: Dict[str, pd.Series] = {}
        if test_dates is not None:
            ic_series["cs_ic"] = cross_sectional_ic(
                test_y, preds, test_dates, "pearson"
            )
            ic_series["cs_rank_ic"] = cross_sectional_ic(
                test_y, preds, test_dates, "spearman"
            )
        return metrics, preds, ic_series
    proba = positive_class_proba(estimator, test_X.to_numpy())
    return classification_metrics(test_y, preds, proba), proba, {}


def run_experiment(
    dataset: Dict[str, Any], model_spec: ModelSpec, dataset_id: str
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

    Raises:
        ValidationError: unknown estimator, disallowed estimator param,
        or the dataset has too few dates for even one walk-forward fold.
    """
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

    splitter = WalkForwardSplit(
        train_window=model_spec.validation.train_window,
        test_window=model_spec.validation.test_window,
        embargo=model_spec.validation.embargo,
    )
    if splitter.n_splits(dates) < 1:
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
            keep = train_df[LABEL_END_COL] < first_test_date
            n_purged_total += int((~keep).sum())
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

        stats = fit_preprocessing(train_df[feature_ids])
        train_X = apply_preprocessing(train_df[feature_ids], stats)
        test_X = apply_preprocessing(test_df[feature_ids], stats)

        estimator = _instantiate(
            estimator_cls, model_spec.estimator.params, model_spec.random_seed
        )
        estimator.fit(train_X.to_numpy(), train_y)

        metrics, prediction_values, fold_ic = _predict_fold(
            model_spec.task,
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
    full_stats = fit_preprocessing(panel[feature_ids])
    full_X = apply_preprocessing(panel[feature_ids], full_stats)
    full_y = panel["target"].to_numpy()
    final_estimator = _instantiate(
        estimator_cls, model_spec.estimator.params, model_spec.random_seed
    )
    final_estimator.fit(full_X.to_numpy(), full_y)

    # model_id generated here (not left to save_model's own default)
    # so the OOS predictions artifact lands in the same
    # SQT_RUNS_DIR/<model_id>/ directory as the model's other files --
    # these predictions are the leakage-safe source
    # modeling.bridge.oos_predictions_to_signal_panel needs to turn a
    # model into an actual strategy backtest (never score_model, whose
    # single as-of snapshot comes from this same final_estimator and
    # would leak if used to "predict" dates it was trained on).
    model_id = new_model_id()
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
