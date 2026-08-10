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
from typing import Any, Dict

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .estimators.registry import get_estimator_class, validate_params
from .features.transforms import apply_preprocessing, fit_preprocessing
from .registry.model_registry import save_model
from .specs import ModelSpec
from .validation.diagnostics import fold_feature_importance, summarize_importance
from .validation.metrics import (
    average_fold_metrics,
    classification_metrics,
    positive_class_proba,
    regression_metrics,
)
from .validation.walk_forward import WalkForwardSplit


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
            + ". TargetSpec currently only builds a continuous forward_return "
            "target -- binarize it yourself (e.g. sign of the forward return) "
            "before calling build_model_dataset if you need a classification target."
        )


def _predict_fold(task: str, estimator: Any, test_X: pd.DataFrame, test_y: Any) -> Dict[str, float]:
    preds = estimator.predict(test_X.to_numpy())
    if task == "regression":
        return regression_metrics(test_y, preds)
    proba = positive_class_proba(estimator, test_X.to_numpy())
    return classification_metrics(test_y, preds, proba)


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
    validate_params(model_spec.task, model_spec.estimator.type, model_spec.estimator.params)

    panel = dataset["panel"]
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

    fold_metrics = []
    fold_importance = []
    for train_pos, test_pos in splitter.split(dates):
        train_dates = dates[train_pos]
        test_dates = dates[test_pos]
        train_df = panel[panel["date"].isin(train_dates)]
        test_df = panel[panel["date"].isin(test_dates)]
        if train_df.empty or test_df.empty:
            continue

        train_y = train_df["target"].to_numpy()
        test_y = test_df["target"].to_numpy()
        # A fold whose training window happens to land entirely on one
        # side of a binary target can't fit a classifier at all (sklearn
        # raises deep inside .fit()) -- skip it, same as an empty
        # train/test slice above, rather than letting the whole
        # experiment crash over one unlucky window.
        if model_spec.task == "classification" and len(np.unique(train_y)) < 2:
            continue

        stats = fit_preprocessing(train_df[feature_ids])
        train_X = apply_preprocessing(train_df[feature_ids], stats)
        test_X = apply_preprocessing(test_df[feature_ids], stats)

        estimator = _instantiate(estimator_cls, model_spec.estimator.params, model_spec.random_seed)
        estimator.fit(train_X.to_numpy(), train_y)

        fold_metrics.append(_predict_fold(model_spec.task, estimator, test_X, test_y))
        fold_importance.append(fold_feature_importance(estimator, feature_ids))

    if not fold_metrics:
        raise ValidationError(
            "run_model_experiment: every walk-forward fold was skipped -- either an empty "
            "train/test slice (the requested universe/date range likely doesn't cover every "
            "entity on every date), or, for classification, a fold whose training window "
            "landed entirely on one class of the binary target."
        )

    oos_metrics = average_fold_metrics(fold_metrics)
    importance_summary = summarize_importance(fold_importance, feature_ids)

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
        preprocessing_stats=full_stats,
    )

    return {
        "model_id": manifest.model_id,
        "oos_metrics": oos_metrics,
        "feature_importance_summary": importance_summary,
        "n_folds": len(fold_metrics),
    }
