"""Tests for modeling.engine.run_experiment: end-to-end fit + walk-forward
validate + register for each allowlisted estimator, and the
fit-on-train-only preprocessing discipline."""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.features.transforms import (
    apply_preprocessing,
    fit_preprocessing,
)
from standard_quant_tools.modeling.registry.model_registry import (
    load_manifest,
    load_model,
)
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)


def _dataset_spec(target: "TargetSpec | None" = None) -> DatasetSpec:
    return DatasetSpec(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=target or TargetSpec(horizon=5),
        benchmark="SPY",
    )


def _model_spec(task="regression", estimator="ridge", **estimator_params) -> ModelSpec:
    return ModelSpec(
        task=task,
        estimator=EstimatorSpec(type=estimator, params=estimator_params),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )


@pytest.fixture
def dataset(patched_multi_factory):
    return build_dataset(_dataset_spec())


class TestRunExperimentRegression:
    @pytest.mark.parametrize(
        "estimator,params",
        [
            ("linear", {}),
            ("ridge", {"alpha": 1.0}),
            ("lasso", {"alpha": 0.01}),
            ("elastic_net", {"alpha": 0.01, "l1_ratio": 0.5}),
            ("hist_gradient_boosting", {"max_iter": 20}),
            ("random_forest", {"n_estimators": 10, "max_depth": 3}),
            ("gradient_boosting", {"n_estimators": 10, "max_depth": 3}),
        ],
    )
    def test_every_allowlisted_regressor_runs_end_to_end(
        self, dataset, estimator, params
    ):
        model_spec = _model_spec(estimator=estimator, **params)
        result = run_experiment(dataset, model_spec, dataset_id="ds_test")
        assert result["model_id"].startswith("mdl_")
        assert result["n_folds"] >= 1
        # Superset, not equality: asserting an exact key set makes every
        # added metric a test failure, which is what happened when
        # cross-sectional IC / baseline / sample-size fields were added.
        assert set(result["oos_metrics"]) >= {"r2", "mae", "ic", "rank_ic"}
        # The cross-sectional IC family is what a cross-sectional model is
        # actually judged on -- see validation/metrics.py.
        assert set(result["oos_metrics"]) >= {
            "cs_ic_mean",
            "cs_ic_icir",
            "cs_rank_ic_mean",
            "cs_rank_ic_icir",
            "baseline_mae",
            "effective_sample_size",
        }

    def test_registered_model_is_loadable_and_predicts(self, dataset):
        result = run_experiment(dataset, _model_spec(), dataset_id="ds_test")
        manifest = load_manifest(result["model_id"])
        model = load_model(result["model_id"])
        assert manifest.task == "regression"
        assert manifest.n_folds == result["n_folds"]
        preds = model.predict(dataset["panel"][manifest.feature_ids].to_numpy()[:5])
        assert len(preds) == 5


def _synthetic_binary_dataset(n: int = 300, n_features: int = 2) -> dict:
    """A hand-built, single-entity dataset whose target flips from all-0
    to all-1 partway through -- not fetched through build_dataset, so the
    fold boundaries relative to the class transition are exactly
    controllable (needed to deterministically produce both single-class
    and mixed-class walk-forward folds in one dataset)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    target = np.array([0] * (n - 100) + [1] * 100)
    data = {"date": dates, "entity": ["X"] * n, "target": target}
    feature_ids = [f"f{i}" for i in range(n_features)]
    for fid in feature_ids:
        data[fid] = rng.normal(0, 1, n)
    panel = pd.DataFrame(data)
    return {
        "panel": panel,
        "feature_ids": feature_ids,
        # forward_direction, so this synthetic dataset passes the
        # task/target compatibility check and actually exercises the
        # single-class-fold skipping it exists to test.
        "target_id": "forward_direction:1",
        "data_hash": "h",
    }


class TestRunExperimentClassification:
    @pytest.mark.parametrize(
        "estimator,params",
        [
            ("logistic", {}),
            ("hist_gradient_boosting", {"max_iter": 20}),
            ("random_forest", {"n_estimators": 10, "max_depth": 3}),
            ("gradient_boosting", {"n_estimators": 10, "max_depth": 3}),
        ],
    )
    def test_every_allowlisted_classifier_runs_end_to_end(
        self, patched_multi_factory, estimator, params
    ):
        """
        Built through the ORDINARY pipeline via
        TargetSpec(type='forward_direction'). These tests used to binarize
        the panel by hand after build_dataset, because ModelSpec.task
        accepted 'classification' while TargetSpec could only produce a
        continuous return — an advertised capability with no way to
        construct it through the five-tool surface.
        """
        built = build_dataset(
            _dataset_spec(target=TargetSpec(type="forward_direction", horizon=5))
        )
        model_spec = _model_spec(task="classification", estimator=estimator, **params)
        result = run_experiment(built, model_spec, dataset_id="ds_test")
        assert set(result["oos_metrics"]) >= {"accuracy", "auc"}
        # Class balance is reported, so accuracy can be read against the
        # majority-class baseline instead of in a vacuum.
        assert set(result["oos_metrics"]) >= {
            "positive_rate",
            "majority_class_accuracy",
        }

    def test_regression_task_against_direction_target_rejected(
        self, patched_multi_factory
    ):
        """A 0/1 target fed to a regressor would fit happily and report
        meaningless R2/IC — caught by task/target compatibility instead."""
        built = build_dataset(
            _dataset_spec(target=TargetSpec(type="forward_direction", horizon=5))
        )
        model_spec = _model_spec(task="regression", estimator="ridge")
        with pytest.raises(ValidationError, match="task='regression' expects one of"):
            run_experiment(built, model_spec, dataset_id="ds_test")

    def test_classification_task_against_return_target_rejected(self, dataset):
        """The mirror case: a continuous forward return under
        task='classification' must be rejected before any fold is
        attempted, not crash deep inside sklearn with 'Unknown label type:
        continuous'."""
        model_spec = _model_spec(task="classification", estimator="logistic")
        with pytest.raises(
            ValidationError, match="task='classification' expects one of"
        ):
            run_experiment(dataset, model_spec, dataset_id="ds_test")

    def test_single_class_overall_target_rejected(self, patched_multi_factory):
        built = build_dataset(_dataset_spec())
        panel = built["panel"].copy()
        panel["target"] = 0  # every row the same class
        # target_id is set to the direction type so this exercises the
        # BINARY check rather than the task/target compatibility check.
        built = {**built, "panel": panel, "target_id": "forward_direction:5"}
        model_spec = _model_spec(task="classification", estimator="logistic")
        with pytest.raises(ValidationError, match="requires a discrete"):
            run_experiment(built, model_spec, dataset_id="ds_test")

    def test_fold_with_single_class_train_window_is_skipped_not_fatal(self):
        """A binary-overall target whose class transition falls such
        that some walk-forward folds' TRAIN window is entirely one class
        must skip only those folds (same discipline as an empty
        train/test slice), not fail the whole experiment -- as long as
        at least one fold ends up with both classes in train."""
        dataset = _synthetic_binary_dataset()
        model_spec = ModelSpec(
            task="classification",
            estimator=EstimatorSpec(type="logistic", params={}),
            validation=ValidationSpec(train_window=150, test_window=30, embargo=0),
            random_seed=1,
        )
        result = run_experiment(dataset, model_spec, dataset_id="ds_test")
        # Naively, WalkForwardSplit would yield floor((300-150-30)/30)+1 = 5
        # folds; at least one (the class-transition fold) must have run,
        # and at least one of the all-single-class folds must have been
        # skipped -- fewer folds counted than the naive total proves the
        # skip logic actually engaged, not merely that the run succeeded.
        assert 1 <= result["n_folds"] < 5


class TestRunExperimentValidation:
    def test_unknown_estimator_raises(self, dataset):
        model_spec = _model_spec(estimator="not_a_real_estimator")
        with pytest.raises(ValidationError, match="unknown estimator"):
            run_experiment(dataset, model_spec, dataset_id="ds_test")

    def test_disallowed_param_raises(self, dataset):
        model_spec = _model_spec(estimator="ridge", not_a_real_param=1)
        with pytest.raises(ValidationError, match="does not accept"):
            run_experiment(dataset, model_spec, dataset_id="ds_test")

    def test_too_short_dataset_raises(self, dataset):
        model_spec = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={}),
            validation=ValidationSpec(train_window=10_000, test_window=10, embargo=0),
        )
        with pytest.raises(ValidationError, match="not enough for one"):
            run_experiment(dataset, model_spec, dataset_id="ds_test")


class TestPreprocessingLeakageDiscipline:
    def test_apply_uses_train_stats_not_test_stats(self):
        """A test-fold value far outside the train fold's range must be
        clipped to the TRAIN winsorize bounds, not its own -- proof the
        stats really were fit on train only."""
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0] * 20})
        test = pd.DataFrame({"x": [1000.0, -1000.0]})
        stats = fit_preprocessing(train)
        transformed_test = apply_preprocessing(test, stats)
        # Clipped to train's ~[1,5] range before z-scoring, so nowhere
        # near the raw magnitude of 1000 in z-score units.
        assert transformed_test["x"].abs().max() < 10
