"""Tests for modeling.registry.model_registry: save/load round-trip."""

import pytest
from sklearn.linear_model import Ridge

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.registry.model_registry import (
    load_manifest,
    load_model,
    load_model_spec,
    load_preprocessing_stats,
    new_model_id,
    save_model,
)
from standard_quant_tools.modeling.specs import EstimatorSpec, ModelSpec, ValidationSpec


def _model_spec() -> ModelSpec:
    return ModelSpec(
        task="regression",
        estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
        validation=ValidationSpec(train_window=100, test_window=20, embargo=0),
        random_seed=3,
    )


class TestSaveLoadRoundTrip:
    def test_manifest_round_trips(self):
        estimator = Ridge(alpha=1.0).fit([[1.0], [2.0], [3.0]], [1.0, 2.0, 3.0])
        model_spec = _model_spec()
        manifest = save_model(
            estimator=estimator,
            model_spec=model_spec,
            feature_ids=["f1", "f2"],
            target_id="forward_return:5",
            dataset_id="ds_abc",
            dataset_hash="deadbeef",
            oos_metrics={"r2": 0.1, "mae": 0.02, "ic": 0.05, "rank_ic": 0.04},
            feature_importance_summary={"f1": {"mean": 0.1, "std": 0.01}},
            n_folds=3,
            preprocessing_stats={"f1": {"lo": 0.0, "hi": 1.0, "mean": 0.5, "std": 0.2}},
        )
        reloaded = load_manifest(manifest.model_id)
        assert reloaded.model_id == manifest.model_id
        assert reloaded.feature_ids == ["f1", "f2"]
        assert reloaded.dataset_id == "ds_abc"
        assert reloaded.n_folds == 3

    def test_model_object_round_trips_and_predicts(self):
        # alpha near zero so the fitted line stays close to y=x, making
        # the post-round-trip prediction assertion below a meaningful
        # check rather than one sensitive to Ridge's shrinkage strength.
        estimator = Ridge(alpha=1e-6).fit([[1.0], [2.0], [3.0], [4.0]], [1.0, 2.0, 3.0, 4.0])
        manifest = save_model(
            estimator=estimator,
            model_spec=_model_spec(),
            feature_ids=["f1"],
            target_id="forward_return:5",
            dataset_id="ds_abc",
            dataset_hash="deadbeef",
            oos_metrics={},
            feature_importance_summary={},
            n_folds=1,
            preprocessing_stats={},
        )
        reloaded_model = load_model(manifest.model_id)
        assert reloaded_model.predict([[5.0]])[0] == pytest.approx(5.0, abs=0.1)

    def test_model_spec_round_trips(self):
        model_spec = _model_spec()
        manifest = save_model(
            estimator=Ridge().fit([[1.0]], [1.0]),
            model_spec=model_spec,
            feature_ids=["f1"],
            target_id="t",
            dataset_id="ds_abc",
            dataset_hash="h",
            oos_metrics={},
            feature_importance_summary={},
            n_folds=1,
            preprocessing_stats={},
        )
        reloaded_spec = load_model_spec(manifest.model_id)
        assert reloaded_spec.estimator.type == "ridge"
        assert reloaded_spec.random_seed == 3

    def test_preprocessing_stats_round_trip(self):
        stats = {"f1": {"lo": 0.0, "hi": 1.0, "mean": 0.5, "std": 0.2}}
        manifest = save_model(
            estimator=Ridge().fit([[1.0]], [1.0]),
            model_spec=_model_spec(),
            feature_ids=["f1"],
            target_id="t",
            dataset_id="ds_abc",
            dataset_hash="h",
            oos_metrics={},
            feature_importance_summary={},
            n_folds=1,
            preprocessing_stats=stats,
        )
        assert load_preprocessing_stats(manifest.model_id) == stats

    def test_explicit_model_id_used_when_given(self):
        model_id = new_model_id()
        manifest = save_model(
            estimator=Ridge().fit([[1.0]], [1.0]),
            model_spec=_model_spec(),
            feature_ids=["f1"],
            target_id="t",
            dataset_id="ds_abc",
            dataset_hash="h",
            oos_metrics={},
            feature_importance_summary={},
            n_folds=1,
            preprocessing_stats={},
            model_id=model_id,
        )
        assert manifest.model_id == model_id


class TestUnregisteredModelRaises:
    def test_load_manifest_unknown_id_raises(self):
        with pytest.raises(ValidationError, match="no registered model"):
            load_manifest("mdl_does_not_exist")

    def test_load_model_unknown_id_raises(self):
        with pytest.raises(ValidationError, match="no registered model"):
            load_model("mdl_does_not_exist")
