"""
Regression tests for `TargetSpec(type="forward_direction")` and the
classification path it unlocks.

`ModelSpec.task` accepted "classification" from the start, but `TargetSpec`
could only build a continuous forward return — so a binary target was only
reachable by mutating the panel by hand AFTER build_dataset, outside the
five-tool agent workflow. Classification was an advertised capability with
no way to construct it. These tests exercise it through the ordinary
pipeline, and cover the JSON-safety and class-balance issues that come with
classification outputs.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools._jsonsafe import sanitize_for_json
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.dataset.builder import build_dataset
from standard_quant_tools.modeling.dataset.target import build_target
from standard_quant_tools.modeling.engine import run_experiment
from standard_quant_tools.modeling.specs import (
    DatasetSpec,
    EstimatorSpec,
    FeatureSpec,
    ModelSpec,
    TargetSpec,
    ValidationSpec,
)


class TestForwardDirectionTarget:
    @staticmethod
    def _close() -> pd.Series:
        idx = pd.date_range("2024-01-01", periods=12, freq="D")
        return pd.Series(
            [100, 101, 102, 101, 100, 103, 104, 103, 105, 106, 107, 108],
            index=idx,
            dtype=float,
        )

    def test_binarizes_the_forward_return(self):
        close = self._close()
        returns = build_target(close, TargetSpec(horizon=3))
        direction = build_target(close, TargetSpec(type="forward_direction", horizon=3))
        both = pd.DataFrame({"r": returns, "d": direction}).dropna()
        assert ((both["r"] > 0) == (both["d"] == 1.0)).all()

    def test_values_are_strictly_zero_or_one(self):
        direction = build_target(
            self._close(), TargetSpec(type="forward_direction", horizon=3)
        )
        assert set(direction.dropna().unique()) <= {0.0, 1.0}

    def test_unresolved_tail_stays_nan_not_labelled_down(self):
        """
        The subtle one. `NaN > threshold` is False, so a naive
        `.astype(float)` would label every unresolved bar 0.0 —
        manufacturing a "went down" observation for bars whose outcome has
        not happened yet, and feeding it to the classifier as fact.
        """
        direction = build_target(
            self._close(), TargetSpec(type="forward_direction", horizon=3)
        )
        assert direction.tail(3).isna().all()
        assert not (direction.tail(3) == 0.0).any()

    def test_threshold_shifts_the_class_boundary(self):
        close = self._close()
        plain = build_target(close, TargetSpec(type="forward_direction", horizon=3))
        strict = build_target(
            close, TargetSpec(type="forward_direction", horizon=3, threshold=0.02)
        )
        assert strict.sum() < plain.sum()

    def test_threshold_rejected_on_forward_return(self):
        with pytest.raises(ValueError, match="forward_direction"):
            TargetSpec(type="forward_return", horizon=5, threshold=0.02)

    def test_non_finite_threshold_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            TargetSpec(type="forward_direction", horizon=5, threshold=float("nan"))

    def test_target_id_records_the_type(self, patched_multi_factory):
        built = build_dataset(_spec(TargetSpec(type="forward_direction", horizon=5)))
        assert built["target_id"] == "forward_direction:5"


def _spec(target: TargetSpec) -> DatasetSpec:
    return DatasetSpec(
        universe=["AAA", "BBB", "CCC"],
        start="2022-01-01",
        end="2023-12-31",
        features=[FeatureSpec(id="technical.rsi"), FeatureSpec(id="market.momentum")],
        target=target,
    )


def _classifier_spec(estimator="logistic", **params) -> ModelSpec:
    return ModelSpec(
        task="classification",
        estimator=EstimatorSpec(type=estimator, params=params),
        validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
        random_seed=1,
    )


class TestClassificationThroughTheNormalPipeline:
    def test_end_to_end_without_touching_the_panel(self, patched_multi_factory):
        built = build_dataset(_spec(TargetSpec(type="forward_direction", horizon=5)))
        result = run_experiment(built, _classifier_spec(), dataset_id="ds_fd")
        assert result["n_folds"] >= 2
        assert set(result["oos_metrics"]) >= {"accuracy", "auc"}

    def test_class_balance_is_reported(self, patched_multi_factory):
        """
        Accuracy is close to meaningless without it: a 95/5 split scores
        0.95 by always predicting the majority class, and `threshold` makes
        that imbalance easy to request.
        """
        built = build_dataset(_spec(TargetSpec(type="forward_direction", horizon=5)))
        result = run_experiment(built, _classifier_spec(), dataset_id="ds_fd2")
        metrics = result["oos_metrics"]
        assert 0.0 <= metrics["positive_rate"] <= 1.0
        assert 0.5 <= metrics["majority_class_accuracy"] <= 1.0

    def test_task_target_mismatch_rejected_both_ways(self, patched_multi_factory):
        direction = build_dataset(
            _spec(TargetSpec(type="forward_direction", horizon=5))
        )
        returns = build_dataset(_spec(TargetSpec(horizon=5)))

        regression = ModelSpec(
            task="regression",
            estimator=EstimatorSpec(type="ridge", params={"alpha": 1.0}),
            validation=ValidationSpec(train_window=150, test_window=30, embargo=5),
            random_seed=1,
        )
        with pytest.raises(ValidationError, match="task='regression' expects one of"):
            run_experiment(direction, regression, dataset_id="ds_fd3")
        with pytest.raises(
            ValidationError, match="task='classification' expects one of"
        ):
            run_experiment(returns, _classifier_spec(), dataset_id="ds_fd4")


class TestJsonSafety:
    """
    modeling_dispatch returned _run_and_record's dict unsanitized, unlike
    agent.tools.dispatch. Classification genuinely produces NaN — AUC on a
    single-class fold, HistGradientBoosting's absent feature importance —
    and json.dumps emits the non-standard `NaN` token for those, which
    strict parsers reject.
    """

    @staticmethod
    def _is_strict_json(payload: str) -> bool:
        try:
            json.loads(
                payload,
                parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)),
            )
            return True
        except ValueError:
            return False

    def test_raw_nan_is_not_strict_json(self):
        assert not self._is_strict_json(json.dumps({"auc": float("nan")}))

    def test_sanitized_output_is_strict_json(self):
        payload = {
            "auc": float("nan"),
            "sortino": float("inf"),
            "importance": {"f0": float("nan")},
            "bands": [1.0, float("-inf")],
        }
        assert self._is_strict_json(json.dumps(sanitize_for_json(payload)))

    def test_sanitizer_preserves_finite_values(self):
        out = sanitize_for_json({"a": 1.5, "b": 0, "c": True, "d": "x"})
        assert out == {"a": 1.5, "b": 0, "c": True, "d": "x"}

    def test_numpy_float32_is_covered(self):
        """np.float64 subclasses float and was already handled; np.float32
        is not and previously survived to the encoder."""
        assert sanitize_for_json(np.float32("nan")) is None

    def test_nested_containers_walked(self):
        out = sanitize_for_json({"t": (float("inf"), 2.0)})
        assert out == {"t": [None, 2.0]}

    def test_both_dispatch_surfaces_share_one_implementation(self):
        """Duplicated sanitizers drift; agent.tools re-exports the shared
        one rather than keeping its own copy."""
        from standard_quant_tools.agent.tools import _sanitize_for_json

        assert _sanitize_for_json is sanitize_for_json

    def test_classification_metrics_survive_strict_json(self, patched_multi_factory):
        built = build_dataset(_spec(TargetSpec(type="forward_direction", horizon=5)))
        result = run_experiment(
            built,
            _classifier_spec("hist_gradient_boosting", max_iter=20),
            dataset_id="ds_fd5",
        )
        # HistGradientBoosting exposes neither coef_ nor
        # feature_importances_, so its importance summary is all-NaN --
        # exactly the case that produced invalid JSON.
        importance = result["feature_importance_summary"]
        assert any(math.isnan(v["mean"]) for v in importance.values())
        assert self._is_strict_json(json.dumps(sanitize_for_json(result)))
