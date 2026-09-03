"""
The two estimators added for sequence and online modelling, and the one
way the architecture wrapper silently lies.

WHAT THESE PIN.

  1. sklearn's `clone` rebuilds an estimator from `get_params()`. The
     architecture here is two scalars, and `hidden_layer_sizes` is not a
     param -- so a clone that did not rederive the tuple would fit
     sklearn's default 100-unit layer while the manifest recorded the width
     the caller asked for. The engine clones per fold, so this is the
     normal path, not an edge case.
  2. The parameter allowlist is a compute budget, not a modelling opinion.
     An unbounded width or iteration count is an agent-triggerable way to
     pin the process.
  3. An SGD classifier defaulting to 'hinge' has no predict_proba, so a
     spec that thresholds a probability must fail where it can be
     explained, not deep inside scoring.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.estimators.neural import (
    PanelMLPClassifier,
    PanelMLPRegressor,
)
from standard_quant_tools.modeling.estimators.registry import (
    get_estimator_class,
    validate_params,
)


class TestTheArchitectureSurvivesCloning:
    def test_a_cloned_regressor_keeps_the_requested_width(self) -> None:
        """
        THE DEFECT THIS EXISTS TO CATCH. `hidden_layer_sizes` is set in
        __init__ but is not a constructor param of this subclass, so clone
        drops it. Without the rederive in fit(), every fold would train
        sklearn's default architecture while the spec said otherwise.
        """
        original = PanelMLPRegressor(n_hidden_units=8, n_hidden_layers=2)
        copy = clone(original)
        assert copy.n_hidden_units == 8
        assert copy.n_hidden_layers == 2

        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 4))
        y = X[:, 0] * 2.0 + rng.normal(0, 0.1, 120)
        copy.fit(X, y)
        # The fitted coefficient shapes are the only honest witness that
        # the network really is 8x2 rather than sklearn's (100,).
        assert copy.hidden_layer_sizes == (8, 8)
        assert [c.shape[1] for c in copy.coefs_[:-1]] == [8, 8]

    def test_a_cloned_classifier_keeps_it_too(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(120, 4))
        y = (X[:, 0] > 0).astype(int)
        fitted = clone(PanelMLPClassifier(n_hidden_units=6, n_hidden_layers=1)).fit(
            X, y
        )
        assert fitted.hidden_layer_sizes == (6,)
        assert fitted.coefs_[0].shape[1] == 6

    def test_the_default_is_still_a_working_network(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.normal(size=(150, 3))
        y = X[:, 0] + rng.normal(0, 0.1, 150)
        model = PanelMLPRegressor(random_state=0, max_iter=400).fit(X, y)
        assert np.corrcoef(model.predict(X), y)[0, 1] > 0.9

    def test_it_is_reproducible_when_seeded(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.normal(size=(150, 3))
        y = X[:, 0] + rng.normal(0, 0.1, 150)
        a = PanelMLPRegressor(random_state=7, max_iter=200).fit(X, y).predict(X)
        b = PanelMLPRegressor(random_state=7, max_iter=200).fit(X, y).predict(X)
        assert np.allclose(a, b)


class TestTheAllowlistIsAComputeBudget:
    def test_an_absurd_width_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="exceeds the maximum"):
            validate_params("regression", "mlp", {"n_hidden_units": 10_000_000})

    def test_an_absurd_depth_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="exceeds the maximum"):
            validate_params("regression", "mlp", {"n_hidden_layers": 50})

    def test_the_ceiling_says_it_is_a_budget_not_an_opinion(self) -> None:
        with pytest.raises(ValidationError, match="resource budget"):
            validate_params("regression", "mlp", {"max_iter": 10_000_000})

    def test_a_hallucinated_param_names_the_real_ones(self) -> None:
        with pytest.raises(ValidationError, match="hidden_layer_sizes"):
            validate_params("regression", "mlp", {"hidden_layer_sizes": [64, 64]})

    def test_a_reasonable_request_passes(self) -> None:
        validate_params(
            "regression",
            "mlp",
            {
                "n_hidden_units": 64,
                "n_hidden_layers": 2,
                "alpha": 1e-3,
                "max_iter": 500,
                "random_state": 11,
            },
        )


class TestSGDIsRegisteredForBothTasks:
    def test_both_tasks_resolve(self) -> None:
        from sklearn.linear_model import SGDClassifier, SGDRegressor

        assert get_estimator_class("regression", "sgd") is SGDRegressor
        assert get_estimator_class("classification", "sgd") is SGDClassifier

    def test_the_robust_loss_is_reachable(self) -> None:
        """Squared error lets one 8-sigma day contribute 64x a 1-sigma one."""
        validate_params("regression", "sgd", {"loss": "huber"})

    def test_an_unknown_loss_lists_the_real_ones(self) -> None:
        with pytest.raises(ValidationError, match="squared_error"):
            validate_params("regression", "sgd", {"loss": "mse"})

    def test_elasticnet_without_a_ratio_is_refused(self) -> None:
        """sklearn's silent 0.15 is a weighting nobody chose."""
        with pytest.raises(ValidationError, match="l1_ratio"):
            validate_params("regression", "sgd", {"penalty": "elasticnet"})

    def test_elasticnet_with_a_ratio_passes(self) -> None:
        validate_params("regression", "sgd", {"penalty": "elasticnet", "l1_ratio": 0.5})

    def test_a_probability_loss_is_reachable_for_classification(self) -> None:
        validate_params("classification", "sgd", {"loss": "log_loss"})

    def test_the_step_schedule_choices_are_named(self) -> None:
        with pytest.raises(ValidationError, match="invscaling"):
            validate_params("regression", "sgd", {"learning_rate": "cosine"})


class TestThroughTheEngine:
    @staticmethod
    def _run(patched_multi_factory, estimator_type, params, lags=None):
        from standard_quant_tools.modeling.agent import (
            BuildModelDatasetInput,
            RunModelExperimentInput,
            build_model_dataset,
            run_model_experiment,
        )
        from standard_quant_tools.modeling.specs import (
            DatasetSpec,
            EstimatorSpec,
            FeatureSpec,
            ModelSpec,
            TargetSpec,
            ValidationSpec,
        )

        built = build_model_dataset(
            BuildModelDatasetInput(
                spec=DatasetSpec(
                    universe=["AAA", "BBB", "CCC"],
                    start="2022-01-01",
                    end="2023-12-31",
                    features=[
                        FeatureSpec(id="technical.rsi", lags=lags or []),
                        FeatureSpec(id="risk.rolling_beta"),
                    ],
                    target=TargetSpec(horizon=5),
                    benchmark="SPY",
                )
            )
        )
        return run_model_experiment(
            RunModelExperimentInput(
                dataset_id=built.dataset_id,
                spec=ModelSpec(
                    task="regression",
                    estimator=EstimatorSpec(type=estimator_type, params=params),
                    validation=ValidationSpec(
                        train_window=150, test_window=30, embargo=5
                    ),
                    random_seed=11,
                ),
            )
        )

    def test_an_mlp_trains_over_a_lag_window(self, patched_multi_factory) -> None:
        """The whole point of §7: a non-linear model over history."""
        result = self._run(
            patched_multi_factory,
            "mlp",
            {
                "n_hidden_units": 8,
                "n_hidden_layers": 1,
                "max_iter": 200,
                "random_state": 0,
            },
            lags=[1, 2, 3],
        )
        assert result.n_folds >= 1
        assert result.model_id.startswith("mdl_")

    def test_sgd_trains(self, patched_multi_factory) -> None:
        result = self._run(
            patched_multi_factory,
            "sgd",
            {"loss": "huber", "alpha": 1e-4, "max_iter": 1000, "random_state": 0},
        )
        assert result.n_folds >= 1

    def test_both_are_scored_and_diagnosed_like_any_other_model(
        self, patched_multi_factory
    ) -> None:
        """
        They are ordinary registry entries, so everything downstream --
        scoring, diagnostics, the backtest bridge -- works with no new code.
        """
        from standard_quant_tools.modeling.agent.models import (
            AnalyzeModelErrorsInput,
        )
        from standard_quant_tools.modeling.agent.tools import analyze_model_errors

        result = self._run(
            patched_multi_factory,
            "mlp",
            {"n_hidden_units": 8, "max_iter": 200, "random_state": 0},
            lags=[1],
        )
        report = analyze_model_errors(AnalyzeModelErrorsInput(model_id=result.model_id))
        assert report.n_rows > 0
        assert report.calibration["slope"] is not None
