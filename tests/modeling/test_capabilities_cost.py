"""
The capability report says what a fit costs and what sets its cores (see the
CHANGELOG entry of 2026-10-04).

An agent chose random_forest from a report that gave it no way to tell that
one forest fit costs about ten thousand ridge fits. Each registration now
declares `fit_cost` (low/medium/high, measured at the estimator's defaults)
and `threads` (one/budget/openmp), and the report carries them beside each
estimator -- not in `ModelAdapter.capabilities`, whose key set is fixed.
"""

import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.adapters import RegressionAdapter
from standard_quant_tools.modeling.capabilities import (
    ESTIMATOR_COST_NOTE,
    estimator_capabilities,
    modeling_capabilities,
)
from standard_quant_tools.modeling.estimators.bounds import EstimatorParamSchema
from standard_quant_tools.modeling.estimators.registry import (
    ESTIMATOR_REGISTRY,
    FIT_COSTS,
    THREAD_KINDS,
    EstimatorCost,
    estimator_cost,
    register_estimator,
    validate_params,
)


def _entry(task, name):
    (entry,) = [
        e for e in estimator_capabilities() if e["task"] == task and e["name"] == name
    ]
    return entry


class TestEveryEstimatorDeclaresIt:
    def test_every_registered_estimator(self):
        """Every estimator this package registers; a test that registers a
        stand-in of its own is not one of them."""
        shipped = ("sklearn", "lightgbm", "xgboost", "standard_quant_tools.modeling")
        for (task, name), cls in ESTIMATOR_REGISTRY.items():
            if not cls.__module__.startswith(shipped):
                continue
            cost = estimator_cost(task, name)
            assert cost is not None, (task, name)
            assert cost.fit_cost in FIT_COSTS
            assert cost.threads in THREAD_KINDS

    def test_the_ones_that_decide_a_choice(self):
        assert estimator_cost("regression", "random_forest") == EstimatorCost(
            "high", "budget"
        )
        assert estimator_cost("regression", "hist_gradient_boosting") == (
            EstimatorCost("medium", "openmp")
        )
        assert estimator_cost("regression", "ridge") == EstimatorCost("low", "one")
        assert estimator_cost("regression", "gradient_boosting") == EstimatorCost(
            "high", "one"
        )
        assert estimator_cost("classification", "logistic") == EstimatorCost(
            "low", "one"
        )

    def test_the_report_carries_it_beside_each_entry(self):
        entry = _entry("regression", "random_forest")
        assert entry["fit_cost"] == "high" and entry["threads"] == "budget"
        report = modeling_capabilities()["estimator_cost"]
        assert report["fit_cost"] == list(FIT_COSTS)
        assert report["threads"] == list(THREAD_KINDS)
        assert report["note"] == ESTIMATOR_COST_NOTE
        assert "15,030-row, 8-feature" in ESTIMATOR_COST_NOTE

    def test_the_adapter_key_set_is_unchanged(self):
        from sklearn.ensemble import RandomForestRegressor

        assert "fit_cost" not in RegressionAdapter().capabilities(RandomForestRegressor)


class TestTheDeclarationIsChecked:
    def test_an_unknown_class_or_kind_is_refused(self):
        class Dummy:
            def fit(self, X, y):
                return self

        schema = EstimatorParamSchema(bounds={})
        for cost in (EstimatorCost("cheap", "one"), EstimatorCost("low", "many")):
            with pytest.raises(ValidationError, match="fit_cost"):
                register_estimator(
                    "regression", "cost_test_dummy", Dummy, schema, cost=cost
                )
        assert ("regression", "cost_test_dummy") not in ESTIMATOR_REGISTRY

    def test_none_is_unmeasured(self):
        class Dummy:
            def fit(self, X, y):
                return self

        register_estimator(
            "regression",
            "cost_test_unmeasured",
            Dummy,
            EstimatorParamSchema(bounds={}),
            cost=EstimatorCost(None, "one"),
        )
        try:
            entry = _entry("regression", "cost_test_unmeasured")
            assert entry["fit_cost"] is None and entry["threads"] == "one"
        finally:
            from standard_quant_tools.modeling.estimators import registry

            for table in (ESTIMATOR_REGISTRY, registry._PARAM_SCHEMAS, registry._COSTS):
                table.pop(("regression", "cost_test_unmeasured"), None)


class TestTheForestAllowlist:
    def test_the_three_new_parameters(self):
        validate_params(
            "regression",
            "random_forest",
            {"max_features": 0.33, "max_samples": 0.5, "min_samples_leaf": 20},
        )
        validate_params("classification", "random_forest", {"max_features": "sqrt"})
        validate_params("regression", "random_forest", {"max_features": None})

    def test_a_whole_number_is_not_read_as_a_share(self):
        """scikit-learn reads an int as a count: max_features=1 is one
        feature per split where 1.0 is all of them."""
        with pytest.raises(ValidationError, match="decimal point"):
            validate_params("regression", "random_forest", {"max_features": 1})
        with pytest.raises(ValidationError, match="decimal point"):
            validate_params("regression", "random_forest", {"max_samples": 2})

    def test_out_of_range_and_unknown_choices_are_refused(self):
        with pytest.raises(ValidationError):
            validate_params("regression", "random_forest", {"max_features": 1.5})
        with pytest.raises(ValidationError):
            validate_params("regression", "random_forest", {"max_features": "all"})
        with pytest.raises(ValidationError):
            validate_params("regression", "random_forest", {"min_samples_leaf": 0})
