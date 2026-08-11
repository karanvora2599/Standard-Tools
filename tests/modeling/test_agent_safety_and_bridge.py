"""
Regression tests for the P1 agent-safety and model->backtest-bridge
findings.

The estimator registry allowlisted parameter NAMES but placed no constraint
on VALUES, which left an agent-triggerable resource-exhaustion path and let
incompatible combinations fail deep inside sklearn. The bridge trusted its
inputs: the caller supplied `task` independently of the model, and the
predictions artifact was consumed without structural validation.
"""

import numpy as np
import pandas as pd
import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.bridge import (
    _validate_predictions_frame,
    oos_predictions_to_signal_panel,
)
from standard_quant_tools.modeling.estimators.bounds import (
    EstimatorParamSchema,
    ParamBound,
)
from standard_quant_tools.modeling.estimators.registry import (
    ESTIMATOR_REGISTRY,
    register_estimator,
    validate_params,
)


class TestResourceBudgets:
    """
    An allowlist of parameter names is not a compute budget. Each of these
    could previously be requested in a single tool call and would pin CPU
    and memory for as long as sklearn kept working.
    """

    @pytest.mark.parametrize(
        "task,name,params",
        [
            ("regression", "random_forest", {"n_estimators": 10_000_000}),
            ("classification", "random_forest", {"n_estimators": 500_000}),
            ("regression", "gradient_boosting", {"n_estimators": 1_000_000}),
            ("regression", "hist_gradient_boosting", {"max_iter": 10**9}),
            ("regression", "random_forest", {"max_depth": 10_000}),
        ],
    )
    def test_runaway_values_rejected(self, task, name, params):
        with pytest.raises(ValidationError, match="exceeds the maximum"):
            validate_params(task, name, params)

    @pytest.mark.parametrize(
        "task,name,params",
        [
            ("regression", "random_forest", {"n_estimators": 200, "max_depth": 8}),
            ("regression", "hist_gradient_boosting", {"max_iter": 500}),
            ("regression", "ridge", {"alpha": 1.0}),
            ("regression", "elastic_net", {"alpha": 0.5, "l1_ratio": 0.5}),
        ],
    )
    def test_realistic_values_accepted(self, task, name, params):
        """The ceilings must not obstruct ordinary research requests."""
        validate_params(task, name, params)

    def test_negative_and_non_finite_rejected(self):
        with pytest.raises(ValidationError, match="below the minimum"):
            validate_params("regression", "ridge", {"alpha": -1.0})
        with pytest.raises(ValidationError, match="finite"):
            validate_params("regression", "ridge", {"alpha": float("inf")})

    def test_wrong_type_rejected(self):
        with pytest.raises(ValidationError, match="must be a number"):
            validate_params("regression", "random_forest", {"n_estimators": "200"})

    def test_bool_is_not_accepted_as_a_count(self):
        """bool subclasses int; True must not pass as n_estimators=1."""
        with pytest.raises(ValidationError, match="must be a number"):
            validate_params("regression", "random_forest", {"n_estimators": True})

    def test_fractional_count_rejected(self):
        with pytest.raises(ValidationError, match="whole number"):
            validate_params("regression", "random_forest", {"n_estimators": 10.5})

    def test_unknown_param_still_rejected(self):
        with pytest.raises(ValidationError, match="does not accept params"):
            validate_params("regression", "ridge", {"nonexistent": 1})


class TestEstimatorCompatibility:
    """
    `penalty` was exposed without `solver`, so an incompatible pair failed
    inside sklearn's .fit() rather than at this boundary.
    """

    def test_l1_with_default_solver_rejected(self):
        with pytest.raises(ValidationError, match="not supported by solver"):
            validate_params("classification", "logistic", {"penalty": "l1"})

    @pytest.mark.parametrize("solver", ["liblinear", "saga"])
    def test_l1_with_compatible_solver_accepted(self, solver):
        validate_params(
            "classification", "logistic", {"penalty": "l1", "solver": solver}
        )

    def test_elasticnet_requires_saga(self):
        with pytest.raises(ValidationError, match="not supported by solver"):
            validate_params(
                "classification",
                "logistic",
                {"penalty": "elasticnet", "solver": "lbfgs"},
            )

    def test_elasticnet_requires_l1_ratio(self):
        with pytest.raises(ValidationError, match="requires l1_ratio"):
            validate_params(
                "classification",
                "logistic",
                {"penalty": "elasticnet", "solver": "saga"},
            )

    def test_elasticnet_fully_specified_accepted(self):
        validate_params(
            "classification",
            "logistic",
            {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5},
        )

    def test_unknown_solver_rejected(self):
        """Caught by the `choices` bound before the compatibility check
        runs — the earlier, more specific error is the right one to
        surface, since an unknown solver is not a compatibility question."""
        with pytest.raises(ValidationError, match="is not one of"):
            validate_params("classification", "logistic", {"solver": "made_up"})


class TestRegistryOverwriteProtection:
    """
    register_estimator silently replaced an existing entry, making the
    allowlist that decides what an agent may run WEAKER than the feature
    registry beside it (register_feature has always required overwrite=True).
    """

    def test_duplicate_registration_rejected(self):
        cls = ESTIMATOR_REGISTRY[("regression", "ridge")]
        with pytest.raises(ValidationError, match="already registered"):
            register_estimator(
                "regression", "ridge", cls, EstimatorParamSchema(bounds={})
            )

    def test_explicit_overwrite_allowed_then_restored(self):
        key = ("regression", "ridge")
        original_cls = ESTIMATOR_REGISTRY[key]
        schema = EstimatorParamSchema(bounds={"alpha": ParamBound("float", 0.0, 10.0)})
        try:
            register_estimator(
                "regression", "ridge", original_cls, schema, overwrite=True
            )
            with pytest.raises(ValidationError, match="exceeds the maximum"):
                validate_params("regression", "ridge", {"alpha": 1e6})
        finally:
            from standard_quant_tools.modeling.estimators.bounds import (
                ALPHA,
                FIT_INTERCEPT,
                MAX_ITER,
            )

            register_estimator(
                "regression",
                "ridge",
                original_cls,
                EstimatorParamSchema(
                    bounds={
                        "alpha": ALPHA,
                        "fit_intercept": FIT_INTERCEPT,
                        "max_iter": MAX_ITER,
                    }
                ),
                overwrite=True,
            )
        validate_params("regression", "ridge", {"alpha": 1e6})


def _frame(**overrides) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "entity": ["AAA", "AAA"],
            "prediction": [0.01, -0.02],
        }
    )
    for key, value in overrides.items():
        base[key] = value
    return base


class TestPredictionArtifactValidation:
    """
    The frame was consumed on trust. The silent failure mattered most:
    duplicate (entity, date) rows overwrote each other in the output dict,
    producing a smaller but perfectly valid-looking signal panel.
    """

    def test_missing_column_named_clearly(self):
        df = _frame().drop(columns=["prediction"])
        with pytest.raises(ValidationError, match="missing column"):
            _validate_predictions_frame(df, "artifact")

    def test_empty_frame_rejected(self):
        with pytest.raises(ValidationError, match="no rows"):
            _validate_predictions_frame(_frame().iloc[:0], "artifact")

    def test_wrong_date_dtype_rejected(self):
        df = _frame()
        df["date"] = ["2024-01-01", "2024-01-02"]
        with pytest.raises(ValidationError, match="must be datetime64"):
            _validate_predictions_frame(df, "artifact")

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_prediction_rejected(self, bad):
        df = _frame()
        df.loc[0, "prediction"] = bad
        with pytest.raises(ValidationError, match="non-finite"):
            _validate_predictions_frame(df, "artifact")

    def test_duplicate_entity_date_rejected(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "entity": ["AAA", "AAA"],
                "prediction": [0.01, 0.99],
            }
        )
        with pytest.raises(ValidationError, match="duplicate"):
            _validate_predictions_frame(df, "artifact")

    def test_valid_frame_passes(self):
        _validate_predictions_frame(_frame(), "artifact")


class TestBridgeTaskResolution:
    def test_requires_exactly_one_of_model_id_or_uri(self):
        with pytest.raises(ValidationError, match="exactly one of model_id"):
            oos_predictions_to_signal_panel()
        with pytest.raises(ValidationError, match="exactly one of model_id"):
            oos_predictions_to_signal_panel(
                oos_predictions_uri="x", model_id="mdl_x", task="regression"
            )

    def test_uri_without_task_rejected(self):
        """Previously `task` was positional-required; now that model_id is
        the preferred entry point, omitting both must still fail clearly."""
        with pytest.raises(ValidationError, match="task is required"):
            oos_predictions_to_signal_panel(oos_predictions_uri="x")
