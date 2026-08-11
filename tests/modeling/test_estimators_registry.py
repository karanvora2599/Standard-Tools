"""Tests for modeling.estimators.registry: get_estimator_class/validate_params
must both raise ValidationError for an unregistered (task, name), never a
raw KeyError -- validate_params is callable independently of
get_estimator_class (a caller could call it first), so it needs its own
existence check, not just a lookup that assumes the pair is already known-good."""

import pytest

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.estimators.registry import (
    ESTIMATOR_REGISTRY,
    get_estimator_class,
    validate_params,
)


class TestGetEstimatorClass:
    def test_known_pair_returns_class(self):
        assert (
            get_estimator_class("regression", "ridge")
            is ESTIMATOR_REGISTRY[("regression", "ridge")]
        )

    def test_unknown_name_raises_validation_error(self):
        with pytest.raises(ValidationError, match="unknown estimator"):
            get_estimator_class("regression", "not_a_real_estimator")

    def test_wrong_task_for_known_name_raises_validation_error(self):
        with pytest.raises(ValidationError, match="unknown estimator"):
            get_estimator_class("classification", "elastic_net")  # regression-only


class TestValidateParams:
    def test_allowed_params_pass(self):
        validate_params("regression", "ridge", {"alpha": 1.0})  # must not raise

    def test_disallowed_param_raises(self):
        with pytest.raises(ValidationError, match="does not accept"):
            validate_params("regression", "ridge", {"not_a_real_param": 1})

    def test_unregistered_pair_raises_validation_error_not_key_error(self):
        """Called standalone (not preceded by get_estimator_class), an
        unregistered (task, name) used to raise a raw KeyError from the
        internal _ALLOWED_PARAMS dict lookup instead of the same clear
        ValidationError get_estimator_class reports for the identical
        condition."""
        with pytest.raises(ValidationError, match="unknown estimator"):
            validate_params("regression", "not_a_real_estimator", {})

    def test_unregistered_task_raises_validation_error_not_key_error(self):
        with pytest.raises(ValidationError, match="unknown estimator"):
            validate_params("not_a_real_task", "ridge", {})
