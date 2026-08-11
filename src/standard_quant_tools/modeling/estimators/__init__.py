"""Importing this package registers every allowlisted estimator into
ESTIMATOR_REGISTRY as a side effect."""

from . import (  # noqa: F401  (registration side effect)
    classification,
    regression,
    trees,
)
from .registry import (
    ESTIMATOR_REGISTRY,
    get_estimator_class,
    register_estimator,
    validate_params,
)

__all__ = [
    "ESTIMATOR_REGISTRY",
    "get_estimator_class",
    "register_estimator",
    "validate_params",
]
