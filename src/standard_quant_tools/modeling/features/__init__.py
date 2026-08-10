"""Importing this package registers every built-in feature into
FEATURE_REGISTRY as a side effect — the same registration-on-import
pattern the individual technical.py/market.py/risk.py/statistical.py/
factors.py modules rely on."""

from . import factors, market, risk, statistical, technical  # noqa: F401  (registration side effect)
from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import FEATURE_REGISTRY, get_feature, list_features, register_feature

__all__ = [
    "FEATURE_REGISTRY",
    "FeatureContext",
    "FeatureDefinition",
    "FeatureScope",
    "TemporalSupport",
    "get_feature",
    "list_features",
    "register_feature",
]
