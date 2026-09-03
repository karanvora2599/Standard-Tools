"""Importing this package registers every built-in feature into
FEATURE_REGISTRY as a side effect — the same registration-on-import
pattern the individual technical.py/market.py/risk.py/statistical.py/
factors.py/network.py modules rely on."""

from . import (  # noqa: F401  (registration side effect)
    factors,
    market,
    network,
    risk,
    statistical,
    technical,
    volume,
)
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
