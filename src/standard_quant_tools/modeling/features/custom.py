"""Extension point for caller-supplied features — a firm's proprietary
alt-data, analyst scores, or internal signals register through the same
FEATURE_REGISTRY every built-in feature uses:

    from standard_quant_tools.modeling.features.custom import (
        FeatureDefinition, FeatureScope, TemporalSupport, register_feature,
    )

    register_feature(FeatureDefinition(
        id="firm.altdata.customer_growth",
        description="...",
        fn=my_feature_fn,
        temporal_support=TemporalSupport.CURRENT_ONLY,
        scope=FeatureScope.ENTITY,
        requires=["Close"],
        lookback=0,
    ))

This module is deliberately thin — register_feature (features.registry)
already does the real work; this is the documented, discoverable entry
point for "how do I add my own feature."
"""

from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import register_feature

__all__ = [
    "FeatureContext",
    "FeatureDefinition",
    "FeatureScope",
    "TemporalSupport",
    "register_feature",
]
