"""
Point-in-time safety: reject a DatasetSpec that would let a CURRENT_ONLY
feature (e.g. today's fundamentals, with no point-in-time-safe historical
provider) leak into a historical training window. This is a real
correctness bug — silent leakage, not a nice-to-have — so the check runs
unconditionally on every build_dataset call.

Nothing in Phase 1 registers a CURRENT_ONLY feature (all built-in
features are PIT_SAFE — price/volume-derived only), but the mechanism is
built now rather than deferred, per the explicit ask to keep the
structure ready for a future fundamentals feature: that feature will be
rejected by construction the moment it's registered as CURRENT_ONLY and
used here, instead of silently leaking.
"""

from typing import List

from standard_quant_tools.error import ValidationError

from ..features.base import FeatureDefinition, TemporalSupport


class PointInTimeViolation(ValidationError):
    """Raised when a DatasetSpec would let a CURRENT_ONLY feature leak
    into a historical training window."""


def check_point_in_time_safety(feature_defs: List[FeatureDefinition]) -> None:
    offending = [
        d.id for d in feature_defs if d.temporal_support == TemporalSupport.CURRENT_ONLY
    ]
    if offending:
        raise PointInTimeViolation(
            f"features {offending} are temporal_support=CURRENT_ONLY and cannot be "
            "used in a historical training dataset — no point-in-time-safe historical "
            "provider exists for them yet. Use only PIT_SAFE features for build_model_dataset."
        )
