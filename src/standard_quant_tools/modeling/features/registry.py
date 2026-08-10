"""
FEATURE_REGISTRY — the catalog `list_features`/`dataset.builder` read
from. Entries are added via `register_feature`, the same
registration-not-monkeypatch pattern `estimators.registry` uses for the
estimator allowlist.
"""

from typing import Dict, List, Optional

from standard_quant_tools.error import ValidationError

from .base import FeatureDefinition

FEATURE_REGISTRY: Dict[str, FeatureDefinition] = {}


def register_feature(definition: FeatureDefinition, *, overwrite: bool = False) -> None:
    """
    Add (or, with overwrite=True, replace) a FeatureDefinition in the
    catalog. This is the extension point a firm's proprietary features
    (features/custom.py) go through, identical to how every built-in
    feature (technical.py, market.py, ...) registers itself on import.

    Raises:
        ValidationError: `definition.id` already registered and
        overwrite=False.
    """
    if definition.id in FEATURE_REGISTRY and not overwrite:
        raise ValidationError(
            f"feature id {definition.id!r} already registered — pass overwrite=True "
            "to replace it, or choose a different id."
        )
    FEATURE_REGISTRY[definition.id] = definition


def get_feature(feature_id: str) -> FeatureDefinition:
    """Raises ValidationError for an unknown id (not KeyError — every
    boundary error in this codebase is a ValidationError)."""
    try:
        return FEATURE_REGISTRY[feature_id]
    except KeyError:
        raise ValidationError(
            f"unknown feature id {feature_id!r} — see list_features() for the catalog."
        ) from None


def list_features(category: Optional[str] = None) -> List[FeatureDefinition]:
    """
    The full catalog, sorted by id for deterministic output, optionally
    filtered to one category (the part of the id before the first '.').
    """
    values = sorted(FEATURE_REGISTRY.values(), key=lambda d: d.id)
    if category is None:
        return values
    prefix = f"{category}."
    return [d for d in values if d.id.startswith(prefix)]
