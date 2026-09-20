"""
PREPROCESSOR_REGISTRY -- the catalog of steps a PreprocessingSpec may name.

The same shape as the feature and estimator registries, for the same
reason: a step named in a JSON spec is looked up here and nowhere else, so
an agent can compose a pipeline from a bounded catalog and cannot smuggle
in a transform the library has not vetted. `register_preprocessor` is the
extension point a firm's own transform goes through, and it refuses to
replace an entry silently, as `register_feature` and `register_estimator`
do.
"""

from __future__ import annotations

from typing import Any, Dict, List

from standard_quant_tools.error import ValidationError

from .base import PreprocessorDefinition

PREPROCESSOR_REGISTRY: Dict[str, PreprocessorDefinition] = {}


def register_preprocessor(
    definition: PreprocessorDefinition, *, overwrite: bool = False
) -> None:
    """
    Add (or, with overwrite=True, replace) a step in the catalog.

    Raises:
        ValidationError: `definition.id` already registered and
        overwrite=False, or the id does not match the class's own.
    """
    if getattr(definition.cls, "id", None) != definition.id:
        raise ValidationError(
            f"preprocessor {definition.id!r}: the class {definition.cls.__name__} "
            f"declares id={getattr(definition.cls, 'id', None)!r}. The registry "
            "key and the class must agree, or a fitted state would name a step "
            "the pipeline cannot rebuild."
        )
    if definition.id in PREPROCESSOR_REGISTRY and not overwrite:
        raise ValidationError(
            f"preprocessor id {definition.id!r} already registered — pass "
            "overwrite=True to replace it, or choose a different id."
        )
    PREPROCESSOR_REGISTRY[definition.id] = definition


def get_preprocessor(step_id: str) -> PreprocessorDefinition:
    """Raises ValidationError for an unknown id, naming the catalog."""
    try:
        return PREPROCESSOR_REGISTRY[step_id]
    except KeyError:
        raise ValidationError(
            f"unknown preprocessing step {step_id!r} — known steps: "
            f"{sorted(PREPROCESSOR_REGISTRY)}. A step must be registered before a "
            "spec can name it; see modeling.preprocessing.register_preprocessor."
        ) from None


def validate_step_params(step_id: str, params: Dict[str, Any]) -> None:
    """Names AND values, against the step's bounded schema -- the same
    boundary `validate_params` gives an estimator."""
    get_preprocessor(step_id).schema_.validate(step_id, params)


def list_preprocessors() -> List[PreprocessorDefinition]:
    return sorted(PREPROCESSOR_REGISTRY.values(), key=lambda d: d.id)


__all__ = [
    "PREPROCESSOR_REGISTRY",
    "get_preprocessor",
    "list_preprocessors",
    "register_preprocessor",
    "validate_step_params",
]
