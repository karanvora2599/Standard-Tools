"""
Preprocessing as a registry of steps and a pipeline of fitted state.

Importing this package registers the built-in steps into
PREPROCESSOR_REGISTRY as a side effect, the way `modeling.estimators` and
`modeling.features` register theirs.
"""

from . import steps  # noqa: F401  (registration side effect)
from .base import FoldContext, Preprocessor, PreprocessorDefinition
from .pipeline import (
    STATE_VERSION,
    apply_pipeline,
    build_step,
    fit_and_apply_pipeline,
    fit_pipeline,
    legacy_stats,
    step_types,
)
from .registry import (
    PREPROCESSOR_REGISTRY,
    get_preprocessor,
    list_preprocessors,
    register_preprocessor,
    validate_step_params,
)

__all__ = [
    "PREPROCESSOR_REGISTRY",
    "STATE_VERSION",
    "FoldContext",
    "Preprocessor",
    "PreprocessorDefinition",
    "apply_pipeline",
    "build_step",
    "fit_and_apply_pipeline",
    "fit_pipeline",
    "get_preprocessor",
    "legacy_stats",
    "list_preprocessors",
    "register_preprocessor",
    "step_types",
    "validate_step_params",
]
