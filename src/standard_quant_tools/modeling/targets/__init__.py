"""
Labels as a registry, with `register_target` as the extension point.

Importing this package registers the built-in labels into TARGET_REGISTRY
as a side effect, the way `modeling.features`, `modeling.estimators` and
`modeling.preprocessing` register theirs.
"""

from . import builtin  # noqa: F401  (registration side effect)
from .base import TargetDefinition, TargetKind
from .builtin import horizon_label_end
from .registry import (
    CROSS_SECTIONAL_TARGETS,
    EXTERNAL_TARGETS,
    TARGET_KINDS,
    TARGET_REGISTRY,
    get_target,
    list_targets,
    register_target,
    targets_for_task,
    validate_target_params,
)

__all__ = [
    "CROSS_SECTIONAL_TARGETS",
    "EXTERNAL_TARGETS",
    "TARGET_KINDS",
    "TARGET_REGISTRY",
    "TargetDefinition",
    "TargetKind",
    "get_target",
    "horizon_label_end",
    "list_targets",
    "register_target",
    "targets_for_task",
    "validate_target_params",
]
