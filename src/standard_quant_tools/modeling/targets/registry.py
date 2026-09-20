"""
TARGET_REGISTRY -- every label this library understands, and the ONE place
that says so.

The same shape as the feature, estimator and preprocessor registries:
`register_target` is how a label gets in, it refuses to replace an entry
silently, and everything that used to read a dict or a Literal reads a
view of this instead. `TARGET_KINDS` and `EXTERNAL_TARGETS` are those
views -- live, so a label registered after import is seen by the spec
validator, the capability report and the generated reference alike.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterator, List, Tuple

from standard_quant_tools.error import ValidationError

from ..tasks import TASKS
from .base import TargetDefinition, TargetKind

TARGET_REGISTRY: Dict[str, TargetDefinition] = {}


def register_target(definition: TargetDefinition, *, overwrite: bool = False) -> None:
    """
    Add (or, with overwrite=True, replace) a label in the registry.

    Raises:
        ValidationError: the id is taken and overwrite=False; a buildable
        label has no builder; an external label HAS one; a task named is
        not a task; the description is too short to be one.
    """
    if definition.id in TARGET_REGISTRY and not overwrite:
        raise ValidationError(
            f"target id {definition.id!r} already registered — pass "
            "overwrite=True to replace it, or choose a different id."
        )
    if not definition.tasks:
        raise ValidationError(
            f"target {definition.id!r} names no task, so it could be declared "
            "and never fitted."
        )
    unknown = [t for t in definition.tasks if t not in TASKS]
    if unknown:
        raise ValidationError(
            f"target {definition.id!r} names task(s) {unknown} that do not "
            f"exist; the tasks are {list(TASKS)}."
        )
    if definition.buildable and definition.builder is None:
        raise ValidationError(
            f"target {definition.id!r} is declared buildable and has no "
            "builder. A label prices can produce needs the function that "
            "produces it."
        )
    if not definition.buildable and definition.builder is not None:
        raise ValidationError(
            f"target {definition.id!r} is declared external-only and carries "
            "a builder. An external label is recorded, never computed: a "
            "bar-derived approximation of a fill probability is a number "
            "with nothing behind it. Register it buildable, or drop the "
            "builder."
        )
    if len(definition.description) < 30:
        raise ValidationError(
            f"target {definition.id!r} carries a description too short to be "
            "one; say what the label measures and what consumes it."
        )
    TARGET_REGISTRY[definition.id] = definition


def get_target(target_id: str) -> TargetDefinition:
    """Raises ValidationError for an unknown id, naming the registry."""
    try:
        return TARGET_REGISTRY[target_id]
    except KeyError:
        raise ValidationError(
            f"unknown target type {target_id!r} — known targets: "
            f"{sorted(TARGET_REGISTRY)}. A label must be registered before a "
            "spec can name it; see modeling.targets.register_target."
        ) from None


def validate_target_params(target_id: str, params: Dict[str, Any]) -> None:
    """Names AND values, against the label's bounded schema."""
    get_target(target_id).param_schema.validate(target_id, params)


def targets_for_task(task: str) -> Tuple[str, ...]:
    """Every label a given task can be fitted against."""
    return tuple(name for name, d in TARGET_REGISTRY.items() if task in d.tasks)


def list_targets() -> List[TargetDefinition]:
    return sorted(TARGET_REGISTRY.values(), key=lambda d: d.id)


class _KindsView(Mapping):
    """`TARGET_KINDS`: {id: TargetKind}, read live off the registry."""

    def __getitem__(self, key: str) -> TargetKind:
        return TARGET_REGISTRY[key].kind

    def __iter__(self) -> Iterator[str]:
        return iter(TARGET_REGISTRY)

    def __len__(self) -> int:
        return len(TARGET_REGISTRY)

    def __repr__(self) -> str:
        return f"TARGET_KINDS({sorted(TARGET_REGISTRY)})"


class _ExternalView(Sequence):
    """`EXTERNAL_TARGETS`: the ids no OHLCV frame can produce, read live."""

    def _ids(self) -> List[str]:
        return [name for name, d in TARGET_REGISTRY.items() if not d.buildable]

    def __getitem__(self, index):
        return self._ids()[index]

    def __len__(self) -> int:
        return len(self._ids())

    def __contains__(self, item: object) -> bool:
        return item in self._ids()

    def __repr__(self) -> str:
        return f"EXTERNAL_TARGETS({self._ids()})"


class _CrossSectionalView:
    """`CROSS_SECTIONAL_TARGETS`: the ids defined against the date's other
    entities. Set-like, read live."""

    def _ids(self) -> frozenset:
        return frozenset(n for n, d in TARGET_REGISTRY.items() if d.cross_sectional)

    def __contains__(self, item: object) -> bool:
        return item in self._ids()

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._ids()))

    def __len__(self) -> int:
        return len(self._ids())


TARGET_KINDS = _KindsView()
EXTERNAL_TARGETS = _ExternalView()
CROSS_SECTIONAL_TARGETS = _CrossSectionalView()

__all__ = [
    "CROSS_SECTIONAL_TARGETS",
    "EXTERNAL_TARGETS",
    "TARGET_KINDS",
    "TARGET_REGISTRY",
    "get_target",
    "list_targets",
    "register_target",
    "targets_for_task",
    "validate_target_params",
]
