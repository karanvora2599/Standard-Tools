"""
ESTIMATOR_REGISTRY: (task, name) -> sklearn class, with an explicit
allowed-params set per entry. engine.py refuses anything not in this
registry — no arbitrary sklearn import, no exec() — and validate_params
rejects any ModelSpec.estimator.params key outside the allowlist for that
estimator, so a caller can't smuggle in an unvetted constructor kwarg.
"""

from typing import Any, Dict, Set, Tuple, Type

from standard_quant_tools.error import ValidationError

ESTIMATOR_REGISTRY: Dict[Tuple[str, str], Type] = {}
_ALLOWED_PARAMS: Dict[Tuple[str, str], Set[str]] = {}


def register_estimator(
    task: str, name: str, cls: Type, allowed_params: Set[str]
) -> None:
    ESTIMATOR_REGISTRY[(task, name)] = cls
    _ALLOWED_PARAMS[(task, name)] = allowed_params


def get_estimator_class(task: str, name: str) -> Type:
    key = (task, name)
    if key not in ESTIMATOR_REGISTRY:
        allowed = sorted(n for t, n in ESTIMATOR_REGISTRY if t == task)
        raise ValidationError(
            f"unknown estimator name={name!r} for task={task!r} — allowed: {allowed}"
        )
    return ESTIMATOR_REGISTRY[key]


def validate_params(task: str, name: str, params: Dict[str, Any]) -> None:
    key = (task, name)
    if key not in _ALLOWED_PARAMS:
        # Callable independently of get_estimator_class (a caller could
        # call validate_params first) -- must not raise a raw KeyError
        # for the same "unknown estimator" condition get_estimator_class
        # already reports as a clear ValidationError.
        allowed = sorted(n for t, n in ESTIMATOR_REGISTRY if t == task)
        raise ValidationError(
            f"unknown estimator name={name!r} for task={task!r} — allowed: {allowed}"
        )
    allowed = _ALLOWED_PARAMS[key]
    unknown = set(params) - allowed
    if unknown:
        raise ValidationError(
            f"estimator {name!r} (task={task!r}) does not accept params {sorted(unknown)} — "
            f"allowed: {sorted(allowed)}"
        )
