"""
ESTIMATOR_REGISTRY: (task, name) -> sklearn class, with an explicit
allowed-params set per entry. engine.py refuses anything not in this
registry — no arbitrary sklearn import, no exec() — and validate_params
rejects any ModelSpec.estimator.params key outside the allowlist for that
estimator, so a caller can't smuggle in an unvetted constructor kwarg.
"""

from typing import Any, Dict, Tuple, Type

from standard_quant_tools.error import ValidationError

from .bounds import EstimatorParamSchema

ESTIMATOR_REGISTRY: Dict[Tuple[str, str], Type] = {}
_PARAM_SCHEMAS: Dict[Tuple[str, str], EstimatorParamSchema] = {}


def register_estimator(
    task: str,
    name: str,
    cls: Type,
    schema: EstimatorParamSchema,
    *,
    overwrite: bool = False,
) -> None:
    """
    Add an estimator to the allowlist.

    `overwrite` defaults to False, matching features.registry's
    register_feature. Previously this silently replaced any existing entry,
    which made the estimator allowlist — the thing that decides what an
    agent is permitted to run at all — weaker than the feature registry it
    sits beside: an accidental re-registration could swap the class behind
    an established name with no error.

    Raises:
        ValidationError: (task, name) already registered and overwrite=False.
    """
    key = (task, name)
    if key in ESTIMATOR_REGISTRY and not overwrite:
        raise ValidationError(
            f"estimator (task={task!r}, name={name!r}) is already registered — pass "
            "overwrite=True to replace it, or choose a different name."
        )
    ESTIMATOR_REGISTRY[key] = cls
    _PARAM_SCHEMAS[key] = schema


def get_estimator_class(task: str, name: str) -> Type:
    key = (task, name)
    if key not in ESTIMATOR_REGISTRY:
        allowed = sorted(n for t, n in ESTIMATOR_REGISTRY if t == task)
        raise ValidationError(
            f"unknown estimator name={name!r} for task={task!r} — allowed: {allowed}"
        )
    return ESTIMATOR_REGISTRY[key]


def validate_params(task: str, name: str, params: Dict[str, Any]) -> None:
    """
    Validate parameter names AND values against the estimator's schema.

    Name-only validation (the previous behavior) still let a caller request
    an unbounded n_estimators/max_iter/max_depth, which is an
    agent-triggerable CPU and memory exhaustion path, and let incompatible
    combinations (logistic penalty vs solver) fail deep inside sklearn
    rather than here.
    """
    key = (task, name)
    if key not in _PARAM_SCHEMAS:
        # Callable independently of get_estimator_class (a caller could
        # call validate_params first) -- must not raise a raw KeyError
        # for the same "unknown estimator" condition get_estimator_class
        # already reports as a clear ValidationError.
        allowed = sorted(n for t, n in ESTIMATOR_REGISTRY if t == task)
        raise ValidationError(
            f"unknown estimator name={name!r} for task={task!r} — allowed: {allowed}"
        )
    _PARAM_SCHEMAS[key].validate(name, params)


def allowed_params(task: str, name: str) -> "list[str]":
    """Parameter names this estimator accepts — used by error messages and
    by any caller that wants to surface the allowlist."""
    key = (task, name)
    if key not in _PARAM_SCHEMAS:
        return []
    return _PARAM_SCHEMAS[key].allowed_names


def validate_param_value(task: str, name: str, param: str, value: Any) -> None:
    """
    Check ONE parameter's value against its bound, without the
    cross-parameter compatibility checks.

    `validate_params` is the whole-dict check and is right for a spec's
    `estimator.params`. A search grid is different: each axis is a list of
    candidates, and a candidate on one axis may only be compatible with a
    candidate on another (logistic's penalty='l1' needs solver='saga' or
    'liblinear'). Validating each axis against the base params would
    therefore refuse grids the engine runs fine, so a grid is checked one
    value at a time against the bound alone, and the combination is left
    to the fit -- which is where the engine already scores an impossible
    candidate as NaN rather than aborting the search.

    Raises:
        ValidationError: unknown estimator, unknown parameter name, or a
        value outside the bound.
    """
    key = (task, name)
    if key not in _PARAM_SCHEMAS:
        allowed = sorted(n for t, n in ESTIMATOR_REGISTRY if t == task)
        raise ValidationError(
            f"unknown estimator name={name!r} for task={task!r} — allowed: {allowed}"
        )
    schema = _PARAM_SCHEMAS[key]
    if param not in schema.bounds:
        raise ValidationError(
            f"estimator {name!r} does not accept param {param!r} — "
            f"allowed: {schema.allowed_names}"
        )
    schema.bounds[param].validate(name, param, value)
