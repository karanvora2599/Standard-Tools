"""
A pipeline: fit a list of steps on training rows, apply the fitted state
anywhere, and carry that state as JSON.

THE STATE IS THE ARTIFACT. `fit_pipeline` returns a plain dict --

    {"version": 1,
     "columns": [...],
     "steps": [{"type": "winsorize", "params": {...}, "state": {...}}, ...]}

-- which the registry writes as `preprocessing_state.json`, content-hashed
beside the estimator, and which `apply_pipeline` rebuilds the steps from.
Nothing about the transform lives outside it: not a Literal in the spec,
not a branch in the engine, not a second file the scoring path has to know
to read. The deployed transform is the validated one because both are this
object.

THE FUSED PATH. The default spec resolves to `winsorize` then `zscore`,
which is exactly `fit_preprocessing`/`apply_preprocessing` -- the pair with
the native kernel that took preprocessing from half a walk-forward run to
a fraction of it. When the resolved steps are that pair, on a float64
frame, the pipeline calls those functions rather than the generic steps,
and reads the state off their statistics. The generic steps are the
reference; a test pins the two paths equal to 1e-12.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.features.transforms import (
    _fit_and_apply_with_stats,
    apply_preprocessing,
    fit_preprocessing,
)

from .base import FoldContext, Preprocessor
from .registry import get_preprocessor

#: The state format. Bumped only if the shape changes; a loader refuses an
#: unknown version rather than guessing.
STATE_VERSION = 1

#: The step pair the fused native path serves. Compared on type and on the
#: parameters that decide the arithmetic.
_DEFAULT_POOLED = (("winsorize", {"lower": 0.01, "upper": 0.99}), ("zscore", {}))


def _resolved_params(step_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    definition = get_preprocessor(step_type)
    return {**definition.default_params, **dict(params or {})}


def build_step(step_type: str, params: Dict[str, Any]) -> Preprocessor:
    """A step instance with the definition's defaults merged under the
    caller's overrides."""
    definition = get_preprocessor(step_type)
    return definition.cls(**_resolved_params(step_type, params))


def _is_default_pooled(steps: Sequence[Tuple[str, Dict[str, Any]]]) -> bool:
    if len(steps) != len(_DEFAULT_POOLED):
        return False
    for (step_type, params), (want_type, want_params) in zip(steps, _DEFAULT_POOLED):
        if step_type != want_type:
            return False
        if {k: float(v) for k, v in params.items()} != want_params:
            return False
    return True


def _normalize(steps: Sequence[Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """(type, resolved params) per step, from StepSpec objects or tuples."""
    out = []
    for step in steps:
        if hasattr(step, "type"):
            step_type, params = step.type, dict(step.params or {})
        else:
            step_type, params = step
        out.append((str(step_type), _resolved_params(str(step_type), params)))
    return out


def _fused_state(columns, stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "columns": list(columns),
        "steps": [
            {
                "type": "winsorize",
                "params": {"lower": 0.01, "upper": 0.99},
                "state": {
                    "lo": {c: stats[c]["lo"] for c in columns},
                    "hi": {c: stats[c]["hi"] for c in columns},
                },
            },
            {
                "type": "zscore",
                "params": {},
                "state": {
                    "mean": {c: stats[c]["mean"] for c in columns},
                    "std": {c: stats[c]["std"] for c in columns},
                },
            },
        ],
    }


def _fused_stats(state: Dict[str, Any]) -> Optional[Dict[str, Dict[str, float]]]:
    """The per-column statistics dict `apply_preprocessing` takes, when the
    state is the default pooled pair; None otherwise."""
    steps = state.get("steps") or []
    if not _is_default_pooled([(s["type"], s.get("params") or {}) for s in steps]):
        return None
    lo, hi = steps[0]["state"]["lo"], steps[0]["state"]["hi"]
    mean, std = steps[1]["state"]["mean"], steps[1]["state"]["std"]
    return {
        c: {"lo": lo[c], "hi": hi[c], "mean": mean[c], "std": std[c]}
        for c in state["columns"]
    }


def legacy_stats(state: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    The `preprocessing_stats.json` projection of a state.

    The per-column winsorize/zscore statistics when the pipeline is the
    default pooled pair -- byte-identical to what `fit_preprocessing`
    persisted before the registry existed -- and empty otherwise, which is
    what a cross-sectional model persisted. Kept for one release so a
    reader of the older file keeps loading; the state file is the record.
    """
    return _fused_stats(state) or {}


def fit_pipeline(
    steps: Sequence[Any], X: pd.DataFrame, ctx: FoldContext
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Fit every step, in order, on `X` -- each on the previous step's output
    -- and return (state, transformed X).

    `X` is the TRAINING rows. That is a rule this function cannot enforce
    and the engine does: a state fitted on rows the estimator is then
    scored against describes a pipeline nobody validated.
    """
    normalized = _normalize(steps)
    columns = list(X.columns)
    if _is_default_pooled(normalized):
        stats = fit_preprocessing(X)
        return _fused_state(columns, stats), apply_preprocessing(X, stats)

    fitted: List[Dict[str, Any]] = []
    current = X
    for step_type, params in normalized:
        step = build_step(step_type, params)
        state = step.fit(current, ctx)
        current = step.transform(current, state, ctx)
        fitted.append({"type": step_type, "params": params, "state": state})
    return {"version": STATE_VERSION, "columns": columns, "steps": fitted}, current


def apply_pipeline(
    state: Dict[str, Any], X: pd.DataFrame, ctx: FoldContext
) -> pd.DataFrame:
    """Apply a fitted state to any rows sharing the fitted columns."""
    if int(state.get("version", -1)) != STATE_VERSION:
        raise ValidationError(
            f"apply_pipeline: unknown preprocessing state version "
            f"{state.get('version')!r}; this library reads {STATE_VERSION}."
        )
    columns = list(state["columns"])
    if list(X.columns) != columns:
        raise ValidationError(
            "apply_pipeline: the rows carry columns "
            f"{list(X.columns)[:8]}{'...' if X.shape[1] > 8 else ''} but the "
            f"state was fitted on {columns[:8]}{'...' if len(columns) > 8 else ''}. "
            "A fitted transform applied to different columns is a different "
            "variable under the same name."
        )
    stats = _fused_stats(state)
    if stats is not None:
        return apply_preprocessing(X, stats)
    current = X
    for entry in state["steps"]:
        step = build_step(entry["type"], entry.get("params") or {})
        current = step.transform(current, entry.get("state") or {}, ctx)
    return current


def fit_and_apply_pipeline(
    steps: Sequence[Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_ctx: FoldContext,
    test_ctx: FoldContext,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Fit on `train`, apply to both, materialising each frame once.

    The fold loop's call. For the default pooled pair it goes through the
    fused native helper, which converts the training block to a contiguous
    matrix once rather than once to fit and once to apply -- the copy the
    native plan measured at 3.7 ms per fold on a 100,000 x 20 block.
    """
    normalized = _normalize(steps)
    if _is_default_pooled(normalized) and list(train.columns) == list(test.columns):
        train_out, test_out, stats = _fit_and_apply_with_stats(train, test)
        return _fused_state(list(train.columns), stats), train_out, test_out
    state, train_out = fit_pipeline(normalized, train, train_ctx)
    return state, train_out, apply_pipeline(state, test, test_ctx)


def step_types(state_or_steps: Any) -> List[str]:
    """The step ids, from a state dict or a step list -- for reports."""
    if isinstance(state_or_steps, dict):
        return [s["type"] for s in state_or_steps.get("steps") or []]
    return [t for t, _ in _normalize(state_or_steps)]


__all__ = [
    "STATE_VERSION",
    "apply_pipeline",
    "build_step",
    "fit_and_apply_pipeline",
    "fit_pipeline",
    "legacy_stats",
    "step_types",
]
