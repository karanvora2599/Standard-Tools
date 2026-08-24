"""
What this runtime can currently do, assembled from the registries themselves.

An agent choosing a model needs to know more than a list of names: whether an
estimator can take sample weights, whether it emits probabilities, whether it
needs query groups, whether it is even installed. The alternative to answering
that is one tool per model, which would grow the modeling surface without
adding a single decision to it.

Everything here is READ OFF the live registries and the model adapters rather
than written down. A newly registered estimator therefore describes itself
correctly without anyone remembering to update a table — which is the failure
mode a hand-maintained capability list always eventually has.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .adapters import available_tasks, get_adapter
from .estimators.registry import ESTIMATOR_REGISTRY, allowed_params
from .features.registry import list_features as _list_features
from .specs import TargetSpec, ValidationSpec


def _literal_options(model: Any, field: str) -> List[str]:
    """The declared choices for a Literal-typed spec field.

    Read from the model rather than duplicated, so adding a target type or a
    validation method shows up here automatically.
    """
    annotation = model.model_fields[field].annotation
    args = getattr(annotation, "__args__", ())
    return [a for a in args if isinstance(a, str)]


def estimator_capabilities() -> List[Dict[str, Any]]:
    """One entry per (task, estimator) actually available in this install."""
    out: List[Dict[str, Any]] = []
    for (task, name), cls in sorted(ESTIMATOR_REGISTRY.items()):
        entry: Dict[str, Any] = {
            "name": name,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "allowed_params": allowed_params(task, name),
        }
        try:
            entry.update(get_adapter(task).capabilities(cls))
        except Exception:  # noqa: BLE001 - a bad adapter must not hide the rest
            entry["task"] = task
        out.append(entry)
    return out


def modeling_capabilities() -> Dict[str, Any]:
    """
    The whole surface: tasks, estimators, features, targets, validation
    schemes, preprocessing, weighting, and which optional libraries are
    present.

    `optional_dependencies` is the part an agent most needs and cannot infer:
    lightgbm and xgboost are not declared dependencies, so the ranking task
    and the fast boosters exist on one machine and not another. Reporting
    absence explicitly is more useful than an estimator list that is silently
    shorter.
    """
    from .estimators import boosting

    features = _list_features()
    by_namespace: Dict[str, int] = {}
    for definition in features:
        by_namespace[definition.id.split(".", 1)[0]] = (
            by_namespace.get(definition.id.split(".", 1)[0], 0) + 1
        )

    return {
        "tasks": available_tasks(),
        "estimators": estimator_capabilities(),
        "features": {
            "count": len(features),
            "by_namespace": dict(sorted(by_namespace.items())),
            "ids": sorted(d.id for d in features),
        },
        "targets": _literal_options(TargetSpec, "type"),
        "validation": {
            "methods": _literal_options(ValidationSpec, "method"),
            "walk_forward_schemes": _literal_options(ValidationSpec, "scheme"),
        },
        "preprocessing": ["pooled", "cross_sectional"],
        "weighting": [
            "none",
            "label_uniqueness",
            "time_decay",
            "uniqueness_and_time_decay",
        ],
        "hyperparameter_search": ["grid", "random"],
        "optional_dependencies": {
            "lightgbm": boosting.HAS_LIGHTGBM,
            "xgboost": boosting.HAS_XGBOOST,
            "native_extension": _native_available(),
        },
    }


def _native_available() -> bool:
    """Whether the compiled fast path is present. Every kernel has a Python
    fallback, so this changes speed and nothing else — but an agent asking
    why a run is slow deserves to be able to find out."""
    try:
        from .features import transforms

        return bool(transforms.HAS_CPP)
    except Exception:  # noqa: BLE001 - absence is the answer, not an error
        return False
