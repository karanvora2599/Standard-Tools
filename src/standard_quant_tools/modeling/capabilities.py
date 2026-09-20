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

from . import calendar as _calendar
from .adapters import available_tasks, get_adapter
from .estimators.registry import (
    ESTIMATOR_REGISTRY,
    allowed_params,
    quantile_support,
)
from .features.registry import list_features as _list_features
from .preprocessing import list_preprocessors
from .specs import (
    TARGET_KINDS,
    PreprocessingSpec,
    SearchSpec,
    TargetSpec,
    ValidationSpec,
)
from .validation import search as _search


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
        support = quantile_support(task, name)
        entry: Dict[str, Any] = {
            "name": name,
            "class": f"{cls.__module__}.{cls.__qualname__}",
            "allowed_params": allowed_params(task, name),
            # The constructor argument that names a quantile, when the
            # estimator can fit one; None otherwise. What decides whether
            # `ModelSpec.quantiles` can be asked of it.
            "quantile_param": support.param if support is not None else None,
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
        # FROM THE REGISTRY, not from the Literal. `TARGET_KINDS` calls
        # itself "the ONE place that says so" and carries `buildable` and a
        # description per label; reporting `_literal_options` instead listed
        # all 18 names flat, and only 6 can be built from a price series.
        # The other 12 raise "cannot be built from a price series" inside
        # build_model_dataset.
        #
        # This is a worse place to be wrong than a tool is. An agent reads
        # the capability report INSTEAD of trying things, so an overstatement
        # here is not one failed call, it is a plan built on a tool that
        # cannot do what it was told.
        "targets": {
            "buildable": sorted(
                name for name, kind in TARGET_KINDS.items() if kind.buildable
            ),
            "external_only": sorted(
                name for name, kind in TARGET_KINDS.items() if not kind.buildable
            ),
            # From the registry view, which a label registered at runtime
            # joins; the field is no longer a Literal to read options off.
            "all": sorted(TARGET_KINDS),
            "note": (
                "`buildable` is what build_model_dataset can derive from a "
                "Close series. `external_only` labels are functions of the "
                "book, of orders or of fills -- nothing in a Close column "
                "determines them -- so they arrive through "
                "register_external_panel, which records what a label IS "
                "rather than recomputing it."
            ),
            "detail": {
                name: {
                    "buildable": kind.buildable,
                    "tasks": list(kind.tasks),
                    "continuous": kind.continuous,
                    "description": kind.description,
                }
                for name, kind in sorted(TARGET_KINDS.items())
            },
        },
        "validation": {
            "methods": _literal_options(ValidationSpec, "method"),
            "walk_forward_schemes": _literal_options(ValidationSpec, "scheme"),
        },
        # FROM THE REGISTRY. This was a hand-written two-item list, which
        # is the failure mode this module's docstring says it exists to
        # avoid; the steps are what a `PreprocessingSpec.steps` pipeline may
        # name, and the schemes are what `normalization` still accepts.
        "preprocessing": {
            "normalization": _literal_options(PreprocessingSpec, "normalization"),
            "steps": [
                {
                    "id": definition.id,
                    "description": definition.description,
                    "params": definition.schema_.allowed_names,
                    "default_params": dict(definition.default_params),
                    "stateless": definition.stateless,
                    "column_wise": definition.column_wise,
                }
                for definition in list_preprocessors()
            ],
        },
        "weighting": [
            "none",
            "label_uniqueness",
            "time_decay",
            "uniqueness_and_time_decay",
        ],
        # From the spec, so a backend added there is reported here. 'tpe'
        # is listed whether or not optuna is installed; the entry under
        # optional_dependencies says whether it can run.
        "hyperparameter_search": _literal_options(SearchSpec, "method"),
        # Widened. It reported three, so an agent could not learn from it
        # that the bloomberg provider or the audit signing path are
        # unavailable in this environment -- and would discover that by
        # making a call that fails.
        "optional_dependencies": {
            "lightgbm": boosting.HAS_LIGHTGBM,
            "xgboost": boosting.HAS_XGBOOST,
            # The search module's own probe, so the capability report and
            # the refusal in run_model_experiment cannot disagree.
            "optuna": _search.optuna_available(),
            # What makes DatasetSpec.calendar resolvable, and with it an
            # intraday interval annualizable.
            "exchange_calendars": _calendar.calendar_available(),
            "native_extension": _native_available(),
            "scipy": _importable("scipy"),
            "numba": _importable("numba"),
            "polars": _importable("polars"),
            "blpapi": _importable("blpapi"),
            "cryptography": _importable("cryptography"),
            # What `FsspecArtifactStore` needs to mirror a verified model
            # package to object storage.
            "fsspec": _importable("fsspec"),
            # What writes `model.skops` beside `model.joblib`: the estimator
            # loadable without executing pickle.
            "skops": _importable("skops"),
            "cvxpy": _importable("cvxpy"),
        },
        # A SIBLING, not an entry in the map above. `optional_dependencies`
        # is name -> bool and callers check it that way; nesting a dict in
        # it broke that contract, which two tests caught immediately.
        "native_extension_detail": _native_detail(),
    }


#: How many symbols a current `_sqt_core` exports. Compared against what
#: actually loaded, so an extension built before the newest kernels reports
#: itself as stale rather than being discovered as an unexplained slowdown.
#: Update this when a kernel is added or removed -- `tests/cpp_bindings`
#: asserts the two agree, so it cannot drift quietly.
#:
#: 42 until the six `_zerocopy` bindings came out; they had no caller in
#: `src/` and wiring them was measured at a mean saving of zero.
_EXPECTED_NATIVE_EXPORTS = 36


def _importable(module: str) -> bool:
    """Whether an optional dependency is present, without importing it.

    `find_spec` rather than `import`: importing torch or blpapi to answer a
    capability question costs seconds and, for a vendor SDK, can try to open
    a session.
    """
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _native_available() -> bool:
    """Whether the compiled fast path is present. Every kernel has a Python
    fallback, so this changes speed and nothing else — but an agent asking
    why a run is slow deserves to be able to find out."""
    try:
        from .features import transforms

        return bool(transforms.HAS_CPP)
    except Exception:  # noqa: BLE001 - absence is the answer, not an error
        return False


def _native_detail() -> Dict[str, Any]:
    """What the loaded extension actually exports.

    A STALE BUILD IS INDISTINGUISHABLE FROM AN ABSENT ONE without this. Every
    kernel is guarded by `hasattr(_cpp_core, "<name>")` at import, which is
    the right design -- a missing symbol falls back silently instead of
    raising -- but it means an extension built before the last two kernels
    were added loads cleanly, reports available, and runs the Python path for
    whatever it does not carry. Observed here: a .pyd built for 3.11 exported
    40 of the 42 symbols and cost about 18x on the two it lacked, with
    nothing saying so.

    So the count and the file come back too. An agent asking why a run is
    slow can compare `exports` against `expected_exports` instead of
    inferring it from a stopwatch.
    """
    detail: Dict[str, Any] = {
        "available": _native_available(),
        "exports": 0,
        "expected_exports": _EXPECTED_NATIVE_EXPORTS,
        "stale": False,
        "path": None,
    }
    try:
        from standard_quant_tools import _sqt_core  # type: ignore[attr-defined]
    except ImportError:
        return detail

    names = [n for n in dir(_sqt_core) if not n.startswith("_")]
    detail["exports"] = len(names)
    detail["path"] = getattr(_sqt_core, "__file__", None)
    detail["stale"] = len(names) < _EXPECTED_NATIVE_EXPORTS
    if detail["stale"]:
        detail["note"] = (
            f"the loaded extension exports {len(names)} symbols where "
            f"{_EXPECTED_NATIVE_EXPORTS} are expected, so some kernels are "
            "falling back to Python silently. Rebuild it."
        )
    return detail
