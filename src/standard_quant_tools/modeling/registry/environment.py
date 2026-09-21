"""
The numerical environment a model was produced in.

The manifest already answers "what source commit created me" through
`git_commit_sha` and `package_version`. It did not answer "what actually
computed me": which numpy, which scikit-learn, which BLAS the matrix work
went through, whether the native extension was loaded and whether it was
the current build, and how many threads the kernels were allowed. Two
machines at the same commit can differ on every one of those, and the
differences show up as coefficients that agree to four digits rather than
twelve, or as a run that is eighteen times slower with nothing saying why.

Everything here is read from the process, never declared. The result is
plain JSON so it travels in `manifest.json` beside the other lineage
fields, and it carries nothing that identifies the machine -- no hostname,
no user, no filesystem path -- because a fingerprint of the numerics is not
a fingerprint of the host.
"""

from __future__ import annotations

import os
import platform
from importlib import metadata
from typing import Any, Dict, Optional

#: The distributions whose versions decide the numbers. `lightgbm`,
#: `xgboost` and `numba` are optional and come back None when absent,
#: which is itself the fact worth recording.
PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "joblib",
    "numba",
    "lightgbm",
    "xgboost",
)

#: Thread caps read by numpy's BLAS, by numba and by this package's own
#: kernels. Unset is recorded as None rather than skipped: "no cap was set"
#: is a different environment from "capped at one".
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "SQT_NUM_THREADS",
)


def _version(distribution: str) -> Optional[str]:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _blas() -> Dict[str, Optional[str]]:
    """The BLAS and LAPACK numpy was built against, from numpy's own
    build record. None when numpy cannot say, which older versions cannot."""
    try:
        import numpy

        config = numpy.show_config(mode="dicts")
    except Exception:  # noqa: BLE001 - absence is the answer, not an error
        return {"blas": None, "lapack": None}
    if not isinstance(config, dict):
        return {"blas": None, "lapack": None}
    dependencies = config.get("Build Dependencies") or {}
    blas = dependencies.get("blas") or {}
    lapack = dependencies.get("lapack") or {}
    return {
        "blas": blas.get("name") if isinstance(blas, dict) else None,
        "lapack": lapack.get("name") if isinstance(lapack, dict) else None,
    }


def environment_fingerprint() -> Dict[str, Any]:
    """
    What computed this model, as JSON.

    Read at registration and stored in `ModelManifest.environment`. The
    native-extension block reuses `capabilities._native_detail`, which is
    the same check `list_modeling_capabilities` reports, so the manifest
    and the capability report cannot disagree about whether the fast path
    was present or stale.
    """
    from ..capabilities import _native_detail

    native = _native_detail()
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "packages": {name: _version(name) for name in PACKAGES},
        "blas": _blas(),
        "native_extension": {
            "available": bool(native.get("available")),
            "exports": int(native.get("exports") or 0),
            "expected_exports": int(native.get("expected_exports") or 0),
            "stale": bool(native.get("stale")),
        },
        "threads": {
            **{name: os.environ.get(name) for name in THREAD_VARIABLES},
            "cpu_count": os.cpu_count(),
        },
    }


def _flatten(mapping: Any, prefix: str = "") -> Dict[str, Any]:
    """A nested fingerprint as one level of dotted keys, so two of them
    can be compared key by key instead of block by block."""
    flat: Dict[str, Any] = {}
    if not isinstance(mapping, dict):
        return flat
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def environment_differences(
    trained: Dict[str, Any], current: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """
    What moved between the environment a model was fitted in and the one
    asking now, as `{"packages.numpy": {"trained": ..., "current": ...}}`.

    Both sides are flattened to dotted keys first, because the interesting
    unit is a single version or a single thread cap, not the block it sits
    in: reporting that `packages` differs says nothing an agent can act on,
    while reporting that `packages.scikit-learn` moved from 1.5.2 to 1.6.0
    names the thing to reinstall.

    A key on one side only is a difference with None on the other, and
    stays one even when the other side recorded None explicitly -- "this
    manifest predates the field" and "this package is not installed" are
    different facts, and collapsing them would hide the first.

    Empty on both sides, or equal on every key, is an empty mapping: the
    caller reads that as "the numerics are the ones that fitted it".
    """
    left = _flatten(trained)
    right = _flatten(current)
    differences: Dict[str, Dict[str, Any]] = {}
    for key in sorted(set(left) | set(right)):
        in_left, in_right = key in left, key in right
        if in_left and in_right and left[key] == right[key]:
            continue
        differences[key] = {"trained": left.get(key), "current": right.get(key)}
    return differences


__all__ = [
    "PACKAGES",
    "THREAD_VARIABLES",
    "environment_differences",
    "environment_fingerprint",
]
