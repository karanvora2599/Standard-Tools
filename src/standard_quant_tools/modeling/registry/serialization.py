"""
A skops bundle beside the joblib: the estimator, loadable without pickle.

WHY A SECOND FORMAT. `model.joblib` is verified against the manifest
before it is deserialized, and the manifest can be signed, so a swapped
binary is refused before it can run. That is a guard around pickle, not
a replacement for it: joblib IS pickle, and pickle executes code from
the file by design. `skops` serializes an sklearn estimator as its
declared state and, at load, constructs only the types the loader was
told to trust -- an unknown type is refused before anything is built.
A model that carries both formats can be loaded either way; the choice
is `load_model(format=...)` or `SQT_MODEL_FORMAT`.

WHAT IS TRUSTED. skops trusts sklearn's own types by default. This
package's estimators -- the numpy Cox model, the quantile wrappers --
are trusted by their module prefix, because they are this code. Anything
else in a bundle is named and refused: the bundle came from somewhere
this registry did not write.

WHEN THERE IS NO BUNDLE. skops cannot serialize everything (a booster
holding a native handle, say). Registration then keeps joblib alone and
the manifest's `formats` says so; asking for skops on such a model is
refused by name rather than answered with joblib.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from standard_quant_tools.error import ValidationError

from .. import artifacts as _artifacts

logger = logging.getLogger(__name__)

TRUSTED_PREFIX = "standard_quant_tools."
FORMAT_ENV = "SQT_MODEL_FORMAT"
FORMATS = ("joblib", "skops")


def skops_available() -> bool:
    try:
        import skops.io  # noqa: F401
    except ImportError:
        return False
    return True


def _require() -> None:
    if not skops_available():
        raise ValidationError(
            "a skops bundle needs the `skops` package "
            "(`pip install standard_quant_tools[skops]`)."
        )


def default_format() -> str:
    """The format `load_model` uses when not told: `SQT_MODEL_FORMAT`, else joblib."""
    value = os.environ.get(FORMAT_ENV, "joblib").strip().lower()
    if value not in FORMATS:
        raise ValidationError(
            f"{FORMAT_ENV}={value!r} is not a model format; the formats are {list(FORMATS)}."
        )
    return value


def dump_estimator(directory: Path, name: str, estimator: Any) -> Optional[str]:
    """
    Write `<name>.skops` under `directory`, atomically; None when skops is
    not installed or cannot serialize this estimator, which is logged and
    recorded on the manifest rather than raised -- the joblib is the
    format every registration has, the bundle is the one it has when it
    can.
    """
    if not skops_available():
        return None
    import skops.io as sio

    buffer = io.BytesIO()
    try:
        sio.dump(estimator, buffer)
    except Exception as exc:  # noqa: BLE001 - any failure means "no bundle"
        logger.warning(
            "[modeling] skops could not serialize %s (%s); the model is "
            "registered with joblib only.",
            type(estimator).__name__,
            exc,
        )
        return None
    path = Path(directory) / f"{name}.skops"
    _artifacts._atomic_write_bytes(path, buffer.getvalue())
    return str(path)


def untrusted_types(path: str) -> List[str]:
    """The types in a bundle beyond skops' defaults, as skops names them."""
    _require()
    import skops.io as sio

    return [str(t) for t in sio.get_untrusted_types(file=str(path))]


def load_estimator(path: str) -> Any:
    """
    Load a bundle, trusting skops' defaults and this package's own types.

    Any other type is refused BY NAME before anything is constructed. That
    is the property the format is for: a bundle that names a type this
    registry never writes did not come from this registry.
    """
    _require()
    import skops.io as sio

    resolved = Path(path)
    if not resolved.exists():
        raise ValidationError(f"artifact not found: {path}")
    beyond_default = untrusted_types(str(resolved))
    foreign = [t for t in beyond_default if not t.startswith(TRUSTED_PREFIX)]
    if foreign:
        raise ValidationError(
            f"{resolved.name} names type(s) outside this package and skops' "
            f"defaults: {foreign[:5]}. A skops bundle is loaded without "
            "executing pickle precisely so an unknown type cannot run code; "
            "this bundle is refused, not loaded."
        )
    return sio.load(str(resolved), trusted=beyond_default)


__all__ = [
    "FORMATS",
    "FORMAT_ENV",
    "TRUSTED_PREFIX",
    "default_format",
    "dump_estimator",
    "load_estimator",
    "skops_available",
    "untrusted_types",
]
