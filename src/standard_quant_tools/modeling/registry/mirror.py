"""
The configured mirror: where a registration is pushed as it happens.

`SQT_MODEL_MIRROR_URL` names an artifact store -- a directory, or a
bucket through fsspec -- that receives every package this process
registers and every promotion it records, immediately after the local
write. Local stays the root the runtime reads from; the mirror is where
another machine finds the package, through `pull_model_package`. A
mirror that cannot be written is an error, not a warning: a registration
that claims to be mirrored and is not is the worse outcome.

Kept apart from `package.py` so the lifecycle log, which `package.py`
reads, can push itself here without a cycle.
"""

from __future__ import annotations

import os
from typing import Optional

from standard_quant_tools.artifact_store import (
    ArtifactStore,
    LocalArtifactStore,
    store_from_url,
)

MIRROR_URL_ENV = "SQT_MODEL_MIRROR_URL"


def mirror_configured() -> bool:
    return bool(os.environ.get(MIRROR_URL_ENV))


def configured_mirror() -> Optional[ArtifactStore]:
    """The store `SQT_MODEL_MIRROR_URL` names, or None."""
    url = os.environ.get(MIRROR_URL_ENV)
    return store_from_url(url) if url else None


def mirror_file(model_id: str, filename: str) -> Optional[str]:
    """Push one local file of a package to the configured mirror; the
    URI it landed at, or None when no mirror is configured."""
    store = configured_mirror()
    if store is None:
        return None
    key = f"{model_id}/{filename}"
    return store.put(key, LocalArtifactStore().get(key))


__all__ = ["MIRROR_URL_ENV", "configured_mirror", "mirror_configured", "mirror_file"]
