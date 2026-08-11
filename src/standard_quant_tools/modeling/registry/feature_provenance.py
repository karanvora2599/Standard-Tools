"""
Per-feature implementation identity.

`ModelManifest.git_commit_sha` pins the repository, which is enough for
built-in features — but not for a feature registered at runtime through
`register_feature`, whose callable may come from a notebook, a firm's
internal package, or anywhere else outside this repo. Two models can share
a git SHA and still have been trained on different implementations of
`custom.my_alpha`.

Hashing each feature function's own source closes that gap for anything
whose source is introspectable, and records an explicit "unavailable"
marker when it is not (a C extension, a functools.partial over a builtin,
a callable defined in an interactive session that inspect can't recover).
An honest marker is the point: it tells a later reader that this feature's
identity was NOT captured, rather than leaving the field silently absent
and indistinguishable from a feature that simply wasn't used.
"""

import hashlib
import inspect
from typing import Dict, List

from ..features.registry import FEATURE_REGISTRY

_UNAVAILABLE = "unavailable"


def feature_implementation_hash(feature_id: str) -> str:
    """SHA-256 (16 hex chars) of the feature function's source, or
    `"unavailable"` when the source cannot be recovered."""
    definition = FEATURE_REGISTRY.get(feature_id)
    if definition is None:
        return _UNAVAILABLE
    try:
        source = inspect.getsource(definition.fn)
    except (OSError, TypeError):
        # OSError: source file not available (interactive session, C
        # extension). TypeError: not a Python-level callable at all.
        return _UNAVAILABLE
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def feature_implementation_hashes(feature_ids: List[str]) -> Dict[str, str]:
    return {fid: feature_implementation_hash(fid) for fid in feature_ids}
