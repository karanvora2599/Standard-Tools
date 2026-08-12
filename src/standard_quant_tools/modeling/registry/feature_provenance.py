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

Scope, stated explicitly: the hash covers the FEATURE FUNCTION'S OWN
SOURCE, not its transitive dependencies. `_technical_rsi` calls
`indicators.momentum.rsi`, so rewriting that shared primitive changes the
computed values while leaving this per-feature hash identical. The
manifest's `git_commit_sha` / `package_version` are the coarser signal for
that case. This field's real job is the one those two cannot do: pinning a
feature registered at runtime from outside the repo.
"""

import hashlib
import inspect
from typing import Any, Dict, List

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


def feature_provenance_from_spec(
    feature_entries: "List[Dict[str, Any]] | None",
) -> Dict[str, Dict[str, Any]]:
    """
    Full per-column provenance, resolved from a DatasetSpec's own feature
    entries: output column -> {feature_id, params, implementation_hash}.

    The panel's columns are `FeatureSpec.output_name`, i.e. the ALIAS when
    one is set. Hashing those names directly (which is what passing
    `feature_ids` to feature_implementation_hashes did) is wrong in two
    ways, and the second is worse than the first:

      - an alias is not a registry id, so `mom_20` resolved to
        "unavailable" and the feature's identity was simply lost — the very
        thing this module exists to record;
      - an alias is an arbitrary caller-supplied string, so aliasing
        `market.momentum` to `"technical.rsi"` looked up and recorded RSI's
        implementation hash for a momentum column. Not a missing record: an
        actively wrong one.

    Keeping the output column, the registry id, the resolved parameters and
    the implementation hash together means a later reader can answer "what
    actually produced this column" without having to re-derive any of it,
    and aliasing can't collide with anything because the alias is the KEY,
    never the lookup.
    """
    from ..features.params import resolve_params

    provenance: Dict[str, Dict[str, Any]] = {}
    for entry in feature_entries or []:
        feature_id = entry.get("id")
        if not feature_id:
            continue
        output_name = entry.get("alias") or feature_id
        definition = FEATURE_REGISTRY.get(feature_id)
        if definition is None:
            # An unregistered id is recorded rather than skipped: the
            # column exists in the panel either way, and a silently absent
            # entry is indistinguishable from a feature that wasn't used.
            provenance[output_name] = {
                "feature_id": feature_id,
                "params": dict(entry.get("params") or {}),
                "implementation_hash": _UNAVAILABLE,
            }
            continue
        try:
            resolved = resolve_params(definition, dict(entry.get("params") or {}))
        except Exception:
            # Provenance must never be the thing that fails a save; the
            # params as requested are still more informative than nothing.
            resolved = dict(entry.get("params") or {})
        provenance[output_name] = {
            "feature_id": feature_id,
            "params": resolved,
            "implementation_hash": feature_implementation_hash(feature_id),
        }
    return provenance
