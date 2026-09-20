"""
Modeling-specific artifact I/O.

Parquet artifacts (dataset panels, prediction frames) reuse
`backtest.artifacts.save_artifact`/`load_artifact` directly — same
SQT_RUNS_DIR root, same atomic-write-then-os.replace + identifier
validation, no reimplementation (dataset/model ids are namespaced with
`ds_`/`mdl_` prefixes so they share the flat SQT_RUNS_DIR/<id>/ layout
backtest runs already use, rather than requiring a nested subdirectory
`save_artifact`'s identifier validation — deliberately, no path
separators allowed — can't express).

Model registry artifacts (manifest.json, model.joblib) aren't
DataFrames, so they get their own small atomic-write helpers here.

They used to MIRROR backtest.artifacts' identifier-validation and
resolved-within-root pattern rather than reach into that module's
underscore-prefixed internals across a package boundary. That was the
right instinct and the wrong remedy: it left two independent copies of a
path-traversal guard, equal only until one of them was hardened. Both now
come from `standard_quant_tools._runspath`, which is neither package's
internals — the same arrangement `_jsonsafe` already uses for the same
reason.

Bytes reach disk through `standard_quant_tools.artifact_store`: one
atomic write and one streaming hash, shared with `backtest.artifacts`,
and a `LocalArtifactStore` the registry's package operations address
by key. The path-based helpers here keep their signatures -- they are
the local filesystem's implementation -- and delegate.
"""

import io
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from standard_quant_tools._jsonsafe import sanitize_for_json
from standard_quant_tools._runspath import (
    resolve_within_runs_dir as _resolved_within_runs_dir,
)
from standard_quant_tools._runspath import runs_dir as _runs_dir
from standard_quant_tools._runspath import validate_identifier as _validate_identifier
from standard_quant_tools.artifact_store import (
    LocalArtifactStore,
    hash_stream,
    write_bytes_atomically,
)
from standard_quant_tools.backtest.artifacts import load_artifact, save_artifact
from standard_quant_tools.error import ValidationError

__all__ = [
    "hash_file",
    "load_artifact",
    "load_joblib",
    "load_json",
    "local_store",
    "run_dir",
    "save_artifact",
    "save_joblib",
    "save_json",
    "verify_file",
]


def run_dir(artifact_id: str) -> Path:
    """
    SQT_RUNS_DIR/<artifact_id> — resolved and confirmed inside the runs
    root before any caller writes to or reads from it. One flat directory
    per id (a `ds_...` dataset id or `mdl_...` model id — the two never
    collide), matching backtest.artifacts' own run_id convention:
    multiple named files (manifest.json, model.joblib, panel.parquet,
    dataset_spec.json, ...) live side by side under the same directory.

    The validator and the containment check both come from `_runspath` now.
    This module had its own copy of the first and open-coded the second, so
    the two artifact stores guarded the same attack with the same code
    written twice -- equal only until one of them was improved.
    """
    _validate_identifier(artifact_id, "artifact_id")
    return _resolved_within_runs_dir(_runs_dir() / artifact_id)


def local_store() -> LocalArtifactStore:
    """The runs directory as an `ArtifactStore`, root resolved per call."""
    return LocalArtifactStore()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    write_bytes_atomically(path, data)


def save_json(directory: Path, name: str, payload: Dict[str, Any]) -> str:
    """
    Persist a JSON artifact under the same JSON-safety contract the agent
    boundary already enforces.

    Python's json.dumps defaults to allow_nan=True and writes bare `NaN` /
    `Infinity` tokens, which are not valid JSON per RFC 8259 and are
    rejected by strict parsers (including some LLM API backends). The
    runtime legitimately produces NaN -- AUC on a single-class fold, ICIR
    with no dispersion, unsupported feature importance -- so manifests were
    being written that this package could read back but many other tools
    could not.

    sanitize_for_json is the same helper modeling_dispatch uses, so a
    manifest on disk and the tool response describing it now agree on how a
    non-finite value is represented (null). allow_nan=False then makes any
    future path that sneaks one through fail loudly at the write rather
    than producing a subtly unparseable file.
    """
    _validate_identifier(name, "name")
    path = directory / f"{name}.json"
    _atomic_write_bytes(
        path,
        json.dumps(
            sanitize_for_json(payload), indent=2, default=str, allow_nan=False
        ).encode("utf-8"),
    )
    return str(path)


def load_json(path: str) -> Dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise ValidationError(f"artifact not found: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def save_joblib(directory: Path, name: str, obj: Any) -> str:
    _validate_identifier(name, "name")
    path = directory / f"{name}.joblib"
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    write_bytes_atomically(path, buffer.getvalue())
    return str(path)


def load_joblib(path: str) -> Any:
    resolved = Path(path)
    if not resolved.exists():
        raise ValidationError(f"artifact not found: {path}")
    return joblib.load(resolved)


# ── Content addressing ──────────────────────────────────────────────────
#
# Each file in a model/dataset directory is written atomically, but that
# only makes each file individually consistent -- it says nothing about
# whether the SET of files still matches what was registered. Every file
# below the manifest is plain JSON or a joblib blob on local disk, so
# anything with write access can edit a feature's period in
# dataset_spec.json, shift a mean in preprocessing_stats.json, or swap
# model.joblib, and every later score_model call would silently use the
# altered version while still reporting the original model_id.
#
# Hashing each artifact and recording the digests in the manifest turns
# the directory from "a collection of atomic files" into a verifiable
# package. The manifest is the root of trust: it is not self-hashing
# (it cannot contain its own digest), so a determined local attacker who
# can edit BOTH an artifact and the manifest is still out of scope --
# closing that requires signing the manifest, the same way
# audit/signing.py does for decision records.


def hash_file(path: Path) -> str:
    """SHA-256 of a file's raw bytes, truncated to 16 hex chars — the same
    digest length audit/hashing.py uses, so provenance identifiers look
    consistent across the two subsystems."""
    resolved = Path(path)
    if not resolved.exists():
        raise ValidationError(f"artifact not found: {resolved}")
    with open(resolved, "rb") as handle:
        return hash_stream(handle)


def verify_file(path: Path, expected: Optional[str], label: str) -> None:
    """
    Raise if `path`'s content hash no longer matches what was recorded.

    `expected=None` means the artifact predates content hashing (a model
    registered by an older version) — verification is skipped rather than
    failing every previously-registered model, which would make an upgrade
    look like mass corruption.
    """
    if expected is None:
        return
    actual = hash_file(path)
    if actual != expected:
        raise ValidationError(
            f"{label} has changed since it was registered "
            f"(expected content hash {expected}, found {actual}): {path}. "
            "A registered model's artifacts are immutable — re-run the "
            "experiment to register a new model rather than editing an "
            "existing one in place."
        )
