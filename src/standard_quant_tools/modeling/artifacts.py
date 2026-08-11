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
DataFrames, so they get their own small atomic-write helpers here,
mirroring backtest.artifacts' exact identifier-validation and
resolved-within-root pattern rather than reaching into that module's
underscore-prefixed internals across a package boundary.
"""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from standard_quant_tools.backtest.artifacts import load_artifact, save_artifact
from standard_quant_tools.error import ValidationError

__all__ = [
    "hash_file",
    "load_artifact",
    "load_joblib",
    "load_json",
    "run_dir",
    "save_artifact",
    "save_joblib",
    "save_json",
    "verify_file",
]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _runs_dir() -> Path:
    return Path(
        os.environ.get(
            "SQT_RUNS_DIR",
            str(Path.home() / ".cache" / "standard_quant_tools" / "runs"),
        )
    )


def _validate_identifier(value: str, field_name: str) -> None:
    if not value or not _IDENTIFIER_RE.match(value):
        raise ValidationError(
            f"{field_name}={value!r} is not a valid identifier — only letters, digits, "
            "'_', and '-' are allowed (no path separators, '..', or empty string)."
        )


def run_dir(artifact_id: str) -> Path:
    """
    SQT_RUNS_DIR/<artifact_id> — resolved and confirmed inside the runs
    root before any caller writes to or reads from it. One flat directory
    per id (a `ds_...` dataset id or `mdl_...` model id — the two never
    collide), matching backtest.artifacts' own run_id convention:
    multiple named files (manifest.json, model.joblib, panel.parquet,
    dataset_spec.json, ...) live side by side under the same directory.
    """
    _validate_identifier(artifact_id, "artifact_id")
    root = _runs_dir().resolve()
    path = (root / artifact_id).resolve()
    if not path.is_relative_to(root):
        raise ValidationError(f"resolved path {path} escapes SQT_RUNS_DIR ({root})")
    return path


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp_path.write_bytes(data)
    os.replace(tmp_path, path)


def save_json(directory: Path, name: str, payload: Dict[str, Any]) -> str:
    _validate_identifier(name, "name")
    path = directory / f"{name}.json"
    _atomic_write_bytes(
        path, json.dumps(payload, indent=2, default=str).encode("utf-8")
    )
    return str(path)


def load_json(path: str) -> Dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        raise ValidationError(f"artifact not found: {path}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def save_joblib(directory: Path, name: str, obj: Any) -> str:
    _validate_identifier(name, "name")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.joblib"
    tmp_path = directory / f".{name}.{uuid.uuid4().hex}.tmp"
    joblib.dump(obj, tmp_path)
    os.replace(tmp_path, path)
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
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


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
