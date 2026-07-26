"""Pluggable storage backend for `AuditWriter`. `LocalFilesystemBackend` is
the only implementation shipped this round -- the interface exists so a
future WORM backend (S3 Object Lock, Azure Immutable Blob) can be dropped in
without touching `AuditWriter`'s chain-hashing/locking orchestration logic.
Building that backend is a deliberately separate, later piece of work; see
Documentation/10_auditability.md for what this seam does and does not cover
today (in particular: only `AuditWriter`'s own read/append/lock/day-listing
operations are backend-routed -- `verify`/`retention`/`export` still read
the local filesystem directly)."""

import os
from pathlib import Path
from typing import Any, List, Protocol

from .paths import _acquire_lock, _iter_day_files, _release_lock


class AuditStorageBackend(Protocol):
    """
    The storage primitives `AuditWriter` needs: acquire/release an
    exclusive lock keyed by a path, read a file's lines, durably append one
    line, check existence, and list which calendar days have a file.
    `AuditWriter` -- not the backend -- owns the "lock, read current state,
    append, unlock" sequencing that keeps the hash chain race-free under
    concurrent writers; a backend only needs to implement each primitive
    correctly for its own storage medium. A backend with its own native
    atomic-append semantics (e.g. conditional PUT / object versioning) can
    make `acquire_lock`/`release_lock` a no-op pair, since the ordering
    guarantee `AuditWriter` relies on would already hold without them.
    """

    def acquire_lock(self, path: Path) -> Any: ...

    def release_lock(self, handle: Any) -> None: ...

    def read_lines(self, path: Path) -> List[str]:
        """Every line in `path` (trailing newline included, same as
        `file.readlines()`), or `[]` if it doesn't exist."""
        ...

    def append_line(self, path: Path, line: str) -> None:
        """Durably append one line (no trailing newline expected on input)
        to `path`, creating it and any parent structure if needed."""
        ...

    def exists(self, path: Path) -> bool: ...

    def list_day_stems(self, audit_dir: Path) -> List[str]:
        """Every day-file stem ("YYYY-MM-DD") with data in `audit_dir`,
        sorted chronologically. Used to find the most recent prior day when
        bootstrapping a new day's chain-index entry."""
        ...


class LocalFilesystemBackend:
    """
    The only backend implemented so far: local disk, cross-process
    advisory locking via a sidecar `.lock` file, and an unconditional
    `fsync` after every append. This is exactly what `AuditWriter` did
    directly before this interface existed, moved here as a seam without
    changing behavior. Explicitly **not** WORM: nothing stops a process
    with filesystem access from writing outside this backend entirely —
    see the top-of-page caveat in `Documentation/10_auditability.md`.
    """

    def acquire_lock(self, path: Path) -> Any:
        lock_path = path.with_name(path.name + ".lock")
        return _acquire_lock(lock_path)

    def release_lock(self, handle: Any) -> None:
        _release_lock(handle)

    def read_lines(self, path: Path) -> List[str]:
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as f:
            return f.readlines()

    def append_line(self, path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def exists(self, path: Path) -> bool:
        return path.exists()

    def list_day_stems(self, audit_dir: Path) -> List[str]:
        return [p.stem for p in _iter_day_files(audit_dir)]
