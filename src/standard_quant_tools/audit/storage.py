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
from typing import Any, List, NamedTuple, Optional, Protocol

from .paths import _acquire_lock, _iter_day_files, _release_lock

#: What a line ends with on disk. The writer has always written in text mode,
#: which spells a newline as the platform's line ending (CRLF on Windows), so
#: an append keeps writing that and no existing file becomes mixed.
_LINE_END = os.linesep.encode("ascii")

#: Bytes read per step when looking backwards for the last line.
_TAIL_BLOCK = 64 * 1024


class LastLine(NamedTuple):
    """The last non-blank line of a file, as the writer needs it.

    `text` is that line with surrounding whitespace removed, or None when
    the file is missing, empty or blank. `terminated` says whether the file
    ends in a newline: a record whose newline was lost is complete but
    unterminated, and so is a record a crash cut off mid-write -- the
    writer tells those apart by whether the text parses. `offset` is the
    byte position the line starts at, which is where a file has to be cut
    to remove a torn fragment; None when the backend cannot say.
    """

    text: Optional[str]
    terminated: bool
    offset: Optional[int]


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

    A backend MAY also provide `read_last_line(path) -> LastLine`. The
    writer needs only the last line of a file before each append, and uses
    that method when it exists; without it the writer reads every line
    through `read_lines`, which is correct but costs time and memory in
    proportion to the day.
    """

    def acquire_lock(self, path: Path) -> Any: ...

    def release_lock(self, handle: Any) -> None: ...

    def read_lines(self, path: Path) -> List[str]:
        """Every line in `path` (trailing newline included, same as
        `file.readlines()`), or `[]` if it doesn't exist."""
        ...

    def append_line(self, path: Path, line: str) -> None:
        """Durably append one line (no trailing newline expected on input)
        to `path`, creating it and any parent structure if needed. When the
        file does not end in a newline, one is written first, so the new
        line never joins the one before it."""
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

    def read_last_line(self, path: Path) -> LastLine:
        """
        The last non-blank line of `path`, found by reading backwards from
        the end.

        The writer needs only this before each append, and it used to read
        the whole day file to get it -- under the lock every other writer
        waits on. A 95 MiB day cost more than half a second and as much
        memory on every tool call. Reading backwards costs the length of
        the last line, whatever the size of the day. See the CHANGELOG
        entry of 2026-09-28.

        CRLF endings and trailing blank lines are skipped. The line is
        decoded strictly: a line that is not UTF-8 raises
        UnicodeDecodeError, which the writer treats like any other
        unreadable tail.
        """
        try:
            handle = open(path, "rb")
        except FileNotFoundError:
            return LastLine(None, True, 0)
        with handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            if end == 0:
                return LastLine(None, True, 0)
            handle.seek(end - 1)
            terminated = handle.read(1) == b"\n"

            # Where the last non-whitespace byte ends.
            content_end = end
            while content_end > 0:
                step = min(_TAIL_BLOCK, content_end)
                handle.seek(content_end - step)
                kept = handle.read(step).rstrip()
                if kept:
                    content_end = content_end - step + len(kept)
                    break
                content_end -= step
            if content_end == 0:
                return LastLine(None, terminated, end)

            # Where the line holding it starts: just after the newline
            # before it, or the start of the file.
            line_start = 0
            position = content_end
            while position > 0:
                step = min(_TAIL_BLOCK, position)
                handle.seek(position - step)
                newline = handle.read(step).rfind(b"\n")
                if newline != -1:
                    line_start = position - step + newline + 1
                    break
                position -= step

            handle.seek(line_start)
            raw = handle.read(content_end - line_start)
        return LastLine(raw.strip().decode("utf-8"), terminated, line_start)

    def append_line(self, path: Path, line: str) -> None:
        """
        Append `line` and the platform line ending, flushed and fsync'd.

        A file whose last byte is not a newline gets one first. Appending
        blindly glued the new record onto a previous record whose newline
        had been lost, producing one line that is neither record -- the
        verifier could not read it and every later write refused to chain
        onto it. The writer refuses before reaching here when the last line
        is a torn fragment rather than a complete record, so this only ever
        separates two complete records.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = line.encode("utf-8") + _LINE_END
        with open(path, "a+b") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size:
                f.seek(size - 1)
                last = f.read(1)
                if last == b"\r":
                    # Half a CRLF: finish it rather than start another.
                    data = b"\n" + data
                elif last != b"\n":
                    data = _LINE_END + data
            # Append mode writes at the end whatever the position.
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    def exists(self, path: Path) -> bool:
        return path.exists()

    def list_day_stems(self, audit_dir: Path) -> List[str]:
        return [p.stem for p in _iter_day_files(audit_dir)]
