"""
The runs directory, and the two checks that keep callers inside it.

`backtest.artifacts` and `modeling.artifacts` both build filesystem paths
from identifiers an LLM can choose -- a `run_id` on a compact-backtest
request, a `ds_...` or `mdl_...` artifact id -- and both had their own copy
of the guard against that. Same regex, same runs-directory lookup, same
error message.

TWO COPIES OF A PATH-TRAVERSAL CHECK IS THE WRONG NUMBER, and not because
of the duplication itself. It is because the two copies are only equal
today. Harden one -- a new escape to reject, a case the regex lets through,
a symlink to resolve differently -- and the other keeps the old behaviour
under the same name, which reads as fixed everywhere and is fixed in one
place. A half-applied security fix is worse than an unapplied one: it
removes the reason to look again.

The asymmetry was already there. `backtest.artifacts` carried the docstring
explaining what the check defends against and layered
`_resolved_within_runs_dir` on top of it as defence in depth;
`modeling.artifacts` had the same validator with no explanation, and
open-coded the containment check inside `run_dir` instead of sharing it.

ONE NAME, ONE FILE. A reference promises that resolving it twice gives the
same value, which needs every spelling that reaches a file to be the only
spelling that does. Windows and macOS fold case, and Windows also drops a
trailing dot or space and reserves device names (`NUL`, `CON`, `COM1`) in
every directory, so `run8/Report` and `run8/report` were one file there
while the store treated them as two keys. Identifiers that name a device
are refused on every platform -- a package made on Linux has to be
pullable on Windows -- and a name that reaches an existing file spelled
differently is refused rather than read or overwritten.

WHAT THE RUNS DIRECTORY KEEPS. Everything, by design: a published value is
never collected, because deleting it would break the promise every holder
of its reference relies on, and an audit record may name it. `sweep` is the
one exception, and it removes only what no reference can name -- the temp
file an interrupted atomic write leaves, and a model or dataset directory
whose registration never reached its commit file.

Lives at the top level for the reason `_jsonsafe` does: both need it and
neither should import the other.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from standard_quant_tools._containment import require_within
from standard_quant_tools._env import env_path
from standard_quant_tools.error import ValidationError

#: The environment variable that relocates the runs root.
RUNS_DIR_ENV = "SQT_RUNS_DIR"

#: A plain slug. Deliberately a whitelist rather than a blacklist of
#: dangerous sequences: '..', '/', '\', ':', and a NUL byte are all excluded
#: by not being letters, digits, '_' or '-', and so is whatever the next
#: platform-specific escape turns out to be.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_-]+$")

#: Names Windows resolves to a device in every directory, compared
#: case-folded against the part of a name before its first dot. `NUL/x`
#: cannot be created and `run9/NUL` opens the null device; refused on every
#: platform so a package made anywhere can be read everywhere.
DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"{port}{digit}" for port in ("com", "lpt") for digit in "0123456789¹²³"}
)

_EXTENDED_LENGTH_PREFIX = "\\\\?\\"


def _default_runs_dir() -> Path:
    return Path.home() / ".cache" / "standard_quant_tools" / "runs"


def runs_dir() -> Path:
    """
    Where artifacts live: `$SQT_RUNS_DIR`, or a cache dir under $HOME.

    Read through `_env.env_path`, so an empty or blank value is the
    default rather than the working directory, a relative path is refused
    rather than read against whichever directory the process happens to be
    in, and a value naming an existing file is refused by name instead of
    surfacing as an OS error from the first write.
    """
    return env_path(RUNS_DIR_ENV, _default_runs_dir)


def is_device_name(name: str) -> bool:
    """Whether Windows would open a device for `name`, extension or not."""
    return name.split(".", 1)[0].casefold() in DEVICE_NAMES


def validate_identifier(value: str, field_name: str) -> None:
    """
    Refuse anything that is not a plain slug.

    run_id/name are LLM-reachable (e.g. `BacktestCompactInput.run_id`) and
    get joined directly into a filesystem path -- so path separators, '..',
    null bytes, and a drive-letter or absolute prefix are rejected here,
    before they can reach the path at all. A Windows device name is refused
    too: it is a slug, but not a file.
    """
    if not value or not _IDENTIFIER_RE.match(value):
        raise ValidationError(
            f"{field_name}={value!r} is not a valid identifier — only letters, "
            "digits, '_', and '-' are allowed (no path separators, '..', or "
            "empty string)."
        )
    if is_device_name(value):
        raise ValidationError(
            f"{field_name}={value!r} is a Windows device name, which no "
            "directory can hold a file or folder under. It is refused on every "
            "platform so what is stored here can be copied anywhere; choose "
            "another name."
        )


def _comparable(path: Path) -> Path:
    return Path(str(path).removeprefix(_EXTENDED_LENGTH_PREFIX))


def _spelled_differently(requested: str, on_disk: str) -> bool:
    """Two spellings the filesystem treats as one name: equal once case is
    folded and a trailing dot or space dropped, and not equal as written."""
    return requested != on_disk and (
        requested.rstrip(". ").casefold() == on_disk.rstrip(". ").casefold()
    )


def _collision(requested: str, on_disk: str, where: Path) -> ValidationError:
    return ValidationError(
        f"{requested!r} names {on_disk!r}, which already exists in {where} "
        "spelled differently. This filesystem treats the two spellings as one "
        "name, so using it would read or replace a value published under the "
        "other. Use the existing spelling, or a name that differs in more "
        "than case."
    )


def require_spelled_as_on_disk(lexical: Path, resolved: Path, root: Path) -> None:
    """
    Refuse a path below `root` that reaches an existing entry under a
    different spelling.

    `resolved` is `lexical` after `Path.resolve()`. Windows resolves an
    existing component to its on-disk spelling, which is compared with the
    requested one component by component; a difference beyond case and a
    trailing dot or space is a link, not a spelling, and is left to the
    containment check. Elsewhere resolve does not canonicalise case (macOS
    folds it anyway), so an existing FILE is also looked up by exact name
    in its directory -- a run directory holds a handful of files, whereas
    listing the runs root on every lookup of a run directory would cost a
    read of every run ever made.
    """
    try:
        below = _comparable(resolved).relative_to(_comparable(root)).parts
    except ValueError:
        return
    if not below:
        return
    requested = Path(lexical).parts[-len(below) :]
    if len(requested) == len(below):
        for index, (mine, theirs) in enumerate(zip(requested, below)):
            if _spelled_differently(mine, theirs):
                raise _collision(
                    mine, theirs, _comparable(root).joinpath(*below[:index])
                )
    if sys.platform == "win32":
        return
    lexical = Path(lexical)
    try:
        if not lexical.is_file():
            return
        names = os.listdir(lexical.parent)
    except OSError:
        return
    if lexical.name in names:
        return
    for name in names:
        if _spelled_differently(lexical.name, name):
            raise _collision(lexical.name, name, lexical.parent)


def resolve_within_runs_dir(path: Path) -> Path:
    """
    Defence in depth on top of `validate_identifier`: confirm the final
    RESOLVED path is inside the runs root before any read or write.

    The validator works on the identifier and this works on the result, so
    a way of building a path that never passes through a validated
    identifier -- a symlink inside the runs directory, a caller assembling
    a path itself -- still has to land inside the root. A path that reaches
    an existing file under another spelling is refused here too.
    """
    root = runs_dir().resolve()
    resolved = Path(path).resolve()
    require_within(
        resolved, root, f"resolved path {resolved} escapes {RUNS_DIR_ENV} ({root})"
    )
    require_spelled_as_on_disk(Path(os.path.abspath(path)), resolved, root)
    return resolved


# ── the sweep ──────────────────────────────────────────────────────────────

#: A model or dataset directory the registry names itself, and the file
#: its registration writes LAST. A directory without that file is one
#: whose registration was interrupted: nothing lists it and nothing can
#: load it.
_COMMIT_FILES = (
    (re.compile(r"^mdl_[0-9a-f]{12}$"), "manifest.json"),
    (re.compile(r"^ds_[0-9a-f]{12}$"), "dataset_meta.json"),
)

#: What an interrupted atomic write leaves: `.<name>.<hex>.tmp`.
_TEMP_FILE_RE = re.compile(r"^\..+\.[0-9a-f]{32}\.tmp$")

#: A published reference's sidecar. A directory holding one is never
#: collected, whatever else it lacks.
_SIDECAR_SUFFIX = "._handoff.json"

#: How old, in hours, a leftover has to be before the sweep will remove it.
#: A day, so a registration or a long conversion in progress is never
#: mistaken for an abandoned one.
DEFAULT_SWEEP_HOURS = 24.0


@dataclass
class SweepReport:
    """What `sweep` found, and whether it removed it."""

    root: Path
    older_than_hours: float
    removed: bool
    #: Temp files left by interrupted writes.
    temp_files: List[Path] = field(default_factory=list)
    #: Model and dataset directories whose registration never committed.
    partial_directories: List[Path] = field(default_factory=list)

    @property
    def candidates(self) -> List[Path]:
        return [*self.temp_files, *self.partial_directories]


def _newest_mtime(path: Path) -> float:
    """The latest modification time of `path` or anything directly in it,
    so a directory still being written counts as fresh."""
    newest = path.stat().st_mtime
    if path.is_dir():
        for entry in path.iterdir():
            try:
                newest = max(newest, entry.stat().st_mtime)
            except OSError:
                continue
    return newest


def _is_junction(path: Path) -> bool:
    check = getattr(path, "is_junction", None)  # Python 3.12+
    return bool(check()) if callable(check) else False


def _commit_file(name: str) -> Optional[str]:
    for pattern, commit in _COMMIT_FILES:
        if pattern.match(name):
            return commit
    return None


def sweep(
    *,
    older_than_hours: float = DEFAULT_SWEEP_HOURS,
    dry_run: bool = True,
    root: Optional[Path] = None,
    now: Optional[float] = None,
) -> SweepReport:
    """
    List -- or, with `dry_run=False`, remove -- what an interrupted write
    left in the runs directory, older than `older_than_hours`.

    Two things only. A dot-prefixed temp file an atomic write never
    renamed into place, in the runs directory or one level down. And a
    `mdl_<id>` or `ds_<id>` directory whose commit file (`manifest.json`,
    `dataset_meta.json`) was never written, holding no published
    reference's sidecar. Nothing a reference or a registered model can
    name is ever a candidate, which is why a time- or size-based
    retention of published values is not offered here: that decision
    breaks references, and belongs to whoever owns them.
    """
    if older_than_hours < 0:
        raise ValidationError(
            f"older_than_hours must be at least 0; got {older_than_hours}."
        )
    base = Path(root) if root is not None else runs_dir()
    report = SweepReport(
        root=base, older_than_hours=float(older_than_hours), removed=not dry_run
    )
    if not base.is_dir():
        return report
    cutoff = (time.time() if now is None else now) - older_than_hours * 3600.0

    def stale(path: Path) -> bool:
        try:
            return _newest_mtime(path) <= cutoff
        except OSError:
            return False

    for entry in sorted(base.iterdir()):
        if entry.is_file() and _TEMP_FILE_RE.match(entry.name) and stale(entry):
            report.temp_files.append(entry)
            continue
        # Never into a link or a junction: what it points at is not the
        # runs directory's to remove.
        if not entry.is_dir() or entry.is_symlink() or _is_junction(entry):
            continue
        children = sorted(entry.iterdir())
        for child in children:
            if child.is_file() and _TEMP_FILE_RE.match(child.name) and stale(child):
                report.temp_files.append(child)
        commit = _commit_file(entry.name)
        if commit is None or (entry / commit).exists():
            continue
        if any(child.name.endswith(_SIDECAR_SUFFIX) for child in children):
            continue
        if stale(entry):
            report.partial_directories.append(entry)

    if not dry_run:
        for path in report.temp_files:
            path.unlink(missing_ok=True)
        for path in report.partial_directories:
            shutil.rmtree(path, ignore_errors=True)
    return report


__all__ = [
    "DEFAULT_SWEEP_HOURS",
    "DEVICE_NAMES",
    "RUNS_DIR_ENV",
    "SweepReport",
    "is_device_name",
    "require_spelled_as_on_disk",
    "resolve_within_runs_dir",
    "runs_dir",
    "sweep",
    "validate_identifier",
]
