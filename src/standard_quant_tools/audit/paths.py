"""Where the audit trail lives on disk, how its files are named/discovered,
and the cross-process advisory locking primitive every writer in this
package uses before touching a day file or the chain index."""

import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import List

from standard_quant_tools._env import env_flag, env_path

logger = logging.getLogger(__name__)


def _audit_enabled() -> bool:
    """
    Whether `dispatch()` writes a decision record: SQT_AUDIT_ENABLED, on
    unless it reads as off.

    Read through the library's one flag reader. This used to count the
    EMPTY string as off, so `SQT_AUDIT_ENABLED=` in a launcher -- the
    usual way of writing "leave the default" -- silently stopped the
    decision log, while a padded " 0 " or "off" left it running against a
    plain intent to stop it. Blank is now the default (on), any case and
    padding of 1/true/yes/on or 0/false/no/off is honoured, and any other
    word is refused by name rather than read as either answer. See the
    CHANGELOG entry of 2026-09-28.
    """
    return env_flag("SQT_AUDIT_ENABLED", True)


class AuditLocationWarning(UserWarning):
    """The audit trail lives somewhere designed to be emptied."""


def _legacy_cache_audit_dir() -> Path:
    """Where the audit trail defaulted to before it moved out of the cache."""
    return Path.home() / ".cache" / "standard_quant_tools" / "audit"


def _is_legacy_cache_location(directory: Path) -> bool:
    """Whether `directory` is that old cache location, however it was
    reached -- by the default keeping an existing trail's home, or by
    SQT_AUDIT_DIR naming it. Either way it is a directory cleanup tools
    empty."""
    try:
        return Path(directory).resolve() == _legacy_cache_audit_dir().resolve()
    except (OSError, RuntimeError):
        return False


_legacy_location_warned = False


def _warn_legacy_location_once(legacy: Path) -> None:
    """Say, once per process, that the trail is in the cache location.

    It was a `logger.warning` on every resolution, which is to say on every
    tool call, sent to a logger the package gives only a NullHandler -- so a
    plain script, the `sqt` CLI and the MCP server never showed it, and a
    host that did configure logging got it once per call. `warnings.warn`
    reaches stderr by default and pytest's warnings summary; the log line is
    kept for hosts that read their logs. See the CHANGELOG entry of
    2026-09-27.
    """
    global _legacy_location_warned
    if _legacy_location_warned:
        return
    _legacy_location_warned = True
    message = (
        f"The audit trail is still under {legacy}, which is a CACHE "
        "directory -- cleanup tools empty it and the XDG spec says anything "
        "there is disposable. It is being used anyway so the existing chain "
        "stays continuous. Move it somewhere durable and set SQT_AUDIT_DIR."
    )
    logger.warning(message)
    # stacklevel 3: past this helper and _audit_dir, to whoever asked where
    # the trail lives.
    warnings.warn(message, AuditLocationWarning, stacklevel=3)


def _audit_dir() -> Path:
    """Where the decision record lives.

    NOT under `~/.cache`, which is where this defaulted. A cache is by
    definition the directory a user is invited to delete: `pip cache purge`,
    every "free up disk space" tool, and half the cleanup scripts on the
    internet empty it, and on Linux the XDG spec says an application must be
    able to recreate anything in there. The audit trail is the opposite kind
    of file -- it is the thing you cannot recreate, and the record a
    regulator or an incident review reads. Storing it somewhere designed to
    be cleared is a retention failure waiting for a disk-space warning.

    `~/.local/state` is the XDG directory for exactly this: data that
    persists between runs and is not a cache and not user documents.
    `SQT_AUDIT_DIR` still overrides, and a deployment should point it at
    something backed up. It is read like every other path setting: blank is
    unset, `~` is expanded, and a relative path or one naming an existing
    file is refused by name -- a relative trail would move with the working
    directory, and a decision log split across two directories is two logs
    with a deletion between them.
    """
    override = env_path("SQT_AUDIT_DIR")
    if override is not None:
        return override

    # An existing trail keeps its home. Changing where this points would
    # otherwise orphan every record already written: the new directory starts
    # empty, the chain appears to begin at genesis, and the index that exists
    # to make a missing day detectable has nothing to compare against. That
    # is the same event as a deletion, and it must not be caused by an
    # upgrade. Say so once, loudly enough to be acted on.
    legacy = _legacy_cache_audit_dir()
    if legacy.exists() and any(legacy.glob("*.jsonl")):
        _warn_legacy_location_once(legacy)
        return legacy

    if sys.platform == "win32":
        base = Path(
            os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
        )
        return base / "standard_quant_tools" / "audit"
    state = os.environ.get("XDG_STATE_HOME")
    root = Path(state) if state else Path.home() / ".local" / "state"
    return root / "standard_quant_tools" / "audit"


_GENESIS_HASH = "0" * 16

# Independent witness log at the audit-dir root: records which calendar days
# had activity and what each day's file *should* chain onto, separately from
# the day files themselves. Without this, deleting an entire day's .jsonl is
# undetectable — the next day's chain would start fresh from genesis with no
# reference to whether a prior day ever existed. To hide the deletion of a
# day that has a later day after it, an attacker now has to rewrite both the
# day file AND this index.
#
# The NEWEST day is different, and nothing inside this directory can make it
# otherwise: a newest day cut short is byte for byte an earlier state of the
# log, and deleting it together with the index's last line leaves a shorter
# trail that verifies clean. No rewrite is needed. What anchors the end is
# outside the directory: the verified head `verify_audit_trail_integrity`
# reports (newest day, its record count and last record_hash, the index's
# length and last hash), recorded somewhere else, or a signed checkpoint,
# which commits to the day's record count, a full SHA-256 digest of its
# records and the index's length.
_INDEX_FILENAME = "_chain_index.jsonl"
_DAY_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")


def _iter_day_files(directory: Path) -> List[Path]:
    """Every daily decision-record file in `directory`, sorted chronologically
    (lexicographic sort on YYYY-MM-DD filenames is chronological). Excludes
    the chain index and any lock/hold sidecar files."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.jsonl") if _DAY_FILE_RE.match(p.name))


# The cross-process lock every writer in this package takes before touching
# a day file or the chain index. It lives in `standard_quant_tools._filelock`
# now, because the model registry's promotion log needs the same one; the
# names are kept here because the storage backend imports them from here.
# Best-effort for the audit writer: `_acquire_lock` returns None when the
# platform offers no lock, and the append then proceeds unlocked rather than
# stopping every tool call.
from standard_quant_tools._filelock import acquire_lock as _acquire_lock  # noqa: E402
from standard_quant_tools._filelock import release_lock as _release_lock  # noqa: E402
