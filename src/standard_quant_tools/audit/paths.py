"""Where the audit trail lives on disk, how its files are named/discovered,
and the cross-process advisory locking primitive every writer in this
package uses before touching a day file or the chain index."""

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _audit_enabled() -> bool:
    return os.environ.get("SQT_AUDIT_ENABLED", "1").lower() not in ("0", "false", "")


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
    something backed up.
    """
    override = os.environ.get("SQT_AUDIT_DIR")
    if override:
        return Path(override)

    # An existing trail keeps its home. Changing where this points would
    # otherwise orphan every record already written: the new directory starts
    # empty, the chain appears to begin at genesis, and the index that exists
    # to make a missing day detectable has nothing to compare against. That
    # is the same event as a deletion, and it must not be caused by an
    # upgrade. Say so once, loudly enough to be acted on.
    legacy = Path.home() / ".cache" / "standard_quant_tools" / "audit"
    if legacy.exists() and any(legacy.glob("*.jsonl")):
        logger.warning(
            "The audit trail is still under %s, which is a CACHE directory -- "
            "cleanup tools empty it and the XDG spec says anything there is "
            "disposable. It is being used anyway so the existing chain stays "
            "continuous. Move it somewhere durable and set SQT_AUDIT_DIR.",
            legacy,
        )
        return legacy

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return base / "standard_quant_tools" / "audit"
    state = os.environ.get("XDG_STATE_HOME")
    root = Path(state) if state else Path.home() / ".local" / "state"
    return root / "standard_quant_tools" / "audit"


_GENESIS_HASH = "0" * 16

# Independent witness log at the audit-dir root: records which calendar days
# had activity and what each day's file *should* chain onto, separately from
# the day files themselves. Without this, deleting an entire day's .jsonl is
# undetectable — the next day's chain would start fresh from genesis with no
# reference to whether a prior day ever existed. An attacker now has to
# consistently rewrite both the day file AND this index to hide a deletion.
_INDEX_FILENAME = "_chain_index.jsonl"
_DAY_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")


def _iter_day_files(directory: Path) -> List[Path]:
    """Every daily decision-record file in `directory`, sorted chronologically
    (lexicographic sort on YYYY-MM-DD filenames is chronological). Excludes
    the chain index and any lock/hold sidecar files."""
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.jsonl") if _DAY_FILE_RE.match(p.name))


def _acquire_lock(lock_path: Path) -> Optional[Any]:
    """
    Best-effort cross-process exclusive lock via a small sidecar file (not
    the growing JSONL file itself — locking a fixed, tiny file avoids the
    platform-specific complexity of byte-range-locking a file whose EOF
    offset keeps moving). Returns an open file handle the caller must pass
    to `_release_lock`, or None if locking isn't available on this platform
    — in which case writes proceed unlocked rather than blocking a tool
    call on a missing OS primitive.
    """
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lf = open(lock_path, "a+b")
    except Exception:
        logger.debug("[audit] advisory file lock unavailable", exc_info=True)
        return None
    try:
        if sys.platform == "win32":
            import msvcrt

            lf.seek(0)
            # msvcrt.locking(LK_LOCK) retries internally for ~10s, then
            # raises OSError -- unlike POSIX fcntl.flock(LOCK_EX) below,
            # which blocks indefinitely. Left as a single attempt, a lock
            # held >10s by another process/thread would raise here, and the
            # blanket except below would silently return None -- letting
            # the caller proceed with NO lock at all, unlike POSIX. Retry
            # in a loop instead so both platforms block indefinitely under
            # contention, matching the OS-level lock's other property on
            # both platforms: it's released automatically if the holder
            # crashes, so this isn't a new hang risk versus POSIX today.
            while True:
                try:
                    msvcrt.locking(lf.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    continue
        else:
            import fcntl

            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        return lf
    except Exception:
        logger.debug("[audit] advisory file lock unavailable", exc_info=True)
        lf.close()
        return None


def _release_lock(lf: Optional[Any]) -> None:
    if lf is None:
        return
    try:
        if sys.platform == "win32":
            import msvcrt

            lf.seek(0)
            msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    finally:
        lf.close()
