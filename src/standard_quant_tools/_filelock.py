"""
One cross-process lock for every read-modify-write this library makes on a
shared file.

The audit writer has always taken one before touching a day file or the
chain index. The model registry's promotion log is the same shape -- read
the current stage, check the move is allowed, append the decision -- and it
took none, so two callers could both validate against the same old stage
and both append, leaving a history in which a model was archived and live
in staging at once. The primitive lives here so both use the same one.

A SIDECAR FILE, NOT THE DATA FILE. Locking a fixed, tiny file avoids the
platform-specific trouble of byte-range-locking a file whose end keeps
moving. Each acquisition opens its own handle, and both OS primitives
(`msvcrt.locking` on Windows, `fcntl.flock` elsewhere) belong to the handle,
so the lock serialises threads of one process as well as separate
processes, and the OS releases it if the holder dies.

WHAT HAPPENS WHEN NO LOCK CAN BE TAKEN is the caller's decision, because the
right answer differs. The audit writer proceeds unlocked (`acquire_lock`
returns None): refusing would stop every tool call on a filesystem without
lock support, and a decision log that stops recording is the worse failure.
A promotion refuses (`exclusive(..., required=True)`): promotions are rare,
and an unserialised one is exactly the race the lock exists to close.

Top-level for the reason `_containment` and `_runspath` are: the audit
package and the model registry both need it and neither should import the
other to get it.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

#: The errors `msvcrt.locking` raises when another handle holds the byte:
#: EACCES from a non-blocking attempt, EDEADLOCK from a blocking one that
#: gave up after its own ten retries. Anything else is a failure to lock,
#: not contention, and retrying it would spin forever.
_CONTENDED = frozenset(
    {errno.EACCES, errno.EAGAIN, errno.EDEADLK, getattr(errno, "EDEADLOCK", -1)}
)

#: Bounds of the wait between attempts on Windows. Short to start with,
#: because a promotion or an audit append holds the lock for milliseconds.
_FIRST_WAIT_SECONDS = 0.001
_LONGEST_WAIT_SECONDS = 0.05


def _lock_windows(handle: Any) -> None:
    import msvcrt

    handle.seek(0)
    # Non-blocking attempts in a loop rather than LK_LOCK: LK_LOCK retries
    # once a second for ten seconds and then raises, so a waiter noticed a
    # released lock up to a second late and gave up on a held one after
    # ten. This blocks until the lock is free, as flock does on POSIX, and
    # notices within a few milliseconds.
    wait = _FIRST_WAIT_SECONDS
    while True:
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError as exc:
            if exc.errno not in _CONTENDED:
                raise
        time.sleep(wait)
        wait = min(wait * 2, _LONGEST_WAIT_SECONDS)


def acquire_lock(lock_path: Path) -> Optional[Any]:
    """
    Block until the exclusive lock on `lock_path` is held, creating the
    file (and its directory) if needed.

    Returns the open handle, which `release_lock` takes, or None when the
    platform or filesystem offers no lock -- the caller then decides
    whether to proceed unlocked or refuse.
    """
    lock_path = Path(lock_path)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+b")
    except Exception:  # noqa: BLE001 - no lock is an answer, not an error
        logger.debug("[filelock] cannot open %s", lock_path, exc_info=True)
        return None
    try:
        if sys.platform == "win32":
            _lock_windows(handle)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle
    except Exception:  # noqa: BLE001 - no lock is an answer, not an error
        logger.debug("[filelock] cannot lock %s", lock_path, exc_info=True)
        handle.close()
        return None


def release_lock(handle: Optional[Any]) -> None:
    """Release and close what `acquire_lock` returned; None is a no-op."""
    if handle is None:
        return
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:  # noqa: BLE001 - closing the handle releases it anyway
        logger.debug("[filelock] unlock failed; closing releases it", exc_info=True)
    finally:
        handle.close()


@contextlib.contextmanager
def exclusive(
    lock_path: Path, *, required: bool = False, purpose: str = "this change"
) -> Iterator[Optional[Any]]:
    """
    Hold the lock on `lock_path` for the body of a `with`.

    `required=True` refuses with a ValidationError when no lock can be
    taken, naming `purpose`, instead of running the body unlocked.
    """
    handle = acquire_lock(Path(lock_path))
    if handle is None and required:
        raise ValidationError(
            f"{purpose} was not made: the lock file {lock_path} could not be "
            "created or locked, and making the change unlocked could "
            "interleave it with another writer's. Check that the directory "
            "is writable and on a filesystem that supports file locks, then "
            "retry."
        )
    try:
        yield handle
    finally:
        release_lock(handle)


__all__ = ["acquire_lock", "exclusive", "release_lock"]
