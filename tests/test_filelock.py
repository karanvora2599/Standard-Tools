"""
The one cross-process lock, shared by the decision log and the promotion log.

Pinned here: it serialises threads of one process (each acquisition opens
its own handle, and the OS lock belongs to the handle), a caller that needs
it refuses when it cannot be had while the audit writer's best-effort use
proceeds, and an error that is not contention fails instead of spinning.
"""

from __future__ import annotations

import errno
import sys
import threading
import time

import pytest

from standard_quant_tools import _filelock
from standard_quant_tools.error import ValidationError


def test_it_serialises_a_read_modify_write_across_threads(tmp_path):
    counter = tmp_path / "counter"
    counter.write_text("0", encoding="utf-8")
    lock = tmp_path / ".counter.lock"
    barrier = threading.Barrier(6)

    def bump(times):
        barrier.wait()
        for _ in range(times):
            with _filelock.exclusive(lock, required=True):
                value = int(counter.read_text(encoding="utf-8"))
                time.sleep(0.0005)
                counter.write_text(str(value + 1), encoding="utf-8")

    threads = [threading.Thread(target=bump, args=(20,)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert counter.read_text(encoding="utf-8") == "120"


def test_a_required_lock_that_cannot_be_had_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(_filelock, "acquire_lock", lambda path: None)
    ran = []
    with pytest.raises(ValidationError, match="the promotion was not made"):
        with _filelock.exclusive(
            tmp_path / ".x.lock", required=True, purpose="the promotion"
        ):
            ran.append(True)
    assert ran == []


def test_a_best_effort_lock_that_cannot_be_had_runs_unlocked(tmp_path, monkeypatch):
    """Null: the audit writer's use -- no lock is an answer, not a refusal."""
    monkeypatch.setattr(_filelock, "acquire_lock", lambda path: None)
    with _filelock.exclusive(tmp_path / ".x.lock") as handle:
        assert handle is None


def test_the_audit_writer_uses_the_same_primitive():
    from standard_quant_tools.audit import paths, storage

    assert paths._acquire_lock is _filelock.acquire_lock
    assert storage._release_lock is _filelock.release_lock


@pytest.mark.skipif(sys.platform != "win32", reason="msvcrt is Windows-only")
def test_an_error_that_is_not_contention_does_not_spin(tmp_path, monkeypatch):
    import msvcrt

    def broken(fileno, mode, nbytes):
        raise OSError(errno.EBADF, "bad file descriptor")

    monkeypatch.setattr(msvcrt, "locking", broken)
    started = time.monotonic()
    assert _filelock.acquire_lock(tmp_path / ".x.lock") is None
    assert time.monotonic() - started < 5
