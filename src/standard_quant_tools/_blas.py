"""
One BLAS thread around the library's own small linear-algebra calls.

numpy and scipy hand matrix factorizations to OpenBLAS, which by default
starts one thread per logical CPU. For the matrices this library factors --
covariance and correlation matrices of tens to a few hundred assets -- that
default is slower than one thread: `eigh` on a 235x235 matrix took 2.6x as
long at 16 threads as at 1, the SVD of 1260 days of 235 returns 2.7x, and a
235x235 `solve` up to 18.5x as long under OpenBLAS 0.3.31. Its answers also change
in the last bits with the thread count, so the same call gave different bits
on machines with different core counts.

`single_threaded_blas()` runs a block on one BLAS thread: faster for these
sizes, and the same bits on any machine with the same BLAS. Large products
that do gain from threads (a lead-lag matrix product over hundreds of names,
a sample covariance's Gram matrix) are left alone.

A BLAS library has one thread setting for the whole process, so the limit is
reference-counted: every concurrent user runs on one thread, and the setting
in force before the first one entered comes back when the last one leaves.
Two users could otherwise interleave their set and restore, and a result's
bits would depend on timing. While any user is inside, BLAS work on other
threads of the process runs on one thread too.

`SQT_BLAS_THREADS` overrides the limit: a whole number of threads, or 0 to
leave BLAS at whatever the process has (the behaviour before the limit).
Unset or blank is 1. It is read when the first user enters.

Without threadpoolctl, or when it finds no BLAS it can control (Apple's
Accelerate, for one), this does nothing. A failure to set or restore the
limit is logged at debug level and never fails the computation it wraps:
the limit is a speed and reproducibility setting, not part of any answer's
definition. Two known gaps: an OpenBLAS built on OpenMP keeps its thread
count per calling thread, so a limit set on one thread does not reach the
others; and a caller changing the BLAS thread count from another thread
while a user is inside is overwritten when the last user leaves.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Environment variable overriding the limit; see the module docstring.
BLAS_THREADS_ENV = "SQT_BLAS_THREADS"

#: The limit when the variable is unset: one thread.
DEFAULT_BLAS_THREADS = 1

_lock = threading.Lock()
_init_lock = threading.Lock()
_users = 0
#: Each limited library with its thread count from before the first user.
_saved: Optional[List[Tuple[Any, Optional[int]]]] = None
_controller: Optional[Any] = None
_unavailable = False


def _get_controller() -> Optional[Any]:
    """The BLAS libraries' threadpoolctl controller, created once after
    scipy's BLAS loads.

    numpy and scipy each bundle their own OpenBLAS; a controller only sees the
    libraries loaded when it was created, so scipy.linalg is imported first.
    Only the BLAS libraries are selected: an OpenMP runtime's thread count is
    not this module's to set or put back.
    """
    global _controller, _unavailable
    if _controller is not None or _unavailable:
        return _controller
    with _init_lock:
        if _controller is not None or _unavailable:
            return _controller
        try:
            import scipy.linalg  # noqa: F401  (loads scipy's own BLAS)
            from threadpoolctl import ThreadpoolController

            controller = ThreadpoolController().select(user_api="blas")
        except Exception as exc:  # noqa: BLE001 - absent or broken: no limit
            logger.debug("BLAS thread limit unavailable: %s", exc)
            _unavailable = True
            return None
        if not controller.lib_controllers:
            logger.debug("BLAS thread limit unavailable: no controllable BLAS")
            _unavailable = True
            return None
        _controller = controller
    return _controller


def blas_thread_limit() -> Optional[int]:
    """The BLAS threads `single_threaded_blas()` runs a block on: 1 unless
    SQT_BLAS_THREADS says otherwise, None when it is 0 (no limit). A value
    that is not a whole number of at least 0 is refused by name."""
    from standard_quant_tools._env import env_int

    value = env_int(BLAS_THREADS_ENV, DEFAULT_BLAS_THREADS, minimum=0)
    return None if value == 0 else value


def _restore(saved: List[Tuple[Any, Optional[int]]]) -> None:
    for library, threads in saved:
        if threads is None:
            continue
        try:
            library.set_num_threads(threads)
        except Exception as exc:  # noqa: BLE001 - never fail the caller
            logger.debug("BLAS thread count not restored: %s", exc)


def _apply(controller: Any, limit: int) -> Optional[List[Tuple[Any, Optional[int]]]]:
    """Set every controlled BLAS to `limit` threads; the libraries with
    their previous counts, or None (nothing left changed) on a failure.

    The library controllers' own get and set, not `controller.limit()`: the
    limiter reads every library's full description on each entry, which is
    twice the cost (8 us against 3.4 us) for the same two calls per library.
    """
    saved: List[Tuple[Any, Optional[int]]] = []
    try:
        for library in controller.lib_controllers:
            saved.append((library, library.get_num_threads()))
            library.set_num_threads(limit)
    except Exception as exc:  # noqa: BLE001 - never fail the caller
        logger.debug("BLAS thread limit not applied: %s", exc)
        _restore(saved)
        return None
    return saved


@contextmanager
def single_threaded_blas() -> Iterator[None]:
    """Run the enclosed block with every loaded BLAS on one thread (or on
    SQT_BLAS_THREADS threads), restoring the process's setting when the last
    concurrent user leaves."""
    global _users, _saved
    with _lock:
        if _users == 0:
            limit = blas_thread_limit()
            controller = _get_controller() if limit is not None else None
            if controller is not None:
                _saved = _apply(controller, limit)
        _users += 1
    try:
        yield
    finally:
        with _lock:
            _users -= 1
            if _users == 0 and _saved is not None:
                saved, _saved = _saved, None
                _restore(saved)


def _after_fork_in_child() -> None:
    """A forked child starts with no users. The threads that were inside
    the limit in the parent do not exist here, so their exits would never
    come, and a lock held by one of them at the fork would never be
    released. BLAS keeps whatever thread count the parent had at the fork."""
    global _lock, _users, _saved
    _lock = threading.Lock()
    _users = 0
    _saved = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


__all__ = [
    "BLAS_THREADS_ENV",
    "DEFAULT_BLAS_THREADS",
    "blas_thread_limit",
    "single_threaded_blas",
]
