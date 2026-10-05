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
sizes, and the same bits on any machine with the same BLAS.

The products that build those matrices run under it too: the sample,
Ledoit-Wolf and EWMA covariances, the `DataFrame.cov()` the optimizers read
(np.cov when nothing is missing; with gaps, pandas' own pairwise loop,
which uses no BLAS), PCA's factor returns, and the network features' and
the lead-lag correlations. Their last bits followed the thread count as
well -- np.cov's on the CI runners' OpenBLAS, the EWMA, factor-return,
network and lead-lag products under 0.3.27 and 0.3.31 at sixteen threads --
and so did every output built from them. At 235 assets and below no whole
call measured was more than 8% slower with them on one thread (PCA's), and
most were faster. Wider, some of these products are slower on one: on a
shared 16-thread machine the factor returns of 2,106 days of 500 assets
took 3.0-7.2x as long and the network features' four products over 1,000
names 1.8-3.4x, which made those whole calls 1.05-1.22x and 1.17-1.50x as
long. That is the price of a whole output, not only its factorizations,
being the same bits on any machine with the same BLAS. Matrix-vector
products outside these blocks (a portfolio's variance, an optimizer's
gradient) keep the caller's setting: they reduce over the assets, and gave
the same bits at every thread count measured. Longer reductions do not:
OpenBLAS splits a dot product of more than 10,000 terms across threads,
and under 0.3.27 a matrix-vector product reducing 100,000 rows at four
threads and more. So the half-life statistics' sums of squares and the
depth slope's run under the limit too, with the rest of the linear algebra
on the library's own paths: `pca_whiten`'s decomposition and projection,
the feature VIFs, and the factor, ADF and Engle-Granger regressions.

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

`openmp_thread_limit(n)` is the same reference-counted limit for the OpenMP
runtimes instead of the BLAS: the threads scikit-learn's histogram gradient
boosting, LightGBM and XGBoost start for one fit or prediction. Those read
the runtime's thread count when a fit or a prediction starts, and at their
default they take every logical CPU. Under the PASSIVE wait policy this
package sets on import, histogram boosting on 16 threads fitted 15,000 rows
of 8 features in 1.5 to 1.7 s, and on one thread in 0.28 to 0.36 s, with
the same predictions. The runtimes keep that count in two different ways,
and the limit follows each:

- Under the MSVC runtime (vcomp) the count set on any thread reaches every
  thread -- `threadpool_limits` on a worker thread was measured reaching
  the main thread -- so it cannot be scoped to one call. It is kept
  process-wide and counted like the BLAS limit: the count in force before
  the first user comes back when the last one leaves, and a user entering
  while another is inside runs on the count already in force.
- GNU libgomp, LLVM's libomp and Intel's runtime keep it per thread, as the
  OpenMP specification has `omp_set_num_threads` do: a count set on one
  thread is not seen by another. Each thread sets its own on its first
  entry and puts back its own on its last exit, so every pooled worker runs
  under the limit and no thread is left limited after its users leave. Kept
  process-wide there, the count of the first thread in would have stayed
  on it whenever another thread left last.
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
    set and put back by `openmp_thread_limit` alone.
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


_omp_lock = threading.Lock()
_omp_users = 0
#: Each OpenMP runtime with its thread count from before the first user.
_omp_saved: Optional[List[Tuple[Any, Optional[int]]]] = None
#: The count the first user set, which later users run on.
_omp_in_force: Optional[int] = None


#: The OpenMP runtimes last found, and how many Python modules were loaded
#: when they were looked for.
_omp_found: Optional[Tuple[int, List[Any]]] = None


def _openmp_libraries() -> List[Any]:
    """The OpenMP runtimes loaded in this process.

    Looked for again whenever a Python module has been imported since the
    last look: an estimator library that brings its own runtime may load
    after the first fit, and a controller sees only the libraries present
    when it was made. Looking costs 3 to 6 ms, which a hyperparameter search
    scoring thousands of candidates would otherwise pay on every one.
    """
    global _omp_found
    import sys

    loaded = len(sys.modules)
    if _omp_found is not None and _omp_found[0] == loaded:
        return _omp_found[1]
    try:
        from threadpoolctl import ThreadpoolController

        libraries = list(
            ThreadpoolController().select(user_api="openmp").lib_controllers
        )
    except Exception as exc:  # noqa: BLE001 - absent or broken: no limit
        logger.debug("OpenMP thread limit unavailable: %s", exc)
        libraries = []
    _omp_found = (len(sys.modules), libraries)
    return libraries


def _set_openmp(libraries: List[Any], threads: int) -> None:
    for library in libraries:
        try:
            library.set_num_threads(threads)
        except Exception as exc:  # noqa: BLE001 - never fail the caller
            logger.debug("OpenMP thread limit not applied: %s", exc)


def _read_openmp(libraries: List[Any]) -> List[Tuple[Any, Optional[int]]]:
    saved: List[Tuple[Any, Optional[int]]] = []
    for library in libraries:
        try:
            saved.append((library, library.get_num_threads()))
        except Exception as exc:  # noqa: BLE001 - never fail the caller
            logger.debug("OpenMP thread count not read: %s", exc)
    return saved


def openmp_count_is_process_wide(library: Any) -> bool:
    """Whether a runtime's thread count, set on one thread, reaches every
    thread: MSVC's vcomp, measured. Every other runtime keeps it per thread,
    as the OpenMP specification has `omp_set_num_threads` do."""
    import ntpath

    # ntpath splits on both separators, so a Windows path is read the same
    # wherever this runs.
    path = str(getattr(library, "filepath", "") or "")
    name = ntpath.basename(path).lower() or str(getattr(library, "prefix", "")).lower()
    return name.startswith("vcomp")


#: Per thread: how deep this thread is inside the limit, and its own counts
#: for the per-thread runtimes from before its first entry.
_omp_local = threading.local()


@contextmanager
def openmp_thread_limit(threads: Optional[int]) -> Iterator[None]:
    """
    Run the enclosed block with every loaded OpenMP runtime at `threads`
    threads, putting each runtime's count back when its users leave. None
    or below 1 leaves the runtimes alone.

    A process-wide runtime is counted across threads: a user entering while
    another is inside runs on the count the first one set, and the count
    from before the first comes back when the last leaves. A per-thread
    runtime is set and put back by each thread for itself, a nested user
    running on its thread's outer count. The estimators this wraps give the
    same predictions at any count, so either changes how long the block
    takes and nothing it returns.
    """
    global _omp_users, _omp_saved, _omp_in_force
    if threads is None or int(threads) < 1:
        yield
        return
    count = int(threads)
    libraries = _openmp_libraries()
    shared = [lib for lib in libraries if openmp_count_is_process_wide(lib)]
    own = [lib for lib in libraries if not openmp_count_is_process_wide(lib)]

    depth = getattr(_omp_local, "depth", 0)
    if depth == 0:
        _omp_local.saved = _read_openmp(own)
        _set_openmp([library for library, _ in _omp_local.saved], count)
    _omp_local.depth = depth + 1
    with _omp_lock:
        if _omp_users == 0:
            _omp_saved = _read_openmp(shared)
            _omp_in_force = count
        _set_openmp([library for library, _ in _omp_saved or []], _omp_in_force)
        _omp_users += 1
    try:
        yield
    finally:
        with _omp_lock:
            _omp_users -= 1
            if _omp_users == 0:
                saved, _omp_saved = _omp_saved or [], None
                _omp_in_force = None
                _restore(saved)
        _omp_local.depth -= 1
        if _omp_local.depth == 0:
            mine, _omp_local.saved = _omp_local.saved or [], None
            _restore(mine)


def _after_fork_in_child() -> None:
    """A forked child starts with no users. The threads that were inside
    the limit in the parent do not exist here, so their exits would never
    come, and a lock held by one of them at the fork would never be
    released. BLAS and OpenMP keep whatever thread counts the parent had at
    the fork."""
    global _lock, _users, _saved, _omp_lock, _omp_users, _omp_saved, _omp_in_force
    _lock = threading.Lock()
    _users = 0
    _saved = None
    _omp_lock = threading.Lock()
    _omp_users = 0
    _omp_saved = None
    _omp_in_force = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


__all__ = [
    "BLAS_THREADS_ENV",
    "DEFAULT_BLAS_THREADS",
    "blas_thread_limit",
    "openmp_thread_limit",
    "single_threaded_blas",
]
