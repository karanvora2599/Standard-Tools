"""Standard quantitative finance tools for backtesting, analysis, and agent-based trading."""

import logging
import os
import sys

__version__ = "0.1.0"

#: Environment variable that forces every kernel onto its Python fallback.
#:
#: WHY THIS EXISTS. Seventeen modules each decide `HAS_CPP` for themselves by
#: probing `_sqt_core`, which is the right design -- a kernel added later
#: falls back per-symbol rather than all-or-nothing. The cost was that the
#: no-extension configuration could not be RUN. Every fallback was reachable
#: only by monkeypatching one module's flag inside one test, so roughly half
#: the C++-adjacent code had no end-to-end coverage at all.
#:
#: That also made the codebase easy to survey wrongly. Instrumenting which
#: functions execute, on a machine where the extension is present, reports
#: every fallback as dead -- and a reachability analysis of this package did
#: exactly that and recommended deleting two of them. `n_workers`'s process
#: pool and eight of ten @njit kernels are not dead; they are the `else:` arm
#: of `if HAS_CPP`. A measurement of what EXECUTES is not a measurement of
#: what is REACHABLE, and the gap between them is one install configuration.
#:
#: So: `SQT_DISABLE_NATIVE=1` makes `_sqt_core` unimportable, every module
#: takes the `except ImportError` branch it already has, and the whole suite
#: can be run against the other configuration. No module needed changing --
#: they all import the same name, so making that one name fail flips all of
#: them at once.
DISABLE_NATIVE_ENV = "SQT_DISABLE_NATIVE"

_TRUTHY = {"1", "true", "yes", "on"}


def native_disabled() -> bool:
    """Whether the compiled extension has been switched off deliberately."""
    return os.environ.get(DISABLE_NATIVE_ENV, "").strip().lower() in _TRUTHY


if native_disabled():
    # `None` in sys.modules makes `import` raise ImportError, which is the
    # exact signal all seventeen call sites already handle. Set before any
    # submodule is imported, so nothing has cached a reference to the real
    # extension by the time it is asked for.
    sys.modules.setdefault(f"{__name__}._sqt_core", None)  # type: ignore[assignment]

# Library-level NullHandler — callers configure handlers; we never emit by default.
logging.getLogger(__name__).addHandler(logging.NullHandler())
