"""Standard quantitative finance tools for backtesting, analysis, and agent-based trading."""

import logging
import os
import sys

__version__ = "0.1.0"

#: The OpenMP runtime's wait policy, defaulted to PASSIVE here.
#:
#: After each parallel region, the MSVC OpenMP runtime (vcomp) keeps its
#: worker threads spinning for about 100 ms, waiting for the next region.
#: That spin kept 12 to 14 of 16 logical CPUs busy on the measuring machine,
#: and the Python that ran after a kernel -- the pandas and numpy around it --
#: ran 1.4x to 2.2x slower beside it. PASSIVE puts the workers to sleep instead:
#: over 35 public calls at 16 threads the geometric-mean speed-up over one
#: thread went from 1.79x to 2.13x. The cost is on kernels called back to
#: back with no Python between them, 6% to 24% slower, and 0.1 to 0.6 ms to
#: wake the workers for a small region. No result changes.
#:
#: The runtime reads the variable once, when it loads, and it loads with
#: `_sqt_core` -- so it is set here, before anything below can import the
#: extension. A value the caller set, in the environment or in `os.environ`
#: before this import, is left alone; blank counts as unset, as it does for
#: every setting this library reads. Being process-wide, it also reaches
#: scikit-learn's OpenMP and any child process. When the runtime is already
#: loaded -- scikit-learn imported first loads its own copy, which the
#: extension then shares -- setting it changes nothing for this process,
#: and a debug-level log line says so. Not a warning: it changes speed only,
#: never an answer, and it follows from ordinary import order a caller may
#: not control, so a warning would fire on every such import (and fail it
#: under `-W error`) for something only a profile needs to know. The outcome
#: is in `_OMP_WAIT_POLICY_DEFAULT`: "set", "caller" or "too_late".
OMP_WAIT_POLICY_ENV = "OMP_WAIT_POLICY"


def _openmp_runtime_loaded() -> "bool | None":
    """Whether the MSVC OpenMP runtime is already in this process; None
    where that cannot be told (any platform but Windows)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        kernel32.GetModuleHandleW.restype = ctypes.c_void_p
        return bool(kernel32.GetModuleHandleW("vcomp140.dll"))
    except Exception:  # noqa: BLE001 - a diagnostic, never an import failure
        return None


def _default_omp_wait_policy() -> str:
    if (os.environ.get(OMP_WAIT_POLICY_ENV) or "").strip():
        return "caller"
    os.environ[OMP_WAIT_POLICY_ENV] = "PASSIVE"
    if _openmp_runtime_loaded():
        logging.getLogger(__name__).debug(
            "OMP_WAIT_POLICY=PASSIVE was set on import, but the OpenMP runtime "
            "(vcomp140.dll) was already loaded -- by a package imported before "
            "standard_quant_tools, such as scikit-learn -- and reads it only "
            "when it loads. This process keeps the runtime's default: workers "
            "spin about 100 ms after each parallel region, slowing the Python "
            "that follows. Results are unaffected. Import standard_quant_tools "
            "first, or set OMP_WAIT_POLICY before Python starts."
        )
        return "too_late"
    return "set"


_OMP_WAIT_POLICY_DEFAULT = _default_omp_wait_policy()

#: Environment variable that forces every kernel onto its Python fallback.
#:
#: WHY THIS EXISTS. Eighteen modules each decide `HAS_CPP` for themselves by
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


def native_disabled() -> bool:
    """Whether the compiled extension has been switched off deliberately.

    Read through the library's one flag reader, the same one the audit
    switches use, so a word means one thing everywhere: 1/true/yes/on and
    0/false/no/off in any case and padding, blank for the default (the
    extension is used), and any other word refused by name. A local `.env`
    is not loaded for this one read -- it runs while the package imports.
    """
    from standard_quant_tools._env import env_flag

    return env_flag(DISABLE_NATIVE_ENV, False, load=False)


_native_off = native_disabled()
if _native_off:
    # `None` in sys.modules makes `import` raise ImportError, which is the
    # exact signal all eighteen call sites already handle. Set before any
    # submodule is imported, so nothing has cached a reference to the real
    # extension by the time it is asked for.
    sys.modules.setdefault(f"{__name__}._sqt_core", None)  # type: ignore[assignment]

# An extension that imports is not necessarily one built from the C++ beside
# this package: an editable install keeps its own compiled copy, which goes
# on answering with old kernels after the sources move on. It is checked
# here, once, against a digest of those sources and refused -- through the
# same `None` in sys.modules as above -- when it does not match, so every
# module falls back together and a NativeBuildWarning says which file and
# how to refresh it. See `_native_build` and the CHANGELOG entry of
# 2026-09-28.
from standard_quant_tools._native_build import (  # noqa: E402,F401
    NativeBuildWarning,
    native_build_status,
    screen_extension,
)

screen_extension(sys.modules[__name__], disabled=_native_off)
del _native_off, screen_extension

# Library-level NullHandler — callers configure handlers; we never emit by default.
logging.getLogger(__name__).addHandler(logging.NullHandler())
