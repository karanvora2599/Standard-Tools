"""Best-effort reproducibility provenance: C++ extension availability, which
build of it ran and which instruction-set path its kernels took, the
compiler and configuration that build was made with, the C runtime and OS
the process runs on, the current git commit, the installed package version,
and a content hash of a registered strategy's source code. All of these fail
silently (return `None`/`False`) rather than raise — provenance is a
nice-to-have, never a reason to break a tool call."""

import functools
import os
import struct
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hashing import hash_payload


def _cpp_available() -> bool:
    try:
        import standard_quant_tools._sqt_core  # type: ignore[attr-defined]  # noqa: F401

        return True
    except ImportError:
        return False


def _native_build_label() -> Optional[str]:
    """Which build of the extension this process runs, as one token: the
    import-time verdict and the short digest of the sources the extension
    was built from (`match:df27c6e4af54`), or why none ran (`absent`,
    `disabled`, `stale:…`). `_cpp_available` says a build ran; this says
    which one."""
    try:
        from standard_quant_tools._native_build import native_build_status

        return native_build_status().label
    except Exception:
        return None


def _native_isa_label() -> Optional[str]:
    """Which instruction-set path the compiled kernels take on this machine
    (`avx2+fma`, `scalar`), or `none` when no extension ran. Recorded beside
    `native_build` because the build alone does not fix the last bits of an
    output: the same binary rounds differently on the AVX2+FMA path than on
    the scalar one."""
    try:
        from standard_quant_tools._native_build import native_isa

        return native_isa()
    except Exception:
        return None


# ── The build behind the label ───────────────────────────────────────────────

#: Build-stamp keys `native_build` already carries, as its short digest.
_LABEL_KEYS = frozenset({"source_digest", "source_files"})

#: The DLLs a Windows binary imports when it links the Universal CRT
#: dynamically (/MD): the CRT itself, or the API sets that forward to it.
_UCRT_IMPORTS = ("ucrtbase.dll", "ucrtbased.dll")
_UCRT_API_SET_PREFIX = "api-ms-win-crt-"


def _json_scalar(value: Any) -> Any:
    """A build-stamp value as JSON writes it back: str, int, bool or None
    unchanged, anything else as its str()."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    return str(value)


def _pe_imports(path: str) -> Optional[List[str]]:
    """
    The DLL names a PE image (a Windows `.pyd` or `.exe`) imports,
    lowercased, in table order; None for a file that is not a PE image.

    Reads the import directory and nothing else. Raises on a file that
    cannot be read or a table that runs off the end of it; the caller turns
    that into "cannot tell".
    """
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:2] != b"MZ":
        return None
    (pe_offset,) = struct.unpack_from("<I", data, 0x3C)
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        return None
    coff = pe_offset + 4
    (n_sections,) = struct.unpack_from("<H", data, coff + 2)
    (optional_size,) = struct.unpack_from("<H", data, coff + 16)
    optional = coff + 20
    (magic,) = struct.unpack_from("<H", data, optional)
    # The data directories start 96 bytes into a PE32 optional header and
    # 112 into a PE32+ one; the import table is the second entry.
    directories = optional + (112 if magic == 0x20B else 96)
    (import_rva,) = struct.unpack_from("<I", data, directories + 8)
    if import_rva == 0:
        return []
    sections: List[Tuple[int, int, int]] = []
    table = optional + optional_size
    for index in range(n_sections):
        virtual_size, address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", data, table + 40 * index + 8
        )
        sections.append((address, max(virtual_size, raw_size), raw_offset))

    def file_offset(rva: int) -> int:
        for address, size, raw_offset in sections:
            if address <= rva < address + size:
                return rva - address + raw_offset
        raise ValueError(f"RVA {rva:#x} lies in no section")

    names: List[str] = []
    entry = file_offset(import_rva)
    while True:
        lookup, _stamp, _chain, name_rva, thunk = struct.unpack_from(
            "<IIIII", data, entry
        )
        if not (lookup or name_rva or thunk):
            return names
        start = file_offset(name_rva)
        end = data.index(b"\0", start)
        names.append(data[start:end].decode("ascii").lower())
        entry += 20


@functools.lru_cache(maxsize=8)
def _crt_linkage(extension_file: Optional[str]) -> Optional[str]:
    """
    How the extension links the C runtime its `exp`, `log` and `erf` come
    from: `dynamic` when it imports the Universal CRT (`ucrtbase.dll` or the
    `api-ms-win-crt-*` sets that forward to it -- MSVC's /MD), `static` when
    it imports neither (/MT: a copy of the CRT's math is inside the binary,
    so the system's `ucrtbase.dll` does not reach the extension's kernels).
    None when it cannot be told: no file, not a Windows image, unreadable.

    The two linkages round differently in places (/MD against /MT moved the
    last bits of 5 of 476 implied volatilities, measured), and nothing in
    the build stamp says which one a binary used, so it is read from the
    binary itself.
    """
    if not extension_file:
        return None
    try:
        imports = _pe_imports(extension_file)
    except Exception:
        return None
    if imports is None:
        return None
    dynamic = any(
        name in _UCRT_IMPORTS or name.startswith(_UCRT_API_SET_PREFIX)
        for name in imports
    )
    return "dynamic" if dynamic else "static"


_native_detail_cache: Tuple[Any, Optional[Dict[str, Any]]] = (None, None)


def _native_detail() -> Optional[Dict[str, Any]]:
    """
    The build facts of the extension this process uses, beyond the source
    digest `native_build` already names: every key of the build stamp
    `_sqt_core.__build_info__` but the digest -- the compiler and its
    version, the configuration (`build_type`), whether host-CPU code
    generation was requested (`native_arch`), the OpenMP version and
    runtime, the PGO phase -- plus `crt_linkage`, read from the binary.

    Why: MSVC with /arch:AVX2, MSVC with SSE2, clang-cl and a PGO build of
    the same sources all carry one `native_build` label, and a clang-cl
    build that contracts multiply-adds differs from the others on most
    kernels. These facts tell such builds apart -- all but two builds by
    the same compiler that differ only in a flag the stamp does not carry,
    such as a contraction flag given on the command line rather than in
    the build files (which the source digest covers).

    None when no extension is in use (every kernel ran its Python path, so
    no build facts apply) or on any failure. The stamp is read defensively:
    a key it lacks is simply absent, and a value that is not a JSON scalar
    is recorded as its str(). Resolved once per import verdict and cached.
    """
    global _native_detail_cache
    try:
        from standard_quant_tools._native_build import native_build_status

        status = native_build_status()
        cached_for, cached = _native_detail_cache
        if cached_for is not status:
            cached = None
            if status.used:
                stamp = dict(status.build or {})
                cached = {
                    str(key): _json_scalar(value)
                    for key, value in sorted(stamp.items(), key=lambda kv: str(kv[0]))
                    if str(key) not in _LABEL_KEYS
                }
                # A stamp that one day reports the linkage itself wins.
                cached.setdefault("crt_linkage", _crt_linkage(status.extension_file))
            _native_detail_cache = (status, cached)
        return dict(cached) if cached is not None else None
    except Exception:
        return None


# ── The platform under the process ───────────────────────────────────────────


def _windows_file_version(path: str) -> Optional[str]:
    """The file version in a Windows binary's version resource,
    `major.minor.build.revision`, or None."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    version = ctypes.WinDLL("version")
    size_of = version.GetFileVersionInfoSizeW
    size_of.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    size_of.restype = wintypes.DWORD
    read = version.GetFileVersionInfoW
    read.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    read.restype = wintypes.BOOL
    query = version.VerQueryValueW
    query.argtypes = [
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.UINT),
    ]
    query.restype = wintypes.BOOL

    size = size_of(path, None)
    if not size:
        return None
    block = ctypes.create_string_buffer(size)
    if not read(path, 0, size, block):
        return None
    pointer = ctypes.c_void_p()
    length = wintypes.UINT()
    if not query(block, "\\", ctypes.byref(pointer), ctypes.byref(length)):
        return None
    if not pointer.value or length.value < 16:
        return None
    # VS_FIXEDFILEINFO: signature, struct version, then the file version as
    # two DWORDs, most significant first.
    signature, _struct, high, low = struct.unpack(
        "<IIII", ctypes.string_at(pointer.value, 16)
    )
    if signature != 0xFEEF04BD:
        return None
    return f"{high >> 16}.{high & 0xFFFF}.{low >> 16}.{low & 0xFFFF}"


def _loaded_ucrt_path() -> Optional[str]:
    """Where the `ucrtbase.dll` this process loaded lives; the system copy
    when it cannot be asked. None off Windows."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32")
    get_handle = kernel32.GetModuleHandleW
    get_handle.argtypes = [wintypes.LPCWSTR]
    get_handle.restype = wintypes.HMODULE
    get_name = kernel32.GetModuleFileNameW
    get_name.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    get_name.restype = wintypes.DWORD

    handle = get_handle("ucrtbase.dll")
    if handle:
        buffer = ctypes.create_unicode_buffer(32768)
        if get_name(handle, buffer, len(buffer)):
            return buffer.value
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    return os.path.join(root, "System32", "ucrtbase.dll")


def _os_build() -> Optional[str]:
    """The OS and its build: `Windows 10.0.26300`, `macOS 14.5`, or on
    Linux the kernel release."""
    import platform

    if sys.platform == "win32":
        version = sys.getwindowsversion()
        return f"Windows {version.major}.{version.minor}.{version.build}"
    if sys.platform == "darwin":
        release = platform.mac_ver()[0]
        return f"macOS {release}" if release else f"Darwin {platform.release()}"
    described = f"{platform.system() or sys.platform} {platform.release()}"
    return described.strip() or None


def _crt_version() -> Optional[str]:
    """
    The C runtime whose `exp`, `log`, `erf` and friends the process calls:
    the loaded `ucrtbase.dll`'s file version on Windows (an OS component,
    updated with it), the glibc version on Linux, and the OS version on
    macOS, whose libm ships with it. None elsewhere or when unknown.
    """
    import platform

    if sys.platform == "win32":
        path = _loaded_ucrt_path()
        version = _windows_file_version(path) if path else None
        return f"ucrtbase {version}" if version else None
    if sys.platform == "darwin":
        release = platform.mac_ver()[0]
        return f"libsystem_m (macOS {release})" if release else None
    try:
        confstr = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, ValueError, OSError):
        confstr = None
    if confstr:
        return confstr
    library, version = platform.libc_ver()
    return f"{library} {version}" if library and version else None


_X86_MACHINES = frozenset({"x86_64", "amd64", "i386", "i486", "i586", "i686", "x86"})


def _crt_fma3_probe() -> Callable[[], Optional[bool]]:
    """
    A cheap callable answering whether the C runtime's math functions take
    their FMA3 code path in this process.

    Windows: the Universal CRT's own answer, `_get_FMA3_enable()`, read each
    time it is asked, since `_set_FMA3_enable` can change it at runtime.
    Linux on x86: glibc picks its FMA variants of `exp`, `log`, `pow`, `sin`
    and the rest when the CPU offers both FMA and AVX2, so it is read from
    the CPU flags the kernel reports. None where neither applies.
    """
    import platform

    if sys.platform == "win32":
        import ctypes

        getter = getattr(ctypes.CDLL("ucrtbase"), "_get_FMA3_enable", None)
        if getter is None:  # not exported outside x64
            return _unknown
        getter.argtypes = []
        getter.restype = ctypes.c_int
        return lambda: bool(getter())
    if sys.platform.startswith("linux") and platform.machine().lower() in (
        _X86_MACHINES
    ):
        flags: Optional[bool] = None
        with open("/proc/cpuinfo", encoding="ascii", errors="replace") as handle:
            for line in handle:
                if line.startswith("flags"):
                    present = set(line.partition(":")[2].split())
                    flags = "fma" in present and "avx2" in present
                    break
        return lambda: flags
    return _unknown


def _machine() -> Optional[str]:
    """The machine architecture, as `platform.machine()` names it."""
    import platform

    return platform.machine() or None


def _unknown() -> Optional[bool]:
    return None


_platform_lock = threading.Lock()
_platform_cache: Optional[Tuple[Dict[str, Any], Callable[[], Optional[bool]]]] = None


def _platform_facts() -> Optional[Dict[str, Any]]:
    """
    The facts about the machine and OS the last bits of an output depend
    on, beyond the extension: `os` (the OS and its build), `machine` (the
    architecture), `crt` (the C runtime and its version) and `crt_fma3`
    (whether that runtime takes its FMA3 code path here). Each is None when
    it cannot be determined.

    These reach the Python path too: `math.exp` and `math.erf` are the C
    runtime's, and its FMA3 path rounds differently from its SSE2 path on
    about one input in a thousand (0.146% of `exp` and 0.112% of `erf`
    inputs, measured), so the same Python on another CPU or another
    Windows update can differ in the last bits.

    Resolved once per process and cached, except `crt_fma3`, which is read
    on every call because the C runtime lets a program switch it.
    """
    global _platform_cache
    try:
        if _platform_cache is None:
            with _platform_lock:
                if _platform_cache is None:
                    facts: Dict[str, Any] = {}
                    for name, probe in (
                        ("os", _os_build),
                        ("machine", _machine),
                        ("crt", _crt_version),
                    ):
                        try:
                            facts[name] = probe()
                        except Exception:
                            facts[name] = None
                    try:
                        fma3 = _crt_fma3_probe()
                    except Exception:
                        fma3 = _unknown
                    _platform_cache = (facts, fma3)
        facts, fma3 = _platform_cache
        try:
            on_fma3 = fma3()
        except Exception:
            on_fma3 = None
        return {**facts, "crt_fma3": on_fma3}
    except Exception:
        return None


_git_sha_cache: Optional[str] = None
_git_sha_resolved = False


def _git_sha() -> Optional[str]:
    """
    Best-effort `git rev-parse HEAD` in the repo containing this package.
    Returns None (never raises) outside a git checkout, without git
    installed, or in any other failure mode. Resolved once per process and
    cached.
    """
    global _git_sha_cache, _git_sha_resolved
    if _git_sha_resolved:
        return _git_sha_cache
    _git_sha_resolved = True
    try:
        import subprocess

        # parents[0]=audit, [1]=standard_quant_tools, [2]=src, [3]=repo root.
        repo_root = Path(__file__).resolve().parents[3]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            _git_sha_cache = result.stdout.strip() or None
    except Exception:
        _git_sha_cache = None
    return _git_sha_cache


def _package_version() -> Optional[str]:
    try:
        from standard_quant_tools import __version__

        return __version__
    except Exception:
        return None


def _strategy_source_hash(model_instance: Any) -> Optional[str]:
    """
    Content hash of a registered strategy's source code, when
    `model_instance` names one via a `strategy` or `strategy_type` field
    (e.g. WalkForwardInput.strategy, BacktestDiagnosticsInput.strategy_type).
    None when neither field is present, the name isn't a registered
    strategy (e.g. a custom-signal tool), or on any lookup failure.
    """
    strategy_name = getattr(model_instance, "strategy", None) or getattr(
        model_instance, "strategy_type", None
    )
    if not strategy_name:
        return None
    try:
        import inspect

        from standard_quant_tools.backtest.strategies import STRATEGY_REGISTRY

        fn = STRATEGY_REGISTRY.get(strategy_name)
        if fn is None:
            return None
        return hash_payload(inspect.getsource(fn))
    except Exception:
        return None
