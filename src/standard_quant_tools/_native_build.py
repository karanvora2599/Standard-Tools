"""
Whether the compiled extension was built from the C++ sources beside it.

AN EXTENSION THAT LOADS IS NOT AN EXTENSION THAT MATCHES. `HAS_CPP` answers
"is `_sqt_core` importable", and that stays truthful in the one situation
where it most needs to say more: an editable install keeps its own compiled
copy in site-packages while its Python modules resolve to the source tree,
so every native fix made after that copy was built is absent while the
Python that calls it is current. The copy is the right extension, just an
old one; it loads, reports itself available, and answers with old code. A
gapped series through such a copy came back entirely NaN where the current
kernels step over the gap, and nothing said why.

So the extension carries a SHA-256 of the native tree it was compiled from
(stamped at build time, see `_cpp/cmake/source_digest.cmake`), and the
package recomputes the same digest over the sources beside it, once, when it
is imported:

- the two agree: the extension is used (`match`);
- no sources beside the package, as in a wheel built without them: nothing
  to compare against, so the extension is trusted (`unchecked`). This
  project's own wheels carry the native tree, so they verify as `match`;
- they differ, or the extension predates the stamp: it is NOT used. Every
  kernel takes the Python path it already has, which matches the sources,
  so results are correct if slower -- and a `NativeBuildWarning` names the
  file that was refused and the command that refreshes it (`stale`,
  `unstamped`);
- the sources could not be read: the check cannot compare, so it says so and
  leaves the extension in use -- a fault in the checker is not evidence that
  the build is old (`unverified`).

When the extension cannot be imported at all, a binary built for a different
CPython ABI sitting in the package directory is reported by both tags
(`abi-mismatch`) instead of being invisible, and one built for this
interpreter that still fails to load is reported with its error
(`unloadable`). No extension anywhere is the ordinary pure-Python install
(`absent`), and `SQT_DISABLE_NATIVE` keeps its meaning (`disabled`).

The check runs once per process, hashes a few dozen files, and never raises:
a failure inside it becomes a verdict, not an import error.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

#: The compiled module's name inside the package.
EXTENSION = "_sqt_core"

#: The native tree, relative to the package directory.
NATIVE_TREE = "_cpp"

#: Which files the digest covers. The same selection as the CMake recipe,
#: matched case-sensitively in both, so both languages pick the same files on
#: every filesystem. A test runs the two on one tree and compares.
_SOURCE_SUFFIXES = (
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".hpp",
    ".hh",
    ".hxx",
    ".h",
    ".inl",
    ".ipp",
    ".cmake",
)
_SOURCE_NAMES = ("CMakeLists.txt",)

#: What a compiled extension file ends in, on any platform CPython supports.
_BINARY_SUFFIXES = (".pyd", ".so")

MATCH = "match"
UNCHECKED = "unchecked"
STALE = "stale"
UNSTAMPED = "unstamped"
UNVERIFIED = "unverified"
ABI_MISMATCH = "abi-mismatch"
UNLOADABLE = "unloadable"
ABSENT = "absent"
DISABLED = "disabled"

#: Verdicts that mean something is wrong with the build in front of this
#: interpreter, and so warn. `absent` and `disabled` are configurations.
WARNED_VERDICTS = frozenset({STALE, UNSTAMPED, UNVERIFIED, ABI_MISMATCH, UNLOADABLE})


class NativeBuildWarning(RuntimeWarning):
    """The compiled extension does not match, or cannot serve, this package.

    Raised once per process at import. The message names the extension file,
    what is wrong with it, and the command that refreshes it; the package has
    already fallen back to its Python paths where the extension could give
    wrong answers, so the warning reports a slowdown, not an error.
    """


class NativeBuildStatus:
    """The verdict on the compiled extension, reached once at import."""

    __slots__ = (
        "verdict",
        "used",
        "extension_file",
        "built_digest",
        "source_digest",
        "source_files",
        "build",
        "interpreter_tag",
        "foreign_binaries",
        "detail",
        "check_ms",
    )

    def __init__(
        self,
        verdict: str,
        used: bool = False,
        extension_file: Optional[str] = None,
        built_digest: Optional[str] = None,
        source_digest: Optional[str] = None,
        source_files: int = 0,
        build: Optional[Dict[str, Any]] = None,
        interpreter_tag: str = "",
        foreign_binaries: Tuple[Tuple[str, str], ...] = (),
        detail: Optional[str] = None,
        check_ms: float = 0.0,
    ) -> None:
        self.verdict = verdict
        self.used = used
        self.extension_file = extension_file
        self.built_digest = built_digest
        self.source_digest = source_digest
        self.source_files = source_files
        self.build = build or {}
        self.interpreter_tag = interpreter_tag
        self.foreign_binaries = foreign_binaries
        self.detail = detail
        self.check_ms = check_ms

    @property
    def label(self) -> str:
        """Verdict and short digest in one token, for a record that has room
        for one field: `match:df27c6e4af54` says which build ran, `stale:…`
        which build was refused, `absent` that there was none."""
        if self.built_digest:
            return f"{self.verdict}:{self.built_digest[:12]}"
        return self.verdict

    def as_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "used": self.used,
            "label": self.label,
            "extension_file": self.extension_file,
            "built_digest": self.built_digest,
            "source_digest": self.source_digest,
            "source_files": self.source_files,
            "build": dict(self.build) or None,
            "interpreter_tag": self.interpreter_tag,
            "foreign_binaries": [
                {"file": path, "tag": tag} for path, tag in self.foreign_binaries
            ],
            "check_ms": round(self.check_ms, 3),
            "detail": self.detail,
        }

    def __repr__(self) -> str:
        return f"NativeBuildStatus({self.label!r}, used={self.used})"


_status: Optional[NativeBuildStatus] = None
_warned: set = set()

#: The directory listing the ABI scan reads. A module attribute so a test can
#: simulate a package directory without building a second interpreter's
#: extension.
_listdir = os.listdir


# ── The digest ───────────────────────────────────────────────────────────────


def _is_source(name: str) -> bool:
    return name in _SOURCE_NAMES or name.endswith(_SOURCE_SUFFIXES)


def _walk_error(exc: OSError) -> None:
    # os.walk skips an unreadable directory silently by default, which would
    # make an unreadable tree hash as a smaller one -- a wrong answer rather
    # than "cannot compare".
    raise exc


def native_sources(native_dir: str) -> List[str]:
    """Every file the digest covers, relative to `native_dir`, '/'-separated
    and sorted bytewise -- the order the CMake recipe sorts in."""
    found: List[str] = []
    for root, _dirs, files in os.walk(native_dir, onerror=_walk_error):
        for name in files:
            if _is_source(name):
                rel = os.path.relpath(os.path.join(root, name), native_dir)
                found.append(rel.replace(os.sep, "/"))
    found.sort(key=lambda path: path.encode("utf-8"))
    return found


def source_digest(native_dir: str) -> Tuple[str, int]:
    """
    The SHA-256 of the native tree, and how many files it covers.

    The recipe `_cpp/cmake/source_digest.cmake` stamps into the extension:
    one `"<sha256 of the file's bytes>  <relative path>\\n"` line per file,
    in sorted path order, hashed together. Raises OSError when a file or
    directory cannot be read; the caller decides what that means.

    CRLF is read as LF before hashing, on both sides. The digest answers
    "was this built from this code", and a checkout that only rewrote line
    endings -- git normalising on checkout, an editor saving with the
    platform's convention -- is the same code; hashing raw bytes called a
    correct build stale and sent every kernel to its slower Python path.
    """
    manifest = hashlib.sha256()
    files = native_sources(native_dir)
    for rel in files:
        with open(os.path.join(native_dir, *rel.split("/")), "rb") as handle:
            content = handle.read().replace(b"\r\n", b"\n")
        file_hash = hashlib.sha256(content).hexdigest()
        manifest.update(f"{file_hash}  {rel}\n".encode("utf-8"))
    return manifest.hexdigest(), len(files)


# ── ABI tags ─────────────────────────────────────────────────────────────────


def _tag_of(suffix: str) -> str:
    """`.cp311-win_amd64.pyd` -> `cp311-win_amd64`; an untagged `.pyd` or
    `.so` has no tag and reads as "untagged"."""
    stem = suffix
    for ending in _BINARY_SUFFIXES:
        if stem.endswith(ending):
            stem = stem[: -len(ending)]
            break
    return stem.strip(".") or "untagged"


def interpreter_tag() -> str:
    """The ABI tag this interpreter's extensions carry, e.g. `cp312-win_amd64`
    or `cpython-312-x86_64-linux-gnu`."""
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    return _tag_of(suffixes[0]) if suffixes else "unknown"


def _extension_binaries(directories: List[str]) -> List[Tuple[str, str]]:
    """Every `_sqt_core.*` binary in the package's directories, as
    `(path, suffix)`. A directory that cannot be listed is skipped: this
    scan only ever adds an explanation, never a failure."""
    prefix = EXTENSION + "."
    found: List[Tuple[str, str]] = []
    for directory in directories:
        try:
            names = _listdir(directory)
        except OSError:
            continue
        for name in sorted(names):
            if name.startswith(prefix) and name.endswith(_BINARY_SUFFIXES):
                found.append((os.path.join(directory, name), name[len(EXTENSION) :]))
    return found


# ── How to refresh ───────────────────────────────────────────────────────────


def _repo_root(package_dir: str) -> Optional[str]:
    """The source checkout this package directory lives in, when it is one:
    `<repo>/src/<package>` with the CMake project at `<repo>`."""
    src = os.path.dirname(package_dir)
    repo = os.path.dirname(src)
    if os.path.basename(src) == "src" and os.path.isfile(
        os.path.join(repo, "CMakeLists.txt")
    ):
        return repo
    return None


def _same_dir(path: Optional[str], directory: str) -> bool:
    if not path:
        return False
    here = os.path.normcase(os.path.dirname(os.path.abspath(path)))
    return here == os.path.normcase(os.path.abspath(directory))


def refresh_instructions(extension_file: Optional[str], package_dir: str) -> str:
    """The exact command that replaces the extension this interpreter loads
    with one built from the sources beside the package."""
    python = sys.executable or "python"
    repo = _repo_root(package_dir)
    if repo is None:
        return (
            "Reinstall the package into this environment: "
            f'"{python}" -m pip install --force-reinstall standard_quant_tools'
        )
    pip = f'"{python}" -m pip install -e "{repo}"'
    tree = os.path.join(repo, f"build{sys.version_info[0]}{sys.version_info[1]}")
    cmake = (
        f'cmake -S "{repo}" -B "{tree}" -DCMAKE_BUILD_TYPE=Release '
        f'-DPython3_EXECUTABLE="{python}" && cmake --build "{tree}" --config Release'
    )
    if extension_file and not _same_dir(extension_file, package_dir):
        # The split install: Python from the checkout, the binary from this
        # environment's site-packages. Rebuilding in the checkout writes to
        # the source tree, which that copy shadows, so only a reinstall into
        # this environment replaces it.
        return (
            "This environment keeps its own compiled copy of the package (an "
            "editable install), which a build in the checkout does not "
            f"replace. Re-run the editable install in this environment: {pip}"
        )
    return (
        f"To build it for this interpreter in the checkout: {cmake} -- or, if "
        "this environment installed the package editable, re-run that install "
        f"here: {pip}"
    )


# ── The screen ───────────────────────────────────────────────────────────────


def _block(package: Any, name: str, module: Any) -> None:
    """Make the extension unimportable for the rest of the process, the same
    way SQT_DISABLE_NATIVE does, so every module that probes it takes the
    `except ImportError` branch it already has. The package attribute goes
    too: `from package import _sqt_core` reads it before sys.modules."""
    sys.modules[name] = None  # type: ignore[assignment]
    if getattr(package, "__dict__", {}).get(EXTENSION) is module:
        delattr(package, EXTENSION)


def _build_facts(module: Any) -> Dict[str, Any]:
    info = getattr(module, "__build_info__", None)
    if info is None:
        return {}
    try:
        return dict(info)
    except (TypeError, ValueError):
        return {}


def _unimportable(
    exc: BaseException, directories: List[str], package_dir: str, tag: str
) -> NativeBuildStatus:
    binaries = _extension_binaries(directories)
    own_suffixes = set(importlib.machinery.EXTENSION_SUFFIXES)
    own = [path for path, suffix in binaries if suffix in own_suffixes]
    foreign = tuple(
        (path, _tag_of(suffix))
        for path, suffix in binaries
        if suffix not in own_suffixes
    )
    if own:
        return NativeBuildStatus(
            UNLOADABLE,
            extension_file=own[0],
            interpreter_tag=tag,
            foreign_binaries=foreign,
            detail=(
                f"standard_quant_tools: the compiled extension {own[0]} is "
                f"built for this interpreter ({tag}) but could not be loaded "
                f"({type(exc).__name__}: {exc}). Every kernel is running its "
                "Python path, which is correct but slower. "
                + refresh_instructions(own[0], package_dir)
            ),
        )
    if foreign:
        listed = "; ".join(f"{path} is built for {other}" for path, other in foreign)
        return NativeBuildStatus(
            ABI_MISMATCH,
            interpreter_tag=tag,
            foreign_binaries=foreign,
            detail=(
                "standard_quant_tools: no compiled extension for this "
                f"interpreter ({tag}), but {listed}, which this interpreter "
                "cannot load. Every kernel is running its Python path, which "
                "is correct but slower. " + refresh_instructions(None, package_dir)
            ),
        )
    return NativeBuildStatus(ABSENT, interpreter_tag=tag)


def _screen(package: Any, disabled: bool) -> NativeBuildStatus:
    name = f"{package.__name__}.{EXTENSION}"
    package_dir = os.path.dirname(os.path.abspath(package.__file__))
    directories = [str(entry) for entry in getattr(package, "__path__", [package_dir])]
    tag = interpreter_tag()

    if disabled:
        return NativeBuildStatus(DISABLED, interpreter_tag=tag)

    # Refused earlier in this process and the package is being imported
    # again (a reload): the refusal stands. Re-screening would find the
    # module blocked and misreport a refused build as one that cannot load.
    if (
        name in sys.modules
        and sys.modules[name] is None
        and _status is not None
        and _status.verdict in (STALE, UNSTAMPED)
    ):
        return _status

    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        return _unimportable(exc, directories, package_dir, tag)

    extension_file = getattr(module, "__file__", None)
    facts = _build_facts(module)
    built = facts.get("source_digest")
    built = built if isinstance(built, str) and built else None

    def status(verdict: str, used: bool, **extra: Any) -> NativeBuildStatus:
        return NativeBuildStatus(
            verdict,
            used=used,
            extension_file=extension_file,
            built_digest=built,
            build=facts,
            interpreter_tag=tag,
            **extra,
        )

    native_dir = os.path.join(package_dir, NATIVE_TREE)
    if not os.path.isdir(native_dir):
        return status(UNCHECKED, True)
    try:
        digest, n_files = source_digest(native_dir)
    except OSError as exc:
        return status(
            UNVERIFIED,
            True,
            detail=(
                f"standard_quant_tools: could not read the C++ sources in "
                f"{native_dir} to check the compiled extension {extension_file} "
                f"against them ({type(exc).__name__}: {exc}). It is in use "
                "UNVERIFIED: if it was built from older sources, its answers "
                "are the older code's."
            ),
        )
    if n_files == 0:
        return status(UNCHECKED, True)

    if built is None:
        _block(package, name, module)
        return status(
            UNSTAMPED,
            False,
            source_digest=digest,
            source_files=n_files,
            detail=(
                f"standard_quant_tools: the compiled extension {extension_file} "
                "was built before extensions recorded their sources, so "
                f"nothing says it matches the C++ sources beside it in "
                f"{native_dir}. It is NOT being used: every kernel is running "
                "its Python path, which matches the sources, so results are "
                "correct but slower. "
                + refresh_instructions(extension_file, package_dir)
            ),
        )
    if built != digest:
        _block(package, name, module)
        return status(
            STALE,
            False,
            source_digest=digest,
            source_files=n_files,
            detail=(
                f"standard_quant_tools: the compiled extension {extension_file} "
                f"does not match the C++ sources beside it in {native_dir}: it "
                f"was built from sources with digest {built[:12]}, and the "
                f"sources there now hash to {digest[:12]}. It is NOT being "
                "used: every kernel is running its Python path, which matches "
                "the sources, so results are correct but slower. "
                + refresh_instructions(extension_file, package_dir)
            ),
        )
    return status(MATCH, True, source_digest=digest, source_files=n_files)


def _warn_once(status: NativeBuildStatus) -> None:
    if status.verdict not in WARNED_VERDICTS or not status.detail:
        return
    if status.verdict in _warned:
        return
    _warned.add(status.verdict)
    try:
        warnings.warn(status.detail, NativeBuildWarning, stacklevel=3)
    except Exception:  # noqa: BLE001 - an escalated warning must not fail the import
        # `-W error` turns the warning into an exception inside `import
        # standard_quant_tools`. The package has already taken the safe
        # path; making it unimportable over a report would be worse than
        # the problem reported, so the message goes to stderr instead.
        try:
            sys.stderr.write(f"NativeBuildWarning: {status.detail}\n")
        except Exception:  # noqa: BLE001 - nowhere left to report it
            pass


def screen_extension(package: Any, disabled: bool = False) -> NativeBuildStatus:
    """
    Decide, once, whether this process uses the compiled extension.

    Called by the package's `__init__` before any module probes
    `_sqt_core`, so a refused extension is refused for all of them at once.
    Records the verdict for `native_build_status()` and warns for the
    verdicts that need a rebuild. Never raises.
    """
    global _status
    started = time.perf_counter()
    try:
        result = _screen(package, disabled)
    except Exception as exc:  # noqa: BLE001 - the check must never break the import
        name = f"{getattr(package, '__name__', '?')}.{EXTENSION}"
        loaded = sys.modules.get(name)
        result = NativeBuildStatus(
            UNVERIFIED,
            used=loaded is not None,
            extension_file=getattr(loaded, "__file__", None),
            interpreter_tag=interpreter_tag(),
            detail=(
                "standard_quant_tools: the check of the compiled extension "
                f"against its sources failed ({type(exc).__name__}: {exc}), so "
                "whether it matches them is unknown."
            ),
        )
    if result is not _status:
        result.check_ms = (time.perf_counter() - started) * 1e3
    _status = result
    _warn_once(result)
    return result


def native_build_status() -> NativeBuildStatus:
    """
    The verdict on the compiled extension this process reached at import:
    which file was loaded, whether it was built from the sources beside the
    package, and the build facts it carries. Reported by
    `describe_effective_config`, `list_modeling_capabilities` and every
    decision record, so a result can say WHICH build computed it.
    """
    if _status is None:
        return NativeBuildStatus(
            UNVERIFIED,
            interpreter_tag=interpreter_tag(),
            detail="the package has not screened its extension in this process",
        )
    return _status
