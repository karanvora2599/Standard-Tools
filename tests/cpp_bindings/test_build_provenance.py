"""
The compiled extension must be the one built from the sources beside it.

An extension that imports is not an extension that matches. An editable
install keeps its own compiled copy in site-packages while its Python
resolves to the source tree, so every native fix made after that copy was
built is absent while the Python calling it is current. `HAS_CPP` stays
True through all of it -- it is the right extension, just an old one -- and
a copy built for another CPython ABI is not an error, it is simply not seen.
Measured through such a copy: RSI(5) over sixteen bars with one missing
returned sixteen NaN where the current kernels return six (the warm-up and
the missing bar), and 196 of this repository's native-backed tests failed.

So the extension carries a digest of its sources, stamped at build time, and
the package recomputes it at import and refuses a mismatch. This file pins:

  - the extension this suite runs against was built from this checkout.
    That test FAILS rather than skips: a refused extension makes every
    `@requires_cpp` test skip, and a suite that skips its way to green on a
    stale build is the failure this file exists to end;
  - the digest itself: stable, sensitive to a byte, blind to everything
    outside the native tree, and the same in Python as in the CMake step
    that stamps it;
  - the verdicts: a match is used silently, a mismatch or an unstamped
    build is refused with a warning that names the file and the command,
    an install with no sources is trusted, and a binary for another ABI is
    named instead of ignored;
  - the check never raises and never makes the package unimportable.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import types
import warnings
from pathlib import Path

import pytest

import standard_quant_tools
from standard_quant_tools import _native_build as nb
from standard_quant_tools._native_build import (
    NativeBuildWarning,
    native_build_status,
    source_digest,
)

PACKAGE_DIR = Path(standard_quant_tools.__file__).resolve().parent
NATIVE_DIR = PACKAGE_DIR / "_cpp"
RECIPE = NATIVE_DIR / "cmake" / "source_digest.cmake"


# ── This checkout's own extension ────────────────────────────────────────────


class TestTheExtensionInUseMatchesThisCheckout:
    def test_the_loaded_extension_was_built_from_these_sources(self):
        """The guard the rest of this directory leans on. A stale build is
        refused at import, which makes every native test SKIP -- so without
        this one failing, a stale build would read as a green run."""
        status = native_build_status()
        if status.verdict in (nb.ABSENT, nb.DISABLED):
            pytest.skip(
                f"no extension in use ({status.verdict}); "
                "tests/test_native_extension.py decides whether that is allowed"
            )
        assert status.verdict == nb.MATCH, status.detail
        assert status.used is True
        digest, n_files = source_digest(str(NATIVE_DIR))
        assert status.built_digest == digest, (
            f"the extension was built from sources {status.built_digest[:12]}, "
            f"this checkout's hash to {digest[:12]}. Rebuild it."
        )
        assert status.source_files == n_files

    def test_the_stamp_is_readable_and_read_only(self):
        _sqt_core = pytest.importorskip("standard_quant_tools._sqt_core")
        info = _sqt_core.__build_info__
        assert set(info) >= {
            "source_digest",
            "source_files",
            "build_type",
            "native_arch",
            "compiler",
            "openmp",
            "pgo",
        }
        assert re.fullmatch(r"[0-9a-f]{64}", info["source_digest"])
        assert info["source_files"] > 20
        assert info["compiler"]
        with pytest.raises(TypeError):
            info["source_digest"] = "0" * 64  # type: ignore[index]

    def test_the_stamp_carries_no_timestamp_and_no_path(self):
        """Two checkouts of one commit, anywhere, stamp the same bytes."""
        _sqt_core = pytest.importorskip("standard_quant_tools._sqt_core")
        text = json.dumps(dict(_sqt_core.__build_info__))
        assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
        assert str(PACKAGE_DIR.parent) not in text
        assert ":\\\\" not in text and '"/' not in text


# ── The digest ───────────────────────────────────────────────────────────────


def _tree(root: Path) -> Path:
    """A small native tree with one file of every kind the recipe covers,
    plus the kinds it must not."""
    native = root / "_cpp"
    files = {
        "CMakeLists.txt": "add_library(x a.cpp)\n",
        "src/a.cpp": "int a() { return 1; }\n",
        "src/b.cc": "int b();\n",
        "src/c.cxx": "int c();\n",
        "src/d.c": "int d;\n",
        "include/sqt/a.hpp": "#pragma once\n",
        "include/sqt/b.hh": "#pragma once\n",
        "include/sqt/c.hxx": "#pragma once\n",
        "include/sqt/d.h": "#pragma once\n",
        "include/sqt/e.inl": "// inline\n",
        "include/sqt/f.ipp": "// impl\n",
        "cmake/recipe.cmake": "set(X 1)\n",
        "sub/CMakeLists.txt": "# nested\n",
        "bindings/bindings.cpp": "// bindings\r\nint main() {}\r\n",
    }
    for rel, text in files.items():
        path = native / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))
    return native


class TestTheDigest:
    def test_it_is_stable_within_and_across_processes(self, tmp_path):
        native = _tree(tmp_path)
        first = source_digest(str(native))
        assert source_digest(str(native)) == first

        script = textwrap.dedent(f"""
            from standard_quant_tools._native_build import source_digest
            print(source_digest({str(native)!r})[0])
        """)
        env = {**os.environ, "SQT_DISABLE_NATIVE": "1"}
        done = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr[-1500:]
        assert done.stdout.strip() == first[0]

    def test_one_changed_byte_changes_it(self, tmp_path):
        native = _tree(tmp_path)
        before, count = source_digest(str(native))
        target = native / "src" / "a.cpp"
        original = target.read_bytes()
        target.write_bytes(original.replace(b"1", b"2"))
        assert source_digest(str(native)) != (before, count)
        target.write_bytes(original)
        assert source_digest(str(native)) == (before, count)

    def test_line_endings_alone_do_not_change_it(self, tmp_path):
        """A checkout that only rewrote line endings -- git normalising on
        checkout, an editor saving CRLF -- is the same code, so the build
        from it is not stale. Only CRLF pairs are folded: a lone carriage
        return is still a changed byte."""
        native = _tree(tmp_path)
        before = source_digest(str(native))
        for path in native.rglob("*"):
            if path.is_file():
                data = path.read_bytes().replace(b"\r\n", b"\n")
                path.write_bytes(data.replace(b"\n", b"\r\n"))
        assert source_digest(str(native)) == before
        for path in native.rglob("*"):
            if path.is_file():
                path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        assert source_digest(str(native)) == before

        target = native / "src" / "a.cpp"
        target.write_bytes(target.read_bytes().replace(b"\n", b"\r"))
        assert source_digest(str(native)) != before

    def test_a_renamed_or_added_file_changes_it(self, tmp_path):
        native = _tree(tmp_path)
        before, count = source_digest(str(native))
        (native / "src" / "a.cpp").rename(native / "src" / "z.cpp")
        assert source_digest(str(native))[0] != before

        native = _tree(tmp_path / "again")
        (native / "src" / "new.hpp").write_text("#pragma once\n", encoding="utf-8")
        after, grown = source_digest(str(native))
        assert after != before and grown == count + 1

    def test_files_outside_the_native_tree_are_ignored(self, tmp_path):
        """Null case: nothing but the native sources moves it -- not the
        Python beside it, not notes, caches or build products inside it."""
        native = _tree(tmp_path)
        before = source_digest(str(native))
        (tmp_path / "module.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "other.cpp").write_text("int z;\n", encoding="utf-8")
        (native / "README.md").write_text("notes\n", encoding="utf-8")
        (native / "src" / "a.cpp.orig").write_text("old\n", encoding="utf-8")
        (native / "src" / "a.obj").write_bytes(b"\x00\x01")
        (native / "__pycache__").mkdir()
        (native / "__pycache__" / "x.pyc").write_bytes(b"\x00")
        (native / "src" / "UPPER.CPP").write_text("int u;\n", encoding="utf-8")
        assert source_digest(str(native)) == before

    def test_it_covers_every_native_source_in_this_checkout(self):
        files = nb.native_sources(str(NATIVE_DIR))
        assert "CMakeLists.txt" in files
        assert "bindings/bindings.cpp" in files
        assert "cmake/source_digest.cmake" in files
        assert "src/build_info.cpp" in files
        assert files == sorted(files, key=lambda p: p.encode("utf-8"))
        on_disk = {
            p.relative_to(NATIVE_DIR).as_posix()
            for p in NATIVE_DIR.rglob("*")
            if p.is_file() and p.suffix in (".cpp", ".hpp")
        }
        assert on_disk <= set(files)

    @pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake")
    def test_python_and_the_cmake_stamp_agree_on_one_tree(self, tmp_path):
        """The two implementations of one recipe, run on the same files. If
        they ever disagree, every build reads as stale."""
        native = _tree(tmp_path)
        (native / "src" / "UPPER.CPP").write_text("int u;\n", encoding="utf-8")
        (native / "notes.txt").write_text("not a source\n", encoding="utf-8")
        header = tmp_path / "stamp.hpp"
        done = subprocess.run(
            [
                "cmake",
                f"-DSQT_NATIVE_DIR={native.as_posix()}",
                f"-DSQT_OUTPUT={header.as_posix()}",
                "-P",
                str(RECIPE),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr
        text = header.read_text(encoding="utf-8")
        digest = re.search(r'SQT_SOURCE_DIGEST "([0-9a-f]{64})"', text).group(1)
        count = int(re.search(r"SQT_SOURCE_FILES (\d+)", text).group(1))
        assert (digest, count) == source_digest(str(native))

    def test_an_unreadable_directory_raises_rather_than_hashing_less(
        self, tmp_path, monkeypatch
    ):
        """A silently skipped directory would hash as a smaller tree -- a
        wrong answer where the right one is "cannot compare"."""
        native = _tree(tmp_path)

        def refusing_walk(top, onerror=None, **kwargs):
            onerror(PermissionError(13, "denied", str(top)))
            return iter(())

        monkeypatch.setattr(nb.os, "walk", refusing_walk)
        with pytest.raises(PermissionError):
            source_digest(str(native))


# ── The verdicts, on a package built in a temporary directory ───────────────


class _FakePackage:
    """A package with a native tree, an extension of our choosing, and
    none of the real package's state, so a refusal here touches nothing
    else in the process."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch, name: str):
        (root / "CMakeLists.txt").write_text("project(x)\n", encoding="utf-8")
        self.dir = root / "src" / name
        self.dir.mkdir(parents=True)
        (self.dir / "__init__.py").write_text("", encoding="utf-8")
        self.native = _tree(self.dir)
        self.module = types.ModuleType(name)
        self.module.__file__ = str(self.dir / "__init__.py")
        self.module.__path__ = [str(self.dir)]  # type: ignore[attr-defined]
        self.ext_name = f"{name}.{nb.EXTENSION}"
        self.monkeypatch = monkeypatch
        monkeypatch.setitem(sys.modules, name, self.module)

    def install(self, where: Path, digest: "str | None") -> types.ModuleType:
        """Put an extension module in place, stamped with `digest`, or
        unstamped when it is None."""
        ext = types.ModuleType(self.ext_name)
        ext.__file__ = str(where / f"{nb.EXTENSION}.cp312-win_amd64.pyd")
        if digest is not None:
            ext.__build_info__ = types.MappingProxyType(  # type: ignore[attr-defined]
                {"source_digest": digest, "source_files": 1, "build_type": "Release"}
            )
        self.monkeypatch.setitem(sys.modules, self.ext_name, ext)
        setattr(self.module, nb.EXTENSION, ext)
        return ext

    def digest(self) -> str:
        return source_digest(str(self.native))[0]


@pytest.fixture
def fresh_state(monkeypatch):
    """The recorded verdict and the warned-once set, restored afterwards."""
    monkeypatch.setattr(nb, "_status", nb._status)
    monkeypatch.setattr(nb, "_warned", set())
    monkeypatch.delenv("SQT_DISABLE_NATIVE", raising=False)


@pytest.fixture
def pkg(tmp_path, monkeypatch, fresh_state, request):
    name = "sqt_provenance_probe_" + re.sub(r"\W", "_", request.node.name)[:40]
    return _FakePackage(tmp_path, monkeypatch, name)


def _screen(pkg: _FakePackage, disabled: bool = False):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        status = nb.screen_extension(pkg.module, disabled=disabled)
    ours = [w for w in caught if issubclass(w.category, NativeBuildWarning)]
    return status, [str(w.message) for w in ours]


class TestAMatchingBuildIsUsed:
    def test_it_is_used_and_says_nothing(self, pkg):
        ext = pkg.install(pkg.dir, pkg.digest())
        status, warned = _screen(pkg)
        assert status.verdict == nb.MATCH and status.used is True
        assert warned == []
        assert sys.modules[pkg.ext_name] is ext
        assert getattr(pkg.module, nb.EXTENSION) is ext
        assert status.label == f"match:{pkg.digest()[:12]}"
        assert nb.native_build_status() is status


class TestAStaleBuildIsRefused:
    def test_an_in_place_build_is_refused_with_the_rebuild_command(self, pkg):
        stale = "0" * 64
        ext = pkg.install(pkg.dir, stale)
        status, warned = _screen(pkg)

        assert status.verdict == nb.STALE and status.used is False
        assert status.built_digest == stale
        assert status.source_digest == pkg.digest()
        # Refused for every module at once, the way SQT_DISABLE_NATIVE is.
        assert sys.modules[pkg.ext_name] is None
        assert not hasattr(pkg.module, nb.EXTENSION)
        with pytest.raises(ImportError):
            importlib.import_module(pkg.ext_name)

        assert len(warned) == 1
        message = warned[0]
        assert ext.__file__ in message
        assert "does not match the C++ sources" in message
        assert "NOT being used" in message
        assert "cmake -S" in message and sys.executable in message
        assert "Python3_EXECUTABLE" in message

    def test_an_editable_installs_own_copy_is_told_to_reinstall(self, pkg, tmp_path):
        """The split install: Python from the checkout, the binary from
        site-packages. A build in the checkout would not replace it, so the
        command is the reinstall, run with this environment's interpreter."""
        site = tmp_path / "venv" / "site-packages" / "probe"
        site.mkdir(parents=True)
        ext = pkg.install(site, "f" * 64)
        status, warned = _screen(pkg)
        assert status.verdict == nb.STALE
        assert ext.__file__ in warned[0]
        repo = str(pkg.dir.parent.parent)
        assert f'-m pip install -e "{repo}"' in warned[0]
        assert "editable install" in warned[0]

    def test_an_unstamped_build_is_refused_when_sources_are_beside_it(self, pkg):
        """Built before extensions carried a stamp: nothing says it matches,
        and the sources are right there to say it might not."""
        pkg.install(pkg.dir, None)
        status, warned = _screen(pkg)
        assert status.verdict == nb.UNSTAMPED and status.used is False
        assert sys.modules[pkg.ext_name] is None
        assert "before extensions recorded their sources" in warned[0]

    def test_it_warns_once_per_process(self, pkg):
        pkg.install(pkg.dir, "0" * 64)
        _, first = _screen(pkg)
        pkg.install(pkg.dir, "0" * 64)
        _, second = _screen(pkg)
        assert len(first) == 1 and second == []

    def test_a_reload_after_a_refusal_keeps_the_refusal(self, pkg):
        """Importing the package again finds the module blocked by the first
        screen. That is the refusal standing, not a build that cannot load."""
        pkg.install(pkg.dir, "0" * 64)
        first, _ = _screen(pkg)
        again, warned = _screen(pkg)
        assert first.verdict == again.verdict == nb.STALE
        assert again is first and warned == []
        assert sys.modules[pkg.ext_name] is None


class TestNoSourcesMeansTrust:
    def test_an_install_with_no_native_tree_is_trusted(self, pkg):
        """A distribution built without the sources has nothing to compare
        against, and the binary came with the Python beside it."""
        shutil.rmtree(pkg.native)
        ext = pkg.install(pkg.dir, "0" * 64)
        status, warned = _screen(pkg)
        assert status.verdict == nb.UNCHECKED and status.used is True
        assert sys.modules[pkg.ext_name] is ext and warned == []

    def test_an_unstamped_build_with_no_native_tree_is_trusted_too(self, pkg):
        """Nothing to compare against, so nothing to refuse it over;
        refusing it would slow every such install for no reason."""
        shutil.rmtree(pkg.native)
        pkg.install(pkg.dir, None)
        status, warned = _screen(pkg)
        assert status.verdict == nb.UNCHECKED and status.used is True
        assert warned == []


class TestTheCheckNeverRaises:
    def test_unreadable_sources_are_reported_and_the_extension_kept(
        self, pkg, monkeypatch
    ):
        """A fault in the checker is not evidence the build is old."""
        ext = pkg.install(pkg.dir, "0" * 64)

        def unreadable(_directory):
            raise PermissionError(13, "denied")

        monkeypatch.setattr(nb, "source_digest", unreadable)
        status, warned = _screen(pkg)
        assert status.verdict == nb.UNVERIFIED and status.used is True
        assert sys.modules[pkg.ext_name] is ext
        assert "could not read the C++ sources" in warned[0]

    def test_a_failure_inside_the_check_becomes_a_verdict(self, pkg, monkeypatch):
        pkg.install(pkg.dir, pkg.digest())

        def broken(*_args, **_kwargs):
            raise RuntimeError("the checker itself broke")

        monkeypatch.setattr(nb, "_screen", broken)
        status, warned = _screen(pkg)
        assert status.verdict == nb.UNVERIFIED
        assert "the checker itself broke" in warned[0]

    def test_warnings_as_errors_does_not_make_the_package_unimportable(
        self, pkg, capsys
    ):
        pkg.install(pkg.dir, "0" * 64)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            status = nb.screen_extension(pkg.module)
        assert status.verdict == nb.STALE and status.used is False
        assert "NativeBuildWarning" in capsys.readouterr().err

    def test_disabled_is_a_configuration_not_a_warning(self, pkg):
        """SQT_DISABLE_NATIVE keeps its meaning: nothing is imported, nothing
        is checked, nothing is said."""
        status, warned = _screen(pkg, disabled=True)
        assert status.verdict == nb.DISABLED and status.used is False
        assert warned == []
        assert pkg.ext_name not in sys.modules


# ── An extension built for another interpreter ───────────────────────────────


def _binary_ending() -> str:
    return ".pyd" if sys.platform == "win32" else ".so"


class TestAnotherInterpretersBinaryIsNamed:
    def test_it_is_reported_with_both_tags_and_the_command(self, pkg, monkeypatch):
        foreign = f"{nb.EXTENSION}.cp29-testplatform{_binary_ending()}"
        monkeypatch.setattr(nb, "_listdir", lambda d: ["__init__.py", foreign])
        status, warned = _screen(pkg)

        assert status.verdict == nb.ABI_MISMATCH and status.used is False
        assert status.foreign_binaries == (
            (str(pkg.dir / foreign), "cp29-testplatform"),
        )
        assert len(warned) == 1
        message = warned[0]
        assert "cp29-testplatform" in message
        assert nb.interpreter_tag() in message
        assert str(pkg.dir / foreign) in message
        assert "Python3_EXECUTABLE" in message and sys.executable in message

    def test_no_binary_at_all_is_the_ordinary_pure_python_install(
        self, pkg, monkeypatch
    ):
        """Null case: a machine without a compiler is not warned at."""
        monkeypatch.setattr(nb, "_listdir", lambda d: ["__init__.py", "_cpp"])
        status, warned = _screen(pkg)
        assert status.verdict == nb.ABSENT and warned == []

    def test_build_products_beside_it_are_not_extensions(self, pkg, monkeypatch):
        names = [
            f"{nb.EXTENSION}.cp29-x.pdb",
            f"{nb.EXTENSION}.cp29-x.lib",
            f"{nb.EXTENSION}.exp",
        ]
        monkeypatch.setattr(nb, "_listdir", lambda d: names)
        status, warned = _screen(pkg)
        assert status.verdict == nb.ABSENT and warned == []

    def test_this_interpreters_binary_that_will_not_load_is_named(
        self, pkg, monkeypatch
    ):
        own = f"{nb.EXTENSION}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
        monkeypatch.setattr(nb, "_listdir", lambda d: [own])
        status, warned = _screen(pkg)
        assert status.verdict == nb.UNLOADABLE
        assert own in warned[0] and "could not be loaded" in warned[0]

    def test_the_tags_are_read_from_the_file_name(self):
        assert nb._tag_of(".cp311-win_amd64.pyd") == "cp311-win_amd64"
        assert nb._tag_of(".cpython-311-x86_64-linux-gnu.so") == (
            "cpython-311-x86_64-linux-gnu"
        )
        assert nb._tag_of(".pyd") == "untagged"
        assert nb.interpreter_tag() == nb._tag_of(
            importlib.machinery.EXTENSION_SUFFIXES[0]
        )


# ── Where native availability is reported ───────────────────────────────────


class TestTheCapabilityReportSaysWhichBuild:
    def test_it_carries_the_import_time_verdict(self):
        from standard_quant_tools.modeling.capabilities import _native_detail

        build = _native_detail()["build"]
        status = native_build_status()
        assert build["verdict"] == status.verdict
        assert build["label"] == status.label
        assert build["built_digest"] == status.built_digest

    def test_a_refused_build_explains_why_it_is_unavailable(self, monkeypatch):
        """Refused means unimportable, which the report used to show as a
        bare `available: false` -- the same answer as no build at all."""
        from standard_quant_tools.modeling.capabilities import _native_detail

        detail = "the compiled extension X.pyd does not match the C++ sources"
        monkeypatch.setattr(
            nb,
            "_status",
            nb.NativeBuildStatus(nb.STALE, built_digest="0" * 64, detail=detail),
        )
        monkeypatch.setitem(sys.modules, "standard_quant_tools._sqt_core", None)
        monkeypatch.delattr(standard_quant_tools, nb.EXTENSION, raising=False)
        report = _native_detail()
        assert report["path"] is None
        assert report["build"]["verdict"] == "stale"
        assert report["note"] == detail


# ── End to end, in a fresh interpreter ───────────────────────────────────────


_REFUSED = """
    import importlib.util, json, sys, warnings
    import numpy as np, pandas as pd

    # Load the real extension ahead of the package and give it a stamp from
    # other sources: what an old copy in site-packages looks like.
    name = "standard_quant_tools._sqt_core"
    spec = importlib.util.spec_from_file_location(name, {path!r})
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    ext.__build_info__ = {{**dict(ext.__build_info__), "source_digest": "0" * 64}}
    sys.modules[name] = ext

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import standard_quant_tools as sqt

    from standard_quant_tools.indicators import momentum, volatility
    from standard_quant_tools.backtest import engine
    from standard_quant_tools.audit.provenance import _cpp_available

    rsi = momentum.rsi(pd.Series(json.loads({close!r}), dtype=float), period=5)
    print(json.dumps({{
        "verdict": sqt.native_build_status().verdict,
        "warned": [str(w.message) for w in caught
                   if w.category.__name__ == "NativeBuildWarning"],
        "has_cpp": [momentum.HAS_CPP, volatility.HAS_CPP, engine.HAS_CPP],
        "cpp_available": _cpp_available(),
        "rsi": [None if np.isnan(v) else round(float(v), 9) for v in rsi],
    }}))
"""

#: Sixteen bars with one missing: the series an old build turned to all-NaN.
_GAPPED = [10, 11, 12, 11, 13, float("nan"), 14, 15, 14, 16, 17, 16, 18, 19, 18, 20]


class TestARefusedBuildFallsBackEverywhere:
    def test_every_module_takes_its_python_path_and_the_answer_is_right(self):
        """The whole chain in a fresh interpreter: the real extension with a
        foreign stamp is refused, every probing module falls back together,
        the audit provenance says no extension ran, and the Python path
        answers the gapped series exactly as the matching extension in this
        process does -- NaN for the warm-up and the missing bar only, where
        the old kernels returned sixteen."""
        import numpy as np
        import pandas as pd

        from standard_quant_tools.indicators import momentum

        _sqt_core = pytest.importorskip("standard_quant_tools._sqt_core")
        native = momentum.rsi(pd.Series(_GAPPED, dtype=float), period=5)
        expected = [None if np.isnan(v) else round(float(v), 9) for v in native]
        assert 0 < expected.count(None) < len(_GAPPED)

        env = {k: v for k, v in os.environ.items() if k != "SQT_DISABLE_NATIVE"}
        script = _REFUSED.format(path=_sqt_core.__file__, close=json.dumps(_GAPPED))
        done = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        got = json.loads(done.stdout.strip().splitlines()[-1])
        assert got["verdict"] == "stale"
        assert len(got["warned"]) == 1 and _sqt_core.__file__ in got["warned"][0]
        assert got["has_cpp"] == [False, False, False]
        assert got["cpp_available"] is False
        assert got["rsi"] == pytest.approx(expected, rel=1e-9)
