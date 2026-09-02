"""
The native extension must be PRESENT where it is expected to be present.

THE GAP THIS CLOSES. 266 tests are decorated `@requires_cpp`, which is
`pytest.mark.skipif(not HAS_CPP)`. That is correct for a contributor with no
C++ toolchain: the library works through its NumPy fallbacks and they should
not be blocked. It is wrong for CI, where a failed or missing build turned
266 tests into skips and the run still reported green.

Three things had to line up for that to be invisible, and all three were:

  - `CMakeLists.txt` downgrades a missing compiler to a WARNING and installs
    the pure-Python package. Right by default, and `SQT_REQUIRE_NATIVE=ON`
    exists to override it -- `pyproject.toml` has described that flag as
    what CI uses since the native layer was added.
  - `ci.yml` never passed it, and never checked the extension imported.
    `build-cpp.yml` did check, but that workflow builds wheels; the one that
    runs the test suite did not.
  - Nothing in the suite itself objected to running without the extension.

The workflow is fixed. This file is the part that survives someone editing
the workflow: with `SQT_EXPECT_NATIVE` set, a missing extension is a
FAILURE, not a skip.

It is deliberately gated on an environment variable rather than always
asserting. A hard requirement would break every contributor without a
compiler, which is the situation the skip markers exist to support. The
point is not that the extension must always be there -- it is that when
something has declared it should be, silence is not an acceptable answer.
"""

from __future__ import annotations

import os

import pytest

EXPECTED = os.environ.get("SQT_EXPECT_NATIVE", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)


def test_the_extension_imports_when_it_is_expected():
    """The check `ci.yml` runs as its own step, repeated here so it also
    holds for anyone who sets SQT_EXPECT_NATIVE locally."""
    if not EXPECTED:
        pytest.skip("SQT_EXPECT_NATIVE is not set; the fallback path is fine")

    from standard_quant_tools import _sqt_core

    assert _sqt_core is not None


def test_has_cpp_is_true_when_the_extension_is_expected():
    """`HAS_CPP` is what the 266 skip markers read. The extension importing
    is not sufficient -- a probe that failed for any other reason leaves
    this False and every native test skips anyway."""
    if not EXPECTED:
        pytest.skip("SQT_EXPECT_NATIVE is not set; the fallback path is fine")

    from standard_quant_tools.agent.runtimes._shared import HAS_CPP

    assert HAS_CPP is True, (
        "SQT_EXPECT_NATIVE is set but HAS_CPP is False, so every "
        "@requires_cpp test is skipping. That is the exact failure this "
        "file exists to make loud: a green run that tested none of the "
        "native paths."
    )


def test_every_module_that_probes_agrees_with_the_others():
    """`HAS_CPP` is probed independently in ~20 modules. They must agree:
    a module whose probe silently failed routes to its fallback while the
    rest go native, and no test compares them."""
    import importlib

    modules = [
        "standard_quant_tools.agent.runtimes._shared",
        "standard_quant_tools.analysis.cointegration",
        "standard_quant_tools.backtest.engine",
        "standard_quant_tools.indicators.panel",
    ]
    flags = {}
    for name in modules:
        module = importlib.import_module(name)
        if hasattr(module, "HAS_CPP"):
            flags[name] = bool(module.HAS_CPP)

    assert flags, "no module exposed HAS_CPP; the probe pattern moved"
    assert len(set(flags.values())) == 1, (
        f"modules disagree about whether the extension is available: {flags}. "
        "One of them routes to a fallback while the others go native."
    )


def test_every_probing_module_imports_without_the_extension():
    """The invariant, tested by BEHAVIOUR rather than by pattern-matching.

    Every probe site must survive the extension being absent: the module
    still imports, `HAS_CPP` is False, and the core handle is None rather
    than undefined. An ImportError that leaves the handle unbound makes the
    module raise NameError at first use instead of taking its fallback --
    which was a real bug here once, fixed by moving the initializers above
    the `try`.

    My first version of this test scanned the source for "HAS_CPP assigned
    inside a try" and flagged three modules that are correct: assigning it
    in BOTH the try and the except is fine, and the thing that must precede
    the try is the core handle, not the flag. Running the import with the
    extension blocked tests what actually matters and cannot be fooled by
    the shape of the code.
    """
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent("""
        import builtins, importlib, sys

        _real = builtins.__import__

        def _blocked(name, globals=None, locals=None, fromlist=(), level=0):
            if name.endswith("_sqt_core") or "_sqt_core" in (fromlist or ()):
                raise ImportError("extension blocked for this check")
            return _real(name, globals, locals, fromlist, level)

        builtins.__import__ = _blocked

        modules = [
            "standard_quant_tools.agent.runtimes._shared",
            "standard_quant_tools.analysis.cointegration",
            "standard_quant_tools.analysis.hurst",
            "standard_quant_tools.backtest.engine",
            "standard_quant_tools.indicators.panel",
            "standard_quant_tools.indicators.momentum",
            "standard_quant_tools.indicators.trend",
        ]
        problems = []
        for name in modules:
            try:
                module = importlib.import_module(name)
            except Exception as exc:
                problems.append(f"{name}: import failed: {type(exc).__name__}: {exc}")
                continue
            flag = getattr(module, "HAS_CPP", "MISSING")
            if flag is not False:
                problems.append(f"{name}: HAS_CPP is {flag!r}, expected False")
        print("PROBLEMS:" + "|".join(problems))
        """)
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = [ln for ln in result.stdout.splitlines() if ln.startswith("PROBLEMS:")]
    assert line, result.stdout[-2000:] + result.stderr[-2000:]
    problems = [p for p in line[0][len("PROBLEMS:") :].split("|") if p]
    assert problems == [], problems


@pytest.mark.skipif(not EXPECTED, reason="only meaningful when native is required")
def test_the_native_gated_tests_actually_ran(pytestconfig):
    """The count itself, so a silent collapse is visible.

    If the extension is expected and present, essentially none of the
    `@requires_cpp` tests should be skipping. This does not enumerate them
    -- it asserts the condition they all read.
    """
    from standard_quant_tools.agent.runtimes._shared import HAS_CPP

    assert HAS_CPP, "the 266 @requires_cpp tests are all skipping"
