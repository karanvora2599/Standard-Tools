"""
The build rules the profile-guided and OpenMP-runtime options depend on.

Each rule here was found missing by running the workflow, not by reading
it, and each failed silently:

  - the instrumented extension imports pgort140.dll, which Python does not
    look for on PATH, so it would not load; the package fell back to Python
    and "training" exercised none of the instrumented code;
  - the counts an instrumented process writes land beside the extension,
    and the linker only folds in counts lying beside the profile, so the
    optimized build linked against an empty profile without a word;
  - FindOpenMP caches the flag it settles on, so switching SQT_OPENMP_LLVM
    in an existing tree kept the previous runtime while the build facts
    named the new one.

Exercising them takes MSVC and a full configure, so this pins the rules in
the build file itself; tests/cpp_bindings/test_build_provenance.py checks
the built extension's facts against the DLL it actually imports.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import standard_quant_tools

CMAKE = Path(standard_quant_tools.__file__).resolve().parent / "_cpp" / "CMakeLists.txt"


@pytest.fixture(scope="module")
def rules() -> str:
    text = CMAKE.read_text(encoding="utf-8")
    # Comments say why; the assertions are about what the file does.
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _block(text: str, opener: str) -> str:
    """The body of the first `if(...)`-style block starting at `opener`, up
    to its matching else()/elseif()/endif() at the same depth."""
    start = text.index(opener)
    depth = 0
    for match in re.finditer(r"\b(if|elseif|else|endif)\s*\(", text[start:]):
        word = match.group(1)
        if word == "if":
            depth += 1
        elif word == "endif":
            depth -= 1
            if depth == 0:
                return text[start : start + match.start()]
        elif depth == 1:
            return text[start : start + match.start()]
    raise AssertionError(f"unterminated block at {opener!r}")


class TestTheProfileGuidedBuild:
    def test_the_instrumented_build_puts_its_runtime_beside_the_extension(self, rules):
        generate = _block(rules, "if(SQT_PGO_GENERATE)\n        find_file")
        assert "pgort140.dll" in generate
        assert re.search(
            r"POST_BUILD\s+COMMAND\s+\"\$\{CMAKE_COMMAND\}\"\s+-E\s+copy_if_different"
            r"\s+\"\$\{SQT_PGORT_DLL\}\"\s+\"\$<TARGET_FILE_DIR:_sqt_core>\"",
            generate,
        ), generate
        assert "FATAL_ERROR" in generate  # no runtime found: refuse, not skip

    def test_a_build_that_is_not_instrumented_removes_that_runtime(self, rules):
        assert re.search(
            r"-E\s+rm\s+-f\s+\"\$<TARGET_FILE_DIR:_sqt_core>/pgort140\.dll\"", rules
        )

    def test_the_optimized_build_merges_the_counts_and_refuses_without_a_profile(
        self, rules
    ):
        use = _block(rules, "if(SQT_PGO_USE)\n            if(NOT EXISTS")
        assert re.search(
            r'if\(NOT EXISTS "\$\{_sqt_pgd\}"\)\s+message\(FATAL_ERROR', use
        )
        assert re.search(r"/merge\s+\"\$\{_sqt_pgc\}\"\s+\"\$\{_sqt_pgd\}\"", use)
        assert 'file(REMOVE "${_sqt_pgc}")' in use  # each count folded in once
        assert '"/USEPROFILE:PGD=${_sqt_pgd}"' in use

    def test_both_halves_name_the_same_profile(self, rules):
        assert '"/GENPROFILE:PGD=${_sqt_pgd}"' in rules
        assert '"/USEPROFILE:PGD=${_sqt_pgd}"' in rules
        assert re.search(r'set\(_sqt_pgd "\$\{CMAKE_CURRENT_BINARY_DIR\}/', rules)

    def test_the_counts_are_looked_for_where_the_extension_writes_them(self, rules):
        assert re.search(
            r'file\(GLOB _sqt_pgc_files "\$\{_pkg_dir\}/_sqt_core!\*\.pgc"\)', rules
        )
        assert re.search(r'RUNTIME_OUTPUT_DIRECTORY_RELEASE\s+"\$\{_pkg_dir\}"', rules)


class TestTheOpenMPRuntimeOption:
    def test_the_option_exists_and_is_off(self, rules):
        assert re.search(r"option\(SQT_OPENMP_LLVM\s+\"[^\"]+\"\s+OFF\)", rules)

    def test_switching_it_drops_the_flag_findopenmp_cached(self, rules):
        on = _block(rules, "if(SQT_OPENMP_LLVM)\n        if(MSVC_VERSION")
        assert re.search(r'set\(OpenMP_CXX_FLAGS "-openmp:llvm" CACHE STRING', on)
        off = rules[
            rules.index('elseif("${OpenMP_CXX_FLAGS}" MATCHES "openmp:llvm")') :
        ]
        off = off[: off.index("endif()")]
        for name in (
            "OpenMP_CXX_FLAGS",
            "OpenMP_CXX_LIB_NAMES",
            "OpenMP_CXX_SPEC_DATE",
        ):
            assert f"unset({name} CACHE)" in off
        # ...and before FindOpenMP runs, or the cached answer is what it reads.
        assert rules.index("option(SQT_OPENMP_LLVM") < rules.index(
            "find_package(OpenMP)"
        )

    def test_the_runtime_fact_comes_from_the_flag_not_the_version(self, rules):
        assert re.search(
            r'if\("\$\{OpenMP_CXX_FLAGS\}" MATCHES "openmp:llvm"\)\s+'
            r'set\(_sqt_openmp_runtime "libomp"\)',
            rules,
        )
        assert 'set(SQT_FACT_OPENMP_RUNTIME \\"${_sqt_openmp_runtime}\\")' in rules
