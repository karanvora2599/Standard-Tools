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

The same holds for the rules that make clang-cl a correct build and keep
every compiler's arithmetic the same, each found by building with it:

  - clang-cl fuses a*b+c into one rounding even under /fp:precise (27 of 44
    sampled kernel outputs moved, 41 tests failed), and drops a plain
    -ffp-contract=off without a word; only /clang:-ffp-contract=off reaches
    it. The no-contraction rule covered two files and now covers every
    unit, with a pragma that survives a global /fp:contract;
  - CMake sets MSVC for clang-cl, so its LLVM OpenMP runtime was stamped
    `vcomp`, and FindOpenMP linked the MSVC toolset's libomp140 import
    library rather than LLVM's own;
  - the C++ suite compiled the shipped sources at baseline ISA with no LTO,
    whatever the extension was built with.
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


NATIVE = CMAKE.parent
REPO = NATIVE.parents[2]
TESTS_CMAKE = REPO / "tests" / "cpp" / "CMakeLists.txt"


def _function(text: str, name: str) -> str:
    start = text.index(f"function({name} ")
    return text[start : text.index("endfunction()", start)]


@pytest.fixture(scope="module")
def suite_rules() -> str:
    if not TESTS_CMAKE.is_file():
        pytest.skip("the C++ suite is not beside this package")
    text = TESTS_CMAKE.read_text(encoding="utf-8")
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


class TestNoContractionInAnyUnit:
    def test_every_unit_includes_the_guard_before_anything_else(self):
        """The pragma governs only what follows it, and the kernels' own
        headers define inline arithmetic, so it comes first."""
        units = sorted((NATIVE / "src").glob("*.cpp"))
        units.append(NATIVE / "bindings" / "bindings.cpp")
        assert len(units) >= 16
        for path in units:
            includes = re.findall(
                r'^\s*#\s*include\s+[<"]([^>"]+)[>"]',
                path.read_text(encoding="utf-8"),
                re.M,
            )
            assert includes[0] == "sqt/fp_contract.hpp", (path.name, includes[:2])

    def test_the_guard_turns_contraction_off_for_clang_and_cl(self):
        text = (NATIVE / "include" / "sqt" / "fp_contract.hpp").read_text(
            encoding="utf-8"
        )
        code = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
        assert re.search(
            r"#if defined\(__clang__\)\s+#pragma STDC FP_CONTRACT OFF\s+"
            r"#elif defined\(_MSC_VER\)\s+#pragma fp_contract\(off\)",
            code,
        ), code

    def test_every_compiler_gets_its_own_spelling_of_the_flag(self, rules):
        """A plain -ffp-contract=off is silently dropped by the clang-cl
        driver; the /clang: prefix is what passes it through."""
        body = _function(rules, "sqt_native_codegen")
        clang_cl = _block(body, "if(SQT_CLANG_CL)")
        assert '"/clang:-ffp-contract=off"' in clang_cl
        cl = body[body.index("elseif(MSVC)") :]
        assert "/fp:precise" in cl[: cl.index("else()")]
        assert "-ffp-contract=off" in cl[cl.index("else()") :]
        assert "sqt_native_codegen(_sqt_core)" in rules

    def test_the_two_file_rule_it_replaced_is_gone(self, rules):
        assert "src/options.cpp src/cusum.cpp" not in rules


class TestTheCppSuiteCompilesWhatShips:
    def test_every_target_takes_the_extensions_codegen(self, suite_rules):
        """Libraries and executables are made in exactly one place each, and
        both places apply sqt_native_codegen through sqt_test_target."""
        assert suite_rules.count("add_library(") == 1
        assert suite_rules.count("add_executable(") == 1
        assert "sqt_test_target(${name})" in _function(
            suite_rules, "sqt_kernel_library"
        )
        assert "sqt_test_target(${name})" in _function(suite_rules, "sqt_cpp_test")
        assert "sqt_native_codegen(${target})" in _function(
            suite_rules, "sqt_test_target"
        )

    def test_the_codegen_carries_the_arch_flag_and_lto(self, rules):
        body = _function(rules, "sqt_native_codegen")
        assert "if(SQT_NATIVE_ARCH)" in body
        assert "/arch:AVX2" in body and "${SQT_GNU_ARCH_FLAG}" in body
        assert "INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE" in body
        assert re.search(
            r'set\(SQT_LTO_RELEASE "\$\{SQT_IPO_SUPPORTED\}" CACHE INTERNAL', rules
        )

    def test_gcc_and_clang_target_x86_64_v3_not_the_build_host(self, rules):
        assert 'check_cxx_compiler_flag("-march=x86-64-v3"' in rules
        assert 'set(SQT_GNU_ARCH_FLAG "-march=x86-64-v3")' in rules
        # -march=native survives only as the default off x86.
        assert rules.count("-march=native") == 1


class TestClangClIsACorrectBuild:
    def test_it_is_told_apart_from_cl(self, rules):
        assert re.search(
            r'if\(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND '
            r'CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"\)',
            rules,
        )

    def test_it_links_llvms_openmp_import_library(self, rules):
        preset = _block(
            rules, "if(SQT_CLANG_CL)\n    get_filename_component(_sqt_llvm_bin"
        )
        assert '"${_sqt_llvm_root}/lib/libomp.lib"' in preset
        assert "set(OpenMP_libomp_LIBRARY" in preset
        assert rules.index("get_filename_component(_sqt_llvm_bin") < rules.index(
            "find_package(OpenMP)"
        )

    def test_its_runtime_is_copied_beside_the_extension_and_installed(self, rules):
        assert re.search(
            r"POST_BUILD\s+COMMAND\s+\"\$\{CMAKE_COMMAND\}\"\s+-E\s+copy_if_different"
            r"\s+\"\$\{SQT_LIBOMP_DLL\}\"\s+\"\$<TARGET_FILE_DIR:_sqt_core>\"",
            rules,
        )
        assert (
            'install(FILES "${SQT_LIBOMP_DLL}" DESTINATION standard_quant_tools)'
            in rules
        )
        assert re.search(
            r"-E\s+rm\s+-f\s+\"\$<TARGET_FILE_DIR:_sqt_core>/libomp\.dll\"", rules
        )

    def test_the_stamp_asks_about_clang_cl_before_cl(self, rules):
        assert re.search(
            r'if\(SQT_CLANG_CL\)\s+set\(_sqt_openmp_runtime "\$\{_sqt_clang_cl_runtime\}"\)'
            r"\s+elseif\(MSVC\)",
            rules,
        )

    def test_cl_only_options_stay_cl_only(self, rules):
        assert "if(MSVC AND NOT SQT_CLANG_CL)\n    if(SQT_OPENMP_LLVM)" in rules
        pgo = _block(rules, "if(SQT_CLANG_CL AND (SQT_PGO_GENERATE OR SQT_PGO_USE))")
        assert "FATAL_ERROR" in pgo

    def test_git_ignores_the_copied_runtime(self):
        ignore = REPO / ".gitignore"
        if not ignore.is_file():
            pytest.skip("not a checkout")
        assert "libomp.dll" in ignore.read_text(encoding="utf-8").splitlines()
