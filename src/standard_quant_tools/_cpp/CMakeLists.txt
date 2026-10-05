# ── Extension module: _sqt_core ───────────────────────────────────────────────
# Builds the pybind11 extension that exposes all C++ fast paths to Python.
# All source files compile into a single shared-library extension (_sqt_core).
#
# Called from the root CMakeLists.txt via add_subdirectory().
# pybind11 and Python3 are already found by the time this file is processed.

# ── Sources ───────────────────────────────────────────────────────────────────

set(SQT_SOURCES
    src/hurst.cpp
    src/indicators.cpp
    src/cointegration.cpp
    src/backtest.cpp
    src/rolling_regression.cpp
    src/monte_carlo.cpp
    src/garch.cpp
    src/signal_state_machines.cpp
    src/isa_dispatch.cpp
    src/rolling_beta_avx2.cpp
    src/panel_stats.cpp
    src/options.cpp
    src/cusum.cpp
    src/correlation.cpp
    src/order_events.cpp
    src/regimes.cpp
    src/build_info.cpp
    bindings/bindings.cpp
)

pybind11_add_module(_sqt_core MODULE ${SQT_SOURCES})

# The package directory the extension is written into (see the output
# directories at the end of this file).
set(_pkg_dir "${CMAKE_SOURCE_DIR}/src/standard_quant_tools")

# ── Which compiler: cl, clang-cl, or a GCC-style driver ─────────────────────
# clang-cl is Clang behind MSVC's command line, and CMake sets MSVC for it:
# every `if(MSVC)` below is true for both. That is right for the flags the
# two drivers share (/O2, /arch:AVX2, /W3) and wrong for three things that
# differ -- whether a*b+c is contracted, which OpenMP runtime is linked, and
# MSVC's profile-guided workflow -- so those test SQT_CLANG_CL. It is cached
# because tests/cpp, a sibling directory, makes the same decisions.
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(SQT_CLANG_CL ON CACHE INTERNAL "The compiler is clang-cl")
else()
    set(SQT_CLANG_CL OFF CACHE INTERNAL "The compiler is clang-cl")
endif()

# ── No floating-point contraction, in every translation unit ────────────────
# The kernels are held to the Python and pandas arithmetic they replace bit
# for bit: options.cpp result for result (a converged flag and an iteration
# count are only comparable if every price along the way is the same
# double), cusum.cpp and correlation.cpp operation for operation, and the
# rest by the bit-identity tests in tests/cpp_bindings. Fusing a*b+c into one
# rounding breaks all of that, and the compilers disagree about doing it:
#
#   - cl: /fp:precise does not contract (VS 2022 and later). Stated anyway,
#     so a /fp:fast earlier on the command line is overridden.
#   - clang-cl: contracts within an expression by default EVEN UNDER
#     /fp:precise. Measured: 27 of 44 sampled kernel outputs moved and 41
#     tests failed; with /clang:-ffp-contract=off all 44 matched cl's. A
#     plain -ffp-contract=off is silently dropped by the clang-cl driver --
#     the /clang: prefix is what passes it through.
#   - GCC and Clang: GCC contracts under -std=gnu++* and Clang within an
#     expression, both of them once -march allows FMA.
#
# This used to cover options.cpp and cusum.cpp only; every unit now gets it,
# through sqt_native_codegen() below, as does every library and test
# executable in tests/cpp. The flag is half of the guard. The other half is
# sqt/fp_contract.hpp, which every unit includes first: its pragma survives
# a global /fp:contract that a later /fp:precise does not undo (measured,
# cl 19.44). FMA instructions in the linked module come only from the
# intrinsics in rolling_beta_avx2.cpp.

# ── AVX2+FMA translation unit (item L: runtime ISA dispatch demo) ────────────
# rolling_beta_avx2.cpp uses AVX2/FMA intrinsics directly and must always be
# compiled with those instructions enabled, INDEPENDENT of SQT_NATIVE_ARCH
# (that flag is opt-in/off-by-default; this file's codegen must be on
# unconditionally for the intrinsics it uses to even compile) -- the actual
# runtime safety comes from isa_dispatch.cpp's CPUID check gating every call
# into this file, not from the compile flag. MSVC has no per-function
# ISA-target attribute (unlike GCC/Clang's __attribute__((target(...)))), so
# isolating these intrinsics into their own translation unit compiled with
# /arch:AVX2 is the only portable way to keep the rest of the module's
# codegen (which must stay safe on non-AVX2 CPUs when SQT_NATIVE_ARCH=OFF)
# unaffected.
#
# -mavx2/-mfma (and /arch:AVX2) are x86/x64-only flags -- passing them on a
# non-x86 toolchain (e.g. GCC/Clang targeting ARM/Apple Silicon) is a hard
# compile error, not a graceful no-op. Only apply them on x86/x64; on any
# other CMAKE_SYSTEM_PROCESSOR, rolling_beta_avx2.cpp's own source-level
# #if guard already compiles it down to a portable (unreachable) stub, so no
# special flags -- and no source-file exclusion -- are needed here.
#
# NOTE: a per-source compile flag confines CODEGEN to this file, not
# INLINING. LTO/IPO is enabled for Release below (and pybind11_add_module
# turns on its own, so /GL appears twice), and inlining across translation
# units is what LTO is for -- so nothing here prevents an AVX2 body being
# inlined into a caller compiled without /arch:AVX2 and reached with no
# CPUID check in front of it. Measured on MSVC 19.44, that does not happen;
# the guarantee is enforced in source rather than left to the optimizer, by
# marking rolling_beta_reduce_avx2 SQT_NOINLINE. See that file's header.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i686|X86|i386)$")
    if(MSVC)
        set_source_files_properties(src/rolling_beta_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(src/rolling_beta_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
endif()

# ── C++ standard ──────────────────────────────────────────────────────────────

target_compile_features(_sqt_core PRIVATE cxx_std_17)

# ── Include directories ───────────────────────────────────────────────────────

target_include_directories(_sqt_core PRIVATE include)

# ── Native-arch optimization (opt-in) ─────────────────────────────────────────
# /arch:AVX2 / -march=x86-64-v3 let the compiler use AVX2, FMA and BMI2 --
# the resulting binary is not guaranteed to run on an older CPU lacking those
# ISA extensions (an "illegal instruction" crash, not a graceful fallback).
# Defaulting this off means a build produced on one machine (a CI runner, a
# shared dev image, a wheel destined for another user's laptop) doesn't
# silently bake in an assumption about the build machine's own CPU. Opt in
# explicitly for local development if you want the extra speed -- the
# published performance figures were built with it on.
#
# On x86-64, GCC and Clang get -march=x86-64-v3 rather than -march=native:
# the same level MSVC's /arch:AVX2 targets (AVX2, FMA, BMI1/2, LZCNT, MOVBE,
# F16C), portable across every x86-64 CPU since Haswell and Excavator, and
# the same codegen on every machine that builds it -- -march=native on an
# AVX-512 host would have turned on instructions no CI runner or second
# laptop has. A compiler too old to know the level name (GCC < 11, Clang
# < 12) gets -march=haswell, the same instruction set. Other architectures
# keep -march=native. Contraction stays off at every level (see above).
option(SQT_NATIVE_ARCH
    "Compile for AVX2+FMA (x86-64-v3) rather than baseline x86-64 (not portable to older CPUs)"
    OFF)
set(SQT_X86 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i686|X86|i386)$")
    set(SQT_X86 ON)
endif()
set(SQT_GNU_ARCH_FLAG "-march=native")
if(NOT MSVC AND SQT_X86)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=x86-64-v3" SQT_HAS_MARCH_X86_64_V3)
    if(SQT_HAS_MARCH_X86_64_V3)
        set(SQT_GNU_ARCH_FLAG "-march=x86-64-v3")
    else()
        set(SQT_GNU_ARCH_FLAG "-march=haswell")
    endif()
endif()
# Cached for tests/cpp, which compiles its libraries the same way.
set(SQT_GNU_ARCH_FLAG "${SQT_GNU_ARCH_FLAG}" CACHE INTERNAL "SQT_NATIVE_ARCH for GCC/Clang")
if(SQT_NATIVE_ARCH)
    if(MSVC)
        message(STATUS "Native arch: ON  — /arch:AVX2 (not portable to CPUs without AVX2/FMA)")
    else()
        message(STATUS "Native arch: ON  — ${SQT_GNU_ARCH_FLAG} (not portable to CPUs without AVX2/FMA)")
    endif()
else()
    message(STATUS "Native arch: OFF — portable codegen (pass -DSQT_NATIVE_ARCH=ON for local speed)")
endif()

# ── Codegen every native target shares ──────────────────────────────────────
# What the compiled arithmetic IS -- contraction off, the target instruction
# set, link-time optimization -- in one place, applied to the extension below
# and to every library and test executable in tests/cpp. The C++ suite used
# to compile the same sources at baseline ISA with no LTO whatever this
# extension was built with, so a defect that only the shipped codegen has
# could not fail it. A function is global once defined, and this file is
# processed before tests/cpp.
function(sqt_native_codegen target)
    if(SQT_CLANG_CL)
        target_compile_options(${target} PRIVATE "/clang:-ffp-contract=off")
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /fp:precise)
    else()
        target_compile_options(${target} PRIVATE -ffp-contract=off)
    endif()
    # Release and RelWithDebInfo, on both families: this was Release-only on
    # MSVC, so a RelWithDebInfo build got native codegen on GCC/Clang and
    # portable codegen on MSVC from the same flag.
    if(SQT_NATIVE_ARCH)
        if(MSVC)
            target_compile_options(${target} PRIVATE
                $<$<CONFIG:Release,RelWithDebInfo>:/arch:AVX2>)
        else()
            target_compile_options(${target} PRIVATE
                $<$<CONFIG:Release,RelWithDebInfo>:${SQT_GNU_ARCH_FLAG}>)
        endif()
    endif()
    if(SQT_LTO_RELEASE)
        set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endfunction()

# ── PGO (Profile-Guided Optimization, opt-in, local-only) ───────────────────
# A two-step workflow, not a single build: (1) build with SQT_PGO_GENERATE=ON,
# run a representative workload to collect a profile, (2) build with
# SQT_PGO_USE=ON to feed that profile back in. See
# Documentation/30_build_guide.md's PGO section for the full local-only workflow
# -- deliberately NOT wired into any CI workflow (too heavy/fragile for a
# simple `cmake -B build && cmake --build build` pipeline: it requires an
# extra training run between two separate builds, and the profile itself is
# workload- and machine-specific). Same "opt-in for local max speed, off by
# default" philosophy as SQT_NATIVE_ARCH above. What a trained profile bought
# on this project's kernels is measured in Documentation/16_performance.md.
option(SQT_PGO_GENERATE "Build an instrumented binary for PGO training (local-only, not for distribution)" OFF)
option(SQT_PGO_USE       "Build using previously-collected PGO profile data (local-only, not for distribution)" OFF)
if(SQT_PGO_GENERATE AND SQT_PGO_USE)
    message(FATAL_ERROR "SQT_PGO_GENERATE and SQT_PGO_USE are mutually exclusive -- run the two-step workflow one flag at a time (see Documentation/30_build_guide.md).")
endif()
if(SQT_CLANG_CL AND (SQT_PGO_GENERATE OR SQT_PGO_USE))
    message(FATAL_ERROR
        "SQT_PGO_GENERATE and SQT_PGO_USE drive cl's profile-guided workflow "
        "(/GENPROFILE, pgort140.dll, pgomgr), which clang-cl and lld-link do "
        "not take. Build the profile-guided extension with cl.")
endif()
if(SQT_PGO_GENERATE)
    message(STATUS "PGO:      GENERATE — instrumented build for training, not for distribution")
elseif(SQT_PGO_USE)
    message(STATUS "PGO:      USE — optimizing with previously-collected profile data")
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
# Release/RelWithDebInfo: full optimisation. The instruction set and the
#   contraction rule are sqt_native_codegen()'s, applied below.
# Debug: no optimisation + debug info so the extension is debuggable with
#   gdb / lldb / WinDbg.

if(MSVC)
    target_compile_options(_sqt_core PRIVATE
        # Optimisation — config-specific
        $<$<CONFIG:Release,RelWithDebInfo>:/O2>
        $<$<CONFIG:Debug>:/Od /Zi>
        # Warnings
        /W3
        # PGO: /GL (whole-program optimization) is required for either half
        # of the instrument/use cycle.
        $<$<BOOL:${SQT_PGO_GENERATE}>:/GL>
        $<$<BOOL:${SQT_PGO_USE}>:/GL>
    )
    # size_t -> int narrowing warnings (C4244/C4267) are suppressed ONLY for
    # bindings.cpp, where pybind11's own API forces some narrowing (e.g.
    # py::ssize_t <-> size_t conversions at the Python boundary) -- every
    # other file compiles under the full warning set, so a real narrowing
    # bug (like the ones eliminated project-wide in the size_t narrowing
    # sweep) surfaces as a build warning instead of being silently hidden.
    set_source_files_properties(bindings/bindings.cpp PROPERTIES
        COMPILE_OPTIONS "/wd4244;/wd4267")

    # PGO on MSVC. What the two linker flags alone got wrong in practice:
    #
    # - THE INSTRUMENTED BINARY COULD NOT BE IMPORTED. It depends on the
    #   toolset's pgort140.dll, and Python (3.8+) does not search PATH for an
    #   extension's dependencies, so the import failed, the package fell back
    #   to Python for every kernel, and training exercised none of the
    #   instrumented code. The DLL is copied beside the extension for the
    #   GENERATE build and removed again by the next build that does not
    #   need it.
    # - WHERE THE PROFILE LIVES. The linker's default puts the .pgd beside
    #   the output -- the package directory, which every build tree shares.
    #   It is named inside this build tree instead, so the profile stays with
    #   the tree that made it and the USE step can check it before linking.
    # - WHERE THE COUNTS LAND. Wherever the .pgd is, each process that loads
    #   the instrumented binary writes its counts as `_sqt_core!N.pgc` beside
    #   the binary, and the linker only folds in count files found beside
    #   the .pgd -- measured, a USE link with the counts elsewhere optimized
    #   against an empty profile without a word. The USE configure merges
    #   them explicitly with pgomgr and deletes them, so a count is folded in
    #   once, and a USE build without a profile is refused rather than
    #   linked: `pgo: use` in the build facts means a profile was used.
    set(_sqt_pgd "${CMAKE_CURRENT_BINARY_DIR}/_sqt_core.pgd")
    get_filename_component(_sqt_linker_dir "${CMAKE_LINKER}" DIRECTORY)
    file(GLOB _sqt_pgc_files "${_pkg_dir}/_sqt_core!*.pgc")
    if(SQT_PGO_GENERATE)
        find_file(SQT_PGORT_DLL NAMES pgort140.dll HINTS "${_sqt_linker_dir}")
        if(NOT SQT_PGORT_DLL)
            message(FATAL_ERROR
                "SQT_PGO_GENERATE: pgort140.dll, the runtime an instrumented "
                "binary loads, was not found beside ${CMAKE_LINKER}. Configure "
                "from a Visual Studio developer prompt, or pass "
                "-DSQT_PGORT_DLL=<path to pgort140.dll>.")
        endif()
        # Counts left by an earlier instrumented binary describe other code.
        if(_sqt_pgc_files)
            file(REMOVE ${_sqt_pgc_files})
        endif()
        target_link_options(_sqt_core PRIVATE /LTCG "/GENPROFILE:PGD=${_sqt_pgd}")
        add_custom_command(TARGET _sqt_core POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${SQT_PGORT_DLL}" "$<TARGET_FILE_DIR:_sqt_core>"
            VERBATIM)
    else()
        if(SQT_PGO_USE)
            if(NOT EXISTS "${_sqt_pgd}")
                message(FATAL_ERROR
                    "SQT_PGO_USE: no profile at ${_sqt_pgd}. Build this tree "
                    "with -DSQT_PGO_GENERATE=ON first and train it (see "
                    "Documentation/30_build_guide.md).")
            endif()
            find_program(SQT_PGOMGR NAMES pgomgr HINTS "${_sqt_linker_dir}")
            if(_sqt_pgc_files AND NOT SQT_PGOMGR)
                message(FATAL_ERROR
                    "SQT_PGO_USE: pgomgr, which folds the training counts into "
                    "the profile, was not found beside ${CMAKE_LINKER}.")
            endif()
            list(LENGTH _sqt_pgc_files _sqt_pgc_count)
            foreach(_sqt_pgc IN LISTS _sqt_pgc_files)
                execute_process(
                    COMMAND "${SQT_PGOMGR}" /merge "${_sqt_pgc}" "${_sqt_pgd}"
                    RESULT_VARIABLE _sqt_merge_rc
                    OUTPUT_VARIABLE _sqt_merge_out
                    ERROR_VARIABLE _sqt_merge_out)
                if(NOT _sqt_merge_rc EQUAL 0)
                    message(FATAL_ERROR
                        "SQT_PGO_USE: pgomgr could not merge ${_sqt_pgc} into "
                        "${_sqt_pgd}:\n${_sqt_merge_out}")
                endif()
                file(REMOVE "${_sqt_pgc}")
            endforeach()
            message(STATUS "PGO:      merged ${_sqt_pgc_count} training count file(s) into ${_sqt_pgd}")
            target_link_options(_sqt_core PRIVATE /LTCG "/USEPROFILE:PGD=${_sqt_pgd}")
            # A profile that gained counts relinks without a source edit.
            set_property(TARGET _sqt_core APPEND PROPERTY LINK_DEPENDS "${_sqt_pgd}")
        endif()
        add_custom_command(TARGET _sqt_core POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E rm -f
                    "$<TARGET_FILE_DIR:_sqt_core>/pgort140.dll"
            VERBATIM)
    endif()
else()
    # Covers GCC on Linux, Apple Clang on macOS, and Clang everywhere
    target_compile_options(_sqt_core PRIVATE
        $<$<CONFIG:Release,RelWithDebInfo>:-O3>
        $<$<CONFIG:Debug>:-O0 -g>
        -Wall
        -Wextra
        $<$<BOOL:${SQT_PGO_GENERATE}>:-fprofile-generate>
        $<$<BOOL:${SQT_PGO_USE}>:-fprofile-use>
        $<$<BOOL:${SQT_PGO_USE}>:-fprofile-correction>
    )
    target_link_options(_sqt_core PRIVATE
        $<$<BOOL:${SQT_PGO_GENERATE}>:-fprofile-generate>
        $<$<BOOL:${SQT_PGO_USE}>:-fprofile-use>
    )
endif()

# ── OpenMP (optional) ─────────────────────────────────────────────────────────
# Used by monte_carlo.cpp's simulate_forward_paths, backtest.cpp's
# batch_run_strategy and options.cpp's chain kernels, where each simulated
# path / test index / contract is independent
# (no shared mutable state) so the loop can run on multiple threads with no
# locking. Not REQUIRED: a build without an OpenMP runtime (e.g. default
# Apple Clang, which ships no OpenMP support) must still succeed —
# the compiler simply won't define _OPENMP, and the affected loops fall
# back to their identical serial code paths, each wrapped in
# #ifdef _OPENMP / #endif at the call site.
#
# There is deliberately no SQT_HAS_OPENMP definition here any more. It was
# a second signal for a question the compiler already answers, and the two
# drifted: backtest.cpp included <omp.h> under SQT_HAS_OPENMP but used
# omp_get_max_threads() under _OPENMP, so compiling it with -fopenmp/
# /openmp but without this definition failed to compile at all.
#
# ── Which OpenMP runtime MSVC links (SQT_OPENMP_LLVM, off by default) ───────
# MSVC's /openmp is OpenMP 2.0 on vcomp140.dll. /openmp:llvm links LLVM's
# runtime, libomp140.x86_64.dll, instead. It is an option and not the
# default for one reason that outweighs any speed: the Visual C++
# Redistributable installs vcomp140.dll and does NOT install libomp140 (the
# 14.50 redistributable's own file list, checked). Visual Studio ships it
# only under VC\Redist\MSVC\<ver>\debug_nonredist, and the runtime package
# that lays it down in System32 is Visual Studio's own debug runtime. An
# extension built this way therefore fails to load on a machine with the
# redistributable and no Visual Studio, and every kernel takes its Python
# path there (the import check reports it as `unloadable`). What it does to
# the speed of the parallel kernels, measured against vcomp, is in
# Documentation/16_performance.md.
#
# What /openmp:llvm does NOT change: `omp simd` is still a hard C7660 under
# it (only /openmp:experimental accepts that directive, and ignores its
# reduction clause), and _OPENMP stays 200203, so CMake reports "2.0" for
# both runtimes -- which is why the build facts below name the runtime
# rather than trusting the version to tell the builds apart.
#
# The flag is preset in the cache rather than passed as OpenMP_RUNTIME_MSVC,
# which only CMake 3.30+ reads; FindOpenMP takes a preset flag as given on
# every CMake this project supports. FindOpenMP also caches the flag it
# settles on, so switching the option in an existing build tree must drop
# that cached answer, or the tree keeps linking the previous runtime while
# the build facts name the new one. The cache is shared with tests/cpp, so
# the C++ suite runs on the same runtime as the extension.
option(SQT_OPENMP_LLVM
    "MSVC only: link LLVM's OpenMP runtime (libomp140, not in the VC++ redistributable) instead of vcomp140"
    OFF)
if(MSVC AND NOT SQT_CLANG_CL)
    if(SQT_OPENMP_LLVM)
        if(MSVC_VERSION LESS 1928)
            message(FATAL_ERROR "SQT_OPENMP_LLVM needs MSVC 19.28 (Visual Studio 2019 16.8) or later for /openmp:llvm.")
        endif()
        if(NOT "${OpenMP_CXX_FLAGS}" STREQUAL "-openmp:llvm")
            unset(OpenMP_CXX_SPEC_DATE CACHE)
        endif()
        set(OpenMP_CXX_FLAGS "-openmp:llvm" CACHE STRING
            "CXX compiler flags for OpenMP parallelization" FORCE)
        set(OpenMP_CXX_LIB_NAMES "" CACHE STRING
            "CXX compiler libraries for OpenMP parallelization" FORCE)
    elseif("${OpenMP_CXX_FLAGS}" MATCHES "openmp:llvm")
        unset(OpenMP_CXX_FLAGS CACHE)
        unset(OpenMP_CXX_LIB_NAMES CACHE)
        unset(OpenMP_CXX_SPEC_DATE CACHE)
    endif()
elseif(SQT_OPENMP_LLVM)
    message(WARNING "SQT_OPENMP_LLVM applies to MSVC's cl only and is ignored by ${CMAKE_CXX_COMPILER_ID}; clang-cl always links LLVM's runtime.")
endif()

# ── clang-cl links LLVM's own OpenMP runtime, and carries it ────────────────
# FindOpenMP looks for clang-cl's import library on the linker's search path,
# where a Visual Studio prompt puts the MSVC toolset's libomp.lib first. That
# one imports libomp140.x86_64.dll -- the runtime SQT_OPENMP_LLVM links, not
# in the VC++ redistributable (see above) -- so the build linked a DLL the
# build facts did not name and most machines do not have. LLVM ships its own
# import library beside clang-cl, for its own libomp.dll (Apache-2.0 with
# LLVM exceptions); that is the one named here, unless the configure was
# given another.
#
# Python 3.8+ does not search PATH for an extension's DLLs: it looks beside
# the extension, beside python.exe, in System32 and in directories added
# with os.add_dll_directory. LLVM's bin directory is none of those, so the
# DLL is copied beside the extension after each link -- the place Windows
# looks first -- and installed with it, so an editable or wheel install
# carries it too. A copy rather than an add_dll_directory in the loader:
# nothing at import has to know where LLVM was installed on the machine that
# built it (the build stamp records no paths), and the extension loads in any
# process that imports it, not only through the package. (One Windows rule
# applies to such a direct load: the extension's own directory is searched
# only when the path it is loaded by uses backslashes -- measured, the same
# file loaded by a forward-slash path did not find its libomp.dll. Imports
# through sys.path are unaffected.) The copy is a build output, ignored by
# git like the extension; a build that does not link it removes it again. A
# missing copy makes the import fail, and the package's import check names
# the DLL (native_build_status(), `unloadable`).
#
# Any other import library (the MSVC toolset's, given explicitly or found on
# the search path) is linked as found and named `libomp140` in the build
# facts, with a warning: an extension so built loads only where Visual Studio
# has installed libomp140.x86_64.dll.
if(SQT_CLANG_CL)
    get_filename_component(_sqt_llvm_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_sqt_llvm_root "${_sqt_llvm_bin}" DIRECTORY)
    if(NOT OpenMP_libomp_LIBRARY AND EXISTS "${_sqt_llvm_root}/lib/libomp.lib")
        set(OpenMP_libomp_LIBRARY "${_sqt_llvm_root}/lib/libomp.lib" CACHE FILEPATH
            "The OpenMP import library clang-cl links: LLVM's own, beside the compiler")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_sqt_core PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP:   found (${OpenMP_CXX_FLAGS}) — simulate_forward_paths/batch_run_strategy will run in parallel")
else()
    message(STATUS "OpenMP:   not found — affected loops will run serially (still correct, just single-threaded)")
endif()

set(_sqt_clang_cl_runtime "")
set(SQT_LIBOMP_DLL "" CACHE INTERNAL "LLVM's libomp.dll, copied beside every binary that links it")
if(SQT_CLANG_CL AND OpenMP_CXX_FOUND)
    # An import library does not say where its DLL is; LLVM's layout does:
    # lib/libomp.lib imports bin/libomp.dll.
    get_filename_component(_sqt_omp_lib_dir "${OpenMP_libomp_LIBRARY}" DIRECTORY)
    get_filename_component(_sqt_libomp_dll "${_sqt_omp_lib_dir}/../bin/libomp.dll" ABSOLUTE)
    if(OpenMP_libomp_LIBRARY AND EXISTS "${_sqt_libomp_dll}")
        set(_sqt_clang_cl_runtime "libomp")
        set(SQT_LIBOMP_DLL "${_sqt_libomp_dll}" CACHE INTERNAL
            "LLVM's libomp.dll, copied beside every binary that links it")
        add_custom_command(TARGET _sqt_core POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                    "${SQT_LIBOMP_DLL}" "$<TARGET_FILE_DIR:_sqt_core>"
            VERBATIM)
        install(FILES "${SQT_LIBOMP_DLL}" DESTINATION standard_quant_tools)
        message(STATUS "OpenMP:   clang-cl links LLVM's runtime (${OpenMP_libomp_LIBRARY}); libomp.dll is copied beside the extension")
    else()
        set(_sqt_clang_cl_runtime "libomp140")
        message(WARNING
            "clang-cl: OpenMP links ${OpenMP_libomp_LIBRARY}, which is not LLVM's "
            "own import library (no libomp.dll in its ../bin). It is taken to be "
            "the MSVC toolset's, whose libomp140.x86_64.dll is not in the VC++ "
            "redistributable: the extension will load only where Visual Studio "
            "installed that DLL. Pass -DOpenMP_libomp_LIBRARY=<LLVM>/lib/libomp.lib "
            "to link LLVM's runtime instead.")
    endif()
endif()
if(WIN32 AND NOT _sqt_clang_cl_runtime STREQUAL "libomp")
    # A clang-cl build's runtime, left beside the extension by an earlier
    # build in the same source tree, is nothing this extension loads.
    add_custom_command(TARGET _sqt_core POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E rm -f "$<TARGET_FILE_DIR:_sqt_core>/libomp.dll"
        VERBATIM)
endif()

# ── Build stamp: which sources and which build produced this binary ─────────
# The extension carries a SHA-256 of the native tree it was compiled from,
# and the package recomputes it over the sources beside it at import. An
# extension that loads but was built from other sources -- the compiled copy
# an editable install keeps in site-packages after the C++ has moved on -- is
# then refused by name instead of answering with old code while reporting
# itself available. See cmake/source_digest.cmake for the recipe and why it
# runs at build time.
#
# The facts that change results or speed are known at configure time and
# land in a small CMake file the stamp step reads. file(CONFIGURE) rewrites it
# only when its text changes, so the stamp is regenerated when a fact moves
# (a reconfigure with another SQT_NATIVE_ARCH, say) on every generator,
# not just those that track command lines.
if(SQT_PGO_GENERATE)
    set(_sqt_pgo "generate")
elseif(SQT_PGO_USE)
    set(_sqt_pgo "use")
else()
    set(_sqt_pgo "off")
endif()
if(SQT_NATIVE_ARCH)
    set(_sqt_native_arch 1)
else()
    set(_sqt_native_arch 0)
endif()
# The runtime is named from the flag or library FindOpenMP settled on, not
# from the version: MSVC reports "2.0" for vcomp and for libomp alike.
# clang-cl is tested first because MSVC is true for it too, and the cl
# branch named its LLVM runtime `vcomp` -- the one full-suite failure of a
# clang-cl build was that mislabel.
set(_sqt_openmp "")
set(_sqt_openmp_runtime "")
if(OpenMP_CXX_FOUND)
    set(_sqt_openmp "${OpenMP_CXX_VERSION}")
    if(SQT_CLANG_CL)
        set(_sqt_openmp_runtime "${_sqt_clang_cl_runtime}")
    elseif(MSVC)
        if("${OpenMP_CXX_FLAGS}" MATCHES "openmp:llvm")
            set(_sqt_openmp_runtime "libomp")
        elseif("${OpenMP_CXX_FLAGS}" MATCHES "openmp:experimental")
            set(_sqt_openmp_runtime "vcomp-experimental")
        else()
            set(_sqt_openmp_runtime "vcomp")
        endif()
    elseif(OpenMP_CXX_LIB_NAMES)
        list(GET OpenMP_CXX_LIB_NAMES 0 _sqt_openmp_lib)
        string(REGEX REPLACE "^lib" "" _sqt_openmp_lib "${_sqt_openmp_lib}")
        set(_sqt_openmp_runtime "lib${_sqt_openmp_lib}")
    else()
        set(_sqt_openmp_runtime "unknown")
    endif()
endif()
set(_sqt_facts_file "${CMAKE_CURRENT_BINARY_DIR}/sqt_build_facts.cmake")
file(CONFIGURE OUTPUT "${_sqt_facts_file}" CONTENT
"set(SQT_FACT_COMPILER \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\")
set(SQT_FACT_NATIVE_ARCH ${_sqt_native_arch})
set(SQT_FACT_OPENMP \"${_sqt_openmp}\")
set(SQT_FACT_OPENMP_RUNTIME \"${_sqt_openmp_runtime}\")
set(SQT_FACT_PGO \"${_sqt_pgo}\")
")

# Every file the digest covers is a dependency of the stamp, so an edit
# followed by a plain `cmake --build` restamps. CONFIGURE_DEPENDS re-runs the
# glob on each build, so a file added or removed is picked up without a
# manual reconfigure. The filter is the recipe's own, case-sensitive on
# every platform.
file(GLOB_RECURSE _sqt_stamp_inputs CONFIGURE_DEPENDS LIST_DIRECTORIES false
     "${CMAKE_CURRENT_SOURCE_DIR}/*")
list(FILTER _sqt_stamp_inputs INCLUDE REGEX
     "(\\.(cpp|cc|cxx|c|hpp|hh|hxx|h|inl|ipp|cmake)|/CMakeLists\\.txt)$")

set(_sqt_stamp_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_sqt_stamp "${_sqt_stamp_dir}/sqt_build_stamp.hpp")
add_custom_command(
    OUTPUT "${_sqt_stamp}"
    COMMAND "${CMAKE_COMMAND}"
            "-DSQT_NATIVE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DSQT_OUTPUT=${_sqt_stamp}"
            "-DSQT_FACTS_FILE=${_sqt_facts_file}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/source_digest.cmake"
    DEPENDS ${_sqt_stamp_inputs} "${_sqt_facts_file}"
    COMMENT "Stamping _sqt_core with the digest of its sources"
    VERBATIM
)
target_sources(_sqt_core PRIVATE "${_sqt_stamp}")
target_include_directories(_sqt_core PRIVATE "${_sqt_stamp_dir}")
# $<CONFIG> is decided at build time under a multi-config generator, which is
# why the configuration is a definition on the one file that reports it and
# not a line in the stamp.
set_property(SOURCE src/build_info.cpp APPEND PROPERTY
             COMPILE_DEFINITIONS "SQT_BUILD_CONFIG=\"$<CONFIG>\"")

# ── Link-time optimization / interprocedural optimization (automatic) ────────
# Unlike SQT_NATIVE_ARCH above, LTO/IPO carries no "illegal instruction on a
# different CPU" portability risk -- it doesn't change the target ISA, only
# how the compiler optimizes across translation-unit boundaries at link
# time. Applied automatically whenever the toolchain supports it (no opt-in
# flag needed), scoped to Release only -- Debug/RelWithDebInfo stay fast to
# build and easy to debug, matching the existing /O2-vs-/Od split above.
#
# One caveat, handled in source rather than here: LTO inlines across
# translation-unit boundaries, which is what makes the per-source
# /arch:AVX2 above a codegen boundary and not an inlining one.
# rolling_beta_reduce_avx2 is marked SQT_NOINLINE so its AVX2 body cannot be
# hoisted into a caller that no CPUID check guards.
include(CheckIPOSupported)
check_ipo_supported(RESULT SQT_IPO_SUPPORTED OUTPUT SQT_IPO_ERROR)
# Cached so tests/cpp links its libraries the way the extension is linked;
# sqt_native_codegen() reads it.
set(SQT_LTO_RELEASE "${SQT_IPO_SUPPORTED}" CACHE INTERNAL
    "LTO for the Release builds of every native target")
if(SQT_IPO_SUPPORTED)
    message(STATUS "LTO/IPO:  enabled for Release builds")
else()
    message(STATUS "LTO/IPO:  not supported by this toolchain (${SQT_IPO_ERROR}) — skipping")
endif()

# Contraction off, the instruction set, LTO: see sqt_native_codegen().
sqt_native_codegen(_sqt_core)

# ── Target properties ─────────────────────────────────────────────────────────
# CXX_VISIBILITY_PRESET hidden + VISIBILITY_INLINES_HIDDEN:
#   Hide all internal symbols by default — required on Linux/macOS to prevent
#   symbol conflicts when multiple extensions sharing internal names (e.g. STL
#   symbols) are loaded into the same interpreter process.
#   pybind11's PYBIND11_MODULE macro marks the module entry point as visible.
#
# Output directories:
#   Drop the compiled extension directly into the Python package so that
#     "from standard_quant_tools import _sqt_core"
#   works from an editable install without any extra install step.
#
#   Extension suffix is platform-managed by pybind11/Python:
#     Windows : _sqt_core.pyd
#     Linux   : _sqt_core.cpython-3XX-x86_64-linux-gnu.so
#     macOS   : _sqt_core.cpython-3XX-darwin.so

# The output (_pkg_dir, set at the top) lands in the SOURCE tree, which is
# what the in-place developer workflow (`cmake --build build`, then run
# pytest against src/) depends on. A wheel is built in an isolated staging
# directory and copies only what CMake INSTALLS, so without this rule
# scikit-build-core produced a wheel with no extension in it -- a silent
# pure-Python package that looked like a successful native build.
install(TARGETS _sqt_core
        LIBRARY DESTINATION standard_quant_tools   # Unix/macOS .so
        RUNTIME DESTINATION standard_quant_tools)  # Windows .pyd

set_target_properties(_sqt_core PROPERTIES
    CXX_VISIBILITY_PRESET     hidden
    VISIBILITY_INLINES_HIDDEN ON
    # Unix/macOS: shared-library output
    LIBRARY_OUTPUT_DIRECTORY                "${_pkg_dir}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG          "${_pkg_dir}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE        "${_pkg_dir}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_pkg_dir}"
    # Windows: .pyd is treated as a runtime DLL
    RUNTIME_OUTPUT_DIRECTORY                "${_pkg_dir}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${_pkg_dir}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${_pkg_dir}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_pkg_dir}"
)
