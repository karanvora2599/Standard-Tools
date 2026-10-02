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
    src/build_info.cpp
    bindings/bindings.cpp
)

pybind11_add_module(_sqt_core MODULE ${SQT_SOURCES})

# The package directory the extension is written into (see the output
# directories at the end of this file).
set(_pkg_dir "${CMAKE_SOURCE_DIR}/src/standard_quant_tools")

# ── Options and CUSUM: the Python arithmetic, operation for operation ────────
# options.cpp is held to the Python option formulas result for result -- a
# converged flag and an iteration count are only comparable between the two
# if every price along the way is the same double. That rules out fusing
# a*b+c into one rounding: Clang contracts within an expression by default
# and GCC does under -std=gnu++*, and either would do so under
# -march=native. MSVC's /fp:precise does not contract since VS 2022; it is
# stated here so a change of default cannot quietly alter it.
#
# cusum.cpp is held to the Python CUSUM loop bit for bit under the same
# rule. Its recursion has no multiply to fuse, so the flag guards against a
# later edit rather than against today's code.
if(MSVC)
    set_source_files_properties(src/options.cpp src/cusum.cpp PROPERTIES COMPILE_OPTIONS "/fp:precise")
else()
    set_source_files_properties(src/options.cpp src/cusum.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

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
# -march=native / /arch:AVX2 tune codegen for the exact CPU that compiles the
# extension -- the resulting binary is not guaranteed to run on a different/
# older CPU lacking those ISA extensions (an "illegal instruction" crash, not
# a graceful fallback). Defaulting this off means a build produced on one
# machine (a CI runner, a shared dev image, a wheel destined for another
# user's laptop) doesn't silently bake in an assumption about the build
# machine's own CPU. Opt in explicitly for local development if you want the
# extra speed -- this session's own measured benchmarks in
# the published performance figures were built with it on.
option(SQT_NATIVE_ARCH
    "Optimize for the host CPU's instruction set (not portable to a different/older CPU)"
    OFF)
if(SQT_NATIVE_ARCH)
    message(STATUS "Native arch: ON  — /arch:AVX2 or -march=native (not portable off this CPU)")
else()
    message(STATUS "Native arch: OFF — portable codegen (pass -DSQT_NATIVE_ARCH=ON for local speed)")
endif()

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
if(SQT_PGO_GENERATE)
    message(STATUS "PGO:      GENERATE — instrumented build for training, not for distribution")
elseif(SQT_PGO_USE)
    message(STATUS "PGO:      USE — optimizing with previously-collected profile data")
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
# Release/RelWithDebInfo: full optimisation, +SIMD tuned for the build
#   machine only if SQT_NATIVE_ARCH is on.
# Debug: no optimisation + debug info so the extension is debuggable with
#   gdb / lldb / WinDbg.

if(MSVC)
    target_compile_options(_sqt_core PRIVATE
        # Optimisation — config-specific
        $<$<CONFIG:Release,RelWithDebInfo>:/O2>
        # Release,RelWithDebInfo -- matching the non-MSVC branch below, which
        # applies -march=native to both. This was Release-only, so a
        # RelWithDebInfo build got native codegen on GCC/Clang and portable
        # codegen on MSVC from the same flag: a benchmark comparing the two
        # compilers in that configuration was not comparing the same build.
        $<$<AND:$<CONFIG:Release,RelWithDebInfo>,$<BOOL:${SQT_NATIVE_ARCH}>>:/arch:AVX2>
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
        $<$<AND:$<CONFIG:Release,RelWithDebInfo>,$<BOOL:${SQT_NATIVE_ARCH}>>:-march=native>
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
if(MSVC)
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
    message(WARNING "SQT_OPENMP_LLVM applies to MSVC only and is ignored by ${CMAKE_CXX_COMPILER_ID}.")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_sqt_core PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP:   found (${OpenMP_CXX_FLAGS}) — simulate_forward_paths/batch_run_strategy will run in parallel")
else()
    message(STATUS "OpenMP:   not found — affected loops will run serially (still correct, just single-threaded)")
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
set(_sqt_openmp "")
set(_sqt_openmp_runtime "")
if(OpenMP_CXX_FOUND)
    set(_sqt_openmp "${OpenMP_CXX_VERSION}")
    if(MSVC)
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
if(SQT_IPO_SUPPORTED)
    set_property(TARGET _sqt_core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    message(STATUS "LTO/IPO:  enabled for Release builds")
else()
    message(STATUS "LTO/IPO:  not supported by this toolchain (${SQT_IPO_ERROR}) — skipping")
endif()

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
