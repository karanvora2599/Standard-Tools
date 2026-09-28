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
    src/build_info.cpp
    bindings/bindings.cpp
)

pybind11_add_module(_sqt_core MODULE ${SQT_SOURCES})

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
# default" philosophy as SQT_NATIVE_ARCH above.
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
    target_link_options(_sqt_core PRIVATE
        $<$<BOOL:${SQT_PGO_GENERATE}>:/LTCG:PGInstrument>
        $<$<BOOL:${SQT_PGO_USE}>:/LTCG:PGOptimize>
    )
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
# Used by monte_carlo.cpp's simulate_forward_paths and backtest.cpp's
# batch_run_strategy, where each simulated path / test index is independent
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
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_sqt_core PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP:   found — simulate_forward_paths/batch_run_strategy will run in parallel")
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
if(OpenMP_CXX_FOUND)
    set(_sqt_openmp "${OpenMP_CXX_VERSION}")
else()
    set(_sqt_openmp "")
endif()
set(_sqt_facts_file "${CMAKE_CURRENT_BINARY_DIR}/sqt_build_facts.cmake")
file(CONFIGURE OUTPUT "${_sqt_facts_file}" CONTENT
"set(SQT_FACT_COMPILER \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\")
set(SQT_FACT_NATIVE_ARCH ${_sqt_native_arch})
set(SQT_FACT_OPENMP \"${_sqt_openmp}\")
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

set(_pkg_dir "${CMAKE_SOURCE_DIR}/src/standard_quant_tools")

# The output above lands in the SOURCE tree, which is what the in-place
# developer workflow (`cmake --build build`, then run pytest against src/)
# depends on. A wheel is built in an isolated staging directory and copies
# only what CMake INSTALLS, so without this rule scikit-build-core produced a
# wheel with no extension in it -- a silent pure-Python package that looked
# like a successful native build.
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
