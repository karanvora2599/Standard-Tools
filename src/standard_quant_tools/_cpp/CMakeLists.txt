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
    bindings/bindings.cpp
)

pybind11_add_module(_sqt_core MODULE ${SQT_SOURCES})

# ── C++ standard ──────────────────────────────────────────────────────────────

target_compile_features(_sqt_core PRIVATE cxx_std_17)

# ── Include directories ───────────────────────────────────────────────────────

target_include_directories(_sqt_core PRIVATE include)

# ── Compiler flags ────────────────────────────────────────────────────────────
# Release/RelWithDebInfo: full optimisation + SIMD tuned for the build machine.
# Debug: no optimisation + debug info so the extension is debuggable with
#   gdb / lldb / WinDbg.  Remove -march=native / /arch:AVX2 for distributable
#   wheels — see build_guide.md Section 9.

if(MSVC)
    target_compile_options(_sqt_core PRIVATE
        # Optimisation — config-specific
        $<$<CONFIG:Release,RelWithDebInfo>:/O2>
        $<$<CONFIG:Release>:/arch:AVX2>   # SIMD; drop for older-hardware targets
        $<$<CONFIG:Debug>:/Od /Zi>
        # Warnings
        /W3
        /wd4244    # size_t → int narrowing from pybind11 internals
        /wd4267    # same
    )
else()
    # Covers GCC on Linux, Apple Clang on macOS, and Clang everywhere
    target_compile_options(_sqt_core PRIVATE
        $<$<CONFIG:Release,RelWithDebInfo>:-O3 -march=native>
        $<$<CONFIG:Debug>:-O0 -g>
        -Wall
        -Wextra
    )
endif()

# ── OpenMP (optional) ─────────────────────────────────────────────────────────
# Only used today by monte_carlo.cpp's simulate_forward_paths, where each
# simulated path is independent (no shared mutable state) so the loop can
# run on multiple threads with no locking. Not REQUIRED: a build without an
# OpenMP runtime (e.g. default Apple Clang, which ships no OpenMP support)
# must still succeed — SQT_HAS_OPENMP simply won't be defined, and the
# affected loop falls back to the identical serial code path, wrapped in
# #ifdef SQT_HAS_OPENMP / #endif at the call site.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_sqt_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_sqt_core PRIVATE SQT_HAS_OPENMP)
    message(STATUS "OpenMP:   found — simulate_forward_paths will run in parallel")
else()
    message(STATUS "OpenMP:   not found — simulate_forward_paths will run serially (still correct, just single-threaded)")
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
