set(SQT_SOURCES
    src/hurst.cpp
    src/indicators.cpp
    bindings/bindings.cpp
)

pybind11_add_module(_sqt_core MODULE ${SQT_SOURCES})

target_include_directories(_sqt_core PRIVATE include)

# ── Compiler flags ────────────────────────────────────────────────────────────
# MSVC (Windows), GCC (Linux), and Apple Clang (macOS) are all handled.
# -march=native / /arch:AVX2 tunes for the build machine; safe for local use.
if(MSVC)
    target_compile_options(_sqt_core PRIVATE
        /O2           # Full optimisation
        /arch:AVX2    # SIMD — remove if targeting older hardware without AVX2
        /W3           # Standard warnings
        /wd4244       # Silence size_t → int narrowing from pybind11 internals
        /wd4267       # Same
    )
else()
    # Covers GCC on Linux, Apple Clang on macOS, and Clang everywhere
    target_compile_options(_sqt_core PRIVATE
        -O3
        -march=native   # Tune for build machine CPU
        -Wall
        -Wextra
    )
endif()

# ── Output directory ──────────────────────────────────────────────────────────
# Drop the compiled extension directly into the Python package so that
#   "from standard_quant_tools import _sqt_core"
# works from an editable install without any extra install step.
#
# Extension suffix is platform-managed by pybind11/Python:
#   Windows : _sqt_core.pyd
#   Linux   : _sqt_core.cpython-3XX-x86_64-linux-gnu.so
#   macOS   : _sqt_core.cpython-3XX-darwin.so
set(_pkg_dir "${CMAKE_SOURCE_DIR}/src/standard_quant_tools")

set_target_properties(_sqt_core PROPERTIES
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
