set(SQT_SOURCES
    src/hurst.cpp
    bindings/bindings.cpp
)

pybind11_add_module(_sqt_core MODULE ${SQT_SOURCES})

target_include_directories(_sqt_core PRIVATE include)

# Compiler flags
if(MSVC)
    # /O2 = full optimisation  /arch:AVX2 = SIMD  /W3 = standard warnings
    # /wd4244,4267 = silence narrowing-conversion noise from pybind11 internals
    target_compile_options(_sqt_core PRIVATE /O2 /arch:AVX2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(_sqt_core PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Place the compiled extension directly in the Python package directory so
# "import standard_quant_tools._sqt_core" works without any install step.
set(_pkg_dir "${CMAKE_SOURCE_DIR}/src/standard_quant_tools")

set_target_properties(_sqt_core PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY                "${_pkg_dir}"
    LIBRARY_OUTPUT_DIRECTORY                "${_pkg_dir}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${_pkg_dir}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${_pkg_dir}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_pkg_dir}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG          "${_pkg_dir}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE        "${_pkg_dir}"
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${_pkg_dir}"
)
