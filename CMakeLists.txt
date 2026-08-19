# 3.19 is the real floor: find_package(... Development.Module) needs 3.18,
# and the multi-value $<CONFIG:Release,RelWithDebInfo>:...> generator
# expressions used in _cpp/CMakeLists.txt need 3.19.
cmake_minimum_required(VERSION 3.19)

# LANGUAGES NONE, then CXX is enabled only once a compiler is actually found.
#
# This file is now on the `pip install` path (see pyproject.toml's
# scikit-build-core backend), and _sqt_core is an OPTIONAL fast path: every
# function it accelerates has a Python/NumPy fallback, and the package is
# designed to be fully usable without it. Declaring `LANGUAGES CXX` up front
# would make CMake demand a working C++ compiler at configure time, so a user
# on a machine without a toolchain could no longer install the library at all
# -- turning an optional accelerator into a hard dependency, which is the
# opposite of what shipping it in the build was meant to achieve.
#
# So: try to build it, and degrade to pure Python when that is not possible.
project(sqt_core VERSION 0.1.0 LANGUAGES NONE)

# Developers and CI want the opposite default -- a silent skip there means a
# green build that quietly tested only the fallback path. SQT_REQUIRE_NATIVE=ON
# turns every graceful skip below into a hard error.
option(SQT_REQUIRE_NATIVE
       "Fail the build if the native extension cannot be built" OFF)

# enable_language(... OPTIONAL), NOT check_language(). check_language runs a
# SEPARATE try-compile that does not inherit the Visual Studio generator's own
# toolchain discovery, so on Windows it reported "no compiler" on a machine
# with a perfectly working MSVC install -- which would have broken the
# ordinary `cmake -S . -B build` developer configure this file has always
# supported. Verified against the previous revision in the same shell: the old
# `LANGUAGES CXX` form found MSVC 19.44.35228.0 while check_language did not.
#
# enable_language(OPTIONAL) uses the real generator, so it agrees with the
# build that follows it.
enable_language(CXX OPTIONAL)
if(NOT CMAKE_CXX_COMPILER_LOADED)
    if(SQT_REQUIRE_NATIVE)
        message(FATAL_ERROR
            "No C++ compiler found and SQT_REQUIRE_NATIVE=ON.")
    endif()
    message(WARNING
        "No C++ compiler found -- installing the pure-Python package only.\n"
        "Every function still works through its NumPy/numba fallback; the "
        "compiled speedups will be unavailable and HAS_CPP will be False.")
    return()
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Find Python ──────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# ── Find pybind11 installed as a Python package ──────────────────────────────
# Works on every OS: Python locates pybind11 regardless of how it was installed.
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_ret
)
if(NOT _pybind11_ret EQUAL 0)
    if(SQT_REQUIRE_NATIVE)
        message(FATAL_ERROR
            "pybind11 cmake directory not found and SQT_REQUIRE_NATIVE=ON.\n"
            "Install it with:  pip install pybind11\n")
    endif()
    # Same reasoning as the missing-compiler case above: skip rather than
    # fail, so the pure-Python package still installs.
    message(WARNING
        "pybind11 not found -- installing the pure-Python package only.\n"
        "Install it with:  pip install pybind11\n")
    return()
endif()
list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
find_package(pybind11 CONFIG REQUIRED)

message(STATUS "System:   ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Python:   ${Python3_EXECUTABLE} (${Python3_VERSION})")
message(STATUS "pybind11: ${pybind11_VERSION}")

# ── Extension module ─────────────────────────────────────────────────────────
add_subdirectory(src/standard_quant_tools/_cpp)

# ── C++ unit tests (opt-in, disabled by default) ─────────────────────────────
option(SQT_BUILD_TESTS "Build C++ unit tests" OFF)
if(SQT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
