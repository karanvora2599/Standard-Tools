cmake_minimum_required(VERSION 3.15)
project(sqt_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Find Python ──────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# ── Find pybind11 (installed as a Python package) ───────────────────────────
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _pybind11_cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
find_package(pybind11 CONFIG REQUIRED)

message(STATUS "Python:   ${Python3_EXECUTABLE}  (${Python3_VERSION})")
message(STATUS "pybind11: ${pybind11_VERSION}")

# ── Extension module ─────────────────────────────────────────────────────────
add_subdirectory(src/standard_quant_tools/_cpp)

# ── C++ unit tests (opt-in) ──────────────────────────────────────────────────
option(SQT_BUILD_TESTS "Build C++ unit tests" OFF)
if(SQT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cpp)
endif()
