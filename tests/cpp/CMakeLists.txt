# ── Shared helpers: compile each .cpp once and link to test + bench ────────────

add_library(sqt_hurst_impl STATIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/hurst.cpp
)
target_include_directories(sqt_hurst_impl PUBLIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)
target_compile_features(sqt_hurst_impl PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(sqt_hurst_impl PRIVATE /O2 /W3)
else()
    target_compile_options(sqt_hurst_impl PRIVATE -O2 -Wall -Wextra)
endif()
# Optional, mirrors sqt_monte_carlo_impl/sqt_backtest_impl above:
# rolling_hurst_into runs each window in parallel when an OpenMP runtime is
# available; without one, SQT_HAS_OPENMP stays undefined and it just runs
# its identical serial loop.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sqt_hurst_impl PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(sqt_hurst_impl PUBLIC SQT_HAS_OPENMP)
endif()

add_library(sqt_indicators_impl STATIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/indicators.cpp
)
target_include_directories(sqt_indicators_impl PUBLIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)
target_compile_features(sqt_indicators_impl PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(sqt_indicators_impl PRIVATE /O2 /W3)
else()
    target_compile_options(sqt_indicators_impl PRIVATE -O2 -Wall -Wextra)
endif()


# ── Unit tests ────────────────────────────────────────────────────────────────

add_executable(test_hurst test_hurst.cpp)
target_link_libraries(test_hurst PRIVATE sqt_hurst_impl)
target_compile_features(test_hurst PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_hurst PRIVATE /O2 /W3)
else()
    target_compile_options(test_hurst PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_hurst COMMAND test_hurst)

add_executable(test_indicators test_indicators.cpp)
target_link_libraries(test_indicators PRIVATE sqt_indicators_impl)
target_compile_features(test_indicators PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_indicators PRIVATE /O2 /W3)
else()
    target_compile_options(test_indicators PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_indicators COMMAND test_indicators)

add_library(sqt_cointegration_impl STATIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/cointegration.cpp
)
target_include_directories(sqt_cointegration_impl PUBLIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)
target_compile_features(sqt_cointegration_impl PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(sqt_cointegration_impl PRIVATE /O2 /W3)
else()
    target_compile_options(sqt_cointegration_impl PRIVATE -O2 -Wall -Wextra)
endif()

add_executable(test_cointegration test_cointegration.cpp)
target_link_libraries(test_cointegration PRIVATE sqt_cointegration_impl)
target_compile_features(test_cointegration PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_cointegration PRIVATE /O2 /W3)
else()
    target_compile_options(test_cointegration PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_cointegration COMMAND test_cointegration)

add_library(sqt_backtest_impl STATIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/backtest.cpp
)
target_include_directories(sqt_backtest_impl PUBLIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)
target_compile_features(sqt_backtest_impl PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(sqt_backtest_impl PRIVATE /O2 /W3)
else()
    target_compile_options(sqt_backtest_impl PRIVATE -O2 -Wall -Wextra)
endif()
# Optional, mirrors sqt_monte_carlo_impl above: batch_run_strategy runs each
# test index in parallel when an OpenMP runtime is available; without one,
# SQT_HAS_OPENMP stays undefined and it just runs its identical serial loop.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sqt_backtest_impl PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(sqt_backtest_impl PUBLIC SQT_HAS_OPENMP)
endif()

add_executable(test_backtest test_backtest.cpp)
target_link_libraries(test_backtest PRIVATE sqt_backtest_impl)
target_compile_features(test_backtest PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_backtest PRIVATE /O2 /W3)
else()
    target_compile_options(test_backtest PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_backtest COMMAND test_backtest)

add_library(sqt_monte_carlo_impl STATIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/monte_carlo.cpp
)
target_include_directories(sqt_monte_carlo_impl PUBLIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)
target_compile_features(sqt_monte_carlo_impl PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(sqt_monte_carlo_impl PRIVATE /O2 /W3)
else()
    target_compile_options(sqt_monte_carlo_impl PRIVATE -O2 -Wall -Wextra)
endif()
# Optional, mirrors src/standard_quant_tools/_cpp/CMakeLists.txt: without an
# OpenMP runtime, SQT_HAS_OPENMP stays undefined and simulate_forward_paths
# just runs its identical serial fallback -- still correct either way.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(sqt_monte_carlo_impl PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(sqt_monte_carlo_impl PUBLIC SQT_HAS_OPENMP)
endif()

add_executable(test_monte_carlo test_monte_carlo.cpp)
target_link_libraries(test_monte_carlo PRIVATE sqt_monte_carlo_impl)
target_compile_features(test_monte_carlo PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_monte_carlo PRIVATE /O2 /W3)
else()
    target_compile_options(test_monte_carlo PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_monte_carlo COMMAND test_monte_carlo)

add_library(sqt_garch_impl STATIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/garch.cpp
)
target_include_directories(sqt_garch_impl PUBLIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)
target_compile_features(sqt_garch_impl PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(sqt_garch_impl PRIVATE /O2 /W3)
else()
    target_compile_options(sqt_garch_impl PRIVATE -O2 -Wall -Wextra)
endif()

add_executable(test_garch test_garch.cpp)
target_link_libraries(test_garch PRIVATE sqt_garch_impl)
target_compile_features(test_garch PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_garch PRIVATE /O2 /W3)
else()
    target_compile_options(test_garch PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_garch COMMAND test_garch)

add_library(sqt_signals_impl STATIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/signal_state_machines.cpp
)
target_include_directories(sqt_signals_impl PUBLIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)
target_compile_features(sqt_signals_impl PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(sqt_signals_impl PRIVATE /O2 /W3)
else()
    target_compile_options(sqt_signals_impl PRIVATE -O2 -Wall -Wextra)
endif()

add_executable(test_signals test_signals.cpp)
target_link_libraries(test_signals PRIVATE sqt_signals_impl)
target_compile_features(test_signals PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_signals PRIVATE /O2 /W3)
else()
    target_compile_options(test_signals PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_signals COMMAND test_signals)

add_library(sqt_rolling_regression_impl STATIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/rolling_regression.cpp
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/isa_dispatch.cpp
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/rolling_beta_avx2.cpp
)
target_include_directories(sqt_rolling_regression_impl PUBLIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)
target_compile_features(sqt_rolling_regression_impl PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(sqt_rolling_regression_impl PRIVATE /O2 /W3)
else()
    target_compile_options(sqt_rolling_regression_impl PRIVATE -O2 -Wall -Wextra)
endif()
# See src/standard_quant_tools/_cpp/CMakeLists.txt's matching block for why
# these flags are x86/x64-gated: they're x86-only and a hard compile error
# on other toolchains, while rolling_beta_avx2.cpp's own source-level #if
# guard already compiles a portable stub everywhere else.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i686|X86|i386)$")
    if(MSVC)
        set_source_files_properties(
            ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/rolling_beta_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2"
        )
    else()
        set_source_files_properties(
            ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/rolling_beta_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma"
        )
    endif()
endif()

add_executable(test_rolling_regression test_rolling_regression.cpp)
target_link_libraries(test_rolling_regression PRIVATE sqt_rolling_regression_impl)
target_compile_features(test_rolling_regression PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_rolling_regression PRIVATE /O2 /W3)
else()
    target_compile_options(test_rolling_regression PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_rolling_regression COMMAND test_rolling_regression)

# ── Fuzz/property harness (correctness/portability pass item 20a) ────────────
# Randomized-input stress test for cointegration.cpp's gauss_elim (via
# ols2/adf_test/engle_granger) and rolling_regression.cpp's cholesky_solve
# (via rolling_factor_loadings) -- fixed seed for deterministic default CI
# runs. Registered as a normal ctest, so it automatically gets ASan/UBSan
# coverage for free via the existing build-and-test-sanitizers CI job.
add_executable(fuzz_cointegration fuzz_cointegration.cpp)
target_link_libraries(fuzz_cointegration PRIVATE
    sqt_cointegration_impl
    sqt_rolling_regression_impl
)
target_compile_features(fuzz_cointegration PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(fuzz_cointegration PRIVATE /O2 /W3)
else()
    target_compile_options(fuzz_cointegration PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_fuzz_cointegration COMMAND fuzz_cointegration)


# ── Performance benchmarks ────────────────────────────────────────────────────
# Not CTest tests — run manually to inspect timing output:
#   Windows  (VS gen) : build\tests\cpp\Release\bench_hurst.exe
#                       build\tests\cpp\Release\bench_backtest.exe
#   Windows  (Ninja)  : build\tests\cpp\bench_hurst.exe
#                       build\tests\cpp\bench_backtest.exe
#   Linux / macOS     : ./build/tests/cpp/bench_hurst
#                       ./build/tests/cpp/bench_backtest

# SQT_NATIVE_ARCH: defined and opt-in (default OFF) in
# src/standard_quant_tools/_cpp/CMakeLists.txt, processed before this file --
# same portability rationale applies here (a bench binary built with
# -march=native/AVX2 baked in isn't guaranteed to run on a different CPU).

add_executable(bench_hurst bench_hurst.cpp)
target_link_libraries(bench_hurst PRIVATE sqt_hurst_impl)
target_compile_features(bench_hurst PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(bench_hurst PRIVATE /O2 /W3)
    if(SQT_NATIVE_ARCH)
        target_compile_options(bench_hurst PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(bench_hurst PRIVATE -O3 -Wall -Wextra)
    if(SQT_NATIVE_ARCH)
        target_compile_options(bench_hurst PRIVATE -march=native)
    endif()
endif()

add_executable(bench_backtest bench_backtest.cpp)
target_link_libraries(bench_backtest PRIVATE sqt_backtest_impl)
target_compile_features(bench_backtest PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(bench_backtest PRIVATE /O2 /W3)
    if(SQT_NATIVE_ARCH)
        target_compile_options(bench_backtest PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(bench_backtest PRIVATE -O3 -Wall -Wextra)
    if(SQT_NATIVE_ARCH)
        target_compile_options(bench_backtest PRIVATE -march=native)
    endif()
endif()
