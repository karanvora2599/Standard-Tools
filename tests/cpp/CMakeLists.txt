# ── Shared helpers: compile each .cpp once and link to test + bench ────────────

add_library(sqt_hurst_impl STATIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/hurst.cpp
)
target_include_directories(sqt_hurst_impl PUBLIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)
target_compile_features(sqt_hurst_impl PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(sqt_hurst_impl PRIVATE /O2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(sqt_hurst_impl PRIVATE -O2 -Wall -Wextra)
endif()

add_library(sqt_indicators_impl STATIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/indicators.cpp
)
target_include_directories(sqt_indicators_impl PUBLIC
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)
target_compile_features(sqt_indicators_impl PUBLIC cxx_std_17)
if(MSVC)
    target_compile_options(sqt_indicators_impl PRIVATE /O2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(sqt_indicators_impl PRIVATE -O2 -Wall -Wextra)
endif()


# ── Unit tests ────────────────────────────────────────────────────────────────

add_executable(test_hurst test_hurst.cpp)
target_link_libraries(test_hurst PRIVATE sqt_hurst_impl)
target_compile_features(test_hurst PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_hurst PRIVATE /O2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(test_hurst PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_hurst COMMAND test_hurst)

add_executable(test_indicators test_indicators.cpp)
target_link_libraries(test_indicators PRIVATE sqt_indicators_impl)
target_compile_features(test_indicators PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_indicators PRIVATE /O2 /W3 /wd4244 /wd4267)
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
    target_compile_options(sqt_cointegration_impl PRIVATE /O2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(sqt_cointegration_impl PRIVATE -O2 -Wall -Wextra)
endif()

add_executable(test_cointegration test_cointegration.cpp)
target_link_libraries(test_cointegration PRIVATE sqt_cointegration_impl)
target_compile_features(test_cointegration PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_cointegration PRIVATE /O2 /W3 /wd4244 /wd4267)
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
    target_compile_options(sqt_backtest_impl PRIVATE /O2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(sqt_backtest_impl PRIVATE -O2 -Wall -Wextra)
endif()

add_executable(test_backtest test_backtest.cpp)
target_link_libraries(test_backtest PRIVATE sqt_backtest_impl)
target_compile_features(test_backtest PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_backtest PRIVATE /O2 /W3 /wd4244 /wd4267)
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
    target_compile_options(sqt_monte_carlo_impl PRIVATE /O2 /W3 /wd4244 /wd4267)
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
    target_compile_options(test_monte_carlo PRIVATE /O2 /W3 /wd4244 /wd4267)
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
    target_compile_options(sqt_garch_impl PRIVATE /O2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(sqt_garch_impl PRIVATE -O2 -Wall -Wextra)
endif()

add_executable(test_garch test_garch.cpp)
target_link_libraries(test_garch PRIVATE sqt_garch_impl)
target_compile_features(test_garch PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_garch PRIVATE /O2 /W3 /wd4244 /wd4267)
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
    target_compile_options(sqt_signals_impl PRIVATE /O2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(sqt_signals_impl PRIVATE -O2 -Wall -Wextra)
endif()

add_executable(test_signals test_signals.cpp)
target_link_libraries(test_signals PRIVATE sqt_signals_impl)
target_compile_features(test_signals PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(test_signals PRIVATE /O2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(test_signals PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_signals COMMAND test_signals)


# ── Performance benchmarks ────────────────────────────────────────────────────
# Not CTest tests — run manually to inspect timing output:
#   Windows  (VS gen) : build\tests\cpp\Release\bench_hurst.exe
#                       build\tests\cpp\Release\bench_backtest.exe
#   Windows  (Ninja)  : build\tests\cpp\bench_hurst.exe
#                       build\tests\cpp\bench_backtest.exe
#   Linux / macOS     : ./build/tests/cpp/bench_hurst
#                       ./build/tests/cpp/bench_backtest

add_executable(bench_hurst bench_hurst.cpp)
target_link_libraries(bench_hurst PRIVATE sqt_hurst_impl)
target_compile_features(bench_hurst PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(bench_hurst PRIVATE /O2 /arch:AVX2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(bench_hurst PRIVATE -O3 -march=native -Wall -Wextra)
endif()

add_executable(bench_backtest bench_backtest.cpp)
target_link_libraries(bench_backtest PRIVATE sqt_backtest_impl)
target_compile_features(bench_backtest PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(bench_backtest PRIVATE /O2 /arch:AVX2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(bench_backtest PRIVATE -O3 -march=native -Wall -Wextra)
endif()
