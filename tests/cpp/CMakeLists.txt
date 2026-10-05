# ── The C++ suite compiles what the extension ships ──────────────────────────
#
# Every library and executable here goes through sqt_test_target(), which
# applies the extension's own codegen -- no floating-point contraction, the
# SQT_NATIVE_ARCH instruction set, LTO in Release -- through
# sqt_native_codegen(), defined in src/standard_quant_tools/_cpp/CMakeLists.txt
# (processed before this file). These libraries used to compile the same
# sources at baseline ISA with no LTO whatever the extension was built with,
# so a defect only the shipped codegen has could not fail this suite.
#
# Optimization follows the extension's too: -O3 where it builds at -O3
# (Release, RelWithDebInfo), -O2 otherwise, so the Debug sanitizer and TSan
# jobs keep the optimization they have always run with. MSVC is /O2 in every
# configuration, as before.
#
# A clang-cl build links LLVM's libomp.dll (see the extension's OpenMP
# section), which Windows will not find from the directory the test binaries
# run in, so one copy is put there. One target does it, and every executable
# depends on it: a copy per executable ran in parallel under Ninja and the
# copies collided on the same file.
if(SQT_LIBOMP_DLL)
    add_custom_target(sqt_test_openmp_runtime
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${SQT_LIBOMP_DLL}" "$<TARGET_FILE_DIR:test_hurst>"
        VERBATIM)
endif()

function(sqt_test_target target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /O2 /W3)
    else()
        target_compile_options(${target} PRIVATE
            $<IF:$<CONFIG:Release,RelWithDebInfo>,-O3,-O2> -Wall -Wextra)
    endif()
    target_compile_features(${target} PUBLIC cxx_std_17)
    sqt_native_codegen(${target})
    get_target_property(_type ${target} TYPE)
    if(_type STREQUAL "EXECUTABLE" AND SQT_LIBOMP_DLL)
        add_dependencies(${target} sqt_test_openmp_runtime)
    endif()
endfunction()

# One kernel library: its sources compiled once, linked to its test (and
# bench), with OpenMP when the kernel has a parallel path, so the suite
# exercises the parallel code -- including the bit-identical-to-serial gates
# -- rather than only the serial fallback. Without an OpenMP runtime the
# compiler leaves _OPENMP undefined and the identical serial loop runs.
find_package(OpenMP)
function(sqt_kernel_library name openmp)
    set(_sources "")
    foreach(_src IN LISTS ARGN)
        list(APPEND _sources ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/${_src})
    endforeach()
    add_library(${name} STATIC ${_sources})
    target_include_directories(${name} PUBLIC
        ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
    )
    sqt_test_target(${name})
    if(openmp AND OpenMP_CXX_FOUND)
        target_link_libraries(${name} PUBLIC OpenMP::OpenMP_CXX)
    endif()
endfunction()

function(sqt_cpp_test name library)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE ${library} ${ARGN})
    sqt_test_target(${name})
endfunction()


# ── Kernel libraries ─────────────────────────────────────────────────────────

# rolling_hurst_into runs each window in parallel.
sqt_kernel_library(sqt_hurst_impl ON hurst.cpp)
sqt_kernel_library(sqt_indicators_impl OFF indicators.cpp)
sqt_kernel_library(sqt_panel_stats_impl OFF panel_stats.cpp)
# batch_engle_granger runs each pair in parallel.
sqt_kernel_library(sqt_cointegration_impl ON cointegration.cpp)
# batch_run_strategy runs each test index in parallel.
sqt_kernel_library(sqt_backtest_impl ON backtest.cpp)
# simulate_forward_paths runs each path in parallel.
sqt_kernel_library(sqt_monte_carlo_impl ON monte_carlo.cpp)
sqt_kernel_library(sqt_garch_impl OFF garch.cpp)
sqt_kernel_library(sqt_signals_impl OFF signal_state_machines.cpp)
sqt_kernel_library(sqt_rolling_regression_impl OFF
    rolling_regression.cpp isa_dispatch.cpp rolling_beta_avx2.cpp)
# The batch-equals-one-contract test exercises the parallel chain path.
sqt_kernel_library(sqt_options_impl ON options.cpp)
sqt_kernel_library(sqt_cusum_impl OFF cusum.cpp)
# The pair loop runs in parallel; the test holds it to one thread's answer.
sqt_kernel_library(sqt_correlation_impl ON correlation.cpp)
sqt_kernel_library(sqt_order_events_impl OFF order_events.cpp)
sqt_kernel_library(sqt_regimes_impl OFF regimes.cpp)

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


# ── Unit tests ────────────────────────────────────────────────────────────────

sqt_cpp_test(test_hurst sqt_hurst_impl)
add_test(NAME cpp_hurst COMMAND test_hurst)

sqt_cpp_test(test_panel_stats sqt_panel_stats_impl)
add_test(NAME cpp_panel_stats COMMAND test_panel_stats)

# omp_policy.hpp is header-only. Built without OpenMP on purpose: the share
# arithmetic is then driven by SQT_NUM_THREADS alone, which the test sets.
# Threads for the callers it runs at once.
find_package(Threads REQUIRED)
sqt_cpp_test(test_omp_policy Threads::Threads)
target_include_directories(test_omp_policy PRIVATE
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)
add_test(NAME cpp_omp_policy COMMAND test_omp_policy)

sqt_cpp_test(test_indicators sqt_indicators_impl)
add_test(NAME cpp_indicators COMMAND test_indicators)

sqt_cpp_test(test_cointegration sqt_cointegration_impl)
add_test(NAME cpp_cointegration COMMAND test_cointegration)

sqt_cpp_test(test_backtest sqt_backtest_impl)
add_test(NAME cpp_backtest COMMAND test_backtest)

sqt_cpp_test(test_monte_carlo sqt_monte_carlo_impl)
add_test(NAME cpp_monte_carlo COMMAND test_monte_carlo)

sqt_cpp_test(test_garch sqt_garch_impl)
add_test(NAME cpp_garch COMMAND test_garch)

sqt_cpp_test(test_signals sqt_signals_impl)
add_test(NAME cpp_signals COMMAND test_signals)

sqt_cpp_test(test_rolling_regression sqt_rolling_regression_impl)
add_test(NAME cpp_rolling_regression COMMAND test_rolling_regression)

# The tests pin the kernel to exact values; the no-contraction rule they
# depend on is sqt_native_codegen()'s, as for every target here.
sqt_cpp_test(test_options sqt_options_impl)
add_test(NAME cpp_options COMMAND test_options)

# The reference loop in the test is held to the kernel's no-contraction rule
# by the same codegen.
sqt_cpp_test(test_cusum sqt_cusum_impl)
add_test(NAME cpp_cusum COMMAND test_cusum)

# The reference here is pandas' nancorr loop, written out; it and the kernel
# compile under one no-contraction rule.
sqt_cpp_test(test_correlation sqt_correlation_impl)
add_test(NAME cpp_correlation COMMAND test_correlation)

sqt_cpp_test(test_order_events sqt_order_events_impl)
add_test(NAME cpp_order_events COMMAND test_order_events)

sqt_cpp_test(test_regimes sqt_regimes_impl)
add_test(NAME cpp_regimes COMMAND test_regimes)

# ── Fuzz/property harness (correctness/portability pass item 20a) ────────────
# Randomized-input stress test for cointegration.cpp's gauss_elim (via
# ols2/adf_test/engle_granger) and rolling_regression.cpp's cholesky_solve
# (via rolling_factor_loadings) -- fixed seed for deterministic default CI
# runs. Registered as a normal ctest, so it automatically gets ASan/UBSan
# coverage for free via the existing build-and-test-sanitizers CI job.
sqt_cpp_test(fuzz_cointegration sqt_cointegration_impl sqt_rolling_regression_impl)
add_test(NAME cpp_fuzz_cointegration COMMAND fuzz_cointegration)


# ── Performance benchmarks ────────────────────────────────────────────────────
# Not CTest tests — run manually to inspect timing output:
#   Windows  (VS gen) : build\tests\cpp\Release\bench_hurst.exe
#                       build\tests\cpp\Release\bench_backtest.exe
#   Windows  (Ninja)  : build\tests\cpp\bench_hurst.exe
#                       build\tests\cpp\bench_backtest.exe
#   Linux / macOS     : ./build/tests/cpp/bench_hurst
#                       ./build/tests/cpp/bench_backtest
#
# Built with the extension's codegen like everything above, so SQT_NATIVE_ARCH
# (opt-in, default OFF) decides their instruction set too -- same portability
# rationale: a bench binary built for AVX2 isn't guaranteed to run on a
# different CPU.

sqt_cpp_test(bench_hurst sqt_hurst_impl)
sqt_cpp_test(bench_backtest sqt_backtest_impl)
