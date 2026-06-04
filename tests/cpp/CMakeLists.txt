# ── Shared helper: compile hurst.cpp once and link both binaries ──────────────

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


# ── Performance benchmarks ────────────────────────────────────────────────────
# Not a CTest test — run manually to inspect timing output:
#   Windows  (VS gen) : build\tests\cpp\Release\bench_hurst.exe
#   Windows  (Ninja)  : build\tests\cpp\bench_hurst.exe
#   Linux / macOS     : ./build/tests/cpp/bench_hurst

add_executable(bench_hurst bench_hurst.cpp)
target_link_libraries(bench_hurst PRIVATE sqt_hurst_impl)
target_compile_features(bench_hurst PRIVATE cxx_std_17)
if(MSVC)
    target_compile_options(bench_hurst PRIVATE /O2 /arch:AVX2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(bench_hurst PRIVATE -O3 -march=native -Wall -Wextra)
endif()
