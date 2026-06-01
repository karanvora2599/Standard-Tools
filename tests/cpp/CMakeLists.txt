add_executable(test_hurst test_hurst.cpp)

target_include_directories(test_hurst PRIVATE
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/include
)

# Compile the implementation directly into the test binary (no shared library)
target_sources(test_hurst PRIVATE
    ${CMAKE_SOURCE_DIR}/src/standard_quant_tools/_cpp/src/hurst.cpp
)

target_compile_features(test_hurst PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(test_hurst PRIVATE /O2 /W3 /wd4244 /wd4267)
else()
    target_compile_options(test_hurst PRIVATE -O2 -Wall -Wextra)
endif()

add_test(NAME cpp_hurst COMMAND test_hurst)
