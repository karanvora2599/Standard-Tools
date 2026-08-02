#pragma once

#include <cstddef>

namespace sqt {

// AVX2+FMA implementation of the 4-accumulator reduction
// rolling_beta_into's recompute_window step needs (item L: runtime ISA
// dispatch demo). Lives in its own translation unit (rolling_beta_avx2.cpp)
// compiled unconditionally with AVX2+FMA codegen enabled, independent of
// the opt-in SQT_NATIVE_ARCH flag -- MSVC has no per-function ISA-target
// attribute (unlike GCC/Clang's __attribute__((target(...)))), so AVX2
// intrinsics require the whole containing translation unit to be compiled
// with /arch:AVX2. Callers MUST check detect_isa_features().avx2 first
// (isa_dispatch.hpp) and only call this when true -- calling it on a CPU
// without AVX2+FMA is an illegal-instruction crash, not a graceful
// fallback.
//
// NOT bit-identical to the scalar accumulation in rolling_regression.cpp's
// recompute_window: SIMD lane accumulation reorders the summation
// (floating-point addition isn't associative) -- verified via a tolerance
// gate in tests/test_cpp_regression.py, not assumed.
//
// @param x, y     Full series arrays (same ones rolling_beta_into received).
// @param start    First index of the window (inclusive).
// @param window   Window length.
// @param cx, cy   Per-window reference points (x[start]/y[start]) already
//                 subtracted from every element before accumulating, same
//                 as the scalar path -- large-baseline catastrophic
//                 cancellation protection carries over unchanged.
// @param Sx, Sy, Sxy, Sxx  Output accumulators (overwritten, not added to).
void rolling_beta_reduce_avx2(
    const double* x,
    const double* y,
    std::size_t   start,
    int           window,
    double        cx,
    double        cy,
    double&       Sx,
    double&       Sy,
    double&       Sxy,
    double&       Sxx);

}  // namespace sqt
