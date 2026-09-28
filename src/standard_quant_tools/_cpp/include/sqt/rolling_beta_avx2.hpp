#pragma once

#include "sqt/platform.hpp"

#include <cstddef>

namespace sqt {

// AVX2+FMA implementation of the 5-accumulator reduction
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
// @param cx, cy   Per-window reference points (the window's newest x and y,
//                 or 0.0 where that value is not finite) already
//                 subtracted from every element before accumulating, same
//                 as the scalar path -- large-baseline catastrophic
//                 cancellation protection carries over unchanged.
// @param Sx, Sy, Sxy, Sxx, Syy  Output accumulators (overwritten, not added
//                 to). Syy is not part of beta: rolling_beta_into watches it,
//                 with Sxx, to tell when a departed outlier has cost the
//                 sliding sums enough digits that they must be rebuilt.
//
// SQT_NOINLINE is not a tuning hint. Release builds enable link-time
// optimization, which inlines across translation units, so nothing in the
// BUILD stops this AVX2 body being hoisted into rolling_beta_into -- which
// is compiled without /arch:AVX2 and is reached with no CPUID check in front
// of it. That would turn the runtime dispatch into decoration and the
// "graceful fallback on an older CPU" into an illegal-instruction crash.
//
// Measured on MSVC 19.44 with SQT_NATIVE_ARCH=OFF, that hoist does NOT
// happen: the linked module contained exactly the vfmadd instructions this
// kernel issued (two at the time), with or without the qualifier. So this is insurance against
// something the toolchain is permitted to do and currently does not, kept
// because the cost is one out-of-line call per window and the alternative is
// relying on an optimizer's present-day choice for a memory-safety property.
SQT_NOINLINE void rolling_beta_reduce_avx2(
    const double* x,
    const double* y,
    std::size_t   start,
    int           window,
    double        cx,
    double        cy,
    double&       Sx,
    double&       Sy,
    double&       Sxy,
    double&       Sxx,
    double&       Syy);

}  // namespace sqt
