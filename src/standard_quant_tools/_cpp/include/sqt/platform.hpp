#pragma once

// Portable `restrict` qualifier. Not part of standard C++, but supported as
// a vendor extension by every compiler this project targets (MSVC, GCC,
// Clang). A `SQT_RESTRICT`-qualified pointer parameter is a promise to the
// compiler that, for the duration of the call, no write through that
// pointer aliases any read through any other pointer parameter -- letting
// the optimizer reorder/vectorize loads and stores it would otherwise have
// to keep in program order defensively. Apply it ONLY where that
// non-aliasing contract is genuinely guaranteed by the call site (see each
// function's own doc comment for why); applying it where aliasing is
// actually possible is undefined behavior, not just a missed optimization.
#if defined(_MSC_VER)
    #define SQT_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define SQT_RESTRICT __restrict__
#else
    #define SQT_RESTRICT
#endif

// Portable "never inline this into a caller" qualifier.
//
// Used for exactly one thing here: a function whose body is compiled with
// an ISA-specific flag (rolling_beta_avx2.cpp, /arch:AVX2) must not be
// inlined into a caller compiled without it. Release builds enable link-time
// optimization, and inlining across translation units is precisely what LTO
// does, so a per-source compile flag confines CODEGEN to that file but says
// nothing about INLINING.
//
// Measured, not assumed -- and the measurement says this is insurance, not a
// live bug. Building with SQT_NATIVE_ARCH=OFF (so only that one file gets
// /arch:AVX2) and disassembling the linked module gives exactly 2 vfmadd
// instructions, the two the kernel itself issues, both with and without this
// qualifier: MSVC 19.44 LTCG leaves the body where it was compiled. What it
// does not do is PROMISE to, and nothing else in the build enforces the
// isolation the file's own header comment claims. The cost is one
// out-of-line call per rolling window, against a body that loops over the
// whole window. Untested on GCC/Clang LTO, which is a separate optimizer
// making its own choices.
#if defined(_MSC_VER)
    #define SQT_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
    #define SQT_NOINLINE __attribute__((noinline))
#else
    #define SQT_NOINLINE
#endif
