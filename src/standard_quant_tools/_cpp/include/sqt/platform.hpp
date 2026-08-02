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
