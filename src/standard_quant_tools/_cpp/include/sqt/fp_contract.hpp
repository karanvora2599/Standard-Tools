#pragma once

// No floating-point contraction anywhere in the extension.
//
// Every translation unit includes this header FIRST, before its own header
// and before any standard header, so the pragma below governs every function
// the unit compiles -- its own, the inline helpers its headers define, and
// the templates it instantiates.
//
// WHY. Contraction fuses a*b + c into one fused multiply-add, which rounds
// once where the source rounds twice. The kernels here are held to Python and
// pandas arithmetic bit for bit, and the fused result differs in the last
// bit often enough to matter: a clang-cl build, which contracts within an
// expression by default even under /fp:precise, changed 27 of 44 sampled
// kernel outputs and failed 41 tests. FMA instructions belong only where a
// kernel asks for them by intrinsic (rolling_beta_avx2.cpp, whose result is
// documented as agreeing with the scalar path to twelve digits, not bit for
// bit); the pragma does not touch intrinsics.
//
// WHY A PRAGMA AS WELL AS THE FLAGS. _cpp/CMakeLists.txt turns contraction
// off on the command line of every unit (/fp:precise for cl,
// /clang:-ffp-contract=off for clang-cl, -ffp-contract=off for GCC and
// Clang). A flag, though, is only as good as the last flag that follows it.
// Measured with cl 19.44: a global /fp:contract is NOT undone by a later
// /fp:precise (11 FMAs either way), while this pragma, in a header included
// first, leaves none. Under clang-cl the pragma alone also leaves none, and
// a plain -ffp-contract=off is silently ignored there -- the driver only
// hears it as /clang:-ffp-contract=off.
//
// What neither can stop: -ffp-contract=fast (or /fp:fast) given after the
// project's own flags, which by definition disregards the pragma. That is a
// build asking for different arithmetic, and its results are its own.
//
// GCC has no pragma for this in C++ (`#pragma STDC FP_CONTRACT` is
// unimplemented and draws -Wunknown-pragmas under -Wall), so it relies on
// -ffp-contract=off alone.
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif
