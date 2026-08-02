#pragma once

namespace sqt {

// Lazily-detected CPU feature flags, used to select between a portable
// scalar kernel and an ISA-specific one at runtime -- so a single wheel/
// build can safely use AVX2 where the actual runtime CPU supports it,
// without baking in an assumption about the build machine's own CPU the
// way the opt-in SQT_NATIVE_ARCH compile flag does.
struct IsaFeatures {
    bool avx2;
    bool fma;
};

// Thread-safe (C++11 magic static), computed once on first call via CPUID
// (plus an OSXSAVE/XGETBV check that the OS has actually enabled AVX
// register-state saving -- CPUID alone can report hardware support even
// when it hasn't). Always returns {false, false} on non-x86 architectures
// (ARM/Apple Silicon, etc.), where CPUID/XGETBV don't exist. Returned by
// value (not by reference) since the struct is two bools -- trivially
// cheap to copy, and by-value avoids ever exposing a reference into
// mutable override state shared across threads.
// Real detection cannot be exercised on a non-AVX2 machine in this
// project's own CI/dev environment -- see force_isa_features_for_testing()
// below for the only practical way to test the scalar-fallback path here.
IsaFeatures detect_isa_features();

// Test-only override: forces detect_isa_features() to return `f` for the
// remainder of the process, regardless of what CPUID actually reports.
// This is the only practical way to exercise the "runs correctly on a
// non-AVX2 CPU" code path from a test running on an actually-AVX2-capable
// machine (physical access to a non-AVX2 CPU isn't available in this
// project's CI/dev environment). Not for production use.
void force_isa_features_for_testing(IsaFeatures f);

// Restores real CPUID-based detection after a force_isa_features_for_testing()
// call -- test cleanup.
void reset_isa_features_override_for_testing();

}  // namespace sqt
