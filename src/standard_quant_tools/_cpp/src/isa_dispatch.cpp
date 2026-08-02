#include "sqt/isa_dispatch.hpp"

#include <atomic>

#if defined(_MSC_VER)
    #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h>
#endif

namespace sqt {

namespace {

IsaFeatures detect_isa_features_real() {
    IsaFeatures f{false, false};

#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    const int max_leaf = regs[0];
    if (max_leaf >= 1) {
        __cpuid(regs, 1);
        // FMA: ECX bit 12.
        f.fma = (regs[2] & (1 << 12)) != 0;
    }
    if (max_leaf >= 7) {
        __cpuidex(regs, 7, 0);
        // AVX2: EBX bit 5.
        f.avx2 = (regs[1] & (1 << 5)) != 0;
    }
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    unsigned int max_leaf = __get_cpuid_max(0, nullptr);
    if (max_leaf >= 1 && __get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        f.fma = (ecx & (1u << 12)) != 0;
    }
    if (max_leaf >= 7 && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        f.avx2 = (ebx & (1u << 5)) != 0;
    }
#endif
    // AVX2 without FMA is possible on real hardware but not useful for the
    // fused-multiply-add kernel this dispatch targets -- treat "avx2" here
    // as "AVX2 and FMA both available," which is what rolling_beta_avx2.cpp
    // actually requires.
    f.avx2 = f.avx2 && f.fma;
    return f;
}

// Override state for testing: an atomic flag plus the forced value. Not a
// magic-static like the real detection below, since it needs to be
// settable/resettable at runtime by tests, not computed once and frozen.
std::atomic<bool> g_override_active{false};
IsaFeatures        g_override_value{false, false};

}  // namespace

const IsaFeatures& detect_isa_features() {
    static const IsaFeatures real = detect_isa_features_real();
    if (g_override_active.load(std::memory_order_acquire)) {
        return g_override_value;
    }
    return real;
}

void force_isa_features_for_testing(IsaFeatures f) {
    g_override_value = f;
    g_override_active.store(true, std::memory_order_release);
}

void reset_isa_features_override_for_testing() {
    g_override_active.store(false, std::memory_order_release);
}

}  // namespace sqt
