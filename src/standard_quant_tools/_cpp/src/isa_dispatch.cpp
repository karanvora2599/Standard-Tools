#include "sqt/isa_dispatch.hpp"

#include <atomic>

// ISA detection only makes sense (and only compiles) on x86/x64 -- CPUID
// and XGETBV are x86-specific instructions. On any other architecture
// (ARM/Apple Silicon, etc.) this whole detection path collapses to
// "AVX2 unavailable," which is correct: rolling_beta_avx2.cpp's own
// translation unit compiles to a portable non-AVX2 stub on non-x86 (see
// that file), so nothing there actually depends on this detection running.
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    #define SQT_ARCH_X86 1
    #include <intrin.h>
    #include <immintrin.h>  // _xgetbv
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
    #define SQT_ARCH_X86 1
    #include <cpuid.h>
#else
    #define SQT_ARCH_X86 0
#endif

namespace sqt {

namespace {

#if SQT_ARCH_X86 && !defined(_MSC_VER)
// Portable XGETBV via inline asm. Avoids requiring this whole translation
// unit be compiled with -mxsave just to call the _xgetbv intrinsic from
// <immintrin.h> -- this file must otherwise stay free of any -m*/ISA
// compile flags, since it's the code that decides whether it's safe to use
// ISA-specific code elsewhere (rolling_beta_avx2.cpp).
unsigned long long xgetbv0() {
    unsigned int eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
}
#endif

IsaFeatures detect_isa_features_real() {
    IsaFeatures f{false, false};

#if SQT_ARCH_X86
    bool osxsave = false;
    unsigned long long xcr0 = 0;

#if defined(_MSC_VER)
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    const int max_leaf = regs[0];
    if (max_leaf >= 1) {
        __cpuid(regs, 1);
        // FMA: ECX bit 12. OSXSAVE: ECX bit 27.
        f.fma = (regs[2] & (1 << 12)) != 0;
        osxsave = (regs[2] & (1 << 27)) != 0;
    }
    if (max_leaf >= 7) {
        __cpuidex(regs, 7, 0);
        // AVX2: EBX bit 5.
        f.avx2 = (regs[1] & (1 << 5)) != 0;
    }
    if (osxsave) xcr0 = _xgetbv(0);
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax, ebx, ecx, edx;
    unsigned int max_leaf = __get_cpuid_max(0, nullptr);
    if (max_leaf >= 1 && __get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        f.fma = (ecx & (1u << 12)) != 0;
        osxsave = (ecx & (1u << 27)) != 0;
    }
    if (max_leaf >= 7 && __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        f.avx2 = (ebx & (1u << 5)) != 0;
    }
    if (osxsave) xcr0 = xgetbv0();
#endif

    // CPUID can report AVX2/FMA hardware support even when the OS hasn't
    // enabled AVX register-state saving (rare but real -- some hypervisors/
    // sandboxes disable it). XCR0 bits 1 (SSE) and 2 (AVX) must both be set
    // for the OS to actually preserve the wide YMM registers across context
    // switches; using AVX2 instructions without that support is undefined
    // behavior, not a graceful fallback.
    const bool os_avx_state = osxsave && ((xcr0 & 0x6ULL) == 0x6ULL);
    if (!os_avx_state) {
        f.avx2 = false;
        f.fma = false;
    }
#endif  // SQT_ARCH_X86

    // AVX2 without FMA is possible on real hardware but not useful for the
    // fused-multiply-add kernel this dispatch targets -- treat "avx2" here
    // as "AVX2 and FMA both available," which is what rolling_beta_avx2.cpp
    // actually requires.
    f.avx2 = f.avx2 && f.fma;
    return f;
}

// Override state for testing: independent atomics for the active flag and
// each forced feature bit, so the {active, avx2, fma} triple can be read
// without ever exposing a plain (non-atomic) struct to concurrent access --
// unlike a single non-atomic IsaFeatures paired with only an atomic
// "active" flag, which is a data race a thread-safety analysis (e.g.
// ThreadSanitizer) would flag even though today's tests only call this
// sequentially.
std::atomic<bool> g_override_active{false};
std::atomic<bool> g_override_avx2{false};
std::atomic<bool> g_override_fma{false};

}  // namespace

IsaFeatures detect_isa_features() {
    static const IsaFeatures real = detect_isa_features_real();
    if (g_override_active.load(std::memory_order_acquire)) {
        return IsaFeatures{g_override_avx2.load(std::memory_order_acquire),
                            g_override_fma.load(std::memory_order_acquire)};
    }
    return real;
}

void force_isa_features_for_testing(IsaFeatures f) {
    // Apply the SAME conflation detect_isa_features_real() applies: "avx2"
    // throughout this project means "AVX2 and FMA both usable", because
    // rolling_beta_avx2.cpp's kernel issues _mm256_fmadd_pd. Without this,
    // forcing {avx2=true, fma=false} would route rolling_beta_into into that
    // kernel on a CPU the override just declared has no FMA -- an illegal
    // instruction, from the very function whose job is to prevent one.
    const bool avx2_usable = f.avx2 && f.fma;
    g_override_avx2.store(avx2_usable, std::memory_order_relaxed);
    g_override_fma.store(f.fma, std::memory_order_relaxed);
    // Release-store the active flag last so it publishes the two stores
    // above to any thread that acquire-loads it in detect_isa_features().
    g_override_active.store(true, std::memory_order_release);
}

void reset_isa_features_override_for_testing() {
    g_override_active.store(false, std::memory_order_release);
}

}  // namespace sqt
