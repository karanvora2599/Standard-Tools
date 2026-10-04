#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

// Shared numerical-robustness helpers used across the native kernels to
// replace ad-hoc fixed thresholds (e.g. `< 1e-14`) and unchecked
// size_t<->int narrowing with a single, documented convention.
namespace sqt::numerics {

// True when `x` is neither NaN nor +/-inf: exactly std::isfinite(x), for a
// per-element loop.
//
// WHY NOT std::isfinite. Under MSVC (cl, not clang-cl) the UCRT implements
// std::isfinite(double) as fpclassify(x) <= 0, and fpclassify as a call to
// _dclass, which is exported from the CRT DLL and is never inlined: every
// test is an indirect call across a DLL boundary, and the Windows x64 ABI
// lets the callee clobber xmm0-xmm5, so a loop keeps its state out of those
// registers around it. In the Wilder kernels, which test three inputs per
// bar, those calls were about a tenth of the kernel's time (CHANGELOG,
// 2026-10-01); replacing them in the per-bar and per-row loops of the
// Bollinger, stochastic, rolling-beta, portfolio, cross-sectional IC and
// standardization and implied-volatility kernels, and in
// is_negligible_pivot and clamp_near_zero_sumsq below, which two of those
// call per bar, made those kernels 1.03-2.5x faster (CHANGELOG,
// 2026-10-04). A test that runs once per call, column or date is left as
// std::isfinite: nothing measured a difference there. clang, GCC and
// clang-cl already inline std::isfinite; this is the same few instructions
// on all of them.
//
// A bit test, not `(x - x) == 0.0` or `std::abs(x) <= DBL_MAX`: those are
// floating-point identities a -ffast-math or -ffinite-math-only build may
// fold to `true`, and an integer comparison of the exponent field cannot be
// folded by any floating-point mode. The exponent is all ones exactly for
// +/-inf and every NaN, which is the definition of not finite.
inline bool is_finite(double x) noexcept {
    std::uint64_t bits;
    std::memcpy(&bits, &x, sizeof bits);
    return (bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL;
}

// True when `x` is a NaN, of either sign and any payload: exactly
// std::isnan(x), for a per-element loop, and inline for the reason
// is_finite is -- MSVC's std::isnan is the same _dclass call into the CRT
// DLL.
//
// The IEEE self-compare, not a bit test like is_finite's: it is one
// ucomisd, where the bit test first moves the value to an integer register,
// and apply_preprocess_stats measured 4-14% faster with it (CHANGELOG,
// 2026-10-02). It is no weaker than std::isnan, which a -ffinite-math-only
// build folds to `false` just the same; this module is never built that way
// (cusum.cpp relies on the same compare).
inline bool is_nan(double x) noexcept { return x != x; }

// Relative-epsilon singularity/pivot test, replacing fixed absolute
// thresholds like `< 1e-14` that don't scale with the input's magnitude.
// `scale` should be a magnitude representative of the *original*
// (pre-elimination) quantity the value is being judged against, so the test
// stays meaningful whether the entries are O(1) or O(1e12).
//
// The threshold is a PURE RATIO. It used to be floored via
// max(abs(scale), 1.0), on the reasoning that a well-conditioned
// small-magnitude system should not be rejected too aggressively -- but the
// floor did precisely the opposite of that intent. For any scale below 1 it
// replaces the relative test with an ABSOLUTE one at rel_eps, so the smaller
// (and therefore safer) the data, the more aggressive the rejection becomes.
// Measured: rolling_beta on x = [1e-8, 2e-8, ...], y = 2x returned NaN for a
// beta that is exactly 2, because W*Sxx = 5e-15 fell under the 1e-12 floor.
// The same analysis on the same data in different units gave different
// answers, which is the one thing a numerical tolerance must never do.
//
// A zero or non-finite scale carries no magnitude information to be relative
// TO, so any nonzero threshold there would be arbitrary; the test degrades to
// "is this exactly zero, or not finite?" rather than silently comparing
// against a fabricated unit scale.
inline bool is_negligible_pivot(double value, double scale, double rel_eps = 1e-12) {
    const double ref = std::abs(scale);
    if (!(ref > 0.0) || !is_finite(ref))
        return !is_finite(value) || value == 0.0;
    return std::abs(value) < rel_eps * ref;
}

// Guards a quantity that is mathematically guaranteed to be >= 0 (e.g. a
// sum of squares / residual sum of squares) but can drift slightly negative
// under floating-point cancellation. This is NOT a blind `max(x, 0)`: if
// the negative magnitude is negligible relative to `scale` (a representative
// magnitude of the terms that fed the subtraction, e.g. the largest
// raw-moment term), it is genuinely floating-point noise and is clamped to
// exactly 0.0. Otherwise the negativity is too large to be noise and
// indicates a real bug -- this throws so the bug surfaces instead of being
// silently hidden.
// Same pure-ratio convention as is_negligible_pivot above, and for the same
// reason: floating-point cancellation noise is proportional to the magnitude
// of the terms that cancelled, so the tolerance has to be too. A max(.., 1.0)
// floor made the tolerance ABSOLUTE for small-magnitude inputs, which quietly
// clamped away drift far larger than real noise on exactly the data where a
// genuine bug is hardest to see.
//
// A NON-FINITE value or scale is returned unchanged rather than thrown on.
// NaN/Inf here is not evidence of a bug in the kernel: it is the ordinary
// consequence of a NaN or Inf bar in the caller's input data, and this
// project's contract for bad data is NaN propagation, not an exception (see
// the validator note in bindings.cpp -- build_dataset's finite-value guard
// already rejects an entire panel over one bad print, which is exactly the
// failure mode a raise from this layer would reintroduce).
// Without this branch, control fell straight through to the throw, because
// `NaN >= 0.0` and `|NaN| < rel_eps*|NaN|` are BOTH false: measured, a
// single NaN price made bollinger_bands raise std::runtime_error for the
// whole series where every other indicator returned NaN for the affected
// bars and kept going.
inline double clamp_near_zero_sumsq(double value, double scale, const char* context,
                                     double rel_eps = 1e-9) {
    if (!is_finite(value) || !is_finite(scale)) return value;
    if (value >= 0.0) return value;
    if (std::abs(value) < rel_eps * std::abs(scale)) return 0.0;
    throw std::runtime_error(
        std::string(context) +
        ": sum-of-squares went unexpectedly negative (value=" + std::to_string(value) +
        ", scale=" + std::to_string(scale) +
        ") -- larger than floating-point noise, indicates a real bug.");
}

// Checked size_t -> int narrowing for the handful of call sites that
// genuinely need a signed `int` (e.g. MSVC OpenMP 2.0's canonical-loop-form
// requirement for signed induction variables). Throws instead of silently
// wrapping when `value` exceeds INT_MAX.
inline int checked_narrow_to_int(std::size_t value, const char* context) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string(context) + ": size " + std::to_string(value) +
                                   " exceeds INT_MAX.");
    }
    return static_cast<int>(value);
}

// Checked multiplication for allocation-size arithmetic (e.g.
// n_simulations * horizon_days). Throws on size_t overflow instead of
// silently under-allocating and corrupting memory.
inline std::size_t checked_mul(std::size_t a, std::size_t b, const char* context) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error(std::string(context) + ": size_t multiplication overflow.");
    }
    return a * b;
}

// Checked size_t -> long long narrowing, for the OpenMP loop bounds and the
// public result fields that are signed 64-bit rather than int.
inline long long checked_narrow_to_ll(std::size_t value, const char* context) {
    if (value > static_cast<std::size_t>(std::numeric_limits<long long>::max())) {
        throw std::overflow_error(std::string(context) + ": size " + std::to_string(value) +
                                   " exceeds LLONG_MAX.");
    }
    return static_cast<long long>(value);
}

}  // namespace sqt::numerics
