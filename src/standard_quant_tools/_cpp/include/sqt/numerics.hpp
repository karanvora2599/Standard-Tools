#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

// Shared numerical-robustness helpers used across the native kernels to
// replace ad-hoc fixed thresholds (e.g. `< 1e-14`) and unchecked
// size_t<->int narrowing with a single, documented convention.
namespace sqt::numerics {

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
    if (!(ref > 0.0) || !std::isfinite(ref))
        return !std::isfinite(value) || value == 0.0;
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
inline double clamp_near_zero_sumsq(double value, double scale, const char* context,
                                     double rel_eps = 1e-9) {
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

inline bool all_finite(const double* arr, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(arr[i])) return false;
    }
    return true;
}

}  // namespace sqt::numerics
