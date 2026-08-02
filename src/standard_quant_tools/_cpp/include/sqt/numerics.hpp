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
// (pre-elimination) matrix -- e.g. the max abs diagonal entry before
// Gaussian elimination or Cholesky decomposition touches it -- so the test
// stays meaningful whether the matrix entries are O(1) or O(1e12). Floored
// at 1.0 so a genuinely well-conditioned small-magnitude system isn't
// rejected too aggressively.
inline bool is_negligible_pivot(double value, double scale, double rel_eps = 1e-12) {
    const double ref = std::max(std::abs(scale), 1.0);
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
inline double clamp_near_zero_sumsq(double value, double scale, const char* context,
                                     double rel_eps = 1e-9) {
    if (value >= 0.0) return value;
    if (std::abs(value) < rel_eps * std::max(std::abs(scale), 1.0)) return 0.0;
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

inline bool all_finite(const double* arr, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(arr[i])) return false;
    }
    return true;
}

}  // namespace sqt::numerics
