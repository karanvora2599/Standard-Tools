#include "sqt/fp_contract.hpp"  // first: no contraction in this unit
#include "sqt/regimes.hpp"

#include "sqt/numerics.hpp"

#include <cmath>
#include <cstddef>

namespace sqt {
namespace {

// numpy's float64 `maximum` against a positive constant floor: a NaN
// propagates, otherwise the larger. A tie returns the floor, which is the
// same double, so which argument a tie returns cannot show.
inline double np_maximum(double a, double floor) {
    const double larger = (a > floor) ? a : floor;
    return (a != a) ? a : larger;
}

// math.pi, and 2 * math.pi as Python forms it (a doubling, so exact).
constexpr double kTwoPi = 2.0 * 3.141592653589793;

// np.allclose(new, old, atol=1e-10) at its default rtol, element by element
// as numpy 2's isclose forms it: |new - old| <= atol + rtol * |old| where
// old is finite, or new == old.
inline bool np_isclose(double now, double before) {
    constexpr double kAtol = 1e-10;
    constexpr double kRtol = 1e-05;
    const bool within = std::fabs(now - before) <= kAtol + kRtol * std::fabs(before);
    return (within && numerics::is_finite(before)) || now == before;
}

}  // namespace

bool regime_em_step(const double* SQT_RESTRICT values, std::size_t n,
                    const double* SQT_RESTRICT exponentials,
                    const double* SQT_RESTRICT means,
                    const double* SQT_RESTRICT variances,
                    const double* SQT_RESTRICT weights, std::size_t k,
                    double* SQT_RESTRICT responsibility,
                    double* SQT_RESTRICT counts,
                    double* SQT_RESTRICT new_means,
                    double* SQT_RESTRICT new_variances,
                    double* SQT_RESTRICT next_exponents,
                    double* SQT_RESTRICT totals) {
    if (k == 0) return true;

    // r[j][i] = weights[j] * (e[j][i] / sqrt(2 * pi * variances[j])): the
    // Python's `weights[k] * (np.exp(...) / math.sqrt(...))`, one row per
    // regime. Elementwise, so the compiler is free to vectorize it.
    for (std::size_t j = 0; j < k; ++j) {
        const double scale = std::sqrt(kTwoPi * variances[j]);
        const double weight = weights[j];
        const double* SQT_RESTRICT e = exponentials + j * n;
        double* SQT_RESTRICT r = responsibility + j * n;
        for (std::size_t i = 0; i < n; ++i) r[i] = weight * (e[i] / scale);
    }

    // Each observation's total over the regimes, left to right from 0.0 as
    // numpy's sum over axis 1 adds them; 1e-300 where it is 0.
    for (std::size_t i = 0; i < n; ++i) totals[i] = 0.0 + responsibility[i];
    for (std::size_t j = 1; j < k; ++j) {
        const double* SQT_RESTRICT r = responsibility + j * n;
        for (std::size_t i = 0; i < n; ++i) totals[i] = totals[i] + r[i];
    }
    for (std::size_t i = 0; i < n; ++i)
        totals[i] = (totals[i] == 0.0) ? 1e-300 : totals[i];
    for (std::size_t j = 0; j < k; ++j) {
        double* SQT_RESTRICT r = responsibility + j * n;
        for (std::size_t i = 0; i < n; ++i) r[i] = r[i] / totals[i];
    }

    // The sums over observations, each in observation order from 0.0 as
    // numpy's sum over axis 0 accumulates them. Regimes are independent
    // chains, so they advance together.
    double count[8];
    double weighted[8];
    for (std::size_t j = 0; j < k; ++j) {
        count[j] = 0.0;
        weighted[j] = 0.0;
    }
    for (std::size_t i = 0; i < n; ++i) {
        const double x = values[i];
        for (std::size_t j = 0; j < k; ++j) {
            const double r = responsibility[j * n + i];
            count[j] += r;
            weighted[j] += r * x;
        }
    }
    double denominator[8];
    double mean[8];
    double spread[8];
    for (std::size_t j = 0; j < k; ++j) {
        counts[j] = count[j];
        denominator[j] = np_maximum(count[j], 1e-12);
        mean[j] = weighted[j] / denominator[j];
        new_means[j] = mean[j];
        spread[j] = 0.0;
    }
    for (std::size_t i = 0; i < n; ++i) {
        const double x = values[i];
        for (std::size_t j = 0; j < k; ++j) {
            const double d = x - mean[j];
            spread[j] += responsibility[j * n + i] * (d * d);
        }
    }
    bool converged = true;
    for (std::size_t j = 0; j < k; ++j) {
        new_variances[j] = np_maximum(spread[j] / denominator[j], 1e-12);
        converged = np_isclose(mean[j], means[j]) && converged;
    }

    // The next step's argument to np.exp: -0.5 * (x - mean) ** 2 / variance.
    for (std::size_t j = 0; j < k; ++j) {
        const double m = mean[j];
        const double variance = new_variances[j];
        double* SQT_RESTRICT out = next_exponents + j * n;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = values[i] - m;
            out[i] = (-0.5 * (d * d)) / variance;
        }
    }
    return converged;
}

}  // namespace sqt
