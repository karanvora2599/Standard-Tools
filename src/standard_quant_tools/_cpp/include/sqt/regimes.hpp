#pragma once

#include "sqt/platform.hpp"

#include <cstddef>

namespace sqt {

/**
 * One EM step of the Gaussian mixture `analysis.stationarity.detect_regimes`
 * fits, after the exponentials.
 *
 * WHY THIS IS NATIVE, AND WHY ONLY THIS. The fit runs 100 iterations on
 * every series it has been measured on (its convergence test compares the
 * means to 1e-10), and each iteration is about 8k + 10 numpy calls over
 * the n x k responsibilities: 23-60 ms on 2,000-5,000 daily returns with
 * 2-4 regimes, 87-95% of the regime tool's call when its prices come from
 * the disk cache. The exponentials stay in numpy, one `np.exp` per regime
 * as the Python path calls it: numpy's float64 exp is its own SIMD routine
 * on some x86 machines and the C library's elsewhere, so a C++ exp would
 * match it on some machines and not others. Everything around it is here,
 * in one call per step.
 *
 * THE SAME ARITHMETIC, NOT A SIMILAR ONE. With e[j][i] the exponential of
 * observation i under regime j, and s_j = sqrt(2 * pi * variances[j]):
 *
 *     r[j][i]  = weights[j] * (e[j][i] / s_j)
 *     t_i      = 0.0 + r[0][i] + r[1][i] + ...   (left to right; 1e-300 if 0)
 *     r[j][i]  = r[j][i] / t_i
 *     counts_j = sum_i r[j][i]                   (in observation order)
 *     means_j  = sum_i (r[j][i] * x[i]) / maximum(counts_j, 1e-12)
 *     vars_j   = maximum(sum_i (r[j][i] * (x[i] - means_j)^2)
 *                        / maximum(counts_j, 1e-12), 1e-12)
 *
 * which is the order numpy evaluates the Python expressions in for 2 <= k
 * <= 7: a sum over axis 0 of a C-ordered (n, k) array accumulates row by
 * row from 0.0, and a sum over axis 1 of fewer than 8 values adds them left
 * to right (measured on numpy 2.0 and 2.4 against explicit loops, n up to
 * 100,003). With k == 1 the (n, 1) column is contiguous and numpy sums it
 * pairwise, and 8 or more values across a row are summed pairwise too, so
 * the binding refuses those; detect_regimes fits 2 to 5. The square is
 * d * d, as numpy squares for `** 2`, and the unit is compiled without
 * contraction. The step is therefore the Python step's bit for bit.
 *
 * It also answers the loop's convergence test, np.allclose(new means, old
 * means, atol=1e-10), as numpy's isclose forms it: |new - old| <= 1e-10 +
 * 1e-5 * |old| where old is finite, or new == old, for every regime. And it
 * writes the next step's exponents from the new means and variances,
 * (-0.5 * (d * d)) / vars_j with d = x[i] - means_j, the argument the Python
 * hands np.exp -- so the caller's loop is one np.exp per regime and one
 * call here per step.
 *
 * Layouts: `exponentials`, `responsibility` and `next_exponents` are
 * row-major (k, n), one contiguous row per regime (the transpose of the
 * Python's (n, k) responsibilities, with the same values). `totals` is n
 * doubles of scratch. Nothing is allocated; k is at most 8 (the binding
 * allows 2 to 7).
 *
 * @return whether every new mean is close to the old one.
 */
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
                    double* SQT_RESTRICT totals);

}  // namespace sqt
