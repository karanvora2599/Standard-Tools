#pragma once

#include <cstddef>
#include <vector>

namespace sqt {

/**
 * GARCH(1,1) conditional variance recursion.
 *
 *   sigma2[0] = max(mean(resid_sq), MIN_SIGMA2)
 *   sigma2[t] = max(omega + alpha*resid_sq[t-1] + beta*sigma2[t-1], MIN_SIGMA2)
 *               for t >= 1
 *
 * Matches _garch11_variance_recursion in analysis/garch.py exactly,
 * including the 1e-12 variance floor (kMinSigma2 below).
 *
 * @param resid_sq  Squared demeaned residuals, length n.
 * @param n         Number of observations.
 * @param omega, alpha, beta  GARCH(1,1) parameters. No bounds are enforced
 *   here -- scipy.optimize's L-BFGS-B bounds already constrain these before
 *   every call from Python, and the recursion is well-defined arithmetic
 *   for any finite input (it never indexes memory using these values), so
 *   there is nothing to guard beyond n==0.
 * @returns  Vector of length n. Empty if n==0.
 */
std::vector<double> garch11_variance_recursion(
    const double* resid_sq,
    std::size_t   n,
    double        omega,
    double        alpha,
    double        beta);

}  // namespace sqt
