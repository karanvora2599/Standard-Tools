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

/** Buffer-writing form of garch11_variance_recursion(). `out` must have
 *  length n. */
void garch11_variance_recursion_into(
    const double* resid_sq,
    std::size_t   n,
    double        omega,
    double        alpha,
    double        beta,
    double*       out);

/**
 * GARCH(1,1) negative log-likelihood -- fuses the variance recursion above
 * with the NLL reduction into a single native call, so a Python-side
 * scipy.optimize objective function evaluation never round-trips a full
 * sigma2 array across the Python/C++ boundary just to immediately reduce
 * it to one scalar. Matches _garch11_neg_loglik in analysis/garch.py
 * exactly, including the soft stationarity penalty:
 *
 *   sigma2[t] = same recursion as garch11_variance_recursion above
 *   nll = 0.5 * sum_t( log(2*pi) + log(sigma2[t]) + resid_sq[t]/sigma2[t] )
 *   if penalize and (alpha+beta) >= 1.0:
 *       nll += 1e6 * ((alpha+beta) - 1.0)^2
 *
 * @param resid_sq  Squared demeaned residuals, length n.
 * @param n         Number of observations.
 * @param omega, alpha, beta  GARCH(1,1) parameters (unconstrained here --
 *   see garch11_variance_recursion's docstring for why).
 * @param penalize  Whether to add the soft non-stationarity penalty.
 * @returns  0.0 if n==0 (matches Python's np.sum over an empty array).
 */
double garch11_neg_loglik(
    const double* resid_sq,
    std::size_t   n,
    double        omega,
    double        alpha,
    double        beta,
    bool          penalize);

/**
 * GARCH(1,1) negative log-likelihood AND its analytic gradient w.r.t.
 * (omega, alpha, beta), computed in the same fused pass as
 * garch11_neg_loglik -- for scipy.optimize's `jac=True` convention (one
 * callable returning both the objective and its gradient), so an
 * optimizer using the gradient pays for exactly one recursion pass per
 * iteration, not two (a naive separate `jac=` callable would redo the
 * whole recursion just to differentiate it).
 *
 * Derived via forward-mode sensitivity tracking through the same
 * recursion: sigma2[0] is a constant (mean of resid_sq, or the floor),
 * so d(sigma2[0])/dtheta = 0 for every parameter. For t>=1, before
 * flooring: d(sigma2[t])/d(omega) = 1 + beta*d(sigma2[t-1])/d(omega);
 * d(sigma2[t])/d(alpha) = resid_sq[t-1] + beta*d(sigma2[t-1])/d(alpha);
 * d(sigma2[t])/d(beta) = sigma2[t-1] + beta*d(sigma2[t-1])/d(beta). If
 * flooring is active at t, sigma2[t] becomes the constant kMinSigma2, so
 * all three sensitivities reset to 0 there (a real, expected
 * non-differentiability at the clamp boundary -- the same kind
 * scipy's own default finite-difference gradient would also see, just
 * not smoothed over). d(NLL)/dtheta = 0.5 * sum_t( (1/sigma2[t] -
 * resid_sq[t]/sigma2[t]^2) * d(sigma2[t])/dtheta ), plus
 * 2e6*((alpha+beta)-1) added to the alpha and beta components when the
 * soft stationarity penalty is active.
 *
 * @param resid_sq  Squared demeaned residuals, length n.
 * @param n         Number of observations.
 * @param omega, alpha, beta, penalize  Same as garch11_neg_loglik.
 * @param out_grad  Caller-provided array of length 3, written as
 *   [d(NLL)/d(omega), d(NLL)/d(alpha), d(NLL)/d(beta)].
 * @returns  The NLL scalar (0.0 and out_grad={0,0,0} if n==0).
 */
double garch11_neg_loglik_grad(
    const double* resid_sq,
    std::size_t   n,
    double        omega,
    double        alpha,
    double        beta,
    bool          penalize,
    double        out_grad[3]);

}  // namespace sqt
