#include "sqt/garch.hpp"

#include <cmath>
#include <numeric>

namespace sqt {

namespace {
constexpr double kMinSigma2 = 1e-12;
constexpr double kPi        = 3.14159265358979323846;
}

std::vector<double> garch11_variance_recursion(
    const double* resid_sq,
    std::size_t   n,
    double        omega,
    double        alpha,
    double        beta)
{
    if (n == 0) return {};

    std::vector<double> sigma2(n);

    const double mean = std::accumulate(resid_sq, resid_sq + n, 0.0) / static_cast<double>(n);
    sigma2[0] = (mean < kMinSigma2) ? kMinSigma2 : mean;

    for (std::size_t t = 1; t < n; ++t) {
        const double s2 = omega + alpha * resid_sq[t - 1] + beta * sigma2[t - 1];
        sigma2[t] = (s2 < kMinSigma2) ? kMinSigma2 : s2;
    }

    return sigma2;
}

double garch11_neg_loglik(
    const double* resid_sq,
    std::size_t   n,
    double        omega,
    double        alpha,
    double        beta,
    bool          penalize)
{
    if (n == 0) return 0.0;

    const double log_2pi = std::log(2.0 * kPi);

    const double mean = std::accumulate(resid_sq, resid_sq + n, 0.0) / static_cast<double>(n);
    double sigma2_prev = (mean < kMinSigma2) ? kMinSigma2 : mean;

    double nll = log_2pi + std::log(sigma2_prev) + resid_sq[0] / sigma2_prev;

    for (std::size_t t = 1; t < n; ++t) {
        const double s2 = omega + alpha * resid_sq[t - 1] + beta * sigma2_prev;
        sigma2_prev = (s2 < kMinSigma2) ? kMinSigma2 : s2;
        nll += log_2pi + std::log(sigma2_prev) + resid_sq[t] / sigma2_prev;
    }
    nll *= 0.5;

    if (penalize) {
        const double persistence = alpha + beta;
        if (persistence >= 1.0) {
            const double d = persistence - 1.0;
            nll += 1.0e6 * d * d;
        }
    }

    return nll;
}

double garch11_neg_loglik_grad(
    const double* resid_sq,
    std::size_t   n,
    double        omega,
    double        alpha,
    double        beta,
    bool          penalize,
    double        out_grad[3])
{
    out_grad[0] = out_grad[1] = out_grad[2] = 0.0;
    if (n == 0) return 0.0;

    const double log_2pi = std::log(2.0 * kPi);

    const double mean = std::accumulate(resid_sq, resid_sq + n, 0.0) / static_cast<double>(n);
    double sigma2_prev = (mean < kMinSigma2) ? kMinSigma2 : mean;
    // d(sigma2_prev)/d(omega, alpha, beta) -- all zero at t=0, since
    // sigma2[0] depends only on resid_sq (data), never on the parameters.
    double g_omega = 0.0, g_alpha = 0.0, g_beta = 0.0;

    double nll = 0.0, grad_omega = 0.0, grad_alpha = 0.0, grad_beta = 0.0;

    const auto accumulate_term = [&](std::size_t t) {
        const double inv_s2 = 1.0 / sigma2_prev;
        nll += log_2pi + std::log(sigma2_prev) + resid_sq[t] * inv_s2;
        // d/d(sigma2) of [log(sigma2) + resid_sq/sigma2] = 1/sigma2 - resid_sq/sigma2^2
        const double dcost_dsigma2 = inv_s2 - resid_sq[t] * inv_s2 * inv_s2;
        grad_omega += dcost_dsigma2 * g_omega;
        grad_alpha += dcost_dsigma2 * g_alpha;
        grad_beta  += dcost_dsigma2 * g_beta;
    };

    accumulate_term(0);

    for (std::size_t t = 1; t < n; ++t) {
        const double raw_s2 = omega + alpha * resid_sq[t - 1] + beta * sigma2_prev;
        if (raw_s2 < kMinSigma2) {
            // Flooring makes sigma2[t] a constant -- its sensitivity to
            // every parameter is exactly zero from here on, same real
            // non-differentiability a finite-difference gradient would
            // also encounter at this boundary.
            g_omega = 0.0;
            g_alpha = 0.0;
            g_beta  = 0.0;
            sigma2_prev = kMinSigma2;
        } else {
            const double new_g_omega = 1.0 + beta * g_omega;
            const double new_g_alpha = resid_sq[t - 1] + beta * g_alpha;
            const double new_g_beta  = sigma2_prev + beta * g_beta;
            g_omega = new_g_omega;
            g_alpha = new_g_alpha;
            g_beta  = new_g_beta;
            sigma2_prev = raw_s2;
        }
        accumulate_term(t);
    }

    nll *= 0.5;
    grad_omega *= 0.5;
    grad_alpha *= 0.5;
    grad_beta  *= 0.5;

    if (penalize) {
        const double persistence = alpha + beta;
        if (persistence >= 1.0) {
            const double d = persistence - 1.0;
            nll += 1.0e6 * d * d;
            const double dpenalty = 2.0e6 * d;
            grad_alpha += dpenalty;
            grad_beta  += dpenalty;
        }
    }

    out_grad[0] = grad_omega;
    out_grad[1] = grad_alpha;
    out_grad[2] = grad_beta;
    return nll;
}

}  // namespace sqt
