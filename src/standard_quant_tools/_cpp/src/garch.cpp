#include "sqt/garch.hpp"

#include <numeric>

namespace sqt {

namespace {
constexpr double kMinSigma2 = 1e-12;
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

}  // namespace sqt
