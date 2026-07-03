#pragma once

#include <cstddef>
#include <vector>

namespace sqt {

/**
 * Incremental sliding-window OLS: y ~ [1, factors_1, ..., factors_k].
 *
 * Uses rank-1 XtX / Xty updates to slide the window in O(k²) per step
 * instead of O(window × k²).  A full recompute is issued every `window`
 * steps to prevent floating-point drift.
 *
 * @param y        Response array, length n.
 * @param factors  Factor matrix flattened row-major (n × k):
 *                   factors[i*k + f] = factor f at bar i.
 * @param n        Number of observations.
 * @param k        Number of factors (NOT counting the intercept).
 * @param window   Rolling window size in bars.
 * @return         Flat row-major array of shape (n, k+1):
 *                   result[i*(k+1) + j] = coefficient j at bar i.
 *                   j=0 is the intercept (alpha); j>0 are factor loadings.
 *                 First (window-1) entries per row contain NaN.
 */
std::vector<double> rolling_factor_loadings(
    const double* y,
    const double* factors,
    std::size_t   n,
    std::size_t   k,
    int           window);

/**
 * Incremental sliding-window OLS beta of y on x.
 *
 *   beta[i] = cov(x, y) / var(x)  over bars [i-window+1, i]
 *
 * Uses O(1) sum updates per step — no per-step loop over the window.
 *
 * @param y       Asset return array, length n.
 * @param x       Benchmark return array, length n.
 * @param n       Number of observations.
 * @param window  Rolling window size in bars.
 * @return        Array of length n; first (window-1) values are NaN.
 */
std::vector<double> rolling_beta(
    const double* y,
    const double* x,
    std::size_t   n,
    int           window);

}  // namespace sqt
