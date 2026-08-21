#pragma once

#include "sqt/platform.hpp"

#include <cstddef>
#include <vector>

namespace sqt {

/**
 * Sliding-window OLS: y ~ [1, factors_1, ..., factors_k].
 *
 * Each window is solved by rank-revealing Householder QR with column
 * pivoting (sqt::qr::lstsq), NOT by the rank-1 XtX/Xty update this comment
 * used to describe. That update was removed because it was wrong twice: its
 * pivot test compared every column against the single largest diagonal of
 * XtX -- the intercept column's, equal to the window length -- so factor
 * values around 1e-6 made the whole window read as singular and the kernel
 * returned all-NaN where the NumPy fallback returned a correct answer; and
 * forming XtX squares the condition number, which is what made the periodic
 * full recompute necessary in the first place. See rolling_regression.cpp
 * for the measured cost of the replacement (per-window QR is 15-60x slower
 * than the update was, paid knowingly).
 *
 * A rank-deficient window yields a NaN row rather than a minimum-norm
 * solution -- one rank policy, shared with analysis/multi_factor.py.
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

/** Buffer-writing form of rolling_factor_loadings(). `out` must have
 *  length n*(k+1). SQT_RESTRICT: `out` is always freshly allocated at every
 *  call site (bindings.cpp), never aliased with y/factors. */
void rolling_factor_loadings_into(
    const double* SQT_RESTRICT y,
    const double* SQT_RESTRICT factors,
    std::size_t   n,
    std::size_t   k,
    int           window,
    double* SQT_RESTRICT       out);

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

/** Buffer-writing form of rolling_beta(). `out` must have length n.
 *  SQT_RESTRICT: `out` is always freshly allocated at every call site
 *  (bindings.cpp), never aliased with y/x. */
void rolling_beta_into(
    const double* SQT_RESTRICT y,
    const double* SQT_RESTRICT x,
    std::size_t   n,
    int           window,
    double* SQT_RESTRICT       out);

}  // namespace sqt
