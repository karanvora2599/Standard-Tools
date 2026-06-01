#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace sqt {

// ── Result type ──────────────────────────────────────────────────────────────

struct HurstResult {
    double      hurst;          // estimated H in [0, 1.5]; NaN on failure
    std::string regime;         // "trending" | "random_walk" | "mean_reverting" | "unknown"
    double      fit_r_squared;  // R² of the log-log OLS fit
    std::string method;         // "dfa" or "rs"
    std::size_t n_obs;          // number of observations used
};

// ── Utility building blocks (also callable from C++ tests) ───────────────────

/**
 * Log-spaced unique integer window sizes in [min_w, max_w].
 * Mirrors numpy.unique(numpy.logspace(...).astype(int)).
 */
std::vector<int> log_sizes(int min_w, int max_w, int n_points = 20);

/**
 * Two-variable OLS (y = a + slope*x) returning (slope, R²).
 * O(m) closed-form normal equations — avoids LAPACK overhead for tiny systems.
 */
std::pair<double, double> ols_slope_r2(
    const std::vector<double>& x,
    const std::vector<double>& y);

// ── Scaling kernels ───────────────────────────────────────────────────────────

/**
 * Detrended Fluctuation Analysis (DFA-1).
 *
 * Integrates the mean-centred series, then for each box size measures the RMS
 * of linearly-detrended residuals.
 * Returns (sizes, fluctuations) arrays ready for log-log OLS.
 */
std::pair<std::vector<double>, std::vector<double>>
dfa(const double* arr, std::size_t n, int min_w, int max_w, int n_points = 20);

/**
 * Classic Rescaled Range (R/S) analysis.
 *
 * Returns (sizes, rs_values) arrays ready for log-log OLS.
 * Upward-biased for short series; prefer DFA for n < 2000.
 */
std::pair<std::vector<double>, std::vector<double>>
rs_analysis(const double* arr, std::size_t n, int min_w, int max_w, int n_points = 20);

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Estimate the Hurst exponent of a return series.
 *
 * @param arr        Contiguous double array (return series — NOT price levels).
 * @param n          Number of elements.
 * @param method     "dfa" (default) or "rs".
 * @param min_window Smallest sub-window for the scaling analysis (default 10).
 * @param max_window Largest sub-window. Pass -1 for auto (n/4 for DFA, n/2 for R/S).
 */
HurstResult hurst_exponent(
    const double*      arr,
    std::size_t        n,
    const std::string& method     = "dfa",
    int                min_window = 10,
    int                max_window = -1);

/**
 * Rolling Hurst exponent computed in a single C++ pass — no Python re-entry
 * per bar.
 *
 * @param arr        Contiguous double array.
 * @param n          Length of arr.
 * @param window     Lookback window in bars (default 200).
 * @param step       Compute every `step` bars; skipped positions hold NaN.
 * @param method     "dfa" or "rs".
 * @param min_window Smallest sub-window for internal scaling.
 * @returns          Vector of length n; first (window-1) elements are NaN.
 */
std::vector<double> rolling_hurst(
    const double*      arr,
    std::size_t        n,
    int                window     = 200,
    int                step       = 1,
    const std::string& method     = "dfa",
    int                min_window = 10);

}  // namespace sqt
