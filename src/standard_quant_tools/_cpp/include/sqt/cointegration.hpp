#pragma once

#include <cstddef>
#include <vector>

namespace sqt {

// ── Result types ──────────────────────────────────────────────────────────────

struct Ols2Result {
    double intercept;
    double slope;
    double r_squared;
    std::vector<double> residuals;  // length n
};

struct AdfResult {
    double statistic;   // t-statistic for the lagged-level coefficient
    int    optimal_lag; // lag selected by IC
    double ic_min;      // minimum information-criterion value at optimal lag
};

struct CointResult {
    // Engle-Granger step 1: OLS
    double intercept;
    double hedge_ratio;    // OLS slope: y0 ≈ intercept + hedge_ratio * y1 + spread

    // Engle-Granger step 2: ADF on the spread
    double adf_statistic;
    int    optimal_lag;
    double p_value;        // MacKinnon (2010) cointegration p-value
    double cv_1pct;        // critical value at 1%  (MacKinnon 2010 Table 2)
    double cv_5pct;        // critical value at 5%
    double cv_10pct;       // critical value at 10%

    // Half-life of mean reversion
    double half_life;      // in bars; +inf when spread is not mean-reverting

    int  n_obs;
    bool cointegrated;     // p_value < 0.05
};

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * 2-variable OLS: y = intercept + slope * x + residuals.
 *
 * Uses closed-form normal equations — avoids LAPACK for this 2×2 system.
 *
 * @param y  Response array (length n).
 * @param x  Predictor array (length n).
 * @param n  Number of observations.
 */
Ols2Result ols2(const double* y, const double* x, std::size_t n);

/**
 * Augmented Dickey-Fuller test with automatic lag selection.
 *
 * Regression: Δy_t = c + φ*y_{t-1} + Σ ψᵢ*Δy_{t-i} + ε_t
 *
 * @param y       Level series (NOT differenced).
 * @param n       Length of y.
 * @param max_lag Maximum lags to test; -1 = auto (⌊12·(n/100)^(1/4)⌋).
 * @param use_aic true = AIC (default), false = BIC for lag selection.
 * @returns       AdfResult with t-statistic and optimal lag.
 */
AdfResult adf_test(const double* y, std::size_t n,
                   int max_lag = -1, bool use_aic = true);

/**
 * Engle-Granger two-step cointegration test.
 *
 * Step 1: OLS regression of y0 on y1 → hedge ratio and spread.
 * Step 2: ADF test on the spread with MacKinnon (2010) critical values.
 * Step 3: AR(1) half-life of the spread.
 *
 * @param y0, y1  Price (or log-price) arrays of equal length n.
 * @param n       Number of observations.
 * @param max_lag ADF max lag; -1 = auto.
 * @param use_aic Use AIC (true, default) or BIC (false) for lag selection.
 */
CointResult engle_granger(
    const double* y0, const double* y1, std::size_t n,
    int max_lag = -1, bool use_aic = true);

}  // namespace sqt
