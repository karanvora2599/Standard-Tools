/**
 * C++ unit tests for sqt::ols2, sqt::adf_test, sqt::engle_granger.
 *
 * Build (all platforms):
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V -R cpp_cointegration
 */

#include "sqt/cointegration.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// ── Tiny assertion helpers ────────────────────────────────────────────────────

static int g_tests_run    = 0;
static int g_tests_failed = 0;

#define CHECK(cond) \
    do { \
        ++g_tests_run; \
        if (!(cond)) { \
            ++g_tests_failed; \
            std::fprintf(stderr, "FAIL  %s  line %d: %s\n", __func__, __LINE__, #cond); \
        } \
    } while (false)

#define CHECK_NEAR(a, b, tol) \
    CHECK(std::abs((a) - (b)) <= (tol))

#define CHECK_TRUE(cond)  CHECK(cond)
#define CHECK_FALSE(cond) CHECK(!(cond))
#define CHECK_NAN(val)    CHECK(std::isnan(val))
#define CHECK_NOT_NAN(val) CHECK(!std::isnan(val))


// ── Synthetic data generators ─────────────────────────────────────────────────

static std::vector<double> linspace(double start, double stop, int n) {
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i)
        out[i] = start + (stop - start) * i / (n - 1);
    return out;
}

// Simple LCG noise
static std::vector<double> lcg_noise(int n, unsigned seed = 42,
                                     double scale = 0.1) {
    std::vector<double> out(n);
    unsigned state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        out[i] = scale * static_cast<double>(static_cast<int>(state))
                         / 2147483648.0;
    }
    return out;
}

// Random walk (cumulative sum of noise)
static std::vector<double> random_walk(int n, unsigned seed = 7) {
    auto noise = lcg_noise(n, seed, 1.0);
    for (int i = 1; i < n; ++i) noise[i] += noise[i - 1];
    return noise;
}

// Mean-reverting AR(1) spread with phi < 1
static std::vector<double> ar1_series(int n, double phi = 0.9,
                                      unsigned seed = 99) {
    auto eps = lcg_noise(n, seed, 0.5);
    std::vector<double> out(n);
    out[0] = eps[0];
    for (int i = 1; i < n; ++i) out[i] = phi * out[i - 1] + eps[i];
    return out;
}


// ── Tests: ols2 ───────────────────────────────────────────────────────────────

static void test_ols2_perfect_line() {
    // y = 2*x + 3  →  intercept=3, slope=2, R²=1
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y = {5.0, 7.0, 9.0, 11.0, 13.0};
    auto r = sqt::ols2(y.data(), x.data(), x.size());
    CHECK_NEAR(r.intercept, 3.0, 1e-9);
    CHECK_NEAR(r.slope,     2.0, 1e-9);
    CHECK_NEAR(r.r_squared, 1.0, 1e-9);
    CHECK(r.residuals.size() == x.size());
    for (auto e : r.residuals)
        CHECK_NEAR(e, 0.0, 1e-9);
}

static void test_ols2_residuals_sum_to_zero() {
    // OLS residuals always sum to zero (intercept in model)
    auto x = linspace(0.0, 10.0, 20);
    auto y = lcg_noise(20, 1, 5.0);
    auto r = sqt::ols2(y.data(), x.data(), x.size());
    double sum = 0.0;
    for (auto e : r.residuals) sum += e;
    CHECK_NEAR(sum, 0.0, 1e-8);
}

static void test_ols2_r2_in_unit_interval() {
    auto x = linspace(0.0, 1.0, 50);
    auto noise = lcg_noise(50, 3, 2.0);
    std::vector<double> y(50);
    for (int i = 0; i < 50; ++i) y[i] = 1.5 * x[i] + noise[i];
    auto r = sqt::ols2(y.data(), x.data(), x.size());
    CHECK(r.r_squared >= 0.0 && r.r_squared <= 1.0);
}

static void test_ols2_flat_predictor() {
    // x constant → ill-conditioned; result should not crash and R²≈0
    std::vector<double> x(30, 5.0);
    auto y = lcg_noise(30, 2, 1.0);
    // Should not throw; R² is undefined/0 for constant x
    // (implementation may return 0 or NaN — just verify no crash)
    auto r = sqt::ols2(y.data(), x.data(), x.size());
    (void)r;
    CHECK_TRUE(true);  // survived without crash
}


// ── Tests: adf_test ───────────────────────────────────────────────────────────

static void test_adf_stationary_series() {
    // Stationary AR(1) near 0 should have very negative ADF stat
    auto s = ar1_series(300, 0.5, 11);
    auto r = sqt::adf_test(s.data(), s.size());
    CHECK(r.statistic < -3.0);  // should comfortably reject H0 of unit root
    CHECK(r.optimal_lag >= 0);
}

static void test_adf_unit_root_series() {
    // Random walk → ADF stat should be close to zero or mildly negative
    auto s = random_walk(300, 5);
    auto r = sqt::adf_test(s.data(), s.size());
    // Cannot guarantee > -3.43 (5% cv), but stat should not be extremely negative
    CHECK(r.statistic > -6.0);  // sanity: not impossibly extreme
}

static void test_adf_lag_nonnegative() {
    auto s = ar1_series(200, 0.8, 42);
    auto r = sqt::adf_test(s.data(), s.size());
    CHECK(r.optimal_lag >= 0);
}

static void test_adf_auto_max_lag() {
    // max_lag=-1 should auto-select; formula: floor(12*(n/100)^0.25)
    // For n=300: floor(12*(3)^0.25) = floor(12*1.316) = floor(15.8) = 15
    auto s = ar1_series(300, 0.7, 13);
    auto r = sqt::adf_test(s.data(), s.size(), -1, true);
    CHECK(r.optimal_lag >= 0 && r.optimal_lag <= 15);
}

static void test_adf_bic_vs_aic() {
    // BIC penalizes more → tends to select fewer lags; both should not crash
    auto s = ar1_series(200, 0.85, 77);
    auto aic = sqt::adf_test(s.data(), s.size(), 10, true);
    auto bic = sqt::adf_test(s.data(), s.size(), 10, false);
    CHECK(aic.optimal_lag >= 0);
    CHECK(bic.optimal_lag >= 0);
    // BIC lag ≤ AIC lag is common but not guaranteed; just check both finite
    CHECK_NOT_NAN(aic.statistic);
    CHECK_NOT_NAN(bic.statistic);
}

static void test_adf_explicit_lag_zero() {
    auto s = ar1_series(100, 0.6, 55);
    auto r = sqt::adf_test(s.data(), s.size(), 0, true);
    CHECK(r.optimal_lag == 0);
    CHECK_NOT_NAN(r.statistic);
}


// ── Tests: engle_granger ──────────────────────────────────────────────────────

static void test_eg_cointegrated_pair() {
    // y1 = rw, y0 = 2*y1 + small_noise → cointegrated by construction
    int n = 400;
    auto rw = random_walk(n, 3);
    auto noise = lcg_noise(n, 7, 0.05);
    std::vector<double> y0(n), y1(n);
    for (int i = 0; i < n; ++i) {
        y1[i] = rw[i];
        y0[i] = 2.0 * rw[i] + noise[i];
    }
    auto r = sqt::engle_granger(y0.data(), y1.data(), n);
    CHECK_NEAR(r.hedge_ratio, 2.0, 0.05);
    CHECK(r.p_value < 0.05);
    CHECK_TRUE(r.cointegrated);
    CHECK(r.half_life > 0.0 && r.half_life < 50.0);
    CHECK(r.n_obs == n);
}

static void test_eg_independent_random_walks() {
    // Two independent random walks should NOT be cointegrated
    int n = 300;
    auto rw1 = random_walk(n, 11);
    auto rw2 = random_walk(n, 99);
    auto r = sqt::engle_granger(rw1.data(), rw2.data(), n);
    // p_value should be higher (not significant at 1%); can't guarantee 5% in finite sample
    CHECK(r.p_value > 0.01);
    CHECK_FALSE(r.cointegrated == true && r.p_value > 0.05);
}

static void test_eg_hedge_ratio_sign() {
    // y0 = -y1 + noise → hedge ratio ≈ -1
    int n = 300;
    auto rw = random_walk(n, 21);
    auto noise = lcg_noise(n, 33, 0.02);
    std::vector<double> y0(n), y1(n);
    for (int i = 0; i < n; ++i) {
        y1[i] = rw[i];
        y0[i] = -rw[i] + noise[i];
    }
    auto r = sqt::engle_granger(y0.data(), y1.data(), n);
    CHECK(r.hedge_ratio < -0.8);
}

static void test_eg_critical_values_ordered() {
    int n = 300;
    auto rw1 = random_walk(n, 17);
    auto rw2 = random_walk(n, 19);
    auto r = sqt::engle_granger(rw1.data(), rw2.data(), n);
    // MacKinnon: cv_1pct < cv_5pct < cv_10pct (all negative)
    CHECK(r.cv_1pct < r.cv_5pct);
    CHECK(r.cv_5pct < r.cv_10pct);
    CHECK(r.cv_10pct < 0.0);
}

static void test_eg_p_value_in_unit_interval() {
    int n = 250;
    auto rw1 = random_walk(n, 31);
    auto rw2 = random_walk(n, 37);
    auto r = sqt::engle_granger(rw1.data(), rw2.data(), n);
    CHECK(r.p_value >= 0.0 && r.p_value <= 1.0);
}

// Independent replica of the MacKinnon (2010) regression-surface algorithm
// (same coefficients cointegration.cpp's internal mackinnon_pvalue uses --
// duplicated here, not called directly, since it has internal linkage and
// this is a separate translation unit) -- catches a regression in the real
// implementation by re-deriving the expected p-value from engle_granger's
// own adf_statistic, independently of how that statistic was computed.
static double mackinnon_pvalue_reference(double t) {
    if (t > 0.92) return 1.0;
    if (t < -18.86) return 0.0;
    double poly;
    if (t <= -2.62) {
        poly = 0.039796 * t * t + 1.5012 * t + 2.92;
    } else {
        poly = ((-0.042377 * t + -0.29198) * t + 0.64695) * t + 2.1945;
    }
    return 0.5 * (1.0 + std::erf(poly / std::sqrt(2.0)));
}

static void test_eg_p_value_matches_mackinnon_regression_surface() {
    // Sweep several AR(1) phi values (via a near-constant y1 so the
    // step-1 OLS just centers y0) to land adf_statistic across a real
    // range, not just one point -- same idea as the Python-side sweep in
    // tests/test_cpp_cointegration.py::TestMackinnonPValueAccuracy.
    const int n = 300;
    // Tiny noise, not an exactly-flat constant -- a truly zero-variance y1
    // makes ols2's X'X singular (residuals become NaN, not just a
    // degenerate-but-defined regression), which would just make every
    // iteration below hit the std::isnan `continue` instead of exercising
    // mackinnon_pvalue at all.
    auto y1_noise = lcg_noise(n, 17, 1e-6);
    std::vector<double> y1(n);
    for (int t = 0; t < n; ++t) y1[t] = 100.0 + y1_noise[t];
    const double phis[] = {0.995, 0.95, 0.9, 0.8, 0.6, 0.3};
    bool saw_mid_range = false;

    for (double phi : phis) {
        auto eps = lcg_noise(n, static_cast<unsigned>(phi * 1000), 0.5);
        std::vector<double> y0(n);
        y0[0] = eps[0];
        for (int t = 1; t < n; ++t) y0[t] = phi * y0[t - 1] + eps[t];
        for (int t = 0; t < n; ++t) y0[t] += 100.0;

        auto r = sqt::engle_granger(y0.data(), y1.data(), n);
        if (std::isnan(r.adf_statistic)) continue;
        if (r.adf_statistic > -18.86 && r.adf_statistic < 0.92) saw_mid_range = true;
        const double expected = mackinnon_pvalue_reference(r.adf_statistic);
        CHECK_NEAR(r.p_value, expected, 1e-9);
    }
    CHECK(saw_mid_range);
}

static void test_eg_intercept_finite() {
    int n = 200;
    auto rw = random_walk(n, 43);
    auto noise = lcg_noise(n, 47, 0.1);
    std::vector<double> y0(n), y1(n);
    for (int i = 0; i < n; ++i) {
        y1[i] = rw[i];
        y0[i] = 1.5 * rw[i] + 10.0 + noise[i];  // large intercept
    }
    auto r = sqt::engle_granger(y0.data(), y1.data(), n);
    CHECK_NOT_NAN(r.intercept);
    CHECK_NEAR(r.intercept, 10.0, 0.5);
}

static void test_eg_n_obs_matches_input() {
    int n = 150;
    auto rw1 = random_walk(n, 51);
    auto rw2 = random_walk(n, 53);
    auto r = sqt::engle_granger(rw1.data(), rw2.data(),
                                static_cast<std::size_t>(n));
    CHECK(r.n_obs == n);
}

static void test_eg_half_life_positive() {
    // Cointegrated pair should have finite, positive half-life
    int n = 400;
    auto rw = random_walk(n, 61);
    auto noise = lcg_noise(n, 67, 0.03);
    std::vector<double> y0(n), y1(n);
    for (int i = 0; i < n; ++i) {
        y1[i] = rw[i];
        y0[i] = rw[i] + noise[i];
    }
    auto r = sqt::engle_granger(y0.data(), y1.data(), n);
    CHECK(r.half_life > 0.0);
    CHECK(!std::isinf(r.half_life));
}


// ── Tests: kalman_filter_1state / kalman_filter_2state ─────────────────────────

static void test_kalman_1state_length() {
    const int n = 200;
    auto x = random_walk(n, 71);
    auto noise = lcg_noise(n, 73, 0.1);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = 1.2 * x[i] + noise[i];

    auto r = sqt::kalman_filter_1state(y.data(), x.data(), n, 1e-4, 1e-3);
    CHECK(r.beta.size() == static_cast<std::size_t>(n));
    CHECK(r.gain.size() == static_cast<std::size_t>(n));
    CHECK(r.innovation.size() == static_cast<std::size_t>(n));
}

static void test_kalman_1state_empty_on_bad_delta() {
    std::vector<double> y = {1.0, 2.0, 3.0};
    std::vector<double> x = {1.0, 2.0, 3.0};
    for (double bad_delta : {0.0, 1.0, -0.1, 1.5}) {
        auto r = sqt::kalman_filter_1state(y.data(), x.data(), 3, bad_delta, 1e-3);
        CHECK(r.beta.empty());
        CHECK(r.gain.empty());
        CHECK(r.innovation.empty());
    }
}

static void test_kalman_1state_empty_on_bad_observation_noise() {
    std::vector<double> y = {1.0, 2.0, 3.0};
    std::vector<double> x = {1.0, 2.0, 3.0};
    for (double bad_noise : {0.0, -1.0}) {
        auto r = sqt::kalman_filter_1state(y.data(), x.data(), 3, 1e-4, bad_noise);
        CHECK(r.beta.empty());
    }
}

static void test_kalman_1state_tracks_true_beta() {
    // y = 1.5*x + small noise -- with a large observation_noise (slow to
    // adapt) but enough bars, beta should converge toward 1.5, not stay
    // at its 0.0 prior.
    const int n = 500;
    auto x = random_walk(n, 81);
    auto noise = lcg_noise(n, 83, 0.05);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = 1.5 * x[i] + noise[i];

    auto r = sqt::kalman_filter_1state(y.data(), x.data(), n, 1e-3, 1e-2);
    CHECK_NEAR(r.beta.back(), 1.5, 0.2);
}

static void test_kalman_2state_length() {
    const int n = 200;
    auto x = random_walk(n, 91);
    auto noise = lcg_noise(n, 93, 0.1);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = 3.0 + 1.2 * x[i] + noise[i];

    auto r = sqt::kalman_filter_2state(y.data(), x.data(), n, 1e-4, 1e-3);
    CHECK(r.alpha.size() == static_cast<std::size_t>(n));
    CHECK(r.beta.size() == static_cast<std::size_t>(n));
    CHECK(r.gain.size() == static_cast<std::size_t>(n));
    CHECK(r.innovation.size() == static_cast<std::size_t>(n));
}

static void test_kalman_2state_empty_on_zero_n() {
    auto r = sqt::kalman_filter_2state(nullptr, nullptr, 0, 1e-4, 1e-3);
    CHECK(r.alpha.empty());
    CHECK(r.beta.empty());
}

static void test_kalman_2state_tracks_true_alpha_beta() {
    const int n = 500;
    auto x = random_walk(n, 101);
    auto noise = lcg_noise(n, 103, 0.05);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = 2.0 + 1.4 * x[i] + noise[i];

    auto r = sqt::kalman_filter_2state(y.data(), x.data(), n, 1e-3, 1e-2);
    CHECK_NEAR(r.alpha.back(), 2.0, 1.0);
    CHECK_NEAR(r.beta.back(), 1.4, 0.2);
}


// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    // ols2
    test_ols2_perfect_line();
    test_ols2_residuals_sum_to_zero();
    test_ols2_r2_in_unit_interval();
    test_ols2_flat_predictor();

    // adf_test
    test_adf_stationary_series();
    test_adf_unit_root_series();
    test_adf_lag_nonnegative();
    test_adf_auto_max_lag();
    test_adf_bic_vs_aic();
    test_adf_explicit_lag_zero();

    // engle_granger
    test_eg_cointegrated_pair();
    test_eg_independent_random_walks();
    test_eg_hedge_ratio_sign();
    test_eg_critical_values_ordered();
    test_eg_p_value_in_unit_interval();
    test_eg_p_value_matches_mackinnon_regression_surface();
    test_eg_intercept_finite();
    test_eg_n_obs_matches_input();
    test_eg_half_life_positive();

    // kalman_filter_1state / kalman_filter_2state
    test_kalman_1state_length();
    test_kalman_1state_empty_on_bad_delta();
    test_kalman_1state_empty_on_bad_observation_noise();
    test_kalman_1state_tracks_true_beta();
    test_kalman_2state_length();
    test_kalman_2state_empty_on_zero_n();
    test_kalman_2state_tracks_true_alpha_beta();

    std::printf("\n%d / %d tests passed.\n",
                g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed > 0 ? 1 : 0;
}
