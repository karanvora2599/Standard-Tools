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
#include "sqt/qr.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
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


// Deterministic per-element pseudo-random in [-1, 1), for building design
// matrices. random_walk() above returns a whole cumulative series, which is
// the wrong shape for filling individual regressor cells.
static double pseudo_random(std::uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    std::uint64_t x = state;
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33;
    return (static_cast<double>(x >> 11) / 9007199254740992.0) * 2.0 - 1.0;
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

static void test_ols2_large_baseline_no_catastrophic_cancellation() {
    // Regression test for ols2's raw-moment cancellation bug (native
    // mirror of TestCppOls2Direct::test_large_baseline_no_catastrophic_cancellation
    // in tests/test_cpp_cointegration.py): an x series with a ~1e9
    // baseline used to make det = s1*sxx - sx*sx compute to exactly 0.0
    // (total cancellation between two ~1e20-magnitude terms), falsely
    // declaring the pair singular. The shift-by-reference-point fix must
    // recover the true slope.
    const int n = 300;
    auto noise_x = lcg_noise(n, 5, 3.0);
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0e9 + noise_x[i];
        y[i] = 10.0 + 1.5 * noise_x[i];  // slope wrt x's *variation* is 1.5
    }
    auto r = sqt::ols2(y.data(), x.data(), x.size());
    CHECK_NOT_NAN(r.slope);
    CHECK_NOT_NAN(r.intercept);
    CHECK_NEAR(r.slope, 1.5, 1e-6);
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

static void test_adf_max_lag_above_old_silent_cap_is_honored() {
    // Regression test: adf_test() used to silently clamp any max_lag
    // above 14 (kMaxK - 2, a fixed max-regressor-count constant) to at
    // most 12, with no error. kMaxK is now removed entirely -- the XtX/
    // Xty/xrow buffers are dynamically sized per candidate lag, and the
    // loop's own data-driven `T < p + 3` break is the sole limiter. This
    // must not crash/error at a max_lag far beyond the old ceiling.
    auto s = ar1_series(600, 0.75, 71);
    auto r = sqt::adf_test(s.data(), s.size(), /*max_lag=*/50, true);
    CHECK_NOT_NAN(r.statistic);
    CHECK(r.optimal_lag >= 0 && r.optimal_lag <= 50);
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

static void test_eg_large_baseline_hedge_ratio_recovered() {
    // engle_granger's step-1 OLS (ols2) at a ~1e9 baseline -- same
    // catastrophic-cancellation regression as
    // test_ols2_large_baseline_no_catastrophic_cancellation, exercised
    // end-to-end through the full two-variable cointegration pipeline.
    int n = 400;
    auto rw = random_walk(n, 3);
    auto noise = lcg_noise(n, 7, 0.05);
    std::vector<double> y0(n), y1(n);
    for (int i = 0; i < n; ++i) {
        y1[i] = 1.0e9 + rw[i];
        y0[i] = 1.8 * y1[i] + noise[i];
    }
    auto r = sqt::engle_granger(y0.data(), y1.data(), n);
    CHECK_NOT_NAN(r.hedge_ratio);
    CHECK_NEAR(r.hedge_ratio, 1.8, 0.1);
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

static void test_eg_critical_values_match_mackinnon_2010_exactly() {
    // Ordering was the ONLY thing asserted about these numbers, here and in
    // tests/cpp_bindings/, which is why the kernel shipped MacKinnon (1991)
    // Table 1 coefficients under a comment naming MacKinnon (2010) Table 2.
    // The 1991 set is also monotonic and also negative, so it satisfied
    // every existing assertion while disagreeing with the statsmodels
    // fallback by up to 0.006.
    //
    // Expected values below are statsmodels 0.14.3's
    //   mackinnoncrit(N=2, regression="c", nobs=n-1)
    // evaluated once and pinned, so this test needs no Python at runtime.
    // n-1, not n, is what coint() passes -- see engle_granger's call site.
    struct Case { int n; double cv1; double cv5; double cv10; };
    const Case cases[] = {
        //  n     1%           5%           10%
        {  51, -4.1288888000, -3.4610612000, -3.1303620000},
        { 101, -4.0093117000, -3.3979133000, -3.0871340000},
        { 251, -3.9407840320, -3.3606795680, -3.0614583200},
        {1001, -3.9074254270, -3.3422469230, -3.0486939200},
    };
    for (const auto& tc : cases) {
        auto rw1 = random_walk(tc.n, 17);
        auto rw2 = random_walk(tc.n, 19);
        auto r = sqt::engle_granger(rw1.data(), rw2.data(),
                                    static_cast<std::size_t>(tc.n));
        CHECK_NEAR(r.cv_1pct,  tc.cv1,  1e-8);
        CHECK_NEAR(r.cv_5pct,  tc.cv5,  1e-8);
        CHECK_NEAR(r.cv_10pct, tc.cv10, 1e-8);
    }
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


// Column-major cell accessor for the nested-RSS designs below.
// lstsq_nested_rss takes A column-major -- see its layout note in qr.hpp.
static inline double& cm(std::vector<double>& A, int T, int r, int c) {
    return A[static_cast<std::size_t>(c) * static_cast<std::size_t>(T) +
             static_cast<std::size_t>(r)];
}

// ── qr::lstsq_nested_rss ─────────────────────────────────────────────────────
//
// The primitive behind adf_test's lag sweep. It claims that ONE unpivoted
// factorization of a T x k design yields the residual sum of squares of every
// first-j-columns submodel. These tests check that claim against the thing it
// replaced: an independent qr::lstsq of each prefix, factorized separately.

static void check_nested_matches_per_prefix(
    const std::vector<double>& design, int T, int k,
    const std::vector<double>& rhs, double tol)
{
    // Reference: factorize each prefix on its own, the old way.
    std::vector<double> ref_rss(static_cast<std::size_t>(k) + 1,
                                std::numeric_limits<double>::quiet_NaN());
    std::vector<unsigned char> ref_ok(static_cast<std::size_t>(k) + 1, 0);
    for (int j = 1; j <= k; ++j) {
        if (T <= j) continue;
        std::vector<double> A(static_cast<std::size_t>(T) * static_cast<std::size_t>(j));
        for (int r = 0; r < T; ++r)
            for (int c = 0; c < j; ++c)
                A[static_cast<std::size_t>(r) * static_cast<std::size_t>(j) +
                  static_cast<std::size_t>(c)] =
                    design[static_cast<std::size_t>(r) * static_cast<std::size_t>(k) +
                           static_cast<std::size_t>(c)];
        std::vector<double> b = rhs;
        std::vector<int> perm(static_cast<std::size_t>(j));
        const auto sol = sqt::qr::lstsq(A.data(), b.data(), T, j, perm.data());
        ref_ok[static_cast<std::size_t>(j)] = sol.full_rank ? 1u : 0u;
        if (sol.full_rank) ref_rss[static_cast<std::size_t>(j)] = sol.rss;
    }

    // Under test: one factorization for all of them. lstsq_nested_rss wants
    // COLUMN-major, so transpose -- which also means this test feeds it a
    // differently-laid-out copy of the same matrix the reference used, and
    // agreement is not an artifact of shared storage.
    std::vector<double> A2(static_cast<std::size_t>(T) * static_cast<std::size_t>(k));
    for (int r = 0; r < T; ++r)
        for (int cc = 0; cc < k; ++cc)
            A2[static_cast<std::size_t>(cc) * static_cast<std::size_t>(T) +
               static_cast<std::size_t>(r)] =
                design[static_cast<std::size_t>(r) * static_cast<std::size_t>(k) +
                       static_cast<std::size_t>(cc)];
    std::vector<double> b2 = rhs;
    std::vector<double> got_rss(static_cast<std::size_t>(k) + 1);
    std::vector<unsigned char> got_ok(static_cast<std::size_t>(k) + 1);
    sqt::qr::lstsq_nested_rss(A2.data(), b2.data(), T, k, got_rss.data(), got_ok.data());

    for (int j = 1; j <= k; ++j) {
        if (T <= j) continue;
        const std::size_t jj = static_cast<std::size_t>(j);
        CHECK(got_ok[jj] == ref_ok[jj]);
        if (!ref_ok[jj]) continue;
        CHECK(got_rss[jj] >= 0.0);  // suffix sum of squares, never negative
        const double scale = std::max(std::abs(ref_rss[jj]), 1e-12);
        CHECK_NEAR(got_rss[jj] / scale, ref_rss[jj] / scale, tol);
    }
}

static void test_nested_rss_matches_independent_fits_random() {
    std::uint64_t state = 20260821;
    for (int trial = 0; trial < 20; ++trial) {
        const int T = 40 + static_cast<int>((pseudo_random(state) + 1.0) * 120.0);
        const int k = 2 + static_cast<int>((pseudo_random(state) + 1.0) * 9.0);
        std::vector<double> design(static_cast<std::size_t>(T) * static_cast<std::size_t>(k));
        std::vector<double> rhs(static_cast<std::size_t>(T));
        for (int r = 0; r < T; ++r) {
            for (int c = 0; c < k; ++c)
                design[static_cast<std::size_t>(r) * static_cast<std::size_t>(k) +
                       static_cast<std::size_t>(c)] = (c == 0) ? 1.0 : pseudo_random(state);
            rhs[static_cast<std::size_t>(r)] = pseudo_random(state);
        }
        check_nested_matches_per_prefix(design, T, k, rhs, 1e-9);
    }
}

static void test_nested_rss_is_monotone_nonincreasing_in_k() {
    // Adding a regressor can never increase the residual sum of squares.
    // A property the identity must satisfy regardless of what the reference
    // says -- and one that a wrong suffix offset would break immediately.
    std::uint64_t state = 4242;
    const int T = 120, k = 8;
    std::vector<double> A(static_cast<std::size_t>(T) * static_cast<std::size_t>(k));
    std::vector<double> b(static_cast<std::size_t>(T));
    for (int r = 0; r < T; ++r) {
        for (int c = 0; c < k; ++c)
            cm(A, T, r, c) = (c == 0) ? 1.0 : pseudo_random(state);
        b[static_cast<std::size_t>(r)] = pseudo_random(state);
    }
    std::vector<double> rss(static_cast<std::size_t>(k) + 1);
    std::vector<unsigned char> ok(static_cast<std::size_t>(k) + 1);
    sqt::qr::lstsq_nested_rss(A.data(), b.data(), T, k, rss.data(), ok.data());
    for (int j = 1; j <= k; ++j)
        CHECK(rss[static_cast<std::size_t>(j)] <=
              rss[static_cast<std::size_t>(j - 1)] + 1e-9);
}

static void test_nested_rss_flags_a_duplicated_column() {
    // Column 3 duplicates column 1, so every prefix through it is rank
    // deficient and every prefix before it is not.
    std::uint64_t state = 77;
    const int T = 80, k = 5;
    std::vector<double> A(static_cast<std::size_t>(T) * static_cast<std::size_t>(k));
    std::vector<double> b(static_cast<std::size_t>(T));
    for (int r = 0; r < T; ++r) {
        cm(A, T, r, 0) = 1.0;
        cm(A, T, r, 1) = pseudo_random(state);
        cm(A, T, r, 2) = pseudo_random(state);
        cm(A, T, r, 3) = cm(A, T, r, 1);   // exact duplicate
        cm(A, T, r, 4) = pseudo_random(state);
        b[static_cast<std::size_t>(r)] = pseudo_random(state);
    }
    std::vector<double> rss(static_cast<std::size_t>(k) + 1);
    std::vector<unsigned char> ok(static_cast<std::size_t>(k) + 1);
    sqt::qr::lstsq_nested_rss(A.data(), b.data(), T, k, rss.data(), ok.data());
    CHECK(ok[1] == 1);
    CHECK(ok[2] == 1);
    CHECK(ok[3] == 1);
    CHECK(ok[4] == 0);  // the duplicate enters here
    CHECK(ok[5] == 0);  // and every longer prefix stays deficient
}

static void test_nested_rss_handles_prefixes_beyond_T() {
    // A design wider than it is tall: short prefixes are still answerable,
    // which is exactly the case adf_test hits on a short series with a large
    // auto max_lag.
    std::uint64_t state = 5;
    const int T = 6, k = 10;
    std::vector<double> A(static_cast<std::size_t>(T) * static_cast<std::size_t>(k));
    std::vector<double> b(static_cast<std::size_t>(T));
    for (int r = 0; r < T; ++r) {
        for (int c = 0; c < k; ++c)
            cm(A, T, r, c) = (c == 0) ? 1.0 : pseudo_random(state);
        b[static_cast<std::size_t>(r)] = pseudo_random(state);
    }
    std::vector<double> rss(static_cast<std::size_t>(k) + 1);
    std::vector<unsigned char> ok(static_cast<std::size_t>(k) + 1);
    sqt::qr::lstsq_nested_rss(A.data(), b.data(), T, k, rss.data(), ok.data());
    for (int j = 1; j < T; ++j) {
        CHECK(!std::isnan(rss[static_cast<std::size_t>(j)]));
        CHECK(rss[static_cast<std::size_t>(j)] >= 0.0);
    }
    for (int j = T + 1; j <= k; ++j)
        CHECK(ok[static_cast<std::size_t>(j)] == 0);
}

static void test_nested_rss_scale_invariant() {
    // The rank verdict must not change when a column is re-expressed in
    // different units -- the reason the routine equilibrates before
    // factorizing. RSS is invariant under column scaling by construction.
    std::uint64_t state = 909;
    const int T = 100, k = 4;
    std::vector<double> A(static_cast<std::size_t>(T) * static_cast<std::size_t>(k));
    std::vector<double> b(static_cast<std::size_t>(T));
    for (int r = 0; r < T; ++r) {
        cm(A, T, r, 0) = 1.0;
        for (int c = 1; c < k; ++c) cm(A, T, r, c) = pseudo_random(state);
        b[static_cast<std::size_t>(r)] = pseudo_random(state);
    }
    std::vector<double> A1 = A, b1 = b;
    std::vector<double> rss1(static_cast<std::size_t>(k) + 1);
    std::vector<unsigned char> ok1(static_cast<std::size_t>(k) + 1);
    sqt::qr::lstsq_nested_rss(A1.data(), b1.data(), T, k, rss1.data(), ok1.data());

    // Same design, columns 1..k-1 rescaled by 1e13 / 1e-13 alternately.
    std::vector<double> A2 = A, b2 = b;
    for (int r = 0; r < T; ++r)
        for (int c = 1; c < k; ++c)
            cm(A2, T, r, c) *= (c % 2 == 0) ? 1e13 : 1e-13;
    std::vector<double> rss2(static_cast<std::size_t>(k) + 1);
    std::vector<unsigned char> ok2(static_cast<std::size_t>(k) + 1);
    sqt::qr::lstsq_nested_rss(A2.data(), b2.data(), T, k, rss2.data(), ok2.data());

    for (int j = 1; j <= k; ++j) {
        CHECK(ok1[static_cast<std::size_t>(j)] == ok2[static_cast<std::size_t>(j)]);
        const double s = std::max(std::abs(rss1[static_cast<std::size_t>(j)]), 1e-12);
        CHECK_NEAR(rss1[static_cast<std::size_t>(j)] / s,
                   rss2[static_cast<std::size_t>(j)] / s, 1e-8);
    }
}

// ── batch_engle_granger ──────────────────────────────────────────────────────
//
// The batch kernel must be indistinguishable from a loop of engle_granger()
// calls -- BIT-identical, not merely close. There is no accumulation across
// pairs, so there is no floating-point reason for any difference, and a
// tolerance here would hide a real indexing bug.

static void check_batch_matches_serial(int n_tickers, int n_bars,
                                        int max_lag, bool use_aic)
{
    std::vector<double> panel(static_cast<std::size_t>(n_tickers) *
                              static_cast<std::size_t>(n_bars));
    for (int t = 0; t < n_tickers; ++t) {
        auto s = random_walk(n_bars, static_cast<unsigned>(17 + 7 * t));
        for (int i = 0; i < n_bars; ++i)
            panel[static_cast<std::size_t>(t) * static_cast<std::size_t>(n_bars) +
                  static_cast<std::size_t>(i)] = s[static_cast<std::size_t>(i)] + 100.0;
    }

    std::vector<int> pairs;
    for (int a = 0; a < n_tickers; ++a)
        for (int b = a + 1; b < n_tickers; ++b) { pairs.push_back(a); pairs.push_back(b); }
    const std::size_t n_pairs = pairs.size() / 2;

    std::vector<double> out(n_pairs * static_cast<std::size_t>(sqt::kBatchCointCols));
    sqt::batch_engle_granger(panel.data(), static_cast<std::size_t>(n_tickers),
                              static_cast<std::size_t>(n_bars), pairs.data(),
                              n_pairs, max_lag, use_aic, out.data());

    for (std::size_t i = 0; i < n_pairs; ++i) {
        const double* y0 = panel.data() +
            static_cast<std::size_t>(pairs[i * 2]) * static_cast<std::size_t>(n_bars);
        const double* y1 = panel.data() +
            static_cast<std::size_t>(pairs[i * 2 + 1]) * static_cast<std::size_t>(n_bars);
        const auto r = sqt::engle_granger(y0, y1, static_cast<std::size_t>(n_bars),
                                           max_lag, use_aic);
        const double* row = out.data() + i * static_cast<std::size_t>(sqt::kBatchCointCols);

        // Exact equality, with the two IEEE special cases spelled out: NaN is
        // never equal to itself, and half_life is legitimately +inf for a
        // spread that is not mean-reverting.
        auto same = [](double a, double b) {
            if (std::isnan(a) && std::isnan(b)) return true;
            return a == b;
        };
        CHECK(same(row[0], r.intercept));
        CHECK(same(row[1], r.hedge_ratio));
        CHECK(same(row[2], r.adf_statistic));
        CHECK(row[3] == static_cast<double>(r.optimal_lag));
        CHECK(same(row[4], r.p_value));
        CHECK(same(row[5], r.cv_1pct));
        CHECK(same(row[6], r.cv_5pct));
        CHECK(same(row[7], r.cv_10pct));
        CHECK(same(row[8], r.half_life));
        CHECK(row[9] == static_cast<double>(r.n_obs));
        CHECK(row[10] == (r.cointegrated ? 1.0 : 0.0));
    }
}

static void test_batch_coint_matches_serial_auto_lag() {
    check_batch_matches_serial(/*n_tickers=*/8, /*n_bars=*/300, /*max_lag=*/-1, true);
}

static void test_batch_coint_matches_serial_fixed_lag_and_bic() {
    check_batch_matches_serial(8, 300, /*max_lag=*/4, /*use_aic=*/false);
}

static void test_batch_coint_matches_serial_short_series() {
    // Short enough that the automatic max-lag cap binds and some candidate
    // lags have no usable degrees of freedom.
    check_batch_matches_serial(6, 40, -1, true);
}

static void test_batch_coint_respects_pair_order() {
    // Row i of the output must correspond to row i of `pairs`, including when
    // the pairs are not in any sorted order and repeat.
    const int n_bars = 200;
    std::vector<double> panel(3 * static_cast<std::size_t>(n_bars));
    for (int t = 0; t < 3; ++t) {
        auto s = random_walk(n_bars, static_cast<unsigned>(101 + t));
        for (int i = 0; i < n_bars; ++i)
            panel[static_cast<std::size_t>(t) * static_cast<std::size_t>(n_bars) +
                  static_cast<std::size_t>(i)] = s[static_cast<std::size_t>(i)] + 100.0;
    }
    const std::vector<int> pairs = {2, 0,  1, 2,  0, 1,  2, 0};
    const std::size_t n_pairs = 4;
    std::vector<double> out(n_pairs * static_cast<std::size_t>(sqt::kBatchCointCols));
    sqt::batch_engle_granger(panel.data(), 3, static_cast<std::size_t>(n_bars),
                              pairs.data(), n_pairs, -1, true, out.data());

    for (std::size_t i = 0; i < n_pairs; ++i) {
        const auto r = sqt::engle_granger(
            panel.data() + static_cast<std::size_t>(pairs[i * 2]) * static_cast<std::size_t>(n_bars),
            panel.data() + static_cast<std::size_t>(pairs[i * 2 + 1]) * static_cast<std::size_t>(n_bars),
            static_cast<std::size_t>(n_bars), -1, true);
        CHECK(out[i * static_cast<std::size_t>(sqt::kBatchCointCols) + 1] == r.hedge_ratio);
    }
    // Rows 0 and 3 are the same pair, so they must be the same numbers.
    for (int col = 0; col < sqt::kBatchCointCols; ++col) {
        const double a = out[0 * static_cast<std::size_t>(sqt::kBatchCointCols) +
                             static_cast<std::size_t>(col)];
        const double d = out[3 * static_cast<std::size_t>(sqt::kBatchCointCols) +
                             static_cast<std::size_t>(col)];
        CHECK(a == d || (std::isnan(a) && std::isnan(d)));
    }
}

static void test_batch_coint_rejects_out_of_range_pair() {
    const int n_bars = 100;
    std::vector<double> panel(2 * static_cast<std::size_t>(n_bars), 1.0);
    std::vector<double> out(static_cast<std::size_t>(sqt::kBatchCointCols));
    for (const std::vector<int>& bad : {std::vector<int>{0, 2},
                                        std::vector<int>{2, 0},
                                        std::vector<int>{-1, 1}}) {
        bool threw = false;
        try {
            sqt::batch_engle_granger(panel.data(), 2, static_cast<std::size_t>(n_bars),
                                      bad.data(), 1, -1, true, out.data());
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK_TRUE(threw);
    }
}

static void test_batch_coint_empty_is_a_noop() {
    std::vector<double> panel(200, 1.0);
    // n_pairs == 0 must return without touching `out` or dereferencing pairs.
    sqt::batch_engle_granger(panel.data(), 2, 100, nullptr, 0, -1, true, nullptr);
    CHECK_TRUE(true);  // reaching here without a crash is the assertion
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    // ols2
    test_ols2_perfect_line();
    test_ols2_residuals_sum_to_zero();
    test_ols2_r2_in_unit_interval();
    test_ols2_flat_predictor();
    test_ols2_large_baseline_no_catastrophic_cancellation();

    // adf_test
    test_adf_stationary_series();
    test_adf_unit_root_series();
    test_adf_lag_nonnegative();
    test_adf_auto_max_lag();
    test_adf_bic_vs_aic();
    test_adf_explicit_lag_zero();
    test_adf_max_lag_above_old_silent_cap_is_honored();

    // engle_granger
    test_eg_cointegrated_pair();
    test_eg_large_baseline_hedge_ratio_recovered();
    test_eg_independent_random_walks();
    test_eg_hedge_ratio_sign();
    // batch_engle_granger
    test_batch_coint_matches_serial_auto_lag();
    test_batch_coint_matches_serial_fixed_lag_and_bic();
    test_batch_coint_matches_serial_short_series();
    test_batch_coint_respects_pair_order();
    test_batch_coint_rejects_out_of_range_pair();
    test_batch_coint_empty_is_a_noop();

    // qr::lstsq_nested_rss (the primitive behind the lag sweep)
    test_nested_rss_matches_independent_fits_random();
    test_nested_rss_is_monotone_nonincreasing_in_k();
    test_nested_rss_flags_a_duplicated_column();
    test_nested_rss_handles_prefixes_beyond_T();
    test_nested_rss_scale_invariant();

    test_eg_critical_values_ordered();
    test_eg_critical_values_match_mackinnon_2010_exactly();
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
