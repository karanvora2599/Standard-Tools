/**
 * C++ unit tests for sqt::hurst — standalone, no external test framework.
 *
 * Build (all platforms):
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run directly:
 *   Windows  : build\tests\cpp\Release\test_hurst.exe
 *              (or build\tests\cpp\test_hurst.exe with Ninja)
 *   Linux    : ./build/tests/cpp/test_hurst
 *   macOS    : ./build/tests/cpp/test_hurst
 *
 * Run via CTest (all platforms):
 *   ctest --test-dir build --config Release -V
 */

#include "sqt/hurst.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
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
#define CHECK_NAN(val)    CHECK(std::isnan(val))
#define CHECK_NOT_NAN(val) CHECK(!std::isnan(val))


// ── Synthetic data generators ─────────────────────────────────────────────────

// Simple LCG pseudo-random, seed-deterministic, no stdlib dependency needed
static std::vector<double> white_noise(int n, unsigned seed = 42) {
    std::vector<double> out(n);
    unsigned state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        out[i] = static_cast<double>(static_cast<int>(state)) / 2147483648.0;
    }
    return out;
}

// Cumulative sum of white noise → random walk returns (H ≈ 0.5 on returns)
static std::vector<double> random_walk_returns(int n, unsigned seed = 42) {
    return white_noise(n, seed);  // returns are already white noise
}

// Linearly increasing series (strong trend → H close to 1 on returns of a ramp)
static std::vector<double> trending_series(int n) {
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) out[i] = static_cast<double>(i) / n;
    return out;
}


// ── Tests: log_sizes ─────────────────────────────────────────────────────────

static void test_log_sizes_basic() {
    auto sizes = sqt::log_sizes(10, 100, 20);
    // All sizes must be in [10, 100]
    for (int s : sizes) {
        CHECK(s >= 10);
        CHECK(s <= 100);
    }
    // Must be sorted and unique
    for (std::size_t i = 1; i < sizes.size(); ++i) {
        CHECK(sizes[i] > sizes[i - 1]);
    }
    // At least one size
    CHECK(!sizes.empty());
}

static void test_log_sizes_single_point() {
    auto sizes = sqt::log_sizes(10, 10, 5);
    CHECK(sizes.size() == 1);
    CHECK(sizes[0] == 10);
}


// ── Tests: ols_slope_r2 ───────────────────────────────────────────────────────

static void test_ols_perfect_fit() {
    // y = 2x + 0  → slope = 2, R² = 1
    std::vector<double> x = {0.0, 1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {0.0, 2.0, 4.0, 6.0, 8.0};
    auto [slope, r2] = sqt::ols_slope_r2(x, y);
    CHECK_NEAR(slope, 2.0, 1e-10);
    CHECK_NEAR(r2,    1.0, 1e-10);
}

static void test_ols_with_intercept() {
    // y = 3x + 1
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {4.0, 7.0, 10.0, 13.0};
    auto [slope, r2] = sqt::ols_slope_r2(x, y);
    CHECK_NEAR(slope, 3.0, 1e-10);
    CHECK_NEAR(r2,    1.0, 1e-10);
}

static void test_ols_flat_line() {
    // y = constant → slope = 0
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {5.0, 5.0, 5.0};
    auto [slope, r2] = sqt::ols_slope_r2(x, y);
    CHECK_NEAR(slope, 0.0, 1e-10);
    // R² = 0 when ss_tot = 0
    CHECK_NEAR(r2, 0.0, 1e-10);
}

static void test_ols_too_few_points() {
    // Single-point input → returns {0, 0}
    std::vector<double> x = {1.0};
    std::vector<double> y = {2.0};
    auto [slope, r2] = sqt::ols_slope_r2(x, y);
    CHECK_NEAR(slope, 0.0, 1e-10);
    CHECK_NEAR(r2,    0.0, 1e-10);
}


// ── Tests: hurst_exponent (DFA) ───────────────────────────────────────────────

static void test_hurst_dfa_too_short() {
    // n < min_window * 4 → NaN
    std::vector<double> data = {0.1, 0.2, 0.3, 0.4};
    auto r = sqt::hurst_exponent(data.data(), data.size(), "dfa", 10, -1);
    CHECK_NAN(r.hurst);
    CHECK(r.regime == "unknown");
}

static void test_hurst_dfa_reasonable_series() {
    // 600-bar white noise — should give a valid H, not NaN
    auto data = white_noise(600);
    auto r = sqt::hurst_exponent(data.data(), data.size(), "dfa", 10, -1);
    CHECK_NOT_NAN(r.hurst);
    CHECK(r.hurst >= 0.0 && r.hurst <= 1.5);
    CHECK(r.n_obs == 600u);
    CHECK(r.method == "dfa");
    // Regime must be one of the three valid strings
    CHECK(r.regime == "trending" || r.regime == "random_walk" || r.regime == "mean_reverting");
}

static void test_hurst_dfa_white_noise_range() {
    // White noise has H ≈ 0.5 — check it lands in a broad expected band
    auto data = white_noise(1000);
    auto r = sqt::hurst_exponent(data.data(), data.size(), "dfa", 10, -1);
    CHECK_NOT_NAN(r.hurst);
    // Broad range [0.2, 0.8] to avoid flaky failures on any RNG
    CHECK(r.hurst >= 0.2 && r.hurst <= 0.8);
}

static void test_hurst_dfa_fit_r2_positive() {
    auto data = white_noise(800);
    auto r = sqt::hurst_exponent(data.data(), data.size(), "dfa", 10, -1);
    // R² should be a valid non-NaN value for a reasonable series
    CHECK_NOT_NAN(r.fit_r_squared);
    CHECK(r.fit_r_squared >= 0.0 && r.fit_r_squared <= 1.0);
}


// ── Tests: hurst_exponent (R/S) ───────────────────────────────────────────────

static void test_hurst_rs_reasonable_series() {
    auto data = white_noise(600);
    auto r = sqt::hurst_exponent(data.data(), data.size(), "rs", 10, -1);
    CHECK_NOT_NAN(r.hurst);
    CHECK(r.hurst >= 0.0 && r.hurst <= 1.5);
    CHECK(r.method == "rs");
}

static void test_hurst_rs_too_short() {
    std::vector<double> data = {0.1, 0.2, 0.3};
    auto r = sqt::hurst_exponent(data.data(), data.size(), "rs", 10, -1);
    CHECK_NAN(r.hurst);
}


// ── Tests: rolling_hurst ──────────────────────────────────────────────────────

static void test_rolling_hurst_length() {
    auto   data = white_noise(500);
    int    n    = static_cast<int>(data.size());
    int    win  = 100;
    auto   out  = sqt::rolling_hurst(data.data(), n, win, 1, "dfa", 10);

    CHECK(static_cast<int>(out.size()) == n);
}

static void test_rolling_hurst_leading_nans() {
    auto data = white_noise(300);
    int  win  = 100;
    auto out  = sqt::rolling_hurst(data.data(), data.size(), win, 1, "dfa", 10);

    // First (window-1) values must be NaN
    for (int i = 0; i < win - 1; ++i)
        CHECK_NAN(out[i]);
}

static void test_rolling_hurst_non_nan_count() {
    const int n    = 300;
    const int win  = 100;
    const int step = 1;
    auto      data = white_noise(n);
    auto      out  = sqt::rolling_hurst(data.data(), n, win, step, "dfa", 10);

    int non_nan = 0;
    for (double v : out) if (!std::isnan(v)) ++non_nan;

    // Expected: n - window + 1 non-NaN values (with step=1)
    const int expected = n - win + 1;
    CHECK(non_nan == expected);
}

static void test_rolling_hurst_step2() {
    const int n    = 400;
    const int win  = 100;
    const int step = 2;
    auto      data = white_noise(n);
    auto      out  = sqt::rolling_hurst(data.data(), n, win, step, "dfa", 10);

    CHECK(static_cast<int>(out.size()) == n);

    // Non-NaN positions should be at i = win-1, win+1, win+3, ...
    for (int i = 0; i < n; ++i) {
        const bool should_have_value = (i >= win - 1) && ((i - (win - 1)) % step == 0);
        if (should_have_value) {
            CHECK_NOT_NAN(out[i]);
        } else {
            CHECK_NAN(out[i]);
        }
    }
}

static void test_rolling_hurst_values_in_range() {
    auto data = white_noise(400);
    auto out  = sqt::rolling_hurst(data.data(), data.size(), 100, 1, "dfa", 10);
    for (double v : out) {
        if (!std::isnan(v)) {
            CHECK(v >= 0.0 && v <= 1.5);
        }
    }
}


// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    // log_sizes
    test_log_sizes_basic();
    test_log_sizes_single_point();

    // ols_slope_r2
    test_ols_perfect_fit();
    test_ols_with_intercept();
    test_ols_flat_line();
    test_ols_too_few_points();

    // hurst_exponent — DFA
    test_hurst_dfa_too_short();
    test_hurst_dfa_reasonable_series();
    test_hurst_dfa_white_noise_range();
    test_hurst_dfa_fit_r2_positive();

    // hurst_exponent — R/S
    test_hurst_rs_reasonable_series();
    test_hurst_rs_too_short();

    // rolling_hurst
    test_rolling_hurst_length();
    test_rolling_hurst_leading_nans();
    test_rolling_hurst_non_nan_count();
    test_rolling_hurst_step2();
    test_rolling_hurst_values_in_range();

    std::printf("\n%d / %d tests passed.\n", g_tests_run - g_tests_failed, g_tests_run);
    return (g_tests_failed == 0) ? 0 : 1;
}
