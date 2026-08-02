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
#include <stdexcept>
#include <string>
#include <utility>
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

static void test_hurst_invalid_method_throws() {
    // Regression test: any method string other than exactly "dfa"/"rs" used
    // to silently fall through to R/S instead of honoring the documented
    // "dfa" or "rs" contract -- a typo (e.g. "DFA", "r_s", "") would run a
    // different estimator than intended while echoing the typo'd string
    // back in the result, not raise.
    auto data = white_noise(600);
    bool threw = false;
    try {
        sqt::hurst_exponent(data.data(), data.size(), "DFA", 10, -1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK_TRUE(threw);

    threw = false;
    try {
        sqt::hurst_exponent(data.data(), data.size(), "", 10, -1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK_TRUE(threw);
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

static void test_rolling_hurst_invalid_method_throws() {
    // Same contract as hurst_exponent -- validated eagerly (before the
    // sliding-window loop) so even an input too short to run any iteration
    // still raises rather than silently returning an all-NaN series.
    std::vector<double> data = {0.1, 0.2, 0.3};  // shorter than any window
    bool threw = false;
    try {
        sqt::rolling_hurst(data.data(), data.size(), 100, 1, "rsx", 10);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

// ── rolling_hurst OpenMP + scratch-buffer path vs. direct hurst_exponent() ──
//
// rolling_hurst_into() internally uses a scratch-buffer sibling
// (hurst_exponent_scratch/dfa_impl) and (when SQT_HAS_OPENMP) runs the
// window loop in parallel -- neither of those internal implementation
// details is exposed, but their combined effect must produce EXACTLY the
// values the public, unchanged hurst_exponent() would produce for the same
// window slice, called directly. This test isolates both risk surfaces at
// once: any scratch-reuse bug or any OpenMP data race would show up as a
// mismatch here. Run this executable under different OMP_NUM_THREADS
// values (1/2/4+) at the process level to additionally confirm exact
// reproducibility regardless of thread count/scheduling.

static void test_rolling_hurst_matches_direct_hurst_exponent_dfa() {
    // Tolerance, not bit-identical: rolling_hurst_into's internal "dfa"
    // path uses dfa_onepass() (item H's one-pass sufficient-statistics
    // reformulation), a genuine floating-point reassociation vs. the
    // public hurst_exponent()'s 3-pass dfa_impl(). Verified across a range
    // of series below, including deliberately ill-conditioned ones.
    auto data = white_noise(500, /*seed=*/2024);
    const int window = 80, step = 3, min_window = 10;
    auto out = sqt::rolling_hurst(data.data(), data.size(), window, step, "dfa", min_window);

    int checked = 0;
    for (int i = window - 1; i < static_cast<int>(data.size()); i += step) {
        auto direct = sqt::hurst_exponent(
            data.data() + (i - window + 1), static_cast<std::size_t>(window),
            "dfa", min_window, -1);
        if (std::isnan(direct.hurst)) {
            CHECK_NAN(out[i]);
        } else {
            CHECK_NEAR(out[i], direct.hurst, 1e-9);
        }
        ++checked;
    }
    CHECK_TRUE(checked > 5);
}

static void test_dfa_onepass_tolerance_ill_conditioned() {
    // Item H's hard numerical-stability gate: dfa_onepass's sum-of-squares
    // style accumulation (Sy, Syy, S_jy) is, in general, less robust to
    // catastrophic cancellation than dfa_impl's deviation-from-mean style
    // (seg_mean subtracted before squaring). Stress-test with series shaped
    // to make that cancellation risk real: a strongly-trending cumulative
    // sum (large magnitude relative to local chunk variation -- exactly
    // what dfa's own Step 1 cumulative-sum transform produces for
    // real return series with any drift) and a near-constant series
    // (tiny variance, so relative error is easily amplified).
    std::vector<std::pair<std::string, std::vector<double>>> cases;

    // Strongly-trending input (large drift -> large-magnitude, near-linear
    // cumulative sum after DFA's own Step-1 transform).
    {
        auto noise = white_noise(400, /*seed=*/3);
        std::vector<double> v(400);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.05 + 0.001 * noise[i];
        cases.emplace_back("strong_trend", v);
    }
    // Near-constant input (tiny variance).
    {
        auto noise = white_noise(400, /*seed=*/5);
        std::vector<double> v(400);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = 1.0 + 1e-8 * noise[i];
        cases.emplace_back("near_constant", v);
    }
    // Ordinary white noise, as a control.
    {
        auto v = white_noise(400, /*seed=*/17);
        cases.emplace_back("white_noise", v);
    }

    for (const auto& [label, data] : cases) {
        const int window = 100, min_window = 10;
        auto out = sqt::rolling_hurst(data.data(), data.size(), window, 5, "dfa", min_window);
        for (int i = window - 1; i < static_cast<int>(data.size()); i += 5) {
            auto direct = sqt::hurst_exponent(
                data.data() + (i - window + 1), static_cast<std::size_t>(window),
                "dfa", min_window, -1);
            if (std::isnan(direct.hurst)) {
                CHECK_NAN(out[i]);
            } else {
                // .hurst is clamped to [0, 1.5] -- an absolute tolerance is
                // appropriate here (no need for a scale-relative one, this
                // isn't a quantity spanning many orders of magnitude).
                CHECK_NEAR(out[i], direct.hurst, 1e-6);
            }
        }
    }
}

static void test_rolling_hurst_matches_direct_hurst_exponent_rs() {
    auto data = white_noise(500, /*seed=*/99);
    const int window = 60, step = 2, min_window = 10;
    auto out = sqt::rolling_hurst(data.data(), data.size(), window, step, "rs", min_window);

    int checked = 0;
    for (int i = window - 1; i < static_cast<int>(data.size()); i += step) {
        auto direct = sqt::hurst_exponent(
            data.data() + (i - window + 1), static_cast<std::size_t>(window),
            "rs", min_window, -1);
        if (std::isnan(direct.hurst)) {
            CHECK_NAN(out[i]);
        } else {
            CHECK(out[i] == direct.hurst);
        }
        ++checked;
    }
    CHECK_TRUE(checked > 5);
}

static void test_rolling_hurst_large_step_matches_direct() {
    // A step that doesn't evenly divide (n - window) exercises the counted
    // rewrite's boundary math (count = (last_i - (window-1))/step + 1).
    auto data = white_noise(300, /*seed=*/7);
    const int window = 50, step = 17, min_window = 10;
    auto out = sqt::rolling_hurst(data.data(), data.size(), window, step, "dfa", min_window);

    int checked = 0;
    for (int i = window - 1; i < static_cast<int>(data.size()); i += step) {
        auto direct = sqt::hurst_exponent(
            data.data() + (i - window + 1), static_cast<std::size_t>(window),
            "dfa", min_window, -1);
        if (std::isnan(direct.hurst)) {
            CHECK_NAN(out[i]);
        } else {
            CHECK_NEAR(out[i], direct.hurst, 1e-9);  // tolerance: see item H comment above
        }
        ++checked;
    }
    CHECK_TRUE(checked > 2);
    // Positions not landed on by the strided walk must stay NaN.
    CHECK_NAN(out[window - 1 + 1]);  // one bar after the first hit, step=17
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
    test_hurst_invalid_method_throws();

    // rolling_hurst
    test_rolling_hurst_length();
    test_rolling_hurst_leading_nans();
    test_rolling_hurst_non_nan_count();
    test_rolling_hurst_step2();
    test_rolling_hurst_values_in_range();
    test_rolling_hurst_invalid_method_throws();
    test_rolling_hurst_matches_direct_hurst_exponent_dfa();
    test_rolling_hurst_matches_direct_hurst_exponent_rs();
    test_rolling_hurst_large_step_matches_direct();
    test_dfa_onepass_tolerance_ill_conditioned();

    std::printf("\n%d / %d tests passed.\n", g_tests_run - g_tests_failed, g_tests_run);
    return (g_tests_failed == 0) ? 0 : 1;
}
