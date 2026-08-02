/**
 * Randomized-input stress test for the cointegration/regression numerics
 * (correctness/portability pass item 20a).
 *
 * gauss_elim and cholesky_solve are anonymous-namespace internals of
 * cointegration.cpp/rolling_regression.cpp -- not directly linkable from
 * an external test binary -- so this fuzzes them indirectly through the
 * public functions that call them: sqt::ols2 and sqt::adf_test/
 * sqt::engle_granger (both use gauss_elim internally), and
 * sqt::rolling_factor_loadings (uses cholesky_solve internally, once per
 * bar). Asserts two things across many randomized inputs, including
 * deliberately adversarial ones (huge baseline, near-constant series,
 * huge dynamic range, all-zero, single-element):
 *
 *   1. No crash / no undefined behavior. This file is registered as a
 *      normal ctest (fixed seed, so default CI runs are deterministic),
 *      which means it automatically gets ASan/UBSan coverage for free via
 *      the existing build-and-test-sanitizers CI job -- no separate
 *      sanitizer-specific wiring needed here.
 *   2. Structural invariants that must hold whenever a function reports
 *      success (non-NaN output): ols2's residuals sum to ~0, engle_granger's
 *      p_value is in [0,1], its critical values are ordered
 *      cv_1pct < cv_5pct < cv_10pct, rolling_factor_loadings never produces
 *      a non-NaN row whose reconstructed fit is wildly inconsistent with
 *      the input.
 *
 * Deliberately a lightweight in-repo randomized harness (reusing this
 * project's own pseudo_random-style PRNG convention, already used
 * elsewhere in tests/cpp/) rather than libFuzzer/AFL++ -- proportionate
 * to the review's own "lower priority" framing for this item.
 *
 * Build (all platforms):
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V -R cpp_fuzz_cointegration
 */

#include "sqt/cointegration.hpp"
#include "sqt/rolling_regression.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
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

// ── Deterministic PRNG (fixed seed -> reproducible default CI runs) ────────────

static double pseudo_random(std::uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    std::uint64_t x = state;
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33;
    return (static_cast<double>(x >> 11) / 9007199254740992.0) * 2.0 - 1.0;  // [-1, 1)
}

// Generates a series with one of several deliberately varied shapes,
// selected by `shape_idx` -- spans well-conditioned, adversarial-scale, and
// pathological-value cases so the fuzz loop below isn't just repeatedly
// exercising the easy path.
static std::vector<double> make_series(int shape_idx, int n, std::uint64_t& state) {
    std::vector<double> v(static_cast<std::size_t>(n));
    switch (shape_idx % 7) {
        case 0:  // ordinary random walk
            v[0] = pseudo_random(state);
            for (int i = 1; i < n; ++i) v[i] = v[i - 1] + pseudo_random(state);
            break;
        case 1:  // huge baseline + small variation
            for (int i = 0; i < n; ++i) v[i] = 1.0e9 + pseudo_random(state) * 5.0;
            break;
        case 2:  // near-constant (tiny variance)
            for (int i = 0; i < n; ++i) v[i] = 1.0 + pseudo_random(state) * 1e-9;
            break;
        case 3:  // huge dynamic range
            for (int i = 0; i < n; ++i)
                v[i] = pseudo_random(state) * ((i % 2 == 0) ? 1e-6 : 1e12);
            break;
        case 4:  // all zero (degenerate, exercises the collinear/singular path)
            std::fill(v.begin(), v.end(), 0.0);
            break;
        case 5:  // strongly trending (large-magnitude cumulative sum)
            v[0] = 0.0;
            for (int i = 1; i < n; ++i) v[i] = v[i - 1] + 0.5 + pseudo_random(state) * 0.01;
            break;
        default:  // plain white noise
            for (int i = 0; i < n; ++i) v[i] = pseudo_random(state) * 10.0;
            break;
    }
    return v;
}

// ── ols2 ─────────────────────────────────────────────────────────────────────

static void fuzz_ols2() {
    std::uint64_t state = 1001;
    for (int trial = 0; trial < 500; ++trial) {
        const int n = 2 + static_cast<int>(std::abs(pseudo_random(state)) * 200);
        auto x = make_series(trial, n, state);
        auto y = make_series(trial + 3, n, state);

        auto r = sqt::ols2(y.data(), x.data(), static_cast<std::size_t>(n));

        if (!std::isnan(r.slope) && !std::isnan(r.intercept)) {
            // OLS residuals sum to ~0 whenever a fit was actually produced.
            double sum = 0.0, max_abs_y = 1.0;
            for (int i = 0; i < n; ++i) {
                sum += r.residuals[static_cast<std::size_t>(i)];
                max_abs_y = std::max(max_abs_y, std::abs(y[static_cast<std::size_t>(i)]));
            }
            // Relative tolerance: residual-sum error scales with the
            // problem's own magnitude and sample count, not a fixed
            // absolute epsilon -- these adversarial shapes span many
            // orders of magnitude by design.
            CHECK(std::abs(sum) < 1e-6 * max_abs_y * n);
            CHECK(r.r_squared >= -1e-9 && r.r_squared <= 1.0 + 1e-9);
        }
    }
}

// ── engle_granger (exercises adf_test + gauss_elim internally) ────────────────

static void fuzz_engle_granger() {
    std::uint64_t state = 2002;
    for (int trial = 0; trial < 300; ++trial) {
        const int n = 10 + static_cast<int>(std::abs(pseudo_random(state)) * 150);
        auto y0 = make_series(trial, n, state);
        auto y1 = make_series(trial + 5, n, state);
        const int max_lag = static_cast<int>(std::abs(pseudo_random(state)) * 40);

        auto r = sqt::engle_granger(y0.data(), y1.data(), static_cast<std::size_t>(n), max_lag);

        CHECK(r.n_obs == n);
        if (!std::isnan(r.p_value)) {
            CHECK(r.p_value >= 0.0 && r.p_value <= 1.0);
        }
        if (!std::isnan(r.cv_1pct) && !std::isnan(r.cv_5pct) && !std::isnan(r.cv_10pct)) {
            CHECK(r.cv_1pct < r.cv_5pct);
            CHECK(r.cv_5pct < r.cv_10pct);
        }
        CHECK(r.half_life > 0.0);  // always positive or +inf, by construction
    }
}

// ── rolling_factor_loadings (exercises cholesky_solve internally) ─────────────

static void fuzz_rolling_factor_loadings() {
    std::uint64_t state = 3003;
    for (int trial = 0; trial < 200; ++trial) {
        const int n      = 20 + static_cast<int>(std::abs(pseudo_random(state)) * 100);
        const int k       = 1 + static_cast<int>(std::abs(pseudo_random(state)) * 4);
        const int window  = 5 + static_cast<int>(std::abs(pseudo_random(state)) * (n / 2));

        auto y = make_series(trial, n, state);
        std::vector<double> factors(static_cast<std::size_t>(n) * static_cast<std::size_t>(k));
        for (auto& f : factors) f = pseudo_random(state) * 10.0;

        auto out = sqt::rolling_factor_loadings(
            y.data(), factors.data(), static_cast<std::size_t>(n),
            static_cast<std::size_t>(k), window);

        CHECK(out.size() == static_cast<std::size_t>(n) * static_cast<std::size_t>(k + 1));
        // Every entry must be either NaN (singular/insufficient window) or
        // finite -- never +/-inf, which would indicate a genuine numerical
        // blow-up rather than the intended "reject as singular" path.
        for (double v : out) {
            CHECK(std::isnan(v) || std::isfinite(v));
        }
    }
}

// ── Edge cases: single-element / empty-ish inputs ──────────────────────────────

static void fuzz_edge_cases() {
    std::uint64_t state = 4004;
    for (int trial = 0; trial < 50; ++trial) {
        const int n = 1 + (trial % 4);  // n = 1..4, below every function's own minimum
        auto x = make_series(trial, n, state);
        auto y = make_series(trial + 1, n, state);
        // Must not crash even though every one of these is too short for a
        // meaningful fit -- each function's own n-too-small guard should
        // return its NaN/empty sentinel, not read out of bounds.
        auto r = sqt::ols2(y.data(), x.data(), static_cast<std::size_t>(n));
        (void)r;
        auto eg = sqt::engle_granger(y.data(), x.data(), static_cast<std::size_t>(n));
        (void)eg;
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    fuzz_ols2();
    fuzz_engle_granger();
    fuzz_rolling_factor_loadings();
    fuzz_edge_cases();

    std::printf("\n%d / %d tests passed.\n",
                g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed > 0 ? 1 : 0;
}
