/**
 * C++ unit tests for sqt::garch11_variance_recursion (GARCH(1,1) kernel).
 *
 * Build (all platforms):
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V -R cpp_garch
 */

#include "sqt/garch.hpp"

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


// ── Tests ─────────────────────────────────────────────────────────────────────

static void test_empty_input_returns_empty() {
    auto result = sqt::garch11_variance_recursion(nullptr, 0, 1e-6, 0.05, 0.9);
    CHECK(result.empty());
}

static void test_returns_correct_length() {
    std::vector<double> resid_sq(200, 1e-4);
    auto result = sqt::garch11_variance_recursion(
        resid_sq.data(), resid_sq.size(), 1e-6, 0.05, 0.9);
    CHECK(result.size() == 200);
}

static void test_first_value_is_mean_of_resid_sq() {
    std::vector<double> resid_sq = {1e-4, 4e-4, 9e-4, 1.6e-3};
    double mean = 0.0;
    for (double v : resid_sq) mean += v;
    mean /= static_cast<double>(resid_sq.size());

    auto result = sqt::garch11_variance_recursion(
        resid_sq.data(), resid_sq.size(), 1e-6, 0.05, 0.9);
    CHECK_NEAR(result[0], mean, 1e-12);
}

static void test_recursion_matches_hand_computation() {
    // sigma2[1] = omega + alpha*resid_sq[0] + beta*sigma2[0]
    std::vector<double> resid_sq = {2e-4, 5e-4, 1e-4};
    const double omega = 1e-6, alpha = 0.08, beta = 0.85;

    auto result = sqt::garch11_variance_recursion(
        resid_sq.data(), resid_sq.size(), omega, alpha, beta);

    const double expected_sigma2_1 = omega + alpha * resid_sq[0] + beta * result[0];
    CHECK_NEAR(result[1], expected_sigma2_1, 1e-12);

    const double expected_sigma2_2 = omega + alpha * resid_sq[1] + beta * result[1];
    CHECK_NEAR(result[2], expected_sigma2_2, 1e-12);
}

static void test_floor_at_min_sigma2() {
    // omega negative and alpha/beta zero drives every subsequent value
    // below the floor -- must clamp to 1e-12, not go negative.
    std::vector<double> resid_sq(30, 0.0);
    auto result = sqt::garch11_variance_recursion(
        resid_sq.data(), resid_sq.size(), -1.0, 0.0, 0.0);
    bool all_floored = true;
    for (double v : result) {
        if (v < 1e-12) { all_floored = false; break; }
    }
    CHECK(all_floored);
}

static void test_single_observation() {
    std::vector<double> resid_sq = {3e-4};
    auto result = sqt::garch11_variance_recursion(
        resid_sq.data(), resid_sq.size(), 1e-6, 0.05, 0.9);
    CHECK(result.size() == 1);
    CHECK_NEAR(result[0], 3e-4, 1e-12);
}


// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test_empty_input_returns_empty();
    test_returns_correct_length();
    test_first_value_is_mean_of_resid_sq();
    test_recursion_matches_hand_computation();
    test_floor_at_min_sigma2();
    test_single_observation();

    std::printf("\n%d / %d tests passed.\n",
                g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed > 0 ? 1 : 0;
}
