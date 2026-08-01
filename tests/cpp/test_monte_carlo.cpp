/**
 * C++ unit tests for sqt::simulate_forward_paths (Monte Carlo moving-block
 * bootstrap kernel).
 *
 * Build (all platforms):
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V -R cpp_monte_carlo
 */

#include "sqt/monte_carlo.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#ifdef SQT_HAS_OPENMP
#include <omp.h>
#endif

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

static void test_returns_correct_size() {
    std::vector<double> values(100, 0.001);
    auto result = sqt::simulate_forward_paths(
        values.data(), values.size(), 30, 200, 10, 10000.0, 1, true);
    CHECK(result.size() == 200 * 30);
}

static void test_empty_on_zero_horizon() {
    std::vector<double> values(100, 0.001);
    auto result = sqt::simulate_forward_paths(
        values.data(), values.size(), 0, 200, 10, 10000.0, 1, true);
    CHECK(result.empty());
}

static void test_empty_on_zero_simulations() {
    std::vector<double> values(100, 0.001);
    auto result = sqt::simulate_forward_paths(
        values.data(), values.size(), 30, 0, 10, 10000.0, 1, true);
    CHECK(result.empty());
}

static void test_empty_on_zero_history() {
    auto result = sqt::simulate_forward_paths(
        nullptr, 0, 30, 200, 10, 10000.0, 1, true);
    CHECK(result.empty());
}

static void test_empty_on_bad_block_size() {
    std::vector<double> values(20, 0.001);
    auto too_small = sqt::simulate_forward_paths(
        values.data(), values.size(), 30, 100, 0, 10000.0, 1, true);
    CHECK(too_small.empty());
    auto too_large = sqt::simulate_forward_paths(
        values.data(), values.size(), 30, 100, 21, 10000.0, 1, true);
    CHECK(too_large.empty());
}

static void test_empty_on_non_positive_initial_capital() {
    std::vector<double> values(20, 0.001);
    auto result = sqt::simulate_forward_paths(
        values.data(), values.size(), 30, 100, 5, 0.0, 1, true);
    CHECK(result.empty());
}

static void test_constant_returns_deterministic_compounding() {
    // Every historical bar has the same return r, so every resampled block
    // -- no matter which start index the RNG happens to draw -- is
    // identical. Every path must equal initial_capital * (1+r)^t exactly,
    // independent of the RNG.
    const double r = 0.002;
    std::vector<double> values(50, r);
    const int horizon = 25;
    const int n_sims  = 10;
    auto result = sqt::simulate_forward_paths(
        values.data(), values.size(), horizon, n_sims, 10, 10000.0, 7, true);
    CHECK(result.size() == static_cast<std::size_t>(horizon * n_sims));
    for (int i = 0; i < n_sims; ++i) {
        double expected = 10000.0;
        for (int t = 0; t < horizon; ++t) {
            expected *= (1.0 + r);
            CHECK_NEAR(result[static_cast<std::size_t>(i) * horizon + t], expected, 1e-6);
        }
    }
}

static void test_same_seed_reproducible() {
    std::vector<double> values(200);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.001 * static_cast<double>((i % 7)) - 0.003;
    }
    auto r1 = sqt::simulate_forward_paths(
        values.data(), values.size(), 40, 50, 15, 10000.0, 123, true);
    auto r2 = sqt::simulate_forward_paths(
        values.data(), values.size(), 40, 50, 15, 10000.0, 123, true);
    CHECK(r1.size() == r2.size());
    bool identical = true;
    for (std::size_t i = 0; i < r1.size(); ++i) {
        if (r1[i] != r2[i]) { identical = false; break; }
    }
    CHECK(identical);
}

static void test_result_independent_of_thread_count() {
#ifdef SQT_HAS_OPENMP
    // A per-path buffer accidentally shared across threads (a data race)
    // would make the result depend on how many threads actually ran the
    // loop. Force 1 thread vs. several for the identical seed+inputs and
    // require bit-identical output either way.
    std::vector<double> values(400);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.0007 * static_cast<double>((i * 53) % 31) - 0.008;
    }

    omp_set_num_threads(1);
    auto r_serial = sqt::simulate_forward_paths(
        values.data(), values.size(), 50, 2000, 20, 10000.0, 55, true);

    omp_set_num_threads(4);
    auto r_parallel = sqt::simulate_forward_paths(
        values.data(), values.size(), 50, 2000, 20, 10000.0, 55, true);

    CHECK(r_serial.size() == r_parallel.size());
    bool identical = true;
    for (std::size_t i = 0; i < r_serial.size(); ++i) {
        if (r_serial[i] != r_parallel[i]) { identical = false; break; }
    }
    CHECK(identical);
#endif
}

static void test_different_paths_diverge() {
    // With genuine return variation and multiple blocks per path, distinct
    // simulated paths within the same call should not all be identical --
    // sanity check that each path draws its own independent block starts.
    std::vector<double> values(300);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.0005 * static_cast<double>((i * 37) % 23) - 0.005;
    }
    auto result = sqt::simulate_forward_paths(
        values.data(), values.size(), 60, 20, 15, 10000.0, 5, true);
    bool any_diff = false;
    for (int i = 1; i < 20 && !any_diff; ++i) {
        for (int t = 0; t < 60; ++t) {
            if (result[static_cast<std::size_t>(i) * 60 + t] != result[t]) {
                any_diff = true;
                break;
            }
        }
    }
    CHECK(any_diff);
}


// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test_returns_correct_size();
    test_empty_on_zero_horizon();
    test_empty_on_zero_simulations();
    test_empty_on_zero_history();
    test_empty_on_bad_block_size();
    test_empty_on_non_positive_initial_capital();
    test_constant_returns_deterministic_compounding();
    test_same_seed_reproducible();
    test_result_independent_of_thread_count();
    test_different_paths_diverge();

    std::printf("\n%d / %d tests passed.\n",
                g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed > 0 ? 1 : 0;
}
