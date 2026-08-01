/**
 * C++ unit tests for sqt::donchian_state_machine and
 * sqt::vwap_reversion_state_machine.
 *
 * Build (all platforms):
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V -R cpp_signals
 */

#include "sqt/signal_state_machines.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
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

static const double kNaN = std::numeric_limits<double>::quiet_NaN();


// ── Tests: donchian_state_machine ───────────────────────────────────────────────

static void test_donchian_empty_input() {
    auto result = sqt::donchian_state_machine(nullptr, nullptr, nullptr, 0);
    CHECK(result.empty());
}

static void test_donchian_enters_long_on_breakout() {
    std::vector<double> close     = {100.0, 101.0, 103.0};
    std::vector<double> entry_max = {100.5, 100.5, 100.5};
    std::vector<double> exit_min  = {90.0,  90.0,  90.0};
    auto result = sqt::donchian_state_machine(
        close.data(), entry_max.data(), exit_min.data(), 3);
    CHECK(result[0] == 0.0);
    CHECK(result[1] == 1.0);
    CHECK(result[2] == 1.0);
}

static void test_donchian_exits_on_breakdown() {
    std::vector<double> close     = {100.0, 101.0, 89.0};
    std::vector<double> entry_max = {100.5, 100.5, 100.5};
    std::vector<double> exit_min  = {90.0,  90.0,  90.0};
    auto result = sqt::donchian_state_machine(
        close.data(), entry_max.data(), exit_min.data(), 3);
    CHECK(result[0] == 0.0);
    CHECK(result[1] == 1.0);
    CHECK(result[2] == 0.0);
}

static void test_donchian_nan_warmup_outputs_zero() {
    std::vector<double> close     = {100.0, 100.0, 100.0};
    std::vector<double> entry_max = {kNaN,  kNaN,  102.0};
    std::vector<double> exit_min  = {kNaN,  kNaN,  98.0};
    auto result = sqt::donchian_state_machine(
        close.data(), entry_max.data(), exit_min.data(), 3);
    CHECK(result[0] == 0.0);
    CHECK(result[1] == 0.0);
}

static void test_donchian_nan_does_not_update_state() {
    // Position opens on bar 1, then bar 2 has NaN exit_min (should NOT
    // close the position or alter state), bar 3 sees a real breakdown.
    std::vector<double> close     = {100.0, 101.0, 95.0,  89.0};
    std::vector<double> entry_max = {100.5, 100.5, 100.5, 100.5};
    std::vector<double> exit_min  = {90.0,  90.0,  kNaN,  90.0};
    auto result = sqt::donchian_state_machine(
        close.data(), entry_max.data(), exit_min.data(), 4);
    CHECK(result[0] == 0.0);
    CHECK(result[1] == 1.0);
    CHECK(result[2] == 0.0);  // NaN bar: output 0.0, but state NOT updated
    CHECK(result[3] == 0.0);  // still in_pos=true carried from bar 1, exits now
}

static void test_donchian_stays_flat_without_breakout() {
    std::vector<double> close     = {100.0, 100.2, 100.1};
    std::vector<double> entry_max = {101.0, 101.0, 101.0};
    std::vector<double> exit_min  = {90.0,  90.0,  90.0};
    auto result = sqt::donchian_state_machine(
        close.data(), entry_max.data(), exit_min.data(), 3);
    CHECK(result[0] == 0.0);
    CHECK(result[1] == 0.0);
    CHECK(result[2] == 0.0);
}


// ── Tests: vwap_reversion_state_machine ─────────────────────────────────────────

static void test_vwap_empty_input() {
    auto result = sqt::vwap_reversion_state_machine(nullptr, nullptr, 0.02, 0);
    CHECK(result.empty());
}

static void test_vwap_enters_long_on_drop_below_threshold() {
    std::vector<double> close = {100.0, 97.0, 99.0};
    std::vector<double> vwap  = {100.0, 100.0, 100.0};
    auto result = sqt::vwap_reversion_state_machine(close.data(), vwap.data(), 0.02, 3);
    CHECK(result[0] == 0.0);
    CHECK(result[1] == 1.0);
    CHECK(result[2] == 1.0);
}

static void test_vwap_exits_on_recovery() {
    std::vector<double> close = {100.0, 97.0, 100.5};
    std::vector<double> vwap  = {100.0, 100.0, 100.0};
    auto result = sqt::vwap_reversion_state_machine(close.data(), vwap.data(), 0.02, 3);
    CHECK(result[0] == 0.0);
    CHECK(result[1] == 1.0);
    CHECK(result[2] == 0.0);
}

static void test_vwap_nan_warmup_outputs_zero() {
    std::vector<double> close = {100.0, 100.0, 100.0};
    std::vector<double> vwap  = {kNaN,  kNaN,  100.0};
    auto result = sqt::vwap_reversion_state_machine(close.data(), vwap.data(), 0.02, 3);
    CHECK(result[0] == 0.0);
    CHECK(result[1] == 0.0);
}

static void test_vwap_stays_flat_without_sufficient_drop() {
    std::vector<double> close = {100.0, 99.0, 98.5};
    std::vector<double> vwap  = {100.0, 100.0, 100.0};
    auto result = sqt::vwap_reversion_state_machine(close.data(), vwap.data(), 0.02, 3);
    CHECK(result[0] == 0.0);
    CHECK(result[1] == 0.0);
    CHECK(result[2] == 0.0);
}


// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test_donchian_empty_input();
    test_donchian_enters_long_on_breakout();
    test_donchian_exits_on_breakdown();
    test_donchian_nan_warmup_outputs_zero();
    test_donchian_nan_does_not_update_state();
    test_donchian_stays_flat_without_breakout();

    test_vwap_empty_input();
    test_vwap_enters_long_on_drop_below_threshold();
    test_vwap_exits_on_recovery();
    test_vwap_nan_warmup_outputs_zero();
    test_vwap_stays_flat_without_sufficient_drop();

    std::printf("\n%d / %d tests passed.\n",
                g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed > 0 ? 1 : 0;
}
