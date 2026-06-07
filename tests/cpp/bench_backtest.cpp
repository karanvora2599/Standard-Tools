/**
 * Performance benchmarks for sqt::run_strategy.
 *
 * Measures median wall-clock time for the main hot paths and asserts
 * conservative upper bounds — intended to catch catastrophic regressions
 * (e.g. accidental debug build, missing optimisation flags), not to enforce
 * tight performance targets.
 *
 * Build (SQT_BUILD_TESTS=ON is required):
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run directly:
 *   Windows  (VS generator) : build\tests\cpp\Release\bench_backtest.exe
 *   Windows  (Ninja)        : build\tests\cpp\bench_backtest.exe
 *   Linux / macOS           : ./build/tests/cpp/bench_backtest
 */

#include "sqt/backtest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>


// ── Timing ───────────────────────────────────────────────────────────────────

using Clock = std::chrono::high_resolution_clock;

template <typename Fn>
static double median_ms(Fn&& fn, int warmup, int reps) {
    for (int i = 0; i < warmup; ++i) fn();

    std::vector<double> times(static_cast<std::size_t>(reps));
    for (int i = 0; i < reps; ++i) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        times[static_cast<std::size_t>(i)] =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::sort(times.begin(), times.end());
    return times[static_cast<std::size_t>(reps) / 2];
}


// ── Synthetic data ────────────────────────────────────────────────────────────

static std::vector<double> make_prices(int n, unsigned seed = 42) {
    std::vector<double> out(static_cast<std::size_t>(n));
    unsigned state = seed;
    double price = 100.0;
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        double ret = static_cast<double>(static_cast<int>(state)) / 2147483648.0 * 0.02;
        price *= (1.0 + ret);
        out[static_cast<std::size_t>(i)] = price;
    }
    return out;
}

static std::vector<double> make_signals(int n, unsigned seed = 99) {
    std::vector<double> out(static_cast<std::size_t>(n));
    unsigned state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        // Alternating long / flat / short signals
        int v = static_cast<int>(state >> 30) % 3;
        out[static_cast<std::size_t>(i)] = (v == 0) ? 1.0 : (v == 1) ? 0.0 : -1.0;
    }
    return out;
}


// ── Result tracker ────────────────────────────────────────────────────────────

static int g_failures = 0;

// Upper bounds are intentionally generous (50-100× above expected Release
// performance) so this only fires on obvious regressions.
static void report(const std::string& label, double ms, double limit_ms) {
    const bool ok = ms <= limit_ms;
    if (!ok) ++g_failures;

    std::printf("  %-50s  %7.3f ms  [limit %5.0f ms]  %s\n",
                label.c_str(), ms, limit_ms, ok ? "PASS" : "FAIL");
}


// ── Benchmark functions ───────────────────────────────────────────────────────

static void bench_run_strategy_long_only() {
    std::printf("run_strategy — long-only (all costs):\n");

    for (int n : {500, 1000, 2000, 5000, 10000}) {
        auto prices  = make_prices(n);
        auto signals = std::vector<double>(static_cast<std::size_t>(n), 1.0);

        int reps = (n <= 2000) ? 50 : 20;
        double ms = median_ms([&] {
            sqt::run_strategy(prices.data(), signals.data(),
                              static_cast<std::size_t>(n),
                              10000.0, 0.001, 0.0005);
        }, 5, reps);

        // Conservative limit: 10 ms per call at n=500 is already 50-100× above
        // what a Release build delivers (~0.05–0.2 ms).
        double limit = 10.0 * (n / 500.0);
        report("long_only  n=" + std::to_string(n), ms, limit);
    }
}

static void bench_run_strategy_mixed_signals() {
    std::printf("\nrun_strategy — mixed L/F/S signals (all costs):\n");

    for (int n : {500, 1000, 2000, 5000}) {
        auto prices  = make_prices(n, 42);
        auto signals = make_signals(n, 99);

        int reps = (n <= 1000) ? 50 : 20;
        double ms = median_ms([&] {
            sqt::run_strategy(prices.data(), signals.data(),
                              static_cast<std::size_t>(n),
                              10000.0, 0.001, 0.0005);
        }, 5, reps);

        double limit = 10.0 * (n / 500.0);
        report("mixed_sig  n=" + std::to_string(n), ms, limit);
    }
}

static void bench_run_strategy_no_costs() {
    std::printf("\nrun_strategy — no transaction costs:\n");

    for (int n : {2000, 5000, 10000}) {
        auto prices  = make_prices(n, 7);
        auto signals = make_signals(n, 13);

        int reps = (n <= 5000) ? 20 : 10;
        double ms = median_ms([&] {
            sqt::run_strategy(prices.data(), signals.data(),
                              static_cast<std::size_t>(n),
                              10000.0, 0.0, 0.0);
        }, 3, reps);

        double limit = 10.0 * (n / 500.0);
        report("no_costs   n=" + std::to_string(n), ms, limit);
    }
}


// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::printf("=================================================================\n");
    std::printf(" sqt::run_strategy  C++ benchmark  (Release build expected)\n");
    std::printf("=================================================================\n\n");

    bench_run_strategy_long_only();
    bench_run_strategy_mixed_signals();
    bench_run_strategy_no_costs();

    std::printf("\n");
    if (g_failures == 0) {
        std::printf("All performance assertions passed.\n");
    } else {
        std::fprintf(stderr, "%d performance assertion(s) failed — check build type.\n",
                     g_failures);
    }
    std::printf("=================================================================\n");
    return (g_failures == 0) ? 0 : 1;
}
