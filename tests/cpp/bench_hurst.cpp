/**
 * Performance benchmarks for sqt::hurst.
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
 *   Windows  (VS generator) : build\tests\cpp\Release\bench_hurst.exe
 *   Windows  (Ninja)        : build\tests\cpp\bench_hurst.exe
 *   Linux / macOS           : ./build/tests/cpp/bench_hurst
 */

#include "sqt/hurst.hpp"

#include <algorithm>
#include <chrono>
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

static std::vector<double> make_noise(int n, unsigned seed = 42) {
    std::vector<double> out(static_cast<std::size_t>(n));
    unsigned state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        out[static_cast<std::size_t>(i)] =
            static_cast<double>(static_cast<int>(state)) / 2147483648.0;
    }
    return out;
}


// ── Result tracker ────────────────────────────────────────────────────────────

static int g_failures = 0;

// Prints one benchmark row and checks the conservative upper bound.
// The limit is intentionally generous (50-100× above expected) so it only
// fires on obvious regressions (debug build, wrong optimisation flags, etc.).
static void report(const std::string& label, double ms, double limit_ms) {
    const bool ok = ms <= limit_ms;
    if (!ok) ++g_failures;

    std::printf("  %-50s  %7.3f ms  [limit %5.0f ms]  %s\n",
                label.c_str(), ms, limit_ms, ok ? "PASS" : "FAIL");
}


// ── Benchmark functions ───────────────────────────────────────────────────────

static void bench_hurst_dfa() {
    std::printf("hurst_exponent — DFA:\n");

    for (int n : {500, 1000, 2000, 5000}) {
        auto data = make_noise(n);
        // Reps: 20 for smaller n, 10 for larger
        int reps = (n <= 1000) ? 20 : 10;
        double ms = median_ms([&] {
            sqt::hurst_exponent(data.data(), static_cast<std::size_t>(n),
                                "dfa", 10, -1);
        }, 3, reps);

        // Upper bounds: 50 ms per call at n=500 is already 100-500× above
        // what a Release build delivers (~0.1–0.5 ms).
        double limit = 50.0 * (n / 500.0);
        report("hurst_dfa  n=" + std::to_string(n), ms, limit);
    }
}

static void bench_hurst_rs() {
    std::printf("\nhurst_exponent — R/S:\n");

    for (int n : {500, 1000, 2000}) {
        auto  data = make_noise(n);
        int   reps = (n <= 1000) ? 20 : 10;
        double ms = median_ms([&] {
            sqt::hurst_exponent(data.data(), static_cast<std::size_t>(n),
                                "rs", 10, -1);
        }, 3, reps);

        double limit = 50.0 * (n / 500.0);
        report("hurst_rs   n=" + std::to_string(n), ms, limit);
    }
}

static void bench_rolling_hurst() {
    std::printf("\nrolling_hurst — DFA:\n");

    struct Case { int n; int window; int step; double limit_ms; };
    const Case cases[] = {
        //  n      window  step   limit
        { 1000,    200,    1,    2000.0 },  // baseline: full step=1 pass
        { 2000,    200,    1,    5000.0 },  // documented headline case
        { 2000,    252,    5,    2000.0 },  // step=5 (5× fewer calls)
        { 5000,    252,    1,   15000.0 },  // large series
    };

    for (const auto& c : cases) {
        auto data = make_noise(c.n);
        int  reps = (c.n <= 2000) ? 5 : 3;
        double ms = median_ms([&] {
            sqt::rolling_hurst(data.data(), static_cast<std::size_t>(c.n),
                               c.window, c.step, "dfa", 10);
        }, 2, reps);

        std::string label = "rolling  n=" + std::to_string(c.n)
                          + "  window=" + std::to_string(c.window)
                          + "  step="   + std::to_string(c.step);
        report(label, ms, c.limit_ms);
    }
}


// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    std::printf("=================================================================\n");
    std::printf(" sqt::hurst  C++ benchmark  (Release build expected)\n");
    std::printf("=================================================================\n\n");

    bench_hurst_dfa();
    bench_hurst_rs();
    bench_rolling_hurst();

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
