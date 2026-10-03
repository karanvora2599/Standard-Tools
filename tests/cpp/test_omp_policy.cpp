/**
 * C++ unit tests for sqt::omp_policy: when a region goes parallel, how many
 * threads it takes, and how the pooled sort shares them among concurrent
 * calls.
 *
 * The decision is tested through omp_policy::detail::decide and share, which
 * take the environment as arguments, so every branch -- the time rule, the
 * SQT_OMP_MIN_WORK override, the thread cap -- is reachable from one
 * process. The cached environment readers and the call counter are tested
 * once, with SQT_NUM_THREADS set before anything reads it.
 *
 * Header-only: built without OpenMP, so OpenMP's own default is one thread
 * and every count below comes from SQT_NUM_THREADS.
 *
 * Build:
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V
 */

#include "sqt/omp_policy.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace op = sqt::omp_policy;

static int g_tests_run = 0;
static int g_tests_failed = 0;

static void expect(bool condition, const char* what) {
    ++g_tests_run;
    if (!condition) {
        ++g_tests_failed;
        std::printf("  FAIL: %s\n", what);
    }
}

static void expect_eq(long long got, long long want, const char* what) {
    ++g_tests_run;
    if (got != want) {
        ++g_tests_failed;
        std::printf("  FAIL: %s (got %lld, want %lld)\n", what, got, want);
    }
}

static void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);  // "" removes it
#else
    if (*value == '\0') unsetenv(name);
    else setenv(name, value, 1);
#endif
}

// ── The time rule ─────────────────────────────────────────────────────────────

static void test_the_threshold_is_serial_time_not_a_unit_count() {
    // The same 30,000 units: 4.2 ms of Engle-Granger, 28.5 us of implied
    // volatility. The first goes parallel, the second does not.
    expect(op::detail::decide(30, 1000, op::cost::engle_granger, 0, -1),
           "30,000 Engle-Granger units (4.2 ms) go parallel");
    expect(!op::detail::decide(60, 500, op::cost::implied_volatility, 0, -1),
           "30,000 implied-volatility units (28.5 us) stay serial");
}

static void test_the_boundary_is_150_microseconds() {
    // 150 us at 5 ns a unit is exactly 30,000 units.
    const op::serial_cost five{5.0, 0.0};
    expect(!op::detail::decide(2, 14'999, five, 0, -1), "149,990 ns stays serial");
    expect(op::detail::decide(2, 15'000, five, 0, -1), "150,000 ns goes parallel");
    // Engle-Granger at 252 bars: 4 pairs are 141 us, 5 are 176 us.
    expect(!op::detail::decide(4, 252, op::cost::engle_granger, 0, -1),
           "4 Engle-Granger pairs of 252 bars stay serial");
    expect(op::detail::decide(5, 252, op::cost::engle_granger, 0, -1),
           "5 Engle-Granger pairs of 252 bars go parallel");
    // A 476-contract implied-volatility chain is 226 us: parallel.
    expect(op::detail::decide(476, 500, op::cost::implied_volatility, 0, -1),
           "a 476-contract implied-volatility chain goes parallel");
}

static void test_a_per_task_cost_counts_for_every_task() {
    // A bootstrap path reseeds its generator whatever its horizon: 1,000
    // paths of 21 days are ~770 us, though only 21,000 units.
    expect(op::detail::decide(1000, 21, op::cost::bootstrap_paths, 0, -1),
           "1,000 x 21 bootstrap paths go parallel");
    expect(!op::detail::decide(100, 21, op::cost::bootstrap_paths, 0, -1),
           "100 x 21 bootstrap paths (77 us) stay serial");
    const op::serial_cost per_task_only{0.0, 75'000.0};
    expect(op::detail::decide(2, 0, per_task_only, 0, -1),
           "two tasks of 75 us each reach the threshold with no units at all");
}

static void test_one_task_or_one_thread_is_always_serial() {
    expect(!op::detail::decide(1, 1'000'000'000, op::cost::engle_granger, 0, -1),
           "one task never goes parallel, however long");
    expect(!op::detail::decide(0, 1'000'000, op::cost::engle_granger, 0, -1),
           "no tasks never go parallel");
    expect(!op::detail::decide(1000, 1000, op::cost::engle_granger, 1, -1),
           "SQT_NUM_THREADS=1 is always serial");
    expect(!op::detail::decide(1000, 1000, op::cost::engle_granger, 1, 0),
           "SQT_NUM_THREADS=1 is serial even with SQT_OMP_MIN_WORK=0");
    expect(op::detail::decide(1000, 1000, op::cost::engle_granger, 2, -1),
           "a cap of two still goes parallel");
}

// ── SQT_OMP_MIN_WORK: the old rule, exactly ───────────────────────────────────

static void test_the_override_is_the_old_unit_rule_and_ignores_cost() {
    // With the override set, the cost is not consulted: the same units
    // decide the same way at a cost of zero and at a cost of a microsecond.
    const op::serial_cost free_unit{0.0, 0.0};
    const op::serial_cost dear_unit{1000.0, 1e9};
    for (const op::serial_cost c : {free_unit, dear_unit, op::cost::greeks}) {
        expect(!op::detail::decide(2, 24'999, c, 0, 50'000),
               "49,998 units stay serial under SQT_OMP_MIN_WORK=50000");
        expect(op::detail::decide(2, 25'000, c, 0, 50'000),
               "50,000 units go parallel under SQT_OMP_MIN_WORK=50000");
        expect(op::detail::decide(2, 0, c, 0, 0),
               "SQT_OMP_MIN_WORK=0 sends any two tasks parallel");
        expect(!op::detail::decide(1, 1'000'000, c, 0, 0),
               "SQT_OMP_MIN_WORK=0 still leaves one task serial");
    }
}

static void test_the_two_argument_form_is_the_old_default() {
    // Every call site states a cost; a call that does not keeps the rule it
    // was written against, 50,000 units.
    expect(!op::detail::decide(2, 24'999, op::serial_cost{0.0, 0.0}, 0,
                               op::kLegacyMinWork),
           "49,998 legacy units stay serial");
    expect(op::detail::decide(2, 25'000, op::serial_cost{0.0, 0.0}, 0,
                              op::kLegacyMinWork),
           "50,000 legacy units go parallel");
}

// ── Sharing the threads (the pooled sort's arithmetic) ────────────────────────

static void test_a_call_alone_gets_exactly_the_configured_count() {
    expect_eq(op::detail::share(0, 16, 0), 0, "no calls counted: OpenMP's default");
    expect_eq(op::detail::share(0, 16, 1), 0, "one call: OpenMP's default");
    expect_eq(op::detail::share(8, 16, 1), 8, "one call: SQT_NUM_THREADS");
    expect_eq(op::detail::share(32, 16, 1), 32,
              "one call: SQT_NUM_THREADS above the core count, as before");
}

static void test_concurrent_calls_split_the_total() {
    expect_eq(op::detail::share(0, 16, 2), 8, "two calls share 16 threads");
    expect_eq(op::detail::share(0, 16, 3), 5, "three calls: 5 each");
    expect_eq(op::detail::share(0, 16, 16), 1, "sixteen calls: 1 each");
    expect_eq(op::detail::share(0, 16, 40), 1, "never fewer than one thread");
    expect_eq(op::detail::share(8, 16, 2), 4,
              "SQT_NUM_THREADS is the total that concurrent calls share");
    expect_eq(op::detail::share(8, 16, 3), 2, "8 threads over 3 calls: 2 each");
}

// ── The counter, through the real environment readers ─────────────────────────

static void test_a_counted_call_is_released_on_every_exit() {
    expect_eq(op::active_calls().load(), 0, "nothing counted at start");
    {
        const op::parallel_call serial(false);
        expect(!serial.parallel(), "a serial call says so");
        expect_eq(op::active_calls().load(), 0, "a serial call is not counted");
        const op::parallel_call parallel(true);
        expect(parallel.parallel(), "a parallel call says so");
        expect_eq(op::active_calls().load(), 1, "a parallel call is counted");
        expect_eq(op::max_threads(), 12, "alone: SQT_NUM_THREADS");
        expect_eq(op::shared_threads(), 12, "the pooled sort alone: SQT_NUM_THREADS");
        {
            const op::parallel_call second(true);
            expect_eq(op::active_calls().load(), 2, "two calls counted");
            expect_eq(op::shared_threads(), 6, "two calls: the pooled sort takes half");
            expect_eq(op::max_threads(), 12,
                      "two calls: every other region still takes SQT_NUM_THREADS");
        }
        expect_eq(op::active_calls().load(), 1, "released at end of scope");
    }
    expect_eq(op::active_calls().load(), 0, "all released");
    try {
        const op::parallel_call counted(true);
        throw std::runtime_error("a worker failed");
    } catch (const std::runtime_error&) {
    }
    expect_eq(op::active_calls().load(), 0, "released when the call throws");
}

static void test_the_environment_readers() {
    expect_eq(op::thread_cap(), 12, "SQT_NUM_THREADS is read");
    expect_eq(op::min_work_override(), -1, "SQT_OMP_MIN_WORK unset: the time rule");
    expect(op::worth_parallel(5, 252, op::cost::engle_granger),
           "worth_parallel applies the time rule");
    expect(!op::worth_parallel(4, 252, op::cost::engle_granger),
           "and its boundary");
    expect(!op::worth_parallel(2, 24'999), "the two-argument form: 50,000 units");
    expect(op::worth_parallel(2, 25'000), "the two-argument form's boundary");
    const op::parallel_call decided(5, 252, op::cost::engle_granger);
    expect(decided.parallel(), "a call constructed from its work decides it");
    expect_eq(op::active_calls().load(), 1, "and is counted");
}

static void test_threads_running_at_once_each_see_their_share() {
    // Four threads each hold a counted call. Each checks only once all four
    // are counted, and none lets go until all four have checked, so every
    // check sees four calls: the pooled sort is offered 12 / 4 = 3 threads,
    // and every other region still all 12.
    constexpr int kCallers = 4;
    std::mutex m;
    std::condition_variable cv;
    int entered = 0;
    int checked = 0;
    int saw_share = 0;
    int saw_total = 0;
    std::vector<std::thread> callers;
    for (int i = 0; i < kCallers; ++i) {
        callers.emplace_back([&] {
            const op::parallel_call call(true);
            std::unique_lock<std::mutex> lock(m);
            ++entered;
            cv.notify_all();
            cv.wait(lock, [&] { return entered == kCallers; });
            if (op::shared_threads() == 12 / kCallers) ++saw_share;
            if (op::max_threads() == 12) ++saw_total;
            ++checked;
            cv.notify_all();
            cv.wait(lock, [&] { return checked == kCallers; });
        });
    }
    for (auto& t : callers) t.join();
    expect_eq(saw_share, kCallers, "each of four pooled sorts saw 3 of 12 threads");
    expect_eq(saw_total, kCallers, "each of four other regions saw all 12");
    expect_eq(op::active_calls().load(), 0, "all four released");
}

int main() {
    // Before anything reads them: the readers cache on first use.
    set_env("SQT_NUM_THREADS", "12");
    set_env("SQT_OMP_MIN_WORK", "");

    std::printf("=== sqt omp_policy tests ===\n");
    test_the_threshold_is_serial_time_not_a_unit_count();
    test_the_boundary_is_150_microseconds();
    test_a_per_task_cost_counts_for_every_task();
    test_one_task_or_one_thread_is_always_serial();
    test_the_override_is_the_old_unit_rule_and_ignores_cost();
    test_the_two_argument_form_is_the_old_default();
    test_a_call_alone_gets_exactly_the_configured_count();
    test_concurrent_calls_split_the_total();
    test_a_counted_call_is_released_on_every_exit();
    test_threads_running_at_once_each_see_their_share();
    test_the_environment_readers();

    std::printf("%d checks, %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
