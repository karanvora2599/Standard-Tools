/**
 * C++ unit tests for sqt::order_queue_ahead and sqt::order_lifetimes.
 *
 * The Python-side tests hold both kernels to the loops in
 * analysis/order_events.py bit for bit on whole sessions. What is asserted
 * here are the planted cases a session-level comparison would find hardest
 * to localize: a CLEAR forgetting orders and levels, -0.0 and +0.0 being one
 * level, a partial fill, a decrement of an order never seen, the snapshot
 * explaining a later cancel, a NaT row, and a duration that leaves int64.
 *
 * Run directly: build\tests\cpp\test_order_events.exe, or via CTest.
 */

#include "sqt/order_events.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

static int g_tests_run = 0;
static int g_tests_failed = 0;

static void expect(bool condition, const char* what) {
    ++g_tests_run;
    if (!condition) {
        ++g_tests_failed;
        std::printf("  FAIL: %s\n", what);
    }
}

namespace {

constexpr std::int8_t A = sqt::kOrderAdd;
constexpr std::int8_t C = sqt::kOrderCancel;
constexpr std::int8_t F = sqt::kOrderFill;
constexpr std::int8_t R = sqt::kOrderClear;
constexpr std::int8_t T = sqt::kOrderOther;
constexpr long long kNaT = std::numeric_limits<long long>::min();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Queue {
    std::vector<double> ahead;
    sqt::QueueAheadCounts counts;
};

Queue run_queue(const std::vector<long long>& orders, std::size_t n_orders,
                const std::vector<std::int8_t>& actions,
                const std::vector<long long>& sides, const std::vector<double>& price,
                const std::vector<double>& size, const std::vector<std::uint8_t>& snap) {
    Queue q;
    q.ahead.assign(actions.size(), -1.0);
    const bool ok = sqt::order_queue_ahead(orders.data(), n_orders, actions.data(),
                                           sides.data(), price.data(), size.data(),
                                           snap.data(), actions.size(), q.ahead.data(),
                                           q.counts);
    expect(ok, "order_queue_ahead succeeds");
    q.ahead.resize(q.counts.n_ahead);
    return q;
}

}  // namespace

static void test_queue_ahead_is_the_size_at_the_level() {
    std::printf("test_queue_ahead_is_the_size_at_the_level\n");
    // Three adds at bid 100, one at bid 101, a partial fill of the first,
    // a cancel of the second, then a fourth add at 100.
    const std::vector<long long> orders = {0, 1, 2, 3, 0, 1, 4};
    const std::vector<std::int8_t> actions = {A, A, A, A, F, C, A};
    const std::vector<long long> sides = {0, 0, 0, 0, 0, 0, 0};
    const std::vector<double> price = {100, 100, 101, 100, 100, 100, 100};
    const std::vector<double> size = {300, 200, 50, 100, 120, 200, 10};
    const std::vector<std::uint8_t> snap(7, 0);
    const Queue q = run_queue(orders, 5, actions, sides, price, size, snap);
    const std::vector<double> want = {0, 300, 0, 500, 280};
    expect(q.ahead == want, "ahead is 0, 300, 0, 500, then 600 - 120 - 200");
    expect(q.counts.n_unseeded_adds == 5, "no snapshot or clear: all unseeded");
    expect(q.counts.n_unseen_decrements == 0, "every decrement was seen");
}

static void test_a_clear_forgets_orders_and_levels() {
    std::printf("test_a_clear_forgets_orders_and_levels\n");
    const std::vector<long long> orders = {0, 0, 1, 0};
    const std::vector<std::int8_t> actions = {A, R, A, C};
    const std::vector<long long> sides = {0, 0, 0, 0};
    const std::vector<double> price = {100, kNaN, 100, 100};
    const std::vector<double> size = {500, kNaN, 100, 500};
    const std::vector<std::uint8_t> snap(4, 0);
    const Queue q = run_queue(orders, 2, actions, sides, price, size, snap);
    const std::vector<double> want = {0, 0};
    expect(q.ahead == want, "the add after the clear joins an empty level");
    expect(q.counts.n_unseeded_adds == 1, "only the add before the clear is unseeded");
    expect(q.counts.n_unseen_decrements == 1, "the cleared order is unseen when cancelled");
    expect(q.counts.unseen_size == 500.0, "with its size");
}

static void test_signed_zero_prices_are_one_level() {
    std::printf("test_signed_zero_prices_are_one_level\n");
    const std::vector<long long> orders = {0, 1, 2};
    const std::vector<std::int8_t> actions = {A, A, A};
    const std::vector<long long> sides = {1, 1, 0};
    const std::vector<double> price = {0.0, -0.0, 0.0};
    const std::vector<double> size = {7, 3, 2};
    const std::vector<std::uint8_t> snap(3, 0);
    const Queue q = run_queue(orders, 3, actions, sides, price, size, snap);
    const std::vector<double> want = {0, 7, 0};
    expect(q.ahead == want, "-0.0 joins +0.0's level; another side is another level");
}

static void test_snapshot_seeds_and_non_finite_rows_are_skipped() {
    std::printf("test_snapshot_seeds_and_non_finite_rows_are_skipped\n");
    const std::vector<long long> orders = {0, 1, 2, 3, 0};
    const std::vector<std::int8_t> actions = {A, A, A, T, F};
    const std::vector<long long> sides = {0, 0, 0, 0, 0};
    const std::vector<double> price = {100, 100, 100, 100, 100};
    const std::vector<double> size = {400, kNaN, 100, 9, 1000};
    const std::vector<std::uint8_t> snap = {1, 0, 0, 0, 0};
    const Queue q = run_queue(orders, 4, actions, sides, price, size, snap);
    const std::vector<double> want = {400};
    expect(q.ahead == want, "the snapshot order is ahead; the NaN-size add is skipped");
    expect(q.counts.n_snapshot_orders == 1, "one snapshot order");
    expect(q.counts.n_unseeded_adds == 0, "the snapshot seeded the book");
}

static void test_lifetimes() {
    std::printf("test_lifetimes\n");
    // order 0: added at 10, cancelled at 25 -> 15
    // order 1: snapshot, filled at 30 -> explained by the snapshot
    // order 2: cancelled without an add -> censored
    // order 3: added at 40 (NaT row before it ignored), filled at 41 -> 1
    // order 4: added, never terminated -> still resting
    // order 5: snapshot, never terminated -> resting at the open
    const std::vector<long long> orders = {0, 1, 0, 1, 2, 3, 3, 3, 4, 5};
    const std::vector<std::int8_t> actions = {A, A, C, F, C, A, A, F, A, A};
    const std::vector<std::uint8_t> snap = {0, 1, 0, 0, 0, 0, 0, 0, 0, 1};
    const std::vector<long long> stamps = {10, 0, 25, 30, 31, kNaT, 40, 41, 50, 0};
    std::vector<long long> filled(10), cancelled(10);
    sqt::LifetimeCounts c;
    expect(sqt::order_lifetimes(orders.data(), 6, actions.data(), snap.data(),
                                stamps.data(), 10, filled.data(), cancelled.data(), c),
           "order_lifetimes succeeds");
    expect(c.n_cancelled == 1 && cancelled[0] == 15, "one cancel after 15");
    expect(c.n_filled == 1 && filled[0] == 1, "one fill after 1");
    expect(c.terminated_from_snapshot == 1, "the snapshot explains one");
    expect(c.terminated_without_an_add == 1, "one is left-censored");
    expect(c.still_resting == 1, "one is still resting");
    expect(c.known_at_open == 1, "one snapshot order never terminated");
    expect(!c.overflowed, "no overflow");
}

static void test_a_duration_outside_int64_is_flagged() {
    std::printf("test_a_duration_outside_int64_is_flagged\n");
    const std::vector<long long> orders = {0, 0};
    const std::vector<std::int8_t> actions = {A, C};
    const std::vector<std::uint8_t> snap = {0, 0};
    const std::vector<long long> stamps = {-(1LL << 62) * 2 + 1,
                                           std::numeric_limits<long long>::max()};
    std::vector<long long> filled(2), cancelled(2);
    sqt::LifetimeCounts c;
    sqt::order_lifetimes(orders.data(), 1, actions.data(), snap.data(), stamps.data(), 2,
                         filled.data(), cancelled.data(), c);
    expect(c.overflowed, "end - start past INT64_MAX is flagged, not wrapped");
}

int main() {
    std::printf("=== sqt order_events tests ===\n");
    test_queue_ahead_is_the_size_at_the_level();
    test_a_clear_forgets_orders_and_levels();
    test_signed_zero_prices_are_one_level();
    test_snapshot_seeds_and_non_finite_rows_are_skipped();
    test_lifetimes();
    test_a_duration_outside_int64_is_flagged();
    std::printf("\n%d assertion(s), %d failed\n", g_tests_run, g_tests_failed);
    return g_tests_failed == 0 ? 0 : 1;
}
