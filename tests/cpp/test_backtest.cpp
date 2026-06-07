/**
 * C++ unit tests for sqt::run_strategy (backtest kernel).
 *
 * Build (all platforms):
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V -R cpp_backtest
 */

#include "sqt/backtest.hpp"

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

#define CHECK_NEAR(a, b, tol) \
    CHECK(std::abs((a) - (b)) <= (tol))

#define CHECK_TRUE(cond)        CHECK(cond)
#define CHECK_FALSE(cond)       CHECK(!(cond))
#define CHECK_INF(val)          CHECK(std::isinf(val))
#define CHECK_FINITE(val)       CHECK(std::isfinite(val))
#define CHECK_NOT_NAN(val)      CHECK(!std::isnan(val))


// ── Tests ─────────────────────────────────────────────────────────────────────

static void test_flat_signal_zero_return() {
    // signal = 0 throughout → no position, no return, equity stays flat
    std::vector<double> prices  = {100.0, 105.0, 110.0, 95.0};
    std::vector<double> signals = {0.0,   0.0,   0.0,   0.0};
    const int n = static_cast<int>(prices.size());
    auto r = sqt::run_strategy(prices.data(), signals.data(), n, 10000.0, 0.001, 0.0005);
    CHECK_NEAR(r.total_return, 0.0, 1e-10);
    CHECK_NEAR(r.final_equity, 10000.0, 1e-8);
    CHECK(r.num_trades == 0);
    CHECK_NEAR(r.win_rate, 0.0, 1e-10);
}

static void test_empty_input() {
    // n=0 → all defaults, no crash
    auto r = sqt::run_strategy(nullptr, nullptr, 0, 10000.0, 0.001, 0.0005);
    CHECK_NEAR(r.final_equity, 10000.0, 1e-10);
    CHECK_NEAR(r.total_return, 0.0, 1e-10);
    CHECK(r.num_trades == 0);
    CHECK(r.equity_curve.empty());
}

static void test_single_bar() {
    // n=1 → no returns possible, equity stays at initial capital
    double price  = 100.0;
    double signal = 1.0;
    auto r = sqt::run_strategy(&price, &signal, 1, 5000.0, 0.001, 0.0005);
    CHECK_NEAR(r.final_equity, 5000.0, 1e-8);
    CHECK_NEAR(r.total_return, 0.0, 1e-10);
    CHECK(r.num_trades == 0);
}

static void test_long_buy_and_hold_no_costs() {
    // signal=1 from bar 1 → executed=[0,1,1], prices go 100→110→121
    // executed[0]=0, executed[1]=1, executed[2]=1
    // returns[1]=(110-100)/100=0.10, returns[2]=(121-110)/110≈0.1
    // strat_ret=[0, 0.10, 0.10] (no costs), equity=[10000, 11000, 12100]
    std::vector<double> prices  = {100.0, 110.0, 121.0};
    std::vector<double> signals = {1.0,   1.0,   1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.0, 0.0);
    CHECK_NEAR(r.total_return, 0.21, 1e-8);
    CHECK_NEAR(r.final_equity, 12100.0, 1e-4);
    CHECK(r.equity_curve.size() == 3);
    CHECK_NEAR(r.equity_curve[0], 10000.0, 1e-6);
    CHECK_NEAR(r.equity_curve[1], 11000.0, 1e-4);
    CHECK_NEAR(r.equity_curve[2], 12100.0, 1e-3);
}

static void test_transaction_costs_reduce_returns() {
    // Same as long buy-and-hold but with costs — result should be lower
    std::vector<double> prices  = {100.0, 110.0, 121.0};
    std::vector<double> signals = {1.0,   1.0,   1.0};
    auto r_free = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.0,   0.0);
    auto r_cost = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.001, 0.0005);
    CHECK(r_cost.total_return < r_free.total_return);
    CHECK(r_cost.final_equity < r_free.final_equity);
}

static void test_max_drawdown_nonpositive() {
    // Prices peak then fall → drawdown should be negative
    std::vector<double> prices  = {100.0, 120.0, 90.0, 110.0};
    std::vector<double> signals = {1.0,   1.0,   1.0,  1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 4, 10000.0, 0.0, 0.0);
    CHECK(r.max_drawdown <= 0.0);
    CHECK(r.max_drawdown > -1.0);  // sensible range
}

static void test_max_drawdown_zero_for_monotone_up() {
    // Prices only go up → no drawdown
    std::vector<double> prices  = {100.0, 105.0, 110.0, 115.0};
    std::vector<double> signals = {1.0,   1.0,   1.0,   1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 4, 10000.0, 0.0, 0.0);
    CHECK_NEAR(r.max_drawdown, 0.0, 1e-10);
}

static void test_sharpe_positive_for_consistently_positive_returns() {
    // All returns positive → Sharpe > 0
    std::vector<double> prices  = {100.0, 101.0, 102.0, 103.0, 104.0};
    std::vector<double> signals = {1.0,   1.0,   1.0,   1.0,   1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 5, 10000.0, 0.0, 0.0);
    CHECK(r.sharpe_ratio > 0.0);
}

static void test_short_position_profits_when_prices_fall() {
    // signal=-1 from bar 0, prices fall → should make money
    std::vector<double> prices  = {100.0, 90.0, 80.0};
    std::vector<double> signals = {-1.0, -1.0, -1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.0, 0.0);
    CHECK(r.total_return > 0.0);
    CHECK(r.final_equity > 10000.0);
}

static void test_one_completed_winning_trade() {
    // Enter long, then exit → 1 winning trade
    // prices=[100,110,120], signals=[1,0,0]
    // executed=[0,1,0]
    // Event at i=1: open long at prices[1]=110
    // Event at i=2: close long at prices[2]=120 → return=(120-110)/110 = 9.09%
    std::vector<double> prices  = {100.0, 110.0, 120.0};
    std::vector<double> signals = {1.0,   0.0,   0.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.0, 0.0);
    CHECK(r.num_trades == 1);
    CHECK_NEAR(r.win_rate, 1.0, 1e-10);
    CHECK_INF(r.profit_factor);  // no losing trades → inf
    CHECK_NEAR(r.avg_trade_return_pct, (120.0 - 110.0) / 110.0 * 100.0, 1e-6);
}

static void test_one_completed_losing_trade() {
    // Enter long, exit at a loss
    // prices=[100,110,90], signals=[1,0,0]
    // Event at i=1: open long at 110; Event at i=2: close at 90 → -18.18%
    std::vector<double> prices  = {100.0, 110.0, 90.0};
    std::vector<double> signals = {1.0,   0.0,   0.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.0, 0.0);
    CHECK(r.num_trades == 1);
    CHECK_NEAR(r.win_rate, 0.0, 1e-10);
    CHECK_NEAR(r.profit_factor, 0.0, 1e-10);  // no winning trades → 0
    const double expected_pct = (90.0 - 110.0) / 110.0 * 100.0;
    CHECK_NEAR(r.avg_trade_return_pct, expected_pct, 1e-6);
}

static void test_no_trades_unclosed_position() {
    // signal=1 throughout and never goes to 0 → position never closed → 0 trades
    std::vector<double> prices  = {100.0, 105.0, 110.0};
    std::vector<double> signals = {1.0,   1.0,   1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.0, 0.0);
    CHECK(r.num_trades == 0);  // trade never closed
}

static void test_sortino_inf_when_no_negative_returns() {
    // Monotone prices up, strategy long → no negative strategy returns → sortino=inf
    std::vector<double> prices  = {100.0, 101.0, 102.0, 103.0};
    std::vector<double> signals = {1.0,   1.0,   1.0,   1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 4, 10000.0, 0.0, 0.0);
    CHECK_INF(r.sortino_ratio);
}

static void test_equity_curve_length_matches_n() {
    std::vector<double> prices  = {100.0, 105.0, 95.0, 110.0, 108.0};
    std::vector<double> signals = {1.0,   1.0,  -1.0,  0.0,   1.0};
    const std::size_t n = prices.size();
    auto r = sqt::run_strategy(prices.data(), signals.data(), n, 10000.0, 0.001, 0.0005);
    CHECK(r.equity_curve.size() == n);
    CHECK_NEAR(r.equity_curve[0], 10000.0, 1e-10);
}

static void test_equity_curve_starts_at_initial_capital() {
    std::vector<double> prices  = {50.0, 55.0};
    std::vector<double> signals = {1.0,  1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 2, 25000.0, 0.0, 0.0);
    CHECK_NEAR(r.equity_curve[0], 25000.0, 1e-8);
}

static void test_calmar_inf_when_no_drawdown() {
    // Monotone up → MDD=0 → calmar=inf
    std::vector<double> prices  = {100.0, 102.0, 104.0, 106.0, 108.0};
    std::vector<double> signals = {1.0,   1.0,   1.0,   1.0,   1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 5, 10000.0, 0.0, 0.0);
    CHECK_INF(r.calmar_ratio);
}

static void test_reversal_trade_long_to_short() {
    // prices=[100,110,90], signals=[1,-1,-1]
    // executed=[0,1,-1]
    // Event at i=1 (pdiff=1): open long at prices[1]=110
    // Event at i=2 (pdiff=-2): close long at prices[2]=90 → return=-18.18%
    //                         open short at prices[2]=90
    std::vector<double> prices  = {100.0, 110.0, 90.0};
    std::vector<double> signals = {1.0,  -1.0,  -1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.0, 0.0);
    CHECK(r.num_trades == 1);  // the long trade completes; short stays open
    CHECK(r.avg_trade_return_pct < 0.0);  // losing long trade
}


// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test_flat_signal_zero_return();
    test_empty_input();
    test_single_bar();
    test_long_buy_and_hold_no_costs();
    test_transaction_costs_reduce_returns();
    test_max_drawdown_nonpositive();
    test_max_drawdown_zero_for_monotone_up();
    test_sharpe_positive_for_consistently_positive_returns();
    test_short_position_profits_when_prices_fall();
    test_one_completed_winning_trade();
    test_one_completed_losing_trade();
    test_no_trades_unclosed_position();
    test_sortino_inf_when_no_negative_returns();
    test_equity_curve_length_matches_n();
    test_equity_curve_starts_at_initial_capital();
    test_calmar_inf_when_no_drawdown();
    test_reversal_trade_long_to_short();

    std::printf("\n%d / %d tests passed.\n",
                g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed > 0 ? 1 : 0;
}
