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
    // Enter long, then exit → 1 winning trade.
    // prices=[100,110,120], signals=[1,0,0]; executed[i]=signals[i-1] -> [0,1,0]
    // Event at i=1 (pdiff=1-0=1): open long, entry_price=ref_price=prices[i-1]=prices[0]=100
    //   (same one-bar-lagged reference price run_strategy's own return calc uses --
    //   confirmed by test_long_buy_and_hold_no_costs above, which is the same
    //   prices[i-1]/prices[i] pairing).
    // Event at i=2 (pdiff=0-1=-1): close long, exit ref_price=prices[i-1]=prices[1]=110
    //   -> return=(110-100)/100 = 10.0%. (prices[2]=120 is never touched --
    //   the position already closed at the i=2 event using the lagged reference
    //   price, not this bar's own value.)
    std::vector<double> prices  = {100.0, 110.0, 120.0};
    std::vector<double> signals = {1.0,   0.0,   0.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.0, 0.0);
    CHECK(r.num_trades == 1);
    CHECK_NEAR(r.win_rate, 1.0, 1e-10);
    CHECK_INF(r.profit_factor);  // no losing trades → inf
    CHECK_NEAR(r.avg_trade_return_pct, (110.0 - 100.0) / 100.0 * 100.0, 1e-6);
}

static void test_one_completed_losing_trade() {
    // Enter long, exit at a loss -- same event shape as the winning-trade test
    // above, so the LOW price must land at index 1 (the exit event's lagged
    // reference price, prices[i-1] at i=2), not index 2, to actually produce a
    // loss: entry_price=prices[0]=100, exit ref_price=prices[1]=90.
    std::vector<double> prices  = {100.0, 90.0, 95.0};
    std::vector<double> signals = {1.0,   0.0,  0.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.0, 0.0);
    CHECK(r.num_trades == 1);
    CHECK_NEAR(r.win_rate, 0.0, 1e-10);
    CHECK_NEAR(r.profit_factor, 0.0, 1e-10);  // no winning trades → 0
    const double expected_pct = (90.0 - 100.0) / 100.0 * 100.0;
    CHECK_NEAR(r.avg_trade_return_pct, expected_pct, 1e-6);
}

static void test_unclosed_position_flushed_as_one_trade_at_final_close() {
    // signal=1 throughout, position never explicitly closed by a signal
    // transition -- this does NOT mean zero trades: run_strategy (matching
    // _build_trade_log's own documented behavior in engine.py) synthesizes a
    // mark-to-market "exit" for any position still open at the final bar,
    // priced at that bar's actual Close (not the lagged ref_price), and
    // deducts only the entry-side cost (no real exit event occurred).
    // Event at i=1: open long, entry_price=ref_price=prices[0]=100.
    // End of loop: position still open -> flush at close_prices[-1]=prices[2]=110.
    // return = (110-100)/100 = 10.0%.
    std::vector<double> prices  = {100.0, 105.0, 110.0};
    std::vector<double> signals = {1.0,   1.0,   1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.0, 0.0);
    CHECK(r.num_trades == 1);
    CHECK_NEAR(r.avg_trade_return_pct, (110.0 - 100.0) / 100.0 * 100.0, 1e-6);
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
    // prices=[100,110,90], signals=[1,-1,-1]; executed[i]=signals[i-1] -> [0,1,-1]
    // Event at i=1 (pdiff=1-0=1): open long, entry_price=ref_price=prices[0]=100.
    // Event at i=2 (pdiff=-1-1=-2, non-zero -> both a close AND a reopen):
    //   close the long using ref_price=prices[i-1]=prices[1]=110
    //     -> return=(110-100)/100 = +10.0% (a WIN, not the loss the reversal's
    //        own bar-2 price of 90 might suggest -- the close uses the lagged
    //        reference price, same convention as every other event here).
    //   immediately reopen short, entry_price=ref_price=prices[1]=110.
    // End of loop: the short is still open -> flushed at close_prices[-1]=prices[2]=90
    //   -> return=(90-110)/110*(-1) = +18.1818...% (also a WIN: price fell while short).
    // Both legs of the reversal turn out to be winners here, not one loser --
    // this is a real, hand-verified consequence of two independent, correct
    // conventions (lagged ref_price for the mid-run close/reopen event, and
    // final-bar-Close for the synthesized flush), not a coincidence to special-case.
    std::vector<double> prices  = {100.0, 110.0, 90.0};
    std::vector<double> signals = {1.0,  -1.0,  -1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 3, 10000.0, 0.0, 0.0);
    CHECK(r.num_trades == 2);  // the long trade completes, then the short is flushed
    CHECK(r.win_rate == 1.0);
    const double long_leg_pct  = (110.0 - 100.0) / 100.0 * 100.0;
    const double short_leg_pct = (90.0 - 110.0) / 110.0 * -1.0 * 100.0;
    CHECK_NEAR(r.avg_trade_return_pct, (long_leg_pct + short_leg_pct) / 2.0, 1e-6);
}

static void test_trade_log_cost_scales_with_leveraged_position_size() {
    // Regression test: the trade log's cost deduction used to be a flat
    // 2*cost_per_unit / 1*cost_per_unit regardless of entry_size, so a 5x
    // leveraged trade paid the exact same cost as a 1x trade even though
    // strat_ret (the equity curve) already scales cost by abs(pdiff) --
    // silently under-costing every leveraged (non-+/-1) SCORE-style
    // position. prices=[100,110,121], signals=[size,0,0]: a real close
    // event at i=2 (pdiff=-size), not a final-bar flush -- both entry and
    // exit legs are costed at abs(entry_size)*cost_per_unit each.
    const double cost_per_unit = 0.01;  // commission_pct + slippage_pct
    std::vector<double> prices = {100.0, 110.0, 121.0};

    std::vector<double> signals_1x = {1.0, 0.0, 0.0};
    auto r1 = sqt::run_strategy(prices.data(), signals_1x.data(), 3, 10000.0,
                                 cost_per_unit, 0.0);
    const double expected_1x = ((110.0 - 100.0) / 100.0 * 1.0 - 2.0 * 1.0 * cost_per_unit) * 100.0;
    CHECK_NEAR(r1.avg_trade_return_pct, expected_1x, 1e-9);
    CHECK_NEAR(expected_1x, 8.0, 1e-9);  // matches the review's own repro numbers

    std::vector<double> signals_5x = {5.0, 0.0, 0.0};
    auto r5 = sqt::run_strategy(prices.data(), signals_5x.data(), 3, 10000.0,
                                 cost_per_unit, 0.0);
    const double expected_5x = ((110.0 - 100.0) / 100.0 * 5.0 - 2.0 * 5.0 * cost_per_unit) * 100.0;
    CHECK_NEAR(r5.avg_trade_return_pct, expected_5x, 1e-9);
    // Before this fix, r5 came out as 48.0 (flat cost, same as r1's 8.0
    // flat cost -> a 6x ratio, not a clean 5x one either) instead of the
    // correctly cost-scaled 40.0 -- both pnl and cost are linear in
    // position size for a single trade, so the fixed formula now produces
    // an exactly 5x relationship between r5 and r1, unlike the old bug.
    CHECK_NEAR(r5.avg_trade_return_pct, 40.0, 1e-9);
    CHECK_NEAR(r5.avg_trade_return_pct, 5.0 * r1.avg_trade_return_pct, 1e-9);
}

static void test_trade_log_resize_cost_is_documented_approximation() {
    // A same-sign RESIZE (1.0 -> 2.5, a single pos_diff event) is a known,
    // documented approximation: this event is treated as closing a
    // 1.0-sized trade AND opening a fresh 2.5-sized one, each independently
    // costed at 2x its own size, rather than the single abs(pdiff)=1.5 the
    // equity curve actually charges for that one event. This test pins
    // down the resulting trade-log values as a known quantity rather than
    // silently drifting if the approximation changes.
    // prices=[100,105,110,108,108], signals=[1,2.5,2.5,0,0];
    // exec_i=signals[i-1] for i=1..4 -> exec=[1,2.5,2.5,0].
    // Event i=1 (pdiff=1): open trade1, entry_price=prices[0]=100, size=1.0.
    // Event i=2 (pdiff=1.5, resize): close trade1 @ ref_price=prices[1]=105,
    //   reopen trade2, entry_price=105, size=2.5.
    // Event i=4 (pdiff=-2.5): close trade2 @ ref_price=prices[3]=108.
    const double cost_per_unit = 0.01;
    std::vector<double> prices  = {100.0, 105.0, 110.0, 108.0, 108.0};
    std::vector<double> signals = {1.0,   2.5,   2.5,   0.0,   0.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 5, 10000.0,
                                cost_per_unit, 0.0);
    CHECK(r.num_trades == 2);  // resize splits into a closed 1.0x + closed 2.5x trade

    const double trade1_pnl = (105.0 - 100.0) / 100.0 * 1.0;
    const double trade1_pct = (trade1_pnl - 2.0 * 1.0 * cost_per_unit) * 100.0;

    const double trade2_pnl = (108.0 - 105.0) / 105.0 * 2.5;
    const double trade2_pct = (trade2_pnl - 2.0 * 2.5 * cost_per_unit) * 100.0;

    CHECK_NEAR(r.avg_trade_return_pct, (trade1_pct + trade2_pct) / 2.0, 1e-9);

    // The trade log's own total realized cost for this sequence is
    // 2*(1.0+2.5)*cost_per_unit = 7*cost_per_unit, vs. the equity curve's
    // own realized cost across the same 3 pos_diff events, sum(abs(pdiff))
    // * cost_per_unit = (1.0+1.5+2.5)*cost_per_unit = 5*cost_per_unit --
    // the two do not match for a resize; this is the documented
    // approximation, not a bug to chase further here.
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
    test_unclosed_position_flushed_as_one_trade_at_final_close();
    test_sortino_inf_when_no_negative_returns();
    test_equity_curve_length_matches_n();
    test_equity_curve_starts_at_initial_capital();
    test_calmar_inf_when_no_drawdown();
    test_reversal_trade_long_to_short();
    test_trade_log_cost_scales_with_leveraged_position_size();
    test_trade_log_resize_cost_is_documented_approximation();

    std::printf("\n%d / %d tests passed.\n",
                g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed > 0 ? 1 : 0;
}
