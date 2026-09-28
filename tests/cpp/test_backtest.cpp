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
#include <cstdint>
#include <cstdio>
#include <limits>
#include <stdexcept>
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

#define CHECK_EQ_SZ(a, b)       CHECK((a) == (b))
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

static void test_profit_factor_zero_over_zero_is_nan() {
    // A flat price series with zero costs produces trades whose return is
    // exactly 0.0 -- neither wins nor losses, so gross_win AND gross_loss
    // are both 0, and the ratio is 0/0.
    //
    // This test used to pin +inf here, read as "no losing trade". But +inf
    // is the x/0 answer with x > 0 (test_one_completed_winning_trade
    // above), and 0/0 is undefined: NaN, the library's convention for every
    // other 0/0 ratio (a Sharpe without dispersion, a Sortino or Calmar of
    // a book that never moved). +inf ranked a set of do-nothing trades
    // FIRST under sort_by="profit_factor". It was 0.0 before that, which
    // reads as "every trade lost". engine.py's _compute_trade_stats applies
    // the same rule, so both backends answer NaN.
    std::vector<double> prices  = {100.0, 100.0, 100.0, 100.0};
    std::vector<double> signals = {1.0,   1.0,   1.0,   1.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 4, 10000.0, 0.0, 0.0);
    CHECK(r.num_trades == 1);
    CHECK_NEAR(r.avg_trade_return_pct, 0.0, 1e-12);
    CHECK(std::isnan(r.profit_factor));

    // run_strategy_summary shares the expression -- both must agree, or
    // batch_run_strategy silently disagrees with run_strategy.
    auto s = sqt::run_strategy_summary(prices.data(), signals.data(), 4, 10000.0, 0.0, 0.0);
    CHECK(std::isnan(s.profit_factor));
    CHECK(s.num_trades == r.num_trades);
}

static void test_profit_factor_with_a_winner_and_a_zero_trade_is_inf() {
    // The null case for the rule above: a zero trade counts on the loss
    // side, but with a gross profit and no gross loss the ratio is x/0 with
    // x > 0 -- +inf, unchanged.
    // executed = [0,1,0,1,0]: lot one 100 -> 110 (+10%), lot two 110 -> 110.
    std::vector<double> prices  = {100.0, 110.0, 110.0, 110.0, 110.0};
    std::vector<double> signals = {1.0,   0.0,   1.0,   0.0,   0.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 5, 10000.0, 0.0, 0.0);
    CHECK(r.num_trades == 2);
    CHECK_INF(r.profit_factor);
    auto s = sqt::run_strategy_summary(prices.data(), signals.data(), 5, 10000.0, 0.0, 0.0);
    CHECK_INF(s.profit_factor);
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
    // The trade log's cost scales with the size transacted, as the equity
    // curve's does: each event pays abs(pdiff) * cost_per_unit, charged
    // against the equity of the bar it is paid on. (A flat cost regardless
    // of size once under-costed every leveraged SCORE-style position.)
    // prices=[100,110,121], signals=[size,0,0] -> executed=[0,size,0]:
    //   bar 1: entry, earns size * 10% and pays size * c
    //   bar 2: exit, pays size * c
    // and the lot is exactly the curve: (1 + 0.1s - sc)(1 - sc) - 1. These
    // pins read 8.0 and 40.0 while costs were charged as a simple fraction
    // of notional (0.1s - 2sc), which is the first-order approximation of
    // the product and overstated the lot against the curve.
    const double c = 0.01;  // commission_pct + slippage_pct
    std::vector<double> prices = {100.0, 110.0, 121.0};

    std::vector<double> signals_1x = {1.0, 0.0, 0.0};
    auto r1 = sqt::run_strategy(prices.data(), signals_1x.data(), 3, 10000.0, c, 0.0);
    const double expected_1x = ((1.0 + 0.1 - c) * (1.0 - c) - 1.0) * 100.0;
    CHECK_NEAR(r1.avg_trade_return_pct, expected_1x, 1e-9);
    CHECK_NEAR(expected_1x, 7.91, 1e-9);
    CHECK_NEAR(r1.avg_trade_return_pct, r1.total_return * 100.0, 1e-9);

    std::vector<double> signals_5x = {5.0, 0.0, 0.0};
    auto r5 = sqt::run_strategy(prices.data(), signals_5x.data(), 3, 10000.0, c, 0.0);
    const double expected_5x = ((1.0 + 0.5 - 5.0 * c) * (1.0 - 5.0 * c) - 1.0) * 100.0;
    CHECK_NEAR(r5.avg_trade_return_pct, expected_5x, 1e-9);
    CHECK_NEAR(expected_5x, 37.75, 1e-9);
    CHECK_NEAR(r5.avg_trade_return_pct, r5.total_return * 100.0, 1e-9);
    // A size-blind cost would have charged 5x the move but 1x the cost.
    const double size_blind = ((1.0 + 0.5 - c) * (1.0 - c) - 1.0) * 100.0;
    CHECK(std::abs(r5.avg_trade_return_pct - size_blind) > 1.0);
}

static void test_trade_log_resize_is_one_lot() {
    // A same-sign RESIZE (1.0 -> 2.5, a single pos_diff event) is an ADD to
    // the SAME lot, not a close-then-reopen: the whole open->resize->close
    // sequence is exactly ONE trade (an earlier reading split it in two and
    // charged 2*(1.0+2.5)*cost_per_unit), and it pays the curve's own
    // open(1.0) + resize(1.5) + close(2.5) units of cost.
    // prices=[100,105,110,108,108], signals=[1,2.5,2.5,0,0];
    // exec_i=signals[i-1] for i=1..4 -> exec=[1,2.5,2.5,0].
    const double cost_per_unit = 0.01;
    std::vector<double> prices  = {100.0, 105.0, 110.0, 108.0, 108.0};
    std::vector<double> signals = {1.0,   2.5,   2.5,   0.0,   0.0};
    auto r = sqt::run_strategy(prices.data(), signals.data(), 5, 10000.0,
                                cost_per_unit, 0.0);
    CHECK(r.num_trades == 1);  // open -> resize -> close is now ONE continuous lot

    // One lot spanning every traded bar, so its return is the curve's:
    //   bar 1: open 1.0,   earns 1.0 * (105/100 - 1), pays 1.0 * c
    //   bar 2: resize 2.5, earns 2.5 * (110/105 - 1), pays 1.5 * c
    //   bar 3: held,       earns 2.5 * (108/110 - 1)
    //   bar 4: close,      pays 2.5 * c
    // The costs are the curve's own open(1.0) + resize(1.5) + close(2.5).
    // This pinned 735/103 = 7.1359% while the lot was priced off its
    // weighted cost basis (103.0) with simple costs; the compounded, costed
    // share of the curve is what the lot actually returned.
    const double c = cost_per_unit;
    const double expected_pct =
        ((1.0 + 0.05 - c) * (1.0 + 2.5 * (110.0 / 105.0 - 1.0) - 1.5 * c) *
             (1.0 + 2.5 * (108.0 / 110.0 - 1.0)) * (1.0 - 2.5 * c) -
         1.0) * 100.0;

    CHECK_NEAR(r.avg_trade_return_pct, expected_pct, 1e-9);
    CHECK_NEAR(r.avg_trade_return_pct, r.total_return * 100.0, 1e-9);
}

static void test_trade_log_is_the_equity_curves_share_across_a_flip() {
    // The invariant the lot accounting establishes: every lot is its share
    // of the equity curve, so the lots multiply back to the curve exactly,
    // costs included. This test used to pin a weaker property -- that the
    // SUM of the lots' simple costs equalled the curve's sum(abs(pdiff)) *
    // cost -- which held while the lots' P&L still disagreed with the curve
    // by the compounding of those costs.
    //
    // executed = [0, 1, 1, 2, 2, -1, -1]: open 1.0 at bar 1, resize to 2.0
    // at bar 3, FLIP to -1.0 at bar 5 (closing 2.0, opening 1.0), still
    // short at the end. Under a close fill the closing lot earns nothing on
    // the flip bar (its exit is at the previous close) and pays its exit
    // cost there, so it ends at equity[4] * (1 - 2c); the short begins at
    // that split and ends at the final equity.
    const double c = 0.02;
    std::vector<double> prices  = {100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 106.0};
    std::vector<double> signals = {1.0,   1.0,   2.0,   2.0,  -1.0,  -1.0,   0.0};
    const std::size_t n = prices.size();
    auto r = sqt::run_strategy(prices.data(), signals.data(), n, 10000.0, c, 0.0);
    CHECK(r.num_trades == 2);

    const auto& eq = r.equity_curve;
    const double split = eq[4] * (1.0 - 2.0 * c);
    const double long_lot  = split / eq[0] - 1.0;
    const double short_lot = eq[n - 1] / split - 1.0;
    CHECK_NEAR(r.avg_trade_return_pct, (long_lot + short_lot) / 2.0 * 100.0, 1e-9);
    // ...and the two multiply back to the curve.
    CHECK_NEAR((1.0 + long_lot) * (1.0 + short_lot), 1.0 + r.total_return, 1e-12);

    // The summary kernel reads the same lots off its running equity.
    auto s = sqt::run_strategy_summary(prices.data(), signals.data(), n, 10000.0, c, 0.0);
    CHECK(s.num_trades == r.num_trades);
    CHECK(s.avg_trade_return_pct == r.avg_trade_return_pct);
}

static void test_trade_log_flip_under_a_two_leg_fill() {
    // Under a two-leg fill the closing lot still holds the overnight leg to
    // the open. executed = [0, 1, -1]: bar 2 flips at the open.
    //   bar 1: entry at open 101, earns 1 * (102/101 - 1), pays c
    //   bar 2: the long earns the overnight leg 102 -> 104 and pays c;
    //          the short earns the intraday leg 104 -> 100 and pays c.
    const double c = 0.001;
    std::vector<double> close   = {100.0, 102.0, 100.0};
    std::vector<double> opens   = {100.0, 101.0, 104.0};
    std::vector<double> signals = {1.0,  -1.0,  -1.0};
    auto r = sqt::run_strategy(close.data(), signals.data(), 3, 10000.0, c, 0.0,
                               252.0, opens.data());
    CHECK(r.num_trades == 2);
    const double e1 = 10000.0 * (1.0 + (102.0 / 101.0 - 1.0) - c);
    const double overnight = 104.0 / 102.0 - 1.0;
    const double split = e1 * (1.0 + overnight - c);
    const double long_lot = split / 10000.0 - 1.0;
    const double short_lot = r.equity_curve[2] / split - 1.0;
    CHECK_NEAR(r.avg_trade_return_pct, (long_lot + short_lot) / 2.0 * 100.0, 1e-9);
    CHECK_NEAR((1.0 + long_lot) * (1.0 + short_lot), 1.0 + r.total_return, 1e-12);
}

// ── run_strategy_summary() vs run_strategy() ────────────────────────────────
//
// run_strategy_summary() is a from-scratch two-pass reimplementation of
// run_strategy()'s 11 scalar fields with zero equity_curve/strat_ret/
// trade_rets array allocation. Every field must match run_strategy()'s
// output exactly (bit-identical, not just close) -- the design guarantees
// this by construction (see backtest.cpp's comment above the function), and
// this is the regression test that actually proves it held.

static double pseudo_random(std::uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    std::uint64_t x = state;
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDULL;
    x ^= x >> 33;
    return (static_cast<double>(x >> 11) / 9007199254740992.0) * 2.0 - 1.0;  // [-1, 1)
}

/**
 * Bit-for-bit agreement, with NaN counted as equal to NaN.
 *
 * Plain == was used here on the reasoning that the +inf-valued fields are
 * fine (IEEE 754 makes +inf == +inf true) and that NaN never arises. It
 * does: a window that opens no position has zero volatility, and a Sharpe
 * ratio over zero volatility is undefined -- the behaviour
 * `TestBothBackendsAgreeOnAnUndefinedSharpe` exists to pin on the Python
 * side. Both kernels then returned NaN, agreed completely, and failed the
 * check, because NaN == NaN is false. Two assertions in the random
 * summary-vs-full comparison failed for that reason alone.
 */
static bool same_value(double a, double b) {
    return (std::isnan(a) && std::isnan(b)) || a == b;
}

static void check_all_fields_match(
    const sqt::BacktestResult& a, const sqt::BacktestResult& b)
{
    CHECK(same_value(a.final_equity, b.final_equity));
    CHECK(same_value(a.total_return, b.total_return));
    CHECK(same_value(a.annualized_vol, b.annualized_vol));
    CHECK(same_value(a.sharpe_ratio, b.sharpe_ratio));
    CHECK(same_value(a.sortino_ratio, b.sortino_ratio));
    CHECK(same_value(a.max_drawdown, b.max_drawdown));
    CHECK(a.num_trades == b.num_trades);
    CHECK(same_value(a.calmar_ratio, b.calmar_ratio));
    CHECK(same_value(a.win_rate, b.win_rate));
    CHECK(same_value(a.profit_factor, b.profit_factor));
    CHECK(same_value(a.avg_trade_return_pct, b.avg_trade_return_pct));
    CHECK(b.equity_curve.empty());  // summary kernel never populates this
}

static void test_run_strategy_summary_matches_run_strategy_random() {
    std::uint64_t state = 777;
    // Random (n, commission, slippage) combos, including leveraged/non-±1
    // signals and occasional zero-price bars.
    for (int trial = 0; trial < 40; ++trial) {
        const int n = 5 + static_cast<int>(std::abs(pseudo_random(state)) * 300);
        std::vector<double> prices(n), signals(n);
        for (int i = 0; i < n; ++i) {
            double p = 50.0 + pseudo_random(state) * 40.0;
            if (trial % 13 == 0 && i == n / 2) p = 0.0;  // occasional zero price
            prices[i]  = p;
            double s = pseudo_random(state) * 3.0;  // leveraged, non-±1 signal
            if (trial % 5 == 0) s = (s > 0.0) ? 1.0 : (s < 0.0 ? -1.0 : 0.0);
            signals[i] = s;
        }
        const double commission = std::abs(pseudo_random(state)) * 0.01;
        const double slippage   = std::abs(pseudo_random(state)) * 0.01;
        const double capital    = 1000.0 + std::abs(pseudo_random(state)) * 9000.0;

        auto full = sqt::run_strategy(prices.data(), signals.data(), n,
                                       capital, commission, slippage);
        auto summ = sqt::run_strategy_summary(prices.data(), signals.data(), n,
                                               capital, commission, slippage);
        check_all_fields_match(full, summ);
    }
}

static void test_run_strategy_summary_edge_cases() {
    // n==0
    {
        auto full = sqt::run_strategy(nullptr, nullptr, 0, 10000.0, 0.001, 0.0005);
        auto summ = sqt::run_strategy_summary(nullptr, nullptr, 0, 10000.0, 0.001, 0.0005);
        check_all_fields_match(full, summ);
    }
    // n==1
    {
        double price = 100.0, signal = 1.0;
        auto full = sqt::run_strategy(&price, &signal, 1, 5000.0, 0.001, 0.0005);
        auto summ = sqt::run_strategy_summary(&price, &signal, 1, 5000.0, 0.001, 0.0005);
        check_all_fields_match(full, summ);
    }
    // all-flat signal
    {
        std::vector<double> prices  = {100.0, 105.0, 110.0, 95.0};
        std::vector<double> signals = {0.0,   0.0,   0.0,   0.0};
        auto full = sqt::run_strategy(prices.data(), signals.data(), 4, 10000.0, 0.001, 0.0005);
        auto summ = sqt::run_strategy_summary(prices.data(), signals.data(), 4, 10000.0, 0.001, 0.0005);
        check_all_fields_match(full, summ);
    }
    // all-short
    {
        std::vector<double> prices  = {100.0, 95.0, 90.0, 92.0, 88.0};
        std::vector<double> signals = {-1.0, -1.0, -1.0, -1.0, -1.0};
        auto full = sqt::run_strategy(prices.data(), signals.data(), 5, 10000.0, 0.001, 0.0005);
        auto summ = sqt::run_strategy_summary(prices.data(), signals.data(), 5, 10000.0, 0.001, 0.0005);
        check_all_fields_match(full, summ);
    }
    // leveraged resize (matches test_trade_log_resize_cost_is_documented_approximation's shape)
    {
        std::vector<double> prices  = {100.0, 105.0, 110.0, 108.0, 108.0};
        std::vector<double> signals = {1.0,   2.5,   2.5,   0.0,   0.0};
        auto full = sqt::run_strategy(prices.data(), signals.data(), 5, 10000.0, 0.01, 0.0);
        auto summ = sqt::run_strategy_summary(prices.data(), signals.data(), 5, 10000.0, 0.01, 0.0);
        check_all_fields_match(full, summ);
    }
}

static void test_no_trade_ratios_are_undefined_not_inf() {
    // A parameter set that never trades has no downside, no drawdown and no
    // trade. Sortino and Calmar are 0/0 and the profit factor has nothing to
    // count, so all three are NaN on both kernels. They were +inf, +inf and
    // 0.0, and the +inf ranked a do-nothing grid row above every real
    // strategy under sort_by="sortino_ratio" or "calmar_ratio".
    std::vector<double> prices  = {100.0, 104.0, 99.0, 107.0, 103.0};
    std::vector<double> signals = {0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> refs    = {100.0, 103.0, 100.0, 105.0, 104.0};
    const std::size_t n = prices.size();

    for (const double* ref : {static_cast<const double*>(nullptr),
                              static_cast<const double*>(refs.data())}) {
        auto r = sqt::run_strategy(prices.data(), signals.data(), n,
                                   10000.0, 0.001, 0.0005, 252.0, ref);
        auto s = sqt::run_strategy_summary(prices.data(), signals.data(), n,
                                           10000.0, 0.001, 0.0005, 252.0, ref);
        CHECK(r.num_trades == 0);
        CHECK(std::isnan(r.sortino_ratio));
        CHECK(std::isnan(r.calmar_ratio));
        CHECK(std::isnan(r.profit_factor));
        check_all_fields_match(r, s);
    }
}

static void test_empty_denominator_follows_the_numerator_sign() {
    // Null case for the rule above: it is the numerator, not the absence of
    // trades, that decides. A position that only ever rose keeps +inf on
    // both kernels (x/0), and a flat book under a NEGATIVE policy rate has a
    // positive excess every bar and no downside, so its Sortino is +inf too.
    std::vector<double> up      = {100.0, 101.0, 103.0, 104.0, 106.0};
    std::vector<double> long_on = {1.0, 1.0, 1.0, 1.0, 1.0};
    auto s_up = sqt::run_strategy_summary(up.data(), long_on.data(), up.size(),
                                          10000.0, 0.0, 0.0);
    CHECK_INF(s_up.sortino_ratio);
    CHECK_INF(s_up.calmar_ratio);
    CHECK_INF(s_up.profit_factor);

    std::vector<double> flat_signal = {0.0, 0.0, 0.0, 0.0, 0.0};
    auto r = sqt::run_strategy(up.data(), flat_signal.data(), up.size(),
                               10000.0, 0.0, 0.0, 252.0, nullptr, -0.02);
    CHECK_INF(r.sortino_ratio);
    CHECK(std::isnan(r.calmar_ratio));  // no growth and no drawdown: 0/0
}

static void test_run_strategy_summary_multi_trade_count() {
    // Hand-constructed multi-trade series: long -> flat -> short -> flat ->
    // long, still open at the end. Targeted check on the new scalar
    // trade-stat accumulation (replacing the old trade_rets vector) beyond
    // the aggregate field-by-field comparison above.
    // exec_i = signals[i-1] for i=1..7 (n=8, so signals[6] is actually
    // consumed as exec_7 -- an n=7 array would leave the last signal
    // element unused, since exec only ever reads up to signals[n-2]).
    std::vector<double> prices  = {100.0, 102.0, 101.0, 98.0, 99.0, 97.0, 100.0, 101.0};
    std::vector<double> signals = {1.0,   1.0,   0.0,   -1.0, -1.0, 0.0,  1.0,   1.0};
    auto summ = sqt::run_strategy_summary(prices.data(), signals.data(), 8,
                                           10000.0, 0.001, 0.0005);
    // trade1: open @ exec_1=1 (i=1), close @ exec_3=0 (i=3) -> closed
    // trade2: open @ exec_4=-1 (i=4), close @ exec_6=0 (i=6) -> closed
    // trade3: open @ exec_7=1 (i=7), still open at end -> flushed
    CHECK(summ.num_trades == 3);
    CHECK_NOT_NAN(summ.avg_trade_return_pct);
}

// ── batch_run_strategy() vs a serial reference loop ─────────────────────────
//
// batch_run_strategy() parallelizes (when _OPENMP) an embarrassingly
// independent loop over test indices, each a pure call to
// run_strategy_summary() with no shared mutable state -- so its output must
// be exactly reproducible regardless of how many threads actually ran it.
// This test doesn't control OMP_NUM_THREADS itself (that's an environment
// variable read once at OpenMP thread-pool creation, not something a single
// process can usefully vary mid-run) --
// tests/cpp_bindings/test_thread_count_determinism.py covers the 1/2/4/8
// thread comparison, one interpreter per count, instead. This test's job is
// simpler: prove the batch path's output matches calling
// run_strategy_summary() directly, test by test, in this process as-built.

static void test_batch_run_strategy_matches_serial_reference() {
    const int n = 300, num_tests = 37;  // enough tests to plausibly span >1 thread
    std::uint64_t state = 4242;
    std::vector<double> prices(n);
    for (int i = 0; i < n; ++i) prices[i] = 50.0 + pseudo_random(state) * 30.0;

    std::vector<double> signals_flat(static_cast<std::size_t>(num_tests) * n);
    for (int t = 0; t < num_tests; ++t) {
        for (int i = 0; i < n; ++i) {
            double s = pseudo_random(state) * 2.0;
            signals_flat[static_cast<std::size_t>(t) * n + i] = s;
        }
    }

    auto batch = sqt::batch_run_strategy(
        prices.data(), signals_flat.data(), n, num_tests, 10000.0, 0.001, 0.0005);
    CHECK(batch.size() == static_cast<std::size_t>(num_tests));

    for (int t = 0; t < num_tests; ++t) {
        auto ref = sqt::run_strategy_summary(
            prices.data(), signals_flat.data() + static_cast<std::size_t>(t) * n,
            n, 10000.0, 0.001, 0.0005);
        check_all_fields_match(ref, batch[static_cast<std::size_t>(t)]);
    }
}

static void test_batch_run_strategy_single_test() {
    // num_tests=1 exercises the `if(num_tests > 1)` OpenMP guard's false
    // branch -- must still produce a correct (serial) result.
    std::vector<double> prices  = {100.0, 105.0, 110.0, 95.0};
    std::vector<double> signals = {1.0,   1.0,   -1.0,  0.0};
    auto batch = sqt::batch_run_strategy(prices.data(), signals.data(), 4, 1,
                                          10000.0, 0.001, 0.0005);
    CHECK(batch.size() == 1);
    auto ref = sqt::run_strategy_summary(prices.data(), signals.data(), 4,
                                          10000.0, 0.001, 0.0005);
    check_all_fields_match(ref, batch[0]);
}

// ── ref_prices: the two-leg fill model ───────────────────────────────────────
//
// run_strategy's `ref_prices` parameter -- the whole fill_price="next_open" /
// "hl2_exploratory" execution model -- had NO C++ test. grep for "ref_prices"
// across tests/cpp/ returned nothing before this block. It is also the
// parameter whose mishandling produced the fill-price defect recorded in this
// file's own history (a lot booked 100 -> 120 against a fill-to-fill
// +19.05%), so leaving it uncovered here was the gap that let that happen.

static void test_ref_prices_null_equals_close_to_close() {
    // Passing ref_prices == prices is NOT the same as passing nullptr in
    // general, but passing nullptr must reproduce the historical
    // close-to-close path exactly -- the default-argument contract.
    std::vector<double> prices  = {100.0, 102.0, 101.0, 105.0, 103.0, 108.0};
    std::vector<double> signals = {1.0, 1.0, 0.0, 1.0, 1.0, 0.0};
    const std::size_t n = prices.size();

    auto explicit_null = sqt::run_strategy(prices.data(), signals.data(), n,
                                            10000.0, 0.001, 0.0005, 252.0, nullptr);
    auto defaulted = sqt::run_strategy(prices.data(), signals.data(), n,
                                        10000.0, 0.001, 0.0005, 252.0);
    CHECK(explicit_null.final_equity == defaulted.final_equity);
    CHECK(explicit_null.num_trades == defaulted.num_trades);
}

static void test_ref_prices_two_leg_decomposition_hand_computed() {
    // The decomposition, restated from backtest.hpp rather than copied from
    // the implementation:
    //     overnight[i] = (ref[i] - close[i-1]) / close[i-1]  at exec[i-1]
    //     intraday[i]  = (close[i] - ref[i])   / ref[i]      at exec[i]
    //     gross[i]     = (1 + exec[i-1]*overnight[i]) * (1 + exec[i]*intraday[i]) - 1
    // with exec[i] = signals[i-1] and exec[i-1] = signals[i-2] (0.0 at i==1).
    //
    // The legs compound. This test used to pin their SUM, which was the
    // defect: g2 below came out 0.0977 where holding 110 -> 121 earns
    // exactly 0.10.
    //
    // Note what this makes true and a naive reading would not expect:
    // setting ref[i] = close[i] does NOT reduce this to the close-to-close
    // path. It zeroes the intraday leg and moves the entire bar onto the
    // overnight leg, which is priced at YESTERDAY's position -- and at i==1
    // yesterday's position is 0 by construction, so bar 1 earns nothing. An
    // earlier version of this test asserted the equality and failed, which
    // is the decomposition behaving exactly as documented.
    std::vector<double> prices  = {100.0, 110.0, 121.0};
    std::vector<double> refs    = {100.0, 105.0, 115.0};
    std::vector<double> signals = {1.0, 1.0, 0.0};
    const std::size_t n = prices.size();

    const double g1 = (1.0 + 0.0 * ((105.0 - 100.0) / 100.0))
                    * (1.0 + 1.0 * ((110.0 - 105.0) / 105.0)) - 1.0;
    const double g2 = (1.0 + 1.0 * ((115.0 - 110.0) / 110.0))
                    * (1.0 + 1.0 * ((121.0 - 115.0) / 115.0)) - 1.0;
    const double expected_equity = 10000.0 * (1.0 + g1) * (1.0 + g2);

    auto r = sqt::run_strategy(prices.data(), signals.data(), n,
                               10000.0, 0.0, 0.0, 252.0, refs.data());
    CHECK_NEAR(r.final_equity, expected_equity, 1e-9);
    // Held through bar 2, the account earns exactly the close-to-close move.
    CHECK_NEAR(g2, 121.0 / 110.0 - 1.0, 1e-15);

    // And it is genuinely different from the close-fill path on this input,
    // so the assertion above is not vacuously satisfiable by ignoring refs.
    auto close_fill = sqt::run_strategy(prices.data(), signals.data(), n,
                                         10000.0, 0.0, 0.0, 252.0, nullptr);
    CHECK(std::abs(close_fill.final_equity - r.final_equity) > 1.0);
}

static void test_ref_prices_trade_log_uses_the_fill_price() {
    // The defect this parameter's doc comment describes: the equity curve
    // used ref_prices[i] while trade accounting still used prices[i-1], so a
    // lot entered at Open=105 and exited at Open=125 was booked 100 -> 120.
    //
    // Enter at bar 1, exit at bar 3. executed[i] = signals[i-1], so
    // signals = {1,0,...} opens at i=1 and closes at i=2.
    std::vector<double> prices = {100.0, 110.0, 120.0, 130.0, 140.0};
    std::vector<double> refs   = {100.0, 105.0, 115.0, 125.0, 135.0};
    std::vector<double> signals = {1.0, 0.0, 0.0, 0.0, 0.0};
    const std::size_t n = prices.size();

    auto r = sqt::run_strategy(prices.data(), signals.data(), n,
                               10000.0, 0.0, 0.0, 252.0, refs.data());
    CHECK(r.num_trades == 1);
    // Entry fills at refs[1] = 105, exit at refs[2] = 115: +9.5238%.
    // Booking it against prices[0]=100 -> prices[1]=110 would give +10.00%.
    CHECK_NEAR(r.avg_trade_return_pct, (115.0 - 105.0) / 105.0 * 100.0, 1e-9);
}

static void test_ref_prices_zero_cost_curve_is_the_hand_built_one() {
    // A hand-built account at zero cost: carry yesterday's position from the
    // prior close to today's fill, re-size at the fill, carry today's
    // position to the close. Entry, a held bar, an exit, a short opened on
    // the fill and flushed at the last close.
    std::vector<double> prices  = {100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 110.0};
    std::vector<double> refs    = {100.0, 101.0, 103.0, 102.0, 106.0, 104.0, 109.0};
    std::vector<double> signals = {0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0};
    const std::size_t n = prices.size();

    std::vector<double> expected(n, 10000.0);
    for (std::size_t i = 1; i < n; ++i) {
        const double held_overnight = (i >= 2) ? signals[i - 2] : 0.0;
        const double held_intraday  = signals[i - 1];
        const double at_fill  = expected[i - 1]
                              * (1.0 + held_overnight * (refs[i] / prices[i - 1] - 1.0));
        expected[i] = at_fill * (1.0 + held_intraday * (prices[i] / refs[i] - 1.0));
    }

    auto r = sqt::run_strategy(prices.data(), signals.data(), n,
                               10000.0, 0.0, 0.0, 252.0, refs.data());
    for (std::size_t i = 0; i < n; ++i) CHECK_NEAR(r.equity_curve[i], expected[i], 1e-9);
    // The unit long lot fills at refs[2] = 103 and exits at refs[4] = 106.
    // Over its span the account grows by exactly the fill-to-fill ratio,
    // which the trade log reports -- the two reconcile at zero cost. With
    // the legs added, the held bar alone put them 0.03 points apart.
    CHECK_NEAR(r.equity_curve[4] / r.equity_curve[1], 106.0 / 103.0, 1e-12);
    auto s = sqt::run_strategy_summary(prices.data(), signals.data(), n,
                                       10000.0, 0.0, 0.0, 252.0, refs.data());
    CHECK_NEAR(s.final_equity, expected[n - 1], 1e-9);
}

static void test_ref_prices_summary_matches_full_random() {
    // run_strategy_summary is what batch_run_strategy / the parameter grid /
    // walk-forward call. It must agree with run_strategy under the fill
    // model too, not only under close-to-close -- the previously tested half.
    std::uint64_t state = 4242;
    for (int trial = 0; trial < 40; ++trial) {
        const std::size_t n = 20 + static_cast<std::size_t>(
            (pseudo_random(state) + 1.0) * 60.0);
        std::vector<double> prices(n), refs(n), signals(n);
        double p = 100.0;
        for (std::size_t i = 0; i < n; ++i) {
            p *= 1.0 + pseudo_random(state) * 0.03;
            prices[i] = p;
            refs[i]   = p * (1.0 + pseudo_random(state) * 0.01);
            const double u = pseudo_random(state);
            signals[i] = (u < -0.33) ? -1.0 : (u > 0.33 ? 1.0 : 0.0);
        }
        const double commission = (pseudo_random(state) + 1.0) * 0.001;
        const double slippage   = (pseudo_random(state) + 1.0) * 0.0005;

        auto full = sqt::run_strategy(prices.data(), signals.data(), n,
                                       10000.0, commission, slippage, 252.0,
                                       refs.data());
        auto summ = sqt::run_strategy_summary(prices.data(), signals.data(), n,
                                               10000.0, commission, slippage,
                                               252.0, refs.data());
        check_all_fields_match(full, summ);
    }
}


// ── batch_backtest_crossover ─────────────────────────────────────────────────
//
// Also entirely uncovered here before this block: the fused grid kernel that
// generates each combination's signal and backtests it without materialising
// a (num_combos x n) matrix.

// Independent reference: build the crossover signal the obvious way and call
// the already-tested run_strategy_summary on it.
static sqt::BacktestResult crossover_reference(
    const std::vector<double>& prices,
    const std::vector<double>& indicators,
    std::size_t n, int fast_row, int slow_row,
    double commission, double slippage, const double* refs)
{
    std::vector<double> signal(n, 0.0);
    const double* fast = indicators.data() + static_cast<std::size_t>(fast_row) * n;
    const double* slow = indicators.data() + static_cast<std::size_t>(slow_row) * n;
    for (std::size_t i = 0; i < n; ++i)
        signal[i] = (fast[i] > slow[i]) ? 1.0 : 0.0;
    return sqt::run_strategy_summary(prices.data(), signal.data(), n,
                                      10000.0, commission, slippage, 252.0, refs);
}

static void test_crossover_matches_per_combination_reference() {
    std::uint64_t state = 31337;
    const std::size_t n = 200;
    const std::size_t n_unique = 5;

    std::vector<double> prices(n);
    double p = 100.0;
    for (std::size_t i = 0; i < n; ++i) {
        p *= 1.0 + pseudo_random(state) * 0.02;
        prices[i] = p;
    }
    // Simple moving averages of several periods, plus a deliberate NaN
    // warm-up prefix on each row -- the pandas comparison yields 0 there and
    // the kernel documents that it matches.
    const int periods[n_unique] = {2, 5, 10, 20, 50};
    std::vector<double> indicators(n_unique * n,
                                    std::numeric_limits<double>::quiet_NaN());
    for (std::size_t r = 0; r < n_unique; ++r) {
        const std::size_t w = static_cast<std::size_t>(periods[r]);
        for (std::size_t i = w - 1; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = i + 1 - w; j <= i; ++j) s += prices[j];
            indicators[r * n + i] = s / static_cast<double>(w);
        }
    }

    std::vector<int> pairs;
    for (int a = 0; a < static_cast<int>(n_unique); ++a)
        for (int b = 0; b < static_cast<int>(n_unique); ++b)
            if (a != b) { pairs.push_back(a); pairs.push_back(b); }
    const std::size_t num_combos = pairs.size() / 2;

    auto results = sqt::batch_backtest_crossover(
        prices.data(), indicators.data(), n, n_unique,
        pairs.data(), num_combos, 10000.0, 0.001, 0.0005, 252.0, nullptr);

    CHECK_EQ_SZ(results.size(), num_combos);
    for (std::size_t c = 0; c < num_combos; ++c) {
        auto expected = crossover_reference(prices, indicators, n,
                                             pairs[c * 2], pairs[c * 2 + 1],
                                             0.001, 0.0005, nullptr);
        check_all_fields_match(expected, results[c]);
    }
}

static void test_crossover_honours_ref_prices() {
    std::uint64_t state = 5150;
    const std::size_t n = 120;
    const std::size_t n_unique = 2;
    std::vector<double> prices(n), refs(n);
    double p = 100.0;
    for (std::size_t i = 0; i < n; ++i) {
        p *= 1.0 + pseudo_random(state) * 0.02;
        prices[i] = p;
        refs[i]   = p * (1.0 + pseudo_random(state) * 0.01);
    }
    std::vector<double> indicators(n_unique * n);
    for (std::size_t i = 0; i < n; ++i) {
        indicators[i]     = prices[i];
        indicators[n + i] = 100.0;
    }
    const std::vector<int> pairs = {0, 1};

    auto results = sqt::batch_backtest_crossover(
        prices.data(), indicators.data(), n, n_unique,
        pairs.data(), 1, 10000.0, 0.001, 0.0005, 252.0, refs.data());
    auto expected = crossover_reference(prices, indicators, n, 0, 1,
                                         0.001, 0.0005, refs.data());
    CHECK_EQ_SZ(results.size(), static_cast<std::size_t>(1));
    check_all_fields_match(expected, results[0]);
}

static void test_crossover_rejects_out_of_range_pair_index() {
    // Previously this was skipped inside the parallel loop with `continue`,
    // leaving that row value-initialised -- which is NOT a neutral result:
    // every real result reports sortino_ratio and profit_factor as +inf when
    // there is no downside and no losing trade, while a default-constructed
    // BacktestResult reports 0.0 for both. A zeroed row is indistinguishable
    // from a genuinely flat strategy. It is a caller error, so it is named
    // as one -- and named BEFORE the region starts, where throwing is safe.
    const std::size_t n = 40;
    std::vector<double> prices(n, 100.0);
    std::vector<double> indicators(2 * n, 1.0);
    for (const std::vector<int>& bad : {std::vector<int>{0, 2},
                                        std::vector<int>{2, 0},
                                        std::vector<int>{-1, 0}}) {
        bool threw = false;
        try {
            sqt::batch_backtest_crossover(prices.data(), indicators.data(), n, 2,
                                           bad.data(), 1, 10000.0, 0.001, 0.0005,
                                           252.0, nullptr);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        CHECK_TRUE(threw);
    }
}

static void test_crossover_empty_inputs() {
    std::vector<double> prices(10, 100.0);
    std::vector<double> indicators(20, 1.0);
    const std::vector<int> pairs = {0, 1};
    CHECK(sqt::batch_backtest_crossover(prices.data(), indicators.data(), 10, 2,
                                         pairs.data(), 0).empty());
    CHECK(sqt::batch_backtest_crossover(nullptr, nullptr, 0, 0,
                                         nullptr, 0).empty());
}

// ── Main ──────────────────────────────────────────────────────────────────────

// ── Risk-free rate ────────────────────────────────────────────────────────────

static void test_risk_free_rate_defaults_to_zero() {
    // The rate is the last defaulted parameter, so every call site that
    // predates it must keep reporting exactly what it always reported.
    const std::vector<double> prices  = {100.0, 101.0, 99.0, 103.0, 102.0, 105.0};
    const std::vector<double> signals = {1.0, 1.0, 0.0, 1.0, 1.0, 0.0};
    auto implicit_rate = sqt::run_strategy(prices.data(), signals.data(),
                                           prices.size());
    auto explicit_zero = sqt::run_strategy(prices.data(), signals.data(),
                                           prices.size(), 10'000.0, 0.001,
                                           0.0005, 252.0, nullptr, 0.0);
    CHECK_NEAR(implicit_rate.sharpe_ratio,  explicit_zero.sharpe_ratio,  0.0);
    CHECK_NEAR(implicit_rate.sortino_ratio, explicit_zero.sortino_ratio, 0.0);
}

static void test_risk_free_rate_lowers_the_ratios() {
    const std::vector<double> prices  = {100.0, 101.0, 102.5, 101.5, 104.0, 106.0};
    const std::vector<double> signals = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    auto at_zero = sqt::run_strategy(prices.data(), signals.data(), prices.size(),
                                     10'000.0, 0.0, 0.0, 252.0, nullptr, 0.0);
    auto at_high = sqt::run_strategy(prices.data(), signals.data(), prices.size(),
                                     10'000.0, 0.0, 0.0, 252.0, nullptr, 0.25);
    CHECK_TRUE(at_high.sharpe_ratio < at_zero.sharpe_ratio);
    // ...and it is a scoring convention, not a cash flow: the equity curve
    // and every return-based field must be untouched.
    CHECK_NEAR(at_high.total_return, at_zero.total_return, 0.0);
    CHECK_NEAR(at_high.final_equity, at_zero.final_equity, 0.0);
    CHECK_NEAR(at_high.max_drawdown, at_zero.max_drawdown, 0.0);
}

static void test_summary_matches_run_strategy_under_a_rate() {
    // The divergence this guards: run_strategy_summary loops from i=1 and
    // seeds the downside sum with bar 0's implicit strat_ret of 0.0, whose
    // EXCESS is -rf/ppy. That seed contributes nothing at rf = 0, so a
    // missing one is invisible until a rate is set -- and every batch entry
    // point reads its numbers from this function.
    //
    // The negative rates are the other half. Bar 0's downside term is
    // min(excess, 0)^2, which is rf^2 for a positive rate and 0 for a
    // negative one; seeding rf^2 regardless put the batch Sortino 3e-6 off
    // the single run at rf = -5%, enough to reorder a grid.
    std::uint64_t state = 4242;
    const double rates[] = {-0.5, -0.05, -0.001, 0.0, 0.01, 0.045, 0.10, 0.25};
    for (double rf : rates) {
        for (int trial = 0; trial < 15; ++trial) {
            const int n = 5 + static_cast<int>(std::abs(pseudo_random(state)) * 200);
            std::vector<double> prices(n), signals(n);
            for (int i = 0; i < n; ++i) {
                prices[i]  = 50.0 + pseudo_random(state) * 40.0;
                signals[i] = pseudo_random(state) * 2.0;
            }
            const double commission = std::abs(pseudo_random(state)) * 0.01;
            const double slippage   = std::abs(pseudo_random(state)) * 0.01;

            auto full = sqt::run_strategy(prices.data(), signals.data(), n,
                                          10'000.0, commission, slippage,
                                          252.0, nullptr, rf);
            auto summ = sqt::run_strategy_summary(prices.data(), signals.data(), n,
                                                  10'000.0, commission, slippage,
                                                  252.0, nullptr, rf);
            check_all_fields_match(full, summ);
        }
    }
}


// ── run_portfolio_simulation ─────────────────────────────────────────────────
//
// Every output element must be defined on every return path. The outputs are
// pre-filled with a sentinel standing in for whatever the caller's buffer held
// -- from Python that was numpy's recycled memory, so a failing run returned
// the previous run's equity curve in its tail.

namespace {

constexpr double kSentinel = 12345.0;

struct PortfolioRun {
    std::vector<double> equity, cash, gross, net, rebal;
    double peak = 0.0;
    sqt::PortfolioSimError err;
    std::size_t n_executed = 0;
};

PortfolioRun run_portfolio(const std::vector<double>& close,
                           const std::vector<double>& exec,
                           const std::vector<double>& weights,
                           const std::vector<long long>& rebal_bars,
                           std::size_t n_bars, std::size_t n_tickers,
                           const sqt::PortfolioCosts& costs) {
    PortfolioRun r;
    const std::size_t n_rebal = rebal_bars.size();
    r.equity.assign(n_bars, kSentinel);
    r.cash.assign(n_bars, kSentinel);
    r.gross.assign(n_bars, kSentinel);
    r.net.assign(n_bars, kSentinel);
    r.rebal.assign(n_rebal * 3, kSentinel);
    const std::vector<double> gaps(n_bars, 1.0);
    r.n_executed = sqt::run_portfolio_simulation(
        close.data(), exec.data(), weights.data(), rebal_bars.data(), gaps.data(),
        n_bars, n_tickers, n_rebal, costs,
        r.equity.data(), r.cash.data(), r.gross.data(), r.net.data(),
        r.rebal.data(), &r.peak, &r.err);
    return r;
}

bool no_sentinel_left(const PortfolioRun& r) {
    for (const auto* v : {&r.equity, &r.cash, &r.gross, &r.net, &r.rebal})
        for (double x : *v)
            if (x == kSentinel) return false;
    return true;
}

// Bars [first, last) of every per-bar output are NaN.
bool bars_nan(const PortfolioRun& r, std::size_t first, std::size_t last) {
    for (std::size_t b = first; b < last; ++b)
        if (!std::isnan(r.equity[b]) || !std::isnan(r.cash[b]) ||
            !std::isnan(r.gross[b]) || !std::isnan(r.net[b]))
            return false;
    return true;
}

bool bars_finite(const PortfolioRun& r, std::size_t first, std::size_t last) {
    for (std::size_t b = first; b < last; ++b)
        if (!std::isfinite(r.equity[b]) || !std::isfinite(r.cash[b]) ||
            !std::isfinite(r.gross[b]) || !std::isfinite(r.net[b]))
            return false;
    return true;
}

bool rebal_rows_nan(const PortfolioRun& r, std::size_t first) {
    for (std::size_t i = first * 3; i < r.rebal.size(); ++i)
        if (!std::isnan(r.rebal[i])) return false;
    return true;
}

sqt::PortfolioCosts free_costs() {
    sqt::PortfolioCosts c;
    c.commission_pct = 0.0;
    c.sell_commission_pct = 0.0;
    c.slippage_pct = 0.0;
    return c;
}

}  // namespace

static void test_portfolio_healthy_run_writes_every_element() {
    // The null case: a run that reaches the end has no NaN anywhere.
    const std::size_t n_bars = 6, n_tickers = 2;
    const std::vector<double> close = {100, 50, 110, 55, 121, 60,
                                       100, 70, 105, 65, 130, 50};
    const std::vector<double> weights = {1.0, 0.0, 0.5, 0.5};
    const std::vector<long long> rebal = {0, 3};
    auto r = run_portfolio(close, close, weights, rebal, n_bars, n_tickers,
                           free_costs());
    CHECK(r.err.status == sqt::kPortfolioOk);
    CHECK(r.n_executed == 2);
    CHECK(no_sentinel_left(r));
    CHECK(bars_finite(r, 0, n_bars));
    CHECK(std::isfinite(r.rebal[3]) && std::isfinite(r.rebal[5]));
    // All-in ticker 0 at 100 with no costs: bar 1 is marked at 110.
    CHECK_NEAR(r.equity[1], 10'000.0 * 110.0 / 100.0, 1e-9);
}

static void test_portfolio_failed_rebalance_leaves_a_nan_tail() {
    // A NaN execution price for a ticker that is being bought stops the run
    // at the rebalance bar, which is never marked.
    const std::size_t n_bars = 6, n_tickers = 2;
    std::vector<double> close = {100, 50, 101, 51, 102, 52,
                                 103, 53, 104, 54, 105, 55};
    std::vector<double> exec = close;
    exec[4 * n_tickers + 0] = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> weights = {0.5, 0.5, 0.6, 0.4};
    const std::vector<long long> rebal = {0, 4};
    auto r = run_portfolio(close, exec, weights, rebal, n_bars, n_tickers,
                           free_costs());
    CHECK(r.err.status == sqt::kPortfolioBadExecPrice);
    CHECK(r.err.bar == 4);
    CHECK(r.n_executed == 1);
    CHECK(no_sentinel_left(r));
    CHECK(bars_finite(r, 0, 4));
    CHECK(bars_nan(r, 4, n_bars));
    CHECK(rebal_rows_nan(r, 1));
}

static void test_portfolio_zero_capital_leaves_every_bar_nan() {
    // Insolvent at the very first rebalance: nothing was ever marked.
    const std::size_t n_bars = 5, n_tickers = 1;
    const std::vector<double> close = {100, 101, 102, 103, 104};
    const std::vector<double> weights = {1.0};
    const std::vector<long long> rebal = {0};
    auto costs = free_costs();
    costs.initial_capital = 0.0;
    auto r = run_portfolio(close, close, weights, rebal, n_bars, n_tickers, costs);
    CHECK(r.err.status == sqt::kPortfolioInsolventAtRebalance);
    CHECK(r.err.bar == 0);
    CHECK(r.n_executed == 0);
    CHECK(no_sentinel_left(r));
    CHECK(bars_nan(r, 0, n_bars));
    CHECK(rebal_rows_nan(r, 0));
}

static void test_portfolio_insolvent_at_bar_marks_that_bar_only() {
    // Fully invested at 100 with no costs: 100 shares, cash exactly 0. A
    // close of 0 at bar 3 marks equity 0 there and stops; bar 3 itself is
    // written (it WAS marked), everything after it is NaN.
    const std::size_t n_bars = 6, n_tickers = 1;
    const std::vector<double> close = {100, 90, 80, 0.0, 70, 60};
    const std::vector<double> weights = {1.0};
    const std::vector<long long> rebal = {0};
    auto r = run_portfolio(close, close, weights, rebal, n_bars, n_tickers,
                           free_costs());
    CHECK(r.err.status == sqt::kPortfolioInsolventAtBar);
    CHECK(r.err.bar == 3);
    CHECK(no_sentinel_left(r));
    CHECK(bars_finite(r, 0, 4));
    CHECK_NEAR(r.equity[3], 0.0, 0.0);
    CHECK(bars_nan(r, 4, n_bars));
}

static void test_portfolio_next_open_trigger_on_the_last_bar() {
    // status 0, but the second row never executes -- there is no following
    // Open to fill at -- so its rebalance row is NaN, not left unwritten.
    const std::size_t n_bars = 5, n_tickers = 1;
    const std::vector<double> close = {100, 101, 102, 103, 104};
    const std::vector<double> open  = {99.5, 100.5, 101.5, 102.5, 103.5};
    const std::vector<double> weights = {1.0, 0.5};
    const std::vector<long long> rebal = {0, 4};
    auto costs = free_costs();
    costs.fill = sqt::kFillNextOpen;
    auto r = run_portfolio(close, open, weights, rebal, n_bars, n_tickers, costs);
    CHECK(r.err.status == sqt::kPortfolioOk);
    CHECK(r.n_executed == 1);
    CHECK(no_sentinel_left(r));
    CHECK(bars_finite(r, 0, n_bars));
    CHECK(std::isfinite(r.rebal[0]));
    CHECK(rebal_rows_nan(r, 1));
}


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
    test_profit_factor_zero_over_zero_is_nan();
    test_profit_factor_with_a_winner_and_a_zero_trade_is_inf();
    test_unclosed_position_flushed_as_one_trade_at_final_close();
    test_sortino_inf_when_no_negative_returns();
    test_equity_curve_length_matches_n();
    test_equity_curve_starts_at_initial_capital();
    test_calmar_inf_when_no_drawdown();
    test_reversal_trade_long_to_short();
    test_trade_log_cost_scales_with_leveraged_position_size();
    test_trade_log_resize_is_one_lot();
    test_trade_log_is_the_equity_curves_share_across_a_flip();
    test_trade_log_flip_under_a_two_leg_fill();
    test_risk_free_rate_defaults_to_zero();
    test_risk_free_rate_lowers_the_ratios();
    test_summary_matches_run_strategy_under_a_rate();
    test_run_strategy_summary_matches_run_strategy_random();
    test_run_strategy_summary_edge_cases();
    test_no_trade_ratios_are_undefined_not_inf();
    test_empty_denominator_follows_the_numerator_sign();
    test_run_strategy_summary_multi_trade_count();
    test_batch_run_strategy_matches_serial_reference();
    test_batch_run_strategy_single_test();

    // ref_prices (fill_price="next_open" / "hl2_exploratory")
    test_ref_prices_null_equals_close_to_close();
    test_ref_prices_two_leg_decomposition_hand_computed();
    test_ref_prices_trade_log_uses_the_fill_price();
    test_ref_prices_zero_cost_curve_is_the_hand_built_one();
    test_ref_prices_summary_matches_full_random();

    // batch_backtest_crossover
    test_crossover_matches_per_combination_reference();
    test_crossover_honours_ref_prices();
    test_crossover_rejects_out_of_range_pair_index();
    test_crossover_empty_inputs();

    // run_portfolio_simulation
    test_portfolio_healthy_run_writes_every_element();
    test_portfolio_failed_rebalance_leaves_a_nan_tail();
    test_portfolio_zero_capital_leaves_every_bar_nan();
    test_portfolio_insolvent_at_bar_marks_that_bar_only();
    test_portfolio_next_open_trigger_on_the_last_bar();

    std::printf("\n%d / %d tests passed.\n",
                g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed > 0 ? 1 : 0;
}
