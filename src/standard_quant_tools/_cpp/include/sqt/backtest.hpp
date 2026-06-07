#pragma once

#include <cstddef>
#include <vector>

namespace sqt {

struct BacktestResult {
    double final_equity;
    double total_return;
    double annualized_vol;
    double sharpe_ratio;
    double sortino_ratio;   // +inf when no negative returns
    double max_drawdown;    // ≤ 0  (convention: min of (equity - peak)/peak)
    double calmar_ratio;    // +inf when max_drawdown == 0
    int    num_trades;      // completed (closed) trades only
    double win_rate;
    double profit_factor;   // +inf when no losing trades
    double avg_trade_return_pct;
    std::vector<double> equity_curve;
};

/**
 * Vectorized backtest kernel — identical algorithm to the Python run_strategy.
 *
 * Execution model (one-bar lag):
 *   executed[0]   = 0
 *   executed[i]   = signals[i-1]          for i ≥ 1
 *   returns[0]    = 0
 *   returns[i]    = (prices[i]-prices[i-1]) / prices[i-1]   for i ≥ 1
 *   pos_diff[i]   = executed[i] - executed[i-1]
 *   strat_ret[i]  = executed[i] * returns[i] - |pos_diff[i]| * (commission + slippage)
 *   equity[i]     = equity[i-1] * (1 + strat_ret[i])
 *
 * Trade log state machine mirrors _build_trade_log in engine.py:
 *   open a trade when pos_diff != 0 and executed != 0;
 *   close it at the next trade event.  Unclosed trades are excluded from stats.
 *
 * @param prices         Close prices, length n.
 * @param signals        Position signals: 1=long, 0=flat, -1=short, length n.
 * @param n              Number of bars.
 * @param initial_capital Starting capital (default 10 000).
 * @param commission_pct  Commission fraction per position unit changed (default 0.001).
 * @param slippage_pct    Slippage fraction per position unit changed (default 0.0005).
 */
BacktestResult run_strategy(
    const double* prices,
    const double* signals,
    std::size_t   n,
    double initial_capital = 10'000.0,
    double commission_pct  = 0.001,
    double slippage_pct    = 0.0005
);

}  // namespace sqt
