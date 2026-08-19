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
 * Trade log state machine mirrors _build_trade_log in engine.py exactly:
 *   open a trade when pos_diff != 0 and executed != 0, recording entry_price
 *   = prices[i-1] (Close one bar before the trade-open event, since
 *   executed[i] = signals[i-1] earns its first return over that close) and
 *   entry_size = exec_i (the raw signal value, not just its sign, so a
 *   leveraged/SCORE signal's trade return scales the same way strat_ret
 *   does); close it at the next trade event, deducting 2*cost_per_unit for
 *   a completed round trip; a position still open at the final bar is
 *   flushed at prices[n-1] with a single cost_per_unit deduction (entry
 *   only — no real exit event occurred).
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
    double slippage_pct    = 0.0005,
    // Bars per year for every annualized metric (volatility, Sharpe,
    // Sortino, Calmar). Was a hard-coded 252 inside the kernel, which is
    // correct only for daily equity bars -- the data and modeling layers
    // now support 1h/5m/1m and 24/7 markets, so an hourly backtest was
    // reporting a "Sharpe" annualized as though its bars were days.
    // Python owns calendar semantics and passes the resolved number here;
    // the kernel stays calendar-agnostic.
    double periods_per_year = 252.0
);

/**
 * Batch backtest: run `num_tests` signal arrays against the same price series.
 *
 * Avoids the Python/C++ boundary crossing overhead that accumulates when
 * calling run_strategy once per parameter combination from Python.
 *
 * @param prices         Close prices, length n.
 * @param signals_flat   Flattened 2-D array, shape (num_tests × n) row-major.
 *                         signals_flat[t*n + i] = signal for test t at bar i.
 * @param n              Number of bars.
 * @param num_tests      Number of signal arrays.
 * @param initial_capital / commission_pct / slippage_pct — same for all tests.
 * @return               Vector of BacktestResult, one per test in input order.
 *                       equity_curve is empty in every result to save memory.
 */
/**
 * Same algorithm and output as run_strategy(), but computes only the 11
 * scalar metrics with zero equity_curve/strat_ret/trade_rets array
 * allocation -- a two-pass design exploiting the fact that strat_ret[i] has
 * no true loop-carried dependency (exec_i = signals[i-1] and the prev_exec
 * needed for pos_diff equal signals[i-2], or 0.0 for i==1, both directly
 * index-derivable): pass 1 fuses the trade-log state machine with running
 * equity/peak/drawdown/mean tracking (no array ever written); pass 2
 * recomputes strat_ret[i] on demand, now that mean is known, to get
 * variance and downside deviation. The returned BacktestResult's
 * equity_curve is always left empty (default-constructed, zero
 * allocation) -- callers needing the curve must call run_strategy() instead.
 *
 * @param prices, signals, n, initial_capital, commission_pct, slippage_pct
 *   Same meaning as run_strategy().
 */
BacktestResult run_strategy_summary(
    const double* prices,
    const double* signals,
    std::size_t   n,
    double initial_capital = 10'000.0,
    double commission_pct  = 0.001,
    double slippage_pct    = 0.0005,
    // Bars per year for every annualized metric (volatility, Sharpe,
    // Sortino, Calmar). Was a hard-coded 252 inside the kernel, which is
    // correct only for daily equity bars -- the data and modeling layers
    // now support 1h/5m/1m and 24/7 markets, so an hourly backtest was
    // reporting a "Sharpe" annualized as though its bars were days.
    // Python owns calendar semantics and passes the resolved number here;
    // the kernel stays calendar-agnostic.
    double periods_per_year = 252.0
);

std::vector<BacktestResult> batch_run_strategy(
    const double* prices,
    const double* signals_flat,
    std::size_t   n,
    std::size_t   num_tests,
    double initial_capital = 10'000.0,
    double commission_pct  = 0.001,
    double slippage_pct    = 0.0005,
    double periods_per_year = 252.0
);

}  // namespace sqt
