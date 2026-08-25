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
 *   = the event's actual FILL price -- prices[i-1] when ref_prices is null
 *   (Close one bar before the trade-open event, since executed[i] =
 *   signals[i-1] earns its first return over that close), or ref_prices[i]
 *   when a fill series was supplied, matching what _build_trade_log is handed
 *   for fill_price="next_open"/"hl2_exploratory" -- and
 *   entry_size = exec_i (the raw signal value, not just its sign, so a
 *   leveraged/SCORE signal's trade return scales the same way strat_ret
 *   does); close it at the next trade event, deducting 2*cost_per_unit for
 *   a completed round trip; a position still open at the final bar is
 *   flushed at prices[n-1] with a single cost_per_unit deduction (entry
 *   only — no real exit event occurred).
 *
 * @param ref_prices     Optional per-bar REFERENCE (fill) price, length n.
 *                       nullptr  -> close-to-close returns, the historical
 *                                   behaviour and fill_price="close".
 *                       non-null -> the two-leg decomposition engine.py uses
 *                                   for fill_price="next_open" (Open) and
 *                                   "hl2_exploratory" ((High+Low)/2):
 *
 *                         overnight[i] = (ref[i] - close[i-1]) / close[i-1]
 *                                        priced at YESTERDAY's position
 *                         intraday[i]  = (close[i] - ref[i]) / ref[i]
 *                                        priced at TODAY's position
 *
 *                       Adding the two legs (rather than compounding them)
 *                       matches engine.py exactly; the product term is the
 *                       only difference from pure close-to-close and is the
 *                       standard overnight/intraday P&L attribution.
 *
 *                       Without this, fill_price != "close" had no native
 *                       path at all, so the MORE REALISTIC execution model
 *                       was also the slow one -- and a native grid could not
 *                       be used for it, which is what let walk-forward
 *                       optimize under one fill model and evaluate under
 *                       another.
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
    double periods_per_year = 252.0,
    // Last and defaulted so every existing positional call site -- the C++
    // benchmarks and gtest suites among them -- keeps compiling unchanged.
    const double* ref_prices = nullptr,
    // Annualized risk-free rate, as a decimal fraction. Subtracted per
    // period (rate / periods_per_year) from every return before Sharpe and
    // Sortino, matching metrics/risk_metrics.py exactly. Defaults to 0.0,
    // which is what this kernel always assumed -- so an unset value cannot
    // change a number that was already reported.
    double risk_free_rate = 0.0
);

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
    double periods_per_year = 252.0,
    // Last and defaulted so every existing positional call site -- the C++
    // benchmarks and gtest suites among them -- keeps compiling unchanged.
    const double* ref_prices = nullptr,
    // Annualized risk-free rate, as a decimal fraction. Subtracted per
    // period (rate / periods_per_year) from every return before Sharpe and
    // Sortino, matching metrics/risk_metrics.py exactly. Defaults to 0.0,
    // which is what this kernel always assumed -- so an unset value cannot
    // change a number that was already reported.
    double risk_free_rate = 0.0
);

/**
 * Fused crossover grid: generate each combination's signal and consume it
 * into the backtest immediately, without ever materializing a
 * (num_combos x n) signal matrix.
 *
 * Profiling a 300-combination x 5,000-bar SMA grid showed where the cost
 * actually sat:
 *
 *     python signal generation   121.4 ms   92.1%
 *     vstack into (combos,bars)    3.2 ms    2.4%
 *     native batch backtest        7.2 ms    5.4%
 *
 * So moving only the backtest into C++ left 92% of the work behind, and the
 * grid computed 600 moving averages where 35 UNIQUE periods were needed --
 * every combination recomputing an average another combination had already
 * produced.
 *
 * The split here follows from that. Python computes each unique indicator
 * ONCE (reusing its own already-verified implementations, so there is no
 * second definition of an indicator to drift), and passes:
 *
 *   @param indicators  (n_unique x n) row-major matrix of precomputed
 *                      indicator series, one row per UNIQUE parameter value.
 *   @param pair_idx    (num_combos x 2) row-major indices into `indicators`:
 *                      {fast_row, slow_row} for each combination.
 *
 * The kernel then writes one combination's signal into a single reusable
 * scratch buffer and backtests it before moving on, so peak memory is
 * O(n_unique * n + n) rather than O(num_combos * n). At the existing
 * 50,000-combination cap over 100,000 bars the old shape was a 40 GB
 * allocation; this never exceeds the indicator matrix.
 *
 * Signal convention matches _sma_signals in strategies.py: 1 when the fast
 * series is strictly above the slow one, else 0. NaN on either side (a
 * warm-up bar) yields 0, exactly as the pandas comparison does.
 */
std::vector<BacktestResult> batch_backtest_crossover(
    const double* prices,
    const double* indicators,
    std::size_t   n,
    std::size_t   n_unique,
    const int*    pair_idx,
    std::size_t   num_combos,
    double initial_capital  = 10'000.0,
    double commission_pct   = 0.001,
    double slippage_pct     = 0.0005,
    double periods_per_year = 252.0,
    const double* ref_prices = nullptr,
    // Annualized risk-free rate, as a decimal fraction. Subtracted per
    // period (rate / periods_per_year) from every return before Sharpe and
    // Sortino, matching metrics/risk_metrics.py exactly. Defaults to 0.0,
    // which is what this kernel always assumed -- so an unset value cannot
    // change a number that was already reported.
    double risk_free_rate = 0.0
);

/**
 * Batch backtest: run `num_tests` signal arrays against the same price series.
 *
 * Avoids the Python/C++ boundary-crossing overhead that accumulates when
 * calling run_strategy once per parameter combination from Python. Each test
 * index is independent, so the loop is OpenMP-parallel under the
 * sqt::omp_policy work threshold.
 *
 * (This comment used to sit two declarations higher up, immediately above
 * run_strategy_summary's own doc block -- so the function it describes had
 * none and the function it appeared to describe had two.)
 *
 * @param prices         Close prices, length n.
 * @param signals_flat   Flattened 2-D array, shape (num_tests × n) row-major.
 *                         signals_flat[t*n + i] = signal for test t at bar i.
 * @param n              Number of bars.
 * @param num_tests      Number of signal arrays.
 * @param initial_capital / commission_pct / slippage_pct / periods_per_year /
 *                       ref_prices — same for all tests, same meaning as
 *                       run_strategy().
 * @return               Vector of BacktestResult, one per test in input order.
 *                       equity_curve is empty in every result to save memory.
 */
// ── Multi-asset portfolio simulation ────────────────────────────────────────
//
// run_strategy is the single-asset case; this is the shared-cash portfolio
// account. The Python engine's per-bar loop had already been optimized hard
// (dense matrices materialized once, positional indexing, a vectorized
// rebalance path) and still cost 124.6 us/bar at 500 tickers, extrapolating
// to ~450 us/bar at 2,000 -- about 0.9 s for one 2,000-bar backtest, which a
// walk-forward or a parameter sweep multiplies by fifty or a hundred.
//
// SCOPE. This deliberately implements ONLY the configuration the Python
// engine itself already carves out as its vectorized fast path:
//
//     commission_model == "pct" and not use_impact_model
//                              and max_adv_participation is None
//
// The per-share commission model has a per-ORDER minimum, the impact model
// needs a per-ticker volatility lookup, and the ADV constraint has to raise
// naming one ticker -- each is a genuinely per-element decision that would
// have to be restated here to be supported, and restating it is how the two
// implementations drift. Everything outside the fast path stays on the
// Python loop, which is unchanged.

// Why a simulation stopped, so the caller can raise the same message it
// always raised rather than a generic one from the kernel.
enum PortfolioSimStatus : int {
    kPortfolioOk = 0,
    kPortfolioBadExecPrice,          // nonpositive/non-finite price, nonzero target
    kPortfolioInsolventAtRebalance,  // equity <= 0 after a rebalance's costs
    kPortfolioLeverageBreach,        // realized gross leverage over the limit
    kPortfolioPositionBreach,        // realized position size over the limit
    kPortfolioInsolventAtBar,        // equity <= 0 from price drift alone
};

struct PortfolioSimError {
    int         status = kPortfolioOk;
    std::size_t bar    = 0;    // bar index where it happened
    int         ticker = -1;   // ticker position, or -1 when not ticker-specific
    double      value  = 0.0;  // the offending quantity, for the message
};

// Where a rebalance executes. Mirrors the Python `fill_price` parameter.
enum PortfolioFill : int {
    kFillClose   = 0,  // execute at this bar's Close
    kFillNextOpen = 1, // execute at the NEXT bar's Open
    kFillHl2      = 2, // execute at this bar's (High+Low)/2
};

struct PortfolioCosts {
    double initial_capital     = 10'000.0;
    double commission_pct      = 0.001;
    // Commission charged on SALES. Defaults equal to commission_pct, which
    // is the symmetric behaviour every caller had before this field existed
    // -- so an unset value cannot change a number. The Python side resolves
    // its own Optional to a concrete rate before calling, so exactly one
    // place decides what "unset" means.
    double sell_commission_pct = 0.001;
    double slippage_pct        = 0.0005;
    double max_gross_leverage  = 1.0;
    double max_position_pct    = 1.0;
    double borrow_fee_bps      = 0.0;
    double margin_interest_rate = 0.0;
    int    fill                = kFillClose;
};

/**
 * Simulate one shared-cash portfolio account.
 *
 * @param close       (n_bars x n_tickers) row-major closes. Equity is always
 *                    marked to Close regardless of where trades execute.
 * @param exec_prices (n_bars x n_tickers) row-major execution prices: the
 *                    Open matrix for kFillNextOpen, (High+Low)/2 for kFillHl2,
 *                    and `close` itself for kFillClose.
 * @param weights     (n_rebal x n_tickers) row-major target weights.
 * @param rebal_bars  (n_rebal) bar index each weight row triggers at, ascending.
 * @param day_gaps    (n_bars) calendar days since the previous bar, for
 *                    financing accrual. day_gaps[0] is unused (1.0 by
 *                    convention). A Friday->Monday gap is 3, not 1.
 * @param out_equity, out_cash, out_gross, out_net  (n_bars) each.
 * @param out_rebal   (n_rebal x 3): turnover_pct, gross_leverage_after,
 *                    n_positions. Only rows for rebalances that actually
 *                    executed are written; see the return value.
 * @param err         Set when the simulation stops early. On any non-ok
 *                    status the output buffers are valid only up to `bar`.
 *
 * @returns the number of rebalances that executed.
 */
std::size_t run_portfolio_simulation(
    const double* close,
    const double* exec_prices,
    const double* weights,
    const long long* rebal_bars,
    const double* day_gaps,
    std::size_t   n_bars,
    std::size_t   n_tickers,
    std::size_t   n_rebal,
    const PortfolioCosts& costs,
    double* out_equity,
    double* out_cash,
    double* out_gross,
    double* out_net,
    double* out_rebal,
    // Peak |single position value| across every bar, in currency. A scalar
    // out-param rather than an (n_bars,) curve: the peak is what a
    // concentration limit is written against, and the per-bar series would
    // cost every caller a payload it reduces immediately. May be null.
    double* out_peak_position,
    PortfolioSimError* err);


std::vector<BacktestResult> batch_run_strategy(
    const double* prices,
    const double* signals_flat,
    std::size_t   n,
    std::size_t   num_tests,
    double initial_capital = 10'000.0,
    double commission_pct  = 0.001,
    double slippage_pct    = 0.0005,
    double periods_per_year = 252.0,
    // Last and defaulted so every existing positional call site -- the C++
    // benchmarks and gtest suites among them -- keeps compiling unchanged.
    const double* ref_prices = nullptr,
    // Annualized risk-free rate, as a decimal fraction. Subtracted per
    // period (rate / periods_per_year) from every return before Sharpe and
    // Sortino, matching metrics/risk_metrics.py exactly. Defaults to 0.0,
    // which is what this kernel always assumed -- so an unset value cannot
    // change a number that was already reported.
    double risk_free_rate = 0.0
);

}  // namespace sqt
