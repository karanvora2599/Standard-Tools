#include "sqt/backtest.hpp"

#include "sqt/numerics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#ifdef SQT_HAS_OPENMP
#include <omp.h>
#endif

namespace sqt {

namespace {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kPPY = 252.0;  // periods per year

    // ── Trade-log position accounting (weighted-average cost basis) ─────────
    //
    // Replaces the old "any pos_diff event closes-then-reopens" model, whose
    // same-sign RESIZE handling (e.g. size 1.0 -> 2.5) double-counted cost --
    // it treated a resize as closing a 1.0-sized trade AND opening a fresh
    // 2.5-sized one, each independently costed at 2*abs(own size), totaling
    // 2*(1.0+2.5)=7*cost_per_unit for a lot the equity curve itself only
    // ever charged sum(abs(pdiff))*cost_per_unit = (1.0+1.5+2.5)=5*cost_per_unit
    // for. PositionState now tracks a genuine weighted-average cost basis
    // across a lot's whole life (open, any same-sign resizes, and the final
    // close), so a resize is a partial ADD (blending cost basis, charging
    // only the incremental amount actually transacted that event) instead of
    // a full close+reopen -- equity P&L and trade-log stats now derive from
    // the same economic events instead of the previous approximation.
    //
    // A full close, a flip (close-then-reopen in one event), and the
    // final-bar flush of a still-open lot are UNCHANGED in total cost/pnl
    // from the old model for any sequence with no intermediate resize (see
    // tests/cpp/test_backtest.cpp's pinned open/close/reversal/flush tests,
    // which are unaffected by this rewrite) -- only the resize case's
    // accounting actually changes.
    struct PositionState {
        double size               = 0.0;  // signed net units held (0 = flat)
        double cost_basis         = 0.0;  // weighted-average entry price of the open lot
        double cost_accrued       = 0.0;  // sum of abs(delta)*cost_per_unit over the lot's life so far
        double realized_pnl_accum = 0.0;  // sum of realized pnl from any partial closes of this lot
    };

    struct TradeCompletion {
        bool   completed  = false;
        double return_pct = 0.0;
    };

    // Applies one bar's position-changing event (exec_i != prev_exec) to
    // `st`. exec_i/prev_exec are used directly wherever a raw *target*
    // position size is needed (never a delta-derived value), matching the
    // original entry_size=exec_i convention. Returns a completed trade's
    // return_pct only when this event fully closes the current lot (a
    // same-sign resize or a partial reduce never completes a trade by
    // itself) -- the caller decides what to do with a completion (push to a
    // vector, or fold into running scalar trade stats).
    TradeCompletion apply_position_event(
        PositionState& st, double exec_i, double prev_exec, double ref_price,
        double cost_per_unit)
    {
        const double pdiff = exec_i - prev_exec;
        TradeCompletion result;
        if (pdiff == 0.0) return result;

        if (st.size != 0.0 && (pdiff > 0.0) != (st.size > 0.0)) {
            // Opposite sign: reduce, fully close, or close-then-flip.
            const double pos_sign    = (st.size > 0.0) ? 1.0 : -1.0;
            const double closing_qty = std::min(std::abs(pdiff), std::abs(st.size));

            st.cost_accrued += closing_qty * cost_per_unit;
            st.realized_pnl_accum += (st.cost_basis != 0.0)
                ? (ref_price - st.cost_basis) / st.cost_basis * (closing_qty * pos_sign)
                : 0.0;
            st.size -= closing_qty * pos_sign;

            if (st.size == 0.0) {
                result.completed  = true;
                result.return_pct = (st.realized_pnl_accum - st.cost_accrued) * 100.0;
                st = PositionState{};
            }
        } else if (st.size != 0.0) {
            // Same sign: a resize/add -- blend cost basis, charge cost only
            // for the incremental amount actually transacted this event.
            const double old_notional = st.size * st.cost_basis;
            st.size += pdiff;
            st.cost_basis     = (old_notional + pdiff * ref_price) / st.size;
            st.cost_accrued  += std::abs(pdiff) * cost_per_unit;
            return result;  // a resize never completes a trade
        }

        if (st.size == 0.0 && exec_i != 0.0) {
            // Opening a fresh lot -- either already flat, or the branch
            // above just fully closed the prior lot (a flip). Uses exec_i
            // directly (the raw target position), not a delta-derived
            // value.
            st.cost_basis    = ref_price;
            st.size          = exec_i;
            st.cost_accrued += std::abs(exec_i) * cost_per_unit;
        }
        return result;
    }

    // Mark-to-market close of a still-open lot at the series' final price
    // (mirrors the original "no real exit event occurred" flush -- entry/
    // resize costs only, already reflected in st.cost_accrued; no
    // additional cost charged here).
    TradeCompletion flush_open_lot(const PositionState& st, double final_price) {
        TradeCompletion result;
        if (st.size == 0.0) return result;
        const double pnl = (st.cost_basis != 0.0)
            ? (final_price - st.cost_basis) / st.cost_basis * st.size
            : 0.0;
        result.completed  = true;
        result.return_pct = (st.realized_pnl_accum + pnl - st.cost_accrued) * 100.0;
        return result;
    }
}

BacktestResult run_strategy(
    const double* prices,
    const double* signals,
    std::size_t   n,
    double initial_capital,
    double commission_pct,
    double slippage_pct)
{
    BacktestResult r{};
    r.equity_curve.resize(n, initial_capital);
    r.final_equity         = initial_capital;
    r.total_return         = 0.0;
    r.annualized_vol       = 0.0;
    r.sharpe_ratio         = 0.0;
    r.sortino_ratio        = kInf;
    r.max_drawdown         = 0.0;
    r.calmar_ratio         = 0.0;
    r.num_trades           = 0;
    r.win_rate             = 0.0;
    r.profit_factor        = 0.0;
    r.avg_trade_return_pct = 0.0;

    if (n == 0) return r;

    const double cost_per_unit = commission_pct + slippage_pct;

    // ── Strategy returns (vectorized formula) ─────────────────────────────────
    // executed[i] = signals[i-1] for i≥1, 0 for i=0
    // returns[i]  = (prices[i] - prices[i-1]) / prices[i-1], 0 for i=0
    // strat_ret[i]= executed[i] * returns[i] - |pos_diff[i]| * cost_per_unit

    std::vector<double> strat_ret(n, 0.0);

    // ── Trade log: weighted-average-cost-basis position accounting ───────────
    // (mirrors _build_trade_log in engine.py) via the shared
    // PositionState/apply_position_event/flush_open_lot helpers above --
    // see their doc comments for the full rationale (this replaces a
    // same-sign-resize approximation that used to double-count cost).
    std::vector<double> trade_rets;  // per-trade return_pct (×100 scale)
    PositionState pos;
    double prev_exec = 0.0;

    for (std::size_t i = 1; i < n; ++i) {
        const double ref_price = prices[i - 1];
        const double ret_i     = (ref_price != 0.0)
            ? (prices[i] - ref_price) / ref_price
            : 0.0;
        const double exec_i = signals[i - 1];
        const double pdiff  = exec_i - prev_exec;
        const double tcost  = std::abs(pdiff) * cost_per_unit;

        strat_ret[i] = exec_i * ret_i - tcost;

        const auto tc = apply_position_event(pos, exec_i, prev_exec, ref_price, cost_per_unit);
        if (tc.completed) trade_rets.push_back(tc.return_pct);

        prev_exec = exec_i;
    }

    // Flush last open trade at final Close price (mirrors Python's
    // synthesized final-bar exit — entry/resize costs only, no real exit event).
    const auto final_tc = flush_open_lot(pos, prices[n - 1]);
    if (final_tc.completed) trade_rets.push_back(final_tc.return_pct);

    // ── Equity curve: cumprod(1 + strat_ret) ──────────────────────────────────
    r.equity_curve[0] = initial_capital;
    for (std::size_t i = 1; i < n; ++i)
        r.equity_curve[i] = r.equity_curve[i - 1] * (1.0 + strat_ret[i]);

    r.final_equity = r.equity_curve[n - 1];
    r.total_return = (r.final_equity - initial_capital) / initial_capital;

    // ── Max drawdown: min of (equity - running_peak) / running_peak ───────────
    double peak = r.equity_curve[0];
    double mdd  = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        if (r.equity_curve[i] > peak) peak = r.equity_curve[i];
        if (peak > 0.0) {
            const double dd = (r.equity_curve[i] - peak) / peak;
            if (dd < mdd) mdd = dd;
        }
    }
    r.max_drawdown = mdd;  // negative convention

    // ── Volatility, Sharpe, Sortino ───────────────────────────────────────────
    // pandas .std() uses sample variance (ddof=1) over all N elements of strat_ret.

    // Loop bounds/accumulation use n (size_t) directly rather than
    // narrowing to int first -- n can exceed INT_MAX for a large series.
    const double n_d = static_cast<double>(n);

    double mean_r = 0.0;
    for (std::size_t i = 0; i < n; ++i) mean_r += strat_ret[i];
    mean_r /= n_d;

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = strat_ret[i] - mean_r;
        sum_sq += d * d;
    }

    const double sample_std = (n > 1) ? std::sqrt(sum_sq / (n_d - 1.0)) : 0.0;
    r.annualized_vol = sample_std * std::sqrt(kPPY);
    r.sharpe_ratio   = (sample_std > 0.0)
        ? (mean_r / sample_std) * std::sqrt(kPPY) : 0.0;

    // Sortino: semi-deviation = sqrt(mean(min(r, 0)^2)) across ALL periods.
    // Sortino & Price (1994) definition — zero contribution from profitable bars.
    {
        double down_sq_sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double d = std::min(strat_ret[i], 0.0);
            down_sq_sum += d * d;
        }
        const double down_dev = std::sqrt(down_sq_sum / n_d) * std::sqrt(kPPY);
        r.sortino_ratio = (down_dev > 0.0) ? (mean_r * kPPY) / down_dev : kInf;
    }

    // ── Calmar: CAGR / |max_drawdown|  (CAGR = (final/initial)^(252/n) - 1) ──
    if (n > 1 && r.final_equity > 0.0 && initial_capital > 0.0) {
        const double ann_ret = std::pow(r.final_equity / initial_capital,
                                        kPPY / static_cast<double>(n)) - 1.0;
        const double abs_mdd = std::abs(mdd);
        r.calmar_ratio = (abs_mdd > 0.0) ? ann_ret / abs_mdd : kInf;
    }

    // ── Trade statistics ──────────────────────────────────────────────────────
    r.num_trades = numerics::checked_narrow_to_int(trade_rets.size(), "run_strategy: num_trades");
    if (r.num_trades > 0) {
        int    n_wins    = 0;
        double gross_win = 0.0, gross_loss = 0.0, sum_tr = 0.0;
        for (double tr : trade_rets) {
            sum_tr += tr;
            if (tr > 0.0) { ++n_wins; gross_win += tr; }
            else gross_loss += std::abs(tr);
        }
        r.win_rate           = static_cast<double>(n_wins) / r.num_trades;
        // gross_loss == 0 means there were no losing trades at all, which is
        // reported as +inf regardless of gross_win -- the convention this
        // file's own test already documents ("no losing trades -> inf",
        // tests/cpp/test_backtest.cpp) and the one engine.py's
        // _compute_trade_stats uses. The previous `gross_win > 0.0` sub-
        // condition made the 0/0 case (every trade returning exactly 0.0,
        // e.g. a flat price series with zero costs) return 0.0 here while
        // Python returned inf -- the same call disagreeing across backends,
        // and inconsistent with this function's own no-losing-trades rule.
        r.profit_factor      = (gross_loss > 0.0) ? gross_win / gross_loss : kInf;
        r.avg_trade_return_pct = sum_tr / r.num_trades;
    }

    return r;
}

// ── run_strategy_summary ────────────────────────────────────────────────────
//
// Same algorithm as run_strategy() above -- every formula, every op order,
// is copied verbatim -- just restructured into two allocation-free passes
// instead of six array-backed passes. strat_ret[i] has no true loop-carried
// dependency: exec_i = signals[i-1], and the prev_exec pdiff needs equals
// signals[i-2] for i>=2 (or 0.0 for i==1), both directly index-derivable.
// Only the trade-log open/close bookkeeping is a genuine sequential state
// machine, and that's preserved exactly as-is in pass 1.
//
// Bit-identical-by-construction with run_strategy(): pass 1's fused
// equity/peak/drawdown/mean tracking processes i=1..n-1 in the same order
// as run_strategy()'s separate equity-curve and max-drawdown loops (the
// underlying arithmetic per step is unchanged, only where the running
// equity value lives -- a scalar here instead of equity_curve[i]). Pass 2's
// sum_sq/down_sq_sum accumulation is seeded with index 0's implicit
// strat_ret[0]=0.0 contribution ((0-mean_r)^2 = mean_r*mean_r exactly, and
// min(0,0)^2=0) before looping i=1..n-1 -- 0.0 + x == x exactly in IEEE 754,
// so this reproduces run_strategy()'s i=0..N-1 accumulation order bit for
// bit, just starting the running sum from the i=0 term's value directly
// instead of adding it as a loop iteration.
BacktestResult run_strategy_summary(
    const double* prices,
    const double* signals,
    std::size_t   n,
    double initial_capital,
    double commission_pct,
    double slippage_pct)
{
    BacktestResult r{};
    r.final_equity         = initial_capital;
    r.total_return         = 0.0;
    r.annualized_vol       = 0.0;
    r.sharpe_ratio         = 0.0;
    r.sortino_ratio        = kInf;
    r.max_drawdown         = 0.0;
    r.calmar_ratio         = 0.0;
    r.num_trades           = 0;
    r.win_rate             = 0.0;
    r.profit_factor        = 0.0;
    r.avg_trade_return_pct = 0.0;

    if (n == 0) return r;

    const double cost_per_unit = commission_pct + slippage_pct;
    // n (size_t) used directly below rather than narrowed to int -- n can
    // exceed INT_MAX for a large series.
    const double n_d = static_cast<double>(n);

    // ── Pass 1: trade-log position accounting + running equity/peak/
    //    drawdown/sum, fused into one forward loop, zero array allocation.
    //    Trade-log accounting uses the same shared PositionState/
    //    apply_position_event/flush_open_lot helpers run_strategy() uses
    //    (see their doc comments above). ───────────────────────────────
    PositionState pos;
    double prev_exec = 0.0;

    // long long (not int): num_trades is bounded by n and accumulated one
    // increment at a time, so it can't itself exceed n -- kept wide here
    // and only checked-narrowed to the public BacktestResult::num_trades
    // (int) field at the end, rather than risking silent int-overflow
    // during accumulation for a pathologically trade-dense huge series.
    long long num_trades = 0;
    long long n_wins     = 0;
    double gross_win  = 0.0, gross_loss = 0.0, sum_tr = 0.0;

    double equity = initial_capital;
    double peak   = initial_capital;
    double mdd    = 0.0;
    double sum_r  = 0.0;

    auto fold_trade = [&](double tr) {
        ++num_trades;
        sum_tr += tr;
        if (tr > 0.0) { ++n_wins; gross_win += tr; }
        else gross_loss += std::abs(tr);
    };

    for (std::size_t i = 1; i < n; ++i) {
        const double ref_price = prices[i - 1];
        const double ret_i     = (ref_price != 0.0)
            ? (prices[i] - ref_price) / ref_price
            : 0.0;
        const double exec_i = signals[i - 1];
        const double pdiff  = exec_i - prev_exec;
        const double tcost  = std::abs(pdiff) * cost_per_unit;

        const double strat_ret_i = exec_i * ret_i - tcost;

        equity *= (1.0 + strat_ret_i);
        if (equity > peak) peak = equity;
        if (peak > 0.0) {
            const double dd = (equity - peak) / peak;
            if (dd < mdd) mdd = dd;
        }

        sum_r += strat_ret_i;

        const auto tc = apply_position_event(pos, exec_i, prev_exec, ref_price, cost_per_unit);
        if (tc.completed) fold_trade(tc.return_pct);

        prev_exec = exec_i;
    }

    // Flush last open trade at final Close price (mirrors run_strategy()).
    const auto final_tc = flush_open_lot(pos, prices[n - 1]);
    if (final_tc.completed) fold_trade(final_tc.return_pct);

    r.final_equity = equity;
    r.total_return = (r.final_equity - initial_capital) / initial_capital;
    r.max_drawdown = mdd;

    const double mean_r = sum_r / n_d;

    // ── Calmar: same formula/placement as run_strategy() ─────────────────────
    if (n > 1 && r.final_equity > 0.0 && initial_capital > 0.0) {
        const double ann_ret = std::pow(r.final_equity / initial_capital,
                                        kPPY / static_cast<double>(n)) - 1.0;
        const double abs_mdd = std::abs(mdd);
        r.calmar_ratio = (abs_mdd > 0.0) ? ann_ret / abs_mdd : kInf;
    }

    // ── Pass 2: recompute strat_ret[i] on demand (no state carried across
    //    iterations -- exec_i/prev_exec are both directly index-derivable)
    //    to get variance and downside deviation now that mean_r is known.
    //    Seeded with index 0's implicit strat_ret[0]=0.0 term. ────────────
    double sum_sq      = mean_r * mean_r;
    double down_sq_sum = 0.0;

    for (std::size_t i = 1; i < n; ++i) {
        const double ref_price = prices[i - 1];
        const double ret_i     = (ref_price != 0.0)
            ? (prices[i] - ref_price) / ref_price
            : 0.0;
        const double exec_i      = signals[i - 1];
        const double prev_exec_i = (i >= 2) ? signals[i - 2] : 0.0;
        const double pdiff       = exec_i - prev_exec_i;
        const double tcost       = std::abs(pdiff) * cost_per_unit;
        const double strat_ret_i = exec_i * ret_i - tcost;

        const double d = strat_ret_i - mean_r;
        sum_sq += d * d;

        const double down_d = std::min(strat_ret_i, 0.0);
        down_sq_sum += down_d * down_d;
    }

    const double sample_std = (n > 1) ? std::sqrt(sum_sq / (n_d - 1.0)) : 0.0;
    r.annualized_vol = sample_std * std::sqrt(kPPY);
    r.sharpe_ratio   = (sample_std > 0.0)
        ? (mean_r / sample_std) * std::sqrt(kPPY) : 0.0;

    const double down_dev = std::sqrt(down_sq_sum / n_d) * std::sqrt(kPPY);
    r.sortino_ratio = (down_dev > 0.0) ? (mean_r * kPPY) / down_dev : kInf;

    // ── Trade statistics ──────────────────────────────────────────────────────
    r.num_trades = numerics::checked_narrow_to_int(
        static_cast<std::size_t>(num_trades), "run_strategy_summary: num_trades");
    if (r.num_trades > 0) {
        r.win_rate           = static_cast<double>(n_wins) / r.num_trades;
        // gross_loss == 0 means there were no losing trades at all, which is
        // reported as +inf regardless of gross_win -- the convention this
        // file's own test already documents ("no losing trades -> inf",
        // tests/cpp/test_backtest.cpp) and the one engine.py's
        // _compute_trade_stats uses. The previous `gross_win > 0.0` sub-
        // condition made the 0/0 case (every trade returning exactly 0.0,
        // e.g. a flat price series with zero costs) return 0.0 here while
        // Python returned inf -- the same call disagreeing across backends,
        // and inconsistent with this function's own no-losing-trades rule.
        r.profit_factor      = (gross_loss > 0.0) ? gross_win / gross_loss : kInf;
        r.avg_trade_return_pct = sum_tr / r.num_trades;
    }

    return r;
}

// ── batch_run_strategy ────────────────────────────────────────────────────────
//
// Each test index t is fully independent: run_strategy_summary() is a pure
// function of its own (prices, signals_flat + t*n, n, ...) slice, with no
// shared mutable state and no RNG -- unlike simulate_forward_paths_into
// (monte_carlo.cpp), no per-thread setup is needed before the loop, so the
// simpler combined `#pragma omp parallel for` form is correct here (that
// file's nested `#pragma omp parallel { ... #pragma omp for ... }` form
// exists specifically to declare a thread-local RNG once per thread, which
// this loop has no equivalent need for). `results` must be pre-sized via
// resize() and written through indexed assignment -- reserve()+push_back()
// is not thread-safe across concurrent writers.
std::vector<BacktestResult> batch_run_strategy(
    const double* prices,
    const double* signals_flat,
    std::size_t   n,
    std::size_t   num_tests,
    double initial_capital,
    double commission_pct,
    double slippage_pct)
{
    std::vector<BacktestResult> results(num_tests);

    // run_strategy_summary() calls numerics::checked_narrow_to_int(), which
    // THROWS on overflow -- and it is invoked from inside the parallel region
    // below. An exception that escapes an OpenMP structured block is
    // undefined behavior (the spec requires it to be caught by the same
    // thread inside the same region); in practice it terminates the process.
    // num_trades is bounded by n, so validating n here -- once, before the
    // region, where a throw is safe -- makes that inner narrowing
    // unreachable rather than merely unlikely.
    (void)numerics::checked_narrow_to_int(n, "batch_run_strategy: bars per test");

    // Signed loop variable: MSVC's OpenMP 2.0 canonical-for-loop form
    // requires a signed integer induction variable, not std::size_t.
    const long long num_tests_ll = static_cast<long long>(num_tests);

#ifdef SQT_HAS_OPENMP
    #pragma omp parallel for schedule(static) if(num_tests > 1)
#endif
    for (long long t = 0; t < num_tests_ll; ++t) {
        results[static_cast<std::size_t>(t)] = run_strategy_summary(
            prices,
            signals_flat + static_cast<std::size_t>(t) * n,
            n,
            initial_capital,
            commission_pct,
            slippage_pct);
    }
    return results;
}

}  // namespace sqt
