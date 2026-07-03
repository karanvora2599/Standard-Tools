#include "sqt/backtest.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sqt {

namespace {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kPPY = 252.0;  // periods per year
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
    const int    N             = static_cast<int>(n);

    // ── Strategy returns (vectorized formula) ─────────────────────────────────
    // executed[i] = signals[i-1] for i≥1, 0 for i=0
    // returns[i]  = (prices[i] - prices[i-1]) / prices[i-1], 0 for i=0
    // strat_ret[i]= executed[i] * returns[i] - |pos_diff[i]| * cost_per_unit

    std::vector<double> strat_ret(n, 0.0);

    // ── Trade log state machine (mirrors _build_trade_log) ────────────────────
    std::vector<double> trade_rets;  // per-trade return_pct (×100 scale)
    bool   has_open    = false;
    double entry_price = 0.0;
    int    entry_dir   = 0;
    double prev_exec   = 0.0;

    for (std::size_t i = 1; i < n; ++i) {
        const double ret_i  = (prices[i - 1] != 0.0)
            ? (prices[i] - prices[i - 1]) / prices[i - 1]
            : 0.0;
        const double exec_i = signals[i - 1];
        const double pdiff  = exec_i - prev_exec;
        const double tcost  = std::abs(pdiff) * cost_per_unit;

        strat_ret[i] = exec_i * ret_i - tcost;

        if (pdiff != 0.0) {
            if (has_open) {
                const double pnl = (prices[i] - entry_price) / entry_price * entry_dir;
                trade_rets.push_back(pnl * 100.0);
                has_open = false;
            }
            if (exec_i != 0.0) {
                entry_price = prices[i];
                entry_dir   = (exec_i > 0.0) ? 1 : -1;
                has_open    = true;
            }
        }

        prev_exec = exec_i;
    }

    // Flush last open trade at final price (mirrors Python _build_trade_log).
    if (has_open && n > 0) {
        const double pnl = (prices[n - 1] - entry_price) / entry_price * entry_dir;
        trade_rets.push_back(pnl * 100.0);
    }

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

    double mean_r = 0.0;
    for (int i = 0; i < N; ++i) mean_r += strat_ret[i];
    mean_r /= N;

    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) {
        const double d = strat_ret[i] - mean_r;
        sum_sq += d * d;
    }

    const double sample_std = (N > 1) ? std::sqrt(sum_sq / (N - 1)) : 0.0;
    r.annualized_vol = sample_std * std::sqrt(kPPY);
    r.sharpe_ratio   = (sample_std > 0.0)
        ? (mean_r / sample_std) * std::sqrt(kPPY) : 0.0;

    // Sortino: semi-deviation = sqrt(mean(min(r, 0)^2)) across ALL periods.
    // Sortino & Price (1994) definition — zero contribution from profitable bars.
    {
        double down_sq_sum = 0.0;
        for (int i = 0; i < N; ++i) {
            const double d = std::min(strat_ret[i], 0.0);
            down_sq_sum += d * d;
        }
        const double down_dev = std::sqrt(down_sq_sum / N) * std::sqrt(kPPY);
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
    r.num_trades = static_cast<int>(trade_rets.size());
    if (r.num_trades > 0) {
        int    n_wins    = 0;
        double gross_win = 0.0, gross_loss = 0.0, sum_tr = 0.0;
        for (double tr : trade_rets) {
            sum_tr += tr;
            if (tr > 0.0) { ++n_wins; gross_win += tr; }
            else gross_loss += std::abs(tr);
        }
        r.win_rate           = static_cast<double>(n_wins) / r.num_trades;
        r.profit_factor      = (gross_loss > 0.0) ? gross_win / gross_loss
                             : (gross_win > 0.0   ? kInf : 0.0);
        r.avg_trade_return_pct = sum_tr / r.num_trades;
    }

    return r;
}

// ── batch_run_strategy ────────────────────────────────────────────────────────

std::vector<BacktestResult> batch_run_strategy(
    const double* prices,
    const double* signals_flat,
    std::size_t   n,
    std::size_t   num_tests,
    double initial_capital,
    double commission_pct,
    double slippage_pct)
{
    std::vector<BacktestResult> results;
    results.reserve(num_tests);
    for (std::size_t t = 0; t < num_tests; ++t) {
        BacktestResult r = run_strategy(
            prices,
            signals_flat + t * n,
            n,
            initial_capital,
            commission_pct,
            slippage_pct);
        // Strip equity curve to save memory — not needed for grid search
        r.equity_curve.clear();
        r.equity_curve.shrink_to_fit();
        results.push_back(std::move(r));
    }
    return results;
}

}  // namespace sqt
