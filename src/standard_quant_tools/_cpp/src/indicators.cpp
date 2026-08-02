#include "sqt/indicators.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

namespace sqt {

namespace {
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}  // namespace


// ── RSI ───────────────────────────────────────────────────────────────────────
//
// Wilder's smoothing: SMA seed for first `period` bars, then exponential
// smoothing with alpha = 1/period.  Matches the Numba reference exactly.

void rsi_into(const double* SQT_RESTRICT prices, std::size_t n, int period,
              double* SQT_RESTRICT out) {
    std::fill(out, out + n, kNaN);

    // period <= 0 would index out[period] with a negative/zero-derived
    // value below — for period < 0 that wraps to a huge size_t via the
    // implicit int->size_t conversion in operator[], an out-of-bounds write.
    // Reject up front rather than relying on downstream arithmetic to stay
    // in range.
    if (period <= 0) return;
    if (static_cast<int>(n) <= period) return;

    // Seed: simple mean of first `period` gains/losses
    double avg_gain = 0.0, avg_loss = 0.0;
    for (int i = 1; i <= period; ++i) {
        const double change = prices[i] - prices[i - 1];
        if (change > 0.0) avg_gain += change;
        else              avg_loss -= change;
    }
    avg_gain /= period;
    avg_loss /= period;

    const auto to_rsi = [](double gain, double loss) -> double {
        if (loss == 0.0) return 100.0;
        const double rs = gain / loss;
        return 100.0 - (100.0 / (1.0 + rs));
    };

    out[period] = to_rsi(avg_gain, avg_loss);

    // Wilder's forward pass
    for (std::size_t i = static_cast<std::size_t>(period) + 1; i < n; ++i) {
        const double change = prices[i] - prices[i - 1];
        const double gain   = (change > 0.0) ? change : 0.0;
        const double loss   = (change < 0.0) ? -change : 0.0;

        avg_gain = (avg_gain * (period - 1) + gain) / period;
        avg_loss = (avg_loss * (period - 1) + loss) / period;

        out[i] = to_rsi(avg_gain, avg_loss);
    }
}

std::vector<double> rsi(const double* prices, std::size_t n, int period) {
    std::vector<double> result(n);
    rsi_into(prices, n, period, result.data());
    return result;
}


// ── ADX ───────────────────────────────────────────────────────────────────────
//
// Wilder's Average Directional Index.  Flat row-major output: (DI+, DI-, ADX).
// Matches _adx_numba in trend.py exactly.

void adx_into(
    const double* SQT_RESTRICT high,
    const double* SQT_RESTRICT low,
    const double* SQT_RESTRICT close,
    std::size_t   n,
    int           period,
    double* SQT_RESTRICT       out)
{
    // 3 columns per bar: DI+, DI-, ADX
    std::fill(out, out + 3 * n, kNaN);

    // period <= 0 would index out[period*3+...] with a negative/zero
    // value and divide by `period` in the Wilder smoothing below.
    if (period <= 0) return;
    if (n < 2 || static_cast<int>(n) <= period) return;

    // Single fused O(1)-auxiliary-memory pass -- no dm_plus/dm_minus/tr/
    // dx_vals arrays (previously 4 full n-sized buffers beyond the output).
    // Wilder's smoothing only ever needs the immediately-previous smoothed
    // sum (atr_s/dmp_s/dmm_s below) plus the CURRENT bar's raw TR/DM
    // value, which is computable inline from high[i]/high[i-1]/low[i]/
    // low[i-1]/close[i-1] with no lookback array; DI/DX's own seed window
    // (Steps 2 and 4 below) only ever needs a running sum of the values
    // seen so far, not the individual values themselves. This performs the
    // exact same sequence of `+=`/smoothing operations in the exact same
    // i=1..n-1 order as the original 4-pass version -- floating-point
    // addition isn't associative, so preserving the *order* of operations
    // (not just the set of values summed) is what makes this bit-identical
    // to the original, not merely numerically close (pinned by an exact-
    // equality regression test in tests/cpp/test_indicators.cpp).
    double atr_s = 0.0, dmp_s = 0.0, dmm_s = 0.0;  // Steps 1-3's Wilder sums
    double dx_seed_sum = 0.0;                       // Step 4's ADX seed accumulator
    double adx_val = 0.0;

    const auto di_p_val = [&]() { return (atr_s != 0.0) ? 100.0 * dmp_s / atr_s : 0.0; };
    const auto di_m_val = [&]() { return (atr_s != 0.0) ? 100.0 * dmm_s / atr_s : 0.0; };

    // Needs `period` DX values to initialise ADX -> starts at bar 2*period-1.
    const std::size_t adx_start = static_cast<std::size_t>(2 * period - 1);

    for (std::size_t i = 1; i < n; ++i) {
        const double up_move   = high[i] - high[i - 1];
        const double down_move = low[i - 1] - low[i];
        const double dm_plus_i  = (up_move > down_move && up_move > 0.0)   ? up_move   : 0.0;
        const double dm_minus_i = (down_move > up_move && down_move > 0.0) ? down_move : 0.0;
        const double tr_i = std::max({
            high[i] - low[i],
            std::abs(high[i] - close[i - 1]),
            std::abs(low[i]  - close[i - 1]),
        });

        if (static_cast<int>(i) <= period) {
            // Step 2: Wilder's seed sums.
            atr_s += tr_i;
            dmp_s += dm_plus_i;
            dmm_s += dm_minus_i;
        } else {
            // Step 3: Wilder's smooth forward.
            atr_s = atr_s - (atr_s / period) + tr_i;
            dmp_s = dmp_s - (dmp_s / period) + dm_plus_i;
            dmm_s = dmm_s - (dmm_s / period) + dm_minus_i;
        }

        if (static_cast<int>(i) < period) continue;  // DI/DX undefined before bar `period`

        const double di_p = di_p_val();
        const double di_m = di_m_val();
        out[i * 3 + 0] = di_p;
        out[i * 3 + 1] = di_m;

        const double di_sum = di_p + di_m;
        const double dx_i = (di_sum != 0.0) ? 100.0 * std::abs(di_p - di_m) / di_sum : 0.0;

        // Step 4: ADX = Wilder's smooth of DX.
        if (i <= adx_start) {
            dx_seed_sum += dx_i;
            if (i == adx_start) {
                adx_val = dx_seed_sum / period;
                out[adx_start * 3 + 2] = adx_val;
            }
        } else {
            adx_val = (adx_val * (period - 1) + dx_i) / period;
            out[i * 3 + 2] = adx_val;
        }
    }
}

std::vector<double> adx(
    const double* high,
    const double* low,
    const double* close,
    std::size_t   n,
    int           period)
{
    std::vector<double> result(3 * n);
    adx_into(high, low, close, n, period, result.data());
    return result;
}


// ── Parabolic SAR ─────────────────────────────────────────────────────────────
//
// State machine with SAR-clamp rules identical to _psar_numba in trend.py.
// Flat row-major output: (SAR, Trend) per bar.

void parabolic_sar_into(
    const double* SQT_RESTRICT high,
    const double* SQT_RESTRICT low,
    std::size_t   n,
    double        af_start,
    double        af_step,
    double        af_max,
    double* SQT_RESTRICT       out)
{
    // 2 columns per bar: SAR, Trend
    std::fill(out, out + 2 * n, kNaN);
    if (n == 0) return;

    // Not a crash risk (af_* only feed floating-point arithmetic below, no
    // indexing) but a nonsensical combination (e.g. af_max < af_start, a
    // negative acceleration factor) silently produces a numerically
    // meaningless SAR series rather than an obviously-wrong one — reject
    // up front, same convention as the period<=0 guards elsewhere in this
    // file.
    if (!std::isfinite(af_start) || !std::isfinite(af_step) || !std::isfinite(af_max) ||
        af_start <= 0.0 || af_step < 0.0 || af_max <= 0.0 || af_max < af_start) {
        return;
    }

    // Bootstrap: assume rising trend from bar 0
    double sar       = low[0];
    double ep        = high[0];
    double af        = af_start;
    bool   is_rising = true;

    out[0] = sar;
    out[1] = 1.0;

    for (std::size_t i = 1; i < n; ++i) {
        const double prev_sar = sar;

        if (is_rising) {
            sar = prev_sar + af * (ep - prev_sar);
            // SAR must stay below the two prior lows
            sar = std::min(sar, low[i - 1]);
            if (i >= 2) sar = std::min(sar, low[i - 2]);

            if (high[i] > ep) {
                ep = high[i];
                af = std::min(af + af_step, af_max);
            }

            if (low[i] < sar) {
                // Bearish reversal
                is_rising = false;
                sar = ep;
                ep  = low[i];
                af  = af_start;
            }
        } else {
            sar = prev_sar - af * (prev_sar - ep);
            // SAR must stay above the two prior highs
            sar = std::max(sar, high[i - 1]);
            if (i >= 2) sar = std::max(sar, high[i - 2]);

            if (low[i] < ep) {
                ep = low[i];
                af = std::min(af + af_step, af_max);
            }

            if (high[i] > sar) {
                // Bullish reversal
                is_rising = true;
                sar = ep;
                ep  = high[i];
                af  = af_start;
            }
        }

        out[i * 2 + 0] = sar;
        out[i * 2 + 1] = is_rising ? 1.0 : -1.0;
    }
}

std::vector<double> parabolic_sar(
    const double* high,
    const double* low,
    std::size_t   n,
    double        af_start,
    double        af_step,
    double        af_max)
{
    std::vector<double> result(2 * n);
    parabolic_sar_into(high, low, n, af_start, af_step, af_max, result.data());
    return result;
}


// ── Wilder's ATR ──────────────────────────────────────────────────────────────
//
// Identical smoothing to RSI and ADX: SMA seed for the first `period` bars,
// then alpha=1/period.  Not the same as a simple rolling mean of TR.

void wilder_atr_into(
    const double* SQT_RESTRICT high,
    const double* SQT_RESTRICT low,
    const double* SQT_RESTRICT close,
    std::size_t   n,
    int           period,
    double* SQT_RESTRICT       out)
{
    std::fill(out, out + n, kNaN);

    // period <= 0 would index out[period-1] with a negative/zero-derived
    // value below — for period <= 0 that wraps to a huge size_t, an
    // out-of-bounds write (this is the exact case the reviewer flagged).
    if (period <= 0) return;
    if (static_cast<int>(n) < period) return;

    // ── True range ────────────────────────────────────────────────────────────
    // Bar 0 has no previous close; use high - low.
    std::vector<double> tr(n);
    tr[0] = high[0] - low[0];
    for (std::size_t i = 1; i < n; ++i) {
        tr[i] = std::max({
            high[i] - low[i],
            std::abs(high[i] - close[i - 1]),
            std::abs(low[i]  - close[i - 1]),
        });
    }

    // ── Seed: SMA of first `period` TR values ─────────────────────────────────
    double atr_val = 0.0;
    for (int i = 0; i < period; ++i) atr_val += tr[i];
    atr_val /= period;
    out[static_cast<std::size_t>(period) - 1] = atr_val;

    // ── Wilder's forward smoothing ────────────────────────────────────────────
    for (std::size_t i = static_cast<std::size_t>(period); i < n; ++i) {
        atr_val = (atr_val * (period - 1) + tr[i]) / period;
        out[i] = atr_val;
    }
}

std::vector<double> wilder_atr(
    const double* high,
    const double* low,
    const double* close,
    std::size_t   n,
    int           period)
{
    std::vector<double> result(n);
    wilder_atr_into(high, low, close, n, period, result.data());
    return result;
}

// ── Bollinger Bands ───────────────────────────────────────────────────────────
//
// Fused sliding-window mean + sample std (ddof=1) in one pass.
// Maintains incremental Sx (sum) and Sxx (sum of squares) so both statistics
// are computed without re-iterating the window.

void bollinger_bands_into(
    const double* SQT_RESTRICT prices,
    std::size_t   n,
    int           period,
    double        num_std,
    double* SQT_RESTRICT       out)
{
    std::fill(out, out + 3 * n, kNaN);
    if (static_cast<int>(n) < period || period < 2) return;

    const double W   = static_cast<double>(period);
    const double dof = W - 1.0;  // ddof=1, matching pandas .std()

    // Raw-moment sums (Sx, Sxx of the *unshifted* prices) suffer
    // catastrophic cancellation for a large-baseline series: e.g. a
    // ~1e9-level price with genuine variance ~0.35 previously produced
    // var = -215.58 here (verified by hand), clamped to std=0 -- Bollinger
    // bands collapsing to a flat moving average, silently. Shifting by a
    // per-window reference `c` close to the window's own values before
    // accumulating keeps Sx'/Sxx' near the *variation* magnitude, not the
    // baseline-squared magnitude, which is what actually causes the
    // cancellation -- same fix class as the shifted-data / two-pass
    // algorithm, kept compatible with this function's O(1)-per-bar sliding
    // update by periodically re-deriving both the shift and the sums from
    // scratch (same "periodic full recompute" idiom rolling_regression.cpp's
    // rolling_factor_loadings already uses to bound floating-point drift).
    double c = 0.0, Sx = 0.0, Sxx = 0.0;
    std::size_t since_refresh = 0;

    auto recompute_window = [&](std::size_t start) {
        c = prices[start];
        Sx = 0.0;
        Sxx = 0.0;
        for (std::size_t j = start; j < start + static_cast<std::size_t>(period); ++j) {
            const double d = prices[j] - c;
            Sx += d;
            Sxx += d * d;
        }
        since_refresh = 0;
    };

    auto write_bands = [&](int i) {
        const double mean = c + Sx / W;
        const double var  = (Sxx - Sx * Sx / W) / dof;
        const double std  = (var > 0.0) ? std::sqrt(var) : 0.0;
        const double bw   = num_std * std;
        const int    o    = i * 3;
        out[o]     = mean + bw;  // upper
        out[o + 1] = mean;       // middle
        out[o + 2] = mean - bw;  // lower
    };

    // Seed first window
    recompute_window(0);
    write_bands(period - 1);

    // Slide
    for (int i = period; i < static_cast<int>(n); ++i) {
        const int old = i - period;
        Sx  += (prices[i] - c) - (prices[old] - c);
        Sxx += (prices[i] - c) * (prices[i] - c) - (prices[old] - c) * (prices[old] - c);
        ++since_refresh;

        if (since_refresh >= static_cast<std::size_t>(period)) {
            recompute_window(static_cast<std::size_t>(old) + 1);
        }

        write_bands(i);
    }
}

std::vector<double> bollinger_bands(
    const double* prices,
    std::size_t   n,
    int           period,
    double        num_std)
{
    std::vector<double> result(3 * n);
    bollinger_bands_into(prices, n, period, num_std, result.data());
    return result;
}


// ── Stochastic Oscillator ─────────────────────────────────────────────────────

void stochastic_oscillator_into(
    const double* SQT_RESTRICT high,
    const double* SQT_RESTRICT low,
    const double* SQT_RESTRICT close,
    std::size_t   n,
    int           k_period,
    int           d_period,
    double* SQT_RESTRICT       out)
{
    std::fill(out, out + 2 * n, kNaN);
    if (static_cast<int>(n) < k_period || k_period < 1) return;

    // d_period <= 0 would make `Sk -= K_vals[i - d_period + 1]` below read
    // out of bounds (i - d_period + 1 >= i + 1 for d_period <= 0) and
    // `Sk / d_period` divide by zero for d_period == 0 — reject up front,
    // same convention as the k_period guard just above.
    if (d_period <= 0) return;

    // Compute %K for each bar from k_period-1 onward.
    //
    // Two monotonic deques of indices give O(1)-amortized sliding
    // max(high)/min(low) per bar -- O(n) total instead of the previous
    // O(n*k_period) full-window rescan on every single bar. Standard
    // sliding-window-extrema technique: max_dq stays high[]-decreasing
    // front-to-back (so its front is always the window max), min_dq stays
    // low[]-increasing (front is always the window min); an index is
    // popped from the back whenever a newer bar makes it permanently
    // irrelevant (that newer bar is both in the window at least as long
    // and at least as extreme), and popped from the front once it slides
    // out of the [i-k_period+1, i] window.
    std::vector<double> K_vals(n, kNaN);
    std::deque<int> max_dq;  // indices, high[] decreasing front-to-back
    std::deque<int> min_dq;  // indices, low[]  increasing front-to-back

    for (int i = 0; i < static_cast<int>(n); ++i) {
        const int window_start = i - k_period + 1;
        while (!max_dq.empty() && max_dq.front() < window_start) max_dq.pop_front();
        while (!min_dq.empty() && min_dq.front() < window_start) min_dq.pop_front();

        while (!max_dq.empty() && high[max_dq.back()] <= high[i]) max_dq.pop_back();
        max_dq.push_back(i);

        while (!min_dq.empty() && low[min_dq.back()] >= low[i]) min_dq.pop_back();
        min_dq.push_back(i);

        if (i >= k_period - 1) {
            const double hi  = high[max_dq.front()];
            const double lo  = low[min_dq.front()];
            const double rng = hi - lo;
            K_vals[i] = (rng > 0.0) ? 100.0 * (close[i] - lo) / rng : 0.0;
        }
    }

    // Compute %D = SMA(%K, d_period) and store both to out
    double Sk = 0.0;
    int    count = 0;
    for (int i = k_period - 1; i < static_cast<int>(n); ++i) {
        out[i * 2] = K_vals[i];  // %K

        Sk += K_vals[i];
        ++count;
        if (count >= d_period) {
            out[i * 2 + 1] = Sk / d_period;  // %D
            Sk -= K_vals[i - d_period + 1];
        }
    }
}

std::vector<double> stochastic_oscillator(
    const double* high,
    const double* low,
    const double* close,
    std::size_t   n,
    int           k_period,
    int           d_period)
{
    std::vector<double> result(2 * n);
    stochastic_oscillator_into(high, low, close, n, k_period, d_period, result.data());
    return result;
}


// ── Fused technical indicators ─────────────────────────────────────────────

TechnicalIndicatorsResult technical_indicators(
    const double* high,
    const double* low,
    const double* close,
    std::size_t   n,
    const TechnicalIndicatorsConfig& config)
{
    TechnicalIndicatorsResult r;

    if (config.compute_rsi) {
        r.rsi.resize(n);
        rsi_into(close, n, config.rsi_period, r.rsi.data());
    }
    if (config.compute_adx) {
        r.adx.resize(3 * n);
        adx_into(high, low, close, n, config.adx_period, r.adx.data());
    }
    if (config.compute_atr) {
        r.atr.resize(n);
        wilder_atr_into(high, low, close, n, config.atr_period, r.atr.data());
    }
    if (config.compute_bollinger) {
        r.bollinger.resize(3 * n);
        bollinger_bands_into(close, n, config.bollinger_period,
                              config.bollinger_num_std, r.bollinger.data());
    }
    if (config.compute_stochastic) {
        r.stochastic.resize(2 * n);
        stochastic_oscillator_into(high, low, close, n, config.stoch_k_period,
                                    config.stoch_d_period, r.stochastic.data());
    }

    return r;
}

}  // namespace sqt
