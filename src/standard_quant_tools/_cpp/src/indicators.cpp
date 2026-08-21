#include "sqt/indicators.hpp"

#include "sqt/numerics.hpp"

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
    if (n <= static_cast<std::size_t>(period)) return;

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
    if (n < 2 || n <= static_cast<std::size_t>(period)) return;

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
    // Computed in size_t space -- 2*period could overflow int for an
    // extreme (if unrealistic) period near INT_MAX/2.
    const std::size_t adx_start = static_cast<std::size_t>(period) * 2 - 1;

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

        if (i <= static_cast<std::size_t>(period)) {
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

        if (i < static_cast<std::size_t>(period)) continue;  // DI/DX undefined before bar `period`

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
    if (n < static_cast<std::size_t>(period)) return;

    // ── True range, computed inline (no O(n) tr[] buffer) ─────────────────────
    // Every TR[i] depends only on high[i]/low[i]/close[i-1] (and high[0]/
    // low[0] for bar 0) -- no lookback beyond the immediately-preceding
    // close -- so it's computable on demand in both the seed and forward-
    // smoothing loops below with O(1) auxiliary memory, exactly like
    // adx_into's existing fused single-pass design above (see that
    // function's own comment for the same technique applied to DM/TR).
    auto tr_at = [&](std::size_t i) {
        return (i == 0)
            ? (high[0] - low[0])
            : std::max({
                  high[i] - low[i],
                  std::abs(high[i] - close[i - 1]),
                  std::abs(low[i]  - close[i - 1]),
              });
    };

    // ── Seed: SMA of first `period` TR values ─────────────────────────────────
    double atr_val = 0.0;
    for (int i = 0; i < period; ++i) atr_val += tr_at(static_cast<std::size_t>(i));
    atr_val /= period;
    out[static_cast<std::size_t>(period) - 1] = atr_val;

    // ── Wilder's forward smoothing ────────────────────────────────────────────
    for (std::size_t i = static_cast<std::size_t>(period); i < n; ++i) {
        atr_val = (atr_val * (period - 1) + tr_at(i)) / period;
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
    // period < 2 checked first (before period is ever cast to size_t
    // below) since a negative period would otherwise wrap to a huge
    // unsigned value in the second comparison.
    if (period < 2 || n < static_cast<std::size_t>(period)) return;

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

    // ── Non-finite bars ───────────────────────────────────────────────────
    // A NaN/Inf price is a MISSING observation, and the window it falls in
    // has no mean or standard deviation -- pandas' rolling(min_periods=
    // period) reports NaN for exactly those windows, and so does this
    // kernel (`nan_in_window`).
    //
    // Inf is folded in with NaN here, unlike stochastic_oscillator_into
    // below, which lets it flow through. That is not an inconsistency: an
    // Inf entering these sums is unrecoverable, because the sliding update
    // subtracts it back out as inf - inf = NaN, so an Inf bar would corrupt
    // every later window exactly the way a NaN does. The deques in the
    // stochastic kernel have no such accumulator and compare against Inf
    // correctly, so there is nothing to protect there. pandas reports an
    // Inf mean and a NaN standard deviation for such a window; this reports
    // NaN for all three bands, which is the same information without a
    // middle band a caller could plot.
    //
    // The second variable is the one that is easy to miss. An O(1) sliding
    // sum cannot un-add a NaN: `Sx += (prices[i]-c) - (prices[old]-c)`
    // leaves Sx as NaN forever once a NaN has entered it, because
    // NaN - NaN is NaN, not 0. So the sums stayed poisoned for up to
    // `period` further bars after the bad bar had already left the window
    // -- until the periodic refresh happened to fire -- and the kernel
    // reported NaN for a stretch of windows containing no bad data at all.
    // `sums_polluted` records that a non-finite value was ever ADDED to the
    // running sums, so the moment the window is clean again the sums are
    // rebuilt from scratch rather than waiting on the refresh cadence.
    std::size_t nan_in_window = 0;
    bool        sums_polluted = false;

    auto recompute_window = [&](std::size_t start) {
        Sx = 0.0;
        Sxx = 0.0;
        nan_in_window = 0;
        for (std::size_t j = start; j < start + static_cast<std::size_t>(period); ++j) {
            if (!std::isfinite(prices[j])) ++nan_in_window;
        }
        // The reference point must itself be finite or it poisons every
        // shifted value in the window. prices[start] is the natural choice
        // (it keeps the shifted values near the window's own variation);
        // fall back to 0.0 only when the window is unevaluable anyway.
        c = std::isfinite(prices[start]) ? prices[start] : 0.0;
        for (std::size_t j = start; j < start + static_cast<std::size_t>(period); ++j) {
            const double d = prices[j] - c;
            Sx += d;
            Sxx += d * d;
        }
        since_refresh = 0;
        sums_polluted = (nan_in_window > 0);
    };

    auto write_bands = [&](std::size_t i) {
        const std::size_t o = i * 3;
        // A window holding a missing observation has no bands to report.
        // Returning NaN here (rather than letting the arithmetic below
        // produce it) also keeps the sums' own pollution state from
        // leaking into the output for windows that are already clean.
        if (nan_in_window > 0) {
            out[o]     = kNaN;
            out[o + 1] = kNaN;
            out[o + 2] = kNaN;
            return;
        }
        const double mean = c + Sx / W;
        const double raw_var = (Sxx - Sx * Sx / W) / dof;
        // A variance is a sum of squares over dof -- mathematically >= 0, but
        // it can drift slightly negative under cancellation. The previous
        // `(var > 0.0) ? sqrt(var) : 0.0` clamped ANY negative value to zero,
        // collapsing the bands onto the moving average with no signal -- the
        // exact silent failure the shift-by-reference-point centering above
        // was introduced to prevent, left undetectable if it ever recurred.
        // clamp_near_zero_sumsq clamps genuine noise and throws on anything
        // larger, so a real regression surfaces instead of hiding. (It
        // passes NaN/Inf straight through rather than throwing -- but this
        // lambda has already returned above for any window containing one,
        // so that path is unreachable from here.)
        const double var = numerics::clamp_near_zero_sumsq(
            raw_var, Sxx / dof, "indicators::bollinger_bands");
        const double std  = (var > 0.0) ? std::sqrt(var) : 0.0;
        const double bw   = num_std * std;
        out[o]     = mean + bw;  // upper
        out[o + 1] = mean;       // middle
        out[o + 2] = mean - bw;  // lower
    };

    // Seed first window
    recompute_window(0);
    write_bands(static_cast<std::size_t>(period) - 1);

    // Slide. size_t throughout (not int): i-period could exceed INT_MAX
    // for a large series, silently wrapping under int arithmetic.
    const std::size_t period_sz = static_cast<std::size_t>(period);
    for (std::size_t i = period_sz; i < n; ++i) {
        const std::size_t old = i - period_sz;
        if (!std::isfinite(prices[old])) --nan_in_window;
        if (!std::isfinite(prices[i])) { ++nan_in_window; sums_polluted = true; }
        Sx  += (prices[i] - c) - (prices[old] - c);
        Sxx += (prices[i] - c) * (prices[i] - c) - (prices[old] - c) * (prices[old] - c);
        ++since_refresh;

        // Second condition: the window just became clean but the sums still
        // carry a NaN/Inf that no subtraction can remove -- rebuild now
        // instead of reporting NaN until the refresh cadence catches up.
        if (since_refresh >= period_sz || (nan_in_window == 0 && sums_polluted)) {
            recompute_window(old + 1);
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
    // k_period < 1 checked first (before it's cast to size_t) since a
    // negative k_period would otherwise wrap to a huge unsigned value.
    if (k_period < 1 || n < static_cast<std::size_t>(k_period)) return;

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
    // Sliding count of bars whose high or low is a missing observation.
    //
    // A NaN can never be a window extreme, and pushing its index into a
    // monotonic deque destroys the deque's whole invariant: every
    // comparison against NaN is false, so `high[max_dq.back()] <= high[i]`
    // never pops it, and it then blocks the eviction of every stale index
    // sitting behind it. The front stops being the window maximum. Measured
    // on a rising series with one NaN high and k_period=5, %K -- bounded
    // 0..100 by construction -- returned 125, 166.67 and 250 from a stale
    // max, then a fabricated 0.0 once the NaN reached the front. That is
    // strictly worse than an exception: it is a confident wrong number in
    // an indicator whose range is its meaning.
    //
    // So NaN indices are never pushed (below), and this counter is what
    // reports the window as unevaluable -- matching pandas'
    // rolling(min_periods=k), where a window containing a missing
    // observation yields NaN rather than a number computed from whichever
    // observations happen to be present. +/-Inf is a real observation, not
    // a missing one, and is left to flow through the arithmetic as pandas
    // does.
    long long nan_in_window = 0;
    // long long (not int) indices/deques: window_start and the loop bound
    // below are derived from n, which can exceed INT_MAX for a large
    // series -- matching the signed-induction-variable precedent already
    // established in backtest.cpp's batch_run_strategy. Signed (not
    // size_t) because window_start = i - k_period + 1 is negative for the
    // first few bars, same as the original int version.
    std::deque<long long> max_dq;  // indices, high[] decreasing front-to-back
    std::deque<long long> min_dq;  // indices, low[]  increasing front-to-back

    const long long n_ll       = static_cast<long long>(n);
    const long long k_period_ll = k_period;

    for (long long i = 0; i < n_ll; ++i) {
        const std::size_t i_sz = static_cast<std::size_t>(i);
        const long long window_start = i - k_period_ll + 1;

        // Slide the missing-observation count: drop the bar that just left
        // the window, then add the bar that just entered it.
        if (window_start >= 1) {
            const std::size_t leaving = static_cast<std::size_t>(window_start - 1);
            if (std::isnan(high[leaving]) || std::isnan(low[leaving])) --nan_in_window;
        }
        const bool high_missing = std::isnan(high[i_sz]);
        const bool low_missing  = std::isnan(low[i_sz]);
        if (high_missing || low_missing) ++nan_in_window;

        while (!max_dq.empty() && max_dq.front() < window_start) max_dq.pop_front();
        while (!min_dq.empty() && min_dq.front() < window_start) min_dq.pop_front();

        // Each deque is gated on ITS OWN series, not on the bar as a whole:
        // a bar with a NaN high but a valid low is still a candidate for
        // the window minimum, and skipping it in min_dq would lose a real
        // extreme once the NaN high aged out of the window.
        if (!high_missing) {
            while (!max_dq.empty() &&
                   high[static_cast<std::size_t>(max_dq.back())] <= high[i_sz]) max_dq.pop_back();
            max_dq.push_back(i);
        }
        if (!low_missing) {
            while (!min_dq.empty() &&
                   low[static_cast<std::size_t>(min_dq.back())] >= low[i_sz]) min_dq.pop_back();
            min_dq.push_back(i);
        }

        if (i >= k_period_ll - 1 && nan_in_window == 0) {
            // nan_in_window == 0 guarantees both deques are non-empty here:
            // every bar of the window was pushed into each of them.
            const double hi  = high[static_cast<std::size_t>(max_dq.front())];
            const double lo  = low[static_cast<std::size_t>(min_dq.front())];
            const double rng = hi - lo;
            K_vals[i_sz] = (rng > 0.0) ? 100.0 * (close[i_sz] - lo) / rng : 0.0;
        }
    }

    // Compute %D = SMA(%K, d_period) and store both to out.
    //
    // `nan_k` is the same problem the sliding sums in bollinger_bands_into
    // have, in a second place: a NaN %K added to the running sum can never
    // be subtracted back out (NaN - NaN is NaN), so one unevaluable window
    // used to poison every %D for the rest of the series. Missing values
    // are counted instead of summed, which both keeps Sk finite and gives
    // %D the same rolling(min_periods=d_period) semantics %K now has. For
    // an all-finite series the adds and subtracts happen in the identical
    // order as before, so this is bit-for-bit unchanged on clean data.
    double    Sk     = 0.0;
    long long count  = 0;
    long long nan_k  = 0;
    const long long d_period_ll = d_period;
    for (long long i = k_period_ll - 1; i < n_ll; ++i) {
        const std::size_t i_sz = static_cast<std::size_t>(i);
        const double k_in = K_vals[i_sz];
        out[i_sz * 2] = k_in;  // %K

        if (std::isnan(k_in)) ++nan_k; else Sk += k_in;
        ++count;
        if (count >= d_period_ll) {
            out[i_sz * 2 + 1] = (nan_k == 0) ? Sk / d_period : kNaN;  // %D
            const double k_out = K_vals[static_cast<std::size_t>(i - d_period_ll + 1)];
            if (std::isnan(k_out)) --nan_k; else Sk -= k_out;
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
