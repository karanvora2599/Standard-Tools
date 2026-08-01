#pragma once

#include <cstddef>
#include <vector>

namespace sqt {

// Every indicator below has a buffer-writing `*_into` variant alongside its
// vector-returning form. The `_into` variants write directly into a
// caller-provided buffer -- used by bindings.cpp to write straight into a
// pre-allocated NumPy array with no intermediate std::vector allocation and
// no copy at the Python/C++ boundary. Native tests and internal callers
// continue to use the vector-returning forms unchanged (each is a thin
// wrapper: allocate, call the `_into` variant, return).

/**
 * RSI — Relative Strength Index (Wilder's smoothing).
 *
 * @param prices  Contiguous close-price array.
 * @param n       Number of elements.
 * @param period  Lookback period (default 14).
 * @returns       Vector of length n; first `period` values are NaN.
 */
std::vector<double> rsi(const double* prices, std::size_t n, int period = 14);

/** Buffer-writing form of rsi(). `out` must have length n. */
void rsi_into(const double* prices, std::size_t n, int period, double* out);

/**
 * ADX — Average Directional Index with DI+ and DI-.
 *
 * Uses Wilder's smoothing identical to the Python/Numba reference.
 * Returns a flat row-major array of length 3*n:
 *   [DI+_0, DI-_0, ADX_0, DI+_1, DI-_1, ADX_1, ...].
 * First `period` rows have NaN in DI+/DI-; ADX starts at row 2*period-1.
 *
 * @param high    Contiguous high-price array (length n).
 * @param low     Contiguous low-price array (length n).
 * @param close   Contiguous close-price array (length n).
 * @param n       Number of bars.
 * @param period  Wilder smoothing period (default 14).
 */
std::vector<double> adx(
    const double* high,
    const double* low,
    const double* close,
    std::size_t   n,
    int           period = 14);

/** Buffer-writing form of adx(). `out` must have length 3*n. */
void adx_into(
    const double* high,
    const double* low,
    const double* close,
    std::size_t   n,
    int           period,
    double*       out);

/**
 * Parabolic SAR — trend-following stop-and-reverse indicator.
 *
 * Returns a flat row-major array of length 2*n:
 *   [SAR_0, Trend_0, SAR_1, Trend_1, ...].
 * Trend: 1.0 = rising (long), -1.0 = falling (short).
 * Bar 0 is bootstrapped: SAR = low[0], EP = high[0], rising.
 *
 * @param high      Contiguous high-price array (length n).
 * @param low       Contiguous low-price array (length n).
 * @param n         Number of bars.
 * @param af_start  Initial acceleration factor (default 0.02).
 * @param af_step   Increment per new extreme point (default 0.02).
 * @param af_max    Maximum acceleration factor (default 0.2).
 *
 * Returns all-NaN if af_start/af_step/af_max are non-finite, af_start <= 0,
 * af_step < 0, af_max <= 0, or af_max < af_start.
 */
std::vector<double> parabolic_sar(
    const double* high,
    const double* low,
    std::size_t   n,
    double        af_start = 0.02,
    double        af_step  = 0.02,
    double        af_max   = 0.2);

/** Buffer-writing form of parabolic_sar(). `out` must have length 2*n. */
void parabolic_sar_into(
    const double* high,
    const double* low,
    std::size_t   n,
    double        af_start,
    double        af_step,
    double        af_max,
    double*       out);

/**
 * Wilder's ATR — Average True Range with Wilder's smoothing.
 *
 * True range:
 *   TR[0] = high[0] - low[0]  (no prior close)
 *   TR[i] = max(H[i]-L[i], |H[i]-C[i-1]|, |L[i]-C[i-1]|)  for i >= 1
 *
 * Seed: ATR[period-1] = mean(TR[0..period-1])
 * Forward: ATR[i] = (ATR[i-1] * (period-1) + TR[i]) / period
 *
 * This is identical to the Wilder smoothing used in RSI and ADX —
 * not the simple rolling-mean ATR produced by pandas .rolling().mean().
 *
 * @param high    Contiguous high-price array (length n).
 * @param low     Contiguous low-price array (length n).
 * @param close   Contiguous close-price array (length n).
 * @param n       Number of bars.
 * @param period  Smoothing period (default 14).
 * @returns       Vector of length n; first period-1 values are NaN.
 */
std::vector<double> wilder_atr(
    const double* high,
    const double* low,
    const double* close,
    std::size_t   n,
    int           period = 14);

/** Buffer-writing form of wilder_atr(). `out` must have length n. */
void wilder_atr_into(
    const double* high,
    const double* low,
    const double* close,
    std::size_t   n,
    int           period,
    double*       out);

/**
 * Bollinger Bands — fused sliding-window mean + std in one pass.
 *
 * Uses incremental sum and sum-of-squares to compute mean and sample std
 * without re-iterating the window.  Equivalent to:
 *   middle = SMA(prices, period)
 *   std    = rolling std (ddof=1) of prices over period
 *   upper  = middle + num_std * std
 *   lower  = middle - num_std * std
 *
 * Returns a flat row-major array of length 3*n:
 *   [upper_0, middle_0, lower_0, upper_1, middle_1, lower_1, ...].
 * First (period-1) triples are NaN.
 *
 * @param prices   Contiguous close-price array (length n).
 * @param n        Number of bars.
 * @param period   Lookback period (default 20).
 * @param num_std  Band width in standard deviations (default 2.0).
 */
std::vector<double> bollinger_bands(
    const double* prices,
    std::size_t   n,
    int           period  = 20,
    double        num_std = 2.0);

/** Buffer-writing form of bollinger_bands(). `out` must have length 3*n. */
void bollinger_bands_into(
    const double* prices,
    std::size_t   n,
    int           period,
    double        num_std,
    double*       out);

/**
 * Stochastic Oscillator — O(n) sliding min/max via monotonic deques (not a
 * per-bar window rescan).
 *
 * %K = 100 * (close - lowest_low) / (highest_high - lowest_low)
 * %D = SMA(%K, d_period)
 *
 * Returns a flat row-major array of length 2*n:
 *   [K_0, D_0, K_1, D_1, ...].
 * First (k_period-1) K values are NaN; first (k_period + d_period - 2)
 * D values are NaN.
 *
 * @param high      Contiguous high-price array (length n).
 * @param low       Contiguous low-price array (length n).
 * @param close     Contiguous close-price array (length n).
 * @param n         Number of bars.
 * @param k_period  Fast period (default 14).
 * @param d_period  Slow period (SMA of K, default 3).
 */
std::vector<double> stochastic_oscillator(
    const double* high,
    const double* low,
    const double* close,
    std::size_t   n,
    int           k_period = 14,
    int           d_period = 3);

/** Buffer-writing form of stochastic_oscillator(). `out` must have length 2*n. */
void stochastic_oscillator_into(
    const double* high,
    const double* low,
    const double* close,
    std::size_t   n,
    int           k_period,
    int           d_period,
    double*       out);

}  // namespace sqt
