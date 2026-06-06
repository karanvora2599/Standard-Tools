#pragma once

#include <cstddef>
#include <vector>

namespace sqt {

/**
 * RSI — Relative Strength Index (Wilder's smoothing).
 *
 * @param prices  Contiguous close-price array.
 * @param n       Number of elements.
 * @param period  Lookback period (default 14).
 * @returns       Vector of length n; first `period` values are NaN.
 */
std::vector<double> rsi(const double* prices, std::size_t n, int period = 14);

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
 */
std::vector<double> parabolic_sar(
    const double* high,
    const double* low,
    std::size_t   n,
    double        af_start = 0.02,
    double        af_step  = 0.02,
    double        af_max   = 0.2);

}  // namespace sqt
