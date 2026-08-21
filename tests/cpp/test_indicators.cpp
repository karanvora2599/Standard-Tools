/**
 * C++ unit tests for sqt::rsi, sqt::adx, sqt::parabolic_sar,
 * sqt::wilder_atr, sqt::bollinger_bands, sqt::stochastic_oscillator and the
 * fused sqt::technical_indicators.
 *
 * Build:
 *   cmake -B build -DSQT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build --config Release
 *
 * Run directly:
 *   Windows : build\tests\cpp\Release\test_indicators.exe
 *   Linux   : ./build/tests/cpp/test_indicators
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V
 */

#include "sqt/indicators.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
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

#define CHECK_NEAR(a, b, tol)  CHECK(std::abs((a) - (b)) <= (tol))
#define CHECK_NAN(val)         CHECK(std::isnan(val))
#define CHECK_NOT_NAN(val)     CHECK(!std::isnan(val))
#define CHECK_EQ(a, b)         CHECK((a) == (b))


// ── Synthetic data helpers ────────────────────────────────────────────────────

// Monotonically increasing prices: 100, 101, 102, ...
static std::vector<double> rising_prices(int n, double start = 100.0) {
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) out[i] = start + i;
    return out;
}

// Monotonically decreasing prices: 200, 199, 198, ...
static std::vector<double> falling_prices(int n, double start = 200.0) {
    std::vector<double> out(n);
    for (int i = 0; i < n; ++i) out[i] = start - i;
    return out;
}

// Simple LCG, seed-deterministic pseudo-random in [0, 1)
static std::vector<double> pseudo_random(int n, unsigned seed = 42) {
    std::vector<double> out(n);
    unsigned state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        out[i] = (static_cast<double>(state & 0x7FFFFFFFu) / 2147483648.0) * 10.0 + 95.0;
    }
    return out;
}

// Synthetic OHLC from a price series: high = close + 0.5, low = close - 0.5
static void ohlc_from_prices(
    const std::vector<double>& close,
    std::vector<double>& high,
    std::vector<double>& low)
{
    const int n = static_cast<int>(close.size());
    high.resize(n);
    low.resize(n);
    for (int i = 0; i < n; ++i) {
        high[i] = close[i] + 0.5;
        low[i]  = close[i] - 0.5;
    }
}


// ── RSI tests ─────────────────────────────────────────────────────────────────

static void test_rsi_nan_prefix() {
    // First `period` values must be NaN
    const int period = 14;
    auto prices = rising_prices(30);
    auto result = sqt::rsi(prices.data(), prices.size(), period);

    CHECK_EQ(static_cast<int>(result.size()), 30);
    for (int i = 0; i < period; ++i) CHECK_NAN(result[i]);
    for (int i = period; i < 30; ++i) CHECK_NOT_NAN(result[i]);
}

static void test_rsi_all_rising_equals_100() {
    // Monotonically rising prices: no losses → RSI = 100.0
    const int period = 5;
    auto prices = rising_prices(20);
    auto result = sqt::rsi(prices.data(), prices.size(), period);

    for (std::size_t i = static_cast<std::size_t>(period); i < result.size(); ++i) {
        CHECK_NEAR(result[i], 100.0, 1e-9);
    }
}

static void test_rsi_all_falling_equals_0() {
    // Monotonically falling prices: no gains → RSI = 0.0
    const int period = 5;
    auto prices = falling_prices(20);
    auto result = sqt::rsi(prices.data(), prices.size(), period);

    for (std::size_t i = static_cast<std::size_t>(period); i < result.size(); ++i) {
        CHECK_NEAR(result[i], 0.0, 1e-9);
    }
}

static void test_rsi_known_value() {
    // prices = {10, 11, 12, 11}, period = 3
    // deltas = {1, 1, -1}
    // avg_gain = (1+1+0)/3 = 2/3, avg_loss = (0+0+1)/3 = 1/3
    // rsi[3] = 100 - 100 / (1 + 2) = 200/3 ≈ 66.6667
    const double prices[] = {10.0, 11.0, 12.0, 11.0};
    auto result = sqt::rsi(prices, 4, 3);

    CHECK_NAN(result[0]);
    CHECK_NAN(result[1]);
    CHECK_NAN(result[2]);
    CHECK_NEAR(result[3], 200.0 / 3.0, 1e-9);
}

static void test_rsi_bounds() {
    // RSI must always be in [0, 100]
    auto prices = pseudo_random(200);
    auto result = sqt::rsi(prices.data(), prices.size(), 14);

    for (auto v : result) {
        if (!std::isnan(v)) {
            CHECK(v >= 0.0);
            CHECK(v <= 100.0);
        }
    }
}

static void test_rsi_short_series() {
    // n <= period → all NaN
    auto prices = rising_prices(5);
    auto r1 = sqt::rsi(prices.data(), 5, 5);  // n == period
    auto r2 = sqt::rsi(prices.data(), 3, 5);  // n < period
    for (auto v : r1) CHECK_NAN(v);
    for (auto v : r2) CHECK_NAN(v);
}

static void test_rsi_empty() {
    auto result = sqt::rsi(nullptr, 0, 14);
    CHECK_EQ(static_cast<int>(result.size()), 0);
}


// ── ADX tests ────────────────────────────────────────────────────────────────

static void test_adx_nan_prefix() {
    // DI+/DI- start at row `period`; ADX starts at row 2*period-1
    const int period = 5;
    auto close = rising_prices(40);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);

    auto result = sqt::adx(high.data(), low.data(), close.data(), 40, period);

    // Rows 0..period-1: DI+ and DI- are NaN
    for (int i = 0; i < period; ++i) {
        CHECK_NAN(result[i * 3 + 0]);
        CHECK_NAN(result[i * 3 + 1]);
    }
    // ADX is NaN before row 2*period-1
    for (int i = 0; i < 2 * period - 1; ++i) {
        CHECK_NAN(result[i * 3 + 2]);
    }
    // DI+ and DI- valid from row `period`
    for (int i = period; i < 40; ++i) {
        CHECK_NOT_NAN(result[i * 3 + 0]);
        CHECK_NOT_NAN(result[i * 3 + 1]);
    }
    // ADX valid from row 2*period-1
    for (int i = 2 * period - 1; i < 40; ++i) {
        CHECK_NOT_NAN(result[i * 3 + 2]);
    }
}

static void test_adx_uptrend_di_plus_dominates() {
    // For a strong up-trend, DI+ > DI- in stable region
    const int period = 5;
    auto close = rising_prices(60, 100.0);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);

    auto result = sqt::adx(high.data(), low.data(), close.data(), 60, period);

    // Check a few bars well past the burn-in period
    for (int i = 2 * period; i < 60; ++i) {
        CHECK(result[i * 3 + 0] > result[i * 3 + 1]);  // DI+ > DI-
    }
}

static void test_adx_bounds() {
    // All valid DI+, DI-, ADX values should be in [0, 100]
    const int period = 7;
    auto close = pseudo_random(150);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);

    auto result = sqt::adx(high.data(), low.data(), close.data(), 150, period);

    for (int i = 0; i < 150; ++i) {
        for (int col = 0; col < 3; ++col) {
            double v = result[i * 3 + col];
            if (!std::isnan(v)) {
                CHECK(v >= 0.0);
                CHECK(v <= 100.0);
        }
        }
    }
}

static void test_adx_short_series() {
    // n <= period → all NaN
    auto close = rising_prices(5);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);

    auto result = sqt::adx(high.data(), low.data(), close.data(), 5, 5);
    for (auto v : result) CHECK_NAN(v);
}

static void test_adx_exact_regression_pin() {
    // Regression pin (Tier 4 item 14 / performance architecture item 4):
    // adx() was rewritten from a 4-array (dm_plus/dm_minus/tr/dx_vals)
    // implementation to a single fused O(1)-auxiliary-memory pass. The
    // rewrite preserves the exact same sequence of floating-point
    // operations in the exact same order (floating-point addition isn't
    // associative, so *order* matters, not just which values get summed),
    // so this must match bit-identically -- these expected values were
    // captured from the verified-correct implementation (cross-checked
    // against test_adx_uptrend_di_plus_dominates/test_adx_bounds/etc.,
    // all passing unchanged) and are pinned here as a permanent guard
    // against any future silent behavior change, intentional or not.
    const int period = 14;
    auto close = pseudo_random(200);  // seed=42 default
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);

    auto result = sqt::adx(high.data(), low.data(), close.data(), 200, period);

    const double tol = 1e-12;
    CHECK_NEAR(result[14 * 3 + 0], 42.45068784543159, tol);
    CHECK_NEAR(result[14 * 3 + 1], 46.9763826288027, tol);
    CHECK_NAN(result[14 * 3 + 2]);
    CHECK_NEAR(result[27 * 3 + 0], 45.11415518667713, tol);
    CHECK_NEAR(result[27 * 3 + 1], 42.15672380278956, tol);
    CHECK_NEAR(result[27 * 3 + 2], 5.134074786054382, tol);
    CHECK_NEAR(result[28 * 3 + 0], 47.39663558209396, tol);
    CHECK_NEAR(result[28 * 3 + 1], 39.72096324053475, tol);
    CHECK_NEAR(result[28 * 3 + 2], 5.396691041837876, tol);
    CHECK_NEAR(result[100 * 3 + 0], 43.52124279434726, tol);
    CHECK_NEAR(result[100 * 3 + 1], 43.005228899066054, tol);
    CHECK_NEAR(result[100 * 3 + 2], 5.354983108893717, tol);
    CHECK_NEAR(result[199 * 3 + 0], 45.42796779633301, tol);
    CHECK_NEAR(result[199 * 3 + 1], 40.314428688191065, tol);
    CHECK_NEAR(result[199 * 3 + 2], 5.04677283529171, tol);
}


// ── Parabolic SAR tests ───────────────────────────────────────────────────────

static void test_psar_bootstrap() {
    // Bar 0: SAR = low[0], Trend = 1.0 (rising)
    const double high[] = {105.0, 106.0};
    const double low[]  = {100.0,  99.0};

    auto result = sqt::parabolic_sar(high, low, 2, 0.02, 0.02, 0.2);

    CHECK_EQ(static_cast<int>(result.size()), 4);
    CHECK_NEAR(result[0], 100.0, 1e-9);  // SAR = low[0]
    CHECK_NEAR(result[1], 1.0,   1e-9);  // Trend = rising
}

static void test_psar_trend_values_are_pm1() {
    // Trend column must be exactly 1.0 or -1.0
    auto close = pseudo_random(100);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);

    auto result = sqt::parabolic_sar(high.data(), low.data(), 100, 0.02, 0.02, 0.2);

    for (int i = 0; i < 100; ++i) {
        double trend = result[i * 2 + 1];
        CHECK(!std::isnan(trend));
        CHECK(trend == 1.0 || trend == -1.0);
    }
}

static void test_psar_rising_trend_sar_below_price() {
    // Strong uptrend: SAR should stay below lows throughout
    auto close = rising_prices(50, 100.0);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);

    auto result = sqt::parabolic_sar(high.data(), low.data(), 50, 0.02, 0.02, 0.2);

    for (int i = 1; i < 50; ++i) {
        double sar   = result[i * 2 + 0];
        double trend = result[i * 2 + 1];
        if (trend == 1.0) {
            CHECK(sar < low[i]);  // SAR below price in uptrend
        }
    }
}

static void test_psar_single_bar() {
    const double high[] = {101.0};
    const double low[]  = {99.0};
    auto result = sqt::parabolic_sar(high, low, 1, 0.02, 0.02, 0.2);

    CHECK_EQ(static_cast<int>(result.size()), 2);
    CHECK_NOT_NAN(result[0]);
    CHECK_NEAR(result[1], 1.0, 1e-9);
}

static void test_psar_empty() {
    auto result = sqt::parabolic_sar(nullptr, nullptr, 0, 0.02, 0.02, 0.2);
    CHECK_EQ(static_cast<int>(result.size()), 0);
}


// ── Wilder's ATR tests ────────────────────────────────────────────────────────

static void test_wilder_atr_nan_prefix() {
    // First period-1 values must be NaN; from period-1 onward must be valid.
    const int period = 5;
    const int n = 30;
    auto close = rising_prices(n);
    std::vector<double> high(n), low(n);
    for (int i = 0; i < n; ++i) { high[i] = close[i] + 0.5; low[i] = close[i] - 0.5; }

    auto result = sqt::wilder_atr(high.data(), low.data(), close.data(), n, period);

    CHECK_EQ(static_cast<int>(result.size()), n);
    for (int i = 0; i < period - 1; ++i) CHECK_NAN(result[i]);
    for (int i = period - 1; i < n; ++i) CHECK_NOT_NAN(result[i]);
}

static void test_wilder_atr_known_value() {
    // Hand-computed example, period = 2:
    //   H=[10,11,12], L=[9,9,10], C=[9.5,10,11]
    //   TR[0] = 10-9 = 1.0
    //   TR[1] = max(11-9, |11-9.5|, |9-9.5|) = max(2, 1.5, 0.5) = 2.0
    //   TR[2] = max(12-10, |12-10|, |10-10|) = max(2, 2, 0) = 2.0
    //   ATR[1] = mean(1.0, 2.0) = 1.5
    //   ATR[2] = (1.5*1 + 2.0) / 2 = 1.75
    const double high[]  = {10.0, 11.0, 12.0};
    const double low[]   = { 9.0,  9.0, 10.0};
    const double close[] = { 9.5, 10.0, 11.0};

    auto result = sqt::wilder_atr(high, low, close, 3, 2);

    CHECK_EQ(static_cast<int>(result.size()), 3);
    CHECK_NAN(result[0]);
    CHECK_NEAR(result[1], 1.5,  1e-12);
    CHECK_NEAR(result[2], 1.75, 1e-12);
}

static void test_wilder_atr_non_negative() {
    // ATR must always be >= 0 for any price series.
    const int n = 200;
    auto close = pseudo_random(n);
    std::vector<double> high(n), low(n);
    for (int i = 0; i < n; ++i) { high[i] = close[i] + 0.5; low[i] = close[i] - 0.5; }

    auto result = sqt::wilder_atr(high.data(), low.data(), close.data(), n, 14);
    for (auto v : result) {
        if (!std::isnan(v)) CHECK(v >= 0.0);
    }
}

static void test_wilder_atr_constant_prices() {
    // When H=L=C for every bar: TR=0 everywhere, so ATR should be 0.
    const int n = 30;
    std::vector<double> h(n, 100.0), l(n, 100.0), c(n, 100.0);

    auto result = sqt::wilder_atr(h.data(), l.data(), c.data(), n, 5);

    for (int i = 4; i < n; ++i) CHECK_NEAR(result[i], 0.0, 1e-12);
}

static void test_wilder_atr_smoothing_recurrence() {
    // Verify: for every i >= period, result[i] == (result[i-1]*(p-1) + TR[i]) / p
    const int period = 7;
    const int n = 50;
    auto close = pseudo_random(n, 17);
    std::vector<double> high(n), low(n);
    for (int i = 0; i < n; ++i) { high[i] = close[i] + 1.0; low[i] = close[i] - 1.0; }

    auto result = sqt::wilder_atr(high.data(), low.data(), close.data(), n, period);

    for (int i = period; i < n; ++i) {
        // Recompute TR[i]
        double tr_i = std::max({
            high[i] - low[i],
            std::abs(high[i] - close[i - 1]),
            std::abs(low[i]  - close[i - 1]),
        });
        double expected = (result[i - 1] * (period - 1) + tr_i) / period;
        CHECK_NEAR(result[i], expected, 1e-10);
    }
}

static void test_wilder_atr_matches_unfused_array_reference_exactly() {
    // Regression pin (correctness/portability pass item 16): wilder_atr_into
    // was rewritten from a std::vector<double> tr(n) precomputed-array
    // implementation to an O(1)-auxiliary-memory fused pass (mirroring
    // adx_into's existing precedent) -- a pure refactor with the exact same
    // arithmetic in the exact same order, not a reassociation, so this must
    // match an independent array-based reference implementation
    // bit-for-bit, not just within a tolerance.
    const int period = 14;
    const int n = 200;
    auto close = pseudo_random(n, 55);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);

    auto fused = sqt::wilder_atr(high.data(), low.data(), close.data(), n, period);

    // Independent reference: precompute tr[] as its own array first (the
    // pre-fusion approach), then run the identical seed/smoothing loops.
    std::vector<double> tr(n);
    tr[0] = high[0] - low[0];
    for (int i = 1; i < n; ++i) {
        tr[i] = std::max({
            high[i] - low[i],
            std::abs(high[i] - close[i - 1]),
            std::abs(low[i]  - close[i - 1]),
        });
    }
    std::vector<double> reference(n, std::numeric_limits<double>::quiet_NaN());
    double atr_val = 0.0;
    for (int i = 0; i < period; ++i) atr_val += tr[i];
    atr_val /= period;
    reference[period - 1] = atr_val;
    for (int i = period; i < n; ++i) {
        atr_val = (atr_val * (period - 1) + tr[i]) / period;
        reference[i] = atr_val;
    }

    for (int i = 0; i < n; ++i) {
        if (std::isnan(reference[i])) {
            CHECK_NAN(fused[i]);
        } else {
            CHECK(fused[i] == reference[i]);  // exact, not CHECK_NEAR
        }
    }
}

static void test_wilder_atr_short_series() {
    // n < period → all NaN; n == period → first valid value at index period-1
    const double h[] = {10.0, 11.0, 12.0};
    const double l[] = { 9.0,  9.5, 10.0};
    const double c[] = { 9.5, 10.0, 11.0};

    // n=3 < period=5 → all NaN
    auto r1 = sqt::wilder_atr(h, l, c, 3, 5);
    for (auto v : r1) CHECK_NAN(v);

    // n=3 == period=3 → exactly one valid value at index 2
    auto r2 = sqt::wilder_atr(h, l, c, 3, 3);
    CHECK_NAN(r2[0]);
    CHECK_NAN(r2[1]);
    CHECK_NOT_NAN(r2[2]);
}

static void test_wilder_atr_empty() {
    auto result = sqt::wilder_atr(nullptr, nullptr, nullptr, 0, 14);
    CHECK_EQ(static_cast<int>(result.size()), 0);
}

static void test_wilder_atr_decays_toward_tr() {
    // If TR becomes constant after the seed period, ATR should converge to that value.
    // We set the first `period` bars to have TR=2, then all remaining to TR=0.
    // ATR should decay toward 0 monotonically after the seed.
    const int period = 5;
    const int n = 100;
    // High - low = 2 for first period bars, then 0 thereafter
    std::vector<double> h(n), l(n, 0.0), c(n, 0.0);
    for (int i = 0; i < period; ++i)  { h[i] = 2.0; }
    for (int i = period; i < n; ++i)  { h[i] = 0.0; }
    // No prev-close gap since all closes = 0

    auto result = sqt::wilder_atr(h.data(), l.data(), c.data(), n, period);

    // ATR at seed: mean of first `period` TR values = 2.0 (or something positive)
    CHECK_NOT_NAN(result[period - 1]);
    CHECK(result[period - 1] > 0.0);

    // From period onward, TR = 0, so ATR should decrease each bar
    for (int i = period; i < n - 1; ++i) {
        CHECK(result[i + 1] <= result[i] + 1e-12);  // non-increasing
    }
    // After many steps, ATR must be close to zero
    CHECK(result[n - 1] < 0.01);
}


// ── Stochastic Oscillator tests ─────────────────────────────────────────────────

// Independent brute-force reference (the O(n*k_period) full-window rescan
// stochastic_oscillator itself used before the monotonic-deque rewrite) --
// deliberately NOT sharing any code with the real implementation, so this
// only agrees with it if the O(n) rewrite is actually correct, not just
// self-consistent.
static std::vector<double> brute_force_stochastic(
    const std::vector<double>& high, const std::vector<double>& low,
    const std::vector<double>& close, int k_period, int d_period)
{
    const int n = static_cast<int>(close.size());
    std::vector<double> result(2 * n, std::numeric_limits<double>::quiet_NaN());
    if (n < k_period || k_period < 1 || d_period <= 0) return result;

    const double kNaNRef = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> K(n, kNaNRef);
    for (int i = k_period - 1; i < n; ++i) {
        // A window holding a missing observation has no extremes to take,
        // so it has no %K -- the same rolling(min_periods=k_period) rule
        // pandas applies, spelled out here rather than inherited from the
        // implementation under test.
        bool missing = false;
        for (int j = i - k_period + 1; j <= i; ++j) {
            if (std::isnan(low[j]) || std::isnan(high[j])) { missing = true; break; }
        }
        if (missing) continue;
        double lo = low[i], hi = high[i];
        for (int j = i - k_period + 1; j <= i; ++j) {
            if (low[j]  < lo) lo = low[j];
            if (high[j] > hi) hi = high[j];
        }
        const double rng = hi - lo;
        K[i] = (rng > 0.0) ? 100.0 * (close[i] - lo) / rng : 0.0;
    }
    for (int i = k_period - 1; i < n; ++i) {
        result[i * 2] = K[i];
        if (i >= k_period - 1 + d_period - 1) {
            double s = 0.0;
            bool   missing = false;
            for (int j = i - d_period + 1; j <= i; ++j) {
                if (std::isnan(K[j])) { missing = true; break; }
                s += K[j];
            }
            result[i * 2 + 1] = missing ? kNaNRef : s / d_period;
        }
    }
    return result;
}

static void check_matches_brute_force(
    const std::vector<double>& high, const std::vector<double>& low,
    const std::vector<double>& close, int k_period, int d_period)
{
    auto result   = sqt::stochastic_oscillator(
        high.data(), low.data(), close.data(), close.size(), k_period, d_period);
    auto expected = brute_force_stochastic(high, low, close, k_period, d_period);
    CHECK_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (std::isnan(expected[i])) { CHECK_NAN(result[i]); }
        else                         { CHECK_NEAR(result[i], expected[i], 1e-9); }
    }
}

static void test_stochastic_matches_brute_force_random() {
    auto close = pseudo_random(300);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);
    check_matches_brute_force(high, low, close, 14, 3);
}

static void test_stochastic_matches_brute_force_monotonic_rising() {
    // Regression test (Tier 4 item 13): the O(n*k_period) rescan was
    // rewritten to O(n) via two monotonic deques for sliding max(high)/
    // min(low). A strictly monotonic series is the adversarial case for
    // that kind of algorithm -- every bar is its own new running extremum,
    // so this exercises the deque's front-eviction (extremum leaving the
    // window k_period bars later), not just its back-insertion logic.
    auto close = rising_prices(60);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);
    check_matches_brute_force(high, low, close, 10, 3);
}

static void test_stochastic_matches_brute_force_monotonic_falling() {
    auto close = falling_prices(60);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);
    check_matches_brute_force(high, low, close, 10, 3);
}

static void test_stochastic_matches_brute_force_spike_exits_window() {
    // A single sharp, isolated spike in high[] must stop being the window
    // max exactly k_period bars later -- exercises front-eviction
    // specifically when the evicted index is NOT the most recently pushed
    // one (the spike sits in the middle of the deque while newer, smaller
    // bars remain behind it).
    auto close = pseudo_random(60, 7);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);
    high[20] += 500.0;
    check_matches_brute_force(high, low, close, 10, 3);
}

static void test_stochastic_k_bounds_0_to_100() {
    auto close = pseudo_random(200, 99);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);
    auto result = sqt::stochastic_oscillator(
        high.data(), low.data(), close.data(), close.size(), 14, 3);
    for (std::size_t i = 0; i < close.size(); ++i) {
        const double k = result[i * 2];
        if (!std::isnan(k)) { CHECK(k >= 0.0); CHECK(k <= 100.0); }
    }
}

static void test_stochastic_close_at_high_yields_k_100() {
    const int n = 30;
    std::vector<double> high(n, 10.0), low(n, 5.0), close(n, 10.0);
    auto result = sqt::stochastic_oscillator(
        high.data(), low.data(), close.data(), n, 5, 3);
    for (int i = 4; i < n; ++i) CHECK_NEAR(result[i * 2], 100.0, 1e-9);
}

static void test_stochastic_empty() {
    auto result = sqt::stochastic_oscillator(nullptr, nullptr, nullptr, 0, 14, 3);
    CHECK_EQ(static_cast<int>(result.size()), 0);
}


// ── Bollinger Bands tests ─────────────────────────────────────────────────────
//
// This kernel had NO direct C++ coverage before -- only an indirect
// equality check inside test_technical_indicators_matches_individual_
// functions, which compares the fused path against this same function and
// so agrees with it whether or not either is right. It is also the most
// intricate sliding-window arithmetic in the file (shifted sums, periodic
// re-centering, a variance that is a difference of large near-equal terms),
// and the place a NaN-handling defect actually shipped.

// Independent brute-force reference: recompute mean and ddof=1 standard
// deviation from scratch for every window, in the obvious two-pass way.
// Shares no code with the incremental implementation, so agreement means
// the O(1) sliding update is correct rather than merely self-consistent.
static std::vector<double> brute_force_bollinger(
    const std::vector<double>& prices, int period, double num_std)
{
    const int n = static_cast<int>(prices.size());
    const double kNaNRef = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> result(3 * static_cast<std::size_t>(n), kNaNRef);
    if (period < 2 || n < period) return result;

    for (int i = period - 1; i < n; ++i) {
        bool missing = false;
        for (int j = i - period + 1; j <= i; ++j) {
            if (!std::isfinite(prices[j])) { missing = true; break; }
        }
        if (missing) continue;  // leave the whole triple NaN

        double sum = 0.0;
        for (int j = i - period + 1; j <= i; ++j) sum += prices[j];
        const double mean = sum / period;
        double ss = 0.0;
        for (int j = i - period + 1; j <= i; ++j) {
            const double d = prices[j] - mean;
            ss += d * d;
        }
        const double sd = std::sqrt(ss / (period - 1));
        const std::size_t o = static_cast<std::size_t>(i) * 3;
        result[o]     = mean + num_std * sd;
        result[o + 1] = mean;
        result[o + 2] = mean - num_std * sd;
    }
    return result;
}

static void check_matches_brute_force_bollinger(
    const std::vector<double>& prices, int period, double num_std, double tol)
{
    auto result   = sqt::bollinger_bands(prices.data(), prices.size(), period, num_std);
    auto expected = brute_force_bollinger(prices, period, num_std);
    CHECK_EQ(result.size(), expected.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (std::isnan(expected[i])) { CHECK_NAN(result[i]); }
        else                         { CHECK_NEAR(result[i], expected[i], tol); }
    }
}

static void test_bollinger_nan_prefix_and_shape() {
    const int period = 20;
    auto prices = pseudo_random(60);
    auto result = sqt::bollinger_bands(prices.data(), prices.size(), period, 2.0);

    CHECK_EQ(result.size(), prices.size() * 3);
    for (int i = 0; i < period - 1; ++i) {
        CHECK_NAN(result[static_cast<std::size_t>(i) * 3]);
        CHECK_NAN(result[static_cast<std::size_t>(i) * 3 + 1]);
        CHECK_NAN(result[static_cast<std::size_t>(i) * 3 + 2]);
    }
    for (std::size_t i = period - 1; i < prices.size(); ++i) {
        CHECK_NOT_NAN(result[i * 3 + 1]);
    }
}

static void test_bollinger_matches_brute_force_random() {
    auto prices = pseudo_random(300);
    check_matches_brute_force_bollinger(prices, 20, 2.0, 1e-9);
}

static void test_bollinger_matches_brute_force_across_refresh_boundary() {
    // The incremental sums are rebuilt from scratch every `period` bars.
    // A series several multiples of `period` long walks that boundary
    // repeatedly, which is where a refresh that re-centres on the wrong
    // start index would show up.
    auto prices = pseudo_random(97, 7);
    check_matches_brute_force_bollinger(prices, 5, 2.0, 1e-9);
    check_matches_brute_force_bollinger(prices, 12, 1.5, 1e-9);
}

static void test_bollinger_constant_series_has_zero_width() {
    std::vector<double> prices(40, 123.5);
    auto result = sqt::bollinger_bands(prices.data(), prices.size(), 10, 2.0);
    for (std::size_t i = 9; i < prices.size(); ++i) {
        CHECK_NEAR(result[i * 3 + 1], 123.5, 1e-12);  // middle
        CHECK_NEAR(result[i * 3],     123.5, 1e-12);  // upper == middle
        CHECK_NEAR(result[i * 3 + 2], 123.5, 1e-12);  // lower == middle
    }
}

static void test_bollinger_large_baseline_no_catastrophic_cancellation() {
    // Pins the shift-by-reference-point centering. Raw-moment sums on a
    // ~1e9-level series with variance ~0.35 previously produced a NEGATIVE
    // variance (measured: -215.58), silently clamped to std=0 -- Bollinger
    // bands collapsing onto the moving average with no signal at all.
    std::vector<double> prices(200);
    for (int i = 0; i < 200; ++i) {
        prices[static_cast<std::size_t>(i)] =
            1.0e9 + ((i * 37) % 13) * 0.1;  // deterministic, spread ~1.2
    }
    auto result = sqt::bollinger_bands(prices.data(), prices.size(), 20, 2.0);
    for (std::size_t i = 19; i < prices.size(); ++i) {
        CHECK_NOT_NAN(result[i * 3 + 1]);
        // A genuinely non-degenerate window must have non-zero width.
        CHECK(result[i * 3] > result[i * 3 + 2]);
    }
    // And the values must still be right, not merely non-degenerate.
    check_matches_brute_force_bollinger(prices, 20, 2.0, 1e-6);
}

static void test_bollinger_short_series_and_bad_period() {
    auto prices = pseudo_random(10);
    for (const double v : sqt::bollinger_bands(prices.data(), prices.size(), 20, 2.0))
        CHECK_NAN(v);                                    // n < period
    for (const double v : sqt::bollinger_bands(prices.data(), prices.size(), 1, 2.0))
        CHECK_NAN(v);                                    // period < 2
    for (const double v : sqt::bollinger_bands(prices.data(), prices.size(), -5, 2.0))
        CHECK_NAN(v);                                    // negative period
    CHECK(sqt::bollinger_bands(nullptr, 0, 20, 2.0).empty());
}

static void test_bollinger_nan_bar_does_not_throw_and_stays_local() {
    // REGRESSION. A single NaN price used to make this raise
    // std::runtime_error for the WHOLE series: clamp_near_zero_sumsq fell
    // through to its throw because `NaN >= 0.0` and `|NaN| < eps*|NaN|` are
    // both false. The documented contract for bad bars in this project is
    // NaN propagation, never an exception.
    //
    // The second half is subtler and is what the brute-force comparison
    // pins: an O(1) sliding sum cannot un-add a NaN (NaN - NaN is NaN), so
    // even after the fix the sums stayed poisoned until the next periodic
    // refresh and the kernel reported NaN for a run of windows containing
    // no bad data at all.
    auto prices = pseudo_random(80);
    prices[30] = std::numeric_limits<double>::quiet_NaN();
    check_matches_brute_force_bollinger(prices, 10, 2.0, 1e-9);

    // Explicitly: NaN for exactly the 10 windows covering bar 30, and a
    // real number on the very next bar.
    auto result = sqt::bollinger_bands(prices.data(), prices.size(), 10, 2.0);
    for (std::size_t i = 30; i <= 39; ++i) CHECK_NAN(result[i * 3 + 1]);
    CHECK_NOT_NAN(result[40 * 3 + 1]);
}

static void test_bollinger_inf_bar_does_not_throw_and_stays_local() {
    // Inf is folded in with NaN here (unlike the stochastic kernel) because
    // an Inf entering the sliding sums is equally unrecoverable: the update
    // subtracts it back out as inf - inf = NaN.
    auto prices = pseudo_random(80);
    prices[30] = std::numeric_limits<double>::infinity();
    check_matches_brute_force_bollinger(prices, 10, 2.0, 1e-9);

    auto result = sqt::bollinger_bands(prices.data(), prices.size(), 10, 2.0);
    for (std::size_t i = 30; i <= 39; ++i) CHECK_NAN(result[i * 3 + 1]);
    CHECK_NOT_NAN(result[40 * 3 + 1]);
}

static void test_bollinger_leading_nan_does_not_poison_the_series() {
    // The seed window starts at bar 0, so a NaN at bar 0 is also the
    // reference point the shifted sums are centred on.
    auto prices = pseudo_random(60);
    prices[0] = std::numeric_limits<double>::quiet_NaN();
    check_matches_brute_force_bollinger(prices, 10, 2.0, 1e-9);
}


// ── Stochastic Oscillator: missing-observation handling ──────────────────────

static void test_stochastic_nan_high_does_not_corrupt_the_deque() {
    // REGRESSION. A NaN was pushed into the monotonic deque like any other
    // value. Every comparison against NaN is false, so it was never popped
    // from the back and it blocked eviction of the stale indices behind it
    // -- the deque front stopped being the window maximum. Measured on this
    // exact input: %K, which is bounded 0..100 by construction, returned
    // 125, 166.67 and 250 from a stale max, then a fabricated 0.0 once the
    // NaN reached the front.
    auto close = rising_prices(20, 9.0);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);
    high[5] = std::numeric_limits<double>::quiet_NaN();

    check_matches_brute_force(high, low, close, 5, 3);

    auto result = sqt::stochastic_oscillator(
        high.data(), low.data(), close.data(), close.size(), 5, 3);
    for (std::size_t i = 0; i < close.size(); ++i) {
        const double k = result[i * 2];
        if (!std::isnan(k)) CHECK(k >= 0.0 && k <= 100.0);
    }
    // NaN for exactly the 5 windows covering bar 5, real again at bar 10.
    for (std::size_t i = 5; i <= 9; ++i) CHECK_NAN(result[i * 2]);
    CHECK_NOT_NAN(result[10 * 2]);
}

static void test_stochastic_nan_low_only_still_yields_the_right_max() {
    // Each deque is gated on its own series: a bar with a valid high but a
    // missing low must still be a max candidate once the low ages out, or
    // the window maximum is silently lost.
    auto close = pseudo_random(40);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);
    low[12] = std::numeric_limits<double>::quiet_NaN();
    check_matches_brute_force(high, low, close, 6, 3);
}

static void test_stochastic_nan_k_does_not_poison_every_later_d() {
    // REGRESSION. %D was a running sum, so one NaN %K entered it and could
    // never be subtracted back out -- every %D for the rest of the series
    // came back NaN, long after the bad bar had left both windows.
    auto close = pseudo_random(60);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);
    high[20] = std::numeric_limits<double>::quiet_NaN();

    check_matches_brute_force(high, low, close, 5, 3);

    auto result = sqt::stochastic_oscillator(
        high.data(), low.data(), close.data(), close.size(), 5, 3);
    CHECK_NOT_NAN(result[59 * 2 + 1]);  // %D recovered well before the end
}

static void test_stochastic_all_nan_series_is_all_nan_not_a_crash() {
    const std::size_t n = 30;
    std::vector<double> nanv(n, std::numeric_limits<double>::quiet_NaN());
    auto result = sqt::stochastic_oscillator(
        nanv.data(), nanv.data(), nanv.data(), n, 5, 3);
    CHECK_EQ(result.size(), n * 2);
    for (const double v : result) CHECK_NAN(v);
}


// ── Fused technical_indicators() tests ────────────────────────────────────────

static void test_technical_indicators_matches_individual_functions() {
    // Regression test (performance architecture item 6): technical_indicators()
    // is pure orchestration over the same *_into kernels the individual
    // rsi()/adx()/wilder_atr()/bollinger_bands()/stochastic_oscillator()
    // functions use -- every field it returns must match calling each
    // function standalone, exactly (not just approximately: same kernel,
    // same inputs, must produce bit-identical output).
    auto close = pseudo_random(300, 7);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);
    const std::size_t n = close.size();

    sqt::TechnicalIndicatorsConfig cfg;
    cfg.compute_rsi        = true;  cfg.rsi_period        = 14;
    cfg.compute_adx        = true;  cfg.adx_period        = 14;
    cfg.compute_atr        = true;  cfg.atr_period        = 14;
    cfg.compute_bollinger  = true;  cfg.bollinger_period  = 20; cfg.bollinger_num_std = 2.0;
    cfg.compute_stochastic = true;  cfg.stoch_k_period    = 14; cfg.stoch_d_period    = 3;

    auto fused = sqt::technical_indicators(high.data(), low.data(), close.data(), n, cfg);

    auto expected_rsi = sqt::rsi(close.data(), n, cfg.rsi_period);
    CHECK_EQ(fused.rsi.size(), expected_rsi.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(expected_rsi[i])) CHECK_NAN(fused.rsi[i]);
        else CHECK_NEAR(fused.rsi[i], expected_rsi[i], 1e-15);
    }

    auto expected_adx = sqt::adx(high.data(), low.data(), close.data(), n, cfg.adx_period);
    CHECK_EQ(fused.adx.size(), expected_adx.size());
    for (std::size_t i = 0; i < expected_adx.size(); ++i) {
        if (std::isnan(expected_adx[i])) CHECK_NAN(fused.adx[i]);
        else CHECK_NEAR(fused.adx[i], expected_adx[i], 1e-15);
    }

    auto expected_atr = sqt::wilder_atr(high.data(), low.data(), close.data(), n, cfg.atr_period);
    CHECK_EQ(fused.atr.size(), expected_atr.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(expected_atr[i])) CHECK_NAN(fused.atr[i]);
        else CHECK_NEAR(fused.atr[i], expected_atr[i], 1e-15);
    }

    auto expected_bb = sqt::bollinger_bands(
        close.data(), n, cfg.bollinger_period, cfg.bollinger_num_std);
    CHECK_EQ(fused.bollinger.size(), expected_bb.size());
    for (std::size_t i = 0; i < expected_bb.size(); ++i) {
        if (std::isnan(expected_bb[i])) CHECK_NAN(fused.bollinger[i]);
        else CHECK_NEAR(fused.bollinger[i], expected_bb[i], 1e-15);
    }

    auto expected_stoch = sqt::stochastic_oscillator(
        high.data(), low.data(), close.data(), n, cfg.stoch_k_period, cfg.stoch_d_period);
    CHECK_EQ(fused.stochastic.size(), expected_stoch.size());
    for (std::size_t i = 0; i < expected_stoch.size(); ++i) {
        if (std::isnan(expected_stoch[i])) CHECK_NAN(fused.stochastic[i]);
        else CHECK_NEAR(fused.stochastic[i], expected_stoch[i], 1e-15);
    }
}

static void test_technical_indicators_only_requested_fields_populated() {
    auto close = pseudo_random(100, 3);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);
    const std::size_t n = close.size();

    sqt::TechnicalIndicatorsConfig cfg;  // everything false/default
    cfg.compute_rsi = true;
    cfg.rsi_period  = 10;

    auto r = sqt::technical_indicators(high.data(), low.data(), close.data(), n, cfg);
    CHECK(!r.rsi.empty());
    CHECK(r.adx.empty());
    CHECK(r.atr.empty());
    CHECK(r.bollinger.empty());
    CHECK(r.stochastic.empty());
}

static void test_technical_indicators_empty_config_returns_all_empty() {
    auto close = pseudo_random(50, 5);
    std::vector<double> high, low;
    ohlc_from_prices(close, high, low);
    sqt::TechnicalIndicatorsConfig cfg;  // all flags default false
    auto r = sqt::technical_indicators(
        high.data(), low.data(), close.data(), close.size(), cfg);
    CHECK(r.rsi.empty());
    CHECK(r.adx.empty());
    CHECK(r.atr.empty());
    CHECK(r.bollinger.empty());
    CHECK(r.stochastic.empty());
}


// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    // RSI
    test_rsi_nan_prefix();
    test_rsi_all_rising_equals_100();
    test_rsi_all_falling_equals_0();
    test_rsi_known_value();
    test_rsi_bounds();
    test_rsi_short_series();
    test_rsi_empty();

    // ADX
    test_adx_nan_prefix();
    test_adx_uptrend_di_plus_dominates();
    test_adx_bounds();
    test_adx_short_series();
    test_adx_exact_regression_pin();

    // Parabolic SAR
    test_psar_bootstrap();
    test_psar_trend_values_are_pm1();
    test_psar_rising_trend_sar_below_price();
    test_psar_single_bar();
    test_psar_empty();

    // Wilder's ATR
    test_wilder_atr_nan_prefix();
    test_wilder_atr_known_value();
    test_wilder_atr_non_negative();
    test_wilder_atr_constant_prices();
    test_wilder_atr_smoothing_recurrence();
    test_wilder_atr_matches_unfused_array_reference_exactly();
    test_wilder_atr_short_series();
    test_wilder_atr_empty();
    test_wilder_atr_decays_toward_tr();

    // Bollinger Bands
    test_bollinger_nan_prefix_and_shape();
    test_bollinger_matches_brute_force_random();
    test_bollinger_matches_brute_force_across_refresh_boundary();
    test_bollinger_constant_series_has_zero_width();
    test_bollinger_large_baseline_no_catastrophic_cancellation();
    test_bollinger_short_series_and_bad_period();
    test_bollinger_nan_bar_does_not_throw_and_stays_local();
    test_bollinger_inf_bar_does_not_throw_and_stays_local();
    test_bollinger_leading_nan_does_not_poison_the_series();

    // Stochastic Oscillator
    test_stochastic_matches_brute_force_random();
    test_stochastic_matches_brute_force_monotonic_rising();
    test_stochastic_matches_brute_force_monotonic_falling();
    test_stochastic_matches_brute_force_spike_exits_window();
    test_stochastic_k_bounds_0_to_100();
    test_stochastic_close_at_high_yields_k_100();
    test_stochastic_empty();
    test_stochastic_nan_high_does_not_corrupt_the_deque();
    test_stochastic_nan_low_only_still_yields_the_right_max();
    test_stochastic_nan_k_does_not_poison_every_later_d();
    test_stochastic_all_nan_series_is_all_nan_not_a_crash();

    // Fused technical_indicators()
    test_technical_indicators_matches_individual_functions();
    test_technical_indicators_only_requested_fields_populated();
    test_technical_indicators_empty_config_returns_all_empty();

    std::fprintf(
        stdout,
        "\n%s  %d / %d tests passed.\n",
        g_tests_failed == 0 ? "PASS" : "FAIL",
        g_tests_run - g_tests_failed,
        g_tests_run);

    return g_tests_failed == 0 ? 0 : 1;
}
