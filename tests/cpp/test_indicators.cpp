/**
 * C++ unit tests for sqt::rsi, sqt::adx, sqt::parabolic_sar.
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

    std::vector<double> K(n, std::numeric_limits<double>::quiet_NaN());
    for (int i = k_period - 1; i < n; ++i) {
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
            for (int j = i - d_period + 1; j <= i; ++j) s += K[j];
            result[i * 2 + 1] = s / d_period;
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
    test_wilder_atr_short_series();
    test_wilder_atr_empty();
    test_wilder_atr_decays_toward_tr();

    // Stochastic Oscillator
    test_stochastic_matches_brute_force_random();
    test_stochastic_matches_brute_force_monotonic_rising();
    test_stochastic_matches_brute_force_monotonic_falling();
    test_stochastic_matches_brute_force_spike_exits_window();
    test_stochastic_k_bounds_0_to_100();
    test_stochastic_close_at_high_yields_k_100();
    test_stochastic_empty();

    std::fprintf(
        stdout,
        "\n%s  %d / %d tests passed.\n",
        g_tests_failed == 0 ? "PASS" : "FAIL",
        g_tests_run - g_tests_failed,
        g_tests_run);

    return g_tests_failed == 0 ? 0 : 1;
}
