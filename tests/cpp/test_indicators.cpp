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

#include <cassert>
#include <cmath>
#include <cstdio>
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

    std::fprintf(
        stdout,
        "\n%s  %d / %d tests passed.\n",
        g_tests_failed == 0 ? "PASS" : "FAIL",
        g_tests_run - g_tests_failed,
        g_tests_run);

    return g_tests_failed == 0 ? 0 : 1;
}
