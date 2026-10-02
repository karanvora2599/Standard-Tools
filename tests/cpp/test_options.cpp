/**
 * C++ unit tests for the chain kernels in options.cpp: implied volatility
 * and Black-Scholes-Merton greeks over a batch.
 *
 * The Python suite (tests/analysis/test_options_batch.py) holds these
 * kernels to the scalar Python functions result for result. What is checked
 * here is what the kernel owes on its own: textbook values, the calculus
 * the greeks claim (finite differences of the kernel's own price), parity,
 * every refusal code, and that a batch -- serial or parallel, per contract
 * or on a spot grid -- answers exactly what one contract at a time does.
 *
 * Run via CTest:
 *   ctest --test-dir build --config Release -V -R cpp_options
 */

#include "sqt/options.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static int g_tests_run = 0;
static int g_tests_failed = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        ++g_tests_run;                                                                \
        if (!(cond)) {                                                                \
            ++g_tests_failed;                                                         \
            std::fprintf(stderr, "FAIL  %s  line %d: %s\n", __func__, __LINE__, #cond); \
        }                                                                             \
    } while (false)

#define CHECK_NEAR(a, b, tol) CHECK(std::abs((a) - (b)) <= (tol))

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

// Bit-for-bit equality that treats two NaNs as equal: "the batch answers
// what one contract answers" includes the refused ones.
bool same(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return true;
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

double price_of(double s, double k, double t, double v, double r, double q, bool call) {
    return sqt::black_scholes_greeks_one(s, k, t, v, r, q, call).price;
}

sqt::ImpliedVolResult solve(double price, double s, double k, double t, double r,
                            double q, bool call) {
    return sqt::implied_volatility_one(price, s, k, t, r, q, call,
                                       sqt::ImpliedVolSettings{});
}

}  // namespace

// ── Greeks ──────────────────────────────────────────────────────────────────

static void test_hull_textbook_prices() {
    // Hull's example: S=42, K=40, T=0.5, r=10%, sigma=20%.
    CHECK_NEAR(price_of(42, 40, 0.5, 0.2, 0.1, 0.0, true), 4.759422392871528, 1e-12);
    CHECK_NEAR(price_of(42, 40, 0.5, 0.2, 0.1, 0.0, false), 0.8085993729000958, 1e-12);
}

static void test_greeks_match_the_library_units() {
    // analysis.derivatives.option_greeks at the same inputs.
    const auto g = sqt::black_scholes_greeks_one(42, 40, 0.5, 0.2, 0.1, 0.0, true);
    CHECK_NEAR(g.delta, 0.7791312909426689, 1e-13);
    CHECK_NEAR(g.gamma, 0.04996267040591185, 1e-13);
    CHECK_NEAR(g.vega, 0.08813415059602853, 1e-13);   // per vol point
    CHECK_NEAR(g.theta, -0.012490663546829112, 1e-13);  // per calendar day
    CHECK_NEAR(g.rho, 0.1398204591336028, 1e-13);     // per rate point
    CHECK_NEAR(g.vanna, -0.009316006786136685, 1e-13);
    CHECK_NEAR(g.volga, 0.0021283288061014413, 1e-13);
    CHECK_NEAR(g.charm, -6.444679447149612e-05, 1e-15);
    CHECK_NEAR(g.speed, -0.007660357766572154, 1e-13);
    CHECK_NEAR(g.d1, 0.7692626281060315, 1e-13);
    CHECK_NEAR(g.d2, 0.627841271868722, 1e-13);
}

static void test_greeks_are_the_derivatives_of_the_price() {
    // Central differences of the kernel's own price, so a sign or a unit
    // error in any one greek fails here without an external oracle.
    const double s = 95, k = 100, t = 0.75, v = 0.3, r = 0.03, q = 0.015;
    for (bool call : {true, false}) {
        const auto g = sqt::black_scholes_greeks_one(s, k, t, v, r, q, call);
        const double hs = 1e-3, hv = 1e-5, ht = 1e-6, hr = 1e-6;
        const double delta = (price_of(s + hs, k, t, v, r, q, call) -
                              price_of(s - hs, k, t, v, r, q, call)) / (2 * hs);
        const double gamma = (price_of(s + hs, k, t, v, r, q, call) -
                              2 * price_of(s, k, t, v, r, q, call) +
                              price_of(s - hs, k, t, v, r, q, call)) / (hs * hs);
        const double vega = (price_of(s, k, t, v + hv, r, q, call) -
                             price_of(s, k, t, v - hv, r, q, call)) / (2 * hv) / 100.0;
        // theta is -dPrice/dT, per calendar day.
        const double theta = -(price_of(s, k, t + ht, v, r, q, call) -
                               price_of(s, k, t - ht, v, r, q, call)) / (2 * ht) / 365.0;
        const double rho = (price_of(s, k, t, v, r + hr, q, call) -
                            price_of(s, k, t, v, r - hr, q, call)) / (2 * hr) / 100.0;
        CHECK_NEAR(g.delta, delta, 1e-7);
        CHECK_NEAR(g.gamma, gamma, 1e-5);
        CHECK_NEAR(g.vega, vega, 1e-8);
        CHECK_NEAR(g.theta, theta, 1e-8);
        CHECK_NEAR(g.rho, rho, 1e-8);
        const double vanna =
            (sqt::black_scholes_greeks_one(s, k, t, v + hv, r, q, call).delta -
             sqt::black_scholes_greeks_one(s, k, t, v - hv, r, q, call).delta) /
            (2 * hv) / 100.0;
        CHECK_NEAR(g.vanna, vanna, 1e-8);
    }
}

static void test_put_call_parity() {
    const double s = 101, k = 97, t = 0.4, v = 0.25, r = -0.005, q = 0.02;
    const auto c = sqt::black_scholes_greeks_one(s, k, t, v, r, q, true);
    const auto p = sqt::black_scholes_greeks_one(s, k, t, v, r, q, false);
    CHECK_NEAR(c.price - p.price, s * std::exp(-q * t) - k * std::exp(-r * t), 1e-12);
    CHECK_NEAR(c.delta - p.delta, std::exp(-q * t), 1e-14);
    CHECK(c.gamma == p.gamma);
    CHECK(c.vega == p.vega);
    CHECK(c.vanna == p.vanna);
    CHECK(c.volga == p.volga);
}

static void test_outside_the_domain_is_nan_in_every_field() {
    const double bad[][6] = {
        {0.0, 100, 1, 0.2, 0.0, 0.0},     // spot
        {100, -1, 1, 0.2, 0.0, 0.0},      // strike
        {100, 100, 0.0, 0.2, 0.0, 0.0},   // time
        {100, 100, 1, 0.0, 0.0, 0.0},     // volatility
        {100, 100, 1, 0.2, kNaN, 0.0},    // rate
        {100, 100, 1, 0.2, 0.0, 11.0},    // yield past MAX_RATE
        {100, 100, 100, 0.2, -9.0, 0.0},  // rate x time past MAX_EXPONENT
        {2e12, 100, 1, 0.2, 0.0, 0.0},    // a price past any market
    };
    for (const auto& b : bad) {
        const auto g = sqt::black_scholes_greeks_one(b[0], b[1], b[2], b[3], b[4], b[5], true);
        CHECK(std::isnan(g.price) && std::isnan(g.delta) && std::isnan(g.gamma) &&
              std::isnan(g.speed) && std::isnan(g.d2));
    }
}

static void test_grid_is_every_contract_at_every_spot() {
    const std::vector<double> spots = {80, 95, 100, 105, 130};
    const std::vector<double> strike = {90, 100, 110};
    const std::vector<double> t = {0.1, 0.5, 2.0};
    const std::vector<double> vol = {0.35, 0.2, 0.25};
    const std::vector<double> rate = {0.01, 0.0, -0.01};
    const std::vector<double> q = {0.0, 0.03, 0.0};
    const std::vector<std::uint8_t> call = {1, 0, 1};
    const std::size_t nc = strike.size(), ns = spots.size();
    std::vector<std::vector<double>> buf(12, std::vector<double>(nc * ns));
    sqt::BlackScholesGreeksOut out{buf[0].data(), buf[1].data(), buf[2].data(),
                                   buf[3].data(), buf[4].data(), buf[5].data(),
                                   buf[6].data(), buf[7].data(), buf[8].data(),
                                   buf[9].data(), buf[10].data(), buf[11].data()};
    sqt::black_scholes_greeks_batch(spots.data(), ns, strike.data(), t.data(), vol.data(),
                                    rate.data(), q.data(), call.data(), nc, true, out);
    for (std::size_t i = 0; i < nc; ++i) {
        for (std::size_t j = 0; j < ns; ++j) {
            const auto g = sqt::black_scholes_greeks_one(spots[j], strike[i], t[i], vol[i],
                                                         rate[i], q[i], call[i] != 0);
            const std::size_t at = i * ns + j;  // row-major (contract, spot)
            CHECK(same(buf[0][at], g.price));
            CHECK(same(buf[2][at], g.gamma));
            CHECK(same(buf[4][at], g.theta));
            CHECK(same(buf[9][at], g.speed));
        }
    }
}

static void test_a_large_batch_answers_what_one_contract_does() {
    // Big enough that omp_policy takes the parallel path when OpenMP is
    // built in: the answer must not depend on how the work was split.
    const std::size_t n = 20000;
    std::vector<double> s(n), k(n), t(n), v(n), r(n), q(n);
    std::vector<std::uint8_t> c(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        s[i] = 100.0;
        k[i] = 50.0 + std::fmod(x * 0.37, 100.0);
        t[i] = 0.01 + std::fmod(x * 0.013, 3.0);
        v[i] = 0.05 + std::fmod(x * 0.0071, 1.2);
        r[i] = -0.01 + std::fmod(x * 0.0003, 0.08);
        q[i] = std::fmod(x * 0.0002, 0.04);
        c[i] = static_cast<std::uint8_t>(i % 2);
    }
    std::vector<std::vector<double>> buf(12, std::vector<double>(n));
    sqt::BlackScholesGreeksOut out{buf[0].data(), buf[1].data(), buf[2].data(),
                                   buf[3].data(), buf[4].data(), buf[5].data(),
                                   buf[6].data(), buf[7].data(), buf[8].data(),
                                   buf[9].data(), buf[10].data(), buf[11].data()};
    sqt::black_scholes_greeks_batch(s.data(), n, k.data(), t.data(), v.data(), r.data(),
                                    q.data(), c.data(), n, false, out);
    bool all_same = true;
    for (std::size_t i = 0; i < n; ++i) {
        const auto g =
            sqt::black_scholes_greeks_one(s[i], k[i], t[i], v[i], r[i], q[i], c[i] != 0);
        all_same = all_same && same(buf[0][i], g.price) && same(buf[1][i], g.delta) &&
                   same(buf[7][i], g.volga) && same(buf[8][i], g.charm);
    }
    CHECK(all_same);

    // Then implied volatility over the same chain, priced at its own vols.
    std::vector<double> price(n), vol(n), err(n);
    std::vector<std::int32_t> iters(n);
    std::vector<std::int8_t> method(n), reason(n);
    std::vector<std::uint8_t> conv(n), at_bound(n);
    for (std::size_t i = 0; i < n; ++i) price[i] = buf[0][i];
    sqt::ImpliedVolBatchOut iv{vol.data(),    err.data(),    iters.data(), method.data(),
                               reason.data(), conv.data(), at_bound.data()};
    sqt::implied_volatility_batch(price.data(), s.data(), k.data(), t.data(), r.data(),
                                  q.data(), c.data(), n, sqt::ImpliedVolSettings{}, iv);
    bool iv_same = true;
    for (std::size_t i = 0; i < n; ++i) {
        const auto one = sqt::implied_volatility_one(price[i], s[i], k[i], t[i], r[i],
                                                     q[i], c[i] != 0,
                                                     sqt::ImpliedVolSettings{});
        iv_same = iv_same && same(vol[i], one.vol) && same(err[i], one.price_error) &&
                  iters[i] == one.iterations && method[i] == one.method &&
                  reason[i] == one.reason && (conv[i] != 0) == one.converged &&
                  (at_bound[i] != 0) == one.at_bound;
    }
    CHECK(iv_same);
}

// ── Implied volatility ──────────────────────────────────────────────────────

static void test_round_trip_recovers_the_volatility() {
    for (bool call : {true, false}) {
        for (double v : {0.05, 0.2, 0.8, 2.5}) {
            const double p = price_of(100, 105, 0.5, v, 0.02, 0.01, call);
            const auto r = solve(p, 100, 105, 0.5, 0.02, 0.01, call);
            CHECK(r.reason == sqt::kIvSolved);
            CHECK(r.converged);
            CHECK(!r.at_bound);
            CHECK_NEAR(r.vol, v, 1e-8);
            CHECK(r.price_error < 1e-9);
        }
    }
}

static void test_hull_solves_in_one_newton_step() {
    const auto r = solve(4.759422392871528, 42, 40, 0.5, 0.1, 0.0, true);
    CHECK(r.reason == sqt::kIvSolved);
    CHECK(r.method == sqt::kIvMethodNewton);
    CHECK(r.iterations == 1);
    // The price was computed with one C runtime's exp and erfc; glibc's
    // differ from MSVC's in the last bit, so the one Newton step lands within
    // a few ulps of 0.2 rather than on it.
    CHECK_NEAR(r.vol, 0.2, 1e-14);
}

static void test_a_price_at_intrinsic_is_a_ceiling() {
    // Deep in the money, priced by the pricer itself: bit-for-bit at the
    // lower bound, so every smaller volatility reproduces it as well.
    const double p = price_of(150, 50, 0.25, 0.2, 0.05, 0.0, true);
    const auto r = solve(p, 150, 50, 0.25, 0.05, 0.0, true);
    CHECK(r.reason == sqt::kIvSolved);
    CHECK(r.at_bound);
    CHECK(r.converged);
    CHECK(r.method == sqt::kIvMethodBisection);
    CHECK(r.iterations == 29);
    CHECK_NEAR(r.vol, 0.41653287996749033, 1e-15);
}

static void test_every_refusal_has_its_own_code() {
    CHECK(solve(0.0, 100, 100, 1, 0, 0, true).reason == sqt::kIvPriceNotPositive);
    CHECK(solve(-1.0, 100, 100, 1, 0, 0, true).reason == sqt::kIvPriceNotPositive);
    CHECK(solve(kNaN, 100, 100, 1, 0, 0, true).reason == sqt::kIvPriceNotFinite);
    CHECK(solve(HUGE_VAL, 100, 100, 1, 0, 0, true).reason == sqt::kIvPriceNotFinite);
    // A call worth more than the spot, and one worth less than intrinsic.
    CHECK(solve(101.0, 100, 100, 1, 0, 0, true).reason == sqt::kIvAboveUpperBound);
    CHECK(solve(9.0, 110, 100, 1, 0, 0, true).reason == sqt::kIvBelowLowerBound);
    // A put worth more than the discounted strike.
    CHECK(solve(99.0, 100, 95, 1, 0.05, 0, false).reason == sqt::kIvAboveUpperBound);
    // Inside the bounds, but only a volatility past 500% reproduces it.
    CHECK(solve(99.0, 100, 100, 1, 0, 0, true).reason == sqt::kIvNoRootInBracket);
    CHECK(solve(5.0, 0.0, 100, 1, 0, 0, true).reason == sqt::kIvInvalidInput);
    CHECK(solve(5.0, 100, 100, 0.0, 0, 0, true).reason == sqt::kIvInvalidInput);
    CHECK(solve(5.0, 100, 100, 1, 0, kNaN, true).reason == sqt::kIvInvalidInput);
    // Each rate inside its bound, the discounted strike past a double.
    CHECK(solve(5.0, 100, 1e12, 77.7, -9.0, 0, true).reason == sqt::kIvNotPriceable);
    const auto r = solve(0.0, 100, 100, 1, 0, 0, true);
    CHECK(std::isnan(r.vol) && std::isnan(r.price_error) && !r.converged &&
          r.method == sqt::kIvMethodNone && r.iterations == 0);
}

static void test_zero_and_negative_rates_and_yields_solve() {
    for (double r : {0.0, -0.02}) {
        for (double q : {0.0, -0.01, 0.04}) {
            const double p = price_of(100, 90, 1.5, 0.3, r, q, false);
            const auto res = solve(p, 100, 90, 1.5, r, q, false);
            CHECK(res.reason == sqt::kIvSolved);
            CHECK_NEAR(res.vol, 0.3, 1e-8);
        }
    }
}

static void test_an_empty_batch_is_a_no_op() {
    // Null case: nothing to solve, nothing written, nothing read.
    sqt::ImpliedVolBatchOut iv{nullptr, nullptr, nullptr, nullptr,
                               nullptr, nullptr, nullptr};
    sqt::implied_volatility_batch(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                  nullptr, 0, sqt::ImpliedVolSettings{}, iv);
    sqt::BlackScholesGreeksOut g{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    sqt::black_scholes_greeks_batch(nullptr, 0, nullptr, nullptr, nullptr, nullptr,
                                    nullptr, nullptr, 0, true, g);
    sqt::black_scholes_greeks_batch(nullptr, 0, nullptr, nullptr, nullptr, nullptr,
                                    nullptr, nullptr, 0, false, g);
    CHECK(true);
}

int main() {
    test_hull_textbook_prices();
    test_greeks_match_the_library_units();
    test_greeks_are_the_derivatives_of_the_price();
    test_put_call_parity();
    test_outside_the_domain_is_nan_in_every_field();
    test_grid_is_every_contract_at_every_spot();
    test_a_large_batch_answers_what_one_contract_does();
    test_round_trip_recovers_the_volatility();
    test_hull_solves_in_one_newton_step();
    test_a_price_at_intrinsic_is_a_ceiling();
    test_every_refusal_has_its_own_code();
    test_zero_and_negative_rates_and_yields_solve();
    test_an_empty_batch_is_a_no_op();

    std::printf("%d/%d checks passed\n", g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed == 0 ? 0 : 1;
}
