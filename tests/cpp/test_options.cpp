/**
 * C++ unit tests for the chain kernels in options.cpp: implied volatility
 * and Black-Scholes-Merton greeks over a batch, and the binomial lattice.
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

// ── Output selection ────────────────────────────────────────────────────────

namespace {

// Twelve greek buffers plus the price_finite flag, for one batch call.
struct Buffers {
    std::vector<std::vector<double>> greek;
    std::vector<std::uint8_t> finite;
    explicit Buffers(std::size_t cells) : greek(12, std::vector<double>(cells)), finite(cells) {}
    // Only the outputs in `mask` get a pointer; the rest are NULL, so a
    // write to an unselected output would fault rather than pass.
    sqt::BlackScholesGreeksOut out(std::uint32_t mask) {
        double* p[12];
        for (std::size_t i = 0; i < 12; ++i)
            p[i] = (mask & (1u << i)) ? greek[i].data() : nullptr;
        return sqt::BlackScholesGreeksOut{
            p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11],
            (mask & sqt::kGreekPriceFinite) ? finite.data() : nullptr};
    }
};

// A book that reaches every branch of the cell loop: calls and puts, a
// contract outside the domain, and -- on the spot axis -- spots outside it
// on both sides of a block boundary.
struct Book {
    std::vector<double> spots, strike, t, vol, rate, q;
    std::vector<std::uint8_t> call;
    Book() {
        for (int j = 0; j < 150; ++j) spots.push_back(40.0 + 0.8 * j);  // 3 blocks of 64
        spots[3] = 0.0;
        spots[63] = kNaN;    // the last spot of the first block
        spots[64] = -5.0;    // the first of the second
        spots[149] = 2e12;   // past any market
        const double k[] = {60, 90, 100, 100, 115, 150, 100, 80};
        const double tt[] = {0.02, 0.25, 1.0, 1.0, 2.5, 0.5, 1.0, 4.0};
        const double v[] = {0.9, 0.2, 0.25, 0.25, 0.4, 1.5, 0.0, 0.05};  // vol 0: refused
        const double r[] = {0.0, 0.03, -0.01, 0.05, 0.07, 0.0, 0.02, 0.01};
        const double d[] = {0.0, 0.01, 0.02, -0.01, 0.0, 0.05, 0.0, 0.03};
        for (int i = 0; i < 8; ++i) {
            strike.push_back(k[i]);
            t.push_back(tt[i]);
            vol.push_back(v[i]);
            rate.push_back(r[i]);
            q.push_back(d[i]);
            call.push_back(static_cast<std::uint8_t>(i % 2));
        }
    }
    std::size_t nc() const { return strike.size(); }
    std::size_t ns() const { return spots.size(); }
    void grid(sqt::BlackScholesGreeksOut out) const {
        sqt::black_scholes_greeks_batch(spots.data(), ns(), strike.data(), t.data(),
                                        vol.data(), rate.data(), q.data(), call.data(),
                                        nc(), true, out);
    }
};

double field(const sqt::BlackScholesGreeks& g, std::size_t i) {
    const double v[12] = {g.price, g.delta, g.gamma, g.vega,  g.theta, g.rho,
                          g.vanna, g.volga, g.charm, g.speed, g.d1,    g.d2};
    return v[i];
}

}  // namespace

static void test_a_grid_longer_than_a_block_is_one_contract_at_a_time() {
    // The grid is cut into blocks of spots per contract; every cell of every
    // block, the refused ones included, is the scalar function's doubles.
    const Book b;
    Buffers buf(b.nc() * b.ns());
    b.grid(buf.out(sqt::kGreeksAll));
    bool all_same = true;
    for (std::size_t i = 0; i < b.nc(); ++i) {
        for (std::size_t j = 0; j < b.ns(); ++j) {
            const auto g = sqt::black_scholes_greeks_one(b.spots[j], b.strike[i], b.t[i],
                                                         b.vol[i], b.rate[i], b.q[i],
                                                         b.call[i] != 0);
            for (std::size_t f = 0; f < 12; ++f)
                all_same = all_same && same(buf.greek[f][i * b.ns() + j], field(g, f));
        }
    }
    CHECK(all_same);
    // The refused contract and the refused spots are NaN, the rest are not.
    CHECK(std::isnan(buf.greek[2][6 * b.ns() + 10]));
    CHECK(std::isnan(buf.greek[2][1 * b.ns() + 63]) && std::isnan(buf.greek[2][1 * b.ns() + 64]));
    CHECK(std::isfinite(buf.greek[2][1 * b.ns() + 62]) && std::isfinite(buf.greek[2][1 * b.ns() + 65]));
}

static void test_a_selection_writes_the_full_calls_doubles() {
    // Each selection -- one greek, the shapes callers use, and an arbitrary
    // mix read at run time -- writes exactly the full call's arrays for the
    // outputs it names, and nothing else (the others are NULL).
    const Book b;
    const std::size_t cells = b.nc() * b.ns();
    Buffers full(cells);
    b.grid(full.out(sqt::kGreeksAll | sqt::kGreekPriceFinite));
    std::vector<std::uint32_t> masks;
    for (std::uint32_t f = 0; f < 12; ++f) masks.push_back(1u << f);
    masks.push_back(sqt::kGreekGamma | sqt::kGreekPriceFinite);
    masks.push_back(sqt::kGreekDelta | sqt::kGreekPriceFinite);
    masks.push_back(sqt::kGreekVanna | sqt::kGreekD1 | sqt::kGreekCharm);
    masks.push_back(sqt::kGreekSpeed | sqt::kGreekVolga | sqt::kGreekPriceFinite);
    masks.push_back(sqt::kGreekPriceFinite);
    for (std::uint32_t mask : masks) {
        Buffers sel(cells);
        b.grid(sel.out(mask));
        bool ok = true;
        for (std::size_t f = 0; f < 12; ++f) {
            if (!(mask & (1u << f))) continue;
            ok = ok && std::memcmp(sel.greek[f].data(), full.greek[f].data(),
                                   cells * sizeof(double)) == 0;
        }
        if (mask & sqt::kGreekPriceFinite) ok = ok && sel.finite == full.finite;
        CHECK(ok);
    }
    // Without grid as well: one cell per contract, at its own spot.
    std::vector<double> s(b.nc(), 104.0);
    s[2] = kNaN;
    Buffers f1(b.nc()), g1(b.nc());
    sqt::black_scholes_greeks_batch(s.data(), b.nc(), b.strike.data(), b.t.data(),
                                    b.vol.data(), b.rate.data(), b.q.data(), b.call.data(),
                                    b.nc(), false, f1.out(sqt::kGreeksAll));
    sqt::black_scholes_greeks_batch(s.data(), b.nc(), b.strike.data(), b.t.data(),
                                    b.vol.data(), b.rate.data(), b.q.data(), b.call.data(),
                                    b.nc(), false,
                                    g1.out(sqt::kGreekDelta | sqt::kGreekPriceFinite));
    CHECK(std::memcmp(f1.greek[1].data(), g1.greek[1].data(), b.nc() * sizeof(double)) == 0);
}

static void test_price_finite_is_whether_the_price_is_finite() {
    // The flag is decided without forming the price: finite exactly when
    // spot * growth and strike * discount are, and d1 is not NaN. Each way
    // a price inside the domain can still fail is planted here, and the
    // flag must say what the price says -- whichever outputs ride with it.
    struct Cell { double s, k, t, v, r, q; };
    const Cell cells[] = {
        {100, 100, 0.5, 0.2, 0.03, 0.0},          // ordinary
        {100, 1e12, 77.7, 0.2, -9.0, 0.0},        // strike * discount past a double
        {1e12, 100, 77.7, 0.2, 0.0, -9.0},        // spot * growth past a double
        {100, 100, 1e-300, 1e-300, 0.0, 0.0},     // d1 = 0 / 0
        {101, 100, 1e-300, 1e-300, 0.0, 0.0},     // d1 = +inf: a finite price
        {99, 100, 1e-300, 1e-300, 0.0, 0.0},      // d1 = -inf: a finite price
        {1e-300, 1e12, 100.0, 100.0, 7.0, -7.0},  // extremes that still price
        {0.0, 100, 0.5, 0.2, 0.0, 0.0},           // outside the domain
        {100, 100, 0.5, 0.2, kNaN, 0.0},          // outside the domain
    };
    const std::size_t n = sizeof(cells) / sizeof(cells[0]);
    std::vector<double> s(n), k(n), t(n), v(n), r(n), q(n);
    for (std::size_t i = 0; i < n; ++i) {
        s[i] = cells[i].s; k[i] = cells[i].k; t[i] = cells[i].t;
        v[i] = cells[i].v; r[i] = cells[i].r; q[i] = cells[i].q;
    }
    int non_finite = 0;
    for (std::uint8_t c : {std::uint8_t{0}, std::uint8_t{1}}) {
        std::vector<std::uint8_t> call(n, c);
        for (std::uint32_t mask : {std::uint32_t{sqt::kGreekPriceFinite},
                                   std::uint32_t{sqt::kGreekGamma | sqt::kGreekPriceFinite},
                                   std::uint32_t{sqt::kGreekDelta | sqt::kGreekPriceFinite},
                                   std::uint32_t{sqt::kGreeksAll | sqt::kGreekPriceFinite}}) {
            Buffers buf(n);
            sqt::black_scholes_greeks_batch(s.data(), n, k.data(), t.data(), v.data(),
                                            r.data(), q.data(), call.data(), n, false,
                                            buf.out(mask));
            for (std::size_t i = 0; i < n; ++i) {
                const double price =
                    sqt::black_scholes_greeks_one(s[i], k[i], t[i], v[i], r[i], q[i], c != 0)
                        .price;
                CHECK((buf.finite[i] != 0) == std::isfinite(price));
                if (mask == sqt::kGreekPriceFinite && !std::isfinite(price)) ++non_finite;
            }
        }
    }
    // Cells 1-3 and the two outside the domain, as a call and as a put:
    // the planted failures are failures, so the agreement above means
    // something.
    CHECK(non_finite == 10);
}

// ── Implied volatility ──────────────────────────────────────────────────────

namespace {

// The square the solver used until the CHANGELOG entry of 2026-10-02: the C
// library's pow(v, 2.0), the way CPython evaluates `v**2`. Kept as the
// reference the change is measured against.
double pow_square(double v) {
    volatile double two = 2.0;
    return std::pow(v, two);
}

}  // namespace

static void test_the_solver_prices_a_volatility_as_the_greeks_do() {
    // The solver squares the volatility by multiplying, as the greeks do,
    // so its model price at any volatility is the greeks' price there to
    // the bit. Started AT the volatility a quote was priced at, Newton takes
    // a step of exactly zero: one iteration, that volatility back, and a
    // price error of exactly zero -- at every volatility, including those
    // whose pow square is not the correctly rounded one.
    const double s = 100.0, k = 104.0, t = 0.75, r = 0.03, q = 0.01;
    // An even grid, and every volatility of a fine scan whose pow square
    // misses the correctly rounded one on this C runtime.
    std::vector<double> vols;
    for (int i = 0; i < 2000; ++i) vols.push_back(0.05 + 2.5 * (i + 0.5) / 2000.0);
    int pow_differs = 0;
    for (int i = 0; i < 200000; ++i) {
        const double vol = 0.05 + 2.5 * i / 200000.0 + 1e-9 * std::sin(i);
        if (pow_square(vol) != vol * vol) {
            ++pow_differs;
            vols.push_back(vol);
        }
    }
    bool all_exact = true;
    for (double vol : vols) {
        for (bool call : {true, false}) {
            const double price = price_of(s, k, t, vol, r, q, call);
            sqt::ImpliedVolSettings settings;
            settings.initial_guess = vol;
            const auto res = sqt::implied_volatility_one(price, s, k, t, r, q, call, settings);
            all_exact = all_exact && res.reason == sqt::kIvSolved &&
                        res.method == sqt::kIvMethodNewton && res.iterations == 1 &&
                        res.vol == vol && res.price_error == 0.0;
        }
    }
    CHECK(all_exact);
    // On the Windows C runtime about 1 square in 2,000 differs; a correctly
    // rounded pow (glibc's) differs on none. Reported, not asserted.
    std::printf("pow(v, 2) != v * v for %d of 200000 volatilities on this runtime\n",
                pow_differs);
}

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

// ── The binomial lattice ────────────────────────────────────────────────────

namespace {

// The lattice as the numpy loop forms it: a fresh array per level, each
// node from the level below, maximum returning its second argument on a tie.
std::vector<double> lattice_reference(const std::vector<double>& up,
                                      const std::vector<double>& down, double spot,
                                      double strike, double sign, double p, double disc,
                                      bool american) {
    const std::size_t steps = up.size() - 1;
    auto max2 = [](double a, double b) { return (a > b || a != a) ? a : b; };
    auto node = [&](std::size_t level, std::size_t i) {
        return (spot * up[level - i]) * down[i];
    };
    std::vector<double> values(steps + 1), levels(6);
    for (std::size_t i = 0; i <= steps; ++i)
        values[i] = max2(sign * (node(steps, i) - strike), 0.0);
    const double q = 1.0 - p;
    for (std::size_t level = steps; level-- > 0;) {
        std::vector<double> next(level + 1);
        for (std::size_t i = 0; i <= level; ++i) {
            next[i] = disc * (p * values[i] + q * values[i + 1]);
            if (american) next[i] = max2(next[i], sign * (node(level, i) - strike));
        }
        values.swap(next);
        if (level <= 2)
            for (std::size_t i = 0; i <= level; ++i)
                levels[level * (level + 1) / 2 + i] = values[i];
    }
    return levels;
}

struct Crr {
    std::vector<double> up, down;
    double p, disc;
};

Crr crr(double t, double vol, double r, double q, std::size_t steps) {
    const double dt = t / static_cast<double>(steps);
    const double u = std::exp(vol * std::sqrt(dt));
    const double d = 1.0 / u;
    const double growth = std::exp((r - q) * dt);
    Crr c{std::vector<double>(steps + 1), std::vector<double>(steps + 1),
          (growth - d) / (u - d), std::exp(-r * dt)};
    for (std::size_t k = 0; k <= steps; ++k) {
        c.up[k] = std::pow(u, static_cast<double>(k));
        c.down[k] = std::pow(d, static_cast<double>(k));
    }
    return c;
}

std::vector<double> lattice(const Crr& c, double spot, double strike, double sign,
                            bool american) {
    std::vector<double> levels(6, kNaN);
    CHECK(sqt::binomial_lattice(c.up.data(), c.down.data(), c.up.size() - 1, spot,
                                strike, sign, c.p, c.disc, american, levels.data()));
    return levels;
}

}  // namespace

// The in-place, one-buffer kernel against a fresh array per level, every
// level 0-2 value bit for bit, at the strike (where a put's payoff is -0.0)
// and either side of it.
static void test_the_lattice_is_the_level_by_level_loop() {
    for (std::size_t steps : {3u, 4u, 10u, 200u, 1001u}) {
        for (double strike : {80.0, 100.0, 125.0}) {
            for (double sign : {1.0, -1.0}) {
                for (bool american : {false, true}) {
                    const Crr c = crr(0.75, 0.35, 0.04, 0.03, steps);
                    const auto got = lattice(c, 100.0, strike, sign, american);
                    const auto want = lattice_reference(c.up, c.down, 100.0, strike, sign,
                                                        c.p, c.disc, american);
                    for (std::size_t i = 0; i < 6; ++i) CHECK(same(got[i], want[i]));
                }
            }
        }
    }
}

// 2,000 steps: the European prices are within a cent of Black-Scholes, and
// an American put on a dividend payer is worth more than the European.
static void test_the_lattice_converges_and_prices_early_exercise() {
    const Crr c = crr(1.0, 0.3, 0.04, 0.03, 2000);
    const double euro_put = lattice(c, 100.0, 110.0, -1.0, false)[0];
    const double amer_put = lattice(c, 100.0, 110.0, -1.0, true)[0];
    const double euro_call = lattice(c, 100.0, 110.0, 1.0, false)[0];
    CHECK_NEAR(euro_put, price_of(100.0, 110.0, 1.0, 0.3, 0.04, 0.03, false), 0.01);
    CHECK_NEAR(euro_call, price_of(100.0, 110.0, 1.0, 0.3, 0.04, 0.03, true), 0.01);
    CHECK(amer_put > euro_put + 0.1);
}

int main() {
    test_hull_textbook_prices();
    test_greeks_match_the_library_units();
    test_greeks_are_the_derivatives_of_the_price();
    test_put_call_parity();
    test_outside_the_domain_is_nan_in_every_field();
    test_grid_is_every_contract_at_every_spot();
    test_a_large_batch_answers_what_one_contract_does();
    test_a_grid_longer_than_a_block_is_one_contract_at_a_time();
    test_a_selection_writes_the_full_calls_doubles();
    test_price_finite_is_whether_the_price_is_finite();
    test_the_solver_prices_a_volatility_as_the_greeks_do();
    test_round_trip_recovers_the_volatility();
    test_hull_solves_in_one_newton_step();
    test_a_price_at_intrinsic_is_a_ceiling();
    test_every_refusal_has_its_own_code();
    test_zero_and_negative_rates_and_yields_solve();
    test_an_empty_batch_is_a_no_op();
    test_the_lattice_is_the_level_by_level_loop();
    test_the_lattice_converges_and_prices_early_exercise();

    std::printf("%d/%d checks passed\n", g_tests_run - g_tests_failed, g_tests_run);
    return g_tests_failed == 0 ? 0 : 1;
}
