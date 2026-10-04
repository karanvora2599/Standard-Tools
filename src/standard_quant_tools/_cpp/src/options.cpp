#include "sqt/fp_contract.hpp"  // first: no contraction in this unit
#include "sqt/options.hpp"

#include "sqt/numerics.hpp"
#include "sqt/omp_policy.hpp"
#include "sqt/platform.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace sqt {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// The constants analysis/options.py solves with. Restated rather than
// shared because the Python module is the contract; tests pin the two
// solvers to each other result for result.
constexpr double kVegaFloor = 1e-8;
constexpr double kBoundTolerance = 1e-9;
constexpr double kSigmaLow = 1e-6;
constexpr double kSigmaHigh = 5.0;
constexpr int kBisectionIterations = 200;
// The bisection may exit on the PRICE tolerance only once its bracket is
// narrower than this, so a flat price in a wide bracket cannot stop it.
constexpr double kPriceExitWidth = 1e-4;

// The pricing domain the Python validators enforce: a magnitude past these
// is a unit error, and exp(rate x time) overflows a float near 710.
constexpr double kMaxPrice = 1e12;
constexpr double kMaxTime = 100.0;
constexpr double kMaxVolatility = 100.0;
constexpr double kMaxRate = 10.0;
constexpr double kMaxExponent = 700.0;

// Work units for omp_policy::worth_parallel, measured against the unit the
// other kernels use (about one arithmetic pass over one bar). One contract
// valued once costs a log, three exps and two erfs -- a few dozen of those
// units; one implied-volatility solve is ten or so valuations with a vega.
// A 476-contract chain therefore goes parallel for the solve and stays
// serial for a single pass of greeks, where thread start-up would cost more
// than the pass itself.
constexpr std::size_t kGreeksWork = 50;
constexpr std::size_t kImpliedVolWork = 500;

// `math.sqrt(2.0)` and `math.sqrt(2.0 * math.pi)` from _special: sqrt is
// correctly rounded everywhere, so these are the same doubles.
const double kSqrt2 = std::sqrt(2.0);
const double kSqrt2Pi = std::sqrt(2.0 * 3.141592653589793);

// _special.norm_cdf and norm_pdf, as Python evaluates them. The CDF is the
// erf form, not erfc: it is what every scalar in the library calls, and it
// rounds differently in the tails, which is exactly where a deep
// out-of-the-money quote is solved.
inline double norm_cdf(double x) { return 0.5 * (1.0 + std::erf(x / kSqrt2)); }
inline double norm_pdf(double x) { return std::exp(-0.5 * x * x) / kSqrt2Pi; }

inline bool finite_in(double v, double low_exclusive, double high) {
    return numerics::is_finite(v) && v > low_exclusive && v <= high;
}

inline bool rates_ok(double rate, double q, double t) {
    return numerics::is_finite(rate) && std::abs(rate) <= kMaxRate &&
           numerics::is_finite(q) && std::abs(q) <= kMaxRate &&
           std::abs(rate * t) <= kMaxExponent && std::abs(q * t) <= kMaxExponent;
}

/** One contract's volatility-independent terms, computed once per solve.
 *  Recomputing them per iteration, as the scalar does, gives the same
 *  doubles: every one is a deterministic function of the inputs. */
struct Contract {
    double spot;
    double strike;
    double t;
    double rate;
    double q;
    double sqrt_t;
    double log_moneyness;
    double disc_q;
    double disc_r;
    bool call;
};

// analysis.options._d1_d2. The volatility is squared by multiplying, as
// that function, derivatives.option_greeks and pricing all do: v * v is the
// correctly rounded square, and the C library's pow(v, 2.0) is not always
// it (on the Windows CRT, about 1 square in 2,000 differs in the last bit).
// Until the CHANGELOG entry of 2026-10-02 this squared through pow, because
// options.py did; both changed together, so the three paths still agree.
inline double iv_d1(const Contract& c, double sigma) {
    return (c.log_moneyness + (c.rate - c.q + 0.5 * sigma * sigma) * c.t) /
           (sigma * c.sqrt_t);
}

// analysis.options.black_scholes_price at `sigma`.
inline double iv_price(const Contract& c, double sigma, double d1) {
    const double d2 = d1 - sigma * c.sqrt_t;
    if (c.call) {
        return c.spot * c.disc_q * norm_cdf(d1) - c.strike * c.disc_r * norm_cdf(d2);
    }
    return c.strike * c.disc_r * norm_cdf(-d2) - c.spot * c.disc_q * norm_cdf(-d1);
}

// black_scholes_price(sigma) - option_price, or NaN when the model price is
// not finite (the scalar's _require_finite_price refusal).
inline double price_diff(const Contract& c, double sigma, double price, bool& priceable) {
    const double model = iv_price(c, sigma, iv_d1(c, sigma));
    if (!numerics::is_finite(model)) {
        priceable = false;
        return kNaN;
    }
    return model - price;
}

ImpliedVolResult refused(std::int8_t reason) {
    ImpliedVolResult r;
    r.vol = kNaN;
    r.price_error = kNaN;
    r.iterations = 0;
    r.method = kIvMethodNone;
    r.reason = reason;
    r.converged = false;
    r.at_bound = false;
    return r;
}

ImpliedVolResult solved(const Contract& c, double price, double sigma, bool converged,
                        int iterations, std::int8_t method, bool at_bound) {
    bool priceable = true;
    const double diff = price_diff(c, sigma, price, priceable);
    if (!priceable) return refused(kIvNotPriceable);
    ImpliedVolResult r;
    r.vol = sigma;
    r.price_error = std::abs(diff);
    r.iterations = static_cast<std::int32_t>(iterations);
    r.method = method;
    r.reason = kIvSolved;
    r.converged = converged;
    r.at_bound = at_bound;
    return r;
}

}  // namespace

SQT_NOINLINE ImpliedVolResult implied_volatility_one(double price, double spot,
                                                     double strike,
                                                     double time_to_expiry,
                                                     double rate, double dividend_yield,
                                                     bool is_call,
                                                     const ImpliedVolSettings& settings) {
    // The scalar's order: a non-positive price is refused before anything
    // about the contract is looked at.
    if (price <= 0.0) return refused(kIvPriceNotPositive);
    if (!finite_in(spot, 0.0, kMaxPrice) || !finite_in(strike, 0.0, kMaxPrice) ||
        !finite_in(time_to_expiry, 0.0, kMaxTime) ||
        !rates_ok(rate, dividend_yield, time_to_expiry)) {
        return refused(kIvInvalidInput);
    }

    Contract c;
    c.spot = spot;
    c.strike = strike;
    c.t = time_to_expiry;
    c.rate = rate;
    c.q = dividend_yield;
    c.call = is_call;
    c.sqrt_t = std::sqrt(time_to_expiry);
    c.log_moneyness = std::log(spot / strike);
    c.disc_q = std::exp(-dividend_yield * time_to_expiry);
    c.disc_r = std::exp(-rate * time_to_expiry);
    // A discounted strike or spot past a double is a rate-unit error the
    // scalar only discovers as a non-finite price mid-solve.
    if (!numerics::is_finite(spot * c.disc_q) || !numerics::is_finite(strike * c.disc_r)) {
        return refused(kIvNotPriceable);
    }
    if (!numerics::is_finite(price)) return refused(kIvPriceNotFinite);

    double lower = 0.0;
    double upper = 0.0;
    if (is_call) {
        lower = std::max(spot * c.disc_q - strike * c.disc_r, 0.0);
        upper = spot * c.disc_q;
    } else {
        lower = std::max(strike * c.disc_r - spot * c.disc_q, 0.0);
        upper = strike * c.disc_r;
    }
    // Equality within the tolerance is INSIDE the bound: the pricer itself
    // produces a deep-in-the-money price bit-for-bit equal to intrinsic.
    const double slack = kBoundTolerance * std::max(std::abs(upper), 1.0);
    if (price < lower - slack) return refused(kIvBelowLowerBound);
    if (price > upper + slack) return refused(kIvAboveUpperBound);
    const bool at_bound = price - lower <= slack;

    bool priceable = true;

    // ── Newton, converged on the step it took ────────────────────────────
    double sigma = settings.initial_guess;
    // At intrinsic every small volatility prices the same; Newton has
    // nothing to divide by and the bisection below finds the ceiling.
    const int newton_iterations = at_bound ? 0 : settings.max_iterations;
    for (int i = 0; i < newton_iterations; ++i) {
        const double d1 = iv_d1(c, sigma);
        const double model = iv_price(c, sigma, d1);
        if (!numerics::is_finite(model)) return refused(kIvNotPriceable);
        const double diff = model - price;
        // analysis.options.black_scholes_greeks' raw vega.
        const double vega = c.spot * c.disc_q * norm_pdf(d1) * c.sqrt_t;
        if (vega < kVegaFloor) break;
        const double step = diff / vega;
        const double candidate = sigma - step;
        if (candidate <= 0.0 || candidate > kSigmaHigh) break;
        sigma = candidate;
        if (std::abs(step) < settings.tol_sigma) {
            return solved(c, price, sigma, true, i + 1, kIvMethodNewton, at_bound);
        }
    }

    // ── Bisection fallback ──────────────────────────────────────────────
    double lo = kSigmaLow;
    double hi = kSigmaHigh;
    if (at_bound) {
        // The largest volatility that still reproduces an intrinsic price
        // to within the bound tolerance.
        int i = 0;
        for (; i < kBisectionIterations; ++i) {
            const double mid = 0.5 * (lo + hi);
            const double diff = price_diff(c, mid, price, priceable);
            if (!priceable) return refused(kIvNotPriceable);
            if (std::abs(diff) <= slack) {
                lo = mid;
            } else {
                hi = mid;
            }
            if ((hi - lo) < settings.tol_sigma) break;
        }
        // The scalar reports `i + 1` whether it broke out or ran out.
        const int used = (i < kBisectionIterations) ? i + 1 : kBisectionIterations;
        return solved(c, price, lo, true, used, kIvMethodBisection, at_bound);
    }
    double diff_lo = price_diff(c, lo, price, priceable);
    const double diff_hi = price_diff(c, hi, price, priceable);
    if (!priceable) return refused(kIvNotPriceable);
    if (diff_lo == 0.0) return solved(c, price, lo, true, 0, kIvMethodBisection, at_bound);
    if (diff_hi == 0.0) return solved(c, price, hi, true, 0, kIvMethodBisection, at_bound);
    if (diff_lo * diff_hi > 0.0) return refused(kIvNoRootInBracket);
    double mid = lo;
    for (int i = 0; i < kBisectionIterations; ++i) {
        mid = 0.5 * (lo + hi);
        const double diff_mid = price_diff(c, mid, price, priceable);
        if (!priceable) return refused(kIvNotPriceable);
        if ((hi - lo) < settings.tol_sigma ||
            (std::abs(diff_mid) < settings.tol && (hi - lo) < kPriceExitWidth)) {
            return solved(c, price, mid, true, i + 1, kIvMethodBisection, at_bound);
        }
        if (diff_lo * diff_mid < 0.0) {
            hi = mid;
        } else {
            lo = mid;
            diff_lo = diff_mid;
        }
    }
    return solved(c, price, mid, false, kBisectionIterations, kIvMethodBisection,
                  at_bound);
}

void implied_volatility_batch(const double* price, const double* spot,
                              const double* strike, const double* time_to_expiry,
                              const double* rate, const double* dividend_yield,
                              const std::uint8_t* is_call, std::size_t n,
                              const ImpliedVolSettings& settings,
                              ImpliedVolBatchOut out) {
    // Each contract writes only its own slot, so the loop needs no
    // reduction and the answer is independent of the thread count.
    const sqt::omp_policy::parallel_call omp_call(n, kImpliedVolWork, sqt::omp_policy::cost::implied_volatility);
    #pragma omp parallel for schedule(guided) \
        if (omp_call.parallel()) \
        num_threads(sqt::omp_policy::max_threads() > 0 \
                        ? sqt::omp_policy::max_threads() : omp_get_max_threads())
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(n); ++k) {
        const auto i = static_cast<std::size_t>(k);
        const ImpliedVolResult r =
            implied_volatility_one(price[i], spot[i], strike[i], time_to_expiry[i],
                                   rate[i], dividend_yield[i], is_call[i] != 0,
                                   settings);
        out.vol[i] = r.vol;
        out.price_error[i] = r.price_error;
        out.iterations[i] = r.iterations;
        out.method[i] = r.method;
        out.reason[i] = r.reason;
        out.converged[i] = static_cast<std::uint8_t>(r.converged ? 1 : 0);
        out.at_bound[i] = static_cast<std::uint8_t>(r.at_bound ? 1 : 0);
    }
}

// Out of line so no caller's loop can be auto-vectorised around it: a
// vectorised exp or erf is not guaranteed to round like the scalar C
// library call the Python formulas make.
SQT_NOINLINE BlackScholesGreeks black_scholes_greeks_one(double spot, double strike,
                                                         double time_to_expiry,
                                                         double volatility, double rate,
                                                         double dividend_yield,
                                                         bool is_call) {
    BlackScholesGreeks g;
    if (!finite_in(spot, 0.0, kMaxPrice) || !finite_in(strike, 0.0, kMaxPrice) ||
        !finite_in(time_to_expiry, 0.0, kMaxTime) ||
        !finite_in(volatility, 0.0, kMaxVolatility) ||
        !rates_ok(rate, dividend_yield, time_to_expiry)) {
        g.price = g.delta = g.gamma = g.vega = g.theta = g.rho = kNaN;
        g.vanna = g.volga = g.charm = g.speed = g.d1 = g.d2 = kNaN;
        return g;
    }
    // analysis.derivatives.option_greeks, term for term. `vol * vol` here,
    // not pow: that function and pricing._black_scholes multiply.
    const double t = time_to_expiry;
    const double vol = volatility;
    const double q = dividend_yield;
    const double sqrt_t = std::sqrt(t);
    const double growth = std::exp(-q * t);
    const double discount = std::exp(-rate * t);
    const double d1 =
        (std::log(spot / strike) + (rate - q + 0.5 * vol * vol) * t) / (vol * sqrt_t);
    const double d2 = d1 - vol * sqrt_t;
    const double pdf_d1 = norm_pdf(d1);

    double price = 0.0;
    double delta = 0.0;
    double rho = 0.0;
    double theta_raw = 0.0;
    double charm_raw = 0.0;
    if (is_call) {
        const double n_d1 = norm_cdf(d1);
        const double n_d2 = norm_cdf(d2);
        // pricing._black_scholes' price, which option_greeks reports.
        price = spot * growth * n_d1 - strike * discount * n_d2;
        delta = growth * n_d1;
        rho = strike * t * discount * n_d2;
        theta_raw = -spot * pdf_d1 * vol * growth / (2.0 * sqrt_t) +
                    q * spot * growth * n_d1 - rate * strike * discount * n_d2;
        charm_raw = -growth * (pdf_d1 * (2.0 * (rate - q) * t - d2 * vol * sqrt_t) /
                                   (2.0 * t * vol * sqrt_t) -
                               q * n_d1);
    } else {
        const double n_md1 = norm_cdf(-d1);
        const double n_md2 = norm_cdf(-d2);
        price = strike * discount * n_md2 - spot * growth * n_md1;
        delta = -growth * n_md1;
        rho = -strike * t * discount * n_md2;
        theta_raw = -spot * pdf_d1 * vol * growth / (2.0 * sqrt_t) -
                    q * spot * growth * n_md1 + rate * strike * discount * n_md2;
        charm_raw = -growth * (pdf_d1 * (2.0 * (rate - q) * t - d2 * vol * sqrt_t) /
                                   (2.0 * t * vol * sqrt_t) +
                               q * n_md1);
    }
    const double gamma = growth * pdf_d1 / (spot * vol * sqrt_t);
    const double vega_raw = spot * growth * pdf_d1 * sqrt_t;
    // Shared by calls and puts: parity is linear in spot and free of vol.
    const double vanna_raw = -growth * pdf_d1 * d2 / vol;
    const double volga_raw = vega_raw * d1 * d2 / vol;
    const double speed = -gamma / spot * (d1 / (vol * sqrt_t) + 1.0);

    g.price = price;
    g.delta = delta;
    g.gamma = gamma;
    g.vega = vega_raw / 100.0;
    g.theta = theta_raw / 365.0;
    g.rho = rho / 100.0;
    g.vanna = vanna_raw / 100.0;
    g.volga = volga_raw / 10000.0;
    g.charm = charm_raw / 365.0;
    g.speed = speed;
    g.d1 = d1;
    g.d2 = d2;
    return g;
}

namespace {

// ── The greeks over a batch, output by output ───────────────────────────
//
// black_scholes_greeks_one above is the definition: the function the scalar
// path and the tests call, and the arithmetic every cell below repeats. The
// batch restructures it three ways, none of which changes a double:
//
// - the terms that depend only on the contract -- sqrt T, both discount
//   factors, the drift -- are formed once per contract (per block of spots,
//   under `grid`) rather than once per cell. Each is the same expression on
//   the same inputs, so the same double;
// - the row-major cell index takes one division per block of spots, not a
//   division and a remainder per cell;
// - an output whose pointer is NULL is neither formed nor written, nor is
//   any transcendental only it needs: gamma takes no erf, delta one, where
//   the full set takes two.
//
// Every expression is written as black_scholes_greeks_one writes it, and a
// hoisted term stands in only for the LEFT-MOST product or sum of an
// expression, which is the sub-expression C++ evaluates first: a hoisted
// `strike * discount` replaces the head of `strike * discount * n_d2`, never
// a middle factor. So `spot * vol * sqrt_t` stays as it is -- (spot * vol) *
// sqrt_t is not spot * (vol * sqrt_t) in floating point.

// The inner loops call log, exp and erf, and a vectorised libm need not
// round like the scalar calls black_scholes_greeks_one makes. This build
// does not vectorise them (no /fp:fast, no -ffast-math, and the loop
// branches); the pragma makes that a statement rather than an observation.
#if defined(__clang__)
#define SQT_SCALAR_LOOP _Pragma("clang loop vectorize(disable)")
#elif defined(_MSC_VER)
#define SQT_SCALAR_LOOP __pragma(loop(no_vector))
#else
#define SQT_SCALAR_LOOP
#endif

// The contract terms and the cell loop are inlined into the task loop.
// Without `grid` a task is one cell, and as a call per task, with the terms
// handed over through memory, the full set ran 12% slower than the loop of
// black_scholes_greeks_one it replaces (measured, MSVC 19.44); inlined, it
// runs faster than that loop.
#if defined(_MSC_VER) && !defined(__clang__)
#define SQT_INLINE_ALWAYS __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define SQT_INLINE_ALWAYS inline __attribute__((always_inline))
#else
#define SQT_INLINE_ALWAYS inline
#endif

// Spots per task under `grid`. The contract terms (two exps and a sqrt) are
// amortised over this many cells, and a book of a few contracts on a long
// spot axis still splits across threads.
constexpr std::size_t kSpotBlock = 64;

// What each intermediate is needed for.
constexpr std::uint32_t kNeedN1 = kGreekPrice | kGreekDelta | kGreekTheta | kGreekCharm;
constexpr std::uint32_t kNeedN2 = kGreekPrice | kGreekRho | kGreekTheta;
constexpr std::uint32_t kNeedPdf = kGreekGamma | kGreekVega | kGreekTheta | kGreekVanna |
                                   kGreekVolga | kGreekCharm | kGreekSpeed;
constexpr std::uint32_t kNeedGrowth = kNeedPdf | kGreekPrice | kGreekDelta |
                                      kGreekPriceFinite;
constexpr std::uint32_t kNeedDiscount = kGreekPrice | kGreekRho | kGreekTheta |
                                        kGreekPriceFinite;

/// The outputs one call writes. `kFixed` is a selection known at compile
/// time -- the ones callers ask for: everything, gamma, delta, the price --
/// so every test below folds away; 0 reads `mask` at run time.
template <std::uint32_t kFixed>
struct Wanted {
    std::uint32_t mask;
    bool operator()(std::uint32_t bits) const {
        return ((kFixed != 0 ? kFixed : mask) & bits) != 0;
    }
};

/// |x| <= DBL_MAX: false for NaN and both infinities, without the CRT call
/// std::isfinite compiles to under MSVC.
inline bool finite_double(double x) {
    return std::abs(x) <= std::numeric_limits<double>::max();
}

/// One contract's spot-independent terms, each named for the expression of
/// black_scholes_greeks_one it is the head of.
struct GreekContract {
    bool ok;    // inside the domain
    bool call;
    double strike;
    double t;
    double vol;
    double rate;
    double q;
    double sqrt_t;               // std::sqrt(t)
    double growth;               // std::exp(-q * t)
    double discount;             // std::exp(-rate * t)
    double drift;                // (rate - q + 0.5 * vol * vol) * t, in d1
    double vol_sqrt_t;           // vol * sqrt_t, d1's denominator
    double strike_discount;      // strike * discount, in the price
    double rho_factor;           // strike * t * discount (a put: -strike * ...)
    double rate_strike_discount; // rate * strike * discount, in theta
    double two_sqrt_t;           // 2.0 * sqrt_t, theta's denominator
    double carry;                // 2.0 * (rate - q) * t, in charm
    double charm_den;            // 2.0 * t * vol * sqrt_t, charm's denominator
};

template <std::uint32_t kFixed>
SQT_INLINE_ALWAYS GreekContract contract_terms(Wanted<kFixed> want, double strike,
                                               double t, double vol, double rate,
                                               double q, bool call) {
    GreekContract c{};
    // black_scholes_greeks_one's domain test, less the spot (tested per
    // cell), written as comparisons alone: every bound is finite, so NaN and
    // both infinities fail them exactly as they fail std::isfinite, which
    // MSVC compiles to a C runtime call.
    c.ok = strike > 0.0 && strike <= kMaxPrice && t > 0.0 && t <= kMaxTime &&
           vol > 0.0 && vol <= kMaxVolatility && std::abs(rate) <= kMaxRate &&
           std::abs(q) <= kMaxRate && std::abs(rate * t) <= kMaxExponent &&
           std::abs(q * t) <= kMaxExponent;
    if (!c.ok) return c;
    c.call = call;
    c.strike = strike;
    c.t = t;
    c.vol = vol;
    c.rate = rate;
    c.q = q;
    c.sqrt_t = std::sqrt(t);
    c.growth = want(kNeedGrowth) ? std::exp(-q * t) : 0.0;
    c.discount = want(kNeedDiscount) ? std::exp(-rate * t) : 0.0;
    c.drift = (rate - q + 0.5 * vol * vol) * t;
    c.vol_sqrt_t = vol * c.sqrt_t;
    c.strike_discount = strike * c.discount;
    c.rho_factor = call ? strike * t * c.discount : -strike * t * c.discount;
    c.rate_strike_discount = rate * strike * c.discount;
    c.two_sqrt_t = 2.0 * c.sqrt_t;
    c.carry = 2.0 * (rate - q) * t;
    c.charm_den = 2.0 * t * vol * c.sqrt_t;
    return c;
}

template <std::uint32_t kFixed>
inline void nan_cell(Wanted<kFixed> want, const BlackScholesGreeksOut& out,
                     std::size_t at) {
    if (want(kGreekPrice)) out.price[at] = kNaN;
    if (want(kGreekDelta)) out.delta[at] = kNaN;
    if (want(kGreekGamma)) out.gamma[at] = kNaN;
    if (want(kGreekVega)) out.vega[at] = kNaN;
    if (want(kGreekTheta)) out.theta[at] = kNaN;
    if (want(kGreekRho)) out.rho[at] = kNaN;
    if (want(kGreekVanna)) out.vanna[at] = kNaN;
    if (want(kGreekVolga)) out.volga[at] = kNaN;
    if (want(kGreekCharm)) out.charm[at] = kNaN;
    if (want(kGreekSpeed)) out.speed[at] = kNaN;
    if (want(kGreekD1)) out.d1[at] = kNaN;
    if (want(kGreekD2)) out.d2[at] = kNaN;
    if (want(kGreekPriceFinite)) out.price_finite[at] = 0;
}

/// One contract inside the domain at `n` consecutive spots, written from
/// cell `at` on. The side (`kCall`) is a template argument, so the choice
/// between the call and the put expressions is made once per contract
/// rather than once per greek.
template <std::uint32_t kFixed, bool kCall>
SQT_INLINE_ALWAYS void greek_side(Wanted<kFixed> want, const GreekContract& c,
                                  const double* spot, std::size_t n, std::size_t at,
                                  const BlackScholesGreeksOut& out) {
    const bool strike_discount_finite = finite_double(c.strike_discount);
    SQT_SCALAR_LOOP
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t cell = at + j;
        const double s = spot[j];
        // finite_in(s, 0, kMaxPrice): NaN and both infinities fail these.
        if (!(s > 0.0 && s <= kMaxPrice)) {
            nan_cell(want, out, cell);
            continue;
        }
        const double d1 = (std::log(s / c.strike) + c.drift) / c.vol_sqrt_t;
        const double d2 = d1 - c.vol_sqrt_t;
        const double spot_growth = s * c.growth;  // the head of the price and of vega
        const double pdf_d1 = want(kNeedPdf) ? norm_pdf(d1) : 0.0;
        double n1 = 0.0;  // N(d1) for a call, N(-d1) for a put
        double n2 = 0.0;  // N(d2) for a call, N(-d2) for a put
        if (want(kNeedN1)) n1 = kCall ? norm_cdf(d1) : norm_cdf(-d1);
        if (want(kNeedN2)) n2 = kCall ? norm_cdf(d2) : norm_cdf(-d2);

        if (want(kGreekPrice)) {
            out.price[cell] = kCall ? spot_growth * n1 - c.strike_discount * n2
                                    : c.strike_discount * n2 - spot_growth * n1;
        }
        if (want(kGreekDelta)) out.delta[cell] = kCall ? c.growth * n1 : -c.growth * n1;
        if (want(kGreekRho)) out.rho[cell] = c.rho_factor * n2 / 100.0;
        if (want(kGreekTheta)) {
            const double decay = -s * pdf_d1 * c.vol * c.growth / c.two_sqrt_t;
            const double theta_raw =
                kCall ? decay + c.q * s * c.growth * n1 - c.rate_strike_discount * n2
                      : decay - c.q * s * c.growth * n1 + c.rate_strike_discount * n2;
            out.theta[cell] = theta_raw / 365.0;
        }
        if (want(kGreekCharm)) {
            const double shape = pdf_d1 * (c.carry - d2 * c.vol * c.sqrt_t) / c.charm_den;
            const double charm_raw = kCall ? -c.growth * (shape - c.q * n1)
                                           : -c.growth * (shape + c.q * n1);
            out.charm[cell] = charm_raw / 365.0;
        }
        if (want(kGreekGamma | kGreekSpeed)) {
            const double gamma = c.growth * pdf_d1 / (s * c.vol * c.sqrt_t);
            if (want(kGreekGamma)) out.gamma[cell] = gamma;
            if (want(kGreekSpeed)) {
                out.speed[cell] = -gamma / s * (d1 / c.vol_sqrt_t + 1.0);
            }
        }
        if (want(kGreekVega | kGreekVolga)) {
            const double vega_raw = spot_growth * pdf_d1 * c.sqrt_t;
            if (want(kGreekVega)) out.vega[cell] = vega_raw / 100.0;
            if (want(kGreekVolga)) out.volga[cell] = vega_raw * d1 * d2 / c.vol / 10000.0;
        }
        if (want(kGreekVanna)) out.vanna[cell] = -c.growth * pdf_d1 * d2 / c.vol / 100.0;
        if (want(kGreekD1)) out.d1[cell] = d1;
        if (want(kGreekD2)) out.d2[cell] = d2;
        // The price is (spot * growth) * N - (strike * discount) * N' for a
        // call, the mirror for a put, with each N in [0, 1] or NaN exactly
        // when d1 is. So it is finite exactly when both heads are and d1 is
        // not NaN: an infinite head gives inf or inf * 0 = NaN, a NaN d1 a
        // NaN, and two finite non-negative terms cannot differ by more than
        // a double holds. No erf is needed to know.
        if (want(kGreekPriceFinite)) {
            out.price_finite[cell] = static_cast<std::uint8_t>(
                strike_discount_finite && finite_double(spot_growth) && d1 == d1 ? 1 : 0);
        }
    }
}

/// One contract at `n` consecutive spots: NaN outside the domain, else the
/// call's or the put's arithmetic.
template <std::uint32_t kFixed>
SQT_INLINE_ALWAYS void greek_cells(Wanted<kFixed> want, const GreekContract& c,
                                   const double* spot, std::size_t n, std::size_t at,
                                   const BlackScholesGreeksOut& out) {
    if (!c.ok) {
        for (std::size_t j = 0; j < n; ++j) nan_cell(want, out, at + j);
    } else if (c.call) {
        greek_side<kFixed, true>(want, c, spot, n, at, out);
    } else {
        greek_side<kFixed, false>(want, c, spot, n, at, out);
    }
}

template <std::uint32_t kFixed>
void greeks_batch(Wanted<kFixed> want, const double* spot, std::size_t n_spots,
                  const double* strike, const double* time_to_expiry,
                  const double* volatility, const double* rate,
                  const double* dividend_yield, const std::uint8_t* is_call,
                  std::size_t n_contracts, bool grid, const BlackScholesGreeksOut& out) {
    const std::size_t total = grid ? n_contracts * n_spots : n_contracts;
    const std::size_t columns = grid ? n_spots : 1;
    // A task is one contract at up to kSpotBlock consecutive spots; without
    // `grid` it is one contract at its own spot.
    const std::size_t block = grid ? kSpotBlock : 1;
    const std::size_t per_row = (columns + block - 1) / block;
    const std::size_t tasks = n_contracts * per_row;
    // Each task writes only its own cells, so the answer is independent of
    // the thread count and of how the tasks are split.
    const sqt::omp_policy::parallel_call omp_call(total, kGreeksWork, sqt::omp_policy::cost::greeks);
    #pragma omp parallel for schedule(guided) \
        if (omp_call.parallel()) \
        num_threads(sqt::omp_policy::max_threads() > 0 \
                        ? sqt::omp_policy::max_threads() : omp_get_max_threads())
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(tasks); ++k) {
        const auto task = static_cast<std::size_t>(k);
        const std::size_t i = per_row == 1 ? task : task / per_row;  // contract
        const std::size_t j0 = per_row == 1 ? 0 : (task - i * per_row) * block;
        const std::size_t n = std::min(block, columns - j0);
        const GreekContract c =
            contract_terms(want, strike[i], time_to_expiry[i], volatility[i], rate[i],
                           dividend_yield[i], is_call[i] != 0);
        greek_cells(want, c, grid ? spot + j0 : spot + i, n, i * columns + j0, out);
    }
}

}  // namespace

void black_scholes_greeks_batch(const double* spot, std::size_t n_spots,
                                const double* strike, const double* time_to_expiry,
                                const double* volatility, const double* rate,
                                const double* dividend_yield,
                                const std::uint8_t* is_call, std::size_t n_contracts,
                                bool grid, BlackScholesGreeksOut out) {
    std::uint32_t mask = 0;
    if (out.price) mask |= kGreekPrice;
    if (out.delta) mask |= kGreekDelta;
    if (out.gamma) mask |= kGreekGamma;
    if (out.vega) mask |= kGreekVega;
    if (out.theta) mask |= kGreekTheta;
    if (out.rho) mask |= kGreekRho;
    if (out.vanna) mask |= kGreekVanna;
    if (out.volga) mask |= kGreekVolga;
    if (out.charm) mask |= kGreekCharm;
    if (out.speed) mask |= kGreekSpeed;
    if (out.d1) mask |= kGreekD1;
    if (out.d2) mask |= kGreekD2;
    if (out.price_finite) mask |= kGreekPriceFinite;
    if (mask == 0) return;
    // The selections callers make, compiled for that selection alone; any
    // other is read at run time. Each writes the same doubles.
    switch (mask) {
        case kGreeksAll:
            greeks_batch(Wanted<kGreeksAll>{mask}, spot, n_spots, strike, time_to_expiry,
                         volatility, rate, dividend_yield, is_call, n_contracts, grid, out);
            break;
        case kGreekGamma:
            greeks_batch(Wanted<kGreekGamma>{mask}, spot, n_spots, strike, time_to_expiry,
                         volatility, rate, dividend_yield, is_call, n_contracts, grid, out);
            break;
        case kGreekGamma | kGreekPriceFinite:
            greeks_batch(Wanted<kGreekGamma | kGreekPriceFinite>{mask}, spot, n_spots,
                         strike, time_to_expiry, volatility, rate, dividend_yield,
                         is_call, n_contracts, grid, out);
            break;
        case kGreekDelta | kGreekPriceFinite:
            greeks_batch(Wanted<kGreekDelta | kGreekPriceFinite>{mask}, spot, n_spots,
                         strike, time_to_expiry, volatility, rate, dividend_yield,
                         is_call, n_contracts, grid, out);
            break;
        case kGreekPrice:
            greeks_batch(Wanted<kGreekPrice>{mask}, spot, n_spots, strike, time_to_expiry,
                         volatility, rate, dividend_yield, is_call, n_contracts, grid, out);
            break;
        default:
            greeks_batch(Wanted<0>{mask}, spot, n_spots, strike, time_to_expiry,
                         volatility, rate, dividend_yield, is_call, n_contracts, grid, out);
            break;
    }
}

}  // namespace sqt
