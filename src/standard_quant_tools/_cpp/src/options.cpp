#include "sqt/options.hpp"

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

// Python's `v**2`, which CPython computes with the C library's pow(v, 2.0).
// That is NOT always v * v -- measured, 112 of 200,000 volatilities in
// [0, 5] differ in the last bit -- and options.py squares the volatility
// this way while derivatives.py multiplies. The exponent is read through a
// volatile so the compiler cannot rewrite the call into a multiply.
inline double py_square(double v) {
    volatile double two = 2.0;
    return std::pow(v, two);
}

inline bool finite_in(double v, double low_exclusive, double high) {
    return std::isfinite(v) && v > low_exclusive && v <= high;
}

inline bool rates_ok(double rate, double q, double t) {
    return std::isfinite(rate) && std::abs(rate) <= kMaxRate &&
           std::isfinite(q) && std::abs(q) <= kMaxRate &&
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

// analysis.options._d1_d2, with the volatility squared through pow.
inline double iv_d1(const Contract& c, double sigma) {
    return (c.log_moneyness + (c.rate - c.q + 0.5 * py_square(sigma)) * c.t) /
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
    if (!std::isfinite(model)) {
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
    if (!std::isfinite(spot * c.disc_q) || !std::isfinite(strike * c.disc_r)) {
        return refused(kIvNotPriceable);
    }
    if (!std::isfinite(price)) return refused(kIvPriceNotFinite);

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
        if (!std::isfinite(model)) return refused(kIvNotPriceable);
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
    #pragma omp parallel for schedule(guided) \
        if (sqt::omp_policy::worth_parallel(n, kImpliedVolWork)) \
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

void black_scholes_greeks_batch(const double* spot, std::size_t n_spots,
                                const double* strike, const double* time_to_expiry,
                                const double* volatility, const double* rate,
                                const double* dividend_yield,
                                const std::uint8_t* is_call, std::size_t n_contracts,
                                bool grid, BlackScholesGreeksOut out) {
    const std::size_t total = grid ? n_contracts * n_spots : n_contracts;
    const std::size_t columns = grid ? n_spots : 1;
    #pragma omp parallel for schedule(guided) \
        if (sqt::omp_policy::worth_parallel(total, kGreeksWork)) \
        num_threads(sqt::omp_policy::max_threads() > 0 \
                        ? sqt::omp_policy::max_threads() : omp_get_max_threads())
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(total); ++k) {
        const auto idx = static_cast<std::size_t>(k);
        const std::size_t i = grid ? idx / columns : idx;  // contract
        const std::size_t j = grid ? idx % columns : idx;  // spot
        const BlackScholesGreeks g = black_scholes_greeks_one(
            spot[j], strike[i], time_to_expiry[i], volatility[i], rate[i],
            dividend_yield[i], is_call[i] != 0);
        out.price[idx] = g.price;
        out.delta[idx] = g.delta;
        out.gamma[idx] = g.gamma;
        out.vega[idx] = g.vega;
        out.theta[idx] = g.theta;
        out.rho[idx] = g.rho;
        out.vanna[idx] = g.vanna;
        out.volga[idx] = g.volga;
        out.charm[idx] = g.charm;
        out.speed[idx] = g.speed;
        out.d1[idx] = g.d1;
        out.d2[idx] = g.d2;
    }
}

}  // namespace sqt
