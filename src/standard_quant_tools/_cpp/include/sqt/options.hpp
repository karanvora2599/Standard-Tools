#pragma once

#include <cstddef>
#include <cstdint>

namespace sqt {

/**
 * Black-Scholes-Merton over a whole option chain in one call.
 *
 * WHY THIS EXISTS. `analysis/options.py` solves one contract per Python
 * call, so a chain of a few hundred contracts is a few hundred boundary
 * crossings, and greeks over a grid of spots are tens of thousands. The
 * kernels here run the same arithmetic over every contract at once.
 *
 * THE SAME ARITHMETIC, NOT A SIMILAR ONE. Every formula below is written
 * operation for operation as the Python it replaces evaluates it -- the
 * same association, the same `volatility**2` through `pow` where the Python
 * squares that way and `vol * vol` where it multiplies, the normal CDF as
 * `0.5 * (1 + erf(x / sqrt(2)))` -- and the translation unit is compiled
 * without floating-point contraction or vectorised math routines. On a
 * toolchain whose libm is the one CPython calls, the results are
 * bit-for-bit the scalar functions', which is what makes a converged flag,
 * an iteration count and a refusal code comparable between the two paths
 * rather than merely close.
 *
 * NOTHING HERE THROWS. A contract the formulas cannot price comes back as a
 * reason code with NaN outputs, because an exception escaping an OpenMP
 * structured block is undefined behaviour and one bad contract must not
 * take the others with it. The Python layer validates first and decides
 * which codes are a refusal of the whole batch.
 */

/// Why an implied-volatility solve did or did not produce a volatility.
/// Codes 1-5 are properties of the QUOTE and are reported per contract;
/// 6 and 7 are properties of the INPUTS the Python layer refuses first.
enum ImpliedVolReason : std::int8_t {
    kIvSolved = 0,
    kIvPriceNotPositive = 1,  // price <= 0: no volatility is identifiable
    kIvPriceNotFinite = 2,    // NaN or +inf: a missing quote
    kIvBelowLowerBound = 3,   // under the volatility -> 0 limit
    kIvAboveUpperBound = 4,   // over the volatility -> infinity limit
    kIvNoRootInBracket = 5,   // inside the bounds, outside [1e-6, 5.0]
    kIvNotPriceable = 6,      // the model price came out non-finite
    kIvInvalidInput = 7,      // spot/strike/time/rate outside the domain
};

/// Which solver produced the volatility, matching the scalar's `method`.
enum ImpliedVolMethod : std::int8_t {
    kIvMethodNone = 0,
    kIvMethodNewton = 1,
    kIvMethodBisection = 2,
};

/// The scalar solver's keyword arguments, with its defaults.
struct ImpliedVolSettings {
    double initial_guess = 0.2;
    double tol = 1e-6;
    int max_iterations = 100;
    double tol_sigma = 1e-8;
};

/// One contract's answer, field for field the scalar's result dict.
struct ImpliedVolResult {
    double vol;
    double price_error;
    std::int32_t iterations;
    std::int8_t method;
    std::int8_t reason;
    bool converged;
    bool at_bound;
};

/**
 * `analysis.options.implied_volatility` for one contract: the no-arbitrage
 * bound check, Newton on vega converged on the volatility step, and the
 * bisection fallback over [1e-6, 5.0] -- including the at-intrinsic
 * bisection that returns the largest volatility still reproducing the
 * price. A quote the scalar would refuse returns its reason code instead.
 */
ImpliedVolResult implied_volatility_one(double price, double spot, double strike,
                                        double time_to_expiry, double rate,
                                        double dividend_yield, bool is_call,
                                        const ImpliedVolSettings& settings);

/// Output buffers for implied_volatility_batch, each of length n.
struct ImpliedVolBatchOut {
    double* vol;
    double* price_error;
    std::int32_t* iterations;
    std::int8_t* method;
    std::int8_t* reason;
    std::uint8_t* converged;
    std::uint8_t* at_bound;
};

/**
 * implied_volatility_one over n contracts. Every input array has length n
 * (the Python layer broadcasts scalars). Runs across contracts in parallel
 * when omp_policy judges the chain large enough; each contract is solved
 * independently, so the answer does not depend on the thread count.
 */
void implied_volatility_batch(const double* price, const double* spot,
                              const double* strike, const double* time_to_expiry,
                              const double* rate, const double* dividend_yield,
                              const std::uint8_t* is_call, std::size_t n,
                              const ImpliedVolSettings& settings,
                              ImpliedVolBatchOut out);

/**
 * The full greek set `analysis.derivatives.option_greeks` returns, in its
 * units: vega, vanna per volatility point, volga per point squared, theta
 * and charm per calendar day, rho per rate point, and the price
 * `analysis.pricing.price_option` gives for model='black_scholes'.
 */
struct BlackScholesGreeks {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
    double vanna;
    double volga;
    double charm;
    double speed;
    double d1;
    double d2;
};

/// One contract at one spot. Inputs outside the pricing domain give NaN
/// in every field.
BlackScholesGreeks black_scholes_greeks_one(double spot, double strike,
                                            double time_to_expiry, double volatility,
                                            double rate, double dividend_yield,
                                            bool is_call);

/// Output buffers for black_scholes_greeks_batch, one per greek.
struct BlackScholesGreeksOut {
    double* price;
    double* delta;
    double* gamma;
    double* vega;
    double* theta;
    double* rho;
    double* vanna;
    double* volga;
    double* charm;
    double* speed;
    double* d1;
    double* d2;
};

/**
 * black_scholes_greeks_one over a batch.
 *
 * grid == false: `spot` has one entry per contract (n_spots == n_contracts)
 * and every output has length n_contracts.
 *
 * grid == true: every contract is valued at every spot, and each output is
 * a row-major (n_contracts, n_spots) array -- the shape a gamma profile or
 * a scenario revaluation reads by row.
 */
void black_scholes_greeks_batch(const double* spot, std::size_t n_spots,
                                const double* strike, const double* time_to_expiry,
                                const double* volatility, const double* rate,
                                const double* dividend_yield,
                                const std::uint8_t* is_call, std::size_t n_contracts,
                                bool grid, BlackScholesGreeksOut out);

}  // namespace sqt
