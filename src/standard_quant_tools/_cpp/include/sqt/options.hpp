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
 * same association, the volatility squared as `vol * vol` (both Python
 * modules multiply; `analysis.options` squared through `pow` until the
 * CHANGELOG entry of 2026-10-02), the normal CDF as
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
///
/// A NULL pointer is an output the caller does not want, and the batch
/// neither computes nor writes it: a gamma profile needs no erf at all, and
/// a hedge needs one per cell where the full set takes two. Every output
/// that IS written is the double the full call writes there -- each is the
/// same expression on the same inputs whichever others are formed beside it.
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
    /// 1 where the cell's price is a finite number, 0 where it is not -- an
    /// input outside the domain, a discounted spot or strike past a double,
    /// or a d1 of 0/0. Decided without forming the price (see options.cpp),
    /// so a caller that wants only gamma can still refuse what a caller of
    /// the full set refuses. Optional, like every other output.
    std::uint8_t* price_finite = nullptr;
};

/// One bit per output of BlackScholesGreeksOut, in its field order: the
/// selector the binding takes. The first twelve are
/// analysis.options_batch.GREEKS in order.
enum GreekOutput : std::uint32_t {
    kGreekPrice = 1u << 0,
    kGreekDelta = 1u << 1,
    kGreekGamma = 1u << 2,
    kGreekVega = 1u << 3,
    kGreekTheta = 1u << 4,
    kGreekRho = 1u << 5,
    kGreekVanna = 1u << 6,
    kGreekVolga = 1u << 7,
    kGreekCharm = 1u << 8,
    kGreekSpeed = 1u << 9,
    kGreekD1 = 1u << 10,
    kGreekD2 = 1u << 11,
    kGreekPriceFinite = 1u << 12,
    kGreeksAll = (1u << 12) - 1,  // the twelve greeks, without the flag
};

/**
 * black_scholes_greeks_one over a batch, for the outputs whose pointers are
 * not NULL.
 *
 * grid == false: `spot` has one entry per contract (n_spots == n_contracts)
 * and every output has length n_contracts.
 *
 * grid == true: every contract is valued at every spot, and each output is
 * a row-major (n_contracts, n_spots) array -- the shape a gamma profile or
 * a scenario revaluation reads by row. The terms that depend only on the
 * contract (sqrt T, both discount factors, the drift) are formed once per
 * contract rather than once per cell: each is the same double either way.
 */
void black_scholes_greeks_batch(const double* spot, std::size_t n_spots,
                                const double* strike, const double* time_to_expiry,
                                const double* volatility, const double* rate,
                                const double* dividend_yield,
                                const std::uint8_t* is_call, std::size_t n_contracts,
                                bool grid, BlackScholesGreeksOut out);

/**
 * Backward induction through a Cox-Ross-Rubinstein lattice: the loop of
 * `analysis.pricing._binomial`, which prices an American option.
 *
 * WHY THIS IS NATIVE. The Python runs one numpy expression per level, so a
 * tree of n steps is about 6n array calls, each over at most n + 1 nodes:
 * 0.95 ms at the 200 steps the pricing tool defaults to and 63 ms at the
 * 5,000 it accepts, 95-98% of the tool's call. Here it is one pass per
 * level over a buffer that stays in L1.
 *
 * THE SAME ARITHMETIC, NOT A SIMILAR ONE. With S[k] = spot * up_powers[k],
 * the node prices of level L are S[L - i] * down_powers[i] -- the product
 * `spot * up_powers[L::-1] * down_powers[:L + 1]` forms, left to right --
 * and, as numpy evaluates the Python:
 *
 *     value[i] = maximum(sign * (price(steps, i) - strike), 0.0)
 *     for L = steps - 1 .. 0:
 *         value[i] = discount * (p * value[i] + (1 - p) * value[i + 1])
 *         if american: value[i] = maximum(value[i], sign * (price(L, i) - strike))
 *
 * with `1 - p` formed once, as the Python forms it once, and `maximum` as
 * numpy's on x86: a NaN propagates, and a tie returns the second argument
 * (np.maximum(0.0, -0.0) is -0.0; a put's payoff at a node priced exactly at
 * the strike is -0.0). The powers come in from numpy, so the one call this
 * would otherwise make to a math library is numpy's on every platform, and
 * the unit is compiled without contraction. On x86 the result is the Python
 * loop's bit for bit. Where numpy's maximum breaks a +0.0/-0.0 tie the other
 * way (Arm's vmaxq returns +0.0), only the sign of a zero can differ.
 *
 * `levels` receives the values of levels 0, 1 and 2 in that order (1 + 2 + 3
 * doubles), which is what the price, delta and gamma read. `steps` must be at
 * least 3, so that level 2 is reached by induction; both power arrays hold
 * steps + 1 values. Nothing is allocated but
 * one buffer of steps + 1 doubles.
 *
 * @return false if that buffer could not be allocated.
 */
bool binomial_lattice(const double* up_powers, const double* down_powers,
                      std::size_t steps, double spot, double strike, double sign,
                      double probability, double discount, bool american,
                      double* levels);

}  // namespace sqt
