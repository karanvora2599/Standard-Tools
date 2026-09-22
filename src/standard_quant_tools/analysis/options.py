"""
Black-Scholes-Merton European option pricing, Greeks, and implied
volatility. Dependency-free: the standard normal CDF/PDF are computed via
`math.erf` (stdlib), not scipy.stats — pricing and Greeks never require
scipy. `implied_volatility` also has no hard scipy dependency: it uses
Newton-Raphson (vega as the derivative) with a bisection fallback over a
practical volatility bracket, the standard robust design for this exact
problem (Newton alone can diverge for deep ITM/OTM options where vega is
tiny).

Scope, stated explicitly: this module prices European options only
(no early exercise) and requires strictly positive time_to_expiry — an
expired or expiring option's value is its intrinsic value
(max(S-K,0) / max(K-S,0)), not something this module's formulas are valid
for; compute that directly rather than calling black_scholes_price with
time_to_expiry=0.
"""

import logging
import math
from typing import Any, Dict, Tuple

from standard_quant_tools._special import (
    norm_cdf,
    norm_pdf,
)
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

_OPTION_TYPES = frozenset({"call", "put"})

#: Below this vega a Newton step is division by noise; bisection takes over.
VEGA_FLOOR = 1e-8
#: A price this close to a no-arbitrage bound, relative to the bound's
#: scale, is AT the bound: the pricer produces such prices itself for a
#: deep-in-the-money option (bit-for-bit equal to intrinsic, D13).
BOUND_TOLERANCE = 1e-9
#: The volatility bracket the bisection searches.
SIGMA_LOW, SIGMA_HIGH = 1e-6, 5.0
#: Largest yield magnitude these formulas will price. A yield is bounded on
#: MAGNITUDE and never on sign: a negative continuous yield is the ordinary
#: case for an FX option's foreign rate and for a commodity whose
#: convenience yield exceeds its storage cost, and a `>= 0` guard refused
#: exactly those. The number matches `analysis.derivatives.MAX_RATE`,
#: restated rather than imported so this module keeps the stdlib-only
#: import graph its header promises; `tests/analysis/test_options.py`
#: pins the two together.
MAX_RATE = 10.0  # 1,000% continuously compounded
_SQRT_2PI = math.sqrt(2.0 * math.pi)


# See `_special`: this had 7 copies across the library, and the ones
# that were not identical disagreed at the edge of the domain.
_norm_cdf = norm_cdf

# See `_special`: this had 3 copies across the library, and the ones
# that were not identical disagreed at the edge of the domain.
_norm_pdf = norm_pdf


def _validate_option_inputs(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    option_type: str,
) -> None:
    if spot <= 0:
        raise ValidationError(f"spot must be > 0, got {spot}")
    if strike <= 0:
        raise ValidationError(f"strike must be > 0, got {strike}")
    if time_to_expiry <= 0:
        raise ValidationError(
            f"time_to_expiry must be > 0, got {time_to_expiry} — an expired/"
            "expiring option's value is its intrinsic value, not something "
            "these Black-Scholes formulas are valid for; compute "
            "max(S-K,0)/max(K-S,0) directly instead"
        )
    if volatility <= 0:
        raise ValidationError(f"volatility must be > 0, got {volatility}")
    if option_type not in _OPTION_TYPES:
        raise ValidationError(
            f"option_type must be one of {sorted(_OPTION_TYPES)}, got {option_type!r}"
        )
    # BOUNDED, not merely positive. Found by fuzzing: `volatility=1e300`
    # passed every check above and then raised
    #
    #     OverflowError: (34, 'Result too large')
    #
    # out of `math.exp` two lines later, naming neither the argument nor the
    # tool. exp(x) overflows a float at about x = 710, and a `vol * sqrt(T)`
    # denominator underflows to exactly zero well before either factor
    # reaches the smallest normal float.
    #
    # These limits sit far outside anything a real market produces -- 10,000%
    # annualized volatility, a hundred-year expiry -- so they reject only
    # inputs that were already a unit error or a typo, never an extreme case
    # somebody meant to ask about.
    #
    # Bounded on MAGNITUDE rather than on the signed value, so a future
    # model-specific sign exemption is not broken by this guard. Writing it
    # against the signed value in pricing.py refused a negative spot outright
    # and broke the Bachelier exemption -- WTI settled at -$37.63 on 20 April
    # 2020, and that is the exact case the model exists for.
    for name, value, high in (
        ("spot", spot, 1e12),
        ("strike", strike, 1e12),
        ("time_to_expiry", time_to_expiry, 100.0),
        ("volatility", volatility, 100.0),
    ):
        numeric = float(value)
        if not math.isfinite(numeric) or abs(numeric) > high:
            raise ValidationError(
                f"{name}={numeric:g} has a magnitude outside what these "
                f"formulas can price (limit {high:g}). A lognormal model's "
                "exp(rate x time) overflows a float at about 710, so a value "
                "this far out is a unit error rather than an extreme case."
            )


def _d1_d2(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> Tuple[float, float]:
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t
    return d1, d2


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> float:
    """
    Black-Scholes-Merton European option price. dividend_yield=0.0 (default)
    is plain Black-Scholes; a non-zero dividend_yield is the Merton (1973)
    continuous-yield extension, and it may be NEGATIVE -- an FX option's
    foreign rate and a commodity's net convenience yield both are.

    Args:
        spot: Current underlying price. Must be > 0.
        strike: Strike price. Must be > 0.
        time_to_expiry: Time to expiry in years (e.g. 0.25 = 3 months). Must be > 0.
        risk_free_rate: Annualized continuously-compounded risk-free rate.
        volatility: Annualized volatility (e.g. 0.20 = 20%). Must be > 0.
        option_type: "call" or "put".
        dividend_yield: Continuous dividend yield, of either sign. Bounded
            on magnitude by MAX_RATE.

    Returns:
        Option price (same currency units as spot/strike).

    Raises:
        ValidationError: spot/strike/time_to_expiry/volatility <= 0,
            abs(dividend_yield) > MAX_RATE, or an unknown option_type.
    """
    _validate_option_inputs(spot, strike, time_to_expiry, volatility, option_type)
    if abs(dividend_yield) > MAX_RATE:
        raise ValidationError(
            f"dividend_yield={dividend_yield} is outside +/-{MAX_RATE:g}. The "
            "bound is on MAGNITUDE and never on sign: a NEGATIVE continuous "
            "yield is the ordinary case for an FX option's foreign rate and "
            "for a commodity whose convenience yield exceeds its storage "
            "cost, and both price normally here. A magnitude beyond this is "
            "a unit error rather than an extreme case."
        )

    d1, d2 = _d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
    )
    disc_q = math.exp(-dividend_yield * time_to_expiry)
    disc_r = math.exp(-risk_free_rate * time_to_expiry)

    if option_type == "call":
        price = spot * disc_q * _norm_cdf(d1) - strike * disc_r * _norm_cdf(d2)
    else:
        price = strike * disc_r * _norm_cdf(-d2) - spot * disc_q * _norm_cdf(-d1)

    logger.debug(
        "[options] price  %s  S=%.4f K=%.4f T=%.4f r=%.4f sigma=%.4f q=%.4f -> %.6f",
        option_type,
        spot,
        strike,
        time_to_expiry,
        risk_free_rate,
        volatility,
        dividend_yield,
        price,
    )
    return price


def black_scholes_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> Dict[str, float]:
    """
    Black-Scholes-Merton Greeks: delta, gamma, vega, theta, rho, plus d1/d2
    (useful to callers that also want the price via black_scholes_price
    without recomputing them).

    Units, stated explicitly (a common source of confusion): vega is the
    price change per 1.0 (100 percentage points) of volatility — divide by
    100 for the conventional "per 1 vol point" quote. theta is per YEAR
    (raw), not per calendar day — divide by 365 for the conventional "daily
    time decay" quote. Both are left raw here rather than pre-scaled, so a
    caller doesn't have to guess which convention this function chose.

    Args/Returns/Raises: same as black_scholes_price, but returns a dict
    with keys "delta", "gamma", "vega", "theta", "rho", "d1", "d2".
    """
    _validate_option_inputs(spot, strike, time_to_expiry, volatility, option_type)
    if abs(dividend_yield) > MAX_RATE:
        raise ValidationError(
            f"dividend_yield={dividend_yield} is outside +/-{MAX_RATE:g}. The "
            "bound is on MAGNITUDE and never on sign: a NEGATIVE continuous "
            "yield is the ordinary case for an FX option's foreign rate and "
            "for a commodity whose convenience yield exceeds its storage "
            "cost, and both price normally here. A magnitude beyond this is "
            "a unit error rather than an extreme case."
        )

    d1, d2 = _d1_d2(
        spot, strike, time_to_expiry, risk_free_rate, volatility, dividend_yield
    )
    sqrt_t = math.sqrt(time_to_expiry)
    disc_q = math.exp(-dividend_yield * time_to_expiry)
    disc_r = math.exp(-risk_free_rate * time_to_expiry)
    pdf_d1 = _norm_pdf(d1)

    gamma = disc_q * pdf_d1 / (spot * volatility * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t

    if option_type == "call":
        delta = disc_q * _norm_cdf(d1)
        theta = (
            -spot * disc_q * pdf_d1 * volatility / (2.0 * sqrt_t)
            - risk_free_rate * strike * disc_r * _norm_cdf(d2)
            + dividend_yield * spot * disc_q * _norm_cdf(d1)
        )
        rho = strike * time_to_expiry * disc_r * _norm_cdf(d2)
    else:
        delta = disc_q * (_norm_cdf(d1) - 1.0)
        theta = (
            -spot * disc_q * pdf_d1 * volatility / (2.0 * sqrt_t)
            + risk_free_rate * strike * disc_r * _norm_cdf(-d2)
            - dividend_yield * spot * disc_q * _norm_cdf(-d1)
        )
        rho = -strike * time_to_expiry * disc_r * _norm_cdf(-d2)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "d1": d1,
        "d2": d2,
    }


def implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    initial_guess: float = 0.2,
    tol: float = 1e-6,
    max_iterations: int = 100,
    tol_sigma: float = 1e-8,
) -> Dict[str, Any]:
    """
    Solve for the Black-Scholes-Merton volatility that reproduces
    option_price, via Newton-Raphson (vega as the derivative) with a
    bisection fallback over [1e-6, 5.0] (500% annualized vol — a deliberately
    generous practical cap) when vega is below a floor or a step leaves the
    bracket. Black-Scholes price is strictly increasing in volatility for
    any fixed inputs, so bisection always converges where a solution exists.

    CONVERGENCE IS DECLARED ON VOLATILITY, NOT ON PRICE. The test used to be
    an absolute price tolerance applied BEFORE a step was taken, so where
    vega is small a volatility wrong by hundreds of points still priced
    inside 1e-6 and the solver returned its initial guess with
    converged=True: four short-dated puts came back at exactly 0.2 for true
    vols of 3.00, 1.20 and 0.45, and over a 700-case grid 28 "converged"
    answers were off by more than 0.01 vol, the worst by 2.80 (findings
    D9). A Newton step is now always taken, and the solver has converged
    when the step it just took moved volatility by less than `tol_sigma`
    -- which is the price tolerance scaled by vega, and is meaningful at
    any vega. `tol` remains the price tolerance the result reports
    `price_error` against; it no longer declares convergence on its own.

    A no-arbitrage bound check runs first: option_price must lie between
    the option's intrinsic-value-only lower bound (volatility -> 0) and its
    upper bound (volatility -> infinity). Equality within BOUND_TOLERANCE
    is admitted -- the pricer itself produces a deep-in-the-money price
    bit-for-bit equal to intrinsic (D13) -- and a price AT the lower bound
    is reported with `at_bound=True`: every volatility at or below the
    returned one reproduces it, so the number is a ceiling, not an
    estimate.

    Args:
        option_price: Observed market price. Must be > 0; 0.0 is what the
            pricer returns when an option is so far from the money that its
            value underflows, and no volatility is identifiable from it.
        spot, strike, time_to_expiry, risk_free_rate, option_type,
            dividend_yield: same as black_scholes_price.
        initial_guess: Starting volatility for Newton-Raphson.
        tol: Price tolerance, reported as `price_error` and used by the
            bisection as an early exit once the bracket is also narrow.
        max_iterations: Cap on Newton-Raphson iterations before falling
            back to bisection (bisection itself always runs up to 200
            iterations if reached).
        tol_sigma: Convergence tolerance on the volatility step.

    Returns:
        Dict with implied_volatility (float), converged (bool), iterations
        (int, iterations actually used in whichever method converged/ran
        last), method ("newton" | "bisection"), price_error (float, the
        absolute pricing error at the returned volatility), at_bound
        (bool).

    Raises:
        ValidationError: option_price <= 0, spot/strike/time_to_expiry <= 0,
            an unknown option_type, abs(dividend_yield) > MAX_RATE, or
            option_price is outside the no-arbitrage bounds achievable at
            any volatility.
    """
    if option_price <= 0:
        raise ValidationError(
            f"option_price must be > 0, got {option_price}. A price of 0.0 is "
            "what the pricer returns when an option is so far from the money "
            "that its value underflows, and no volatility is identifiable "
            "from it -- the quote carries no information about the smile."
        )
    if spot <= 0:
        raise ValidationError(f"spot must be > 0, got {spot}")
    if strike <= 0:
        raise ValidationError(f"strike must be > 0, got {strike}")
    if time_to_expiry <= 0:
        raise ValidationError(f"time_to_expiry must be > 0, got {time_to_expiry}")
    if option_type not in _OPTION_TYPES:
        raise ValidationError(
            f"option_type must be one of {sorted(_OPTION_TYPES)}, got {option_type!r}"
        )
    if abs(dividend_yield) > MAX_RATE:
        raise ValidationError(
            f"dividend_yield={dividend_yield} is outside +/-{MAX_RATE:g}. The "
            "bound is on MAGNITUDE and never on sign: a NEGATIVE continuous "
            "yield is the ordinary case for an FX option's foreign rate and "
            "for a commodity whose convenience yield exceeds its storage "
            "cost, and both price normally here. A magnitude beyond this is "
            "a unit error rather than an extreme case."
        )

    disc_q = math.exp(-dividend_yield * time_to_expiry)
    disc_r = math.exp(-risk_free_rate * time_to_expiry)
    if option_type == "call":
        lower = max(spot * disc_q - strike * disc_r, 0.0)
        upper = spot * disc_q
    else:
        lower = max(strike * disc_r - spot * disc_q, 0.0)
        upper = strike * disc_r
    # Equality within a tolerance is INSIDE the bound (D13): a strict `<`
    # refused 77 of 700 prices the pricer had itself produced, because a
    # deep-in-the-money call prices bit-for-bit equal to intrinsic.
    slack = BOUND_TOLERANCE * max(abs(upper), 1.0)
    if not (lower - slack <= option_price <= upper + slack):
        raise ValidationError(
            f"option_price={option_price:.6f} is outside the no-arbitrage "
            f"range [{lower:.6f}, {upper:.6f}] for these inputs — no "
            "volatility can reproduce this price"
        )
    at_bound = option_price - lower <= slack

    def _price_diff(sigma: float) -> float:
        return (
            black_scholes_price(
                spot,
                strike,
                time_to_expiry,
                risk_free_rate,
                sigma,
                option_type,
                dividend_yield,
            )
            - option_price
        )

    def _result(sigma: float, converged: bool, iterations: int, method: str):
        return {
            "implied_volatility": float(sigma),
            "converged": bool(converged),
            "iterations": int(iterations),
            "method": method,
            "price_error": abs(_price_diff(sigma)),
            "at_bound": bool(at_bound),
        }

    def _vega(sigma: float) -> float:
        return black_scholes_greeks(
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            sigma,
            option_type,
            dividend_yield,
        )["vega"]

    # ── Newton, converged on the step it took ───────────────────────────
    sigma = initial_guess
    if at_bound:
        # At intrinsic, every small volatility prices the same: the
        # bisection finds the largest one that still does, which is the
        # honest ceiling; Newton has nothing to divide by.
        max_iterations = 0
    for i in range(max_iterations):
        diff = _price_diff(sigma)
        vega = _vega(sigma)
        if vega < VEGA_FLOOR:
            break
        step = diff / vega
        candidate = sigma - step
        if candidate <= 0 or candidate > SIGMA_HIGH:
            break
        sigma = candidate
        if abs(step) < tol_sigma:
            logger.debug(
                "[options] implied_vol  newton converged  sigma=%.6f  iters=%d",
                sigma,
                i + 1,
            )
            return _result(sigma, True, i + 1, "newton")

    # ── Bisection fallback ──────────────────────────────────────────────
    lo, hi = SIGMA_LOW, SIGMA_HIGH
    if at_bound:
        # Every volatility below some level reproduces an intrinsic price to
        # within the bound tolerance; the answer is the largest that does.
        for i in range(200):
            mid = 0.5 * (lo + hi)
            if abs(_price_diff(mid)) <= slack:
                lo = mid
            else:
                hi = mid
            if (hi - lo) < tol_sigma:
                break
        return _result(lo, True, i + 1, "bisection")
    diff_lo = _price_diff(lo)
    diff_hi = _price_diff(hi)
    if diff_lo == 0.0:
        return _result(lo, True, 0, "bisection")
    if diff_hi == 0.0:
        return _result(hi, True, 0, "bisection")
    if diff_lo * diff_hi > 0:
        raise ValidationError(
            "option_price passed the no-arbitrage bound check but no root "
            "was found in the [1e-6, 5.0] volatility bracket — check inputs"
        )
    mid = lo
    for i in range(200):
        mid = 0.5 * (lo + hi)
        diff_mid = _price_diff(mid)
        # Converged when the bracket is narrower than the volatility
        # tolerance, or when the price is inside `tol` AND the bracket is
        # already narrow enough that the price tolerance means something.
        if (hi - lo) < tol_sigma or (abs(diff_mid) < tol and (hi - lo) < 1e-4):
            logger.debug(
                "[options] implied_vol  bisection converged  sigma=%.6f  iters=%d",
                mid,
                i + 1,
            )
            return _result(mid, True, i + 1, "bisection")
        if diff_lo * diff_mid < 0:
            hi = mid
        else:
            lo, diff_lo = mid, diff_mid

    logger.debug("[options] implied_vol  bisection did not converge  sigma=%.6f", mid)
    return _result(mid, False, 200, "bisection")
