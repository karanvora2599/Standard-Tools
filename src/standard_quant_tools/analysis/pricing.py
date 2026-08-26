"""
Option pricing models behind one declarative spec.

The expansion plan is emphatic about this and it is the same rule that keeps
`STRATEGY_REGISTRY` from becoming twelve backtest tools: the model families
go behind **one `PricingSpec` with a `model` field**, not seven tools. A
caller picks a model the way they pick an estimator, and adding Heston later
is a registry entry rather than a thirteenth name to learn.

THE MODELS, AND WHEN EACH IS THE RIGHT ONE:

- **black_scholes** — equities. Lognormal spot, continuous dividend yield.
  The default, and wrong in the tails by construction.
- **black_76** — options on FUTURES. Same lognormal maths, but the
  underlying is a forward, so there is no carry term. Using Black-Scholes on
  a future by passing the futures price as spot double-counts the carry.
- **bachelier** — NORMAL rather than lognormal. The model to use when the
  underlying can go negative, which is not a hypothetical: WTI settled at
  -$37.63 on 20 April 2020 and every lognormal model on the street returned
  a domain error that day.
- **binomial** — Cox-Ross-Rubinstein, and the only one here that prices an
  AMERICAN option. Early exercise has no closed form, so a caller who needs
  it needs a lattice.

WHAT IS DELIBERATELY NOT HERE. Local vol, SVI, SABR and Heston are in the
plan and are not implemented. Each of them is CALIBRATED to a surface rather
than evaluated from parameters, so they need an option chain -- and no
provider in this library serves one. Adding them now would mean shipping a
calibration routine with nothing to calibrate against, which is the same
empty box the point-in-time module already warns about. The `model` field is
the extension point; they are entries it does not have yet.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

#: Every model this spec accepts. `american` is a property of the model
#: rather than a flag, because only the lattice can price one.
MODELS = ("black_scholes", "black_76", "bachelier", "binomial")

#: Models that can price an American option. Everything else is European by
#: construction, and silently pricing an American option with a European
#: formula understates it by exactly the early-exercise premium -- which is
#: largest precisely when it matters, deep in the money on a dividend payer.
AMERICAN_CAPABLE = ("binomial",)

#: Steps in the binomial lattice. 200 puts the CRR price within about a cent
#: of Black-Scholes on a European option, which is the convergence check
#: worth having: a lattice that does NOT converge to the closed form on the
#: case where both apply is wired wrong.
DEFAULT_BINOMIAL_STEPS = 200


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _validate(spot, strike, time_to_expiry, volatility, option_type, model):
    if model not in MODELS:
        raise ValidationError(
            f"unknown pricing model {model!r}; expected one of {list(MODELS)}"
        )
    if option_type not in ("call", "put"):
        raise ValidationError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )
    if time_to_expiry <= 0:
        raise ValidationError(
            "time_to_expiry must be positive. An expired option is worth its "
            "intrinsic value and does not need a model."
        )
    if volatility <= 0:
        raise ValidationError("volatility must be positive")
    if model != "bachelier" and strike <= 0:
        # Bachelier is exempt for the same reason it is exempt on spot: a
        # negative strike is not a typo under a normal model. WTI options
        # traded at negative strikes in April 2020, and a check that refused
        # them would refuse the exact case the model exists for.
        raise ValidationError(
            f"{model!r} is lognormal and needs a positive strike (got "
            f"{strike}). Use model='bachelier' for a normal underlying."
        )
    if model != "bachelier" and spot <= 0:
        raise ValidationError(
            f"{model!r} is a LOGNORMAL model and cannot price a "
            f"non-positive underlying (got spot={spot}). This is not an edge "
            "case: WTI settled at -$37.63 on 20 April 2020. Use "
            "model='bachelier', which is normal rather than lognormal."
        )


def price_option(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    option_type: str = "call",
    model: str = "black_scholes",
    dividend_yield: float = 0.0,
    american: bool = False,
    steps: int = DEFAULT_BINOMIAL_STEPS,
) -> Dict[str, Any]:
    """
    One option price, and the greeks that come with it.

    `volatility` means different things in different models and the
    difference is not cosmetic. For the lognormal models it is a RELATIVE
    volatility -- a fraction of the underlying per year. For `bachelier` it
    is an ABSOLUTE one, in the underlying's own units. Passing 0.30 to
    Bachelier on a $80 future means 30 cents of annual vol, not 30%, and the
    resulting price is wrong by two orders of magnitude. Said here because
    no type system catches it.
    """
    # BOUNDED, not merely positive. Fuzzing raised OverflowError out of this
    # function at volatility=1e300: exp(r*T) overflows a float at about
    # r*T = 710, and `vol * sqrt(T)` underflows to exactly zero well before
    # either factor reaches the smallest normal float. The limits below sit
    # far outside anything a real market produces, so they reject only
    # inputs that were already wrong.
    # Bounded on MAGNITUDE, never on sign. Bounding the signed value
    # rejected a negative spot outright and broke the Bachelier exemption --
    # the whole reason that model is here. WTI settled at -$37.63 on 20
    # April 2020, and a guard that refuses it refuses the exact case the
    # model exists for. The sign checks below are model-aware and stay
    # there; this only has to stop 1e300 reaching exp().
    for _name, _value, _high in (
        ("spot", spot, 1e12),
        ("strike", strike, 1e12),
        ("time_to_expiry", time_to_expiry, 100.0),
        ("volatility", volatility, 100.0),
    ):
        _numeric = float(_value)
        if not math.isfinite(_numeric) or abs(_numeric) > _high:
            raise ValidationError(
                f"{_name}={_numeric:g} has a magnitude outside what these "
                f"models can price (limit {_high:g}). A lognormal model's "
                "exp(rate x time) overflows a float at about 710, so a value "
                "this far out is a unit error rather than an extreme case."
            )

    _validate(spot, strike, time_to_expiry, volatility, option_type, model)
    if american and model not in AMERICAN_CAPABLE:
        raise ValidationError(
            f"model={model!r} prices EUROPEAN options only, and american=True "
            f"was requested. Early exercise has no closed form; use "
            f"model='binomial'. Pricing it with a European formula would "
            "understate the option by exactly the early-exercise premium, "
            "which is largest deep in the money on a dividend payer -- the "
            "case where you asked."
        )

    if model == "binomial":
        return _binomial(
            spot,
            strike,
            time_to_expiry,
            volatility,
            risk_free_rate,
            option_type,
            dividend_yield,
            american,
            steps,
        )
    if model == "bachelier":
        return _bachelier(
            spot, strike, time_to_expiry, volatility, risk_free_rate, option_type
        )
    # black_76 is Black-Scholes on a forward: no carry, because the forward
    # already contains it. Modelled as a zero dividend yield against a
    # discounted-forward spot rather than as separate arithmetic.
    carry = 0.0 if model == "black_76" else dividend_yield
    return _black_scholes(
        spot,
        strike,
        time_to_expiry,
        volatility,
        risk_free_rate,
        option_type,
        carry,
        forward=(model == "black_76"),
    )


def _black_scholes(
    spot, strike, t, vol, rate, option_type, dividend_yield, forward=False
) -> Dict[str, Any]:
    discount = math.exp(-rate * t)
    # For Black-76 the "spot" IS the forward, so it is not grown by carry --
    # it is discounted. Getting this backwards double-counts the carry, and
    # the error grows with time to expiry, which is why it survives a
    # near-dated test.
    drift = 0.0 if forward else (rate - dividend_yield)
    growth = 1.0 if forward else math.exp(-dividend_yield * t)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (drift + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t

    if option_type == "call":
        price = (
            discount * (spot * _norm_cdf(d1) - strike * _norm_cdf(d2))
            if forward
            else spot * growth * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
        )
        delta = discount * _norm_cdf(d1) if forward else growth * _norm_cdf(d1)
        rho = strike * t * discount * _norm_cdf(d2)
    else:
        price = (
            discount * (strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1))
            if forward
            else strike * discount * _norm_cdf(-d2) - spot * growth * _norm_cdf(-d1)
        )
        delta = -discount * _norm_cdf(-d1) if forward else -growth * _norm_cdf(-d1)
        rho = -strike * t * discount * _norm_cdf(-d2)

    gamma = (
        (growth if not forward else discount) * _norm_pdf(d1) / (spot * vol * sqrt_t)
    )
    vega = spot * (growth if not forward else discount) * _norm_pdf(d1) * sqrt_t
    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        # Per 1% of vol and per calendar day, which is how a desk quotes them.
        "vega": float(vega / 100.0),
        "rho": float(rho / 100.0),
        "d1": float(d1),
        "d2": float(d2),
        "model": "black_76" if forward else "black_scholes",
    }


def _bachelier(spot, strike, t, vol, rate, option_type) -> Dict[str, Any]:
    """
    Normal (arithmetic) model. The underlying may be negative.

    `vol` here is an ABSOLUTE volatility in the underlying's units, not a
    fraction. This is the model's defining difference and the easiest thing
    to get wrong about it.
    """
    discount = math.exp(-rate * t)
    sqrt_t = math.sqrt(t)
    d = (spot - strike) / (vol * sqrt_t)
    if option_type == "call":
        price = discount * (
            (spot - strike) * _norm_cdf(d) + vol * sqrt_t * _norm_pdf(d)
        )
        delta = discount * _norm_cdf(d)
    else:
        price = discount * (
            (strike - spot) * _norm_cdf(-d) + vol * sqrt_t * _norm_pdf(d)
        )
        delta = -discount * _norm_cdf(-d)
    gamma = discount * _norm_pdf(d) / (vol * sqrt_t)
    vega = discount * sqrt_t * _norm_pdf(d)
    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega / 100.0),
        "rho": float(-t * price / 100.0),
        "d1": float(d),
        "d2": float(d),
        "model": "bachelier",
    }


def _binomial(
    spot, strike, t, vol, rate, option_type, dividend_yield, american, steps
) -> Dict[str, Any]:
    """
    Cox-Ross-Rubinstein lattice. The only model here that prices early
    exercise.

    Greeks come from the lattice itself rather than from a bumped reprice:
    the first two time steps already contain the three spot nodes a central
    difference needs, so delta and gamma are exact for the tree rather than
    being a finite difference of it.
    """
    if steps < 10:
        raise ValidationError("binomial: steps must be at least 10")
    dt = t / steps
    up = math.exp(vol * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp((rate - dividend_yield) * dt)
    if not down < growth < up:
        raise ValidationError(
            f"binomial: no arbitrage-free probability exists for these inputs "
            f"(u={up:.4f}, d={down:.4f}, growth={growth:.4f}). The lattice is "
            "too coarse for this rate and volatility -- raise `steps`."
        )
    probability = (growth - down) / (up - down)
    discount = math.exp(-rate * dt)
    sign = 1.0 if option_type == "call" else -1.0

    prices = [spot * (up ** (steps - i)) * (down**i) for i in range(steps + 1)]
    values = [max(sign * (p - strike), 0.0) for p in prices]

    node_cache: Dict[int, list] = {}
    for step in range(steps - 1, -1, -1):
        prices = [spot * (up ** (step - i)) * (down**i) for i in range(step + 1)]
        values = [
            discount * (probability * values[i] + (1.0 - probability) * values[i + 1])
            for i in range(step + 1)
        ]
        if american:
            values = [max(v, sign * (prices[i] - strike)) for i, v in enumerate(values)]
        if step <= 2:
            node_cache[step] = (list(prices), list(values))

    price = values[0]
    p1, v1 = node_cache[1]
    p2, v2 = node_cache[2]
    delta = (v1[0] - v1[1]) / (p1[0] - p1[1])
    upper = (v2[0] - v2[1]) / (p2[0] - p2[1])
    lower = (v2[1] - v2[2]) / (p2[1] - p2[2])
    gamma = (upper - lower) / ((p2[0] - p2[2]) / 2.0)

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": None,
        "rho": None,
        "d1": None,
        "d2": None,
        "model": "binomial",
        "american": bool(american),
        "steps": int(steps),
        "notes": [
            "vega and rho are not returned: the lattice has no closed form "
            "for them, and a bumped reprice would be a different number from "
            "the analytic ones the other models report. Price with "
            "black_scholes for a European vega."
        ],
    }


__all__ = [
    "AMERICAN_CAPABLE",
    "DEFAULT_BINOMIAL_STEPS",
    "MODELS",
    "price_option",
]
