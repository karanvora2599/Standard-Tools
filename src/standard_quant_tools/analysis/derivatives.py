"""
What you do with a price once you have one.

`pricing.py` answers "what is this option worth". Everything here answers a
question that only arises AFTER that: what the position's risk looks like,
whether the quoted surface is internally consistent, what the market is
saying about the move, and what the hedge actually costs to run.

THE ORGANISING FACT is that an option price is a statement about a
distribution, and almost every mistake in this area comes from reading it as
a statement about a level. `expected_move` returns a one-standard-deviation
move and says so, because the number gets quoted as "the expected move" and
then read as a bound -- it is exceeded about a third of the time by
construction. `volatility_cone` exists because "IV is 30" is meaningless
without knowing that this name's 30-day realized vol has been between 18 and
55 for the past two years.

NO SCIPY, consistent with the rest of the package. Every distribution
function here is a closed form or an erf-based identity that `math` carries.

THE LIMITS, stated once rather than rediscovered:

- **Greeks are model derivatives, not market sensitivities.** Vega is the
  derivative with respect to the ONE volatility number in the formula. On a
  real book with a smile, shocking every strike's vol by the same amount is
  not a thing the market does, so a vega sum across strikes overstates what
  a realistic vol move costs.
- **The smile fit is a fit.** It interpolates and it does not extrapolate.
  Asking for the implied vol of a strike outside the fitted range returns a
  refusal rather than a polynomial's opinion, because a quadratic taken
  beyond its data will happily produce a negative variance.
- **`simulate_delta_hedge` is a single path unless you give it many.** One
  path tells you what happened, not what to expect; the dispersion across
  paths is the whole point, and the hedging error is proportional to
  1/sqrt(n_hedges) rather than to 1/n_hedges.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.analysis.pricing import price_option
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

#: Trading days in a year. Used to annualize realized volatility and to
#: convert a hedging schedule into a time step.
TRADING_DAYS = 252

#: The horizons a volatility cone is built over, in trading days. Chosen to
#: bracket the listed expiries an equity option trader actually quotes --
#: one week, two weeks, a month, two months, a quarter, half a year.
CONE_HORIZONS = (5, 10, 21, 42, 63, 126)

#: Smile fits below this many strikes are not fits, they are interpolation
#: with extra steps. A quadratic through three points is exact and tells you
#: nothing about whether the shape is real.
MIN_SMILE_STRIKES = 5


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _positive(value: Any, name: str) -> float:
    if value is None:
        raise ValidationError(f"{name} is required and was not given")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValidationError(f"{name} must be positive and finite, got {value!r}")
    return value


#: Magnitudes beyond which an input is not an extreme case, it is a
#: mistake. Every one is set far outside anything a real market produces,
#: so a rejection here never refuses a question somebody meant to ask.
#:
#: The reason they exist is arithmetic rather than taste: exp(r*T) overflows
#: a float at about r*T = 710, and a `vol * sqrt(T)` denominator underflows
#: to exactly zero well before either factor reaches the smallest normal
#: float. Found by fuzzing, which raised OverflowError and
#: ZeroDivisionError out of five tools at 1e300 and 1e-300.
MAX_VOLATILITY = 100.0  # 10,000% annualized
MIN_TIME_TO_EXPIRY = 1e-8  # about a third of a second
MAX_TIME_TO_EXPIRY = 100.0  # years; no listed contract is close
MAX_RATE = 10.0  # 1,000% continuously compounded
MAX_PRICE = 1e12


def _bounded(
    value: float, name: str, *, low: float, high: float, unit: str = ""
) -> float:
    """
    A finite value inside a plausible range, or a refusal that names the
    bound it broke.

    `_positive` admits 1e300, and 1e300 overflows `exp` two lines later
    with an error naming neither the argument nor the tool. This refuses
    first, with both.
    """
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value!r}")
    if not low <= value <= high:
        raise ValidationError(
            f"{name}={value:g}{unit} is outside the range this can price "
            f"({low:g} to {high:g}{unit}). That is not a conservative "
            "limit -- it sits far outside anything a real market produces, "
            "so a value beyond it is a unit error or a typo rather than an "
            "extreme case. Note that a lognormal model's exp(rate x time) "
            "overflows a float at about 710."
        )
    return value


def _option_inputs(
    *,
    spot: Optional[float] = None,
    strike: Optional[float] = None,
    time_to_expiry: Optional[float] = None,
    volatility: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    dividend_yield: Optional[float] = None,
) -> None:
    """Bound every option argument that was supplied. Order matters only
    for which error a caller sees first."""
    # Magnitude, for the reason recorded in pricing.py: a signed bound here
    # would refuse the negative underlying a normal model exists to price.
    if spot is not None:
        _bounded(abs(float(spot)), "spot magnitude", low=1e-8, high=MAX_PRICE)
    if strike is not None:
        _bounded(abs(float(strike)), "strike magnitude", low=1e-8, high=MAX_PRICE)
    if time_to_expiry is not None:
        _bounded(
            time_to_expiry,
            "time_to_expiry",
            low=MIN_TIME_TO_EXPIRY,
            high=MAX_TIME_TO_EXPIRY,
            unit=" years",
        )
    if volatility is not None:
        _bounded(volatility, "volatility", low=1e-8, high=MAX_VOLATILITY)
    if risk_free_rate is not None:
        _bounded(risk_free_rate, "risk_free_rate", low=-MAX_RATE, high=MAX_RATE)
    if dividend_yield is not None:
        _bounded(dividend_yield, "dividend_yield", low=-MAX_RATE, high=MAX_RATE)


# ── greeks beyond the first order ───────────────────────────────────────


def option_greeks(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> Dict[str, Any]:
    """
    The full Black-Scholes greek set, including the second-order ones that
    explain why a delta-hedged book still loses money.

    `price_option` returns delta, gamma, vega, theta and rho. Those tell you
    the first-order risk. They do not tell you how that risk CHANGES, which
    is what actually gets a hedged position into trouble:

    - **vanna** -- how delta moves when vol moves. A short-vol position that
      is delta-flat today is not delta-flat after a vol spike, and vanna is
      how much rehedging that spike forces.
    - **volga** (vomma) -- how vega moves when vol moves. Vega is largest
      at the money, so a wing option's vega GROWS as vol rises; volga is why
      short wings lose more than the vega number suggested.
    - **charm** -- how delta moves with time alone. It is why a Friday
      delta-flat book opens Monday short, with no move in the underlying.
    - **speed** -- how gamma moves with spot, which matters for the size of
      the rehedge on a large gap.

    UNITS ARE STATED PER GREEK because there is no convention and the
    mismatch is a real source of error. Vega and volga are per one
    volatility POINT (0.01), theta and charm are per calendar day, and vanna
    is per one vol point per unit of spot.

    THE HONEST LIMIT: every number here is the derivative of one model with
    one volatility. A book with a smile does not experience a parallel vol
    shift, so summing vega across strikes and multiplying by an expected vol
    move overstates the P&L -- the wings move less than the at-the-money in
    most vol regimes, and sometimes the other way.
    """
    spot = _positive(spot, "spot")
    strike = _positive(strike, "strike")
    t = _positive(time_to_expiry, "time_to_expiry")
    vol = _positive(volatility, "volatility")
    _option_inputs(
        spot=spot,
        strike=strike,
        time_to_expiry=t,
        volatility=vol,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    option_type = str(option_type).lower()
    if option_type not in ("call", "put"):
        raise ValidationError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )

    rate = float(risk_free_rate)
    q = float(dividend_yield)
    sqrt_t = math.sqrt(t)
    growth = math.exp(-q * t)
    discount = math.exp(-rate * t)

    d1 = (math.log(spot / strike) + (rate - q + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    pdf_d1 = _norm_pdf(d1)

    if option_type == "call":
        delta = growth * _norm_cdf(d1)
        rho = strike * t * discount * _norm_cdf(d2)
        theta_raw = (
            -spot * pdf_d1 * vol * growth / (2.0 * sqrt_t)
            + q * spot * growth * _norm_cdf(d1)
            - rate * strike * discount * _norm_cdf(d2)
        )
        charm_raw = -growth * (
            pdf_d1
            * (2.0 * (rate - q) * t - d2 * vol * sqrt_t)
            / (2.0 * t * vol * sqrt_t)
            - q * _norm_cdf(d1)
        )
    else:
        delta = -growth * _norm_cdf(-d1)
        rho = -strike * t * discount * _norm_cdf(-d2)
        theta_raw = (
            -spot * pdf_d1 * vol * growth / (2.0 * sqrt_t)
            - q * spot * growth * _norm_cdf(-d1)
            + rate * strike * discount * _norm_cdf(-d2)
        )
        charm_raw = -growth * (
            pdf_d1
            * (2.0 * (rate - q) * t - d2 * vol * sqrt_t)
            / (2.0 * t * vol * sqrt_t)
            + q * _norm_cdf(-d1)
        )

    gamma = growth * pdf_d1 / (spot * vol * sqrt_t)
    vega_raw = spot * growth * pdf_d1 * sqrt_t

    # Vanna and volga are the same for calls and puts -- put-call parity is
    # linear in spot and independent of vol, so every second derivative that
    # touches vol is shared.
    vanna_raw = -growth * pdf_d1 * d2 / vol
    volga_raw = vega_raw * d1 * d2 / vol
    speed = -gamma / spot * (d1 / (vol * sqrt_t) + 1.0)

    return {
        "price": float(
            price_option(
                spot=spot,
                strike=strike,
                time_to_expiry=t,
                volatility=vol,
                risk_free_rate=rate,
                option_type=option_type,
                dividend_yield=q,
            )["price"]
        ),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega_raw / 100.0),
        "theta": float(theta_raw / 365.0),
        "rho": float(rho / 100.0),
        "vanna": float(vanna_raw / 100.0),
        "volga": float(volga_raw / 10000.0),
        "charm": float(charm_raw / 365.0),
        "speed": float(speed),
        "d1": float(d1),
        "d2": float(d2),
        "moneyness": float(spot / strike),
        "units": {
            "delta": "change in price per $1 of spot",
            "gamma": "change in delta per $1 of spot",
            "vega": "change in price per 1 volatility point (0.01)",
            "theta": "change in price per calendar day",
            "rho": "change in price per 1 rate point (0.01)",
            "vanna": "change in delta per 1 volatility point",
            "volga": "change in vega per 1 volatility point",
            "charm": "change in delta per calendar day",
            "speed": "change in gamma per $1 of spot",
        },
        "warnings": [
            "Every greek here is the derivative of ONE model at ONE "
            "volatility. Summing vega across a book with a smile and "
            "multiplying by an expected vol move overstates the P&L: the "
            "wings do not move point-for-point with the at-the-money.",
        ],
    }


# ── multi-leg positions ─────────────────────────────────────────────────


def analyze_strategy(
    legs: Sequence[Dict[str, Any]],
    *,
    spot: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    spot_range: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    The payoff, the breakevens and the aggregate greeks of a multi-leg
    position.

    Each leg is a dict with `option_type` ('call', 'put' or 'stock'),
    `strike`, `quantity` (negative is short), and for the option legs
    `volatility` and `time_to_expiry`. The payoff is computed AT EXPIRY --
    intrinsic value only -- while the greeks are computed at today's spot,
    because those are the two questions actually asked of a structure and
    they are asked at different times.

    BREAKEVENS ARE FOUND NUMERICALLY, by scanning the payoff for sign
    changes and bisecting. A closed form exists for each named structure and
    would have to be written once per structure; this works for an arbitrary
    combination of legs, which is the point of accepting a leg list rather
    than a strategy name.

    THE LIMIT WORTH KNOWING: max profit and max loss are reported over the
    SCANNED RANGE, not over all possible spots. A short call has unbounded
    loss and the scan cannot say so by finding it, so the result flags when
    the extreme sits at the edge of the range -- that is what "unbounded"
    looks like from inside a numerical scan, and it is reported rather than
    silently returned as a finite number.
    """
    if not legs:
        raise ValidationError("analyze_strategy: no legs given")
    spot = _positive(spot, "spot")

    parsed: List[Dict[str, Any]] = []
    for i, leg in enumerate(legs):
        kind = str(leg.get("option_type", "")).lower()
        if kind not in ("call", "put", "stock"):
            raise ValidationError(
                f"leg {i}: option_type must be 'call', 'put' or 'stock', "
                f"got {leg.get('option_type')!r}"
            )
        quantity = float(leg.get("quantity", 1.0))
        if quantity == 0:
            raise ValidationError(f"leg {i}: quantity of zero is not a position")
        entry = {"option_type": kind, "quantity": quantity}
        if kind == "stock":
            entry["strike"] = 0.0
        else:
            entry["strike"] = _positive(leg.get("strike"), f"leg {i} strike")
            entry["volatility"] = _positive(
                leg.get("volatility", 0.0), f"leg {i} volatility"
            )
            entry["time_to_expiry"] = _positive(
                leg.get("time_to_expiry", 0.0), f"leg {i} time_to_expiry"
            )
        parsed.append(entry)

    if spot_range is None:
        strikes = [leg["strike"] for leg in parsed if leg["option_type"] != "stock"]
        anchor = max([spot] + strikes) if strikes else spot
        low = min([spot] + strikes) * 0.5 if strikes else spot * 0.5
        grid = np.linspace(max(0.01, low), anchor * 1.5, 401)
    else:
        grid = np.asarray(sorted(float(x) for x in spot_range), dtype=float)
        if grid.size < 3:
            raise ValidationError(
                "spot_range needs at least 3 points to find a breakeven"
            )

    # Net premium paid today. A long position costs money, so the payoff
    # curve sits below the intrinsic curve by exactly this.
    net_premium = 0.0
    totals = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for leg in parsed:
        if leg["option_type"] == "stock":
            net_premium += leg["quantity"] * spot
            totals["delta"] += leg["quantity"]
            continue
        greeks = option_greeks(
            spot=spot,
            strike=leg["strike"],
            time_to_expiry=leg["time_to_expiry"],
            volatility=leg["volatility"],
            risk_free_rate=risk_free_rate,
            option_type=leg["option_type"],
            dividend_yield=dividend_yield,
        )
        net_premium += leg["quantity"] * greeks["price"]
        for key in totals:
            totals[key] += leg["quantity"] * greeks[key]

    payoff = np.zeros_like(grid)
    for leg in parsed:
        if leg["option_type"] == "stock":
            payoff += leg["quantity"] * grid
        elif leg["option_type"] == "call":
            payoff += leg["quantity"] * np.maximum(grid - leg["strike"], 0.0)
        else:
            payoff += leg["quantity"] * np.maximum(leg["strike"] - grid, 0.0)
    profit = payoff - net_premium

    breakevens = _find_breakevens(grid, profit)
    max_profit_i = int(np.argmax(profit))
    max_loss_i = int(np.argmin(profit))
    at_edge = lambda i: i in (0, len(grid) - 1)  # noqa: E731

    warnings: List[str] = []
    if at_edge(max_profit_i):
        warnings.append(
            "Maximum profit sits at the edge of the scanned range, which is "
            "what an UNBOUNDED payoff looks like from inside a numerical "
            "scan. Read it as 'unbounded above', not as this number."
        )
    if at_edge(max_loss_i):
        warnings.append(
            "Maximum loss sits at the edge of the scanned range -- read it "
            "as unbounded. A short call or short stock leg has no worst "
            "case, and the scan cannot find one that does not exist."
        )
    if not breakevens:
        warnings.append(
            "No breakeven in the scanned range: the position is profitable "
            "or loss-making everywhere it was evaluated."
        )
    warnings.append(
        "Payoff is AT EXPIRY (intrinsic only); the greeks are at today's "
        "spot. Between now and expiry the position is worth neither."
    )

    return {
        "n_legs": len(parsed),
        "net_premium": float(net_premium),
        "position": "debit" if net_premium > 0 else "credit",
        "breakevens": [float(b) for b in breakevens],
        "max_profit": float(profit[max_profit_i]),
        "max_profit_at_spot": float(grid[max_profit_i]),
        "max_profit_unbounded": bool(at_edge(max_profit_i)),
        "max_loss": float(profit[max_loss_i]),
        "max_loss_at_spot": float(grid[max_loss_i]),
        "max_loss_unbounded": bool(at_edge(max_loss_i)),
        "greeks": {k: float(v) for k, v in totals.items()},
        "payoff_curve": [
            {"spot": float(s), "profit": float(p)}
            for s, p in zip(grid[::20], profit[::20])
        ],
        "warnings": warnings,
    }


def _find_breakevens(grid: np.ndarray, profit: np.ndarray) -> List[float]:
    """Sign changes in the profit curve, refined by linear interpolation."""
    out: List[float] = []
    for i in range(len(grid) - 1):
        a, b = profit[i], profit[i + 1]
        if a == 0.0:
            out.append(float(grid[i]))
        elif a * b < 0:
            # Linear interpolation is exact here: the payoff is piecewise
            # linear in spot, and a sign change between two grid points that
            # straddle no strike lies on one segment.
            out.append(float(grid[i] - a * (grid[i + 1] - grid[i]) / (b - a)))
    return out


# ── the surface ─────────────────────────────────────────────────────────


def fit_volatility_smile(
    strikes: Sequence[float],
    implied_vols: Sequence[float],
    *,
    forward: float,
    time_to_expiry: float,
) -> Dict[str, Any]:
    """
    Fit the smile as a quadratic in log-moneyness, and report whether the
    fit is a description or an artefact.

    QUADRATIC IN LOG-MONEYNESS, not in strike. The smile is approximately
    symmetric in log(K/F) and emphatically not in K -- a parabola in strike
    puts its vertex in the wrong place and makes the skew depend on the
    level of the underlying, so the same shape refit after a 10% rally looks
    like a different market.

    The three coefficients are the three things a trader actually quotes:
    `atm_vol` is the level, `skew` is the slope at the money (per unit of
    log-moneyness, so it is comparable across expiries and underlyings), and
    `curvature` is the smile's convexity, which is what a butterfly prices.

    ARBITRAGE IS CHECKED, not assumed. A fitted smile can imply a negative
    risk-neutral density, which means the quotes it was fitted to admit a
    butterfly arbitrage or -- far more often -- that one of them is stale.
    The check is Durrleman's condition evaluated across the fitted range,
    and a violation is reported with the strike where it happens rather than
    returned as a clean fit.

    IT IS CONCAVITY THAT BREAKS THE DENSITY, not convexity, which is the
    opposite of the intuition that a "violent" smile is the dangerous one.
    Durrleman's g carries a +w''/2 term, so a strongly CONVEX smile pushes
    it further above zero; a butterfly arbitrage is literally a concave
    price in strike, and it is negative curvature that drives g below zero.
    A smile with a curvature of +25 passes this check and one with -4 fails
    it.

    IT DOES NOT EXTRAPOLATE. `implied_vol_at` refuses strikes outside the
    fitted range, because a quadratic continued into the wings produces
    negative variance at a perfectly ordinary distance from the money.
    """
    k = np.asarray([float(x) for x in strikes], dtype=float)
    v = np.asarray([float(x) for x in implied_vols], dtype=float)
    if k.size != v.size:
        raise ValidationError(
            f"strikes and implied_vols have different lengths ({k.size} vs {v.size})"
        )
    mask = np.isfinite(k) & np.isfinite(v) & (k > 0) & (v > 0)
    k, v = k[mask], v[mask]
    if k.size < MIN_SMILE_STRIKES:
        raise ValidationError(
            f"fit_volatility_smile: {k.size} usable strikes, and a quadratic "
            f"needs at least {MIN_SMILE_STRIKES} to be a fit rather than an "
            "interpolation. Three points determine a parabola exactly and "
            "tell you nothing about whether the shape is real."
        )
    forward = _positive(forward, "forward")
    t = _positive(time_to_expiry, "time_to_expiry")

    x = np.log(k / forward)
    order = np.argsort(x)
    x, v, k = x[order], v[order], k[order]

    design = np.column_stack([np.ones_like(x), x, x**2])
    coefficients, *_ = np.linalg.lstsq(design, v, rcond=None)
    c0, c1, c2 = (float(c) for c in coefficients)

    fitted = design @ coefficients
    residual = v - fitted
    total_ss = float(((v - v.mean()) ** 2).sum())
    r_squared = float(1.0 - (residual**2).sum() / total_ss) if total_ss > 0 else 1.0

    violations = _durrleman_violations(c0, c1, c2, x, t)

    warnings: List[str] = []
    if violations:
        warnings.append(
            f"The fitted smile implies a NEGATIVE risk-neutral density at "
            f"{len(violations)} of the sampled log-moneyness points (nearest "
            f"the money at k={violations[0]['strike']:.2f}). That is a "
            "butterfly arbitrage in the fitted surface. Usually one quote is "
            "stale rather than the market being free money -- check the "
            "inputs before trading it."
        )
    if r_squared < 0.9:
        warnings.append(
            f"R-squared of {r_squared:.2f}: a quadratic does not describe "
            "this smile. Common causes are a mixed-expiry input or a "
            "genuinely bimodal distribution ahead of a binary event, and "
            "neither is fixed by fitting a higher polynomial."
        )
    warnings.append(
        f"Fitted over strikes {k.min():.2f} to {k.max():.2f}. The fit does "
        "not extrapolate: a quadratic continued into the wings reaches "
        "negative variance at an ordinary distance from the money."
    )

    return {
        "n_strikes": int(k.size),
        "forward": float(forward),
        "time_to_expiry": float(t),
        "atm_vol": float(c0),
        "skew": float(c1),
        "curvature": float(c2),
        "r_squared": r_squared,
        "residual_std": float(residual.std(ddof=1)) if k.size > 3 else 0.0,
        "strike_range": [float(k.min()), float(k.max())],
        "arbitrage_violations": violations,
        "fitted": [
            {"strike": float(kk), "observed_vol": float(vv), "fitted_vol": float(ff)}
            for kk, vv, ff in zip(k, v, fitted)
        ],
        "warnings": warnings,
    }


def _durrleman_violations(c0, c1, c2, x, t) -> List[Dict[str, Any]]:
    """
    Durrleman's condition: where it is negative, the fitted smile implies a
    negative probability density, which is a butterfly arbitrage.

    Evaluated on the fitted range only. Outside it the fit is not claimed to
    hold, so a violation there is a statement about the polynomial rather
    than about the quotes.
    """
    out: List[Dict[str, Any]] = []
    grid = np.linspace(float(x.min()), float(x.max()), 121)
    for xi in grid:
        vol = c0 + c1 * xi + c2 * xi * xi
        if vol <= 0:
            out.append({"log_moneyness": float(xi), "reason": "negative fitted vol"})
            continue
        w = vol * vol * t  # total implied variance
        dw = 2.0 * vol * (c1 + 2.0 * c2 * xi) * t
        d2w = 2.0 * t * ((c1 + 2.0 * c2 * xi) ** 2 + vol * 2.0 * c2)
        g = (
            (1.0 - xi * dw / (2.0 * w)) ** 2
            - dw * dw / 4.0 * (1.0 / w + 0.25)
            + d2w / 2.0
        )
        if g < 0:
            out.append(
                {
                    "log_moneyness": float(xi),
                    "durrleman_g": float(g),
                    "reason": "negative implied density",
                }
            )
    # Nearest the money first: that is the violation a trader can act on.
    out.sort(key=lambda d: abs(d["log_moneyness"]))
    for entry in out:
        entry["strike"] = float(math.exp(entry["log_moneyness"]))
    return out[:10]


def volatility_cone(
    prices: pd.Series,
    *,
    horizons: Sequence[int] = CONE_HORIZONS,
    current_implied: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Where today's implied volatility sits inside this name's own history of
    realized volatility, horizon by horizon.

    "IV is 30" means nothing on its own. It means something once you know
    this underlying's 30-day realized vol has spent the last two years
    between 18 and 55 with a median of 26 -- then 30 is the 62nd percentile
    and mildly rich, rather than a number.

    THE CONE SHAPE IS THE INFORMATION. Short-horizon realized vol has a much
    wider distribution than long-horizon, because it averages fewer days.
    A cone that is NOT wider at the short end usually means the sample is
    too short for the long horizons to have independent observations, and
    that is reported as `independent_windows` per horizon -- below about 10,
    the percentiles are describing a handful of overlapping windows.

    OVERLAPPING WINDOWS ARE USED and this matters. Rolling 21-day vol from
    daily data gives one observation per day, but only n/21 of them are
    independent. The percentiles are still the right estimate of the
    marginal distribution; the CONFIDENCE in them is much lower than the
    observation count suggests, which is why the independent count is
    returned next to it.
    """
    values = pd.Series(prices).astype(float).dropna()
    if len(values) < 60:
        raise ValidationError(
            f"volatility_cone: {len(values)} observations is not enough "
            "history for a cone. The shortest horizon needs many "
            "non-overlapping windows to have a distribution at all."
        )
    returns = np.log(values / values.shift(1)).dropna()
    n = len(returns)

    rows: List[Dict[str, Any]] = []
    for horizon in sorted({int(h) for h in horizons}):
        if horizon < 2 or horizon > n // 3:
            continue
        rolling = returns.rolling(horizon).std(ddof=1) * math.sqrt(TRADING_DAYS)
        realized = rolling.dropna().to_numpy()
        if realized.size < 5:
            continue
        independent = int(n // horizon)
        row = {
            "horizon_days": horizon,
            "n_windows": int(realized.size),
            "independent_windows": independent,
            "min": float(np.min(realized)),
            "p10": float(np.percentile(realized, 10)),
            "p25": float(np.percentile(realized, 25)),
            "median": float(np.percentile(realized, 50)),
            "p75": float(np.percentile(realized, 75)),
            "p90": float(np.percentile(realized, 90)),
            "max": float(np.max(realized)),
            "current": float(realized[-1]),
        }
        if current_implied and horizon in current_implied:
            iv = float(current_implied[horizon])
            row["implied_vol"] = iv
            row["implied_percentile"] = float((realized < iv).mean() * 100.0)
            row["implied_vs_median"] = float(iv - row["median"])
        rows.append(row)

    if not rows:
        raise ValidationError(
            "volatility_cone: no horizon had enough windows. Every requested "
            f"horizon was under 2 days or over a third of the {n} returns "
            "available."
        )

    warnings: List[str] = []
    thin = [r["horizon_days"] for r in rows if r["independent_windows"] < 10]
    if thin:
        warnings.append(
            f"Horizons {thin} have fewer than 10 INDEPENDENT windows. Their "
            "percentiles are computed from overlapping samples and describe "
            "a handful of distinct periods -- read them as indicative."
        )
    warnings.append(
        "Realized volatility is close-to-close and therefore misses "
        "overnight gaps' contribution to intraday risk. For a cone meant to "
        "be compared against intraday implied, a Garman-Klass or Yang-Zhang "
        "estimator is the better input."
    )
    return {"n_returns": int(n), "cone": rows, "warnings": warnings}


def analyze_vol_term_structure(
    implied_by_expiry: Dict[float, float],
) -> Dict[str, Any]:
    """
    Whether the volatility term structure is in contango or backwardation,
    and what the forward volatilities between expiries are.

    THE FORWARD VOL IS THE POINT. A trader looking at 30-day IV of 25 and
    60-day IV of 28 is not being offered 28 for the second month -- they are
    being offered whatever makes the total variance add up, which is
    sqrt((28^2*60 - 25^2*30)/30) = 30.6. Calendar spreads are priced off
    that number, not off the quoted levels, and the difference is large
    enough to reverse the sign of a trade.

    A NEGATIVE FORWARD VARIANCE IS AN ARBITRAGE and is reported as one.
    Total variance must be non-decreasing in maturity; if the quotes say
    otherwise, a calendar spread is free money, which in practice means one
    of the quotes is stale.

    Keys are years to expiry, values are implied volatilities as decimals.
    """
    if len(implied_by_expiry) < 2:
        raise ValidationError(
            "analyze_vol_term_structure: a term structure needs at least two "
            "expiries."
        )
    points = sorted(
        (float(t), float(v))
        for t, v in implied_by_expiry.items()
        if float(t) > 0 and float(v) > 0
    )
    if len(points) < 2:
        raise ValidationError(
            "analyze_vol_term_structure: fewer than two expiries had a "
            "positive maturity and a positive volatility."
        )

    forwards: List[Dict[str, Any]] = []
    violations: List[Dict[str, Any]] = []
    for (t1, v1), (t2, v2) in zip(points, points[1:]):
        w1, w2 = v1 * v1 * t1, v2 * v2 * t2
        gap = t2 - t1
        forward_variance = (w2 - w1) / gap
        entry = {
            "from_expiry": t1,
            "to_expiry": t2,
            "spot_vol_near": v1,
            "spot_vol_far": v2,
            "forward_variance": float(forward_variance),
            "forward_vol": (
                float(math.sqrt(forward_variance)) if forward_variance > 0 else None
            ),
        }
        forwards.append(entry)
        if forward_variance <= 0:
            violations.append(entry)

    slope = points[-1][1] - points[0][1]
    if slope > 0.005:
        shape = "contango"
    elif slope < -0.005:
        shape = "backwardation"
    else:
        shape = "flat"

    warnings: List[str] = []
    if violations:
        warnings.append(
            f"{len(violations)} interval(s) have NEGATIVE forward variance, "
            "which is a calendar arbitrage: total implied variance must be "
            "non-decreasing in maturity. In practice this means a stale "
            "quote rather than free money -- check the near expiry first, "
            "it is usually the illiquid one."
        )
    if shape == "backwardation":
        warnings.append(
            "Backwardation -- near-dated vol above far-dated -- normally "
            "means the market is pricing a known near-term event or is "
            "already in a stress regime. Selling the front against the back "
            "on 'mean reversion' is short exactly that event."
        )
    warnings.append(
        "Forward vol is what a calendar spread actually prices, and it is "
        "not the difference between the quoted legs. Trading off the quoted "
        "levels can reverse the sign of the position."
    )

    return {
        "n_expiries": len(points),
        "shape": shape,
        "slope": float(slope),
        "term_structure": [{"expiry_years": t, "implied_vol": v} for t, v in points],
        "forward_vols": forwards,
        "arbitrage_violations": violations,
        "warnings": warnings,
    }


# ── consistency checks ──────────────────────────────────────────────────


def check_put_call_parity(
    *,
    call_price: float,
    put_price: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    tolerance_bps: float = 25.0,
) -> Dict[str, Any]:
    """
    Whether a call and a put on the same strike and expiry are mutually
    consistent, and what the violation is worth if they are not.

    C - P = S*exp(-qT) - K*exp(-rT). This is a NO-MODEL identity: it follows
    from the payoffs alone and holds whatever the volatility, whatever the
    distribution, whatever the model. That is what makes it the right first
    check on a quoted chain -- a violation is a data problem or an
    opportunity, and never a modelling disagreement.

    THE USUAL CAUSE IS NOT ARBITRAGE. In order of likelihood: the two quotes
    are from different timestamps, one leg is a stale last-traded price
    rather than a mid, the dividend assumption is wrong, or the underlying
    is hard to borrow (which shows up as an apparent parity violation of
    exactly the borrow cost). Each is checked against below by reporting the
    IMPLIED dividend yield and the implied forward, so the failure mode is
    identifiable rather than just flagged.

    The violation is reported in basis points of the strike, which is the
    scale a trader can compare against a bid-ask spread.
    """
    spot = _positive(spot, "spot")
    strike = _positive(strike, "strike")
    t = _positive(time_to_expiry, "time_to_expiry")
    _option_inputs(
        spot=spot,
        strike=strike,
        time_to_expiry=t,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    call_price = float(call_price)
    put_price = float(put_price)
    if call_price < 0 or put_price < 0:
        raise ValidationError("option prices cannot be negative")

    discount = math.exp(-float(risk_free_rate) * t)
    growth = math.exp(-float(dividend_yield) * t)

    left = call_price - put_price
    right = spot * growth - strike * discount
    violation = left - right
    violation_bps = violation / strike * 1e4

    # If parity is to hold exactly, what dividend yield and what forward
    # would the quotes imply? Either being implausible identifies the cause.
    implied_growth = (call_price - put_price + strike * discount) / spot
    implied_q = -math.log(implied_growth) / t if implied_growth > 0 else float("nan")
    implied_forward = (call_price - put_price) / discount + strike

    breached = abs(violation_bps) > float(tolerance_bps)
    warnings: List[str] = []
    if breached:
        warnings.append(
            f"Parity is off by {violation_bps:.1f} bps of strike. Before "
            "reading this as an arbitrage, rule out the four things that "
            "cause it far more often: quotes from different timestamps, a "
            "last-traded price standing in for a mid, a wrong dividend "
            f"assumption (parity holds exactly at q={implied_q:.4f}), and a "
            "hard-to-borrow underlying, where the apparent violation is the "
            "borrow cost."
        )
        if violation > 0:
            warnings.append(
                "The call is rich relative to the put: the conversion (short "
                "call, long put, long stock) is the direction that captures "
                "it, financing costs and borrow permitting."
            )
        else:
            warnings.append(
                "The put is rich relative to the call: the reversal (long "
                "call, short put, short stock) is the direction, which "
                "requires a locate and pays the borrow."
            )
    warnings.append(
        "Put-call parity is model-free -- it follows from the payoffs and "
        "holds under any distribution. It is a data-quality check first and "
        "a trade signal a distant second."
    )

    return {
        "call_minus_put": float(left),
        "forward_minus_strike_pv": float(right),
        "violation": float(violation),
        "violation_bps_of_strike": float(violation_bps),
        "within_tolerance": not breached,
        "tolerance_bps": float(tolerance_bps),
        "implied_dividend_yield": (
            float(implied_q) if math.isfinite(implied_q) else None
        ),
        "implied_forward": float(implied_forward),
        "warnings": warnings,
    }


def implied_forward_price(
    *,
    spot: float,
    time_to_expiry: float,
    risk_free_rate: float,
    dividend_yield: float = 0.0,
    borrow_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    The forward price implied by carry, with every component broken out.

    F = S * exp((r - q - b) * T). The decomposition is the reason to have
    this as a tool rather than a line of arithmetic: when a quoted future
    disagrees with the computed forward, the question is always WHICH term
    is wrong, and a single number cannot answer it.

    BORROW IS SEPARATE FROM DIVIDEND on purpose. Both reduce the forward and
    both are often folded into one "carry" number, but they behave
    differently: the dividend is a known cash amount for a listed name, and
    the borrow is a floating rate that can move 500 bps in a day on a
    squeezed stock. A model that cannot tell them apart attributes a borrow
    spike to a dividend surprise.
    """
    spot = _positive(spot, "spot")
    t = _positive(time_to_expiry, "time_to_expiry")
    # `dividend_yield` goes through `_option_inputs` alongside the other two
    # rates rather than being trusted. Left out, it was the one term in
    # `r - q - b` with no bound at all: q=1e5 returned a forward of exactly
    # 0.0 as though the position were worthless, and q=-800 escaped as a
    # bare OverflowError from math.exp rather than as a ValidationError
    # naming the argument. Both are the failure `_bounded` exists to stop.
    _option_inputs(
        spot=spot,
        time_to_expiry=t,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    _bounded(borrow_rate, "borrow_rate", low=-MAX_RATE, high=MAX_RATE)
    r, q, b = float(risk_free_rate), float(dividend_yield), float(borrow_rate)
    net_carry = r - q - b
    forward = spot * math.exp(net_carry * t)

    return {
        "spot": float(spot),
        "forward": float(forward),
        "time_to_expiry": float(t),
        "net_carry_rate": float(net_carry),
        "basis": float(forward - spot),
        "basis_pct": float((forward / spot - 1.0) * 100.0),
        "components": {
            "financing": float(spot * (math.exp(r * t) - 1.0)),
            "dividend": float(-spot * (1.0 - math.exp(-q * t))),
            "borrow": float(-spot * (1.0 - math.exp(-b * t))),
        },
        "warnings": [
            "Borrow is reported separately from the dividend because they "
            "behave differently: a listed name's dividend is a known cash "
            "amount, while borrow is a floating rate that can move hundreds "
            "of basis points in a day on a squeezed stock. Folding them "
            "into one carry number attributes a borrow spike to a dividend "
            "surprise.",
            "This is the CARRY forward. A quoted future differing from it is "
            "the market disagreeing about one of the three components, most "
            "often borrow.",
        ],
    }


# ── what the market is saying, and what the hedge costs ─────────────────


def expected_move(
    *,
    spot: float,
    implied_vol: float,
    days: float,
    realized_moves: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    The move the option market is pricing over a horizon, with the standard
    misreading pre-empted.

    THE NUMBER IS ONE STANDARD DEVIATION, and it gets quoted as "the
    expected move" and then read as a bound. It is not a bound. Under the
    model's own assumptions the move exceeds it about 32% of the time --
    roughly one earnings print in three -- and a strategy that sells the
    straddle because "the move is priced at 5%" is short exactly that third.

    Both conventions are returned. The straddle approximation
    (0.8 * S * sigma * sqrt(T)) is what a desk quotes because it is what the
    at-the-money straddle costs; the one-standard-deviation move
    (S * sigma * sqrt(T)) is the distributional statement. They differ by
    about 20% and confusing them is a real source of mispriced event trades.

    Pass `realized_moves` -- past absolute moves over the same horizon, as
    decimals -- to get the honest comparison: how often the market's number
    was actually exceeded historically, rather than how often the lognormal
    says it should be.
    """
    spot = _positive(spot, "spot")
    vol = _positive(implied_vol, "implied_vol")
    days = _positive(days, "days")
    _option_inputs(spot=spot, volatility=vol)
    _bounded(days, "days", low=1e-6, high=MAX_TIME_TO_EXPIRY * 365.0, unit=" days")
    t = days / 365.0
    sigma_move = spot * vol * math.sqrt(t)
    straddle = 0.8 * sigma_move

    result: Dict[str, Any] = {
        "spot": float(spot),
        "implied_vol": float(vol),
        "days": float(days),
        "one_sd_move": float(sigma_move),
        "one_sd_move_pct": float(sigma_move / spot * 100.0),
        "straddle_approximation": float(straddle),
        "straddle_approximation_pct": float(straddle / spot * 100.0),
        "upper_1sd": float(spot + sigma_move),
        "lower_1sd": float(spot - sigma_move),
        "theoretical_exceedance_pct": 31.7,
    }
    warnings = [
        "This is a ONE STANDARD DEVIATION move, not a bound. The model's "
        "own assumptions have it exceeded about 32% of the time -- one "
        "event in three. 'The move is priced at 5%' is a distributional "
        "statement, and selling the straddle on it is short that third.",
        "Two conventions are returned and they differ by ~20%: the straddle "
        "approximation is what the at-the-money straddle costs, the 1-sd "
        "move is the distributional number. Mixing them misprices event "
        "trades.",
    ]

    if realized_moves is not None:
        moves = np.asarray([abs(float(m)) for m in realized_moves], dtype=float)
        moves = moves[np.isfinite(moves)]
        if moves.size >= 3:
            implied_pct = sigma_move / spot
            exceeded = float((moves > implied_pct).mean() * 100.0)
            result["realized"] = {
                "n_observations": int(moves.size),
                "median_move_pct": float(np.median(moves) * 100.0),
                "mean_move_pct": float(moves.mean() * 100.0),
                "max_move_pct": float(moves.max() * 100.0),
                "exceeded_implied_pct": exceeded,
            }
            if moves.size < 8:
                warnings.append(
                    f"The exceedance rate is computed from {moves.size} past "
                    "moves. At that count the standard error is larger than "
                    "the difference it is being used to detect."
                )
            elif exceeded > 45:
                warnings.append(
                    f"The market's implied move was exceeded {exceeded:.0f}% "
                    "of the time historically, against a theoretical 32%. "
                    "Either this name's moves are fatter-tailed than "
                    "lognormal (usual) or the option is cheap (rare)."
                )
            elif exceeded < 20:
                warnings.append(
                    f"The implied move was exceeded only {exceeded:.0f}% of "
                    "the time historically, against a theoretical 32%. The "
                    "straddle has been rich on this name -- which is also "
                    "what a sample that happens to exclude a crisis looks "
                    "like."
                )
    result["warnings"] = warnings
    return result


def simulate_delta_hedge(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    implied_vol: float,
    realized_vol: float,
    risk_free_rate: float = 0.0,
    option_type: str = "call",
    n_hedges: int = 21,
    n_paths: int = 500,
    transaction_cost_bps: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    What a delta-hedged option position actually earns, when the volatility
    you hedge at is not the volatility that shows up.

    THE THEORETICAL P&L IS KNOWN and this measures the gap to it. A
    continuously hedged option earns (sigma_realized^2 - sigma_implied^2)
    times the dollar gamma, integrated over the life -- so selling a
    30-vol option into 20-vol realized makes money in expectation. That is
    the trade. What the closed form does not tell you is the DISPERSION,
    and the dispersion is what determines whether the trade is sized
    correctly.

    DISCRETE HEDGING ERROR SCALES AS 1/sqrt(n_hedges), not 1/n_hedges. Going
    from daily to twice-daily hedging cuts the standard deviation of the
    error by 29%, not by half, while doubling the transaction cost. That
    tradeoff is the reason to simulate rather than to compute, and the
    result reports both sides of it.

    PATH DEPENDENCE IS THE POINT. The same realized volatility earns
    different amounts depending on WHERE the underlying spent its time: gamma
    is concentrated at the money, so a path that oscillates around the strike
    collects far more than one that trends away and realizes the same vol.
    The percentiles across paths show that spread directly.
    """
    spot = _positive(spot, "spot")
    strike = _positive(strike, "strike")
    t = _positive(time_to_expiry, "time_to_expiry")
    iv = _positive(implied_vol, "implied_vol")
    rv = _positive(realized_vol, "realized_vol")
    _option_inputs(
        spot=spot,
        strike=strike,
        time_to_expiry=t,
        volatility=iv,
        risk_free_rate=risk_free_rate,
    )
    _bounded(rv, "realized_vol", low=1e-8, high=MAX_VOLATILITY)
    n_hedges = max(1, int(n_hedges))
    n_paths = max(1, int(n_paths))
    option_type = str(option_type).lower()
    if option_type not in ("call", "put"):
        raise ValidationError(
            f"option_type must be 'call' or 'put', got {option_type!r}"
        )

    dt = t / n_hedges
    rng = np.random.default_rng(int(seed))
    cost_rate = float(transaction_cost_bps) / 1e4

    entry = price_option(
        spot=spot,
        strike=strike,
        time_to_expiry=t,
        volatility=iv,
        risk_free_rate=risk_free_rate,
        option_type=option_type,
    )
    premium = float(entry["price"])

    # Short the option, hedge it long delta. The sign convention is stated
    # because the P&L flips with it and "the hedged P&L" is ambiguous.
    pnl = np.zeros(n_paths)
    costs = np.zeros(n_paths)
    for p in range(n_paths):
        s = spot
        cash = premium  # sold the option
        shares = 0.0
        path_cost = 0.0
        for step in range(n_hedges):
            remaining = t - step * dt
            greeks = option_greeks(
                spot=s,
                strike=strike,
                time_to_expiry=max(remaining, 1e-8),
                volatility=iv,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
            )
            target = greeks["delta"]  # long delta hedges the short option
            trade = target - shares
            traded_value = abs(trade) * s
            path_cost += traded_value * cost_rate
            cash -= trade * s + traded_value * cost_rate
            shares = target
            # Advance one step under the REALIZED vol, not the implied one.
            z = rng.standard_normal()
            s *= math.exp(
                (risk_free_rate - 0.5 * rv * rv) * dt + rv * math.sqrt(dt) * z
            )
            cash *= math.exp(risk_free_rate * dt)
        intrinsic = (
            max(s - strike, 0.0) if option_type == "call" else max(strike - s, 0.0)
        )
        pnl[p] = cash + shares * s - intrinsic
        costs[p] = path_cost

    # The continuous-hedging expectation, for comparison. Approximated at
    # the initial dollar gamma, which is exact only if spot does not move --
    # it is a reference point, not a prediction, and is labelled as one.
    dollar_gamma = 0.5 * entry["gamma"] * spot * spot
    theoretical = dollar_gamma * (iv * iv - rv * rv) * t

    warnings = [
        "Sign convention: SHORT the option, delta-hedged long. A positive "
        "P&L means implied exceeded realized by more than the hedging "
        "friction cost.",
        f"Discrete hedging error scales as 1/sqrt(n_hedges). Doubling from "
        f"{n_hedges} to {2 * n_hedges} rehedges cuts the P&L standard "
        "deviation by about 29%, not by half, while doubling the "
        "transaction cost.",
        "The theoretical number is evaluated at the INITIAL dollar gamma. "
        "It is a reference point, not a prediction -- real gamma varies "
        "along the path, which is most of why the simulated distribution is "
        "as wide as it is.",
    ]
    if n_paths < 100:
        warnings.append(
            f"{n_paths} paths: the percentiles are noisy. The mean is the "
            "only number here with much precision at this count."
        )

    return {
        "option_premium": premium,
        "implied_vol": float(iv),
        "realized_vol": float(rv),
        "n_hedges": n_hedges,
        "n_paths": n_paths,
        "mean_pnl": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "std_pnl": float(pnl.std(ddof=1)) if n_paths > 1 else 0.0,
        "p05_pnl": float(np.percentile(pnl, 5)),
        "p95_pnl": float(np.percentile(pnl, 95)),
        "worst_pnl": float(pnl.min()),
        "best_pnl": float(pnl.max()),
        "win_rate": float((pnl > 0).mean()),
        "mean_transaction_cost": float(costs.mean()),
        "theoretical_continuous_pnl": float(theoretical),
        "pnl_as_pct_of_premium": (
            float(pnl.mean() / premium * 100.0) if premium > 0 else None
        ),
        "warnings": warnings,
    }


def option_risk_scenarios(
    *,
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.0,
    option_type: str = "call",
    quantity: float = 1.0,
    spot_shocks: Sequence[float] = (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20),
    vol_shocks: Sequence[float] = (-0.10, -0.05, 0.0, 0.05, 0.10),
    days_forward: float = 0.0,
) -> Dict[str, Any]:
    """
    A full revaluation grid over spot and volatility, rather than a greek
    approximation of one.

    WHY REVALUE RATHER THAN APPROXIMATE. The delta-gamma estimate of P&L
    under a 20% move is wrong by the third and higher order terms, and for
    an option those are not small: a Taylor expansion around today's spot
    misses most of the convexity that makes the position worth holding. This
    reprices the option at every node instead, so the number is exact under
    the model.

    THE CORRELATION BETWEEN THE AXES IS NOT MODELLED and that is the
    grid's main limitation. Spot down 20% and vol unchanged is a cell in
    this table and is not a market state that occurs -- equity vol rises
    when spot falls, reliably enough that the joint scenario is the only one
    worth reading. The diagonal is flagged for that reason: the realistic
    stress path runs down-and-left to up-and-right, not across a row.

    `days_forward` decays the position before revaluing, which is how a
    weekend or an overnight gap should actually be stressed.
    """
    spot = _positive(spot, "spot")
    strike = _positive(strike, "strike")
    t = _positive(time_to_expiry, "time_to_expiry")
    vol = _positive(volatility, "volatility")
    _option_inputs(
        spot=spot,
        strike=strike,
        time_to_expiry=t,
        volatility=vol,
        risk_free_rate=risk_free_rate,
    )
    quantity = float(quantity)
    remaining = t - float(days_forward) / 365.0
    if remaining <= 0:
        raise ValidationError(
            f"days_forward={days_forward} takes the option past expiry "
            f"({t * 365:.1f} days away). Revaluing an expired option is a "
            "payoff calculation, not a scenario."
        )

    base = price_option(
        spot=spot,
        strike=strike,
        time_to_expiry=t,
        volatility=vol,
        risk_free_rate=risk_free_rate,
        option_type=option_type,
    )["price"]
    base_value = quantity * base

    grid: List[Dict[str, Any]] = []
    worst = None
    for ds in spot_shocks:
        shocked_spot = spot * (1.0 + float(ds))
        if shocked_spot <= 0:
            continue
        row: Dict[str, Any] = {"spot_shock_pct": float(ds) * 100.0, "cells": []}
        for dv in vol_shocks:
            shocked_vol = vol + float(dv)
            if shocked_vol <= 0:
                continue
            value = (
                quantity
                * price_option(
                    spot=shocked_spot,
                    strike=strike,
                    time_to_expiry=remaining,
                    volatility=shocked_vol,
                    risk_free_rate=risk_free_rate,
                    option_type=option_type,
                )["price"]
            )
            pnl = value - base_value
            cell = {
                "vol_shock": float(dv),
                "value": float(value),
                "pnl": float(pnl),
                "pnl_pct_of_base": (
                    float(pnl / base_value * 100.0) if base_value != 0 else None
                ),
            }
            row["cells"].append(cell)
            if worst is None or pnl < worst["pnl"]:
                worst = {
                    "pnl": float(pnl),
                    "spot_shock_pct": float(ds) * 100.0,
                    "vol_shock": float(dv),
                }
        grid.append(row)

    return {
        "base_value": float(base_value),
        "quantity": quantity,
        "days_forward": float(days_forward),
        "grid": grid,
        "worst_case": worst,
        "warnings": [
            "Every cell is a full REVALUATION, not a delta-gamma "
            "approximation. Under a 20% move the Taylor estimate misses "
            "most of the convexity, which for an option is the part worth "
            "owning.",
            "The two axes are shocked INDEPENDENTLY and the market does not "
            "move that way. Equity vol rises when spot falls; 'spot -20%, "
            "vol unchanged' is a cell in this table and not a state of the "
            "world. Read the down-spot/up-vol diagonal, not a row.",
        ],
    }


__all__ = [
    "CONE_HORIZONS",
    "TRADING_DAYS",
    "analyze_strategy",
    "analyze_vol_term_structure",
    "check_put_call_parity",
    "expected_move",
    "fit_volatility_smile",
    "implied_forward_price",
    "option_greeks",
    "option_risk_scenarios",
    "simulate_delta_hedge",
    "volatility_cone",
]
