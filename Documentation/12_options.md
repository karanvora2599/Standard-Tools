# Options Pricing, Greeks & Implied Volatility

`standard_quant_tools.analysis.options` — Black-Scholes-Merton pricing, Greeks, and implied volatility for **European options only** (no early exercise). Dependency-free: the standard normal CDF/PDF are computed via `math.erf` (stdlib), not scipy — pricing and Greeks never require scipy. `implied_volatility` also has no hard scipy dependency: Newton-Raphson (vega as the derivative) with a bisection fallback over a practical volatility bracket.

---

## Pricing

```python
from standard_quant_tools.analysis.options import black_scholes_price

# Hull's textbook example: S=42, K=40, T=0.5y, r=10%, sigma=20%
call = black_scholes_price(42, 40, 0.5, 0.10, 0.20, "call")
put  = black_scholes_price(42, 40, 0.5, 0.10, 0.20, "put")
print(call, put)   # ~4.76, ~0.81
```

`dividend_yield` (default `0.0`) extends plain Black-Scholes to the Merton (1973) continuous-dividend-yield case — pass a nonzero value for a dividend-paying underlying or an index/FX rate differential:

```python
call_with_div = black_scholes_price(42, 40, 0.5, 0.10, 0.20, "call", dividend_yield=0.03)
```

**A yield is bounded on magnitude, never on sign.** `dividend_yield` is
accepted anywhere in `[-MAX_RATE, MAX_RATE]` (`MAX_RATE = 10.0`, matching
`analysis.derivatives`), because a negative continuous yield is the ordinary
case for an FX option's foreign rate and for a commodity whose convenience
yield exceeds its storage cost. A `>= 0` guard refused exactly those.

**Scope, stated explicitly:** `time_to_expiry` must be strictly `> 0`. An expired or expiring option's value is its intrinsic value (`max(S-K, 0)` / `max(K-S, 0)`) — not something these formulas are valid for. Compute that directly rather than calling `black_scholes_price` with `time_to_expiry=0`.

---

## Greeks

```python
from standard_quant_tools.analysis.options import black_scholes_greeks

greeks = black_scholes_greeks(42, 40, 0.5, 0.10, 0.20, "call")
print(greeks)
# {'delta': 0.779, 'gamma': 0.050, 'vega': 8.813, 'theta': -4.559, 'rho': 13.982, 'd1': 0.769, 'd2': 0.628}
```

**Units, stated explicitly (a common source of confusion):**
- `vega` is the price change per **1.0** (100 percentage points) of volatility — divide by 100 for the conventional "per 1 vol point" quote.
- `theta` is per **year** (raw), not per calendar day — divide by 365 for the conventional "daily time decay" quote.

Both are left raw rather than pre-scaled, so nothing is silently rescaled behind your back. `d1`/`d2` are included so a caller who also wants the price doesn't have to recompute them.

Every Greek formula here is cross-validated in `tests/analysis/test_options.py` against a finite-difference derivative of `black_scholes_price` itself (e.g. `delta ≈ (price(S+h) - price(S-h)) / 2h`), not just trusted as textbook formulas typed in correctly.

---

## Implied Volatility

```python
from standard_quant_tools.analysis.options import implied_volatility

result = implied_volatility(
    option_price=4.759422392871528,
    spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10, option_type="call",
)
print(result)
# {'implied_volatility': 0.2, 'converged': True, 'iterations': 1, 'method': 'newton',
#  'price_error': 0.0, 'at_bound': False}
```

**Solve method:** Newton-Raphson (vega as the derivative) with a bisection fallback over `[1e-6, 5.0]` (500% annualized vol — a deliberately generous practical cap) when vega falls below `VEGA_FLOOR` or a step leaves that bracket. Newton alone is not robust here: vega can be tiny for deep ITM/OTM options, making a raw Newton step overshoot or divide by ~zero. Bisection is slower but guaranteed to converge whenever a solution exists in the bracket, since Black-Scholes price is strictly increasing in volatility for any fixed inputs.

**Convergence is declared on volatility, not on price.** The test used to be an absolute price tolerance (`tol`) applied *before* any step was taken, so where vega is small a volatility wrong by hundreds of points still priced inside `1e-6` and the solver returned its initial guess with `converged=True`: four short-dated puts came back at exactly `0.2` for true vols of 3.00, 1.20 and 0.45, and over a 700-case grid 28 of the 549 "converged" answers were off by more than 0.01 vol. A Newton step is now always taken, and the solver has converged when the step it just took moved volatility by less than `tol_sigma` (default `1e-8`) — which is the price tolerance scaled by vega, and is meaningful at any vega. Bisection converges on the width of its bracket. `tol` remains the price tolerance the result reports `price_error` against; it no longer declares convergence on its own.

**No-arbitrage bound check runs first:** `option_price` must lie between the option's lower bound (volatility → 0) and upper bound (volatility → ∞); a price outside that range raises `ValidationError` immediately rather than searching for a volatility that can't exist. Equality within `BOUND_TOLERANCE` is *inside* the bound — the pricer itself produces a deep-in-the-money price bit-for-bit equal to intrinsic, and a strict `<` refused 77 of 700 such prices. A price at intrinsic is solved by bisection for the largest volatility that still reproduces it and comes back with `at_bound=True`: every volatility at or below that number prices the same, so it is a ceiling rather than an estimate. A price of `0.0` is refused with the reason: it is what the pricer returns when an option is so far from the money that its value underflows, and no volatility is identifiable from it.

```python
from standard_quant_tools.error import ValidationError

try:
    implied_volatility(option_price=50.0, spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10)
except ValidationError as e:
    print(e)   # "... is outside the no-arbitrage range [3.950823, 42.000000] ..."
               # the lower bound is the discounted intrinsic, not zero
```

---

## Via Agent Tools

Two tools carry this module onto the agent surface, registered in
`get_agent_tools()` and `dispatch()` like every other tool in the library. They
are two of the derivatives runtime's twelve — the other ten are in
[21_derivatives.md](21_derivatives.md).

```python
from standard_quant_tools.agent.tools import get_option_pricing, get_implied_volatility, dispatch
from standard_quant_tools.agent.models import OptionPricingInput, ImpliedVolatilityInput

result = get_option_pricing(OptionPricingInput(
    spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10,
    volatility=0.20, option_type="call",
))
print(result.price, result.greeks.delta)

# Or via dispatch(), same as any other tool:
result = dispatch("get_option_pricing", {
    "spot": 42, "strike": 40, "time_to_expiry": 0.5,
    "risk_free_rate": 0.10, "volatility": 0.20, "option_type": "call",
})

iv_result = get_implied_volatility(ImpliedVolatilityInput(
    option_price=4.76, spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10,
))
print(iv_result.implied_volatility)
```

`get_option_pricing` bundles price + all five Greeks in one call (avoiding two
separate round trips for a common combined need); `get_implied_volatility` is
the reverse direction (price known, volatility unknown).

**Two things the tool does that this module does not.** `get_option_pricing`
takes `model` (`black_scholes`, `black_76`, `bachelier`, `binomial`) and
`american`, so it reaches `analysis/pricing.py`'s lattice — the one model here
that prices early exercise — and it is not European-only the way this module
is. And its greeks are **scaled to the conventional quote**: `vega` per one
volatility point (0.01) and `theta` per calendar day, where
`black_scholes_greeks` above returns both raw. The same inputs give
`vega 8.813` from the function and `0.088134` from the tool; neither is wrong
and they are not the same number.

---

## Error Handling

```python
from standard_quant_tools.error import ValidationError

# spot/strike/time_to_expiry/volatility <= 0 or above their magnitude limits
# (1e12 for a price, 100 for a year count or a volatility), and an unknown
# option_type, all raise ValidationError with a message naming the offending
# field and value. A negative dividend_yield does not: it is bounded on
# magnitude at MAX_RATE and priced as given.
```

`black_scholes_price`/`black_scholes_greeks`/`implied_volatility` never raise anything other than `ValidationError` — there is no network call, external API, or optional dependency in this module to fail in a different way.
