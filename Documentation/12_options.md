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
# {'implied_volatility': 0.2, 'converged': True, 'iterations': 1, 'method': 'newton'}
```

**Solve method:** Newton-Raphson (vega as the derivative) with a bisection fallback over `[1e-6, 5.0]` (500% annualized vol — a deliberately generous practical cap) when Newton fails to converge or steps outside that bracket. Newton alone is not robust here: vega can be tiny for deep ITM/OTM options, making a raw Newton step overshoot or divide by ~zero. Bisection is slower but guaranteed to converge whenever a solution exists in the bracket, since Black-Scholes price is strictly increasing in volatility for any fixed inputs.

**No-arbitrage bound check runs first:** `option_price` must lie strictly between the option's lower bound (volatility → 0) and upper bound (volatility → ∞); a price outside that range raises `ValidationError` immediately rather than searching for a volatility that can't exist.

```python
from standard_quant_tools.error import ValidationError

try:
    implied_volatility(option_price=50.0, spot=42, strike=40, time_to_expiry=0.5, risk_free_rate=0.10)
except ValidationError as e:
    print(e)   # "... is outside the no-arbitrage range (0.000000, 42.000000) ..."
```

---

## Via Agent Tools

Two tools, registered in `get_agent_tools()` and `dispatch()` like every other tool in the library:

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

`get_option_pricing` bundles price + all five Greeks in one call (avoiding two separate round trips for a common combined need); `get_implied_volatility` is the reverse direction (price known, volatility unknown).

---

## Error Handling

```python
from standard_quant_tools.error import ValidationError

# spot/strike/time_to_expiry/volatility <= 0, an unknown option_type, or a
# negative dividend_yield all raise ValidationError with a message naming
# the offending field and value.
```

`black_scholes_price`/`black_scholes_greeks`/`implied_volatility` never raise anything other than `ValidationError` — there is no network call, external API, or optional dependency in this module to fail in a different way.
