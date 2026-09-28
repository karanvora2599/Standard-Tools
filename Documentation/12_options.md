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

**So is `risk_free_rate`, and so is each rate times the time.** The rate had
no bound at all: `r=-1e300` raised a bare `OverflowError`, `r=nan` priced to
NaN (a null price at the agent surface, with no warning), and `r=1e300`
returned a call worth exactly the spot. Both rates are now refused unless
finite and within `±MAX_RATE`, and each product `rate × time_to_expiry` is
refused above `MAX_EXPONENT = 700` (matching `analysis.derivatives`) — each
factor can be inside its own bound while the product is not: `r=-9` at
`T=100` asks `exp()` for 900. Negative rates inside the bound price
normally. A price that still comes out non-finite is refused rather than
returned.

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

## Whole Chains in One Call

`standard_quant_tools.analysis.options_batch` — the functions above over a
whole chain: one call per chain instead of one per contract. Every argument
broadcasts, numpy-style, so what varies across the chain is an array and
what the chain shares is a scalar. `is_call` is `True` for a call and
`False` for a put (0/1 accepted); for `option_type` strings pass
`is_call=(np.asarray(option_types) == "call")`.

```python
import numpy as np
from standard_quant_tools.analysis.options_batch import (
    implied_volatility_batch, black_scholes_greeks_batch, zero_gamma_spot,
)

strikes = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
quotes = np.array([11.2, 7.1, 3.9, 1.8, 0.0])        # the last one underflowed
iv = implied_volatility_batch(quotes, 100.0, strikes, 0.25, 0.04, 0.01, True)
iv["implied_volatility"]   # array; NaN where refused
iv["reason"]               # ["solved", ..., "price_not_positive"]
iv["refusals"]             # {"price_not_positive": 1}

greeks = black_scholes_greeks_batch(100.0, strikes[:4], 0.25, iv["implied_volatility"][:4], 0.04, 0.01)
greeks["delta"], greeks["gamma"], greeks["theta"]   # per contract, option_greeks' units

grid = black_scholes_greeks_batch(np.linspace(80, 120, 61), strikes[:4], 0.25, 0.2, 0.04, grid=True)
grid["gamma"].shape        # (4, 61): every contract at every spot
```

**The same numbers, contract for contract.** `implied_volatility_batch` is
`implied_volatility`'s algorithm — the bound check with `BOUND_TOLERANCE`,
Newton on vega converged on `tol_sigma`, the bisection over `[1e-6, 5.0]`,
the at-intrinsic ceiling flagged `at_bound`, the same keyword defaults — and
returns the volatility, `converged`, `iterations`, `method`, `price_error`
and `at_bound` the scalar returns for each contract. `black_scholes_greeks_batch`
returns `analysis.derivatives.option_greeks`' full set in its units — vega
and vanna per volatility point, volga per point squared, theta and charm per
calendar day, rho per rate point, speed, `d1`, `d2` — and `price_option`'s
price. Both run as a compiled kernel with OpenMP across contracts (gated on
total work, like every kernel here) and fall back to numpy under
`SQT_DISABLE_NATIVE=1` or a stale extension; `path` says which ran. The
fallback takes `exp`, `log`, `pow` and `erf` from `math` rather than numpy,
because numpy's vectorised transcendentals can round differently in the last
bit on some CPUs and a last-bit difference in a price is enough to change an
iteration count. The tests hold all three — kernel, fallback and scalar —
to the same results on randomized chains (1e-12 relative on every value,
identical flags, iteration counts and reasons); on this machine they agree
to the bit.

**A quote that cannot be solved is reported; an input that cannot be priced
refuses the batch.** A chain routinely holds quotes no volatility can
reproduce, and one of them must not take the other 475 with it:

| `reason` | The quote | The scalar |
|---|---|---|
| `solved` | a volatility was found | returns it |
| `price_not_positive` | `<= 0`, what an option so far out that its value underflows is quoted at | raises |
| `price_not_finite` | NaN or +inf — a missing quote | raises (as outside the range) |
| `below_lower_bound` / `above_upper_bound` | outside what any volatility produces | raises |
| `no_root_in_bracket` | inside the bounds, but only a volatility past 500% reproduces it | raises |

Those contracts come back with NaN volatility and `converged=False`. An
input outside the pricing domain — spot, strike or time not positive or past
its magnitude limit, a rate or yield past `MAX_RATE`, `rate × time` past
`MAX_EXPONENT`, a discounted strike past a double — is a unit error in the
arrays rather than a fact about one quote, and raises the scalar function's
own `ValidationError`, prefixed with the contract it found it in
(`"implied_volatility_batch: contract 3: strike must be > 0, got -5.0"`, or
`contract (1, 2)` for a 2-D chain).

**`zero_gamma_spot(strike, time_to_expiry, volatility, quantity, spot_low=,
spot_high=)`** is where a book's aggregate signed gamma, `Σ quantity × gamma`,
changes sign inside a bracket. `quantity` carries sign and size (multipliers
folded in; a dealer-positioning convention is a choice of signs the caller
makes); gamma is the same for a call and a put, so the type is not an input,
and dollar gamma crosses at the same spot. The book is valued on `n_grid`
spots (default 201) in one grid call, and each sign change is refined by
Brent's method — one batched evaluation of the whole book per step, about
five steps where bisection needs thirty. **No crossing is invented**: a book
that is long (or short) gamma across the whole bracket returns
`zero_gamma_spot=None` with a `reason` saying which side of zero it sits on,
and a bracket where every gamma underflows to exactly zero — far from the
strikes, close to expiry — is gamma being absent, not a crossing. Every
crossing found is in `crossings`; `zero_gamma_spot` is the one nearest
`reference_spot` (default: the bracket's midpoint). Two crossings closer than
one grid step cancel and are not seen, which is why `grid_step` is returned.

Measured throughput is in [16_performance.md](16_performance.md#option-chains).

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
separate round trips for a common combined need) under every closed-form
model; the binomial lattice returns delta and gamma only, with `vega`,
`theta` and `rho` null and a note saying why. `get_implied_volatility` is
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
# (1e12 for a price, 100 for a year count or a volatility), a risk_free_rate
# or dividend_yield that is not finite or exceeds MAX_RATE in magnitude, a
# rate x time_to_expiry above MAX_EXPONENT, and an unknown option_type, all
# raise ValidationError with a message naming the offending field and value.
# A negative rate or dividend_yield does not: both are bounded on magnitude
# and priced as given.
```

`black_scholes_price`/`black_scholes_greeks`/`implied_volatility` never raise anything other than `ValidationError` — there is no network call, external API, or optional dependency in this module to fail in a different way.
