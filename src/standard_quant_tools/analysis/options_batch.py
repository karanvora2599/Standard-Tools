"""
Black-Scholes-Merton over a whole option chain in one call.

`analysis.options` answers for one contract per call, and a chain is
hundreds of contracts: one implied-volatility solve is about 22 us and one
gamma about half a microsecond, so a 476-contract surface paid 476 Python
calls to solve and a gamma profile over 61 spots paid 29,036. The functions
here take arrays -- broadcast against scalars, the way numpy broadcasts --
and answer for every contract at once:

- `implied_volatility_batch` -- the scalar `implied_volatility`, contract
  for contract.
- `black_scholes_greeks_batch` -- the price and the full greek set of
  `analysis.derivatives.option_greeks`, per contract or on a grid of spots,
  or only the greeks a caller names.
- `zero_gamma_spot` -- where a book's aggregate signed gamma changes sign,
  or a statement that it does not inside the bracket.

THE SAME NUMBERS, NOT CLOSE ONES. Each batch function is the scalar
function's arithmetic, operation for operation, in both of its paths: the
compiled kernel (`_sqt_core`) and the numpy fallback beside it. The
fallback does its arithmetic in numpy but takes exp, log and erf from the
`math` module -- the functions the scalar formulas call -- because
numpy's own vectorised transcendentals may round differently in the last
bit on some CPUs, and a last-bit difference in a price is enough to move an
iteration count or tip a quote across a no-arbitrage bound. Tests hold all
three paths to the same doubles on randomized chains.

WHAT A BATCH REFUSES, AND WHAT IT REPORTS. Two kinds of bad input, handled
differently on purpose:

- An input outside the pricing DOMAIN -- a spot of zero, a negative time, a
  rate of 1e300, a rate x time past what exp() takes -- is a unit error in
  the caller's arrays, and the batch raises the scalar function's own
  `ValidationError`, prefixed with the contract that broke it.
- A QUOTE no volatility can reproduce -- a price of zero, a missing (NaN)
  price, one outside the no-arbitrage bounds, or one only a volatility
  past 500% reaches -- is a fact about the market data, and a chain
  routinely holds some. Those come back per contract as a `reason`, with
  NaN volatility, and never refuse the rest of the chain.

The compiled path is used when the extension is built, matches its
sources, and exports the kernel; `SQT_DISABLE_NATIVE=1` and a stale
extension both fall back here, with the same answers.
"""

from __future__ import annotations

import logging
import math
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from standard_quant_tools._special import norm_cdf_array, norm_pdf_array
from standard_quant_tools.analysis import options as _scalar
from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = True
except ImportError:
    pass

#: Why a contract's implied volatility is or is not reported, by the code
#: the kernel returns. The first is success; the next five are properties of
#: the quote and are reported per contract; the last two are properties of
#: the inputs and refuse the batch.
REASONS = (
    "solved",
    "price_not_positive",
    "price_not_finite",
    "below_lower_bound",
    "above_upper_bound",
    "no_root_in_bracket",
    "not_priceable",
    "invalid_input",
)
_NOT_PRICEABLE = 6
_INVALID_INPUT = 7

#: The solver that produced each volatility, as the scalar names it.
METHODS = ("none", "newton", "bisection")

#: Every array `black_scholes_greeks_batch` returns, in
#: `analysis.derivatives.option_greeks`' units.
GREEKS = (
    "price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
    "charm",
    "speed",
    "d1",
    "d2",
)

GREEK_UNITS = {
    "delta": "change in price per $1 of spot",
    "gamma": "change in delta per $1 of spot",
    "vega": "change in price per 1 volatility point (0.01)",
    "theta": "change in price per calendar day",
    "rho": "change in price per 1 rate point (0.01)",
    "vanna": "change in delta per 1 volatility point",
    "volga": "change in vega per 1 volatility point",
    "charm": "change in delta per calendar day",
    "speed": "change in gamma per $1 of spot",
}

# The domain the scalar validators enforce, restated from analysis.options
# (spot/strike/time/volatility magnitudes) so the vectorised mask and the
# scalar refusal it hands over to agree on every contract.
_MAX_PRICE = 1e12
_MAX_TIME = 100.0
_MAX_VOLATILITY = 100.0
#: The bisection may exit on the PRICE tolerance only once its bracket is
#: narrower than this, as in the scalar solver.
_PRICE_EXIT_WIDTH = 1e-4
#: An exponent at or below this cannot carry a spot or strike of at most
#: `_MAX_PRICE` past the largest double: `_MAX_PRICE * exp(x)` is then at
#: least a factor e^2 short of it, far more than the last-bit error of
#: `math.exp` and of the product. About 680.15.
_DISCOUNT_SCREEN = math.log(sys.float_info.max) - math.log(_MAX_PRICE) - 2.0


# ── the transcendentals, from `math` ────────────────────────────────────


def _from_math(fn: Callable[..., float], nargs: int = 1) -> Callable[..., np.ndarray]:
    """
    An elementwise version of a `math` function that returns float arrays.

    Why not numpy's own: see the module docstring. `np.exp` and `np.log`
    agree with `math.exp` and `math.log` on most machines and differ in the
    last bit on some (numpy dispatches SIMD implementations by CPU), while
    this is the scalar path's function by construction.
    """
    ufunc = np.frompyfunc(fn, nargs, 1)

    def apply(*args: Any) -> np.ndarray:
        arrays = [np.asarray(a, dtype=float) for a in args]
        shape = np.broadcast_shapes(*(a.shape for a in arrays))
        if int(np.prod(shape)) == 0:
            return np.empty(shape, dtype=float)
        return np.asarray(ufunc(*arrays), dtype=float)

    return apply


_exp = _from_math(math.exp)
_log = _from_math(math.log)
# Every formula below squares a volatility as `v * v`, the way both scalar
# modules do. (analysis.options squared through pow(v, 2.0) until the
# CHANGELOG entry of 2026-10-02, and this module with it.)


def _native(name: str) -> Any:
    """The compiled kernel, when the extension is present and exports it.

    Per symbol, so an extension that predates a kernel falls back for that
    kernel alone."""
    if HAS_CPP and _cpp_core is not None:
        return getattr(_cpp_core, name, None)
    return None


# ── inputs ──────────────────────────────────────────────────────────────


def _floats(value: Any, name: str, fn: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValidationError(
            f"{fn}: {name} must be a number or an array of numbers, got "
            f"{value!r:.80}"
        ) from None


def _setting(value: Any, name: str, fn: str, minimum: Optional[float] = None) -> float:
    """A finite scalar configuration value, at or above `minimum`."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{fn}: {name} must be a number, got {value!r}") from None
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        floor = "" if minimum is None else f" and >= {minimum:g}"
        raise ValidationError(f"{fn}: {name} must be finite{floor}, got {value!r}")
    return number


def _whole(value: Any, name: str, fn: str, minimum: int) -> int:
    """A whole number at or above `minimum`; a bool is not one."""
    try:
        number = int(value)
        whole = not isinstance(value, bool) and number == value
    except (TypeError, ValueError, OverflowError):
        whole = False
    if not whole or number < minimum:
        raise ValidationError(
            f"{fn}: {name} must be a whole number >= {minimum}, got {value!r}"
        )
    return number


def _booleans(value: Any, fn: str) -> np.ndarray:
    """`is_call` as a boolean array; 0/1 numbers are accepted, words are not."""
    arr = np.asarray(value)
    if arr.dtype == bool:
        return arr
    if arr.dtype.kind in "iuf" and bool(np.all((arr == 0) | (arr == 1))):
        return arr.astype(bool)
    raise ValidationError(
        f"{fn}: is_call must be True (a call) or False (a put) per contract, "
        f"got an array of dtype {arr.dtype}. For option_type strings, pass "
        "is_call=(np.asarray(option_types) == 'call')."
    )


def _broadcast(
    fn: str, **named: np.ndarray
) -> Tuple[Tuple[int, ...], List[np.ndarray]]:
    """Broadcast the per-contract arrays and flatten them, C-contiguous."""
    try:
        shape = np.broadcast_shapes(*(a.shape for a in named.values()))
    except ValueError:
        shapes = ", ".join(f"{k}{np.shape(v)}" for k, v in named.items())
        raise ValidationError(
            f"{fn}: the per-contract arrays do not broadcast against each "
            f"other ({shapes}). Pass one entry per contract, or a scalar for "
            "a value every contract shares."
        ) from None
    # An array already in the full shape is flattened in place; only the
    # ones that broadcast are copied out.
    return shape, [
        np.ascontiguousarray(
            a if a.shape == shape else np.broadcast_to(a, shape)
        ).reshape(-1)
        for a in named.values()
    ]


def _where(shape: Tuple[int, ...], i: int, what: str = "contract") -> str:
    """A flat index as the caller indexes it."""
    if len(shape) <= 1:
        return f"{what} {i}"
    return f"{what} {tuple(int(x) for x in np.unravel_index(i, shape))}"


def _refuse_first(
    bad: np.ndarray,
    shape: Tuple[int, ...],
    fn: str,
    check: Callable[[int], None],
    what: str = "contract",
) -> None:
    """
    Raise the scalar function's own refusal for the first bad contract.

    The mask finds WHICH contract; the scalar validator says WHAT is wrong
    with it, in the words and with the remedy every single-contract call
    already gives. Nothing here restates a message.
    """
    i = int(np.flatnonzero(bad)[0])
    where = _where(shape, i, what)
    try:
        check(i)
    except ValidationError as err:
        raise ValidationError(f"{fn}: {where}: {err}") from None
    raise ValidationError(  # pragma: no cover - the mask and the check agree
        f"{fn}: {where} is outside the domain these formulas can price."
    )


def _outside_domain(
    spot: np.ndarray,
    strike: np.ndarray,
    t: np.ndarray,
    rate: np.ndarray,
    q: np.ndarray,
    vol: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Every contract the scalar validators would refuse, as one mask.

    Each test is written so NaN and both infinities fail it -- a comparison
    with NaN is False and every bound is finite -- so no separate
    finiteness pass is needed.
    """
    with np.errstate(invalid="ignore", over="ignore"):
        ok = (
            _positive_up_to(spot, _MAX_PRICE)
            & _positive_up_to(strike, _MAX_PRICE)
            & _positive_up_to(t, _MAX_TIME)
            & (np.abs(rate) <= _scalar.MAX_RATE)
            & (np.abs(q) <= _scalar.MAX_RATE)
            & (np.abs(rate * t) <= _scalar.MAX_EXPONENT)
            & (np.abs(q * t) <= _scalar.MAX_EXPONENT)
        )
        if vol is not None:
            ok &= _positive_up_to(vol, _MAX_VOLATILITY)
    return ~ok


def _positive_up_to(x: np.ndarray, high: float) -> np.ndarray:
    """0 < x <= high, False for NaN and for either infinity."""
    return (x > 0) & (x <= high)


def _validate_domain(
    fn: str,
    shape: Tuple[int, ...],
    spot: np.ndarray,
    strike: np.ndarray,
    t: np.ndarray,
    rate: np.ndarray,
    q: np.ndarray,
    vol: Optional[np.ndarray] = None,
) -> None:
    bad = _outside_domain(spot, strike, t, rate, q, vol)
    if not bad.any():
        return

    def check(i: int) -> None:
        _scalar._validate_option_inputs(
            float(spot[i]),
            float(strike[i]),
            float(t[i]),
            1.0 if vol is None else float(vol[i]),
            "call",
        )
        _scalar._validate_rates(float(rate[i]), float(q[i]), float(t[i]))

    _refuse_first(bad, shape, fn, check)


def _validate_discounting(
    fn: str,
    shape: Tuple[int, ...],
    spot: np.ndarray,
    strike: np.ndarray,
    t: np.ndarray,
    rate: np.ndarray,
    q: np.ndarray,
) -> None:
    """
    A discounted strike or spot past a double.

    Each factor can be inside its bound and the product not: a strike of
    1e12 discounted at r=-9 over 77 years is 1e316. The scalar finds this as
    a non-finite price part-way through a solve; a batch refuses it before
    any contract is solved, with the same words.

    Called after `_validate_domain`, so every spot and strike is at most
    `_MAX_PRICE` and every exponent is finite. The exact test -- the
    product, with the exponential from `math` -- runs only on contracts
    whose exponent is past `_DISCOUNT_SCREEN`, since no other contract can
    overflow; on a real chain that is none of them, and the test costs two
    multiplies per contract instead of two `math.exp` calls. The kernel's
    own `not_priceable` code is not a substitute: it reports a price of zero
    as `price_not_positive` before it looks at the discounting, where this
    refuses the batch.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        near = (-rate * t > _DISCOUNT_SCREEN) | (-q * t > _DISCOUNT_SCREEN)
    if not near.any():
        return
    at = np.flatnonzero(near)
    with np.errstate(over="ignore", invalid="ignore"):
        overflow = ~np.isfinite(strike[at] * _exp(-rate[at] * t[at])) | ~np.isfinite(
            spot[at] * _exp(-q[at] * t[at])
        )
    bad = np.zeros(near.shape, dtype=bool)
    bad[at[overflow]] = True
    if bad.any():

        def check(i: int) -> None:
            _scalar._require_finite_price(float("inf"), "the discounted strike or spot")

        _refuse_first(bad, shape, fn, check)


# ── implied volatility ──────────────────────────────────────────────────


def implied_volatility_batch(
    option_price: Any,
    spot: Any,
    strike: Any,
    time_to_expiry: Any,
    risk_free_rate: Any,
    dividend_yield: Any = 0.0,
    is_call: Any = True,
    *,
    initial_guess: float = 0.2,
    tol: float = 1e-6,
    max_iterations: int = 100,
    tol_sigma: float = 1e-8,
) -> Dict[str, Any]:
    """
    The Black-Scholes-Merton implied volatility of every contract in a
    chain, in one call.

    THE SCALAR'S ALGORITHM, CONTRACT FOR CONTRACT. For each contract this is
    `analysis.options.implied_volatility`: the no-arbitrage bound check
    (equality within `BOUND_TOLERANCE` admitted), Newton on vega converged
    on the VOLATILITY step (`tol_sigma`), the bisection over [1e-6, 5.0]
    when vega falls below `VEGA_FLOOR` or a step leaves the bracket, and the
    at-intrinsic bisection that returns the largest volatility still
    reproducing the price, flagged `at_bound`. Same bounds, same
    tolerances, same keyword defaults -- and the same doubles: the
    volatility, the iteration count, the method and the price error of every
    contract are the ones the scalar returns for it.

    Every argument broadcasts: pass arrays for what varies across the chain
    and scalars for what does not. `is_call` is True for a call and False
    for a put (0/1 are accepted).

    A QUOTE THAT CANNOT BE SOLVED IS REPORTED, NOT RAISED. Where the scalar
    raises for the quote itself, the batch reports that contract's `reason`
    and a NaN volatility and solves the rest:

    - `price_not_positive` -- a price <= 0, which is what an option so far
      from the money that its value underflows is quoted at;
    - `price_not_finite` -- NaN or +inf, a missing quote;
    - `below_lower_bound` / `above_upper_bound` -- outside what any
      volatility can produce;
    - `no_root_in_bracket` -- inside the bounds, but only a volatility past
      500% (or below 1e-6) reproduces it.

    An input outside the pricing domain (spot, strike or time not positive
    or past its magnitude limit, a rate or yield past `MAX_RATE`, a rate x
    time past `MAX_EXPONENT`, a discounted strike past a double) refuses the
    whole batch with the scalar's own `ValidationError`, naming the contract:
    that is a unit error in the arrays, not a fact about one quote.

    Returns:
        Dict of arrays in the broadcast shape -- `implied_volatility`,
        `converged`, `iterations`, `method` ("newton", "bisection", or
        "none" where refused), `price_error`, `at_bound`, `reason` -- plus
        `n_contracts`, `n_solved`, `refusals` (reason -> count, refused
        reasons only) and `path` ("C++" or "python").

    Raises:
        ValidationError: an input outside the pricing domain, arrays that do
            not broadcast, an `is_call` that is not boolean, or a solver
            setting outside its range.
    """
    fn = "implied_volatility_batch"
    guess = _setting(initial_guess, "initial_guess", fn)
    if not 0.0 < guess <= _MAX_VOLATILITY:
        raise ValidationError(
            f"{fn}: initial_guess must be a volatility in (0, "
            f"{_MAX_VOLATILITY:g}], got {initial_guess!r}. It is the volatility "
            "the first Newton step is priced at."
        )
    tol = _setting(tol, "tol", fn, minimum=0.0)
    tol_sigma = _setting(tol_sigma, "tol_sigma", fn, minimum=0.0)
    max_iterations = _whole(max_iterations, "max_iterations", fn, minimum=0)

    shape, (price, s, k, t, r, q, call) = _broadcast(
        fn,
        option_price=_floats(option_price, "option_price", fn),
        spot=_floats(spot, "spot", fn),
        strike=_floats(strike, "strike", fn),
        time_to_expiry=_floats(time_to_expiry, "time_to_expiry", fn),
        risk_free_rate=_floats(risk_free_rate, "risk_free_rate", fn),
        dividend_yield=_floats(dividend_yield, "dividend_yield", fn),
        is_call=_booleans(is_call, fn),
    )
    _validate_domain(fn, shape, s, k, t, r, q)
    _validate_discounting(fn, shape, s, k, t, r, q)

    kernel = _native("implied_volatility_batch")
    if kernel is not None:
        raw = kernel(
            price,
            s,
            k,
            t,
            r,
            q,
            call.astype(np.uint8),
            guess,
            tol,
            # The kernel counts in int32. Newton leaves by converging or by
            # stepping out of the bracket long before either bound, so the
            # cap changes no answer.
            min(max_iterations, 2**31 - 1),
            tol_sigma,
        )
        path = "C++"
    else:
        raw = _implied_volatility_python(
            price, s, k, t, r, q, call, guess, tol, max_iterations, tol_sigma
        )
        path = "python"

    reason = np.asarray(raw["reason"], dtype=np.int8)
    # The validators above refuse both of these first. Kept so a solver that
    # ever disagreed with them refuses, rather than reporting an input
    # problem as though it were a fact about one quote.
    broken = (reason == _INVALID_INPUT) | (reason == _NOT_PRICEABLE)
    if broken.any():  # pragma: no cover
        i = int(np.flatnonzero(broken)[0])
        raise ValidationError(
            f"{fn}: {_where(shape, i)} could not be priced "
            f"({REASONS[int(reason[i])]}). Check the rate units."
        )

    names = np.asarray(REASONS)[reason]
    counts = {
        REASONS[c]: int(n) for c, n in zip(*np.unique(reason, return_counts=True)) if c
    }
    logger.debug(
        "[options_batch] implied_volatility_batch  n=%d  refused=%s  path=%s",
        reason.size,
        counts,
        path,
    )
    return {
        "implied_volatility": np.asarray(
            raw["implied_volatility"], dtype=float
        ).reshape(shape),
        "converged": np.asarray(raw["converged"]).astype(bool).reshape(shape),
        "iterations": np.asarray(raw["iterations"]).astype(np.int64).reshape(shape),
        "method": np.asarray(METHODS)[np.asarray(raw["method"], dtype=np.int8)].reshape(
            shape
        ),
        "price_error": np.asarray(raw["price_error"], dtype=float).reshape(shape),
        "at_bound": np.asarray(raw["at_bound"]).astype(bool).reshape(shape),
        "reason": names.reshape(shape),
        "n_contracts": int(reason.size),
        "n_solved": int((reason == 0).sum()),
        "refusals": counts,
        "path": path,
    }


class _Chain:
    """The volatility-independent terms of the contracts still being solved.

    Computed once; the scalar recomputes them per iteration, which gives the
    same doubles since each is a deterministic function of the inputs."""

    def __init__(self, s, k, t, r, q, call, price):
        self.s, self.k, self.t, self.r, self.q = s, k, t, r, q
        self.call, self.price = call, price
        self.sqrt_t = np.sqrt(t)
        self.log_moneyness = _log(s / k)
        self.disc_q = _exp(-q * t)
        self.disc_r = _exp(-r * t)

    def take(self, idx: np.ndarray) -> "_Chain":
        sub = object.__new__(_Chain)
        for name, value in vars(self).items():
            setattr(sub, name, value[idx])
        return sub

    def d1(self, sigma: np.ndarray) -> np.ndarray:
        # analysis.options._d1_d2, term for term.
        return (
            self.log_moneyness + (self.r - self.q + 0.5 * sigma * sigma) * self.t
        ) / (sigma * self.sqrt_t)

    def model(self, sigma: np.ndarray, d1: np.ndarray) -> np.ndarray:
        """analysis.options.black_scholes_price at `sigma`."""
        d2 = d1 - sigma * self.sqrt_t
        n1 = norm_cdf_array(np.where(self.call, d1, -d1))
        n2 = norm_cdf_array(np.where(self.call, d2, -d2))
        call = self.s * self.disc_q * n1 - self.k * self.disc_r * n2
        put = self.k * self.disc_r * n2 - self.s * self.disc_q * n1
        return np.where(self.call, call, put)

    def diff(self, sigma: np.ndarray) -> np.ndarray:
        return self.model(sigma, self.d1(sigma)) - self.price


def _implied_volatility_python(
    price: np.ndarray,
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    call: np.ndarray,
    initial_guess: float,
    tol: float,
    max_iterations: int,
    tol_sigma: float,
) -> Dict[str, np.ndarray]:
    """
    The numpy fallback: the scalar solver's control flow, run on every
    still-unsolved contract at once.

    Each phase keeps a mask of the contracts still in it and steps them
    together, so a contract takes exactly the steps it would alone -- the
    same Newton iterates, the same bisection midpoints -- and leaves the
    phase at the iteration it would have. What changes is only that 476
    contracts share each step's array arithmetic.
    """
    n = price.size
    vol = np.full(n, np.nan)
    err = np.full(n, np.nan)
    iters = np.zeros(n, dtype=np.int32)
    method = np.zeros(n, dtype=np.int8)
    reason = np.zeros(n, dtype=np.int8)
    conv = np.zeros(n, dtype=bool)
    at_bound = np.zeros(n, dtype=bool)

    # The scalar's order: a non-positive price first, then the domain.
    with np.errstate(invalid="ignore"):
        reason[price <= 0] = 1
    reason[(reason == 0) & _outside_domain(s, k, t, r, q)] = _INVALID_INPUT
    live = np.flatnonzero(reason == 0)
    chain = _Chain(s[live], k[live], t[live], r[live], q[live], call[live], price[live])
    with np.errstate(over="ignore", invalid="ignore"):
        s_dq = chain.s * chain.disc_q
        k_dr = chain.k * chain.disc_r
    not_priceable = ~np.isfinite(s_dq) | ~np.isfinite(k_dr)
    reason[live[not_priceable]] = _NOT_PRICEABLE
    missing = ~not_priceable & ~np.isfinite(chain.price)
    reason[live[missing]] = 2

    with np.errstate(invalid="ignore", over="ignore"):
        lower = np.where(
            chain.call, np.maximum(s_dq - k_dr, 0.0), np.maximum(k_dr - s_dq, 0.0)
        )
        upper = np.where(chain.call, s_dq, k_dr)
        slack = _scalar.BOUND_TOLERANCE * np.maximum(np.abs(upper), 1.0)
        below = chain.price < lower - slack
        above = chain.price > upper + slack
    pending = ~not_priceable & ~missing
    reason[live[pending & below]] = 3
    reason[live[pending & ~below & above]] = 4
    solvable = pending & ~below & ~above

    keep = np.flatnonzero(solvable)
    idx = live[keep]  # positions in the full batch
    chain = chain.take(keep)
    slack = slack[keep]
    bound = (chain.price - lower[keep]) <= slack
    at_bound[idx] = bound

    final = np.full(idx.size, np.nan)  # the volatility each contract returns
    done = np.zeros(idx.size, dtype=bool)

    def finish(where: np.ndarray, sigma: Any, used: Any, how: int, ok: bool) -> None:
        final[where] = sigma
        iters[idx[where]] = used
        method[idx[where]] = how
        conv[idx[where]] = ok
        done[where] = True

    # ── Newton, converged on the step it took ───────────────────────────
    sigma = np.full(idx.size, initial_guess)
    newton = ~bound  # at intrinsic, Newton has nothing to divide by
    for i in range(max_iterations):
        active = np.flatnonzero(newton)
        if active.size == 0:
            break
        sub = chain.take(active)
        d1 = sub.d1(sigma[active])
        diff = sub.model(sigma[active], d1) - sub.price
        vega = sub.s * sub.disc_q * norm_pdf_array(d1) * sub.sqrt_t
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            step = diff / vega
            candidate = sigma[active] - step
        leave = (
            (vega < _scalar.VEGA_FLOOR)
            | (candidate <= 0)
            | (candidate > _scalar.SIGMA_HIGH)
        )
        moved = active[~leave]
        sigma[moved] = candidate[~leave]
        converged = moved[np.abs(step[~leave]) < tol_sigma]
        if converged.size:
            finish(converged, sigma[converged], i + 1, 1, True)
        newton[active[leave]] = False
        newton[converged] = False

    # ── Bisection at intrinsic: the largest volatility that still prices ─
    ab = np.flatnonzero(bound & ~done)
    if ab.size:
        sub = chain.take(ab)
        lo = np.full(ab.size, _scalar.SIGMA_LOW)
        hi = np.full(ab.size, _scalar.SIGMA_HIGH)
        used = np.full(ab.size, 200, dtype=np.int32)
        going = np.ones(ab.size, dtype=bool)
        for i in range(200):
            a = np.flatnonzero(going)
            if a.size == 0:
                break
            mid = 0.5 * (lo[a] + hi[a])
            inside = np.abs(sub.take(a).diff(mid)) <= slack[ab][a]
            lo[a[inside]] = mid[inside]
            hi[a[~inside]] = mid[~inside]
            narrow = a[(hi[a] - lo[a]) < tol_sigma]
            used[narrow] = i + 1
            going[narrow] = False
        finish(ab, lo, used, 2, True)

    # ── Bisection fallback ──────────────────────────────────────────────
    rest = np.flatnonzero(~done)
    if rest.size:
        sub = chain.take(rest)
        lo = np.full(rest.size, _scalar.SIGMA_LOW)
        hi = np.full(rest.size, _scalar.SIGMA_HIGH)
        diff_lo = sub.diff(lo)
        diff_hi = sub.diff(hi)
        at_lo = diff_lo == 0.0
        at_hi = ~at_lo & (diff_hi == 0.0)
        if at_lo.any():
            finish(rest[at_lo], _scalar.SIGMA_LOW, 0, 2, True)
        if at_hi.any():
            finish(rest[at_hi], _scalar.SIGMA_HIGH, 0, 2, True)
        no_root = ~at_lo & ~at_hi & (diff_lo * diff_hi > 0)
        reason[idx[rest[no_root]]] = 5
        done[rest[no_root]] = True
        going = ~at_lo & ~at_hi & ~no_root
        mid = lo.copy()
        for i in range(200):
            a = np.flatnonzero(going)
            if a.size == 0:
                break
            mid[a] = 0.5 * (lo[a] + hi[a])
            diff_mid = sub.take(a).diff(mid[a])
            width = hi[a] - lo[a]
            stop = (width < tol_sigma) | (
                (np.abs(diff_mid) < tol) & (width < _PRICE_EXIT_WIDTH)
            )
            if stop.any():
                finish(rest[a[stop]], mid[a[stop]], i + 1, 2, True)
                going[a[stop]] = False
            a, diff_mid = a[~stop], diff_mid[~stop]
            down = diff_lo[a] * diff_mid < 0
            hi[a[down]] = mid[a[down]]
            lo[a[~down]] = mid[a[~down]]
            diff_lo[a[~down]] = diff_mid[~down]
        left = np.flatnonzero(going)
        if left.size:
            finish(rest[left], mid[left], 200, 2, False)

    solved = np.flatnonzero(done & (reason[idx] == 0))
    if solved.size:
        vol[idx[solved]] = final[solved]
        err[idx[solved]] = np.abs(chain.take(solved).diff(final[solved]))
    return {
        "implied_volatility": vol,
        "price_error": err,
        "iterations": iters,
        "method": method,
        "reason": reason,
        "converged": conv,
        "at_bound": at_bound,
    }


# ── greeks ──────────────────────────────────────────────────────────────


def black_scholes_greeks_batch(
    spot: Any,
    strike: Any,
    time_to_expiry: Any,
    volatility: Any,
    risk_free_rate: Any,
    dividend_yield: Any = 0.0,
    is_call: Any = True,
    *,
    grid: bool = False,
    greeks: Optional[Union[str, Iterable[str]]] = None,
) -> Dict[str, Any]:
    """
    The Black-Scholes-Merton price and full greek set of every contract in
    a batch, in one call.

    THE NUMBERS `analysis.derivatives.option_greeks` RETURNS, in its units:
    delta, gamma and speed per $1 of spot; vega and vanna per volatility
    POINT (0.01); volga per point squared; theta and charm per calendar
    DAY; rho per rate point -- and `price`, which is `analysis.pricing.
    price_option`'s for model='black_scholes'. Contract for contract they
    are the same doubles. (`analysis.options.black_scholes_greeks` returns
    vega and theta raw -- per 1.0 of vol and per year; multiply by 100 and
    365 to compare.)

    Every argument broadcasts. `is_call` is True for a call and False for a
    put.

    `grid=True` values every contract at every spot. `spot` is then a 1-D
    array of spot levels, and each output has shape (*contracts, n_spots):
    row i is contract i across the grid, which is the shape a gamma profile
    or a scenario revaluation reads.

    `greeks` names the outputs to compute -- one name or several, from
    `GREEKS` -- and None (the default) is all of them. Only those are
    computed and returned: a gamma profile, `greeks=("gamma",)`, takes no
    erf at all and a twelfth of the memory, and a hedge, `greeks=("delta",)`,
    one erf per cell where the full set takes two. Each selected array is
    the full call's array for that name, bit for bit, and a selection
    refuses exactly the inputs the full call refuses, with the same words
    -- including a price that comes out non-finite, which is detected
    without forming the price.

    Refuses, with the scalar's `ValidationError` naming the contract, any
    input outside the pricing domain (the rules of `black_scholes_price`,
    which are also `price_option`'s) and any contract whose price comes out
    non-finite.

    Returns:
        Dict with one array per selected entry of `GREEKS` (every entry by
        default), `units` for those of them that have one, and `path`.

    Raises:
        ValidationError: an input outside the pricing domain, a price that
            is not finite, or a `greeks` selection that names nothing or
            names something not in `GREEKS`.
    """
    fn = "black_scholes_greeks_batch"
    names = _selection(greeks, fn)
    contract = dict(
        strike=_floats(strike, "strike", fn),
        time_to_expiry=_floats(time_to_expiry, "time_to_expiry", fn),
        volatility=_floats(volatility, "volatility", fn),
        risk_free_rate=_floats(risk_free_rate, "risk_free_rate", fn),
        dividend_yield=_floats(dividend_yield, "dividend_yield", fn),
        is_call=_booleans(is_call, fn),
    )
    s_in = _floats(spot, "spot", fn)
    if grid:
        if s_in.ndim != 1:
            raise ValidationError(
                f"{fn}: with grid=True, spot is the 1-D grid of spot levels "
                f"every contract is valued at; got shape {s_in.shape}."
            )
        spots = np.ascontiguousarray(s_in)
        with np.errstate(invalid="ignore"):
            bad = ~_positive_up_to(spots, _MAX_PRICE)
        if bad.any():
            _refuse_first(
                bad,
                spots.shape,
                fn,
                lambda j: _scalar._validate_option_inputs(
                    float(spots[j]), 1.0, 1.0, 1.0, "call"
                ),
                what="spot",
            )
        shape, (k, t, v, r, q, call) = _broadcast(fn, **contract)
        _validate_domain(fn, shape, np.ones_like(k), k, t, r, q, v)
        out_shape = shape + (spots.size,)
    else:
        shape, (spots, k, t, v, r, q, call) = _broadcast(fn, spot=s_in, **contract)
        _validate_domain(fn, shape, spots, k, t, r, q, v)
        out_shape = shape

    # A selection without the price gets a flag of where it would be
    # non-finite instead, so it refuses what the full call refuses.
    values = _greeks_arrays(
        spots, k, t, v, r, q, call, grid, names, price_finite="price" not in names
    )
    if "price" in values:
        unpriceable = ~np.isfinite(values["price"].reshape(-1))
    else:
        unpriceable = ~values["price_finite"].reshape(-1)
    if unpriceable.any():
        first = int(np.flatnonzero(unpriceable)[0])
        where = _where(out_shape, first)
        if "price" in values:
            price = float(values["price"].reshape(-1)[first])
        else:
            # The one cell, priced: the refusal quotes the value the full
            # call would have, and a cell is the same double alone.
            i, j = divmod(first, spots.size) if grid else (first, first)
            one = slice(i, i + 1)
            price = float(
                _greeks_arrays(
                    spots[j : j + 1],
                    k[one],
                    t[one],
                    v[one],
                    r[one],
                    q[one],
                    call[one],
                    False,
                    ("price",),
                )["price"][0]
            )
        _scalar._require_finite_price(price, f"{fn}: {where}")
    result: Dict[str, Any] = {name: values[name].reshape(out_shape) for name in names}
    result["units"] = {
        name: words for name, words in GREEK_UNITS.items() if name in names
    }
    result["path"] = values["path"]
    return result


def _selection(greeks: Any, fn: str) -> Tuple[str, ...]:
    """The `greeks` argument as names, in `GREEKS` order; None is all."""
    if greeks is None:
        return GREEKS
    try:
        asked = [greeks] if isinstance(greeks, str) else list(greeks)
    except TypeError:
        asked = [greeks]
    unknown = [
        name for name in asked if not isinstance(name, str) or name not in GREEKS
    ]
    if unknown:
        raise ValidationError(
            f"{fn}: greeks must name entries of GREEKS ({', '.join(GREEKS)}), "
            f"got {unknown[0]!r}. Pass greeks=None for all of them."
        )
    if not asked:
        raise ValidationError(
            f"{fn}: greeks selects nothing. Name the greeks to compute, e.g. "
            "greeks=('gamma',), or pass greeks=None for all of them."
        )
    return tuple(name for name in GREEKS if name in asked)


#: The kernel's output bits: one per entry of GREEKS in order, then the
#: price_finite flag.
_GREEK_BITS = {name: 1 << i for i, name in enumerate(GREEKS)}
_PRICE_FINITE_BIT = 1 << len(GREEKS)


def _greeks_arrays(
    spots: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    call: np.ndarray,
    grid: bool,
    names: Tuple[str, ...] = GREEKS,
    price_finite: bool = False,
) -> Dict[str, Any]:
    """
    Validated, flat inputs in; the arrays named in `names` out, flat
    (contracts x spots, row-major, under `grid`). With `price_finite`, also
    a boolean array that is True where the price is a finite number.
    """
    kernel = _native("black_scholes_greeks_batch")
    if kernel is not None:
        bits = sum(_GREEK_BITS[name] for name in names)
        if price_finite:
            bits |= _PRICE_FINITE_BIT
        raw = kernel(spots, k, t, v, r, q, call.astype(np.uint8), bool(grid), bits)
        out: Dict[str, Any] = {
            name: np.asarray(raw[name]).reshape(-1) for name in names
        }
        if price_finite:
            out["price_finite"] = np.asarray(raw["price_finite"]).reshape(-1) != 0
        out["path"] = "C++"
        return out
    if grid:
        column = (slice(None), None)
        spots_b = spots[None, :]
        k, t, v, r, q, call = (x[column] for x in (k, t, v, r, q, call))
    else:
        spots_b = spots
    values = _greeks_python(spots_b, k, t, v, r, q, call, names, price_finite)
    shape = np.broadcast(spots_b, k).shape
    wanted = names + (("price_finite",) if price_finite else ())
    out = {name: np.broadcast_to(values[name], shape).reshape(-1) for name in wanted}
    out["path"] = "python"
    return out


def _greeks_python(
    spot,
    strike,
    t,
    vol,
    rate,
    q,
    call,
    names: Tuple[str, ...] = GREEKS,
    price_finite: bool = False,
) -> Dict[str, np.ndarray]:
    """
    The numpy fallback: `option_greeks` term for term, elementwise, for the
    outputs in `names`.

    Both the call and the put expressions are formed and `call` picks one;
    each is the scalar's own association, so the selected one is the
    scalar's double. An intermediate no selected output needs is not
    formed, and a formed one is the same expression on the same inputs, so
    a selection is the full call's arrays for those names.
    """
    want = set(names)
    out: Dict[str, np.ndarray] = {}
    sqrt_t = np.sqrt(t)
    growth = discount = pdf_d1 = n1 = n2 = None
    if price_finite or want & _NEEDS_GROWTH:
        growth = _exp(-q * t)
    if price_finite or want & _NEEDS_DISCOUNT:
        discount = _exp(-rate * t)
    # `vol * vol`: option_greeks and pricing._black_scholes multiply.
    d1 = (_log(spot / strike) + (rate - q + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    if want & _NEEDS_PDF:
        pdf_d1 = norm_pdf_array(d1)
    if want & _NEEDS_N1:  # N(d1) for a call, N(-d1) for a put
        n1 = norm_cdf_array(np.where(call, d1, -d1))
    if want & _NEEDS_N2:
        n2 = norm_cdf_array(np.where(call, d2, -d2))
    if "price" in want:
        out["price"] = np.where(
            call,
            spot * growth * n1 - strike * discount * n2,
            strike * discount * n2 - spot * growth * n1,
        )
    if "delta" in want:
        out["delta"] = np.where(call, growth * n1, -growth * n1)
    if "gamma" in want or "speed" in want:
        gamma = growth * pdf_d1 / (spot * vol * sqrt_t)
        out["gamma"] = gamma
        out["speed"] = -gamma / spot * (d1 / (vol * sqrt_t) + 1.0)
    if "vega" in want or "volga" in want:
        vega_raw = spot * growth * pdf_d1 * sqrt_t
        out["vega"] = vega_raw / 100.0
        out["volga"] = vega_raw * d1 * d2 / vol / 10000.0
    if "theta" in want:
        decay = -spot * pdf_d1 * vol * growth / (2.0 * sqrt_t)
        theta_raw = np.where(
            call,
            decay + q * spot * growth * n1 - rate * strike * discount * n2,
            decay - q * spot * growth * n1 + rate * strike * discount * n2,
        )
        out["theta"] = theta_raw / 365.0
    if "rho" in want:
        rho = np.where(call, strike * t * discount * n2, -strike * t * discount * n2)
        out["rho"] = rho / 100.0
    if "vanna" in want:
        out["vanna"] = -growth * pdf_d1 * d2 / vol / 100.0
    if "charm" in want:
        shape_term = (
            pdf_d1
            * (2.0 * (rate - q) * t - d2 * vol * sqrt_t)
            / (2.0 * t * vol * sqrt_t)
        )
        charm_raw = np.where(
            call, -growth * (shape_term - q * n1), -growth * (shape_term + q * n1)
        )
        out["charm"] = charm_raw / 365.0
    out["d1"] = d1
    out["d2"] = d2
    if price_finite:
        if "price" in out:
            out["price_finite"] = np.isfinite(out["price"])
        else:
            # The kernel's test, and the same fact: the price is
            # (spot * growth) * N - (strike * discount) * N' with each N in
            # [0, 1], or NaN exactly when d1 is -- finite exactly when both
            # heads are and d1 is not NaN.
            with np.errstate(over="ignore", invalid="ignore"):
                out["price_finite"] = (
                    np.isfinite(spot * growth)
                    & np.isfinite(strike * discount)
                    & ~np.isnan(d1)
                )
    return out


# What each intermediate of `_greeks_python` is needed for.
_NEEDS_N1 = {"price", "delta", "theta", "charm"}
_NEEDS_N2 = {"price", "rho", "theta"}
_NEEDS_PDF = {"gamma", "vega", "theta", "vanna", "volga", "charm", "speed"}
_NEEDS_GROWTH = _NEEDS_PDF | {"price", "delta"}
_NEEDS_DISCOUNT = {"price", "rho", "theta"}


# ── where a book's gamma changes sign ───────────────────────────────────


def zero_gamma_spot(
    strike: Any,
    time_to_expiry: Any,
    volatility: Any,
    quantity: Any,
    *,
    spot_low: float,
    spot_high: float,
    risk_free_rate: Any = 0.0,
    dividend_yield: Any = 0.0,
    reference_spot: Optional[float] = None,
    n_grid: int = 201,
    xtol: float = 1e-10,
) -> Dict[str, Any]:
    """
    The spot at which a book's aggregate signed gamma crosses zero inside
    [spot_low, spot_high] -- or the statement that it does not.

    Net gamma is sum_i quantity_i x gamma_i(S), with each contract's gamma
    from `black_scholes_greeks_batch` at its own strike, expiry and
    volatility. `quantity` carries the sign and the size: positive long,
    negative short, contract multipliers folded in. A dealer-positioning
    convention (calls long, puts short) is a choice of signs the caller
    makes here; nothing in this function assumes one. Gamma is the same for
    a call and a put, so the option type is not an input. Dollar gamma
    (gamma x S^2) crosses zero at the same spot, since S^2 > 0.

    HOW. Net gamma is evaluated on `n_grid` spots spread evenly across the
    bracket in one grid call, and every sign change between neighbouring
    spots is a bracket. Each bracket is then narrowed by Brent's method --
    inverse quadratic interpolation that falls back to bisection whenever a
    step would not shrink the bracket fast enough, scipy's `brentq` -- with
    one batched evaluation of the whole book per step, until it is narrower
    than `xtol` relative to the spot. Net gamma is smooth between strikes,
    which is where Brent converges superlinearly: a handful of evaluations
    where bisection needs thirty.

    NO CROSSING IS INVENTED. A book whose net gamma keeps one sign across
    the bracket returns `zero_gamma_spot=None`, `crossing_found=False` and
    the reason, with the net gamma at both ends so the caller can see which
    side of zero the book sits. Spots where every contract's gamma underflows
    to exactly zero -- far from every strike, close to expiry -- are not a
    crossing either: gamma is absent there, not changing sign.

    THE LIMIT OF A SCAN. Two crossings closer together than one grid step
    cancel between neighbouring spots and are not seen; `grid_step` is
    returned so that is checkable, and a finer `n_grid` resolves them. When
    several crossings are found, `zero_gamma_spot` is the one nearest
    `reference_spot` (the bracket's midpoint if none is given) and every
    one is listed in `crossings`.

    Raises:
        ValidationError: a contract outside the pricing domain, a quantity
            that is not finite, a book with no non-zero position, a bracket
            that is not 0 < spot_low < spot_high <= 1e12, or n_grid < 3.
    """
    fn = "zero_gamma_spot"
    low = _setting(spot_low, "spot_low", fn)
    high = _setting(spot_high, "spot_high", fn)
    if not 0.0 < low < high <= _MAX_PRICE:
        raise ValidationError(
            f"{fn}: the bracket must satisfy 0 < spot_low < spot_high <= "
            f"{_MAX_PRICE:g}, got [{spot_low!r}, {spot_high!r}]."
        )
    # Three spots is two intervals: the least a sign change can be placed in.
    n_grid = _whole(n_grid, "n_grid", fn, minimum=3)
    xtol = _setting(xtol, "xtol", fn, minimum=0.0)
    if xtol == 0.0:
        raise ValidationError(
            f"{fn}: xtol must be > 0; a root search needs a width to stop at"
        )
    anchor_given = (
        None
        if reference_spot is None
        else _setting(reference_spot, "reference_spot", fn)
    )

    shape, (k, t, v, r, q, qty) = _broadcast(
        fn,
        strike=_floats(strike, "strike", fn),
        time_to_expiry=_floats(time_to_expiry, "time_to_expiry", fn),
        volatility=_floats(volatility, "volatility", fn),
        risk_free_rate=_floats(risk_free_rate, "risk_free_rate", fn),
        dividend_yield=_floats(dividend_yield, "dividend_yield", fn),
        quantity=_floats(quantity, "quantity", fn),
    )
    if k.size == 0:
        raise ValidationError(f"{fn}: the book is empty; there is no gamma to cross.")
    _validate_domain(fn, shape, np.ones_like(k), k, t, r, q, v)
    if not np.isfinite(qty).all():
        i = int(np.flatnonzero(~np.isfinite(qty))[0])
        raise ValidationError(
            f"{fn}: {_where(shape, i)}: quantity must be finite, got {qty[i]!r}"
        )
    if not (qty != 0).any():
        raise ValidationError(
            f"{fn}: every quantity is zero, so the book has no gamma at any "
            "spot. Pass the signed position sizes."
        )
    call = np.ones(k.size, dtype=bool)  # gamma does not depend on it

    def net_gamma(spots: np.ndarray) -> np.ndarray:
        # Gamma alone: the full set's gamma, without the two erfs per cell
        # and eleven other arrays it has no use for.
        gamma = _greeks_arrays(spots, k, t, v, r, q, call, True, ("gamma",))["gamma"]
        # Summed row by row down each column, a fixed order: a BLAS dot
        # would be free to split it by thread, and the two paths' identical
        # gammas would no longer give identical sums.
        return (qty[:, None] * gamma.reshape(k.size, spots.size)).sum(axis=0)

    grid = np.linspace(low, high, n_grid)
    path = "C++" if _native("black_scholes_greeks_batch") is not None else "python"
    values = net_gamma(grid)
    signs = np.sign(values)

    brackets: List[Tuple[float, float, float, float, bool]] = []
    last = None  # index of the last spot with non-zero net gamma
    for j in range(n_grid):
        if signs[j] == 0:
            continue
        if last is not None and signs[j] != signs[last]:
            # Two or more exact zeros between them is gamma underflowing,
            # not gamma at zero; the crossing is somewhere inside that
            # stretch. One exact zero is a crossing that fell on the grid.
            brackets.append(
                (grid[last], grid[j], values[last], values[j], j - last > 2)
            )
        last = j

    def at(spot: float) -> float:
        return float(net_gamma(np.array([spot]))[0])

    crossings: List[float] = []
    evaluations = 0
    for lo, hi, value_lo, value_hi, _ in brackets:
        root, used = _brent(at, lo, hi, value_lo, value_hi, xtol)
        crossings.append(root)
        evaluations += used

    warnings: List[str] = [
        f"Crossings are located by scanning {n_grid} spots {grid[1] - grid[0]:.6g} "
        "apart and refining each sign change. Two crossings closer than one "
        "step cancel and are not seen; raise n_grid to resolve them."
    ]
    if any(b[4] for b in brackets):
        warnings.append(
            "Net gamma underflows to exactly zero on a stretch between opposite "
            "signs -- every contract is too far from its strike, too close to "
            "expiry, to carry any. The crossing reported sits inside that "
            "stretch; any spot in it is equally the answer."
        )

    reason: Optional[str] = None
    chosen: Optional[float] = None
    if crossings:
        anchor = 0.5 * (low + high) if anchor_given is None else anchor_given
        chosen = min(crossings, key=lambda x: (abs(x - anchor), x))
        if len(crossings) > 1:
            warnings.append(
                f"Net gamma changes sign {len(crossings)} times in the bracket. "
                f"zero_gamma_spot is the crossing nearest {anchor:.6g}; every "
                "crossing is in `crossings`, and which one matters depends on "
                "where spot is going."
            )
    else:
        nonzero = signs[signs != 0]
        if nonzero.size == 0:
            reason = (
                "Net gamma is exactly zero at every scanned spot: every "
                "contract's gamma underflows across the bracket, so there is "
                "no gamma to change sign. Move the bracket toward the strikes."
            )
        else:
            side = (
                "positive (long gamma)" if nonzero[0] > 0 else "negative (short gamma)"
            )
            reason = (
                f"Net gamma is {side} at every scanned spot in [{low:.6g}, "
                f"{high:.6g}], so there is no crossing in this bracket and none "
                "is reported. Widen the bracket, or check the signs of the "
                "quantities -- a book of only long options is long gamma "
                "everywhere."
            )
        warnings.append(reason)

    return {
        "zero_gamma_spot": None if chosen is None else float(chosen),
        "crossing_found": bool(crossings),
        "crossings": [float(x) for x in crossings],
        "n_crossings": len(crossings),
        "net_gamma_at_low": float(values[0]),
        "net_gamma_at_high": float(values[-1]),
        "bracket": [low, high],
        "n_grid": n_grid,
        "grid_step": float(grid[1] - grid[0]),
        "refinement_evaluations": evaluations,
        "reason": reason,
        "path": path,
        "warnings": warnings,
    }


def _brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    fa: float,
    fb: float,
    xtol: float,
    max_evaluations: int = 100,
) -> Tuple[float, int]:
    """
    A root of `f` in [a, b], where f(a) and f(b) have opposite signs.

    Brent's method as scipy's `brentq` writes it: inverse quadratic
    interpolation, or a secant step, accepted only while it shrinks the
    bracket faster than bisection would, and bisection otherwise -- so it
    converges wherever bisection does and much faster where `f` is smooth.
    Stops when the bracket is narrower than `xtol` relative to the root, or
    at an exact zero. Returns the root and the evaluations of `f` it made.
    """
    x_pre, x_cur, f_pre, f_cur = a, b, fa, fb
    x_blk = f_blk = s_pre = s_cur = 0.0
    used = 0
    while True:
        if f_pre != 0.0 and f_cur != 0.0 and (f_pre < 0.0) != (f_cur < 0.0):
            x_blk, f_blk = x_pre, f_pre
            s_pre = s_cur = x_cur - x_pre
        if abs(f_blk) < abs(f_cur):
            x_pre, x_cur, x_blk = x_cur, x_blk, x_cur
            f_pre, f_cur, f_blk = f_cur, f_blk, f_cur
        delta = 0.5 * xtol * max(1.0, abs(x_cur))
        s_bis = 0.5 * (x_blk - x_cur)
        if f_cur == 0.0 or abs(s_bis) < delta or used >= max_evaluations:
            return x_cur, used
        if abs(s_pre) > delta and abs(f_cur) < abs(f_pre):
            if x_pre == x_blk:  # secant
                s_try = -f_cur * (x_cur - x_pre) / (f_cur - f_pre)
            else:  # inverse quadratic interpolation
                d_pre = (f_pre - f_cur) / (x_pre - x_cur)
                d_blk = (f_blk - f_cur) / (x_blk - x_cur)
                s_try = (
                    -f_cur
                    * (f_blk * d_blk - f_pre * d_pre)
                    / (d_blk * d_pre * (f_blk - f_pre))
                )
            if 2.0 * abs(s_try) < min(abs(s_pre), 3.0 * abs(s_bis) - delta):
                s_pre, s_cur = s_cur, s_try  # a good short step
            else:
                s_pre = s_cur = s_bis
        else:
            s_pre = s_cur = s_bis
        x_pre, f_pre = x_cur, f_cur
        if abs(s_cur) > delta:
            x_cur += s_cur
        else:
            x_cur += delta if s_bis > 0 else -delta
        f_cur = f(x_cur)
        used += 1


__all__ = [
    "GREEKS",
    "GREEK_UNITS",
    "METHODS",
    "REASONS",
    "black_scholes_greeks_batch",
    "implied_volatility_batch",
    "zero_gamma_spot",
]
