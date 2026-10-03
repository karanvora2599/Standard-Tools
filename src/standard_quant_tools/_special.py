"""
The special functions this library writes out because scipy is not a
dependency -- written out once.

WHY THIS FILE EXISTS. Nine modules had a private copy of at least one of
these. Counted before this file: `_norm_cdf` seven times, `_norm_pdf` three,
`_norm_ppf` twice, and `_betainc`/`_betacf`/`_f_sf` twice each. The normal
CDF copies were genuinely identical -- it is one exact line and there is
nothing to get wrong. The others had drifted, and drifted in the direction
that matters: two copies of the same algorithm disagreeing about what to do
at the edge of their domain.

    `_norm_ppf(p)` at p = 1.0
        backtest.robustness      returned +inf
        backtesting.overfitting  raised ValidationError

    `_f_sf(f, d1, d2)` at d2 = 0
        analysis.diagnostics     returned 1.0   (guarded)
        analysis.structure       returned 0.0   (unguarded)

Neither divergence was reachable through a public entry point at the time
of writing -- `structure`'s call site guards `df_den <= 0` first, and
`robustness` only reaches p = 1.0 at around 1e17 trials. That is what makes
them worth collapsing rather than worth arguing about: nothing was broken,
and nothing held the two halves together either, so the next edit to one of
them was free to make it broken.

A p-value of 0.0 from a test with no denominator degrees of freedom is the
bad case. It is not an error and not a NaN -- it is maximum significance,
returned in the ordinary shape, for a test that had nothing to measure.

WHICH VARIANT WON. The stricter one, every time. Refusing an input the
algorithm has no answer for beats returning an infinity that looks like a
number, which is the same call `numeric_contract` makes about series and
`analysis/_series.py` makes about the four `_clean` helpers.

The private per-module names stay as thin aliases, so anything importing
`_norm_cdf` from where it used to live still gets one.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from standard_quant_tools.error import ValidationError

__all__ = [
    "betacf",
    "betainc",
    "f_sf",
    "f_sf_array",
    "norm_cdf",
    "norm_cdf_array",
    "norm_pdf",
    "norm_pdf_array",
    "norm_ppf",
]

_SQRT2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


# ── normal distribution ─────────────────────────────────────────────────


def norm_cdf(x: float) -> float:
    """Standard normal CDF. Exact via `math.erf`, so there is no accuracy
    tradeoff being made here and no reason for a second implementation."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def norm_pdf(x: float) -> float:
    """Standard normal density."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


_erf_each = np.frompyfunc(math.erf, 1, 1)
_exp_each = np.frompyfunc(math.exp, 1, 1)


def norm_cdf_array(x: Any) -> np.ndarray:
    """
    `norm_cdf` over an array, element for element the double `norm_cdf`
    returns.

    numpy has no erf, so `math.erf` is applied per element and the rest of
    the formula -- `0.5 * (1 + erf(x / sqrt(2)))` -- is numpy arithmetic,
    which rounds exactly as the scalar's does. This was
    `np.vectorize(norm_cdf)`: the same numbers, with the whole formula run
    as a Python call per element. The option-chain batch in
    `analysis.options_batch` evaluates two of these per contract.

    The empty case is handled explicitly: the CDF of no observations is no
    observations, not an error. (`np.vectorize` raised on a size-0 input,
    which is why this guard exists.)
    """
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        return np.empty(values.shape, dtype=float)
    erf = np.asarray(_erf_each(values / _SQRT2), dtype=float)
    return np.asarray(0.5 * (1.0 + erf), dtype=float)


def norm_pdf_array(x: Any) -> np.ndarray:
    """`norm_pdf` over an array, element for element the scalar's double:
    the exponential is `math.exp` per element, since numpy's own `exp` may
    round the last bit differently on some CPUs."""
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        return np.empty(values.shape, dtype=float)
    exp = np.asarray(_exp_each(-0.5 * values * values), dtype=float)
    return np.asarray(exp / _SQRT_2PI, dtype=float)


def norm_ppf(p: float) -> float:
    """
    Inverse normal CDF, by Acklam's rational approximation.

    Accurate to about 1.15e-9 across the whole range, which is well past
    what any statistic in this library needs. Written out rather than
    imported because scipy is not a declared dependency.

    Raises on p outside (0, 1) exclusive, rather than returning a signed
    infinity. The quantile genuinely is infinite at the bounds, but an
    infinity returned into a Sharpe ratio or a critical value is a number
    that stops being one several steps later, with nothing marking where.

    Raises:
        ValidationError: p is not strictly between 0 and 1.
    """
    if not 0.0 < p < 1.0:
        raise ValidationError(
            f"norm_ppf: p must be strictly in (0, 1), got {p!r}. The normal "
            "quantile is infinite at the bounds; if p reached 0 or 1 by "
            "rounding (1 - 1/n collapses to 1.0 above about 1e16), the "
            "caller's n is the thing to fix."
        )

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


# ── incomplete beta, and the F tail that uses it ────────────────────────

# The continued fraction's floor, convergence tolerance and iteration cap,
# shared by `betacf` and its array form.
_BETACF_TINY = 1e-300
_BETACF_TOLERANCE = 1e-14
_BETACF_ITERATIONS = 300


def _lgamma_ratio(a: float, b: float) -> float:
    """log(Gamma(a + b) / (Gamma(a) Gamma(b))), the leading terms of the
    incomplete beta's front factor, in the order `betainc` adds them up --
    one definition for the scalar and the array form."""
    return math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)


def betacf(a: float, b: float, x: float, iterations: int = _BETACF_ITERATIONS) -> float:
    """
    Continued fraction for the incomplete beta, by modified Lentz.

    `tiny` is 1e-300 rather than 1e-30: it exists to keep a denominator away
    from exact zero, and the larger floor perturbs the result at a magnitude
    the algorithm can actually see.
    """
    tiny = _BETACF_TINY
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETACF_TOLERANCE:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), via the continued fraction on
    whichever side of the symmetry point converges."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(_lgamma_ratio(a, b) + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return (
        1.0
        - math.exp(_lgamma_ratio(a, b) + b * math.log(1.0 - x) + a * math.log(x))
        * betacf(b, a, 1.0 - x)
        / b
    )


def f_sf(statistic: float, d1: float, d2: float) -> float:
    """
    Upper tail of the F distribution: P(F > statistic).

    By the identity P(F > f) = I_{d2/(d2 + d1 f)}(d2/2, d1/2).

    Degenerate degrees of freedom return 1.0, not 0.0. One of the two copies
    this replaces checked only `statistic <= 0`, so a zero denominator dof
    fell through to `betainc(0, ., 0)` and came back 0.0 -- a p-value of
    zero, maximum significance, from a test with nothing to measure. The
    result is also clamped into [0, 1], because a continued fraction that
    stops early can otherwise leave it a hair outside.
    """
    if statistic <= 0 or d1 <= 0 or d2 <= 0:
        return 1.0
    x = d2 / (d2 + d1 * statistic)
    return float(max(0.0, min(1.0, betainc(d2 / 2.0, d1 / 2.0, x))))


# Degrees of freedom the array form computes itself; anything else -- zero,
# negative, NaN, infinite, or outside these bounds -- is handed to `f_sf`
# element by element. Inside them nothing in the scalar path can raise:
# `lgamma` stays far from its poles and its overflow, the front factor's
# exponent stays far below `math.exp`'s overflow at about 709.8, and every
# divisor is positive. Real tests' degrees of freedom sit well inside.
_F_SF_ARRAY_MIN_DOF = 2.0**-40
_F_SF_ARRAY_MAX_DOF = 2.0**40

# Below these sizes a numpy pass costs more than the Python loop it
# replaces: each continued-fraction iteration is some forty numpy calls,
# about 40 microseconds whatever the length, paid until the slowest element
# converges. A batch smaller than `_F_SF_ARRAY_MIN_BATCH` is `f_sf` per
# element (measured break-even: 100-400 elements), and once the iterating
# set is down to `_BETACF_ARRAY_TAIL` elements they are finished by
# `betacf` from the start. Both hand-offs are the scalar itself, so neither
# can change a bit.
_F_SF_ARRAY_MIN_BATCH = 128
_BETACF_ARRAY_TAIL = 32


def _each(function: Any, values: np.ndarray) -> np.ndarray:
    """A `math` function per element: numpy's own SIMD `exp`/`log` may round
    the last bit differently from the platform libm on some CPUs (AVX-512
    among them), and the array form has to be the scalar's double."""
    return np.fromiter(
        map(function, values.tolist()), dtype=np.float64, count=values.size
    )


def _floor_at_tiny(values: np.ndarray) -> None:
    """`if abs(v) < tiny: v = tiny`, in place, for every element."""
    np.copyto(values, _BETACF_TINY, where=np.abs(values) < _BETACF_TINY)


def _betacf_array(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    `betacf` over 1-D arrays, element for element the scalar's double.

    The same operations in the same order, each element stopping at the
    iteration its scalar loop would break on. An element that converges
    leaves the working set, so the later iterations run only on the
    elements still iterating. numpy's add, multiply and divide round as
    Python's float arithmetic does, and the floor test `abs(d) < tiny` is
    the same comparison (a NaN fails it in both).

    The loop body works in place, into a few buffers, which is 1.5x faster
    than allocating each intermediate. Each step is commented with the
    scalar statement it computes; where an operation's operands appear
    swapped, it is an add or a multiply, which IEEE arithmetic makes exact
    in either order. The last `_BETACF_ARRAY_TAIL` elements still iterating
    are handed to `betacf` itself.
    """
    out = np.empty(x.shape[0], dtype=np.float64)
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones(x.shape[0], dtype=np.float64)
    # d = 1.0 - qab * x / qap; floor; d = 1.0 / d
    d = qab * x
    d /= qap
    np.subtract(1.0, d, out=d)
    _floor_at_tiny(d)
    np.divide(1.0, d, out=d)
    h = d.copy()
    index = np.arange(x.shape[0])
    for m in range(1, _BETACF_ITERATIONS + 1):
        if index.size <= _BETACF_ARRAY_TAIL:
            # a, b and x are still the inputs for these elements; `betacf`
            # runs them from the first iteration, so this is its double.
            out[index] = [
                betacf(p, q, r) for p, q, r in zip(a.tolist(), b.tolist(), x.tolist())
            ]
            return out
        m2 = float(2 * m)
        fm = float(m)
        a_m2 = a + m2
        # aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        aa = b - fm
        aa *= fm
        aa *= x
        scratch = qam + m2
        scratch *= a_m2
        aa /= scratch
        # d = 1.0 + aa * d; floor
        d *= aa
        d += 1.0
        _floor_at_tiny(d)
        # c = 1.0 + aa / c; floor
        np.divide(aa, c, out=c)
        c += 1.0
        _floor_at_tiny(c)
        # d = 1.0 / d; h *= d * c
        np.divide(1.0, d, out=d)
        np.multiply(d, c, out=scratch)
        h *= scratch
        # aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        np.add(a, fm, out=aa)
        np.negative(aa, out=aa)
        np.add(qab, fm, out=scratch)
        aa *= scratch
        aa *= x
        np.add(qap, m2, out=scratch)
        np.multiply(a_m2, scratch, out=scratch)
        aa /= scratch
        # d = 1.0 + aa * d; floor; c = 1.0 + aa / c; floor; d = 1.0 / d
        d *= aa
        d += 1.0
        _floor_at_tiny(d)
        np.divide(aa, c, out=c)
        c += 1.0
        _floor_at_tiny(c)
        np.divide(1.0, d, out=d)
        # delta = d * c; h *= delta; converged where abs(delta - 1.0) < 1e-14
        np.multiply(d, c, out=scratch)
        h *= scratch
        scratch -= 1.0
        np.abs(scratch, out=scratch)
        done = scratch < _BETACF_TOLERANCE
        if done.any():
            out[index[done]] = h[done]
            going = ~done
            index = index[going]
            a, b, x = a[going], b[going], x[going]
            qab, qap, qam = qab[going], qap[going], qam[going]
            c, d, h = c[going], d[going], h[going]
    out[index] = h
    return out


def _betainc_inside_array(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """`betainc` over 1-D arrays with 0 < x < 1 and a, b inside the array
    form's bounds, element for element the scalar's double."""
    # One lgamma triple per distinct (a, b) -- a handful in practice, where
    # the scalar computed it per element. Same function, same order.
    pairs = np.empty(a.shape[0], dtype=np.complex128)
    pairs.real = a
    pairs.imag = b
    distinct, which = np.unique(pairs, return_inverse=True)
    ratio = np.array(
        [
            _lgamma_ratio(p, q)
            for p, q in zip(distinct.real.tolist(), distinct.imag.tolist())
        ],
        dtype=np.float64,
    )[which.reshape(-1)]
    log_x = _each(math.log, x)
    log_1mx = _each(math.log, 1.0 - x)

    out = np.empty(x.shape[0], dtype=np.float64)
    left = x < (a + 1.0) / (a + b + 2.0)
    if left.any():
        al, bl, xl = a[left], b[left], x[left]
        front = _each(math.exp, ratio[left] + al * log_x[left] + bl * log_1mx[left])
        out[left] = front * _betacf_array(al, bl, xl) / al
    right = ~left
    if right.any():
        ar, br, xr = a[right], b[right], x[right]
        front = _each(math.exp, ratio[right] + br * log_1mx[right] + ar * log_x[right])
        out[right] = 1.0 - front * _betacf_array(br, ar, 1.0 - xr) / br
    return out


def f_sf_array(statistic: Any, d1: Any, d2: Any) -> np.ndarray:
    """
    `f_sf` over arrays: element for element the double `f_sf` returns.

    The arguments broadcast together and are taken as doubles. The result
    is `f_sf` to the bit on every input, including the ones `f_sf` treats
    specially, because those ARE `f_sf`: an element with a non-positive,
    NaN or infinite argument, or degrees of freedom outside
    [2**-40, 2**40], is computed by calling it. Where `f_sf` would raise on
    such an element, this raises the same error at the first one.

    Everything else runs as array passes -- the masked continued fraction
    of `_betacf_array`, the lgamma terms once per distinct pair of degrees
    of freedom, and `math.log` / `math.exp` per element (see `_each`) -- so
    a lead-lag search with tens of thousands of p-values is one pass, not
    tens of thousands of Python calls. Fewer than `_F_SF_ARRAY_MIN_BATCH`
    such elements are `f_sf` per element too, since a pass costs more than
    the loop at that size. The clamp into [0, 1] is Python's
    `max(0.0, min(1.0, p))` written out, which a NaN does not survive: it
    returns 1.0 for NaN, where `np.clip` would return NaN.
    """
    statistic_b, d1_b, d2_b = np.broadcast_arrays(
        np.asarray(statistic, dtype=np.float64),
        np.asarray(d1, dtype=np.float64),
        np.asarray(d2, dtype=np.float64),
    )
    shape = statistic_b.shape
    stat = statistic_b.reshape(-1)
    dof1 = d1_b.reshape(-1)
    dof2 = d2_b.reshape(-1)
    result = np.empty(stat.shape[0], dtype=np.float64)

    with np.errstate(invalid="ignore"):
        regular = (
            np.isfinite(stat)
            & (stat > 0)
            & (dof1 >= _F_SF_ARRAY_MIN_DOF)
            & (dof1 <= _F_SF_ARRAY_MAX_DOF)
            & (dof2 >= _F_SF_ARRAY_MIN_DOF)
            & (dof2 <= _F_SF_ARRAY_MAX_DOF)
        )
    if np.count_nonzero(regular) < _F_SF_ARRAY_MIN_BATCH:
        regular[:] = False
    scalar = np.flatnonzero(~regular)
    if scalar.size:
        result[scalar] = [
            f_sf(s, a, b)
            for s, a, b in zip(
                stat[scalar].tolist(), dof1[scalar].tolist(), dof2[scalar].tolist()
            )
        ]

    if regular.any():
        with np.errstate(all="ignore"):
            s, n1, n2 = stat[regular], dof1[regular], dof2[regular]
            x = n2 / (n2 + n1 * s)
            p = np.where(x <= 0, 0.0, 1.0)
            inside = (x > 0) & (x < 1)
            if inside.any():
                p[inside] = _betainc_inside_array(
                    n2[inside] / 2.0, n1[inside] / 2.0, x[inside]
                )
            p = np.where(p < 1.0, p, 1.0)
            result[regular] = np.where(p > 0.0, p, 0.0)
    return result.reshape(shape)
