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
    "norm_cdf",
    "norm_cdf_array",
    "norm_pdf",
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


def norm_cdf_array(x: Any) -> np.ndarray:
    """
    `norm_cdf` over an array. `math.erf` is scalar and numpy has no erf, so
    this is a vectorize wrapper -- the same one `multi_factor` used.

    The empty case is handled explicitly because `np.vectorize` infers its
    output dtype by calling the function once, and on a size-0 input there
    is nothing to call it with: it raises `ValueError: cannot call
    vectorize on size 0 inputs unless otypes is set`. The CDF of no
    observations is no observations, not an error.
    """
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        return np.empty(values.shape, dtype=float)
    return np.vectorize(norm_cdf)(values).astype(float)


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


def betacf(a: float, b: float, x: float, iterations: int = 300) -> float:
    """
    Continued fraction for the incomplete beta, by modified Lentz.

    `tiny` is 1e-300 rather than 1e-30: it exists to keep a denominator away
    from exact zero, and the larger floor perturbs the result at a magnitude
    the algorithm can actually see.
    """
    tiny = 1e-300
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
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b), via the continued fraction on
    whichever side of the symmetry point converges."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return (
        1.0
        - math.exp(
            math.lgamma(a + b)
            - math.lgamma(a)
            - math.lgamma(b)
            + b * math.log(1.0 - x)
            + a * math.log(x)
        )
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
