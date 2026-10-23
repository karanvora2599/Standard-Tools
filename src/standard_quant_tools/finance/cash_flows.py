"""
The time value of a stream of cash: NPV at a rate, every real IRR, MIRR,
and the Macaulay and modified duration, convexity and DV01 at a yield.

A stream is a set of (time, amount) pairs, the time in years from its first
flow. It is built one of two ways. `periodic_cash_flows` takes a list: flow
k is at k / periods_per_year, so flow 0 is NOW and is not discounted --
numpy-financial's convention, and not Excel's NPV, which discounts its
first value a whole period. `dated_cash_flows` takes {date: amount} and a
day count (the library's, `delta_one.daycount`); time runs from the first
date, which is XNPV's and XIRR's convention under ACT/365F.

A RATE COMPOUNDS ONE OF THREE WAYS, and the same number means different
things under each: 'periodic' at periods_per_year a year (a bond yield
quoted semiannually), 'annual' (an effective annual rate, XIRR's), and
'continuous'. Internally every rate is held as u, the continuously
compounded rate, so a discount factor is exp(-u t) whatever the quote, and
each answer is converted back into the caller's convention.

EVERY REAL IRR, NOT ONE. An IRR is a root of NPV(u) = sum c_i exp(-u t_i),
an exponential sum, and it can have none, one or several. A solver that
returns the root nearest its starting guess answers a different question
on every start. `rate_roots` isolates all of them instead:

- Descartes' rule holds for an exponential sum (Laguerre's extension):
  the number of real roots is at most the number V of sign changes of the
  amounts in time order, and has V's parity. V = 0 has no root and V = 1
  exactly one.
- Every root lies inside bounds read from the amounts (a Cauchy bound on
  the dominant term at each end), so the search range is finite and known.
- For V >= 2, Rolle's theorem isolates them: multiplied by exp(u t_j),
  where t_j ends the first run of one sign, the sum keeps its roots and its
  derivative has one term and one sign change fewer. The derivative's roots
  split the range into pieces on which the sum is monotone, so each piece
  holds at most one root, found by bisection. The recursion is V - 1 deep.
- A critical point where the sum is zero (to rounding) is a double root:
  the flows touch zero NPV there without crossing it.

Up to MAX_EXACT_SIGN_CHANGES sign changes the search is exact in that
sense. Beyond it -- flows that change sign more than twenty times, which
real projects do not -- the roots are bracketed on a grid of GRID_POINTS
rates and the search says so, since two roots closer than the grid can
be missed in pairs.

AN IRR IS REPORTED ONLY WHEN IT IS UNIQUE. With two roots (the
Lorie-Savage pump, -1,600 / +10,000 / -10,000, at 25% and 400%) neither is
"the" IRR, and choosing one is choosing an answer. `analyze_cash_flows`
then gives NaN with every root listed, and states Norstrom's (1972)
sufficient condition: when the cumulative flows in time order change sign
exactly once and end non-zero, there is exactly one IRR above 0%.

MIRR discounts the negative flows to time 0 at a finance rate and
compounds the positive ones to the last flow's time at a reinvestment
rate; the MIRR is the rate that grows the first into the second over the
horizon -- Excel's MIRR for annual flows.

DURATION IS OF THE PRESENT VALUE. Macaulay duration is the PV-weighted
mean time of the flows, sum t PV / sum PV; modified duration is
-(1/P) dP/dy, which is Macaulay / (1 + y/m) for a rate compounded m times
a year and Macaulay itself under continuous compounding; convexity is
(1/P) d2P/dy2; DV01 is the change in value for one basis point of yield,
modified duration x P x 1e-4. A bond's duration is of its promised flows,
so its price is NOT one of the flows: with the price included the
present value is zero at the yield and a duration divides by it.

Refused, by name: no flow, more than MAX_FLOWS, a non-finite amount or one
beyond MAX_ABS_AMOUNT, every amount zero, an unknown compounding or day
count, dated flows without a day count, two keys naming one date, and a
rate at or below -100% per compounding period.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from standard_quant_tools.delta_one.daycount import CONVENTIONS
from standard_quant_tools.delta_one.daycount import _convention as _day_count_name
from standard_quant_tools.delta_one.daycount import to_date, year_fraction
from standard_quant_tools.error import ValidationError

__all__ = [
    "COMPOUNDINGS",
    "GRID_POINTS",
    "MAX_ABS_AMOUNT",
    "MAX_ABS_RATE",
    "MAX_EXACT_SIGN_CHANGES",
    "MAX_FLOWS",
    "CashFlowSchedule",
    "RateRoots",
    "analyze_cash_flows",
    "cumulative_sign_changes",
    "dated_cash_flows",
    "log_rate",
    "periodic_cash_flows",
    "rate_from_log",
    "rate_roots",
    "root_status",
    "sign_changes",
]

#: How a rate compounds: at the flow frequency, once a year, or continuously.
COMPOUNDINGS = ("periodic", "annual", "continuous")
#: The most flows one stream takes: 166 years of monthly flows.
MAX_FLOWS = 2_000
#: The largest amount taken, a quadrillion in any currency. A larger one
#: is a unit mistake, and its square in a convexity sum would overflow.
MAX_ABS_AMOUNT = 1e15
#: A rate is a decimal, so 10 is 1,000%. A larger one is a percentage
#: passed where a decimal was meant.
MAX_ABS_RATE = 10.0
#: Up to this many sign changes the roots are isolated exactly (Rolle);
#: beyond it they are bracketed on a grid.
MAX_EXACT_SIGN_CHANGES = 20
#: Rates on the fallback grid, spaced evenly in asinh(u / 0.01): about
#: 0.003% apart near zero and 0.3% of the rate far from it.
GRID_POINTS = 8_193

_EPS = float(np.finfo(float).eps)
#: Elements of one evaluation block, so a long grid never builds a matrix
#: larger than about 32 MB.
_BLOCK = 4_000_000


@dataclass(frozen=True)
class CashFlowSchedule:
    """A stream of cash: each flow's time in years from the first, and how
    a rate on it compounds."""

    times: Tuple[float, ...]
    amounts: Tuple[float, ...]
    compounding: str
    periods_per_year: int
    day_count: Optional[str] = None
    first_date: Optional[str] = None
    last_date: Optional[str] = None


@dataclass(frozen=True)
class RateRoots:
    """Every real root of an NPV, as continuously compounded rates."""

    #: Ascending. A rate r in the caller's convention is `rate_from_log`.
    log_rates: Tuple[float, ...]
    #: 'exact' (Rolle isolation), 'grid' (bracketed on GRID_POINTS rates),
    #: or 'none' (no sign change, so no search was needed).
    method: str
    #: Sign changes of the amounts in time order: at most that many roots.
    sign_changes: int
    #: The u range every root lies in, from the Cauchy bounds; None when
    #: there is no root to look for.
    search_range: Optional[Tuple[float, float]] = None


# ── building a schedule ──────────────────────────────────────────────────


def _check_compounding(compounding: Any, periods_per_year: Any) -> Tuple[str, int]:
    if compounding not in COMPOUNDINGS:
        raise ValidationError(
            f"compounding={compounding!r} is not one of {list(COMPOUNDINGS)}: "
            "'periodic' compounds periods_per_year times a year, 'annual' "
            "once (an effective annual rate, XIRR's), 'continuous' always."
        )
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, (int, np.integer))
        or not 1 <= int(periods_per_year) <= 365
    ):
        raise ValidationError(
            f"periods_per_year={periods_per_year!r} must be a whole number "
            "from 1 to 365: 1 for annual flows, 2 semiannual, 4 quarterly, "
            "12 monthly."
        )
    return str(compounding), int(periods_per_year)


def _check_amounts(
    amounts: Sequence[Any], field: str, labels: Optional[Sequence[str]] = None
) -> List[float]:
    """Finite, bounded, not all zero; a bad entry named by its position,
    or by its date when `labels` gives them."""
    values = [float(a) for a in amounts]

    def where(indexes: List[int]) -> str:
        if labels is None:
            return f"positions {indexes[:5]}"
        return f"dates {[labels[i] for i in indexes[:5]]}"

    if not values:
        raise ValidationError(f"{field} is empty; give at least one flow.")
    if len(values) > MAX_FLOWS:
        raise ValidationError(
            f"{field} has {len(values)} flows, more than the {MAX_FLOWS} one "
            "stream takes. Aggregate them to a coarser period first."
        )
    bad = [i for i, v in enumerate(values) if not math.isfinite(v)]
    if bad:
        raise ValidationError(
            f"{field} is not finite at {where(bad)}. A flow is an "
            "amount of cash; drop a period with none or give it as 0.0."
        )
    big = [i for i, v in enumerate(values) if abs(v) > MAX_ABS_AMOUNT]
    if big:
        raise ValidationError(
            f"{field} has amounts beyond {MAX_ABS_AMOUNT:g} at {where(big)} "
            f"(e.g. {values[big[0]]:g}). That is a unit mistake; "
            "state the flows in thousands or millions."
        )
    if all(v == 0.0 for v in values):
        raise ValidationError(
            f"every amount in {field} is zero: there is nothing to value and "
            "every rate is an IRR."
        )
    return values


def periodic_cash_flows(
    flows: Sequence[float],
    *,
    periods_per_year: int = 1,
    compounding: Optional[str] = None,
) -> CashFlowSchedule:
    """
    Flows at regular periods: flow k at k / periods_per_year years, so flow
    0 is now. `compounding` defaults to 'periodic' -- at the flows' own
    frequency, the way a bond's yield is quoted.
    """
    compounding, m = _check_compounding(compounding or "periodic", periods_per_year)
    amounts = _check_amounts(flows, "flows")
    times = tuple(k / m for k in range(len(amounts)))
    return CashFlowSchedule(times, tuple(amounts), compounding, m)


def dated_cash_flows(
    flows: Mapping[Any, float],
    *,
    day_count: Optional[str],
    compounding: Optional[str] = None,
    periods_per_year: int = 1,
) -> CashFlowSchedule:
    """
    Flows on dates: each one's time is the year fraction from the earliest
    date under `day_count` (required -- the same dates are different times
    under ACT/360 and ACT/365F). `compounding` defaults to 'annual', XIRR's
    convention; 'periodic' compounds periods_per_year times a year.
    """
    if day_count is None:
        raise ValidationError(
            "dated flows need a day_count: the same dates are different "
            "times under different conventions, and a rate is quoted under "
            f"one. One of {sorted(CONVENTIONS)}; XIRR uses 'ACT/365F'."
        )
    compounding, m = _check_compounding(compounding or "annual", periods_per_year)
    if not flows:
        raise ValidationError("dated_flows is empty; give at least one.")
    by_date: Dict[Any, float] = {}
    for key, amount in flows.items():
        day = to_date(key, f"dated flow date {key!r}")
        if day in by_date:
            raise ValidationError(
                f"two dated flows fall on {day.isoformat()} (one is {key!r}). "
                "Add them into one amount for that date."
            )
        by_date[day] = amount
    days = sorted(by_date)
    amounts = _check_amounts(
        [by_date[d] for d in days], "dated_flows", [d.isoformat() for d in days]
    )
    times = tuple(year_fraction(days[0], d, convention=day_count) for d in days)
    return CashFlowSchedule(
        times,
        tuple(amounts),
        compounding,
        m,
        day_count=_day_count_name(day_count),
        first_date=days[0].isoformat(),
        last_date=days[-1].isoformat(),
    )


# ── rates ────────────────────────────────────────────────────────────────


def _per_year(compounding: str, periods_per_year: int) -> Optional[int]:
    """Compounding periods a year; None when continuous."""
    if compounding == "continuous":
        return None
    return 1 if compounding == "annual" else int(periods_per_year)


def log_rate(
    rate: float, compounding: str, periods_per_year: int = 1, name: str = "rate"
) -> float:
    """
    The continuously compounded rate u equivalent to `rate` under the
    compounding: a discount factor at time t is exp(-u t).
    """
    value = float(rate)
    if not math.isfinite(value) or abs(value) > MAX_ABS_RATE:
        raise ValidationError(
            f"{name}={rate!r} must be a finite decimal within "
            f"+/-{MAX_ABS_RATE:g} (0.05 is 5%); a percentage passed as 5 is "
            "500%."
        )
    m = _per_year(compounding, periods_per_year)
    if m is None:
        return value
    if value <= -m:
        raise ValidationError(
            f"{name}={rate!r} is -100% per compounding period or below "
            f"({compounding}, {m} a year), which has no discount factor. A "
            "rate above it is needed."
        )
    return m * math.log1p(value / m)


def rate_from_log(u: float, compounding: str, periods_per_year: int = 1) -> float:
    """`log_rate`'s inverse: u in the caller's convention. A root past a
    float's range comes back infinite rather than raising."""
    m = _per_year(compounding, periods_per_year)
    if m is None:
        return float(u)
    with np.errstate(over="ignore"):
        return float(m * np.expm1(u / m))


# ── signs ────────────────────────────────────────────────────────────────


def sign_changes(amounts: Sequence[float]) -> int:
    """Sign changes in a sequence, zeros skipped: Descartes' bound."""
    signs = [1 if a > 0 else -1 for a in amounts if a != 0]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def cumulative_sign_changes(amounts: Sequence[float]) -> Tuple[int, bool]:
    """
    Sign changes of the running sums, and whether Norstrom's condition
    holds: exactly one change, with the final sum non-zero.
    """
    running, sums = [], []
    for amount in amounts:
        running.append(amount)
        sums.append(math.fsum(running))
    changes = sign_changes(sums)
    return changes, changes == 1 and sums[-1] != 0.0


# ── the roots ────────────────────────────────────────────────────────────


def _merged(times: Sequence[float], amounts: Sequence[float]):
    """Distinct times ascending, amounts on one time added, zeros dropped.
    Two dates 30/360 counts as one time are one term of the sum."""
    grouped: Dict[float, List[float]] = {}
    for t, a in zip(times, amounts):
        grouped.setdefault(float(t), []).append(float(a))
    t_out, c_out = [], []
    for t in sorted(grouped):
        total = math.fsum(grouped[t])
        if total != 0.0:
            t_out.append(t)
            c_out.append(total)
    return np.asarray(t_out, dtype=float), np.asarray(c_out, dtype=float)


def _evaluate(u: np.ndarray, t: np.ndarray, c: np.ndarray):
    """
    sum c_i exp(-u t_i) at each u, scaled by exp(-max_i(-u t_i)) -- a
    positive continuous factor, so the sign and the roots are the sum's --
    and the same scaling of sum |c_i| exp(-u t_i), the size rounding is
    measured against. Plain sums rather than a matrix product, so the bits
    do not depend on a BLAS's threads.
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    value = np.empty(u.size)
    scale = np.empty(u.size)
    step = max(1, _BLOCK // max(1, t.size))
    magnitude = np.abs(c)
    for start in range(0, u.size, step):
        exponent = -np.multiply.outer(u[start : start + step], t)
        weights = np.exp(exponent - exponent.max(axis=1, keepdims=True))
        value[start : start + step] = (weights * c).sum(axis=1)
        scale[start : start + step] = (weights * magnitude).sum(axis=1)
    return value, scale


def _tolerance(n_terms: int) -> float:
    """How close to zero, relative to the scaled magnitude, rounding can
    leave a sum of n terms that is exactly zero."""
    return 16.0 * max(1, n_terms) * _EPS


def _bisect(t, c, a, b, ga) -> List[float]:
    """Each bracket [a_k, b_k] with a sign change, bisected to the last
    bits together: a fixed procedure, so the same input gives the same
    bits."""
    if a.size == 0:
        return []
    a, b, side = a.astype(float), b.astype(float), np.sign(ga)
    for _ in range(256):
        mid = 0.5 * (a + b)
        if np.all((b - a) <= 4.0 * _EPS * np.maximum(1.0, np.abs(mid))):
            break
        value, _scale = _evaluate(mid, t, c)
        sign = np.sign(value)
        right = sign == side  # the root is in [mid, b]
        a, b = np.where(right, mid, a), np.where(right, b, mid)
        exact = sign == 0
        a, b = np.where(exact, mid, a), np.where(exact, mid, b)
    return [float(x) for x in 0.5 * (a + b)]


def _roots_between(t, c, points: np.ndarray) -> List[float]:
    """The roots in [points[0], points[-1]] when the sum is monotone
    between consecutive points: a point where it is zero to rounding, and
    one root in each piece whose ends differ in sign."""
    value, scale = _evaluate(points, t, c)
    zero = np.abs(value) <= _tolerance(t.size) * scale
    value = np.where(zero, 0.0, value)
    roots = [float(p) for p in points[zero]]
    crosses = (value[:-1] * value[1:]) < 0
    roots += _bisect(
        t, c, points[:-1][crosses], points[1:][crosses], value[:-1][crosses]
    )
    return sorted(set(roots))


def _isolate(t: np.ndarray, c: np.ndarray, lo: float, hi: float) -> List[float]:
    """Every root in [lo, hi] by Rolle isolation (the module docstring)."""
    signs = np.sign(c)
    changes = np.flatnonzero(signs[1:] != signs[:-1])
    if changes.size == 0:
        return []
    critical: List[float] = []
    if changes.size > 1:
        j = int(changes[0])  # the last term of the first run of one sign
        keep = np.arange(t.size) != j
        derivative = -(t[keep] - t[j]) * c[keep]
        derivative = derivative / np.max(np.abs(derivative))
        critical = _isolate(t[keep], derivative, lo, hi)
    points = np.asarray([lo, *critical, hi], dtype=float)
    return _roots_between(t, c, points)


def _grid(t: np.ndarray, c: np.ndarray, lo: float, hi: float) -> List[float]:
    """Roots bracketed on GRID_POINTS rates, dense near zero."""
    scale = 0.01
    z = np.linspace(np.arcsinh(lo / scale), np.arcsinh(hi / scale), GRID_POINTS)
    points = scale * np.sinh(z)
    points[0], points[-1] = lo, hi
    return _roots_between(t, c, np.maximum.accumulate(points))


def _bounds(t: np.ndarray, c: np.ndarray) -> Tuple[float, float]:
    """
    A u range holding every root. In x = exp(-u) the sum is
    sum c_i x^t_i; for x >= 1 the last term dominates once x^(t_n -
    t_{n-1}) > sum_{i<n} |c_i| / |c_n|, and for x <= 1 the first once
    x^(t_1 - t_0) < |c_0| / sum_{i>0} |c_i|. Widened by a hair so a root
    on a bound is not lost to rounding.
    """
    a = np.abs(c)
    big = max(0.0, math.log(math.fsum(a[:-1]) / a[-1]) / (t[-1] - t[-2]))
    small = min(0.0, math.log(a[0] / math.fsum(a[1:])) / (t[1] - t[0]))
    lo, hi = -big, -small
    pad = 1e-9 * (1.0 + hi - lo)
    return lo - pad, hi + pad


def rate_roots(times: Sequence[float], amounts: Sequence[float]) -> RateRoots:
    """
    Every real u with sum amounts_i exp(-u times_i) = 0, ascending: the
    IRRs of the flows as continuously compounded rates (the module
    docstring says how they are found and when the answer is exact).
    """
    t, c = _merged(times, amounts)
    if c.size:
        # Scaled so the largest is 1; an amount 1e-308 of the largest is
        # then below a float and leaves the sum, as it would its value.
        c = c / np.max(np.abs(c))
        t, c = t[c != 0.0], c[c != 0.0]
    changes = sign_changes(list(c))
    if changes == 0 or c.size < 2:
        return RateRoots((), "none", changes)
    lo, hi = _bounds(t, c)
    if changes <= MAX_EXACT_SIGN_CHANGES:
        roots, method = _isolate(t, c, lo, hi), "exact"
    else:
        roots, method = _grid(t, c, lo, hi), "grid"
    return RateRoots(tuple(roots), method, changes, (lo, hi))


# ── the analysis ─────────────────────────────────────────────────────────


def _total(values: Any) -> float:
    """The correctly rounded sum, so the order of the flows cannot move a
    bit; a sum with an overflowed term is infinite or NaN rather than the
    ValueError `math.fsum` raises on inf + -inf."""
    array = np.asarray(values, dtype=float)
    if np.isfinite(array).all():
        return math.fsum(array)
    with np.errstate(invalid="ignore"):
        return float(np.sum(array))


def _present_values(t: np.ndarray, c: np.ndarray, u: float) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore"):
        return c * np.exp(-u * t)


def root_status(roots: RateRoots) -> Tuple[str, str]:
    """
    Whether the roots give one IRR: ('unique' | 'none' | 'multiple' |
    'uncertified', the reason there is no single one, '' when there is).
    """
    n = len(roots.log_rates)
    if roots.method == "grid":
        return (
            "uncertified",
            f"the flows change sign {roots.sign_changes} times, more than the "
            f"{MAX_EXACT_SIGN_CHANGES} the exact search takes, so the roots "
            f"were bracketed on a grid of {GRID_POINTS} rates and found {n}; "
            "two roots closer than the grid can be missed in pairs, so no "
            "root is certified unique",
        )
    if n == 1:
        return "unique", ""
    if n == 0:
        if roots.sign_changes == 0:
            return (
                "none",
                "every flow has the same sign, so no rate makes the NPV zero "
                "and there is no IRR",
            )
        return (
            "none",
            "no real rate makes the NPV zero (its roots are complex), so "
            "there is no IRR; compare NPVs at a stated rate instead",
        )
    return (
        "multiple",
        f"the flows have {n} IRRs, all listed, so none of them is the IRR; "
        "decide on the NPV at a stated rate, or on the MIRR",
    )


def analyze_cash_flows(
    schedule: CashFlowSchedule,
    *,
    rate: Optional[float] = None,
    finance_rate: Optional[float] = None,
    reinvest_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    NPV at `rate`, every real IRR (the IRR itself only when unique), MIRR at
    the finance and reinvestment rates, and the Macaulay and modified
    duration, convexity and DV01 of the flows at `rate` as a yield.

    Every rate is in the schedule's compounding. A number not asked for is
    None: NPV and the durations without `rate`, MIRR without its two rates.
    A number asked for and not defined is NaN: the IRR unless it is unique
    (`irr_status` says why, from `root_status`), a duration when the
    present value at the rate is zero, a MIRR without a negative and a
    positive flow at different times.
    """
    comp, m = schedule.compounding, schedule.periods_per_year
    t = np.asarray(schedule.times, dtype=float)
    c = np.asarray(schedule.amounts, dtype=float)
    notes: List[str] = []
    horizon = float(t[-1]) if t.size else 0.0

    roots = rate_roots(t, c)
    rates = [rate_from_log(u, comp, m) for u in roots.log_rates]
    status, _why = root_status(roots)
    # In time order, one amount per time: what both rules of signs read.
    cumulative, norstrom = cumulative_sign_changes(list(_merged(t, c)[1]))

    out: Dict[str, Any] = {
        "n_flows": int(c.size),
        "horizon_years": horizon,
        "total": math.fsum(c),
        "compounding": comp,
        "periods_per_year": m,
        "day_count": schedule.day_count,
        "first_date": schedule.first_date,
        "last_date": schedule.last_date,
        "irr": rates[0] if status == "unique" else float("nan"),
        "irr_status": status,
        "irr_roots": rates,
        "sign_changes": roots.sign_changes,
        "cumulative_sign_changes": cumulative,
        "norstrom_holds": norstrom,
        "root_search": roots.method,
        "search_range": (
            None
            if roots.search_range is None
            else [rate_from_log(u, comp, m) for u in roots.search_range]
        ),
        # None: not asked for (no rate, no MIRR rates). NaN: asked for and
        # not defined on these flows.
        "rate": None,
        "npv": None,
        "macaulay_duration": None,
        "modified_duration": None,
        "convexity": None,
        "dv01": None,
        "mirr": None,
        "warnings": notes,
    }
    if roots.method == "exact" and norstrom and status == "multiple":
        # Norstrom guarantees one root above 0%; the others are below it.
        notes.append(
            "Norstrom's condition holds, so exactly one root is above 0%; "
            "the others are at or below 0%."
        )

    if rate is not None:
        u = log_rate(rate, comp, m, "rate")
        pv = _present_values(t, c, u)
        price = _total(pv)
        out["rate"] = float(rate)
        out["npv"] = price
        magnitude = _total(np.abs(pv))
        nan = float("nan")
        out.update(
            macaulay_duration=nan, modified_duration=nan, convexity=nan, dv01=nan
        )
        if price != 0.0 and abs(price) > _tolerance(c.size) * magnitude:
            macaulay = _total(t * pv) / price
            per_year = _per_year(comp, m)
            if per_year is None:
                modified = macaulay
                convexity = _total(t * t * pv) / price
            else:
                growth = 1.0 + float(rate) / per_year
                modified = macaulay / growth
                convexity = _total(pv * t * (t + 1.0 / per_year)) / (
                    price * growth * growth
                )
            out.update(
                macaulay_duration=macaulay,
                modified_duration=modified,
                convexity=convexity,
                dv01=modified * price * 1e-4,
            )
            if np.any(pv > 0) and np.any(pv < 0):
                notes.append(
                    "The flows change sign, so the Macaulay duration is a "
                    "sensitivity rather than a time inside the horizon; read "
                    "modified_duration and dv01."
                )

    if finance_rate is not None and reinvest_rate is not None:
        out["mirr"] = _mirr(t, c, horizon, finance_rate, reinvest_rate, comp, m)
    return out


def _mirr(t, c, horizon, finance_rate, reinvest_rate, comp, m) -> float:
    """NaN without a negative and a positive flow, or with no time between
    the first flow and the last."""
    u_finance = log_rate(finance_rate, comp, m, "finance_rate")
    u_reinvest = log_rate(reinvest_rate, comp, m, "reinvest_rate")
    negative, positive = c < 0, c > 0
    if not negative.any() or not positive.any() or horizon <= 0.0:
        return float("nan")
    with np.errstate(over="ignore", under="ignore"):
        cost = -_total(c[negative] * np.exp(-u_finance * t[negative]))
        terminal = _total(c[positive] * np.exp(u_reinvest * (horizon - t[positive])))
    if not (cost > 0.0 and terminal > 0.0 and math.isfinite(terminal / cost)):
        return float("nan")
    return rate_from_log(math.log(terminal / cost) / horizon, comp, m)
