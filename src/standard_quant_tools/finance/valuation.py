"""
A discounted cash flow value of forecast free cash flows, the reverse --
what a target value implies -- and how the answer moves with its two
assumptions.

    value          = sum_{t=1..N} FCF_t / (1 + r)^t  +  TV / (1 + r)^N
    Gordon         TV = FCF_N (1 + g) / (r - g)          (needs g < r)
    exit multiple  TV = M x metric_N                      (metric_N: EBITDA,
                                                           say; FCF_N by default)

Each forecast flow is at its year's end and r is an effective annual rate.
With FCF growing at g from FCF_1 the sum and the Gordon value add up to
FCF_1 / (r - g) exactly -- the planted check -- whatever N is.

THE TERMINAL SHARE IS ALWAYS REPORTED: PV(TV) / value. A ten-year forecast
at 9% with 3% terminal growth is routinely 60-80% terminal value, and then
the valuation is a statement about g and r rather than about the
forecast. Each value also comes back as the other method's assumption --
the exit multiple a Gordon value implies (TV / FCF_N) and the perpetual
growth an exit multiple implies ((TV r - FCF_N) / (TV + FCF_N)) -- since
an 18x exit that implies 7% growth forever is a contradiction worth seeing.

THE REVERSE DCF solves for the one assumption a `target_value` implies:
the discount rate, the terminal growth or the exit multiple. The rate and
the growth are found by scanning their whole admissible range -- r in
(g, 10] for Gordon and (-0.99, 10] for a multiple, g in (-1, r) -- for
every sign change of value - target, and polishing each bracket with
Brent's method; the bracket that held the answer and the range searched
are both reported. A target the model cannot reach over the range has no
answer, and a target reached twice (negative flows can do that) has two,
both listed and neither chosen. The multiple is linear in the value and is
solved in closed form.

THE SENSITIVITY GRID is the value at each discount rate against each
terminal growth (or exit multiple): by default the solved or given value
+/-2 points of rate in 1-point steps, +/-1 point of growth in half-point
steps, or the multiple x 0.8 ... 1.2. A Gordon cell with g >= r has no
value and is NaN.

Refused, naming the argument: no forecast or more than MAX_YEARS, a
non-finite or out-of-range input, Gordon growth at or above the discount
rate, an input of the other terminal method, the parameter being solved
for passed as well, a target without `solve_for` or the reverse.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from standard_quant_tools.error import ValidationError

__all__ = [
    "MAX_GRID",
    "MAX_YEARS",
    "SOLVABLE",
    "TERMINAL_METHODS",
    "check_dcf_inputs",
    "dcf_valuation",
]

#: The longest forecast taken, in years.
MAX_YEARS = 200
#: The most rows, and columns, of a sensitivity grid.
MAX_GRID = 15
TERMINAL_METHODS = ("gordon", "exit_multiple")
SOLVABLE = ("discount_rate", "terminal_growth", "exit_multiple")

#: The largest rate taken or searched: 10 is 1,000%.
_RATE_CEILING = 10.0
#: The lowest discount rate a multiple-based search tries: -99%.
_RATE_FLOOR = -0.99
_MAX_ABS_AMOUNT = 1e15
_MAX_MULTIPLE = 1_000.0
_SCAN_POINTS = 601
_EPS = float(np.finfo(float).eps)


# ── checks ───────────────────────────────────────────────────────────────


def _finite(value: Any, name: str, *, low: float, high: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not low <= number <= high:
        raise ValidationError(
            f"{name}={value!r} must be a finite number in [{low:g}, {high:g}]."
        )
    return number


def _rate(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not -1.0 < number <= _RATE_CEILING:
        raise ValidationError(
            f"{name}={value!r} must be a decimal rate above -1 and at most "
            f"{_RATE_CEILING:g} (0.09 is 9%); a percentage passed as 9 is 900%."
        )
    return number


def _amounts(values: Sequence[Any]) -> np.ndarray:
    flows = [float(v) for v in values]
    if not flows:
        raise ValidationError("cash_flows is empty; give the forecast for years 1..N.")
    if len(flows) > MAX_YEARS:
        raise ValidationError(
            f"cash_flows has {len(flows)} years, more than the {MAX_YEARS} "
            "taken; the terminal value carries the years after the forecast."
        )
    bad = [i for i, v in enumerate(flows) if not math.isfinite(v)]
    if bad:
        raise ValidationError(
            f"cash_flows is not finite at years {[i + 1 for i in bad[:5]]}."
        )
    big = [i for i, v in enumerate(flows) if abs(v) > _MAX_ABS_AMOUNT]
    if big:
        raise ValidationError(
            f"cash_flows exceeds {_MAX_ABS_AMOUNT:g} in absolute value at years "
            f"{[i + 1 for i in big[:5]]}: a unit mistake; state the forecast "
            "in thousands or millions."
        )
    return np.asarray(flows, dtype=float)


def _gordon_room(growth: float, rate: float) -> None:
    if not growth < rate:
        raise ValidationError(
            f"terminal_growth={growth:g} is not below discount_rate={rate:g}. "
            "A Gordon value FCF_N (1 + g) / (r - g) is infinite at g = r and "
            "negative above it: no business outgrows its discount rate "
            "forever. Lower terminal_growth, raise discount_rate, or use "
            "terminal_method='exit_multiple'."
        )


# ── the value ────────────────────────────────────────────────────────────


def _total(values: Any) -> float:
    """The correctly rounded sum; infinite or NaN, rather than the
    ValueError `math.fsum` raises, when a discount factor overflowed."""
    array = np.asarray(values, dtype=float)
    if np.isfinite(array).all():
        return math.fsum(array)
    with np.errstate(invalid="ignore"):
        return float(np.sum(array))


def _discount_factors(rate: float, n_years: int) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore"):
        return np.exp(-np.arange(1, n_years + 1) * math.log1p(rate))


def _terminal(flows, method, growth, multiple, metric, rate) -> float:
    if method == "gordon":
        return float(flows[-1]) * (1.0 + growth) / (rate - growth)
    return multiple * metric


def _value(flows, method, rate, growth, multiple, metric) -> Dict[str, Any]:
    factors = _discount_factors(rate, flows.size)
    pv_years = flows * factors
    pv_forecast = _total(pv_years)
    terminal = _terminal(flows, method, growth, multiple, metric, rate)
    pv_terminal = terminal * float(factors[-1])
    value = pv_forecast + pv_terminal
    share = pv_terminal / value if value != 0.0 else float("nan")
    return {
        "value": value,
        "pv_forecast": pv_forecast,
        "terminal_value": terminal,
        "pv_terminal": pv_terminal,
        "terminal_share": share,
        "pv_by_year": [float(x) for x in pv_years],
    }


def _value_only(flows, method, rate, growth, multiple, metric) -> float:
    factors = _discount_factors(rate, flows.size)
    terminal = _terminal(flows, method, growth, multiple, metric, rate)
    return _total(flows * factors) + terminal * float(factors[-1])


# ── the reverse ──────────────────────────────────────────────────────────


def _scan(
    residual: Callable[[float], float], points: np.ndarray
) -> List[Tuple[float, Tuple[float, float]]]:
    """Every root of `residual` over the points: a point where it is zero,
    and each sign change polished by Brent's method; with its bracket."""
    with np.errstate(all="ignore"):
        values = np.array([residual(float(p)) for p in points])
    finite = np.isfinite(values)
    out: List[Tuple[float, Tuple[float, float]]] = []
    for i in np.flatnonzero(finite & (values == 0.0)):
        out.append((float(points[i]), (float(points[i]), float(points[i]))))
    for i in range(points.size - 1):
        a, b, fa, fb = points[i], points[i + 1], values[i], values[i + 1]
        if not (finite[i] and finite[i + 1]) or fa == 0.0 or fb == 0.0:
            continue
        if (fa < 0.0) != (fb < 0.0):
            root = brentq(residual, a, b, xtol=1e-15, rtol=4 * _EPS, maxiter=500)
            out.append((float(root), (float(a), float(b))))
    return sorted(out)


def _rate_points(method: str, growth: Optional[float]) -> np.ndarray:
    if method == "gordon":
        assert growth is not None
        gap = _RATE_CEILING - growth
        return growth + gap * np.geomspace(1e-12, 1.0, _SCAN_POINTS)
    logs = np.linspace(math.log1p(_RATE_FLOOR), math.log1p(_RATE_CEILING), _SCAN_POINTS)
    points = np.expm1(logs)
    points[0], points[-1] = _RATE_FLOOR, _RATE_CEILING
    return points


def _growth_points(rate: float) -> np.ndarray:
    # g = r - (1 + r) q, q in (0, 1): from just under r down to just above -1.
    return np.sort(rate - (1.0 + rate) * np.geomspace(1e-12, 1.0 - 1e-9, _SCAN_POINTS))


# ── the call ─────────────────────────────────────────────────────────────


def _given(name: str, value: Any, solve_for: Optional[str], method: str) -> None:
    if solve_for == name and value is not None:
        raise ValidationError(
            f"solve_for={name!r} solves for {name}, and {name}={value!r} was "
            f"given too. Drop {name}, or drop solve_for and target_value."
        )
    if solve_for != name and value is None:
        raise ValidationError(
            f"{name} is required here (terminal_method={method!r}), or "
            f"solve_for={name!r} with a target_value to imply it from."
        )


def check_dcf_inputs(
    *,
    terminal_method: str = "gordon",
    discount_rate: Optional[float] = None,
    terminal_growth: Optional[float] = None,
    exit_multiple: Optional[float] = None,
    terminal_metric: Optional[float] = None,
    target_value: Optional[float] = None,
    solve_for: Optional[str] = None,
) -> None:
    """
    The rules between `dcf_valuation`'s inputs, refused by name: each
    method's own inputs and not the other's, the solved input not given as
    well, a target with `solve_for` and not without, Gordon growth below
    the rate. The agent tool's input model runs the same check.
    """
    if terminal_method not in TERMINAL_METHODS:
        raise ValidationError(
            f"terminal_method={terminal_method!r} is not one of "
            f"{list(TERMINAL_METHODS)}."
        )
    if solve_for is not None and solve_for not in SOLVABLE:
        raise ValidationError(
            f"solve_for={solve_for!r} is not one of {list(SOLVABLE)}."
        )
    if (target_value is None) != (solve_for is None):
        given = "target_value" if target_value is not None else "solve_for"
        raise ValidationError(
            f"{given} was given alone: a reverse DCF needs target_value (the "
            "value to explain) and solve_for (the assumption that explains "
            "it) together."
        )
    other = {"gordon": "exit_multiple", "exit_multiple": "gordon"}[terminal_method]
    if terminal_method == "gordon":
        for name, value in (
            ("exit_multiple", exit_multiple),
            ("terminal_metric", terminal_metric),
        ):
            if value is not None:
                raise ValidationError(
                    f"{name}={value!r} belongs to terminal_method='{other}', "
                    "and this call is 'gordon'. Drop it, or switch methods."
                )
        if solve_for == "exit_multiple":
            raise ValidationError(
                "solve_for='exit_multiple' needs terminal_method="
                "'exit_multiple'; a Gordon value has no multiple to solve for."
            )
        _given("terminal_growth", terminal_growth, solve_for, terminal_method)
    else:
        if terminal_growth is not None:
            raise ValidationError(
                f"terminal_growth={terminal_growth!r} belongs to "
                "terminal_method='gordon', and this call is 'exit_multiple'. "
                "Drop it; the result reports the growth the multiple implies."
            )
        if solve_for == "terminal_growth":
            raise ValidationError(
                "solve_for='terminal_growth' needs terminal_method='gordon'; "
                "an exit multiple has no growth rate to solve for."
            )
        _given("exit_multiple", exit_multiple, solve_for, terminal_method)
    _given("discount_rate", discount_rate, solve_for, terminal_method)
    if (
        terminal_method == "gordon"
        and discount_rate is not None
        and terminal_growth is not None
    ):
        _gordon_room(float(terminal_growth), float(discount_rate))


def dcf_valuation(
    cash_flows: Sequence[float],
    *,
    discount_rate: Optional[float] = None,
    terminal_method: str = "gordon",
    terminal_growth: Optional[float] = None,
    exit_multiple: Optional[float] = None,
    terminal_metric: Optional[float] = None,
    target_value: Optional[float] = None,
    solve_for: Optional[str] = None,
    sensitivity_rates: Optional[Sequence[float]] = None,
    sensitivity_terminal: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    The DCF value of `cash_flows` (years 1..N, year end), its terminal
    value and share, the reverse solve when a `target_value` is given, and
    the sensitivity grid. A number that does not apply to the call (the
    other method's inputs, a solve nobody asked for) is None; one that
    applies and is not defined is NaN: the value when the reverse solve
    has no single answer, the terminal share of a zero value, a Gordon grid
    cell with g >= r.
    """
    flows = _amounts(cash_flows)
    check_dcf_inputs(
        terminal_method=terminal_method,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        exit_multiple=exit_multiple,
        terminal_metric=terminal_metric,
        target_value=target_value,
        solve_for=solve_for,
    )
    method = terminal_method

    rate = None if discount_rate is None else _rate(discount_rate, "discount_rate")
    growth = (
        None if terminal_growth is None else _rate(terminal_growth, "terminal_growth")
    )
    multiple = (
        None
        if exit_multiple is None
        else _finite(exit_multiple, "exit_multiple", low=0.0, high=_MAX_MULTIPLE)
    )
    if multiple is not None and multiple <= 0.0:
        raise ValidationError(
            f"exit_multiple={exit_multiple!r} must be positive; a multiple of "
            "zero says the business is worth nothing at year N."
        )
    metric = (
        float(flows[-1])
        if terminal_metric is None
        else _finite(
            terminal_metric,
            "terminal_metric",
            low=-_MAX_ABS_AMOUNT,
            high=_MAX_ABS_AMOUNT,
        )
    )
    if method == "gordon" and rate is not None and growth is not None:
        _gordon_room(growth, rate)
    target = (
        None
        if target_value is None
        else _finite(
            target_value,
            "target_value",
            low=-_MAX_ABS_AMOUNT,
            high=_MAX_ABS_AMOUNT,
        )
    )

    notes: List[str] = []
    nan = float("nan")
    out: Dict[str, Any] = {
        "n_years": int(flows.size),
        "terminal_method": method,
        # None: does not apply to this call. NaN: applies and is not defined.
        "terminal_metric": metric if method == "exit_multiple" else None,
        "solve_for": solve_for,
        "target_value": target,
        "implied": None if solve_for is None else nan,
        "implied_roots": [],
        "implied_bracket": None,
        "search_range": None,
        "warnings": notes,
    }

    if solve_for is not None:
        assert target is not None
        if solve_for == "exit_multiple":
            assert rate is not None
            factors = _discount_factors(rate, flows.size)
            gap = target - _total(flows * factors)
            denominator = float(factors[-1]) * metric
            if denominator == 0.0:
                notes.append(
                    "The terminal metric is zero, so no multiple reaches the "
                    "target: every multiple gives the same value."
                )
            else:
                multiple = gap / denominator
                out["implied"] = multiple
                out["implied_roots"] = [multiple]
                if multiple <= 0.0:
                    notes.append(
                        f"The implied exit multiple is {multiple:.4g}: the "
                        "forecast alone is worth more than the target, so the "
                        "target prices the business at year N below nothing."
                    )
        else:
            if solve_for == "discount_rate":
                points = _rate_points(method, growth)

                def residual(r: float) -> float:
                    return (
                        _value_only(flows, method, r, growth, multiple, metric) - target
                    )

            else:
                assert rate is not None
                points = _growth_points(rate)

                def residual(g: float) -> float:
                    return (
                        _value_only(flows, method, rate, g, multiple, metric) - target
                    )

            found = _scan(residual, points)
            out["search_range"] = [float(points[0]), float(points[-1])]
            out["implied_roots"] = [root for root, _bracket in found]
            if len(found) == 1:
                root, bracket = found[0]
                out["implied"] = root
                out["implied_bracket"] = list(bracket)
                if solve_for == "discount_rate":
                    rate = root
                else:
                    growth = root
            elif not found:
                notes.append(
                    f"No {solve_for} in [{points[0]:.6g}, {points[-1]:.6g}] "
                    f"gives a value of {target:g}: the target is outside what "
                    "the forecast and the terminal assumption can reach."
                )
            else:
                notes.append(
                    f"{len(found)} values of {solve_for} give a value of "
                    f"{target:g} (implied_roots), so none of them is the "
                    "implied one; the forecast changes sign."
                )

    solved = rate is not None and (
        (method == "gordon" and growth is not None)
        or (method == "exit_multiple" and multiple is not None)
    )
    gordon = method == "gordon"
    out["discount_rate"] = nan if rate is None else rate
    out["terminal_growth"] = (growth if growth is not None else nan) if gordon else None
    out["exit_multiple"] = (
        None if gordon else (multiple if multiple is not None else nan)
    )
    if not solved:
        out.update(
            value=nan,
            pv_forecast=nan,
            terminal_value=nan,
            pv_terminal=nan,
            terminal_share=nan,
            pv_by_year=[],
            implied_exit_multiple=nan if gordon else None,
            implied_terminal_growth=None if gordon else nan,
            sensitivity=None,
        )
        return out

    assert rate is not None
    result = _value(flows, method, rate, growth, multiple, metric)
    out.update(result)
    last = float(flows[-1])
    terminal = result["terminal_value"]
    if gordon:
        out["implied_exit_multiple"] = terminal / last if last != 0.0 else nan
        out["implied_terminal_growth"] = None
    else:
        out["implied_exit_multiple"] = None
        out["implied_terminal_growth"] = (
            (terminal * rate - last) / (terminal + last)
            if terminal + last != 0.0
            else nan
        )
    share = result["terminal_share"]
    if math.isfinite(share) and share > 0.75:
        notes.append(
            f"The terminal value is {share:.0%} of the value, so the answer is "
            "mostly the terminal assumption; read the sensitivity grid."
        )
    if method == "gordon" and last <= 0.0:
        notes.append(
            "The last forecast flow is not positive, so the Gordon terminal "
            "value carries that loss forward forever."
        )
    out["sensitivity"] = _grid(
        flows,
        method,
        rate,
        growth,
        multiple,
        metric,
        sensitivity_rates,
        sensitivity_terminal,
    )
    return out


def _axis(
    given: Optional[Sequence[float]],
    default: List[float],
    name: str,
    valid: Callable[[float], bool],
    rule: str,
) -> List[float]:
    """The caller's grid axis, every value checked; or the default, kept
    to the values that are valid -- a default never refuses the call."""
    if given is None:
        return [v for v in default if math.isfinite(v) and valid(v)]
    chosen = [float(v) for v in given]
    if len(chosen) > MAX_GRID:
        raise ValidationError(
            f"{name} has {len(chosen)} values, more than the {MAX_GRID} a grid "
            "axis takes."
        )
    bad = [v for v in chosen if not (math.isfinite(v) and valid(v))]
    if bad:
        raise ValidationError(f"{name} has {bad[:5]}; {rule}.")
    return chosen


def _is_rate(value: float) -> bool:
    return -1.0 < value <= _RATE_CEILING


def _grid(
    flows, method, rate, growth, multiple, metric, rates_given, terminal_given
) -> Dict[str, Any]:
    rate_rule = f"a rate is a decimal above -1 and at most {_RATE_CEILING:g}"
    rates = _axis(
        rates_given,
        [rate + step for step in (-0.02, -0.01, 0.0, 0.01, 0.02)],
        "sensitivity_rates",
        _is_rate,
        rate_rule,
    )
    if method == "gordon":
        assert growth is not None
        terminal = _axis(
            terminal_given,
            [growth + step for step in (-0.01, -0.005, 0.0, 0.005, 0.01)],
            "sensitivity_terminal",
            _is_rate,
            rate_rule,
        )
    else:
        assert multiple is not None
        terminal = _axis(
            terminal_given,
            [multiple * k for k in (0.8, 0.9, 1.0, 1.1, 1.2)],
            "sensitivity_terminal",
            lambda k: 0.0 < k <= _MAX_MULTIPLE,
            f"a multiple is in (0, {_MAX_MULTIPLE:g}]",
        )
    values: List[List[float]] = []
    for r in rates:
        row = []
        for x in terminal:
            if method == "gordon":
                cell = (
                    _value_only(flows, method, r, x, None, metric)
                    if x < r
                    else float("nan")
                )
            else:
                cell = _value_only(flows, method, r, None, x, metric)
            row.append(cell)
        values.append(row)
    return {
        "discount_rates": rates,
        "terminal": terminal,
        "terminal_axis": "terminal_growth" if method == "gordon" else "exit_multiple",
        "values": values,
    }
