"""
Where a portfolio's return over its benchmark came from: Brinson-Fachler
allocation, selection and interaction, per segment and per period, linked
over the periods so they add up to the cumulative active return.

ONE PERIOD (Brinson, Hood & Beebower 1986; Brinson & Fachler 1985). With
portfolio and benchmark weights w_p, w_b and returns r_p, r_b per segment,
R_p = sum w_p r_p and R_b = sum w_b r_b, and for each segment

    allocation   (w_p - w_b) (r_b - R_b)
    selection    w_b (r_p - r_b)
    interaction  (w_p - w_b) (r_p - r_b)

which sum over the segments to R_p - R_b exactly. Fachler's form measures
allocation against the benchmark's total, so overweighting a segment that
beat the benchmark is rewarded and one that rose less than it is not;
because both weight sets sum to one, the R_b term adds nothing to the
total, only to the split across segments.

MANY PERIODS. Single-period effects do not add up to the cumulative active
return, prod(1 + R_p) - prod(1 + R_b), because returns compound. Each
period's effects are scaled by a coefficient that makes them do so:

    Carino (1999)    k_t / K, k_t = ln((1 + R_p,t) / (1 + R_b,t)) /
                     (R_p,t - R_b,t) and K the same of the cumulative
                     returns (1 / (1 + R) where the two are equal)
    Menchero (2000)  M + a_t, M = ((R_P - R_B) / T) / ((1 + R_P)^(1/T) -
                     (1 + R_B)^(1/T)) and a_t the least-squares correction
                     proportional to period t's active return

and either sums to R_P - R_B to rounding, period by period and segment by
segment. Carino's scales a period by its own log-to-arithmetic ratio;
Menchero's keeps the coefficients as close to one constant as the
identity allows.

WHAT IS FILLED AND WHAT IS NOT. A segment missing from a weight map holds
nothing there (weight 0). A segment the portfolio holds none of needs no
portfolio return: its selection and interaction are zero, so the
benchmark's is used, and the result says so. Every other return is
required -- a benchmark return for every segment either side holds, a
portfolio return wherever the portfolio holds any -- because a missing one
has no neutral value: zero is a return.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

from standard_quant_tools.error import ValidationError

__all__ = ["LINKINGS", "WEIGHT_TOLERANCE", "brinson_fachler"]

_WHO = "brinson_fachler"

LINKINGS = ("carino", "menchero")

#: How far a period's weights may sum from one.
WEIGHT_TOLERANCE = 1e-6


def _log_ratio_coefficient(ours: float, theirs: float) -> float:
    """ln((1 + ours) / (1 + theirs)) / (ours - theirs), stably, and its
    limit 1 / (1 + theirs) when the two are equal."""
    x = (ours - theirs) / (1.0 + theirs)
    if x == 0.0:
        return 1.0 / (1.0 + theirs)
    return math.log1p(x) / (x * (1.0 + theirs))


def _linking(
    portfolio: Sequence[float], benchmark: Sequence[float], method: str
) -> List[float]:
    """Each period's coefficient, so that sum_t c_t (R_p,t - R_b,t) is the
    cumulative active return."""
    total_p = math.prod(1.0 + r for r in portfolio) - 1.0
    total_b = math.prod(1.0 + r for r in benchmark) - 1.0
    if method == "carino":
        whole = _log_ratio_coefficient(total_p, total_b)
        return [
            _log_ratio_coefficient(p, b) / whole for p, b in zip(portfolio, benchmark)
        ]
    periods = len(portfolio)
    active = [p - b for p, b in zip(portfolio, benchmark)]
    if total_p == total_b:
        m = (1.0 + total_p) ** ((periods - 1) / periods)
    else:
        m = ((total_p - total_b) / periods) / (
            (1.0 + total_p) ** (1.0 / periods) - (1.0 + total_b) ** (1.0 / periods)
        )
    squares = sum(a * a for a in active)
    if squares == 0.0:
        return [m] * periods
    scale = ((total_p - total_b) - m * sum(active)) / squares
    return [m + scale * a for a in active]


def _check_weights(weights: Mapping[str, float], field: str, period: Any) -> None:
    total = math.fsum(weights.values())
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ValidationError(
            f"{_WHO}: {field} for period {period} sum to {total:.10g}, not 1. "
            "Brinson's effects add up to the active return only when each "
            "side's weights sum to one; put the rest in a cash segment (with "
            "its return) rather than leaving it out."
        )


def brinson_fachler(
    periods: Sequence[Any],
    portfolio_weights: Mapping[Any, Mapping[str, float]],
    benchmark_weights: Mapping[Any, Mapping[str, float]],
    portfolio_returns: Mapping[Any, Mapping[str, float]],
    benchmark_returns: Mapping[Any, Mapping[str, float]],
    *,
    linking: str = "carino",
) -> Dict[str, Any]:
    """
    Brinson-Fachler effects per segment and period, linked over `periods`
    (in that order) by `linking`. Each map is {period: {segment: value}},
    returns as decimals per period.

    Raises:
        ValidationError: a period one map lacks, weights that do not sum to
        one, a return a held segment needs and does not have, a total
        return at or below -100%, or an unknown linking.
    """
    if linking not in LINKINGS:
        raise ValidationError(
            f"{_WHO}: linking={linking!r}; expected one of {list(LINKINGS)}."
        )
    maps = {
        "portfolio_weights": portfolio_weights,
        "benchmark_weights": benchmark_weights,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
    }
    wanted = list(periods)
    if not wanted:
        raise ValidationError(f"{_WHO}: no periods to attribute.")
    for field, mapping in maps.items():
        missing = [p for p in wanted if p not in mapping]
        extra = [p for p in mapping if p not in set(wanted)]
        if missing or extra:
            raise ValidationError(
                f"{_WHO}: {field} must cover exactly the same periods as the "
                f"others; it lacks {missing[:5]} and has {extra[:5]} besides."
            )

    notes: List[str] = []
    filled: List[str] = []
    rows: List[Dict[str, Any]] = []
    for period in wanted:
        wp = portfolio_weights[period]
        wb = benchmark_weights[period]
        rp = portfolio_returns[period]
        rb = benchmark_returns[period]
        _check_weights(wp, "portfolio_weights", period)
        _check_weights(wb, "benchmark_weights", period)
        segments = list(dict.fromkeys([*wb, *wp]))
        held = [s for s in segments if wp.get(s, 0.0) != 0.0 or wb.get(s, 0.0) != 0.0]
        lacking_b = [s for s in held if s not in rb]
        lacking_p = [s for s in held if wp.get(s, 0.0) != 0.0 and s not in rp]
        if lacking_b or lacking_p:
            parts = []
            if lacking_b:
                parts.append(f"benchmark_returns lacks {lacking_b[:5]}")
            if lacking_p:
                parts.append(f"portfolio_returns lacks {lacking_p[:5]}")
            raise ValidationError(
                f"{_WHO}: period {period}: {' and '.join(parts)}, held there. "
                "A missing return has no neutral value -- zero is a return -- "
                "so give the segment's return for the period."
            )
        r_p_seg = {
            s: float(rp[s]) if s in rp and wp.get(s, 0.0) != 0.0 else float(rb[s])
            for s in held
        }
        filled.extend(f"{period}:{s}" for s in held if wp.get(s, 0.0) == 0.0)
        total_b = math.fsum(float(wb.get(s, 0.0)) * float(rb[s]) for s in held)
        total_p = math.fsum(float(wp.get(s, 0.0)) * r_p_seg[s] for s in held)
        for side, value in (("portfolio", total_p), ("benchmark", total_b)):
            if not value > -1.0:
                raise ValidationError(
                    f"{_WHO}: period {period}: the {side} returned {value:.6g}, "
                    "a loss of everything or more, which no linking can compound."
                )
        effects: Dict[str, Dict[str, float]] = {}
        for s in held:
            active_weight = float(wp.get(s, 0.0)) - float(wb.get(s, 0.0))
            excess = r_p_seg[s] - float(rb[s])
            # `+ 0.0` turns a product's -0.0 into 0.0 and changes no other
            # number: a segment with no effect reads as zero, unsigned.
            effects[s] = {
                "allocation": active_weight * (float(rb[s]) - total_b) + 0.0,
                "selection": float(wb.get(s, 0.0)) * excess + 0.0,
                "interaction": active_weight * excess + 0.0,
            }
        rows.append(
            {
                "period": period,
                "portfolio_return": total_p,
                "benchmark_return": total_b,
                "active_return": total_p - total_b,
                "segments": effects,
            }
        )
    if filled:
        notes.append(
            f"{_WHO}: the portfolio held none of {len(filled)} segment-period(s) "
            f"(e.g. {filled[:3]}); the benchmark's return stands in for its "
            "own there, so their selection and interaction are zero."
        )

    coefficients = _linking(
        [row["portfolio_return"] for row in rows],
        [row["benchmark_return"] for row in rows],
        linking,
    )
    linked: Dict[str, Dict[str, float]] = {}
    for row, c in zip(rows, coefficients):
        row["linking_coefficient"] = c
        for effect in ("allocation", "selection", "interaction"):
            row[effect] = math.fsum(e[effect] for e in row["segments"].values())
        for s, e in row["segments"].items():
            e["total"] = e["allocation"] + e["selection"] + e["interaction"]
            into = linked.setdefault(
                s, {"allocation": 0.0, "selection": 0.0, "interaction": 0.0}
            )
            for effect in ("allocation", "selection", "interaction"):
                into[effect] += c * e[effect]
    for e in linked.values():
        e["total"] = e["allocation"] + e["selection"] + e["interaction"]

    cumulative_p = math.prod(1.0 + row["portfolio_return"] for row in rows) - 1.0
    cumulative_b = math.prod(1.0 + row["benchmark_return"] for row in rows) - 1.0
    totals = {
        effect: math.fsum(e[effect] for e in linked.values())
        for effect in ("allocation", "selection", "interaction")
    }
    return {
        "linking": linking,
        "n_periods": len(rows),
        "n_segments": len(linked),
        "portfolio_return": cumulative_p,
        "benchmark_return": cumulative_b,
        "active_return": cumulative_p - cumulative_b,
        "allocation": totals["allocation"],
        "selection": totals["selection"],
        "interaction": totals["interaction"],
        "sum_of_effects": math.fsum(totals.values()),
        "by_segment": linked,
        "by_period": rows,
        "warnings": notes,
    }
