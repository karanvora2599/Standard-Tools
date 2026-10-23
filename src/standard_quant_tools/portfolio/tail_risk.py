"""
Which positions a portfolio's tail comes from: Euler component expected
shortfall and kernel-smoothed component VaR, from a scenario matrix.

THE INPUT is a matrix of scenarios x assets -- historical returns on their
dates, simulated returns, or each position's P&L under a full revaluation
-- and a weight per asset. The portfolio's P&L in scenario s is
P_s = sum_i w_i X_si: a return when X holds returns and the weights are
fractions of the book, currency when X holds P&L per unit and the weights
are quantities (1.0 for a column that is already one position's P&L).

VaR AND ES ARE LOSSES, POSITIVE, exactly as `var_historical` and `cvar`
report them: VaR is minus the empirical (1 - confidence) quantile of P
(numpy's linear interpolation), ES minus the mean of the scenarios at or
below it. On the same P the two numbers here are those functions' bits.

COMPONENT ES IS EXACT (Tasche 1999). ES is the mean loss over the tail set
T = {s : P_s <= q}, and scaling every weight by a positive factor scales q
and leaves T unchanged, so ES is homogeneous of degree one and Euler's
theorem splits it by asset:

    ES = sum_i w_i (-mean_{s in T} X_si) = sum_i component_i,

the marginal of asset i being minus its mean in the portfolio's tail. The
sum is an identity: it holds to rounding on any input. A negative component
is an asset that GAINS, on average, when the portfolio is in its tail -- a
hedge, whether or not it was meant as one.

COMPONENT VaR IS SMOOTHED (Epperlein and Smillie 2006). VaR's marginal is
E[-X_i | P = q], a conditional expectation on a set of probability zero:
the one scenario sitting at the quantile is a sample of size one. A
Gaussian kernel weights every scenario by its distance from q,

    marginal_i = -sum_s K((P_s - q)/h) X_si / sum_s K((P_s - q)/h),

and the components w_i marginal_i sum, exactly, to the kernel-smoothed VaR
-sum_s K P_s / sum_s K -- reported as `var_kernel` beside the empirical
`var`, so the gap between them (the smoothing's bias, O(h^2)) is visible
rather than rescaled away. The bandwidth h is Silverman's rule of thumb on
the portfolio's P&L, 0.9 min(sd, IQR/1.34) n^(-1/5) (the standard
deviation alone when the IQR is zero), unless the caller gives one; it is
reported either way, with the kernel's effective number of scenarios
(sum K)^2 / sum K^2.

THE TAIL COUNT IS ALWAYS REPORTED. ES at 99% on 250 scenarios is the mean of
three, and every component with it; below `CVAR_MIN_TAIL_OBSERVATIONS` the
result carries the warning every other expected shortfall in the library
does.

Under a multivariate normal with zero mean both converge to the closed
form: component_i = w_i (Sigma w)_i / sigma_p times z for VaR and
phi(z) / (1 - confidence) for ES.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import (
    _check_confidence,
    _cvar_tail,
    cvar_tail_warning,
    has_no_dispersion,
)
from standard_quant_tools.portfolio.construction import _finite_weights

__all__ = ["MIN_SCENARIOS", "decompose_tail_risk", "silverman_bandwidth"]

#: Fewer scenarios than this is not a distribution: at any confidence of
#: 90% or more the tail is the single worst one.
MIN_SCENARIOS = 10

#: Below this many effective scenarios near the VaR, component VaR is an
#: average of a handful of points, and the result says so.
MIN_KERNEL_SCENARIOS = 10

#: How many offending names a refusal lists before counting the rest.
_SHOWN = 5


def _listed(items: Any) -> str:
    items = list(items)
    shown = ", ".join(repr(i) for i in items[:_SHOWN])
    more = len(items) - _SHOWN
    return f"[{shown}{f' and {more} more' if more > 0 else ''}]"


def silverman_bandwidth(values: np.ndarray) -> float:
    """
    Silverman's rule of thumb, 0.9 min(sd, IQR/1.34) n^(-1/5), for a
    Gaussian kernel on `values` (Silverman 1986, eq. 3.31). The standard
    deviation alone when the interquartile range is zero, so a sample whose
    middle half is one value still gets a positive bandwidth.
    """
    array = np.asarray(values, dtype=np.float64)
    sd = float(array.std(ddof=1))
    upper, lower = np.percentile(array, [75.0, 25.0])
    iqr = float(upper - lower)
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    return 0.9 * spread * array.size ** (-0.2)


def _scenario_matrix(scenarios: pd.DataFrame, order: List[str], who: str) -> np.ndarray:
    """The scenarios as a C-ordered float64 matrix with the columns in
    `order`, or a refusal naming what is wrong with them."""
    if not isinstance(scenarios, pd.DataFrame):
        raise ValidationError(
            f"{who}: scenarios must be a frame of scenarios (rows) by assets "
            f"(columns), got a {type(scenarios).__name__}."
        )
    columns = [str(c) for c in scenarios.columns]
    if len(set(columns)) != len(columns):
        repeated = sorted({c for c in columns if columns.count(c) > 1})
        raise ValidationError(
            f"{who}: the scenarios name the asset(s) {_listed(repeated)} more "
            "than once; each asset is one column."
        )
    known = set(columns)
    named = set(order)
    absent = [a for a in order if a not in known]
    unweighted = [c for c in columns if c not in named]
    if absent or unweighted:
        parts = []
        if absent:
            parts.append(f"weights name {_listed(absent)}, which have no scenarios")
        if unweighted:
            parts.append(
                f"the scenarios have {_listed(unweighted)}, which have no weight"
            )
        raise ValidationError(
            f"{who}: the weights and the scenarios must name the same assets; "
            f"{' and '.join(parts)}. Give every asset a weight (0.0 to hold "
            "none of it) and a scenario column -- a position without "
            "scenarios cannot be in the tail, and one left out would leave the "
            "tail of a different portfolio."
        )
    if len(scenarios) < MIN_SCENARIOS:
        raise ValidationError(
            f"{who}: {len(scenarios)} scenario(s) is too few for a tail; give at "
            f"least {MIN_SCENARIOS}. At any confidence of 90% or more fewer "
            "than ten leave a tail of the single worst one."
        )
    frame = scenarios.copy()
    frame.columns = columns
    try:
        matrix = np.ascontiguousarray(frame[order].to_numpy(dtype=np.float64))
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{who}: the scenarios hold values that are not numbers ({exc})."
        ) from exc
    bad = ~np.isfinite(matrix)
    if bad.any():
        rows, cols = np.nonzero(bad)
        where = [f"{order[c]} in scenario {r}" for r, c in zip(rows[:_SHOWN], cols)]
        raise ValidationError(
            f"{who}: {int(bad.sum())} scenario value(s) are missing or not "
            f"finite: {where}. A scenario needs a value for every asset; drop "
            "the scenario or supply the value."
        )
    return matrix


def decompose_tail_risk(
    scenarios: pd.DataFrame,
    weights: Mapping[str, float],
    *,
    confidence: float = 0.95,
    bandwidth: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Portfolio VaR and ES from a scenario matrix, each split by asset.

    Args:
        scenarios: scenarios (rows) x assets (columns): returns or P&L per
            unit. Its columns must be exactly the weights' assets.
        weights: asset -> weight (a fraction of the book on returns, a
            quantity on P&L).
        confidence: the VaR and ES confidence, strictly between 0 and 1.
        bandwidth: the kernel bandwidth for component VaR, in the units of
            the portfolio's P&L; None for Silverman's rule on it.

    Returns a dict: `var`, `es` (losses, positive, as `var_historical` and
    `cvar` give them), `var_kernel` (what the component VaRs sum to),
    `n_tail_observations`, the bandwidth and its rule, the kernel's
    effective scenario count, the sums of the components, `by_asset` (per
    asset: weight, marginal and component ES and VaR and their shares),
    `n_hedges`, counts and `warnings`. A share whose total is zero is NaN.
    """
    who = "decompose_tail_risk"
    _check_confidence(confidence)
    series = _finite_weights(weights, "weights", who)
    series.index = [str(a) for a in series.index]
    order = list(series.index)
    if not order:
        raise ValidationError(f"{who}: weights names no asset.")
    if bandwidth is not None and not (
        isinstance(bandwidth, (int, float))
        and math.isfinite(bandwidth)
        and bandwidth > 0
    ):
        raise ValidationError(
            f"{who}: bandwidth must be a finite positive number in the units of "
            f"the portfolio's P&L, got {bandwidth!r}. Omit it for Silverman's "
            "rule."
        )
    X = _scenario_matrix(scenarios, order, who)
    w = series.to_numpy(dtype=np.float64)
    n, m = X.shape

    # Summed asset by asset in the weights' order, with no BLAS product: the
    # portfolio's P&L is then one definite sum, the same bits whatever the
    # matrix's memory layout or the caller's threads, and a caller holding
    # the same columns can form the same P to hand to `cvar`.
    P = np.zeros(n, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(m):
            P += w[i] * X[:, i]
    if not np.all(np.isfinite(P)):
        raise ValidationError(
            f"{who}: the portfolio's P&L is beyond the float range in some "
            "scenario. Give weights as fractions of the book or quantities, "
            "and scenarios in return or currency units."
        )
    if has_no_dispersion(P):
        raise ValidationError(
            f"{who}: the portfolio's P&L is the same in every scenario, so it "
            "has no tail to decompose. Check the weights (all zero?) and the "
            "scenarios."
        )

    # The library's own estimator: the same threshold and the same tail as
    # `cvar` and `var_historical` take on this P&L, so these are their bits.
    threshold, tail = _cvar_tail(pd.Series(P), confidence)
    mask = P <= threshold
    n_tail = int(mask.sum())
    var = float(-threshold)
    es = float(-tail.mean())

    if bandwidth is None:
        h = silverman_bandwidth(P)
        rule = "silverman"
    else:
        h = float(bandwidth)
        rule = "given"
    u = (P - threshold) / h
    kernel = np.exp(-0.5 * u * u)
    kernel_total = float(kernel.sum())
    if not kernel_total > 0:
        raise ValidationError(
            f"{who}: at bandwidth {h:.6g} no scenario is within reach of the "
            f"VaR ({var:.6g}): every kernel weight underflowed to zero. Give a "
            "wider bandwidth, in the units of the portfolio's P&L, or omit it "
            "for Silverman's rule."
        )

    with single_threaded_blas():
        tail_mean = X[mask].mean(axis=0)
        kernel_mean = (kernel @ X) / kernel_total
        kernel_pnl = float(kernel @ P) / kernel_total
        effective = kernel_total**2 / float(kernel @ kernel)

    marginal_es = -tail_mean
    component_es = w * marginal_es
    marginal_var = -kernel_mean
    component_var = w * marginal_var
    var_kernel = -kernel_pnl

    def share(part: float, whole: float) -> float:
        return float(part / whole) if whole != 0 else float("nan")

    rows = [
        {
            "asset": name,
            "weight": float(w[i]),
            "marginal_es": float(marginal_es[i]),
            "component_es": float(component_es[i]),
            "es_share": share(float(component_es[i]), es),
            "marginal_var": float(marginal_var[i]),
            "component_var": float(component_var[i]),
            "var_share": share(float(component_var[i]), var_kernel),
        }
        for i, name in enumerate(order)
    ]
    rows.sort(key=lambda r: r["component_es"], reverse=True)
    hedges = [r["asset"] for r in rows if r["component_es"] < 0]

    warnings: List[str] = []
    caution = cvar_tail_warning("es", n_tail, n, confidence)
    if caution:
        warnings.append(
            f"{caution} Every component_es is an average over the same "
            f"{n_tail} scenario(s)."
        )
    if effective < MIN_KERNEL_SCENARIOS:
        warnings.append(
            f"Component VaR rests on about {effective:.1f} effective scenarios "
            f"near the VaR (bandwidth {h:.6g}, rule {rule!r}), so each "
            "component is an average of a handful of points. More scenarios "
            "or a wider bandwidth give it more."
        )
    if not es > 0:
        warnings.append(
            f"The expected shortfall is {es:.6g}: the portfolio's tail is not a "
            "loss, so a share of it is a share of a gain (or of zero) and reads "
            "with the sign flipped."
        )
    if hedges:
        warnings.append(
            f"{len(hedges)} position(s) have a NEGATIVE component ES "
            f"({hedges[:_SHOWN]}): on average they gain when the portfolio is "
            "in its tail, so they are hedges whether or not they were meant as "
            "one."
        )
    return {
        "n_scenarios": int(n),
        "n_assets": int(m),
        "confidence": float(confidence),
        "var": var,
        "es": es,
        "var_kernel": var_kernel,
        "n_tail_observations": n_tail,
        "bandwidth": h,
        "bandwidth_rule": rule,
        "kernel_effective_scenarios": float(effective),
        "sum_of_es_components": math.fsum(float(c) for c in component_es),
        "sum_of_var_components": math.fsum(float(c) for c in component_var),
        "by_asset": rows,
        "n_hedges": len(hedges),
        "warnings": warnings,
    }
