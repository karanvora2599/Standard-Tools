"""
Capital across return streams, and how much of the answer the sample chose.

A SLEEVE is anything with a return stream -- a strategy, a model, a manager,
a fund -- and the question is what share of the capital each one gets. The
assets inside a sleeve are its own business; nothing here looks through it.
The streams arrive already on one calendar (`portfolio.streams.
align_return_streams` puts them there), one column per sleeve.

THE RULES, each the library's own where one exists:

- `equal` -- 1/N.
- `inverse_volatility` -- w_i proportional to 1/sigma_i, from the
  covariance's diagonal.
- `risk_parity` -- equal risk contributions, or the caller's risk budgets
  (`portfolio.construction.risk_parity`).
- `hrp` -- hierarchical risk parity on the returns themselves
  (`portfolio.construction.hierarchical_risk_parity`), whose tree is built
  from their sample correlation whatever covariance the risk is reported on.
- `min_variance` and `max_sharpe` -- the exact solves of
  `portfolio.general_optimizer.optimize_portfolio` on the covariance and the
  sleeves' mean returns, against `cash_rate`.
- `shrunk_sharpe` -- the same tangency, on means whose Sharpe ratios are
  shrunk toward their pooled value first (see `shrunk_sharpe_ratios`).

The covariance is `portfolio.covariance.estimate_covariance` with the
caller's method. Every rule's risk -- contributions, ex-ante volatility --
is read off that one matrix, so two rules are compared on the same risk
model.

CONSTRAINTS, in the order they act. `max_weight` caps the rule's fully
invested weights: the optimizers take it as a bound, the others are capped
and the excess water-filled to the uncapped sleeves pro rata. A
`target_volatility` then scales the gross, refused when it would need more
gross than `max_leverage`; what is not invested is cash, and a gross above
one borrows at the cash rate. `capacity` (currency per sleeve, against
`capital`) caps last: the excess goes to cash, or is water-filled to the
sleeves with room.

WHAT THE SAMPLE CHOSE. A block bootstrap refits the whole rule on circular
block resamples of the dated rows, every sleeve's return on a date moved
together, and reports each weight's 5th and 95th percentiles and the mean
L1 distance of the resampled weights from the point weights. The draws are
widened about their mean by `_resampling.circular_block_scale`, as the
library's other block-bootstrap intervals are. A max-Sharpe allocation from
a few years of data moves most of its capital between resamples; that is
the number to read before the weights.

THE IN-SAMPLE SHARPE IS IN-SAMPLE: the rule chose its weights on the very
returns it is scored on. `walk_forward.evaluate_allocation_rule` is the
test of a rule; this is the allocation.

The linear algebra runs on one BLAS thread (see `_blas`), so the same input
gives the same bits on any machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools._resampling import (
    circular_block_indices,
    circular_block_scale,
)
from standard_quant_tools.backtest.robustness import sharpe_standard_error_factor
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import (
    annualized_sharpe,
    has_no_dispersion,
)
from standard_quant_tools.numeric_contract import (
    require_finite_scalar,
    require_periods_per_year,
)
from standard_quant_tools.portfolio import construction
from standard_quant_tools.portfolio.covariance import METHODS as COVARIANCE_METHODS
from standard_quant_tools.portfolio.covariance import estimate_covariance
from standard_quant_tools.portfolio.general_optimizer import optimize_portfolio

__all__ = [
    "CAPACITY_EXCESS",
    "COVARIANCE_METHODS",
    "METHODS",
    "MIN_SHRUNK_SLEEVES",
    "SleeveRule",
    "allocate_across_sleeves",
    "fit_sleeve_rule",
    "shrunk_sharpe_ratios",
    "water_fill",
]

METHODS = (
    "equal",
    "inverse_volatility",
    "risk_parity",
    "hrp",
    "min_variance",
    "max_sharpe",
    "shrunk_sharpe",
)

#: The rules solved by the optimizer, the only ones that can hold a short.
OPTIMIZED = ("min_variance", "max_sharpe", "shrunk_sharpe")

#: Where capital above a sleeve's capacity goes.
CAPACITY_EXCESS = ("cash", "redistribute")

#: James-Stein toward a mean estimated from the same sleeves improves on the
#: raw estimates only from four of them: its shrinkage is (k - 3) / Q.
MIN_SHRUNK_SLEEVES = 4

#: A weight within this share of its cap is at the cap.
_AT_CAP = 1e-12

#: A target volatility may need this much more gross than max_leverage
#: before it is refused: the scale is a ratio of two rounded numbers.
_LEVERAGE_SLACK = 1e-12

#: The bootstrap reports its bands only when at least this share of its
#: resamples could be fitted.
_MIN_FITTED_SHARE = 0.5


@dataclass(frozen=True)
class SleeveRule:
    """
    An allocation rule: the method, its covariance, and its constraints.

    `risk_budgets` (sleeve -> positive risk share, normalized) is read by
    `risk_parity` only. `long_only=False` lets the three optimized rules
    short a sleeve; the others are long by construction. `max_weight` is the
    largest fully invested weight (and, with shorts, the largest short).
    `target_volatility` (annual, in the covariance's units) scales the
    gross, up to `max_leverage`. `capacity` is sleeve -> the most capital it
    takes, in the currency of `capital`; `capacity_excess` sends what it
    cuts to "cash" or "redistribute"s it to the sleeves with room.
    """

    method: str = "risk_parity"
    covariance_method: str = "ledoit_wolf"
    halflife: float = 60.0
    risk_budgets: Optional[Mapping[str, float]] = None
    long_only: bool = True
    max_weight: Optional[float] = None
    target_volatility: Optional[float] = None
    max_leverage: float = 1.0
    capital: Optional[float] = None
    capacity: Optional[Mapping[str, float]] = None
    capacity_excess: str = "cash"

    def __post_init__(self) -> None:
        who = "allocation rule"
        if self.method not in METHODS:
            raise ValidationError(
                f"{who}: method={self.method!r} is not one of {list(METHODS)}."
            )
        if self.covariance_method not in COVARIANCE_METHODS:
            raise ValidationError(
                f"{who}: covariance_method={self.covariance_method!r} is not one "
                f"of {list(COVARIANCE_METHODS)}."
            )
        require_finite_scalar(self.halflife, "halflife", who)
        if not self.halflife > 0:
            raise ValidationError(
                f"{who}: halflife={self.halflife!r} must be a positive number of "
                "periods; 60 is the default."
            )
        if self.risk_budgets is not None:
            if self.method != "risk_parity":
                raise ValidationError(
                    f"{who}: risk_budgets are read by method='risk_parity' only, "
                    f"and method={self.method!r}. Drop risk_budgets, or use "
                    "risk_parity."
                )
            bad = {
                str(k): v
                for k, v in self.risk_budgets.items()
                if not (math.isfinite(float(v)) and float(v) > 0)
            }
            if bad:
                raise ValidationError(
                    f"{who}: every risk budget must be a finite positive share, "
                    f"got {bad}. A zero budget is an exclusion: leave the sleeve "
                    "out of the streams instead."
                )
        if not self.long_only and self.method not in OPTIMIZED:
            raise ValidationError(
                f"{who}: long_only=False lets an optimized rule short a sleeve, "
                f"and method={self.method!r} is long by construction. Use one "
                f"of {list(OPTIMIZED)}, or keep long_only=True."
            )
        if self.max_weight is not None:
            require_finite_scalar(self.max_weight, "max_weight", who)
            if not 0.0 < self.max_weight <= 1.0:
                raise ValidationError(
                    f"{who}: max_weight={self.max_weight!r} must be in (0, 1]: "
                    "it is the largest share of the capital one sleeve takes."
                )
        if self.target_volatility is not None:
            require_finite_scalar(self.target_volatility, "target_volatility", who)
            if not self.target_volatility > 0:
                raise ValidationError(
                    f"{who}: target_volatility={self.target_volatility!r} must "
                    "be positive (annual, e.g. 0.10)."
                )
        require_finite_scalar(self.max_leverage, "max_leverage", who)
        if not self.max_leverage > 0:
            raise ValidationError(
                f"{who}: max_leverage={self.max_leverage!r} must be positive; "
                "1.0 is the capital itself."
            )
        if self.capital is not None:
            require_finite_scalar(self.capital, "capital", who)
            if not self.capital > 0:
                raise ValidationError(
                    f"{who}: capital={self.capital!r} must be a positive amount."
                )
        if self.capacity is not None:
            if self.capital is None:
                raise ValidationError(
                    f"{who}: capacity is in currency and needs the capital it "
                    "is a share of. Give capital, e.g. capital=10_000_000."
                )
            if not self.long_only:
                raise ValidationError(
                    f"{who}: capacity caps a long allocation; with "
                    "long_only=False cap the sleeves through max_weight."
                )
            bad = {
                str(k): v
                for k, v in self.capacity.items()
                if not (math.isfinite(float(v)) and float(v) > 0)
            }
            if bad:
                raise ValidationError(
                    f"{who}: every capacity must be a finite positive amount, got "
                    f"{bad}. A sleeve that takes nothing is left out of the "
                    "streams, not given a capacity of zero."
                )
        if self.capacity_excess not in CAPACITY_EXCESS:
            raise ValidationError(
                f"{who}: capacity_excess={self.capacity_excess!r} is not one of "
                f"{list(CAPACITY_EXCESS)}."
            )


# ── pieces ───────────────────────────────────────────────────────────────


def water_fill(
    weights: np.ndarray, caps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Long weights with each at most its cap, the excess handed to the
    sleeves still under theirs, pro rata to their weights.

    Each pass caps every sleeve above its cap and spreads what was cut over
    the uncapped sleeves holding a positive weight, so their proportions to
    one another are kept; a pass caps at least one more sleeve, so it ends
    within N passes. Returns (weights, capped, left over): `capped` marks
    the sleeves held at their cap, and what no sleeve had room for is left
    over, which the caller puts in cash. The weights' sum is kept to
    rounding whenever nothing is left over.
    """
    w = np.array(weights, dtype=float)
    caps = np.asarray(caps, dtype=float)
    start = float(w.sum())
    capped = np.zeros(w.size, dtype=bool)
    for _ in range(w.size + 1):
        over = w > caps * (1.0 + _AT_CAP)
        if not over.any():
            break
        excess = float(np.sum(w[over] - caps[over]))
        w[over] = caps[over]
        capped |= over
        room = (~capped) & (w > 0.0)
        if not room.any():
            break
        w[room] += excess * w[room] / float(w[room].sum())
    capped |= w >= caps * (1.0 - _AT_CAP)
    capped &= np.isfinite(caps)
    return w, capped, max(start - float(w.sum()), 0.0)


def _covariance_frame(
    frame: pd.DataFrame, rule: SleeveRule, periods_per_year: int
) -> Tuple[pd.DataFrame, List[str]]:
    """The annual covariance of the sleeves, by name, and its warnings."""
    out = estimate_covariance(
        frame,
        method=rule.covariance_method,
        halflife=rule.halflife,
        periods_per_year=periods_per_year,
    )
    names = [str(c) for c in frame.columns]
    matrix = out["matrix"]
    cov = pd.DataFrame(
        [[float(matrix[a][b]) for b in names] for a in names],
        index=names,
        columns=names,
    )
    return cov, list(out.get("warnings", []))


def shrunk_sharpe_ratios(
    values: np.ndarray,
    correlation: np.ndarray,
    *,
    rf_per_period: float,
    periods_per_year: int,
    what: str = "shrunk_sharpe",
) -> Dict[str, Any]:
    """
    James-Stein shrinkage of the sleeves' Sharpe ratios toward their pooled
    value, on the Lo/Mertens standard errors.

    Each sleeve's per-period Sharpe ratio SR_i of its return over the cash
    rate has the standard error sqrt(1 - skew SR + (kurtosis - 1)/4 SR^2) /
    sqrt(T - 1) (Lo 2002; Mertens 2002; `sharpe_standard_error_factor`).
    The estimates are correlated as the sleeves are, so their covariance is
    taken as V = D R D, D the standard errors and R the returns' correlation
    -- the first-order covariance of two Sharpe estimates. The pooled value
    is the precision-weighted mean m = 1'V^-1 SR / 1'V^-1 1, and with
    Q = (SR - m)'V^-1 (SR - m) the positive-part James-Stein estimate is

        SR_i* = m + (1 - c)(SR_i - m),   c = min(1, (k - 3) / Q),

    k the number of sleeves (Efron & Morris 1973). When the Sharpe ratios
    are equal, Q is a chi-squared on k - 1 degrees of freedom at any sample
    length: c is 1 -- full pooling -- with probability P(chi2_{k-1} <= k - 3),
    and its mean rises toward 1 as k grows. When they differ, Q grows with
    the sample and c falls toward 0.

    Returns the Sharpe ratios, their standard errors and the shrunk ones
    (all annualized), `shrinkage` (c), `pooled_sharpe` (m, annualized) and
    the dispersion statistic Q. Refused below `MIN_SHRUNK_SLEEVES` sleeves,
    for a sleeve with no dispersion, and when V is singular.
    """
    T, k = values.shape
    if k < MIN_SHRUNK_SLEEVES:
        raise ValidationError(
            f"{what}: {k} sleeves, and James-Stein shrinkage toward the sleeves' "
            f"own pooled Sharpe ratio needs at least {MIN_SHRUNK_SLEEVES}: its "
            "shrinkage is (k - 3) / Q, which is not positive below that. Use "
            "max_sharpe, or a rule that reads no means."
        )
    excess = values - rf_per_period
    mean = excess.mean(axis=0)
    std = excess.std(axis=0, ddof=1)
    flat = [i for i in range(k) if has_no_dispersion(excess[:, i], float(std[i]))]
    if flat:
        raise ValidationError(
            f"{what}: sleeve(s) at position {flat} have no dispersion, so their "
            "Sharpe ratios are undefined. Leave them out of the streams."
        )
    sr = mean / std
    centred = (excess - mean) / std
    skew = (centred**3).mean(axis=0)
    kurtosis = (centred**4).mean(axis=0)
    se = np.array(
        [
            sharpe_standard_error_factor(
                float(sr[i]), float(skew[i]), float(kurtosis[i])
            )
            for i in range(k)
        ]
    ) / math.sqrt(T - 1)
    V = se[:, None] * correlation * se[None, :]
    ones = np.ones(k)
    try:
        with single_threaded_blas():
            precision_ones = np.linalg.solve(V, ones)
            m = float(precision_ones @ sr) / float(precision_ones @ ones)
            d = sr - m
            Q = float(d @ np.linalg.solve(V, d))
    except np.linalg.LinAlgError as exc:
        raise ValidationError(
            f"{what}: the Sharpe estimates' covariance is singular ({exc}): two "
            "sleeves move as one. Drop one of them."
        ) from exc
    shrinkage = 1.0 if not Q > 0 else min(1.0, (k - 3) / Q)
    shrunk = m + (1.0 - shrinkage) * d
    root = math.sqrt(periods_per_year)
    return {
        "sharpe": sr * root,
        "standard_error": se * root,
        "shrunk_sharpe": shrunk * root,
        "shrinkage": float(shrinkage),
        "pooled_sharpe": float(m * root),
        "dispersion_statistic": Q,
    }


def _optimized(
    method: str,
    values: np.ndarray,
    cov: pd.DataFrame,
    rule: SleeveRule,
    *,
    cash_rate: float,
    periods_per_year: int,
    what: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """min_variance, max_sharpe or shrunk_sharpe through the exact solver."""
    names = list(cov.columns)
    cap = rule.max_weight
    lower = 0.0 if rule.long_only else (-cap if cap is not None else -math.inf)
    upper = cap if cap is not None else math.inf
    details: Dict[str, Any] = {}
    kwargs: Dict[str, Any] = {}
    if method != "min_variance":
        if method == "max_sharpe":
            mu = values.mean(axis=0) * periods_per_year
        else:
            matrix = cov.to_numpy()
            sd = np.sqrt(np.diag(matrix))
            shrink = shrunk_sharpe_ratios(
                values,
                matrix / np.outer(sd, sd),
                rf_per_period=cash_rate / periods_per_year,
                periods_per_year=periods_per_year,
                what=f"{what}: shrunk_sharpe",
            )
            details["shrinkage"] = {
                key: (
                    {n: float(v) for n, v in zip(names, value)}
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in shrink.items()
            }
            mu = cash_rate + shrink["shrunk_sharpe"] * sd
        if rule.long_only and not float(np.max(mu)) > cash_rate:
            raise ValidationError(
                f"{what}: method={method!r} needs a sleeve whose "
                f"{'mean' if method == 'max_sharpe' else 'shrunk mean'} return "
                f"is above cash_rate={cash_rate:g}, and none is over these "
                f"{values.shape[0]} periods (the highest is {float(np.max(mu)):.4g} "
                "a year). No long-only mix then has a positive Sharpe ratio. Use "
                "min_variance or risk_parity, which read no means."
            )
        kwargs = {
            "expected_returns": dict(zip(names, (float(x) for x in mu))),
            "risk_free_rate": cash_rate,
        }
    try:
        out = optimize_portfolio(
            cov,
            objective="min_variance" if method == "min_variance" else "max_sharpe",
            lower_bounds=lower,
            upper_bounds=upper,
            **kwargs,
        )
    except ValidationError as exc:
        raise ValidationError(f"{what}: method={method!r}: {exc}") from exc
    if not out["weights"]:
        raise ValidationError(
            f"{what}: method={method!r} reached no answer: {out['solver']}."
        )
    weights = np.array([float(out["weights"][n]) for n in names])
    notes = [] if out["certified"] else [f"{what}: {method}: {out['solver']}."]
    details["certified"] = bool(out["certified"])
    binding = [
        {"sleeve": str(b["name"]), "cap": "max_weight", "limit": float(b["bound"])}
        for b in out["binding_constraints"]
        if b["constraint"] in ("upper_bound", "lower_bound")
        and b["bound"] is not None
        and b["bound"] != 0.0
    ]
    return weights, binding, notes, details


def _rule_weights(
    frame: pd.DataFrame,
    values: np.ndarray,
    cov: pd.DataFrame,
    rule: SleeveRule,
    *,
    cash_rate: float,
    periods_per_year: int,
    what: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """The rule's fully invested weights, max_weight applied."""
    names = list(cov.columns)
    n = len(names)
    method = rule.method
    if rule.max_weight is not None and rule.max_weight * n < 1.0 - 1e-12:
        raise ValidationError(
            f"{what}: max_weight={rule.max_weight:g} across {n} sleeves holds at "
            f"most {rule.max_weight * n:.4g} of the capital, and the rule is "
            f"fully invested. Raise max_weight to at least {1.0 / n:.4g}."
        )
    if method in OPTIMIZED:
        return _optimized(
            method,
            values,
            cov,
            rule,
            cash_rate=cash_rate,
            periods_per_year=periods_per_year,
            what=what,
        )
    notes: List[str] = []
    details: Dict[str, Any] = {}
    if method == "equal":
        weights = np.full(n, 1.0 / n)
    elif method == "inverse_volatility":
        checked, notes = construction._covariance_frame_with_notes(cov, what)
        inverse = 1.0 / np.sqrt(np.diag(checked.to_numpy()))
        weights = inverse / inverse.sum()
    elif method == "risk_parity":
        budget = None
        if rule.risk_budgets is not None:
            given = {str(k) for k in rule.risk_budgets}
            missing = [s for s in names if s not in given]
            unknown = sorted(given - set(names))
            if missing or unknown:
                raise ValidationError(
                    f"{what}: risk_budgets must name every sleeve once; missing "
                    f"{missing[:5]}, not a sleeve {unknown[:5]}."
                )
            budget = [float(rule.risk_budgets[s]) for s in names]
        out = construction.risk_parity(cov, budget=budget)
        if not out["converged"]:
            raise ValidationError(
                f"{what}: risk parity did not converge in {out['iterations']} "
                f"iterations (largest risk-share error {out['max_share_error']:.2e}); "
                "the last iterate is not a risk parity allocation. The covariance "
                "is near singular: drop a redundant sleeve."
            )
        weights = np.array([float(out["weights"][s]) for s in names])
        notes = [w for w in out["warnings"] if "not positive semi-definite" in w]
    else:  # hrp
        out = construction.hierarchical_risk_parity(
            frame, periods_per_year=periods_per_year
        )
        weights = np.array([float(out["weights"][s]) for s in names])
    binding: List[Dict[str, Any]] = []
    if rule.max_weight is not None:
        weights, capped, _left = water_fill(weights, np.full(n, rule.max_weight))
        binding = [
            {"sleeve": s, "cap": "max_weight", "limit": float(rule.max_weight)}
            for s, c in zip(names, capped)
            if c
        ]
        if binding and method == "risk_parity":
            notes.append(
                f"{what}: max_weight held {[b['sleeve'] for b in binding]} at "
                f"{rule.max_weight:g} and handed the excess to the others pro "
                "rata, so the risk contributions are no longer the budgets."
            )
    return weights, binding, notes, details


def fit_sleeve_rule(
    frame: pd.DataFrame,
    rule: SleeveRule,
    *,
    cash_rate: float = 0.0,
    periods_per_year: int = 252,
    capital: Optional[float] = None,
    what: str = "allocate_across_sleeves",
) -> Dict[str, Any]:
    """
    One fit of `rule` on `frame` (rows are dates, columns sleeves).

    Returns `weights` (an array in the columns' order, as fractions of the
    capital; the rest is cash), `covariance` (the annual matrix the rule's
    risk is read off), `binding` (the caps that bind: each a sleeve, the cap
    -- "max_weight" or "capacity" -- and its limit as a weight), `scale`
    (the target-volatility multiplier, or None), `left_in_cash` (what the
    capacity cut and nobody took), `notes` and `details` (the solver's
    certificate flag, the shrinkage report). `capital` overrides the rule's
    own, for a caller whose capital has moved since.
    """
    periods_per_year = require_periods_per_year(periods_per_year, what)
    cash_rate = require_finite_scalar(cash_rate, "cash_rate", what)
    names = [str(c) for c in frame.columns]
    if len(names) < 2:
        raise ValidationError(
            f"{what}: {len(names)} sleeve; an allocation needs at least two streams."
        )
    values = np.ascontiguousarray(frame.to_numpy(dtype=float))
    with single_threaded_blas():
        cov, cov_notes = _covariance_frame(frame, rule, periods_per_year)
        weights, binding, notes, details = _rule_weights(
            frame,
            values,
            cov,
            rule,
            cash_rate=cash_rate,
            periods_per_year=periods_per_year,
            what=what,
        )
        matrix = cov.to_numpy()
        scale = None
        if rule.target_volatility is not None:
            volatility = math.sqrt(max(float(weights @ matrix @ weights), 0.0))
            if not volatility > 0:
                raise ValidationError(
                    f"{what}: the rule's allocation has no volatility, so no "
                    "scale reaches target_volatility. Drop the target."
                )
            scale = rule.target_volatility / volatility
            gross = scale * float(np.abs(weights).sum())
            if gross > rule.max_leverage * (1.0 + _LEVERAGE_SLACK):
                raise ValidationError(
                    f"{what}: target_volatility={rule.target_volatility:g} needs "
                    f"a gross of {gross:.4g} (the {rule.method} allocation's "
                    f"volatility is {volatility:.4g}, scaled {scale:.4g}x), above "
                    f"max_leverage={rule.max_leverage:g}. Lower the target to at "
                    f"most {volatility * rule.max_leverage / float(np.abs(weights).sum()):.4g}, "
                    "or raise max_leverage if the leverage is meant."
                )
            weights = weights * scale
    left = 0.0
    if rule.capacity is not None:
        unknown = sorted({str(k) for k in rule.capacity} - set(names))
        if unknown:
            raise ValidationError(
                f"{what}: capacity names {unknown[:5]}, which are not sleeves; "
                f"the sleeves are {names[:8]}."
            )
        base = rule.capital if capital is None else float(capital)
        caps = np.array(
            [float(rule.capacity.get(s, math.inf)) / base for s in names], dtype=float
        )
        if rule.capacity_excess == "cash":
            capped = weights > caps * (1.0 + _AT_CAP)
            left = float(np.sum(weights[capped] - caps[capped]))
            weights = np.minimum(weights, caps)
        else:
            weights, capped, left = water_fill(weights, caps)
        binding += [
            {"sleeve": s, "cap": "capacity", "limit": float(c)}
            for s, c, hit in zip(names, caps, capped)
            if hit
        ]
        if left > 0 and rule.capacity_excess == "redistribute":
            notes.append(
                f"{what}: every sleeve with room is at its capacity, so "
                f"{left:.4g} of the capital stays in cash."
            )
        if (
            scale is not None
            and binding
            and any(b["cap"] == "capacity" for b in binding)
        ):
            notes.append(
                f"{what}: capacity moved capital after the target-volatility "
                "scaling, so the ex-ante volatility is not the target."
            )
    return {
        "weights": weights,
        "covariance": matrix,
        "binding": binding,
        "scale": scale,
        "left_in_cash": left,
        "notes": cov_notes + notes,
        "details": details,
    }


# ── the allocation report ────────────────────────────────────────────────


def _block_size(n: int, block_size: Optional[int]) -> int:
    """The bootstrap's block: n^(1/3) by default, at most half the sample."""
    size = max(1, int(round(n ** (1.0 / 3.0)))) if block_size is None else block_size
    return max(1, min(int(size), max(n // 2, 1)))


def _bounds(rule: SleeveRule, names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """The bounds every allocation of `rule` keeps, per sleeve: what a
    widened bootstrap draw is held inside. max_weight bounds the weights
    only when no target volatility rescales them afterwards."""
    n = len(names)
    cap = rule.max_weight if rule.target_volatility is None else None
    upper = np.full(n, math.inf if cap is None else cap)
    if rule.capacity is not None and rule.capital is not None:
        for i, s in enumerate(names):
            if s in rule.capacity:
                upper[i] = min(upper[i], float(rule.capacity[s]) / rule.capital)
    if rule.long_only:
        lower = np.zeros(n)
    else:
        lower = np.full(n, -math.inf if cap is None else -cap)
    return lower, upper


def _bootstrap_weights(
    values: np.ndarray,
    names: Sequence[str],
    rule: SleeveRule,
    *,
    cash_rate: float,
    periods_per_year: int,
    n_bootstrap: int,
    block_size: int,
    random_seed: int,
    what: str,
) -> Tuple[Optional[np.ndarray], int, Optional[str]]:
    """The rule refitted on circular block resamples of the rows: the draws
    (one row each), how many could not be fitted, and the first reason."""
    T = values.shape[0]
    rng = np.random.default_rng(int(random_seed))
    draws: List[np.ndarray] = []
    failed = 0
    reason: Optional[str] = None
    for _ in range(n_bootstrap):
        index = circular_block_indices(T, block_size, rng)
        sample = pd.DataFrame(values[index], columns=list(names))
        try:
            fit = fit_sleeve_rule(
                sample,
                rule,
                cash_rate=cash_rate,
                periods_per_year=periods_per_year,
                what=what,
            )
        except ValidationError as exc:
            failed += 1
            reason = reason or str(exc)
            continue
        draws.append(fit["weights"])
    if not draws:
        return None, failed, reason
    return np.vstack(draws), failed, reason


def allocate_across_sleeves(
    frame: pd.DataFrame,
    rule: SleeveRule,
    *,
    cash_rate: float = 0.0,
    periods_per_year: int = 252,
    n_bootstrap: int = 500,
    block_size: Optional[int] = None,
    random_seed: int = 0,
    what: str = "allocate_across_sleeves",
) -> Dict[str, Any]:
    """
    The allocation of `rule` on the aligned sleeve returns `frame`, and how
    much of it the sample decided.

    Returns per sleeve the weight, its risk contribution (w_i (Sigma w)_i /
    sigma, annual, summing to the ex-ante volatility) and risk share, and
    the 5th and 95th percentile of its bootstrap weight; for the whole, the
    cash weight, gross and net, the ex-ante volatility, the effective number
    of sleeves by weight ((sum |w|)^2 / sum w^2) and by risk (1 / sum of
    squared risk shares), the binding caps, the in-sample Sharpe ratio of
    the allocation's return over the cash rate, the bootstrap's stability
    (mean L1 distance of a resampled allocation from this one) and its
    draws, block and scale. With `capital` in the rule, the currency per
    sleeve and in cash.
    """
    periods_per_year = require_periods_per_year(periods_per_year, what)
    names = [str(c) for c in frame.columns]
    fit = fit_sleeve_rule(
        frame, rule, cash_rate=cash_rate, periods_per_year=periods_per_year, what=what
    )
    weights, matrix = fit["weights"], fit["covariance"]
    values = np.ascontiguousarray(frame.to_numpy(dtype=float))
    rf = cash_rate / periods_per_year
    with single_threaded_blas():
        marginal = matrix @ weights
        variance = float(weights @ marginal)
        excess = (values - rf) @ weights
    volatility = math.sqrt(max(variance, 0.0))
    nan = float("nan")
    if volatility > 0:
        contributions = weights * marginal / volatility
        shares = contributions / volatility
        effective_risk = 1.0 / float(np.sum(shares**2))
    else:
        contributions = np.full(len(names), nan)
        shares = contributions
        effective_risk = nan
    gross = float(np.abs(weights).sum())
    squares = float(np.sum(weights**2))
    effective_n = gross**2 / squares if squares > 0 else nan

    T = values.shape[0]
    block = _block_size(T, block_size)
    n_bootstrap = int(n_bootstrap)
    low = high = None
    stability = nan
    failed = 0
    reason = None
    scale = circular_block_scale(T, block)
    notes = list(fit["notes"])
    if n_bootstrap > 0:
        draws, failed, reason = _bootstrap_weights(
            values,
            names,
            rule,
            cash_rate=cash_rate,
            periods_per_year=periods_per_year,
            n_bootstrap=n_bootstrap,
            block_size=block,
            random_seed=random_seed,
            what=what,
        )
        fitted = 0 if draws is None else draws.shape[0]
        if draws is not None and fitted >= max(2, _MIN_FITTED_SHARE * n_bootstrap):
            centre = draws.mean(axis=0)
            draws = centre + scale * (draws - centre)
            draws = np.clip(draws, *_bounds(rule, names))
            low = np.percentile(draws, 5.0, axis=0)
            high = np.percentile(draws, 95.0, axis=0)
            stability = float(np.abs(draws - weights).sum(axis=1).mean())
        if failed:
            notes.append(
                f"{what}: the rule could not be fitted on {failed} of "
                f"{n_bootstrap} resamples"
                + (
                    "; the bands are from the rest."
                    if low is not None
                    else ", more than half, so there are no bands."
                )
                + f" The first reason: {reason}"
            )
        if n_bootstrap < 100:
            notes.append(
                f"{what}: {n_bootstrap} resamples put each band's 5th and 95th "
                "percentiles on a handful of draws; 500 is the default."
            )
    capital = rule.capital
    capital_by_sleeve = None
    cash_amount = None
    if capital is not None:
        amounts = weights * capital
        for b in fit["binding"]:
            if b["cap"] == "capacity":
                amounts[names.index(b["sleeve"])] = float(rule.capacity[b["sleeve"]])
        capital_by_sleeve = {s: float(a) for s, a in zip(names, amounts)}
        cash_amount = float(capital - math.fsum(amounts))
    by = {s: i for i, s in enumerate(names)}
    return {
        "method": rule.method,
        "covariance_method": rule.covariance_method,
        "n_sleeves": len(names),
        "n_observations": int(T),
        "periods_per_year": periods_per_year,
        "weights": {s: float(weights[by[s]]) for s in names},
        "cash_weight": float(1.0 - weights.sum()),
        "gross_weight": gross,
        "net_weight": float(weights.sum()),
        "capital_by_sleeve": capital_by_sleeve,
        "cash_amount": cash_amount,
        "risk_contributions": {s: float(contributions[by[s]]) for s in names},
        "risk_shares": {s: float(shares[by[s]]) for s in names},
        "ex_ante_volatility": volatility,
        "effective_n": effective_n,
        "effective_n_risk": effective_risk,
        "target_volatility_scale": fit["scale"],
        "binding_caps": fit["binding"],
        "in_sample_sharpe": annualized_sharpe(excess, periods_per_year),
        "weight_p5": None if low is None else {s: float(low[by[s]]) for s in names},
        "weight_p95": None if high is None else {s: float(high[by[s]]) for s in names},
        "stability_l1": stability,
        "n_bootstrap": n_bootstrap,
        "n_bootstrap_failed": failed,
        "block_size": block,
        "bootstrap_scale": scale,
        "random_seed": int(random_seed),
        "certified": fit["details"].get("certified"),
        "shrinkage": fit["details"].get("shrinkage"),
        "warnings": notes,
    }
