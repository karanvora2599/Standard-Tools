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

REGIMES, when the rule has a `RegimeSpec`. Every period of the window has
a STATE, a number known at its end: by default the EWMA volatility of the
sleeves' equal-weight composite through that period, or the value of a
column the caller names (`state_column`, which is then not a sleeve). The
window's states are labelled into two or three regimes -- by the library's
Gaussian mixture (`analysis.stationarity.detect_regimes`) or by the
window's own quantiles -- regime 0 holding the lowest states. A period's
return is paired with the label of the period BEFORE it, the state known
when it began: so the moments of regime r are those of the returns that
followed a period in r, a return never chooses its own label (a large
move would otherwise vote itself into the volatile regime and inflate it),
and the current regime -- the window's last label -- is the one the next
period starts in. The rule's mean and covariance are then those of the
returns in the current regime, shrunk toward the whole window's by
n_r / (n_r + n0); HRP clusters the regime's rows. Fewer than
`min_observations` such returns, a state with no dispersion or one the
mixture cannot fit, and the rule is fitted unconditioned, with a note. All
of it is read from the window: walked forward, the regime rule sees
nothing its unconditioned twin does not.

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
from standard_quant_tools.analysis.stationarity import detect_regimes
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
    "REGIME_LABELS",
    "RegimeSpec",
    "SleeveRule",
    "allocate_across_sleeves",
    "ewma_volatility",
    "fit_sleeve_rule",
    "label_regimes",
    "shrunk_sharpe_ratios",
    "sleeve_positions",
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

#: How a window's states are labelled: the library's Gaussian mixture, or
#: the window's own quantiles of the state.
REGIME_LABELS = ("mixture", "quantile")


@dataclass(frozen=True)
class RegimeSpec:
    """
    How a rule conditions its estimates on the current regime, read from
    its window alone (see the module docstring, REGIMES).

    `state_column` names a column of the returns as the state -- it is then
    not a sleeve -- and without one the state is the EWMA volatility of the
    sleeves' equal-weight composite at half-life `halflife` periods.
    `labels` is "mixture" or "quantile", into `n_regimes` (2 or 3) regimes.
    `shrinkage` is n0: the regime's moments get the weight n_r / (n_r + n0)
    and the whole window's the rest. Fewer than `min_observations` returns
    in the current regime, and the rule is fitted unconditioned.

    THE DEFAULTS ARE MEASURED, on planted sleeves walked forward with a
    252-period window, monthly refits and 10 bps a unit of turnover, 30
    ten-year histories a case, each rule against itself unconditioned (the
    CHANGELOG entry that added this has the table). On two-regime Markov
    sleeves (calm and stressed, different means and correlations) the
    default spec gained 0.050 +- 0.007 of Sharpe ratio for risk parity at
    a volatility target, 0.013 +- 0.002 for risk parity and 0.015 +- 0.005
    for minimum variance; on sleeves with no regimes it cost those rules at
    most 0.006, and cost max Sharpe 0.03 to 0.06 -- whose gain on the regime
    sleeves, 0.043 +- 0.028, is not distinguishable from none. The mixture
    beat the quantiles where there were no regimes (it finds one regime
    there; the quantiles always cut the window in half), a 10-period
    half-life beat 21 and 42 on the regime sleeves (the state must turn
    before the regime does), n0 = 126 cost 30 to 45% less than 63 without
    regimes and kept three quarters of its gain with them, and three
    regimes cost more than two where there were none and gained no more
    where there were.
    """

    n_regimes: int = 2
    labels: str = "mixture"
    state_column: Optional[str] = None
    halflife: float = 10.0
    shrinkage: float = 126.0
    min_observations: int = 20

    def __post_init__(self) -> None:
        who = "allocation rule: regime"
        if isinstance(self.n_regimes, bool) or self.n_regimes not in (2, 3):
            raise ValidationError(
                f"{who}: n_regimes={self.n_regimes!r} must be 2 or 3; more "
                "regimes than that leave each a handful of a window's periods."
            )
        if self.labels not in REGIME_LABELS:
            raise ValidationError(
                f"{who}: labels={self.labels!r} is not one of {list(REGIME_LABELS)}."
            )
        if self.state_column is not None and not str(self.state_column):
            raise ValidationError(
                f"{who}: state_column is empty; name a column of the returns, "
                "or omit it for the EWMA volatility of the sleeves' mix."
            )
        require_finite_scalar(self.halflife, "halflife", who)
        if not self.halflife > 0:
            raise ValidationError(
                f"{who}: halflife={self.halflife!r} must be a positive number of "
                "periods; 10 is the default."
            )
        require_finite_scalar(self.shrinkage, "shrinkage", who)
        if not self.shrinkage >= 0:
            raise ValidationError(
                f"{who}: shrinkage={self.shrinkage!r} must be at least 0: it is "
                "n0, the periods' worth of weight the whole window keeps."
            )
        if (
            isinstance(self.min_observations, bool)
            or not isinstance(self.min_observations, (int, np.integer))
            or self.min_observations < 2
        ):
            raise ValidationError(
                f"{who}: min_observations={self.min_observations!r} must be a "
                "whole number of at least 2."
            )


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
    `regime` conditions the rule's estimates on the current regime.
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
    regime: Optional[RegimeSpec] = None

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
        if self.regime is not None and not isinstance(self.regime, RegimeSpec):
            raise ValidationError(
                f"{who}: regime is a {type(self.regime).__name__}; give a "
                "RegimeSpec, or None for an unconditioned rule."
            )
        if (
            self.regime is not None
            and self.regime.state_column is not None
            and (
                self.regime.state_column in (self.risk_budgets or {})
                or self.regime.state_column in (self.capacity or {})
            )
        ):
            raise ValidationError(
                f"{who}: state_column={self.regime.state_column!r} is the "
                "regime's state, not a sleeve, and risk_budgets or capacity "
                "names it. Drop it from those, or choose another state."
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


# ── regimes ──────────────────────────────────────────────────────────────


def sleeve_positions(frame: pd.DataFrame, rule: SleeveRule, what: str) -> List[int]:
    """
    The positions of the frame's columns that are sleeves: every column but
    the rule's regime state, refused by name when the frame lacks it.
    """
    names = [str(c) for c in frame.columns]
    state = None if rule.regime is None else rule.regime.state_column
    if state is None:
        return list(range(len(names)))
    if state not in names:
        raise ValidationError(
            f"{what}: regime state_column={state!r} is not a column of the "
            f"returns, whose columns are {names[:8]}. Name one of them, or "
            "omit state_column for the EWMA volatility of the sleeves' "
            "equal-weight mix."
        )
    return [i for i, name in enumerate(names) if name != state]


def ewma_volatility(values: np.ndarray, halflife: float) -> np.ndarray:
    """
    Each period's EWMA volatility of `values` through that period, per
    period: sqrt(sum_j a^(s-j) x_j^2 / sum_j a^(s-j)), a = 0.5^(1/halflife),
    the sums over the periods up to and including s. Normalized by the
    weights actually present, so the first periods are averages of the few
    squares there are rather than a guess at what came before.
    """
    decay = 0.5 ** (1.0 / float(halflife))
    out = np.empty(values.size)
    weighted = 0.0
    total = 0.0
    for i, x in enumerate(values):
        weighted = decay * weighted + float(x) * float(x)
        total = decay * total + 1.0
        out[i] = math.sqrt(weighted / total)
    return out


def label_regimes(
    state: np.ndarray, spec: RegimeSpec
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Each period's regime from its state, 0 the regime of the lowest
    states, or None and why the states cannot be labelled.

    "quantile" cuts at the window's own k-quantiles of the state (a state
    on a cut takes the lower label). "mixture" is `detect_regimes`, whose
    regimes are sorted by the volatility OF THE STATE; they are re-sorted
    here by the state's mean, so regime 0 is the calm one when the state
    is a volatility and the low one for any other state.
    """
    k = spec.n_regimes
    if has_no_dispersion(state):
        return None, "the state does not vary over the window"
    if spec.labels == "quantile":
        cuts = np.quantile(state, np.arange(1, k) / k)
        labels = (state[:, None] > cuts[None, :]).sum(axis=1)
    else:
        try:
            out = detect_regimes(pd.Series(state), n_regimes=k)
        except ValidationError as exc:
            return None, f"the mixture cannot be fitted: {exc}"
        means = np.array([float(r["mean"]) for r in out["regimes"]])
        remap = np.empty(k, dtype=int)
        remap[np.argsort(means, kind="stable")] = np.arange(k)
        labels = remap[np.asarray(out["labels"], dtype=int)]
    return labels.astype(int), None


def _regime_moments(
    frame: pd.DataFrame,
    values: np.ndarray,
    cov: pd.DataFrame,
    state: Optional[np.ndarray],
    rule: SleeveRule,
    *,
    periods_per_year: int,
    what: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], List[str]]:
    """
    The current regime's moments, shrunk toward the window's, or None when
    the rule is to be fitted unconditioned; the report; the notes.

    `cov` is the whole window's covariance, already estimated. The regime's
    rows are those whose PREVIOUS period carries the current label.
    """
    spec = rule.regime
    assert spec is not None
    k = spec.n_regimes
    if state is None:
        state = ewma_volatility(values.mean(axis=1), spec.halflife)
        source = "ewma_volatility"
        current_state = float(state[-1]) * math.sqrt(periods_per_year)
    else:
        source = str(spec.state_column)
        current_state = float(state[-1])
    report: Dict[str, Any] = {
        "state": source,
        "labels": spec.labels,
        "n_regimes": k,
        "current_state": current_state,
        "current_regime": None,
        "periods_per_regime": None,
        "regime_periods": 0,
        "shrink_weight": None,
        "persistence": None,
        "conditioned": False,
    }
    labels, why = label_regimes(state, spec)
    if labels is None:
        return (
            None,
            report,
            [f"{what}: regime: {why}, so the rule is fitted unconditioned."],
        )
    current = int(labels[-1])
    previous = labels[:-1]
    rows = np.flatnonzero(previous == current) + 1
    n_regime = int(rows.size)
    switches = int(np.count_nonzero(np.diff(labels)))
    report.update(
        current_regime=current,
        periods_per_regime=[int(c) for c in np.bincount(previous, minlength=k)],
        regime_periods=n_regime,
        persistence=1.0 - switches / max(labels.size - 1, 1),
    )
    if n_regime < spec.min_observations:
        return (
            None,
            report,
            [
                f"{what}: regime: {n_regime} of the window's returns follow a "
                f"period in the current regime {current}, fewer than "
                f"min_observations={spec.min_observations}, so the rule is "
                "fitted unconditioned."
            ],
        )
    weight = n_regime / (n_regime + float(spec.shrinkage))
    subset = frame.iloc[rows]
    regime_cov, notes = _covariance_frame(subset, rule, periods_per_year)
    conditioned = pd.DataFrame(
        weight * regime_cov.to_numpy() + (1.0 - weight) * cov.to_numpy(),
        index=cov.index,
        columns=cov.columns,
    )
    report.update(shrink_weight=weight, conditioned=True)
    moments = {
        "covariance": conditioned,
        "mean": weight * values[rows].mean(axis=0)
        + (1.0 - weight) * values.mean(axis=0),
        "frame": subset,
        "values": np.ascontiguousarray(values[rows]),
        "weight": weight,
        "whole_covariance": cov,
    }
    return moments, report, [f"{what}: regime: {n}" for n in notes]


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
    moments: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """min_variance, max_sharpe or shrunk_sharpe through the exact solver.

    With a regime's `moments`, `cov` is already the conditioned covariance;
    max_sharpe reads the conditioned mean, and shrunk_sharpe blends the
    James-Stein Sharpe ratios of the regime's rows with the whole window's
    by the regime's weight (the shrinkage reported is the regime's)."""
    names = list(cov.columns)
    cap = rule.max_weight
    lower = 0.0 if rule.long_only else (-cap if cap is not None else -math.inf)
    upper = cap if cap is not None else math.inf
    details: Dict[str, Any] = {}
    kwargs: Dict[str, Any] = {}
    if method != "min_variance":
        if method == "max_sharpe":
            if moments is None:
                mu = values.mean(axis=0) * periods_per_year
            else:
                mu = moments["mean"] * periods_per_year
        else:
            matrix = cov.to_numpy()
            sd = np.sqrt(np.diag(matrix))
            if moments is None:
                shrink = shrunk_sharpe_ratios(
                    values,
                    matrix / np.outer(sd, sd),
                    rf_per_period=cash_rate / periods_per_year,
                    periods_per_year=periods_per_year,
                    what=f"{what}: shrunk_sharpe",
                )
                shrunk = shrink["shrunk_sharpe"]
            else:
                shrink, shrunk = _regime_shrunk_sharpe(
                    values,
                    moments,
                    cash_rate=cash_rate,
                    periods_per_year=periods_per_year,
                    what=what,
                )
            details["shrinkage"] = {
                key: (
                    {n: float(v) for n, v in zip(names, value)}
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in shrink.items()
            }
            mu = cash_rate + shrunk * sd
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


def _regime_shrunk_sharpe(
    values: np.ndarray,
    moments: Dict[str, Any],
    *,
    cash_rate: float,
    periods_per_year: int,
    what: str,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """The regime's James-Stein report and the annual Sharpe ratios the
    tangency reads: the regime rows' shrunk Sharpes and the whole window's,
    blended by the regime's weight, as the mean and covariance are."""

    def james_stein(rows: np.ndarray, matrix: np.ndarray) -> Dict[str, Any]:
        sd = np.sqrt(np.diag(matrix))
        return shrunk_sharpe_ratios(
            rows,
            matrix / np.outer(sd, sd),
            rf_per_period=cash_rate / periods_per_year,
            periods_per_year=periods_per_year,
            what=f"{what}: shrunk_sharpe",
        )

    weight = moments["weight"]
    regime = james_stein(moments["values"], moments["covariance"].to_numpy())
    whole = james_stein(values, moments["whole_covariance"].to_numpy())
    blended = weight * regime["shrunk_sharpe"] + (1.0 - weight) * whole["shrunk_sharpe"]
    return regime, blended


def _rule_weights(
    frame: pd.DataFrame,
    values: np.ndarray,
    cov: pd.DataFrame,
    rule: SleeveRule,
    *,
    cash_rate: float,
    periods_per_year: int,
    what: str,
    moments: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str], Dict[str, Any]]:
    """The rule's fully invested weights, max_weight applied. With a
    regime's `moments`, `cov` is the conditioned covariance and HRP
    clusters the regime's rows."""
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
            moments=moments,
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
            frame if moments is None else moments["frame"],
            periods_per_year=periods_per_year,
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
    certificate flag, the shrinkage report, and with a regime the regime
    report). `capital` overrides the rule's own, for a caller whose
    capital has moved since. With a regime `state_column` the frame holds
    the state too: `sleeves` names the columns the weights are for.
    """
    periods_per_year = require_periods_per_year(periods_per_year, what)
    cash_rate = require_finite_scalar(cash_rate, "cash_rate", what)
    names = [str(c) for c in frame.columns]
    state: Optional[np.ndarray] = None
    if rule.regime is not None and rule.regime.state_column is not None:
        keep = sleeve_positions(frame, rule, what)
        state = frame.iloc[:, names.index(rule.regime.state_column)].to_numpy(
            dtype=float
        )
        frame = frame.iloc[:, keep]
        names = [names[i] for i in keep]
    if len(names) < 2:
        raise ValidationError(
            f"{what}: {len(names)} sleeve; an allocation needs at least two streams."
        )
    values = np.ascontiguousarray(frame.to_numpy(dtype=float))
    with single_threaded_blas():
        cov, cov_notes = _covariance_frame(frame, rule, periods_per_year)
        moments: Optional[Dict[str, Any]] = None
        regime_report: Optional[Dict[str, Any]] = None
        if rule.regime is not None:
            moments, regime_report, regime_notes = _regime_moments(
                frame,
                values,
                cov,
                state,
                rule,
                periods_per_year=periods_per_year,
                what=what,
            )
            cov_notes = cov_notes + regime_notes
            if moments is not None:
                cov = moments["covariance"]
        weights, binding, notes, details = _rule_weights(
            frame,
            values,
            cov,
            rule,
            cash_rate=cash_rate,
            periods_per_year=periods_per_year,
            what=what,
            moments=moments,
        )
        if regime_report is not None:
            details["regime"] = regime_report
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
        "sleeves": names,
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
    columns = [str(c) for c in frame.columns]
    fit = fit_sleeve_rule(
        frame, rule, cash_rate=cash_rate, periods_per_year=periods_per_year, what=what
    )
    names = fit["sleeves"]
    weights, matrix = fit["weights"], fit["covariance"]
    # Every column, the regime's state with them: what a resample redraws.
    everything = np.ascontiguousarray(frame.to_numpy(dtype=float))
    values = (
        everything
        if len(names) == len(columns)
        else np.ascontiguousarray(everything[:, sleeve_positions(frame, rule, what)])
    )
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
            everything,
            columns,
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
        "regime": fit["details"].get("regime"),
        "warnings": notes,
    }
