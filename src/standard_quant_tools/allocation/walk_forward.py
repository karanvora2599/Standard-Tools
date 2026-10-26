"""
An allocation rule tested the only way that tests it: walked forward.

`sleeves.allocate_across_sleeves` fits a rule on a sample and scores it on
the same sample, which is why its Sharpe ratio is labelled in-sample. Here
the rule is fitted on a trailing window, held while the sleeves' returns
drift its weights, refitted every `rebalance_every` periods at a cost per
unit of turnover, and scored only on the periods after each fit.

THE RULE SEES ITS WINDOW AND NOTHING ELSE. At a rebalance on row t the rule
is handed a new frame built from a copy of rows t - W .. t - 1 -- its own
array, its own index -- so nothing reachable from what it is given holds a
later row: not a view into the full sample, not its index. A rule that
tries to read past its window finds nothing there, so it chooses exactly
what a rule that never tried would, and every weight chosen at or before t
and every out-of-sample return on or before t is a function of the rows up
to t alone. Changing a later return changes none of them.

THE BOOK, period by period. Before the first fit the capital is cash. At a
rebalance the book trades from its drifted weights to the rule's, paying
cost_bps / 10,000 per unit of turnover sum_i |w_i - w_i,drifted| (the cash
leg is not a trade) out of the capital, so the period's return is

    (1 - cost)(1 + sum_i w_i r_i + c rf) - 1,

c = 1 - sum_i w_i the cash, earning (or, when negative, paying) the cash
rate rf = cash_rate / periods_per_year. Between rebalances nothing trades:
w_i becomes w_i (1 + r_i) / (1 + g) and c becomes c (1 + rf) / (1 + g), g
the period's return before the cost. A rebalance at which the rule cannot
be fitted -- a max-Sharpe window in which no sleeve beat cash, a target
volatility that needs more leverage than allowed -- trades nothing and is
counted, with its date and reason.

WHAT IS COMPARED. The same mechanics run for a benchmark rule (equal
weight by default, on the rule's covariance estimate and window and none
of its constraints) and for the rule fitted ONCE on the full sample and
held -- the look-ahead version, whose weights saw the periods they are
scored on. All three are scored on the same out-of-sample periods, by their
Sharpe ratios over the cash rate. The difference of the rule's and the
benchmark's Sharpe ratios (Sharpe ratios, not mean returns: a rule that
runs at another volatility is not penalized for it) has a circular
block-bootstrap interval on the paired periods, widened by
`_resampling.circular_block_scale` as the library's other paired intervals
are. The deflated Sharpe ratio of the rule's out-of-sample stream is taken
at the caller's own count of the rules and settings tried.

Nothing here selects a rule or a window: each call tests one, and choosing
the best of several by this result is itself a selection the trial count
must include.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools._resampling import (
    circular_block_indices,
    circular_block_scale,
)
from standard_quant_tools.backtesting.overfitting import deflated_sharpe_ratio
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.return_metrics import (
    annualized_volatility,
    cagr,
    equity_from_returns,
)
from standard_quant_tools.metrics.risk_metrics import annualized_sharpe, max_drawdown
from standard_quant_tools.numeric_contract import (
    require_finite_scalar,
    require_periods_per_year,
    require_positive_int,
)

from .sleeves import (
    METHODS,
    SleeveRule,
    _block_size,
    fit_sleeve_rule,
    sleeve_positions,
)

__all__ = [
    "MIN_OUT_OF_SAMPLE",
    "evaluate_allocation_rule",
    "regime_summary",
    "walk_rule_forward",
    "walk_sleeve_rule",
]

#: The fewest out-of-sample periods taken: the deflated Sharpe ratio's own
#: minimum, below which every statistic here is wider than what it measures.
MIN_OUT_OF_SAMPLE = 20

#: How many unfitted rebalances a warning names before counting the rest.
_SHOWN = 3

#: A rule: the trailing window (a frame of its own) and the capital at the
#: rebalance (None without one) in, the weights in the window's column
#: order out. A refusal (`ValidationError`) leaves the book as it is.
Rule = Callable[[pd.DataFrame, Optional[float]], Any]


def walk_rule_forward(
    frame: pd.DataFrame,
    rule: Rule,
    *,
    estimation_window: int,
    rebalance_every: int,
    cost_bps: float = 0.0,
    cash_rate: float = 0.0,
    periods_per_year: int = 252,
    capital: Optional[float] = None,
    what: str = "walk_rule_forward",
) -> Dict[str, Any]:
    """
    The out-of-sample path of `rule` over the aligned sleeve returns.

    Rebalances fall on rows W, W + k, W + 2k, ... (W the estimation window,
    k `rebalance_every`), each fitted on the W rows before it. Returns the
    out-of-sample `returns` (an array, one per row from W on) on `dates`,
    the `targets` the rule chose (a frame, one row per rebalance it fitted),
    the `turnover` and `cost` of each, the `unfitted` rebalances (date,
    reason), and the book's final drifted `weights`. `held` and
    `held_cash` are what the book held after every rebalance, fitted or
    not: the target, or the drifted weights it kept.
    """
    periods_per_year = require_periods_per_year(periods_per_year, what)
    window = require_positive_int(estimation_window, "estimation_window", what)
    every = require_positive_int(rebalance_every, "rebalance_every", what)
    cost_bps = require_finite_scalar(cost_bps, "cost_bps", what, minimum=0.0)
    cash_rate = require_finite_scalar(cash_rate, "cash_rate", what)
    # Row-contiguous whatever the frame's layout, so a row is the same bits
    # however the caller built the frame.
    values = np.ascontiguousarray(frame.to_numpy(dtype=float))
    names = [str(c) for c in frame.columns]
    index = frame.index
    T, n = values.shape
    if T - window < MIN_OUT_OF_SAMPLE:
        raise ValidationError(
            f"{what}: {T} shared periods and an estimation_window of {window} "
            f"leave {max(T - window, 0)} out of sample, fewer than "
            f"{MIN_OUT_OF_SAMPLE}. Shorten the window or give longer streams."
        )
    rf = cash_rate / periods_per_year
    kappa = cost_bps / 10_000.0
    weights = np.zeros(n)
    cash = 1.0
    nav = 1.0
    returns = np.empty(T - window)
    chosen: List[np.ndarray] = []
    chosen_dates: List[Any] = []
    turnover: List[float] = []
    costs: List[float] = []
    unfitted: List[Tuple[str, str]] = []
    held: List[np.ndarray] = []
    held_cash: List[float] = []
    held_dates: List[Any] = []
    for t in range(window, T):
        cost = 0.0
        if (t - window) % every == 0:
            # A frame of its own: copied rows and a copied index, so nothing
            # the rule can reach from it holds a row after t - 1.
            trailing = pd.DataFrame(
                values[t - window : t].copy(),
                index=index[t - window : t].copy(),
                columns=list(names),
            )
            target = None
            try:
                target = rule(trailing, None if capital is None else capital * nav)
            except ValidationError as exc:
                unfitted.append((_label(index[t]), str(exc)))
            if target is not None:
                target = _checked_target(target, names, index[t], what)
                traded = math.fsum(np.abs(target - weights))
                cost = kappa * traded
                weights = target.copy()
                cash = 1.0 - math.fsum(target)
                chosen.append(target)
                chosen_dates.append(index[t])
                turnover.append(traded)
                costs.append(cost)
            held.append(weights.copy())
            held_cash.append(cash)
            held_dates.append(index[t])
        r = values[t]
        # Correctly rounded: the same bits in any order and on any BLAS.
        gross = math.fsum(weights * r) + cash * rf
        growth = (1.0 - cost) * (1.0 + gross)
        if not growth > 0.0:
            raise ValidationError(
                f"{what}: on {_label(index[t])} the book lost all of its capital "
                f"(a period return of {growth - 1.0:.4g}); a levered allocation "
                "that loses more than everything has no return after it. Lower "
                "the leverage."
            )
        returns[t - window] = growth - 1.0
        nav *= growth
        weights = weights * (1.0 + r) / (1.0 + gross)
        cash = cash * (1.0 + rf) / (1.0 + gross)
    targets = pd.DataFrame(
        np.vstack(chosen) if chosen else np.empty((0, n)),
        index=pd.Index(chosen_dates, name=index.name),
        columns=names,
    )
    return {
        "dates": index[window:],
        "returns": returns,
        "targets": targets,
        "turnover": turnover,
        "cost": costs,
        "unfitted": unfitted,
        "weights": weights,
        "cash": cash,
        "held": pd.DataFrame(
            np.vstack(held) if held else np.empty((0, n)),
            index=pd.Index(held_dates, name=index.name),
            columns=names,
        ),
        "held_cash": np.array(held_cash, dtype=float),
    }


def walk_sleeve_rule(
    frame: pd.DataFrame,
    rule: SleeveRule,
    *,
    estimation_window: int,
    rebalance_every: int,
    cost_bps: float = 0.0,
    cash_rate: float = 0.0,
    periods_per_year: int = 252,
    what: str = "walk_rule_forward",
    reports: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    `walk_rule_forward` of a `SleeveRule`, refitted by `fit_sleeve_rule` on
    each trailing window, at the rule's capital.

    With a regime `state_column`, the window the rule is handed holds the
    state with the sleeves, and the book holds none of it: the rule's
    weights get a zero there, which adds nothing to any sum the walk takes,
    so the returns are the bits of a walk on the sleeves alone. The targets
    and holdings come back over the sleeves. `reports` collects the regime
    report of every fit.
    """
    columns = [str(c) for c in frame.columns]
    keep = sleeve_positions(frame, rule, what)
    padded = len(keep) != len(columns)

    def fit(window: pd.DataFrame, capital: Optional[float]) -> np.ndarray:
        out = fit_sleeve_rule(
            window,
            rule,
            cash_rate=cash_rate,
            periods_per_year=periods_per_year,
            capital=capital,
            what=what,
        )
        if reports is not None and "regime" in out["details"]:
            reports.append(out["details"]["regime"])
        if not padded:
            return out["weights"]
        full = np.zeros(len(columns))
        full[keep] = out["weights"]
        return full

    path = walk_rule_forward(
        frame,
        fit,
        estimation_window=estimation_window,
        rebalance_every=rebalance_every,
        cost_bps=cost_bps,
        cash_rate=cash_rate,
        periods_per_year=periods_per_year,
        capital=rule.capital,
        what=what,
    )
    if padded:
        path["targets"] = path["targets"].iloc[:, keep]
        path["held"] = path["held"].iloc[:, keep]
        path["weights"] = path["weights"][keep]
    return path


def regime_summary(reports: List[Dict[str, Any]], n_regimes: int) -> Dict[str, Any]:
    """What the regime did across a walk's fits: how many were
    conditioned, how many fell back to the unconditioned rule, how many
    fits each regime was current at, the mean shrink weight and the mean
    persistence of the window's labels."""
    conditioned = [r for r in reports if r["conditioned"]]
    labelled = [r for r in reports if r["current_regime"] is not None]
    current = [0] * n_regimes
    for r in labelled:
        current[int(r["current_regime"])] += 1
    nan = float("nan")
    return {
        "n_fits": len(reports),
        "n_conditioned": len(conditioned),
        "n_unconditioned": len(reports) - len(conditioned),
        "fits_per_regime": current,
        "mean_shrink_weight": (
            float(np.mean([r["shrink_weight"] for r in conditioned]))
            if conditioned
            else nan
        ),
        "mean_persistence": (
            float(np.mean([r["persistence"] for r in labelled])) if labelled else nan
        ),
    }


def _label(stamp: Any) -> str:
    if isinstance(stamp, pd.Timestamp):
        return str(stamp.date()) if stamp == stamp.normalize() else stamp.isoformat()
    return str(stamp)


def _checked_target(target: Any, names: List[str], stamp: Any, what: str) -> np.ndarray:
    """A rule's answer as an array in the sleeves' order, or a refusal."""
    if isinstance(target, dict):
        missing = [s for s in names if s not in target]
        if missing:
            raise ValidationError(
                f"{what}: the rule's weights on {_label(stamp)} leave out {missing[:5]}."
            )
        target = [target[s] for s in names]
    array = np.asarray(target, dtype=float).reshape(-1)
    if array.size != len(names) or not np.all(np.isfinite(array)):
        raise ValidationError(
            f"{what}: the rule's weights on {_label(stamp)} are not {len(names)} "
            "finite numbers, one per sleeve."
        )
    return array


def _summary(path: Dict[str, Any], rf: float, periods_per_year: int) -> Dict[str, Any]:
    """One stream's out-of-sample numbers."""
    r = path["returns"]
    series = pd.Series(r, index=path["dates"])
    equity = equity_from_returns(series)
    years = r.size / periods_per_year
    total_turnover = math.fsum(path["turnover"])
    with single_threaded_blas():
        sharpe = annualized_sharpe(r - rf, periods_per_year)
    return {
        "sharpe": sharpe,
        "annual_return": float(cagr(equity, periods_per_year)),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "volatility": float(annualized_volatility(series, periods_per_year)),
        "max_drawdown": float(max_drawdown(equity)),
        "total_turnover": total_turnover,
        "annual_turnover": total_turnover / years,
        "total_cost": math.fsum(path["cost"]),
        "n_rebalances": len(path["turnover"]) + len(path["unfitted"]),
        "n_unfitted": len(path["unfitted"]),
    }


def _paired_sharpe_difference(
    x: np.ndarray,
    y: np.ndarray,
    *,
    periods_per_year: int,
    n_bootstrap: int,
    block_size: int,
    random_seed: int,
    confidence: float,
) -> Dict[str, Any]:
    """Sharpe(x) - Sharpe(y) with a circular block-bootstrap interval on the
    paired periods, the draws widened about their mean, and a two-sided
    percentile p-value read off the same draws."""
    n = x.size
    observed = annualized_sharpe(x, periods_per_year) - annualized_sharpe(
        y, periods_per_year
    )
    nan = float("nan")
    out = {
        "observed": observed,
        "lower": nan,
        "upper": nan,
        "p_value": nan,
        "scale": circular_block_scale(n, block_size),
        "n_failed": 0,
    }
    if n_bootstrap <= 0 or not math.isfinite(observed):
        return out
    rng = np.random.default_rng(int(random_seed))
    draws = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        index = circular_block_indices(n, block_size, rng)
        draws[i] = annualized_sharpe(x[index], periods_per_year) - annualized_sharpe(
            y[index], periods_per_year
        )
    usable = draws[np.isfinite(draws)]
    out["n_failed"] = int(n_bootstrap - usable.size)
    if usable.size < max(2, n_bootstrap // 2):
        return out
    if out["scale"] != 1.0:
        centre = float(usable.mean())
        usable = centre + out["scale"] * (usable - centre)
    alpha = (1.0 - confidence) / 2.0
    out["lower"] = float(np.percentile(usable, alpha * 100.0))
    out["upper"] = float(np.percentile(usable, (1.0 - alpha) * 100.0))
    if observed == 0.0 and np.all(usable == 0.0):
        out["p_value"] = 1.0
    else:
        far = float(np.mean(usable <= 0.0) if observed > 0 else np.mean(usable >= 0.0))
        out["p_value"] = float(min(1.0, 2.0 * max(far, 1.0 / usable.size)))
    return out


def evaluate_allocation_rule(
    frame: pd.DataFrame,
    rule: SleeveRule,
    *,
    n_trials: int,
    benchmark_rule: str = "equal",
    estimation_window: int = 252,
    rebalance_every: int = 21,
    cost_bps: float = 0.0,
    cash_rate: float = 0.0,
    periods_per_year: int = 252,
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    random_seed: int = 0,
    confidence: float = 0.95,
    what: str = "evaluate_allocation_rule",
) -> Dict[str, Any]:
    """
    `rule` walked forward over the aligned sleeve returns `frame`, against a
    benchmark rule and against itself fitted on the full sample.

    Returns `dates` and the out-of-sample `rule_returns` and
    `benchmark_returns`, the rule's chosen weights (`targets`, a frame),
    the summaries of the `rule`, the `benchmark` and the `full_sample` fit
    (Sharpe over the cash rate, annual return, total return, volatility,
    maximum drawdown, turnover, cost, rebalances; the full-sample one None
    when the rule cannot be fitted on every period), the `sharpe_difference`
    (rule minus benchmark, with its interval at `confidence` and a
    p-value), the `look_ahead_gap` (the full-sample fit's Sharpe minus the
    rule's), the full-sample weights, the `deflated` Sharpe ratio of the
    rule's stream at `n_trials`, the `unfitted` rebalances and warnings.
    A regime rule adds its `regime` summary (`regime_summary`); its state
    column, if it has one, is in the rule's windows and in no book. With
    `benchmark_rule` the rule's own method, the benchmark is its
    unconditioned twin (without its constraints).
    """
    n_trials = require_positive_int(n_trials, "n_trials", what)
    periods_per_year = require_periods_per_year(periods_per_year, what)
    if benchmark_rule not in METHODS:
        raise ValidationError(
            f"{what}: benchmark_rule={benchmark_rule!r} is not one of {list(METHODS)}."
        )
    if not 0.0 < confidence < 1.0:
        raise ValidationError(f"{what}: confidence={confidence!r} must be in (0, 1).")
    benchmark = SleeveRule(
        method=benchmark_rule,
        covariance_method=rule.covariance_method,
        halflife=rule.halflife,
    )
    settings = dict(
        estimation_window=estimation_window,
        rebalance_every=rebalance_every,
        cost_bps=cost_bps,
        cash_rate=cash_rate,
        periods_per_year=periods_per_year,
        what=what,
    )
    # A regime's state column rides in the rule's windows and is no sleeve:
    # the benchmark and the held full-sample fit trade the sleeves alone.
    keep = sleeve_positions(frame, rule, what)
    sleeves = frame if len(keep) == frame.shape[1] else frame.iloc[:, keep]
    reports: List[Dict[str, Any]] = []
    path = walk_sleeve_rule(frame, rule, reports=reports, **settings)
    bench = walk_sleeve_rule(sleeves, benchmark, **settings)
    warnings: List[str] = []
    rf = cash_rate / periods_per_year
    held_summary: Optional[Dict[str, Any]] = None
    held_weights: Optional[np.ndarray] = None
    try:
        whole = fit_sleeve_rule(
            frame.copy(),
            rule,
            cash_rate=cash_rate,
            periods_per_year=periods_per_year,
            what=what,
        )
    except ValidationError as exc:
        warnings.append(
            f"{what}: the rule cannot be fitted on the full sample, so there is "
            f"no look-ahead comparison: {exc}"
        )
    else:
        warnings += whole["notes"]
        held_weights = whole["weights"]
        held = walk_rule_forward(
            sleeves,
            lambda _window, _capital: held_weights,
            capital=rule.capital,
            **settings,
        )
        held_summary = _summary(held, rf, periods_per_year)
    regime = None
    if rule.regime is not None:
        regime = regime_summary(reports, rule.regime.n_regimes)
        if regime["n_unconditioned"]:
            warnings.append(
                f"{what}: at {regime['n_unconditioned']} of {regime['n_fits']} "
                "fits the regime could not be conditioned on (too few of the "
                "window's returns in the current regime, or a state that could "
                "not be labelled) and the rule was fitted unconditioned."
            )

    rule_summary = _summary(path, rf, periods_per_year)
    bench_summary = _summary(bench, rf, periods_per_year)
    n_oos = int(path["returns"].size)
    block = _block_size(n_oos, block_size)
    with single_threaded_blas():
        difference = _paired_sharpe_difference(
            path["returns"] - rf,
            bench["returns"] - rf,
            periods_per_year=periods_per_year,
            n_bootstrap=int(n_bootstrap),
            block_size=block,
            random_seed=random_seed,
            confidence=confidence,
        )
    deflated: Dict[str, Any] = {}
    try:
        deflated = deflated_sharpe_ratio(
            pd.Series(path["returns"] - rf),
            n_trials=n_trials,
            periods_per_year=periods_per_year,
        )
    except ValidationError as exc:
        warnings.append(f"{what}: no deflated Sharpe ratio: {exc}")
    gap = (
        float("nan")
        if held_summary is None
        else held_summary["sharpe"] - rule_summary["sharpe"]
    )
    if path["unfitted"]:
        shown = "; ".join(f"{d}: {why}" for d, why in path["unfitted"][:_SHOWN])
        more = len(path["unfitted"]) - _SHOWN
        warnings.append(
            f"{what}: the rule could not be fitted at {len(path['unfitted'])} of "
            f"{rule_summary['n_rebalances']} rebalances, which traded nothing "
            f"and held the book (cash before the first fit). {shown}"
            + (f"; and {more} more." if more > 0 else ".")
        )
    if held_summary is not None and math.isfinite(gap) and gap > 0.25:
        warnings.append(
            f"{what}: fitted once on the full sample, the rule's Sharpe ratio on "
            f"the same out-of-sample periods is {held_summary['sharpe']:.2f} "
            f"against {rule_summary['sharpe']:.2f} walked forward: {gap:.2f} of "
            "an in-sample result is the fit seeing the periods it is scored on."
        )
    if math.isfinite(difference["lower"]) and (
        difference["lower"] <= 0.0 <= difference["upper"]
    ):
        warnings.append(
            f"{what}: the {confidence:.0%} interval of the Sharpe difference "
            f"against {benchmark_rule}, [{difference['lower']:.2f}, "
            f"{difference['upper']:.2f}], contains zero: out of sample the rule "
            "is not distinguishable from the benchmark rule."
        )
    deflated_warnings = list(deflated.get("warnings", []))
    warnings += [w for w in deflated_warnings if "deflated Sharpe probability" in w]
    return {
        "dates": path["dates"],
        "rule_returns": path["returns"],
        "benchmark_returns": bench["returns"],
        "targets": path["targets"],
        "rule": rule_summary,
        "benchmark": bench_summary,
        "full_sample": held_summary,
        "full_sample_weights": (
            None
            if held_weights is None
            else {str(c): float(w) for c, w in zip(sleeves.columns, held_weights)}
        ),
        "look_ahead_gap": gap,
        "sharpe_difference": difference,
        "confidence": confidence,
        "block_size": block,
        "n_bootstrap": int(n_bootstrap),
        "deflated": deflated,
        "n_trials": n_trials,
        "unfitted": path["unfitted"],
        "n_out_of_sample": n_oos,
        "regime": regime,
        "warnings": warnings,
    }
