"""
Choosing among allocation rules as you go, and testing the choosing.

Picking the rule with the best walk-forward record and quoting that record
is a selection on the test set: the record is the winner's, chosen because
it won. Measured on sleeves where no rule can beat equal weight, the
hindsight-best of six rules showed 0.06 of Sharpe ratio over a selector
that had to choose before each period, and 0.18 on sleeves where one rule
really was better and the selector could not tell which. This module runs
the honest version -- a selector that, at each rebalance, chooses from what
the rules had ALREADY done -- and reports it beside the number a hindsight
pick would have shown.

THE NESTED WALK. Every candidate rule is walked forward over the whole
sample (`walk_forward.walk_sleeve_rule`: fitted on a trailing window,
drifting, rebalanced every k periods at a cost), which gives each its
out-of-sample returns. The selector rebalances on the same rows. At the
rebalance on row t it scores each candidate by the Sharpe ratio, over the
cash rate, of that candidate's realized out-of-sample returns on rows
t - L .. t - 1 (L the `selection_window`) -- strictly before t, each of
them a return the candidate earned after its own fit -- picks, and trades
its own book from its drifted weights to what the chosen candidate holds at
t (its new target, or its drifted book when it could not be fitted there),
paying cost_bps on the turnover. With `combine="top_k"` -- the default, k
the better half of the candidates -- it holds the equal blend of the k best
candidates' books; `combine="best"` holds the one. Every return of the
selector on row s is therefore a function of the rows up to s alone;
changing any later row changes no choice, weight or return at or before s.

HYSTERESIS. A selector that follows the trailing leader on noise switches
often and pays for each switch. `switch_margin` is the annual Sharpe ratio
a challenger must beat the incumbent by before the selector leaves it; with
top_k, the worst member of the held set is replaced by the best outsider
only past the same margin.

THE WARM-UP. Until a candidate has L out-of-sample periods behind it there
is nothing to score, and the selector holds the first candidate. Those
periods are reported and are not scored: every summary, difference, the
hindsight pick, the PBO and the deflated Sharpe ratios are over the scored
periods, which start at the first rebalance with L periods of history.

WHAT IS COMPARED, on the scored periods:

- the selector against the equal-weight rule walked forward the same way,
  by the paired Sharpe difference with its circular block-bootstrap
  interval (`walk_forward._paired_sharpe_difference`);
- the HINDSIGHT best candidate -- the one with the highest Sharpe ratio
  over these periods, which no one could have known to hold -- and its
  Sharpe minus the selector's: the part of a picked-the-winner backtest
  that was the picking;
- the probability of backtest overfitting over the candidates' streams
  (`backtesting.overfitting.probability_of_backtest_overfitting`): how
  often the in-sample best of these candidates lands in the bottom half
  out of sample. Near 0.5 or above, choosing among them by their record is
  no better than choosing at random;
- the deflated Sharpe ratio of the selector's stream at the caller's
  `n_trials`, and of the hindsight winner at n_trials times the number of
  candidates: what an honest claim for the winner must survive.

The linear algebra runs on one BLAS thread (see `_blas`), so the same input
gives the same bits on any machine.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.backtesting.overfitting import (
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from standard_quant_tools.error import ValidationError
from standard_quant_tools.metrics.risk_metrics import annualized_sharpe
from standard_quant_tools.numeric_contract import (
    require_finite_scalar,
    require_periods_per_year,
    require_positive_int,
)

from .sleeves import SleeveRule, _block_size
from .walk_forward import (
    MIN_OUT_OF_SAMPLE,
    _label,
    _paired_sharpe_difference,
    _summary,
    walk_sleeve_rule,
)

__all__ = [
    "COMBINE",
    "DEFAULT_COMBINE",
    "DEFAULT_SELECTION_WINDOW",
    "DEFAULT_SWITCH_MARGIN",
    "MAX_CANDIDATES",
    "PBO_SPLITS",
    "RESERVED_NAMES",
    "default_top_k",
    "run_selector",
    "select_allocation_rule",
    "walk_candidates",
]

#: How the selector holds what it chose: the single best candidate, or the
#: equal blend of the k best.
COMBINE = ("best", "top_k")

#: The most candidates one selection takes. Past a dozen the selection is
#: itself a search, and the trials it spends are the caller's to count.
MAX_CANDIDATES = 12

#: The defaults, each from the measurements on planted sleeves (six rules,
#: 40 histories of ten years a case, 10 bps a unit of turnover) recorded in
#: the CHANGELOG entry that added this module. Paired against the single
#: best at a 252-period window and a 0.5 margin, the equal blend of the
#: better half of six rules gained 0.041 +- 0.011 of Sharpe ratio where no
#: rule had an edge and 0.022 +- 0.011 where one sleeve had a modest one,
#: and gave up 0.064 +- 0.014 where one rule had a large edge -- of a 0.35
#: gain over equal weight. A margin of 0.5 beat no margin in every case
#: (by 0.015 to 0.039), and a 252-period window beat 126 and 504 in three
#: of the four.
DEFAULT_SELECTION_WINDOW = 252
DEFAULT_SWITCH_MARGIN = 0.5
DEFAULT_COMBINE = "top_k"


def default_top_k(n_candidates: int) -> int:
    """The better half: k = n // 2, at least 1 (which is the single best)."""
    return max(1, int(n_candidates) // 2)


#: The CSCV split of the candidates' streams: C(8, 4) = 70 combinations.
PBO_SPLITS = 8

#: Names the published returns panel uses for the selector and the
#: benchmark, which a candidate therefore cannot take.
RESERVED_NAMES = ("selector", "benchmark")

#: How many candidates a warning lists before counting the rest.
_SHOWN = 3


# ── the candidates ───────────────────────────────────────────────────────


def _checked_candidates(
    candidates: Sequence[Tuple[str, SleeveRule]], what: str, minimum: int
) -> Tuple[List[str], List[SleeveRule]]:
    names = [str(name) for name, _rule in candidates]
    rules = [rule for _name, rule in candidates]
    if len(rules) < minimum:
        raise ValidationError(
            f"{what}: {len(rules)} candidate rule(s), and a selection needs at "
            f"least {minimum}: with one there is nothing to choose. "
            "evaluate_allocation_rule tests a single rule."
        )
    if len(rules) > MAX_CANDIDATES:
        raise ValidationError(
            f"{what}: {len(rules)} candidate rules, more than the "
            f"{MAX_CANDIDATES} one selection takes. Narrow the list; every "
            "candidate tried is a trial to count."
        )
    empty = [i for i, name in enumerate(names) if not name]
    if empty:
        raise ValidationError(
            f"{what}: the candidates at positions {empty} have empty names. "
            "Name each one."
        )
    counts: Dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    repeated = sorted(name for name, count in counts.items() if count > 1)
    if repeated:
        raise ValidationError(
            f"{what}: the candidate name(s) {repeated[:_SHOWN]} are repeated; "
            "each candidate's stream and choices are reported by name. Give "
            "each its own, e.g. 'risk_parity_ewma'."
        )
    reserved = [name for name in names if name in RESERVED_NAMES]
    if reserved:
        raise ValidationError(
            f"{what}: {reserved} is reserved for the {reserved[0]}'s own "
            "stream in what is published. Rename the candidate."
        )
    for name, rule in zip(names, rules):
        if not isinstance(rule, SleeveRule):
            raise ValidationError(
                f"{what}: candidate {name!r} is a {type(rule).__name__}, not a "
                "SleeveRule."
            )
    return names, rules


def walk_candidates(
    frame: pd.DataFrame,
    candidates: Sequence[Tuple[str, SleeveRule]],
    *,
    estimation_window: int,
    rebalance_every: int,
    cost_bps: float = 0.0,
    cash_rate: float = 0.0,
    periods_per_year: int = 252,
    what: str = "select_allocation_rule",
    minimum: int = 2,
) -> Dict[str, Any]:
    """
    Every candidate and the equal-weight rule walked forward on the same
    rows with the same settings.

    A regime candidate's state column rides in that candidate's windows
    only; the sleeves are the columns that are no candidate's state.
    Returns `names`, `sleeves` (the sleeve frame), the candidates' `paths`
    in order and the `benchmark` path.
    """
    names, rules = _checked_candidates(candidates, what, minimum)
    columns = [str(c) for c in frame.columns]
    states = []
    for name, rule in zip(names, rules):
        state = None if rule.regime is None else rule.regime.state_column
        if state is None:
            continue
        if state not in columns:
            raise ValidationError(
                f"{what}: candidate {name!r} names regime state_column="
                f"{state!r}, which is not a column of the returns "
                f"{columns[:8]}. Name one of them, or omit state_column."
            )
        states.append(state)
    keep = [i for i, c in enumerate(columns) if c not in states]
    sleeves = frame if len(keep) == len(columns) else frame.iloc[:, keep]
    if sleeves.shape[1] < 2:
        raise ValidationError(
            f"{what}: {sleeves.shape[1]} sleeve(s) once the regime state "
            f"columns {sorted(set(states))} are set aside; an allocation needs "
            "at least two."
        )
    settings = dict(
        estimation_window=estimation_window,
        rebalance_every=rebalance_every,
        cost_bps=cost_bps,
        cash_rate=cash_rate,
        periods_per_year=periods_per_year,
        what=what,
    )
    paths = []
    for name, rule in zip(names, rules):
        state = None if rule.regime is None else rule.regime.state_column
        own = sleeves if state is None else frame.iloc[:, keep + [columns.index(state)]]
        try:
            paths.append(walk_sleeve_rule(own, rule, **settings))
        except ValidationError as exc:
            raise ValidationError(f"{what}: candidate {name!r}: {exc}") from exc
    benchmark = walk_sleeve_rule(sleeves, SleeveRule(method="equal"), **settings)
    return {"names": names, "sleeves": sleeves, "paths": paths, "benchmark": benchmark}


# ── the selector ─────────────────────────────────────────────────────────


def _key(score: float) -> float:
    """A score to rank by: an undefined Sharpe ranks below every other."""
    return score if math.isfinite(score) else -math.inf


def _choose(
    scores: np.ndarray,
    incumbent: Tuple[int, ...],
    *,
    size: int,
    margin: float,
) -> Tuple[int, ...]:
    """
    The candidates held after this rebalance: the incumbent set, filled to
    `size` with the best outsiders, then each worst member swapped for the
    best outsider while the outsider's score beats the member's by more
    than `margin`. Ties go to the earlier candidate. A candidate whose
    Sharpe is undefined is never brought in, and one held is the first to
    leave.
    """
    order = sorted(range(scores.size), key=lambda j: (-_key(scores[j]), j))
    members = list(incumbent)
    for j in order:
        if len(members) >= size:
            break
        if j not in members and math.isfinite(scores[j]):
            members.append(j)
    while True:
        outside = [j for j in order if j not in members and math.isfinite(scores[j])]
        if not outside:
            break
        challenger = outside[0]
        worst = min(members, key=lambda j: (_key(scores[j]), -j))
        if _key(scores[challenger]) > _key(scores[worst]) + margin:
            members[members.index(worst)] = challenger
        else:
            break
    return tuple(sorted(members))


def run_selector(
    values: np.ndarray,
    paths: Sequence[Dict[str, Any]],
    *,
    estimation_window: int,
    rebalance_every: int,
    selection_window: int,
    switch_margin: float,
    size: int = 1,
    cost_bps: float = 0.0,
    cash_rate: float = 0.0,
    periods_per_year: int = 252,
    what: str = "select_allocation_rule",
) -> Dict[str, Any]:
    """
    The selector's own book over the sleeve returns `values` (rows are
    periods, columns sleeves), given the candidates' walks `paths` on the
    same rows.

    Rebalances fall on the candidates' rows W, W + k, ...; at each the
    selector scores the candidates on their out-of-sample returns in the
    `selection_window` rows before it (none before the first L rows of
    history, when it holds the first candidate), keeps or changes what it
    holds (`_choose`), and trades to the equal blend of the chosen
    candidates' books at that row. The book's arithmetic is
    `walk_rule_forward`'s, expression for expression, so a selector that
    always holds one candidate earns that candidate's returns to the bit.

    Returns `returns` (one per row from W on), the chosen `choices` (a
    tuple of candidate positions per rebalance), `selecting` (whether that
    rebalance scored anything), the `targets` traded to (an array, one row
    per rebalance), each rebalance's `turnover` and `cost`, and
    `first_scored` (the row of the first rebalance that scored, or None).
    """
    W = estimation_window
    k = rebalance_every
    L = selection_window
    T, n = values.shape
    m = len(paths)
    rf = cash_rate / periods_per_year
    kappa = cost_bps / 10_000.0
    streams = np.column_stack([p["returns"] for p in paths])
    held = [p["held"].to_numpy(dtype=float) for p in paths]
    held_cash = [np.asarray(p["held_cash"], dtype=float) for p in paths]
    weights = np.zeros(n)
    cash = 1.0
    returns = np.empty(T - W)
    choices: List[Tuple[int, ...]] = []
    selecting: List[bool] = []
    targets: List[np.ndarray] = []
    turnover: List[float] = []
    costs: List[float] = []
    incumbent: Tuple[int, ...] = (0,)
    first_scored: Optional[int] = None
    for t in range(W, T):
        cost = 0.0
        if (t - W) % k == 0:
            i = (t - W) // k
            scored = t - L >= W
            if scored:
                if first_scored is None:
                    first_scored = t
                window = streams[t - L - W : t - W] - rf
                with single_threaded_blas():
                    scores = np.array(
                        [
                            annualized_sharpe(window[:, j], periods_per_year)
                            for j in range(m)
                        ]
                    )
                incumbent = _choose(scores, incumbent, size=size, margin=switch_margin)
            if len(incumbent) == 1:
                target = held[incumbent[0]][i].copy()
                target_cash = float(held_cash[incumbent[0]][i])
            else:
                target = np.vstack([held[j][i] for j in incumbent]).mean(axis=0)
                target_cash = float(np.mean([held_cash[j][i] for j in incumbent]))
            traded = math.fsum(np.abs(target - weights))
            cost = kappa * traded
            weights = target.copy()
            cash = target_cash
            choices.append(incumbent)
            selecting.append(scored)
            targets.append(target)
            turnover.append(traded)
            costs.append(cost)
        r = values[t]
        # walk_rule_forward's expressions, so one candidate is its bits.
        gross = math.fsum(weights * r) + cash * rf
        growth = (1.0 - cost) * (1.0 + gross)
        if not growth > 0.0:
            raise ValidationError(
                f"{what}: on row {t} the selector's book lost all of its "
                f"capital (a period return of {growth - 1.0:.4g}). Lower the "
                "candidates' leverage."
            )
        returns[t - W] = growth - 1.0
        weights = weights * (1.0 + r) / (1.0 + gross)
        cash = cash * (1.0 + rf) / (1.0 + gross)
    return {
        "returns": returns,
        "choices": choices,
        "selecting": selecting,
        "targets": np.vstack(targets) if targets else np.empty((0, n)),
        "turnover": turnover,
        "cost": costs,
        "first_scored": first_scored,
    }


# ── the report ───────────────────────────────────────────────────────────


def _scored_path(
    path: Dict[str, Any], start: int, first_date: Any, dates: pd.Index
) -> Dict[str, Any]:
    """
    A walk's path cut to the scored rows: its returns from `start` on, and
    the rebalances from the one on `first_date`. The turnover and costs
    belong to the fitted rebalances and the unfitted list to the others,
    each in rebalance order, so both are placed by position in the list of
    every rebalance (`held`'s index).
    """
    rebalances = list(path["held"].index)
    first = rebalances.index(first_date)
    fitted = set(path["targets"].index)
    fitted_at = [p for p, d in enumerate(rebalances) if d in fitted]
    unfitted_at = [p for p, d in enumerate(rebalances) if d not in fitted]
    return {
        "returns": path["returns"][start:],
        "dates": dates[start:],
        "turnover": [x for x, p in zip(path["turnover"], fitted_at) if p >= first],
        "cost": [x for x, p in zip(path["cost"], fitted_at) if p >= first],
        "unfitted": [u for u, p in zip(path["unfitted"], unfitted_at) if p >= first],
    }


def _deflated(
    returns: np.ndarray, n_trials: int, periods_per_year: int, who: str
) -> Tuple[Dict[str, Any], Optional[str]]:
    try:
        return (
            deflated_sharpe_ratio(
                pd.Series(returns),
                n_trials=n_trials,
                periods_per_year=periods_per_year,
            ),
            None,
        )
    except ValidationError as exc:
        return {}, f"no deflated Sharpe ratio for the {who}: {exc}"


def select_allocation_rule(
    frame: pd.DataFrame,
    candidates: Sequence[Tuple[str, SleeveRule]],
    *,
    n_trials: int,
    estimation_window: int = 252,
    rebalance_every: int = 21,
    cost_bps: float = 0.0,
    cash_rate: float = 0.0,
    periods_per_year: int = 252,
    selection_window: int = DEFAULT_SELECTION_WINDOW,
    switch_margin: float = DEFAULT_SWITCH_MARGIN,
    combine: str = DEFAULT_COMBINE,
    top_k: Optional[int] = None,
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    random_seed: int = 0,
    confidence: float = 0.95,
    what: str = "select_allocation_rule",
) -> Dict[str, Any]:
    """
    A selector among the `candidates` -- (name, rule) pairs -- walked
    forward over the aligned sleeve returns `frame`, against equal weight
    and against the candidate hindsight would have picked.

    Returns the scored `dates` and the selector's, the benchmark's and each
    candidate's returns on them (`selector_returns`, `benchmark_returns`,
    `candidate_returns`, a frame), the full streams from the first fit on
    (`streams`, a frame: the warm-up included), the selector's targets at
    each rebalance (`targets`, a frame) and `choices` (date, names,
    whether it scored); the summaries of the `selector`, `benchmark` and
    each candidate (`candidates`); `times_chosen`, `n_switches`; the
    `sharpe_difference` selector minus benchmark with its interval and
    p-value; the `hindsight_best` candidate and `hindsight_gap` (its Sharpe
    minus the selector's); the `pbo` report; the `deflated` Sharpe ratio of
    the selector at `n_trials` and `hindsight_deflated` of the winner at
    n_trials times the number of candidates; and warnings.
    """
    n_trials = require_positive_int(n_trials, "n_trials", what)
    periods_per_year = require_periods_per_year(periods_per_year, what)
    window = require_positive_int(estimation_window, "estimation_window", what)
    every = require_positive_int(rebalance_every, "rebalance_every", what)
    L = require_positive_int(selection_window, "selection_window", what)
    if L < 2:
        raise ValidationError(
            f"{what}: selection_window={L} scores each candidate on one period, "
            "which has no Sharpe ratio. Give at least 2; "
            f"{DEFAULT_SELECTION_WINDOW} is the default."
        )
    switch_margin = require_finite_scalar(
        switch_margin, "switch_margin", what, minimum=0.0
    )
    cost_bps = require_finite_scalar(cost_bps, "cost_bps", what, minimum=0.0)
    cash_rate = require_finite_scalar(cash_rate, "cash_rate", what)
    if combine not in COMBINE:
        raise ValidationError(
            f"{what}: combine={combine!r} is not one of {list(COMBINE)}."
        )
    if not 0.0 < confidence < 1.0:
        raise ValidationError(f"{what}: confidence={confidence!r} must be in (0, 1).")
    m = len(_checked_candidates(candidates, what, 2)[0])
    size = 1
    if combine == "top_k":
        size = (
            default_top_k(m)
            if top_k is None
            else require_positive_int(top_k, "top_k", what)
        )
        if not size < m:
            raise ValidationError(
                f"{what}: top_k={size} with {m} candidates blends them all at "
                "every rebalance, which selects nothing. Give a top_k below "
                f"{m}, or leave it out for the better half ({default_top_k(m)})."
            )
    elif top_k is not None:
        raise ValidationError(
            f"{what}: top_k={top_k} is read by combine='top_k' only, and "
            "combine='best'. Drop top_k, or use combine='top_k'."
        )
    T = int(frame.shape[0])
    if T - window < MIN_OUT_OF_SAMPLE:
        raise ValidationError(
            f"{what}: {T} shared periods and an estimation_window of {window} "
            f"leave {max(T - window, 0)} out of sample, fewer than "
            f"{MIN_OUT_OF_SAMPLE}, before any selection. Shorten the window or "
            "give longer streams."
        )
    first = window + -(-L // every) * every
    if T - first < MIN_OUT_OF_SAMPLE:
        most = max(T - window - MIN_OUT_OF_SAMPLE, 0)
        raise ValidationError(
            f"{what}: selection_window={L} needs {L} out-of-sample periods of "
            f"each candidate before the first choice, which falls on row "
            f"{first} of {T} (estimation_window={window}, rebalance_every="
            f"{every}), leaving {max(T - first, 0)} periods to score, fewer "
            f"than {MIN_OUT_OF_SAMPLE}. Shorten selection_window (to at most "
            f"about {most}) or the estimation window, or give longer streams."
        )

    walks = walk_candidates(
        frame,
        candidates,
        estimation_window=window,
        rebalance_every=every,
        cost_bps=cost_bps,
        cash_rate=cash_rate,
        periods_per_year=periods_per_year,
        what=what,
    )
    names = walks["names"]
    sleeves = walks["sleeves"]
    paths = walks["paths"]
    bench = walks["benchmark"]
    values = np.ascontiguousarray(sleeves.to_numpy(dtype=float))
    run = run_selector(
        values,
        paths,
        estimation_window=window,
        rebalance_every=every,
        selection_window=L,
        switch_margin=switch_margin,
        size=size,
        cost_bps=cost_bps,
        cash_rate=cash_rate,
        periods_per_year=periods_per_year,
        what=what,
    )
    first_row = run["first_scored"]
    assert first_row is not None and first_row == first  # by the check above
    start = first_row - window
    dates = bench["dates"]
    first_date = sleeves.index[first_row]
    rf = cash_rate / periods_per_year
    warnings: List[str] = []

    rebalance_dates = list(bench["held"].index)
    every_rebalance = pd.DataFrame(run["targets"], index=rebalance_dates)
    selector_path = {
        "returns": run["returns"],
        "dates": dates,
        "turnover": run["turnover"],
        "cost": run["cost"],
        "unfitted": [],
        "targets": every_rebalance,
        "held": every_rebalance,
    }
    scored = {
        "selector": _scored_path(selector_path, start, first_date, dates),
        "benchmark": _scored_path(bench, start, first_date, dates),
    }
    candidate_scored = [_scored_path(p, start, first_date, dates) for p in paths]
    summaries = {
        name: _summary(path, rf, periods_per_year) for name, path in scored.items()
    }
    candidate_summaries = {
        name: _summary(path, rf, periods_per_year)
        for name, path in zip(names, candidate_scored)
    }

    selector_returns = scored["selector"]["returns"]
    bench_returns = scored["benchmark"]["returns"]
    candidate_frame = pd.DataFrame(
        np.column_stack([p["returns"] for p in candidate_scored]),
        index=dates[start:],
        columns=names,
    )
    n_scored = int(selector_returns.size)
    block = _block_size(n_scored, block_size)
    with single_threaded_blas():
        difference = _paired_sharpe_difference(
            selector_returns - rf,
            bench_returns - rf,
            periods_per_year=periods_per_year,
            n_bootstrap=int(n_bootstrap),
            block_size=block,
            random_seed=random_seed,
            confidence=confidence,
        )

    # What was chosen, at the rebalances that scored.
    picks = [c for c, s in zip(run["choices"], run["selecting"]) if s]
    times_chosen = {name: 0 for name in names}
    for chosen in picks:
        for j in chosen:
            times_chosen[names[j]] += 1
    held_before = [c for c, s in zip(run["choices"], run["selecting"]) if not s]
    previous = held_before[-1] if held_before else None
    switches = 0
    for chosen in picks:
        if previous is not None and chosen != previous:
            switches += 1
        previous = chosen

    sharpes = np.array([candidate_summaries[n]["sharpe"] for n in names])
    hindsight: Optional[str] = None
    gap = float("nan")
    hindsight_deflated: Dict[str, Any] = {}
    if np.isfinite(sharpes).any():
        best = int(np.nanargmax(np.where(np.isfinite(sharpes), sharpes, -np.inf)))
        hindsight = names[best]
        gap = float(sharpes[best]) - summaries["selector"]["sharpe"]
        hindsight_deflated, why = _deflated(
            candidate_frame[hindsight].to_numpy() - rf,
            n_trials * len(names),
            periods_per_year,
            f"hindsight best {hindsight!r}",
        )
        if why:
            warnings.append(f"{what}: {why}")
    deflated, why = _deflated(
        selector_returns - rf, n_trials, periods_per_year, "selector"
    )
    if why:
        warnings.append(f"{what}: {why}")

    pbo: Optional[Dict[str, Any]] = None
    try:
        with single_threaded_blas():
            pbo = probability_of_backtest_overfitting(
                candidate_frame - rf, n_splits=PBO_SPLITS
            )
    except ValidationError as exc:
        warnings.append(f"{what}: no probability of backtest overfitting: {exc}")

    # ── what the caller is told ──
    if held_before:
        warnings.append(
            f"{what}: the selector held the first candidate, {names[0]!r}, for "
            f"the {start} periods from {_label(dates[0])} until "
            f"{_label(first_date)}, before any candidate had "
            f"selection_window={L} periods of out-of-sample history. Those "
            "periods are not scored."
        )
    unfitted = {
        name: len(p["unfitted"])
        for name, p in zip(names, candidate_scored)
        if p["unfitted"]
    }
    if unfitted:
        warnings.append(
            f"{what}: some candidates could not be fitted at some scored "
            f"rebalances and held their books there: {unfitted}."
        )
    if math.isfinite(difference["lower"]) and (
        difference["lower"] <= 0.0 <= difference["upper"]
    ):
        warnings.append(
            f"{what}: the {confidence:.0%} interval of the selector's Sharpe "
            f"difference against equal weight, [{difference['lower']:.2f}, "
            f"{difference['upper']:.2f}], contains zero: out of sample the "
            "selection is not distinguishable from equal weight."
        )
    if hindsight is not None and math.isfinite(gap):
        warnings.append(
            f"{what}: HINDSIGHT: {hindsight!r} had the best Sharpe ratio over "
            f"the scored periods, {float(sharpes[names.index(hindsight)]):.2f} "
            f"against the selector's {summaries['selector']['sharpe']:.2f}. That "
            "is what a backtest that picked the winner afterwards would report; "
            "a claim for it must survive the deflation at "
            f"{n_trials * len(names)} trials."
        )
    if pbo is not None and pbo["pbo"] >= 0.5:
        warnings.append(
            f"{what}: the probability of backtest overfitting over the "
            f"candidates is {pbo['pbo']:.2f}: the candidate with the best record "
            "on half the periods lands in the bottom half on the other half at "
            "least as often as not, so choosing among them by their record is "
            "no better than choosing at random."
        )
    warnings += [
        w for w in deflated.get("warnings", []) if "deflated Sharpe probability" in w
    ]

    targets = pd.DataFrame(
        run["targets"],
        index=pd.Index(rebalance_dates, name=sleeves.index.name),
        columns=[str(c) for c in sleeves.columns],
    )
    streams = pd.DataFrame(
        {"selector": run["returns"], "benchmark": bench["returns"]},
        index=dates,
    )
    for name, path in zip(names, paths):
        streams[name] = path["returns"]
    return {
        "names": names,
        "dates": dates[start:],
        "first_scored_date": first_date,
        "n_warm_up": int(start),
        "selector_returns": selector_returns,
        "benchmark_returns": bench_returns,
        "candidate_returns": candidate_frame,
        "streams": streams,
        "targets": targets,
        "choices": [
            (d, [names[j] for j in c], bool(s))
            for d, c, s in zip(rebalance_dates, run["choices"], run["selecting"])
        ],
        "selector": summaries["selector"],
        "benchmark": summaries["benchmark"],
        "candidates": candidate_summaries,
        "times_chosen": times_chosen,
        "n_selections": len(picks),
        "n_switches": switches,
        "last_choice": [names[j] for j in run["choices"][-1]],
        "sharpe_difference": difference,
        "confidence": confidence,
        "block_size": block,
        "n_bootstrap": int(n_bootstrap),
        "hindsight_best": hindsight,
        "hindsight_gap": gap,
        "pbo": pbo,
        "deflated": deflated,
        "hindsight_deflated": hindsight_deflated,
        "n_trials": n_trials,
        "n_out_of_sample": n_scored,
        "selection_window": L,
        "switch_margin": switch_margin,
        "combine": combine,
        "top_k": size if combine == "top_k" else None,
        "warnings": warnings,
    }
