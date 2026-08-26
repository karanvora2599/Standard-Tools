"""
What the equity curve does not say.

A backtest reports one path. That path happened once, in one order, and
almost every question worth asking about it is a question about the paths
that did not happen: how much of the outcome was sequence rather than edge,
whether the losses arrive in runs a real account could not sit through, and
whether a strategy with this many trades and this win rate is
distinguishable from a coin.

WHY RESAMPLING TRADES IS DIFFERENT FROM RESAMPLING RETURNS. A trade is the
natural unit of a strategy's decision -- one entry, one exit, one bet. The
returns inside it are not independent draws, they are the anatomy of a
single decision. Resampling trades therefore answers "what if the same edge
had dealt these bets in a different order", which is the question, while
resampling daily returns answers something closer to "what if this were a
different strategy".

WHAT THIS CANNOT DO, said once. Reordering trades destroys any real
dependence BETWEEN them -- and there usually is some, because a strategy
that loses in a regime loses several times in that regime. The resampled
distribution is therefore optimistic about clustering, which is exactly why
`analyze_trade_clustering` measures the clustering in the ORIGINAL order
rather than trusting the bootstrap to reproduce it.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _trades(values: Sequence[float], who: str, minimum: int = 20) -> np.ndarray:
    array = np.asarray([float(v) for v in values], dtype=float)
    array = array[np.isfinite(array)]
    if array.size < minimum:
        raise ValidationError(
            f"{who}: {array.size} usable trades, and this needs at least "
            f"{minimum}. Below that every statistic here is a description of "
            "a handful of events rather than an estimate."
        )
    return array


def monte_carlo_trade_paths(
    trade_returns: Sequence[float],
    *,
    n_paths: int = 2000,
    seed: int = 0,
    starting_equity: float = 1.0,
) -> Dict[str, Any]:
    """
    The distribution of outcomes the same edge could have produced, by
    reshuffling the trades.

    THE BACKTEST'S DRAWDOWN IS ONE DRAW. A strategy with a 20% maximum
    drawdown in its backtest does not have a 20% maximum drawdown; it has a
    DISTRIBUTION of them, and the backtested one is a single sample from it.
    Reordering the same trades -- same edge, same win rate, same trade sizes,
    different sequence -- produces that distribution directly, and the
    backtested drawdown routinely sits near the middle of it while the 95th
    percentile is half again as deep.

    THAT PERCENTILE IS THE NUMBER TO SIZE ON. A position sized so the
    backtested drawdown is survivable is sized on the median outcome, and
    half of all realizations are worse.

    RESHUFFLING, NOT RESAMPLING WITH REPLACEMENT. Every path here contains
    exactly the same trades in a different order, so the total return is
    identical across paths and only the PATH differs. That isolates sequence
    risk from edge uncertainty, which is the point -- a bootstrap with
    replacement mixes the two, and the drawdown distribution it produces is
    then partly about having drawn a different strategy.
    """
    array = _trades(trade_returns, "monte_carlo_trade_paths")
    n_paths = max(100, int(n_paths))
    rng = np.random.default_rng(int(seed))

    def _path_stats(order: np.ndarray) -> tuple:
        equity = starting_equity * np.cumprod(1.0 + order)
        peak = np.maximum.accumulate(equity)
        drawdown = equity / peak - 1.0
        underwater = float((drawdown < 0).mean())
        return float(drawdown.min()), float(equity[-1]), underwater

    observed = _path_stats(array)
    drawdowns = np.empty(n_paths)
    finals = np.empty(n_paths)
    underwaters = np.empty(n_paths)
    for i in range(n_paths):
        drawdowns[i], finals[i], underwaters[i] = _path_stats(rng.permutation(array))

    percentile_of_observed = float((drawdowns > observed[0]).mean() * 100.0)

    warnings: List[str] = []
    p95 = float(np.percentile(drawdowns, 5))  # 5th pct of a negative number
    if observed[0] > p95:
        warnings.append(
            f"The backtested drawdown of {observed[0]:.1%} sits at the "
            f"{percentile_of_observed:.0f}th percentile of what this same "
            f"edge produces in a different order. The 95th percentile is "
            f"{p95:.1%}. Sizing so the backtested drawdown is survivable is "
            "sizing on a lucky ordering."
        )
    warnings.append(
        "Paths are RESHUFFLES, so every one holds the same trades and ends "
        "at the same total return. This isolates SEQUENCE risk from edge "
        "uncertainty; it says nothing about whether the edge is real."
    )
    warnings.append(
        "Reordering destroys real dependence BETWEEN trades, and there "
        "usually is some -- a strategy that loses in a regime loses several "
        "times in it. The reshuffled distribution is therefore optimistic "
        "about clustering."
    )
    if array.size < 50:
        warnings.append(
            f"{array.size} trades. The reshuffle can only produce orderings "
            "of what is there, so a short trade history gives a narrow and "
            "overconfident distribution."
        )

    return {
        "n_trades": int(array.size),
        "n_paths": n_paths,
        "observed_max_drawdown": observed[0],
        "observed_final_equity": observed[1],
        "observed_drawdown_percentile": percentile_of_observed,
        "median_max_drawdown": float(np.median(drawdowns)),
        "p05_max_drawdown": float(np.percentile(drawdowns, 5)),
        "p95_max_drawdown": float(np.percentile(drawdowns, 95)),
        "worst_max_drawdown": float(drawdowns.min()),
        "mean_fraction_underwater": float(underwaters.mean()),
        "final_equity": float(finals[0]),
        "warnings": warnings,
    }


def analyze_trade_clustering(trade_returns: Sequence[float]) -> Dict[str, Any]:
    """
    Whether wins and losses arrive in RUNS, measured in the original order.

    THE ORDER IS THE DATA HERE, which is what separates this from every
    other statistic about a trade list. A 55% win rate is survivable if the
    losses are scattered and unholdable if they arrive eleven in a row --
    and the win rate is identical in both cases.

    THE TEST IS A RUNS TEST. Under the null that wins and losses are
    independent draws, the number of runs has a known mean and variance, so
    the observed count becomes a z-score. FEWER runs than expected means
    clustering (streaks); MORE means alternation, which is rarer and usually
    means the strategy is reacting to its own last outcome.

    WHY IT MATTERS MORE THAN IT LOOKS. Clustering is what makes a drawdown
    deep, and it is also what a reshuffling bootstrap destroys -- so the
    Monte Carlo drawdown distribution is optimistic by exactly the amount
    measured here. Read the two together.
    """
    array = _trades(trade_returns, "analyze_trade_clustering")
    signs = array > 0
    wins = int(signs.sum())
    losses = int(array.size - wins)
    if wins == 0 or losses == 0:
        raise ValidationError(
            "analyze_trade_clustering: every trade has the same sign, so "
            "there are no runs to test."
        )

    runs = int(1 + (signs[1:] != signs[:-1]).sum())
    n = array.size
    expected = 2.0 * wins * losses / n + 1.0
    variance = 2.0 * wins * losses * (2.0 * wins * losses - n) / (n * n * (n - 1.0))
    z = (runs - expected) / math.sqrt(variance) if variance > 0 else float("nan")
    p_value = (
        float(math.erfc(abs(z) / math.sqrt(2.0))) if math.isfinite(z) else float("nan")
    )

    # The longest actual streaks, which is what a person experiences.
    longest_loss = longest_win = current = 0
    previous: Optional[bool] = None
    for sign in signs:
        if sign == previous:
            current += 1
        else:
            current, previous = 1, bool(sign)
        if sign:
            longest_win = max(longest_win, current)
        else:
            longest_loss = max(longest_loss, current)

    clustered = bool(math.isfinite(z) and z < -1.96)
    alternating = bool(math.isfinite(z) and z > 1.96)

    warnings: List[str] = []
    if clustered:
        warnings.append(
            f"Wins and losses CLUSTER (runs z = {z:.2f}, p = {p_value:.4f}): "
            f"there are fewer runs than independence predicts, and the "
            f"longest losing streak is {longest_loss} trades. Clustering is "
            "what makes a drawdown deep, and it is exactly what a "
            "trade-reshuffling Monte Carlo destroys -- so read that "
            "drawdown distribution as optimistic."
        )
    elif alternating:
        warnings.append(
            f"Wins and losses ALTERNATE more than independence predicts "
            f"(runs z = {z:.2f}). That is rarer than clustering and usually "
            "means the strategy is reacting to its own last outcome -- "
            "check for a position-sizing or state-carrying bug."
        )
    else:
        warnings.append(
            f"Runs z = {z:.2f}: no detectable clustering. Wins and losses "
            "are consistent with independent draws at this sample size."
        )
    warnings.append(
        f"The longest losing streak was {longest_loss} trades and the "
        f"longest winning streak {longest_win}. A win rate says nothing "
        "about either, and the streak is what a real account has to sit "
        "through."
    )

    return {
        "n_trades": int(n),
        "n_wins": wins,
        "n_losses": losses,
        "win_rate": float(wins / n),
        "n_runs": runs,
        "expected_runs": float(expected),
        "runs_z_score": float(z) if math.isfinite(z) else None,
        "p_value": float(p_value) if math.isfinite(p_value) else None,
        "clustered": clustered,
        "alternating": alternating,
        "longest_losing_streak": longest_loss,
        "longest_winning_streak": longest_win,
        "warnings": warnings,
    }


def compare_against_random(
    trade_returns: Sequence[float],
    *,
    n_simulations: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Whether this strategy beats a coin flip that traded the same instrument
    the same number of times.

    THE NULL THAT MATTERS. A strategy is usually compared against zero, or
    against buy-and-hold. Neither is the right comparison after a search:
    the question is whether the ENTRY DECISIONS added anything over choosing
    at random, and a strategy can beat zero purely by holding a rising asset
    with no skill in the timing at all.

    THE CONSTRUCTION. Each simulation keeps the same trade OUTCOMES but
    assigns their signs at random with the same overall win rate, which
    preserves the return distribution and destroys any relationship between
    the decision and the result. If the real strategy's Sharpe sits inside
    that distribution, the entries were not doing the work.

    THE LIMITATION IS REAL AND WORTH STATING: preserving the win rate means
    the null already has the strategy's win rate, so this tests whether the
    SEQUENCING and SIZING of wins and losses beat random -- not whether the
    win rate itself does. A strategy whose entire edge is a 55% win rate on
    symmetric bets will look indistinguishable from this null, correctly:
    that edge lives in the win rate, which the null was given.

    IT IS CONSERVATIVE, measured. Because the null keeps both the magnitudes
    and the win rate, it is a tight comparison: on genuinely skill-free
    trades it fired 1 time in 150 against a nominal 5%. That direction is
    the safe one -- it will not manufacture significance -- but it means a
    non-rejection here is weaker evidence of no skill than a nominal 5%
    test would suggest. It retains power against the case it was built for:
    a strategy whose wins are systematically larger than its losses came
    back at p < 0.001.
    """
    array = _trades(trade_returns, "compare_against_random")
    n_simulations = max(100, int(n_simulations))
    rng = np.random.default_rng(int(seed))

    magnitudes = np.abs(array)
    win_rate = float((array > 0).mean())

    def _sharpe(values: np.ndarray) -> float:
        std = float(values.std(ddof=1))
        if std <= 0:
            return float("nan")
        return float(values.mean() / std)

    observed = _sharpe(array)
    if not math.isfinite(observed):
        raise ValidationError(
            "compare_against_random: the trade returns have no dispersion, "
            "so there is no ratio to compare."
        )

    simulated = np.empty(n_simulations)
    for i in range(n_simulations):
        signs = np.where(rng.random(array.size) < win_rate, 1.0, -1.0)
        simulated[i] = _sharpe(rng.permutation(magnitudes) * signs)
    usable = simulated[np.isfinite(simulated)]
    p_value = float((usable >= observed).mean())

    warnings: List[str] = []
    if p_value > 0.10:
        warnings.append(
            f"p = {p_value:.3f}: this strategy's per-trade Sharpe sits inside "
            "what random sign assignment produces at the same win rate. The "
            "SEQUENCING and SIZING of its wins and losses are not adding "
            "anything detectable."
        )
    warnings.append(
        "The null KEEPS the strategy's win rate, so this does not test "
        "whether the win rate itself beats chance -- it tests whether "
        "anything beyond it does. A strategy whose entire edge is a 55% win "
        "rate on symmetric bets will look indistinguishable here, and that "
        "is the correct answer to the question asked."
    )
    warnings.append(
        "Beating zero is not the same as beating random. A strategy can beat "
        "zero purely by holding a rising asset with no skill in the timing."
    )

    return {
        "n_trades": int(array.size),
        "n_simulations": n_simulations,
        "win_rate": win_rate,
        "observed_per_trade_sharpe": observed,
        "random_median": float(np.median(usable)),
        "random_p95": float(np.percentile(usable, 95)),
        "p_value": p_value,
        "beats_random_at_05": bool(p_value < 0.05),
        "warnings": warnings,
    }


def exposure_attribution(
    returns: Sequence[float],
    exposure: Sequence[float],
    *,
    periods_per_year: int = TRADING_DAYS,
) -> Dict[str, Any]:
    """
    How much of the return came from being RIGHT versus from being INVESTED.

    THE DECOMPOSITION. A strategy's return is the product of its exposure
    and the market's move. Split it and you can see whether the P&L came
    from timing -- holding more before up moves -- or simply from average
    exposure to an asset that rose. The second is beta with extra steps, and
    it is available for a management fee of zero.

    THE TIMING TERM IS THE COVARIANCE between exposure and the subsequent
    return. Positive means the strategy was larger before good periods,
    which is the only part of the result that is actually skill. It is
    usually far smaller than people expect, and often negative in strategies
    that look profitable.

    TIME IN MARKET IS REPORTED because it changes what the Sharpe means. A
    strategy invested 20% of the time with a Sharpe of 1.5 and one invested
    100% of the time with a Sharpe of 1.5 are different propositions: the
    first has capital idle that has to earn something, and its Sharpe
    ignores that entirely.
    """
    r = np.asarray([float(v) for v in returns], dtype=float)
    e = np.asarray([float(v) for v in exposure], dtype=float)
    if r.size != e.size:
        raise ValidationError(
            f"exposure_attribution: {r.size} returns against {e.size} "
            "exposures. They must be parallel."
        )
    mask = np.isfinite(r) & np.isfinite(e)
    r, e = r[mask], e[mask]
    if r.size < 30:
        raise ValidationError(
            f"exposure_attribution: {r.size} usable observations, needs 30."
        )

    strategy = e * r
    mean_exposure = float(e.mean())
    # E[e*r] = E[e]E[r] + Cov(e, r): the passive part and the timing part.
    passive = mean_exposure * float(r.mean())
    timing = float(strategy.mean() - passive)
    total = float(strategy.mean())

    invested = float((np.abs(e) > 1e-12).mean())
    long_share = float((e > 0).mean())

    warnings: List[str] = []
    if total != 0 and timing / total < 0.2:
        warnings.append(
            f"Only {timing / total:.0%} of the mean return comes from TIMING "
            "-- being larger before good periods. The rest is average "
            "exposure to an asset that rose, which is beta and is available "
            "for nothing."
        )
    if timing < 0:
        # Fires on ANY negative timing, not only when the total is still
        # positive. The original condition was `timing < 0 < total`, which
        # went silent in the worse case -- a strategy whose timing is
        # negative AND whose total is negative had nothing said about it,
        # exactly when the diagnosis matters most.
        warnings.append(
            "The timing contribution is NEGATIVE: the strategy was "
            "systematically smaller before good periods and larger before "
            "bad ones."
            + (
                " It is profitable in spite of its timing, not because of it."
                if total > 0
                else " The timing is part of why it lost money -- inverting "
                "the exposure signal is the first thing to check."
            )
        )
    if invested < 0.5:
        warnings.append(
            f"Invested only {invested:.0%} of the time. A Sharpe computed on "
            "these returns ignores what the idle capital earns or costs, so "
            "it is not comparable with a fully-invested strategy's."
        )

    return {
        "n_observations": int(r.size),
        "mean_exposure": mean_exposure,
        "fraction_invested": invested,
        "fraction_long": long_share,
        "total_mean_return": total,
        "passive_contribution": passive,
        "timing_contribution": timing,
        "timing_share": float(timing / total) if total != 0 else None,
        "annualized_total": float(total * periods_per_year),
        "annualized_passive": float(passive * periods_per_year),
        "annualized_timing": float(timing * periods_per_year),
        "exposure_return_correlation": (
            float(np.corrcoef(e, r)[0, 1]) if e.std() > 0 and r.std() > 0 else None
        ),
        "warnings": warnings,
    }


def break_even_cost(
    trade_returns: Sequence[float],
    *,
    current_cost_bps: float = 0.0,
) -> Dict[str, Any]:
    """
    The per-trade cost at which this edge disappears.

    THE NUMBER EVERY BACKTEST SHOULD REPORT AND ALMOST NONE DO. A backtest
    is run at some assumed cost, and the assumption is usually inherited
    rather than chosen. What decides whether the result survives contact
    with a real broker is not whether it is profitable at 5 basis points --
    it is how far above 5 the break-even sits.

    THE RATIO IS THE ANSWER. A strategy breaking even at 8bp when you
    modelled 5bp has 1.6x of headroom, and a single bad fill, a widening
    spread or a venue change eats it. One breaking even at 80bp is robust to
    all of those. The rule of thumb worth carrying is that under about 2x
    the assumed cost, the backtest is a statement about the cost assumption
    rather than about the strategy.

    WHAT THIS DOES NOT MODEL: impact. The break-even is computed as a flat
    per-trade charge, so it answers "what fixed cost kills this" and not
    "what happens when I trade larger". Impact grows with size and is not
    flat, so a strategy with headroom here can still fail on capacity --
    `get_capacity_report` and `estimate_trade_cost` are the tools for that.
    """
    array = _trades(trade_returns, "break_even_cost")
    current = float(current_cost_bps) / 1e4

    gross_mean = float(array.mean()) + current
    if gross_mean <= 0:
        return {
            "n_trades": int(array.size),
            "current_cost_bps": float(current_cost_bps),
            "mean_return_gross": float(gross_mean),
            "break_even_cost_bps": 0.0,
            "headroom_multiple": 0.0,
            "warnings": [
                "The strategy does not make money even BEFORE costs -- the "
                "gross mean trade return is non-positive. There is no "
                "break-even cost to report."
            ],
        }

    break_even_bps = float(gross_mean * 1e4)
    headroom = (
        float(break_even_bps / current_cost_bps) if current_cost_bps > 0 else None
    )

    # A cost applies to every trade, so its effect on the Sharpe is a shift
    # of the mean with the dispersion unchanged.
    std = float(array.std(ddof=1))
    gross = array + current
    sharpe_at = []
    for cost_bps in (0.0, 5.0, 10.0, 25.0, 50.0, 100.0):
        net = gross - cost_bps / 1e4
        sharpe_at.append(
            {
                "cost_bps": cost_bps,
                "mean_return": float(net.mean()),
                "per_trade_sharpe": float(net.mean() / std) if std > 0 else None,
                "profitable": bool(net.mean() > 0),
            }
        )

    warnings: List[str] = []
    if headroom is not None and headroom < 2:
        warnings.append(
            f"Break-even is {break_even_bps:.1f} bps against an assumed "
            f"{current_cost_bps:.1f} bps -- only {headroom:.1f}x of headroom. "
            "Below about 2x, the backtest is a statement about the cost "
            "ASSUMPTION rather than about the strategy: one bad fill, a "
            "widening spread or a venue change eats the edge."
        )
    elif headroom is not None:
        warnings.append(
            f"Break-even is {break_even_bps:.1f} bps against an assumed "
            f"{current_cost_bps:.1f} bps, {headroom:.1f}x of headroom."
        )
    warnings.append(
        "This is a FLAT per-trade charge and does not model impact. Impact "
        "grows with size, so a strategy with headroom here can still fail on "
        "capacity -- get_capacity_report and estimate_trade_cost answer that "
        "question."
    )
    return {
        "n_trades": int(array.size),
        "current_cost_bps": float(current_cost_bps),
        "mean_return_net": float(array.mean()),
        "mean_return_gross": float(gross_mean),
        "break_even_cost_bps": break_even_bps,
        "headroom_multiple": headroom,
        "sensitivity": sharpe_at,
        "warnings": warnings,
    }


__all__ = [
    "break_even_cost",
    "TRADING_DAYS",
    "analyze_trade_clustering",
    "compare_against_random",
    "exposure_attribution",
    "monte_carlo_trade_paths",
]
