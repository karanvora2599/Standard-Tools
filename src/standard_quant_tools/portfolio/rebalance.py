"""
Getting from the weights you have to the weights you want.

Every optimizer in this library returns a target weight vector, and every
one of them implicitly assumes you arrive there instantly and for free. You
do not. A transition has two costs that pull in opposite directions:

- **Trade fast** and you pay market impact, which grows with the fraction of
  a day's volume you take.
- **Trade slow** and you hold the old portfolio longer, which costs whatever
  the new one was supposed to earn.

There is no single right answer, so this returns the SCHEDULE and both costs
rather than one number. A caller who wants it done today can see what today
costs.

THE FAILURE THIS EXISTS TO SURFACE. An optimizer can emit a 5% target weight
in a name whose daily volume cannot support it. Nothing downstream notices:
the weight vector is valid, the backtest fills at the close, and the
position is simply never attainable in the size the model assumed. Here it
shows up as a name that has not converged by the end of the horizon, with
the number of days it would actually take.

THE IMPACT MODEL IS THE SQUARE ROOT LAW. Cost in basis points scales with
the square root of participation -- take 4x the volume and pay 2x the
impact per share, not 4x. It is the standard empirical form.

IT IS `sqrt_impact_bps` IN A DIFFERENT PARAMETERIZATION, AND THAT IS WORTH
BEING PRECISE ABOUT, because this docstring used to claim it was "the same
one `estimate_trade_cost` already uses" and a reader could check that
against the code and find a term missing. `backtest.costs.sqrt_impact_bps`
is `coefficient * volatility * sqrt(participation) * 1e4`. This is
`impact_coefficient * sqrt(participation)`. The volatility term is not
absent -- it is folded into the coefficient, exactly:

    impact_coefficient == coefficient * volatility * 1e4

The two agree to floating point under that identity, and a test pins it so
they cannot drift. The parameterization here is the one that fits the
question: this function plans a schedule across many names over many days
and is not given a per-name volatility, so it takes the one number a trader
can actually quote -- basis points at full participation.

WHAT THE DEFAULT IMPLIES. Under that identity `DEFAULT_IMPACT_COEFFICIENT`
of 10 bps is `coefficient=1.0` at a per-bar volatility of 0.001, which is
about 1.6% annualized. No equity is that quiet. Against a name at a more
ordinary 2% daily the canonical model charges 20x more, so the default is
a floor and not a central estimate. A caller who has a volatility should
convert it through the identity above and pass the result.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)

#: Basis points of impact at 100% participation, under the square root law.
#: Widely quoted between 5 and 20 bps for liquid US equities; 10 is the
#: middle and is a MODEL rather than a measurement. A caller who has
#: calibrated their own should pass it.
#:
#: In `sqrt_impact_bps` terms this is `coefficient * volatility * 1e4`, so
#: 10 is coefficient 1.0 at a per-bar volatility of 0.001 -- quieter than
#: any real equity. See the module docstring: treat it as a floor.
DEFAULT_IMPACT_COEFFICIENT = 10.0

#: Fraction of a day's volume this will plan to take. Above ~20% the square
#: root law stops describing reality -- you are no longer trading alongside
#: the day's flow, you are the day's flow -- so planning past it produces
#: cost estimates that are wrong in the optimistic direction.
DEFAULT_MAX_PARTICIPATION = 0.10


def _as_vector(weights: Dict[str, float], names: Sequence[str]) -> np.ndarray:
    return np.array([float(weights.get(n, 0.0)) for n in names], dtype=float)


def plan_rebalance(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    *,
    portfolio_value: float,
    adv: Optional[Dict[str, float]] = None,
    max_participation: float = DEFAULT_MAX_PARTICIPATION,
    max_days: int = 10,
    urgency: float = 0.5,
    impact_coefficient: float = DEFAULT_IMPACT_COEFFICIENT,
) -> Dict[str, Any]:
    """
    A day-by-day path from current holdings to target weights.

    `urgency` in [0, 1] is the only judgement call, and it is exposed rather
    than chosen: 1.0 trades as fast as the participation cap allows, which
    minimises the time spent holding the wrong portfolio and maximises
    impact. 0.0 spreads the transition evenly across `max_days`. The
    schedule reports the cost of the choice either way, so the caller can
    see the trade rather than be handed one side of it.

    `adv` is average daily dollar volume per name. Without it, participation
    cannot be computed, so the plan is returned WITHOUT impact costs and says
    so -- rather than defaulting to a number that would look like an estimate.
    """
    if portfolio_value <= 0:
        raise ValidationError("plan_rebalance: portfolio_value must be positive")
    if not 0.0 <= urgency <= 1.0:
        raise ValidationError("plan_rebalance: urgency must be between 0 and 1")
    if max_days < 1:
        raise ValidationError("plan_rebalance: max_days must be at least 1")
    if not 0.0 < max_participation <= 1.0:
        raise ValidationError("plan_rebalance: max_participation must be in (0, 1]")

    names = sorted(set(current_weights) | set(target_weights))
    if not names:
        raise ValidationError("plan_rebalance: no positions on either side")

    start = _as_vector(current_weights, names)
    target = _as_vector(target_weights, names)
    gap = target - start
    total_turnover = float(np.abs(gap).sum())

    if total_turnover == 0.0:
        return {
            "names": names,
            "n_days": 0,
            "schedule": [],
            "total_turnover": 0.0,
            "total_cost_bps": 0.0,
            "total_cost_dollars": 0.0,
            "converged": True,
            "unreachable": [],
            "warnings": ["already at target: nothing to trade"],
        }

    # How much of the remaining gap to close each day. urgency=1 closes it
    # all on day 1 (subject to the participation cap); urgency=0 spreads it
    # evenly. In between is a geometric decay, which is what a trader
    # actually does: more early, tapering.
    per_day = _daily_fractions(max_days, urgency)

    have_adv = adv is not None
    adv_vector = (
        np.array([float((adv or {}).get(n, 0.0)) for n in names], dtype=float)
        if have_adv
        else None
    )

    weights = start.copy()
    schedule: List[Dict[str, Any]] = []
    cumulative_dollars = 0.0
    cumulative_notional = 0.0

    for day, fraction in enumerate(per_day, start=1):
        remaining = target - weights
        wanted = remaining * fraction

        if have_adv:
            # Cap each name's daily trade at max_participation of its ADV.
            allowed_notional = adv_vector * max_participation
            wanted_notional = np.abs(wanted) * portfolio_value
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.where(
                    wanted_notional > 0,
                    np.minimum(
                        1.0, allowed_notional / np.maximum(wanted_notional, 1e-12)
                    ),
                    1.0,
                )
            wanted = wanted * scale

        traded_notional = np.abs(wanted) * portfolio_value
        if have_adv:
            with np.errstate(divide="ignore", invalid="ignore"):
                participation = np.where(
                    adv_vector > 0, traded_notional / adv_vector, np.nan
                )
            # Square root law, weighted by how much of the day's trading
            # each name represents. This is `sqrt_impact_bps` with the
            # volatility folded into the coefficient -- see the module
            # docstring for the identity, and test_impact_model_identity
            # for the assertion that keeps the two equal. It stays written
            # out here because that function is scalar and this is a
            # names-by-days sweep.
            impact = impact_coefficient * np.sqrt(np.nan_to_num(participation))
            day_bps = (
                float((impact * traded_notional).sum() / traded_notional.sum())
                if traded_notional.sum() > 0
                else 0.0
            )
        else:
            participation = np.full(len(names), np.nan)
            day_bps = float("nan")

        weights = weights + wanted
        # Accumulate DOLLARS, not rates. Summing each day's average basis
        # points would add five rates and call it a cost -- see the blended
        # figure below.
        day_dollars = (
            0.0 if np.isnan(day_bps) else float((impact * traded_notional).sum() / 1e4)
        )
        cumulative_dollars += day_dollars
        cumulative_notional += float(traded_notional.sum())
        distance = float(np.abs(target - weights).sum())

        schedule.append(
            {
                "day": day,
                "turnover": float(np.abs(wanted).sum()),
                "traded_notional": float(traded_notional.sum()),
                "max_participation_used": (
                    float(np.nanmax(participation)) if have_adv else None
                ),
                "impact_bps": None if np.isnan(day_bps) else day_bps,
                "impact_dollars": None if not have_adv else day_dollars,
                "cumulative_cost_dollars": (
                    None if not have_adv else float(cumulative_dollars)
                ),
                "distance_to_target": distance,
                "weights": {n: float(w) for n, w in zip(names, weights)},
            }
        )
        if distance < 1e-9:
            break

    residual = target - weights
    unreachable = _unreachable(
        names, residual, adv_vector, portfolio_value, max_participation, have_adv
    )

    return {
        "names": names,
        "n_days": len(schedule),
        "schedule": schedule,
        "total_turnover": total_turnover,
        # The BLENDED rate: total impact dollars over total notional traded.
        # Not the sum of the daily rates, which would add five rates and
        # report them as a cost -- and would say trading everything at once
        # is cheaper than spreading it, which is backwards under a square
        # root impact law.
        "total_cost_bps": (
            None
            if not have_adv
            else (
                float(cumulative_dollars / cumulative_notional * 1e4)
                if cumulative_notional > 0
                else 0.0
            )
        ),
        "total_cost_dollars": None if not have_adv else float(cumulative_dollars),
        "converged": bool(np.abs(residual).sum() < 1e-6),
        "residual_distance": float(np.abs(residual).sum()),
        "unreachable": unreachable,
        "warnings": _warnings(
            schedule, unreachable, have_adv, max_days, max_participation
        ),
    }


def _daily_fractions(max_days: int, urgency: float) -> List[float]:
    """
    What fraction of the REMAINING gap to close on each day.

    Expressed as a fraction of what is left rather than of the original gap,
    so a day that gets capped by liquidity does not silently drop the
    untraded part -- it rolls into the next day by construction.
    """
    if urgency >= 1.0:
        return [1.0] * max_days
    if urgency <= 0.0:
        # Even in absolute terms: 1/n of the original on day 1, then 1/(n-1)
        # of what remains, and so on.
        return [1.0 / (max_days - i) for i in range(max_days)]
    # Geometric taper between the two, which is what a trader does: more
    # early, less later.
    rate = 0.2 + 0.75 * urgency
    return [rate] * (max_days - 1) + [1.0]


def _unreachable(
    names, residual, adv_vector, portfolio_value, max_participation, have_adv
) -> List[Dict[str, Any]]:
    """
    Names still short of target when the horizon runs out, with how long
    they would actually need.

    This is the number an optimizer cannot tell you. A 5% target in a name
    whose ADV supports 0.4% a day is not a position you hold; it is a
    position you spend twelve days acquiring, during which it is not the
    portfolio that was optimized.
    """
    out = []
    for i, name in enumerate(names):
        if abs(residual[i]) < 1e-9:
            continue
        entry: Dict[str, Any] = {
            "name": name,
            "residual_weight": float(residual[i]),
        }
        if have_adv and adv_vector[i] > 0:
            daily_capacity = adv_vector[i] * max_participation / portfolio_value
            entry["days_needed"] = float(abs(residual[i]) / daily_capacity)
            entry["daily_capacity_weight"] = float(daily_capacity)
        out.append(entry)
    return sorted(out, key=lambda e: -abs(e["residual_weight"]))


def _warnings(
    schedule, unreachable, have_adv, max_days, max_participation
) -> List[str]:
    out: List[str] = []
    if not have_adv:
        out.append(
            "no `adv` was supplied, so participation and impact could not be "
            "computed and the schedule is a pure weight path. The costs are "
            "reported as null rather than as zero -- an unpriced transition "
            "is not a free one."
        )
    if unreachable:
        worst = unreachable[0]
        detail = (
            f" It would need about {worst['days_needed']:.0f} days at "
            f"{max_participation:.0%} participation."
            if "days_needed" in worst
            else ""
        )
        out.append(
            f"{len(unreachable)} name(s) did not reach target within "
            f"{max_days} days, the largest being {worst['name']!r} at "
            f"{worst['residual_weight']:+.2%} short.{detail} Until then the "
            "portfolio you hold is not the portfolio that was optimized."
        )
    if have_adv and schedule:
        hottest = max(
            (s for s in schedule if s["max_participation_used"] is not None),
            key=lambda s: s["max_participation_used"],
            default=None,
        )
        if hottest and hottest["max_participation_used"] > 0.2:
            out.append(
                f"day {hottest['day']} plans to take "
                f"{hottest['max_participation_used']:.0%} of a name's daily "
                "volume. Above about 20% the square root law stops describing "
                "reality -- you are no longer trading alongside the day's flow, "
                "you are the day's flow -- so the cost estimate is optimistic."
            )
    return out


__all__ = [
    "DEFAULT_IMPACT_COEFFICIENT",
    "DEFAULT_MAX_PARTICIPATION",
    "plan_rebalance",
]
