"""
Monitoring a spread on a live feed, in a world where a tool call returns.

THE ARCHITECTURAL PROBLEM, AND WHY THE SHAPE IS WHAT IT IS. A monitor wants
to be a subscription: open it, and it tells you when something happens. A
tool call cannot be that. It is asked a question and it answers, and there
is nowhere for a long-running loop to live between calls.

So the state is the return value. `new_spread_monitor` produces a plain
JSON-safe dict; `update_spread_monitor` takes that dict plus whatever ticks
have arrived since the last call and hands back a new one, with any alert
attached. The caller -- an agent, a cron loop, a `while True` in a script --
holds the state between calls. That is the only shape that works here, and
it has two properties a subscription does not: the state is inspectable at
every step, and a monitor can be paused, serialized, moved to another
process and resumed without losing its baseline.

ONE MONITOR, THREE CHANNELS, FIVE JOBS. The roadmap this came from asked
for five separate monitors -- live basis, ETF NAV, index arbitrage, roll
spread, and a generic cross-instrument spread. They are not five
computations. Four are `(a/b - 1)` in basis points and differ only in what
a and b are CALLED, which is the caller's business rather than the
library's; the fifth is a difference in points. Five tools for three
formulas would mint near-identical names for one rearrangement, which is
the mistake `solve_carry` exists to avoid. So the CHANNEL says how the two
series combine, the LABEL says what they are, and `CHANNEL_USES` records
which roadmap monitor each channel serves.

O(1) PER OBSERVATION, WHICH IS THE POINT. `detect_basis_dislocation` reruns
CUSUM over the whole history every call: correct, and quadratic if you call
it on every tick. This carries the accumulators forward instead, so a
million updates cost a million constant-time steps rather than a million
passes over a growing array. On a live feed that is the difference between
keeping up and falling behind.

THE BASELINE FREEZES AFTER WARM-UP, DELIBERATELY. A monitor that keeps
updating its own definition of normal adapts to the dislocation it is
supposed to report and goes quiet exactly when it matters -- the same
failure the batch detector's reference window exists to avoid. The
consequence is real and stated rather than hidden: after a genuine regime
change the monitor stays triggered until it is reset, because by its own
baseline the world is still abnormal. That is a decision for whoever is
watching, so `reset_spread_monitor` is explicit rather than automatic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from standard_quant_tools.analysis.liquidity_events import (
    DEFAULT_SLACK,
    DEFAULT_THRESHOLD,
    DEGENERATE_BASELINE_CV,
)
from standard_quant_tools.error import ValidationError

from ._numbers import finite, non_negative, positive

__all__ = [
    "CHANNEL_USES",
    "CHANNELS",
    "MONITOR_VERSION",
    "new_spread_monitor",
    "reset_spread_monitor",
    "update_spread_monitor",
]

#: Bumped when the state's shape changes. A monitor resumed from a state
#: written by older code would otherwise carry accumulators that mean
#: something slightly different, and the drift would be invisible.
MONITOR_VERSION = 3

#: Observations gathered before the baseline is fixed. Below about this the
#: standard deviation is too noisy to standardize against, and a detector
#: built on it fires on its own estimation error.
DEFAULT_WARMUP = 60

#: CUSUM threshold for a STREAM, which is not the same problem the batch
#: detector solves and does not take the same number.
#:
#: `liquidity_events.DEFAULT_THRESHOLD` is 9.0, calibrated for a ~5% false
#: alarm rate over a window of KNOWN length -- its reference window is
#: 0.3*n, so its baseline sharpens as the series grows. A stream has no
#: known length and a baseline frozen at `warmup` observations, so the
#: statistic keeps accumulating against a fixed scale and the false alarm
#: rate climbs without bound. Measured here on pure noise, 200 trials:
#:
#:      tested obs      thr=9      thr=15     thr=20
#:            150        2.5%       0.0%       0.0%
#:            430       10.5%       1.0%       0.0%
#:          1,500       31.0%       2.0%       0.5%
#:          5,000       45.0%       7.0%       2.0%
#:
#: 45% is not a detector. Detection power pays almost nothing for the move:
#: 100% at a 1 sd shift and above at every threshold tested, and the only
#: loss is at 0.5 sd (78% -> 53%), which is a marginal signal either way.
#:
#: This is a rate per stream, not per observation, and it still grows with
#: length -- a monitor left running for a million updates will eventually
#: fire on noise. That is inherent to a fixed threshold on an unbounded
#: stream, not something a constant can fix; reset the monitor periodically
#: if the horizon is long.
STREAMING_THRESHOLD = 15.0

#: How the two series combine into the number being watched. THREE, not
#: five, because four of the roadmap's monitors are the same arithmetic
#: under different names.
CHANNELS: Dict[str, str] = {
    "relative_bps": (
        "(primary / reference - 1) * 10,000. The general case: any two "
        "instruments whose prices are comparable in level. Positive means "
        "primary is dear to reference."
    ),
    "annualized_bps": (
        "ln(primary / reference) / T * 10,000, so the number is a RATE and "
        "is comparable across expiries. Needs a time to expiry on every "
        "tick; without one a March and a December contract are not on the "
        "same scale and a single baseline spans both."
    ),
    "absolute_points": (
        "primary - reference, in the underlying's own points. For spreads "
        "quoted in points rather than as a ratio -- a calendar spread is "
        "the usual one, and expressing it in bps of a 6000 index would "
        "compress the whole signal into the third decimal."
    ),
}

#: Which of the roadmap's five monitors each channel serves, and what to
#: pass as primary and reference. Kept as data rather than as five tools:
#: the arithmetic is shared and only the naming differs, so a caller needs
#: the mapping rather than a separate schema for each.
CHANNEL_USES: Dict[str, str] = {
    "relative_bps": (
        "live basis (primary=future, reference=spot); ETF NAV "
        "(primary=ETF price, reference=NAV); index arbitrage "
        "(primary=basket value, reference=index level); any cross-instrument "
        "spread between two comparable prices."
    ),
    "annualized_bps": (
        "live basis where the series spans more than one expiry, which is "
        "the case a bps-of-spot channel gets wrong: it STEPS at every roll "
        "and the detector faithfully reports the step."
    ),
    "absolute_points": (
        "roll spread (primary=next contract, reference=front), and any "
        "spread the market quotes in points."
    ),
}


def new_spread_monitor(
    *,
    channel: str = "relative_bps",
    label: str = "spread",
    warmup: int = DEFAULT_WARMUP,
    threshold: float = STREAMING_THRESHOLD,
    slack: float = DEFAULT_SLACK,
) -> Dict[str, Any]:
    """
    An empty monitor, ready to be fed.

    `channel` decides how the two series combine; `label` says what they
    are and appears in any alert. The two are separate because the
    arithmetic is shared across jobs and the naming is not -- an ETF premium
    and a cash-futures basis are the same computation and want different
    words in the message.
    """
    if channel not in CHANNELS:
        raise ValidationError(
            f"channel={channel!r} must be one of {sorted(CHANNELS)}. There "
            "are three because there are three formulas: a ratio in basis "
            "points, an annualized rate, and a difference in points."
        )
    if int(warmup) < 10:
        raise ValidationError(
            f"warmup={warmup} is too short. A baseline standard deviation "
            "from fewer than about ten observations is mostly estimation "
            "error, and a detector standardized against it fires on that."
        )
    return {
        "version": MONITOR_VERSION,
        "channel": channel,
        "label": str(label),
        "warmup": int(warmup),
        "threshold": positive(threshold, "threshold"),
        "slack": non_negative(slack, "slack"),
        "n": 0,
        # Welford's online mean and sum of squared deviations, so the
        # baseline is computed without keeping the warm-up observations --
        # and without the cancellation that `sum_sq/n - mean**2` suffers.
        # Measured on 70 identical ticks, that naive form returned a
        # standard deviation of 1.4e-06 rather than 0, which slipped past a
        # `std <= 0` guard and produced a CUSUM statistic of 1.7 BILLION.
        "mean": 0.0,
        "m2": 0.0,
        "baseline_mean": None,
        "baseline_std": None,
        # Set at freeze time when the warm-up saw no real variation. The
        # detector still runs -- a calm period before a real shock is the
        # case this is for -- but every statistic against such a baseline is
        # arithmetic rather than evidence, and says so.
        "degenerate_baseline": False,
        "up": 0.0,
        "down": 0.0,
        "peak": 0.0,
        "triggered": False,
        "first_crossing_at": None,
        "n_alerts": 0,
        "last_value": None,
    }


def update_spread_monitor(
    state: Dict[str, Any],
    *,
    primary: Sequence[float],
    reference: Sequence[float],
    time_to_expiry: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Feed everything that arrived since the last call.

    `primary` and `reference` are the two legs, in the order the sign
    convention expects: positive means primary is dear to reference. For a
    basis that is future over spot; for an ETF, price over NAV; for a
    calendar spread, the next contract over the front.

    Returns `{"state": ..., "alert": ..., ...}`. The state is what to pass
    back next time; the alert is None until the accumulated statistic
    crosses, and then describes the crossing once. It does NOT re-fire on
    every subsequent update -- a monitor that alerts on every tick of a
    sustained dislocation is a monitor somebody turns off.

    Batches are equivalent to single updates. Feeding a hundred ticks in one
    call and a hundred calls of one tick produce the same state, because the
    accumulators are carried rather than recomputed. That is what makes the
    call frequency a deployment choice rather than a modelling one.
    """
    state = _validated(state)
    channel = state["channel"]
    primary_values = _floats(primary, "primary")
    reference_values = _floats(reference, "reference")
    if len(primary_values) != len(reference_values):
        raise ValidationError(
            f"primary has {len(primary_values)} observations and reference "
            f"has {len(reference_values)}; an update must carry both legs of "
            "every tick it reports."
        )
    if not primary_values:
        raise ValidationError(
            "an update with no observations. Pass at least one tick, or do "
            "not call -- an empty update would advance nothing and its "
            "returned state would be indistinguishable from a stalled feed."
        )

    expiries: Optional[List[float]] = None
    if channel == "annualized_bps":
        if time_to_expiry is None:
            raise ValidationError(
                "the annualized_bps channel needs time_to_expiry on every "
                "update -- that is what makes it a rate rather than a level. "
                "Without it a March and a December contract share one "
                "baseline, which is not a comparison."
            )
        expiries = _floats(time_to_expiry, "time_to_expiry")
        if len(expiries) != len(primary_values):
            raise ValidationError(
                f"time_to_expiry has {len(expiries)} values against "
                f"{len(primary_values)} ticks."
            )
    elif time_to_expiry is not None:
        raise ValidationError(
            f"time_to_expiry was given to the {channel!r} channel, which "
            "does not use it. Use annualized_bps, or drop the argument -- "
            "silently ignoring it would leave the caller believing the "
            "channel was one thing while it was another."
        )

    alert: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    crossed_this_call = False

    for index, (a, b) in enumerate(zip(primary_values, reference_values)):
        value = _channel_value(channel, a, b, expiries, index)

        state["n"] += 1
        state["last_value"] = value

        if state["n"] <= state["warmup"]:
            # Welford: one pass, O(1), and stable.
            delta = value - state["mean"]
            state["mean"] += delta / state["n"]
            state["m2"] += delta * (value - state["mean"])
            if state["n"] == state["warmup"]:
                _freeze_baseline(state, warnings)
            continue

        mean = state["baseline_mean"]
        std = state["baseline_std"]
        if std is None or std <= 0:
            # A zero baseline has no z at all, so this observation cannot be
            # tested. It used to `continue` and nothing else, which meant a
            # warm-up spent on a stalled or halted feed left the monitor
            # DEAF FOREVER: the baseline never re-froze, and 50 ticks at
            # +100 bps, then +1,000, then +10,000, all returned
            # triggered=False with a statistic of 0.0.
            #
            # `_freeze_baseline`'s own warning says "The monitor still runs
            # -- a calm period before a real shock is exactly the case it is
            # for", and named a stale feed as the way to produce this. It
            # was wrong in precisely that case. So keep accumulating and
            # re-freeze as soon as the window has real dispersion.
            _refreeze_if_possible(state, value, warnings)
            continue

        z = (value - mean) / std
        state["up"] = max(0.0, state["up"] + z - state["slack"])
        state["down"] = max(0.0, state["down"] - z - state["slack"])
        statistic = max(state["up"], state["down"])
        if statistic > state["peak"]:
            state["peak"] = statistic

        if not state["triggered"] and statistic >= state["threshold"]:
            direction = "above" if state["up"] >= state["down"] else "below"
            unit = "points" if channel == "absolute_points" else "bps"
            state["triggered"] = True
            state["first_crossing_at"] = state["n"]
            state["n_alerts"] += 1
            crossed_this_call = True
            alert = {
                "observation": state["n"],
                "value": value,
                "statistic": statistic,
                "direction": direction,
                "channel": channel,
                "baseline_mean": mean,
                "baseline_std": std,
                "shift_in_baseline_sd": (value - mean) / std,
                "degenerate_baseline": bool(state.get("degenerate_baseline")),
                "message": (
                    f"{state['label']}: the spread has shifted {direction} "
                    f"its baseline of {mean:.1f} {unit} and stayed there. "
                    f"Now {value:.1f} {unit}. Rule out the mechanical causes "
                    "first -- a contract roll, a dividend going ex, a move "
                    "in the financing curve -- before treating this as a "
                    "dislocation."
                ),
            }

    if state["n"] < state["warmup"]:
        warnings.append(
            f"Still warming up: {state['n']} of {state['warmup']} "
            "observations. Nothing can trigger until the baseline is fixed, "
            "and a quiet monitor here says nothing about the market."
        )
    if crossed_this_call and state.get("degenerate_baseline"):
        warnings.append(
            "This alert is measured against a baseline that saw no "
            "variation, so its statistic is arithmetic rather than "
            "evidence. Something did change; how MUCH it changed, in "
            "standard deviations, is not a number this window can support."
        )
    if state["triggered"] and not crossed_this_call:
        warnings.append(
            "Already triggered on an earlier update and NOT re-alerting. The "
            "baseline is frozen, so a genuine regime change leaves this "
            "monitor triggered until it is reset -- by its own definition of "
            "normal the world is still abnormal. reset_spread_monitor when "
            "the new level is the one to watch from."
        )
    if channel == "relative_bps":
        warnings.append(
            "The relative_bps channel STEPS at a contract roll, because the "
            "two legs then carry different amounts of time. Over a series "
            "spanning more than one expiry use annualized_bps -- otherwise "
            "the detector will faithfully report the roll."
        )

    return {
        "state": state,
        "alert": alert,
        "channel": channel,
        "triggered": bool(state["triggered"]),
        "n_observations": int(state["n"]),
        "warming_up": bool(state["n"] < state["warmup"]),
        "current_value": state["last_value"],
        "statistic": float(max(state["up"], state["down"])),
        "peak_statistic": float(state["peak"]),
        "threshold": float(state["threshold"]),
        "baseline_mean": state["baseline_mean"],
        "baseline_std": state["baseline_std"],
        "degenerate_baseline": bool(state.get("degenerate_baseline")),
        "warnings": warnings,
    }


def reset_spread_monitor(
    state: Dict[str, Any], *, keep_baseline: bool = False
) -> Dict[str, Any]:
    """
    Clear the accumulators, and optionally relearn what normal is.

    `keep_baseline=True` acknowledges the alert and keeps watching against
    the SAME normal -- right when the shift was a one-off that has passed.
    `False` relearns from the next `warmup` observations, which is right
    when the new level is the level to watch from. They are different
    decisions and the tool does not guess: acknowledging a spike and
    accepting a regime change look identical in the accumulators and are
    opposite conclusions about the market.
    """
    state = _validated(state)
    fresh = dict(state)
    fresh.update(
        {
            "up": 0.0,
            "down": 0.0,
            "peak": 0.0,
            "triggered": False,
            "first_crossing_at": None,
        }
    )
    if not keep_baseline:
        fresh.update(
            {
                "n": 0,
                "mean": 0.0,
                "m2": 0.0,
                "baseline_mean": None,
                "baseline_std": None,
                "degenerate_baseline": False,
            }
        )
    return fresh


# ── internals ───────────────────────────────────────────────────────────


def _channel_value(
    channel: str,
    primary: float,
    reference: float,
    expiries: Optional[List[float]],
    index: int,
) -> float:
    """One tick, combined the way this channel says."""
    if channel == "absolute_points":
        return primary - reference

    if reference <= 0:
        raise ValidationError(
            f"observation {index} has reference={reference!r}. The "
            f"{channel!r} channel divides by it, so a non-positive reference "
            "has no value. Use absolute_points for a spread that does not "
            "need a positive denominator."
        )
    if primary <= 0:
        raise ValidationError(
            f"observation {index} has primary={primary!r}, which the "
            f"{channel!r} channel cannot take a ratio of."
        )

    if channel == "relative_bps":
        return (primary / reference - 1.0) * 10_000.0

    t = expiries[index]  # type: ignore[index]
    if t <= 0:
        raise ValidationError(
            f"observation {index} has time_to_expiry={t!r}. A contract at or "
            "past expiry has no annualized spread; roll the monitor onto the "
            "next contract instead."
        )
    return math.log(primary / reference) / t * 10_000.0


def _refreeze_if_possible(
    state: Dict[str, Any], value: float, warnings: List[str]
) -> None:
    """
    Try again on a baseline that came out degenerate.

    A frozen baseline is normally fixed for the life of the monitor, which
    is what makes the statistic comparable across the stream. A baseline of
    zero standard deviation is the exception: it can never standardize
    anything, so holding it is not stability, it is silence. This keeps
    feeding Welford past the warm-up and re-freezes the moment the window
    has dispersion to measure against.
    """
    # Welford again, on its own accumulators, the same one pass the warm-up
    # uses at line 264 -- not a second variance formula.
    count = int(state.get("degenerate_n", 0)) + 1
    mean = float(state.get("degenerate_mean", 0.0))
    m2 = float(state.get("degenerate_m2", 0.0))
    delta = value - mean
    mean += delta / count
    m2 += delta * (value - mean)
    state["degenerate_n"], state["degenerate_mean"], state["degenerate_m2"] = (
        count,
        mean,
        m2,
    )
    if count < 2 or m2 <= 0.0:
        return
    std = math.sqrt(m2 / (count - 1))
    coefficient = abs(std / mean) if mean else (0.0 if std == 0 else float("inf"))
    if std <= 0 or coefficient < DEGENERATE_BASELINE_CV:
        return
    state["baseline_mean"] = float(mean)
    state["baseline_std"] = float(std)
    state["degenerate_baseline"] = False
    state["up"] = 0.0
    state["down"] = 0.0
    warnings.append(
        f"The warm-up baseline had no dispersion, so it could not "
        f"standardize anything. It has been re-frozen on the first "
        f"{count} observations that did "
        f"(mean {mean:.4f}, sd {std:.6g}), and the CUSUM accumulators were "
        "reset with it. Everything before this point was untestable, not "
        "quiet."
    )


def _freeze_baseline(state: Dict[str, Any], warnings: List[str]) -> None:
    """Fix the baseline from the warm-up, then never touch it again."""
    n = state["n"]
    mean = state["mean"]
    # ddof=1, matching every other dispersion figure in this package.
    variance = state["m2"] / (n - 1) if n > 1 else 0.0
    std = math.sqrt(max(variance, 0.0))
    state["baseline_mean"] = float(mean)
    state["baseline_std"] = float(std)

    # The library's own threshold, not a new one. `liquidity_events`
    # measured this exact failure: a frozen-spread series produced a CUSUM
    # peak of 286,431 while moving from 1.00 bps to 1.02 bps, because a
    # near-zero denominator turns any move into an enormous z.
    coefficient = abs(std / mean) if mean else (0.0 if std == 0 else float("inf"))
    state["degenerate_baseline"] = bool(coefficient < DEGENERATE_BASELINE_CV)
    if state["degenerate_baseline"]:
        warnings.append(
            f"The warm-up window saw effectively no variation (mean "
            f"{mean:.4f}, sd {std:.6g}), so it cannot standardize anything "
            "and no observation can be tested against it. Rather than run "
            "on a zero denominator -- or, as this did before, fall silent "
            "forever -- the monitor keeps accumulating and re-freezes the "
            "baseline as soon as the window has real dispersion, resetting "
            "the accumulators with it. A feed that was stale or halted "
            "through the warm-up produces exactly this, and the ticks until "
            "the re-freeze are untestable rather than quiet."
        )


def _validated(state: Any) -> Dict[str, Any]:
    """A monitor state, or a refusal naming what is wrong with it."""
    if not isinstance(state, dict):
        raise ValidationError(
            f"state must be the dict returned by new_spread_monitor, got "
            f"{type(state).__name__}."
        )
    missing = [
        key
        for key in ("version", "channel", "warmup", "threshold", "slack", "n")
        if key not in state
    ]
    if missing:
        raise ValidationError(
            f"state is missing {missing}. Pass back the `state` from the "
            "previous update, not the whole result."
        )
    if state["version"] != MONITOR_VERSION:
        raise ValidationError(
            f"state was written by monitor version {state['version']} and "
            f"this is version {MONITOR_VERSION}. The accumulators would "
            "carry forward meaning something slightly different, so it is "
            "refused rather than resumed. Create a new monitor."
        )
    if state["channel"] not in CHANNELS:
        raise ValidationError(
            f"state carries channel={state['channel']!r}, which is not one "
            f"of {sorted(CHANNELS)}."
        )
    return dict(state)


def _floats(values: Sequence[float], name: str) -> List[float]:
    if values is None:
        raise ValidationError(f"{name} is required and was not given")
    try:
        out = [finite(v, f"{name}[{i}]") for i, v in enumerate(values)]
    except TypeError:
        raise ValidationError(f"{name} must be a sequence of numbers") from None
    return out
