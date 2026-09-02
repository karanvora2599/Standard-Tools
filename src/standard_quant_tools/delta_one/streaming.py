"""
Monitoring a basis on a live feed, in a world where a tool call returns.

THE ARCHITECTURAL PROBLEM, AND WHY THE SHAPE IS WHAT IT IS. A monitor wants
to be a subscription: open it, and it tells you when something happens. A
tool call cannot be that. It is asked a question and it answers, and there
is nowhere for a long-running loop to live between calls.

So the state is the return value. `new_basis_monitor` produces a plain
JSON-safe dict; `update_basis_monitor` takes that dict plus whatever ticks
have arrived since the last call and hands back a new one, with any alert
attached. The caller -- an agent, a cron loop, a `while True` in a script --
holds the state between calls. That is the only shape that works here, and
it has two properties a subscription does not: the state is inspectable at
every step, and a monitor can be paused, serialized, moved to another
process and resumed without losing its baseline.

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
watching, so `reset_basis_monitor` is explicit rather than automatic.
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
    "MONITOR_VERSION",
    "new_basis_monitor",
    "reset_basis_monitor",
    "update_basis_monitor",
]

#: Bumped when the state's shape changes. A monitor resumed from a state
#: written by older code would otherwise carry accumulators that mean
#: something slightly different, and the drift would be invisible.
MONITOR_VERSION = 2

#: Observations gathered before the baseline is fixed. Below about this the
#: standard deviation is too noisy to standardize against, and a detector
#: built on it fires on its own estimation error.
DEFAULT_WARMUP = 60


def new_basis_monitor(
    *,
    label: str = "basis",
    warmup: int = DEFAULT_WARMUP,
    threshold: float = DEFAULT_THRESHOLD,
    slack: float = DEFAULT_SLACK,
    annualized: bool = False,
) -> Dict[str, Any]:
    """
    An empty monitor, ready to be fed.

    `annualized` says which channel is being watched: with it, updates must
    carry a time to expiry and the basis is `ln(F/S)/T` in bps, comparable
    across expiries. Without it the channel is `(F/S - 1)` in bps of spot,
    which is fine within one contract and STEPS at a roll -- and a step is
    exactly what this is built to flag, so a monitor left un-annualized
    across a roll will report the roll.
    """
    if int(warmup) < 10:
        raise ValidationError(
            f"warmup={warmup} is too short. A baseline standard deviation "
            "from fewer than about ten observations is mostly estimation "
            "error, and a detector standardized against it fires on that."
        )
    return {
        "version": MONITOR_VERSION,
        "label": str(label),
        "warmup": int(warmup),
        "threshold": positive(threshold, "threshold"),
        "slack": non_negative(slack, "slack"),
        "annualized": bool(annualized),
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


def update_basis_monitor(
    state: Dict[str, Any],
    *,
    spot: Sequence[float],
    futures: Sequence[float],
    time_to_expiry: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Feed everything that arrived since the last call.

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
    spot_values = _floats(spot, "spot")
    futures_values = _floats(futures, "futures")
    if len(spot_values) != len(futures_values):
        raise ValidationError(
            f"spot has {len(spot_values)} observations and futures has "
            f"{len(futures_values)}; an update must carry both sides of every "
            "tick it reports."
        )
    if not spot_values:
        raise ValidationError(
            "an update with no observations. Pass at least one tick, or do "
            "not call -- an empty update would advance nothing and its "
            "returned state would be indistinguishable from a stalled feed."
        )

    expiries: Optional[List[float]] = None
    if state["annualized"]:
        if time_to_expiry is None:
            raise ValidationError(
                "this monitor watches the ANNUALIZED basis, so every update "
                "needs time_to_expiry. Without it the two channels would be "
                "mixed inside one baseline, which is not a comparison."
            )
        expiries = _floats(time_to_expiry, "time_to_expiry")
        if len(expiries) != len(spot_values):
            raise ValidationError(
                f"time_to_expiry has {len(expiries)} values against "
                f"{len(spot_values)} ticks."
            )
    elif time_to_expiry is not None:
        raise ValidationError(
            "time_to_expiry was given to a monitor that is not annualized. "
            "Create it with annualized=True, or drop the argument -- "
            "silently ignoring it would leave the caller believing the "
            "channel was one thing while it was another."
        )

    alert: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    crossed_this_call = False

    for index, (s, f) in enumerate(zip(spot_values, futures_values)):
        if s <= 0:
            raise ValidationError(
                f"observation {index} has spot={s!r}; the basis in bps of "
                "spot is undefined there."
            )
        if state["annualized"]:
            t = expiries[index]  # type: ignore[index]
            if t <= 0:
                raise ValidationError(
                    f"observation {index} has time_to_expiry={t!r}. A "
                    "contract at or past expiry has no annualized basis; "
                    "roll the monitor onto the next contract instead."
                )
            value = math.log(f / s) / t * 10_000.0
        else:
            value = (f / s - 1.0) * 10_000.0

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
            # Exactly zero has no z at all, so there is nothing to test
            # rather than something enormous to report.
            continue

        z = (value - mean) / std
        state["up"] = max(0.0, state["up"] + z - state["slack"])
        state["down"] = max(0.0, state["down"] - z - state["slack"])
        statistic = max(state["up"], state["down"])
        if statistic > state["peak"]:
            state["peak"] = statistic

        if not state["triggered"] and statistic >= state["threshold"]:
            state["triggered"] = True
            state["first_crossing_at"] = state["n"]
            state["n_alerts"] += 1
            crossed_this_call = True
            alert = {
                "observation": state["n"],
                "value": value,
                "statistic": statistic,
                "direction": "above" if state["up"] >= state["down"] else "below",
                "baseline_mean": mean,
                "baseline_std": std,
                "shift_in_baseline_sd": (value - mean) / std,
                "degenerate_baseline": bool(state.get("degenerate_baseline")),
                "message": (
                    f"{state['label']}: the basis has shifted "
                    f"{'above' if state['up'] >= state['down'] else 'below'} "
                    f"its baseline of {mean:.1f} bps and stayed there. Now "
                    f"{value:.1f} bps. Rule out a contract roll, a dividend "
                    "going ex and a move in the financing curve before "
                    "treating this as a dislocation."
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
            "normal the world is still abnormal. reset_basis_monitor when "
            "the new level is the one to watch from."
        )

    return {
        "state": state,
        "alert": alert,
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


def reset_basis_monitor(
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
            f"{mean:.4f}, sd {std:.6g}). The monitor still runs -- a calm "
            "period before a real shock is exactly the case it is for -- but "
            "every statistic measured against this baseline is arithmetic "
            "rather than evidence, and any alert will read as enormous "
            "because it is dividing by almost nothing. A feed that was "
            "stale or halted through the warm-up produces exactly this."
        )


def _validated(state: Any) -> Dict[str, Any]:
    """A monitor state, or a refusal naming what is wrong with it."""
    if not isinstance(state, dict):
        raise ValidationError(
            f"state must be the dict returned by new_basis_monitor, got "
            f"{type(state).__name__}."
        )
    missing = [
        key
        for key in ("version", "warmup", "threshold", "slack", "annualized", "n")
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
    return dict(state)


def _floats(values: Sequence[float], name: str) -> List[float]:
    if values is None:
        raise ValidationError(f"{name} is required and was not given")
    try:
        out = [finite(v, f"{name}[{i}]") for i, v in enumerate(values)]
    except TypeError:
        raise ValidationError(f"{name} must be a sequence of numbers") from None
    return out
