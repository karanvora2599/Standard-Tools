"""
resolve_params — the single validation point between a caller-supplied
`FeatureSpec.params` dict and the feature function it is splatted into.

Without this, `params: Dict[str, object]` was completely unrestricted and
went straight through as `**params`, which produced two distinct failures:

1. **Future leakage through a negative window.** `market.momentum` and
   `volume.obv_roc` pass `lookback` directly to
   `Series.pct_change(periods=lookback)`. pandas accepts a negative period
   and computes `x[t] / x[t + |lookback|] - 1` — i.e. the feature at t
   reads a price from t+|lookback|. The feature's declared
   `TemporalSupport.PIT_SAFE` is a static property of the FORMULA, so the
   point-in-time gate kept passing while the resolved parameters made the
   feature non-causal. PIT safety has to be a property of the RESOLVED
   feature, not just its label.

2. **Unknown parameter names became raw TypeErrors.** A typo'd or invented
   key surfaced as `fn() got an unexpected keyword argument 'lookbak'`
   from inside the feature, not as a modeling validation error naming the
   feature and its accepted parameters.

Validation is derived from each feature's own `default_params` rather than
a separate hand-maintained schema, so a newly registered feature (including
a firm's custom one) is covered automatically instead of being forgotten.
"""

import math
from typing import Any, Dict

from standard_quant_tools.error import ValidationError

from .base import FeatureDefinition

# Parameter names that denote a number of bars of history. These are the
# ones a negative or zero value turns into either future leakage or an
# empty window, so they carry the strictest rule.
_WINDOW_PARAM_NAMES = frozenset(
    {"lookback", "period", "window", "span", "fast", "slow", "signal", "horizon"}
)

# Upper bound on any bar-count parameter. Deliberately far above anything a
# real model would use (100k daily bars is ~400 years) -- this exists to
# stop an agent turning one valid-looking tool call into an unbounded
# rolling computation, the same reason the estimator registry caps
# n_estimators/max_depth, not to express an opinion about useful window
# lengths.
_MAX_WINDOW_BARS = 100_000


def _is_window_param(name: str) -> bool:
    return name in _WINDOW_PARAM_NAMES or name.endswith(
        ("_period", "_window", "_lookback")
    )


def resolve_params(
    definition: FeatureDefinition, requested: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge `requested` onto `definition.default_params` and validate the
    result.

    Rules, in order:
      - every requested key must be one the feature actually accepts;
      - a value must match the broad type of that parameter's default
        (bool/int/float/str), so a string doesn't reach an arithmetic
        window;
      - any bar-count parameter must be a strictly positive integer —
        this is the rule that closes the negative-lookback leak;
      - any other numeric value must be finite.

    Raises:
        ValidationError: unknown parameter name, wrong type, non-positive
        bar count, or a non-finite numeric value.
    """
    unknown = sorted(set(requested) - set(definition.default_params))
    if unknown:
        accepted = sorted(definition.default_params) or ["(none)"]
        raise ValidationError(
            f"feature {definition.id!r}: unknown parameter(s) {unknown}. "
            f"Accepted parameter(s): {accepted}."
        )

    resolved = {**definition.default_params, **requested}

    for name, value in resolved.items():
        default = definition.default_params[name]

        # bool is a subclass of int in Python -- check it first so a
        # boolean flag isn't silently treated as a bar count of 1.
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValidationError(
                    f"feature {definition.id!r}: parameter {name!r} must be a bool, "
                    f"got {value!r} ({type(value).__name__})."
                )
            continue

        if isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValidationError(
                    f"feature {definition.id!r}: parameter {name!r} must be a number, "
                    f"got {value!r} ({type(value).__name__})."
                )
            if not math.isfinite(float(value)):
                raise ValidationError(
                    f"feature {definition.id!r}: parameter {name!r} must be finite, "
                    f"got {value!r}."
                )
            # Integer-ness is derived from the DEFAULT's type, not from the
            # parameter's name. The name-based rule below still applies the
            # stricter >= 1 window semantics, but it only recognised a
            # fixed vocabulary -- so `refit_every=1.5` sailed through as a
            # generic finite number and later reached `range(window, n+1,
            # refit_every)`, which raises a raw TypeError from inside
            # numpy/Python rather than a modeling validation error naming
            # the feature and parameter.
            if isinstance(default, int) and not isinstance(default, bool):
                if isinstance(value, float) and not float(value).is_integer():
                    raise ValidationError(
                        f"feature {definition.id!r}: parameter {name!r} must be a whole "
                        f"number (its default {default!r} is an integer), got {value!r}."
                    )
                value = int(value)
                resolved[name] = value

            if _is_window_param(name):
                if isinstance(value, float) and not float(value).is_integer():
                    raise ValidationError(
                        f"feature {definition.id!r}: parameter {name!r} is a number of "
                        f"bars and must be a whole number, got {value!r}."
                    )
                if int(value) < 1:
                    raise ValidationError(
                        f"feature {definition.id!r}: parameter {name!r} must be >= 1, got "
                        f"{value!r}. A zero or negative window is not merely invalid — "
                        f"pandas interprets a negative period as a FORWARD window, which "
                        f"would make this feature read future prices while still being "
                        f"declared point-in-time safe."
                    )
                if int(value) > _MAX_WINDOW_BARS:
                    raise ValidationError(
                        f"feature {definition.id!r}: parameter {name!r}={value!r} exceeds "
                        f"the maximum supported window of {_MAX_WINDOW_BARS:,} bars. "
                        "Estimator parameters already carry compute ceilings; feature "
                        "windows need them for the same reason — one tool call should "
                        "not be able to request an unbounded rolling computation. This "
                        "is a bound on obviously pathological input, not a view on what "
                        "window length is sensible."
                    )
                resolved[name] = int(value)
            continue

        if isinstance(default, str) and not isinstance(value, str):
            raise ValidationError(
                f"feature {definition.id!r}: parameter {name!r} must be a string, "
                f"got {value!r} ({type(value).__name__})."
            )

    return resolved


def resolved_lookback(definition: FeatureDefinition, resolved: Dict[str, Any]) -> int:
    """
    Bars of history this feature actually consumes GIVEN its resolved
    parameters.

    `FeatureDefinition.lookback` is a static number recorded at
    registration time against the default parameters, so it stays 20 for
    `market.momentum` even when called with `lookback=500`. Callers that
    need to size a history window (scoring, warm-up budgeting) need the
    resolved value, not the declared one.
    """
    window_values = [
        int(v)
        for k, v in resolved.items()
        if _is_window_param(k)
        and isinstance(v, (int, float))
        and not isinstance(v, bool)
    ]
    return (
        max([definition.lookback, *window_values])
        if window_values
        else definition.lookback
    )


# ── Warm-up of a recursive feature ──────────────────────────────────────
#
# `resolved_lookback` counts the bars to a feature's FIRST OUTPUT. For a
# finite-window feature that is also the bar from which its value no longer
# depends on where the history starts. For a recursive smoother it is not:
# an EMA or a Wilder average carries its start value forward with weight
# (1 - alpha)^n, so an RSI computed from a history starting in 2016 and one
# starting in 2010 disagree for over a hundred bars after the first output
# -- rows that are not NaN, so nothing drops them. A scoring panel is
# rebuilt from `as_of - lookback_days`, a different start than the training
# build's, so an under-warmed recursive feature is a train/serve skew as
# well as a dirty first stretch.

#: The start value's residual weight at which a recursive feature counts as
#: warm. Measured over six seeds and three start points, at this tolerance
#: the truncated and full-history values of every smoother below agree to
#: within 3e-4 of the feature's standard deviation (the RSI at period 50 is
#: the worst); at 1e-3 the RSI still differed by 1.3e-3 of it.
WARMUP_TOLERANCE = 1e-4


def smoother_decay_bars(alpha: float, tolerance: float = WARMUP_TOLERANCE) -> int:
    """
    Bars until a first-order recursive smoother with coefficient `alpha`
    has forgotten its start value to within `tolerance`: the least n with
    (1 - alpha)^n <= tolerance. EMA(span s) has alpha = 2 / (s + 1), about
    (s + 1) / 2 * ln(1 / tolerance) bars; Wilder(p) has alpha = 1 / p,
    about p * ln(1 / tolerance).
    """
    if not 0.0 < alpha < 1.0:
        return 0 if alpha >= 1.0 else _MAX_WINDOW_BARS
    return int(math.ceil(math.log(tolerance) / math.log(1.0 - alpha)))


def _chained_decay_bars(alpha: float, tolerance: float) -> int:
    """
    Two smoothers of the same `alpha` in series -- ADX, a Wilder average of
    a DX built from Wilder averages -- forget their start as
    (1 + n * alpha) * (1 - alpha)^n: the second stage keeps averaging in
    the first stage's decaying error. The least n below `tolerance`.
    """
    n = smoother_decay_bars(alpha, tolerance)
    while (1.0 + n * alpha) * (1.0 - alpha) ** n > tolerance:
        n += 1
    return n


def _psar_warmup(params: Dict[str, Any], tolerance: float) -> int:
    """
    Parabolic SAR is a state machine, not a smoother, and has no decay rate
    to put in a closed form: its warm-up is the bars until two runs started
    at different points reverse on the same bar and agree from then on.
    Measured over 450 start pairs at the default af_start=0.02 that took a
    median of 6 bars, 38 at the 95th percentile and 81 at the most; 100 is
    declared, and scaled up for a slower start, which trends longer before
    it reverses. Not a function of `tolerance`: once coupled the two runs
    are identical.
    """
    af_start = float(params.get("af_start", 0.02))
    scale = max(1.0, 0.02 / af_start) if af_start > 0 else 1.0
    return int(math.ceil(100 * scale))


#: The built-in features whose value depends on where the history starts,
#: and the closed form of how long that lasts at their resolved parameters.
#: Chained stages add: the output is warm once its last stage has forgotten
#: a start that was itself still warming. Every other built-in feature is a
#: finite window (or a running sum whose differences do not depend on its
#: start), for which the first output is already warm.
_RECURSIVE_WARMUP: Dict[str, Any] = {
    # Wilder average of gains and losses, seeded after `period` bars.
    "technical.rsi": lambda p, eps: int(p["period"])
    + smoother_decay_bars(1.0 / p["period"], eps),
    # Wilder average of the true range, seeded after `period` bars.
    "risk.atr_pct": lambda p, eps: int(p["period"])
    + smoother_decay_bars(1.0 / p["period"], eps),
    # Wilder DI after `period` bars, Wilder ADX over DX after `period` more.
    "technical.adx": lambda p, eps: 2 * int(p["period"])
    + _chained_decay_bars(1.0 / p["period"], eps),
    # MACD line (the slower EMA dominates the faster), then the signal EMA.
    "technical.macd_histogram": lambda p, eps: int(p["slow"])
    + smoother_decay_bars(2.0 / (max(p["fast"], p["slow"]) + 1.0), eps)
    + smoother_decay_bars(2.0 / (p["signal"] + 1.0), eps),
    "market.psar_trend": _psar_warmup,
}
# The same smoothers divided by the bar's own Close, which adds no memory.
_RECURSIVE_WARMUP["technical.macd_histogram_pct"] = _RECURSIVE_WARMUP[
    "technical.macd_histogram"
]


def is_recursive(definition: FeatureDefinition) -> bool:
    """Whether this feature's value depends on where its history starts for
    longer than its first output takes to appear."""
    return definition.id in _RECURSIVE_WARMUP


def resolved_warmup(
    definition: FeatureDefinition,
    resolved: Dict[str, Any],
    tolerance: float = WARMUP_TOLERANCE,
) -> int:
    """
    Bars of history after which this feature's value no longer depends on
    where the history started, to within `tolerance` of the start value's
    weight, GIVEN its resolved parameters.

    A second quantity beside `resolved_lookback`, not a replacement: that
    one is the bars to the first output, which is what a panel's row loss
    is counted in. This one is what a history window must cover for the
    value at its end to be the value a longer history would give -- the
    number to size a scoring window with. Equal to `resolved_lookback` for
    a finite-window feature; longer for a recursive smoother (EMA, Wilder)
    and for Parabolic SAR. A custom feature is taken at its declared
    lookback, since nothing here knows its formula.
    """
    if not 0.0 < tolerance < 1.0:
        raise ValidationError(
            f"warm-up tolerance must lie strictly between 0 and 1, got {tolerance!r}."
        )
    first_output = resolved_lookback(definition, resolved)
    rule = _RECURSIVE_WARMUP.get(definition.id)
    if rule is None:
        return first_output
    try:
        return max(first_output, int(rule(resolved, tolerance)))
    except (KeyError, TypeError, ZeroDivisionError):
        # A definition registered over a built-in id with parameters the
        # built-in's formula does not have: its own formula is unknown here.
        return first_output
