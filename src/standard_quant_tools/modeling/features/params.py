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
