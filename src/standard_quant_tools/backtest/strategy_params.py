"""
One validation contract for every strategy in STRATEGY_REGISTRY.

The modeling runtime already has this (modeling/features/params.py's
resolve_params): a declared schema per feature, checked once at the boundary,
so a bad parameter fails with a message naming the feature and the parameter
rather than as a pandas/numpy error several frames deep. The classic strategy
registry had nothing equivalent -- not one of its eight strategies validated a
single parameter -- and the consequence was not merely cosmetic:

    momentum_timeseries(lookback=-20)

reached ``Close.pct_change(periods=-20)`` unchecked, and a NEGATIVE period
makes pandas look FORWARD. Standing at bar 25 it returns
``close[25]/close[45] - 1``, so the signal for bar 25 is computed from bar
45's price. That is direct look-ahead, produced by an ordinary-looking
integer, in the layer this library exists to make trustworthy. It is
reachable from the agent surface because BacktestInput.parameters is an
unconstrained ``Dict[str, Any]``.

The same class of hole existed elsewhere in the registry: NaN thresholds
(which make every comparison False, silently flattening a strategy to
never-in-market), inverted RSI bands, non-positive windows, and negative
Bollinger multipliers that swap the bands' meaning.

Design notes:

  - Windows are POSITIVE INTEGERS. Zero and negative are rejected outright;
    the look-ahead above is why a negative is a correctness bug rather than
    merely an odd input.
  - Thresholds must be FINITE. NaN fails every comparison, so it does not
    make a strategy stricter -- it makes it silently inert, which looks
    exactly like a strategy that honestly found no trades.
  - Cross-parameter relations are checked (fast < slow, oversold <
    overbought), since each value can be individually valid while the pair
    is nonsense.
  - Unknown parameter names are REJECTED. Every strategy signature ends in
    ``**_``, so a typo or a hallucinated parameter was silently swallowed
    and the strategy ran on its defaults -- the caller believing it had
    configured something it had not.
"""

import math
from typing import Any, Dict, Optional, Tuple

from standard_quant_tools.error import ValidationError

# A window longer than this is a request no realistic series can serve, and
# is more likely a unit mix-up (days vs minutes) than an intent. Mirrors the
# spirit of modeling's _MAX_WINDOW_BARS.
_MAX_WINDOW_BARS = 100_000


class _Param:
    """One parameter's contract. `kind` is 'window' (a positive integer bar
    count) or 'number' (a finite float within an optional range)."""

    def __init__(
        self,
        kind: str,
        default: Any,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> None:
        self.kind = kind
        self.default = default
        self.minimum = minimum
        self.maximum = maximum


# Relations that must hold BETWEEN parameters, as (left, right, why).
# Checked only after each value is individually valid.
_RELATIONS: Dict[str, Tuple[Tuple[str, str, str], ...]] = {
    "sma_crossover": (
        (
            "fast_period",
            "slow_period",
            "the fast average must be shorter than the slow one, or the "
            "crossover has no meaning",
        ),
    ),
    "macd_crossover": (
        (
            "fast",
            "slow",
            "MACD is fast EMA minus slow EMA; with fast >= slow the line is "
            "inverted or identically zero",
        ),
    ),
    "rsi_mean_reversion": (
        (
            "oversold",
            "overbought",
            "oversold must sit below overbought, or the bands cross and every "
            "bar is simultaneously both",
        ),
    ),
}

STRATEGY_PARAM_SCHEMA: Dict[str, Dict[str, _Param]] = {
    "sma_crossover": {
        "fast_period": _Param("window", 10),
        "slow_period": _Param("window", 30),
    },
    "rsi_mean_reversion": {
        "period": _Param("window", 14),
        "oversold": _Param("number", 30, minimum=0.0, maximum=100.0),
        "overbought": _Param("number", 70, minimum=0.0, maximum=100.0),
    },
    "macd_crossover": {
        "fast": _Param("window", 12),
        "slow": _Param("window", 26),
        "signal": _Param("window", 9),
    },
    "bollinger_reversion": {
        "period": _Param("window", 20),
        # A negative multiplier puts the "upper" band below the "lower" one,
        # inverting every entry and exit while still producing plausible
        # output. Zero collapses both bands onto the mean.
        "num_std": _Param("number", 2.0, minimum=1e-9, maximum=100.0),
    },
    "donchian_breakout": {
        "entry_period": _Param("window", 20),
        "exit_period": _Param("window", 10),
    },
    "momentum_timeseries": {
        # The look-ahead parameter. See the module docstring.
        "lookback": _Param("window", 90),
        "threshold": _Param("number", 0.0, minimum=-1.0, maximum=100.0),
    },
    "vwap_reversion": {
        "period": _Param("window", 20),
        "entry_threshold": _Param("number", 0.02, minimum=0.0, maximum=100.0),
    },
    "adx_trend": {
        "adx_period": _Param("window", 14),
        "adx_threshold": _Param("number", 25.0, minimum=0.0, maximum=100.0),
    },
}


def _resolve_window(strategy: str, name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{strategy}: {name} must be a positive whole number of bars, got "
            f"{type(value).__name__} ({value!r})"
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(
                f"{strategy}: {name} must be a finite positive whole number of "
                f"bars, got {value!r}"
            )
        if not float(value).is_integer():
            raise ValidationError(
                f"{strategy}: {name} counts BARS and must be a whole number, "
                f"got {value!r}"
            )
    value = int(value)
    if value < 1:
        raise ValidationError(
            f"{strategy}: {name} must be >= 1, got {value}. A negative period "
            "is not merely invalid — pandas reads a negative pct_change/shift "
            "period as a FORWARD window, so this bar's signal would be "
            "computed from future prices and the backtest would contain "
            "look-ahead by construction."
        )
    if value > _MAX_WINDOW_BARS:
        raise ValidationError(
            f"{strategy}: {name}={value} exceeds the maximum supported window "
            f"({_MAX_WINDOW_BARS} bars)"
        )
    return value


def _resolve_number(strategy: str, name: str, value: Any, spec: _Param) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{strategy}: {name} must be a number, got "
            f"{type(value).__name__} ({value!r})"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(
            f"{strategy}: {name} must be finite, got {value!r}. NaN compares "
            "False against every threshold test, so it does not make the "
            "strategy stricter — it silently makes it inert, which is "
            "indistinguishable from a strategy that honestly found no trades."
        )
    if spec.minimum is not None and number < spec.minimum:
        raise ValidationError(
            f"{strategy}: {name} must be >= {spec.minimum}, got {number}"
        )
    if spec.maximum is not None and number > spec.maximum:
        raise ValidationError(
            f"{strategy}: {name} must be <= {spec.maximum}, got {number}"
        )
    return number


def resolve_strategy_params(
    strategy: str,
    params: Optional[Dict[str, Any]] = None,
    check_relations: bool = True,
) -> Dict[str, Any]:
    """
    Validate and normalize one strategy's parameters, returning the full
    resolved set (supplied values merged over the declared defaults).

    `check_relations` splits the contract into the two things it actually
    enforces, because they are not equally universal:

      - PER-VALUE checks (always on) cover what makes a run WRONG: a
        negative window silently reading future prices, a NaN threshold
        silently making the strategy inert, an unknown name silently
        ignored. These are applied on every path, including inside
        STRATEGY_REGISTRY itself, so no call site can skip them.

      - CROSS-PARAMETER relations (fast < slow, oversold < overbought)
        describe a combination that is economically meaningless but not
        leaky or silently misleading -- an inverted crossover just scores
        badly. A parameter GRID legitimately sweeps a rectangle that
        contains such pairs (fast in 10..50 × slow in 10..50 covers plenty
        of fast >= slow), and backtest_grid does not catch per-combination
        errors, so enforcing relations there would abort the whole sweep
        rather than let the search score those points and move on.

    So relations are enforced where a single configuration is deliberately
    requested (the agent tools), and skipped where a search is enumerating
    a space.

    Raises:
        ValidationError: unknown strategy, unknown parameter name, a window
            that is not a positive whole number, a non-finite or
            out-of-range threshold, or (when check_relations) a violated
            cross-parameter relation.
    """
    if strategy not in STRATEGY_PARAM_SCHEMA:
        raise ValidationError(
            f"unknown strategy {strategy!r}. Available: "
            f"{sorted(STRATEGY_PARAM_SCHEMA)}"
        )
    schema = STRATEGY_PARAM_SCHEMA[strategy]
    supplied = dict(params or {})

    unknown = sorted(set(supplied) - set(schema))
    if unknown:
        raise ValidationError(
            f"{strategy}: unknown parameter(s) {unknown}. Valid parameters: "
            f"{sorted(schema)}. Every strategy signature ends in **_, so an "
            "unrecognized name was previously accepted silently and the "
            "strategy ran on its default instead — the request looked "
            "configured when it was not."
        )

    resolved: Dict[str, Any] = {}
    for name, spec in schema.items():
        if name not in supplied:
            resolved[name] = spec.default
            continue
        value = supplied[name]
        if spec.kind == "window":
            resolved[name] = _resolve_window(strategy, name, value)
        else:
            resolved[name] = _resolve_number(strategy, name, value, spec)

    if not check_relations:
        return resolved
    for left, right, why in _RELATIONS.get(strategy, ()):
        if resolved[left] >= resolved[right]:
            raise ValidationError(
                f"{strategy}: {left}={resolved[left]} must be < "
                f"{right}={resolved[right]} — {why}"
            )
    return resolved
