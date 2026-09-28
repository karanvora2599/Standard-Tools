"""
Scalar guards for this package, delegating rather than reimplementing.

WHY THIS FILE EXISTS. Every module here validates the same three shapes --
a finite number, a non-negative one, a strictly positive one -- and each of
them had written its own copy. Thirteen of them, across eight modules,
which is exactly the drift this library warns about elsewhere: they were
already not identical, because the local ones accepted `True` as the number
1.0 while `require_finite_scalar` rejects a bool outright.

ONE ENTRY POINT FOR ALL FOUR. `require_finite_scalar` in `numeric_contract`
checks the TYPE first -- a bool or a string is refused rather than coerced
-- and finiteness BEFORE any range comparison, for the reason its own
docstring gives: every comparison against NaN is False, so a guard written
as `if x <= 0` never fires for NaN and the NaN flows on into a result that
carries no error and no numbers. Every guard below passes through it.

`positive` used to be `analysis.derivatives._positive`, on the argument
that `implied_forward_price` validates its own spot with it and the two
should agree on what a valid price IS. They agreed on the range and not on
the type: `_positive` calls `float(value)`, so `positive(True)` returned 1.0
and `positive('5')` returned 5.0 while `finite` and `non_negative` beside it
refused both -- and `etf_fair_value(etf_price='100.3', nav=True)` reported
a premium of 993,000 bps. This one is `require_finite_scalar` plus the
strict bound, which is STRICTER than `_positive` on every input, so anything
accepted here is still accepted by `implied_forward_price` downstream.
`bounded` wraps `analysis.derivatives._bounded` the same way, keeping its
range and its message.
"""

from __future__ import annotations

from typing import Any

from standard_quant_tools.analysis.derivatives import _bounded
from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import require_finite_scalar

__all__ = ["bounded", "finite", "non_negative", "positive"]


def finite(value: Any, name: str, func: str = "delta_one") -> float:
    """A number, finite, of any sign."""
    return require_finite_scalar(value, name, func)


def non_negative(value: Any, name: str, func: str = "delta_one") -> float:
    """A number, finite, at or above zero."""
    return require_finite_scalar(value, name, func, minimum=0.0)


def positive(value: Any, name: str, func: str = "delta_one") -> float:
    """A number, finite, strictly above zero."""
    if value is None:
        raise ValidationError(f"{name} is required and was not given")
    number = require_finite_scalar(value, name, func)
    if number <= 0:
        raise ValidationError(f"{func}: {name} must be positive, got {number!r}")
    return number


def bounded(
    value: Any,
    name: str,
    *,
    low: float,
    high: float,
    unit: str = "",
    func: str = "delta_one",
) -> float:
    """A number, finite, inside a range set far outside any real market."""
    return _bounded(
        require_finite_scalar(value, name, func), name, low=low, high=high, unit=unit
    )
