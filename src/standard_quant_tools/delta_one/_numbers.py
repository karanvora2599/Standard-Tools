"""
Scalar guards for this package, delegating rather than reimplementing.

WHY THIS FILE EXISTS. Every module here validates the same three shapes --
a finite number, a non-negative one, a strictly positive one -- and each of
them had written its own copy. Thirteen of them, across eight modules,
which is exactly the drift this library warns about elsewhere: they were
already not identical, because the local ones accepted `True` as the number
1.0 while `require_finite_scalar` rejects a bool outright.

NOTHING IS IMPLEMENTED HERE. `require_finite_scalar` in `numeric_contract`
already checks finiteness BEFORE any range comparison, for the reason its
own docstring gives: every comparison against NaN is False, so a guard
written as `if x <= 0` never fires for NaN and the NaN flows on into a
result that carries no error and no numbers. `_positive` in
`analysis.derivatives` is the strictest scalar check in the library and is
what `implied_forward_price` itself uses, so the carry math in this package
and the carry math it calls agree on what a valid price IS.

This module only adapts their signatures to one shape, so a call site does
not have to name the calling function twice.
"""

from __future__ import annotations

from typing import Any

# Private by name, shared by intent: `_positive` is what implied_forward_price
# validates its own spot with, and this package's carry functions call that
# function. Using a second definition here would let the two disagree about
# a price one of them accepts.
from standard_quant_tools.analysis.derivatives import _bounded as bounded
from standard_quant_tools.analysis.derivatives import _positive as positive
from standard_quant_tools.numeric_contract import require_finite_scalar

__all__ = ["bounded", "finite", "non_negative", "positive"]


def finite(value: Any, name: str, func: str = "delta_one") -> float:
    """A number, finite, of any sign."""
    return require_finite_scalar(value, name, func)


def non_negative(value: Any, name: str, func: str = "delta_one") -> float:
    """A number, finite, at or above zero."""
    return require_finite_scalar(value, name, func, minimum=0.0)
