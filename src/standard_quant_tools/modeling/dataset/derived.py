"""
Features computed from other features, on the same row.

Applied to ONE ENTITY'S frame, between the missing-value policy and the
lags, for the two reasons the builder gives at that point: a lag of a
derived column should be a lag of the filled inputs, which it is only if
the fill precedes the derivation; and everything here stays inside one
entity's own rows, so nothing can reach another symbol's history.

Every operator is pointwise -- a row-wise function of columns on the same
row. There is no window, no state and nothing fitted, which is what makes
a derived column cost no warm-up beyond its inputs' and makes it incapable
of leaking: there is nothing to fit, so nothing can be fitted on the wrong
rows.

A ZERO DENOMINATOR GIVES NaN, NOT AN INFINITY. An infinity is the quieter
failure of the two: it survives into the panel, is "finite" to nobody's
check, and most imputations turn it into a very large real number that the
model then fits. NaN is the value the rest of this pipeline already knows
how to talk about -- the missing-value policy sees it, the coverage report
counts it, and the fold check refuses a column that is all of it.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError


def _ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = left / right
    return np.where(np.isfinite(out), out, np.nan)


#: The bounded operator set. Pointwise, stateless, warm-up free.
OPERATORS = {
    "ratio": _ratio,
    "difference": lambda left, right: left - right,
    "product": lambda left, right: left * right,
    "sum": lambda left, right: left + right,
}

#: Refused by name rather than offered. A residual is FITTED, and fitting
#: it over an entity's whole history at build time fits on the test window
#: too. `preprocessing/` is where a step is fitted per fold on training
#: rows only, which is the only place this can happen honestly.
FITTED_OPERATORS = {
    "residual_ols": (
        "a residual is a FITTED quantity: computing it here would regress "
        "over the entity's whole history, including the rows the model is "
        "about to be tested on. A preprocessing step is fitted per fold on "
        "training rows only, which is where a residualisation belongs."
    ),
}


def apply_derived(
    frame: pd.DataFrame, derived: Sequence[Any]
) -> pd.DataFrame:
    """
    Add each derived column to one entity's feature frame, in order.

    The spec has already checked that every input exists by the time it is
    read -- inputs may name a base feature or a derived feature defined
    EARLIER -- so this evaluates straight down the list and a cycle cannot
    arrive here. The check is repeated anyway, because this function is
    also reachable from scoring, where the frame is rebuilt and a column
    could be absent for a reason the spec could not see.
    """
    if not derived:
        return frame
    out = frame
    for spec in derived:
        operator = OPERATORS.get(spec.op)
        if operator is None:
            reason = FITTED_OPERATORS.get(spec.op)
            raise ValidationError(
                f"derived feature {spec.name!r}: unknown operator {spec.op!r}"
                + (f" -- {reason}" if reason else f"; expected one of {sorted(OPERATORS)}")
            )
        missing = [name for name in spec.inputs if name not in out.columns]
        if missing:
            raise ValidationError(
                f"derived feature {spec.name!r} reads {missing}, which this "
                f"frame does not carry. It has {sorted(out.columns)}."
            )
        left = out[spec.inputs[0]].to_numpy(dtype=float)
        right = out[spec.inputs[1]].to_numpy(dtype=float)
        # Assigned into a copy rather than in place: `out` is the caller's
        # frame on the first pass, and a derived feature must not mutate
        # the columns another entity's pass will read.
        out = out.assign(**{spec.name: operator(left, right)})
    return out


def derived_output_names(derived: Sequence[Any]) -> list:
    """The columns `apply_derived` will add, in order."""
    return [spec.name for spec in derived]


def derived_warmup_note(derived: Sequence[Any]) -> Dict[str, str]:
    """Why a derived feature adds no warm-up, for the reports that say
    where a panel can start."""
    return {
        spec.name: (
            f"{spec.op} of {spec.inputs[0]} and {spec.inputs[1]}, on the same "
            "row — so it first has a value exactly where both inputs do, and "
            "costs no warm-up of its own."
        )
        for spec in derived
    }


__all__ = [
    "FITTED_OPERATORS",
    "OPERATORS",
    "apply_derived",
    "derived_output_names",
    "derived_warmup_note",
]
