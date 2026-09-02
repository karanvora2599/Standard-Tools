"""
One answer to "is this series usable", for the modules that each had their own.

WHY THIS FILE EXISTS. Four modules here defined a private `_clean` with the
same name and the same job -- drop what cannot be measured, then refuse if
too little is left -- and gave four different answers to the same input. A
single positive infinity in a 300-point return series produced:

    structure.detect_change_points      0 breaks became 3 breaks
    diagnostics.ljung_box               p_value became NaN
    stationarity.run_stationarity_tests LinAlgError: SVD did not converge
    inference.bootstrap_statistic       silently dropped, answered on 299

The first is the one that matters. An infinity is not a measurement; it is
a division that should not have happened upstream, most often a zero or
negative price reaching `pct_change`. Here it did not surface as an error
or as a NaN -- it manufactured three regime changes in a series that has
none, and returned them in the ordinary result shape with ordinary-looking
break dates. Nothing downstream could tell that apart from a real finding.

The fourth is quieter and still wrong: dropping the infinity means the
answer is computed on a different sample than the caller passed, and the
result says nothing about it.

WHAT IT DOES. `require_finite_series` in `numeric_contract` is already the
library's answer to this question, and its rejection message names how many
infinities there are, where they are, and what upstream mistake usually
produces them. This module does not restate that check -- it calls it, and
adds only the two things the four private copies had that it does not: a
NaN drop, and a per-caller minimum-length floor.

NaN is treated differently from infinity ON PURPOSE. A NaN is a gap -- an
untraded day, a series that starts later than its neighbours -- and
dropping it is the correct reading of the data. An infinity is a
corruption, and there is no correct reading of it.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.numeric_contract import require_finite_series

__all__ = ["clean_series"]


def clean_series(
    series: Union[pd.Series, Sequence[float], Any],
    name: str,
    func: str,
    *,
    minimum: int = 1,
    as_array: bool = False,
    note: Optional[str] = None,
) -> Union[pd.Series, np.ndarray]:
    """
    Numeric, finite, NaN-free and long enough -- or a ValidationError saying
    which of those failed.

    `minimum` is the caller's floor, because the floors are genuinely
    different: a change point detector needs enough points either side of a
    break, and an asymptotic test statistic needs enough for its
    distribution to mean anything. `note` appends the caller's own reason
    for that floor, which is worth more to whoever hits it than the number.

    `as_array` returns a numpy array for the callers that go straight into
    numpy from here; the index is not meaningful to them.
    """
    raw = series if isinstance(series, pd.Series) else pd.Series(series)
    values = raw.astype(float)

    # Infinities are rejected here, by the library's own contract, with its
    # own message. NaN survives this call and is dropped below.
    require_finite_series(values, name, func, allow_empty=True)

    values = values.dropna()
    if len(values) < minimum:
        detail = f" {note}" if note else ""
        raise ValidationError(
            f"{func}: {len(values)} usable observations, and this needs at "
            f"least {minimum}.{detail}"
        )
    return values.to_numpy() if as_array else values
