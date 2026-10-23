"""
A duration's seconds, counted from its integer, the same on every pandas.

`Series.dt.total_seconds()` is pandas' arithmetic, and pandas 3.1 changed
it. Up to 3.0 it divided each duration's integer by the units in a second
in one step; from 3.1 it adds the whole seconds to the fraction left over,
and nudges a value that lands on a whole second. The two round differently
wherever a duration has a part under a second: of 200,000 random
nanosecond durations under 2**40 ns (about 18 minutes), 309 read another
last bit, and of 200,000 past 2**53 ns (104 days), 49,691. Daily and
minute stamps are whole seconds, which both forms give exactly; a stamp
with a part under a second -- what an external panel or a point-in-time
record can carry -- did not get one answer across pandas versions.

`duration_seconds` forms every value as `analysis.order_events` forms an
order's lifetime, from the integer: the whole seconds plus the exact
fraction left over, added in floating point. That is pandas 3.1's sum
without the nudge, it is the same bits whichever unit the durations are
stored in (pandas 2 parses a timestamp string to nanoseconds, pandas 3 to
microseconds when nothing finer is written), and it does not depend on the
pandas installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from standard_quant_tools.error import ValidationError


def duration_seconds(deltas: Any) -> np.ndarray:
    """
    Each duration in `deltas` (a timedelta Series, Index or array) as
    float seconds, counted from its integer; NaN for NaT, as
    `total_seconds()` gives it.

    Raises:
        ValidationError: the values are not durations, or are stored in a
            unit pandas does not use (it uses s, ms, us and ns).
    """
    # The rule has one definition, the order-event module's. Imported here
    # rather than at the top, so a module that needs a duration does not
    # import the analysis package with it.
    from standard_quant_tools.analysis.order_events import _seconds

    values = np.asarray(deltas)
    if values.dtype.kind != "m":
        raise ValidationError(
            f"duration_seconds: expected durations (timedelta64), got dtype "
            f"{values.dtype}. Subtract two timestamps to get one."
        )
    unit, step = np.datetime_data(values.dtype)
    seconds = _seconds(values.view(np.int64), unit) if step == 1 else None
    if seconds is None:
        raise ValidationError(
            f"duration_seconds: durations stored as {values.dtype} are in a "
            "unit pandas does not use; cast them to timedelta64[ns] (or s, "
            "ms, us) first."
        )
    missing = np.isnat(values)
    if missing.any():
        seconds = np.where(missing, np.nan, seconds)
    return seconds


__all__ = ["duration_seconds"]
