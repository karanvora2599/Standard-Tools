"""
How the indicators read a missing bar, and why an infinity is refused.

ONE RULE, EVERY DOOR, BOTH BACKENDS. `rsi`, `wilder_atr`, `adx`,
`bollinger_bands`, `stochastic_oscillator` and `technical_indicators_panel`
all read their inputs the same way:

  * NaN is a MISSING BAR. The Wilder recursions (RSI, Wilder's ATR, ADX)
    skip it: the indicator is computed over the bars that are present, as
    if the missing ones had been dropped, and is NaN at the missing ones.
    The windowed indicators (Bollinger Bands, stochastic %K and %D) are NaN
    for every window that holds a missing bar and resume at the first one
    that does not -- pandas' rolling(min_periods=period).
  * +/-inf is REFUSED, by name. An infinity is not a price. A window's mean
    and range have no meaning with one in it, and a recursion cannot skip
    what it has already absorbed: inf - inf is NaN on the next smoothing
    step, so one infinite bar used to leave every later RSI, ATR and ADX
    NaN, and a stochastic %D that never recovered.

The native kernels and the Python fallbacks implement the NaN rule
operation for operation, so which backend served a call is not visible in
its answer. Before the CHANGELOG entry of 2026-09-28 the single-series
functions refused NaN while the panel answered it, and the two backends
disagreed about where a gap fell: the native RSI went NaN for the rest of
the series after a NaN in its seed window and read one in its forward pass
as an unchanged price, and the Numba ADX never recovered where the native
one did.
"""

from __future__ import annotations

import numpy as np

from standard_quant_tools.error import ValidationError


def refuse_infinities(values: np.ndarray, name: str, func: str) -> None:
    """Raise ValidationError if `values` holds +/-inf; NaN passes as a gap."""
    bad = np.isinf(np.asarray(values, dtype=np.float64))
    if bad.any():
        raise ValidationError(
            f"{func}: {name} contains {int(bad.sum())} infinite value(s). An "
            "infinity is not a price, and a smoothed indicator would carry it "
            "into every later bar. Replace it with NaN to mark the bar as "
            "missing -- a missing bar is skipped -- or drop the bar."
        )
