"""Target construction. Phase 1 supports forward_return only (ModelSpec.type
is a Literal, so an unsupported type is already rejected at the Pydantic
boundary before this function is ever called)."""

import pandas as pd

from ..specs import TargetSpec


def build_target(close: pd.Series, spec: TargetSpec) -> pd.Series:
    """
    Forward return over `spec.horizon` bars: value at date t is
    (close[t+horizon] - close[t]) / close[t] — the return an entity earns
    starting at t, not the trailing return ending at t. Implemented as
    pct_change(periods=horizon).shift(-horizon): pct_change gives the
    trailing return ending at t+horizon, and shift(-horizon) pulls that
    value back to sit on row t, which is exactly the forward return.
    """
    return close.pct_change(periods=spec.horizon).shift(-spec.horizon)
