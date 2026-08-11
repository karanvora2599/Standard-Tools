"""Target construction. Phase 1 supports forward_return only (ModelSpec.type
is a Literal, so an unsupported type is already rejected at the Pydantic
boundary before this function is ever called)."""

import pandas as pd

from ..specs import TargetSpec


def build_target(close: pd.Series, spec: TargetSpec) -> pd.Series:
    """
    Build the supervised target.

    `forward_return` — the return an entity earns starting at t, not the
    trailing return ending at t: (close[t+horizon] - close[t]) / close[t].
    Implemented as pct_change(periods=horizon).shift(-horizon): pct_change
    gives the trailing return ending at t+horizon, and shift(-horizon)
    pulls that value back onto row t, which is exactly the forward return.

    `forward_direction` — that same forward return binarized to 1.0/0.0
    against `spec.threshold`. This exists so task='classification' is
    reachable through the ordinary five-tool pipeline: ModelSpec.task has
    always ACCEPTED 'classification', but TargetSpec could only build a
    continuous return, so a binary target could only be obtained by
    mutating the panel by hand outside the agent workflow — a documented
    capability with no way to construct it.

    NaN is preserved rather than being binarized. The final `horizon` rows
    have no forward return at all, and `NaN > threshold` is False, so a
    naive `.astype(float)` would silently label every one of them 0.0 —
    manufacturing a "went down" observation for bars whose outcome simply
    has not happened yet. Alignment drops NaN rows instead.
    """
    forward_return = close.pct_change(periods=spec.horizon).shift(-spec.horizon)
    if spec.type == "forward_return":
        return forward_return
    direction = (forward_return > spec.threshold).astype(float)
    return direction.where(forward_return.notna())


def build_label_end_dates(close: pd.Series, spec: TargetSpec) -> pd.Series:
    """
    The date of the LAST bar each row's target actually observes.

    Row t's forward return reads close[t+horizon], so its label is only
    fully determined once bar t+horizon has printed. Walk-forward
    validation must therefore purge any training row whose label end lands
    on or after the first test date, or the model trains on labels built
    from test-period prices.

    Returned as an explicit per-row timestamp rather than being inferred
    from an integer offset, because `horizon` counts THIS ENTITY'S OWN
    bars: with missing trading days or entities on different calendars
    (a mid-history IPO, a halted symbol, a foreign listing), t+horizon
    entity bars is not generally t+horizon global panel dates. Purging on
    an integer embargo silently under-purges exactly in those cases.

    The final `horizon` rows have no label end (their target is NaN and
    they are dropped during alignment anyway), so they are NaT here.
    """
    end_dates = pd.Series(pd.NaT, index=close.index, dtype="datetime64[ns]")
    if spec.horizon < len(close):
        end_dates.iloc[: len(close) - spec.horizon] = close.index[spec.horizon :]
    return end_dates
