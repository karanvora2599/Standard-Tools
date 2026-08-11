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
