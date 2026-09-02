"""
Target construction.

Two kinds live here. Most targets are ENTITY-LOCAL: they need only that
entity's own price history, so they are built inside the per-entity loop
alongside the features. Two are CROSS-SECTIONAL — a rank within the date,
and a return measured against the universe average — and cannot be built
until every entity is stacked into one panel; those are applied afterwards
by `apply_cross_sectional_target`, which the builder calls once.

TargetSpec.type is a Literal, so an unsupported name is rejected at the
Pydantic boundary before anything here runs.
"""

import numpy as np
import pandas as pd

from ..specs import TargetSpec

# Targets that cannot be computed from one entity's prices alone.
CROSS_SECTIONAL_TARGETS = frozenset(
    {"forward_return_rank", "forward_return_market_neutral"}
)


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
    if spec.type in CROSS_SECTIONAL_TARGETS:
        # Placeholder: the raw forward return, which
        # apply_cross_sectional_target turns into the requested quantity
        # once the panel exists. Returning it (rather than NaN) means the
        # alignment step drops exactly the rows it always did.
        return forward_return
    if spec.type == "forward_return_vol_scaled":
        return forward_return / _horizon_volatility(close, spec)
    if spec.type == "triple_barrier":
        return _triple_barrier(close, spec)
    direction = (forward_return > spec.threshold).astype(float)
    return direction.where(forward_return.notna())


def _horizon_volatility(close: pd.Series, spec: TargetSpec) -> pd.Series:
    """
    Trailing volatility of this entity, scaled to the target horizon.

    Uses returns up to and INCLUDING bar t, so the scale applied to row t's
    forward return is known at t — the divisor must not be built from the
    same future the numerator measures, or the target leaks its own answer.

    Zero volatility (a halted or synthetic-constant series) becomes NaN
    rather than dividing by zero: an entity with no variation has no
    meaningful volatility-scaled return, and alignment drops the row like
    any other missing value.
    """
    returns = close.pct_change(fill_method=None)
    vol = returns.rolling(spec.vol_window, min_periods=spec.vol_window).std()
    scaled = vol * np.sqrt(float(spec.horizon))
    return scaled.where(scaled > 0)


def _triple_barrier(close: pd.Series, spec: TargetSpec) -> pd.Series:
    """
    Which barrier the price touches first within the horizon.

    Three NOMINAL classes, not an ordered scale:

        1.0  upper barrier touched first  ("up")
        0.0  lower barrier touched first  ("down")
        2.0  neither touched within the horizon  ("went nowhere")

    "Went nowhere" is a real and common outcome, and a plain up/down label
    silently folds it into "down", teaching the model something false.

    The specific numbering is not arbitrary. It has to be integer-valued,
    because sklearn reads a float target whose values are 0.0/0.5/1.0 as
    CONTINUOUS and refuses to fit any classifier to it — the obvious
    encoding does not work at all. Given integers, "up" is deliberately 1
    so that positive_class_proba keeps returning P(up): that probability is
    what the downstream signal path consumes as a score, and any ordering
    that put "flat" at class 1 would hand it P(nothing happened).

    The barrier defaults to trailing volatility scaled to the horizon
    rather than a fixed percentage, because a fixed 5% barrier is a coin
    flip in a quiet name and unreachable in a volatile one — the same
    label would mean different things for different entities.

    Only closes are examined, not intrabar highs and lows. That makes this
    a conservative barrier test: a level touched and reversed within a
    single bar is not counted. Stated rather than hidden, because the
    alternative reads a high/low the entity may not have printed at a
    tradeable moment.
    """
    prices = close.to_numpy(dtype=float)
    n = prices.size
    horizon = int(spec.horizon)
    out = pd.Series(np.nan, index=close.index, dtype=float)
    if n <= horizon or horizon < 1:
        return out

    if spec.barrier > 0:
        width = np.full(n, float(spec.barrier))
    else:
        width = _horizon_volatility(close, spec).to_numpy(dtype=float)

    # future[t, k] is the return from bar t to bar t+1+k.
    windows = np.lib.stride_tricks.sliding_window_view(prices, horizon)
    entry = prices[: n - horizon]
    with np.errstate(invalid="ignore", divide="ignore"):
        forward = windows[1:] / entry[:, None] - 1.0

    band = width[: n - horizon][:, None]
    touched_up = forward >= band
    touched_down = forward <= -band
    # argmax returns 0 for an all-False row, so the "any" masks below are
    # what separate "touched at bar 0" from "never touched".
    first_up = np.where(touched_up.any(axis=1), touched_up.argmax(axis=1), horizon)
    first_down = np.where(
        touched_down.any(axis=1), touched_down.argmax(axis=1), horizon
    )

    labels = np.full(n - horizon, 2.0)  # neither barrier touched
    labels[first_up < first_down] = 1.0
    labels[first_down < first_up] = 0.0
    # A NaN barrier (volatility warm-up) leaves the row unlabelled.
    labels[~np.isfinite(band[:, 0])] = np.nan
    out.iloc[: n - horizon] = labels
    return out


def apply_cross_sectional_target(
    panel: pd.DataFrame, spec: TargetSpec, target_col: str = "target"
) -> pd.DataFrame:
    """
    Turn the stacked forward-return column into a cross-sectional target.

    Called once by the builder after every entity is in one frame, because
    both of these targets are defined against the OTHER entities on the
    same date and simply do not exist per entity.

      forward_return_rank — the return's rank within its date, mapped to
        [-0.5, 0.5]. This is the target that matches the scorecard: the
        model is judged on cross-sectional rank IC, so training it to
        predict a rank rather than a magnitude aligns the two. It also
        removes the fat tail that lets a handful of extreme returns
        dominate a squared-error loss.

      forward_return_market_neutral — the return minus that date's
        equal-weighted mean across entities. Takes the market factor out
        of the LABEL, rather than leaving it in and hoping the model
        learns to ignore it. What remains is the relative performance a
        cross-sectional model is actually supposed to forecast.

    A date with a single entity has no cross-section: its rank is
    undefined and its market-relative return is exactly zero by
    construction, which is not a measurement. Those rows are set to NaN and
    dropped, the same rule cross_sectional_ic applies when scoring.
    """
    if spec.type not in CROSS_SECTIONAL_TARGETS or panel.empty:
        return panel

    out = panel.copy()
    grouped = out.groupby("date")[target_col]
    if spec.type == "forward_return_market_neutral":
        values = out[target_col] - grouped.transform("mean")
    else:
        # rank -> [0, 1] by (rank - 1) / (n - 1), then centered on zero, so
        # the target is symmetric and scale-free regardless of how many
        # entities happen to be present that day.
        ranks = grouped.rank(method="average")
        counts = grouped.transform("count")
        values = (ranks - 1.0) / (counts - 1.0) - 0.5
        values = values.where(counts > 1)

    single_entity = grouped.transform("count") <= 1
    out[target_col] = values.where(~single_entity)
    return out


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
