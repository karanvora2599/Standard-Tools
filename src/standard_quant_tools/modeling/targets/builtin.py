"""
The built-in labels: six prices can produce, twelve they cannot.

The arithmetic is the arithmetic `dataset/target.py` had, moved rather
than rewritten; that module is now the dispatcher over this registry and
re-exports the same names. The tests that pinned each label's numbers --
the vol-scaled denominator using no future bar, the barrier walk on a
known path, the rank centred on zero -- run unchanged against these.

Every buildable label takes the entity's full OHLCV and reads what it
needs; every builder here needs only Close, which `requires` says.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from standard_quant_tools.modeling.features.transforms import (
    cross_sectional_counts,
    rank_within_date,
)

from .base import TargetDefinition
from .registry import register_target


def _forward_return(close: pd.Series, spec: Any) -> pd.Series:
    """
    (close[t+horizon] - close[t]) / close[t]: the return an entity earns
    starting at t, not the trailing return ending at t. pct_change gives
    the trailing return ending at t+horizon, and shift(-horizon) pulls it
    back onto row t.

    `fill_method=None`, explicitly. Without it pandas pads across a data
    gap and FABRICATES a supervised label: close=[100, 101, nan, nan, 90]
    with horizon=2 gave row 1 a forward return of +0.01 -- correct is NaN,
    and the next real print is -10%.
    """
    return close.pct_change(periods=spec.horizon, fill_method=None).shift(-spec.horizon)


def _horizon_volatility(close: pd.Series, spec: Any) -> pd.Series:
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


def _triple_barrier(close: pd.Series, spec: Any) -> pd.Series:
    """
    Which barrier the price touches first within the horizon.

    Three NOMINAL classes, not an ordered scale:

        1.0  upper barrier touched first  ("up")
        0.0  lower barrier touched first  ("down")
        2.0  neither touched within the horizon  ("went nowhere")

    "Went nowhere" is a real and common outcome, and a plain up/down label
    silently folds it into "down", teaching the model something false.

    The numbering has to be integer-valued, because sklearn reads a float
    target whose values are 0.0/0.5/1.0 as CONTINUOUS and refuses to fit
    any classifier to it. Given integers, "up" is deliberately 1 so that
    positive_class_proba keeps returning P(up).

    The barrier defaults to trailing volatility scaled to the horizon
    rather than a fixed percentage: a fixed 5% barrier is a coin flip in a
    quiet name and unreachable in a volatile one.

    Only closes are examined, not intrabar highs and lows -- a conservative
    barrier test, stated rather than hidden.
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

    windows = np.lib.stride_tricks.sliding_window_view(prices, horizon)
    entry = prices[: n - horizon]
    with np.errstate(invalid="ignore", divide="ignore"):
        forward = windows[1:] / entry[:, None] - 1.0

    band = width[: n - horizon][:, None]
    touched_up = forward >= band
    touched_down = forward <= -band
    first_up = np.where(touched_up.any(axis=1), touched_up.argmax(axis=1), horizon)
    first_down = np.where(
        touched_down.any(axis=1), touched_down.argmax(axis=1), horizon
    )

    labels = np.full(n - horizon, 2.0)
    labels[first_up < first_down] = 1.0
    labels[first_down < first_up] = 0.0
    labels[~np.isfinite(band[:, 0])] = np.nan
    out.iloc[: n - horizon] = labels
    return out


def horizon_label_end(ohlcv: pd.DataFrame, spec: Any, context: Any = None) -> pd.Series:
    """
    The date of the LAST bar each row's target observes: `horizon` bars
    ahead on THIS ENTITY'S OWN calendar.

    Returned as an explicit per-row timestamp rather than inferred from an
    integer offset, because with missing trading days or entities on
    different calendars, t+horizon entity bars is not generally t+horizon
    global panel dates, and purging on an integer embargo under-purges
    exactly there. The final `horizon` rows have no label end and are NaT.
    The default `label_end_builder` for every registered label.
    """
    close = ohlcv["Close"]
    end_dates = pd.Series(pd.NaT, index=close.index, dtype="datetime64[ns]")
    if spec.horizon < len(close):
        end_dates.iloc[: len(close) - spec.horizon] = close.index[spec.horizon :]
    return end_dates


# ── builders: (ohlcv, spec, context) -> Series ─────────────────────────


def _build_forward_return(ohlcv, spec, context=None):
    return _forward_return(ohlcv["Close"], spec)


def _build_forward_direction(ohlcv, spec, context=None):
    # NaN is preserved rather than binarized: `NaN > threshold` is False,
    # so a naive astype(float) would label every unresolved final row 0.0
    # -- "went down" for bars whose outcome has not happened yet.
    forward = _forward_return(ohlcv["Close"], spec)
    return (forward > spec.threshold).astype(float).where(forward.notna())


def _build_vol_scaled(ohlcv, spec, context=None):
    close = ohlcv["Close"]
    return _forward_return(close, spec) / _horizon_volatility(close, spec)


def _build_triple_barrier(ohlcv, spec, context=None):
    return _triple_barrier(ohlcv["Close"], spec)


# ── cross-sectional stages: (panel, spec, target_col) -> panel ─────────


def _stage_rank(panel: pd.DataFrame, spec: Any, target_col: str) -> pd.DataFrame:
    """
    The return's rank within its date, mapped to [-0.5, 0.5].

    (rank - 1) / (n - 1), then centred, so the target is symmetric and
    scale-free regardless of how many entities are present that day.
    Through the shared per-date ranking kernel, the same one the ensemble
    combiner and the feature report use. A date with a single entity has
    no rank and becomes NaN; the builder drops those rows and says so.
    """
    out = panel.copy()
    column = out[[target_col]]
    dates = out["date"].to_numpy()
    ranks = rank_within_date(column, dates)[target_col]
    counts = cross_sectional_counts(column, dates)[target_col]
    with np.errstate(invalid="ignore", divide="ignore"):
        values = (ranks - 1.0) / (counts - 1.0) - 0.5
    out[target_col] = values.where(counts > 1)
    return out


def _stage_market_neutral(panel: pd.DataFrame, spec: Any, target_col: str) -> pd.DataFrame:
    """The return minus that date's equal-weighted mean across entities:
    the market factor taken out of the LABEL rather than left in and
    hoped away. A one-name cross-section has a market-relative return of
    exactly zero by construction, which is not a measurement, so it is
    NaN and dropped."""
    out = panel.copy()
    grouped = out.groupby("date")[target_col]
    values = out[target_col] - grouped.transform("mean")
    counts = grouped.transform("count")
    out[target_col] = values.where(counts > 1)
    return out


# ── registration ─────────────────────────────────────────────────────────

_CONTINUOUS = ("regression", "ranking")

register_target(
    TargetDefinition(
        id="forward_return",
        description="The return from t to t+horizon.",
        tasks=_CONTINUOUS,
        buildable=True,
        continuous=True,
        builder=_build_forward_return,
    )
)
register_target(
    TargetDefinition(
        id="forward_direction",
        description="That forward return binarized against `threshold`.",
        tasks=("classification",),
        buildable=True,
        continuous=False,
        builder=_build_forward_direction,
    )
)
register_target(
    TargetDefinition(
        id="forward_return_vol_scaled",
        description="Forward return over the entity's own trailing volatility.",
        tasks=_CONTINUOUS,
        buildable=True,
        continuous=True,
        builder=_build_vol_scaled,
    )
)
register_target(
    TargetDefinition(
        id="forward_return_rank",
        description="Its rank within the date's cross-section, in [-0.5, 0.5].",
        tasks=_CONTINUOUS,
        buildable=True,
        continuous=True,
        builder=_build_forward_return,
        cross_sectional_stage=_stage_rank,
    )
)
register_target(
    TargetDefinition(
        id="forward_return_market_neutral",
        description="Forward return minus that date's equal-weighted mean.",
        tasks=_CONTINUOUS,
        buildable=True,
        continuous=True,
        builder=_build_forward_return,
        cross_sectional_stage=_stage_market_neutral,
    )
)
register_target(
    TargetDefinition(
        id="triple_barrier",
        description="Which barrier is touched first: up, down, or neither.",
        tasks=("classification",),
        buildable=True,
        continuous=False,
        builder=_build_triple_barrier,
    )
)

# ── microstructure labels, computed where the book is ──────────────────
#
# EXTERNAL-ONLY IS NOT A GAP. A markout, a fill probability or a time to
# fill is a function of the order book and of orders, not of closing
# prices. `build_target` refuses them by name and says to compute them
# where the book is and register the panel -- which is a real answer,
# whereas a bar-derived approximation would be a number with nothing
# behind it.

for _id, _tasks, _continuous, _description in (
    (
        "future_mid_return",
        _CONTINUOUS,
        True,
        "Return of the MIDPOINT over the horizon. Not the same as a "
        "trade-price return: the mid moves without a trade and is where "
        "a passive order is measured from.",
    ),
    (
        "future_microprice_return",
        _CONTINUOUS,
        True,
        "Return of the size-weighted touch price. Leads the mid when the "
        "book is lopsided, which is exactly when the mid is least "
        "informative.",
    ),
    (
        "future_markout",
        _CONTINUOUS,
        True,
        "Mid move measured FROM a fill, signed by the side taken. The "
        "standard read on whether a trade was well-placed.",
    ),
    (
        "next_mid_direction",
        ("classification",),
        False,
        "Whether the midpoint's next move is up or down.",
    ),
    (
        "future_spread",
        _CONTINUOUS,
        True,
        "The quoted spread at t+horizon. A liquidity forecast rather "
        "than a price one -- what it will COST to cross, not where the "
        "price goes.",
    ),
    (
        "future_depth",
        _CONTINUOUS,
        True,
        "Resting size at t+horizon. What will be THERE to trade against, "
        "which a spread forecast does not answer -- a tight quote for a "
        "hundred shares and a tight quote for fifty thousand cost the "
        "same to cross and are not the same liquidity.",
    ),
    (
        "future_ofi",
        _CONTINUOUS,
        True,
        "Signed order-flow imbalance over the horizon, from book "
        "updates. Predicting FLOW rather than price: the quantity that "
        "moves the price, one step earlier.",
    ),
    (
        "future_volume",
        _CONTINUOUS,
        True,
        "Traded volume over the horizon. Bar volume can approximate this "
        "at daily frequency, but not at the horizons this exists for, "
        "where the question is how much prints in the next thirty "
        "seconds.",
    ),
    (
        "future_trade_intensity",
        _CONTINUOUS,
        True,
        "Trades per unit time over the horizon. Distinct from volume: "
        "one block and two hundred odd lots are the same volume and "
        "completely different information.",
    ),
    (
        "fill_probability",
        ("classification",),
        False,
        "Whether a passive order resting at a stated level fills within "
        "the horizon. Needs queue position and cancellations, so no "
        "bar-derived series can produce it.",
    ),
    (
        "time_to_fill",
        ("regression",),
        True,
        "How long that order waits before filling. CENSORED by "
        "construction -- an order that never fills has no time, and "
        "recording it as the horizon rather than as unfilled biases "
        "every estimate toward patience.",
    ),
    (
        "adverse_selection",
        _CONTINUOUS,
        True,
        "How much the mid moves against a fill after it happens. The "
        "cost of being the one who was willing to trade.",
    ),
):
    register_target(
        TargetDefinition(
            id=_id,
            description=_description,
            tasks=_tasks,
            buildable=False,
            continuous=_continuous,
        )
    )

__all__ = ["_horizon_volatility", "_triple_barrier", "horizon_label_end"]
