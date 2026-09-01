"""
Signal-to-weight construction: convert a SCORE signal panel (arbitrary
per-ticker alpha values, date x ticker — see SignalType in agent/models.py)
into a TARGET_WEIGHT panel consumable by run_portfolio_simulation
(backtest/portfolio_engine.py).

Every function takes and returns a pd.DataFrame with the same (date x
ticker) shape: dates as the index, one column per ticker. Output weights are
scaled so each row's gross exposure (sum of |weight|) equals `gross_leverage`
— the same convention run_portfolio_simulation's max_gross_leverage check
already expects.

Deferred, not built here: beta-neutral, sector-neutral, risk-parity, and
optimizer-generated weights. Each needs infrastructure this repo doesn't have
yet (per-ticker beta/sector metadata, a QP solver) — adding them would be new
scope, not "finishing" the SCORE-to-weight conversion path that was already
partially built (the SignalType enum's SCORE variant had no consumer until
this module).
"""

import logging

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

logger = logging.getLogger(__name__)


def _check_scores(scores: pd.DataFrame) -> None:
    if scores.empty:
        raise ValidationError("scores panel is empty")
    if scores.isna().any().any():
        raise ValidationError(
            "scores panel contains NaN — every ticker must have a value at every date"
        )
    # NaN was rejected; infinity was not, and it is worse here. An inf score
    # survives ranking (it simply sorts first), then poisons a z-score — the
    # mean and standard deviation of a column containing inf are both NaN, so
    # EVERY weight in that cross-section becomes NaN, not just the offending
    # one. Verified: dollar_neutral on a panel with a single inf returned an
    # all-NaN row.
    infinite = np.isinf(scores.to_numpy(dtype=float))
    if infinite.any():
        raise ValidationError(
            f"scores panel contains {int(infinite.sum())} non-finite (infinite) "
            "value(s). An infinite score does not merely rank first — it makes "
            "the column's mean and standard deviation NaN, so every weight in "
            "that cross-section becomes NaN rather than just the one."
        )


def _check_gross_leverage(gross_leverage: float) -> float:
    """
    Gross leverage scales the whole weight vector, so it was the one number
    here that could invert or erase a book without any individual weight
    looking wrong: a negative value flips every position (turning the
    strategy into its own opposite), zero erases the book, and NaN makes
    every weight NaN. None of the three were checked, and a NaN would have
    passed a `< 0` guard anyway.
    """
    if isinstance(gross_leverage, bool) or not isinstance(gross_leverage, (int, float)):
        raise ValidationError(
            f"gross_leverage must be a number, got {type(gross_leverage).__name__}"
        )
    value = float(gross_leverage)
    if not np.isfinite(value):
        raise ValidationError(f"gross_leverage must be finite, got {gross_leverage!r}")
    if value <= 0:
        raise ValidationError(
            f"gross_leverage must be > 0, got {value}. Zero erases the book and "
            "a negative value flips every position, turning the strategy into "
            "its own opposite while each individual weight still looks "
            "well-formed."
        )
    return value


def rank_weighted(scores: pd.DataFrame, gross_leverage: float = 1.0) -> pd.DataFrame:
    """
    Weight each name proportional to its cross-sectional rank, centered on
    the row's mean rank (long the top-ranked, short the bottom-ranked, ~zero
    weight near the median), then scaled so sum(|weight|) == gross_leverage.
    Centering on the mean rank makes sum(weight) == 0 automatically.
    """
    _check_scores(scores)
    _check_gross_leverage(gross_leverage)
    ranks = scores.rank(axis=1, method="average")
    centered = ranks.sub(ranks.mean(axis=1), axis=0)
    gross = centered.abs().sum(axis=1)
    gross_safe = gross.where(gross > 1e-12, other=1.0)
    return centered.div(gross_safe, axis=0) * gross_leverage


def equal_weight_top_bottom(
    scores: pd.DataFrame,
    n_long: int,
    n_short: int,
    gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """
    Long the top n_long scores and short the bottom n_short scores each row,
    equal-weighted within each side so sum(|weight|) == gross_leverage.
    """
    _check_scores(scores)
    _check_gross_leverage(gross_leverage)
    if n_long <= 0 and n_short <= 0:
        raise ValidationError("at least one of n_long/n_short must be > 0")
    n_tickers = scores.shape[1]
    if n_long + n_short > n_tickers:
        raise ValidationError(
            f"n_long + n_short ({n_long + n_short}) exceeds the number of tickers ({n_tickers})"
        )

    # Split gross_leverage 50/50 between the two sides only when BOTH sides
    # are active. A long-only (n_short=0) or short-only (n_long=0) request
    # must allocate the FULL gross_leverage to its one active side -- always
    # halving it regardless would silently size a long-only portfolio at
    # half the requested gross exposure.
    if n_long > 0 and n_short > 0:
        long_w = gross_leverage / (2 * n_long)
        short_w = gross_leverage / (2 * n_short)
    elif n_long > 0:
        long_w = gross_leverage / n_long
        short_w = 0.0
    else:  # n_short > 0 (n_long == 0) -- validated above that at least one is > 0
        long_w = 0.0
        short_w = gross_leverage / n_short

    # Ranked in one pass rather than sorted per row. The `iterrows()` loop
    # this replaces did a full `sort_values` and two label-based `.loc`
    # assignments per date, and measured 1.42s on a 2499x500 panel against
    # 0.05-0.10s for `rank_weighted` and `zscore_normalized` doing the same
    # shape of work vectorized -- the outlier was the loop, not the task.
    #
    # `na_option="bottom"` reproduces `sort_values`, which places NaN last
    # in a descending sort and therefore lets an unscored name be picked as
    # a short. That is pre-existing behaviour, kept deliberately rather
    # than quietly corrected here.
    ranks = scores.rank(axis=1, ascending=False, method="first", na_option="bottom")
    n_columns = scores.shape[1]

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    if n_long > 0:
        weights = weights.mask(ranks <= n_long, long_w)
    # Applied second so that, if the two selections overlap, the short wins
    # -- the order the sequential `.loc` assignments resolved it in.
    if n_short > 0:
        weights = weights.mask(ranks > n_columns - n_short, -short_w)
    return weights


def zscore_normalized(
    scores: pd.DataFrame, gross_leverage: float = 1.0
) -> pd.DataFrame:
    """
    Weight each name proportional to its cross-sectional z-score, scaled so
    sum(|weight|) == gross_leverage. A row with zero cross-sectional std
    (every ticker scored identically) gets all-zero weight rather than a
    division-by-zero blowup — there is no cross-sectional signal to act on.
    """
    _check_scores(scores)
    _check_gross_leverage(gross_leverage)
    mean = scores.mean(axis=1)
    std = scores.std(axis=1)
    degenerate = std <= 1e-12
    std_safe = std.where(~degenerate, other=1.0)
    z = scores.sub(mean, axis=0).div(std_safe, axis=0)
    z = z.where(~degenerate, other=0.0)
    gross = z.abs().sum(axis=1)
    gross_safe = gross.where(gross > 1e-12, other=1.0)
    return z.div(gross_safe, axis=0) * gross_leverage


def vol_scaled(
    scores: pd.DataFrame,
    returns_df: pd.DataFrame,
    lookback: int = 20,
    gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """
    Divide each name's raw score by its trailing realized volatility
    (rolling std of returns_df over `lookback` bars) before the same
    cross-sectional gross-leverage normalization zscore_normalized uses —
    so an equally-scored high-vol name ends up with a smaller weight than a
    low-vol one. Bars before `lookback` observations exist (NaN rolling std)
    get zero weight for that name on that date, not a division blowup.
    """
    _check_scores(scores)
    _check_gross_leverage(gross_leverage)
    missing = [c for c in scores.columns if c not in returns_df.columns]
    if missing:
        raise ValidationError(f"returns_df is missing columns for: {missing}")

    # Rolling window FIRST, on returns_df's own (daily) frequency, THEN
    # reindex onto scores.index -- reindexing before rolling would silently
    # turn a `lookback`-bar volatility window into `lookback` SCORE-DATE
    # observations, e.g. a "20-bar" window becomes ~20 months of history
    # when scores are submitted monthly against daily returns.
    vol = returns_df[scores.columns].rolling(lookback).std()
    vol = vol.reindex(scores.index)
    vol_safe = vol.where(vol > 1e-12, other=np.nan)
    adjusted = (scores / vol_safe).fillna(0.0)
    gross = adjusted.abs().sum(axis=1)
    gross_safe = gross.where(gross > 1e-12, other=1.0)
    return adjusted.div(gross_safe, axis=0) * gross_leverage


def dollar_neutral(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process any weight panel so sum(weight) == 0 per row (equal dollar
    long and short), by subtracting each row's mean weight from every
    position, THEN rescaling each row back to its own original
    sum(|weight|) — mean-centering alone preserves every pairwise weight
    difference exactly (only the common offset changes) but does not
    preserve gross exposure, contradicting this module's own stated
    invariant that every sizing function returns the requested gross
    leverage. Rescaling by a single positive per-row scalar preserves the
    zero-sum property exactly (sum(c*x_i) = c*sum(x_i) = 0 for any c).
    """
    original_gross = weights.abs().sum(axis=1)
    centered = weights.sub(weights.mean(axis=1), axis=0)
    centered_gross = centered.abs().sum(axis=1)
    centered_gross_safe = centered_gross.where(centered_gross > 1e-12, other=1.0)
    return centered.div(centered_gross_safe, axis=0).mul(original_gross, axis=0)
