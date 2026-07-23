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


def rank_weighted(scores: pd.DataFrame, gross_leverage: float = 1.0) -> pd.DataFrame:
    """
    Weight each name proportional to its cross-sectional rank, centered on
    the row's mean rank (long the top-ranked, short the bottom-ranked, ~zero
    weight near the median), then scaled so sum(|weight|) == gross_leverage.
    Centering on the mean rank makes sum(weight) == 0 automatically.
    """
    _check_scores(scores)
    ranks = scores.rank(axis=1, method="average")
    centered = ranks.sub(ranks.mean(axis=1), axis=0)
    gross = centered.abs().sum(axis=1)
    gross_safe = gross.where(gross > 1e-12, other=1.0)
    return centered.div(gross_safe, axis=0) * gross_leverage


def equal_weight_top_bottom(
    scores: pd.DataFrame, n_long: int, n_short: int, gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """
    Long the top n_long scores and short the bottom n_short scores each row,
    equal-weighted within each side so sum(|weight|) == gross_leverage.
    """
    _check_scores(scores)
    if n_long <= 0 and n_short <= 0:
        raise ValidationError("at least one of n_long/n_short must be > 0")
    n_tickers = scores.shape[1]
    if n_long + n_short > n_tickers:
        raise ValidationError(
            f"n_long + n_short ({n_long + n_short}) exceeds the number of tickers ({n_tickers})"
        )

    long_w = gross_leverage / (2 * n_long) if n_long > 0 else 0.0
    short_w = gross_leverage / (2 * n_short) if n_short > 0 else 0.0

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for date, row in scores.iterrows():
        ordered = row.sort_values(ascending=False)
        if n_long > 0:
            weights.loc[date, ordered.index[:n_long]] = long_w
        if n_short > 0:
            weights.loc[date, ordered.index[-n_short:]] = -short_w
    return weights


def zscore_normalized(scores: pd.DataFrame, gross_leverage: float = 1.0) -> pd.DataFrame:
    """
    Weight each name proportional to its cross-sectional z-score, scaled so
    sum(|weight|) == gross_leverage. A row with zero cross-sectional std
    (every ticker scored identically) gets all-zero weight rather than a
    division-by-zero blowup — there is no cross-sectional signal to act on.
    """
    _check_scores(scores)
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
    missing = [c for c in scores.columns if c not in returns_df.columns]
    if missing:
        raise ValidationError(f"returns_df is missing columns for: {missing}")

    vol = returns_df[scores.columns].reindex(scores.index).rolling(lookback).std()
    vol_safe = vol.where(vol > 1e-12, other=np.nan)
    adjusted = (scores / vol_safe).fillna(0.0)
    gross = adjusted.abs().sum(axis=1)
    gross_safe = gross.where(gross > 1e-12, other=1.0)
    return adjusted.div(gross_safe, axis=0) * gross_leverage


def dollar_neutral(weights: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process any weight panel so sum(weight) == 0 per row (equal dollar
    long and short), by subtracting each row's mean weight from every
    position. Preserves every pairwise weight difference exactly — only the
    common offset changes.
    """
    return weights.sub(weights.mean(axis=1), axis=0)
