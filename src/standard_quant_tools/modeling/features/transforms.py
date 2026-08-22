"""Feature-column transforms applied inside dataset.builder / engine.py.
engine.py fits these on train-fold statistics only and applies the same
fitted stats to the test fold — never fit on the full frame first — the
leakage discipline validation/walk_forward.py's split boundary exists to
protect."""

from typing import Dict

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip `series` to its own [lower, upper] quantiles."""
    if not (0.0 <= lower < upper <= 1.0):
        raise ValidationError(
            f"winsorize: need 0 <= lower < upper <= 1, got ({lower}, {upper})"
        )
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def zscore_time_series(series: pd.Series) -> pd.Series:
    """Z-score `series` against its own mean/std (one entity across dates)."""
    mean, std = series.mean(), series.std()
    if not std or pd.isna(std):
        return series * 0.0
    return (series - mean) / std


def zscore_cross_sectional(
    panel: pd.DataFrame, column: str, date_col: str = "date"
) -> pd.Series:
    """Z-score `column` within each date's cross-section (all entities on
    the same date share one mean/std) — a different shape of operation
    than zscore_time_series, so it takes the long panel plus a column
    name rather than a single Series."""
    grouped = panel.groupby(date_col)[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return (panel[column] - mean) / std


def fit_preprocessing(train: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Fit per-column winsorize bounds (1st/99th percentile) + zscore
    mean/std on `train` only — the fold-boundary leakage discipline
    engine.py exists to enforce: these stats are computed once per fold
    from the training rows, then the SAME stats are applied to both train
    and test via apply_preprocessing, never refit on test.
    """
    stats: Dict[str, Dict[str, float]] = {}
    for col in train.columns:
        lo, hi = float(train[col].quantile(0.01)), float(train[col].quantile(0.99))
        clipped = train[col].clip(lower=lo, upper=hi)
        mean, std = float(clipped.mean()), float(clipped.std())
        if not std or pd.isna(std):
            std = 1.0
        stats[col] = {"lo": lo, "hi": hi, "mean": mean, "std": std}
    return stats


def apply_preprocessing(
    df: pd.DataFrame, stats: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Apply stats produced by fit_preprocessing (fit on train) to any
    frame — train or test — sharing the same feature columns."""
    out = df.copy()
    for col, s in stats.items():
        clipped = out[col].clip(lower=s["lo"], upper=s["hi"])
        out[col] = (clipped - s["mean"]) / s["std"]
    return out


def standardize_cross_sectional(
    frame: pd.DataFrame, dates: np.ndarray, clip_sigma: float = 3.0
) -> pd.DataFrame:
    """
    Standardize every column WITHIN each date's cross-section.

    WHY THIS IS A DIFFERENT ANSWER, NOT A DIFFERENT FLAVOUR. Pooled
    z-scoring (fit_preprocessing) computes one mean and standard deviation
    over the whole training panel, which leaves the market factor sitting
    inside every feature: on a day the whole market rallies, every entity's
    momentum reads high together, and a model fed those features can score
    well by learning "today was an up day" rather than "this name is strong
    relative to its peers". For a model whose scorecard is cross-sectional
    IC, that is the wrong thing to have learned. Standardizing within the
    date removes the common component by construction, so what reaches the
    estimator is each entity's position relative to its peers that day.

    NO FOLD-BOUNDARY PROBLEM. Unlike the pooled statistics, these are not
    fitted on train and applied to test: each date is standardized using
    only its own cross-section, which is contemporaneous information — on
    the test date you genuinely do know every entity's features for that
    date. So there is nothing here to leak across the split.

    CLIPPING RATHER THAN QUANTILE WINSORIZING. The pooled path clips to the
    1st/99th percentile, which is meaningful over tens of thousands of
    pooled rows and meaningless within one date: the 1st percentile of a
    20-name cross-section is just its minimum, so "winsorizing" would clip
    the extreme observation to itself and do nothing at all. Clipping at
    `clip_sigma` standard deviations after standardizing is the transform
    that actually bounds an outlier at this sample size.

    A date whose cross-section is constant (or has one usable entity) has
    no dispersion to divide by; those rows become 0.0 — the value they are
    standardized to be, since every entity sits exactly at the mean.
    """
    if frame.empty:
        return frame.copy()

    values = frame.to_numpy(dtype=np.float64, copy=True)
    codes = pd.factorize(np.asarray(dates), sort=False)[0]
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    block = values[order]

    starts = np.flatnonzero(np.r_[True, codes_sorted[1:] != codes_sorted[:-1]])
    counts = np.diff(np.r_[starts, codes_sorted.size]).astype(np.float64)
    widths = counts[:, None]

    # Per-date mean and (ddof=1) standard deviation, one reduceat pass per
    # statistic over the whole block rather than a groupby per column.
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.add.reduceat(block, starts, axis=0) / widths
        centered = block - np.repeat(mean, counts.astype(np.int64), axis=0)
        sum_sq = np.add.reduceat(centered * centered, starts, axis=0)
        variance = sum_sq / np.maximum(widths - 1.0, 1.0)
        std = np.sqrt(variance)
        std = np.where(std > 0.0, std, np.nan)
        standardized = centered / np.repeat(std, counts.astype(np.int64), axis=0)

    # A flat cross-section leaves every entity exactly at the mean.
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    if clip_sigma > 0:
        np.clip(standardized, -clip_sigma, clip_sigma, out=standardized)

    out = np.empty_like(standardized)
    out[order] = standardized
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)
