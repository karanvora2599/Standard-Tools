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


def apply_preprocessing(df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Apply stats produced by fit_preprocessing (fit on train) to any
    frame — train or test — sharing the same feature columns."""
    out = df.copy()
    for col, s in stats.items():
        clipped = out[col].clip(lower=s["lo"], upper=s["hi"])
        out[col] = (clipped - s["mean"]) / s["std"]
    return out
