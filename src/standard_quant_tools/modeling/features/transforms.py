"""Feature-column transforms applied inside dataset.builder / engine.py.
engine.py fits these on train-fold statistics only and applies the same
fitted stats to the test fold — never fit on the full frame first — the
leakage discipline validation/walk_forward.py's split boundary exists to
protect."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

# Optional native fast path, on the same terms as the rest of the package:
# the extension may be absent, and the Python below stays the reference
# implementation and the test oracle. Preprocessing was measured at 47-56%
# of a walk-forward run -- more than the estimator fit and the metrics
# combined -- which is why it is the part that got a kernel.
_cpp_core: Any = None
HAS_CPP = False
try:
    from standard_quant_tools import (
        _sqt_core as _cpp_core,  # type: ignore[attr-defined]
    )

    HAS_CPP = hasattr(_cpp_core, "fit_preprocess_stats")
except ImportError:
    pass

_WINSOR_LOW = 0.01
_WINSOR_HIGH = 0.99


def _native_matrix(frame: pd.DataFrame) -> Optional[np.ndarray]:
    """
    The frame as a C-contiguous float64 matrix, or None if it is not the
    shape the kernel accepts.

    A non-numeric column is the disqualifying case: `to_numpy(dtype=float)`
    would raise on it, and the Python path handles it (or fails with a
    clearer message) perfectly well.
    """
    if frame.empty or frame.shape[1] == 0:
        return None
    try:
        return np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
    except (TypeError, ValueError):
        return None


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
    matrix = _native_matrix(train) if HAS_CPP else None
    if matrix is not None:
        native = _cpp_core.fit_preprocess_stats(matrix, _WINSOR_LOW, _WINSOR_HIGH)
        return {
            col: {
                "lo": float(native["lo"][i]),
                "hi": float(native["hi"][i]),
                "mean": float(native["mean"][i]),
                "std": float(native["std"][i]),
            }
            for i, col in enumerate(train.columns)
        }

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
    # The native path transforms the WHOLE matrix in one fused pass, so it
    # only applies when the frame is exactly the fitted columns in the
    # fitted order. A frame carrying extra columns, or a partial stats dict,
    # goes down the per-column path, which copies the untouched columns
    # through unchanged — that is a real calling convention here (the engine
    # passes df[feature_ids]) and not worth a second kernel.
    columns: List[str] = list(stats)
    if HAS_CPP and list(df.columns) == columns:
        matrix = _native_matrix(df)
        if matrix is not None:
            transformed = _cpp_core.apply_preprocess_stats(
                matrix,
                np.array([stats[c]["lo"] for c in columns], dtype=np.float64),
                np.array([stats[c]["hi"] for c in columns], dtype=np.float64),
                np.array([stats[c]["mean"] for c in columns], dtype=np.float64),
                np.array([stats[c]["std"] for c in columns], dtype=np.float64),
            )
            return pd.DataFrame(transformed, index=df.index, columns=df.columns)

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
    # Validated on the PYTHON side so both paths agree. The native kernel
    # already rejects a negative clip_sigma; without this the same call
    # raised ValueError on a machine with the extension built and silently
    # skipped clipping on one without it -- a backend divergence, and the
    # kind that only shows up as two users getting different numbers.
    if not (clip_sigma >= 0.0):
        raise ValidationError(
            f"standardize_cross_sectional: clip_sigma must be >= 0, got "
            f"{clip_sigma}. Use 0.0 to disable clipping."
        )
    if frame.empty:
        return frame.copy()

    codes = pd.factorize(np.asarray(dates), sort=False)[0]
    if HAS_CPP and hasattr(_cpp_core, "standardize_by_date"):
        matrix = _native_matrix(frame)
        if matrix is not None:
            standardized = _cpp_core.standardize_by_date(
                matrix,
                np.ascontiguousarray(codes.astype(np.int64)),
                int(codes.max()) + 1 if codes.size else 0,
                float(clip_sigma),
            )
            return pd.DataFrame(standardized, index=frame.index, columns=frame.columns)

    values = frame.to_numpy(dtype=np.float64, copy=True)
    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    block = values[order]

    starts = np.flatnonzero(np.r_[True, codes_sorted[1:] != codes_sorted[:-1]])
    counts = np.diff(np.r_[starts, codes_sorted.size]).astype(np.float64)

    # Per-date mean and (ddof=1) standard deviation, one reduceat pass per
    # statistic over the whole block rather than a groupby per column.
    #
    # NaN IS SKIPPED BY THE MOMENTS AND PRESERVED IN THE OUTPUT. It did not
    # used to be: a plain reduceat propagates one NaN into the whole date's
    # mean, and the final non-finite sweep then mapped every entity in that
    # date to 0.0 -- so one missing name reported every OTHER name as
    # sitting exactly at the cross-sectional mean. That is the specific
    # fabricated observation panel_stats.hpp says this must not produce.
    #
    # It was known, deliberate and documented as a wart in panel_stats.cpp,
    # on the reasoning that "in practice it never fires: alignment drops NaN
    # rows before the panel reaches the engine." `load_external_panel` broke
    # that premise -- an externally computed panel keeps its warm-up NaNs --
    # so the wart became reachable and had to go.
    row_counts = counts.astype(np.int64)
    present = ~np.isnan(block)
    n_valid = np.add.reduceat(present, starts, axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = np.add.reduceat(np.where(present, block, 0.0), starts, axis=0)
        mean = total / np.where(n_valid > 0.0, n_valid, np.nan)
        centered = block - np.repeat(mean, row_counts, axis=0)
        sum_sq = np.add.reduceat(
            np.where(present, centered * centered, 0.0), starts, axis=0
        )
        variance = sum_sq / np.maximum(n_valid - 1.0, 1.0)
        std = np.sqrt(variance)
        std = np.where(std > 0.0, std, np.nan)
        standardized = centered / np.repeat(std, row_counts, axis=0)

    # A PRESENT entity in a cross-section with no dispersion -- flat, or a
    # single usable name -- sits exactly at the mean, so it is 0.0 rather
    # than NaN, which would drop the whole date downstream. An ABSENT one
    # stays absent.
    standardized = np.where(
        present, np.where(np.isfinite(standardized), standardized, 0.0), np.nan
    )
    if clip_sigma > 0:
        np.clip(standardized, -clip_sigma, clip_sigma, out=standardized)

    out = np.empty_like(standardized)
    out[order] = standardized
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)
