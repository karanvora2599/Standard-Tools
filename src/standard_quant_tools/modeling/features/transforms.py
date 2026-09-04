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


# `zscore_time_series` and `zscore_cross_sectional` both stood here and both
# are gone. The second was the predecessor `standardize_cross_sectional` was
# built ON, left behind disagreeing with its own successor -- NaN for a
# constant cross-section where the successor gives 0.0. The first survived
# only because that one's docstring named it by contrast; with the contrast
# deleted it had no reference in the repo at all, no test and no mention in
# any document.
#
# The cross-sectional work lives in `standardize_cross_sectional`, which
# takes a whole frame rather than a column and carries the sigma clipping
# and the NaN rule that make it correct on a real panel.


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


def rank_within_date(frame: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
    """
    Average rank of every value within its own date's cross-section.

    `Series.rank(method="average")` per column per date: 1-based, ties take
    the mean of the ordinals they span, NaN is skipped by the ranking and
    preserved in the output. A missing name does not shift the ranks of the
    names that are present.

    This is the one place in the modelling layer where VECTORISATION LOST.
    pandas' `groupby.rank` is already the good implementation -- a numpy
    rewrite measured 395 ms against its 407 ms at three columns, and slower
    at five -- so the only way past it was the kernel, which does the same
    work in 22 ms. That is why this exists as a kernel and almost nothing
    else in this layer does.

    Three callers rank within a date and each had its own copy of the
    pandas idiom: the ensemble combiner, the feature report's rank turnover,
    and the cross-sectional rank target.
    """
    if frame.empty or frame.shape[1] == 0:
        return frame.copy()

    codes, uniques = pd.factorize(np.asarray(dates), sort=False)
    if HAS_CPP and hasattr(_cpp_core, "rank_by_date"):
        matrix = _native_matrix(frame)
        if matrix is not None:
            ranked = _cpp_core.rank_by_date(
                matrix,
                np.ascontiguousarray(codes.astype(np.int64)),
                int(len(uniques)),
            )
            return pd.DataFrame(ranked, index=frame.index, columns=frame.columns)

    grouped = frame.groupby(codes, sort=False)
    return grouped.rank(method="average")


def cross_sectional_counts(frame: pd.DataFrame, dates: np.ndarray) -> pd.DataFrame:
    """
    How many entities carried a value for each column on each date.

    The companion to `rank_within_date`: a rank means nothing without the
    size of the cross-section it was taken in, and every caller needs both.
    """
    codes, uniques = pd.factorize(np.asarray(dates), sort=False)
    n_dates = int(len(uniques))
    values = frame.to_numpy(dtype=np.float64)
    present = ~np.isnan(values)
    counts = np.empty_like(values)
    for column in range(values.shape[1]):
        per_date = np.bincount(codes[present[:, column]], minlength=n_dates).astype(
            np.float64
        )
        counts[:, column] = per_date[codes]
    return pd.DataFrame(counts, index=frame.index, columns=frame.columns)


def fit_and_apply_preprocessing(
    train: pd.DataFrame, test: pd.DataFrame
) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """
    Fit winsorize/zscore statistics on `train` and apply them to both,
    materialising each frame exactly ONCE.

    Identical in result to `fit_preprocessing` followed by two
    `apply_preprocessing` calls. It exists because that sequence converts
    the training block to a C-contiguous matrix twice: `to_numpy` returns
    the column block transposed, so it is F-contiguous, and the
    `ascontiguousarray` the kernel needs is a full copy. On a 100,000 x 20
    training block that copy measured 3.7 ms, paid twice per fold, on every
    fold of every walk-forward run and every candidate of every
    hyperparameter search.

    Falls back to the public pair whenever the fast path does not apply, so
    there is exactly one implementation of the arithmetic.
    """
    train_matrix = _native_matrix(train) if HAS_CPP else None
    test_matrix = _native_matrix(test) if HAS_CPP else None
    same_columns = list(train.columns) == list(test.columns)
    if train_matrix is None or test_matrix is None or not same_columns:
        stats = fit_preprocessing(train)
        return apply_preprocessing(train, stats), apply_preprocessing(test, stats)

    native = _cpp_core.fit_preprocess_stats(train_matrix, _WINSOR_LOW, _WINSOR_HIGH)
    lo = np.asarray(native["lo"], dtype=np.float64)
    hi = np.asarray(native["hi"], dtype=np.float64)
    mean = np.asarray(native["mean"], dtype=np.float64)
    std = np.asarray(native["std"], dtype=np.float64)
    return (
        pd.DataFrame(
            _cpp_core.apply_preprocess_stats(train_matrix, lo, hi, mean, std),
            index=train.index,
            columns=train.columns,
        ),
        pd.DataFrame(
            _cpp_core.apply_preprocess_stats(test_matrix, lo, hi, mean, std),
            index=test.index,
            columns=test.columns,
        ),
    )


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
