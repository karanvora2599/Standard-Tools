"""
The built-in preprocessing steps.

The first three reproduce the two schemes that existed before the registry,
exactly. `winsorize` then `zscore` IS `fit_preprocessing` /
`apply_preprocessing` -- the pooled path -- and a test pins the pair equal
to those functions to 1e-12, on the Python path and on the native one.
`cross_sectional_standardize` IS `standardize_cross_sectional`. The
arithmetic is not reimplemented: the pipeline routes the default pair to
the fused native kernel when the extension is present, and the steps here
are the reference the kernel is tested against.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.estimators.bounds import (
    EstimatorParamSchema,
    ParamBound,
)
from standard_quant_tools.modeling.features.transforms import (
    standardize_cross_sectional,
)

from .base import FoldContext, Preprocessor, PreprocessorDefinition
from .registry import register_preprocessor


def _per_column(state: Dict[str, Any], key: str, columns) -> np.ndarray:
    """One state entry per column, in the frame's column order."""
    return np.array([state[key][c] for c in columns], dtype=np.float64)


class Winsorize(Preprocessor):
    """
    Clip each column to its training-fold quantiles.

    The bounds are quantiles of the TRAINING rows and are applied unchanged
    to the test rows -- a test row beyond the training 99th percentile is
    clipped to the training value, never to its own fold's. pandas' linear
    quantile, NaN skipped, which is what `fit_preprocessing` computed and
    what the native kernel matches.
    """

    id = "winsorize"
    stateless = False
    column_wise = True

    def fit(self, X: pd.DataFrame, ctx: FoldContext) -> Dict[str, Any]:
        lower = float(self.params["lower"])
        upper = float(self.params["upper"])
        if not lower < upper:
            raise ValidationError(
                f"winsorize: lower={lower} must be below upper={upper}."
            )
        return {
            "lo": {c: float(X[c].quantile(lower)) for c in X.columns},
            "hi": {c: float(X[c].quantile(upper)) for c in X.columns},
        }

    def transform(self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext):
        lo = _per_column(state, "lo", X.columns)
        hi = _per_column(state, "hi", X.columns)
        values = X.to_numpy(dtype=np.float64)
        # np.clip with NaN bounds (an all-NaN training column) would clip
        # everything to NaN; pandas' clip treats a NaN bound as no bound,
        # which is the behaviour fit_preprocessing had. Reproduce that.
        clipped = np.where(np.isnan(lo), values, np.maximum(values, lo))
        clipped = np.where(np.isnan(hi), clipped, np.minimum(clipped, hi))
        # A NaN value stays NaN: np.maximum(nan, lo) is nan already.
        return pd.DataFrame(clipped, index=X.index, columns=X.columns)


class ZScore(Preprocessor):
    """
    Centre and scale each column by its training-fold mean and standard
    deviation (ddof=1, NaN skipped). A column with no dispersion -- or too
    few rows to measure any -- is scaled by 1.0 rather than dividing by
    zero, which is what `fit_preprocessing` did.
    """

    id = "zscore"
    stateless = False
    column_wise = True

    def fit(self, X: pd.DataFrame, ctx: FoldContext) -> Dict[str, Any]:
        mean: Dict[str, float] = {}
        std: Dict[str, float] = {}
        for c in X.columns:
            column_mean = float(X[c].mean())
            column_std = float(X[c].std())
            if not column_std or pd.isna(column_std):
                column_std = 1.0
            mean[c] = column_mean
            std[c] = column_std
        return {"mean": mean, "std": std}

    def transform(self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext):
        mean = _per_column(state, "mean", X.columns)
        std = _per_column(state, "std", X.columns)
        values = (X.to_numpy(dtype=np.float64) - mean) / std
        return pd.DataFrame(values, index=X.index, columns=X.columns)


class CrossSectionalStandardize(Preprocessor):
    """
    Standardize every column within each date's cross-section.

    Stateless: each date is normalized against its own cross-section,
    contemporaneous information a live model also has, so nothing is
    fitted and nothing crosses the fold boundary. The arithmetic is
    `standardize_cross_sectional`, kernel-backed when the extension is
    present -- see that function for why clipping at `clip_sigma` replaces
    quantile winsorizing at this sample size.
    """

    id = "cross_sectional_standardize"
    stateless = True
    column_wise = True

    def fit(self, X: pd.DataFrame, ctx: FoldContext) -> Dict[str, Any]:
        return {}

    def transform(self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext):
        if ctx.dates is None or len(ctx.dates) != len(X):
            raise ValidationError(
                "cross_sectional_standardize needs one date per row: got "
                f"{0 if ctx.dates is None else len(ctx.dates)} dates for {len(X)} rows."
            )
        return standardize_cross_sectional(X, ctx.dates, float(self.params["clip_sigma"]))


class RobustScale(Preprocessor):
    """
    Centre by the training median and scale by the median absolute
    deviation.

    The estimator `zscore` is not: a single 8-sigma day moves a mean and
    dominates a standard deviation, and financial features have those. The
    median and the MAD are moved by neither. `scale_to_normal` multiplies
    the MAD by 1.4826, the constant that makes it a consistent estimate of
    the standard deviation when the data really are Gaussian, so the two
    steps agree in scale on well-behaved data and differ only where it
    matters. A column with no dispersion scales by 1.0, as `zscore` does.
    """

    id = "robust_scale"
    stateless = False
    column_wise = True

    def fit(self, X: pd.DataFrame, ctx: FoldContext) -> Dict[str, Any]:
        factor = 1.4826 if bool(self.params["scale_to_normal"]) else 1.0
        center: Dict[str, float] = {}
        scale: Dict[str, float] = {}
        for c in X.columns:
            median = float(X[c].median())
            mad = float((X[c] - median).abs().median()) * factor
            if not mad or pd.isna(mad):
                mad = 1.0
            center[c] = median
            scale[c] = mad
        return {"center": center, "scale": scale}

    def transform(self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext):
        center = _per_column(state, "center", X.columns)
        scale = _per_column(state, "scale", X.columns)
        values = (X.to_numpy(dtype=np.float64) - center) / scale
        return pd.DataFrame(values, index=X.index, columns=X.columns)


class QuantileTransform(Preprocessor):
    """
    Map each column through its training-fold empirical distribution --
    rank-gauss when `output='normal'`, a uniform [0, 1] otherwise.

    The reference is a grid of `n_quantiles` training quantiles rather than
    the whole training column, so the state stays small; a value is placed
    on that grid by linear interpolation. Beyond the training range a value
    maps to the grid's end, which is the honest answer for "further out than
    anything seen": the transform has no basis to say how far. Ties in the
    grid are collapsed to their mean probability so a flat region of the
    distribution does not map to an arbitrary point in it. NaN stays NaN.
    """

    id = "quantile_transform"
    stateless = False
    column_wise = True

    #: How far into the tails the normal map may go. Clipping the CDF here
    #: bounds the output at about +/-5.2 rather than letting a value at the
    #: grid's end become infinite.
    _EPS = 1e-7

    def fit(self, X: pd.DataFrame, ctx: FoldContext) -> Dict[str, Any]:
        n_quantiles = int(self.params["n_quantiles"])
        quantiles: Dict[str, Any] = {}
        probabilities: Dict[str, Any] = {}
        for c in X.columns:
            values = X[c].to_numpy(dtype=np.float64)
            values = values[~np.isnan(values)]
            if values.size == 0:
                quantiles[c] = None
                probabilities[c] = None
                continue
            grid = np.linspace(0.0, 1.0, min(n_quantiles, values.size))
            raw = np.quantile(values, grid)
            # Collapse tied quantile values to one knot at their mean
            # probability, so np.interp sees a strictly increasing axis.
            unique = np.unique(raw)
            knots = np.array([grid[raw == v].mean() for v in unique])
            quantiles[c] = [float(v) for v in unique]
            probabilities[c] = [float(p) for p in knots]
        return {"quantiles": quantiles, "probabilities": probabilities}

    def transform(self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext):
        from scipy.stats import norm

        output = str(self.params["output"])
        out = np.empty(X.shape, dtype=np.float64)
        for j, c in enumerate(X.columns):
            values = X[c].to_numpy(dtype=np.float64)
            knots = state["quantiles"][c]
            if knots is None:
                out[:, j] = np.nan
                continue
            if len(knots) == 1:
                cdf = np.where(np.isnan(values), np.nan, 0.5)
            else:
                cdf = np.interp(
                    values, np.asarray(knots), np.asarray(state["probabilities"][c])
                )
            if output == "normal":
                cdf = np.clip(cdf, self._EPS, 1.0 - self._EPS)
                out[:, j] = norm.ppf(cdf)
            else:
                out[:, j] = cdf
            out[np.isnan(values), j] = np.nan
        return pd.DataFrame(out, index=X.index, columns=X.columns)


class MissingIndicator(Preprocessor):
    """
    Add one `<column>__missing` indicator (1.0 where the value is NaN) per
    input column, keeping the originals.

    Stateless and deterministic in its column set: an indicator for EVERY
    input column, not only the ones that happened to be missing in this
    fold's training rows, because a column set that depended on the fold
    could not be summarized across folds or applied at scoring. On a panel
    that alignment already made complete the indicators are all zero, which
    costs a column and nothing else. Meaningful once
    `DatasetSpec.missing.policy` lets NaN reach the engine, and typically
    paired with `impute` after it.
    """

    id = "missing_indicator"
    stateless = True
    column_wise = True

    SUFFIX = "__missing"

    def fit(self, X: pd.DataFrame, ctx: FoldContext) -> Dict[str, Any]:
        return {}

    def transform(self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext):
        indicators = X.isna().astype(np.float64)
        indicators.columns = [f"{c}{self.SUFFIX}" for c in X.columns]
        return pd.concat([X, indicators], axis=1)


class Impute(Preprocessor):
    """
    Fill NaN with a statistic of the TRAINING rows -- median, mean, or a
    constant.

    The fill is fitted on the training fold and applied unchanged to the
    test fold, so a missing test value receives the training median and
    never the test fold's own, which would be a statistic of rows the model
    is about to be scored on. A column that is entirely NaN in training has
    no median; it falls back to `fill_value`, which is also what
    `strategy='constant'` uses everywhere.
    """

    id = "impute"
    stateless = False
    column_wise = True

    def fit(self, X: pd.DataFrame, ctx: FoldContext) -> Dict[str, Any]:
        strategy = str(self.params["strategy"])
        fallback = float(self.params["fill_value"])
        fill: Dict[str, float] = {}
        for c in X.columns:
            if strategy == "median":
                value = float(X[c].median())
            elif strategy == "mean":
                value = float(X[c].mean())
            else:
                value = fallback
            fill[c] = fallback if pd.isna(value) else value
        return {"fill": fill}

    def transform(self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext):
        fill = _per_column(state, "fill", X.columns)
        values = X.to_numpy(dtype=np.float64)
        return pd.DataFrame(
            np.where(np.isnan(values), fill, values), index=X.index, columns=X.columns
        )


class PCAWhiten(Preprocessor):
    """
    Replace the columns with their leading principal components, scaled to
    unit variance on the training fold.

    NOT column-wise: every output depends on every input, so a consumer
    that drops a column from a fitted matrix has to refit this step. The
    components are fitted on the training rows only -- the mean, the
    rotation and the scale -- and applied unchanged to the test rows, so a
    test-fold covariance never enters the basis. Each component's sign is
    fixed so its largest-magnitude loading is positive, which makes the fit
    reproducible from its inputs rather than from the SVD's choice.

    Refuses NaN rather than guessing: a covariance of a matrix with holes
    is not defined, and `impute` exists to put before it.
    """

    id = "pca_whiten"
    stateless = False
    column_wise = False

    def fit(self, X: pd.DataFrame, ctx: FoldContext) -> Dict[str, Any]:
        matrix = X.to_numpy(dtype=np.float64)
        n_rows, n_features = matrix.shape
        k = int(self.params["n_components"])
        if k > n_features:
            raise ValidationError(
                f"pca_whiten: n_components={k} exceeds the {n_features} input "
                "column(s). There are not that many directions to keep."
            )
        if np.isnan(matrix).any():
            raise ValidationError(
                "pca_whiten cannot fit on missing values: a covariance of a "
                "matrix with holes is not defined. Put an `impute` step before "
                "it, or keep the dataset's `drop` missing-data policy."
            )
        if n_rows < 2:
            raise ValidationError("pca_whiten needs at least two training rows.")
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:k].copy()
        # A reproducible sign: the largest-magnitude loading of each
        # component is positive.
        for i in range(components.shape[0]):
            pivot = int(np.argmax(np.abs(components[i])))
            if components[i, pivot] < 0:
                components[i] *= -1.0
        variance = (singular[:k] ** 2) / float(n_rows - 1)
        total = float((singular**2).sum() / float(n_rows - 1)) or 1.0
        if bool(self.params["whiten"]):
            scale = np.where(variance > 0.0, 1.0 / np.sqrt(np.where(variance > 0, variance, 1.0)), 1.0)
        else:
            scale = np.ones(k, dtype=np.float64)
        return {
            "mean": [float(v) for v in mean],
            "components": [[float(v) for v in row] for row in components],
            "scale": [float(v) for v in scale],
            "explained_variance_ratio": [float(v / total) for v in variance],
        }

    def transform(self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext):
        mean = np.asarray(state["mean"], dtype=np.float64)
        components = np.asarray(state["components"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        projected = (X.to_numpy(dtype=np.float64) - mean) @ components.T * scale
        columns = [f"pc{i + 1}" for i in range(components.shape[0])]
        return pd.DataFrame(projected, index=X.index, columns=columns)


QUANTILE = ParamBound("float", 0.0, 1.0)

register_preprocessor(
    PreprocessorDefinition(
        id=Winsorize.id,
        description=(
            "Clip each column to its training-fold quantiles, so a single "
            "extreme print cannot set the scale for everything that follows."
        ),
        cls=Winsorize,
        schema=EstimatorParamSchema(bounds={"lower": QUANTILE, "upper": QUANTILE}),
        default_params={"lower": 0.01, "upper": 0.99},
    )
)
register_preprocessor(
    PreprocessorDefinition(
        id=ZScore.id,
        description=(
            "Centre and scale each column by its training-fold mean and "
            "standard deviation. Leaves the market factor inside every "
            "feature; pair with cross_sectional_standardize for a model "
            "judged on cross-sectional IC."
        ),
        cls=ZScore,
        schema=EstimatorParamSchema(bounds={}),
        default_params={},
    )
)
register_preprocessor(
    PreprocessorDefinition(
        id=CrossSectionalStandardize.id,
        description=(
            "Standardize within each date's cross-section and clip at "
            "clip_sigma, so what reaches the model is each entity's position "
            "relative to its peers that day. Stateless: nothing crosses the "
            "fold boundary."
        ),
        cls=CrossSectionalStandardize,
        schema=EstimatorParamSchema(
            bounds={
                "clip_sigma": ParamBound(
                    "float",
                    0.0,
                    100.0,
                    note="Standard deviations to clip at after standardizing; 0 disables.",
                )
            }
        ),
        default_params={"clip_sigma": 3.0},
    )
)

register_preprocessor(
    PreprocessorDefinition(
        id=RobustScale.id,
        description=(
            "Centre by the training median and scale by the median absolute "
            "deviation, which a single extreme print cannot move; "
            "scale_to_normal makes the MAD a consistent estimate of the "
            "standard deviation on Gaussian data."
        ),
        cls=RobustScale,
        schema=EstimatorParamSchema(bounds={"scale_to_normal": ParamBound("bool")}),
        default_params={"scale_to_normal": True},
    )
)
register_preprocessor(
    PreprocessorDefinition(
        id=QuantileTransform.id,
        description=(
            "Map each column through its training-fold empirical distribution: "
            "rank-gauss for output='normal', a uniform [0, 1] for 'uniform'. "
            "Removes the shape of the distribution entirely, tails included."
        ),
        cls=QuantileTransform,
        schema=EstimatorParamSchema(
            bounds={
                "n_quantiles": ParamBound(
                    "int", 10, 10_000, note="Knots in the reference grid; the state grows with it."
                ),
                "output": ParamBound("str", choices=("normal", "uniform")),
            }
        ),
        default_params={"n_quantiles": 1000, "output": "normal"},
    )
)
register_preprocessor(
    PreprocessorDefinition(
        id=MissingIndicator.id,
        description=(
            "Add a <column>__missing indicator (1.0 where NaN) for every input "
            "column, keeping the originals. Meaningful once the dataset's "
            "missing-data policy lets NaN reach the engine; pair with impute."
        ),
        cls=MissingIndicator,
        schema=EstimatorParamSchema(bounds={}),
        default_params={},
    )
)
register_preprocessor(
    PreprocessorDefinition(
        id=Impute.id,
        description=(
            "Fill NaN with the training fold's median or mean, or a constant. "
            "A missing test value receives the TRAINING statistic, never the "
            "test fold's own."
        ),
        cls=Impute,
        schema=EstimatorParamSchema(
            bounds={
                "strategy": ParamBound("str", choices=("median", "mean", "constant")),
                "fill_value": ParamBound("float", -1e12, 1e12),
            }
        ),
        default_params={"strategy": "median", "fill_value": 0.0},
    )
)
register_preprocessor(
    PreprocessorDefinition(
        id=PCAWhiten.id,
        description=(
            "Replace the columns with their leading n_components principal "
            "components, fitted on the training fold and scaled to unit "
            "variance when whiten is set. Not column-wise: every output depends "
            "on every input. Refuses NaN; put impute before it."
        ),
        cls=PCAWhiten,
        schema=EstimatorParamSchema(
            bounds={
                "n_components": ParamBound(
                    "int", 1, 64, note="Bounded like every other width in the registry."
                ),
                "whiten": ParamBound("bool"),
            }
        ),
        default_params={"n_components": 8, "whiten": True},
    )
)

__all__ = [
    "CrossSectionalStandardize",
    "Impute",
    "MissingIndicator",
    "PCAWhiten",
    "QuantileTransform",
    "RobustScale",
    "Winsorize",
    "ZScore",
]
