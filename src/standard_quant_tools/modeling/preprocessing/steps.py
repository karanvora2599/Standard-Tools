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

__all__ = ["CrossSectionalStandardize", "Winsorize", "ZScore"]
