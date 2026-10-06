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

An infinite value is refused by `winsorize` and `zscore` with the default
fit's own refusal, in the rows they are fitted on and in the rows they
transform. The default pair goes through the fused path, which refuses an
infinity in the training rows and clips one in the rows it applies to, as
it always has; the two agree on every finite frame.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from standard_quant_tools._blas import single_threaded_blas
from standard_quant_tools.error import ValidationError
from standard_quant_tools.modeling.estimators.bounds import (
    EstimatorParamSchema,
    ParamBound,
)
from standard_quant_tools.modeling.features.transforms import (
    _refuse_infinite_training_values,
    cross_sectional_counts,
    rank_within_date,
    standardize_cross_sectional,
)

from .base import FoldContext, Preprocessor, PreprocessorDefinition
from .registry import register_preprocessor


def _refuse_infinite_rows(X: pd.DataFrame, func: str) -> None:
    """
    Refuse +/-inf in the rows a fitted step is about to transform, worded
    as the default fit's refusal (`_refuse_infinite_training_values`) is.

    The fit refuses an infinity among the training rows; this is the same
    door at transform, where an infinity would come out of `zscore` as an
    infinite feature value and out of `winsorize` clipped to a bound, as
    though it were an ordinary extreme.
    """
    numeric = X.select_dtypes(include="number")
    if numeric.shape[1] == 0:
        return
    bad = np.isinf(numeric.to_numpy(dtype=np.float64))
    if not bad.any():
        return
    per_column = bad.sum(axis=0)
    named = [
        f"{col!r} ({int(count)})"
        for col, count in zip(numeric.columns, per_column)
        if count
    ]
    raise ValidationError(
        f"{func}: the rows to transform hold infinite values in column(s) "
        f"{', '.join(named[:5])}{' and more' if len(named) > 5 else ''}. An "
        "infinity has no z-score and lies beyond every fitted bound, so the "
        "transformed value would be infinite, or clipped to a bound as though "
        "it were an ordinary extreme. Mark the value missing with NaN, which "
        "the transform keeps as NaN, or repair or drop the rows."
    )


def _per_column(state: Dict[str, Any], key: str, columns) -> np.ndarray:
    """One state entry per column, in the frame's column order."""
    return np.array([state[key][c] for c in columns], dtype=np.float64)


def _column_quantiles(X: pd.DataFrame, lower: float, upper: float):
    """
    `{c: float(X[c].quantile(lower))}` and the same for `upper`, every
    column, the doubles those per-column calls return -- from one
    `X.quantile([lower, upper])`.

    Two `Series.quantile` calls per column were the whole of the fit; one
    frame call does every column and both bounds in one pass. pandas
    computes each column's quantiles with one numpy percentile call over
    that column, and with both bounds in the call the column is partitioned
    at both positions at once. That puts the same value at each position,
    but values that compare equal -- -0.0 and +0.0, or two NaNs -- can land
    in a different order, and an interpolated bound can then differ in its
    bits, though only when it is zero or NaN. Those bounds are recomputed
    the per-column way, so the state is the per-column state to the bit.

    Frames the frame call might treat differently from a column at a time
    -- repeated labels, where `X[c]` is itself a frame, or any column that
    is not float64 -- keep the per-column calls.
    """
    columns = X.columns
    one_call = (
        len(columns) > 0
        and columns.is_unique
        and all(dtype == np.float64 for dtype in X.dtypes)
    )
    if not one_call:
        return (
            {c: float(X[c].quantile(lower)) for c in columns},
            {c: float(X[c].quantile(upper)) for c in columns},
        )
    bounds = X.quantile([lower, upper]).to_numpy(dtype=np.float64)
    lo: Dict[Any, float] = {}
    hi: Dict[Any, float] = {}
    for k, c in enumerate(columns):
        low, high = bounds[0, k], bounds[1, k]
        if low == 0.0 or np.isnan(low):
            low = X[c].quantile(lower)
        if high == 0.0 or np.isnan(high):
            high = X[c].quantile(upper)
        lo[c] = float(low)
        hi[c] = float(high)
    return lo, hi


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
        _refuse_infinite_training_values(X, "winsorize")
        lo, hi = _column_quantiles(X, lower, upper)
        return {"lo": lo, "hi": hi}

    def transform(self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext):
        _refuse_infinite_rows(X, "winsorize")
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
        _refuse_infinite_training_values(X, "zscore")
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
        _refuse_infinite_rows(X, "zscore")
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
        return standardize_cross_sectional(
            X, ctx.dates, float(self.params["clip_sigma"])
        )


class CrossSectionalRank(Preprocessor):
    """
    Replace each value with its rank inside that date's cross-section,
    mapped to [-0.5, 0.5].

    WHY A RANK AND NOT A STANDARDIZATION. `cross_sectional_standardize`
    subtracts a mean and divides by a standard deviation, and both are
    moved by the same fat tails the features have: one 8-sigma name sets
    the scale for every other name that day. A rank is moved by none of
    them -- it is immune to the distribution's shape entirely, and only
    the ORDER survives, which for a model judged on cross-sectional IC is
    the part being scored.

    This is the argument the library already makes on the label side.
    `forward_return_rank` "matches how the model is SCORED, which is the
    cross-sectional rank IC, and is immune to a fat-tailed return
    distribution", and the measured failure on the feature side is on
    record in `Documentation/15_modeling.md`: a feature whose
    cross-sectional standard deviation ran 0.23 to 22.4, where "a rank of
    it sorted names by price level as much as by momentum".

    THE MAPPING IS THE LABEL'S. `(rank - 1) / (n - 1) - 0.5`, exactly what
    `targets/builtin._stage_rank` applies, so a ranked feature and a ranked
    target are on the same scale and a coefficient between them means what
    it looks like. Raw ranks would not be: a 30-name date ranks 1..30 and a
    500-name date 1..500, so the same feature would carry a different scale
    on every date.

    Stateless, for the reason `cross_sectional_standardize` is: each date
    is ranked against its own cross-section, which is contemporaneous
    information a live model also has, so nothing is fitted and nothing
    crosses the fold boundary.

    A date with one entity has no cross-section and becomes NaN rather than
    a fabricated 0.0 -- the same decision `_stage_rank` makes, where it says
    a one-name rank "is not a measurement". The impute step then treats it
    like any other missing value, and a panel of mostly single-entity dates
    produces an all-NaN column, which the fold check refuses by name.
    """

    id = "cross_sectional_rank"
    stateless = True
    column_wise = True

    def fit(self, X: pd.DataFrame, ctx: FoldContext) -> Dict[str, Any]:
        return {}

    def transform(self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext):
        if ctx.dates is None or len(ctx.dates) != len(X):
            raise ValidationError(
                "cross_sectional_rank needs one date per row: got "
                f"{0 if ctx.dates is None else len(ctx.dates)} dates for "
                f"{len(X)} rows."
            )
        if X.empty or X.shape[1] == 0:
            return X.copy()
        dates = np.asarray(ctx.dates)
        ranks = rank_within_date(X, dates)
        counts = cross_sectional_counts(X, dates)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = (ranks - 1.0) / (counts - 1.0) - 0.5
        return values.where(counts > 1)


class GroupDemean(Preprocessor):
    """
    Subtract each date's GROUP mean, so what reaches the model is the
    entity's position within its own sector rather than its sector's.

    A momentum feature on a panel of banks and miners carries the
    industry's move as well as the name's; a model fitted on it learns the
    industry rotation and reports it as stock selection. Demeaning within
    the group removes the part every name in that group shared.

    THE GROUPS ARE A PARAMETER AND NOT A LOOKUP. A step that asked a
    provider for a sector at fit time would neutralise differently next
    month against the same panel, so a registered model would not
    reproduce: its manifest would describe a pipeline whose behaviour lives
    outside it. It is also the survivorship shape this repo documents for
    universe membership — today's classification applied to history. Passed
    in, the map is validated at the spec boundary, hashed into the dataset
    and model identity, and written into `preprocessing_state.json` with
    everything else. `sector_groups` builds one from a provider, once, and
    returns the warning that belongs with it.

    Stateless, for the reason `cross_sectional_standardize` is: each date's
    groups are formed from that date's own cross-section, which is
    contemporaneous information a live model also has, so nothing crosses
    the fold boundary.

    A GROUP OF ONE, AND AN ENTITY WITH NO GROUP, ARE BOTH NaN. A singleton
    group's mean is the name itself, so demeaning it gives exactly 0.0 for
    every such row — a fabricated "average for its sector" that destroys
    the feature while looking like a measurement. NaN is the same call
    `targets/builtin._stage_rank` makes where it says a one-name
    cross-section "is not a measurement", and the same one
    `cross_sectional_rank` makes. The impute step then treats it like any
    other hole, the engine reports it in `missing_rate_train`, and a column
    that is NaN in every training row of a fold is refused by name.
    """

    id = "group_demean"
    stateless = True
    column_wise = True

    def fit(self, X: pd.DataFrame, ctx: FoldContext) -> Dict[str, Any]:
        return {}

    def transform(self, X: pd.DataFrame, state: Dict[str, Any], ctx: FoldContext):
        if ctx.dates is None or len(ctx.dates) != len(X):
            raise ValidationError(
                "group_demean needs one date per row: got "
                f"{0 if ctx.dates is None else len(ctx.dates)} dates for "
                f"{len(X)} rows."
            )
        if ctx.entities is None:
            raise ValidationError(
                "group_demean needs the entity of each row to look its group "
                "up, and this panel carries none. A panel without entity "
                "identity has no cross-section to neutralise within."
            )
        groups = self.params.get("groups") or {}
        if not groups:
            raise ValidationError(
                "group_demean was given no `groups` map. The groups are a "
                "parameter rather than a lookup on purpose — a sector read "
                "at fit time would neutralise differently next month and a "
                "registered model would not reproduce. Build one with "
                "`sector_groups` and pass it in."
            )
        if X.empty or X.shape[1] == 0:
            return X.copy()

        labels = np.array(
            [str(groups.get(str(entity), "")) for entity in ctx.entities],
            dtype=object,
        )
        keys = [np.asarray(ctx.dates), labels]
        grouped = X.groupby(keys, sort=False)
        demeaned = X - grouped.transform("mean")
        # A group of one has no within-group position to report, and an
        # unmapped entity has no group at all. Both are NaN rather than a
        # fabricated zero.
        demeaned = demeaned.where(grouped.transform("count") > 1)
        demeaned[labels == ""] = np.nan
        return demeaned


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
            #
            # QUADRATIC IN n_quantiles AS A LIST COMPREHENSION. `grid[raw
            # == v]` rescans the whole of `raw` once per unique value --
            # 10^6 comparisons per column at the default 1,000 and 10^8 at
            # the 10,000 ceiling, for a step whose whole job is a table
            # lookup. `return_inverse` gives each element's bucket in one
            # pass and `bincount` sums the grid into those buckets, so the
            # mean per knot is two linear passes over the same arrays:
            # O(q log q) for the sort inside `unique`, and nothing else.
            # The knots are identical -- the same groups, the same means.
            unique, inverse = np.unique(raw, return_inverse=True)
            knots = np.bincount(inverse, weights=grid) / np.bincount(inverse)
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
        # One BLAS thread, as the walk-forward pools give every fit: the
        # full-panel refit runs off the pools on the caller's threads, and
        # from 20,000 rows of 30 features the decomposition's last bits
        # differed there between one thread and two or four.
        with single_threaded_blas():
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
            scale = np.where(
                variance > 0.0,
                1.0 / np.sqrt(np.where(variance > 0, variance, 1.0)),
                1.0,
            )
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
        centred = X.to_numpy(dtype=np.float64) - mean
        # The projection on one BLAS thread too: at 20,000 rows its last
        # bits differed between one thread and sixteen.
        with single_threaded_blas():
            projected = centred @ components.T * scale
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
        id=CrossSectionalRank.id,
        description=(
            "Replace each value with its rank inside that date's "
            "cross-section, mapped to [-0.5, 0.5] — the same mapping the "
            "forward_return_rank TARGET uses, so feature and label share a "
            "scale. Immune to the fat tails that move a mean and a standard "
            "deviation, so pair it with a rank target for a model judged on "
            "cross-sectional IC. Stateless: nothing crosses the fold "
            "boundary. A one-entity date has no cross-section and becomes "
            "NaN."
        ),
        cls=CrossSectionalRank,
        schema=EstimatorParamSchema(bounds={}),
        default_params={},
    )
)

register_preprocessor(
    PreprocessorDefinition(
        id=GroupDemean.id,
        description=(
            "Subtract each date's GROUP mean, so the model sees an entity's "
            "position within its sector rather than its sector's move. "
            "`groups` is an entity -> label map and is a PARAMETER, not a "
            "lookup: a sector read at fit time would neutralise differently "
            "next month, so a registered model would not reproduce, and it "
            "would apply today's classification to history. Stateless. A "
            "group of one, and an entity with no group, are NaN rather than "
            "a fabricated zero."
        ),
        cls=GroupDemean,
        schema=EstimatorParamSchema(bounds={}),
        default_params={"groups": {}},
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
                    "int",
                    10,
                    10_000,
                    note="Knots in the reference grid; the state grows with it.",
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
    "CrossSectionalRank",
    "GroupDemean",
    "CrossSectionalStandardize",
    "Impute",
    "MissingIndicator",
    "PCAWhiten",
    "QuantileTransform",
    "RobustScale",
    "Winsorize",
    "ZScore",
]
