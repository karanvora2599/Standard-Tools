"""
Universe-scope PCA-derived factor features — wraps analysis.pca.pca_returns,
the same cross-sectional factor tool run_pca_analysis (agent/tools.py)
exposes as a standalone report, reused here as model features instead.
PCA needs the whole universe's return panel at once, not one symbol at a
time, which is why these declare scope=UNIVERSE (see features/base.py's
docstring) rather than fitting the entity-scope fn(ohlcv, ...) contract
every other feature in this package uses.

Both features share one rolling-refit design: refitting PCA on every
single bar would be wasted work for a value (a factor's own composition)
that doesn't move much day to day, so PCA is refit only every
`refit_every` bars over a trailing `window`-bar panel, and the fitted PC1
is held fixed until the next refit. Each refit calls pca_returns with
method="power_iteration" rather than the default full SVD -- since only
PC1 is ever needed here, power iteration is meaningfully cheaper (it
solves for just the requested component instead of every singular
triplet regardless of how many were asked for), with no accuracy cost
for a real, factor-structured universe where PC1's eigenvalue is
well-separated from the rest (see pca_returns's `method` docstring for
the one case -- near-degenerate eigenvalues -- where this wouldn't hold,
which doesn't apply here since only the single dominant component is used):

  factors.pca_loading      — each entity's PC1 loading, forward-filled
                              between refits.
  factors.pca_factor_return — that date's realized return projected onto
                              the currently-held PC1 loadings (dot
                              product), so it updates every bar even
                              though the loadings themselves only change
                              at each refit.
"""

import numpy as np
import pandas as pd

from standard_quant_tools.analysis.pca import pca_returns as _pca_returns
from standard_quant_tools.error import ValidationError

from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import register_feature


def _validate_window_params(window: int, refit_every: int, feature_id: str) -> None:
    """Both loop bounds below use `window`/`refit_every` as a `range()`
    step — an unvalidated refit_every=0 crashes with Python's cryptic
    `ValueError: range() arg 3 must not be zero` deep inside feature
    computation instead of a clear, attributable error; window<2 would
    let a single-observation slice reach pca_returns, which needs at
    least 2 observations to fit."""
    if window < 2:
        raise ValidationError(f"{feature_id}: window must be >= 2, got {window}")
    if refit_every < 1:
        raise ValidationError(f"{feature_id}: refit_every must be >= 1, got {refit_every}")


def _pca_loading(
    returns_panel: pd.DataFrame,
    context: FeatureContext,
    window: int = 252,
    refit_every: int = 21,
) -> pd.DataFrame:
    _validate_window_params(window, refit_every, "factors.pca_loading")
    n = len(returns_panel)
    out = pd.DataFrame(np.nan, index=returns_panel.index, columns=returns_panel.columns)
    for end in range(window, n + 1, refit_every):
        window_slice = returns_panel.iloc[end - window : end]
        result = _pca_returns(window_slice, n_components=1, method="power_iteration")
        out.iloc[end - 1] = result["loadings"]["PC1"]
    return out.ffill()


def _pca_factor_return(
    returns_panel: pd.DataFrame,
    context: FeatureContext,
    window: int = 252,
    refit_every: int = 21,
) -> pd.DataFrame:
    _validate_window_params(window, refit_every, "factors.pca_factor_return")
    n = len(returns_panel)
    values = pd.Series(np.nan, index=returns_panel.index)
    current_loadings = None
    for i in range(n):
        if i + 1 >= window and (i + 1 - window) % refit_every == 0:
            window_slice = returns_panel.iloc[i + 1 - window : i + 1]
            result = _pca_returns(window_slice, n_components=1, method="power_iteration")
            current_loadings = result["loadings"]["PC1"]
        if current_loadings is not None:
            values.iloc[i] = float(returns_panel.iloc[i].to_numpy() @ current_loadings.to_numpy())
    return pd.DataFrame({col: values for col in returns_panel.columns}, index=returns_panel.index)


register_feature(
    FeatureDefinition(
        id="factors.pca_loading",
        description="Entity's loading on PC1 of the universe return panel, "
        "refit every `refit_every` bars and forward-filled between refits.",
        fn=_pca_loading,
        default_params={"window": 252, "refit_every": 21},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.UNIVERSE,
        requires=["Close"],
        lookback=252,
    )
)
register_feature(
    FeatureDefinition(
        id="factors.pca_factor_return",
        description="That date's realized universe return projected onto the "
        "currently-held PC1 loadings — a shared macro factor (same value for "
        "every entity that date).",
        fn=_pca_factor_return,
        default_params={"window": 252, "refit_every": 21},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.UNIVERSE,
        requires=["Close"],
        lookback=252,
    )
)
