"""
Universe-scope correlation-NETWORK features: where an entity sits in the
graph its universe forms, rather than how much of the dominant factor it
carries.

WHAT IS DELIBERATELY NOT HERE. Eigenvector centrality of the correlation
matrix is, up to normalisation, the leading eigenvector of that matrix --
which is exactly what `factors.pca_loading` already computes and has
computed since this package had features at all. Registering it again under
a graph-theoretic name would be the same number twice with two
explanations, so the two features below are chosen because they are NOT
recoverable from PC1.

  network.avg_correlation -- an entity's mean correlation to the rest of
  the universe. PC1 loading is signed and variance-weighted, so a
  high-volatility name loads heavily whether or not it moves WITH anything;
  mean correlation is scale-free and answers the different question of how
  much company a name keeps. In a one-factor universe the two agree, and
  the gap between them is informative precisely when it is not one-factor.

  network.mst_degree -- the entity's degree in the minimum spanning tree of
  the correlation-distance matrix (Mantegna's construction). This is LOCAL
  topology: a hub is a name that other names route through, which is a
  statement about the graph's shape and not about any global factor. A
  universe can have a flat PC1 and a highly centralised tree, and a
  universe's tree can reorganise while PC1 barely moves.

THE DISTANCE IS NOT THE CORRELATION. Edges are weighted by
d = sqrt(2(1 - rho)), the standard metric for this: it is a true distance
(zero only when rho is 1, and satisfying the triangle inequality), which a
raw correlation or its negation is not. A spanning tree built on a
non-metric weight is a tree over nothing in particular.

REFIT COST. Both follow the rolling-refit design `factors.py` established
for the same reason -- a correlation structure does not move meaningfully
day to day, and recomputing an N-by-N correlation plus a spanning tree on
every bar is work spent on a value that barely changes. They are refit
every `refit_every` bars over a trailing `window` and held fixed between,
so the reported value is always the one most recently ESTIMATED, never
interpolated toward a future one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError

from .base import FeatureContext, FeatureDefinition, FeatureScope, TemporalSupport
from .registry import register_feature

#: Below this a correlation matrix over the window is too noisy to build a
#: tree from -- the estimate's own error exceeds the differences between
#: edges, so the tree is a sample of noise rather than a structure.
MIN_WINDOW = 20

#: A tree needs at least two nodes to have an edge at all.
MIN_ENTITIES = 2


def _validate(window: int, refit_every: int, feature_id: str) -> None:
    if window < MIN_WINDOW:
        raise ValidationError(
            f"{feature_id}: window must be >= {MIN_WINDOW}, got {window}. A "
            "correlation matrix estimated on fewer observations has an error "
            "larger than the differences between its entries, so the network "
            "built from it describes sampling noise."
        )
    if refit_every < 1:
        raise ValidationError(
            f"{feature_id}: refit_every must be >= 1, got {refit_every}"
        )


#: Above this ratio of |column mean| to column standard deviation, centring
#: a column throws away digits the input never carried -- `1e8 + x` stores
#: `x` to about 1e-8 absolute, so no algorithm recovers it. The fast path
#: below is exact well inside this and disagreed with pandas at 1e-8 beyond
#: it, so it defers rather than differ. A returns panel, which is what this
#: module is given, sits at a ratio near zero.
_MAX_MEAN_TO_STD = 1e3


def _pairwise_correlation(frame: pd.DataFrame, min_periods: int):
    """
    `DataFrame.corr(min_periods=...)` as four matrix products.

    `min_periods` forces pandas down its pairwise `nancorr` path, which
    walks column pairs in a Python-level loop: 188 ms for 1,000 entities
    over a 126-bar window, against 62 ms here for the same numbers and the
    same NaN pattern.

    Each column is centred on its own mean before the products. Correlation
    is invariant to a per-column shift, and without it the identity
    `E[x^2] - E[x]^2` cancels: on a panel of price levels around 1e8 the
    variance came out NEGATIVE and the result was `inf`.

    Returns None when the panel is too ill-conditioned for that centring to
    be exact, so the caller falls back to pandas rather than quietly
    returning a different number.
    """
    values = frame.to_numpy(dtype=np.float64)
    present = np.isfinite(values)
    counts = present.sum(axis=0)
    if not counts.any():
        return None

    filled = np.where(present, values, 0.0)
    column_mean = filled.sum(axis=0) / np.where(counts > 0, counts, 1)
    spread = np.sqrt(
        np.maximum(
            (np.where(present, values * values, 0.0)).sum(axis=0)
            / np.where(counts > 0, counts, 1)
            - column_mean**2,
            0.0,
        )
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        conditioning = np.abs(column_mean) / np.where(spread > 0, spread, np.inf)
    if np.nanmax(conditioning, initial=0.0) > _MAX_MEAN_TO_STD:
        return None

    centred = np.where(present, values - column_mean, 0.0)
    mask = present.astype(np.float64)
    n_pairs = mask.T @ mask
    sum_x = centred.T @ mask
    sum_y = sum_x.T
    sum_xy = centred.T @ centred
    sum_xx = (centred * centred).T @ mask
    sum_yy = sum_xx.T
    with np.errstate(invalid="ignore", divide="ignore"):
        covariance = sum_xy / n_pairs - (sum_x / n_pairs) * (sum_y / n_pairs)
        var_x = np.maximum(sum_xx / n_pairs - (sum_x / n_pairs) ** 2, 0.0)
        var_y = np.maximum(sum_yy / n_pairs - (sum_y / n_pairs) ** 2, 0.0)
        out = covariance / np.sqrt(var_x * var_y)
    out[n_pairs < min_periods] = np.nan
    return pd.DataFrame(out, index=frame.columns, columns=frame.columns)


def _correlation(window_slice: pd.DataFrame):
    """Correlation over one window, or None if it is not estimable."""
    usable = window_slice.dropna(axis=1, how="all")
    if usable.shape[1] < MIN_ENTITIES:
        return None
    matrix = _pairwise_correlation(usable, MIN_WINDOW)
    if matrix is None:
        matrix = usable.corr(min_periods=MIN_WINDOW)
    if matrix.isna().all(axis=None):
        return None
    return matrix


def _avg_correlation_at(window_slice: pd.DataFrame):
    """Each entity's mean correlation to the others in this window."""
    matrix = _correlation(window_slice)
    if matrix is None:
        return None
    values = matrix.to_numpy(dtype="float64").copy()
    # The diagonal is 1 by construction and would drag every entity's mean
    # toward its own self-correlation, which says nothing about anything.
    np.fill_diagonal(values, np.nan)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(values, axis=1)
    return pd.Series(means, index=matrix.columns)


def _prim_degrees(distance: np.ndarray) -> np.ndarray:
    """
    Spanning-tree degree per node, by Prim over a DENSE distance matrix.

    WHY NOT `scipy.sparse.csgraph.minimum_spanning_tree`. It reads a stored
    zero as "no edge", and a perfectly correlated pair has Mantegna distance
    sqrt(2(1-1)) = 0 -- so the single edge the construction most wants is
    the one silently discarded. Two entities that move identically returned
    degrees {0, 0}, and a two-node spanning tree has exactly one edge, so
    the only possible answer was {1, 1}. In a larger universe the edge count
    stayed right and the tree rerouted around the duplicate, corrupting
    precisely the topology this feature exists to measure. Reachable through
    a dual listing, a symbol repeated in a universe, or stale repeated
    prices.

    Prim carries no sparsity convention, so zero is an ordinary weight. It
    is also faster here: the matrix is dense by construction, and building
    a CSR graph to run Kruskal over it is work with nothing to show for it.
    """
    n = distance.shape[0]
    degrees = np.zeros(n, dtype="float64")
    if n < 2:
        return degrees
    in_tree = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    best[0] = 0.0
    for _ in range(n):
        candidate = np.where(in_tree, np.inf, best)
        node = int(np.argmin(candidate))
        if not np.isfinite(candidate[node]):
            break  # a disconnected component; the rest have no edge to add
        in_tree[node] = True
        if parent[node] >= 0:
            degrees[node] += 1.0
            degrees[parent[node]] += 1.0
        weights = distance[node]
        closer = (~in_tree) & (weights < best)
        best[closer] = weights[closer]
        parent[closer] = node
    return degrees


def _mst_degree_at(window_slice: pd.DataFrame):
    """Each entity's degree in the correlation-distance spanning tree."""
    matrix = _correlation(window_slice)
    if matrix is None:
        return None
    rho = matrix.to_numpy(dtype="float64")
    # An unestimable pair gets the maximum distance rather than zero: zero
    # is the CLOSEST possible pair, so using it for "never measured" would
    # pull the tree straight through a pair nobody observed.
    rho = np.where(np.isfinite(rho), rho, -1.0)
    distance = np.sqrt(np.clip(2.0 * (1.0 - rho), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    return pd.Series(_prim_degrees(distance), index=matrix.columns)


def _rolling_network(
    returns_panel: pd.DataFrame,
    window: int,
    refit_every: int,
    at_window,
    feature_id: str,
) -> pd.DataFrame:
    """Refit `at_window` every `refit_every` bars, hold it between refits."""
    _validate(window, refit_every, feature_id)
    n = len(returns_panel)
    out = pd.DataFrame(np.nan, index=returns_panel.index, columns=returns_panel.columns)
    for end in range(window, n + 1, refit_every):
        values = at_window(returns_panel.iloc[end - window : end])
        if values is None:
            continue
        # Assigned at bar end-1, computed from bars end-window..end-1
        # inclusive -- only data available at that bar, matching the
        # convention factors.py uses.
        out.iloc[end - 1] = values.reindex(out.columns)
    return out.ffill()


def _avg_correlation(
    returns_panel: pd.DataFrame,
    context: FeatureContext,
    window: int = 126,
    refit_every: int = 21,
) -> pd.DataFrame:
    return _rolling_network(
        returns_panel,
        window,
        refit_every,
        _avg_correlation_at,
        "network.avg_correlation",
    )


def _mst_degree(
    returns_panel: pd.DataFrame,
    context: FeatureContext,
    window: int = 126,
    refit_every: int = 21,
) -> pd.DataFrame:
    return _rolling_network(
        returns_panel, window, refit_every, _mst_degree_at, "network.mst_degree"
    )


register_feature(
    FeatureDefinition(
        id="network.avg_correlation",
        description="Entity's mean correlation to the rest of the universe "
        "over a trailing window, refit every `refit_every` bars. Scale-free, "
        "unlike a PC1 loading: it says how much company a name keeps rather "
        "than how much variance it contributes.",
        fn=_avg_correlation,
        default_params={"window": 126, "refit_every": 21},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.UNIVERSE,
        requires=["Close"],
        lookback=126,
    )
)
register_feature(
    FeatureDefinition(
        id="network.mst_degree",
        description="Entity's degree in the minimum spanning tree of the "
        "universe's correlation-distance matrix (Mantegna). Local topology, "
        "not a global factor: a hub is a name others route through, which a "
        "PC1 loading cannot express.",
        fn=_mst_degree,
        default_params={"window": 126, "refit_every": 21},
        temporal_support=TemporalSupport.PIT_SAFE,
        scope=FeatureScope.UNIVERSE,
        requires=["Close"],
        lookback=126,
    )
)

__all__ = ["MIN_ENTITIES", "MIN_WINDOW"]
