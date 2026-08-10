"""
build_dataset: DatasetSpec -> long feature(/target) panel plus lineage
metadata (the FeatureFrame this phase actually returns — a dict, not a
new class, since nothing yet needs FeatureFrame to be more than "panel +
a few identifiers").

Fetches each universe symbol's OHLCV once (provider dict-comprehension,
mirrors run_pca_analysis in agent/tools.py — not fetch_returns_sync,
which returns Close-only returns and features here need full OHLC),
computes every requested feature (entity-scope per symbol, universe-scope
once over the shared return panel — see features/base.py's docstring),
optionally builds the forward-return target, runs the point-in-time
safety check, and returns the panel with a content hash for the audit
trail.

`include_target=False` is scoring.py's path: score_model wants a
prediction as of a date where the forward-return target hasn't happened
yet (it needs `horizon` bars of future data past "today"), so it must
skip target construction entirely rather than have those most-recent
rows silently dropped by a target-based dropna.
"""

import hashlib
from typing import Any, Dict

import pandas as pd

from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import require_finite_array

from ..features.base import FeatureContext, FeatureScope
from ..features.registry import get_feature
from ..specs import DatasetSpec
from .alignment import build_returns_panel, stack_features_only, stack_long
from .leakage import check_point_in_time_safety
from .target import build_target


def _fetch_ohlcv(provider: Any, symbol: str, start: str, end: str) -> pd.DataFrame:
    """A raw provider exception (network error, unknown ticker, rate
    limit, ...) would otherwise propagate with no indication of WHICH
    symbol in a multi-symbol universe caused it -- wrap it in a
    ValidationError that names the symbol, keeping the original exception
    as the cause for anyone inspecting the traceback."""
    try:
        df = provider.get_ohlcv(symbol, start, end)
    except Exception as exc:
        raise ValidationError(
            f"build_model_dataset: failed to fetch OHLCV for {symbol!r} ({start} to "
            f"{end}): {exc}"
        ) from exc
    if df.empty:
        raise ValidationError(f"build_model_dataset: no OHLCV data returned for {symbol!r}")
    return df


def build_dataset(spec: DatasetSpec, include_target: bool = True) -> Dict[str, Any]:
    """
    Raises:
        ValidationError: unknown feature id, a universe/benchmark symbol
        that failed to fetch or returned empty data, a feature that
        produced a non-finite (inf) value, or no rows survive
        feature(/target) alignment.
        PointInTimeViolation: a requested feature is CURRENT_ONLY.
    """
    feature_defs = [get_feature(fs.id) for fs in spec.features]
    check_point_in_time_safety(feature_defs)

    provider = DataFactory.get_provider()
    ohlcv_by_entity = {
        symbol: _fetch_ohlcv(provider, symbol, spec.start, spec.end) for symbol in spec.universe
    }

    benchmark_df = _fetch_ohlcv(provider, spec.benchmark, spec.start, spec.end)
    context = FeatureContext(benchmark_close=benchmark_df["Close"])

    close_by_entity = {symbol: df["Close"] for symbol, df in ohlcv_by_entity.items()}
    returns_panel = build_returns_panel(close_by_entity)

    # Universe-scope features are computed once, over the shared panel —
    # not once per entity, since PCA needs every entity's returns at once.
    universe_outputs: Dict[str, pd.DataFrame] = {}
    for fs, definition in zip(spec.features, feature_defs):
        if definition.scope == FeatureScope.UNIVERSE:
            params = {**definition.default_params, **fs.params}
            universe_outputs[fs.id] = definition.fn(returns_panel, context, **params)

    per_entity_features: Dict[str, pd.DataFrame] = {}
    target_by_entity: Dict[str, pd.Series] = {}
    for symbol, ohlcv in ohlcv_by_entity.items():
        columns: Dict[str, pd.Series] = {}
        for fs, definition in zip(spec.features, feature_defs):
            if definition.scope == FeatureScope.ENTITY:
                params = {**definition.default_params, **fs.params}
                columns[fs.id] = definition.fn(ohlcv, context, **params)
            else:
                columns[fs.id] = universe_outputs[fs.id][symbol]
        per_entity_features[symbol] = pd.DataFrame(columns)
        if include_target:
            target_by_entity[symbol] = build_target(ohlcv["Close"], spec.target)

    if include_target:
        long_panel = stack_long(per_entity_features, target_by_entity)
    else:
        long_panel = stack_features_only(per_entity_features)

    if long_panel.empty:
        raise ValidationError(
            "build_model_dataset: no rows survive feature"
            + ("/target" if include_target else "")
            + " alignment — check that start/end covers enough history for the "
            "requested features' lookback windows."
        )

    # dropna() (inside stack_long/stack_features_only) removes NaN but
    # not +/-inf -- a degenerate feature computation (e.g. division by a
    # near-zero denominator somewhere upstream) could otherwise feed inf
    # straight into sklearn, which either raises a cryptic error or
    # silently produces garbage. Same enforcement point this codebase
    # already uses pervasively elsewhere (indicators, analysis, backtest).
    numeric_cols = [fs.id for fs in spec.features] + (["target"] if include_target else [])
    for col in numeric_cols:
        require_finite_array(long_panel[col].to_numpy(dtype=float), col, "build_model_dataset")

    data_hash = hashlib.sha256(
        pd.util.hash_pandas_object(long_panel, index=True).to_numpy().tobytes()
    ).hexdigest()

    result: Dict[str, Any] = {
        "panel": long_panel,
        "entities": sorted(ohlcv_by_entity.keys()),
        "feature_ids": [fs.id for fs in spec.features],
        "data_hash": data_hash,
        "spec_hash": hashlib.sha256(spec.model_dump_json().encode()).hexdigest(),
    }
    result["target_id"] = f"{spec.target.type}:{spec.target.horizon}" if include_target else None
    return result
