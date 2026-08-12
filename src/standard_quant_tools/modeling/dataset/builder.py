"""
build_dataset: DatasetSpec -> long feature(/target) panel plus lineage
metadata (the FeatureFrame this phase actually returns — a dict, not a
new class, since nothing yet needs FeatureFrame to be more than "panel +
a few identifiers").

Fetches each universe symbol's OHLCV once — concurrently, via
dataset/fetch.py, from the provider and at the interval the DatasetSpec
names (not fetch_returns_sync, which returns Close-only returns and
features here need full OHLC) — computes every requested feature
(entity-scope per symbol, universe-scope once over the shared return
panel — see features/base.py's docstring), optionally builds the
forward-return target, runs the point-in-time safety check, collects
coverage/provenance warnings (dataset/coverage.py), and returns the panel
with a content hash for the audit trail.

`include_target=False` is scoring.py's path: score_model wants a
prediction as of a date where the forward-return target hasn't happened
yet (it needs `horizon` bars of future data past "today"), so it must
skip target construction entirely rather than have those most-recent
rows silently dropped by a target-based dropna.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from standard_quant_tools.audit.hashing import hash_dataframe
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import require_finite_array

from ..features.base import FeatureContext, FeatureScope
from ..features.params import resolve_params
from ..features.registry import get_feature
from ..specs import DatasetSpec
from .alignment import build_returns_panel, stack_features_only, stack_long
from .coverage import (
    alignment_warnings,
    entity_coverage_warnings,
    intersection_warnings,
    interval_warnings,
    provider_guarantee_warnings,
)
from .fetch import fetch_universe_ohlcv
from .leakage import check_point_in_time_safety
from .target import build_label_end_dates, build_target

logger = logging.getLogger(__name__)


def _check_required_columns(
    ohlcv: pd.DataFrame, symbol: str, feature_defs: list, feature_names: list
) -> None:
    """
    Enforce each feature's declared `requires` against the fetched frame.

    FeatureDefinition.requires was purely informational: a provider (or a
    custom one) returning a frame without 'Volume' produced a raw KeyError
    from inside whichever feature happened to touch it first, naming the
    column but not the feature, the symbol, or the fact that the provider
    was the problem.
    """
    available = set(ohlcv.columns)
    problems = []
    for definition, name in zip(feature_defs, feature_names):
        missing = [c for c in definition.requires if c not in available]
        if missing:
            problems.append(f"{name!r} requires {missing}")
    if problems:
        raise ValidationError(
            f"build_model_dataset: OHLCV for {symbol!r} is missing column(s) needed by "
            f"the requested features — {'; '.join(problems)}. "
            f"Provider returned columns: {sorted(available)}."
        )


def _fetch_ohlcv(
    provider: Any, symbol: str, start: str, end: str, interval: str = "1d"
) -> pd.DataFrame:
    """Single-symbol fetch (the benchmark). A raw provider exception
    (network error, unknown ticker, rate limit, unsupported interval, ...)
    would otherwise propagate with no indication of WHICH symbol in a
    multi-symbol universe caused it -- wrap it in a ValidationError that
    names the symbol, keeping the original exception as the cause for
    anyone inspecting the traceback.

    The universe itself goes through dataset/fetch.py, which fetches
    concurrently and reports every failing symbol at once."""
    try:
        df = provider.get_ohlcv(symbol, start, end, interval)
    except Exception as exc:
        raise ValidationError(
            f"build_model_dataset: failed to fetch OHLCV for {symbol!r} ({start} to "
            f"{end}, interval={interval!r}): {exc}"
        ) from exc
    if df.empty:
        raise ValidationError(
            f"build_model_dataset: no OHLCV data returned for {symbol!r}"
        )
    return df


def _provider_metadata(provider: Any, symbol: str, interval: str) -> Optional[Any]:
    """The provider's own honest self-report about what it does and does not
    guarantee. Best-effort: a custom or mocked provider need not implement
    get_metadata, and failing the whole dataset build because a provenance
    NOTE could not be read would be the wrong trade. Fetched once for a
    representative symbol -- provider/adjusted/survivorship/point-in-time are
    provider-level properties, and only `timezone` is symbol-dependent."""
    getter = getattr(provider, "get_metadata", None)
    if getter is None:
        return None
    try:
        return getter(symbol, interval)
    except Exception as exc:  # noqa: BLE001 — provenance is not worth failing over
        logger.debug("[modeling] provider metadata unavailable: %s", exc)
        return None


def dataset_spec_hash(spec: DatasetSpec) -> str:
    """
    Canonical content hash of a DatasetSpec.

    Extracted so the build side and the verify side (agent/tools.py's
    run_model_experiment, which re-derives this to detect an edited
    dataset_spec.json) cannot drift apart — two independent inlined
    expressions of "the spec hash" is precisely how an integrity check
    quietly stops checking anything.

    model_dump_json() is canonical enough here because pydantic serializes
    a model's fields in declaration order, so the same spec always produces
    the same JSON regardless of the order kwargs were passed in.
    """
    return hashlib.sha256(spec.model_dump_json().encode()).hexdigest()


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

    # Resolved ONCE, up front, before any data is fetched: a bad parameter
    # should fail immediately rather than after a slow multi-symbol
    # download. This is also the point-in-time gate for parameter VALUES --
    # check_point_in_time_safety above only inspects each feature's static
    # TemporalSupport label, which stays PIT_SAFE even when a negative
    # lookback turns the formula into a forward-looking one.
    resolved_params = [
        resolve_params(definition, fs.params)
        for fs, definition in zip(spec.features, feature_defs)
    ]

    # Provider and interval come from the spec rather than DataFactory's
    # defaults: the runtime previously could only ever build a dataset from
    # yfinance daily bars, and -- worse -- the resulting model recorded
    # neither, so its lineage could not say what it had been trained on.
    # Both are part of DatasetSpec, so both are hashed into spec_hash,
    # bundled into the model, and reused verbatim by scoring.
    provider = DataFactory.get_provider(spec.provider)
    ohlcv_by_entity = fetch_universe_ohlcv(
        provider, list(spec.universe), spec.start, spec.end, spec.interval
    )

    benchmark_df = _fetch_ohlcv(
        provider, spec.benchmark, spec.start, spec.end, spec.interval
    )
    context = FeatureContext(benchmark_close=benchmark_df["Close"])

    close_by_entity = {symbol: df["Close"] for symbol, df in ohlcv_by_entity.items()}
    returns_panel = build_returns_panel(close_by_entity)

    # ── Coverage / provenance diagnostics ─────────────────────────────────
    # Collected here, once the data is in hand but before any of it is
    # consumed, and returned to the caller rather than logged: these change
    # how the OOS metrics should be read, and a log line is not part of the
    # tool result an agent sees.
    has_universe_scope = any(
        definition.scope == FeatureScope.UNIVERSE for definition in feature_defs
    )
    warnings: List[str] = []
    warnings.extend(interval_warnings(spec.interval))
    warnings.extend(
        provider_guarantee_warnings(
            _provider_metadata(provider, spec.universe[0], spec.interval)
        )
    )
    warnings.extend(entity_coverage_warnings(ohlcv_by_entity, spec.start, spec.end))
    warnings.extend(
        intersection_warnings(ohlcv_by_entity, returns_panel, has_universe_scope)
    )

    # Universe-scope features are computed once, over the shared panel —
    # not once per entity, since PCA needs every entity's returns at once.
    # Keyed on output_name, not id: the same feature can now appear more
    # than once (at different parameters) under different aliases, so the
    # id is no longer a unique key for its computed output.
    universe_outputs: Dict[str, pd.DataFrame] = {}
    for fs, definition, params in zip(spec.features, feature_defs, resolved_params):
        if definition.scope == FeatureScope.UNIVERSE:
            universe_outputs[fs.output_name] = definition.fn(
                returns_panel, context, **params
            )

    per_entity_features: Dict[str, pd.DataFrame] = {}
    target_by_entity: Dict[str, pd.Series] = {}
    label_end_by_entity: Dict[str, pd.Series] = {}
    feature_names = [fs.output_name for fs in spec.features]
    for symbol, ohlcv in ohlcv_by_entity.items():
        _check_required_columns(ohlcv, symbol, feature_defs, feature_names)
        columns: Dict[str, pd.Series] = {}
        for fs, definition, params in zip(spec.features, feature_defs, resolved_params):
            if definition.scope == FeatureScope.ENTITY:
                columns[fs.output_name] = definition.fn(ohlcv, context, **params)
            else:
                columns[fs.output_name] = universe_outputs[fs.output_name][symbol]
        per_entity_features[symbol] = pd.DataFrame(columns)
        if include_target:
            target_by_entity[symbol] = build_target(ohlcv["Close"], spec.target)
            # Recorded per row, per entity, from that entity's OWN bar
            # index -- see build_label_end_dates for why an integer offset
            # against the global date axis is not equivalent.
            label_end_by_entity[symbol] = build_label_end_dates(
                ohlcv["Close"], spec.target
            )

    if include_target:
        long_panel, drop_attribution = stack_long(
            per_entity_features, target_by_entity, label_end_by_entity
        )
    else:
        long_panel, drop_attribution = stack_features_only(per_entity_features)

    if long_panel.empty:
        # The attribution turns a dead end into a diagnosis: "no rows
        # survive" previously left the caller to guess which of their
        # features was too long for the window they asked for.
        raise ValidationError(
            "build_model_dataset: no rows survive feature"
            + ("/target" if include_target else "")
            + " alignment — check that start/end covers enough history for the "
            "requested features' lookback windows. Rows missing per column, out "
            f"of {drop_attribution['rows_before_alignment']}: "
            + ", ".join(
                f"{name}={counts['n_missing']}"
                for name, counts in sorted(
                    drop_attribution["per_feature"].items(),
                    key=lambda kv: -kv[1]["n_missing"],
                )
            )
        )

    entities_fetched = sorted(ohlcv_by_entity.keys())
    # Entities that actually reached the panel, not the ones fetched. The
    # two differ whenever a symbol's history is shorter than the feature
    # lookbacks plus the target horizon, and reporting the fetched list made
    # a dataset look like it covered a universe the model never saw.
    entities_surviving = sorted(long_panel["entity"].unique().tolist())
    warnings.extend(
        alignment_warnings(drop_attribution, entities_fetched, entities_surviving)
    )

    # dropna() (inside stack_long/stack_features_only) removes NaN but
    # not +/-inf -- a degenerate feature computation (e.g. division by a
    # near-zero denominator somewhere upstream) could otherwise feed inf
    # straight into sklearn, which either raises a cryptic error or
    # silently produces garbage. Same enforcement point this codebase
    # already uses pervasively elsewhere (indicators, analysis, backtest).
    numeric_cols = [fs.output_name for fs in spec.features] + (
        ["target"] if include_target else []
    )
    for col in numeric_cols:
        require_finite_array(
            long_panel[col].to_numpy(dtype=float), col, "build_model_dataset"
        )

    # audit.hash_dataframe, not a local pd.util.hash_pandas_object call.
    # hash_pandas_object is a per-ROW digest that never sees column labels,
    # so two panels with identical numbers under entirely different feature
    # columns hash the same. The audit package was explicitly fixed for
    # exactly this collision; duplicating the pre-fix version here
    # reintroduced it in the modeling lineage.
    data_hash = hash_dataframe(long_panel)

    result: Dict[str, Any] = {
        "panel": long_panel,
        # Entities in the PANEL. This used to be the fetched list, which
        # silently overstated coverage for any symbol whose history did not
        # survive alignment.
        "entities": entities_surviving,
        "entities_fetched": entities_fetched,
        # BuildModelDatasetResult.warnings has existed since the first
        # version of the tool surface and was never populated by anything.
        "warnings": warnings,
        # Per-column row loss, so "3,000 rows" can be read as the warm-up
        # that was asked for or as one feature eating the panel.
        "drop_attribution": drop_attribution,
        # Panel column names (alias where given, id otherwise) -- these are
        # what the model is actually trained on and what the manifest must
        # record, not the underlying registry ids.
        "feature_ids": [fs.output_name for fs in spec.features],
        "data_hash": data_hash,
        "spec_hash": dataset_spec_hash(spec),
    }
    result["target_id"] = (
        f"{spec.target.type}:{spec.target.horizon}" if include_target else None
    )
    return result
