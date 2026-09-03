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
from standard_quant_tools.data.bundle import DataBundle, validate_bundle
from standard_quant_tools.data.factory import DataFactory
from standard_quant_tools.error import ValidationError
from standard_quant_tools.validation import (
    memoized_input_checks,
    require_finite_array,
)

from ..features.base import FeatureContext, FeatureScope
from ..features.params import resolve_params
from ..features.registry import get_feature
from ..specs import DatasetSpec
from .alignment import build_returns_panel, stack_features_only, stack_long
from .lags import expand_lags, expanded_feature_ids, lags_by_output_name
from .coverage import (
    alignment_warnings,
    entity_coverage_warnings,
    intersection_warnings,
    interval_warnings,
    provider_guarantee_warnings,
)
from .fetch import fetch_universe_ohlcv
from .leakage import check_point_in_time_safety
from .panel_features import compute_panel_features
from .target import (
    CROSS_SECTIONAL_TARGETS,
    apply_cross_sectional_target,
    build_label_end_dates,
    build_target,
)

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


def _provider_contract(provider: Any, frame_kind: str = "bars") -> Optional[Any]:
    """
    The provider's own declared temporal contract, or None.

    Best-effort for the same reason as `_provider_metadata`: a duck-typed
    or partially-stubbed provider need not implement
    `get_temporal_contract`, and failing a dataset build because a
    PROVENANCE note could not be read would be the wrong trade. Falling
    back to None hands `DataBundle` the inference path, which is weaker but
    honest about being weaker.
    """
    from standard_quant_tools.data.temporal import TemporalContract

    getter = getattr(provider, "get_temporal_contract", None)
    if getter is None:
        return None
    try:
        contract = getter(frame_kind)
    except Exception as exc:  # noqa: BLE001 -- provenance is not worth failing over
        logger.debug("[modeling] temporal contract unavailable: %s", exc)
        return None
    # A duck-typed or mocked provider hands back something that is not a
    # contract at all, and a MagicMock in particular answers every
    # attribute truthily -- which would report a bundle as point-in-time
    # safe because nothing said otherwise. Checking the type is what keeps
    # the verdict from being vacuously clean.
    if not isinstance(contract, TemporalContract):
        logger.debug(
            "[modeling] %r returned a non-contract from get_temporal_contract",
            type(provider).__name__,
        )
        return None
    return contract


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


def _check_entity_output(
    result: Any, column: str, feature_id: str, ohlcv: pd.DataFrame, symbol: str
) -> pd.Series:
    """
    Enforce the documented ENTITY feature contract: a pd.Series aligned to
    the entity's own OHLCV index.

    The contract was documented for custom features but never checked --
    the result went straight into a DataFrame constructor, so a callable
    returning the wrong type, length or index failed later as a generic
    pandas error with no mention of which feature caused it. Making an
    already-documented interface actually enforced, and naming the feature
    when it isn't met.
    """
    if not isinstance(result, pd.Series):
        raise ValidationError(
            f"feature {feature_id!r} (column {column!r}) must return a pandas Series "
            f"for an ENTITY-scope feature, got {type(result).__name__} for {symbol!r}."
        )
    # The contract is that the index is a SUBSET of the entity's bars --
    # not that it is identical. A feature legitimately produces fewer rows
    # than it consumes: risk.rolling_beta works from returns, which lose
    # the first bar to pct_change, so it returns n-1 values for n bars.
    # Panel assembly is index-aligned, so a subset lands on the right bars
    # and the absent ones become NaN, which the alignment step then handles.
    #
    # What is NOT safe is an index carrying labels the entity does not
    # have: those either silently introduce rows or, worse, indicate the
    # feature computed against the wrong entity entirely.
    extra = result.index.difference(ohlcv.index)
    if len(extra) > 0:
        raise ValidationError(
            f"feature {feature_id!r} (column {column!r}) returned {len(extra)} "
            f"index label(s) that are not in {symbol!r}'s OHLCV index, e.g. "
            f"{[str(x) for x in extra[:3]]}. A feature's output must be indexed by "
            "the bars it was given — anything else would either introduce rows the "
            "entity does not have or indicate it was computed against different data."
        )
    return result


def _check_universe_output(
    result: Any, column: str, feature_id: str, returns_panel: pd.DataFrame
) -> pd.DataFrame:
    """
    Enforce the documented UNIVERSE feature contract: a pd.DataFrame with
    one column per entity, indexed like the shared returns panel. Same
    rationale as _check_entity_output -- a missing column here surfaced as
    a bare KeyError when the per-entity loop indexed into it.
    """
    if not isinstance(result, pd.DataFrame):
        raise ValidationError(
            f"feature {feature_id!r} (column {column!r}) must return a pandas DataFrame "
            f"for a UNIVERSE-scope feature (one column per entity), got "
            f"{type(result).__name__}."
        )
    missing = [c for c in returns_panel.columns if c not in result.columns]
    if missing:
        raise ValidationError(
            f"feature {feature_id!r} (column {column!r}) returned no values for "
            f"entities {missing}. A UNIVERSE feature must produce a column for every "
            "entity in the panel."
        )
    if not result.index.equals(returns_panel.index):
        raise ValidationError(
            f"feature {feature_id!r} (column {column!r}) returned a DataFrame whose "
            "index does not match the shared returns panel's."
        )
    return result


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
    # interval carried into the context so a feature that ANNUALIZES scales
    # by the right constant instead of assuming daily bars -- see
    # features/risk.py::_annualization.
    context = FeatureContext(
        benchmark_close=benchmark_df["Close"], interval=spec.interval
    )

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

    # ── What this data can and cannot support, as one verdict ─────────────
    # The panel is collected into a DataBundle and validated as a UNIT. A
    # frame is only half the fact; the other half is what its source can
    # say about WHEN each row became knowable, and the pairing is the thing
    # worth carrying forward. `join_point_in_time` later attaches records
    # to this panel, and the honest answer to "was that join safe" depends
    # on the contract the bars arrived under, not on the join alone.
    #
    # `require_pit=False` is deliberate and is not a lowered bar. No shipped
    # provider reports `point_in_time=True`, so requiring it would refuse
    # every build this runtime has ever done. The verdict is recorded and
    # surfaced as warnings instead, which is the difference between a
    # caller who knows what they are joining onto and one who finds out
    # from a model that looks prescient.
    # ASK THE PROVIDER for its contract rather than inferring one from the
    # frame. Inference reads COLUMNS, which can only say what is present --
    # never what the source guarantees -- so a provider that genuinely
    # serves point-in-time data would still have been reported as though it
    # did not, and the caveat would fire on every build regardless of the
    # truth. A warning that is always on carries no information, which is
    # the property `tests/modeling/test_data_runtime_architecture.py` pins.
    bundle = DataBundle(f"dataset:{dataset_spec_hash(spec)[:12]}")
    bundle.add(
        "bars",
        returns_panel,
        _provider_contract(provider, "bars"),
        source=spec.provider,
        entity_scoped=True,
    )
    # The verdict is PROVENANCE, not a warning, and the distinction is the
    # point. `provider_guarantee_warnings` above already says whether the
    # provider guarantees point-in-time data, which is the informative
    # version of this fact; re-deriving it from a returns panel would add a
    # caveat that fires on EVERY build, because a wide frame of returns
    # structurally has no `available_time` column and never will. A warning
    # that is always on carries no information -- the property
    # `test_a_clean_provider_and_universe_produce_no_warnings` pins.
    #
    # So the bundle is recorded rather than shouted. It travels with the
    # dataset, and `join_point_in_time` later attaches records to exactly
    # this panel, where "was that join safe" depends on the contract these
    # bars arrived under.
    bundle_verdict = validate_bundle(bundle, require_pit=False)

    # Universe-scope features are computed once, over the shared panel —
    # not once per entity, since PCA needs every entity's returns at once.
    # Keyed on output_name, not id: the same feature can now appear more
    # than once (at different parameters) under different aliases, so the
    # id is no longer a unique key for its computed output.
    universe_outputs: Dict[str, pd.DataFrame] = {}
    for fs, definition, params in zip(spec.features, feature_defs, resolved_params):
        if definition.scope == FeatureScope.UNIVERSE:
            universe_outputs[fs.output_name] = _check_universe_output(
                definition.fn(returns_panel, context, **params),
                fs.output_name,
                definition.id,
                returns_panel,
            )

    per_entity_features: Dict[str, pd.DataFrame] = {}
    target_by_entity: Dict[str, pd.Series] = {}
    label_end_by_entity: Dict[str, pd.Series] = {}
    # {horizon name: {entity: Series}} -- every requested horizon, off the
    # same features. Empty for a single-horizon spec, which is every spec
    # written before `horizons` existed.
    extra_targets: Dict[str, Dict[str, pd.Series]] = {}
    extra_label_ends: Dict[str, Dict[str, pd.Series]] = {}
    feature_names = [fs.output_name for fs in spec.features]
    # {column: its lags} for the per-entity expansion below, and the
    # full expanded column list, generated in ONE place so the panel's
    # column order, X's column order and the importance vector's order
    # cannot drift apart.
    lags_requested = lags_by_output_name(spec.features)
    expanded_names = expanded_feature_ids(spec.features)
    # Every feature function re-validates the OHLCV columns it is handed,
    # which is correct at a public boundary and pure repeat work here: this
    # loop passes the SAME ohlcv["Close"] to each of N features for each of
    # M entities, after _check_required_columns has already inspected it.
    # The scope keeps the checks (an object still has to pass once) and
    # drops the N-1 repeats -- measured at 12% of the build at 50 entities
    # and 18% at 100.
    # Technical indicators for the whole universe in one native call, when
    # every entity shares an index (see panel_features for why that guard
    # is required rather than merely convenient). Returns {} when it does
    # not apply, and the loop below then computes those features per entity
    # exactly as before.
    panel_outputs = compute_panel_features(
        spec.features, feature_defs, resolved_params, ohlcv_by_entity
    )
    with memoized_input_checks():
        for symbol, ohlcv in ohlcv_by_entity.items():
            _check_required_columns(ohlcv, symbol, feature_defs, feature_names)
            columns: Dict[str, pd.Series] = {}
            for fs, definition, params in zip(
                spec.features, feature_defs, resolved_params
            ):
                if fs.output_name in panel_outputs:
                    columns[fs.output_name] = _check_entity_output(
                        panel_outputs[fs.output_name][symbol],
                        fs.output_name,
                        definition.id,
                        ohlcv,
                        symbol,
                    )
                elif definition.scope == FeatureScope.ENTITY:
                    columns[fs.output_name] = _check_entity_output(
                        definition.fn(ohlcv, context, **params),
                        fs.output_name,
                        definition.id,
                        ohlcv,
                        symbol,
                    )
                else:
                    columns[fs.output_name] = universe_outputs[fs.output_name][symbol]
            # Lags are added HERE -- on one entity's own frame, on its
            # own bar index -- so a shift can only reach that entity's
            # earlier rows. After stack_long it would cross the entity
            # boundary and hand one symbol another's history, which
            # produces a plausible panel and no visible symptom.
            per_entity_features[symbol] = expand_lags(
                pd.DataFrame(columns), lags_requested
            )
            if include_target:
                target_by_entity[symbol] = build_target(ohlcv["Close"], spec.target)
                # Recorded per row, per entity, from that entity's OWN bar
                # index -- see build_label_end_dates for why an integer
                # offset against the global date axis is not equivalent.
                label_end_by_entity[symbol] = build_label_end_dates(
                    ohlcv["Close"], spec.target
                )
                # One label per requested horizon, off the SAME features
                # -- but ONLY when more than one was asked for. A
                # single-horizon spec is every spec written before
                # `horizons` existed, and emitting `target__h5` beside
                # `target` for those would duplicate a column and change the
                # panel shape of every dataset in existence to no purpose.
                # `spec.target` is cloned rather than mutated so each build
                # reads a genuine TargetSpec -- every target builder takes
                # one and reads `horizon` off it.
                for horizon, name in zip(
                    spec.target.horizons if len(spec.target.horizons or []) > 1 else [],
                    spec.target.horizon_names,
                ):
                    at_horizon = spec.target.model_copy(
                        update={"horizon": horizon, "horizons": [horizon]}
                    )
                    extra_targets.setdefault(name, {})[symbol] = build_target(
                        ohlcv["Close"], at_horizon
                    )
                    extra_label_ends.setdefault(name, {})[symbol] = (
                        build_label_end_dates(ohlcv["Close"], at_horizon)
                    )

    if include_target:
        long_panel, drop_attribution = stack_long(
            per_entity_features,
            target_by_entity,
            label_end_by_entity,
            extra_targets,
            extra_label_ends,
        )
        # A rank within the date, and a return measured against the
        # universe average, are defined against the OTHER entities present
        # that day, so they cannot exist until every entity is in one
        # frame. Applied here, then the rows it could not define (a date
        # carrying a single entity has no cross-section) are dropped, so
        # the panel keeps its "no NaN targets" contract.
        if spec.target.type in CROSS_SECTIONAL_TARGETS:
            long_panel = apply_cross_sectional_target(long_panel, spec.target)
            n_before = len(long_panel)
            long_panel = long_panel[long_panel["target"].notna()]
            n_dropped = n_before - len(long_panel)
            if n_dropped:
                warnings.append(
                    f"NOTE: target {spec.target.type!r} is cross-sectional, so "
                    f"{n_dropped} row(s) on dates carrying a single entity were "
                    "dropped — a one-name cross-section has no rank and a "
                    "market-relative return of exactly zero by construction, "
                    "neither of which is a measurement."
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
        "feature_ids": expanded_names,
        "data_hash": data_hash,
        "spec_hash": dataset_spec_hash(spec),
        # What the frames in this dataset can and cannot support, as one
        # verdict from `data.bundle`. See the bundle construction above.
        "temporal_bundle": bundle_verdict,
    }
    result["target_id"] = (
        f"{spec.target.type}:{spec.target.horizon}" if include_target else None
    )
    # The same shape register_external_panel records, so `_select_target`
    # does not need to know whether the panel was built here or handed in.
    # Empty for a single-horizon spec: there is one label, it is `target`,
    # and there is nothing to select between. Reporting a one-entry list
    # would advertise a choice that does not exist.
    result["targets"] = (
        [
            {
                "name": name,
                "column": f"target__{name}",
                "horizon": horizon,
                "target_type": spec.target.type,
                "label_end_column": f"label_end_date__{name}",
            }
            for horizon, name in zip(
                spec.target.horizons or [], spec.target.horizon_names
            )
        ]
        if include_target and len(spec.target.horizons or []) > 1
        else []
    )
    return result
