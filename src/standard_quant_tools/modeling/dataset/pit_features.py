"""
Point-in-time features in the dataset build: the gate, the fetch, the
transform and the join.

THE GATE COMES FIRST. A feature of scope POINT_IN_TIME reads a record set
the provider must stamp with availability times, and whether it can is a
fact about the provider that `get_temporal_contract(frame_kind)` states
without a fetch. The builder asks that before it fetches a single bar, so
a spec that could never have been built is refused for the price of a
method call rather than after a universe has been downloaded -- which is
what `data.temporal.require_pit` exists for and what turned
`TemporalSupport.CURRENT_ONLY` from a label nothing used into a line the
builder draws.

THE JOIN COMES LAST. Records are attached to the STACKED panel, by
availability time, with the staleness bound each feature declares; the
features are never computed per entity from bars because they are not
bars. A row the join could not supply -- nobody had the filing yet, or the
last one is too old -- is missing in the same sense any other missing
feature is, and the dataset's `missing.policy` decides what happens to it.

WHAT TRAVELS WITH THE DATASET. The record set and the contract it arrived
under go into the `DataBundle` beside the bars, so the verdict on a
dataset says what its fundamentals could and could not support, and the
restatements OBSERVED in the pull are reported as the measurement the
provider's `unknown` revisions encoding is waiting on.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from standard_quant_tools.data.temporal import TemporalContract, require_pit
from standard_quant_tools.error import ValidationError

from ..features.base import FeatureContext, FeatureDefinition, FeatureScope
from .point_in_time import (
    AVAILABLE_TIME,
    ENTITY,
    EVENT_TIME,
    asof_join,
    coverage_report,
    observed_revisions,
    validate_pit_frame,
)

#: The value column a point-in-time transform returns.
VALUE = "value"

#: Calendar days of records requested before the panel's first date on top
#: of the deepest staleness any requested feature accepts: a fiscal year,
#: so a year-over-year transform has a prior filing for the first rows.
RECORD_HISTORY_DAYS = 400

#: One requested point-in-time feature: its spec, definition and resolved
#: parameters.
PitRequest = Tuple[Any, FeatureDefinition, Dict[str, Any]]


def point_in_time_requests(
    specs: Sequence[Any],
    definitions: Sequence[FeatureDefinition],
    params: Sequence[Dict[str, Any]],
) -> List[PitRequest]:
    """The requested features of POINT_IN_TIME scope, with a refusal for
    the one thing the scope cannot do: a lag in bars."""
    requests: List[PitRequest] = []
    for spec, definition, resolved in zip(specs, definitions, params):
        if definition.scope != FeatureScope.POINT_IN_TIME:
            continue
        if getattr(spec, "lags", None):
            raise ValidationError(
                f"build_model_dataset: feature {spec.output_name!r} is a "
                "point-in-time feature and cannot be lagged in bars -- its "
                "rows are filings, not sessions, and a lag of k bars would "
                "mean a different number of filings for every entity. Request "
                "the prior period as its own feature instead."
            )
        requests.append((spec, definition, resolved))
    return requests


def gate_point_in_time(
    provider: Any,
    requests: Sequence[PitRequest],
    *,
    contract_getter: Callable[[Any, str], Optional[TemporalContract]],
    purpose: str,
) -> Dict[str, TemporalContract]:
    """
    The provider's contract per requested frame kind, or a refusal by
    name -- before anything is fetched.
    """
    contracts: Dict[str, TemporalContract] = {}
    for frame_kind in sorted({d.frame_kind for _s, d, _p in requests}):
        ids = sorted(
            spec.output_name for spec, d, _p in requests if d.frame_kind == frame_kind
        )
        contract = contract_getter(provider, frame_kind)
        if contract is None:
            raise ValidationError(
                f"{purpose}: feature(s) {ids} read point-in-time {frame_kind!r} "
                f"records, and provider {type(provider).__name__!r} declares no "
                "temporal contract for that frame kind. Only a provider that "
                "says when each record became knowable can serve them; "
                "PolygonProvider serves 'fundamentals'."
            )
        require_pit(contract, f"{purpose}: feature(s) {ids} ({frame_kind!r})")
        contracts[frame_kind] = contract
    return contracts


def _fetch_records(
    provider: Any,
    frame_kind: str,
    fields: List[str],
    universe: Sequence[str],
    start: str,
    end: str,
    purpose: str,
) -> pd.DataFrame:
    try:
        records = provider.get_point_in_time_records(
            list(universe), frame_kind, fields, start, end
        )
    except NotImplementedError as exc:
        raise ValidationError(
            f"{purpose}: provider {type(provider).__name__!r} declares a "
            f"point-in-time contract for {frame_kind!r} but does not serve the "
            f"records: {exc}"
        ) from exc
    if not isinstance(records, pd.DataFrame):
        raise ValidationError(
            f"{purpose}: provider {type(provider).__name__!r} returned "
            f"{type(records).__name__} for point-in-time {frame_kind!r} records; "
            "expected a DataFrame in the point_in_time schema."
        )
    return validate_pit_frame(records, name=f"{frame_kind} records")


def _transform(
    definition: FeatureDefinition,
    output_name: str,
    records: pd.DataFrame,
    context: FeatureContext,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """One feature's value series, in the record schema, named for the panel."""
    transform_params = {k: v for k, v in params.items() if k != "max_staleness_days"}
    out = definition.fn(records, context, **transform_params)
    if not isinstance(out, pd.DataFrame):
        raise ValidationError(
            f"feature {definition.id!r} returned {type(out).__name__}; a "
            "point-in-time transform returns a DataFrame with columns "
            f"{[ENTITY, EVENT_TIME, AVAILABLE_TIME, VALUE]}."
        )
    missing = [c for c in (ENTITY, EVENT_TIME, AVAILABLE_TIME, VALUE) if c not in out]
    if missing:
        raise ValidationError(
            f"feature {definition.id!r} returned a frame without {missing}; a "
            "point-in-time transform keeps the record schema and adds "
            f"{VALUE!r}."
        )
    out = out[[ENTITY, EVENT_TIME, AVAILABLE_TIME, VALUE]].rename(
        columns={VALUE: output_name}
    )
    return validate_pit_frame(out, name=f"feature {definition.id!r}")


def join_point_in_time_features(
    panel: pd.DataFrame,
    provider: Any,
    spec: Any,
    requests: Sequence[PitRequest],
    contracts: Dict[str, TemporalContract],
    context: FeatureContext,
    *,
    keep_missing: bool,
    purpose: str = "build_model_dataset",
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]], List[str], Dict[str, Any]]:
    """
    Fetch each frame kind once, transform every requested feature, join
    each onto `panel` by availability time, and drop (or, under `keep`,
    keep) the rows the join could not supply.

    Returns (panel, per-feature drop attribution, warnings, frames by
    kind for the bundle).
    """
    warnings: List[str] = []
    frames: Dict[str, Any] = {}
    columns: List[str] = []
    by_kind: Dict[str, List[PitRequest]] = {}
    for request in requests:
        by_kind.setdefault(request[1].frame_kind, []).append(request)

    for frame_kind, kind_requests in sorted(by_kind.items()):
        fields = sorted({f for _s, d, _p in kind_requests for f in d.fields})
        deepest = max(int(p["max_staleness_days"]) for _s, _d, p in kind_requests)
        start = (
            pd.Timestamp(spec.start) - pd.Timedelta(days=deepest + RECORD_HISTORY_DAYS)
        ).strftime("%Y-%m-%d")
        records = _fetch_records(
            provider, frame_kind, fields, spec.universe, start, spec.end, purpose
        )
        frames[frame_kind] = {"records": records, "contract": contracts[frame_kind]}
        dropped = int(records.attrs.get("n_dropped_without_available_time", 0) or 0)
        if dropped:
            warnings.append(
                f"NOTE: {dropped} {frame_kind} record(s) carried no availability "
                "time and were left out rather than dated to the period they "
                "describe."
            )
        seen = observed_revisions(records)
        if seen["n_restated"]:
            warnings.append(
                f"NOTE: {seen['n_restated']} of {seen['n_facts']} {frame_kind} "
                "fact(s) in the pulled records carry more than one version, so "
                "restatements arrived as rows and the join reads the version "
                "current at each date. The provider declares revisions="
                f"{contracts[frame_kind].revisions!r}; this is the observation "
                "that encoding is waiting on."
            )
        for fs, definition, params in kind_requests:
            name = fs.output_name
            values = _transform(definition, name, records, context, params)
            staleness = pd.Timedelta(days=int(params["max_staleness_days"]))
            panel = asof_join(panel, values, fields=[name], max_staleness=staleness)
            columns.append(name)
            warnings.extend(coverage_report(panel, [name]))

    attribution: Dict[str, Dict[str, int]] = {}
    if columns:
        missing = panel[columns].isna()
        n_missing_per_row = missing.sum(axis=1)
        for name in columns:
            attribution[name] = {
                "n_missing": int(missing[name].sum()),
                "n_sole_missing": int((missing[name] & (n_missing_per_row == 1)).sum()),
            }
        if not keep_missing:
            panel = panel.loc[n_missing_per_row == 0].reset_index(drop=True)
    return panel, attribution, warnings, frames


__all__ = [
    "RECORD_HISTORY_DAYS",
    "VALUE",
    "gate_point_in_time",
    "join_point_in_time_features",
    "point_in_time_requests",
]
