"""
The interconnect: typed references that cross every runtime boundary.

WHY THIS EXISTS RATHER THAN BRIDGE TOOLS. The first attempt at moving a
model's predictions into a backtest was a bespoke tool that knew about both
sides. That approach does not scale: with N producers and M consumers it
needs N x M bridges, every one of which has to be written, tested and kept
in step with both ends. Worse, each bridge is a place where the two sides'
assumptions can quietly diverge.

A reference makes it N + M. A producer publishes a value once and gets back
a string. Any consumer in any runtime resolves that string. Neither side
knows the other exists, and nothing has to be transcribed through a context
window to get from one to the other.

WHAT A REFERENCE IS. `sqt://<kind>/<run_id>/<name>` — a content KIND, and
where the bytes live. The kind is the part that earns its keep: it is
checked on resolve, so handing a trade log to something expecting an equity
curve fails immediately and by name, rather than several frames deep in
pandas with a message about a missing column. Untyped URIs could not do
that, which is why the raw artifact paths this library already hands back
are accepted but reported as `unknown` kind.

WHY REFERENCES AND NOT SHARED DISPATCH. A reference is a value, so it
survives the process boundary between two agents in the multi-agent
orchestrator, it shows up in the audit log as an input to the second call,
and it carries no execution rights — holding a reference to a backtest's
equity curve lets you READ that curve from any runtime and still does not
let you run a backtest.

BULK INPUTS ONLY. A reference is for data too large to be worth moving
through a model's context: a signal panel, a weight panel, an equity curve,
a prediction frame. Small results stay inline, because a reference to a
Sharpe ratio would be indirection with no benefit and one more thing that
can dangle.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from standard_quant_tools.backtest.artifacts import (
    _resolved_within_runs_dir,
    _runs_dir,
    _validate_identifier,
    load_artifact,
    save_artifact,
)
from standard_quant_tools.error import ValidationError

SCHEME = "sqt"

#: kind -> what the resolved value is, and how it is stored.
#:
#: "frame" kinds round-trip as Parquet. "mapping" kinds are nested dicts
#: ({ticker: {date: value}}) that several tools take directly as input;
#: they are stored as a two-level frame and rebuilt on resolve, so the
#: caller gets back exactly the shape the tool wants rather than something
#: it has to reshape and possibly reshape wrongly.
KINDS: Dict[str, Dict[str, str]] = {
    "equity_curve": {
        "storage": "series",
        "description": "Account value per bar, as produced by any backtest.",
    },
    "trade_log": {
        "storage": "frame",
        "description": "One row per completed trade.",
    },
    "signal_panel": {
        "storage": "mapping",
        "description": (
            "{ticker: {date: value}} where value is -1, 0 or 1. What "
            "run_signal_panel_backtest consumes."
        ),
    },
    "weight_panel": {
        "storage": "mapping",
        "description": (
            "{ticker: {date: weight}} as fractions of account equity. What "
            "run_portfolio_simulation consumes."
        ),
    },
    "score_panel": {
        "storage": "mapping",
        "description": (
            "{ticker: {date: score}} of unrestricted alpha scores, before "
            "any conversion into weights."
        ),
    },
    "returns_panel": {
        "storage": "frame",
        "description": "Wide frame of per-asset returns, indexed by date.",
    },
    "price_panel": {
        "storage": "frame",
        "description": "Wide frame of prices or a stacked OHLCV panel.",
    },
    "predictions": {
        "storage": "frame",
        "description": (
            "Long frame of (date, entity, prediction) — what "
            "run_model_experiment persists out of sample."
        ),
    },
    "feature_panel": {
        "storage": "frame",
        "description": "Computed features, entity by date.",
    },
    "indicator_panel": {
        "storage": "frame",
        "description": "Technical indicator values across a universe.",
    },
}

_REF_RE = re.compile(
    rf"^{SCHEME}://(?P<kind>[a-z_]+)/(?P<run_id>[A-Za-z0-9_-]+)/(?P<name>[A-Za-z0-9_-]+)$"
)

#: Sidecar recording what one published reference is. Written beside the
#: data rather than encoded only in the string, so a reference that has
#: been copied, logged or truncated can still be identified from disk --
#: and so `describe()` can report the producing runtime, which the string
#: itself deliberately does not carry (a value should not have to be
#: rewritten because the code that made it moved).
#:
#: ONE FILE PER ARTIFACT, not one catalogue per run_id. A shared catalogue
#: has to be read, updated and rewritten, so two agents publishing
#: different names under the same run_id race and the loser entry
#: disappears -- leaving a live reference that resolves to data of unknown
#: kind. With a fleet of agents that is a routine interleaving rather than
#: an exotic one. A file per artifact has no read-modify-write and so has
#: nothing to lose.


def _sidecar_name(name: str) -> str:
    return f"{name}._handoff.json"


@dataclass(frozen=True)
class Reference:
    """One published value: what it is, and where."""

    ref: str
    kind: str
    run_id: str
    name: str

    @property
    def is_typed(self) -> bool:
        return self.kind in KINDS


def parse(ref: str) -> Reference:
    """Split a reference, or explain why it is not one."""
    match = _REF_RE.match(str(ref).strip())
    if match is None:
        raise ValidationError(
            f"{ref!r} is not a handoff reference. The shape is "
            f"'{SCHEME}://<kind>/<run_id>/<name>', e.g. "
            f"'{SCHEME}://signal_panel/run123/predictions'. A raw artifact "
            "path from an older tool result is accepted by resolve() but "
            "carries no kind, so it cannot be type-checked."
        )
    kind = match.group("kind")
    if kind not in KINDS:
        raise ValidationError(
            f"unknown reference kind {kind!r}; expected one of "
            f"{sorted(KINDS)}. The kind is what makes a mismatched handoff "
            "fail by name instead of several frames deep."
        )
    return Reference(
        ref=ref, kind=kind, run_id=match.group("run_id"), name=match.group("name")
    )


def _mapping_to_frame(mapping: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(mapping)
    frame.index.name = "date"
    return frame


def _frame_to_mapping(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    return {
        str(column): {
            str(index): float(value) for index, value in frame[column].dropna().items()
        }
        for column in frame.columns
    }


def publish(
    data: Any,
    kind: str,
    run_id: str,
    name: str,
    producer: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """
    Store a bulk value and return the reference any runtime can resolve.

    `producer` is recorded but never required. A value is not owned by the
    runtime that made it -- recording it helps a human read a lineage, and
    nothing enforces it, because a consumer that cared where a panel came
    from would be coupled to exactly the thing references remove.

    `overwrite` DEFAULTS TO FALSE, unlike the raw artifact store it sits
    on. A reference is a promise that resolving it twice yields the same
    value, and replacing what one agent published because another picked
    the same (run_id, name) breaks that promise for every holder of the old
    reference -- including holders already recorded in an audit log. With
    many agents choosing ids independently that collision is routine, so it
    fails loudly and the caller picks a fresh id.
    """
    if kind not in KINDS:
        raise ValidationError(f"unknown kind {kind!r}; expected one of {sorted(KINDS)}")
    _validate_identifier(run_id, "run_id")
    _validate_identifier(name, "name")

    storage = KINDS[kind]["storage"]
    if storage == "mapping":
        if not isinstance(data, dict) or not data:
            raise ValidationError(
                f"kind {kind!r} expects a non-empty " "{ticker: {date: value}} mapping"
            )
        payload: Any = _mapping_to_frame(data)
    elif storage == "series":
        payload = (
            data.to_frame(name=getattr(data, "name", None) or "value")
            if isinstance(data, pd.Series)
            else data
        )
    else:
        payload = data

    try:
        save_artifact(payload, run_id, name, overwrite=overwrite)
    except ValidationError as exc:
        if "already exists" not in str(exc):
            raise
        raise ValidationError(
            f"a value is already published at run_id={run_id!r} "
            f"name={name!r}. A reference promises that resolving it twice "
            "gives the same value, so this will not replace it. Choose a "
            "fresh run_id, or pass overwrite=True only if you genuinely "
            "mean to invalidate every existing holder of that reference."
        ) from exc

    sidecar = _resolved_within_runs_dir(_runs_dir() / run_id / _sidecar_name(name))
    sidecar.write_text(
        json.dumps({"kind": kind, "producer": producer}, indent=1), encoding="utf-8"
    )

    return f"{SCHEME}://{kind}/{run_id}/{name}"


def _load_path(run_id: str, name: str) -> pd.DataFrame:
    path = _resolved_within_runs_dir(_runs_dir() / run_id / f"{name}.parquet")
    if not path.exists():
        raise ValidationError(
            f"no stored value at run_id={run_id!r} name={name!r}. A "
            "reference outlives nothing: if the runs directory was cleared "
            "or the value was never published, there is nothing to resolve."
        )
    return load_artifact(str(path))


def resolve(ref: str, expect: Optional[str] = None) -> Any:
    """
    Load what a reference points at, in whatever runtime is asking.

    `expect` is the type check that makes this safe to use as a general
    interconnect. Passing a trade log where an equity curve was wanted
    fails here, naming both kinds, instead of surfacing later as a missing
    column in a drawdown calculation.

    A raw artifact path (what several tools returned before references
    existed) is accepted and loaded, but cannot be type-checked -- so
    `expect` against one is refused rather than silently skipped.
    """
    text = str(ref).strip()
    if not text.startswith(f"{SCHEME}://"):
        if expect is not None:
            raise ValidationError(
                f"{ref!r} is a raw artifact path, which carries no kind, so "
                f"it cannot be checked against expect={expect!r}. Publish it "
                "with a kind, or drop the expectation and check the shape "
                "yourself."
            )
        return load_artifact(
            str(_resolved_within_runs_dir(pd.io.common.stringify_path(ref)))
        )

    reference = parse(text)
    if expect is not None and reference.kind != expect:
        raise ValidationError(
            f"expected a {expect!r} reference but {ref!r} is a "
            f"{reference.kind!r}. {KINDS[expect]['description']} "
            f"What was passed: {KINDS[reference.kind]['description']}"
        )

    frame = _load_path(reference.run_id, reference.name)
    storage = KINDS[reference.kind]["storage"]
    if storage == "mapping":
        return _frame_to_mapping(frame)
    if storage == "series":
        squeezed = frame.squeeze("columns")
        if isinstance(squeezed, pd.DataFrame):
            raise ValidationError(
                f"{ref!r} has {len(frame.columns)} columns; a "
                f"{reference.kind!r} is a single series."
            )
        return squeezed
    return frame


def describe(ref: str) -> Dict[str, Any]:
    """What a reference points at, without loading all of it."""
    reference = parse(ref)
    frame = _load_path(reference.run_id, reference.name)
    sidecar = _resolved_within_runs_dir(
        _runs_dir() / reference.run_id / _sidecar_name(reference.name)
    )
    producer = None
    if sidecar.exists():
        try:
            producer = json.loads(sidecar.read_text(encoding="utf-8")).get("producer")
        except Exception:
            producer = None
    path = _resolved_within_runs_dir(
        _runs_dir() / reference.run_id / f"{reference.name}.parquet"
    )
    return {
        "ref": reference.ref,
        "kind": reference.kind,
        "description": KINDS[reference.kind]["description"],
        "producer": producer,
        # So a consumer can prove it read what the producer wrote. Across a
        # fleet those are different processes at different times, and
        # "same reference" is only as good as "same bytes".
        "content_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "index_start": str(frame.index[0]) if len(frame) else None,
        "index_end": str(frame.index[-1]) if len(frame) else None,
    }


def kinds() -> Dict[str, str]:
    """Every content kind a reference can carry, and what it means."""
    return {kind: meta["description"] for kind, meta in KINDS.items()}


__all__ = [
    "KINDS",
    "Reference",
    "describe",
    "kinds",
    "parse",
    "publish",
    "resolve",
]
