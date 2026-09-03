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
    "tick_tape": {
        "storage": "frame",
        "description": (
            "Individual trades, timestamp-indexed, with `price` and `size` "
            "columns -- those exact names, because the microstructure tools "
            "refuse without them. What the tick tools measure from, and "
            "what no OHLCV row can be turned back into."
        ),
    },
    "quote_panel": {
        "storage": "frame",
        "description": (
            "Top-of-book quotes, timestamp-indexed, with `bid_price` and "
            "`ask_price` columns -- those exact names, not `bid`/`ask`, "
            "because the microstructure tools refuse without them. Top of "
            "book ONLY -- depth is an `order_book_panel`, a different kind. "
            "Queue position and per-order resting size are in neither: they "
            "need an order-level feed."
        ),
    },
    "data_bundle": {
        "storage": "frame",
        "description": (
            "A MANIFEST of other references, one row per frame: its kind, "
            "its ref, its source and what that source guarantees about "
            "when its rows became knowable. Holds no data itself, which is "
            "what keeps a bundle immutable -- assembling one cannot copy "
            "or diverge from the frames it names."
        ),
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
    "order_book_panel": {
        "storage": "external",
        "description": (
            "L2 depth snapshots -- `timestamp`, then `bid_price_{i}` / "
            "`bid_size_{i}` / `ask_price_{i}` / `ask_size_{i}` per level, "
            "level 0 the touch. EXTERNAL ONLY: a book is registered where it "
            "lies and read in batches, never copied into the runs directory, "
            "so resolving one returns a handle rather than a frame."
        ),
    },
    "order_event_panel": {
        "storage": "external",
        "description": (
            "Order-by-order events -- `timestamp`, `order_id`, `action` "
            "(A add, C cancel, M modify, F fill, T trade, R clear), `side`, "
            "`price`, `size`. A strictly deeper feed than an "
            "`order_book_panel`: depth aggregates size per price level, and "
            "that aggregation is what makes queue position, order lifetime "
            "and a true cancellation rate impossible to recover. EXTERNAL "
            "ONLY, and larger than a book by orders of magnitude."
        ),
    },
    "event_panel": {
        "storage": "external",
        "description": (
            "Rows in event time carrying the point-in-time contract: "
            "`event_time` and `available_time`, which are different columns "
            "because using the first as the second is the leak "
            "point_in_time.py exists to prevent. EXTERNAL ONLY."
        ),
    },
}

#: Kinds that MAY live outside the runs directory, registered by path.
#:
#: Two of these (`order_book_panel`, `event_panel`) can only be external --
#: nothing in this library produces one, so there is no in-memory publish
#: path to preserve. The other two exist in both forms on purpose:
#: `fetch_tick_tape` publishes a tape it fetched, and the same kind of tape
#: bought from a vendor is the same content addressed a different way. One
#: kind, two storages, rather than an `external_tick_tape` that would double
#: the taxonomy and let a consumer accept one and refuse the other.
EXTERNAL_KINDS = frozenset(
    {
        "order_book_panel",
        "order_event_panel",
        "event_panel",
        "tick_tape",
        "quote_panel",
    }
)

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
    if KINDS[kind]["storage"] == "external":
        raise ValidationError(
            f"kind {kind!r} cannot be published from memory -- it is stored "
            "by reference to a file that stays where it is. Use "
            "publish_external(path=...), which records a pointer and a "
            "schema instead of copying the bytes. A book or an event tape "
            "large enough to need this kind is large enough that copying it "
            "into the runs directory is the thing to avoid."
        )
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


def _read_sidecar(run_id: str, name: str) -> Dict[str, Any]:
    """What was recorded beside a published value, or an empty dict."""
    sidecar = _resolved_within_runs_dir(_runs_dir() / run_id / _sidecar_name(name))
    if not sidecar.exists():
        return {}
    try:
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a damaged sidecar is not fatal
        return {}
    return loaded if isinstance(loaded, dict) else {}


def publish_external(
    path: str,
    kind: str,
    run_id: str,
    name: str,
    producer: Optional[str] = None,
    fmt: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[str, Any]:
    """
    Register data that stays where it is, and return a reference to it.

    The counterpart to `publish` for datasets too large to copy. What lands
    in the runs directory is the sidecar and nothing else -- no Parquet, no
    second copy -- so registering a forty-gigabyte book costs a schema read.

    THE IMMUTABILITY PROMISE IS WEAKER HERE, and deliberately visible rather
    than quietly dropped. `publish` can promise that resolving a reference
    twice yields the same bytes, because this library wrote those bytes and
    refuses to overwrite them. An external file belongs to the caller and can
    be re-extracted underneath a live reference. The sidecar therefore
    records a fingerprint at registration, and `describe` re-reads it, so a
    file that moved or changed is REPORTED rather than silently resolving to
    something else. The `(run_id, name)` collision rule is unchanged: a
    second registration under the same pair is refused, because two datasets
    answering to one reference is the failure that rule exists to prevent.

    Returns the reference and the `ExternalDataset` handle, so a caller that
    just registered something does not have to resolve it to learn its shape.
    """
    from standard_quant_tools.data import external as _external

    if kind not in KINDS:
        raise ValidationError(f"unknown kind {kind!r}; expected one of {sorted(KINDS)}")
    if kind not in EXTERNAL_KINDS:
        raise ValidationError(
            f"kind {kind!r} cannot be registered by path; it is published "
            f"from memory with publish(). Kinds that may live outside the "
            f"runs directory: {sorted(EXTERNAL_KINDS)}."
        )
    _validate_identifier(run_id, "run_id")
    _validate_identifier(name, "name")

    handle = _external.inspect(path, kind=kind, fmt=fmt)
    problems = _external.check_schema(kind, handle.columns)
    if problems:
        raise ValidationError(
            f"{handle.path.name} does not have the shape a {kind!r} needs: "
            + "; ".join(problems)
            + ". Registering it anyway would move the failure from here to "
            "whatever first tried to read a column that is not there."
        )

    sidecar = _resolved_within_runs_dir(_runs_dir() / run_id / _sidecar_name(name))
    if sidecar.exists() and not overwrite:
        raise ValidationError(
            f"a dataset is already registered at run_id={run_id!r} "
            f"name={name!r}. A reference names one dataset; pointing it at a "
            "second would change what every existing holder resolves. Choose "
            "a fresh run_id, or pass overwrite=True to invalidate them."
        )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(
            {
                "kind": kind,
                "producer": producer,
                "storage": "external",
                "path": str(handle.path),
                "format": handle.fmt,
                "rows": handle.rows,
                "columns": list(handle.columns),
                "dtypes": handle.dtypes,
                "n_files": handle.n_files,
                "size_bytes": handle.size_bytes,
                "fingerprint": handle.fingerprint,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return f"{SCHEME}://{kind}/{run_id}/{name}", handle


def _resolve_external(reference: "Reference", sidecar: Dict[str, Any]) -> Any:
    """Rebuild the handle a registration recorded, checking it still fits."""
    from standard_quant_tools.data import external as _external

    stored = sidecar.get("path")
    if not stored:
        raise ValidationError(
            f"{reference.ref!r} is registered as external but its sidecar "
            "records no path. The registration is unusable; register the "
            "dataset again under a fresh name."
        )
    try:
        # The row count recorded at registration is reused rather than
        # recomputed. For CSV that count was a full scan, and paying for it
        # again on every resolve would be a full read inside the mechanism
        # built to avoid them. It is only trusted while the fingerprint
        # still matches -- past that it describes bytes that are gone.
        handle = _external.inspect(
            str(stored),
            kind=reference.kind,
            fmt=sidecar.get("format"),
            known_rows=sidecar.get("rows"),
            count_rows=False,
        )
        if handle.fingerprint != sidecar.get("fingerprint"):
            handle = _external.inspect(
                str(stored), kind=reference.kind, fmt=sidecar.get("format")
            )
        return handle
    except ValidationError as exc:
        raise ValidationError(
            f"{reference.ref!r} points at {stored}, which cannot be read now "
            f"-- {exc} Nothing was copied when it was registered, so a "
            "reference to external data is only as good as the file it names."
        ) from exc


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

    # The SIDECAR decides the storage, not the kind's default. A `tick_tape`
    # exists both ways -- fetched and copied, or registered where it lies --
    # and only the registration knows which this one is.
    sidecar = _read_sidecar(reference.run_id, reference.name)
    if sidecar.get("storage") == "external":
        return _resolve_external(reference, sidecar)

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
    recorded = _read_sidecar(reference.run_id, reference.name)
    producer = recorded.get("producer")

    if recorded.get("storage") == "external":
        handle = _resolve_external(reference, recorded)
        moved = handle.fingerprint != recorded.get("fingerprint")
        return {
            "ref": reference.ref,
            "kind": reference.kind,
            "description": KINDS[reference.kind]["description"],
            "producer": producer,
            "storage": "external",
            "path": str(handle.path),
            "format": handle.fmt,
            # NOT a content hash, and the name says so. See
            # data/external.py::fingerprint for why the weaker check that
            # always runs beats the stronger one nobody would wait for.
            "fingerprint": handle.fingerprint,
            "changed_since_registration": bool(moved),
            "rows": handle.rows,
            "columns": [str(c) for c in handle.columns],
            "n_files": handle.n_files,
            "size_bytes": handle.size_bytes,
            "index_start": None,
            "index_end": None,
        }

    frame = _load_path(reference.run_id, reference.name)
    path = _resolved_within_runs_dir(
        _runs_dir() / reference.run_id / f"{reference.name}.parquet"
    )
    return {
        "ref": reference.ref,
        "kind": reference.kind,
        "description": KINDS[reference.kind]["description"],
        "producer": producer,
        "storage": "local",
        # So a consumer can prove it read what the producer wrote. Across a
        # fleet those are different processes at different times, and
        # "same reference" is only as good as "same bytes". An externally
        # registered dataset gets a `fingerprint` instead, which is a
        # weaker claim deliberately spelled with a different key.
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
    "EXTERNAL_KINDS",
    "KINDS",
    "Reference",
    "describe",
    "kinds",
    "parse",
    "publish",
    "publish_external",
    "resolve",
]
