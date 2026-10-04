"""verify_replay(): re-run a recorded tool call and compare data/output
hashes against what was originally stored.

Covers BOTH agent surfaces. The tool registry used to be hardcoded to
`agent.tools._TOOL_DISPATCH`, so a `run_model_experiment` record — which
modeling_dispatch had faithfully written to the audit log — could not be
replayed at all: it failed with "Unknown tool". The modeling runtime is
deliberately independent of the 46-tool registry, so replay resolves
against each in turn rather than either importing the other.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from standard_quant_tools.error import ValidationError

from . import provenance as _provenance
from .context import DATA_SOURCE_HASH_VERSION, _data_sources_var, _ReplaySources
from .hashing import ROUNDED_SIGNIFICANT_DIGITS, hash_payload, round_floats
from .models import ReplayResult

# Identifiers minted fresh on every modeling run: `ds_` + 12 hex for a
# dataset, `mdl_` + 12 hex for a model (see modeling.artifacts).
_VOLATILE_ID_RE = re.compile(r"\b(?:ds|mdl)_[0-9a-f]{12}\b")


def _resolve_tool(tool_name: str) -> Tuple[Any, Any, str]:
    """
    Find `tool_name` in either agent surface.

    Local imports: both tool packages import this one, so importing them
    back at module load time would be circular.
    """
    from standard_quant_tools.agent.tools import _TOOL_DISPATCH

    if tool_name in _TOOL_DISPATCH:
        fn, model_cls = _TOOL_DISPATCH[tool_name]
        return fn, model_cls, "agent"

    from standard_quant_tools.modeling.agent.tools import MODELING_TOOL_DISPATCH

    if tool_name in MODELING_TOOL_DISPATCH:
        fn, model_cls = MODELING_TOOL_DISPATCH[tool_name]
        return fn, model_cls, "modeling"

    # The feature lab is the third surface. Its records used to fail here
    # with 'Unknown tool', so the most expensive call in that runtime
    # could be neither replayed nor pre-validated.
    from standard_quant_tools.modeling.agent.feature_tools import (
        FEATURE_TOOL_DISPATCH,
    )

    if tool_name in FEATURE_TOOL_DISPATCH:
        fn, model_cls = FEATURE_TOOL_DISPATCH[tool_name]
        return fn, model_cls, "feature_lab"

    raise ValueError(
        f"Unknown tool {tool_name!r} in decision record — not found in the agent, "
        f"modeling or feature_lab tool registries."
    )


def _has_volatile_identifiers(obj: Any) -> bool:
    """True when any string in `obj` carries a run-specific dataset/model id."""
    if isinstance(obj, str):
        return bool(_VOLATILE_ID_RE.search(obj))
    if isinstance(obj, dict):
        return any(_has_volatile_identifiers(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_volatile_identifiers(v) for v in obj)
    return False


def normalize_identifiers(obj: Any) -> Any:
    """
    Replace run-specific dataset/model ids with a stable placeholder.

    Applied to BOTH the recorded and the replayed output before comparison,
    so the comparison asks whether the substance reproduced rather than
    whether two UUIDs happened to match. Deliberately narrow: only the
    `ds_`/`mdl_` identifier pattern is rewritten, including where it appears
    inside an artifact path — a genuine change to any metric, feature list
    or fold count still shows up as a mismatch.
    """
    if isinstance(obj, str):
        return _VOLATILE_ID_RE.sub("<run_id>", obj)
    if isinstance(obj, dict):
        return {k: normalize_identifiers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_identifiers(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(normalize_identifiers(v) for v in obj)
    return obj


# Internal alias kept so call sites read consistently with the other
# underscore-prefixed helpers in this module.
_normalize_identifiers = normalize_identifiers


def _detail_differences(
    name: str, then: Dict[str, Any], now: Optional[Dict[str, Any]]
) -> List[str]:
    """One line per fact in the recorded `then` that `now` does not repeat.

    The record is the authority on what it vouched for: a fact it carries
    is compared, and a fact it does not carry -- one a later release began
    recording -- is not, so adding a fact never turns an older record's
    verdict."""
    if not isinstance(now, dict):
        return [f"{name}: recorded {then!r}, now {now!r}"]
    return [
        f"{name}.{key}: recorded {then[key]!r}, now {now.get(key)!r}"
        for key in sorted(then)
        if then[key] != now.get(key)
    ]


def _build_differences(record: Dict[str, Any]) -> List[str]:
    """
    How the native build, instruction-set path and platform that wrote
    `record` differ from the ones replaying it, one line per fact that
    differs.

    `native_build` and `native_isa`: a record that does not carry one counts
    as different, since it cannot vouch for having run where the replay
    runs.

    `native_detail` (compiler, configuration, OpenMP runtime, PGO, CRT
    linkage) and `platform` (OS, machine, C runtime version, its FMA3
    path): each fact the record carries is compared, because two builds of
    the same sources share one `native_build` label and the same build on
    another C runtime or CPU can differ in the last bits. A record without
    them -- written before they existed, or with no extension in use, which
    `native_build` already says -- is judged on `native_build` and
    `native_isa` alone, exactly as before.

    Empty means the same build on the same path and, as far as the record
    says, the same platform: the one case in which the exact output hash is
    promised to reproduce.
    """
    current = {
        "native_build": _provenance._native_build_label(),
        "native_isa": _provenance._native_isa_label(),
    }
    differences: List[str] = []
    for name, now in current.items():
        then = record.get(name)
        if then is None:
            differences.append(f"{name}: not recorded, now {now!r}")
        elif then != now:
            differences.append(f"{name}: recorded {then!r}, now {now!r}")
    for name, probe in (
        ("native_detail", _provenance._native_detail),
        ("platform", _provenance._platform_facts),
    ):
        recorded = record.get(name)
        if isinstance(recorded, dict):
            differences.extend(_detail_differences(name, recorded, probe()))
    return differences


def _records_build_facts(record: Dict[str, Any]) -> bool:
    """Whether `record` names the compiler and platform, not only the
    sources and the instruction-set path."""
    return isinstance(record.get("native_detail"), dict) or isinstance(
        record.get("platform"), dict
    )


def _redacted_input_fields(node: Any, prefix: str = "") -> List[str]:
    """
    Names of input fields holding a redaction placeholder.

    Matches the `<redacted:...>` form redaction.py's _placeholder_for emits;
    the two are coupled by that format, which is why this looks for the
    marker rather than trying to re-derive which fields the configured policy
    would have scrubbed (that policy can change between the write and the
    replay, so the RECORD is the authority, not the current configuration).
    """
    found: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(_redacted_input_fields(value, f"{prefix}{key}."))
    elif isinstance(node, list):
        for item in node:
            found.extend(_redacted_input_fields(item, prefix))
    elif isinstance(node, str) and node.startswith("<redacted:"):
        found.append(prefix.rstrip("."))
    return sorted(set(found))


_SourceKey = Tuple[str, str, str, str]

#: How one data source compared: the hashes agree (`same`), agree only
#: under the other pandas's representation (`respelled`), differ in a form
#: that does not depend on pandas or between digests compared as given
#: (`changed`), differ in the earlier form under every representation tried
#: (`undecided`), or were taken in forms that cannot be compared
#: (`uncompared`).
_SAME, _RESPELLED, _CHANGED, _UNDECIDED, _UNCOMPARED = (
    "same",
    "respelled",
    "changed",
    "undecided",
    "uncompared",
)


def _source_key(entry: Dict[str, Any]) -> _SourceKey:
    return (entry["symbol"], entry["start"], entry["end"], entry["interval"])


def _hash_version(entry: Dict[str, Any]) -> Optional[int]:
    """The form of a data source's `content_hash`: 1 for an entry without
    `content_hash_version` (every entry written before the key existed),
    otherwise what it says; None for a value that is not a version."""
    version = entry.get("content_hash_version")
    if version is None:
        return 1
    if isinstance(version, bool):
        return None
    try:
        return int(version)
    except (TypeError, ValueError):
        return None


def _compare_source(
    key: _SourceKey,
    old: Optional[Dict[str, Any]],
    new: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    """
    One `data_source_matches` entry -- the recorded and the replayed hash of
    one data source, compared like with like -- and how it compared.

    A version-2 record is compared with the replay's canonical hash. A
    version-1 record is compared with `hash_dataframe` of the replayed
    frame, as read and then under the other pandas's dtype names and
    datetime resolutions: `new_hash` is the value as read, and
    `reproduced_with` names the representation that matched when it was not
    the one read. Two digests of the same declared form that the replay did
    not take itself -- a provider outside this library reporting its own --
    are compared as given. Forms that cannot be bridged (an unversioned
    digest against a versioned one, or a version this release does not
    know) are not compared, and `match` is None.

    `hash_version` is the form the record's hash is in: 1, 2, or None for a
    value that names no version this release knows.
    """
    symbol, start, end, interval = key
    present = old if old is not None else new
    entry: Dict[str, Any] = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "interval": interval,
        "old_hash": old.get("content_hash") if old is not None else None,
        "new_hash": new.get("content_hash") if new is not None else None,
        "match": False,
        "hash_version": _hash_version(present) if present is not None else None,
        "reproduced_with": None,
    }
    if old is None or new is None:
        # Fetched by only one of the two: a difference whatever the form.
        return entry, _CHANGED
    old_version = _hash_version(old)
    if old_version == 1 and "legacy_content_hash" in new:
        entry["new_hash"] = new["legacy_content_hash"]
        if entry["new_hash"] == entry["old_hash"]:
            entry["match"] = True
            return entry, _SAME
        variant = new.get("legacy_variant")
        if variant is not None:
            entry["match"] = True
            entry["reproduced_with"] = variant
            return entry, _RESPELLED
        return entry, _UNDECIDED
    known = old_version in (1, DATA_SOURCE_HASH_VERSION)
    if known and old_version == _hash_version(new):
        entry["match"] = entry["old_hash"] == entry["new_hash"]
        return entry, _SAME if entry["match"] else _CHANGED
    entry["match"] = None
    return entry, _UNCOMPARED


def _named(entries: List[Dict[str, Any]], limit: int = 5) -> str:
    names = [
        f"{m['symbol']} {m['start']} -> {m['end']} ({m['interval']})" for m in entries
    ]
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _data_source_notes(
    compared: List[Tuple[Dict[str, Any], str]], output_moved: bool
) -> List[str]:
    """What the data-source comparison says, one note per kind of finding."""
    kinds: Dict[str, List[Dict[str, Any]]] = {}
    for entry, kind in compared:
        kinds.setdefault(kind, []).append(entry)
    notes: List[str] = []

    respelled = kinds.get(_RESPELLED, [])
    if respelled:
        hows = "; ".join(sorted({m["reproduced_with"] for m in respelled}))
        notes.append(
            f"The content hash recorded for {_named(respelled)} is the earlier "
            "form, `hash_dataframe`, which covers how pandas spells each "
            "column's dtype and stores its datetimes. The replayed data "
            f"reproduces it with {hows}, so it is the data that was recorded; "
            "only the representation the hash was taken over differs, as it "
            "does between pandas 2 and pandas 3."
        )
    if kinds.get(_CHANGED):
        if output_moved:
            notes.append(
                "Underlying data changed and the output changed accordingly — "
                "the provider likely revised historical values."
            )
        else:
            notes.append(
                "Underlying data changed but the output is unaffected "
                "(e.g. a scale- or shift-invariant metric) — worth a closer look."
            )
    undecided = kinds.get(_UNDECIDED, [])
    if undecided:
        import pandas as pd

        notes.append(
            f"The content hash recorded for {_named(undecided)} is the earlier "
            "form, `hash_dataframe`, which depends on how pandas spells each "
            "column's dtype and stores its datetimes. The replayed data does "
            "not reproduce it as read, with text columns spelled `object`, "
            "`str` or `string`, or with its datetime columns and index at "
            "`[ns]`, `[us]`, `[ms]` or `[s]`. The record does not say which "
            f"pandas hashed it, and this replay runs pandas {pd.__version__}, "
            "so this check cannot tell a revision by the provider from a "
            "difference in how that pandas represented the same values. A "
            "call recorded now carries a hash that does not depend on the "
            "pandas version."
        )
    uncompared = kinds.get(_UNCOMPARED, [])
    if uncompared:
        notes.append(
            f"The record and the replay hashed {_named(uncompared)} in "
            "different forms (an unversioned digest reported by a provider "
            "outside this library against a versioned one, or a content-hash "
            "version this release, which writes version "
            f"{DATA_SOURCE_HASH_VERSION}, does not know), so the two were not "
            "compared."
        )
    if output_moved and not (kinds.get(_CHANGED) or undecided):
        notes.append(
            "Output changed even though input data is identical — "
            "code/logic likely changed since the record was written."
            if not uncompared
            else "Output changed, and every input data source that could be "
            "compared is identical — code/logic likely changed since the "
            "record was written."
        )
    return notes


def verify_replay(record: Dict[str, Any]) -> ReplayResult:
    """
    Re-run a recorded tool call and compare data + output hashes against
    what was stored. A data-source mismatch with a matching output usually
    means the provider revised historical data; an output mismatch with
    matching data sources means the code/logic changed since the record
    was written.

    The hashes behind that verdict come back with it — the output hash the
    replay produced, the one the record stored, and both hashes for every
    data source — so "the code changed" arrives with what changed rather
    than as a bare claim.

    Each data source is compared in the form its record took (see
    `_compare_source`). A record written before `content_hash_version`
    existed holds `hash_dataframe`, whose value depends on the pandas
    version; it is checked against the replayed frame's `hash_dataframe` as
    read and under the other pandas's dtype names and datetime resolutions,
    and a miss under all of them is reported as undecided between a revised
    value and a pandas difference, not as a revision.
    """
    fn, model_cls, surface = _resolve_tool(record["tool_name"])
    tool_name = record["tool_name"]

    # A REDACTED input cannot be replayed. The record stores
    # _redact(raw_input, fields), so a redacted field holds a placeholder
    # rather than the original value — reconstructing the call from it would
    # re-run a DIFFERENT call and then compare its output against the
    # original's hash. That comparison is guaranteed to mismatch, and the
    # mismatch would read as evidence of drift rather than as the artefact of
    # redaction that it is. Refused explicitly instead.
    redacted_fields = _redacted_input_fields(record.get("input", {}))
    if redacted_fields:
        raise ValidationError(
            f"decision record {record.get('request_id')} is not replayable: "
            f"input field(s) {redacted_fields} were redacted, so the original "
            "call cannot be reconstructed. Replaying with the placeholder "
            "would run a different call and report the inevitable hash "
            "mismatch as drift. Redaction and exact replay are in tension by "
            "construction; a record needs one or the other."
        )

    # A call that FAILED originally is a first-class outcome, not an absence
    # of one. Replaying it and letting the exception escape reports an error
    # in the replay machinery, when what actually reproduced is the original
    # failure — which is the correct result.
    original_status = record.get("status", "ok")
    original_error = record.get("error_type")

    # The record's data sources by key. A version-1 entry's hash is handed
    # to the replay's providers, which hash the frame they fetch that way
    # too, so the comparison below is made like with like.
    old_by_key = {_source_key(s): s for s in record.get("data_sources", [])}
    replay_sources = _ReplaySources(
        legacy={
            key: entry["content_hash"]
            for key, entry in old_by_key.items()
            if _hash_version(entry) == 1 and entry.get("content_hash") is not None
        }
    )
    token_data = _data_sources_var.set(replay_sources)
    try:
        result_obj = fn(model_cls(**record["input"]))
        new_output = result_obj.model_dump()
        new_sources = list(_data_sources_var.get() or [])
    except Exception as exc:
        if original_status == "error":
            reproduced = type(exc).__name__ == original_error
            return ReplayResult(
                request_id=record.get("request_id", ""),
                tool_name=tool_name,
                output_match=reproduced,
                notes=[
                    "The original call FAILED, and the replay failed too. "
                    f"Original error: {original_error}; replay error: "
                    f"{type(exc).__name__}. "
                    + (
                        "The same failure reproduced, which is a successful "
                        "replay of a failed call."
                        if reproduced
                        else "A DIFFERENT failure occurred, so something has "
                        "changed since the record was written."
                    )
                ],
            )
        raise
    finally:
        _data_sources_var.reset(token_data)

    if original_status == "error":
        # The failure did not reproduce, which is an answer about the call,
        # not about the replay machinery. This arm used to name a field
        # that does not exist on ReplayResult, so it raised TypeError and
        # the caller was told the replay could not run — see the CHANGELOG
        # entry of 2026-09-22. There is no stored output hash to compare
        # against (the original produced no output), so what the replay
        # produced this time is reported on its own.
        return ReplayResult(
            request_id=record.get("request_id", ""),
            tool_name=tool_name,
            output_match=False,
            new_output_hash=hash_payload(new_output),
            notes=[
                f"The original call failed with {original_error}, but the "
                "replay SUCCEEDED. The failure no longer reproduces — the "
                "code, the data or the environment has changed since."
            ],
        )

    stored_output_hash = record.get("output_hash")
    new_output_hash = hash_payload(new_output)
    output_match: Optional[bool] = (
        new_output_hash == stored_output_hash
        if stored_output_hash is not None
        else None
    )

    notes: List[str] = []
    normalized_hash: Optional[str] = None

    # ── Semantic comparison for surfaces with non-deterministic ids ──────
    # Modeling mints a fresh UUID-based dataset_id/model_id on every run and
    # embeds it in artifact paths, so a byte-identical re-run NEVER matches
    # literally -- every modeling replay would report a false mismatch, which
    # is worse than no replay support at all because it looks like evidence
    # of drift. Re-compare with those identifiers normalized away, so the
    # question becomes "did the SUBSTANCE reproduce" rather than "were the
    # random ids the same".
    if output_match is False and _has_volatile_identifiers(new_output):
        normalized_hash = hash_payload(_normalize_identifiers(new_output))
        stored_normalized = record.get("output_hash_normalized")
        if stored_normalized is not None:
            output_match = normalized_hash == stored_normalized
            notes.append(
                "Compared with run-specific identifiers (dataset_id/model_id and the "
                "artifact paths containing them) normalized away — these are freshly "
                "minted per run and never reproduce literally."
            )
        else:
            # Recorded before normalized hashing existed: the literal
            # mismatch cannot be distinguished from a real one.
            output_match = None
            notes.append(
                "This record predates normalized output hashing, and its output "
                "contains run-specific identifiers that never reproduce literally — "
                "so a literal mismatch here is not evidence of drift either way. "
                "Re-record to get a comparable hash."
            )

    # ── The twelve-significant-digit comparison ──────────────────────────
    # An output hash is bit-exact only for the same native build on the
    # same instruction-set path and platform: the AVX2+FMA reduction fuses
    # each multiply-add and sums in four lanes, so it rounds differently
    # from the scalar loop, and a different compiler, OpenMP runtime, CRT
    # linkage, C runtime version or CRT FMA3 path may move the last bits
    # too. Across those the promise is twelve significant digits. So when
    # the exact hash misses, the rounded one decides between "reproduced to
    # twelve digits elsewhere" and a real difference -- and on the SAME
    # build, path and platform a miss keeps the verdict it always had.
    digits = ROUNDED_SIGNIFICANT_DIGITS
    rounded_match: Optional[bool] = None
    new_rounded_hash: Optional[str] = None
    stored_rounded_hash = record.get("output_hash_rounded")
    build_differences: List[str] = []
    if output_match is False:
        build_differences = _build_differences(record)
        new_rounded_hash = hash_payload(
            round_floats(_normalize_identifiers(new_output), digits)
        )
        where = "; ".join(build_differences)
        if stored_rounded_hash is None:
            notes.append(
                "This record predates the rounded output hash, so a "
                "difference in the last bits -- which a different native "
                "build or instruction-set path produces by itself -- cannot "
                f"be told apart from a real one. {digits}-digit comparison "
                "is only possible for records written since it existed."
                + (f" The build differs: {where}." if where else "")
            )
        else:
            rounded_match = new_rounded_hash == stored_rounded_hash
            if rounded_match and build_differences:
                notes.append(
                    f"Reproduced to {digits} significant digits, not bit for "
                    f"bit, on a different build, instruction-set path or "
                    f"platform ({where}). That is the contract across "
                    "builds: an output hash is bit-exact only for the same "
                    "native build on the same instruction-set path and "
                    "platform, and elsewhere the outputs agree to "
                    f"{digits} significant digits. It is not evidence that "
                    "the code changed."
                )
            elif rounded_match and _records_build_facts(record):
                notes.append(
                    f"The output agrees to {digits} significant digits but "
                    "not bit for bit, on the same native build, "
                    "instruction-set path, compiler and platform as far as "
                    "the record names them. Something none of them covers "
                    "moved the last bits: compiler flags given outside the "
                    "build files, the Python-side libraries (NumPy, SciPy, "
                    "pandas) or the BLAS they load."
                )
            elif rounded_match:
                notes.append(
                    f"The output agrees to {digits} significant digits but "
                    "not bit for bit, on the same native build and "
                    "instruction-set path. The build label names the C++ "
                    "sources, not the compiler, its flags or the Python-side "
                    "libraries, so something it does not record moved the "
                    "last bits."
                )
            elif build_differences:
                notes.append(
                    f"The output differs beyond {digits} significant digits "
                    f"as well, so the different build, path or platform "
                    f"({where}) does not account for it on its own. (Two "
                    "values a last bit apart can still round apart at a "
                    "rounding boundary, so this is strong rather than "
                    "conclusive.)"
                )
    reproduced_elsewhere = bool(rounded_match) and bool(build_differences)
    output_moved = output_match is False and not reproduced_elsewhere

    new_by_key = {_source_key(s): s for s in new_sources}
    # Iterate the union of old and new keys, not just new_sources — a key
    # present in the original record but absent from the replay (e.g. the
    # tool no longer fetches a symbol/range it used to) must still be
    # reported, not silently dropped just because the loop only walked
    # what the replay happened to touch.
    compared = [
        _compare_source(key, old_by_key.get(key), new_by_key.get(key))
        for key in sorted(set(old_by_key) | set(new_by_key))
    ]
    data_matches: List[Dict[str, Any]] = [entry for entry, _kind in compared]
    notes.extend(_data_source_notes(compared, output_moved))

    return ReplayResult(
        request_id=record["request_id"],
        tool_name=tool_name,
        output_match=output_match,
        data_source_matches=data_matches,
        notes=notes,
        new_output_hash=new_output_hash,
        stored_output_hash=stored_output_hash,
        new_output_hash_normalized=normalized_hash,
        rounded_output_match=rounded_match,
        new_output_hash_rounded=new_rounded_hash,
        stored_output_hash_rounded=stored_rounded_hash,
        build_differences=build_differences,
    )
