"""Hash-chain tamper-evidence verification: a single day file in isolation
(`verify_audit_log_integrity`), or the full cross-day trail -- the chain
index's own chain plus every day file it attests to
(`verify_audit_trail_integrity`).

`scripts/verify_audit_log.py` is a stdlib-only copy of this module for an
auditor who will not install the package. Every change here is mirrored
there, and `tests/audit/test_standalone_verifier.py` holds the two to the
same answers."""

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

from .hashing import hash_payload
from .paths import _GENESIS_HASH, _INDEX_FILENAME, _audit_dir, _iter_day_files

# ── Lines written before the record hash was taken over the written form ────
#
# Until the CHANGELOG entry of 2026-09-27 the writer hashed a record's live
# values and then wrote pydantic's JSON of them, which spells a NaN or an
# infinity as `null`. Such a line can never reproduce its stored hash, so the
# day reported "content altered" for ever although nobody touched it -- and
# the only way to silence that was to edit the log. Those lines are already
# on disk and cannot be rewritten.
#
# What CAN be done without rewriting anything: try putting NaN, +inf or -inf
# back where the line now holds `null` inside `input`, and hash that. If one
# such restoration reproduces the stored hash, the line is exactly what the
# old writer produced for a non-finite input. That is sound in the one
# direction that matters: a line is accepted only when a restoration hashes
# to the stored value, so an edit (1.0 -> null, 3.0 -> 4.0, anything else)
# still fails unless the editor also found a second preimage of a 64-bit
# hash. The search is bounded -- restorations of one null first, then two,
# and so on, up to a fixed number of trial hashes and bytes hashed per line
# -- so a line with many nulls costs a bounded time and simply stays
# reported when no restoration within the bound explains it.
#
# Only non-finite floats are tried. They are the only divergence a JSON
# caller could reach; the others (timestamps, bytes, sets, integer keys)
# needed a Python caller and are left reported.

#: Trial hashes spent on one line, at most.
_EXPLAIN_MAX_TRIALS = 4096
#: Serialised bytes hashed for one line, at most -- a record carrying a large
#: input gets proportionally fewer trials.
_EXPLAIN_MAX_BYTES = 16 * 1024 * 1024
_NON_FINITE = (float("nan"), float("inf"), float("-inf"))


def _null_paths(node: Any, path: Tuple[Any, ...] = ()) -> Iterator[Tuple[Any, ...]]:
    if node is None:
        yield path
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _null_paths(value, path + (key,))
    elif isinstance(node, list):
        for position, value in enumerate(node):
            yield from _null_paths(value, path + (position,))


def _set_at(node: Any, path: Tuple[Any, ...], value: Any) -> None:
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value


def _spell(path: Tuple[Any, ...], value: float) -> str:
    where = "input" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}" for part in path
    )
    token = "NaN" if value != value else ("Infinity" if value > 0 else "-Infinity")
    return f"{where}={token}"


def _non_finite_explanation(record: Dict[str, Any]) -> Optional[str]:
    """Where restoring NaN/+inf/-inf in place of a `null` inside `input`
    reproduces the stored record_hash, spelled for a reader; None when no
    restoration within the bound does. See the block comment above."""
    claimed = record.get("record_hash")
    source = record.get("input")
    if not isinstance(claimed, str) or not isinstance(source, dict):
        return None
    nulls = [path for path in _null_paths(source) if path]
    if not nulls:
        return None
    size = max(1, len(json.dumps(record)))
    budget = min(_EXPLAIN_MAX_TRIALS, max(1, _EXPLAIN_MAX_BYTES // size))
    trial = {**record, "input": json.loads(json.dumps(source)), "record_hash": None}
    spent = 0
    for count in range(1, len(nulls) + 1):
        for chosen in itertools.combinations(nulls, count):
            for values in itertools.product(_NON_FINITE, repeat=count):
                for path, value in zip(chosen, values):
                    _set_at(trial["input"], path, value)
                if hash_payload(trial) == claimed:
                    return ", ".join(_spell(p, v) for p, v in zip(chosen, values))
                spent += 1
                if spent >= budget:
                    return None
            for path in chosen:
                _set_at(trial["input"], path, None)
    return None


def _non_finite_note(
    path: Path, lineno: int, record: Dict[str, Any], where: str
) -> str:
    return (
        f"{path.name} line {lineno} (request_id={record.get('request_id')}): "
        f"record_hash={record.get('record_hash')!r} does not match the line "
        f"as it now reads, but restoring {where} reproduces it exactly. The "
        "line is as it was first written, by a writer that hashed a "
        "non-finite input value and then wrote it as null; it was not "
        "altered. Records written since carry such values as the strings "
        "'NaN', 'Infinity' and '-Infinity'."
    )


class _LineCheck(NamedTuple):
    """One record's two checks, as `_walk_day` found them."""

    lineno: int
    record: Dict[str, Any]
    expected_prev: Any
    claimed_prev: Any
    claimed_hash: Any
    recomputed: str
    link_holds: bool
    content_holds: bool
    #: Set when the content held only because restoring a non-finite input
    #: value reproduced the stored hash.
    non_finite_explanation: Optional[str]


def _walk_day(path: Path, expected_prev_hash: Optional[str]) -> Iterator[_LineCheck]:
    """
    Every record of one day file with its link and content checked -- the
    walk `verify_audit_log_integrity` reports from and the signed
    checkpoint recomputes a day through, so the two cannot disagree about
    which records hold.

    `expected_prev_hash=None` takes the first record's own claim as the
    day's head, for a day the chain index never witnessed; the head is
    then unchecked and everything after it still is. A line that is not
    JSON raises, as it always has here.
    """
    prev_hash = expected_prev_hash
    first = True
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            claimed_prev = record.get("prev_record_hash")
            if first and prev_hash is None:
                prev_hash = claimed_prev
            first = False
            recomputed = hash_payload({**record, "record_hash": None})
            claimed_hash = record.get("record_hash")
            explanation: Optional[str] = None
            content_holds = recomputed == claimed_hash
            if not content_holds:
                explanation = _non_finite_explanation(record)
                content_holds = explanation is not None
            yield _LineCheck(
                lineno=lineno,
                record=record,
                expected_prev=prev_hash,
                claimed_prev=claimed_prev,
                claimed_hash=claimed_hash,
                recomputed=recomputed,
                link_holds=claimed_prev == prev_hash,
                content_holds=content_holds,
                non_finite_explanation=explanation,
            )
            prev_hash = claimed_hash or prev_hash


def verify_audit_log_integrity(
    path: Union[str, Path],
    expected_prev_hash: str = _GENESIS_HASH,
    notes: Optional[List[str]] = None,
) -> List[str]:
    """
    Walk a single day's JSONL audit file and confirm its hash chain is
    intact. Returns a list of human-readable problems (empty if the file is
    clean or doesn't exist). Detects a record whose content was edited after
    the fact, or a record removed/reordered/inserted — as long as every
    later record in the file wasn't *also* rewritten to match, which is a
    fundamentally unreachable guarantee without an external, independently
    stored anchor (e.g. signing the last hash of each day into a separate
    system) — this function does not attempt that.

    Args:
        expected_prev_hash: the chain head this file's FIRST record should
            claim as its prev_record_hash. Defaults to the genesis hash,
            correct when verifying a file in isolation (or the very first
            day file the audit trail ever wrote). When verifying a file as
            part of the larger cross-day trail, pass the chain index's
            claimed chain_head for this day instead — see
            verify_audit_trail_integrity, which does this automatically —
            so a wholesale-regenerated day file with an internally
            consistent but fabricated starting point is still caught.
        notes: when given, a line accepted only because restoring a
            non-finite input value reproduced its stored hash is described
            here instead of being silently passed -- such a line is not a
            problem, but a reader of the verdict should know it was found.
    """
    path = Path(path)
    if not path.exists():
        return []
    problems: List[str] = []
    for check in _walk_day(path, expected_prev_hash):
        request_id = check.record.get("request_id")
        if not check.link_holds:
            problems.append(
                f"line {check.lineno} (request_id={request_id}): "
                f"prev_record_hash={check.claimed_prev!r} does not match the "
                f"preceding record's hash {check.expected_prev!r} — chain broken "
                "(a record was edited, removed, reordered, or inserted)."
            )
        if not check.content_holds:
            problems.append(
                f"line {check.lineno} (request_id={request_id}): "
                f"record_hash={check.claimed_hash!r} does not match its own "
                f"recomputed content hash {check.recomputed!r} — this line's "
                "content was altered after it was written."
            )
        elif check.non_finite_explanation is not None and notes is not None:
            notes.append(
                _non_finite_note(
                    path, check.lineno, check.record, check.non_finite_explanation
                )
            )
    return problems


def _last_record_hash(path: Path) -> Optional[str]:
    """
    The `record_hash` on a day file's last non-blank line — the chain head
    the NEXT indexed day has to claim. None when the file is missing, empty,
    or its last line cannot be read as a record: that is a different
    complaint, already made by the per-record walk above, and guessing a
    tail here would turn one problem into two.
    """
    if not path.exists():
        return None
    last_line: Optional[str] = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line is None:
            return None
        parsed = json.loads(last_line)
    except (OSError, ValueError):
        return None
    return parsed.get("record_hash") if isinstance(parsed, dict) else None


def verify_audit_trail_integrity(
    audit_dir: Optional[Union[str, Path]] = None,
    notes: Optional[List[str]] = None,
) -> List[str]:
    """
    Verify the FULL cross-day audit trail, not just one file: the
    independent chain-index witness log's own hash chain, that every day
    file the index attests to still exists on disk (and the reverse — a day
    file present with no matching index entry, for any date at or after the
    index's earliest entry), each day file's own internal record chain
    seeded with the chain head the index claims for that day — so a
    wholesale-regenerated day file with a fabricated-but-internally-
    consistent chain is still caught, which verify_audit_log_integrity(path)
    alone (with its default genesis-hash assumption) cannot detect — and,
    since the CHANGELOG entry of 2026-09-22, each day's ENDING hash against
    the next indexed day's recorded chain head.

    That last check is what closes the hole the head check leaves open: a
    day's head is published in plaintext in the chain index sitting next to
    the day files, so an attacker who rewrites a day's records and
    re-derives its chain from that real head produces a file that is
    internally consistent AND correctly seeded. Only where that day ENDS
    gives it away, and the next day's index entry — written before the
    rewrite and chained into the index's own hash chain — is the
    independent record of where it should have ended.

    Days before the chain index's earliest entry (audit activity that
    predates this feature, or an audit directory with no index at all) are
    NOT cross-day-linked, by design — retroactively rewriting old records to
    link them in would itself be indistinguishable from tampering. Verify
    those individually with verify_audit_log_integrity(path) instead.

    Returns a list of human-readable problems (empty if everything's clean,
    including the case where the audit directory doesn't exist yet). When
    `notes` is given it receives what was found and is not a problem, as in
    verify_audit_log_integrity.
    """
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    problems: List[str] = []
    index_path = directory / _INDEX_FILENAME

    index_entries: List[Dict[str, Any]] = []
    if index_path.exists():
        prev_index_hash = _GENESIS_HASH
        with open(index_path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                claimed_prev = entry.get("prev_index_hash")
                if claimed_prev != prev_index_hash:
                    problems.append(
                        f"chain index line {lineno} (date={entry.get('date')}): "
                        f"prev_index_hash={claimed_prev!r} does not match the "
                        f"preceding entry's hash {prev_index_hash!r} — index "
                        "chain broken (an entry was edited, removed, "
                        "reordered, or inserted)."
                    )
                recomputed = hash_payload({**entry, "index_hash": None})
                claimed_hash = entry.get("index_hash")
                if recomputed != claimed_hash:
                    problems.append(
                        f"chain index line {lineno} (date={entry.get('date')}): "
                        f"index_hash={claimed_hash!r} does not match its own "
                        f"recomputed content hash {recomputed!r} — this entry "
                        "was altered after it was written."
                    )
                prev_index_hash = claimed_hash or prev_index_hash
                index_entries.append(entry)

    indexed_dates = {e["date"] for e in index_entries if e.get("date")}
    on_disk_dates = {p.stem for p in _iter_day_files(directory)}

    for date in sorted(indexed_dates - on_disk_dates):
        problems.append(
            f"chain index attests to activity on {date}, but {date}.jsonl "
            "no longer exists on disk — likely deleted."
        )

    if indexed_dates:
        earliest_indexed_date = min(indexed_dates)
        unindexed_days = {
            d
            for d in on_disk_dates
            if d >= earliest_indexed_date and d not in indexed_dates
        }
        for date in sorted(unindexed_days):
            problems.append(
                f"{date}.jsonl exists on disk with no corresponding chain "
                "index entry (the index entry may have been removed, or "
                "this file was created outside the normal write path)."
            )

    prev_date: Optional[str] = None
    prev_tail: Optional[str] = None
    for entry in index_entries:
        date = entry.get("date")
        if date not in on_disk_dates:
            # Already reported above. The tail of a file that is gone is
            # unknown, so the next day is not accused of re-chaining.
            prev_date, prev_tail = date, None
            continue
        day_path = directory / f"{date}.jsonl"
        expected_head = entry.get("chain_head", _GENESIS_HASH)
        if prev_tail is not None and expected_head != prev_tail:
            problems.append(
                f"chain index says {date} chains onto {expected_head}, but "
                f"{prev_date}.jsonl ends at {prev_tail} — a day was re-chained "
                "from its published head (the head of every day is in "
                "plaintext in the chain index, so a rewritten day can be made "
                "to start exactly where the index says; where it ENDS is what "
                "gives it away)."
            )
        problems.extend(
            verify_audit_log_integrity(
                day_path, expected_prev_hash=expected_head, notes=notes
            )
        )
        prev_date, prev_tail = date, _last_record_hash(day_path)

    return problems
