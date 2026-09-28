"""Hash-chain tamper-evidence verification: a single day file in isolation
(`verify_audit_log_integrity`), or the full cross-day trail -- the chain
index's own chain plus every day file it attests to
(`verify_audit_trail_integrity`).

`scripts/verify_audit_log.py` is a stdlib-only copy of this module for an
auditor who will not install the package. Every change here is mirrored
there, and `tests/audit/test_standalone_verifier.py` holds the two to the
same answers -- the same problems, notes and verified head, word for word."""

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union

from .hashing import hash_payload
from .paths import (
    _DAY_FILE_RE,
    _GENESIS_HASH,
    _INDEX_FILENAME,
    _audit_dir,
    _iter_day_files,
)

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


# ── Reading a line that may not be a record ─────────────────────────────────
#
# A line that is not a record used to raise out of the verifier: a truncated
# line, a flipped byte, `{not json` or `[1,2,3]` each aborted the walk, so a
# single junk byte in one day hid every finding after it -- an attacker who
# altered day 19 only had to damage day 18. Each line is now read on its
# own: bytes, then UTF-8, then JSON, then "is it an object". A line that
# fails any step is reported where it is and the walk goes on. See the
# CHANGELOG entry of 2026-09-28.
#
# Lines are split on b"\n" alone, in binary, so a CRLF file and an LF file
# read identically and a stray carriage return inside a damaged line cannot
# split it in two.

_JSON_KINDS = {list: "array", str: "string", int: "number", float: "number"}


def _json_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "true/false"
    if value is None:
        return "null"
    return _JSON_KINDS.get(type(value), type(value).__name__)


def _read_line(raw: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """`(object, None)` for a line holding a JSON object, else `(None, why)`."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, f"not UTF-8 text at byte {exc.start}"
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"not JSON: {exc.msg} at column {exc.colno}"
    except (ValueError, RecursionError):
        return None, "not JSON that can be read"
    if not isinstance(value, dict):
        return None, f"a JSON {_json_kind(value)}, not an object"
    return value, None


def _lines(path: Path) -> Iterator[Tuple[int, bytes]]:
    """Every non-blank line of `path` with its 1-based line number."""
    with open(path, "rb") as f:
        for lineno, raw in enumerate(f, start=1):
            if raw.strip():
                yield lineno, raw


class _LineCheck(NamedTuple):
    """One line's checks, as `_walk_day` found them."""

    lineno: int
    #: The parsed record; empty when the line is unreadable.
    record: Dict[str, Any]
    expected_prev: Any
    claimed_prev: Any
    claimed_hash: Any
    recomputed: Optional[str]
    link_holds: bool
    content_holds: bool
    #: Set when the content held only because restoring a non-finite input
    #: value reproduced the stored hash.
    non_finite_explanation: Optional[str]
    #: Why the line is not a record, when it is not one.
    unreadable: Optional[str] = None
    #: The unreadable line this record follows. Its link cannot be checked
    #: across it, so `link_holds` means nothing then.
    link_unchecked_after: Optional[int] = None


def _walk_day(path: Path, expected_prev_hash: Optional[str]) -> Iterator[_LineCheck]:
    """
    Every non-blank line of one day file with its link and content checked
    -- the walk `verify_audit_log_integrity` reports from and the signed
    checkpoint recomputes a day through, so the two cannot disagree about
    which records hold.

    `expected_prev_hash=None` takes the first record's own claim as the
    day's head, for a day the chain index never witnessed; the head is
    then unchecked and everything after it still is. A line that is not a
    record is yielded with `unreadable` set, and the walk continues.
    """
    prev_hash = expected_prev_hash
    first = True
    after_unreadable: Optional[int] = None
    for lineno, raw in _lines(path):
        record, reason = _read_line(raw)
        if record is None:
            yield _LineCheck(
                lineno=lineno,
                record={},
                expected_prev=prev_hash,
                claimed_prev=None,
                claimed_hash=None,
                recomputed=None,
                link_holds=False,
                content_holds=False,
                non_finite_explanation=None,
                unreadable=reason,
            )
            first = False
            prev_hash = None
            after_unreadable = lineno
            continue
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
            link_holds=after_unreadable is None and claimed_prev == prev_hash,
            content_holds=content_holds,
            non_finite_explanation=explanation,
            link_unchecked_after=after_unreadable,
        )
        after_unreadable = None
        prev_hash = claimed_hash or prev_hash


class _DayResult(NamedTuple):
    """One day file's verdict, as the trail check needs it."""

    problems: List[str]
    #: The subset of `problems` about lines that are not records.
    unreadable: List[str]
    notes: List[str]
    #: Non-blank lines, readable or not -- the day's record count.
    lines: int
    #: record_hash of the last non-blank line, when that line is a record.
    tail: Optional[str]
    #: Readable records carrying a record_hash field at all.
    hashed: int


def _check_day(path: Path, expected_prev_hash: Optional[str]) -> _DayResult:
    problems: List[str] = []
    unreadable: List[str] = []
    notes: List[str] = []
    lines = 0
    hashed = 0
    tail: Optional[str] = None
    for check in _walk_day(path, expected_prev_hash):
        lines += 1
        if check.unreadable is not None:
            message = (
                f"{path.name} line {check.lineno}: not a readable record "
                f"({check.unreadable}). The line was damaged or altered after "
                "it was written; its content cannot be checked, and neither "
                "can the link from the record after it."
            )
            problems.append(message)
            unreadable.append(message)
            tail = None
            continue
        request_id = check.record.get("request_id")
        if "record_hash" in check.record:
            hashed += 1
        if not check.link_holds and check.link_unchecked_after is None:
            problems.append(
                f"{path.name} line {check.lineno} (request_id={request_id}): "
                f"prev_record_hash={check.claimed_prev!r} does not match the "
                f"preceding record's hash {check.expected_prev!r} — chain broken "
                "(a record was edited, removed, reordered, or inserted)."
            )
        if not check.content_holds:
            problems.append(
                f"{path.name} line {check.lineno} (request_id={request_id}): "
                f"record_hash={check.claimed_hash!r} does not match its own "
                f"recomputed content hash {check.recomputed!r} — this line's "
                "content was altered after it was written."
            )
        elif check.non_finite_explanation is not None:
            notes.append(
                _non_finite_note(
                    path, check.lineno, check.record, check.non_finite_explanation
                )
            )
        tail = check.claimed_hash if isinstance(check.claimed_hash, str) else None
    return _DayResult(problems, unreadable, notes, lines, tail, hashed)


def _empty_head() -> Dict[str, Any]:
    return {
        "newest_date": None,
        "records": 0,
        "record_hash": None,
        "total_records": 0,
        "index_entries": None,
        "index_hash": None,
    }


def describe_head(head: Dict[str, Any]) -> str:
    """
    One sentence naming what a verification ran through -- the newest day,
    its record count and last record_hash, and the chain index's length and
    last hash -- for recording somewhere this directory cannot reach.

    That is the only anchor for the END of the trail short of a signed
    checkpoint: a newest day cut short is byte for byte an earlier state of
    the log, and deleting it with the index's last line leaves a shorter
    trail that verifies clean. A head written down elsewhere is what the
    next verification's head is compared with.
    """
    if not head.get("records") and head.get("newest_date") is None:
        return "Verified through: nothing -- there is no record to verify."
    where = head.get("newest_date") or "this file"
    sentence = (
        f"Verified through {where}: {head.get('records')} record(s), last "
        f"record_hash {head.get('record_hash')}"
    )
    if head.get("index_entries") is not None:
        sentence += (
            f"; {head.get('total_records')} record(s) in all; chain index "
            f"{head.get('index_entries')} entry(ies), last index_hash "
            f"{head.get('index_hash')}"
        )
    return (
        sentence + ". Keep this head somewhere this directory cannot reach: a "
        "newest day cut short, or deleted together with its chain index "
        "entry, verifies clean against the files alone."
    )


def verify_audit_log_integrity(
    path: Union[str, Path],
    expected_prev_hash: str = _GENESIS_HASH,
    notes: Optional[List[str]] = None,
    head: Optional[Dict[str, Any]] = None,
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

    A line that is not a record -- not UTF-8, not JSON, not a JSON object,
    cut off -- is reported as a problem where it is, and the rest of the
    file is still checked.

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
        head: when given, filled with what the check ran through (see
            `describe_head`): the file's record count and last record_hash.
    """
    path = Path(path)
    if head is not None:
        head.update(_empty_head())
    if not path.exists():
        return []
    day = _check_day(path, expected_prev_hash)
    if notes is not None:
        notes.extend(day.notes)
    if head is not None:
        head.update(
            newest_date=path.stem if _DAY_FILE_RE.match(path.name) else None,
            records=day.lines,
            record_hash=day.tail,
            total_records=day.lines,
        )
    return day.problems


class _Index(NamedTuple):
    #: Readable entries with their line numbers, in file order.
    entries: List[Tuple[int, Dict[str, Any]]]
    #: Non-blank lines, readable or not.
    lines: int
    #: index_hash of the last non-blank line, when that line is an entry.
    last_hash: Optional[str]


def _read_index(index_path: Path, problems: List[str]) -> _Index:
    """The chain index's entries, with its own hash chain checked. An
    unreadable line is a problem, names no date, and leaves the link from
    the entry after it uncheckable."""
    entries: List[Tuple[int, Dict[str, Any]]] = []
    lines = 0
    last_hash: Optional[str] = None
    if not index_path.exists():
        return _Index(entries, lines, last_hash)
    prev_index_hash: Optional[str] = _GENESIS_HASH
    for lineno, raw in _lines(index_path):
        lines += 1
        entry, reason = _read_line(raw)
        if entry is None:
            problems.append(
                f"chain index line {lineno}: not a readable entry ({reason}). "
                "The line was damaged or altered after it was written; the "
                "day it indexed is unknown, and the link from the entry after "
                "it cannot be checked."
            )
            prev_index_hash = None
            last_hash = None
            continue
        claimed_prev = entry.get("prev_index_hash")
        if prev_index_hash is not None and claimed_prev != prev_index_hash:
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
        last_hash = claimed_hash if isinstance(claimed_hash, str) else None
        entries.append((lineno, entry))
    return _Index(entries, lines, last_hash)


def verify_audit_trail_integrity(
    audit_dir: Optional[Union[str, Path]] = None,
    notes: Optional[List[str]] = None,
    head: Optional[Dict[str, Any]] = None,
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

    DAYS BEFORE THE INDEX'S EARLIEST ENTRY are opened too (see the
    CHANGELOG entry of 2026-09-28; they used to be skipped unread, so a file
    planted there verified clean). Each is checked on its own from the
    genesis hash, which is how days were chained before the index existed.
    When the index began, its first entry recorded where the trail stood --
    the last record_hash of the newest day before it, or the genesis hash
    when there was none -- so the newest earlier day must end exactly
    there, and when the index records that nothing came before it, a day
    file dated before it cannot have been there. Earlier days than the
    newest are reported in `notes` as unanchored. A day whose records
    predate the hash chain itself (no record carries a record_hash) has
    nothing to verify and is noted, not reported.

    A LINE THAT IS NOT A RECORD, in a day file or in the index, is reported
    where it is and the walk goes on; it used to abort the whole
    verification.

    A DAY INDEXED TWICE in a row from the same head is the trace of a first
    write that failed after its index entry was appended, which an earlier
    release could do; it is noted once and the day is verified against the
    later entry, rather than accused of re-chaining itself.

    THE NEWEST DAY cannot be anchored from inside this directory: cut short,
    or deleted with the index's last line, it is a valid earlier state of
    the trail. `head`, when given, is filled with what the verification ran
    through (see `describe_head`) so it can be recorded somewhere else.

    Returns a list of human-readable problems (empty if everything's clean,
    including the case where the audit directory doesn't exist yet). When
    `notes` is given it receives what was found and is not a problem, as in
    verify_audit_log_integrity.
    """
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    problems: List[str] = []
    found: List[str] = []
    index = _read_index(directory / _INDEX_FILENAME, problems)

    dated = [
        (lineno, entry)
        for lineno, entry in index.entries
        if isinstance(entry.get("date"), str) and entry.get("date")
    ]
    indexed_dates = {entry["date"] for _, entry in dated}
    day_files = _iter_day_files(directory)
    on_disk_dates = {p.stem for p in day_files}

    # Consecutive entries for one date are one day indexed more than once.
    groups: List[Tuple[str, List[Tuple[int, Dict[str, Any]]]]] = []
    for lineno, entry in dated:
        if groups and groups[-1][0] == entry["date"]:
            groups[-1][1].append((lineno, entry))
        else:
            groups.append((entry["date"], [(lineno, entry)]))

    for date in sorted(indexed_dates - on_disk_dates):
        last = max(i for i, (d, _) in enumerate(groups) if d == date)
        chain_head = groups[last][1][-1][1].get("chain_head")
        following = next((g for g in groups[last + 1 :] if g[0] != date), None)
        if following is not None and following[1][0][1].get("chain_head") == (
            chain_head
        ):
            problems.append(
                f"chain index attests to activity on {date}, but {date}.jsonl "
                f"does not exist, and the next indexed day, {following[0]}, "
                "continues from the same head that entry records. Either no "
                f"record was ever written on {date} -- an earlier release "
                "wrote a day's index entry before its first record, so a "
                "first write that failed left the entry behind -- or "
                f"{date}.jsonl was deleted before {following[0]} began."
            )
        else:
            problems.append(
                f"chain index attests to activity on {date}, but {date}.jsonl "
                "no longer exists on disk — likely deleted."
            )

    results: Dict[str, _DayResult] = {}
    earliest = min(indexed_dates) if indexed_dates else None

    if earliest is not None:
        for date in sorted(
            d for d in on_disk_dates if d >= earliest and d not in indexed_dates
        ):
            problems.append(
                f"{date}.jsonl exists on disk with no corresponding chain "
                "index entry (the index entry may have been removed, or "
                "this file was created outside the normal write path)."
            )
            day = _check_day(directory / f"{date}.jsonl", None)
            results[date] = day
            problems.extend(day.problems)
            found.extend(day.notes)

    # Days before the index began.
    before = [p for p in day_files if earliest is None or p.stem < earliest]
    for day_path in before:
        day = _check_day(day_path, _GENESIS_HASH)
        results[day_path.stem] = day
        if day.lines and not day.hashed:
            problems.extend(day.unreadable)
            found.append(
                f"{day_path.name} predates the hash chain: none of its "
                f"{day.lines} record(s) carries a record_hash, so there is "
                "nothing in it to verify."
            )
            continue
        problems.extend(day.problems)
        found.extend(day.notes)
    if before and earliest is not None:
        anchor = next(entry for _, entry in dated if entry["date"] == earliest)
        first_head = anchor.get("chain_head")
        if first_head == _GENESIS_HASH:
            for day_path in before:
                if results[day_path.stem].hashed:
                    problems.append(
                        f"{day_path.name} is dated before the chain index's "
                        f"first entry ({earliest}), and that entry records "
                        "that nothing came before it (its chain_head is the "
                        "genesis hash). A day that existed when the index "
                        "began would be where it starts, so this file "
                        "appeared afterwards: history added in front of the "
                        "trail, or a file copied in from elsewhere."
                    )
        else:
            newest_before = before[-1]
            tail = results[newest_before.stem].tail
            if tail != first_head:
                problems.append(
                    f"the chain index's first entry ({earliest}) records that "
                    f"the trail continued from {first_head}, but "
                    f"{newest_before.name}, the last day before it, ends at "
                    f"{tail} — a day before the index was altered, cut short "
                    "or replaced after the index began."
                )
            unanchored = [p.name for p in before[:-1] if results[p.stem].hashed]
            if unanchored:
                found.append(
                    f"{len(unanchored)} day file(s) before {newest_before.name} "
                    f"({unanchored[0]} .. {unanchored[-1]}) predate the chain "
                    "index and are not anchored by it: each was checked on its "
                    "own, but nothing outside them records where it ended."
                )

    prev_date: Optional[str] = None
    prev_tail: Optional[str] = None
    for date, group in groups:
        entry = group[-1][1]
        if len(group) > 1:
            where = ", ".join(str(lineno) for lineno, _ in group)
            if len({repr(e.get("chain_head")) for _, e in group}) == 1:
                found.append(
                    f"chain index lines {where} all index {date} from the same "
                    f"head {entry.get('chain_head')}: the first write of that "
                    "day failed after its index entry was written (an earlier "
                    "release wrote the entry before the record), and the next "
                    "write indexed the day again. No record is missing; the "
                    "day is verified once, against the last of them."
                )
            else:
                problems.append(
                    f"chain index lines {where} index {date} more than once, "
                    "from different heads — the index was edited, or the "
                    "previous day changed after this one began. The day is "
                    "verified against the last of them."
                )
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
        day = _check_day(day_path, expected_head)
        results[date] = day
        problems.extend(day.problems)
        found.extend(day.notes)
        prev_date, prev_tail = date, day.tail

    if notes is not None:
        notes.extend(found)
    if head is not None:
        head.update(_empty_head())
        head["index_entries"] = index.lines
        head["index_hash"] = index.last_hash
        head["total_records"] = sum(results[p.stem].lines for p in day_files)
        if day_files:
            newest = results[day_files[-1].stem]
            head["newest_date"] = day_files[-1].stem
            head["records"] = newest.lines
            head["record_hash"] = newest.tail
    return problems
