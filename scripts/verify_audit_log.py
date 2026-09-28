#!/usr/bin/env python3
"""
Standalone, dependency-free verifier for the standard_quant_tools audit
trail. Deliberately does NOT import the standard_quant_tools package (or
any third-party library — only the Python standard library) so an external
auditor can run this against an exported log bundle without installing the
project or its dependencies:

    python verify_audit_log.py <audit_dir>              # full cross-day trail
    python verify_audit_log.py --file <path.jsonl>       # one day file only

Exit code 0 = clean, 1 = one or more problems found (printed to stdout).
Either way the last line printed names the head the check ran through --
the newest day, its record count and last record_hash, and the chain
index's length and last hash. Keep it somewhere the audited directory
cannot reach: a newest day cut short, or deleted with its index entry,
verifies clean against the files alone, and a head recorded elsewhere is
what shows it.

This is a deliberate reimplementation, not an import, of the equivalent
logic in the standard_quant_tools.audit package (hashing.hash_payload,
verify.verify_audit_log_integrity, verify.verify_audit_trail_integrity).
That is a known duplication-by-design maintenance risk: any future change
to those functions' behavior — especially hash_payload's canonicalization —
must be mirrored here, or this script will silently disagree with the real
library about what counts as tampered. tests/audit/test_standalone_verifier.py
is the parity check that catches that drift; if you change the library's
hashing or verification, run that test before assuming this script still
agrees with it.
"""

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

_GENESIS_HASH = "0" * 16
_INDEX_FILENAME = "_chain_index.jsonl"
_DAY_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")


def hash_payload(obj: Any) -> str:
    """Must stay byte-for-byte identical to audit.hash_payload's
    canonicalization — see tests/audit/test_standalone_verifier.py."""
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _iter_day_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.jsonl") if _DAY_FILE_RE.match(p.name))


# Lines written by a writer that hashed a non-finite input value (NaN,
# +inf, -inf) and then wrote it as null can never reproduce their stored
# hash, although nobody touched them. A line is accepted as such only when
# restoring a non-finite value in place of some nulls inside `input`
# reproduces the stored hash exactly -- an edit cannot pass without a second
# preimage of the hash. The search is bounded per line. Mirrors
# audit.verify._non_finite_explanation exactly, bounds and order included.
_EXPLAIN_MAX_TRIALS = 4096
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


# A line that is not a record -- not UTF-8, not JSON, not a JSON object --
# is reported where it is and the walk goes on, so one damaged line cannot
# hide what comes after it. Lines are split on b"\n" alone, in binary.
# Mirrors audit.verify._read_line and _lines exactly, wording included.
_JSON_KINDS = {list: "array", str: "string", int: "number", float: "number"}


def _json_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "true/false"
    if value is None:
        return "null"
    return _JSON_KINDS.get(type(value), type(value).__name__)


def _read_line(raw: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
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
    with open(path, "rb") as f:
        for lineno, raw in enumerate(f, start=1):
            if raw.strip():
                yield lineno, raw


class _DayResult(NamedTuple):
    problems: List[str]
    unreadable: List[str]
    notes: List[str]
    lines: int
    tail: Optional[str]
    hashed: int


def _check_day(path: Path, expected_prev_hash: Optional[str]) -> _DayResult:
    """One day file's records, each link and content checked. Mirrors
    audit.verify._walk_day and _check_day exactly."""
    problems: List[str] = []
    unreadable: List[str] = []
    notes: List[str] = []
    lines = 0
    hashed = 0
    tail: Optional[str] = None
    prev_hash = expected_prev_hash
    first = True
    after_unreadable: Optional[int] = None
    for lineno, raw in _lines(path):
        lines += 1
        record, reason = _read_line(raw)
        if record is None:
            message = (
                f"{path.name} line {lineno}: not a readable record "
                f"({reason}). The line was damaged or altered after "
                "it was written; its content cannot be checked, and neither "
                "can the link from the record after it."
            )
            problems.append(message)
            unreadable.append(message)
            tail = None
            first = False
            prev_hash = None
            after_unreadable = lineno
            continue
        claimed_prev = record.get("prev_record_hash")
        if first and prev_hash is None:
            prev_hash = claimed_prev
        first = False
        request_id = record.get("request_id")
        if "record_hash" in record:
            hashed += 1
        if after_unreadable is None and claimed_prev != prev_hash:
            problems.append(
                f"{path.name} line {lineno} (request_id={request_id}): "
                f"prev_record_hash={claimed_prev!r} does not match the "
                f"preceding record's hash {prev_hash!r} — chain broken "
                "(a record was edited, removed, reordered, or inserted)."
            )
        recomputed = hash_payload({**record, "record_hash": None})
        claimed_hash = record.get("record_hash")
        if recomputed != claimed_hash:
            explanation = _non_finite_explanation(record)
            if explanation is None:
                problems.append(
                    f"{path.name} line {lineno} (request_id={request_id}): "
                    f"record_hash={claimed_hash!r} does not match its own "
                    f"recomputed content hash {recomputed!r} — this line's "
                    "content was altered after it was written."
                )
            else:
                notes.append(_non_finite_note(path, lineno, record, explanation))
        after_unreadable = None
        prev_hash = claimed_hash or prev_hash
        tail = claimed_hash if isinstance(claimed_hash, str) else None
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
    """One sentence naming what a verification ran through. Mirrors
    audit.verify.describe_head exactly."""
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


def verify_log_file(
    path: Path,
    expected_prev_hash: str = _GENESIS_HASH,
    notes: Optional[List[str]] = None,
    head: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Verify one day file's internal hash chain in isolation. Mirrors
    audit.verify_audit_log_integrity exactly, `notes` and `head` included."""
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
    entries: List[Tuple[int, Dict[str, Any]]]
    lines: int
    last_hash: Optional[str]


def _read_index(index_path: Path, problems: List[str]) -> _Index:
    """Mirrors audit.verify._read_index exactly."""
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


def verify_trail(
    directory: Path,
    notes: Optional[List[str]] = None,
    head: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Verify the full cross-day trail: the chain index's own hash chain,
    that every day file the index attests to still exists (and vice versa),
    each day file's internal chain seeded with the index's claimed starting
    point, each day's ending hash against the next indexed day's recorded
    chain head, and the days before the index began against where its first
    entry says the trail stood. A line that is not a record is reported and
    the walk goes on; a day indexed twice from the same head is noted once.
    Mirrors audit.verify_audit_trail_integrity exactly, `notes` and `head`
    included."""
    directory = Path(directory)
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone hash-chain verifier for a standard_quant_tools "
        "audit trail. Stdlib-only, no project install required."
    )
    parser.add_argument(
        "audit_dir",
        nargs="?",
        default=None,
        help="Root directory of the audit trail (contains *.jsonl day files "
        "and _chain_index.jsonl). Ignored if --file is given.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Verify a single day's .jsonl in isolation instead of the full "
        "cross-day trail.",
    )
    args = parser.parse_args(argv)

    notes: List[str] = []
    head: Dict[str, Any] = {}
    if args.file is not None:
        problems = verify_log_file(args.file, notes=notes, head=head)
    elif args.audit_dir is not None:
        problems = verify_trail(Path(args.audit_dir), notes=notes, head=head)
    else:
        parser.error("either audit_dir or --file is required")  # exits the process

    if not problems:
        print("OK — no integrity problems found.")
    else:
        print(f"{len(problems)} problem(s) found:")
        for p in problems:
            print(f"  - {p}")
    if notes:
        print(f"{len(notes)} note(s), not problems:")
        for n in notes:
            print(f"  - {n}")
    print(describe_head(head))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
