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

This is a deliberate reimplementation, not an import, of the equivalent
logic in src/standard_quant_tools/audit/ (hashing.hash_payload,
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
from typing import Any, Dict, Iterator, List, Optional, Tuple

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


def verify_log_file(
    path: Path,
    expected_prev_hash: str = _GENESIS_HASH,
    notes: Optional[List[str]] = None,
) -> List[str]:
    """Verify one day file's internal hash chain in isolation. Mirrors
    audit.verify_audit_log_integrity exactly, `notes` included."""
    if not path.exists():
        return []
    problems: List[str] = []
    prev_hash = expected_prev_hash
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            claimed_prev = record.get("prev_record_hash")
            if claimed_prev != prev_hash:
                problems.append(
                    f"{path.name} line {lineno} (request_id="
                    f"{record.get('request_id')}): prev_record_hash="
                    f"{claimed_prev!r} does not match the preceding record's "
                    f"hash {prev_hash!r} — chain broken (a record was "
                    "edited, removed, reordered, or inserted)."
                )
            recomputed = hash_payload({**record, "record_hash": None})
            claimed_hash = record.get("record_hash")
            if recomputed != claimed_hash:
                explanation = _non_finite_explanation(record)
                if explanation is None:
                    problems.append(
                        f"{path.name} line {lineno} (request_id="
                        f"{record.get('request_id')}): record_hash="
                        f"{claimed_hash!r} does not match its own recomputed "
                        f"content hash {recomputed!r} — this line's content "
                        "was altered after it was written."
                    )
                elif notes is not None:
                    notes.append(_non_finite_note(path, lineno, record, explanation))
            prev_hash = claimed_hash or prev_hash
    return problems


def _last_record_hash(path: Path) -> Optional[str]:
    """The record_hash on a day file's last non-blank line -- the chain head
    the next indexed day has to claim. None when the file is missing, empty
    or its last line cannot be read as a record. Mirrors
    audit.verify._last_record_hash exactly."""
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


def verify_trail(directory: Path, notes: Optional[List[str]] = None) -> List[str]:
    """Verify the full cross-day trail: the chain index's own hash chain,
    that every day file the index attests to still exists (and vice versa),
    each day file's internal chain seeded with the index's claimed starting
    point, and each day's ending hash against the next indexed day's
    recorded chain head -- a day rewritten and re-chained from the head the
    index publishes is internally consistent and correctly seeded, so its
    tail is the only thing that gives it away. Mirrors
    audit.verify_audit_trail_integrity exactly."""
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
            verify_log_file(day_path, expected_prev_hash=expected_head, notes=notes)
        )
        prev_date, prev_tail = date, _last_record_hash(day_path)

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
    if args.file is not None:
        problems = verify_log_file(args.file, notes=notes)
    elif args.audit_dir is not None:
        problems = verify_trail(Path(args.audit_dir), notes=notes)
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
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
