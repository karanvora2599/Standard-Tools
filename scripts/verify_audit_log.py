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
logic in src/standard_quant_tools/audit.py (hash_payload,
verify_audit_log_integrity, verify_audit_trail_integrity). That is a known
duplication-by-design maintenance risk: any future change to those
functions' behavior — especially hash_payload's canonicalization — must be
mirrored here, or this script will silently disagree with the real library
about what counts as tampered. tests/test_standalone_verifier.py is the
parity check that catches that drift; if you change hash_payload in
audit.py, run that test before assuming this script still agrees with it.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_GENESIS_HASH = "0" * 16
_INDEX_FILENAME = "_chain_index.jsonl"
_DAY_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")


def hash_payload(obj: Any) -> str:
    """Must stay byte-for-byte identical to audit.hash_payload's
    canonicalization — see tests/test_standalone_verifier.py."""
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _iter_day_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.glob("*.jsonl") if _DAY_FILE_RE.match(p.name))


def verify_log_file(path: Path, expected_prev_hash: str = _GENESIS_HASH) -> List[str]:
    """Verify one day file's internal hash chain in isolation. Mirrors
    audit.verify_audit_log_integrity exactly."""
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
                problems.append(
                    f"{path.name} line {lineno} (request_id="
                    f"{record.get('request_id')}): record_hash="
                    f"{claimed_hash!r} does not match its own recomputed "
                    f"content hash {recomputed!r} — this line's content was "
                    "altered after it was written."
                )
            prev_hash = claimed_hash or prev_hash
    return problems


def verify_trail(directory: Path) -> List[str]:
    """Verify the full cross-day trail: the chain index's own hash chain,
    that every day file the index attests to still exists (and vice versa),
    and each day file's internal chain seeded with the index's claimed
    starting point. Mirrors audit.verify_audit_trail_integrity exactly."""
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

    for entry in index_entries:
        date = entry.get("date")
        if date not in on_disk_dates:
            continue  # already reported above
        day_path = directory / f"{date}.jsonl"
        expected_head = entry.get("chain_head", _GENESIS_HASH)
        problems.extend(verify_log_file(day_path, expected_prev_hash=expected_head))

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

    if args.file is not None:
        problems = verify_log_file(args.file)
    elif args.audit_dir is not None:
        problems = verify_trail(Path(args.audit_dir))
    else:
        parser.error("either audit_dir or --file is required")  # exits the process

    if not problems:
        print("OK — no integrity problems found.")
        return 0

    print(f"{len(problems)} problem(s) found:")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
