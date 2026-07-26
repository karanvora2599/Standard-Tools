"""Hash-chain tamper-evidence verification: a single day file in isolation
(`verify_audit_log_integrity`), or the full cross-day trail -- the chain
index's own chain plus every day file it attests to
(`verify_audit_trail_integrity`)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .hashing import hash_payload
from .paths import _GENESIS_HASH, _INDEX_FILENAME, _audit_dir, _iter_day_files


def verify_audit_log_integrity(
    path: Union[str, Path], expected_prev_hash: str = _GENESIS_HASH
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
    """
    path = Path(path)
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
                    f"line {lineno} (request_id={record.get('request_id')}): "
                    f"prev_record_hash={claimed_prev!r} does not match the "
                    f"preceding record's hash {prev_hash!r} — chain broken "
                    "(a record was edited, removed, reordered, or inserted)."
                )
            recomputed = hash_payload({**record, "record_hash": None})
            claimed_hash = record.get("record_hash")
            if recomputed != claimed_hash:
                problems.append(
                    f"line {lineno} (request_id={record.get('request_id')}): "
                    f"record_hash={claimed_hash!r} does not match its own "
                    f"recomputed content hash {recomputed!r} — this line's "
                    "content was altered after it was written."
                )
            prev_hash = claimed_hash or prev_hash
    return problems


def verify_audit_trail_integrity(
    audit_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Verify the FULL cross-day audit trail, not just one file: the
    independent chain-index witness log's own hash chain, that every day
    file the index attests to still exists on disk (and the reverse — a day
    file present with no matching index entry, for any date at or after the
    index's earliest entry), and each day file's own internal record chain
    seeded with the chain head the index claims for that day — so a
    wholesale-regenerated day file with a fabricated-but-internally-
    consistent chain is still caught, which verify_audit_log_integrity(path)
    alone (with its default genesis-hash assumption) cannot detect.

    Days before the chain index's earliest entry (audit activity that
    predates this feature, or an audit directory with no index at all) are
    NOT cross-day-linked, by design — retroactively rewriting old records to
    link them in would itself be indistinguishable from tampering. Verify
    those individually with verify_audit_log_integrity(path) instead.

    Returns a list of human-readable problems (empty if everything's clean,
    including the case where the audit directory doesn't exist yet).
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

    for entry in index_entries:
        date = entry.get("date")
        if date not in on_disk_dates:
            continue  # already reported above
        day_path = directory / f"{date}.jsonl"
        expected_head = entry.get("chain_head", _GENESIS_HASH)
        problems.extend(
            verify_audit_log_integrity(day_path, expected_prev_hash=expected_head)
        )

    return problems
