"""Legal hold, retention-based garbage collection, and read-only sealing --
the operational side of running the audit trail long-term without either
growing unbounded or losing evidence carelessly. Nothing here runs
automatically: deletion only ever happens via an explicit `gc(dry_run=False)`
call, never as a side effect of writing or verifying records."""

import json
import logging
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Union

from .paths import _audit_dir, _iter_day_files

logger = logging.getLogger(__name__)


def _hold_path(date: str, directory: Path) -> Path:
    return directory / f"{date}.jsonl.hold"


def hold_day(
    date: str,
    audit_dir: Optional[Union[str, Path]] = None,
    reason: Optional[str] = None,
) -> Path:
    """
    Place a legal/retention hold on a calendar day (`date`, "YYYY-MM-DD"),
    protecting it from `gc()` regardless of `retention_days`. Idempotent —
    holding an already-held day just overwrites the reason/timestamp. The
    hold sidecar (`<date>.jsonl.hold`) is written even if that day's audit
    file doesn't exist yet (e.g. placing a hold ahead of expected activity),
    since a hold is a statement about a date, not a file.
    """
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    directory.mkdir(parents=True, exist_ok=True)
    hold_path = _hold_path(date, directory)
    payload = {
        "date": date,
        "reason": reason,
        "held_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(hold_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return hold_path


def release_hold(date: str, audit_dir: Optional[Union[str, Path]] = None) -> bool:
    """Remove a hold on `date`, if one exists. Returns whether a hold was
    actually removed (False if the day was never held)."""
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    hold_path = _hold_path(date, directory)
    if not hold_path.exists():
        return False
    hold_path.unlink()
    return True


def is_held(date: str, audit_dir: Optional[Union[str, Path]] = None) -> bool:
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    return _hold_path(date, directory).exists()


def _retention_days_from_env() -> Optional[int]:
    raw = os.environ.get("SQT_AUDIT_RETENTION_DAYS")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("[audit] SQT_AUDIT_RETENTION_DAYS=%r is not an integer", raw)
        return None


def gc_candidates(
    audit_dir: Optional[Union[str, Path]] = None,
    retention_days: Optional[int] = None,
) -> List[str]:
    """
    Calendar dates (day-file stems) eligible for deletion under a retention
    policy: on-disk day files strictly older than `retention_days`,
    excluding any date currently under a legal hold (see `hold_day`).

    `retention_days=None` (the default) reads `SQT_AUDIT_RETENTION_DAYS`;
    if that's unset too, returns `[]` — no retention window configured
    means nothing is ever a deletion candidate, not "delete everything."
    """
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    days = retention_days if retention_days is not None else _retention_days_from_env()
    if days is None:
        return []
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return sorted(
        p.stem
        for p in _iter_day_files(directory)
        if p.stem < cutoff and not is_held(p.stem, directory)
    )


def gc(
    audit_dir: Optional[Union[str, Path]] = None,
    retention_days: Optional[int] = None,
    dry_run: bool = True,
) -> List[str]:
    """
    Delete day files past the retention window (see `gc_candidates`).
    `dry_run=True` (the default) previews candidates without deleting
    anything. Deletion (`dry_run=False`) is never triggered automatically
    anywhere in this library — only through an explicit call to this
    function (e.g. from `sqt gc --confirm`).

    Note: a deleted day file is a real, permanent deletion. After it,
    `verify_audit_trail_integrity()`/`sqt verify` will correctly report
    that date as "likely deleted" — the hash chain has no way to
    distinguish a legitimate, policy-driven deletion from tampering, since
    both leave the same evidence (a missing file the index still attests
    to). Export a bundle (`export_bundle`) for anything you may need to
    produce later *before* running `gc(dry_run=False)` on it.
    """
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    candidates = gc_candidates(directory, retention_days)
    if dry_run:
        return candidates
    deleted: List[str] = []
    for date in candidates:
        path = directory / f"{date}.jsonl"
        try:
            path.unlink()
            deleted.append(date)
        except FileNotFoundError:
            pass
    return deleted


def seal_day(date: str, audit_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Mark a calendar day's audit file read-only via `os.chmod`, signaling
    it's closed and shouldn't be written to again. This is explicitly
    **not** WORM: anyone with sufficient OS-level privilege can chmod the
    file writable again, and on Windows `os.chmod` can only toggle the
    read-only attribute (no separate owner/group/other bits). It's a
    deployer-scheduled operational safeguard against *accidental*
    modification — e.g. a nightly cron calling `sqt seal <yesterday>` —
    not something this library does automatically at day rollover, since
    rollover timing isn't guaranteed to coincide with a running process.

    Only the day file itself is sealed. The chain index (`_chain_index.jsonl`)
    stays writable, since it's shared across every day including future
    ones — there's no meaningful way to seal "this date's entries" within a
    single shared append-only file.

    Raises:
        FileNotFoundError: no audit file exists for `date`.
    """
    directory = Path(audit_dir) if audit_dir else _audit_dir()
    day_path = directory / f"{date}.jsonl"
    if not day_path.exists():
        raise FileNotFoundError(f"No audit file for {date} in {directory}")
    os.chmod(day_path, stat.S_IREAD)
    return day_path
