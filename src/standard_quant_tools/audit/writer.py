"""AuditWriter: the append-only, hash-chained, fsync'd JSONL writer -- one
file per UTC day, plus the cross-day chain-index witness log that links
each new day's first record onto the previous active day's last hash."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .hashing import hash_payload
from .models import DecisionRecord
from .paths import (
    _GENESIS_HASH,
    _INDEX_FILENAME,
    _acquire_lock,
    _audit_dir,
    _iter_day_files,
    _release_lock,
)


class AuditWriter:
    """Append-only JSONL writer, one file per UTC day."""

    def __init__(self, audit_dir: Optional[Union[str, Path]] = None):
        self._dir = Path(audit_dir) if audit_dir else _audit_dir()

    def _path_for(self, when: datetime) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        return self._dir / f"{when.strftime('%Y-%m-%d')}.jsonl"

    def _last_record_hash_in_file(self, path: Path) -> Optional[str]:
        """Hash of the last line in `path`, or None if the file doesn't
        exist or has no valid lines. Must be called while the relevant write
        lock is held, since it establishes a chain link a new record commits
        to."""
        if not path.exists():
            return None
        last_line: Optional[str] = None
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line is None:
            return None
        try:
            return json.loads(last_line).get("record_hash")
        except Exception:
            return None

    def _last_index_hash(self, index_path: Path) -> str:
        """Hash of the last line in the chain index, or the genesis hash if
        it doesn't exist/is empty. Must be called while the index lock is
        held."""
        if not index_path.exists():
            return _GENESIS_HASH
        last_line: Optional[str] = None
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line is None:
            return _GENESIS_HASH
        try:
            return json.loads(last_line).get("index_hash") or _GENESIS_HASH
        except Exception:
            return _GENESIS_HASH

    def _chain_head_before(self, day_path: Path) -> str:
        """The record_hash a NEW day file's first record should chain onto:
        the last record_hash of the most recent existing day file strictly
        before `day_path`, or the genesis hash if there is no earlier day
        file (this is the very first day the audit trail has ever seen
        activity)."""
        candidates = [p for p in _iter_day_files(self._dir) if p.name < day_path.name]
        if not candidates:
            return _GENESIS_HASH
        return self._last_record_hash_in_file(candidates[-1]) or _GENESIS_HASH

    def _bootstrap_new_day(self, day_path: Path) -> str:
        """
        Called once, immediately before the first record of a new calendar
        day's file is written. Computes the chain head this new day should
        link onto and records that linkage in the independent chain-index
        witness log (itself hash-chained) BEFORE the day file gains its
        first record, so the index and the day file can be cross-checked
        against each other later (verify_audit_trail_integrity) — an
        attacker who deletes/regenerates a day file now also has to rewrite
        a second, independent artifact to hide it.

        Returns the chain head so the caller can commit to it as the new
        day's first record's prev_record_hash.
        """
        index_path = self._dir / _INDEX_FILENAME
        index_lock_path = index_path.with_name(index_path.name + ".lock")
        ilf = _acquire_lock(index_lock_path)
        try:
            chain_head = self._chain_head_before(day_path)
            entry: Dict[str, Any] = {
                "date": day_path.stem,
                "chain_head": chain_head,
                "prev_index_hash": self._last_index_hash(index_path),
                "index_hash": None,
            }
            entry["index_hash"] = hash_payload(entry)
            with open(index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())
        finally:
            _release_lock(ilf)
        return chain_head

    def write(self, record: DecisionRecord) -> Path:
        when = datetime.now(timezone.utc)
        path = self._path_for(when)
        lock_path = path.with_name(path.name + ".lock")

        # Day-lock is always acquired before the index-lock taken (only)
        # inside _bootstrap_new_day — a fixed lock order, so this can never
        # deadlock against a concurrent writer doing the same thing.
        lf = _acquire_lock(lock_path)
        try:
            is_new_day_file = not path.exists()
            if is_new_day_file:
                record.prev_record_hash = self._bootstrap_new_day(path)
            else:
                record.prev_record_hash = (
                    self._last_record_hash_in_file(path) or _GENESIS_HASH
                )
            # Hash over the record with record_hash itself left unset, so
            # the chain link (prev_record_hash) and the record's own content
            # are both covered without the field hashing itself.
            record.record_hash = hash_payload(
                {**record.model_dump(exclude={"record_hash"}), "record_hash": None}
            )
            with open(path, "a", encoding="utf-8") as f:
                f.write(record.model_dump_json() + "\n")
                f.flush()
                os.fsync(f.fileno())
        finally:
            _release_lock(lf)
        return path
