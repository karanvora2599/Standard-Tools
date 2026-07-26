"""AuditWriter: the append-only, hash-chained, fsync'd JSONL writer -- one
file per UTC day, plus the cross-day chain-index witness log that links
each new day's first record onto the previous active day's last hash.

Storage is delegated to a pluggable `AuditStorageBackend` (default:
`LocalFilesystemBackend`) -- AuditWriter owns the chain-hashing and lock
sequencing, the backend owns the actual read/append/lock primitives for
whatever medium it targets."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .hashing import hash_payload
from .models import DecisionRecord
from .paths import _GENESIS_HASH, _INDEX_FILENAME, _audit_dir
from .storage import AuditStorageBackend, LocalFilesystemBackend


class AuditWriter:
    """Append-only JSONL writer, one file per UTC day."""

    def __init__(
        self,
        audit_dir: Optional[Union[str, Path]] = None,
        backend: Optional[AuditStorageBackend] = None,
    ):
        self._dir = Path(audit_dir) if audit_dir else _audit_dir()
        self._backend: AuditStorageBackend = (
            backend if backend is not None else LocalFilesystemBackend()
        )

    def _path_for(self, when: datetime) -> Path:
        return self._dir / f"{when.strftime('%Y-%m-%d')}.jsonl"

    def _last_line(self, path: Path) -> Optional[str]:
        last_line: Optional[str] = None
        for line in self._backend.read_lines(path):
            if line.strip():
                last_line = line
        return last_line

    def _last_record_hash_in_file(self, path: Path) -> Optional[str]:
        """Hash of the last line in `path`, or None if the file doesn't
        exist or has no valid lines. Must be called while the relevant write
        lock is held, since it establishes a chain link a new record commits
        to."""
        last_line = self._last_line(path)
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
        last_line = self._last_line(index_path)
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
        candidates = sorted(
            stem
            for stem in self._backend.list_day_stems(self._dir)
            if stem < day_path.stem
        )
        if not candidates:
            return _GENESIS_HASH
        last_path = self._dir / f"{candidates[-1]}.jsonl"
        return self._last_record_hash_in_file(last_path) or _GENESIS_HASH

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
        ilf = self._backend.acquire_lock(index_path)
        try:
            chain_head = self._chain_head_before(day_path)
            entry: Dict[str, Any] = {
                "date": day_path.stem,
                "chain_head": chain_head,
                "prev_index_hash": self._last_index_hash(index_path),
                "index_hash": None,
            }
            entry["index_hash"] = hash_payload(entry)
            self._backend.append_line(index_path, json.dumps(entry, sort_keys=True))
        finally:
            self._backend.release_lock(ilf)
        return chain_head

    def write(self, record: DecisionRecord) -> Path:
        when = datetime.now(timezone.utc)
        path = self._path_for(when)

        # Day-lock is always acquired before the index-lock taken (only)
        # inside _bootstrap_new_day — a fixed lock order, so this can never
        # deadlock against a concurrent writer doing the same thing.
        lf = self._backend.acquire_lock(path)
        try:
            is_new_day_file = not self._backend.exists(path)
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
            self._backend.append_line(path, record.model_dump_json())
        finally:
            self._backend.release_lock(lf)
        return path
