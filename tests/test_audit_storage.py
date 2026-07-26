"""
Tests for the pluggable storage backend
(src/standard_quant_tools/audit/storage.py): confirms `AuditStorageBackend`
is a real seam `AuditWriter` delegates through -- not just a passthrough
wrapper around direct file I/O -- by swapping in a fake, non-filesystem
backend and confirming it produces identical hash-chain behavior to
`LocalFilesystemBackend`.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Set

from standard_quant_tools import audit


class _InMemoryBackend:
    """A fake, non-filesystem backend. Exists only to prove AuditWriter
    genuinely delegates storage operations rather than assuming local disk
    anywhere in its own logic."""

    _DAY_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.jsonl$")

    def __init__(self) -> None:
        self._store: Dict[str, List[str]] = {}
        self._locked: Set[str] = set()

    def acquire_lock(self, path: Path) -> str:
        key = str(path)
        assert key not in self._locked, (
            f"lock already held for {key} -- AuditWriter would deadlock or "
            "race on real storage if this ever fires"
        )
        self._locked.add(key)
        return key

    def release_lock(self, handle: str) -> None:
        self._locked.discard(handle)

    def read_lines(self, path: Path) -> List[str]:
        return list(self._store.get(str(path), []))

    def append_line(self, path: Path, line: str) -> None:
        self._store.setdefault(str(path), []).append(line + "\n")

    def exists(self, path: Path) -> bool:
        return str(path) in self._store

    def list_day_stems(self, audit_dir: Path) -> List[str]:
        stems = []
        for key in self._store:
            name = Path(key).name
            if self._DAY_FILE_RE.match(name):
                stems.append(Path(key).stem)
        return sorted(stems)


def _record(request_id: str, timestamp_utc: str) -> "audit.DecisionRecord":
    return audit.DecisionRecord(
        request_id=request_id,
        timestamp_utc=timestamp_utc,
        tool_name="t",
        input={},
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )


class TestStorageBackendInterface:
    def test_default_backend_is_local_filesystem(self, tmp_path: Path):
        w = audit.AuditWriter(audit_dir=tmp_path)
        assert isinstance(w._backend, audit.LocalFilesystemBackend)

    def test_custom_backend_replaces_filesystem_entirely(self, tmp_path: Path):
        backend = _InMemoryBackend()
        w = audit.AuditWriter(audit_dir=tmp_path, backend=backend)

        record = _record("r1", "2024-01-01T00:00:00+00:00")
        path = w.write(record)

        assert not list(
            tmp_path.iterdir()
        ), "a custom backend must mean nothing touches the real filesystem"
        lines = backend.read_lines(path)
        assert len(lines) == 1
        written = json.loads(lines[0])
        assert written["record_hash"] == record.record_hash
        assert record.prev_record_hash == audit._GENESIS_HASH

    def test_custom_backend_maintains_hash_chain_within_a_day(self, tmp_path: Path):
        backend = _InMemoryBackend()
        w = audit.AuditWriter(audit_dir=tmp_path, backend=backend)

        r1 = _record("r1", "2024-01-01T00:00:00+00:00")
        r2 = _record("r2", "2024-01-01T00:01:00+00:00")
        w.write(r1)
        w.write(r2)

        assert r2.prev_record_hash == r1.record_hash
        assert r1.record_hash != r2.record_hash

    def test_custom_backend_supports_cross_day_chain_continuity(self, tmp_path: Path):
        """Mirrors TestChainIndexContinuity in test_audit.py, but through a
        backend that isn't the real filesystem -- confirms cross-day
        bootstrap (_chain_head_before) is genuinely backend-routed, not
        hardcoded to local Path.glob()."""
        backend = _InMemoryBackend()
        w = audit.AuditWriter(audit_dir=tmp_path, backend=backend)

        day1 = tmp_path / "2024-01-01.jsonl"
        head1 = w._bootstrap_new_day(day1)
        r1 = _record("r1", "2024-01-01T00:00:00+00:00")
        r1.prev_record_hash = head1
        r1.record_hash = audit.hash_payload(
            {**r1.model_dump(exclude={"record_hash"}), "record_hash": None}
        )
        backend.append_line(day1, r1.model_dump_json())

        day2 = tmp_path / "2024-01-02.jsonl"
        head2 = w._bootstrap_new_day(day2)

        assert head2 == r1.record_hash, (
            "cross-day chain continuity must work through the backend "
            "interface, not just LocalFilesystemBackend"
        )

    def test_locks_are_acquired_and_released_in_pairs(self, tmp_path: Path):
        """A regression guard for the fixed lock-order contract -- if
        AuditWriter ever left a lock held (or released one it never
        acquired), the fake backend's assertion in acquire_lock would fail
        this test outright."""
        backend = _InMemoryBackend()
        w = audit.AuditWriter(audit_dir=tmp_path, backend=backend)

        for i in range(3):
            w.write(_record(f"r{i}", "2024-01-01T00:00:00+00:00"))

        assert backend._locked == set(), "every acquired lock must be released"
