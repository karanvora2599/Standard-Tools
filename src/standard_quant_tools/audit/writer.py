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
from typing import Any, Callable, Dict, Optional, Union

from standard_quant_tools.error import AuditIntegrityError

from .hashing import hash_payload
from .json_native import to_json_native
from .models import DecisionRecord
from .paths import _GENESIS_HASH, _INDEX_FILENAME, _audit_dir
from .storage import AuditStorageBackend, LastLine, LocalFilesystemBackend

#: How a parsed JSON value that is not an object is named in a refusal.
_JSON_KINDS = {list: "array", str: "string", int: "number", float: "number"}


def _json_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "true/false"
    if value is None:
        return "null"
    return _JSON_KINDS.get(type(value), type(value).__name__)


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

    def _last_line(self, path: Path) -> LastLine:
        """The last non-blank line of `path`, through the backend.

        A backend that can read a file backwards (`read_last_line`) is asked
        for the tail alone; one that cannot is read line by line, which is
        what every backend did before and stays correct, only slower.
        """
        read_last_line = getattr(self._backend, "read_last_line", None)
        if callable(read_last_line):
            return read_last_line(path)
        lines = self._backend.read_lines(path)
        if not lines:
            return LastLine(None, True, 0)
        terminated = lines[-1].endswith("\n")
        for line in reversed(lines):
            if line.strip():
                return LastLine(line.strip(), terminated, None)
        return LastLine(None, terminated, None)

    def _read_tail(self, path: Path, what: str) -> Optional[Dict[str, Any]]:
        """
        The last entry of `path` as a JSON object, or None when the file is
        missing or holds nothing. Must be called while the lock over `path`
        is held, since the answer becomes a chain link.

        An UNREADABLE last line raises. It used to return None, which the
        callers turned into the genesis hash — so a corrupted tail made the
        writer silently START A NEW CHAIN and keep appending as though the
        trail had just begun:

            valid record
            valid record
            CORRUPTED LINE
            next tool call -> prev_record_hash = GENESIS

        "The file does not exist yet" and "the file exists and I cannot
        read its tail" are completely different states: the first is a
        legitimate genesis, the second means the trail is already damaged.
        Continuing to extend a damaged chain produces a tamper-evident log
        that is no longer evidence of anything, which is worse than refusing
        the write.

        Two kinds of unreadable tail are told apart, because they have
        different remedies. A last line with no newline after it that does
        not parse is a record a crash or a full disk cut off mid-write: the
        refusal says where the fragment starts, so the file can be cut back
        to its last complete record. A last line that does parse but has
        lost only its newline is a complete record, and is NOT refused --
        the backend writes the missing newline before the next line.
        """
        try:
            tail = self._last_line(path)
        except ValueError as exc:  # UnicodeDecodeError: bytes that are not text
            raise AuditIntegrityError(
                f"audit chain is corrupt: the last line of {path} is not "
                f"UTF-8 text ({exc}). Refusing to append — extending a chain "
                "whose tail cannot be read would restart it and destroy the "
                "trail's evidential value. Copy the file somewhere safe, "
                "remove the damaged line, and run `sqt verify`."
            ) from exc
        if tail.text is None:
            return None
        try:
            parsed = json.loads(tail.text)
        except ValueError as exc:
            if not tail.terminated:
                where = (
                    f" It starts at byte {tail.offset}."
                    if tail.offset is not None
                    else ""
                )
                raise AuditIntegrityError(
                    f"audit chain is corrupt: the last line of {path} was cut "
                    f"off mid-{what} — it has no newline and does not parse "
                    f"({exc}), which is what a crash or a full disk during a "
                    f"write leaves behind.{where} Refusing to append, because "
                    "nothing can chain onto a fragment. To continue: copy the "
                    "file somewhere safe, truncate it to the byte where the "
                    "fragment starts (the end of the last complete line), and "
                    "run `sqt verify`. The fragment was never a whole "
                    f"{what}, so no complete {what} is lost by removing it."
                ) from exc
            raise AuditIntegrityError(
                f"audit chain is corrupt: the last line of {path} is not valid "
                f"JSON ({exc}). Refusing to append — extending a chain whose "
                "tail cannot be read would silently restart it from genesis "
                "and destroy the trail's evidential value. Repair or archive "
                "this file before continuing; `sqt verify` names the line."
            ) from exc
        if not isinstance(parsed, dict):
            raise AuditIntegrityError(
                f"audit chain is corrupt: the last line of {path} is a JSON "
                f"{_json_kind(parsed)}, not a {what}. Refusing to append. "
                "Repair or archive this file before continuing; `sqt verify` "
                "names the line."
            )
        return parsed

    def _last_record_hash_in_file(self, path: Path) -> Optional[str]:
        """
        Hash of the last record in `path`, or None if the file doesn't exist
        or is empty. Must be called while the relevant write lock is held,
        since it establishes a chain link a new record commits to. An
        unreadable last line raises (see `_read_tail`).
        """
        parsed = self._read_tail(path, "record")
        if parsed is None:
            return None
        record_hash = parsed.get("record_hash")
        if not record_hash:
            raise AuditIntegrityError(
                f"audit chain is corrupt: the last record in {path} carries no "
                "record_hash, so the next record has nothing to chain onto. "
                "Refusing to append."
            )
        return record_hash

    def _last_index_entry(self, index_path: Path) -> Optional[Dict[str, Any]]:
        """
        The chain index's last entry, or None if it doesn't exist/is empty.
        Must be called while the index lock is held.

        Same fail-closed rule as the day file: an unreadable index tail is
        corruption, not a fresh start. The index is the independent witness
        that makes a deleted day file detectable, so silently re-genesising
        it removes the second artifact an attacker would otherwise have to
        forge.
        """
        entry = self._read_tail(index_path, "entry")
        if entry is not None and not entry.get("index_hash"):
            raise AuditIntegrityError(
                f"audit chain index is corrupt: the last entry in {index_path} "
                "carries no index_hash. Refusing to append."
            )
        return entry

    def _last_index_hash(self, index_path: Path) -> str:
        """Hash of the chain index's last entry, or the genesis hash when it
        has none. Must be called while the index lock is held."""
        entry = self._last_index_entry(index_path)
        return entry["index_hash"] if entry is not None else _GENESIS_HASH

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
        # The PREVIOUS day's lock is held while its tail is read. Without it
        # there is a cross-midnight race:
        #
        #   23:59:59  writer A holds yesterday's lock, about to append
        #   00:00:00  writer B creates today's file and reads yesterday's
        #             tail — seeing the record BEFORE A's append
        #   00:00:01  writer A appends
        #
        # Today's first record would then chain onto a record that is no
        # longer yesterday's last one, forking the chain at the day boundary
        # while every individual record still verified.
        #
        # Lock ordering is safe: write() already holds TODAY's day lock, and
        # this only ever reaches BACKWARDS to a strictly earlier stem, so
        # every holder acquires locks in the same (newest -> older)
        # direction and no cycle can form.
        prev_lock = self._backend.acquire_lock(last_path)
        try:
            return self._last_record_hash_in_file(last_path) or _GENESIS_HASH
        finally:
            self._backend.release_lock(prev_lock)

    def _bootstrap_new_day(
        self,
        day_path: Path,
        prepare: Optional[Callable[[str], Any]] = None,
    ) -> str:
        """
        Called once, immediately before the first record of a new calendar
        day's file is written. Computes the chain head this new day should
        link onto and records that linkage in the independent chain-index
        witness log (itself hash-chained) BEFORE the day file gains its
        first record, so the index and the day file can be cross-checked
        against each other later (verify_audit_trail_integrity) — an
        attacker who deletes/regenerates a day file now also has to rewrite
        a second, independent artifact to hide it.

        `prepare`, when given, is called with the chain head BEFORE the
        index is touched; the writer builds and serialises the day's first
        record there. If it raises, the index is left exactly as it was.
        The index entry used to be appended first, so a first record that
        could not be serialised left an entry for a day with no file: the
        trail reported the day deleted, and the retry indexed the day a
        second time and was then accused of re-chaining it, for good.

        IDEMPOTENT. When the index's last entry is already this date -- a
        first write that failed after its entry was appended, by an earlier
        release or by a crash between the two appends -- that entry's head
        is reused and nothing is appended, so a day is indexed once.

        Returns the chain head so the caller can commit to it as the new
        day's first record's prev_record_hash.
        """
        index_path = self._dir / _INDEX_FILENAME
        ilf = self._backend.acquire_lock(index_path)
        try:
            last = self._last_index_entry(index_path)
            if (
                last is not None
                and last.get("date") == day_path.stem
                and isinstance(last.get("chain_head"), str)
            ):
                chain_head: str = last["chain_head"]
                if prepare is not None:
                    prepare(chain_head)
                return chain_head
            chain_head = self._chain_head_before(day_path)
            if prepare is not None:
                prepare(chain_head)
            entry: Dict[str, Any] = {
                "date": day_path.stem,
                "chain_head": chain_head,
                "prev_index_hash": (
                    last["index_hash"] if last is not None else _GENESIS_HASH
                ),
                "index_hash": None,
            }
            entry["index_hash"] = hash_payload(entry)
            self._backend.append_line(index_path, json.dumps(entry, sort_keys=True))
        finally:
            self._backend.release_lock(ilf)
        return chain_head

    @staticmethod
    def _serialised(record: DecisionRecord) -> str:
        """Hash `record` over the line it will be written as, and return
        that line.

        The hash is taken over the line AS IT WILL BE READ BACK, not over
        the live objects the line was made from. The verifier hashes
        `json.loads(line)`, so the two agree by construction: hashing the
        objects let any value the JSON writer spells differently (a NaN
        written as null, a timestamp, a set, an integer key) leave a record
        whose stored hash could never be reproduced -- a day reported as
        tampered for ever. For a record already made of JSON-native values
        the parsed form equals the objects, so its hash is bit-identical to
        the old rule and every day file on disk verifies exactly as before.
        See the CHANGELOG entry of 2026-09-27.
        """
        # Hash over the record with record_hash itself left unset, so the
        # chain link (prev_record_hash) and the record's own content are
        # both covered without the field hashing itself.
        payload = json.loads(record.model_dump_json(exclude={"record_hash"}))
        record.record_hash = hash_payload({**payload, "record_hash": None})
        return record.model_dump_json()

    def write(self, record: DecisionRecord) -> Path:
        # The recorded values are made JSON-native before anything is
        # hashed or written, whoever built the record: a numpy value in a
        # free-form input made the JSON writer raise, and the record was
        # dropped. `dispatch` normalises the input already; doing it again
        # here costs nothing (the conversion is idempotent) and covers a
        # record written directly through this class.
        record.input = to_json_native(record.input)
        record.data_sources = to_json_native(record.data_sources)

        when = datetime.now(timezone.utc)
        path = self._path_for(when)

        # Day-lock is always acquired before the index-lock taken (only)
        # inside _bootstrap_new_day — a fixed lock order, so this can never
        # deadlock against a concurrent writer doing the same thing.
        lf = self._backend.acquire_lock(path)
        try:
            line: Optional[str] = None
            if not self._backend.exists(path):

                def _prepare(chain_head: str) -> None:
                    nonlocal line
                    record.prev_record_hash = chain_head
                    line = self._serialised(record)

                self._bootstrap_new_day(path, prepare=_prepare)
            else:
                record.prev_record_hash = (
                    self._last_record_hash_in_file(path) or _GENESIS_HASH
                )
                line = self._serialised(record)
            assert line is not None
            self._backend.append_line(path, line)
        finally:
            self._backend.release_lock(lf)
        return path
