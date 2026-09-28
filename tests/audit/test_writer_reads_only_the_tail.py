"""
The writer reads a day's last line and nothing else, repairs a lost
newline, still refuses a torn record, and never indexes a day it did not
write.

Every append used to read the WHOLE day file to find its last line, under
the lock every other writer waits on -- half a second and as much memory
per tool call on a 95 MiB day. A complete record whose newline was lost was
glued to the next one, producing a line that is neither record: the
verifier could not read it and every later write refused. And the index
entry for a new day was appended before the day's first record was built,
so a first write that failed left the day indexed with no file, and the
retry indexed it again. See the CHANGELOG entry of 2026-09-28.
"""

import json
import random
from pathlib import Path
from typing import List, Optional

import pytest

from standard_quant_tools import audit
from standard_quant_tools.audit import storage as storage_module
from standard_quant_tools.audit.storage import LocalFilesystemBackend
from standard_quant_tools.audit.writer import AuditWriter
from standard_quant_tools.error import AuditIntegrityError


def _record(i: int) -> "audit.DecisionRecord":
    return audit.DecisionRecord(
        request_id=f"r{i}",
        timestamp_utc="2024-01-01T00:00:00+00:00",
        tool_name=f"t{i}",
        input={"i": i},
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )


def _old_last_line(path: Path) -> Optional[str]:
    """How the writer found the last line before: every line, read."""
    last: Optional[str] = None
    for line in LocalFilesystemBackend().read_lines(path):
        if line.strip():
            last = line
    return last.strip() if last is not None else None


def _index_lines(directory: Path) -> List[dict]:
    index = directory / audit._INDEX_FILENAME
    if not index.exists():
        return []
    return [json.loads(x) for x in index.read_text("utf-8").splitlines() if x.strip()]


class TestTheTailIsReadBackwards:
    @pytest.mark.parametrize(
        "content",
        [
            b'{"a": 1}\n{"b": 2}\n',
            b'{"a": 1}\r\n{"b": 2}\r\n',
            b'{"a": 1}\n{"b": 2}\n\n  \n',
            b'{"a": 1}\n{"b": 2}',
            b'{"a": 1}\n{"b": 2',
            b'{"only": true}\n',
            b'{"only": true}',
            b"",
            b"\n  \n\r\n",
            b'{"a": 1}\n' + b'{"long": "' + b"x" * 200_000 + b'"}\n',
            '{"a": 1}\n{"name": "Zoë – ✓"}\n'.encode("utf-8"),
        ],
        ids=[
            "LF",
            "CRLF",
            "trailing blank lines",
            "no final newline",
            "torn final line",
            "one line",
            "one line without newline",
            "empty",
            "only blanks",
            "a line longer than one block",
            "non-ASCII",
        ],
    )
    def test_it_finds_what_reading_every_line_found(self, tmp_path, content):
        path = tmp_path / "day.jsonl"
        path.write_bytes(content)

        found = LocalFilesystemBackend().read_last_line(path)

        assert found.text == _old_last_line(path)
        assert found.terminated == (not content or content.endswith(b"\n"))
        if found.text is not None:
            assert path.read_bytes()[found.offset :].strip().decode() == found.text

    def test_random_trails_agree_with_the_old_reading(self, tmp_path):
        rng = random.Random(20260928)
        for case in range(200):
            lines = [
                json.dumps({"n": rng.randint(0, 10**6), "s": "y" * rng.randint(0, 300)})
                for _ in range(rng.randint(0, 6))
            ]
            ending = rng.choice(["\n", "\r\n"])
            text = ending.join(lines)
            if lines and rng.random() < 0.7:
                text += ending
            text += rng.choice(["", "\n", "  \n", "\r\n\r\n"])
            path = tmp_path / f"case{case}.jsonl"
            path.write_bytes(text.encode("utf-8"))

            assert LocalFilesystemBackend().read_last_line(path).text == (
                _old_last_line(path)
            ), text

    def test_a_missing_file_has_no_tail(self, tmp_path):
        found = LocalFilesystemBackend().read_last_line(tmp_path / "absent.jsonl")
        assert found.text is None

    def test_the_writer_never_reads_a_whole_day(self, tmp_path, monkeypatch):
        """The writer used `read_lines` -- every line of the day -- to find
        the last one, on every call."""
        writer = AuditWriter(audit_dir=tmp_path)
        writer.write(_record(1))

        def _whole_file(self, path):
            raise AssertionError(f"read every line of {path}")

        monkeypatch.setattr(LocalFilesystemBackend, "read_lines", _whole_file)
        writer.write(_record(2))
        writer.write(_record(3))

        assert audit.verify_audit_trail_integrity(tmp_path) == []

    def test_a_large_day_costs_the_length_of_its_last_line(self, tmp_path, monkeypatch):
        path = tmp_path / "big.jsonl"
        with open(path, "wb") as f:
            for i in range(20_000):
                f.write(json.dumps({"i": i, "pad": "z" * 400}).encode() + b"\n")
        assert path.stat().st_size > 8 * 1024 * 1024

        read = [0]
        real_open = open

        class _Counting:
            def __init__(self, handle):
                self._handle = handle

            def read(self, *args):
                data = self._handle.read(*args)
                read[0] += len(data)
                return data

            def __getattr__(self, name):
                return getattr(self._handle, name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._handle.__exit__(*exc)

        monkeypatch.setattr(
            storage_module,
            "open",
            lambda *a, **k: _Counting(real_open(*a, **k)),
            raising=False,
        )
        found = LocalFilesystemBackend().read_last_line(path)

        assert json.loads(found.text)["i"] == 19_999
        assert read[0] < 256 * 1024, read[0]


class TestALostNewlineIsRepaired:
    def _written_day(self, tmp_path: Path) -> Path:
        writer = AuditWriter(audit_dir=tmp_path)
        return Path(writer.write(_record(1)))

    @pytest.mark.parametrize("strip", [b"\r\n", b"\n"], ids=["whole ending", "LF only"])
    def test_the_next_record_starts_its_own_line(self, tmp_path, strip):
        """The next record was glued onto the complete one before it."""
        day = self._written_day(tmp_path)
        data = day.read_bytes()
        ending = b"\r\n" if data.endswith(b"\r\n") else b"\n"
        cut = data[: -len(ending)] if strip == b"\r\n" else data[:-1]
        day.write_bytes(cut)

        AuditWriter(audit_dir=tmp_path).write(_record(2))

        lines = [x for x in day.read_bytes().splitlines() if x.strip()]
        assert len(lines) == 2
        assert [json.loads(x)["request_id"] for x in lines] == ["r1", "r2"]
        assert audit.verify_audit_trail_integrity(tmp_path) == []

    def test_an_intact_day_gains_no_blank_line(self, tmp_path):
        """Null case: a terminated day is appended to exactly as before."""
        day = self._written_day(tmp_path)
        AuditWriter(audit_dir=tmp_path).write(_record(2))
        data = day.read_bytes()
        assert b"\n\n" not in data and b"\r\n\r\n" not in data
        assert len(data.splitlines()) == 2


class TestATornRecordIsStillRefused:
    def test_a_cut_off_record_is_refused_with_where_to_cut(self, tmp_path):
        writer = AuditWriter(audit_dir=tmp_path)
        day = Path(writer.write(_record(1)))
        whole = day.stat().st_size
        with open(day, "ab") as f:
            f.write(b'{"request_id": "r2", "record_ha')

        with pytest.raises(AuditIntegrityError) as refused:
            writer.write(_record(3))
        message = str(refused.value)
        assert "cut off" in message and f"byte {whole}" in message
        assert "truncate" in message

        # The remedy the message names works.
        with open(day, "r+b") as f:
            f.truncate(whole)
        writer.write(_record(3))
        assert audit.verify_audit_trail_integrity(tmp_path) == []

    @pytest.mark.parametrize(
        "tail",
        [b"[1, 2, 3]\n", b"42\n", b'{"request_id": "\xff\xfe"}\n'],
        ids=["an array", "a number", "not UTF-8"],
    )
    def test_a_tail_that_is_not_a_record_is_refused_as_corruption(self, tmp_path, tail):
        """An array or a bare number used to escape as AttributeError, and
        bytes that are not UTF-8 as UnicodeDecodeError -- both of which the
        dispatch wrapper swallows as an ordinary write failure."""
        writer = AuditWriter(audit_dir=tmp_path)
        day = Path(writer.write(_record(1)))
        with open(day, "ab") as f:
            f.write(tail)

        with pytest.raises(AuditIntegrityError, match="corrupt"):
            writer.write(_record(2))


class TestAFailedFirstWriteIndexesNothing:
    def test_a_first_record_that_cannot_be_built_leaves_the_index_alone(
        self, tmp_path, monkeypatch
    ):
        """The index entry was appended before the record was serialised,
        so this left a day indexed with no file -- reported as deleted, and
        indexed again by the next write."""
        real = audit.DecisionRecord.model_dump_json
        calls = []

        def _fails_once(self, *args, **kwargs):
            calls.append(self.request_id)
            if len(calls) == 1:
                raise ValueError("this record cannot be written")
            return real(self, *args, **kwargs)

        monkeypatch.setattr(audit.DecisionRecord, "model_dump_json", _fails_once)
        writer = AuditWriter(audit_dir=tmp_path)

        with pytest.raises(ValueError):
            writer.write(_record(1))
        assert _index_lines(tmp_path) == []
        assert audit._iter_day_files(tmp_path) == []

        writer.write(_record(2))
        assert len(_index_lines(tmp_path)) == 1
        assert audit.verify_audit_trail_integrity(tmp_path) == []

    def test_an_entry_already_left_for_today_is_reused(self, tmp_path):
        """The state an earlier release left behind: today's entry is the
        index's last line and today's file does not exist. The next write
        chains from that entry instead of indexing the day a second time."""
        writer = AuditWriter(audit_dir=tmp_path)
        day = Path(writer.write(_record(1)))
        entry = _index_lines(tmp_path)[-1]
        day.unlink()

        writer.write(_record(2))

        assert _index_lines(tmp_path) == [entry]
        first = json.loads(day.read_text(encoding="utf-8").splitlines()[0])
        assert first["prev_record_hash"] == entry["chain_head"]
        assert audit.verify_audit_trail_integrity(tmp_path) == []

    def test_an_ordinary_first_write_indexes_the_day_once(self, tmp_path):
        """Null case."""
        writer = AuditWriter(audit_dir=tmp_path)
        writer.write(_record(1))
        writer.write(_record(2))

        assert len(_index_lines(tmp_path)) == 1
        assert audit.verify_audit_trail_integrity(tmp_path) == []
