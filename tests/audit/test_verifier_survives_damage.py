"""
Both verifiers read every line on its own, report the trail's head, open
the days before the chain index, and recognise a day indexed twice.

A line that is not a record used to raise out of the verifier, so a single
junk byte in one day hid every finding after it: an attacker who altered
day 3 only had to damage day 2. Days dated before the chain index's first
entry were never opened, so history planted in front of the trail verified
clean. The newest day could be cut short with nothing to show for it, and
nothing said what a clean verification had actually covered. A first write
that failed after its index entry left the day indexed twice, and the
second entry was then reported as a re-chained day on every run.

Every case is run through the library and through the stdlib-only script
an auditor's bundle carries, and the two must agree word for word --
problems, notes and head. See the CHANGELOG entry of 2026-09-28.
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple

import pytest

from standard_quant_tools import audit, cli
from standard_quant_tools.audit.writer import AuditWriter

from .. import REPO_ROOT

DAYS = ("2024-03-01", "2024-03-02", "2024-03-03")


@pytest.fixture(scope="module")
def standalone() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_audit_log_standalone_damage",
        REPO_ROOT / "scripts" / "verify_audit_log.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sealed(record: "audit.DecisionRecord") -> "audit.DecisionRecord":
    payload = json.loads(record.model_dump_json(exclude={"record_hash"}))
    record.record_hash = audit.hash_payload({**payload, "record_hash": None})
    return record


def _record(name: str, prev: str, **inputs: Any) -> "audit.DecisionRecord":
    record = audit.DecisionRecord(
        request_id=f"r-{name}",
        timestamp_utc="2024-03-01T00:00:00+00:00",
        tool_name=name,
        input=inputs or {"quantity": 200},
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )
    record.prev_record_hash = prev
    return _sealed(record)


def _write_day(
    directory: Path, date: str, n: int = 2, newline: str = "\n"
) -> List["audit.DecisionRecord"]:
    """One day bootstrapped through the real writer, carrying `n` records."""
    day = directory / f"{date}.jsonl"
    prev = AuditWriter(audit_dir=directory)._bootstrap_new_day(day)
    records = []
    for i in range(n):
        record = _record(f"{date}-{i}", prev)
        prev = record.record_hash
        records.append(record)
    with open(day, "w", encoding="utf-8", newline="") as f:
        f.write("".join(r.model_dump_json() + newline for r in records))
    return records


def _write_trail(directory: Path, newline: str = "\n") -> None:
    for date in DAYS:
        _write_day(directory, date, newline=newline)


def _both(
    standalone: ModuleType, directory: Path
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """Problems, notes and head -- asserting the two verifiers agree."""
    notes: List[str] = []
    head: Dict[str, Any] = {}
    problems = audit.verify_audit_trail_integrity(directory, notes=notes, head=head)
    mirrored_notes: List[str] = []
    mirrored_head: Dict[str, Any] = {}
    mirrored = standalone.verify_trail(
        directory, notes=mirrored_notes, head=mirrored_head
    )
    assert mirrored == problems
    assert mirrored_notes == notes
    assert mirrored_head == head
    return problems, notes, head


def _alter_day_three(directory: Path) -> None:
    """A real edit: one payload changed, the stored hashes left behind."""
    day = directory / f"{DAYS[2]}.jsonl"
    lines = day.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["input"]["quantity"] = 999999
    lines[1] = json.dumps(record)
    day.write_text("\n".join(lines) + "\n", encoding="utf-8")


JUNK = {
    "not json": b"{not json\n",
    "a JSON array": b"[1,2,3]\n",
    "a JSON string": b'"just a string"\n',
    "an invalid utf-8 byte": b'{"request_id": "\xff"}\n',
    "a cut-off line": b'{"request_id": "r-x", "record_hash": "ab\n',
}


class TestOneDamagedLineHidesNothingElse:
    @pytest.mark.parametrize("junk", list(JUNK.values()), ids=list(JUNK))
    def test_the_damage_and_the_later_edit_are_both_reported(
        self, standalone, tmp_path, junk
    ):
        """This raised out of both verifiers before anything about day 3
        was said."""
        _write_trail(tmp_path)
        _alter_day_three(tmp_path)
        with open(tmp_path / f"{DAYS[1]}.jsonl", "ab") as f:
            f.write(junk)

        problems, _, _ = _both(standalone, tmp_path)

        assert any(
            f"{DAYS[1]}.jsonl line 3: not a readable record" in p for p in problems
        ), problems
        assert any(
            f"{DAYS[2]}.jsonl line 2" in p and "altered" in p for p in problems
        ), problems

    def test_a_damaged_line_in_the_middle_does_not_break_the_next_link(
        self, standalone, tmp_path
    ):
        """The record after a damaged line cannot have its link checked
        across it -- the damage is the finding, not a second "chain broken"."""
        _write_trail(tmp_path)
        day = tmp_path / f"{DAYS[1]}.jsonl"
        lines = day.read_bytes().split(b"\n")
        lines.insert(1, b"{garbage")
        day.write_bytes(b"\n".join(lines))

        problems, _, _ = _both(standalone, tmp_path)

        assert len(problems) == 1, problems
        assert "line 2: not a readable record" in problems[0]

    def test_a_damaged_chain_index_line_is_reported_and_the_walk_goes_on(
        self, standalone, tmp_path
    ):
        _write_trail(tmp_path)
        _alter_day_three(tmp_path)
        index = tmp_path / audit._INDEX_FILENAME
        with open(index, "ab") as f:
            f.write(b"[1]\n")

        problems, _, _ = _both(standalone, tmp_path)

        assert any(
            "chain index line 4: not a readable entry" in p for p in problems
        ), problems
        assert any(f"{DAYS[2]}.jsonl line 2" in p for p in problems), problems

    def test_sqt_verify_exits_one_and_names_both_without_a_traceback(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path))
        _write_trail(tmp_path)
        _alter_day_three(tmp_path)
        with open(tmp_path / f"{DAYS[1]}.jsonl", "ab") as f:
            f.write(b"[1,2,3]\n")

        assert cli.main(["verify"]) == 1
        out = capsys.readouterr().out
        assert "not a readable record" in out and "altered" in out

    def test_the_standalone_script_exits_one_on_the_same_trail(
        self, standalone, tmp_path, capsys
    ):
        _write_trail(tmp_path)
        with open(tmp_path / f"{DAYS[0]}.jsonl", "ab") as f:
            f.write(b"{not json\n")

        assert standalone.main([str(tmp_path)]) == 1
        assert "not a readable record" in capsys.readouterr().out

    @pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["LF", "CRLF"])
    def test_a_clean_trail_reports_nothing_in_either(
        self, standalone, tmp_path, newline
    ):
        """Null case: reading in binary and splitting on the newline alone
        reads a CRLF trail exactly as it reads an LF one."""
        _write_trail(tmp_path, newline=newline)

        problems, notes, _ = _both(standalone, tmp_path)

        assert problems == [] and notes == []

    def test_a_single_day_file_survives_a_damaged_line_too(self, standalone, tmp_path):
        records = _write_day(tmp_path, DAYS[0], n=3)
        day = tmp_path / f"{DAYS[0]}.jsonl"
        with open(day, "ab") as f:
            f.write(b"\x00\x01garbage\n")

        problems = audit.verify_audit_log_integrity(day)

        assert problems == standalone.verify_log_file(day)
        assert len(problems) == 1 and "line 4" in problems[0]
        assert records  # the three real records raised no problem


class TestTheHeadIsReported:
    def test_an_untouched_trail_reports_its_true_head(self, standalone, tmp_path):
        _write_trail(tmp_path)
        last = json.loads(
            (tmp_path / f"{DAYS[2]}.jsonl").read_text(encoding="utf-8").splitlines()[-1]
        )
        index_lines = (
            (tmp_path / audit._INDEX_FILENAME).read_text(encoding="utf-8").splitlines()
        )

        problems, _, head = _both(standalone, tmp_path)

        assert problems == []
        assert head == {
            "newest_date": DAYS[2],
            "records": 2,
            "record_hash": last["record_hash"],
            "total_records": 6,
            "index_entries": 3,
            "index_hash": json.loads(index_lines[-1])["index_hash"],
        }

    @pytest.mark.parametrize(
        "cut",
        ["truncate the newest day", "delete it and its index line", "rewrite it"],
    )
    def test_what_the_files_cannot_show_the_head_does(self, standalone, tmp_path, cut):
        """None of these is a problem the files can show -- a shorter trail
        is a valid earlier state of the log. The head each verification
        reports is what differs, which is why it is reported: recorded
        elsewhere, it is compared with the next one."""
        _write_trail(tmp_path)
        _, _, before = _both(standalone, tmp_path)
        newest = tmp_path / f"{DAYS[2]}.jsonl"
        if cut == "truncate the newest day":
            lines = newest.read_text(encoding="utf-8").splitlines()
            newest.write_text(lines[0] + "\n", encoding="utf-8")
        elif cut == "delete it and its index line":
            newest.unlink()
            index = tmp_path / audit._INDEX_FILENAME
            lines = index.read_text(encoding="utf-8").splitlines()
            index.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        else:
            head = json.loads(
                (tmp_path / audit._INDEX_FILENAME)
                .read_text(encoding="utf-8")
                .splitlines()[-1]
            )["chain_head"]
            forged = _record("forged", head, quantity=1)
            newest.write_text(forged.model_dump_json() + "\n", encoding="utf-8")

        problems, _, after = _both(standalone, tmp_path)

        assert problems == []
        assert after != before

    def test_the_cli_and_the_script_print_the_head_last(
        self, standalone, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path))
        _write_trail(tmp_path)

        assert cli.main(["verify"]) == 0
        cli_last = capsys.readouterr().out.strip().splitlines()[-1]
        assert standalone.main([str(tmp_path)]) == 0
        script_last = capsys.readouterr().out.strip().splitlines()[-1]

        assert cli_last == script_last
        assert cli_last.startswith(f"Verified through {DAYS[2]}: 2 record(s)")
        assert "3 entry(ies)" in cli_last

    def test_a_single_file_reports_its_own_head(self, standalone, tmp_path):
        records = _write_day(tmp_path, DAYS[0], n=3)
        day = tmp_path / f"{DAYS[0]}.jsonl"
        head: Dict[str, Any] = {}
        mirrored: Dict[str, Any] = {}

        audit.verify_audit_log_integrity(day, head=head)
        standalone.verify_log_file(day, head=mirrored)

        assert head == mirrored
        assert head["records"] == 3
        assert head["record_hash"] == records[-1].record_hash
        assert head["index_entries"] is None


class TestHistoryAddedInFront:
    def _plant_before(self, directory: Path, name: str = "2020-01-01.jsonl") -> None:
        planted = _record("planted", audit._GENESIS_HASH)
        (directory / name).write_text(planted.model_dump_json() + "\n", "utf-8")

    def test_a_day_planted_before_an_index_that_records_nothing_before_it(
        self, standalone, tmp_path
    ):
        """The index's first entry chains from the genesis hash, so nothing
        existed before it; a day file dated earlier appeared afterwards.
        This verified clean -- the file was never opened."""
        _write_trail(tmp_path)
        self._plant_before(tmp_path)

        problems, _, _ = _both(standalone, tmp_path)

        assert len(problems) == 1, problems
        assert "2020-01-01.jsonl is dated before the chain index's first entry" in (
            problems[0]
        )

    def test_junk_planted_before_the_index_is_opened_and_reported(
        self, standalone, tmp_path
    ):
        _write_trail(tmp_path)
        (tmp_path / "2020-01-01.jsonl").write_text(
            '{"request_id":"FABRICATED","record_hash":"nonsense"}\n{not json at all\n',
            encoding="utf-8",
        )

        problems, _, _ = _both(standalone, tmp_path)

        assert any("not a readable record" in p for p in problems), problems
        assert any("dated before the chain index" in p for p in problems), problems

    def test_the_last_day_before_the_index_must_end_where_the_index_began(
        self, standalone, tmp_path
    ):
        """A genuine upgrade: the earlier day existed first, and the index's
        first entry recorded its tail. Cutting that day short afterwards is
        now visible -- the index said where it ended."""
        earlier = tmp_path / "2023-12-31.jsonl"
        first = _record("older-0", audit._GENESIS_HASH)
        second = _record("older-1", first.record_hash)
        earlier.write_text(
            first.model_dump_json() + "\n" + second.model_dump_json() + "\n",
            encoding="utf-8",
        )
        _write_trail(tmp_path)
        assert _both(standalone, tmp_path)[0] == []

        earlier.write_text(first.model_dump_json() + "\n", encoding="utf-8")
        problems, _, _ = _both(standalone, tmp_path)

        assert len(problems) == 1, problems
        assert "2023-12-31.jsonl, the last day before it, ends at" in problems[0]

    def test_older_days_before_the_index_are_noted_as_unanchored(
        self, standalone, tmp_path
    ):
        for name in ("2023-12-30", "2023-12-31"):
            record = _record(name, audit._GENESIS_HASH)
            (tmp_path / f"{name}.jsonl").write_text(
                record.model_dump_json() + "\n", encoding="utf-8"
            )
        _write_trail(tmp_path)

        problems, notes, _ = _both(standalone, tmp_path)

        assert problems == []
        assert any("2023-12-30.jsonl" in n and "not anchored" in n for n in notes)

    def test_days_from_before_the_hash_chain_are_noted_not_reported(
        self, standalone, tmp_path
    ):
        """Records written before the chain existed carry no record_hash at
        all; there is nothing in them to verify, and an index begun after
        them chained from genesis because of it. (The file is placed after
        the trail is written only because today's writer refuses to chain
        onto a record without a hash; the index's first head is genesis
        either way.)"""
        _write_trail(tmp_path)
        (tmp_path / "2023-01-01.jsonl").write_text(
            json.dumps({"request_id": "old", "tool_name": "t"}) + "\n",
            encoding="utf-8",
        )

        problems, notes, _ = _both(standalone, tmp_path)

        assert problems == []
        assert any("predates the hash chain" in n for n in notes)


class TestADayIndexedTwice:
    def _index_entries(self, directory: Path) -> List[Dict[str, Any]]:
        index = directory / audit._INDEX_FILENAME
        return [json.loads(x) for x in index.read_text("utf-8").splitlines()]

    def _append_index_entry(self, directory: Path, date: str, head: str) -> None:
        """What an earlier release left when a day's first write failed
        after its index entry: an entry, chained into the index properly."""
        entries = self._index_entries(directory)
        entry = {
            "date": date,
            "chain_head": head,
            "prev_index_hash": entries[-1]["index_hash"],
            "index_hash": None,
        }
        entry["index_hash"] = audit.hash_payload(entry)
        with open(directory / audit._INDEX_FILENAME, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

    def test_the_same_day_indexed_twice_from_one_head_is_a_note(
        self, standalone, tmp_path
    ):
        """The retry indexed the day again. That used to be reported as the
        day re-chaining itself, on every verification for good."""
        _write_day(tmp_path, DAYS[0])
        day = tmp_path / f"{DAYS[1]}.jsonl"
        head = AuditWriter(audit_dir=tmp_path)._bootstrap_new_day(day)
        self._append_index_entry(tmp_path, DAYS[1], head)
        record = _record("retry", head)
        day.write_text(record.model_dump_json() + "\n", encoding="utf-8")

        problems, notes, _ = _both(standalone, tmp_path)

        assert problems == []
        assert len(notes) == 1 and "lines 2, 3 all index" in notes[0]

    def test_the_same_day_indexed_from_two_heads_is_a_problem(
        self, standalone, tmp_path
    ):
        _write_day(tmp_path, DAYS[0])
        day = tmp_path / f"{DAYS[1]}.jsonl"
        head = AuditWriter(audit_dir=tmp_path)._bootstrap_new_day(day)
        self._append_index_entry(tmp_path, DAYS[1], "f" * 16)
        record = _record("retry", "f" * 16)
        day.write_text(record.model_dump_json() + "\n", encoding="utf-8")

        problems, _, _ = _both(standalone, tmp_path)

        assert any("from different heads" in p for p in problems), problems
        assert head != "f" * 16

    def test_an_indexed_day_with_no_file_before_one_continuing_from_its_head(
        self, standalone, tmp_path
    ):
        """No record was ever written that day, or the file was deleted
        before the next day began; the files cannot say which, so it stays a
        problem -- once, and saying both."""
        _write_day(tmp_path, DAYS[0])
        gone = tmp_path / f"{DAYS[1]}.jsonl"
        head = AuditWriter(audit_dir=tmp_path)._bootstrap_new_day(gone)
        later = tmp_path / f"{DAYS[2]}.jsonl"
        later_head = AuditWriter(audit_dir=tmp_path)._bootstrap_new_day(later)
        assert later_head == head
        later.write_text(_record("later", head).model_dump_json() + "\n", "utf-8")

        problems, _, _ = _both(standalone, tmp_path)

        assert len(problems) == 1, problems
        assert "Either no record was ever written" in problems[0]
        assert "re-chained" not in problems[0]

    def test_a_deleted_day_with_records_is_still_called_deleted(
        self, standalone, tmp_path
    ):
        """Null case for the wording above: the next day continued from the
        deleted day's own tail, so something WAS there."""
        _write_trail(tmp_path)
        (tmp_path / f"{DAYS[1]}.jsonl").unlink()

        problems, _, _ = _both(standalone, tmp_path)

        assert any("likely deleted" in p for p in problems), problems
        assert not any("Either no record" in p for p in problems)
