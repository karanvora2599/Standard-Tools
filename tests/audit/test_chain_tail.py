"""
A day rewritten and re-chained from its own published head.

The chain index records where each day STARTS. That head is in plaintext
in `_chain_index.jsonl`, sitting next to the day files, so an attacker who
rewrites a day's records and re-derives the chain from that real head
produces a file that is internally consistent and correctly seeded — which
is exactly what every verifier used to check and nothing more. What gives
it away is where the day ENDS: day N's last `record_hash` is what the
index recorded as day N+1's head, before the rewrite, in a separate
hash-chained artifact. These tests plant that rewrite and require both the
library verifier and the standalone one the auditor bundle ships to name
it — and require the honest trails, and the deleted day, to stay clean of
that accusation (see the CHANGELOG entry of 2026-09-22).
"""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from standard_quant_tools import audit

from .. import REPO_ROOT


@pytest.fixture(scope="module")
def standalone() -> ModuleType:
    """The stdlib-only verifier that travels inside an auditor bundle,
    loaded as a module — the second implementation this check has to hold
    in, and the one an auditor actually runs."""
    spec = importlib.util.spec_from_file_location(
        "verify_audit_log_standalone_tail",
        REPO_ROOT / "scripts" / "verify_audit_log.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash_record(record: "audit.DecisionRecord") -> str:
    return audit.hash_payload(
        {**record.model_dump(exclude={"record_hash"}), "record_hash": None}
    )


def _record_chained_onto(tool_name: str, prev_hash: str) -> "audit.DecisionRecord":
    """One decision record, hashed the way AuditWriter.write() hashes it."""
    record = audit.DecisionRecord(
        request_id=f"r-{tool_name}",
        timestamp_utc="2024-01-01T00:00:00+00:00",
        tool_name=tool_name,
        input={"quantity": 200},
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )
    record.prev_record_hash = prev_hash
    record.record_hash = _hash_record(record)
    return record


def _write_three_honest_days(directory: Path) -> None:
    """Three consecutive days, each bootstrapped through the real writer
    (so the chain index is the real one) and each carrying two records
    chained onto the head the index recorded."""
    writer = audit.AuditWriter(audit_dir=directory)
    for day in ("2024-03-01", "2024-03-02", "2024-03-03"):
        day_path = directory / f"{day}.jsonl"
        head = writer._bootstrap_new_day(day_path)
        first = _record_chained_onto(f"{day}-first", head)
        second = _record_chained_onto(f"{day}-second", first.record_hash)
        day_path.write_text(
            first.model_dump_json() + "\n" + second.model_dump_json() + "\n",
            encoding="utf-8",
        )


def _index_head_for(directory: Path, date: str) -> str:
    """The chain head the index publishes for a date — what an attacker
    reads before rewriting that day."""
    index_path = directory / audit._INDEX_FILENAME
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            if entry.get("date") == date:
                return entry["chain_head"]
    raise AssertionError(f"no chain index entry for {date}")


def _rewrite_day_from_its_published_head(directory: Path, date: str) -> None:
    """The attack: alter a payload and re-derive the whole day's chain from
    the head the index publishes for it. Every record hashes correctly,
    every prev_record_hash matches, and the first record starts exactly
    where the index says it should."""
    day_path = directory / f"{date}.jsonl"
    original = [
        json.loads(line)
        for line in day_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prev_hash = _index_head_for(directory, date)
    forged_lines = []
    for raw in original:
        record = audit.DecisionRecord(**raw)
        record.input = {"quantity": 999999}
        record.prev_record_hash = prev_hash
        record.record_hash = _hash_record(record)
        prev_hash = record.record_hash
        forged_lines.append(record.model_dump_json())
    day_path.write_text("\n".join(forged_lines) + "\n", encoding="utf-8")


class TestAnHonestTrailStaysClean:
    def test_three_consecutive_days_report_no_problems(self, tmp_path: Path):
        _write_three_honest_days(tmp_path)
        assert audit.verify_audit_trail_integrity(tmp_path) == []

    def test_the_standalone_verifier_agrees_the_trail_is_clean(
        self, standalone: ModuleType, tmp_path: Path
    ):
        _write_three_honest_days(tmp_path)
        assert standalone.verify_trail(tmp_path) == []
        assert standalone.main([str(tmp_path)]) == 0

    def test_a_day_whose_last_record_is_appended_normally_stays_clean(
        self, tmp_path: Path
    ):
        """The null case for the tail check: growing a day the way the
        writer grows it — appending onto the last record — moves the tail,
        and nothing downstream has been written yet, so there is nothing to
        contradict."""
        _write_three_honest_days(tmp_path)
        last_day = tmp_path / "2024-03-03.jsonl"
        tail = json.loads(last_day.read_text(encoding="utf-8").splitlines()[-1])
        appended = _record_chained_onto("late-call", tail["record_hash"])
        with open(last_day, "a", encoding="utf-8") as f:
            f.write(appended.model_dump_json() + "\n")

        assert audit.verify_audit_trail_integrity(tmp_path) == []


class TestARewrittenDayIsNamedByItsTail:
    def test_the_library_names_the_rewritten_day_against_the_next_days_head(
        self, tmp_path: Path
    ):
        _write_three_honest_days(tmp_path)
        honest_tail = json.loads(
            (tmp_path / "2024-03-02.jsonl").read_text(encoding="utf-8").splitlines()[-1]
        )["record_hash"]
        _rewrite_day_from_its_published_head(tmp_path, "2024-03-02")

        problems = audit.verify_audit_trail_integrity(tmp_path)

        assert len(problems) == 1, (
            "the rewritten day is internally consistent and correctly "
            "seeded, so the tail is the only thing that can report it: "
            f"{problems}"
        )
        problem = problems[0]
        forged_tail = json.loads(
            (tmp_path / "2024-03-02.jsonl").read_text(encoding="utf-8").splitlines()[-1]
        )["record_hash"]
        assert forged_tail != honest_tail
        assert "2024-03-02" in problem and "2024-03-03" in problem
        assert forged_tail in problem and honest_tail in problem
        assert "re-chained" in problem

    def test_the_rewritten_day_passes_every_check_that_existed_before_the_tail(
        self, tmp_path: Path
    ):
        """Why the tail check had to exist: seeded with the index's head,
        the forged day verifies clean on its own, and so does the index's
        own chain. Nothing but the comparison between them catches it."""
        _write_three_honest_days(tmp_path)
        _rewrite_day_from_its_published_head(tmp_path, "2024-03-02")

        forged_day = tmp_path / "2024-03-02.jsonl"
        assert (
            audit.verify_audit_log_integrity(
                forged_day,
                expected_prev_hash=_index_head_for(tmp_path, "2024-03-02"),
            )
            == []
        )
        assert json.loads(forged_day.read_text(encoding="utf-8").splitlines()[0])[
            "input"
        ] == {"quantity": 999999}

    def test_the_bundled_standalone_verifier_reports_it_identically(
        self, standalone: ModuleType, tmp_path: Path
    ):
        """The auditor runs the shipped script, not the library. The two
        have to say the same thing, word for word, or the bundle is a
        weaker artifact than the system it came from."""
        _write_three_honest_days(tmp_path)
        _rewrite_day_from_its_published_head(tmp_path, "2024-03-02")

        assert standalone.verify_trail(tmp_path) == audit.verify_audit_trail_integrity(
            tmp_path
        )
        assert standalone.main([str(tmp_path)]) == 1


class TestTheTailCheckAccusesNobodyElse:
    def test_a_deleted_day_is_reported_as_deleted_and_not_as_re_chained(
        self, tmp_path: Path
    ):
        """A day file that is gone has no tail to compare, so the day after
        it must not be accused of re-chaining on top of the deletion it is
        innocent of."""
        _write_three_honest_days(tmp_path)
        (tmp_path / "2024-03-02.jsonl").unlink()

        problems = audit.verify_audit_trail_integrity(tmp_path)

        assert any("2024-03-02" in p and "no longer exists" in p for p in problems)
        assert not any("re-chained" in p for p in problems)

    def test_a_day_that_predates_the_index_is_not_compared_to_it(self, tmp_path: Path):
        """Days before the index's earliest entry are deliberately not
        cross-day-linked. A file sitting there with its own genesis chain
        is not a re-chained day."""
        earlier = tmp_path / "2023-12-31.jsonl"
        _write_three_honest_days(tmp_path)
        standalone_record = _record_chained_onto("older-call", audit._GENESIS_HASH)
        earlier.write_text(standalone_record.model_dump_json() + "\n", encoding="utf-8")

        assert audit.verify_audit_trail_integrity(tmp_path) == []

    def test_an_empty_day_file_does_not_produce_a_tail_accusation(self, tmp_path: Path):
        """A bootstrapped day whose first record was never written has no
        tail. That is a gap in the evidence, not evidence of a rewrite, and
        the next day is not blamed for it."""
        writer = audit.AuditWriter(audit_dir=tmp_path)
        first_day = tmp_path / "2024-04-01.jsonl"
        head = writer._bootstrap_new_day(first_day)
        first_day.write_text("", encoding="utf-8")
        second_day = tmp_path / "2024-04-02.jsonl"
        second_head = writer._bootstrap_new_day(second_day)
        assert second_head == head
        record = _record_chained_onto("second-day-call", second_head)
        second_day.write_text(record.model_dump_json() + "\n", encoding="utf-8")

        assert not any(
            "re-chained" in p for p in audit.verify_audit_trail_integrity(tmp_path)
        )
