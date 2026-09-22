"""
What the provenance tools say when the answer is not simply "fine".

Three of the four things pinned here produced a reassuring answer that was
not true, and the fourth produced an alarming one that was not true either
(see the CHANGELOG entry of 2026-09-22):

  - verifying ONE day checked it against the genesis hash instead of the
    head the chain index recorded for it, so every day after the first was
    reported as a broken chain, permanently, on a log nobody had touched;
  - an empty audit directory verified `intact=True, problems=[]`, so "the
    trail is intact" and "there is no trail" -- and "nothing is being
    recorded at all" -- were one answer;
  - an export whose date range covered no day file wrote a zip holding a
    manifest, a README and a verifier, reported a plausible size and an ok
    status, and was indistinguishable from a real bundle;
  - `explain_decision` returned fifteen of the record's nineteen fields,
    among them not the hash of the strategy source that ran -- the one
    field that ties a run to the code behind it.

Every record here is written by a REAL dispatch into a temporary audit
directory, for the same reason `test_provenance_tools.py` does it: a
hand-forged record would test the readers against a fiction and keep
passing while the writer moved.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from standard_quant_tools import audit
from standard_quant_tools.agent.runtimes.meta import tools as meta_tools
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.audit import signing
from standard_quant_tools.audit.paths import (
    _GENESIS_HASH,
    _INDEX_FILENAME,
    _audit_dir,
    _iter_day_files,
)
from standard_quant_tools.audit.writer import AuditWriter
from standard_quant_tools.error import ValidationError

#: Two planted calendar days. The writer files each record under the UTC
#: day it was written, so a trail that crosses midnight cannot be built in
#: one test run without planting the filename -- and a trail that never
#: crosses midnight is exactly the case the broken single-day check
#: happened to get right.
DAY_ONE = "2026-03-01"
DAY_TWO = "2026-03-02"


@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path, monkeypatch):
    """A fresh audit directory per test, so one test's records cannot make
    another's verdict."""
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
    return tmp_path / "audit"


def _plant_day(monkeypatch, date: str) -> None:
    """Send every record written from here on into `date`'s day file.

    Only the FILENAME moves. The chain head each new day commits to, the
    index entry that witnesses it and every record hash are computed by
    the writer exactly as they are in production -- which is the point,
    since what is being verified is that linkage.
    """
    monkeypatch.setattr(
        AuditWriter,
        "_path_for",
        lambda self, when, _date=date: self._dir / f"{_date}.jsonl",
    )


def _records_in(day_file: Path) -> list:
    lines = [
        line
        for line in day_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [json.loads(line) for line in lines]


def _record_a_call(
    tool: str = "list_strategies",
    payload: Optional[Dict[str, Any]] = None,
    *,
    day: Optional[str] = None,
) -> Dict[str, Any]:
    """Make a real, offline tool call and return the record it wrote."""
    dispatch(tool, {"strategy_type": "sma_crossover"} if payload is None else payload)
    directory = _audit_dir()
    day_files = _iter_day_files(directory)
    assert day_files, "the call wrote no audit record"
    path = directory / f"{day}.jsonl" if day else day_files[-1]
    return _records_in(path)[-1]


def _index_heads() -> Dict[str, str]:
    """Each day's recorded chain head, as the witness log states it."""
    index_path = _audit_dir() / _INDEX_FILENAME
    entries = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {entry["date"]: entry["chain_head"] for entry in entries}


def _edit_a_line(day_file: Path, lineno: int = 0) -> None:
    """Rewrite one record's content in place, leaving its hash behind."""
    lines = day_file.read_text(encoding="utf-8").strip().split("\n")
    record = json.loads(lines[lineno])
    record["duration_ms"] = 999999.0
    lines[lineno] = json.dumps(record)
    day_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestVerifyingOneDayWorksPastTheFirst:
    """A day's first record chains onto the PREVIOUS day's last one, and
    the chain index is where that head is written down. Checking the file
    against the genesis hash instead called every day but the first
    broken."""

    def test_the_second_recorded_day_verifies_clean(self, monkeypatch):
        _plant_day(monkeypatch, DAY_ONE)
        _record_a_call(day=DAY_ONE)
        _plant_day(monkeypatch, DAY_TWO)
        _record_a_call(day=DAY_TWO)

        # The premise: day two really does chain onto day one, so genesis
        # is the wrong thing to measure it against. Without this the test
        # could pass on a trail of one day and prove nothing.
        assert _index_heads()[DAY_TWO] != _GENESIS_HASH

        result = dispatch("verify_audit_integrity", {"date": DAY_TWO})
        assert result["problems"] == []
        assert result["intact"] is True
        assert result["verdict"] == "intact"

    def test_an_edited_line_in_the_second_day_still_breaks_it(self, monkeypatch):
        """The fix must not be "pass the head and stop looking". Seeding
        the chain correctly is what makes a real break meaningful."""
        _plant_day(monkeypatch, DAY_ONE)
        _record_a_call(day=DAY_ONE)
        _plant_day(monkeypatch, DAY_TWO)
        _record_a_call(day=DAY_TWO)
        _record_a_call(day=DAY_TWO)

        _edit_a_line(_audit_dir() / f"{DAY_TWO}.jsonl")

        result = dispatch("verify_audit_integrity", {"date": DAY_TWO})
        assert result["intact"] is False
        assert result["verdict"] == "tampered"
        assert result["problems"]

    def test_a_day_the_index_never_witnessed_says_what_it_assumed(self, monkeypatch):
        """No index entry is a real state -- a day predating the witness
        log. Genesis is then the right assumption, and the result says it
        was made rather than presenting it as a checked fact."""
        _plant_day(monkeypatch, DAY_ONE)
        _record_a_call(day=DAY_ONE)
        (_audit_dir() / _INDEX_FILENAME).unlink()

        result = dispatch("verify_audit_integrity", {"date": DAY_ONE})
        assert result["problems"] == []
        assert any("genesis" in note for note in result["notes"])


class TestTheVerdictSeparatesAnEmptyTrailFromAnIntactOne:
    def test_recording_switched_off_is_not_an_intact_trail(self, monkeypatch):
        """SQT_AUDIT_ENABLED gates WRITING only, so with it off the
        verifier finds no index and no day files and used to report the
        same thing it reports for a clean log."""
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "0")
        result = dispatch("verify_audit_integrity", {})
        assert result["verdict"] == "recording_disabled"
        assert result["recording_enabled"] is False
        assert result["problems"] == []
        assert any("SQT_AUDIT_ENABLED" in note for note in result["notes"])

    def test_an_empty_directory_with_recording_on_has_no_trail(self):
        result = dispatch("verify_audit_integrity", {})
        assert result["verdict"] == "no_trail"
        assert result["recording_enabled"] is True
        assert result["problems"] == []

    def test_records_that_verify_are_intact(self):
        _record_a_call()
        result = dispatch("verify_audit_integrity", {})
        assert result["verdict"] == "intact"
        assert result["recording_enabled"] is True
        assert result["intact"] is True

    def test_an_edited_record_is_tampered(self):
        _record_a_call()
        _record_a_call()
        _record_a_call()
        _edit_a_line(_iter_day_files(_audit_dir())[-1])

        result = dispatch("verify_audit_integrity", {})
        assert result["verdict"] == "tampered"
        assert result["intact"] is False
        assert result["problems"]

    def test_the_four_verdicts_are_the_declared_ones(self):
        """Each verdict above is a value of the same Literal, so a renamed
        state cannot pass unnoticed by half the callers."""
        from standard_quant_tools.agent import VerifyAuditIntegrityResult

        declared = VerifyAuditIntegrityResult.model_fields["verdict"].annotation
        assert set(getattr(declared, "__args__", ())) == {
            "intact",
            "tampered",
            "no_trail",
            "recording_disabled",
        }


@pytest.mark.skipif(not audit.HAS_CRYPTOGRAPHY, reason="cryptography is not installed")
class TestTheSignatureStateNamesWhichFailureItIs:
    """A checkpoint check has six ways of not saying "valid", and they call
    for completely different responses. Collapsed into one boolean, "nobody
    ever signed this day" arrived looking exactly like "this day was
    forged"."""

    def _public_key(self, tmp_path) -> str:
        private_bytes, public_bytes = signing.generate_keypair()
        (tmp_path / "audit.key").write_bytes(private_bytes)
        (tmp_path / "audit.pub").write_bytes(public_bytes)
        return str(tmp_path / "audit.pub")

    def test_a_day_nobody_anchored_is_not_reported_as_broken(self, tmp_path):
        _record_a_call()
        date = _iter_day_files(_audit_dir())[-1].stem

        result = dispatch(
            "verify_audit_integrity",
            {"date": date, "public_key_path": self._public_key(tmp_path)},
        )
        assert result["signature_state"] == "no_checkpoint"
        assert result["checkpoint_signature_valid"] is None
        assert result["verdict"] == "intact"
        assert any("never anchored" in note for note in result["notes"])

    def test_verifying_a_signed_day_twice_reports_the_record_it_appended(
        self, tmp_path
    ):
        """Every verification is itself a recorded call, so a signed day
        that is still being written to drifts past its checkpoint by
        design. The result names the cause and the remedy instead of
        presenting the drift as evidence."""
        _record_a_call()
        date = _iter_day_files(_audit_dir())[-1].stem
        public_key = self._public_key(tmp_path)
        signing.checkpoint_and_sign(
            date, audit_dir=_audit_dir(), key_path=tmp_path / "audit.key"
        )

        first = dispatch(
            "verify_audit_integrity",
            {"date": date, "public_key_path": public_key},
        )
        assert first["signature_state"] == "valid"
        assert first["checkpoint_signature_valid"] is True

        # The call above wrote its own record into the day it verified.
        second = dispatch(
            "verify_audit_integrity",
            {"date": date, "public_key_path": public_key},
        )
        assert second["signature_state"] == "content_drift"
        assert second["checkpoint_signature_valid"] is False
        assert any("appends a record" in note for note in second["notes"])
        assert any("yesterday" in note for note in second["notes"])


class TestAnEmptyExportIsRefused:
    def test_a_range_covering_no_day_is_refused_and_writes_nothing(self, tmp_path):
        _record_a_call()
        out = tmp_path / "nothing.zip"
        with pytest.raises(ValidationError) as exc:
            dispatch(
                "export_audit_bundle",
                {
                    "start_date": "1999-01-01",
                    "end_date": "1999-01-02",
                    "out_path": str(out),
                },
            )
        assert "describe_audit_log" in str(exc.value)
        assert not out.exists(), "a refused export still wrote a bundle"

    def test_a_real_range_reports_what_the_bundle_holds(self, tmp_path):
        _record_a_call()
        _record_a_call()
        date = _iter_day_files(_audit_dir())[-1].stem

        result = dispatch(
            "export_audit_bundle",
            {
                "start_date": date,
                "end_date": date,
                "out_path": str(tmp_path / "bundle.zip"),
            },
        )
        assert result["day_files"] >= 1
        assert result["record_count"] >= 1
        assert result["size_bytes"] > 0


class TestExplainDecisionReturnsTheWholeRecord:
    def test_a_registered_strategys_source_hash_crosses(self):
        """`provenance.py` hashes a registered strategy's source precisely
        so a run can be tied to the code that ran; the tool dropped it, so
        the only way to have it was to read the day file."""
        record = _record_a_call(payload={"strategy_type": "sma_crossover"})
        result = dispatch("explain_decision", {"request_id": record["request_id"]})

        assert record["strategy_source_hash"] is not None
        assert result["strategy_source_hash"] == record["strategy_source_hash"]

    def test_a_call_that_names_no_strategy_carries_none(self):
        record = _record_a_call("list_stress_scenarios", {})
        result = dispatch("explain_decision", {"request_id": record["request_id"]})
        assert result["strategy_source_hash"] is None

    def test_the_remaining_three_fields_match_the_stored_record(self):
        record = _record_a_call()
        result = dispatch("explain_decision", {"request_id": record["request_id"]})

        assert result["prev_record_hash"] == record["prev_record_hash"]
        assert result["output_hash_normalized"] == record["output_hash_normalized"]
        assert result["n_workers"] == record["n_workers"]


class TestReplayComesWithWhatChanged:
    def test_each_data_source_match_carries_both_hashes(self, monkeypatch):
        """`verify_replay` computes the recorded and the re-fetched hash of
        every input and the tool kept neither, so `code_changed` arrived
        without the evidence it was decided from. Injected here rather than
        fetched, because a real hash difference needs a provider that
        revises its history."""
        record = _record_a_call()
        real = meta_tools._verify_replay

        def _with_hashes(rec):
            result = real(rec)
            result.data_source_matches = [
                {
                    "symbol": "AAPL",
                    "start": "2024-01-01",
                    "end": "2024-06-01",
                    "interval": "1d",
                    "old_hash": "1111111111111111",
                    "new_hash": "2222222222222222",
                    "match": False,
                }
            ]
            return result

        monkeypatch.setattr(meta_tools, "_verify_replay", _with_hashes)
        result = dispatch("replay_decision", {"request_id": record["request_id"]})

        match = result["data_source_matches"][0]
        assert match["old_hash"] == "1111111111111111"
        assert match["new_hash"] == "2222222222222222"
        assert match["matches"] is False

    def test_the_output_hashes_cross(self):
        record = _record_a_call()
        result = dispatch("replay_decision", {"request_id": record["request_id"]})

        assert result["verdict"] == "reproduced"
        assert result["stored_output_hash"] == record["output_hash"]
        assert result["new_output_hash"] == result["stored_output_hash"]
