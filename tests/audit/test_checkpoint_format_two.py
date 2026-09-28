"""
A signed checkpoint commits to the record count, a full SHA-256 digest of
the day's records and the chain index's length.

The record chain is 64-bit -- every `record_hash` is SHA-256 cut to 16 hex
characters -- and stays that way, because every record already written
carries one. A checkpoint that signed only the day's last record_hash
therefore committed to 64 bits of the day, and nothing about the index: a
newer day deleted together with its index line left an older signed day
reading "valid". Format 2 signs the count, a 256-bit digest and the
index's length at signing. Format 1 checkpoints still verify by the rule
they were signed under. See the CHANGELOG entry of 2026-09-28.
"""

import json
from pathlib import Path
from typing import List, Tuple

import pytest

from standard_quant_tools import audit, cli
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.audit.signing import _records_digest
from standard_quant_tools.audit.writer import AuditWriter

pytestmark = pytest.mark.skipif(
    not audit.HAS_CRYPTOGRAPHY, reason="cryptography is not installed"
)

DAY, LATER = "2024-05-01", "2024-05-02"


def _sealed(record: "audit.DecisionRecord") -> "audit.DecisionRecord":
    payload = json.loads(record.model_dump_json(exclude={"record_hash"}))
    record.record_hash = audit.hash_payload({**payload, "record_hash": None})
    return record


def _build(directory: Path, date: str, n: int = 3) -> List["audit.DecisionRecord"]:
    directory.mkdir(parents=True, exist_ok=True)
    day = directory / f"{date}.jsonl"
    prev = AuditWriter(audit_dir=directory)._bootstrap_new_day(day)
    records = []
    for i in range(n):
        record = audit.DecisionRecord(
            request_id=f"{date}-{i}",
            timestamp_utc=f"{date}T00:0{i}:00+00:00",
            tool_name="list_strategies",
            input={"n": i},
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        record.prev_record_hash = prev
        prev = _sealed(record).record_hash
        records.append(record)
    day.write_text("".join(r.model_dump_json() + "\n" for r in records), "utf-8")
    return records


@pytest.fixture
def keys(tmp_path) -> Tuple[Path, Path]:
    private_bytes, public_bytes = audit.generate_keypair()
    private_path, public_path = tmp_path / "k.private", tmp_path / "k.public"
    private_path.write_bytes(private_bytes)
    public_path.write_bytes(public_bytes)
    return private_path, public_path


@pytest.fixture
def audit_dir(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "audit"
    monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
    return directory


def _state(directory: Path, keys, date: str = DAY) -> "audit.CheckpointVerification":
    return audit.verify_checkpoint(date, keys[1], audit_dir=directory)


def _sign_format_one(directory: Path, private_path: Path, date: str = DAY) -> None:
    """A checkpoint as the previous release signed it."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    last = json.loads((directory / f"{date}.jsonl").read_text("utf-8").splitlines()[-1])
    entry = next(
        json.loads(x)
        for x in (directory / audit._INDEX_FILENAME).read_text("utf-8").splitlines()
        if json.loads(x)["date"] == date
    )
    checkpoint = {
        "date": date,
        "final_record_hash": last["record_hash"],
        "index_hash": entry["index_hash"],
        "signed_at_utc": "2026-01-01T00:00:00+00:00",
    }
    canonical = json.dumps(checkpoint, sort_keys=True).encode("utf-8")
    key = Ed25519PrivateKey.from_private_bytes(private_path.read_bytes())
    (directory / f"{date}.checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8"
    )
    (directory / f"{date}.checkpoint.sig").write_text(
        key.sign(canonical).hex(), encoding="utf-8"
    )


def _drop_later_day(directory: Path) -> None:
    """Delete the newer day and the index's last line -- two truncations,
    no hashing, and a trail that verifies clean on its own."""
    (directory / f"{LATER}.jsonl").unlink()
    index = directory / audit._INDEX_FILENAME
    lines = index.read_text("utf-8").splitlines()
    index.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")


class TestFormatTwoAnchorsWhatFormatOneCouldNot:
    def test_an_untouched_day_is_valid_under_format_two(self, audit_dir, keys):
        """Null case."""
        _build(audit_dir, DAY)
        audit.checkpoint_and_sign(DAY, audit_dir=audit_dir, key_path=keys[0])
        found = _state(audit_dir, keys)
        assert (found.state, found.version, found.records_signed) == ("valid", 2, 3)

    def test_removing_later_index_entries_is_altered(self, audit_dir, keys):
        """Signed after the later day was indexed, the checkpoint knows the
        index held two lines. Format 1 reads this trail as valid."""
        _build(audit_dir, DAY)
        _build(audit_dir, LATER)
        audit.checkpoint_and_sign(DAY, audit_dir=audit_dir, key_path=keys[0])
        _drop_later_day(audit_dir)

        assert audit.verify_audit_trail_integrity(audit_dir) == []
        found = _state(audit_dir, keys)
        assert found.state == "altered"
        assert "fewer than the 2" in (found.detail or "")

    def test_the_same_removal_under_a_format_one_checkpoint_stays_valid(
        self, audit_dir, keys
    ):
        """Why format 2 exists, and proof that format 1 is judged as it
        always was: nothing it signed changed."""
        _build(audit_dir, DAY)
        _build(audit_dir, LATER)
        _sign_format_one(audit_dir, keys[0])
        _drop_later_day(audit_dir)

        found = _state(audit_dir, keys)
        assert (found.state, found.version) == ("valid", 1)

    def test_a_rewritten_index_behind_the_signed_length_is_altered(
        self, audit_dir, keys
    ):
        _build(audit_dir, DAY)
        _build(audit_dir, LATER)
        audit.checkpoint_and_sign(DAY, audit_dir=audit_dir, key_path=keys[0])
        index = audit_dir / audit._INDEX_FILENAME
        lines = index.read_text("utf-8").splitlines()
        last = json.loads(lines[-1])
        last["chain_head"] = "e" * 16
        last["index_hash"] = None
        last["index_hash"] = audit.hash_payload(last)
        index.write_text("\n".join([lines[0], json.dumps(last)]) + "\n", "utf-8")

        found = _state(audit_dir, keys)
        assert found.state == "altered" and "line 2" in (found.detail or "")

    def test_a_truncated_day_is_altered_with_the_count(self, audit_dir, keys):
        _build(audit_dir, DAY)
        audit.checkpoint_and_sign(DAY, audit_dir=audit_dir, key_path=keys[0])
        day = audit_dir / f"{DAY}.jsonl"
        day.write_text(
            "\n".join(day.read_text("utf-8").splitlines()[:2]) + "\n", "utf-8"
        )

        found = _state(audit_dir, keys)
        assert found.state == "altered"
        assert "3 record(s) were signed and only 2" in (found.detail or "")

    def test_an_appended_record_is_extended(self, audit_dir, keys):
        records = _build(audit_dir, DAY)
        audit.checkpoint_and_sign(DAY, audit_dir=audit_dir, key_path=keys[0])
        extra = audit.DecisionRecord(
            request_id="late",
            timestamp_utc=f"{DAY}T00:09:00+00:00",
            tool_name="list_strategies",
            input={},
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        extra.prev_record_hash = records[-1].record_hash
        with open(audit_dir / f"{DAY}.jsonl", "a", encoding="utf-8") as f:
            f.write(_sealed(extra).model_dump_json() + "\n")

        found = _state(audit_dir, keys)
        assert (found.state, found.records_signed, found.records_after) == (
            "extended",
            3,
            1,
        )

    def test_a_copy_with_converted_line_endings_is_still_valid(self, audit_dir, keys):
        """The digest is over the parsed records, not the bytes, so a copy
        whose line endings were converted verifies."""
        _build(audit_dir, DAY)
        audit.checkpoint_and_sign(DAY, audit_dir=audit_dir, key_path=keys[0])
        day = audit_dir / f"{DAY}.jsonl"
        data = day.read_bytes().replace(b"\r\n", b"\n")
        day.write_bytes(data.replace(b"\n", b"\r\n"))

        assert _state(audit_dir, keys).state == "valid"

    def test_a_format_this_release_does_not_read_is_unavailable(self, audit_dir, keys):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        _build(audit_dir, DAY)
        checkpoint = {"checkpoint_version": 9, "date": DAY}
        key = Ed25519PrivateKey.from_private_bytes(keys[0].read_bytes())
        (audit_dir / f"{DAY}.checkpoint.json").write_text(json.dumps(checkpoint))
        (audit_dir / f"{DAY}.checkpoint.sig").write_text(
            key.sign(json.dumps(checkpoint, sort_keys=True).encode()).hex()
        )

        found = _state(audit_dir, keys)
        assert found.state == "unavailable" and "format 9" in (found.detail or "")


class TestTheDigest:
    def test_it_is_a_full_sha256_that_moves_with_any_field(self):
        records = [{"a": 1, "record_hash": "x" * 16}, {"b": [1.5, "NaN"]}]
        digest = _records_digest(records)
        assert len(digest) == 64
        assert _records_digest([{"a": 2, "record_hash": "x" * 16}, records[1]]) != (
            digest
        )
        assert _records_digest([records[0]]) != digest

    def test_key_order_does_not_move_it(self):
        assert _records_digest([{"a": 1, "b": 2}]) == _records_digest(
            [{"b": 2, "a": 1}]
        )


class TestTheSurfacesReportTheFormat:
    def test_the_meta_tool_reports_the_format_and_the_head(
        self, audit_dir, keys, monkeypatch
    ):
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
        _build(audit_dir, DAY)
        audit.checkpoint_and_sign(DAY, audit_dir=audit_dir, key_path=keys[0])

        result = dispatch(
            "verify_audit_integrity", {"date": DAY, "public_key_path": str(keys[1])}
        )

        assert result["signature_state"] == "valid"
        assert result["checkpoint_version"] == 2
        assert result["head_date"] == DAY and result["head_record_count"] == 3

    def test_the_meta_tool_says_when_a_checkpoint_is_format_one(
        self, audit_dir, keys, monkeypatch
    ):
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
        _build(audit_dir, DAY)
        _sign_format_one(audit_dir, keys[0])

        result = dispatch(
            "verify_audit_integrity", {"date": DAY, "public_key_path": str(keys[1])}
        )

        assert result["checkpoint_version"] == 1
        assert any("format 1" in note for note in result["notes"])

    def test_the_trail_check_reports_the_head_fields(self, audit_dir, monkeypatch):
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
        records = _build(audit_dir, DAY)

        result = dispatch("verify_audit_integrity", {})

        assert result["head_date"] == DAY
        assert result["head_record_count"] == 3
        assert result["head_record_hash"] == records[-1].record_hash
        assert result["index_entries"] == 1 and result["index_head_hash"]
        assert any(n.startswith(f"Verified through {DAY}") for n in result["notes"])

    def test_sqt_verify_checkpoint_prints_the_head(self, audit_dir, keys, capsys):
        _build(audit_dir, DAY)
        audit.checkpoint_and_sign(DAY, audit_dir=audit_dir, key_path=keys[0])

        code = cli.main(["verify", "--checkpoint", DAY, "--pubkey", str(keys[1])])

        out = capsys.readouterr().out
        assert code == 0
        assert f"Verified through {DAY}: 3 record(s)" in out
        assert "Signature valid." in out
