"""
A signed checkpoint commits to what the day's records RECOMPUTE to.

The checkpoint used to be compared with the `record_hash` the day's last
line claims. Nothing recomputed it, so an edit that left the stored hashes
alone changed nothing the checkpoint looked at: an edited day verified
"valid", `verify_checkpoint_signature` said True, and `sqt verify
--checkpoint` -- which ran INSTEAD of the chain check -- printed "Signature
valid." with exit 0. A day signed after it had been edited verified valid
too, and a day cut short read as the same benign "drift" as one that had
grown.

Pinned here: an edited, truncated or rewritten day is "altered"; a day that
only grew is "extended"; signing refuses a day whose chain does not hold; a
checkpoint signed before any of this still verifies; the CLI runs the chain
check as well and names the state; the meta runtime carries the new states.
See the CHANGELOG entry of 2026-09-27.
"""

import json
from pathlib import Path
from typing import Callable, List, Tuple

import pytest

from standard_quant_tools import audit, cli
from standard_quant_tools.audit.signing import CHECKPOINT_FAILURES
from standard_quant_tools.audit.writer import AuditWriter
from standard_quant_tools.error import ValidationError

pytestmark = pytest.mark.skipif(
    not audit.HAS_CRYPTOGRAPHY, reason="cryptography is not installed"
)

DATE = "2024-05-01"


def _record(i: int, date: str = DATE) -> "audit.DecisionRecord":
    return audit.DecisionRecord(
        request_id=f"r{i}",
        timestamp_utc=f"{date}T00:0{i}:00+00:00",
        tool_name="get_option_pricing",
        input={"n": i},
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )


def _seal_hash(record: "audit.DecisionRecord") -> str:
    payload = json.loads(record.model_dump_json(exclude={"record_hash"}))
    return audit.hash_payload({**payload, "record_hash": None})


def _build_day(
    audit_dir: Path, n: int = 3, date: str = DATE
) -> List["audit.DecisionRecord"]:
    """A day of `n` chained records, indexed as the writer indexes one."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    day = audit_dir / f"{date}.jsonl"
    prev = AuditWriter(audit_dir=audit_dir)._bootstrap_new_day(day)
    records = []
    for i in range(n):
        record = _record(i, date)
        record.prev_record_hash = prev
        record.record_hash = _seal_hash(record)
        prev = record.record_hash
        records.append(record)
    day.write_text("".join(r.model_dump_json() + "\n" for r in records), "utf-8")
    return records


def _append(audit_dir: Path, after: "audit.DecisionRecord", i: int = 7) -> None:
    record = _record(i)
    record.prev_record_hash = after.record_hash
    record.record_hash = _seal_hash(record)
    with open(audit_dir / f"{DATE}.jsonl", "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")


def _rewrite(audit_dir: Path, change: Callable[[list], None]) -> None:
    """Edit the day's lines in place, leaving every stored hash behind."""
    day = audit_dir / f"{DATE}.jsonl"
    lines = [json.loads(x) for x in day.read_text(encoding="utf-8").splitlines()]
    change(lines)
    day.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")


@pytest.fixture
def keys(tmp_path: Path) -> Tuple[Path, Path]:
    private_bytes, public_bytes = audit.generate_keypair()
    private_path = tmp_path / "signing.private"
    public_path = tmp_path / "signing.public"
    private_path.write_bytes(private_bytes)
    public_path.write_bytes(public_bytes)
    return private_path, public_path


@pytest.fixture
def audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "audit"
    monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
    return directory


def _sign(audit_dir: Path, keys: Tuple[Path, Path]) -> None:
    audit.checkpoint_and_sign(DATE, audit_dir=audit_dir, key_path=keys[0])


def _found(audit_dir: Path, keys: Tuple[Path, Path]) -> "audit.CheckpointVerification":
    found = audit.verify_checkpoint(DATE, keys[1], audit_dir=audit_dir)
    assert audit.verify_checkpoint_state(DATE, keys[1], audit_dir=audit_dir) == (
        found.state
    )
    assert audit.verify_checkpoint_signature(DATE, keys[1], audit_dir=audit_dir) is (
        found.state == "valid"
    )
    return found


class TestADayChangedAfterSigningIsAltered:
    def test_an_edit_in_the_middle_that_leaves_the_hashes_alone(self, audit_dir, keys):
        _build_day(audit_dir)
        _sign(audit_dir, keys)
        _rewrite(audit_dir, lambda lines: lines[1]["input"].update(n=999))

        found = _found(audit_dir, keys)
        assert found.state == "altered"
        assert "line 2" in (found.detail or "")

    def test_an_edit_to_the_last_record(self, audit_dir, keys):
        _build_day(audit_dir)
        _sign(audit_dir, keys)
        _rewrite(audit_dir, lambda lines: lines[-1].update(status="error"))

        assert _found(audit_dir, keys).state == "altered"

    def test_a_day_cut_short(self, audit_dir, keys):
        """Removing the last record used to read as the same benign drift
        as appending one."""
        _build_day(audit_dir)
        _sign(audit_dir, keys)
        _rewrite(audit_dir, lambda lines: lines.pop())

        assert _found(audit_dir, keys).state == "altered"

    def test_a_day_rewritten_from_its_published_head(self, audit_dir, keys):
        """Internally consistent and correctly seeded, so the chain alone
        passes it; only the signed endpoint gives it away."""
        records = _build_day(audit_dir)
        _sign(audit_dir, keys)
        prev = records[0].prev_record_hash
        forged = []
        for i in range(3):
            record = _record(i + 10)
            record.prev_record_hash = prev
            record.record_hash = _seal_hash(record)
            prev = record.record_hash
            forged.append(record)
        (audit_dir / f"{DATE}.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in forged), "utf-8"
        )

        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert _found(audit_dir, keys).state == "altered"

    def test_an_edit_before_the_signed_point_hidden_behind_an_append(
        self, audit_dir, keys
    ):
        records = _build_day(audit_dir)
        _sign(audit_dir, keys)
        _append(audit_dir, records[-1])
        _rewrite(audit_dir, lambda lines: lines[0]["input"].update(n=5))

        assert _found(audit_dir, keys).state == "altered"

    def test_an_edit_after_the_signed_point(self, audit_dir, keys):
        records = _build_day(audit_dir)
        _sign(audit_dir, keys)
        _append(audit_dir, records[-1])
        _rewrite(audit_dir, lambda lines: lines[-1]["input"].update(n=5))

        found = _found(audit_dir, keys)
        assert found.state == "altered"
        assert found.records_signed == 3

    def test_a_changed_chain_index_entry(self, audit_dir, keys):
        _build_day(audit_dir)
        _sign(audit_dir, keys)
        index = audit_dir / audit._INDEX_FILENAME
        entry = json.loads(index.read_text(encoding="utf-8"))
        entry["chain_head"] = "f" * 16
        index.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        assert _found(audit_dir, keys).state == "altered"

    def test_altered_is_a_failure(self):
        assert "altered" in CHECKPOINT_FAILURES
        assert "extended" not in CHECKPOINT_FAILURES


class TestADayThatOnlyGrewIsExtended:
    def test_an_untouched_day_is_valid(self, audit_dir, keys):
        """Null case."""
        _build_day(audit_dir)
        _sign(audit_dir, keys)

        found = _found(audit_dir, keys)
        assert found.state == "valid"
        assert (found.records_signed, found.records_after) == (3, 0)

    def test_a_record_appended_after_signing(self, audit_dir, keys):
        records = _build_day(audit_dir)
        _sign(audit_dir, keys)
        _append(audit_dir, records[-1])

        found = _found(audit_dir, keys)
        assert found.state == "extended"
        assert (found.records_signed, found.records_after) == (3, 1)


class TestSigningRefusesADayThatDoesNotHold:
    def test_a_day_edited_before_signing_is_not_signed(self, audit_dir, keys):
        """It used to be signed, and then verified "valid" for ever."""
        _build_day(audit_dir)
        _rewrite(audit_dir, lambda lines: lines[0]["input"].update(n=5))

        with pytest.raises(ValidationError, match="sqt verify"):
            _sign(audit_dir, keys)
        assert not (audit_dir / f"{DATE}.checkpoint.json").exists()
        assert not (audit_dir / f"{DATE}.checkpoint.sig").exists()

    def test_a_day_with_no_records_is_not_signed(self, audit_dir, keys):
        audit_dir.mkdir(parents=True)
        with pytest.raises(ValidationError, match="nothing to"):
            _sign(audit_dir, keys)

    def test_an_intact_day_is_signed_at_its_recomputed_end(self, audit_dir, keys):
        """Null case. The checkpoint written is format 2: besides the
        endpoint and the day's index entry it commits to the record count,
        a full SHA-256 digest of the records and the chain index's length.
        This used to pin the four-field format 1 shape, which signing no
        longer writes (format 1 checkpoints still verify; see
        TestACheckpointSignedBeforeStillVerifies)."""
        records = _build_day(audit_dir)
        _sign(audit_dir, keys)

        checkpoint = json.loads(
            (audit_dir / f"{DATE}.checkpoint.json").read_text(encoding="utf-8")
        )
        assert checkpoint["final_record_hash"] == records[-1].record_hash
        assert set(checkpoint) == {
            "checkpoint_version",
            "date",
            "record_count",
            "final_record_hash",
            "day_digest",
            "index_hash",
            "index_len",
            "index_head",
            "signed_at_utc",
        }
        assert checkpoint["checkpoint_version"] == 2
        assert checkpoint["record_count"] == len(records)
        assert len(checkpoint["day_digest"]) == 64
        assert checkpoint["index_len"] == 1


def _sign_the_old_way(audit_dir: Path, private_path: Path) -> None:
    """A checkpoint as the previous release wrote it: the hash the day's
    last line claims and the index entry's hash, signed as they stood."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    day = audit_dir / f"{DATE}.jsonl"
    last = json.loads(day.read_text(encoding="utf-8").splitlines()[-1])
    index = audit_dir / audit._INDEX_FILENAME
    entry = json.loads(index.read_text(encoding="utf-8").splitlines()[-1])
    checkpoint = {
        "date": DATE,
        "final_record_hash": last["record_hash"],
        "index_hash": entry["index_hash"],
        "signed_at_utc": "2026-01-01T00:00:00+00:00",
    }
    canonical = json.dumps(checkpoint, sort_keys=True).encode("utf-8")
    key = Ed25519PrivateKey.from_private_bytes(private_path.read_bytes())
    (audit_dir / f"{DATE}.checkpoint.json").write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True), encoding="utf-8"
    )
    (audit_dir / f"{DATE}.checkpoint.sig").write_text(
        key.sign(canonical).hex(), encoding="utf-8"
    )


class TestACheckpointSignedBeforeStillVerifies:
    def test_an_old_checkpoint_on_an_untouched_day_is_valid(self, audit_dir, keys):
        _build_day(audit_dir)
        _sign_the_old_way(audit_dir, keys[0])

        assert _found(audit_dir, keys).state == "valid"

    def test_an_old_checkpoint_over_a_line_written_for_a_nan_is_valid(
        self, audit_dir, keys
    ):
        """The previous writer hashed a NaN input and wrote it as null, so
        that line never reproduces its stored hash. It is recognised by
        restoring the NaN, the same way the chain check recognises it, and
        the day the signature covers is still the day on disk."""
        audit_dir.mkdir(parents=True)
        day = audit_dir / f"{DATE}.jsonl"
        record = _record(0)
        record.input = {"spot": [100.0, float("nan")]}
        record.prev_record_hash = AuditWriter(audit_dir=audit_dir)._bootstrap_new_day(
            day
        )
        record.record_hash = audit.hash_payload(
            {**record.model_dump(exclude={"record_hash"}), "record_hash": None}
        )
        day.write_text(record.model_dump_json() + "\n", encoding="utf-8")
        _sign_the_old_way(audit_dir, keys[0])

        assert _found(audit_dir, keys).state == "valid"


class TestTheCliChecksTheChainAndNamesTheState:
    def _verify(self, audit_dir, keys, capsys) -> Tuple[int, str]:
        capsys.readouterr()
        code = cli.main(["verify", "--checkpoint", DATE, "--pubkey", str(keys[1])])
        return code, capsys.readouterr().out

    def test_an_edited_day_fails_with_its_state_named(self, audit_dir, keys, capsys):
        """This printed "Signature valid." and exited 0."""
        _build_day(audit_dir)
        _sign(audit_dir, keys)
        _rewrite(audit_dir, lambda lines: lines[1]["input"].update(n=999))

        code, out = self._verify(audit_dir, keys, capsys)
        assert code == 1
        assert "altered" in out
        assert "Signature valid." not in out
        assert "problem(s) found" in out

    def test_a_grown_day_exits_three_with_its_state_named(
        self, audit_dir, keys, capsys
    ):
        records = _build_day(audit_dir)
        _sign(audit_dir, keys)
        _append(audit_dir, records[-1])

        code, out = self._verify(audit_dir, keys, capsys)
        assert code == 3
        assert "extended" in out

    def test_an_untouched_day_passes(self, audit_dir, keys, capsys):
        """Null case: the wording a script may already look for is kept."""
        _build_day(audit_dir)
        _sign(audit_dir, keys)

        code, out = self._verify(audit_dir, keys, capsys)
        assert code == 0
        assert "Signature valid." in out
        assert "OK" in out

    def test_a_broken_chain_elsewhere_fails_a_valid_checkpoint(
        self, audit_dir, keys, capsys
    ):
        """The chain is checked every time, so `verify --checkpoint` cannot
        pass a trail that `verify` fails."""
        _build_day(audit_dir)
        _sign(audit_dir, keys)
        _build_day(audit_dir, date="2024-05-02")
        later = audit_dir / "2024-05-02.jsonl"
        line = json.loads(later.read_text(encoding="utf-8").splitlines()[0])
        line["status"] = "error"
        rest = later.read_text(encoding="utf-8").splitlines()[1:]
        later.write_text("\n".join([json.dumps(line), *rest]) + "\n", encoding="utf-8")

        code, out = self._verify(audit_dir, keys, capsys)
        assert code == 1
        assert "Checkpoint 2024-05-01: valid" in out


class TestTheMetaToolReportsTheNewStates:
    def test_an_edited_signed_day_is_altered_and_a_failed_check(self, audit_dir, keys):
        """It reported `signature_state="valid"` and
        `checkpoint_signature_valid=True` beside a tampered chain."""
        from standard_quant_tools.agent.tools import dispatch

        _build_day(audit_dir)
        _sign(audit_dir, keys)
        _rewrite(audit_dir, lambda lines: lines[1]["input"].update(n=999))

        result = dispatch(
            "verify_audit_integrity",
            {"date": DATE, "public_key_path": str(keys[1])},
        )
        assert result["signature_state"] == "altered"
        assert result["checkpoint_signature_valid"] is False
        assert result["records_after_checkpoint"] is None
        assert result["verdict"] == "tampered"

    def test_an_untouched_signed_day_is_valid(self, audit_dir, keys):
        """Null case."""
        from standard_quant_tools.agent.tools import dispatch

        _build_day(audit_dir)
        _sign(audit_dir, keys)

        result = dispatch(
            "verify_audit_integrity",
            {"date": DATE, "public_key_path": str(keys[1])},
        )
        assert result["signature_state"] == "valid"
        assert result["checkpoint_signature_valid"] is True
        assert result["records_after_checkpoint"] == 0
        assert result["verdict"] == "intact"
