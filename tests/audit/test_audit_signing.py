"""
Tests for Ed25519 checkpoint signing
(src/standard_quant_tools/audit/signing.py). Requires the optional
`cryptography` dependency -- skipped automatically if it isn't installed,
matching this repo's pattern for other optional deps (e.g. blpapi).
"""

import json
from pathlib import Path

import pytest

from standard_quant_tools import audit
from standard_quant_tools.audit import signing as signing_module

pytestmark = pytest.mark.skipif(
    not audit.HAS_CRYPTOGRAPHY, reason="cryptography is not installed"
)


def _write_day_with_one_record(
    audit_dir: Path, date: str = "2024-01-01"
) -> "audit.DecisionRecord":
    w = audit.AuditWriter(audit_dir=audit_dir)
    day_path = audit_dir / f"{date}.jsonl"
    head = w._bootstrap_new_day(day_path)
    record = audit.DecisionRecord(
        request_id="r1",
        timestamp_utc=f"{date}T00:00:00+00:00",
        tool_name="t1",
        input={},
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )
    record.prev_record_hash = head
    record.record_hash = audit.hash_payload(
        {**record.model_dump(exclude={"record_hash"}), "record_hash": None}
    )
    day_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    return record


class TestKeypairGeneration:
    def test_generate_keypair_returns_32_byte_raw_keys(self):
        private_bytes, public_bytes = audit.generate_keypair()
        assert len(private_bytes) == 32
        assert len(public_bytes) == 32

    def test_each_call_generates_a_distinct_keypair(self):
        priv1, pub1 = audit.generate_keypair()
        priv2, pub2 = audit.generate_keypair()
        assert priv1 != priv2
        assert pub1 != pub2


class TestCheckpointSignRoundTrip:
    def test_sign_and_verify_round_trip_succeeds(self, tmp_path: Path):
        _write_day_with_one_record(tmp_path)
        private_bytes, public_bytes = audit.generate_keypair()
        priv_path = tmp_path.parent / "priv.key"
        pub_path = tmp_path.parent / "pub.key"
        priv_path.write_bytes(private_bytes)
        pub_path.write_bytes(public_bytes)

        checkpoint_path = audit.checkpoint_and_sign(
            "2024-01-01", audit_dir=tmp_path, key_path=priv_path
        )
        assert checkpoint_path.exists()
        assert (tmp_path / "2024-01-01.checkpoint.sig").exists()
        assert (
            audit.verify_checkpoint_signature(
                "2024-01-01", pub_path, audit_dir=tmp_path
            )
            is True
        )

    def test_checkpoint_contains_expected_fields(self, tmp_path: Path):
        record = _write_day_with_one_record(tmp_path)
        private_bytes, _ = audit.generate_keypair()
        priv_path = tmp_path.parent / "priv.key"
        priv_path.write_bytes(private_bytes)

        checkpoint_path = audit.checkpoint_and_sign(
            "2024-01-01", audit_dir=tmp_path, key_path=priv_path
        )
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        assert checkpoint["date"] == "2024-01-01"
        assert checkpoint["final_record_hash"] == record.record_hash
        assert "index_hash" in checkpoint
        assert "signed_at_utc" in checkpoint

    def test_signer_callback_used_instead_of_key_file(self, tmp_path: Path):
        _write_day_with_one_record(tmp_path)
        private_bytes, public_bytes = audit.generate_keypair()
        pub_path = tmp_path.parent / "pub.key"
        pub_path.write_bytes(public_bytes)

        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )

        private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
        calls = []

        def signer(payload: bytes) -> bytes:
            calls.append(payload)
            return private_key.sign(payload)

        audit.checkpoint_and_sign("2024-01-01", audit_dir=tmp_path, signer=signer)
        assert len(calls) == 1
        assert (
            audit.verify_checkpoint_signature(
                "2024-01-01", pub_path, audit_dir=tmp_path
            )
            is True
        )


class TestCheckpointTamperDetection:
    def test_wholesale_day_file_rewrite_after_signing_fails_verification(
        self, tmp_path: Path
    ):
        """The class of attack plain hash-chain verification cannot catch
        on its own (see verify.py's docstrings): an attacker deletes the
        day file and writes a brand-new, internally self-consistent one
        starting from genesis. The forged file's final_record_hash differs
        from what was signed, so checkpoint verification -- unlike
        verify_audit_log_integrity alone -- catches it."""
        _write_day_with_one_record(tmp_path)
        private_bytes, public_bytes = audit.generate_keypair()
        priv_path = tmp_path.parent / "priv.key"
        pub_path = tmp_path.parent / "pub.key"
        priv_path.write_bytes(private_bytes)
        pub_path.write_bytes(public_bytes)
        audit.checkpoint_and_sign("2024-01-01", audit_dir=tmp_path, key_path=priv_path)

        day_path = tmp_path / "2024-01-01.jsonl"
        day_path.unlink()
        forged = audit.DecisionRecord(
            request_id="forged",
            timestamp_utc="2024-01-01T00:00:00+00:00",
            tool_name="t1",
            input={},
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        forged.prev_record_hash = audit._GENESIS_HASH
        forged.record_hash = audit.hash_payload(
            {**forged.model_dump(exclude={"record_hash"}), "record_hash": None}
        )
        day_path.write_text(forged.model_dump_json() + "\n", encoding="utf-8")

        assert (
            audit.verify_checkpoint_signature(
                "2024-01-01", pub_path, audit_dir=tmp_path
            )
            is False
        )

    def test_altering_a_record_without_recomputing_its_hash_is_a_chain_break(
        self, tmp_path: Path
    ):
        """Editing a record's content but leaving its stored record_hash
        field untouched is exactly the "content altered" case
        verify_audit_log_integrity() already catches -- checkpoint
        verification alone does NOT re-validate the chain itself, it only
        anchors the chain's endpoint, so the two are meant to be run
        together, not as substitutes for each other. This test documents
        that boundary rather than asserting checkpoint verification catches
        it too."""
        _write_day_with_one_record(tmp_path)
        private_bytes, public_bytes = audit.generate_keypair()
        priv_path = tmp_path.parent / "priv.key"
        pub_path = tmp_path.parent / "pub.key"
        priv_path.write_bytes(private_bytes)
        pub_path.write_bytes(public_bytes)
        audit.checkpoint_and_sign("2024-01-01", audit_dir=tmp_path, key_path=priv_path)

        day_path = tmp_path / "2024-01-01.jsonl"
        line = json.loads(day_path.read_text(encoding="utf-8").splitlines()[0])
        line["status"] = "tampered"  # record_hash left stale on purpose
        day_path.write_text(json.dumps(line) + "\n", encoding="utf-8")

        # The chain-integrity check (a different function) is the one that
        # must catch this -- and does.
        problems = audit.verify_audit_log_integrity(day_path)
        assert problems and any("record_hash" in p for p in problems)

    def test_appending_a_record_after_signing_fails_verification(self, tmp_path: Path):
        record = _write_day_with_one_record(tmp_path)
        private_bytes, public_bytes = audit.generate_keypair()
        priv_path = tmp_path.parent / "priv.key"
        pub_path = tmp_path.parent / "pub.key"
        priv_path.write_bytes(private_bytes)
        pub_path.write_bytes(public_bytes)
        audit.checkpoint_and_sign("2024-01-01", audit_dir=tmp_path, key_path=priv_path)

        day_path = tmp_path / "2024-01-01.jsonl"
        r2 = audit.DecisionRecord(
            request_id="r2",
            timestamp_utc="2024-01-01T00:01:00+00:00",
            tool_name="t2",
            input={},
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        r2.prev_record_hash = record.record_hash
        r2.record_hash = audit.hash_payload(
            {**r2.model_dump(exclude={"record_hash"}), "record_hash": None}
        )
        with open(day_path, "a", encoding="utf-8") as f:
            f.write(r2.model_dump_json() + "\n")

        assert (
            audit.verify_checkpoint_signature(
                "2024-01-01", pub_path, audit_dir=tmp_path
            )
            is False
        )

    def test_tampering_the_stored_checkpoint_itself_fails_verification(
        self, tmp_path: Path
    ):
        _write_day_with_one_record(tmp_path)
        private_bytes, public_bytes = audit.generate_keypair()
        priv_path = tmp_path.parent / "priv.key"
        pub_path = tmp_path.parent / "pub.key"
        priv_path.write_bytes(private_bytes)
        pub_path.write_bytes(public_bytes)
        checkpoint_path = audit.checkpoint_and_sign(
            "2024-01-01", audit_dir=tmp_path, key_path=priv_path
        )

        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint["final_record_hash"] = "tampered0000000"
        checkpoint_path.write_text(
            json.dumps(checkpoint, sort_keys=True), encoding="utf-8"
        )

        assert (
            audit.verify_checkpoint_signature(
                "2024-01-01", pub_path, audit_dir=tmp_path
            )
            is False
        )

    def test_wrong_public_key_fails_verification(self, tmp_path: Path):
        _write_day_with_one_record(tmp_path)
        private_bytes, _ = audit.generate_keypair()
        _, other_public_bytes = audit.generate_keypair()
        priv_path = tmp_path.parent / "priv.key"
        wrong_pub_path = tmp_path.parent / "wrong_pub.key"
        priv_path.write_bytes(private_bytes)
        wrong_pub_path.write_bytes(other_public_bytes)
        audit.checkpoint_and_sign("2024-01-01", audit_dir=tmp_path, key_path=priv_path)

        assert (
            audit.verify_checkpoint_signature(
                "2024-01-01", wrong_pub_path, audit_dir=tmp_path
            )
            is False
        )


class TestCheckpointMissingArtifacts:
    def test_verify_missing_checkpoint_returns_false(self, tmp_path: Path):
        _, public_bytes = audit.generate_keypair()
        pub_path = tmp_path.parent / "pub.key"
        pub_path.write_bytes(public_bytes)
        assert (
            audit.verify_checkpoint_signature(
                "2099-01-01", pub_path, audit_dir=tmp_path
            )
            is False
        )

    def test_verify_missing_pubkey_file_returns_false(self, tmp_path: Path):
        _write_day_with_one_record(tmp_path)
        private_bytes, _ = audit.generate_keypair()
        priv_path = tmp_path.parent / "priv.key"
        priv_path.write_bytes(private_bytes)
        audit.checkpoint_and_sign("2024-01-01", audit_dir=tmp_path, key_path=priv_path)

        assert (
            audit.verify_checkpoint_signature(
                "2024-01-01", tmp_path.parent / "nonexistent.pub", audit_dir=tmp_path
            )
            is False
        )


class TestMissingSigningKey:
    def test_checkpoint_and_sign_without_key_raises_clear_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("SQT_AUDIT_SIGNING_KEY_PATH", raising=False)
        _write_day_with_one_record(tmp_path)
        with pytest.raises(FileNotFoundError, match="No signing key found"):
            audit.checkpoint_and_sign("2024-01-01", audit_dir=tmp_path)

    def test_checkpoint_and_sign_with_nonexistent_key_path_raises(self, tmp_path: Path):
        _write_day_with_one_record(tmp_path)
        with pytest.raises(FileNotFoundError):
            audit.checkpoint_and_sign(
                "2024-01-01",
                audit_dir=tmp_path,
                key_path=tmp_path.parent / "no-such-key",
            )

    def test_signing_key_path_env_var_is_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        _write_day_with_one_record(tmp_path)
        private_bytes, public_bytes = audit.generate_keypair()
        priv_path = tmp_path.parent / "priv.key"
        pub_path = tmp_path.parent / "pub.key"
        priv_path.write_bytes(private_bytes)
        pub_path.write_bytes(public_bytes)
        monkeypatch.setenv("SQT_AUDIT_SIGNING_KEY_PATH", str(priv_path))

        audit.checkpoint_and_sign("2024-01-01", audit_dir=tmp_path)  # no key_path
        assert (
            audit.verify_checkpoint_signature(
                "2024-01-01", pub_path, audit_dir=tmp_path
            )
            is True
        )


class TestCryptographyNotInstalled:
    """Confirms the import-guard error path without requiring an
    environment where cryptography is actually missing."""

    def test_require_cryptography_raises_clear_error_when_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(signing_module, "HAS_CRYPTOGRAPHY", False)
        with pytest.raises(ImportError, match=r"standard_quant_tools\[signing\]"):
            audit.generate_keypair()

    def test_checkpoint_and_sign_raises_when_cryptography_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(signing_module, "HAS_CRYPTOGRAPHY", False)
        with pytest.raises(ImportError):
            audit.checkpoint_and_sign("2024-01-01", audit_dir=tmp_path)

    def test_verify_checkpoint_signature_raises_when_cryptography_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(signing_module, "HAS_CRYPTOGRAPHY", False)
        with pytest.raises(ImportError):
            audit.verify_checkpoint_signature(
                "2024-01-01", tmp_path / "pub.key", audit_dir=tmp_path
            )
