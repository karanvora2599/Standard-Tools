"""
Three answers the audit trail computed and then threw away.

- A replay of a call that FAILED originally and SUCCEEDS now is a real
  answer — the failure no longer reproduces. It used to come back as
  "replay could not run", because the arm that returns it named a field
  ReplayResult does not have and raised a TypeError on the way out.
- A replay's verdict used to arrive without the hashes it was made from,
  so "the output changed" could not be checked or quoted.
- A checkpoint check answered False for a day that was never signed, for
  the wrong public key, for damaged signature bytes and for a record
  appended after signing — four situations calling for four different
  responses, reported identically.

And what the auditor bundle carries: the checkpoint sidecars that are the
only evidence surviving a wholesale rewrite, and a count of what went in
so a bundle of nothing is recognizable as one. See the CHANGELOG entry of
2026-09-22.
"""

import json
import zipfile
from pathlib import Path
from typing import Tuple

import pytest

from standard_quant_tools import audit
from standard_quant_tools.audit.replay import verify_replay

# A pure-computation tool: replaying it needs no provider, no network and
# no cache, so these tests exercise the replay machinery rather than the
# weather.
_PRICING_INPUT = {
    "spot": 100.0,
    "strike": 100.0,
    "time_to_expiry": 1.0,
    "risk_free_rate": 0.03,
    "volatility": 0.2,
}


def _pricing_record(**overrides) -> dict:
    record = {
        "request_id": "r-pricing",
        "timestamp_utc": "2024-01-01T00:00:00+00:00",
        "tool_name": "get_option_pricing",
        "input": dict(_PRICING_INPUT),
        "data_sources": [],
        "status": "ok",
    }
    record.update(overrides)
    return record


class TestAFailureThatNoLongerReproduces:
    def test_it_comes_back_as_a_result_rather_than_a_machinery_error(self):
        record = _pricing_record(status="error", error_type="ValidationError")

        result = verify_replay(record)

        assert isinstance(result, audit.ReplayResult)
        assert result.output_match is False
        assert any("no longer reproduces" in note for note in result.notes)
        assert any("replay SUCCEEDED" in note for note in result.notes)

    def test_it_reports_what_the_replay_produced_this_time(self):
        """There is no stored output to compare against — the original
        produced none — so the hash of what ran now is the whole of the
        evidence, and it is returned rather than discarded."""
        result = verify_replay(
            _pricing_record(status="error", error_type="ValidationError")
        )

        assert result.new_output_hash and len(result.new_output_hash) == 16
        assert result.stored_output_hash is None

    def test_a_failure_that_still_reproduces_is_unchanged(self):
        """The other arm of the same branch: the original failed, the
        replay fails the same way, and that is a successful replay of a
        failed call. This answer was already correct and stays word for
        word what it was."""
        rejected_input = dict(_PRICING_INPUT, option_type="straddle")
        # Recorded as "ok", the replay's own exception escapes — which is
        # how the failure type is discovered rather than assumed.
        try:
            verify_replay(_pricing_record(input=rejected_input))
            pytest.skip("the input was accepted; the failure path cannot be exercised")
        except Exception as exc:  # noqa: BLE001 - the type is the fixture
            failure_type = type(exc).__name__

        result = verify_replay(
            _pricing_record(
                input=rejected_input, status="error", error_type=failure_type
            )
        )
        assert result.output_match is True
        assert any("reproduced" in note for note in result.notes)

    def test_a_different_failure_is_still_reported_as_a_change(self):
        result = verify_replay(
            _pricing_record(
                input=dict(_PRICING_INPUT, option_type="straddle"),
                status="error",
                error_type="AnEntirelyDifferentError",
            )
        )
        assert result.output_match is False
        assert any("DIFFERENT failure" in note for note in result.notes)


class TestTheVerdictArrivesWithItsEvidence:
    def test_a_matching_replay_reports_both_hashes(self):
        reference = verify_replay(_pricing_record())
        assert reference.stored_output_hash is None  # nothing was recorded to compare

        result = verify_replay(_pricing_record(output_hash=reference.new_output_hash))

        assert result.output_match is True
        assert result.new_output_hash == reference.new_output_hash
        assert result.stored_output_hash == reference.new_output_hash

    def test_a_mismatch_says_which_hash_moved_to_which(self):
        result = verify_replay(_pricing_record(output_hash="0123456789abcdef"))

        assert result.output_match is False
        assert result.stored_output_hash == "0123456789abcdef"
        assert result.new_output_hash != result.stored_output_hash
        assert any("code/logic likely changed" in note for note in result.notes)

    def test_an_output_without_volatile_identifiers_needs_no_normalized_hash(self):
        """The null case: the normalized hash exists for outputs carrying
        per-run dataset/model ids, and stays None when the literal
        comparison was the only one made."""
        result = verify_replay(_pricing_record(output_hash="0123456789abcdef"))
        assert result.new_output_hash_normalized is None


# ── Signed checkpoints: which of seven states ─────────────────────────────────


def _write_day(audit_dir: Path, date: str = "2024-05-01") -> "audit.DecisionRecord":
    writer = audit.AuditWriter(audit_dir=audit_dir)
    day_path = audit_dir / f"{date}.jsonl"
    head = writer._bootstrap_new_day(day_path)
    record = audit.DecisionRecord(
        request_id="r1",
        timestamp_utc=f"{date}T00:00:00+00:00",
        tool_name="get_option_pricing",
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


def _keypair_files(directory: Path, name: str = "signing") -> Tuple[Path, Path]:
    private_bytes, public_bytes = audit.generate_keypair()
    private_path = directory / f"{name}.private"
    public_path = directory / f"{name}.public"
    private_path.write_bytes(private_bytes)
    public_path.write_bytes(public_bytes)
    return private_path, public_path


def _state_and_bool(date: str, public_key_path: Path, audit_dir: Path) -> str:
    """The state, having required that the boolean gate agrees with it. The
    two are one function and one derived bit; nothing may make them
    disagree."""
    state = audit.verify_checkpoint_state(date, public_key_path, audit_dir=audit_dir)
    assert audit.verify_checkpoint_signature(
        date, public_key_path, audit_dir=audit_dir
    ) is (state == "valid")
    return state


@pytest.mark.skipif(not audit.HAS_CRYPTOGRAPHY, reason="cryptography is not installed")
class TestCheckpointStateNamesTheCause:
    def test_valid_immediately_after_signing(self, tmp_path: Path):
        _write_day(tmp_path)
        private_path, public_path = _keypair_files(tmp_path.parent)
        audit.checkpoint_and_sign(
            "2024-05-01", audit_dir=tmp_path, key_path=private_path
        )

        assert _state_and_bool("2024-05-01", public_path, tmp_path) == "valid"

    def test_no_checkpoint_when_the_day_was_never_anchored(self, tmp_path: Path):
        _write_day(tmp_path)
        private_path, public_path = _keypair_files(tmp_path.parent)
        audit.checkpoint_and_sign(
            "2024-05-01", audit_dir=tmp_path, key_path=private_path
        )
        (tmp_path / "2024-05-01.checkpoint.json").unlink()

        assert _state_and_bool("2024-05-01", public_path, tmp_path) == "no_checkpoint"

    def test_no_signature_when_the_sig_file_is_gone(self, tmp_path: Path):
        _write_day(tmp_path)
        private_path, public_path = _keypair_files(tmp_path.parent)
        audit.checkpoint_and_sign(
            "2024-05-01", audit_dir=tmp_path, key_path=private_path
        )
        (tmp_path / "2024-05-01.checkpoint.sig").unlink()

        assert _state_and_bool("2024-05-01", public_path, tmp_path) == "no_signature"

    def test_key_mismatch_when_verified_with_a_different_public_key(
        self, tmp_path: Path
    ):
        _write_day(tmp_path)
        private_path, _ = _keypair_files(tmp_path.parent, "signer")
        _, other_public_path = _keypair_files(tmp_path.parent, "someone_else")
        audit.checkpoint_and_sign(
            "2024-05-01", audit_dir=tmp_path, key_path=private_path
        )

        assert _state_and_bool("2024-05-01", other_public_path, tmp_path) == (
            "key_mismatch"
        )

    def test_corrupt_signature_when_the_sig_bytes_are_damaged(self, tmp_path: Path):
        _write_day(tmp_path)
        private_path, public_path = _keypair_files(tmp_path.parent)
        audit.checkpoint_and_sign(
            "2024-05-01", audit_dir=tmp_path, key_path=private_path
        )
        sig_path = tmp_path / "2024-05-01.checkpoint.sig"
        intact = sig_path.read_text(encoding="utf-8")
        sig_path.write_text(intact[: len(intact) // 2], encoding="utf-8")

        assert _state_and_bool("2024-05-01", public_path, tmp_path) == (
            "corrupt_signature"
        )

    def test_content_drift_when_a_record_is_appended_after_signing(
        self, tmp_path: Path
    ):
        """The ordinary, innocent case that used to read as a compromise: a
        day is still being written to, so its final record hash moves past
        what was signed. The signature itself is untouched and still
        verifies."""
        record = _write_day(tmp_path)
        private_path, public_path = _keypair_files(tmp_path.parent)
        audit.checkpoint_and_sign(
            "2024-05-01", audit_dir=tmp_path, key_path=private_path
        )

        later = audit.DecisionRecord(
            request_id="r2",
            timestamp_utc="2024-05-01T00:01:00+00:00",
            tool_name="get_option_pricing",
            input={},
            cpp_available=False,
            duration_ms=1.0,
            status="ok",
        )
        later.prev_record_hash = record.record_hash
        later.record_hash = audit.hash_payload(
            {**later.model_dump(exclude={"record_hash"}), "record_hash": None}
        )
        with open(tmp_path / "2024-05-01.jsonl", "a", encoding="utf-8") as f:
            f.write(later.model_dump_json() + "\n")

        assert _state_and_bool("2024-05-01", public_path, tmp_path) == "content_drift"

    def test_unavailable_when_the_public_key_file_is_unreadable(self, tmp_path: Path):
        _write_day(tmp_path)
        private_path, _ = _keypair_files(tmp_path.parent)
        audit.checkpoint_and_sign(
            "2024-05-01", audit_dir=tmp_path, key_path=private_path
        )
        broken_key = tmp_path.parent / "broken.public"
        broken_key.write_bytes(b"this is not an ed25519 key")

        assert _state_and_bool("2024-05-01", broken_key, tmp_path) == "unavailable"

    def test_a_missing_public_key_file_is_also_unavailable(self, tmp_path: Path):
        _write_day(tmp_path)
        private_path, _ = _keypair_files(tmp_path.parent)
        audit.checkpoint_and_sign(
            "2024-05-01", audit_dir=tmp_path, key_path=private_path
        )

        assert (
            _state_and_bool("2024-05-01", tmp_path.parent / "no_such.public", tmp_path)
            == "unavailable"
        )

    def test_every_state_reached_here_is_one_of_the_seven(self, tmp_path: Path):
        """A state nobody declared is a state nobody can handle."""
        declared = {
            "valid",
            "no_checkpoint",
            "no_signature",
            "key_mismatch",
            "corrupt_signature",
            "content_drift",
            "unavailable",
        }
        _write_day(tmp_path)
        private_path, public_path = _keypair_files(tmp_path.parent)
        assert (
            audit.verify_checkpoint_state("2024-05-01", public_path, audit_dir=tmp_path)
            in declared
        )
        audit.checkpoint_and_sign(
            "2024-05-01", audit_dir=tmp_path, key_path=private_path
        )
        assert (
            audit.verify_checkpoint_state("2024-05-01", public_path, audit_dir=tmp_path)
            in declared
        )


# ── What the auditor bundle carries ───────────────────────────────────────────


class TestTheBundleCarriesTheSidecars:
    @pytest.mark.skipif(
        not audit.HAS_CRYPTOGRAPHY, reason="cryptography is not installed"
    )
    def test_a_signed_day_travels_with_its_checkpoint_and_signature(
        self, tmp_path: Path
    ):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        _write_day(audit_dir)
        private_path, _ = _keypair_files(tmp_path)
        audit.checkpoint_and_sign(
            "2024-05-01", audit_dir=audit_dir, key_path=private_path
        )
        out_path = tmp_path / "bundle.zip"

        audit.export_bundle("2024-05-01", "2024-05-01", out_path, audit_dir)

        with zipfile.ZipFile(out_path) as zf:
            names = set(zf.namelist())
            manifest = json.loads(zf.read("manifest.json"))
        assert "2024-05-01.checkpoint.json" in names
        assert "2024-05-01.checkpoint.sig" in names
        listed = {f["name"] for f in manifest["files"]}
        assert {"2024-05-01.checkpoint.json", "2024-05-01.checkpoint.sig"} <= listed

    def test_an_unsigned_day_exports_with_neither_and_without_complaint(
        self, tmp_path: Path
    ):
        """Signing is optional, so a day with no sidecars is the ordinary
        case and not a problem to report."""
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        _write_day(audit_dir)
        out_path = tmp_path / "bundle.zip"

        result = audit.export_bundle("2024-05-01", "2024-05-01", out_path, audit_dir)

        with zipfile.ZipFile(out_path) as zf:
            names = set(zf.namelist())
        assert not any("checkpoint" in name for name in names)
        assert "2024-05-01.jsonl" in names
        assert result.day_files == 1

    def test_the_bundled_readme_does_not_call_signing_unimplemented(
        self, tmp_path: Path
    ):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        _write_day(audit_dir)
        out_path = tmp_path / "bundle.zip"
        audit.export_bundle("2024-05-01", "2024-05-01", out_path, audit_dir)

        with zipfile.ZipFile(out_path) as zf:
            readme = zf.read("README.txt").decode("utf-8")
        assert "planned but not yet implemented" not in readme
        assert "checkpoint.sig" in readme
        assert "out of band" in readme


class TestTheBundleSaysHowMuchItHolds:
    def test_a_real_range_reports_its_days_and_records(self, tmp_path: Path):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        _write_day(audit_dir, "2024-05-01")
        _write_day(audit_dir, "2024-05-02")
        out_path = tmp_path / "bundle.zip"

        result = audit.export_bundle("2024-05-01", "2024-05-02", out_path, audit_dir)

        assert result.day_files == 2
        assert result.record_count == 2
        with zipfile.ZipFile(out_path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
        assert manifest["day_files"] == 2
        assert manifest["record_count"] == 2

    def test_a_range_covering_no_day_reports_zero(self, tmp_path: Path):
        """A bundle of nothing is a well-formed zip of a README, a manifest
        and a verifier — the same shape as a real one. The count is what
        tells them apart."""
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        _write_day(audit_dir, "2024-05-01")
        out_path = tmp_path / "bundle.zip"

        result = audit.export_bundle("2030-01-01", "2030-01-31", out_path, audit_dir)

        assert result.day_files == 0
        assert result.record_count == 0
        with zipfile.ZipFile(out_path) as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["day_files"] == 0
            assert not any(name.endswith("2024-05-01.jsonl") for name in zf.namelist())

    def test_the_result_still_passes_as_the_path_it_wrote(self, tmp_path: Path):
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        _write_day(audit_dir)
        out_path = tmp_path / "bundle.zip"

        result = audit.export_bundle("2024-05-01", "2024-05-01", out_path, audit_dir)

        assert Path(result) == out_path
        assert str(result) == str(out_path)
        assert result.path.exists()
