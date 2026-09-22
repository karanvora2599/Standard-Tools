"""
describe_audit_log and find_decisions: the two doors into the decision log.

WHY THESE TWO EXIST is the thing worth pinning. Every other provenance
tool takes something the caller already has to possess -- a request id, a
date range -- and `dispatch()` returns the payload alone, so an in-process
caller never saw an id and three of the five tools could not be reached at
all. A test that only checked the fields would pass while that hole stayed
open, so the tests here follow the chain: record real calls, find their
ids, and hand one to `explain_decision`.

The other half is configuration. A record count of zero means something
entirely different under recording-off than under recording-on, and the
seven settings that decide which it is were reported by nothing. They are
pinned here against a monkeypatched environment -- including the one that
must never be reported at all.

Every test runs against a REAL log written by real dispatches into a
throwaway directory. A fixture that forged records would test these
readers against a fiction and keep passing while the writer drifted.
"""

import json
import os
import stat

import pytest

from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.audit.paths import _audit_dir, _iter_day_files
from standard_quant_tools.audit.retention import hold_day, release_hold, seal_day
from standard_quant_tools.error import ValidationError

#: A planted day far enough in the past that any retention window at all
#: makes it a candidate, and far enough from today that no real record can
#: land in it.
OLD_DAY = "2020-01-01"

#: The salt is the one setting that must never cross the boundary. A
#: distinctive string so a substring search over the whole serialized
#: result is conclusive.
SALT = "sentinel-salt-value-9f3a2b"


@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path, monkeypatch):
    """A fresh decision log per test, so one test's records cannot answer
    another's question."""
    directory = tmp_path / "audit"
    monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
    for name in (
        "SQT_AUDIT_FAIL_CLOSED",
        "SQT_AUDIT_REDACT_FIELDS",
        "SQT_AUDIT_REDACT_SALT",
        "SQT_AUDIT_RETENTION_DAYS",
        "SQT_AUDIT_SIGNING_KEY_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    yield directory
    # A sealed day file is read-only, and a read-only file in a temp tree
    # is a cleanup failure on Windows rather than a test failure here.
    for path in directory.glob("*.jsonl"):
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        except OSError:  # pragma: no cover - best effort
            pass


def _record_a_call(**overrides) -> str:
    """One real, offline dispatch; its request id."""
    dispatch("list_strategies", {"strategy_type": "sma_crossover", **overrides})
    day_files = _iter_day_files(_audit_dir())
    assert day_files, "the call wrote no audit record"
    last = day_files[-1].read_text(encoding="utf-8").strip().split("\n")[-1]
    return json.loads(last)["request_id"]


def _record_a_failure() -> None:
    """A call that raises. The record is written either way, and nothing
    could read one back before find_decisions."""
    with pytest.raises(ValidationError):
        dispatch("explain_decision", {"request_id": "0" * 32})


def _plant_an_old_day(directory) -> str:
    """A day file dated in the past, so a retention window has something to
    be a candidate."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{OLD_DAY}.jsonl"
    path.write_text(
        json.dumps(
            {
                "request_id": "a" * 32,
                "timestamp_utc": f"{OLD_DAY}T00:00:00+00:00",
                "tool_name": "list_strategies",
                "status": "ok",
                "duration_ms": 1.0,
                # The writer chains a new day onto the previous day's last
                # record_hash and refuses to append when there is none, so
                # a planted day carries one or every later write fails.
                "prev_record_hash": "0" * 16,
                "record_hash": "f" * 16,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return OLD_DAY


class TestTheSettingsThatDecideWhatAnEmptyLogMeans:
    def test_all_seven_audit_settings_are_reported(self, monkeypatch, tmp_path):
        monkeypatch.setenv("SQT_AUDIT_FAIL_CLOSED", "1")
        monkeypatch.setenv("SQT_AUDIT_REDACT_FIELDS", "account_id,client.ssn")
        monkeypatch.setenv("SQT_AUDIT_REDACT_SALT", SALT)
        monkeypatch.setenv("SQT_AUDIT_RETENTION_DAYS", "30")
        monkeypatch.setenv("SQT_AUDIT_SIGNING_KEY_PATH", str(tmp_path / "key"))

        result = dispatch("describe_audit_log", {})

        assert result["audit_dir"] == str(_audit_dir())
        assert result["recording_enabled"] is True
        assert result["fail_closed"] is True
        assert result["redacted_fields"] == ["account_id", "client.ssn"]
        assert result["redaction_salt_set"] is True
        assert result["retention_days"] == 30
        assert result["signing_configured"] is True
        assert isinstance(result["signing_available"], bool)

    def test_the_redaction_salt_never_appears_in_the_result(self, monkeypatch):
        """The salt exists to make a redaction placeholder unrecoverable,
        so a tool that printed it would undo the redaction it reports."""
        monkeypatch.setenv("SQT_AUDIT_REDACT_SALT", SALT)
        monkeypatch.setenv("SQT_AUDIT_REDACT_FIELDS", "account_id")
        _record_a_call()

        result = dispatch("describe_audit_log", {"include_days": True})

        assert result["redaction_salt_set"] is True
        assert SALT not in json.dumps(result)

    def test_the_defaults_are_reported_when_nothing_is_set(self):
        result = dispatch("describe_audit_log", {})
        assert result["recording_enabled"] is True
        assert result["fail_closed"] is False
        assert result["retention_days"] is None
        assert result["redacted_fields"] == []
        assert result["redaction_salt_set"] is False
        assert result["signing_configured"] is False

    def test_recording_off_is_said_rather_than_shown_as_an_empty_trail(
        self, monkeypatch
    ):
        """Zero records under recording-off is 'nothing was recorded', not
        'nothing happened' -- see the CHANGELOG entry of 2026-09-22."""
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "0")
        result = dispatch("describe_audit_log", {})
        assert result["recording_enabled"] is False
        assert any("Recording is OFF" in w for w in result["warnings"])

    def test_an_unsalted_redaction_is_warned_about(self, monkeypatch):
        monkeypatch.setenv("SQT_AUDIT_REDACT_FIELDS", "account_id")
        result = dispatch("describe_audit_log", {})
        assert any("brute-forceable" in w for w in result["warnings"])


class TestWhatTheTrailHolds:
    def test_the_records_that_were_written_are_counted(self):
        _record_a_call()
        _record_a_call()
        result = dispatch("describe_audit_log", {})
        assert result["days"] == 1
        # This call's own record is appended AFTER the tool returns, so
        # the two it counts are the two that preceded it.
        assert result["total_records"] >= 2
        assert result["total_bytes"] > 0
        assert result["oldest_date"] == result["newest_date"]

    def test_a_capped_listing_does_not_change_the_totals(self, isolated_audit_log):
        """A listing that had quietly narrowed the counts above it would be
        worse than no listing, because it reads as authoritative."""
        _record_a_call()
        _plant_an_old_day(isolated_audit_log)

        capped = dispatch("describe_audit_log", {"include_days": True, "max_days": 1})

        assert capped["days"] == 2
        assert len(capped["day_summaries"]) == 1
        listed = sum(day["records"] for day in capped["day_summaries"])
        assert capped["total_records"] > listed
        assert capped["oldest_date"] == OLD_DAY
        assert capped["day_summaries"][0]["date"] == capped["newest_date"]
        assert any("newest are listed" in note for note in capped["notes"])

        full = dispatch("describe_audit_log", {"include_days": True})
        assert len(full["day_summaries"]) == 2

    def test_the_per_day_breakdown_spans_the_days_records(self):
        _record_a_call()
        result = dispatch("describe_audit_log", {"include_days": True})
        day = result["day_summaries"][0]
        assert day["records"] >= 1
        assert day["bytes"] > 0
        assert day["first_utc"] and day["last_utc"]
        assert day["first_utc"] <= day["last_utc"]

    def test_summary_only_omits_the_per_day_detail(self):
        _record_a_call()
        result = dispatch("describe_audit_log", {})
        assert result["day_summaries"] == []
        assert result["days"] == 1


class TestHoldsSealsAndSignaturesAreReportedNeverPlaced:
    def test_a_held_day_is_flagged_and_is_not_a_candidate(
        self, isolated_audit_log, monkeypatch
    ):
        date = _plant_an_old_day(isolated_audit_log)
        monkeypatch.setenv("SQT_AUDIT_RETENTION_DAYS", "1")
        hold_day(date, isolated_audit_log, reason="an open matter")

        result = dispatch("describe_audit_log", {"include_days": True})

        held = [d for d in result["day_summaries"] if d["date"] == date]
        assert held and held[0]["held"] is True
        assert date not in result["gc_candidate_dates"]

    def test_a_released_day_reappears_as_a_candidate(
        self, isolated_audit_log, monkeypatch
    ):
        """The release goes through the library, not through a tool: lifting
        a hold is the precondition for a deletion and stays off the tool
        surface."""
        date = _plant_an_old_day(isolated_audit_log)
        monkeypatch.setenv("SQT_AUDIT_RETENTION_DAYS", "1")
        hold_day(date, isolated_audit_log)
        assert date not in dispatch("describe_audit_log", {})["gc_candidate_dates"]

        assert release_hold(date, isolated_audit_log) is True

        result = dispatch("describe_audit_log", {"include_days": True})
        assert date in result["gc_candidate_dates"]
        released = [d for d in result["day_summaries"] if d["date"] == date]
        assert released and released[0]["held"] is False

    def test_no_candidate_without_a_retention_window(self, isolated_audit_log):
        _plant_an_old_day(isolated_audit_log)
        result = dispatch("describe_audit_log", {})
        assert result["gc_candidate_dates"] == []
        assert any(
            "nothing is ever a deletion candidate" in w for w in result["warnings"]
        )

    def test_the_preview_never_removes_the_file(self, isolated_audit_log, monkeypatch):
        date = _plant_an_old_day(isolated_audit_log)
        monkeypatch.setenv("SQT_AUDIT_RETENTION_DAYS", "1")
        result = dispatch("describe_audit_log", {})
        assert date in result["gc_candidate_dates"]
        assert (isolated_audit_log / f"{date}.jsonl").exists()

    def test_a_sealed_day_is_flagged(self, isolated_audit_log):
        date = _plant_an_old_day(isolated_audit_log)
        seal_day(date, isolated_audit_log)

        result = dispatch("describe_audit_log", {"include_days": True})

        sealed = [d for d in result["day_summaries"] if d["date"] == date]
        assert sealed and sealed[0]["sealed"] is True
        today = [d for d in result["day_summaries"] if d["date"] != date]
        assert all(d["sealed"] is False for d in today)

    def test_a_signed_day_is_flagged(self, isolated_audit_log, tmp_path):
        from standard_quant_tools.audit import signing

        if not signing.HAS_CRYPTOGRAPHY:
            pytest.skip("Ed25519 checkpoint signing needs the cryptography package")

        _record_a_call()
        date = _iter_day_files(_audit_dir())[-1].stem
        private_key, _public_key = signing.generate_keypair()
        key_path = tmp_path / "audit.key"
        key_path.write_bytes(private_key)
        signing.checkpoint_and_sign(
            date, key_path=key_path, audit_dir=isolated_audit_log
        )

        result = dispatch("describe_audit_log", {"include_days": True})

        signed = [d for d in result["day_summaries"] if d["date"] == date]
        assert signed and signed[0]["checkpoint_signed"] is True

    def test_an_unsigned_trail_says_what_the_chain_cannot_prove(self):
        _record_a_call()
        result = dispatch("describe_audit_log", {"include_days": True})
        assert any("wholesale rewrite" in w for w in result["warnings"])


class TestFindDecisionsIsTheDoorToTheRequestId:
    def test_a_recorded_error_is_found_and_ok_excludes_it(self):
        _record_a_failure()

        errors = dispatch("find_decisions", {"status": "error"})
        assert errors["n_matches"] >= 1
        assert all(m["status"] == "error" for m in errors["matches"])
        assert any(m["tool_name"] == "explain_decision" for m in errors["matches"])
        assert any(m["error_type"] for m in errors["matches"])

        ok = dispatch("find_decisions", {"status": "ok"})
        assert all(m["status"] == "ok" for m in ok["matches"])
        assert not any(
            m["request_id"] == errors["matches"][0]["request_id"] for m in ok["matches"]
        )

    def test_the_returned_id_feeds_explain_decision(self):
        _record_a_call()
        found = dispatch("find_decisions", {"tool_name": "list_strategies"})
        assert found["n_matches"] >= 1

        explained = dispatch(
            "explain_decision", {"request_id": found["matches"][0]["request_id"]}
        )
        assert explained["tool_name"] == "list_strategies"

    def test_a_tool_filter_matches_exactly(self):
        _record_a_call()
        found = dispatch("find_decisions", {"tool_name": "list_strategies"})
        assert all(m["tool_name"] == "list_strategies" for m in found["matches"])

        none = dispatch("find_decisions", {"tool_name": "list_strategie"})
        assert none["n_matches"] == 0
        assert any("does not" in w or "not a tool" in w for w in none["warnings"])

    def test_the_date_filters_skip_whole_files(self, isolated_audit_log):
        _record_a_call()
        _plant_an_old_day(isolated_audit_log)

        narrowed = dispatch("find_decisions", {"start_date": "2024-01-01"})
        assert narrowed["days_skipped"] == 1
        assert narrowed["days_scanned"] == 1
        assert all(m["timestamp_utc"] >= "2024" for m in narrowed["matches"])

        outside = dispatch(
            "find_decisions", {"start_date": "2019-01-01", "end_date": "2019-12-31"}
        )
        assert outside["n_matches"] == 0
        assert outside["total_scanned"] == 0
        assert any("range" in w for w in outside["warnings"])

    def test_the_limit_truncates(self):
        for _ in range(3):
            _record_a_call()
        result = dispatch("find_decisions", {"limit": 1})
        assert result["n_matches"] == 1
        assert result["truncated"] is True

        whole = dispatch("find_decisions", {"limit": 500})
        assert whole["truncated"] is False
        assert whole["scan_complete"] is True
        assert whole["n_matches"] >= 3

    def test_newest_first_is_the_default_and_can_be_reversed(self):
        first = _record_a_call()
        _record_a_call()
        newest = dispatch("find_decisions", {"tool_name": "list_strategies"})
        oldest = dispatch(
            "find_decisions",
            {"tool_name": "list_strategies", "newest_first": False},
        )
        assert oldest["matches"][0]["request_id"] == first
        assert newest["matches"][0]["request_id"] != first

    def test_a_malformed_date_is_refused_by_name(self):
        with pytest.raises(ValidationError) as exc:
            dispatch("find_decisions", {"start_date": "01/01/2024"})
        assert "YYYY-MM-DD" in str(exc.value)

    def test_a_reversed_range_is_refused(self):
        with pytest.raises(ValidationError) as exc:
            dispatch(
                "find_decisions",
                {"start_date": "2026-02-01", "end_date": "2026-01-01"},
            )
        assert "after" in str(exc.value)

    def test_the_result_carries_no_request_id_of_its_own(self):
        """dispatch() returns the payload and nothing else; the id belongs
        to the record, and find_decisions is how one is obtained."""
        _record_a_call()
        result = dispatch("find_decisions", {})
        assert "request_id" not in result
        assert all("request_id" in m for m in result["matches"])
