"""
Regression tests for Pass 5 of the full-codebase audit: audit write policy
and replay honesty.

The theme is that an audit trail must be honest about its own limits — about
whether a record was written, about whether a record can be replayed at all,
and about what a failed call reproducing actually means.
"""

import uuid
from datetime import datetime, timezone

import pytest

from standard_quant_tools.error import AuditIntegrityError, ValidationError


class TestAuditWritePolicy:
    """
    Fail-open is the right DEFAULT for an analytics library — a full disk
    should not destroy a legitimate result the caller already paid to
    compute. It is not right under a governance regime, where an action taken
    without a record of it is exactly what the trail exists to prevent.
    """

    def test_fail_open_is_the_default(self, monkeypatch):
        from standard_quant_tools.audit.dispatch import _audit_fail_closed

        monkeypatch.delenv("SQT_AUDIT_FAIL_CLOSED", raising=False)
        assert _audit_fail_closed() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
    def test_fail_closed_is_selectable(self, monkeypatch, value):
        from standard_quant_tools.audit.dispatch import _audit_fail_closed

        monkeypatch.setenv("SQT_AUDIT_FAIL_CLOSED", value)
        assert _audit_fail_closed() is True

    @pytest.mark.parametrize("value", ["0", "false", "", "no"])
    def test_other_values_stay_fail_open(self, monkeypatch, value):
        from standard_quant_tools.audit.dispatch import _audit_fail_closed

        monkeypatch.setenv("SQT_AUDIT_FAIL_CLOSED", value)
        assert _audit_fail_closed() is False

    def test_integrity_errors_are_never_swallowed(self):
        """
        The interaction that matters. The writer refuses to extend a chain
        whose tail it cannot read (Pass 1), and the dispatch wrapper catches
        `Exception` broadly around the write — so without an explicit
        passthrough, that refusal would have been logged as an ordinary write
        failure and the corruption would stay invisible, which is precisely
        the state the writer's check exists to prevent.

        Asserted structurally: AuditIntegrityError must be re-raised before
        the broad handler, and must not be a subclass of anything the
        fail-open branch would catch first.
        """
        import inspect

        from standard_quant_tools.audit import dispatch

        source = inspect.getsource(dispatch._run_and_record)
        integrity_at = source.index("except AuditIntegrityError")
        broad_at = source.index("except Exception:", integrity_at)
        assert integrity_at < broad_at, (
            "AuditIntegrityError must be handled BEFORE the broad handler, "
            "or corruption is swallowed as a write failure"
        )
        assert "raise" in source[integrity_at:broad_at]

    def test_integrity_error_is_not_a_validation_error(self):
        """It is not a statement about the caller's input, so it must not be
        caught by handlers written for bad input."""
        assert not issubclass(AuditIntegrityError, ValidationError)


class TestRedactionAndReplayAreInTension:
    """
    The record stores `_redact(raw_input, fields)`, so a redacted field holds
    a placeholder rather than the original value. Reconstructing the call from
    it would run a DIFFERENT call and then compare the result against the
    original's hash — a guaranteed mismatch that reads as evidence of drift
    rather than as the artefact of redaction it actually is.
    """

    def test_redacted_fields_are_detected(self):
        from standard_quant_tools.audit.replay import _redacted_input_fields

        record_input = {
            "symbol": "AAPL",
            "account_id": "<redacted:a1b2c3d4>",
            "nested": {"secret": "<redacted:deadbeef>"},
        }
        found = _redacted_input_fields(record_input)
        assert "account_id" in found
        assert any("secret" in f for f in found)

    def test_a_clean_input_reports_nothing_redacted(self):
        from standard_quant_tools.audit.replay import _redacted_input_fields

        assert _redacted_input_fields({"symbol": "AAPL", "period": "1y"}) == []

    def test_replaying_a_redacted_record_is_refused(self):
        from standard_quant_tools.audit.replay import verify_replay

        record = {
            "request_id": str(uuid.uuid4()),
            "tool_name": "get_technical_analysis",
            "input": {"symbol": "<redacted:a1b2c3d4>"},
            "status": "ok",
        }
        with pytest.raises(ValidationError, match="not replayable"):
            verify_replay(record)

    def test_the_refusal_explains_the_tension(self):
        """A caller should learn that the record needs one or the other, not
        merely that this one failed."""
        from standard_quant_tools.audit.replay import verify_replay

        record = {
            "request_id": "r1",
            "tool_name": "get_technical_analysis",
            "input": {"symbol": "<redacted:a1b2c3d4>"},
            "status": "ok",
        }
        with pytest.raises(ValidationError, match="redact"):
            verify_replay(record)


class TestFailedCallsReplayAsFirstClassOutcomes:
    """
    A call that failed originally is an outcome, not the absence of one.
    Letting the replay exception escape reported an error in the replay
    machinery, when what actually reproduced was the original failure — which
    is a SUCCESSFUL replay.
    """

    def _record(self, error_type, status="error"):
        return {
            "request_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "tool_name": "get_technical_analysis",
            # an input this tool's own schema rejects, so the replay raises
            # deterministically without needing a network provider
            "input": {"symbol": "AAPL", "indicators": ["not_a_real_indicator"]},
            "status": status,
            "error_type": error_type,
        }

    def test_same_failure_reproducing_is_a_successful_replay(self):
        from standard_quant_tools.audit.replay import verify_replay

        # Provoke a real failure by handing the tool an unparseable date; the
        # exact exception type is discovered, then asserted as reproduced.
        probe = self._record("PLACEHOLDER")
        try:
            verify_replay(probe)
            pytest.skip("input did not fail; cannot exercise the failure path")
        except ValidationError:
            raise
        except Exception as exc:
            actual = type(exc).__name__

        result = verify_replay(self._record(actual))
        assert result.output_match is True
        assert any("reproduced" in n for n in result.notes)

    def test_a_different_failure_is_reported_as_a_change(self):
        from standard_quant_tools.audit.replay import verify_replay

        result = verify_replay(self._record("SomeOtherErrorType"))
        assert result.output_match is False
        assert any("DIFFERENT failure" in n for n in result.notes)
