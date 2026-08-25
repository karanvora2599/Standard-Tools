"""
The five provenance tools: explain_decision, replay_decision,
compare_decisions, verify_audit_integrity, export_audit_bundle.

Every test here runs against a REAL audit log written into a tmp dir by
calling real tools, not against a hand-built record. The log's whole value
is that dispatch() writes it automatically and consistently; a fixture that
forged records would test the readers against a fiction and pass while the
writer drifted.

The verdict logic is the part worth the most attention. A backtest that
returns a different number later is not evidence of a bug -- the default
provider guarantees neither point-in-time values nor stable adjusted
prices -- so the tests below pin the distinction between the data moving
and the code moving, including the case where both are true.
"""

import json

import pytest

from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.audit.paths import _audit_dir, _iter_day_files
from standard_quant_tools.error import ValidationError


@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path, monkeypatch):
    """A fresh audit directory per test, so one test's records cannot make
    another's integrity check pass or fail."""
    monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
    return tmp_path / "audit"


def _record_a_call(**overrides) -> str:
    """Make a real, offline tool call and return its request id."""
    payload = {"strategy_type": "sma_crossover", **overrides}
    dispatch("list_strategies", payload)
    day_files = _iter_day_files(_audit_dir())
    assert day_files, "the call wrote no audit record"
    last = day_files[-1].read_text(encoding="utf-8").strip().split("\n")[-1]
    return json.loads(last)["request_id"]


class TestExplainDecision:
    def test_reports_the_call_that_was_made(self):
        request_id = _record_a_call()
        result = dispatch("explain_decision", {"request_id": request_id})
        assert result["tool_name"] == "list_strategies"
        assert result["status"] == "ok"
        assert result["input"]["strategy_type"] == "sma_crossover"

    def test_reports_the_execution_path(self):
        """The one field nothing else can reconstruct afterwards: the
        C++/Numba/Python choice is made at call time and falls back
        transparently."""
        result = dispatch("explain_decision", {"request_id": _record_a_call()})
        assert result["execution_path"] in {"C++", "Python/Numba"}

    def test_carries_the_code_state_it_ran_under(self):
        result = dispatch("explain_decision", {"request_id": _record_a_call()})
        assert result["package_version"]
        assert result["record_hash"]

    def test_an_unknown_id_is_a_caller_error(self):
        with pytest.raises(ValidationError):
            dispatch("explain_decision", {"request_id": "0" * 32})


class TestReplayDecision:
    def test_an_offline_tool_reproduces(self):
        """list_strategies reads no market data, so nothing can have moved
        underneath it. Anything but 'reproduced' here is the library
        disagreeing with itself."""
        result = dispatch("replay_decision", {"request_id": _record_a_call()})
        assert result["verdict"] == "reproduced"
        assert result["output_match"] is True

    def test_code_changed_when_inputs_hold_but_output_moves(self, monkeypatch):
        """The only verdict that implicates the library. Forced here by
        making the replayed call return something else while its inputs are
        untouched."""
        request_id = _record_a_call()

        real = dispatch.__globals__["_verify_replay"]

        def _mismatch(record):
            result = real(record)
            result.output_match = False
            result.data_source_matches = []
            return result

        monkeypatch.setitem(dispatch.__globals__, "_verify_replay", _mismatch)
        result = dispatch("replay_decision", {"request_id": request_id})
        assert result["verdict"] == "code_changed"
        assert any("points at the code" in note for note in result["notes"])

    def test_data_changed_takes_priority_over_code_changed(self, monkeypatch):
        """When an input hash no longer matches, a different output is
        EXPECTED. Reporting that as a code problem would turn every
        provider revision into a false bug report."""
        request_id = _record_a_call()
        real = dispatch.__globals__["_verify_replay"]

        def _data_moved(record):
            result = real(record)
            result.output_match = False
            result.data_source_matches = [
                {
                    "symbol": "AAPL",
                    "start": "2024-01-01",
                    "end": "2024-06-01",
                    "interval": "1d",
                    "match": False,
                }
            ]
            return result

        monkeypatch.setitem(dispatch.__globals__, "_verify_replay", _data_moved)
        result = dispatch("replay_decision", {"request_id": request_id})
        assert result["verdict"] == "data_changed"
        assert any("says nothing about the library" in n for n in result["notes"])

    def test_a_missing_output_hash_is_not_comparable_not_a_mismatch(self, monkeypatch):
        request_id = _record_a_call()
        real = dispatch.__globals__["_verify_replay"]

        def _uncomparable(record):
            result = real(record)
            result.output_match = None
            return result

        monkeypatch.setitem(dispatch.__globals__, "_verify_replay", _uncomparable)
        result = dispatch("replay_decision", {"request_id": request_id})
        assert result["verdict"] == "not_comparable"

    def test_a_replay_that_cannot_run_reports_failed(self, monkeypatch):
        request_id = _record_a_call()

        def _explode(record):
            raise RuntimeError("provider unreachable")

        monkeypatch.setitem(dispatch.__globals__, "_verify_replay", _explode)
        result = dispatch("replay_decision", {"request_id": request_id})
        assert result["verdict"] == "failed"
        assert "provider unreachable" in result["notes"][0]


class TestCompareDecisions:
    def test_identical_calls_are_reproductions(self):
        first = _record_a_call()
        second = _record_a_call()
        result = dispatch(
            "compare_decisions", {"request_id_a": first, "request_id_b": second}
        )
        assert result["same_tool"] and result["same_input"] and result["same_output"]
        assert any("reproductions of each other" in s for s in result["summary"])

    def test_different_inputs_explain_a_different_output(self):
        first = _record_a_call(strategy_type="sma_crossover")
        second = _record_a_call(strategy_type="rsi_mean_reversion")
        result = dispatch(
            "compare_decisions", {"request_id_a": first, "request_id_b": second}
        )
        assert result["same_input"] is False
        assert any("different inputs" in s for s in result["summary"])

    def test_comparing_a_record_with_itself_is_rejected(self):
        request_id = _record_a_call()
        with pytest.raises(Exception) as exc:
            dispatch(
                "compare_decisions",
                {"request_id_a": request_id, "request_id_b": request_id},
            )
        assert "same record" in str(exc.value)


class TestVerifyAuditIntegrity:
    def test_a_clean_trail_verifies(self):
        _record_a_call()
        result = dispatch("verify_audit_integrity", {})
        assert result["intact"] is True
        assert result["problems"] == []

    def test_an_edited_record_breaks_the_chain(self):
        """The property the whole log rests on: changing a past line
        invalidates every line after it."""
        _record_a_call()
        _record_a_call()
        _record_a_call()

        day_file = _iter_day_files(_audit_dir())[-1]
        lines = day_file.read_text(encoding="utf-8").strip().split("\n")
        tampered = json.loads(lines[0])
        tampered["duration_ms"] = 999999.0
        lines[0] = json.dumps(tampered)
        day_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = dispatch("verify_audit_integrity", {})
        assert result["intact"] is False
        assert result["problems"]

    def test_a_single_day_check_says_what_it_cannot_see(self):
        """Verifying one file in isolation cannot detect a deleted day.
        Saying so is the difference between a check and a false assurance."""
        _record_a_call()
        date = _iter_day_files(_audit_dir())[-1].stem
        result = dispatch("verify_audit_integrity", {"date": date})
        assert result["scope"] == date
        assert any("MISSING" in note for note in result["notes"])

    def test_a_chain_only_check_states_its_limit(self):
        _record_a_call()
        date = _iter_day_files(_audit_dir())[-1].stem
        result = dispatch("verify_audit_integrity", {"date": date})
        assert result["checkpoint_signature_valid"] is None
        assert any("wholesale rewrite" in note for note in result["notes"])

    def test_an_unknown_date_is_rejected(self):
        _record_a_call()
        with pytest.raises(ValidationError) as exc:
            dispatch("verify_audit_integrity", {"date": "1999-01-01"})
        assert "no audit file" in str(exc.value)

    def test_a_malformed_date_never_reaches_the_filesystem(self):
        """The date is LLM-reachable and is joined into a path."""
        _record_a_call()
        with pytest.raises(ValidationError) as exc:
            dispatch("verify_audit_integrity", {"date": "../../etc/passwd"})
        assert "YYYY-MM-DD" in str(exc.value)

    def test_a_public_key_without_a_date_is_rejected(self):
        with pytest.raises(Exception) as exc:
            dispatch("verify_audit_integrity", {"public_key_path": "key.pub"})
        assert "signed per" in str(exc.value)


class TestExportAuditBundle:
    def test_writes_a_bundle_and_rewrites_no_existing_record(self, tmp_path):
        """The export call is itself audited, so the day file GROWS -- what
        must not happen is an existing line changing. Appending is how the
        log works; rewriting is what it exists to detect."""
        _record_a_call()
        day_file = _iter_day_files(_audit_dir())[-1]
        before = day_file.read_text(encoding="utf-8")
        date = day_file.stem

        out = tmp_path / "bundle.zip"
        result = dispatch(
            "export_audit_bundle",
            {"start_date": date, "end_date": date, "out_path": str(out)},
        )
        assert out.exists() and result["size_bytes"] > 0

        after = day_file.read_text(encoding="utf-8")
        assert after.startswith(before), "export rewrote an existing record"
        assert dispatch("verify_audit_integrity", {})["intact"] is True

    def test_says_that_verifying_a_copy_is_not_verifying_the_log(self, tmp_path):
        _record_a_call()
        date = _iter_day_files(_audit_dir())[-1].stem
        result = dispatch(
            "export_audit_bundle",
            {
                "start_date": date,
                "end_date": date,
                "out_path": str(tmp_path / "b.zip"),
            },
        )
        assert any("The bundle is a copy" in note for note in result["notes"])

    def test_a_reversed_range_is_rejected(self, tmp_path):
        with pytest.raises(Exception) as exc:
            dispatch(
                "export_audit_bundle",
                {
                    "start_date": "2026-02-01",
                    "end_date": "2026-01-01",
                    "out_path": str(tmp_path / "b.zip"),
                },
            )
        assert "precedes" in str(exc.value)


class TestRetentionStaysOffTheToolSurface:
    def test_no_tool_can_delete_seal_or_hold_a_record(self):
        """The deliberate omission. An agent that can destroy the record of
        its own decisions is not audited by it, so these stay CLI-only."""
        from standard_quant_tools.agent.tools import _TOOL_DISPATCH

        # Exact names, not substrings: run_buy_and_hold is a backtest, and
        # a substring rule would forbid it while catching nothing real.
        forbidden = {
            "gc_audit_log",
            "seal_audit_day",
            "hold_audit_day",
            "release_audit_hold",
            "generate_audit_keypair",
            "anchor_audit_day",
        }
        assert forbidden.isdisjoint(_TOOL_DISPATCH)

        # And the retention functions themselves must stay unreachable
        # through dispatch, whatever a future tool might be named.
        from standard_quant_tools.audit import retention

        reachable = {fn for fn, _model in _TOOL_DISPATCH.values()}
        for name in ("gc", "seal_day", "hold_day", "release_hold"):
            assert getattr(retention, name) not in reachable, name
