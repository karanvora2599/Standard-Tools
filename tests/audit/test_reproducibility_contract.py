"""
What an output hash promises, and what replay says when it is not kept.

An output hash is bit-exact for the same native build on the same
instruction-set path. Across builds or paths the outputs agree to twelve
significant digits and no further: the AVX2+FMA reduction fuses each
multiply-add and sums in four lanes, so it rounds differently from the
scalar loop (measured by the C++ suite on rolling_beta), and a different
compiler or OpenMP runtime may move the last bits too.

Replay compared the exact hash alone, so the same answer replayed on
another build or another CPU read "code_changed". Each new record now also
carries `output_hash_rounded` (floats rounded to twelve significant digits)
and `native_isa`, and replay tells the cases apart:

  same build, exact match                 -> reproduced
  different build or path, rounded match  -> reproduced_to_12_digits
  same build and path, exact miss         -> today's verdict, unchanged
  a record from before the rounded hash   -> today's verdict, with a note

The rounding is defined exactly, and its edges are pinned first.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from pydantic import BaseModel

from standard_quant_tools import _native_build, audit
from standard_quant_tools.agent import tools as agent_tools
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.audit import provenance
from standard_quant_tools.audit.dispatch import _run_and_record
from standard_quant_tools.audit.hashing import hash_payload, round_floats
from standard_quant_tools.audit.replay import normalize_identifiers
from standard_quant_tools.audit.writer import AuditWriter

# ── The rounding ─────────────────────────────────────────────────────────────


class TestRoundFloats:
    def test_twelve_significant_digits_at_any_magnitude(self):
        assert round_floats(1.23456789012345) == 1.23456789012
        assert round_floats(1234.56789012345) == 1234.56789012
        assert round_floats(1.23456789012345e-9) == 1.23456789012e-9
        assert round_floats(-9.87654321098765e20) == -9.87654321099e20

    def test_an_exact_tie_rounds_to_even(self):
        """Both are exactly representable, so the twelfth digit is a true
        tie: 2 stays, 3 goes up to the even 4."""
        assert round_floats(1234567890125.0) == 1234567890120.0
        assert round_floats(1234567890135.0) == 1234567890140.0

    def test_negative_zero_becomes_zero(self):
        rounded = round_floats(-0.0)
        assert rounded == 0.0 and math.copysign(1.0, rounded) == 1.0
        assert hash_payload(round_floats({"x": -0.0})) == hash_payload({"x": 0.0})
        # The exact hash does tell them apart, which is the point.
        assert hash_payload({"x": -0.0}) != hash_payload({"x": 0.0})

    def test_nan_and_infinities_are_kept(self):
        assert math.isnan(round_floats(float("nan")))
        assert round_floats(math.inf) == math.inf
        assert round_floats(-math.inf) == -math.inf

    def test_integers_and_booleans_are_untouched(self):
        big = 12345678901234567  # more digits than any float keeps
        assert round_floats(big) == big and type(round_floats(big)) is int
        assert round_floats(True) is True
        assert round_floats(np.int64(big)) == big

    def test_strings_none_and_keys_are_untouched(self):
        payload = {"1.23456789012345": "1.23456789012345", "none": None}
        assert round_floats(payload) == payload

    def test_nested_containers_are_rounded_at_any_depth(self):
        payload = {"a": [1.0000000000001, {"b": (2.00000000000001, 3)}]}
        assert round_floats(payload) == {"a": [1.0, {"b": [2.0, 3]}]}

    def test_numpy_values_are_converted_first(self):
        assert round_floats(np.array([1.23456789012345, 2.0])) == [
            1.23456789012,
            2.0,
        ]
        assert round_floats(np.float64(1.23456789012345)) == 1.23456789012

    def test_a_last_bit_difference_hashes_the_same(self):
        x = 0.1 + 0.2
        y = math.nextafter(x, 1.0)
        assert hash_payload(x) != hash_payload(y)
        assert hash_payload(round_floats(x)) == hash_payload(round_floats(y))

    def test_a_difference_in_the_eleventh_digit_does_not(self):
        """Null case: the rounded hash forgives the last bits and nothing
        more."""
        x = 1.2345678901
        assert hash_payload(round_floats(x)) != hash_payload(round_floats(x + 1e-10))


# ── The record ───────────────────────────────────────────────────────────────


class _ProbeInput(BaseModel):
    window: int = 60


class _ProbeOutput(BaseModel):
    beta: float
    betas: List[float]
    n_obs: int


TOOL = "probe_rolling_beta"
HERE_BUILD = "match:aaaaaaaaaaaa"
HERE_ISA = "avx2+fma"

#: What the recorded call returned.
ORIGINAL: Dict[str, Any] = {
    "beta": 1.1043287719382044,
    "betas": [0.9831234567890123, 1.0123456789012345],
    "n_obs": 2461,
}


def _last_bits_moved(output: Dict[str, Any]) -> Dict[str, Any]:
    """The same answer from different arithmetic: every float one unit in
    the last place away, as the AVX2+FMA and scalar reductions differ."""
    return {
        **output,
        "beta": math.nextafter(output["beta"], math.inf),
        "betas": [math.nextafter(v, -math.inf) for v in output["betas"]],
    }


def _ninth_digit_moved(output: Dict[str, Any]) -> Dict[str, Any]:
    return {**output, "beta": output["beta"] * (1.0 + 1e-9)}


@pytest.fixture
def audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "audit"
    monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
    return directory


@pytest.fixture
def replay_returns(monkeypatch: pytest.MonkeyPatch):
    """Register the probe tool and choose what its replay returns. Replay
    runs here on build HERE_BUILD and path HERE_ISA."""
    state: Dict[str, Any] = {"output": dict(ORIGINAL)}

    def tool(model: _ProbeInput) -> _ProbeOutput:
        return _ProbeOutput(**state["output"])

    monkeypatch.setitem(agent_tools._TOOL_DISPATCH, TOOL, (tool, _ProbeInput))
    monkeypatch.setattr(provenance, "_native_build_label", lambda: HERE_BUILD)
    monkeypatch.setattr(provenance, "_native_isa_label", lambda: HERE_ISA)

    def choose(output: Dict[str, Any]) -> None:
        state["output"] = output

    return choose


def _planted(
    request_id: str = "r1",
    native_build: Any = HERE_BUILD,
    native_isa: Any = HERE_ISA,
    with_rounded: bool = True,
) -> Dict[str, Any]:
    """A record of ORIGINAL as the writer stores it; a field passed as None
    is absent, as on a record written before it existed."""
    record: Dict[str, Any] = {
        "request_id": request_id,
        "timestamp_utc": "2026-10-01T00:00:00+00:00",
        "tool_name": TOOL,
        "input": {"window": 60},
        "data_sources": [],
        "cpp_available": True,
        "duration_ms": 1.0,
        "status": "ok",
        "output_hash": hash_payload(ORIGINAL),
        "output_hash_normalized": hash_payload(normalize_identifiers(ORIGINAL)),
    }
    if with_rounded:
        record["output_hash_rounded"] = hash_payload(
            round_floats(normalize_identifiers(ORIGINAL))
        )
    if native_build is not None:
        record["native_build"] = native_build
    if native_isa is not None:
        record["native_isa"] = native_isa
    return record


class TestANewRecordCarriesTheContract:
    def test_the_rounded_hash_and_the_isa_path_are_written(
        self, audit_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The planted records below are built the way the writer builds
        one; this is the check that they are."""
        monkeypatch.setattr(_native_build, "native_isa", lambda: "scalar")
        _run_and_record(TOOL, lambda m: _ProbeOutput(**ORIGINAL), _ProbeInput())
        day = audit._iter_day_files(audit_dir)[-1]
        record = json.loads(day.read_text(encoding="utf-8").splitlines()[-1])
        assert record["output_hash"] == hash_payload(ORIGINAL)
        assert record["output_hash_rounded"] == hash_payload(
            round_floats(normalize_identifiers(ORIGINAL))
        )
        assert record["native_isa"] == "scalar"
        assert audit.verify_audit_trail_integrity(audit_dir) == []

    def test_a_failed_call_has_no_output_to_hash(self, audit_dir: Path):
        def boom(model):
            raise ValueError("no")

        with pytest.raises(ValueError):
            _run_and_record(TOOL, boom, _ProbeInput())
        day = audit._iter_day_files(audit_dir)[-1]
        record = json.loads(day.read_text(encoding="utf-8").splitlines()[-1])
        assert record["output_hash"] is None
        assert record["output_hash_rounded"] is None


class TestReplayTellsTheCasesApart:
    def test_same_build_exact_match_reproduces(self, replay_returns):
        """Null case: nothing moved, so nothing below the exact hash is
        consulted."""
        result = audit.verify_replay(_planted())
        assert result.output_match is True
        assert result.rounded_output_match is None
        assert result.build_differences == []

    def test_a_different_build_matching_to_twelve_digits(self, replay_returns):
        replay_returns(_last_bits_moved(ORIGINAL))
        result = audit.verify_replay(_planted(native_build="match:bbbbbbbbbbbb"))
        assert result.output_match is False
        assert result.rounded_output_match is True
        assert result.build_differences == [
            f"native_build: recorded 'match:bbbbbbbbbbbb', now {HERE_BUILD!r}"
        ]
        assert any("Reproduced to 12 significant digits" in n for n in result.notes)
        assert not any("code/logic likely changed" in n for n in result.notes)

    def test_a_different_isa_path_matching_to_twelve_digits(self, replay_returns):
        """The same build on a CPU without AVX2 takes the scalar path."""
        replay_returns(_last_bits_moved(ORIGINAL))
        result = audit.verify_replay(_planted(native_isa="scalar"))
        assert result.rounded_output_match is True
        assert result.build_differences == [
            f"native_isa: recorded 'scalar', now {HERE_ISA!r}"
        ]
        assert not any("code/logic likely changed" in n for n in result.notes)

    def test_same_build_and_path_exact_miss_reads_as_before(self, replay_returns):
        """On the build and path that wrote it the exact hash is promised,
        so missing it keeps its verdict even when twelve digits agree."""
        replay_returns(_last_bits_moved(ORIGINAL))
        result = audit.verify_replay(_planted())
        assert result.output_match is False
        assert result.rounded_output_match is True
        assert result.build_differences == []
        assert any("code/logic likely changed" in n for n in result.notes)

    def test_a_different_build_differing_beyond_twelve_digits(self, replay_returns):
        replay_returns(_ninth_digit_moved(ORIGINAL))
        result = audit.verify_replay(_planted(native_build="match:bbbbbbbbbbbb"))
        assert result.output_match is False
        assert result.rounded_output_match is False
        assert any("differs beyond 12 significant digits" in n for n in result.notes)
        assert any("code/logic likely changed" in n for n in result.notes)

    def test_an_old_record_falls_back_with_a_note(self, replay_returns):
        """Written before the rounded hash, the ISA path and the build label:
        nothing to compare to twelve digits, so the verdict is today's and
        a note says why."""
        replay_returns(_last_bits_moved(ORIGINAL))
        record = _planted(native_build=None, native_isa=None, with_rounded=False)
        result = audit.verify_replay(record)
        assert result.output_match is False
        assert result.rounded_output_match is None
        assert result.stored_output_hash_rounded is None
        assert any("predates the rounded output hash" in n for n in result.notes)
        assert any("code/logic likely changed" in n for n in result.notes)

    def test_a_record_that_predates_the_isa_field_counts_as_another_path(
        self, replay_returns
    ):
        """It cannot vouch for having run where the replay runs."""
        replay_returns(_last_bits_moved(ORIGINAL))
        result = audit.verify_replay(_planted(native_isa=None))
        assert result.build_differences == [
            f"native_isa: not recorded, now {HERE_ISA!r}"
        ]
        assert result.rounded_output_match is True


class TestTheReplayToolSaysWhich:
    def _write(self, record: Dict[str, Any]) -> str:
        AuditWriter().write(audit.DecisionRecord(**record))
        return record["request_id"]

    def test_reproduced_to_12_digits_on_another_build(
        self, audit_dir: Path, replay_returns
    ):
        rid = self._write(_planted(native_build="match:bbbbbbbbbbbb"))
        replay_returns(_last_bits_moved(ORIGINAL))
        result = dispatch("replay_decision", {"request_id": rid})
        assert result["verdict"] == "reproduced_to_12_digits"
        assert result["output_match"] is False
        assert result["rounded_output_match"] is True
        assert result["stored_output_hash_rounded"] == (
            result["new_output_hash_rounded"]
        )
        assert result["build_differences"]

    def test_the_same_build_still_reads_code_changed(
        self, audit_dir: Path, replay_returns
    ):
        rid = self._write(_planted())
        replay_returns(_last_bits_moved(ORIGINAL))
        result = dispatch("replay_decision", {"request_id": rid})
        assert result["verdict"] == "code_changed"
        assert result["build_differences"] == []

    def test_an_exact_match_is_still_reproduced(self, audit_dir: Path, replay_returns):
        rid = self._write(_planted(native_build="match:bbbbbbbbbbbb"))
        result = dispatch("replay_decision", {"request_id": rid})
        assert result["verdict"] == "reproduced"
        assert result["rounded_output_match"] is None

    def test_an_old_record_reads_code_changed_with_the_note(
        self, audit_dir: Path, replay_returns
    ):
        rid = self._write(
            _planted(native_build=None, native_isa=None, with_rounded=False)
        )
        replay_returns(_last_bits_moved(ORIGINAL))
        result = dispatch("replay_decision", {"request_id": rid})
        assert result["verdict"] == "code_changed"
        assert any("predates the rounded output hash" in n for n in result["notes"])
