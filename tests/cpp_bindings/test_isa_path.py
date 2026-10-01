"""
Which instruction-set path the compiled rolling kernels take on this
machine, as the extension's own CPU detection decides it.

`native_build` says which build ran, and that does not fix the last bits of
an output: one binary takes the AVX2+FMA reduction on a CPU that has it and
the scalar loop on one that does not, and the two round differently (the
C++ suite measures it on rolling_beta: most betas differ in the last bits,
all agree to twelve significant digits). So the extension answers which
path it takes, and every decision record names it beside the build.

It is a property of the machine, not of the binary, so it is asked at
runtime and is no part of the build stamp or the source digest.

The binding cannot force a path: the override that does lives in the C++
test suite only, so the cross-path measurement runs there and not here.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import BaseModel

from standard_quant_tools import _native_build as nb
from standard_quant_tools import audit
from standard_quant_tools.audit.dispatch import _run_and_record

_sqt_core = pytest.importorskip(
    "standard_quant_tools._sqt_core", reason="native extension not built"
)

PATHS = {"avx2+fma", "scalar"}


class TestTheExtensionNamesItsPath:
    def test_it_is_one_of_the_two(self):
        assert _sqt_core.isa_path() in PATHS

    def test_it_is_asked_rather_than_stamped(self):
        """The stamp describes the binary; the path depends on the CPU the
        binary lands on, so it must not be frozen into the stamp."""
        assert "isa_path" not in _sqt_core.__build_info__
        assert not any("avx2+fma" in str(v) for v in _sqt_core.__build_info__.values())

    def test_the_package_reports_what_the_extension_says(self):
        if not nb.native_build_status().used:
            pytest.skip("the extension is present but not in use")
        assert nb.native_isa() == _sqt_core.isa_path()


class TestWithoutAKernelThereIsNoPath:
    def test_no_extension_in_use_is_none(self, monkeypatch: pytest.MonkeyPatch):
        """Null case: every kernel ran its Python path, so no
        instruction-set path was taken."""
        monkeypatch.setattr(nb, "_status", nb.NativeBuildStatus(nb.ABSENT))
        assert nb.native_isa() == nb.ISA_NONE

    def test_an_extension_that_cannot_say_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        if not nb.native_build_status().used:
            pytest.skip("the extension is present but not in use")
        monkeypatch.delattr(_sqt_core, "isa_path")
        assert nb.native_isa() == nb.ISA_UNKNOWN


class _Probe(BaseModel):
    payload: Dict[str, Any] = {}


class TestTheRecordNamesThePath:
    def test_beside_the_build(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        directory = tmp_path / "audit"
        monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
        monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
        _run_and_record("probe_tool", lambda model: {"ok": True}, _Probe())
        day = audit._iter_day_files(directory)[-1]
        record = json.loads(day.read_text(encoding="utf-8").splitlines()[-1])
        assert record["native_build"] == nb.native_build_status().label
        assert record["native_isa"] == nb.native_isa()
