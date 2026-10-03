"""
A decision record names the compiler and the platform, and replay compares
them.

`native_build` is the import verdict and the short digest of the C++
sources. MSVC with /arch:AVX2, MSVC with SSE2, clang-cl and a PGO build of
the same sources all carry the same label, so a replay on any of them read
as "the same build" -- including a clang-cl build that contracts
multiply-adds and differs in the last bits on most kernels. Nor did the
record say which C runtime the process called: the CRT linkage (/MD or
/MT), the version of `ucrtbase.dll` (an OS component) and whether the CRT
takes its FMA3 code path all move the last bits, the last two even on the
pure-Python path through `math`.

New records carry `native_detail` (the build stamp's compiler, build_type,
native_arch, openmp, openmp_runtime and pgo, plus the CRT linkage read from
the binary) and `platform` (os, machine, crt, crt_fma3). Replay counts a
difference in any recorded fact as a build or platform difference, so a bit
mismatch that agrees to twelve digits reads `reproduced_to_12_digits` and
`sqt replay` exits 3. A record without the fields is judged exactly as
before. See the CHANGELOG entry of 2026-10-02.
"""

import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from standard_quant_tools import _native_build as nb
from standard_quant_tools import audit, cli
from standard_quant_tools.agent import tools as agent_tools
from standard_quant_tools.agent.tools import dispatch
from standard_quant_tools.audit import provenance
from standard_quant_tools.audit.dispatch import _run_and_record
from standard_quant_tools.audit.hashing import hash_payload, round_floats
from standard_quant_tools.audit.replay import normalize_identifiers
from standard_quant_tools.audit.writer import AuditWriter

from .test_standalone_verifier import _load_standalone_module

DATE = "2024-05-01"
PLATFORM_KEYS = {"os", "machine", "crt", "crt_fma3"}


@pytest.fixture(scope="module")
def standalone():
    return _load_standalone_module()


@pytest.fixture
def audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "audit"
    monkeypatch.setenv("SQT_AUDIT_DIR", str(directory))
    monkeypatch.setenv("SQT_AUDIT_ENABLED", "1")
    return directory


class _Probe(BaseModel):
    payload: Dict[str, Any] = {}


def _last_line(directory: Path) -> Dict[str, Any]:
    day = audit._iter_day_files(directory)[-1]
    return json.loads(day.read_text(encoding="utf-8").splitlines()[-1])


# ── What a new record carries ────────────────────────────────────────────────


class TestANewRecordNamesTheBuildAndThePlatform:
    def test_both_facts_are_written_and_the_day_verifies(
        self, audit_dir: Path, standalone
    ):
        _run_and_record("probe_tool", lambda model: {"ok": True}, _Probe())
        record = _last_line(audit_dir)
        assert record["native_detail"] == provenance._native_detail()
        assert set(record["platform"]) == PLATFORM_KEYS
        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert standalone.verify_trail(audit_dir) == []

    def test_the_build_detail_is_the_stamp_without_the_digest(self):
        status = nb.native_build_status()
        if not status.used:
            pytest.skip("no compiled extension in use")
        detail = provenance._native_detail()
        assert detail is not None
        assert "source_digest" not in detail and "source_files" not in detail
        for key, value in status.build.items():
            if key not in ("source_digest", "source_files"):
                assert detail[key] == value
        assert detail["crt_linkage"] in ("dynamic", "static", None)

    def test_no_extension_in_use_records_no_build_detail(
        self, audit_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Null case: the Python path ran, so no build facts apply -- but
        the platform still does, since `math` calls the C runtime."""
        monkeypatch.setattr(nb, "_status", nb.NativeBuildStatus(nb.ABSENT))
        _run_and_record("probe_tool", lambda model: {"ok": True}, _Probe())
        record = _last_line(audit_dir)
        assert record["native_build"] == "absent"
        assert record["native_detail"] is None
        assert set(record["platform"]) == PLATFORM_KEYS

    def test_a_refused_build_records_no_build_detail(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        stale = nb.NativeBuildStatus(
            nb.STALE,
            used=False,
            built_digest="0a1b2c3d4e5f" + "0" * 52,
            build={"compiler": "MSVC 19.44.35228.0"},
        )
        monkeypatch.setattr(nb, "_status", stale)
        assert provenance._native_detail() is None

    def test_the_stamp_is_read_defensively(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A stamp with fewer keys, or a value JSON cannot hold, is
        recorded as far as it goes rather than failing the record. A key the
        stamp itself carries wins over the one read from the binary."""
        odd = nb.NativeBuildStatus(
            nb.MATCH,
            used=True,
            extension_file=str(tmp_path / "missing.pyd"),
            build={
                "source_digest": "d" * 64,
                "compiler": "Clang 23.1.2",
                "openmp_runtime": None,
                "flags": ("/O2", "/fp:precise"),
            },
        )
        monkeypatch.setattr(nb, "_status", odd)
        detail = provenance._native_detail()
        assert detail == {
            "compiler": "Clang 23.1.2",
            "crt_linkage": None,
            "flags": str(("/O2", "/fp:precise")),
            "openmp_runtime": None,
        }
        json.dumps(detail)

        stamped = nb.NativeBuildStatus(
            nb.MATCH, used=True, build={"crt_linkage": "static"}
        )
        monkeypatch.setattr(nb, "_status", stamped)
        assert provenance._native_detail() == {"crt_linkage": "static"}

    def test_a_record_carries_a_copy_not_the_cache(self):
        first = provenance._platform_facts()
        assert first is not None
        first["os"] = "tampered"
        assert provenance._platform_facts()["os"] != "tampered"


class TestThePlatformFacts:
    def test_their_shape(self):
        facts = provenance._platform_facts()
        assert facts is not None and set(facts) == PLATFORM_KEYS
        assert facts["crt_fma3"] in (True, False, None)

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows facts")
    def test_on_windows_the_loaded_ucrt_and_its_own_fma3_answer(self):
        import ctypes

        facts = provenance._platform_facts()
        version = sys.getwindowsversion()
        assert facts["os"] == (
            f"Windows {version.major}.{version.minor}.{version.build}"
        )
        assert re.fullmatch(r"ucrtbase \d+\.\d+\.\d+\.\d+", facts["crt"])
        getter = getattr(ctypes.CDLL("ucrtbase"), "_get_FMA3_enable", None)
        if getter is not None:
            assert facts["crt_fma3"] is bool(getter())

    @pytest.mark.skipif(sys.platform != "win32", reason="the UCRT's switch")
    def test_the_fma3_path_is_read_when_asked_not_once(self):
        """The CRT lets a program switch its FMA3 path off, and the record
        written after that must say so."""
        import ctypes

        ucrt = ctypes.CDLL("ucrtbase")
        getter = getattr(ucrt, "_get_FMA3_enable", None)
        setter = getattr(ucrt, "_set_FMA3_enable", None)
        if getter is None or setter is None or not getter():
            pytest.skip("this CPU or CRT has no FMA3 path to switch")
        provenance._platform_facts()
        try:
            setter(0)
            assert provenance._platform_facts()["crt_fma3"] is False
        finally:
            setter(1)
        assert provenance._platform_facts()["crt_fma3"] is True

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="glibc")
    def test_on_linux_the_glibc_version(self):
        crt = provenance._platform_facts()["crt"]
        assert crt is None or crt.startswith("glibc ")

    def test_a_probe_that_fails_is_none_not_a_failed_call(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(provenance, "_platform_cache", None)

        def broken() -> Optional[str]:
            raise OSError("no version resource")

        monkeypatch.setattr(provenance, "_crt_version", broken)
        monkeypatch.setattr(provenance, "_crt_fma3_probe", broken)
        facts = provenance._platform_facts()
        assert facts["crt"] is None and facts["crt_fma3"] is None
        assert facts["os"] is not None


# ── The CRT linkage, read from the binary ────────────────────────────────────


def _tiny_pe(imports: List[str], pe32_plus: bool = True) -> bytes:
    """A minimal PE image whose import table names `imports`, in order."""
    section_rva, section_file = 0x1000, 0x200
    descriptors = 20 * (len(imports) + 1)
    table, names = b"", b""
    for name in imports:
        name_rva = section_rva + descriptors + len(names)
        table += struct.pack("<IIIII", 0, 0, 0, name_rva, 0)
        names += name.encode("ascii") + b"\0"
    table += b"\0" * 20
    body = table + names

    dos = bytearray(64)
    dos[:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 64)
    optional_size = 240 if pe32_plus else 224
    coff = struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, optional_size, 0x2022)
    optional = bytearray(optional_size)
    struct.pack_into("<H", optional, 0, 0x20B if pe32_plus else 0x10B)
    directories = 112 if pe32_plus else 96
    if imports:
        struct.pack_into("<II", optional, directories + 8, section_rva, len(table))
    section = struct.pack(
        "<8sIIIIIIHHI",
        b".idata",
        len(body),
        section_rva,
        len(body),
        section_file,
        0,
        0,
        0,
        0,
        0,
    )
    header = bytes(dos) + b"PE\0\0" + coff + bytes(optional) + section
    return header + b"\0" * (section_file - len(header)) + body


class TestTheCrtLinkage:
    @pytest.mark.parametrize("pe32_plus", [True, False])
    def test_an_import_of_the_ucrt_is_dynamic(self, tmp_path: Path, pe32_plus: bool):
        binary = tmp_path / f"md_{pe32_plus}.pyd"
        binary.write_bytes(
            _tiny_pe(
                [
                    "python312.dll",
                    "VCRUNTIME140.dll",
                    "api-ms-win-crt-math-l1-1-0.dll",
                    "KERNEL32.dll",
                ],
                pe32_plus,
            )
        )
        assert provenance._pe_imports(str(binary)) == [
            "python312.dll",
            "vcruntime140.dll",
            "api-ms-win-crt-math-l1-1-0.dll",
            "kernel32.dll",
        ]
        assert provenance._crt_linkage(str(binary)) == "dynamic"

    def test_ucrtbase_named_directly_is_dynamic(self, tmp_path: Path):
        binary = tmp_path / "direct.pyd"
        binary.write_bytes(_tiny_pe(["ucrtbase.dll", "KERNEL32.dll"]))
        assert provenance._crt_linkage(str(binary)) == "dynamic"

    def test_no_crt_import_is_static(self, tmp_path: Path):
        """/MT: the CRT's math is compiled into the binary."""
        binary = tmp_path / "mt.pyd"
        binary.write_bytes(_tiny_pe(["python312.dll", "KERNEL32.dll"]))
        assert provenance._crt_linkage(str(binary)) == "static"
        empty = tmp_path / "no_imports.pyd"
        empty.write_bytes(_tiny_pe([]))
        assert provenance._crt_linkage(str(empty)) == "static"

    def test_what_cannot_be_told_is_none(self, tmp_path: Path):
        elf = tmp_path / "_sqt_core.so"
        elf.write_bytes(b"\x7fELF" + b"\0" * 60)
        truncated = tmp_path / "cut.pyd"
        truncated.write_bytes(_tiny_pe(["ucrtbase.dll"])[:0x150])
        assert provenance._crt_linkage(str(elf)) is None
        assert provenance._crt_linkage(str(truncated)) is None
        assert provenance._crt_linkage(str(tmp_path / "missing.pyd")) is None
        assert provenance._crt_linkage(None) is None

    @pytest.mark.skipif(sys.platform != "win32", reason="a Windows extension")
    def test_the_extension_in_use_can_be_told(self):
        status = nb.native_build_status()
        if not status.used or not status.extension_file:
            pytest.skip("no compiled extension in use")
        assert provenance._crt_linkage(status.extension_file) in ("dynamic", "static")


# ── A day written before the fields ──────────────────────────────────────────

_NEW_FIELDS = {"native_detail", "platform"}


def _record(request_id: str) -> "audit.DecisionRecord":
    return audit.DecisionRecord(
        request_id=request_id,
        timestamp_utc=f"{DATE}T00:00:00+00:00",
        tool_name="probe",
        input={"value": 1},
        cpp_available=True,
        native_build="match:d938ad9d2eae",
        native_isa="avx2+fma",
        duration_ms=1.0,
        status="ok",
    )


def _write_before_the_fields(day: Path, records: List["audit.DecisionRecord"]) -> None:
    """Append records as a writer that did not know the new fields did:
    chained, hashed over the line as written, and without the keys."""
    prev = AuditWriter(audit_dir=day.parent)._bootstrap_new_day(day)
    with open(day, "a", encoding="utf-8") as handle:
        for record in records:
            record.prev_record_hash = prev
            payload = json.loads(
                record.model_dump_json(exclude={"record_hash", *_NEW_FIELDS})
            )
            record_hash = audit.hash_payload({**payload, "record_hash": None})
            handle.write(json.dumps({**payload, "record_hash": record_hash}) + "\n")
            prev = record_hash


class TestADayWrittenBeforeTheFieldsStillVerifies:
    def test_old_records_verify_and_new_ones_chain_onto_them(
        self, audit_dir: Path, standalone, monkeypatch: pytest.MonkeyPatch
    ):
        audit_dir.mkdir(parents=True)
        day = audit_dir / f"{DATE}.jsonl"
        _write_before_the_fields(day, [_record("old-1"), _record("old-2")])
        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert standalone.verify_trail(audit_dir) == []

        monkeypatch.setattr(
            AuditWriter, "_path_for", lambda self, when: self._dir / f"{DATE}.jsonl"
        )
        _run_and_record("probe_tool", lambda model: {"ok": True}, _Probe())

        lines = [json.loads(x) for x in day.read_text(encoding="utf-8").splitlines()]
        assert [_NEW_FIELDS <= set(x) for x in lines] == [False, False, True]
        assert lines[2]["prev_record_hash"] == lines[1]["record_hash"]
        assert audit.verify_audit_trail_integrity(audit_dir) == []
        assert standalone.verify_trail(audit_dir) == []

    @pytest.mark.skipif(not audit.HAS_CRYPTOGRAPHY, reason="cryptography")
    def test_a_checkpoint_over_old_and_new_records_verifies(
        self, audit_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        audit_dir.mkdir(parents=True)
        day = audit_dir / f"{DATE}.jsonl"
        _write_before_the_fields(day, [_record("old-1")])
        private_bytes, public_bytes = audit.generate_keypair()
        private_path = tmp_path / "signing.private"
        public_path = tmp_path / "signing.public"
        private_path.write_bytes(private_bytes)
        public_path.write_bytes(public_bytes)

        audit.checkpoint_and_sign(DATE, audit_dir=audit_dir, key_path=private_path)
        before = audit.verify_checkpoint(DATE, public_path, audit_dir=audit_dir)
        assert before.state == "valid"

        monkeypatch.setattr(
            AuditWriter, "_path_for", lambda self, when: self._dir / f"{DATE}.jsonl"
        )
        _run_and_record("probe_tool", lambda model: {"ok": True}, _Probe())
        audit.checkpoint_and_sign(DATE, audit_dir=audit_dir, key_path=private_path)
        after = audit.verify_checkpoint(DATE, public_path, audit_dir=audit_dir)
        assert after.state == "valid"

    def test_an_old_record_reads_back_without_the_facts(self, tmp_path: Path):
        day = tmp_path / f"{DATE}.jsonl"
        _write_before_the_fields(day, [_record("old-1")])
        reread = audit.DecisionRecord(**json.loads(day.read_text(encoding="utf-8")))
        assert reread.native_detail is None and reread.platform is None


# ── Replay ───────────────────────────────────────────────────────────────────


class _ProbeInput(BaseModel):
    window: int = 60


class _ProbeOutput(BaseModel):
    beta: float
    betas: List[float]


TOOL = "probe_build_facts"
HERE_BUILD = "match:aaaaaaaaaaaa"
HERE_ISA = "avx2+fma"
HERE_DETAIL: Dict[str, Any] = {
    "build_type": "Release",
    "compiler": "MSVC 19.44.35228.0",
    "crt_linkage": "dynamic",
    "native_arch": True,
    "openmp": "2.0",
    "openmp_runtime": "vcomp",
    "pgo": "off",
}
HERE_PLATFORM: Dict[str, Any] = {
    "os": "Windows 10.0.26300",
    "machine": "AMD64",
    "crt": "ucrtbase 10.0.26100.9444",
    "crt_fma3": True,
}
ORIGINAL: Dict[str, Any] = {
    "beta": 1.1043287719382044,
    "betas": [0.9831234567890123, 1.0123456789012345],
}


def _last_bits_moved(output: Dict[str, Any]) -> Dict[str, Any]:
    """The same answer from different arithmetic, one ulp away."""
    return {
        "beta": math.nextafter(output["beta"], math.inf),
        "betas": [math.nextafter(v, -math.inf) for v in output["betas"]],
    }


def _ninth_digit_moved(output: Dict[str, Any]) -> Dict[str, Any]:
    return {**output, "beta": output["beta"] * (1.0 + 1e-9)}


@pytest.fixture
def replay_returns(monkeypatch: pytest.MonkeyPatch):
    """Register the probe tool and choose what its replay returns. Replay
    runs here on HERE_BUILD, HERE_ISA, HERE_DETAIL and HERE_PLATFORM."""
    state: Dict[str, Any] = {"output": dict(ORIGINAL)}

    def tool(model: _ProbeInput) -> _ProbeOutput:
        return _ProbeOutput(**state["output"])

    monkeypatch.setitem(agent_tools._TOOL_DISPATCH, TOOL, (tool, _ProbeInput))
    monkeypatch.setattr(provenance, "_native_build_label", lambda: HERE_BUILD)
    monkeypatch.setattr(provenance, "_native_isa_label", lambda: HERE_ISA)
    monkeypatch.setattr(provenance, "_native_detail", lambda: dict(HERE_DETAIL))
    monkeypatch.setattr(provenance, "_platform_facts", lambda: dict(HERE_PLATFORM))

    def choose(output: Dict[str, Any]) -> None:
        state["output"] = output

    return choose


_ABSENT = object()


def _planted(
    request_id: str = "r1",
    native_build: Any = HERE_BUILD,
    native_isa: Any = HERE_ISA,
    native_detail: Any = _ABSENT,
    platform: Any = _ABSENT,
) -> Dict[str, Any]:
    """A record of ORIGINAL as the writer stores it, written on this build
    and platform unless a fact is passed. Passing None for a field drops
    its key, as on a record written before the field existed."""
    record: Dict[str, Any] = {
        "request_id": request_id,
        "timestamp_utc": "2026-10-02T00:00:00+00:00",
        "tool_name": TOOL,
        "input": {"window": 60},
        "data_sources": [],
        "cpp_available": True,
        "duration_ms": 1.0,
        "status": "ok",
        "output_hash": hash_payload(ORIGINAL),
        "output_hash_normalized": hash_payload(normalize_identifiers(ORIGINAL)),
        "output_hash_rounded": hash_payload(
            round_floats(normalize_identifiers(ORIGINAL))
        ),
        "native_detail": dict(HERE_DETAIL),
        "platform": dict(HERE_PLATFORM),
    }
    if native_build is not None:
        record["native_build"] = native_build
    if native_isa is not None:
        record["native_isa"] = native_isa
    for name, value in (("native_detail", native_detail), ("platform", platform)):
        if value is None:
            record.pop(name)
        elif value is not _ABSENT:
            record[name] = value
    return record


def _with(field: str, key: str, value: Any) -> Dict[str, Any]:
    """The planted record with one fact changed."""
    base = dict(HERE_DETAIL if field == "native_detail" else HERE_PLATFORM)
    base[key] = value
    return _planted(**{field: base})


ONE_FACT_DIFFERS = [
    pytest.param("native_detail", "compiler", "Clang 23.1.2", id="compiler"),
    pytest.param("native_detail", "native_arch", False, id="native_arch"),
    pytest.param("native_detail", "openmp_runtime", "libomp", id="openmp_runtime"),
    pytest.param("native_detail", "pgo", "use", id="pgo"),
    pytest.param("native_detail", "crt_linkage", "static", id="crt_linkage"),
    pytest.param("platform", "crt", "ucrtbase 10.0.22621.3672", id="crt_version"),
    pytest.param("platform", "crt_fma3", False, id="crt_fma3"),
    pytest.param("platform", "os", "Windows 10.0.22631", id="os_build"),
    pytest.param("platform", "machine", "ARM64", id="machine"),
]


def _here(field: str, key: str) -> Any:
    return (HERE_DETAIL if field == "native_detail" else HERE_PLATFORM)[key]


class TestReplayComparesTheFacts:
    @pytest.mark.parametrize("field, key, recorded", ONE_FACT_DIFFERS)
    def test_one_differing_fact_is_a_build_or_platform_difference(
        self, replay_returns, field: str, key: str, recorded: Any
    ):
        replay_returns(_last_bits_moved(ORIGINAL))
        result = audit.verify_replay(_with(field, key, recorded))
        assert result.output_match is False
        assert result.rounded_output_match is True
        assert result.build_differences == [
            f"{field}.{key}: recorded {recorded!r}, now {_here(field, key)!r}"
        ]
        assert any("Reproduced to 12 significant digits" in n for n in result.notes)
        assert not any("code/logic likely changed" in n for n in result.notes)

    def test_every_fact_the_same_keeps_the_verdict(self, replay_returns):
        """Null case: on the same build, path and platform the exact hash is
        promised, so a miss still reads as a change -- and the note names
        what the record cannot see rather than the compiler it now does."""
        replay_returns(_last_bits_moved(ORIGINAL))
        result = audit.verify_replay(_planted())
        assert result.build_differences == []
        assert result.rounded_output_match is True
        assert any("code/logic likely changed" in n for n in result.notes)
        assert any("none of them covers" in n for n in result.notes)

    def test_a_record_without_the_fields_is_judged_as_before(
        self, replay_returns, monkeypatch: pytest.MonkeyPatch
    ):
        """An older record carries neither field, and replay running on a
        different compiler AND a different platform still compares only
        what the record carries -- the verdict and the note are today's."""
        monkeypatch.setattr(
            provenance,
            "_native_detail",
            lambda: {**HERE_DETAIL, "compiler": "Clang 23.1.2"},
        )
        monkeypatch.setattr(
            provenance,
            "_platform_facts",
            lambda: {**HERE_PLATFORM, "crt": "glibc 2.35", "crt_fma3": False},
        )
        replay_returns(_last_bits_moved(ORIGINAL))
        result = audit.verify_replay(_planted(native_detail=None, platform=None))
        assert result.build_differences == []
        assert any("code/logic likely changed" in n for n in result.notes)
        assert any("The build label names the C++ sources" in n for n in result.notes)

    def test_no_extension_then_or_now_compares_the_platform_alone(
        self, replay_returns, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(provenance, "_native_build_label", lambda: "absent")
        monkeypatch.setattr(provenance, "_native_isa_label", lambda: "none")
        monkeypatch.setattr(provenance, "_native_detail", lambda: None)
        replay_returns(_last_bits_moved(ORIGINAL))
        same = _planted(native_build="absent", native_isa="none", native_detail=None)
        same["native_detail"] = None  # written by this release, Python path
        assert audit.verify_replay(same).build_differences == []

        other_crt = dict(same, platform={**HERE_PLATFORM, "crt_fma3": False})
        result = audit.verify_replay(other_crt)
        assert result.build_differences == [
            "platform.crt_fma3: recorded False, now True"
        ]
        assert result.rounded_output_match is True

    def test_an_extension_then_and_none_now(
        self, replay_returns, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(provenance, "_native_build_label", lambda: "absent")
        monkeypatch.setattr(provenance, "_native_detail", lambda: None)
        replay_returns(_last_bits_moved(ORIGINAL))
        result = audit.verify_replay(_planted())
        assert result.build_differences == [
            f"native_build: recorded {HERE_BUILD!r}, now 'absent'",
            f"native_detail: recorded {HERE_DETAIL!r}, now None",
        ]

    def test_a_fact_only_the_replay_knows_is_not_a_difference(self, replay_returns):
        """The record is the authority on what it vouched for: a fact a
        later release began recording does not turn its verdict."""
        replay_returns(_last_bits_moved(ORIGINAL))
        fewer = {k: v for k, v in HERE_DETAIL.items() if k != "crt_linkage"}
        result = audit.verify_replay(_planted(native_detail=fewer))
        assert result.build_differences == []

    def test_an_exact_match_on_another_compiler_is_reproduced(self, replay_returns):
        result = audit.verify_replay(_with("native_detail", "compiler", "Clang 23.1.2"))
        assert result.output_match is True
        assert result.rounded_output_match is None
        assert result.build_differences == []

    def test_another_compiler_does_not_excuse_a_real_difference(self, replay_returns):
        replay_returns(_ninth_digit_moved(ORIGINAL))
        result = audit.verify_replay(_with("native_detail", "compiler", "Clang 23.1.2"))
        assert result.rounded_output_match is False
        assert any("differs beyond 12 significant digits" in n for n in result.notes)
        assert any("code/logic likely changed" in n for n in result.notes)


class TestTheVerdictsReachTheToolAndTheCli:
    @staticmethod
    def _write(record: Dict[str, Any]) -> str:
        AuditWriter().write(audit.DecisionRecord(**record))
        return record["request_id"]

    @staticmethod
    def _write_old(audit_dir: Path, record: Dict[str, Any]) -> str:
        """The record as a writer from before the fields left it: no keys."""
        audit_dir.mkdir(parents=True, exist_ok=True)
        day = audit_dir / f"{DATE}.jsonl"
        _write_before_the_fields(day, [audit.DecisionRecord(**record)])
        line = json.loads(day.read_text(encoding="utf-8").splitlines()[-1])
        assert not _NEW_FIELDS & set(line)
        return record["request_id"]

    @pytest.mark.parametrize(
        "field, key, recorded",
        [
            pytest.param("native_detail", "compiler", "Clang 23.1.2", id="compiler"),
            pytest.param("native_detail", "native_arch", False, id="native_arch"),
            pytest.param("platform", "crt", "ucrtbase 10.0.22621.3672", id="crt"),
            pytest.param("platform", "crt_fma3", False, id="crt_fma3"),
        ],
    )
    def test_a_different_fact_is_reproduced_to_12_digits_and_exit_3(
        self, audit_dir: Path, replay_returns, field: str, key: str, recorded: Any
    ):
        rid = self._write(_with(field, key, recorded))
        replay_returns(_last_bits_moved(ORIGINAL))
        result = dispatch("replay_decision", {"request_id": rid})
        assert result["verdict"] == "reproduced_to_12_digits"
        assert result["build_differences"] == [
            f"{field}.{key}: recorded {recorded!r}, now {_here(field, key)!r}"
        ]
        assert cli.main(["replay", rid]) == 3

    def test_the_same_facts_still_read_code_changed_and_exit_1(
        self, audit_dir: Path, replay_returns
    ):
        rid = self._write(_planted())
        replay_returns(_last_bits_moved(ORIGINAL))
        assert dispatch("replay_decision", {"request_id": rid})["verdict"] == (
            "code_changed"
        )
        assert cli.main(["replay", rid]) == 1

    def test_an_old_record_reads_as_before(self, audit_dir: Path, replay_returns):
        rid = self._write_old(audit_dir, _planted(native_detail=None, platform=None))
        replay_returns(_last_bits_moved(ORIGINAL))
        assert dispatch("replay_decision", {"request_id": rid})["verdict"] == (
            "code_changed"
        )
        assert cli.main(["replay", rid]) == 1

    def test_an_exact_match_exits_0(self, audit_dir: Path, replay_returns):
        rid = self._write(_with("platform", "crt_fma3", False))
        assert dispatch("replay_decision", {"request_id": rid})["verdict"] == (
            "reproduced"
        )
        assert cli.main(["replay", rid]) == 0

    def test_explain_shows_the_facts(self, audit_dir: Path):
        _run_and_record("probe_tool", lambda model: {"ok": True}, _Probe())
        record = _last_line(audit_dir)
        result = dispatch("explain_decision", {"request_id": record["request_id"]})
        assert result["native_detail"] == record["native_detail"]
        assert result["platform"] == record["platform"]

    def test_explain_of_an_old_record_says_none(self, audit_dir: Path):
        rid = self._write_old(audit_dir, _planted(native_detail=None, platform=None))
        result = dispatch("explain_decision", {"request_id": rid})
        assert result["native_detail"] is None and result["platform"] is None
