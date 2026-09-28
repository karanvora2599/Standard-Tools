"""
An audit bundle never replaces a file, and the tool that writes one writes
only where the operator said it may.

`export_bundle` opened its destination for writing and silently overwrote
whatever was there, an earlier bundle included; the agent-facing tool
checked for an existing file first, which left a window in which one could
appear. The tool took any absolute path as given, so a model-chosen string
could drop a zip anywhere this process can write. See the CHANGELOG entry
of 2026-09-28.
"""

import json
import os
from pathlib import Path

import pytest

from standard_quant_tools import audit, cli
from standard_quant_tools.agent.runtimes.meta.tools import _contained_bundle_path
from standard_quant_tools.audit import export as export_module
from standard_quant_tools.audit.writer import AuditWriter
from standard_quant_tools.error import ValidationError

DAY = "2024-01-01"


def _trail(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    day = directory / f"{DAY}.jsonl"
    record = audit.DecisionRecord(
        request_id="r1",
        timestamp_utc=f"{DAY}T00:00:00+00:00",
        tool_name="t",
        input={},
        cpp_available=False,
        duration_ms=1.0,
        status="ok",
    )
    record.prev_record_hash = AuditWriter(audit_dir=directory)._bootstrap_new_day(day)
    payload = json.loads(record.model_dump_json(exclude={"record_hash"}))
    record.record_hash = audit.hash_payload({**payload, "record_hash": None})
    day.write_text(record.model_dump_json() + "\n", encoding="utf-8")


def _leftovers(directory: Path):
    return [p.name for p in directory.iterdir() if p.name.endswith(".partial")]


class TestTheLibraryNeverReplacesABundle:
    def test_an_existing_file_is_refused_and_left_as_it_was(self, tmp_path):
        """It was overwritten."""
        _trail(tmp_path / "audit")
        out = tmp_path / "bundle.zip"
        out.write_bytes(b"an earlier bundle")

        with pytest.raises(ValidationError, match="already exists"):
            audit.export_bundle(DAY, DAY, out, tmp_path / "audit")

        assert out.read_bytes() == b"an earlier bundle"
        assert _leftovers(tmp_path) == []

    def test_a_file_that_appears_while_the_zip_is_built_is_not_replaced(
        self, tmp_path, monkeypatch
    ):
        """The check and the create are one step: a destination created
        after the first look is still refused."""
        _trail(tmp_path / "audit")
        out = tmp_path / "bundle.zip"
        real = export_module._publish_exclusively

        def _raced(built, target):
            target.write_bytes(b"arrived in between")
            return real(built, target)

        monkeypatch.setattr(export_module, "_publish_exclusively", _raced)
        with pytest.raises(ValidationError, match="already exists"):
            audit.export_bundle(DAY, DAY, out, tmp_path / "audit")

        assert out.read_bytes() == b"arrived in between"
        assert _leftovers(tmp_path) == []

    def test_without_hard_links_the_create_is_still_exclusive(
        self, tmp_path, monkeypatch
    ):
        def _no_links(*args, **kwargs):
            raise PermissionError("hard links are not supported here")

        monkeypatch.setattr(export_module.os, "link", _no_links)
        _trail(tmp_path / "audit")
        out = tmp_path / "bundle.zip"

        audit.export_bundle(DAY, DAY, out, tmp_path / "audit")
        assert out.stat().st_size > 0
        with pytest.raises(ValidationError, match="already exists"):
            audit.export_bundle(DAY, DAY, out, tmp_path / "audit")
        assert _leftovers(tmp_path) == []

    def test_a_new_destination_is_written_whole(self, tmp_path):
        """Null case."""
        _trail(tmp_path / "audit")
        out = tmp_path / "bundle.zip"
        result = audit.export_bundle(DAY, DAY, out, tmp_path / "audit")
        assert result.path == out and result.record_count == 1
        assert _leftovers(tmp_path) == []

    def test_sqt_export_refuses_an_existing_file(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "audit"))
        _trail(tmp_path / "audit")
        out = tmp_path / "bundle.zip"
        out.write_bytes(b"keep me")

        assert (
            cli.main(["export", "--start", DAY, "--end", DAY, "--out", str(out)]) == 1
        )
        assert "already exists" in capsys.readouterr().err
        assert out.read_bytes() == b"keep me"


class TestTheToolWritesOnlyInsideTheFence:
    @pytest.fixture(autouse=True)
    def _stores(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        monkeypatch.setenv("SQT_AUDIT_DIR", str(tmp_path / "shared" / "audit"))
        (tmp_path / "shared" / "audit").mkdir(parents=True)
        (tmp_path / "shared" / "handoff").mkdir(parents=True)
        (tmp_path / "elsewhere").mkdir()
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", str(tmp_path / "shared"))

    def test_an_absolute_path_outside_every_listed_directory_is_refused(self, tmp_path):
        """It was written wherever it pointed."""
        with pytest.raises(ValidationError, match="SQT_EXTERNAL_DIRS"):
            _contained_bundle_path(str(tmp_path / "elsewhere" / "b.zip"))

    def test_a_listed_directory_is_allowed(self, tmp_path):
        """Null case."""
        target = tmp_path / "shared" / "handoff" / "b.zip"
        assert _contained_bundle_path(str(target)) == target.resolve()

    def test_the_audit_directory_is_refused_even_when_listed(self, tmp_path):
        with pytest.raises(ValidationError, match="the audit directory"):
            _contained_bundle_path(str(tmp_path / "shared" / "audit" / "b.zip"))

    def test_the_runs_directory_outside_bundles_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "SQT_EXTERNAL_DIRS",
            os.pathsep.join([str(tmp_path / "shared"), str(tmp_path / "runs")]),
        )
        (tmp_path / "runs" / "run-1").mkdir(parents=True)
        with pytest.raises(ValidationError, match="runs directory"):
            _contained_bundle_path(str(tmp_path / "runs" / "run-1" / "b.zip"))

    def test_the_bundles_folder_takes_an_absolute_path_too(self, tmp_path):
        bundles = tmp_path / "runs" / "bundles"
        bundles.mkdir(parents=True)
        assert (
            _contained_bundle_path(str(bundles / "b.zip"))
            == (bundles / "b.zip").resolve()
        )

    @pytest.mark.parametrize(
        "path", ["\\\\fileserver\\share\\b.zip", "//fileserver/share/b.zip"]
    )
    def test_a_network_path_is_refused_before_anything_opens_it(self, path):
        with pytest.raises(ValidationError, match="network or device"):
            _contained_bundle_path(path)

    def test_a_bare_name_still_lands_in_bundles(self, tmp_path):
        resolved = _contained_bundle_path("audit-2024.zip")
        assert resolved == (tmp_path / "runs" / "bundles" / "audit-2024.zip").resolve()
