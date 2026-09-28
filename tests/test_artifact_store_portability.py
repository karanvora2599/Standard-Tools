"""
A key names one file on every platform, and a write that must not replace
anything never does.

Windows drops a trailing dot, folds case and opens a device for `NUL` or
`COM1` in every directory, so `run8/rep.` silently replaced `run8/rep` and
the second artifact was invisible to `list()`, while `run9/NUL` could not
be written at all. The grammar refuses the spellings that are not portable
on every platform, and the local store refuses a key that reaches an
existing file under another spelling.

`write_bytes_exclusively` is the write a published reference needs: atomic
like `write_bytes_atomically`, and exclusive -- of any number of writers
racing to one name exactly one succeeds, and the existing bytes are never
touched.
"""

from __future__ import annotations

import errno
import os
import sys
import threading
from uuid import uuid4

import pytest

from standard_quant_tools import _runspath
from standard_quant_tools import artifact_store as store_module
from standard_quant_tools.artifact_store import (
    FsspecArtifactStore,
    LocalArtifactStore,
    fsspec_available,
    validate_key,
    write_bytes_atomically,
    write_bytes_exclusively,
)
from standard_quant_tools.error import ValidationError


def _case_insensitive(directory) -> bool:
    probe = directory / f"CaseProbe{uuid4().hex[:6]}"
    probe.write_bytes(b"")
    try:
        return (directory / probe.name.lower()).exists()
    finally:
        probe.unlink()


class TestTheKeyGrammarIsPortable:
    @pytest.mark.parametrize(
        "key",
        [
            "run8/rep.",
            "run8/x.json..",
            "run8/a..b",
            "NUL/data.bin",
            "con/x",
            "run9/NUL",
            "run9/CON.json",
            "run9/com1.txt",
            "run9/lpt9",
            "run9/aux.tar.gz",
        ],
    )
    def test_an_unportable_key_is_refused_everywhere(self, key):
        """Planted: every one of these validated, and on Windows each either
        collided with another key or could not be written."""
        with pytest.raises(ValidationError):
            validate_key(key)

    @pytest.mark.parametrize("value", ["NUL", "con", "Aux", "COM1", "lpt3", "prn"])
    def test_a_device_name_is_not_an_identifier(self, value):
        with pytest.raises(ValidationError, match="device name"):
            _runspath.validate_identifier(value, "run_id")

    @pytest.mark.parametrize(
        "key",
        [
            "run8/report.v2.json",
            "ds_abc/features.parquet",
            "run8/console.json",
            "run8/nullable.parquet",
            "com10/x.json",
            "run8/model.joblib",
        ],
    )
    def test_ordinary_keys_round_trip(self, tmp_path, key):
        """Null: a name that merely starts like a device name is a name."""
        store = LocalArtifactStore(tmp_path / "store")
        store.put(key, b"payload")
        assert store.get(key) == b"payload"
        assert store.list(key.split("/")[0]) == [key]


class TestASecondSpellingIsRefused:
    def test_a_key_differing_only_in_case_does_not_replace_the_first(self, tmp_path):
        root = tmp_path / "store"
        root.mkdir()
        if not _case_insensitive(root):
            pytest.skip("this filesystem keeps case apart; the two keys are two files")
        store = LocalArtifactStore(root)
        store.put("run8/report", b"FIRST")
        with pytest.raises(ValidationError, match="spelled differently"):
            store.put("run8/Report", b"SECOND")
        with pytest.raises(ValidationError, match="spelled differently"):
            store.get("run8/Report")
        assert store.get("run8/report") == b"FIRST"
        assert store.list("run8") == ["run8/report"]

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows folds run directories")
    def test_a_run_id_differing_only_in_case_is_refused(self, tmp_path, monkeypatch):
        import pandas as pd

        from standard_quant_tools.backtest.artifacts import save_artifact

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        series = pd.Series([1.0, 2.0], name="x")
        save_artifact(series, "RunA", "equity")
        with pytest.raises(ValidationError, match="already exists"):
            save_artifact(series, "runa", "equity")
        with pytest.raises(ValidationError, match="spelled differently"):
            save_artifact(series, "runa", "other")

    def test_distinct_names_in_one_run_are_untouched(self, tmp_path):
        """Null: the check fires on a second spelling, not on a neighbour."""
        store = LocalArtifactStore(tmp_path / "store")
        store.put("run8/report", b"A")
        store.put("run8/report.v2", b"B")
        assert store.get("run8/report") == b"A"
        assert store.get("run8/report.v2") == b"B"


class TestExclusiveWrites:
    def test_it_writes_once_and_never_replaces(self, tmp_path):
        target = tmp_path / "run" / "value.bin"
        assert write_bytes_exclusively(target, b"FIRST") is True
        assert write_bytes_exclusively(target, b"SECOND") is False
        assert target.read_bytes() == b"FIRST"
        assert not list(target.parent.glob(".*.tmp"))

    def test_racing_writers_have_exactly_one_winner(self, tmp_path):
        for trial in range(30):
            target = tmp_path / f"t{trial}" / "value.bin"
            barrier = threading.Barrier(6)
            won = []

            def write(payload):
                barrier.wait()
                if write_bytes_exclusively(target, payload):
                    won.append(payload)

            threads = [
                threading.Thread(target=write, args=(f"writer-{i}".encode(),))
                for i in range(6)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=60)
            assert len(won) == 1, won
            assert target.read_bytes() == won[0]
            assert not list(target.parent.glob(".*.tmp"))

    def test_without_hard_links_it_is_still_exclusive(self, tmp_path, monkeypatch):
        """The fallback for a filesystem that has no hard links (FAT, some
        network mounts) keeps both promises."""

        def no_links(src, dst):
            raise OSError(errno.EPERM, "hard links are not supported here")

        monkeypatch.setattr(store_module.os, "link", no_links)
        target = tmp_path / "run" / "value.bin"
        assert write_bytes_exclusively(target, b"FIRST") is True
        assert write_bytes_exclusively(target, b"SECOND") is False
        assert target.read_bytes() == b"FIRST"
        assert not list(target.parent.glob(".*.tmp"))

    def test_a_real_link_failure_is_not_mistaken_for_no_links(
        self, tmp_path, monkeypatch
    ):
        def broken(src, dst):
            raise OSError(errno.EIO, "the disk said no")

        monkeypatch.setattr(store_module.os, "link", broken)
        with pytest.raises(OSError, match="the disk said no"):
            write_bytes_exclusively(tmp_path / "run" / "value.bin", b"X")
        assert not list((tmp_path / "run").glob(".*.tmp"))

    @pytest.mark.parametrize("writer", ["atomically", "exclusively"])
    def test_a_failed_cleanup_does_not_replace_the_real_error(
        self, tmp_path, monkeypatch, writer
    ):
        """Planted: the temp file's removal runs in a `finally`; when it
        failed too (a Windows sharing violation) its PermissionError was
        what the caller saw, not the failure that stopped the write."""
        from pathlib import Path

        def the_write_fails(*args):
            raise RuntimeError("the failure that stopped the write")

        real_unlink = Path.unlink

        def the_cleanup_fails(self, missing_ok=False):
            if self.name.endswith(".tmp"):
                raise PermissionError(13, "the file is in use")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(store_module.os, "replace", the_write_fails)
        monkeypatch.setattr(store_module.os, "link", the_write_fails)
        monkeypatch.setattr(Path, "unlink", the_cleanup_fails)
        write = (
            write_bytes_atomically
            if writer == "atomically"
            else write_bytes_exclusively
        )
        with pytest.raises(RuntimeError, match="the failure that stopped the write"):
            write(tmp_path / "value.bin", b"DATA")

    def test_a_failed_cleanup_does_not_fail_a_completed_write(
        self, tmp_path, monkeypatch
    ):
        from pathlib import Path

        real_unlink = Path.unlink

        def the_cleanup_fails(self, missing_ok=False):
            if self.name.endswith(".tmp"):
                raise PermissionError(13, "the file is in use")
            return real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", the_cleanup_fails)
        target = tmp_path / "value.bin"
        assert write_bytes_exclusively(target, b"DATA") is True
        assert target.read_bytes() == b"DATA"

    def test_the_replacing_write_still_replaces(self, tmp_path):
        """Null: `write_bytes_atomically` is the write that replaces."""
        target = tmp_path / "value.bin"
        write_bytes_atomically(target, b"FIRST")
        write_bytes_atomically(target, b"SECOND")
        assert target.read_bytes() == b"SECOND"


@pytest.mark.skipif(sys.platform != "win32", reason="a Windows sharing violation")
class TestATransientRefusalIsRetried:
    @staticmethod
    def _sharing_violation():
        error = PermissionError(13, "The process cannot access the file")
        error.winerror = 32
        return error

    def test_a_brief_refusal_is_waited_out(self, tmp_path, monkeypatch):
        real_replace = os.replace
        refusals = {"left": 2}

        def busy_then_free(src, dst):
            if refusals["left"]:
                refusals["left"] -= 1
                raise self._sharing_violation()
            return real_replace(src, dst)

        monkeypatch.setattr(store_module.os, "replace", busy_then_free)
        target = tmp_path / "value.bin"
        write_bytes_atomically(target, b"DATA")
        assert target.read_bytes() == b"DATA"

    def test_a_refusal_that_persists_is_a_validation_error(self, tmp_path, monkeypatch):
        def always_busy(src, dst):
            raise self._sharing_violation()

        monkeypatch.setattr(store_module.os, "replace", always_busy)
        monkeypatch.setattr(store_module, "_FIRST_REPLACE_WAIT", 0.0001)
        with pytest.raises(ValidationError, match="could not be replaced"):
            write_bytes_atomically(tmp_path / "value.bin", b"DATA")
        assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.skipif(not fsspec_available(), reason="fsspec is not installed")
class TestAPullRefusesNamesThisMachineCannotKeepApart:
    def test_a_trailing_dot_twin_is_refused_before_anything_is_written(
        self, tmp_path, monkeypatch
    ):
        """Planted: an object store holds `model.joblib` and `model.joblib.`
        as two keys; on Windows the second, which no digest covers, replaced
        the first."""
        from standard_quant_tools.modeling.registry.package import pull_model_package

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        model_id = f"mdl_{uuid4().hex[:12]}"
        remote = FsspecArtifactStore(f"memory://sqt-portability/{uuid4().hex}")
        remote.put(f"{model_id}/manifest.json", b"{}")
        remote.put(f"{model_id}/model.joblib", b"registered")
        remote._fs.pipe_file(f"{remote._root}/{model_id}/model.joblib.", b"planted")
        with pytest.raises(ValidationError, match="holds both"):
            pull_model_package(model_id, remote)
        assert not (tmp_path / "runs" / model_id).exists()

    def test_names_equal_once_case_is_folded_are_refused(self, tmp_path, monkeypatch):
        from standard_quant_tools.modeling.registry.package import pull_model_package

        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        model_id = f"mdl_{uuid4().hex[:12]}"
        remote = FsspecArtifactStore(f"memory://sqt-portability/{uuid4().hex}")
        remote.put(f"{model_id}/manifest.json", b"{}")
        remote.put(f"{model_id}/model.joblib", b"one")
        remote.put(f"{model_id}/MODEL.joblib", b"two")
        with pytest.raises(ValidationError, match="holds both"):
            pull_model_package(model_id, remote)
        assert not (tmp_path / "runs" / model_id).exists()
