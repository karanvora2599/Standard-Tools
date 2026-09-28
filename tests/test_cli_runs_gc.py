"""
`sqt runs gc`: what an interrupted write left behind, and nothing else.

The runs directory keeps every published value for good, by design -- a
reference promises that resolving it twice gives the same value, and a
collector that deleted one would break every holder, including the ones an
audit record names. What it may collect is what no reference can name: a
temp file an atomic write never renamed into place, and a model or dataset
directory whose registration never reached its commit file. Dry-run by
default, and only past an age threshold, so a registration or conversion in
progress is never mistaken for an abandoned one.
"""

from __future__ import annotations

import os
import time
from uuid import uuid4

import pytest

from standard_quant_tools import _runspath, cli

TWO_DAYS = 2 * 24 * 3600.0


@pytest.fixture
def runs(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setenv("SQT_RUNS_DIR", str(root))
    return root


def _age(path, seconds: float) -> None:
    """Backdate `path`, and everything directly in it, by `seconds`."""
    stamp = time.time() - seconds
    for entry in [path, *(path.iterdir() if path.is_dir() else [])]:
        os.utime(entry, (stamp, stamp))


def _temp_name(name: str) -> str:
    return f".{name}.{uuid4().hex}.tmp"


def _leftovers(runs):
    """An orphaned model directory and a stale temp file, both two days old."""
    orphan = runs / f"mdl_{uuid4().hex[:12]}"
    orphan.mkdir()
    (orphan / "model.joblib").write_bytes(b"half a registration")
    _age(orphan, TWO_DAYS)
    run = runs / "run1"
    run.mkdir()
    temp = run / _temp_name("eq.parquet")
    temp.write_bytes(b"never renamed into place")
    _age(temp, TWO_DAYS)
    return orphan, temp


def _keepers(runs):
    """Everything the sweep must never touch."""
    registered = runs / f"mdl_{uuid4().hex[:12]}"
    registered.mkdir()
    (registered / "model.joblib").write_bytes(b"model")
    (registered / "manifest.json").write_text("{}", encoding="utf-8")
    (registered / ".promotions.lock").write_bytes(b"")
    _age(registered, TWO_DAYS)

    dataset = runs / f"ds_{uuid4().hex[:12]}"
    dataset.mkdir()
    (dataset / "panel.parquet").write_bytes(b"panel")
    (dataset / "dataset_meta.json").write_text("{}", encoding="utf-8")
    _age(dataset, TWO_DAYS)

    published = runs / "run2"
    published.mkdir()
    (published / "curve.parquet").write_bytes(b"value")
    (published / "curve._handoff.json").write_text("{}", encoding="utf-8")
    _age(published, TWO_DAYS)

    # A registration still in progress: no commit file yet, but fresh.
    fresh = runs / f"mdl_{uuid4().hex[:12]}"
    fresh.mkdir()
    (fresh / "model.joblib").write_bytes(b"being written")
    _age(fresh, 60.0)

    # No commit file, but a published reference lives in it.
    referenced = runs / f"ds_{uuid4().hex[:12]}"
    referenced.mkdir()
    (referenced / "x.parquet").write_bytes(b"value")
    (referenced / "x._handoff.json").write_text("{}", encoding="utf-8")
    _age(referenced, TWO_DAYS)

    # Named like a model, but not an id the registry mints.
    lookalike = runs / "mdl_my_backtest"
    lookalike.mkdir()
    (lookalike / "equity.parquet").write_bytes(b"value")
    _age(lookalike, TWO_DAYS)
    return [registered, dataset, published, fresh, referenced, lookalike]


def _snapshot(paths):
    return {
        str(p): sorted(child.name for child in p.iterdir()) for p in paths if p.exists()
    }


class TestItCollectsOnlyLeftovers:
    def test_a_dry_run_lists_and_removes_nothing(self, runs):
        """Planted: an orphaned model directory and a stale temp file."""
        orphan, temp = _leftovers(runs)
        report = cli.cmd_runs_gc()
        assert report.removed is False
        assert report.partial_directories == [orphan]
        assert report.temp_files == [temp]
        assert orphan.exists() and temp.exists()

    def test_confirm_removes_them(self, runs):
        orphan, temp = _leftovers(runs)
        report = cli.cmd_runs_gc(confirm=True)
        assert report.removed is True
        assert not orphan.exists() and not temp.exists()

    def test_the_command_line_is_dry_run_by_default(self, runs, capsys):
        orphan, temp = _leftovers(runs)
        assert cli.main(["runs", "gc"]) == 0
        out = capsys.readouterr().out
        assert "dry-run" in out and orphan.name in out and temp.name in out
        assert orphan.exists() and temp.exists()
        assert cli.main(["runs", "gc", "--confirm"]) == 0
        assert not orphan.exists() and not temp.exists()


class TestItNeverTouchesWhatAReferenceCanName:
    def test_registered_published_fresh_and_referenced_are_kept(self, runs):
        """Null: a registered model, a dataset, a published value, a
        registration in progress, a directory holding a reference, and a
        run that only looks like a model id -- all untouched, even with
        --confirm."""
        keepers = _keepers(runs)
        before = _snapshot(keepers)
        report = cli.cmd_runs_gc(confirm=True)
        assert report.candidates == []
        assert _snapshot(keepers) == before

    def test_the_threshold_protects_recent_leftovers(self, runs):
        orphan, temp = _leftovers(runs)
        report = cli.cmd_runs_gc(confirm=True, older_than_hours=72)
        assert report.candidates == []
        assert orphan.exists() and temp.exists()

    def test_a_negative_threshold_is_refused(self, runs):
        with pytest.raises(ValueError, match="at least 0"):
            _runspath.sweep(older_than_hours=-1)

    def test_an_empty_or_missing_runs_directory_is_nothing_to_do(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "never_created"))
        assert cli.main(["runs", "gc"]) == 0
        assert "Nothing to collect" in capsys.readouterr().out
        assert not (tmp_path / "never_created").exists()
