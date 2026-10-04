"""
`scripts/mutation_testing.py` leaves the working tree as it found it (see
the CHANGELOG entry of 2026-10-04).

The harness mutates a source file, runs the tests that should notice, and
puts the file back. It used to put it back by re-encoding decoded text,
which rewrote line endings: an LF working copy came back CRLF on Windows
and a CRLF one came back LF elsewhere. `git status` then listed every
mutated file as modified with nothing for `git diff` to show, and the next
run refused to start because its clean-tree check read `git status`.

Each case here builds a throwaway repository with one line-ending setup --
`core.autocrlf` on and off, LF and CRLF working copies, and this
repository's own `.gitattributes` rule -- applies a mutation through the
harness, and checks that the file is byte-identical afterwards, with the
same modification time, that `git status` is empty, and that a second run
is allowed to start. The refusal is checked the other way: work git would
lose (an edit, a staged change, an untracked file) still stops a run, and a
file that differs from the index only in its stat or its line endings does
not. No test here runs pytest inside the harness; the restore is the
property under test.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parent.parent / "scripts" / "mutation_testing.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

SOURCE_LF = (
    b"def guard(value):\n"
    b"    if value > 10:\n"
    b"        raise ValueError(value)\n"
    b"    return value\n"
)
OLD = "    if value > 10:\n        raise ValueError(value)\n"
NEW = "    if value > 10:\n        pass\n"


def _load_harness():
    spec = importlib.util.spec_from_file_location("mutation_testing", HARNESS)
    assert spec and spec.loader, f"could not load {HARNESS}"
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: the dataclass under `from __future__
    # import annotations` resolves its annotations through `sys.modules`.
    sys.modules["mutation_testing"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop("mutation_testing", None)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


@pytest.fixture(autouse=True)
def _isolated_git(tmp_path_factory, monkeypatch):
    """The developer's and the machine's git configuration stay out of it:
    a system-wide `core.autocrlf=true` (Git for Windows' default) would
    otherwise decide the case a test thinks it is setting up."""
    empty = tmp_path_factory.mktemp("gitconfig") / "global"
    empty.write_bytes(b"")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def _repository(
    root: Path, autocrlf: str, committed: bytes, attributes: str = ""
) -> Path:
    """A repository holding `module.py` with `committed` as its blob, checked
    out again under `autocrlf` so the working copy has whatever line endings
    that setting gives it."""
    root.mkdir()
    _git(root, "init", "-q")
    for key, value in (
        ("user.name", "test"),
        ("user.email", "test@example.invalid"),
        ("commit.gpgsign", "false"),
        ("core.safecrlf", "false"),
        ("core.autocrlf", "false"),
    ):
        _git(root, "config", key, value)
    if attributes:
        (root / ".gitattributes").write_bytes(attributes.encode())
    source = root / "module.py"
    source.write_bytes(committed)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    _git(root, "config", "core.autocrlf", autocrlf)
    source.unlink()
    _git(root, "checkout", "--", "module.py")
    assert _git(root, "status", "--porcelain") == ""
    return source


CASES = {
    # (core.autocrlf, the committed blob, .gitattributes, working copy)
    "autocrlf off, LF": ("false", SOURCE_LF, "", b"\n"),
    "autocrlf on, LF blob checked out as CRLF": ("true", SOURCE_LF, "", b"\r\n"),
    "autocrlf off, CRLF blob": (
        "false",
        SOURCE_LF.replace(b"\n", b"\r\n"),
        "",
        b"\r\n",
    ),
    "autocrlf input, LF": ("input", SOURCE_LF, "", b"\n"),
    "autocrlf on, eol=lf attribute (this repository)": (
        "true",
        SOURCE_LF,
        "* text=auto eol=lf\n*.py text eol=lf\n",
        b"\n",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_run_leaves_the_file_byte_identical_and_git_clean(harness, tmp_path, case):
    """The bytes, the modification time and `git status` after the restore
    are what they were before the mutation, and the mutated file kept its
    own line endings while the mutation was live."""
    autocrlf, committed, attributes, newline = CASES[case]
    source = _repository(tmp_path / "repo", autocrlf, committed, attributes)
    before_bytes = source.read_bytes()
    assert before_bytes == SOURCE_LF.replace(b"\n", newline), "setup"
    # A modification time well in the past, so an unchanged one is not a
    # coincidence of the clock's resolution.
    os.utime(source, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    before_mtime = source.stat().st_mtime_ns
    _git(tmp_path / "repo", "status", "--porcelain")  # refresh the index

    mutation = harness.Mutation("guard: drop the raise", source, OLD, NEW, "-")
    with harness.applied(mutation) as live:
        assert live
        during = source.read_bytes()
        assert during == before_bytes.replace(
            OLD.encode().replace(b"\n", newline), NEW.encode().replace(b"\n", newline)
        )
        assert b"raise" not in during

    assert source.read_bytes() == before_bytes
    assert source.stat().st_mtime_ns == before_mtime
    assert _git(tmp_path / "repo", "status", "--porcelain") == ""
    # The second run is allowed to start, and restores just as cleanly.
    harness.require_clean_tree([source], root=tmp_path / "repo")
    with harness.applied(mutation) as live:
        assert live
    assert source.read_bytes() == before_bytes
    assert _git(tmp_path / "repo", "status", "--porcelain") == ""


def test_a_test_run_that_raises_still_restores(harness, tmp_path):
    """The restore is in a `finally`: an exception from the test run (a
    timeout, Ctrl-C) leaves the original bytes behind, not the mutation."""
    source = _repository(tmp_path / "repo", "true", SOURCE_LF)
    before = source.read_bytes()
    mutation = harness.Mutation("guard: drop the raise", source, OLD, NEW, "-")
    with pytest.raises(KeyboardInterrupt):
        with harness.applied(mutation):
            raise KeyboardInterrupt
    assert source.read_bytes() == before
    assert _git(tmp_path / "repo", "status", "--porcelain") == ""


def test_an_anchor_that_does_not_match_once_writes_nothing(harness, tmp_path):
    """A drifted anchor is reported as skipped by the caller; the file is
    never opened for writing, so even its modification time is untouched."""
    source = _repository(tmp_path / "repo", "false", SOURCE_LF)
    os.utime(source, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))
    mutation = harness.Mutation("absent", source, "no such line\n", "x\n", "-")
    with harness.applied(mutation) as live:
        assert live is False
    assert source.read_bytes() == SOURCE_LF
    assert source.stat().st_mtime_ns == 1_600_000_000_000_000_000


class TestTheRefusalReadsContent:
    """Work `git checkout --` would discard still stops a run; a difference
    git itself does not count as a change no longer does."""

    def test_an_edit_is_refused(self, harness, tmp_path):
        source = _repository(tmp_path / "repo", "false", SOURCE_LF)
        source.write_bytes(SOURCE_LF + b"# work in progress\n")
        with pytest.raises(SystemExit, match=r"module\.py \(modified\)"):
            harness.require_clean_tree([source], root=tmp_path / "repo")

    def test_a_staged_change_is_refused(self, harness, tmp_path):
        source = _repository(tmp_path / "repo", "false", SOURCE_LF)
        source.write_bytes(SOURCE_LF + b"# staged\n")
        _git(tmp_path / "repo", "add", "module.py")
        with pytest.raises(SystemExit, match=r"staged, not committed"):
            harness.require_clean_tree([source], root=tmp_path / "repo")

    def test_an_untracked_file_is_refused(self, harness, tmp_path):
        _repository(tmp_path / "repo", "false", SOURCE_LF)
        extra = tmp_path / "repo" / "new.py"
        extra.write_bytes(SOURCE_LF)
        with pytest.raises(SystemExit, match=r"not tracked by git"):
            harness.require_clean_tree([extra], root=tmp_path / "repo")

    def test_a_touched_file_is_not_refused(self, harness, tmp_path):
        source = _repository(tmp_path / "repo", "false", SOURCE_LF)
        source.write_bytes(SOURCE_LF)
        os.utime(source, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
        harness.require_clean_tree([source], root=tmp_path / "repo")

    def test_a_line_ending_rewrite_git_normalizes_away_is_not_refused(
        self, harness, tmp_path
    ):
        """The state the old restore left behind: an `eol=lf` file rewritten
        with CRLF. `git status` lists it (its size changed) but its content
        is the index's once normalized, so it is not work to protect."""
        source = _repository(
            tmp_path / "repo", "true", SOURCE_LF, "* text=auto eol=lf\n"
        )
        source.write_bytes(SOURCE_LF.replace(b"\n", b"\r\n"))
        harness.require_clean_tree([source], root=tmp_path / "repo")
