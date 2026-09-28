"""
`SQT_RUNS_DIR` read the way every path setting is read.

It was `os.environ.get(VAR, default)`, so a variable set to nothing -- the
`VAR=` accident in a unit file or a compose file -- was the path `""`, the
process working directory, and the artifact store moved to wherever the
client happened to launch the server. A relative value moved with every
`os.chdir`, so a reference stopped resolving after one; `~` was a literal
directory named `~`; and a value naming a file surfaced as a raw
`FileExistsError` from the first write. Blank is now the default, `~` is
home, and a relative path or a file is refused by name.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from standard_quant_tools import _runspath
from standard_quant_tools.backtest.artifacts import save_artifact
from standard_quant_tools.error import ValidationError


@pytest.fixture
def home_and_cwd(tmp_path, monkeypatch):
    """A fake home directory, and a working directory that is somewhere else."""
    home = tmp_path / "home"
    cwd = tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(cwd)
    return home, cwd


def _series() -> pd.Series:
    return pd.Series([1.0, 2.0, 3.0], name="equity")


class TestABlankValueIsTheDefault:
    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_an_artifact_lands_under_home_not_in_the_working_directory(
        self, home_and_cwd, monkeypatch, blank
    ):
        """Planted: `SQT_RUNS_DIR=` used to write into the working directory."""
        home, cwd = home_and_cwd
        monkeypatch.setenv("SQT_RUNS_DIR", blank)
        uri = Path(save_artifact(_series(), "run1", "equity"))
        default = home / ".cache" / "standard_quant_tools" / "runs"
        assert uri.is_relative_to(default.resolve())
        assert not (cwd / "run1").exists()

    def test_tilde_is_home(self, home_and_cwd, monkeypatch):
        home, cwd = home_and_cwd
        monkeypatch.setenv("SQT_RUNS_DIR", "~/sqt_runs")
        assert _runspath.runs_dir() == home / "sqt_runs"
        assert not (cwd / "~").exists()


class TestAValueThatCannotBeTheRootIsRefused:
    def test_a_relative_path_is_refused_by_name(self, home_and_cwd, monkeypatch):
        _home, cwd = home_and_cwd
        monkeypatch.setenv("SQT_RUNS_DIR", "rel/x")
        with pytest.raises(ValidationError, match="SQT_RUNS_DIR is a relative path"):
            save_artifact(_series(), "run1", "equity")
        assert not (cwd / "rel").exists()

    def test_a_file_is_refused_by_name_not_by_the_os(self, tmp_path, monkeypatch):
        """Planted: this was `FileExistsError [WinError 183]` from mkdir."""
        a_file = tmp_path / "not_a_directory"
        a_file.write_text("x", encoding="utf-8")
        monkeypatch.setenv("SQT_RUNS_DIR", str(a_file))
        with pytest.raises(ValidationError, match="names an existing file"):
            save_artifact(_series(), "run1", "equity")

    def test_the_refusal_does_not_echo_the_value(self, monkeypatch):
        monkeypatch.setenv("SQT_RUNS_DIR", "relative-zz-planted-zz")
        with pytest.raises(ValidationError) as excinfo:
            _runspath.runs_dir()
        assert "zz-planted-zz" not in str(excinfo.value)


class TestAnAbsoluteValueIsUnchanged:
    def test_it_is_the_root(self, tmp_path, monkeypatch):
        """Null."""
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        uri = Path(save_artifact(_series(), "run1", "equity"))
        assert uri.is_relative_to((tmp_path / "runs").resolve())


class TestADotEnvValueIsInForceFromTheFirstRead:
    def test_the_first_read_already_sees_it(self, tmp_path, monkeypatch):
        """Planted: the runs root used to change mid-process, when some
        other reader happened to load `.env` first. Loading is stood in for
        here, so the test does not depend on where python-dotenv looks."""
        from standard_quant_tools import config

        target = tmp_path / "runs_from_dotenv"
        monkeypatch.delenv("SQT_RUNS_DIR")

        def load_env(dotenv_path=None):
            monkeypatch.setenv("SQT_RUNS_DIR", str(target))
            return True

        monkeypatch.setattr(config, "load_env", load_env)
        assert _runspath.runs_dir() == target
