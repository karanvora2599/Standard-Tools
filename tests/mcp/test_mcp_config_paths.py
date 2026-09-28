"""
The server's startup check reads the storage settings the way the library
does, and says where external data may be read from.

The check used to treat a blank value as unset while the library treated it
as the working directory, and it expanded `~` while the library did not --
so it probed one directory and the library wrote to another. It now reads
through the same helper: blank is the default (and the warning names the
default, not the working directory), `~` is home for both, and a relative
path or a file stops the server at startup.

The bearer token is sent in an HTTP header, which carries ASCII; a token
with any other character could never be presented, and is refused at
startup without being echoed.
"""

from __future__ import annotations

import pytest

from standard_quant_tools.mcp.config import TOKEN_ENV_VAR, report, resolve


@pytest.fixture(autouse=True)
def _no_ambient_token(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV_VAR, raising=False)


@pytest.fixture
def home(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))
    return home


class TestTheStartupCheckProbesWhereTheLibraryWrites:
    def test_tilde_is_the_same_directory_for_both(self, home, monkeypatch):
        from standard_quant_tools._runspath import runs_dir

        monkeypatch.setenv("SQT_RUNS_DIR", "~/sqt_runs")
        config = resolve([])
        assert config.runs_dir == home / "sqt_runs" == runs_dir()
        assert (home / "sqt_runs").is_dir()

    def test_a_relative_path_stops_the_server(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("SQT_RUNS_DIR", "rel")
        with pytest.raises(SystemExit, match="SQT_RUNS_DIR is a relative path"):
            resolve([])
        assert not (tmp_path / "rel").exists()

    def test_a_file_stops_the_server(self, tmp_path, monkeypatch):
        a_file = tmp_path / "a_file"
        a_file.write_text("x", encoding="utf-8")
        monkeypatch.setenv("SQT_CACHE_DIR", str(a_file))
        with pytest.raises(SystemExit, match="SQT_CACHE_DIR names an existing file"):
            resolve([])

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_a_blank_value_is_the_default_and_the_warning_says_which(
        self, home, monkeypatch, blank
    ):
        """Planted: the warning said the default was relative to the working
        directory, which was only ever true of the blank case -- and the
        library then did use the working directory."""
        monkeypatch.setenv("SQT_RUNS_DIR", blank)
        config = resolve([])
        assert config.runs_dir is None
        (warning,) = [w for w in config.warnings if "SQT_RUNS_DIR" in w]
        assert "working directory" not in warning
        assert str(home / ".cache" / "standard_quant_tools" / "runs") in warning

    def test_an_absolute_value_is_probed_and_used(self, tmp_path, monkeypatch):
        """Null."""
        monkeypatch.setenv("SQT_RUNS_DIR", str(tmp_path / "runs"))
        config = resolve([])
        assert config.runs_dir == tmp_path / "runs"
        assert not any("SQT_RUNS_DIR" in w for w in config.warnings)


class TestTheReportNamesTheExternalFence:
    def test_the_listed_directories_are_reported(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", str(tmp_path / "vendor"))
        config = resolve([])
        assert config.external_dirs == (tmp_path / "vendor",)
        report(config, tool_count=1, schema_bytes_total=1024)
        err = capsys.readouterr().err
        line = next(ln for ln in err.splitlines() if "SQT_EXTERNAL_DIRS" in ln)
        assert str(tmp_path / "vendor") in line and "runs directory" in line

    def test_unset_is_reported_as_the_runs_directory_only(self, monkeypatch, capsys):
        monkeypatch.delenv("SQT_EXTERNAL_DIRS", raising=False)
        config = resolve([])
        assert config.external_dirs == ()
        report(config, tool_count=1, schema_bytes_total=1024)
        assert "the runs directory only" in capsys.readouterr().err

    def test_a_relative_entry_stops_the_server(self, monkeypatch):
        monkeypatch.setenv("SQT_EXTERNAL_DIRS", "relative/vendor")
        with pytest.raises(SystemExit, match="SQT_EXTERNAL_DIRS is a relative path"):
            resolve([])


class TestTheTokenCharacterSet:
    @pytest.mark.parametrize("token", ["tökén-secret", "emoji-\U0001f600"])
    def test_a_token_no_header_can_carry_is_refused(self, monkeypatch, token):
        monkeypatch.setenv(TOKEN_ENV_VAR, token)
        with pytest.raises(SystemExit, match="not printable ASCII") as excinfo:
            resolve(["--transport", "http"])
        assert token not in str(excinfo.value)

    def test_a_base64_token_is_accepted(self, monkeypatch):
        """Null."""
        monkeypatch.setenv(TOKEN_ENV_VAR, "q2V+/xZ3aW5kb3dz-_.~=")
        config = resolve(["--transport", "http"])
        assert config.auth_token == "q2V+/xZ3aW5kb3dz-_.~="
