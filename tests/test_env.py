"""
One reading of an environment setting.

Three readers in this library disagreed about what `VAR=` means -- unset,
the working directory, or "off" -- and none of them refused a misspelled
flag. The helpers in `standard_quant_tools._env` are the single reading, and
these tests are its table: blank is unset, a relative path is refused by
name, an unknown word is refused rather than read as either answer, and a
refusal never repeats the value it refused.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from standard_quant_tools import _env
from standard_quant_tools._env import (
    FALSE_WORDS,
    TRUE_WORDS,
    env_flag,
    env_int,
    env_path,
    env_paths,
    env_str,
)
from standard_quant_tools.error import ValidationError

NAME = "ENV_HELPER_PROBE"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(NAME, raising=False)


class TestBlankIsUnset:
    @pytest.mark.parametrize("raw", ["", "   ", "\t\n"])
    def test_blank_reads_as_unset_everywhere(self, monkeypatch, tmp_path, raw):
        monkeypatch.setenv(NAME, raw)
        assert env_str(NAME) is None
        assert env_path(NAME, default=tmp_path) == tmp_path
        assert env_paths(NAME) == ()
        assert env_flag(NAME, default=True) is True
        assert env_int(NAME, default=8) == 8

    def test_unset_gives_every_default(self, tmp_path):
        assert env_str(NAME) is None
        assert env_path(NAME) is None
        assert env_path(NAME, default=tmp_path) == tmp_path
        assert env_path(NAME, default=lambda: tmp_path / "later") == tmp_path / "later"
        assert env_paths(NAME) == ()
        assert env_flag(NAME, default=False) is False
        assert env_int(NAME) is None
        assert env_int(NAME, default=3) == 3

    def test_padding_is_stripped(self, monkeypatch):
        monkeypatch.setenv(NAME, "  joblib  ")
        assert env_str(NAME) == "joblib"


class TestPaths:
    def test_an_absolute_directory_is_returned(self, monkeypatch, tmp_path):
        monkeypatch.setenv(NAME, f"  {tmp_path}  ")
        assert env_path(NAME) == tmp_path

    def test_a_path_that_does_not_exist_yet_is_accepted_and_not_created(
        self, monkeypatch, tmp_path
    ):
        later = tmp_path / "not" / "yet"
        monkeypatch.setenv(NAME, str(later))
        assert env_path(NAME) == later
        assert not later.exists()

    def test_tilde_is_the_home_directory(self, monkeypatch):
        monkeypatch.setenv(NAME, "~/sqt-env-probe")
        assert env_path(NAME) == Path.home() / "sqt-env-probe"

    @pytest.mark.parametrize("raw", ["rel/x", "runs", "./here", "../up"])
    def test_a_relative_path_is_refused_by_name_without_its_value(
        self, monkeypatch, raw
    ):
        """The anchor would be the working directory, which the launcher
        chose and `os.chdir` moves."""
        monkeypatch.setenv(NAME, raw)
        with pytest.raises(ValidationError, match="relative path") as caught:
            env_path(NAME)
        assert NAME in str(caught.value)
        assert raw not in str(caught.value)

    def test_a_directory_setting_naming_a_file_is_refused(self, monkeypatch, tmp_path):
        existing = tmp_path / "a-file.txt"
        existing.write_text("x", encoding="utf-8")
        monkeypatch.setenv(NAME, str(existing))
        with pytest.raises(ValidationError, match="names an existing file") as caught:
            env_path(NAME)
        assert str(existing) not in str(caught.value)

    def test_a_file_setting_naming_a_directory_is_refused(self, monkeypatch, tmp_path):
        monkeypatch.setenv(NAME, str(tmp_path))
        with pytest.raises(ValidationError, match="names a directory"):
            env_path(NAME, kind="file")

    def test_a_file_setting_naming_a_file_is_returned(self, monkeypatch, tmp_path):
        key = tmp_path / "key.pem"
        key.write_text("x", encoding="utf-8")
        monkeypatch.setenv(NAME, str(key))
        assert env_path(NAME, kind="file") == key


class TestPathLists:
    def test_entries_are_split_on_the_platform_separator(self, monkeypatch, tmp_path):
        first, second = tmp_path / "a", tmp_path / "b"
        monkeypatch.setenv(NAME, os.pathsep.join([str(first), f" {second} "]))
        assert env_paths(NAME) == (first, second)

    def test_empty_entries_and_repeats_are_dropped(self, monkeypatch, tmp_path):
        entry = str(tmp_path / "a")
        monkeypatch.setenv(
            NAME, os.pathsep.join(["", entry, "  ", entry, ""]) + os.pathsep
        )
        assert env_paths(NAME) == (tmp_path / "a",)

    def test_a_relative_entry_is_refused_by_position(self, monkeypatch, tmp_path):
        monkeypatch.setenv(NAME, os.pathsep.join([str(tmp_path), "rel/inbox"]))
        with pytest.raises(ValidationError, match="entry 2 of 2") as caught:
            env_paths(NAME)
        message = str(caught.value)
        assert NAME in message
        assert "rel/inbox" not in message
        assert str(tmp_path) not in message


class TestFlags:
    @pytest.mark.parametrize("word", sorted(TRUE_WORDS))
    def test_every_true_word_padded_and_in_any_case(self, monkeypatch, word):
        for spelled in (word, word.upper(), f"  {word.title()} "):
            monkeypatch.setenv(NAME, spelled)
            assert env_flag(NAME, default=False) is True

    @pytest.mark.parametrize("word", sorted(FALSE_WORDS))
    def test_every_false_word_padded_and_in_any_case(self, monkeypatch, word):
        for spelled in (word, word.upper(), f"  {word.title()} "):
            monkeypatch.setenv(NAME, spelled)
            assert env_flag(NAME, default=True) is False

    @pytest.mark.parametrize("word", ["flase", "enabled", "2", "y"])
    def test_an_unknown_word_is_refused_not_read_as_either(self, monkeypatch, word):
        monkeypatch.setenv(NAME, word)
        with pytest.raises(ValidationError, match="neither on nor off") as caught:
            env_flag(NAME, default=True)
        assert NAME in str(caught.value)
        assert repr(word) not in str(caught.value)


class TestIntegers:
    def test_a_padded_integer_is_read(self, monkeypatch):
        monkeypatch.setenv(NAME, " 30 ")
        assert env_int(NAME, default=8) == 30

    def test_a_non_integer_is_refused_without_its_value(self, monkeypatch):
        monkeypatch.setenv(NAME, "4abc")
        with pytest.raises(ValidationError, match="not an integer") as caught:
            env_int(NAME, default=8)
        assert "4abc" not in str(caught.value)

    def test_the_bounds_are_enforced(self, monkeypatch):
        monkeypatch.setenv(NAME, "-5")
        with pytest.raises(ValidationError, match="below its minimum of 1"):
            env_int(NAME, minimum=1)
        monkeypatch.setenv(NAME, "70000")
        with pytest.raises(ValidationError, match="above its maximum of 65535"):
            env_int(NAME, maximum=65535)
        monkeypatch.setenv(NAME, "1")
        assert env_int(NAME, minimum=1, maximum=65535) == 1


class TestADotEnvIsLoadedFirst:
    """A value supplied by a local `.env` is in force from the first read of
    any setting, not from whichever reader happened to load the file."""

    def test_every_reader_loads_before_it_reads(self, monkeypatch, tmp_path):
        from standard_quant_tools import config

        calls = []

        def fake_load_env(*_args, **_kwargs):
            calls.append(True)
            os.environ[NAME] = str(tmp_path)
            return True

        monkeypatch.setattr(config, "load_env", fake_load_env)
        assert env_str(NAME) == str(tmp_path)
        assert calls

    def test_load_false_does_not(self, monkeypatch):
        from standard_quant_tools import config

        def fail(*_args, **_kwargs):
            raise AssertionError("load_env ran for a reader that asked it not to")

        monkeypatch.setattr(config, "load_env", fail)
        assert env_str(NAME, load=False) is None

    def test_a_failing_load_is_not_a_setting(self, monkeypatch):
        from standard_quant_tools import config

        def broken(*_args, **_kwargs):
            raise OSError("unreadable .env")

        monkeypatch.setattr(config, "load_env", broken)
        monkeypatch.setenv(NAME, "on")
        assert env_flag(NAME, default=False) is True


def test_nothing_else_is_exported():
    assert set(_env.__all__) == {
        "FALSE_WORDS",
        "TRUE_WORDS",
        "env_flag",
        "env_int",
        "env_path",
        "env_paths",
        "env_str",
    }
